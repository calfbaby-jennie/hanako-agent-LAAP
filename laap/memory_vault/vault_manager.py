"""Per-agent durable Memory Vault.

The upstream LAAP modules have long depended on ``laap.memory_vault`` while the
package was absent from the published tree. This implementation restores the
contract with SQLite and an optional SQLCipher backend. Agent names are
normalized into separate database files; callers never choose arbitrary paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

try:  # Optional at runtime; stdlib sqlite remains a functional local vault.
    from pysqlcipher3 import dbapi2 as _sqlcipher  # type: ignore
    _SQLCIPHER_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    _sqlcipher = None
    _SQLCIPHER_AVAILABLE = False

VAULT_DIR = os.environ.get(
    "LAAP_VAULT_DIR", str(Path.home() / ".laap" / "memory_vault")
)
_AGENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _open_vault_connection(db_path: str | Path, key_hex: str = ""):
    """Open a row-addressable SQLite/SQLCipher connection.

    The key is applied only when SQLCipher is installed. A standard SQLite
    connection is intentionally reported as unencrypted through
    ``_SQLCIPHER_AVAILABLE`` so callers can expose the security state instead
    of silently claiming encryption.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _SQLCIPHER_AVAILABLE:
        conn = _sqlcipher.connect(str(path), timeout=30, check_same_thread=False)
        if key_hex:
            conn.execute(f"PRAGMA key = \"x'{key_hex}'\"")
    else:
        conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


class VaultManager:
    """Thread-safe, per-agent memory vault manager."""

    def __init__(self, vault_dir: str | Path | None = None) -> None:
        self.vault_dir = str(Path(vault_dir or VAULT_DIR).expanduser())
        Path(self.vault_dir).mkdir(parents=True, exist_ok=True)
        self._vault_cache: dict[str, tuple[str, str]] = {}
        self._cache_lock = threading.RLock()

    @staticmethod
    def _agent_key(agent_name: str) -> str:
        clean = _AGENT_RE.sub("_", str(agent_name or "aris")).strip("._") or "aris"
        return clean[:96]

    def _get_vault(self, agent_name: str) -> tuple[str, str]:
        agent = self._agent_key(agent_name)
        with self._cache_lock:
            cached = self._vault_cache.get(agent)
            if cached:
                return cached
            db_path = str(Path(self.vault_dir) / f"{agent}_vault.db")
            # Deterministic local key avoids a second secret store while keeping
            # per-agent separation when SQLCipher is available.
            seed = f"laap-vault:{agent}:{Path(self.vault_dir).resolve()}".encode()
            key_hex = hashlib.sha256(seed).hexdigest()
            conn = _open_vault_connection(db_path, key_hex)
            try:
                self._ensure_schema(conn)
                conn.commit()
            finally:
                conn.close()
            self._vault_cache[agent] = (db_path, key_hex)
            return db_path, key_hex

    init_for_agent = _get_vault

    @staticmethod
    def _ensure_schema(conn) -> None:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                scope TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_scope
                ON memories(agent_name, scope, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_created
                ON memories(created_at DESC);
        """)

    def store(
        self, agent_name: str, scope: str, content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not str(content).strip():
            raise ValueError("content must be non-empty")
        scope = str(scope or "episodic").strip().lower()
        if scope not in {"episodic", "semantic", "procedural", "working", "meta"}:
            raise ValueError(f"unsupported memory scope: {scope}")
        agent = self._agent_key(agent_name)
        db_path, key_hex = self._get_vault(agent)
        now = time.time()
        memory_id = f"mem_{uuid.uuid4().hex[:16]}"
        conn = _open_vault_connection(db_path, key_hex)
        try:
            self._ensure_schema(conn)
            conn.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?)",
                (memory_id, agent, scope, str(content),
                 json.dumps(metadata or {}, ensure_ascii=False, default=str), now, now),
            )
            conn.commit()
        finally:
            conn.close()
        return {"stored": True, "memory_id": memory_id, "scope": scope, "agent_name": agent}

    def retrieve(
        self, agent_name: str, query: str, scope: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        agent = self._agent_key(agent_name)
        db_path, key_hex = self._get_vault(agent)
        terms = [term for term in re.split(r"\s+", str(query).strip()) if term][:8]
        clauses = ["agent_name = ?"]
        params: list[Any] = [agent]
        if scope:
            clauses.append("scope = ?")
            params.append(str(scope).lower())
        if terms:
            clauses.append("(" + " OR ".join("content LIKE ?" for _ in terms) + ")")
            params.extend(f"%{term}%" for term in terms)
        params.append(max(1, min(int(limit), 200)))
        conn = _open_vault_connection(db_path, key_hex)
        try:
            self._ensure_schema(conn)
            rows = conn.execute(
                "SELECT * FROM memories WHERE " + " AND ".join(clauses)
                + " ORDER BY created_at DESC LIMIT ?", params,
            ).fetchall()
        finally:
            conn.close()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.pop("metadata_json"))
            except Exception:
                item["metadata"] = {}
            result.append(item)
        return result

    def consolidate(self, agent_name: Optional[str] = None) -> dict[str, Any]:
        agents = [self._agent_key(agent_name)] if agent_name else sorted(
            path.name[:-9] for path in Path(self.vault_dir).glob("*_vault.db")
        )
        summaries = {}
        for agent in agents:
            db_path, key_hex = self._get_vault(agent)
            conn = _open_vault_connection(db_path, key_hex)
            try:
                self._ensure_schema(conn)
                total, earliest, latest = conn.execute(
                    "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM memories WHERE agent_name = ?",
                    (agent,),
                ).fetchone()
                scopes = {
                    row[0]: row[1] for row in conn.execute(
                        "SELECT scope, COUNT(*) FROM memories WHERE agent_name = ? GROUP BY scope",
                        (agent,),
                    ).fetchall()
                }
            finally:
                conn.close()
            summaries[agent] = {
                "total": total, "by_scope": scopes,
                "earliest": earliest, "latest": latest,
            }
        return {"agents": summaries, "encrypted": _SQLCIPHER_AVAILABLE}


vault_manager = VaultManager()
