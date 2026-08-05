"""LAAP — Sub-agent delegation tool (simulated)."""

from __future__ import annotations
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from laap.tools.base import ToolResult

logger = logging.getLogger("laap.tools.delegate")


def _get_storage_dir() -> Path:
    """Return the directory used for LAAP JSON storage."""
    path = Path(os.environ.get("LAAP_STORAGE_DIR", ".laap"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _delegate_file() -> Path:
    return _get_storage_dir() / "delegate.json"


def _load_registry() -> Dict[str, Any]:
    default_registry: Dict[str, Any] = {"tasks": {}}
    file_path = _delegate_file()
    if not file_path.exists():
        return default_registry
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load delegate registry: {exc}")
        return default_registry
    return data if isinstance(data, dict) and isinstance(data.get("tasks"), dict) else default_registry


def _save_registry(data: Dict[str, Any]) -> None:
    file_path = _delegate_file()
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sub_agent_spawn(goal: str, context: Optional[str] = None) -> ToolResult:
    """Spawn a simulated sub-agent task and return its task_id."""
    try:
        data = _load_registry()
        task_id = uuid.uuid4().hex[:12]
        data["tasks"][task_id] = {
            "id": task_id,
            "goal": goal,
            "context": context or "",
            "status": "pending",
            "result": None,
            "created_at": time.time(),
        }
        _save_registry(data)
        return ToolResult(
            success=True,
            output=task_id,
            metadata={"task_id": task_id, "status": "pending"},
        )
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


def sub_agent_collect(task_id: str, timeout: int = 30) -> ToolResult:
    """Collect the result of a simulated sub-agent task."""
    try:
        data = _load_registry()
        task = data["tasks"].get(task_id)
        if not task:
            return ToolResult(
                success=False,
                output="",
                error=f"Task not found: {task_id}",
                metadata={"task_id": task_id},
            )

        # Simulate sub-agent execution: mark as completed and echo the goal.
        if task.get("status") == "pending":
            task["status"] = "completed"
            task["result"] = {
                "summary": f"Simulated sub-agent completed goal: {task['goal']}",
                "context": task.get("context", ""),
            }
            task["completed_at"] = time.time()
            _save_registry(data)

        return ToolResult(
            success=True,
            output=json.dumps(task["result"], ensure_ascii=False),
            metadata={
                "task_id": task_id,
                "status": task["status"],
                "goal": task["goal"],
            },
        )
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))
