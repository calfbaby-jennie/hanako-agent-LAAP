"""LAAP — Simple kanban task board backed by a JSON file."""

from __future__ import annotations
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from laap.tools.base import ToolResult

logger = logging.getLogger("laap.tools.kanban")


def _get_storage_dir() -> Path:
    """Return the directory used for LAAP JSON storage."""
    path = Path(os.environ.get("LAAP_STORAGE_DIR", ".laap"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _kanban_file() -> Path:
    return _get_storage_dir() / "kanban.json"


def _load_kanban() -> Dict[str, List[Dict[str, str]]]:
    empty: Dict[str, List[Dict[str, str]]] = {"tasks": []}
    file_path = _kanban_file()
    if not file_path.exists():
        return empty
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load kanban: {exc}")
        return empty
    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        return data
    return empty


def _save_kanban(data: Dict[str, List[Dict[str, str]]]) -> None:
    file_path = _kanban_file()
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def task_create(title: str, description: str = "", status: str = "todo") -> ToolResult:
    """Create a new kanban task."""
    try:
        data = _load_kanban()
        task = {
            "id": uuid.uuid4().hex[:12],
            "title": title,
            "description": description,
            "status": status,
        }
        data["tasks"].append(task)
        _save_kanban(data)
        return ToolResult(
            success=True,
            output=json.dumps(task, ensure_ascii=False),
            metadata={"task_id": task["id"]},
        )
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


def task_list(status: Optional[str] = None) -> ToolResult:
    """List kanban tasks, optionally filtered by status."""
    try:
        data = _load_kanban()
        tasks = data["tasks"]
        if status:
            tasks = [t for t in tasks if t.get("status") == status]
        return ToolResult(
            success=True,
            output=json.dumps(tasks, ensure_ascii=False),
            metadata={"count": len(tasks), "status_filter": status},
        )
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


def task_update(
    task_id: str,
    status: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> ToolResult:
    """Update an existing kanban task."""
    try:
        data = _load_kanban()
        for task in data["tasks"]:
            if task.get("id") == task_id:
                if status is not None:
                    task["status"] = status
                if title is not None:
                    task["title"] = title
                if description is not None:
                    task["description"] = description
                _save_kanban(data)
                return ToolResult(
                    success=True,
                    output=json.dumps(task, ensure_ascii=False),
                    metadata={"task_id": task_id},
                )
        return ToolResult(
            success=False,
            output="",
            error=f"Task not found: {task_id}",
            metadata={"task_id": task_id},
        )
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))
