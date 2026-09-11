"""Isaac Lab task discovery tools."""

from __future__ import annotations

from typing import Any

from isaaclab_mcp.discovery import discover_tasks
from isaaclab_mcp.settings import load_settings


def register_task_tools(mcp: Any) -> None:
    """Register task-oriented tools."""

    @mcp.tool("list_isaac_lab_tasks")
    def list_isaac_lab_tasks(keyword: str | None = None, limit: int = 200) -> dict[str, Any]:
        """List statically declared Isaac Lab task IDs, optionally filtered by keyword."""
        settings = load_settings()
        if settings.isaaclab_path is None:
            return {"ok": False, "error": "ISAACLAB_PATH is not configured", "tasks": []}

        tasks = discover_tasks(settings.isaaclab_path, keyword=keyword, limit=limit)
        return {
            "ok": settings.isaaclab_path.is_dir(),
            "discovery_mode": "static_source",
            "isaaclab_path": str(settings.isaaclab_path),
            "keyword": keyword,
            "count": len(tasks),
            "limit": limit,
            "tasks": tasks,
        }
