"""IsaacLab-MCP tool registration."""

from __future__ import annotations

from typing import Any

from isaaclab_mcp.tools.system import register_system_tools
from isaaclab_mcp.tools.tasks import register_task_tools
from isaaclab_mcp.tools.training import register_training_tools


def register_all_tools(mcp: Any) -> None:
    """Register every public MCP tool."""
    register_system_tools(mcp)
    register_task_tools(mcp)
    register_training_tools(mcp)
