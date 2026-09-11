"""Capability and environment status tools."""

from __future__ import annotations

from typing import Any

from isaaclab_mcp.status import collect_status


def register_system_tools(mcp: Any) -> None:
    """Register status-oriented tools."""

    @mcp.tool("get_isaac_lab_capabilities")
    def get_isaac_lab_capabilities() -> dict[str, Any]:
        """Return the implemented IsaacLab-MCP capability boundary."""
        return {
            "server_stage": "training-authoring",
            "named_tools": {
                "get_isaac_lab_capabilities": "implemented",
                "get_isaac_lab_status": "implemented and read-only",
                "list_isaac_lab_tasks": "implemented using static source discovery",
                "design_lifting_training": "implemented; creates a validated RL design",
                "create_lifting_training_project": "implemented; preview defaults to true",
                "validate_lifting_training_project": "implemented; checks schema and asset readiness",
            },
            "runtime_execution": {
                "support": "manual CLI task is implemented; MCP job-control tools are not implemented",
                "required_route": "C:\\isaacsim\\python.bat",
                "verified_tasks": ["Isaac-Lift-Cube-Dofbot-v0", "Isaac-Reach-TM6S-Lift-Proxy-v0"],
                "planned_named_tools": ["train", "play", "job status", "cancel job", "metrics"],
            },
            "limitations": [
                "Static task discovery does not guarantee dynamically generated registry entries.",
                "No tool starts Isaac Sim, training, or a live environment in this skeleton.",
                "Generated design packets require validated company assets before runtime code generation.",
            ],
        }

    @mcp.tool("get_isaac_lab_status")
    def get_isaac_lab_status() -> dict[str, Any]:
        """Inspect the configured Isaac Lab checkout without modifying it."""
        return collect_status()
