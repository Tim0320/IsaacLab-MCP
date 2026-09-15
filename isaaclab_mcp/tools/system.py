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
            "server_stage": "restricted-dofbot-job-control",
            "named_tools": {
                "get_isaac_lab_capabilities": "implemented",
                "get_isaac_lab_status": "implemented and read-only",
                "list_isaac_lab_tasks": "implemented using static source discovery",
                "design_lifting_training": "implemented; creates a validated RL design",
                "create_lifting_training_project": "implemented; preview defaults to true",
                "validate_lifting_training_project": "implemented; checks schema and asset readiness",
                "validate_environment_contract": "implemented; read-only Sim-to-Lab contract validation",
                "validate_evidence_bundle": "implemented; read-only verification-target validation",
                "validate_scene_change_request": "implemented; read-only Lab-to-Sim request validation",
                "submit_dofbot_training_run": "implemented; allow-listed visible Dofbot runner with mandatory RecordVideo",
                "get_training_run_status": "implemented; reads durable job state and artifacts",
                "cancel_training_run": "implemented; requests cancellation through a worker-owned control file",
            },
            "runtime_execution": {
                "support": "MCP job control is limited to two visible Dofbot tasks; other tasks remain manual CLI only",
                "required_route": "C:\\isaacsim\\python.bat",
                "verified_tasks": ["Isaac-Lift-Cube-Dofbot-v0", "Isaac-Reach-TM6S-Lift-Proxy-v0"],
                "implemented_named_tools": ["submit_dofbot_training_run", "get_training_run_status", "cancel_training_run"],
            },
            "limitations": [
                "Static task discovery does not guarantee dynamically generated registry entries.",
                "The runner does not evaluate policy quality; Verification Agent evidence remains a separate gate.",
                "Generated design packets require validated company assets before runtime code generation.",
            ],
        }

    @mcp.tool("get_isaac_lab_status")
    def get_isaac_lab_status() -> dict[str, Any]:
        """Inspect the configured Isaac Lab checkout without modifying it."""
        return collect_status()
