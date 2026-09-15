"""MCP tools for the restricted visible Dofbot training runner."""

from __future__ import annotations

from typing import Any, Literal

from isaaclab_mcp.evaluation_jobs import cancel_evaluation, evaluate_training_run, get_evaluation_status
from isaaclab_mcp.training_jobs import cancel_training_run, get_training_run_status, submit_dofbot_training_run


def register_job_tools(mcp: Any) -> None:
    """Register narrow job-control tools; never expose arbitrary shell execution."""

    @mcp.tool("submit_dofbot_training_run")
    def submit_dofbot_training_run_tool(
        environment_contract: dict[str, Any],
        task_id: Literal["Isaac-Grasp-Cube-Dofbot-v0", "Isaac-Lift-Cube-Dofbot-v0"],
        run_name: str,
        num_envs: int = 4,
        max_iterations: int = 40,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Start an allow-listed Dofbot run with visible Kit and a mandatory 15-second RecordVideo clip."""
        return submit_dofbot_training_run(
            environment_contract,
            task_id=task_id,
            run_name=run_name,
            num_envs=num_envs,
            max_iterations=max_iterations,
            seed=seed,
        )

    @mcp.tool("get_training_run_status")
    def get_training_run_status_tool(job_id: str) -> dict[str, Any]:
        """Read a durable training-job state and its produced artifact paths."""
        return get_training_run_status(job_id)

    @mcp.tool("cancel_training_run")
    def cancel_training_run_tool(job_id: str) -> dict[str, Any]:
        """Ask the worker to stop only the launcher process it owns."""
        return cancel_training_run(job_id)

    @mcp.tool("evaluate_training_run")
    def evaluate_training_run_tool(job_id: str, num_envs: int = 4, steps: int = 800) -> dict[str, Any]:
        """Evaluate one completed Dofbot job in visible Kit and save a 15-second MP4 plus EvidenceBundle."""
        return evaluate_training_run(job_id, num_envs=num_envs, steps=steps)

    @mcp.tool("get_evaluation_status")
    def get_evaluation_status_tool(evaluation_id: str) -> dict[str, Any]:
        """Read durable metrics, evidence paths, and PASS or FIX_REQUIRED state for one evaluation."""
        return get_evaluation_status(evaluation_id)

    @mcp.tool("cancel_evaluation")
    def cancel_evaluation_tool(evaluation_id: str) -> dict[str, Any]:
        """Ask the evaluation worker to stop only the Isaac Sim process it owns."""
        return cancel_evaluation(evaluation_id)
