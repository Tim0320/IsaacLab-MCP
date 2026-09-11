"""MCP tools for designing and writing lifting training projects."""

from __future__ import annotations

from typing import Any, Literal

from isaaclab_mcp.people_rl import design_people_rl_program
from isaaclab_mcp.training_design import design_lifting_training
from isaaclab_mcp.training_projects import create_training_project, validate_training_project


def register_training_tools(mcp: Any) -> None:
    """Register non-interactive training authoring tools."""

    @mcp.tool("design_lifting_training")
    def design_lifting_training_tool(
        project_name: str,
        use_case: str,
        mechanism_asset_path: str | None = None,
        payload_asset_path: str | None = None,
        cable_asset_path: str | None = None,
        environment_asset_path: str | None = None,
        lift_joint_name: str | None = None,
        end_effector_body_name: str | None = None,
        payload_mass_kg: float = 1.0,
        safe_working_load_kg: float | None = None,
        initial_height_m: float = 0.05,
        target_height_m: float = 0.5,
        max_lift_speed_mps: float = 0.25,
        episode_length_s: float = 10.0,
        num_envs: int = 512,
        attachment_mode: Literal["fixed_joint_curriculum", "contact_hook", "cable"] = "fixed_joint_curriculum",
    ) -> dict[str, Any]:
        """Design assets, RL signals, curriculum, PPO settings, metrics, and readiness for a lifting task."""
        design = design_lifting_training(
            project_name=project_name,
            use_case=use_case,
            mechanism_asset_path=mechanism_asset_path,
            payload_asset_path=payload_asset_path,
            cable_asset_path=cable_asset_path,
            environment_asset_path=environment_asset_path,
            lift_joint_name=lift_joint_name,
            end_effector_body_name=end_effector_body_name,
            payload_mass_kg=payload_mass_kg,
            safe_working_load_kg=safe_working_load_kg,
            initial_height_m=initial_height_m,
            target_height_m=target_height_m,
            max_lift_speed_mps=max_lift_speed_mps,
            episode_length_s=episode_length_s,
            num_envs=num_envs,
            attachment_mode=attachment_mode,
        )
        return design.model_dump(mode="json")

    @mcp.tool("design_people_rl_program")
    def design_people_rl_program_tool() -> dict[str, Any]:
        """Report the runnable, imitation-only, and missing G1 people-interaction skills."""
        return design_people_rl_program().model_dump(mode="json")

    @mcp.tool("create_lifting_training_project")
    def create_lifting_training_project(
        design: dict[str, Any], preview: bool = True, overwrite: bool = False
    ) -> dict[str, Any]:
        """Preview or create a training design packet under the configured training root."""
        return create_training_project(design, preview=preview, overwrite=overwrite)

    @mcp.tool("validate_lifting_training_project")
    def validate_lifting_training_project(project_name: str) -> dict[str, Any]:
        """Validate a generated project's files, schema, asset references, and training readiness."""
        return validate_training_project(project_name)
