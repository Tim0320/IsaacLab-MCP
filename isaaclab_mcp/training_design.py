"""Structured lifting-task design for Isaac Lab training projects."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AssetRequirement(BaseModel):
    """One simulation asset and the properties it must provide."""

    model_config = ConfigDict(extra="forbid")

    role: str
    required: bool
    reference: str | None
    reference_status: Literal["missing", "local_exists", "local_missing", "external_unverified", "not_required"]
    requirements: list[str]


class SignalTerm(BaseModel):
    """One action, observation, reward, termination, or randomization term."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    weight: float | None = None
    expression: str
    purpose: str


class PpoConfig(BaseModel):
    """Initial RSL-RL PPO settings based on the local Isaac Lab lift template."""

    model_config = ConfigDict(extra="forbid")

    library: Literal["rsl_rl"] = "rsl_rl"
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    save_interval: int = 50
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 128, 64])
    activation: str = "elu"
    learning_rate: float = 1.0e-4
    gamma: float = 0.98
    lam: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.006


class LiftingTrainingDesign(BaseModel):
    """Canonical training design packet written by IsaacLab-MCP."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_name: str
    display_name: str
    task_id: str
    use_case: str
    workflow: Literal["manager_based_single_agent"] = "manager_based_single_agent"
    attachment_mode: str
    simulation: dict[str, Any]
    control: dict[str, Any]
    assets: list[AssetRequirement]
    observations: list[SignalTerm]
    rewards: list[SignalTerm]
    terminations: list[SignalTerm]
    randomization: list[SignalTerm]
    curriculum: list[dict[str, Any]]
    algorithm: PpoConfig
    evaluation_metrics: list[str]
    assumptions: list[str]
    missing_requirements: list[str]
    readiness: dict[str, bool]


def normalize_project_name(value: str) -> str:
    """Convert a user-facing project name into a safe ASCII directory slug."""
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not normalized:
        raise ValueError("project_name must contain at least one ASCII letter or number")
    if len(normalized) > 64:
        raise ValueError("project_name must be 64 characters or fewer after normalization")
    return normalized


def _task_class_name(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("-"))


def asset_reference_status(reference: str | None, required: bool = True) -> str:
    """Classify a local or remote asset reference without opening it."""
    if reference is None or not reference.strip():
        return "missing" if required else "not_required"
    value = reference.strip()
    if value.startswith(("omniverse://", "http://", "https://")):
        return "external_unverified"
    return "local_exists" if Path(value).expanduser().exists() else "local_missing"


def design_lifting_training(
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
) -> LiftingTrainingDesign:
    """Create a validated manager-based RL design for lifting a payload."""
    slug = normalize_project_name(project_name)
    if not use_case.strip():
        raise ValueError("use_case must not be empty")
    if payload_mass_kg <= 0:
        raise ValueError("payload_mass_kg must be greater than zero")
    if safe_working_load_kg is not None and safe_working_load_kg < payload_mass_kg:
        raise ValueError("safe_working_load_kg must be greater than or equal to payload_mass_kg")
    if target_height_m <= initial_height_m:
        raise ValueError("target_height_m must be greater than initial_height_m")
    if max_lift_speed_mps <= 0:
        raise ValueError("max_lift_speed_mps must be greater than zero")
    if episode_length_s <= 0:
        raise ValueError("episode_length_s must be greater than zero")
    if not 1 <= num_envs <= 16384:
        raise ValueError("num_envs must be between 1 and 16384")

    mechanism_status = asset_reference_status(mechanism_asset_path)
    payload_status = asset_reference_status(payload_asset_path)
    cable_required = attachment_mode == "cable"
    cable_status = asset_reference_status(cable_asset_path, required=cable_required)
    environment_status = asset_reference_status(environment_asset_path, required=False)

    missing = []
    if mechanism_status != "local_exists":
        missing.append("mechanism_asset_path must resolve to a validated local articulated USD or URDF")
    if payload_status != "local_exists":
        missing.append("payload_asset_path must resolve to a validated local rigid-body USD")
    if cable_required and cable_status != "local_exists":
        missing.append("cable_asset_path must resolve to a validated cable asset when attachment_mode is cable")
    if not lift_joint_name:
        missing.append("lift_joint_name is required and must match the articulation joint exactly")
    if not end_effector_body_name:
        missing.append("end_effector_body_name is required and must identify the hook or lifting point")
    if safe_working_load_kg is None:
        missing.append("safe_working_load_kg is required before safety-gated training")

    required_asset_ready = (
        mechanism_status == "local_exists"
        and payload_status == "local_exists"
        and (not cable_required or cable_status == "local_exists")
    )
    identifiers_ready = bool(lift_joint_name and end_effector_body_name)
    load_limit_ready = safe_working_load_kg is not None
    asset_ready = required_asset_ready and identifiers_ready and load_limit_ready

    assets = [
        AssetRequirement(
            role="lifting_mechanism",
            required=True,
            reference=mechanism_asset_path,
            reference_status=mechanism_status,
            requirements=[
                "articulated USD or URDF with correct metre scale",
                "rigid bodies with collision, mass, and inertia",
                "one named controllable lift joint with position and velocity limits",
                "stable base or mounting transform",
            ],
        ),
        AssetRequirement(
            role="hook_or_end_effector",
            required=False,
            reference=mechanism_asset_path,
            reference_status=asset_reference_status(mechanism_asset_path, required=False),
            requirements=[
                "named rigid body for the lifting point",
                "collision geometry separated from visual geometry",
                "attachment frame aligned with the payload connection point",
            ],
        ),
        AssetRequirement(
            role="payload",
            required=True,
            reference=payload_asset_path,
            reference_status=payload_status,
            requirements=[
                "rigid-body USD with collision geometry",
                f"verified nominal mass near {payload_mass_kg:g} kg",
                "centre of mass and inertia matching the real payload",
                "named attachment point or contact surface",
            ],
        ),
        AssetRequirement(
            role="cable_or_line",
            required=cable_required,
            reference=cable_asset_path,
            reference_status=cable_status,
            requirements=[
                "validated length, diameter, stiffness, damping, and breaking load",
                "stable attachment to the spool or mechanism and the hook",
                "collision filtering that avoids self-collision instability",
            ],
        ),
        AssetRequirement(
            role="workcell_environment",
            required=False,
            reference=environment_asset_path,
            reference_status=environment_status,
            requirements=[
                "ground or support surface",
                "obstacle collision geometry",
                "restricted-zone markers for unsafe-collision termination",
            ],
        ),
    ]

    observations = [
        SignalTerm(
            name="lift_joint_position",
            kind="observation",
            expression="normalized joint position",
            purpose="tell the policy the current lifting height",
        ),
        SignalTerm(
            name="lift_joint_velocity",
            kind="observation",
            expression="joint velocity [m/s or rad/s according to joint type]",
            purpose="let the policy brake before overshoot",
        ),
        SignalTerm(
            name="payload_to_target",
            kind="observation",
            expression="target_position - payload_position in mechanism root frame [m]",
            purpose="provide direction and remaining distance",
        ),
        SignalTerm(
            name="payload_velocity",
            kind="observation",
            expression="payload linear and angular velocity",
            purpose="measure swing and unstable motion",
        ),
        SignalTerm(
            name="hook_to_payload",
            kind="observation",
            expression="payload_position - hook_position [m]",
            purpose="confirm attachment geometry and detect separation",
        ),
        SignalTerm(
            name="previous_action",
            kind="observation",
            expression="last normalized action",
            purpose="support smooth control",
        ),
    ]

    rewards = [
        SignalTerm(
            name="target_distance",
            kind="reward",
            weight=8.0,
            expression="1 - tanh(distance_to_target / 0.15)",
            purpose="dense guidance toward the requested lift pose",
        ),
        SignalTerm(
            name="upward_progress",
            kind="reward",
            weight=2.0,
            expression="clip(current_height - previous_height, -0.05, 0.05)",
            purpose="reward useful vertical progress early in learning",
        ),
        SignalTerm(
            name="stable_payload",
            kind="reward",
            weight=1.0,
            expression="exp(-payload_angular_speed / 0.5)",
            purpose="reduce swinging and rotation",
        ),
        SignalTerm(
            name="success_bonus",
            kind="reward",
            weight=20.0,
            expression="target_error < 0.03 m and speed < 0.05 m/s for 0.5 s",
            purpose="make stable completion more valuable than passing through the target",
        ),
        SignalTerm(
            name="action_rate",
            kind="penalty",
            weight=-0.01,
            expression="square(action - previous_action)",
            purpose="discourage abrupt control changes",
        ),
        SignalTerm(
            name="energy",
            kind="penalty",
            weight=-0.001,
            expression="absolute(effort * joint_velocity)",
            purpose="discourage wasteful actuation",
        ),
        SignalTerm(
            name="unsafe_collision",
            kind="penalty",
            weight=-20.0,
            expression="contact with restricted bodies or zones",
            purpose="teach safe workcell behaviour",
        ),
        SignalTerm(
            name="overload",
            kind="penalty",
            weight=-10.0,
            expression="estimated load > safe_working_load_kg",
            purpose="keep commands within equipment rating",
        ),
        SignalTerm(
            name="dropped_load",
            kind="penalty",
            weight=-25.0,
            expression="payload below drop height or detached unexpectedly",
            purpose="strongly reject unsafe failure",
        ),
    ]

    terminations = [
        SignalTerm(
            name="success",
            kind="termination",
            expression="target error and speed remain inside tolerance for 0.5 s",
            purpose="finish only after a stable lift",
        ),
        SignalTerm(
            name="dropped_load",
            kind="termination",
            expression="payload falls below the configured minimum height",
            purpose="end unsafe episodes immediately",
        ),
        SignalTerm(
            name="unsafe_collision",
            kind="termination",
            expression="forbidden contact is detected",
            purpose="prevent learning through dangerous collisions",
        ),
        SignalTerm(
            name="overload",
            kind="termination",
            expression="load exceeds equipment rating",
            purpose="enforce the safe working load",
        ),
        SignalTerm(
            name="timeout",
            kind="termination",
            expression=f"episode time >= {episode_length_s:g} s",
            purpose="bound rollout length and penalize stalled behaviour",
        ),
    ]

    randomization = [
        SignalTerm(
            name="payload_mass",
            kind="domain_randomization",
            expression="nominal mass multiplied by uniform(0.8, 1.2)",
            purpose="reduce sensitivity to payload variation",
        ),
        SignalTerm(
            name="joint_friction",
            kind="domain_randomization",
            expression="nominal friction multiplied by uniform(0.8, 1.2)",
            purpose="cover mechanism wear and modelling error",
        ),
        SignalTerm(
            name="motor_strength",
            kind="domain_randomization",
            expression="nominal drive strength multiplied by uniform(0.9, 1.1)",
            purpose="cover actuator variation",
        ),
        SignalTerm(
            name="payload_start_pose",
            kind="domain_randomization",
            expression="small bounded position and yaw offset",
            purpose="prevent memorizing one initial placement",
        ),
        SignalTerm(
            name="sensor_noise",
            kind="domain_randomization",
            expression="bounded zero-mean observation noise",
            purpose="improve robustness to real sensors",
        ),
    ]

    curriculum = [
        {
            "phase": 1,
            "name": "rigid_attachment",
            "goal": "learn vertical motion and braking with the payload attached by a fixed joint",
            "advance_when": "success_rate >= 0.90 and drop_rate <= 0.01",
        },
        {
            "phase": 2,
            "name": "compliant_attachment",
            "goal": "introduce swing, small attachment compliance, and randomized payload mass",
            "advance_when": "success_rate >= 0.85 and unsafe_collision_rate <= 0.01",
        },
        {
            "phase": 3,
            "name": "contact_or_cable",
            "goal": "use validated hook contact or cable dynamics after stable simpler phases",
            "advance_when": "human review confirms stable force and contact behaviour",
        },
    ]

    assumptions = [
        "The first policy controls one lifting axis with normalized velocity commands.",
        "A fixed attachment is used first because cable and hook contact greatly increase training difficulty.",
        "Reward weights are starting values and must be tuned from measured episode metrics.",
        "Simulation success does not establish real-machine safety or complete Sim2Real deployment.",
    ]

    return LiftingTrainingDesign(
        project_name=slug,
        display_name=project_name.strip(),
        task_id=f"Isaac-{_task_class_name(slug)}-v0",
        use_case=use_case.strip(),
        attachment_mode=attachment_mode,
        simulation={
            "num_envs": num_envs,
            "episode_length_s": episode_length_s,
            "physics_dt_s": 0.01,
            "control_decimation": 2,
            "initial_height_m": initial_height_m,
            "target_height_m": target_height_m,
        },
        control={
            "action": "lift_velocity_command",
            "normalized_range": [-1.0, 1.0],
            "max_lift_speed_mps": max_lift_speed_mps,
            "lift_joint_name": lift_joint_name,
            "end_effector_body_name": end_effector_body_name,
            "payload_mass_kg": payload_mass_kg,
            "safe_working_load_kg": safe_working_load_kg,
        },
        assets=assets,
        observations=observations,
        rewards=rewards,
        terminations=terminations,
        randomization=randomization,
        curriculum=curriculum,
        algorithm=PpoConfig(),
        evaluation_metrics=[
            "success_rate",
            "drop_rate",
            "unsafe_collision_rate",
            "overload_rate",
            "mean_time_to_target_s",
            "mean_target_error_m",
            "peak_payload_angular_speed_rad_s",
            "mean_episode_return",
        ],
        assumptions=assumptions,
        missing_requirements=missing,
        readiness={"design_ready": True, "asset_ready": asset_ready, "training_ready": asset_ready},
    )
