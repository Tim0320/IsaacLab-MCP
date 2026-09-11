"""Design a staged Unitree G1 people-interaction RL program."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class PeopleSkillPlan(BaseModel):
    """One independently trainable skill in the people-interaction program."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["run", "pick", "chase"]
    task_id: str
    asset_config: str
    learning_route: str
    status: Literal["rl_ready", "imitation_ready", "runtime_missing"]
    actions: list[str]
    observations: list[str]
    rewards: list[str]
    success_metrics: list[str]
    blockers: list[str]


class PeopleRLProgram(BaseModel):
    """Capability and curriculum packet for a humanoid skill program."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    robot: str
    runtime_baseline: str
    architecture: str
    autonomy_loop: list[str]
    curriculum: list[dict[str, str]]
    skills: list[PeopleSkillPlan]
    safety_rules: list[str]
    readiness: dict[str, bool]


def design_people_rl_program() -> PeopleRLProgram:
    """Describe what the local Isaac Lab 3.0.0 runtime can train today.

    The three behaviors deliberately remain separate skills.  Their action spaces
    and learning routes differ enough that treating them as one PPO task would
    hide missing rewards and make failures difficult to diagnose.
    """

    skills = [
        PeopleSkillPlan(
            name="run",
            task_id="Isaac-Velocity-Flat-G1-v0",
            asset_config="isaaclab_assets.G1_MINIMAL_CFG",
            learning_route="RSL-RL PPO",
            status="rl_ready",
            actions=["normalized G1 joint-position targets"],
            observations=[
                "base linear and angular velocity",
                "projected gravity",
                "commanded planar velocity",
                "joint position and velocity",
                "previous action",
            ],
            rewards=[
                "track planar velocity",
                "track yaw velocity",
                "upright posture and foot timing",
                "penalize torque, acceleration, sliding, and abrupt actions",
            ],
            success_metrics=["velocity tracking error", "fall rate", "episode length"],
            blockers=[],
        ),
        PeopleSkillPlan(
            name="pick",
            task_id="Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
            asset_config="isaaclab_assets.G1_29DOF_CFG",
            learning_route="teleoperation demonstrations plus Robomimic BC-RNN",
            status="imitation_ready",
            actions=["two wrist poses", "14 hand-joint targets", "lower-body locomotion command"],
            observations=[
                "robot joints and link poses",
                "left and right wrist poses",
                "hand-joint state",
                "object pose",
                "previous action",
            ],
            rewards=[],
            success_metrics=["object grasped", "object lifted", "object placed without dropping"],
            blockers=[
                "the Isaac Lab task has rewards=None",
                "the task has no RSL-RL runner configuration",
                "demonstrations or a new reward-based pick task are required before autonomous RL training",
            ],
        ),
        PeopleSkillPlan(
            name="chase",
            task_id="Isaac-Chase-Person-G1-v0",
            asset_config="G1 locomotion agent plus a Reallusion Worker visual and synchronized collision capsule",
            learning_route="project-owned RSL-RL PPO moving-target task",
            status="rl_ready",
            actions=["normalized G1 joint-position targets"],
            observations=[
                "target XY position and heading error in the G1 yaw frame",
                "G1 proprioception",
            ],
            rewards=[
                "positive progress toward the target",
                "close the remaining distance to the target",
                "face the target",
                "reward verified physical contact and penalize falls",
            ],
            success_metrics=["person-contact success rate", "contact force", "time to contact", "fall rate"],
            blockers=[],
        ),
    ]

    return PeopleRLProgram(
        robot="Unitree G1 humanoid",
        runtime_baseline="Isaac Lab 3.0.0 with Isaac Sim 6.0.1",
        architecture="hierarchical skill policies with an explicit high-level skill selector",
        autonomy_loop=[
            "observe robot, object, and target state",
            "select run, pick, or chase skill from the current goal",
            "execute the selected low-level policy",
            "apply fall detection and require a measured person-contact event",
            "score the outcome and retry only the failed skill stage",
        ],
        curriculum=[
            {
                "phase": "1",
                "skill": "run",
                "advance_when": "velocity tracking is stable and fall rate is below 5%",
            },
            {
                "phase": "2",
                "skill": "pick",
                "advance_when": "grasp-and-lift success is at least 80% in fixed-base evaluation",
            },
            {
                "phase": "3",
                "skill": "chase",
                "advance_when": "physical person-contact success is at least 80% in deterministic evaluation",
            },
            {
                "phase": "4",
                "skill": "integration",
                "advance_when": "the selector completes run-to-object, pick, and person-contact sequences without a fall",
            },
        ],
        skills=skills,
        safety_rules=[
            "the chase target exists only inside simulation",
            "terminate falls from root orientation or height instead of torso contact",
            "count success only when the dedicated person-contact sensor reaches 1 N",
            "keep the moving collision capsule synchronized with the visible person",
            "keep manipulation stationary until the fixed-base pick skill is reliable",
        ],
        readiness={"run": True, "pick": False, "chase": True, "integrated_autonomy": False},
    )
