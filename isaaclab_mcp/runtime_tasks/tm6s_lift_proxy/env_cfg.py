"""Isaac Lab configuration for the TM6S lift-proxy task.

The supplied TM6S asset has no gripper or end-effector collision geometry.  The
task therefore trains ``tool0`` pose tracking in an elevated workspace.  It is
an executable first-stage control task, not evidence of physical grasping.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

DEFAULT_TM6S_USD_PATH = Path(r"E:\CMC\mod\TMroboot\tm6s.usd")


def resolve_tm6s_usd_path() -> Path:
    """Return the configured TM6S USD path and fail before Kit spawning if absent."""
    path = Path(os.getenv("TM6S_USD_PATH", str(DEFAULT_TM6S_USD_PATH))).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"TM6S USD not found: {path}. Set TM6S_USD_PATH to the source asset.")
    return path


TM6S_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(resolve_tm6s_usd_path()),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_1_joint": 0.0,
            "shoulder_2_joint": -math.pi / 2,
            "elbow_joint": math.pi / 2,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=2.0,
            stiffness=400.0,
            damping=40.0,
        ),
    },
)


@configclass
class TM6SLiftProxyEnvCfg(ReachEnvCfg):
    """Train the TM6S end-effector toward elevated lift-zone pose commands."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = TM6S_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 64
        self.scene.env_spacing = 3.0

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_1_joint",
                "shoulder_2_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            scale=0.25,
            use_default_offset=True,
        )

        self.commands.ee_pose.body_name = "tool0"
        self.commands.ee_pose.resampling_time_range = (3.0, 3.0)
        self.commands.ee_pose.ranges.pos_x = (0.25, 0.55)
        self.commands.ee_pose.ranges.pos_y = (-0.35, 0.10)
        self.commands.ee_pose.ranges.pos_z = (0.55, 1.10)
        self.commands.ee_pose.ranges.roll = (0.0, 0.0)
        self.commands.ee_pose.ranges.pitch = (0.0, 0.0)
        self.commands.ee_pose.ranges.yaw = (-math.pi, math.pi)
        self.commands.ee_pose.debug_vis = False

        for reward_name in (
            "end_effector_position_tracking",
            "end_effector_position_tracking_fine_grained",
            "end_effector_orientation_tracking",
        ):
            reward = getattr(self.rewards, reward_name)
            reward.params["asset_cfg"].body_names = ["tool0"]
        self.rewards.end_effector_orientation_tracking.weight = -0.02

        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 6.0


@configclass
class TM6SLiftProxyEnvCfg_PLAY(TM6SLiftProxyEnvCfg):
    """Small deterministic configuration for interactive evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
