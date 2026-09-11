"""Isaac Lab manager-based cube-lift task for the default Dofbot asset."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from . import mdp as dofbot_mdp

DOFBOT_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots/Yahboom/Dofbot/dofbot.usd"

# Center policy exploration on the verified cube-centered grasp pose.  The grasp
# pre-training task starts here with open fingers; the lift task keeps the same
# action meaning but permits a wider motion range.
DOFBOT_ARM_ACTION_OFFSET = {
    "joint1": -1.50,
    "joint2": -0.040,
    "joint3": -1.570,
    "joint4": -0.939,
    "Wrist_Twist_RevoluteJoint": 0.080,
}
DOFBOT_ARM_ACTION_SCALE = {
    # Preserve the grasping geometry while giving the two pitch joints enough
    # range to raise the payload.  Wide base/wrist exploration mostly causes
    # avoidable drops on this small robot.
    "joint1": 0.05,
    "joint2": 0.60,
    "joint3": 0.05,
    "joint4": 0.50,
    "Wrist_Twist_RevoluteJoint": 0.20,
}
DOFBOT_GRASP_RESET_JOINT_POS = {
    "joint1": -1.50,
    "joint2": 0.278,
    "joint3": -1.495,
    "joint4": -1.018,
    "Wrist_Twist_RevoluteJoint": -1.059,
}


DOFBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=DOFBOT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="min",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint[1-2]": 0.0,
            "joint[3-4]": -math.pi / 3,
            "Wrist_Twist_RevoluteJoint": 0.0,
            "Finger_.*_01_RevoluteJoint": 0.0,
            "Finger_.*_0[2-3]_RevoluteJoint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-4]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Twist_RevoluteJoint"],
            effort_limit_sim=0.1,
            velocity_limit_sim=5.0,
            stiffness=1000.0,
            damping=10.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Finger_.*_01_RevoluteJoint"],
            effort_limit_sim=0.5,
            velocity_limit_sim=5.0,
            stiffness=6000.0,
            damping=1000.0,
        ),
        "passive_fingers": ImplicitActuatorCfg(
            joint_names_expr=["Finger_.*_0[2-3]_RevoluteJoint"],
            effort_limit_sim=0.0,
            velocity_limit_sim=5.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


@configclass
class DofbotCubeLiftEnvCfg(LiftEnvCfg):
    """Lift a small cube with Dofbot's articulated two-finger gripper."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = DOFBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Both curriculum stages begin from the same collision-free approach
        # pose.  This keeps the grasp policy's action meaning intact when its
        # checkpoint is transferred to lift training.
        self.scene.robot.init_state.joint_pos = {
            **DOFBOT_GRASP_RESET_JOINT_POS,
            "Finger_.*_01_RevoluteJoint": 0.0,
            "Finger_.*_0[2-3]_RevoluteJoint": 0.0,
        }
        self.scene.num_envs = 64
        self.scene.env_spacing = 1.5

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint[1-4]", "Wrist_Twist_RevoluteJoint"],
            scale=DOFBOT_ARM_ACTION_SCALE,
            offset=DOFBOT_ARM_ACTION_OFFSET,
            use_default_offset=False,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Finger_Left_01_RevoluteJoint", "Finger_Right_01_RevoluteJoint"],
            open_command_expr={
                "Finger_Left_01_RevoluteJoint": 0.0,
                "Finger_Right_01_RevoluteJoint": 0.0,
            },
            close_command_expr={
                "Finger_Left_01_RevoluteJoint": -0.50,
                "Finger_Right_01_RevoluteJoint": 0.50,
            },
        )

        self.commands.object_pose.body_name = "link4"
        self.commands.object_pose.ranges.pos_x = (0.12, 0.22)
        self.commands.object_pose.ranges.pos_y = (-0.08, 0.08)
        self.commands.object_pose.ranges.pos_z = (0.14, 0.28)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (-math.pi, math.pi)
        self.commands.object_pose.debug_vis = False

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.16, 0.0, 0.065), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.4, 0.4, 0.4),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max",
                    restitution_combine_mode="min",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=100.0,
                    max_linear_velocity=10.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )
        self.scene.pedestal = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pedestal",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.16, 0.0, 0.02)),
            spawn=sim_utils.CuboidCfg(
                size=(0.012, 0.012, 0.04),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25)),
            ),
        )

        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.002, 0.002),
            "y": (-0.002, 0.002),
            "z": (0.0, 0.0),
        }
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link4",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.132)),
                )
            ],
        )

        fingertip_cfg = SceneEntityCfg("robot", body_names=["Finger_Left_03", "Finger_Right_03"])
        self.observations.policy.object_position = ObsTerm(
            func=dofbot_mdp.object_to_fingertip_midpoint,
            params={
                "grasp_height_offset": 0.035,
                "robot_cfg": fingertip_cfg,
            },
        )
        self.observations.policy.fingertip_gap = ObsTerm(
            func=dofbot_mdp.fingertip_gap,
            params={"robot_cfg": fingertip_cfg},
        )
        self.rewards.reaching_object = RewTerm(
            func=dofbot_mdp.fingertip_object_distance,
            weight=4.0,
            params={
                "std": 0.25,
                "grasp_height_offset": 0.035,
                "robot_cfg": fingertip_cfg,
            },
        )
        self.rewards.gripper_action_timing = RewTerm(
            func=dofbot_mdp.gripper_action_timing,
            weight=1.0,
            params={
                "distance_threshold": 0.06,
                "grasp_height_offset": 0.035,
                "robot_cfg": fingertip_cfg,
            },
        )
        self.rewards.stable_grasp = RewTerm(
            func=dofbot_mdp.stable_grasp,
            weight=4.0,
            params={
                "distance_threshold": 0.06,
                "minimum_gap": 0.036,
                "maximum_gap": 0.055,
                "minimum_object_height": 0.045,
                "grasp_height_offset": 0.035,
                "robot_cfg": fingertip_cfg,
            },
        )
        self.rewards.grasped_height_progress = RewTerm(
            func=dofbot_mdp.grasped_height_progress,
            weight=12.0,
            params={
                "base_height": 0.065,
                "target_height": 0.105,
                "distance_threshold": 0.06,
                "minimum_gap": 0.036,
                "maximum_gap": 0.055,
                "grasp_height_offset": 0.035,
                "robot_cfg": fingertip_cfg,
            },
        )
        self.rewards.grasped_lift_pose_progress = RewTerm(
            func=dofbot_mdp.grasped_lift_pose_progress,
            weight=8.0,
            params={
                # Joint-space result of the visibly verified lift motion.
                "target_joint_positions": [0.435, -1.287],
                "std": 0.5,
                "distance_threshold": 0.06,
                "minimum_gap": 0.036,
                "maximum_gap": 0.055,
                "grasp_height_offset": 0.035,
                "robot_cfg": SceneEntityCfg("robot", joint_names=["joint2", "joint4"]),
                "fingertip_robot_cfg": fingertip_cfg,
            },
        )
        self.rewards.object_fallen = RewTerm(
            func=dofbot_mdp.object_fallen,
            weight=-5.0,
            params={"minimum_height": 0.045},
        )
        self.rewards.gripper_switch = RewTerm(
            func=dofbot_mdp.gripper_switch,
            weight=-0.25,
        )
        # A miss only resets that vectorized environment.  PPO keeps running and
        # immediately receives a fresh object for the next grasp attempt.
        self.terminations.object_dropping.params["minimum_height"] = 0.045
        self.rewards.lifting_object.params["minimal_height"] = 0.075
        self.rewards.object_goal_tracking.params["minimal_height"] = 0.075
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = 0.075

        self.observations.policy.enable_corruption = False
        # Long enough for open -> align -> close -> reopen -> retry behavior.
        self.episode_length_s = 20.0


@configclass
class DofbotCubeGraspEnvCfg(DofbotCubeLiftEnvCfg):
    """Stage-one curriculum: learn repeatable grasping before lifting."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.scale = {
            "joint1": 0.05,
            "joint2": 0.08,
            "joint3": 0.05,
            "joint4": 0.08,
            "Wrist_Twist_RevoluteJoint": 0.20,
        }
        self.rewards.reaching_object.weight = 6.0
        self.rewards.gripper_action_timing.weight = 2.0
        self.rewards.stable_grasp.weight = 12.0
        self.rewards.grasped_height_progress.weight = 0.0
        self.rewards.grasped_lift_pose_progress.weight = 0.0
        self.rewards.lifting_object.weight = 0.0
        self.rewards.object_goal_tracking.weight = 0.0
        self.rewards.object_goal_tracking_fine_grained.weight = 0.0
        self.episode_length_s = 8.0


@configclass
class DofbotCubeLiftEnvCfg_PLAY(DofbotCubeLiftEnvCfg):
    """Small deterministic configuration for interactive evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
