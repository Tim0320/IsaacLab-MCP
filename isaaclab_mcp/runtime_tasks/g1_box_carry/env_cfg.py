"""Manager-based RL configuration for running while carrying a 15 kg box."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

from . import mdp

PALMS = SceneEntityCfg(
    "robot",
    body_names=["left_palm_link", "right_palm_link"],
    preserve_order=True,
)
BOX = SceneEntityCfg("box")
ANCHORS = (mdp.LEFT_ANCHOR_B, mdp.RIGHT_ANCHOR_B)


@configclass
class G1BoxCarryEnvCfg(G1FlatEnvCfg):
    """Train G1 to retain a bilateral 15 kg payload while moving forward."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 512
        self.scene.env_spacing = 4.0
        self.scene.robot = self.scene.robot.copy()
        self.scene.robot.init_state.joint_pos.update(
            {
                "left_shoulder_pitch_joint": -1.30,
                "right_shoulder_pitch_joint": -1.30,
                ".*_elbow_pitch_joint": 2.00,
                "left_five_joint": -1.0,
                "left_three_joint": -1.0,
                "left_six_joint": -1.0,
                "left_four_joint": -1.0,
                "right_five_joint": 1.0,
                "right_three_joint": 1.0,
                "right_six_joint": 1.0,
                "right_four_joint": 1.0,
            }
        )
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CarryBox",
            spawn=sim_utils.CuboidCfg(
                size=(0.28, 0.34, 0.24),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.35, 0.05),
                    metallic=0.1,
                    roughness=0.45,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.345, 0.0, 0.813)),
        )

        # Keep all 37 policy outputs checkpoint-compatible, but prevent the old
        # locomotion policy's arm swing from immediately tearing open the grasp.
        self.actions.joint_pos.scale = {
            ".*_hip_.*_joint": 0.5,
            "torso_joint": 0.5,
            ".*_knee_joint": 0.5,
            ".*_ankle_.*_joint": 0.5,
            ".*_shoulder_.*_joint": 0.03,
            ".*_elbow_.*_joint": 0.03,
            ".*_(five|three|six|four|zero|one|two)_joint": 0.02,
        }

        # Deterministic upright resets keep the dynamic box aligned with both palms.
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_box = EventTerm(
            func=mdp.reset_box_to_carry_pose,
            mode="reset",
            params={
                "box_position_from_env_origin_m": (0.345, 0.0, 0.813),
                "box_cfg": BOX,
            },
        )
        self.events.bimanual_grasp = EventTerm(
            func=mdp.apply_bimanual_grasp_forces,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "stiffness_npm": 1800.0,
                "damping_nspm": 120.0,
                "rotational_stiffness_nmprad": 45.0,
                "rotational_damping_nmsprad": 8.0,
                "max_force_n": 350.0,
                "break_distance_m": 0.40,
                "anchor_offsets_b": ANCHORS,
                "robot_cfg": PALMS,
                "box_cfg": BOX,
            },
        )

        # Start below sprint speed: first retain the 15 kg payload, then learn to accelerate.
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.rel_standing_envs = 0.50
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.35)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.rewards.bimanual_hold = RewTerm(
            func=mdp.bimanual_hold_reward,
            weight=10.0,
            params={"std": 0.08, "robot_cfg": PALMS, "box_cfg": BOX, "anchor_offsets_b": ANCHORS},
        )
        self.rewards.load_balance = RewTerm(
            func=mdp.load_balance_reward,
            weight=1.0,
            params={"std": 0.04, "robot_cfg": PALMS, "box_cfg": BOX, "anchor_offsets_b": ANCHORS},
        )
        self.rewards.box_height = RewTerm(
            func=mdp.box_height_reward,
            weight=2.0,
            params={"target_height_from_root_m": 0.052, "std": 0.10, "robot_cfg": SceneEntityCfg("robot"), "box_cfg": BOX},
        )
        self.rewards.box_upright = RewTerm(func=mdp.box_upright_reward, weight=1.0, params={"box_cfg": BOX})
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.joint_deviation_arms.weight = -0.25
        self.rewards.joint_deviation_fingers.weight = -0.10
        self.rewards.termination_penalty.weight = -100.0

        self.terminations.grasp_lost = DoneTerm(
            func=mdp.bimanual_grasp_lost,
            params={
                "break_distance_m": 0.40,
                "minimum_box_height_m": 0.30,
                "grace_steps": 50,
                "robot_cfg": PALMS,
                "box_cfg": BOX,
                "anchor_offsets_b": ANCHORS,
            },
        )
        self.episode_length_s = 20.0


@configclass
class G1BoxCarryRunEnvCfg(G1BoxCarryEnvCfg):
    """Second curriculum stage: accelerate only after bilateral carry is learned."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.80)
        self.rewards.track_lin_vel_xy_exp.weight = 4.0


@configclass
class G1BoxCarryEnvCfg_PLAY(G1BoxCarryEnvCfg):
    """One-environment deterministic hold-curriculum evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (3.2, 3.2, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.75)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.20)


@configclass
class G1BoxCarryRunEnvCfg_PLAY(G1BoxCarryRunEnvCfg):
    """One-environment deterministic visible evaluation at the run-stage speed."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (3.2, 3.2, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.75)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.60, 0.60)
