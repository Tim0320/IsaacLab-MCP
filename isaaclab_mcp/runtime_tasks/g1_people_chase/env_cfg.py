"""Manager-based RL configuration for G1 moving-person contact pursuit."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

from . import mdp

PERSON_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/person_target",
    markers={
        "person": sim_utils.UsdFileCfg(
            usd_path=f"{NVIDIA_NUCLEUS_DIR}/Assets/Characters/Reallusion/Worker/Worker.usd",
            scale=(0.01, 0.01, 0.01),
        )
    },
)


@configclass
class G1PeopleChaseEnvCfg(G1FlatEnvCfg):
    """Train G1 to physically contact a moving person proxy without falling."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1024
        self.scene.env_spacing = 8.0
        self.scene.person_proxy = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/PersonContactProxy",
            spawn=sim_utils.CapsuleCfg(
                radius=0.3,
                height=1.2,
                axis="Z",
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=75.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.0),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 1.0)),
        )
        self.scene.person_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/PersonContactProxy",
            update_period=0.0,
            history_length=1,
        )
        self.commands.base_velocity = mdp.MovingPersonCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),
            simple_heading=True,
            debug_vis=True,
            position_success_threshold=2.0,
            orbit_radius_m=3.0,
            target_speed_mps=0.65,
            desired_distance_m=0.0,
            distance_gain=1.0,
            max_forward_speed_mps=1.0,
            max_lateral_speed_mps=0.5,
            heading_control_stiffness=0.5,
            max_yaw_rate_rps=1.0,
            ranges=mdp.MovingPersonCommandCfg.Ranges(
                pos_x=(-3.0, 3.0),
                pos_y=(-3.0, 3.0),
                heading=(-math.pi, math.pi),
            ),
            goal_pose_visualizer_cfg=PERSON_MARKER_CFG,
        )

        # Keep the inherited velocity-tracking terms so the official G1
        # locomotion checkpoint remains useful during chase fine-tuning.
        self.rewards.contact_proximity = RewTerm(
            func=mdp.contact_proximity,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 1.5},
        )
        self.rewards.face_target = RewTerm(
            func=mdp.face_target,
            weight=0.5,
            params={"command_name": "base_velocity", "std": 0.6},
        )
        self.rewards.target_progress = RewTerm(
            func=mdp.target_progress,
            weight=2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.person_contact = RewTerm(
            func=mdp.person_contact_reward,
            weight=50.0,
            params={"sensor_cfg": SceneEntityCfg("person_contact"), "force_threshold_n": 1.0},
        )
        self.rewards.termination_penalty = RewTerm(
            func=base_mdp.is_terminated_term,
            weight=-200.0,
            params={"term_keys": ["fallen_orientation", "fallen_height"]},
        )

        # Torso contact is now a valid way to touch the person. Detect falls from
        # root orientation and height so a successful touch is not mislabeled.
        self.terminations.base_contact = None
        self.terminations.fallen_orientation = DoneTerm(
            func=base_mdp.bad_orientation,
            params={"limit_angle": 0.8},
        )
        self.terminations.fallen_height = DoneTerm(
            func=base_mdp.root_height_below_minimum,
            params={"minimum_height": 0.45},
        )
        self.terminations.person_contact = DoneTerm(
            func=mdp.person_contact_success,
            params={"sensor_cfg": SceneEntityCfg("person_contact"), "force_threshold_n": 1.0},
        )
        self.episode_length_s = 20.0


@configclass
class G1PeopleChaseEnvCfg_PLAY(G1PeopleChaseEnvCfg):
    """Small deterministic configuration for visible evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
