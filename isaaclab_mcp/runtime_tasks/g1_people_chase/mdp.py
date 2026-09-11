"""Command, reward, and termination terms for G1 person-contact pursuit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.commands.commands_cfg import UniformPose2dCommandCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


@configclass
class MovingPersonCommandCfg(UniformPose2dCommandCfg):
    """Configuration for a moving person with synchronized visual and collision representations."""

    class_type: type | str = "isaaclab_mcp.runtime_tasks.g1_people_chase.moving_person_command:MovingPersonCommand"
    orbit_radius_m: float = 3.0
    target_speed_mps: float = 0.65
    desired_distance_m: float = 0.0
    distance_gain: float = 1.0
    closing_speed_mps: float = 0.25
    max_forward_speed_mps: float = 1.0
    max_lateral_speed_mps: float = 0.5
    heading_control_stiffness: float = 0.5
    max_yaw_rate_rps: float = 1.0
    collision_proxy_asset_name: str = "person_proxy"
    collision_proxy_center_height_m: float = 1.0
    contact_sensor_name: str = "person_contact"
    contact_force_threshold_n: float = 1.0


def target_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Planar distance from G1 to the moving person target."""
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    delta = command.pos_command_w[:, :2] - robot.data.root_pos_w.torch[:, :2]
    return torch.linalg.vector_norm(delta, dim=1)


def contact_proximity(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Provide a dense approach signal that grows as G1 closes on the person."""
    distance = target_distance(env, command_name, asset_cfg)
    return 1.0 - torch.tanh(distance / std)


def target_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward G1 velocity projected toward the person."""
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    delta = command.pos_command_w[:, :2] - robot.data.root_pos_w.torch[:, :2]
    direction = delta / torch.clamp(torch.linalg.vector_norm(delta, dim=1, keepdim=True), min=1.0e-6)
    progress = torch.sum(robot.data.root_lin_vel_w.torch[:, :2] * direction, dim=1)
    return torch.clamp(progress, min=-1.0, max=1.0)


def face_target(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward heading alignment with the current person position."""
    heading_error = env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-torch.square(heading_error / std))


def person_contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("person_contact"),
) -> torch.Tensor:
    """Return the strongest force measured by the ground-clear person proxy."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_magnitudes = torch.linalg.vector_norm(contact_sensor.data.net_forces_w.torch, dim=-1)
    return force_magnitudes.flatten(start_dim=1).amax(dim=1)


def person_contact_reward(
    env: ManagerBasedRLEnv,
    force_threshold_n: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("person_contact"),
) -> torch.Tensor:
    """Reward only verified physical contact with the moving person proxy."""
    return (person_contact_force(env, sensor_cfg) >= force_threshold_n).float()


def person_contact_success(
    env: ManagerBasedRLEnv,
    force_threshold_n: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("person_contact"),
) -> torch.Tensor:
    """End an episode successfully only after verified physical person contact."""
    return person_contact_force(env, sensor_cfg) >= force_threshold_n
