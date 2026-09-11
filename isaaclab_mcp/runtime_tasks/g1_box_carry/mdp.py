"""Physics coupling, rewards, and termination terms for G1 bimanual box carry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_conjugate, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


LEFT_ANCHOR_B = (0.0, 0.17, 0.0)
RIGHT_ANCHOR_B = (0.0, -0.17, 0.0)


def _resolved_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    return env_ids.to(device=env.device, dtype=torch.int32)


def _anchor_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    anchor_offsets_b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return palm and box-anchor positions/velocities for both hands."""
    robot: Articulation = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]
    offsets_b = torch.tensor(anchor_offsets_b, device=env.device, dtype=torch.float32)
    offsets_b = offsets_b.unsqueeze(0).expand(env.num_envs, -1, -1)
    box_quat = box.data.root_quat_w.torch.unsqueeze(1).expand(-1, 2, -1)
    offsets_w = quat_apply(box_quat, offsets_b)
    anchor_pos_w = box.data.root_pos_w.torch.unsqueeze(1) + offsets_w
    anchor_vel_w = box.data.root_lin_vel_w.torch.unsqueeze(1) + torch.cross(
        box.data.root_ang_vel_w.torch.unsqueeze(1).expand_as(offsets_w), offsets_w, dim=-1
    )
    palm_pos_w = robot.data.body_pos_w.torch[:, robot_cfg.body_ids]
    palm_vel_w = robot.data.body_lin_vel_w.torch[:, robot_cfg.body_ids]
    return palm_pos_w, palm_vel_w, anchor_pos_w, anchor_vel_w, offsets_w


def reset_box_to_carry_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    box_position_from_env_origin_m: tuple[float, float, float],
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> None:
    """Reset the box upright between the palms after a deterministic robot reset."""
    env_ids = _resolved_env_ids(env, env_ids)
    box: RigidObject = env.scene[box_cfg.name]
    position = torch.tensor(box_position_from_env_origin_m, device=env.device).expand(len(env_ids), -1)
    box_pos = env.scene.env_origins[env_ids] + position
    box_quat = torch.zeros((len(env_ids), 4), device=env.device)
    box_quat[:, 3] = 1.0
    box_pose = torch.cat((box_pos, box_quat), dim=-1)
    box_velocity = torch.zeros((len(env_ids), 6), device=env.device)
    box.write_root_pose_to_sim_index(root_pose=box_pose, env_ids=env_ids)
    box.write_root_velocity_to_sim_index(root_velocity=box_velocity, env_ids=env_ids)
    box.permanent_wrench_composer.reset(env_ids=env_ids)


def apply_bimanual_grasp_forces(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    stiffness_npm: float,
    damping_nspm: float,
    rotational_stiffness_nmprad: float,
    rotational_damping_nmsprad: float,
    max_force_n: float,
    break_distance_m: float,
    anchor_offsets_b: tuple[tuple[float, float, float], tuple[float, float, float]],
    robot_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
) -> None:
    """Apply equal-and-opposite compliant forces from both palms to a dynamic box."""
    selected = _resolved_env_ids(env, env_ids)
    robot: Articulation = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]
    palm_pos, palm_vel, anchor_pos, anchor_vel, offsets_w = _anchor_state(
        env, robot_cfg, box_cfg, anchor_offsets_b
    )
    palm_pos = palm_pos[selected]
    palm_vel = palm_vel[selected]
    anchor_pos = anchor_pos[selected]
    anchor_vel = anchor_vel[selected]
    offsets_w = offsets_w[selected]

    displacement = palm_pos - anchor_pos
    distances = torch.linalg.vector_norm(displacement, dim=-1)
    both_attached = (distances < break_distance_m).all(dim=1, keepdim=True).unsqueeze(-1)
    forces_on_box = stiffness_npm * displacement + damping_nspm * (palm_vel - anchor_vel)
    force_norm = torch.clamp(torch.linalg.vector_norm(forces_on_box, dim=-1, keepdim=True), min=1.0e-6)
    forces_on_box *= torch.clamp(max_force_n / force_norm, max=1.0)
    forces_on_box *= both_attached

    box_quat = box.data.root_quat_w.torch[selected]
    robot_quat = robot.data.root_quat_w.torch[selected]
    desired_quat = yaw_quat(robot_quat)
    orientation_error = axis_angle_from_quat(quat_mul(desired_quat, quat_conjugate(box_quat)))
    relative_ang_vel = box.data.root_ang_vel_w.torch[selected] - robot.data.root_ang_vel_w.torch[selected]
    orientation_torque = (
        rotational_stiffness_nmprad * orientation_error
        - rotational_damping_nmsprad * relative_ang_vel
    )
    orientation_torque *= both_attached.squeeze(1)

    box_force = forces_on_box.sum(dim=1, keepdim=True)
    box_torque = (torch.cross(offsets_w, forces_on_box, dim=-1).sum(dim=1) + orientation_torque).unsqueeze(1)
    palm_torques = -0.5 * orientation_torque.unsqueeze(1).expand(-1, 2, -1)
    palm_body_ids = torch.tensor(robot_cfg.body_ids, device=env.device, dtype=torch.int32)
    box_body_ids = torch.tensor([0], device=env.device, dtype=torch.int32)

    robot.permanent_wrench_composer.set_forces_and_torques_index(
        forces=-forces_on_box,
        torques=palm_torques,
        body_ids=palm_body_ids,
        env_ids=selected,
        is_global=True,
    )
    box.permanent_wrench_composer.set_forces_and_torques_index(
        forces=box_force,
        torques=box_torque,
        body_ids=box_body_ids,
        env_ids=selected,
        is_global=True,
    )


def bimanual_anchor_error(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    anchor_offsets_b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> torch.Tensor:
    palm_pos, _, anchor_pos, _, _ = _anchor_state(env, robot_cfg, box_cfg, anchor_offsets_b)
    return torch.linalg.vector_norm(palm_pos - anchor_pos, dim=-1)


def bimanual_hold_reward(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    anchor_offsets_b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> torch.Tensor:
    errors = bimanual_anchor_error(env, robot_cfg, box_cfg, anchor_offsets_b)
    return torch.exp(-torch.square(errors / std)).prod(dim=1)


def load_balance_reward(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    anchor_offsets_b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> torch.Tensor:
    errors = bimanual_anchor_error(env, robot_cfg, box_cfg, anchor_offsets_b)
    return torch.exp(-torch.abs(errors[:, 0] - errors[:, 1]) / std)


def box_height_reward(
    env: ManagerBasedRLEnv,
    target_height_from_root_m: float,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]
    relative_height = box.data.root_pos_w.torch[:, 2] - robot.data.root_pos_w.torch[:, 2]
    return torch.exp(-torch.square((relative_height - target_height_from_root_m) / std))


def box_upright_reward(
    env: ManagerBasedRLEnv,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    box: RigidObject = env.scene[box_cfg.name]
    local_up = torch.zeros((env.num_envs, 3), device=env.device)
    local_up[:, 2] = 1.0
    world_up = quat_apply(box.data.root_quat_w.torch, local_up)
    return torch.clamp(world_up[:, 2], min=0.0, max=1.0)


def bimanual_grasp_lost(
    env: ManagerBasedRLEnv,
    break_distance_m: float,
    minimum_box_height_m: float,
    grace_steps: int,
    robot_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    anchor_offsets_b: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> torch.Tensor:
    box: RigidObject = env.scene[box_cfg.name]
    errors = bimanual_anchor_error(env, robot_cfg, box_cfg, anchor_offsets_b)
    separated = (errors > break_distance_m).any(dim=1)
    too_low = box.data.root_pos_w.torch[:, 2] < minimum_box_height_m
    return (env.episode_length_buf > grace_steps) & (separated | too_low)
