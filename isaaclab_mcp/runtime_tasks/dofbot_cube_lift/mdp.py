"""Dofbot-specific observation and reward functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def object_to_fingertip_midpoint(
    env: ManagerBasedRLEnv,
    grasp_height_offset: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Finger_Left_03", "Finger_Right_03"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Vector from the distal-finger midpoint to the desired grasp point."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    fingertip_positions = robot.data.body_pos_w.torch[:, robot_cfg.body_ids]
    fingertip_midpoint = fingertip_positions.mean(dim=1)
    grasp_point = object_asset.data.root_pos_w.torch.clone()
    grasp_point[:, 2] += grasp_height_offset
    return quat_apply_inverse(robot.data.root_quat_w.torch, grasp_point - fingertip_midpoint)


def fingertip_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    grasp_height_offset: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Finger_Left_03", "Finger_Right_03"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the real distal-finger midpoint approaching the grasp point."""
    displacement = object_to_fingertip_midpoint(
        env,
        grasp_height_offset=grasp_height_offset,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    )
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    return 1.0 - torch.tanh(distance / std)


def gripper_action_timing(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    grasp_height_offset: float,
    gripper_action_index: int = -1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Finger_Left_03", "Finger_Right_03"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward closing near the object and penalize closing in empty space."""
    displacement = object_to_fingertip_midpoint(
        env,
        grasp_height_offset=grasp_height_offset,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    )
    near_object = torch.linalg.vector_norm(displacement, dim=-1) < distance_threshold
    closing = env.action_manager.action[:, gripper_action_index] < 0.0
    return torch.where(
        closing,
        torch.where(near_object, 1.0, -1.0),
        0.0,
    )


def fingertip_gap(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Finger_Left_03", "Finger_Right_03"]),
) -> torch.Tensor:
    """Distance between the two distal finger bodies."""
    robot: Articulation = env.scene[robot_cfg.name]
    fingertip_positions = robot.data.body_pos_w.torch[:, robot_cfg.body_ids]
    return torch.linalg.vector_norm(fingertip_positions[:, 0] - fingertip_positions[:, 1], dim=-1).unsqueeze(-1)


def stable_grasp(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    minimum_gap: float,
    maximum_gap: float,
    minimum_object_height: float,
    grasp_height_offset: float,
    gripper_action_index: int = -1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Finger_Left_03", "Finger_Right_03"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward a sustained, closed grasp around the object instead of an empty close."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    fingertip_positions = robot.data.body_pos_w.torch[:, robot_cfg.body_ids]
    gap = torch.linalg.vector_norm(fingertip_positions[:, 0] - fingertip_positions[:, 1], dim=-1)
    displacement = object_to_fingertip_midpoint(
        env,
        grasp_height_offset=grasp_height_offset,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    )
    close_to_object = torch.linalg.vector_norm(displacement, dim=-1) < distance_threshold
    object_supported = object_asset.data.root_pos_w.torch[:, 2] > minimum_object_height
    object_between_fingers = (gap > minimum_gap) & (gap < maximum_gap)
    closing = env.action_manager.action[:, gripper_action_index] < 0.0
    return (closing & close_to_object & object_supported & object_between_fingers).float()


def grasped_height_progress(
    env: ManagerBasedRLEnv,
    base_height: float,
    target_height: float,
    distance_threshold: float,
    minimum_gap: float,
    maximum_gap: float,
    grasp_height_offset: float,
    gripper_action_index: int = -1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Finger_Left_03", "Finger_Right_03"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Give dense lift progress only while the object remains inside the closed gripper."""
    grasp_quality = stable_grasp(
        env,
        distance_threshold=distance_threshold,
        minimum_gap=minimum_gap,
        maximum_gap=maximum_gap,
        minimum_object_height=base_height - 0.02,
        grasp_height_offset=grasp_height_offset,
        gripper_action_index=gripper_action_index,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    )
    object_asset: RigidObject = env.scene[object_cfg.name]
    height_progress = torch.clamp(
        (object_asset.data.root_pos_w.torch[:, 2] - base_height) / (target_height - base_height),
        min=0.0,
        max=1.0,
    )
    return grasp_quality * height_progress


def grasped_lift_pose_progress(
    env: ManagerBasedRLEnv,
    target_joint_positions: list[float],
    std: float,
    distance_threshold: float,
    minimum_gap: float,
    maximum_gap: float,
    grasp_height_offset: float,
    gripper_action_index: int = -1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint2", "joint4"]),
    fingertip_robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Finger_Left_03", "Finger_Right_03"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Guide a held object toward a known-safe lift pose before sparse lift rewards fire."""
    grasp_quality = stable_grasp(
        env,
        distance_threshold=distance_threshold,
        minimum_gap=minimum_gap,
        maximum_gap=maximum_gap,
        minimum_object_height=0.045,
        grasp_height_offset=grasp_height_offset,
        gripper_action_index=gripper_action_index,
        robot_cfg=fingertip_robot_cfg,
        object_cfg=object_cfg,
    )
    robot: Articulation = env.scene[robot_cfg.name]
    joint_positions = robot.data.joint_pos.torch[:, robot_cfg.joint_ids]
    target = torch.tensor(
        target_joint_positions,
        dtype=joint_positions.dtype,
        device=joint_positions.device,
    )
    pose_error = torch.linalg.vector_norm(joint_positions - target, dim=-1)
    return grasp_quality * (1.0 - torch.tanh(pose_error / std))


def object_fallen(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return one after the object has been knocked below its training support."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    return (object_asset.data.root_pos_w.torch[:, 2] < minimum_height).float()


def gripper_switch(
    env: ManagerBasedRLEnv,
    gripper_action_index: int = -1,
) -> torch.Tensor:
    """Penalize rapid binary gripper toggling so each grasp attempt can settle."""
    closing = env.action_manager.action[:, gripper_action_index] < 0.0
    previously_closing = env.action_manager.prev_action[:, gripper_action_index] < 0.0
    return (closing != previously_closing).float()
