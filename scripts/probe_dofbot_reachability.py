"""Search Dofbot joint actions for a reachable cube-centered gripper pose."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arm-action-scale", type=float, required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--iterations", type=int, default=12)
parser.add_argument("--settle-steps", type=int, default=20)
parser.add_argument("--lift-delta", type=float, default=0.2)
parser.add_argument("--lift-search-iterations", type=int, default=0)
parser.add_argument("--lift-steps", type=int, default=60)
parser.add_argument("--hold-steps", type=int, default=60)
parser.add_argument("--target-z-offset", type=float, default=0.0)
parser.add_argument("--object-z-offset", type=float, default=0.0)
parser.add_argument("--object-scale", type=float)
parser.add_argument("--pedestal-width", type=float, default=0.10)
parser.add_argument("--approach-action", type=float, nargs=5)
parser.add_argument("--fixed-grasp-action", type=float, nargs=5)
parser.add_argument("--gripper-effort-limit", type=float)
parser.add_argument("--left-close-command", type=float)
parser.add_argument("--right-close-command", type=float)
parser.add_argument("--disable-self-collisions", action="store_true")
parser.add_argument("--ee-frame-mode", choices=("current", "wrist"), default="current")
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObjectCfg

from isaaclab_mcp.runtime_tasks import register_tasks
from isaaclab_mcp.runtime_tasks.dofbot_cube_lift import TASK_ID
from isaaclab_mcp.runtime_tasks.dofbot_cube_lift.env_cfg import DofbotCubeLiftEnvCfg


def _finger_geometry(robot, cube, left_id: int, right_id: int):
    left = robot.data.body_pos_w.torch[:, left_id]
    right = robot.data.body_pos_w.torch[:, right_id]
    midpoint = 0.5 * (left + right)
    cube_position = cube.data.root_pos_w.torch
    return (
        torch.linalg.vector_norm(midpoint - cube_position, dim=-1),
        torch.linalg.vector_norm(left - right, dim=-1),
        midpoint,
        cube_position,
    )


def _write_result(result: dict) -> None:
    if args_cli.output:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    register_tasks()
    cfg = DofbotCubeLiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.episode_length_s = 120.0
    cfg.actions.arm_action.scale = args_cli.arm_action_scale
    if args_cli.disable_self_collisions:
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    if args_cli.gripper_effort_limit is not None:
        cfg.scene.robot.actuators["gripper"].effort_limit_sim = args_cli.gripper_effort_limit
    if args_cli.left_close_command is not None:
        cfg.actions.gripper_action.close_command_expr["Finger_Left_01_RevoluteJoint"] = args_cli.left_close_command
    if args_cli.right_close_command is not None:
        cfg.actions.gripper_action.close_command_expr["Finger_Right_01_RevoluteJoint"] = args_cli.right_close_command
    object_init_pos = cfg.scene.object.init_state.pos
    if args_cli.object_scale is not None:
        cfg.scene.object.spawn.scale = (args_cli.object_scale,) * 3
    cfg.scene.object.init_state.pos = (
        object_init_pos[0],
        object_init_pos[1],
        object_init_pos[2] + args_cli.object_z_offset,
    )
    if args_cli.object_z_offset > 0.0:
        cfg.scene.pedestal = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pedestal",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(object_init_pos[0], object_init_pos[1], args_cli.object_z_offset / 2.0)
            ),
            spawn=sim_utils.CuboidCfg(
                size=(
                    args_cli.pedestal_width,
                    args_cli.pedestal_width,
                    args_cli.object_z_offset,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25)),
            ),
        )
    if args_cli.ee_frame_mode == "wrist":
        cfg.scene.ee_frame.target_frames[0].prim_path = "{ENV_REGEX_NS}/Robot/link5/Wrist_Twist$"
        cfg.scene.ee_frame.target_frames[0].offset.pos = (0.0, 0.0, 0.066)
    cfg.events.reset_object_position.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
    }

    env = gym.make(TASK_ID, cfg=cfg)
    robot = env.unwrapped.scene["robot"]
    cube = env.unwrapped.scene["object"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    left_id = robot.find_bodies("Finger_Left_03")[0][0]
    right_id = robot.find_bodies("Finger_Right_03")[0][0]
    mean = torch.zeros(5, device=env.unwrapped.device)
    std = torch.ones(5, device=env.unwrapped.device) * 0.8
    elite_count = max(4, args_cli.num_envs // 8)
    target_open_gap = 0.0603
    best_objective = torch.tensor(torch.inf, device=env.unwrapped.device)
    best_distance = torch.tensor(torch.inf, device=env.unwrapped.device)
    best_action = torch.zeros(5, device=env.unwrapped.device)

    try:
        torch.manual_seed(42)
        approach_action = (
            None
            if args_cli.approach_action is None
            else torch.tensor(args_cli.approach_action, device=env.unwrapped.device)
        )
        fixed_grasp_action = (
            None
            if args_cli.fixed_grasp_action is None
            else torch.tensor(args_cli.fixed_grasp_action, device=env.unwrapped.device)
        )
        if fixed_grasp_action is not None:
            best_action = fixed_grasp_action
        search_iterations = 0 if fixed_grasp_action is not None else args_cli.iterations
        for _ in range(search_iterations):
            candidates = torch.clamp(
                mean + std * torch.randn(args_cli.num_envs, 5, device=env.unwrapped.device),
                -1.0,
                1.0,
            )
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, -1] = 1.0
            env.reset()
            cube_before_approach = cube.data.root_pos_w.torch.clone()
            if approach_action is not None:
                actions[:, :5] = approach_action
                for _ in range(max(args_cli.settle_steps, 40)):
                    env.step(actions)
            actions[:, :5] = candidates
            for _ in range(args_cli.settle_steps):
                env.step(actions)

            _, gap, midpoint, cube_position = _finger_geometry(robot, cube, left_id, right_id)
            target_position = cube_position.clone()
            target_position[:, 2] += args_cli.target_z_offset
            target_distance = torch.linalg.vector_norm(midpoint - target_position, dim=-1)
            approach_cube_displacement = torch.linalg.vector_norm(cube_position - cube_before_approach, dim=-1)
            objective = target_distance + 2.0 * torch.abs(gap - target_open_gap) + 5.0 * approach_cube_displacement
            elite_ids = torch.topk(objective, elite_count, largest=False).indices
            elites = candidates[elite_ids]
            mean = elites.mean(dim=0)
            std = torch.clamp(elites.std(dim=0, unbiased=False), min=0.03)
            iteration_best, iteration_best_id = objective.min(dim=0)
            if iteration_best < best_objective:
                best_objective = iteration_best
                best_distance = target_distance[iteration_best_id]
                best_action = candidates[iteration_best_id].clone()

        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        actions[:, :5] = best_action
        actions[:, -1] = 1.0
        env.reset()
        if approach_action is not None:
            actions[:, :5] = approach_action
            for _ in range(max(args_cli.settle_steps, 40)):
                env.step(actions)
            actions[:, :5] = best_action
        for _ in range(max(args_cli.settle_steps, 40)):
            env.step(actions)
        open_distance, open_gap, open_midpoint, cube_before_close = _finger_geometry(robot, cube, left_id, right_id)
        open_joint_positions = robot.data.joint_pos.torch[0].clone()
        open_applied_torque = robot.data.applied_torque.torch[0].clone()
        ee_position = ee_frame.data.target_pos_w.torch[..., 0, :]
        ee_midpoint_offset = torch.linalg.vector_norm(ee_position - open_midpoint, dim=-1)
        verified_target = cube_before_close.clone()
        verified_target[:, 2] += args_cli.target_z_offset
        verified_target_distance = torch.linalg.vector_norm(open_midpoint - verified_target, dim=-1)

        actions[:, -1] = -1.0
        for _ in range(40):
            env.step(actions)
        closed_distance, closed_gap, _, cube_after_close = _finger_geometry(robot, cube, left_id, right_id)
        closed_joint_positions = robot.data.joint_pos.torch[0].clone()
        closed_applied_torque = robot.data.applied_torque.torch[0].clone()
        cube_close_displacement = torch.linalg.vector_norm(cube_after_close - cube_before_close, dim=-1)

        best_lift_action = best_action.clone()
        best_lift_objective = torch.tensor(torch.inf, device=env.unwrapped.device)
        if args_cli.lift_search_iterations > 0:
            lift_mean = best_action.clone()
            lift_std = torch.tensor([0.05, 0.30, 0.30, 0.30, 0.05], device=env.unwrapped.device)
            lift_target_delta = torch.tensor([0.0, 0.0, args_cli.lift_delta], device=env.unwrapped.device)
            for _ in range(args_cli.lift_search_iterations):
                lift_candidates = torch.clamp(
                    lift_mean + lift_std * torch.randn(args_cli.num_envs, 5, device=env.unwrapped.device),
                    -1.0,
                    1.0,
                )
                actions[:, :5] = best_action
                actions[:, -1] = 1.0
                env.reset()
                if approach_action is not None:
                    actions[:, :5] = approach_action
                    for _ in range(max(args_cli.settle_steps, 40)):
                        env.step(actions)
                    actions[:, :5] = best_action
                for _ in range(max(args_cli.settle_steps, 40)):
                    env.step(actions)
                actions[:, -1] = -1.0
                for _ in range(40):
                    env.step(actions)
                _, _, grasp_midpoint, lift_cube_start = _finger_geometry(robot, cube, left_id, right_id)
                lift_target = grasp_midpoint + lift_target_delta
                actions[:, :5] = lift_candidates
                maximum_candidate_height = lift_cube_start[:, 2].clone()
                for _ in range(args_cli.lift_steps):
                    env.step(actions)
                    _, _, _, lift_cube_position = _finger_geometry(robot, cube, left_id, right_id)
                    maximum_candidate_height = torch.maximum(maximum_candidate_height, lift_cube_position[:, 2])
                _, _, lifted_midpoint, lift_cube_end = _finger_geometry(robot, cube, left_id, right_id)
                lift_target_distance = torch.linalg.vector_norm(lifted_midpoint - lift_target, dim=-1)
                lift_gain = lift_cube_end[:, 2] - lift_cube_start[:, 2]
                horizontal_cube_displacement = torch.linalg.vector_norm(
                    lift_cube_end[:, :2] - lift_cube_start[:, :2], dim=-1
                )
                retained_distance, _, _, _ = _finger_geometry(robot, cube, left_id, right_id)
                lift_objective = (
                    lift_target_distance
                    - 3.0 * torch.clamp(lift_gain, min=0.0, max=args_cli.lift_delta)
                    + 2.0 * horizontal_cube_displacement
                    + 2.0 * torch.clamp(retained_distance - 0.05, min=0.0)
                )
                lift_elite_ids = torch.topk(lift_objective, elite_count, largest=False).indices
                lift_elites = lift_candidates[lift_elite_ids]
                lift_mean = lift_elites.mean(dim=0)
                lift_std = torch.clamp(lift_elites.std(dim=0, unbiased=False), min=0.02)
                iteration_lift_best, iteration_lift_best_id = lift_objective.min(dim=0)
                if iteration_lift_best < best_lift_objective:
                    best_lift_objective = iteration_lift_best
                    best_lift_action = lift_candidates[iteration_lift_best_id].clone()

        actions[:, :5] = best_action
        actions[:, -1] = 1.0
        env.reset()
        if approach_action is not None:
            actions[:, :5] = approach_action
            for _ in range(max(args_cli.settle_steps, 40)):
                env.step(actions)
            actions[:, :5] = best_action
        for _ in range(max(args_cli.settle_steps, 40)):
            env.step(actions)
        actions[:, -1] = -1.0
        for _ in range(40):
            env.step(actions)
        final_grasp_distance, final_closed_gap, _, final_cube_after_close = _finger_geometry(
            robot, cube, left_id, right_id
        )
        actions[:, :5] = best_lift_action
        maximum_lift_height = final_cube_after_close[:, 2].clone()
        minimum_lift_distance = final_grasp_distance.clone()
        for _ in range(args_cli.lift_steps):
            env.step(actions)
            lift_distance, _, _, lift_cube_position = _finger_geometry(robot, cube, left_id, right_id)
            maximum_lift_height = torch.maximum(maximum_lift_height, lift_cube_position[:, 2])
            minimum_lift_distance = torch.minimum(minimum_lift_distance, lift_distance)
        lift_end_cube_position = cube.data.root_pos_w.torch.clone()
        for _ in range(args_cli.hold_steps):
            env.step(actions)
        hold_end_cube_position = cube.data.root_pos_w.torch.clone()
        maximum_lift_gain = maximum_lift_height - final_cube_after_close[:, 2]
        final_lift_gain = hold_end_cube_position[:, 2] - final_cube_after_close[:, 2]

        result = {
            "state": "finished",
            "task_id": TASK_ID,
            "device": str(env.unwrapped.device),
            "arm_action_scale": args_cli.arm_action_scale,
            "ee_frame_mode": args_cli.ee_frame_mode,
            "num_envs": args_cli.num_envs,
            "iterations": args_cli.iterations,
            "settle_steps": args_cli.settle_steps,
            "lift_delta": args_cli.lift_delta,
            "lift_search_iterations": args_cli.lift_search_iterations,
            "lift_steps": args_cli.lift_steps,
            "hold_steps": args_cli.hold_steps,
            "target_z_offset": args_cli.target_z_offset,
            "object_z_offset": args_cli.object_z_offset,
            "object_scale": args_cli.object_scale,
            "pedestal_width_m": args_cli.pedestal_width,
            "gripper_effort_limit_nm": args_cli.gripper_effort_limit,
            "left_close_command_rad": args_cli.left_close_command,
            "right_close_command_rad": args_cli.right_close_command,
            "self_collisions_enabled": not args_cli.disable_self_collisions,
            "approach_normalized_arm_action": (None if approach_action is None else approach_action.cpu().tolist()),
            "best_normalized_arm_action": best_action.cpu().tolist(),
            "best_normalized_lift_action": best_lift_action.cpu().tolist(),
            "best_lift_objective": (None if torch.isinf(best_lift_objective) else float(best_lift_objective.item())),
            "robot_body_names": list(robot.body_names),
            "robot_body_paths_env0": list(robot.root_view.link_paths[0]),
            "best_search_objective": (None if torch.isinf(best_objective) else float(best_objective.item())),
            "best_search_distance_m": (None if torch.isinf(best_distance) else float(best_distance.item())),
            "joint_names": list(robot.joint_names),
            "open_joint_positions_rad": open_joint_positions.cpu().tolist(),
            "closed_joint_positions_rad": closed_joint_positions.cpu().tolist(),
            "open_applied_torque_nm": open_applied_torque.cpu().tolist(),
            "closed_applied_torque_nm": closed_applied_torque.cpu().tolist(),
            "verified_open_distance_m": float(open_distance.min().item()),
            "verified_target_distance_m": float(verified_target_distance.min().item()),
            "verified_closed_distance_m": float(closed_distance.min().item()),
            "verified_arm_joint_positions_rad": robot.data.joint_pos.torch[0, :5].cpu().tolist(),
            "verified_gripper_midpoint_local_m": (open_midpoint[0] - env.unwrapped.scene.env_origins[0]).cpu().tolist(),
            "verified_cube_position_local_m": (cube_before_close[0] - env.unwrapped.scene.env_origins[0])
            .cpu()
            .tolist(),
            "open_fingertip_gap_m": float(open_gap.min().item()),
            "closed_fingertip_gap_m": float(closed_gap.min().item()),
            "final_closed_fingertip_gap_m": float(final_closed_gap.min().item()),
            "ee_frame_to_fingertip_midpoint_m": float(ee_midpoint_offset.max().item()),
            "maximum_cube_displacement_while_closing_m": float(cube_close_displacement.max().item()),
            "maximum_cube_height_after_close_m": float(cube_after_close[:, 2].max().item()),
            "maximum_cube_height_during_lift_m": float(maximum_lift_height.max().item()),
            "maximum_cube_lift_gain_m": float(maximum_lift_gain.max().item()),
            "cube_height_at_lift_end_m": float(lift_end_cube_position[:, 2].max().item()),
            "cube_height_after_hold_m": float(hold_end_cube_position[:, 2].max().item()),
            "final_cube_lift_gain_m": float(final_lift_gain.max().item()),
            "lifted_over_2cm_env_count": int((final_lift_gain > 0.02).sum().item()),
            "minimum_gripper_cube_distance_during_lift_m": float(minimum_lift_distance.min().item()),
        }
        _write_result(result)
        print("DOFBOT_REACHABILITY=" + json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
