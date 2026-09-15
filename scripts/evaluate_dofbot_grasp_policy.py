"""Evaluate whether a Dofbot policy reaches, closes on, and lifts the cube."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--task-stage", choices=("grasp", "lift"), default="grasp")
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video-length", type=int, default=750)
parser.add_argument("--video-dir", type=Path, required=True)
parser.add_argument("--capture", type=Path)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "headless", False):
    parser.error("This project requires a visible Kit window; --headless is not allowed.")
if not 500 <= args_cli.video_length <= 1000:
    parser.error("--video-length must produce a 10-20 second clip at 50 FPS (500-1000 frames).")
if args_cli.steps < args_cli.video_length:
    parser.error("--steps must be greater than or equal to --video-length.")

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

from isaaclab_mcp.runtime_tasks import register_tasks
from isaaclab_mcp.runtime_tasks.dofbot_cube_lift import GRASP_TASK_ID, TASK_ID
from isaaclab_mcp.runtime_tasks.dofbot_cube_lift.agents.rsl_rl_ppo_cfg import (
    DofbotCubeGraspPPORunnerCfg,
    DofbotCubeLiftPPORunnerCfg,
)
from isaaclab_mcp.runtime_tasks.dofbot_cube_lift.env_cfg import (
    DofbotCubeGraspEnvCfg,
    DofbotCubeLiftEnvCfg,
)


def _write_output(payload: dict) -> None:
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    progress = {"state": "starting", "checkpoint": str(args_cli.checkpoint.resolve())}
    _write_output(progress)
    register_tasks()
    grasp_stage = args_cli.task_stage == "grasp"
    task_id = GRASP_TASK_ID if grasp_stage else TASK_ID
    cfg = DofbotCubeGraspEnvCfg() if grasp_stage else DofbotCubeLiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed

    video_dir = args_cli.video_dir.resolve()
    video_dir.mkdir(parents=True, exist_ok=True)
    base_env = gym.make(task_id, cfg=cfg, render_mode="rgb_array")
    base_env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=str(video_dir),
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(base_env, clip_actions=None)
    progress["state"] = "environment_ready"
    _write_output(progress)
    agent_cfg = DofbotCubeGraspPPORunnerCfg() if grasp_stage else DofbotCubeLiftPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    progress["state"] = "runner_ready"
    _write_output(progress)
    runner.load(str(args_cli.checkpoint.resolve()))
    progress["state"] = "checkpoint_loaded"
    _write_output(progress)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    cube = env.unwrapped.scene["object"]
    left_id = robot.find_bodies("Finger_Left_03")[0][0]
    right_id = robot.find_bodies("Finger_Right_03")[0][0]
    obs = env.get_observations()
    initial_cube_height = cube.data.root_pos_w.torch[:, 2].clone()

    minimum_gripper_cube_distance = torch.full((args_cli.num_envs,), torch.inf, device=env.unwrapped.device)
    maximum_cube_height = torch.full((args_cli.num_envs,), -torch.inf, device=env.unwrapped.device)
    grasp_attempts = torch.zeros(args_cli.num_envs, dtype=torch.long, device=env.unwrapped.device)
    previous_close = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    open_streak = torch.full_like(grasp_attempts, 5)
    stable_grasp_steps = torch.zeros_like(grasp_attempts)
    maximum_stable_grasp_steps = torch.zeros_like(grasp_attempts)
    lifted_and_retained = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    close_steps = 0
    close_near_steps = 0
    total_samples = 0
    episode_completions = 0
    reward_sum = 0.0
    action_sum = torch.zeros(6, device=env.unwrapped.device)
    action_sq_sum = torch.zeros(6, device=env.unwrapped.device)

    try:
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)

                left = robot.data.body_pos_w.torch[:, left_id]
                right = robot.data.body_pos_w.torch[:, right_id]
                gripper_midpoint = 0.5 * (left + right)
                cube_position = cube.data.root_pos_w.torch
                gripper_cube_distance = torch.linalg.vector_norm(cube_position - gripper_midpoint, dim=-1)
                fingertip_gap = torch.linalg.vector_norm(left - right, dim=-1)
                cube_height = cube_position[:, 2]
                commanded_close = actions[:, -1] < 0.0
                near_cube = gripper_cube_distance < 0.06
                stable_grasp = (
                    commanded_close
                    & near_cube
                    & (fingertip_gap > 0.036)
                    & (fingertip_gap < 0.055)
                    & (cube_height > 0.045)
                )
                # Count a new attempt only after the gripper stayed open for at
                # least five policy steps; this rejects one-step sign chatter.
                grasp_attempts += commanded_close & ~previous_close & (open_streak >= 5)
                stable_grasp_steps = torch.where(stable_grasp, stable_grasp_steps + 1, 0)
                maximum_stable_grasp_steps = torch.maximum(maximum_stable_grasp_steps, stable_grasp_steps)
                lifted_and_retained |= stable_grasp & (cube_height > initial_cube_height + 0.02)
                open_streak = torch.where(commanded_close, 0, open_streak + 1)
                previous_close = commanded_close

                minimum_gripper_cube_distance = torch.minimum(minimum_gripper_cube_distance, gripper_cube_distance)
                maximum_cube_height = torch.maximum(maximum_cube_height, cube_height)
                close_steps += int(commanded_close.sum().item())
                close_near_steps += int((commanded_close & near_cube).sum().item())
                total_samples += args_cli.num_envs
                episode_completions += int(dones.sum().item())
                reward_sum += float(rewards.sum().item())
                action_sum += actions.sum(dim=0)
                action_sq_sum += actions.square().sum(dim=0)

        action_mean = action_sum / total_samples
        action_variance = torch.clamp(action_sq_sum / total_samples - action_mean.square(), min=0.0)
        held_grasp = maximum_stable_grasp_steps >= 25
        result = {
            "state": "finished",
            "task_id": task_id,
            "checkpoint": str(args_cli.checkpoint.resolve()),
            "device": str(env.unwrapped.device),
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "seed": args_cli.seed,
            "video_length": args_cli.video_length,
            "video_directory": str(video_dir),
            "total_policy_samples": total_samples,
            "episode_completions": episode_completions,
            "mean_reward_per_step": reward_sum / total_samples,
            "minimum_gripper_cube_distance_m": float(minimum_gripper_cube_distance.min().item()),
            "median_minimum_gripper_cube_distance_m": float(minimum_gripper_cube_distance.median().item()),
            "maximum_cube_height_m": float(maximum_cube_height.max().item()),
            "grasp_attempt_count": int(grasp_attempts.sum().item()),
            "environments_with_grasp_attempt": int((grasp_attempts > 0).sum().item()),
            "environments_held_grasp_for_0_5s": int(held_grasp.sum().item()),
            "held_grasp_success_rate": float(held_grasp.float().mean().item()),
            "maximum_stable_grasp_duration_s": float(maximum_stable_grasp_steps.max().item() * env.unwrapped.step_dt),
            "environments_lifted_2cm_and_retained": int(lifted_and_retained.sum().item()),
            "close_command_fraction": close_steps / total_samples,
            "close_near_cube_fraction": close_near_steps / total_samples,
            "close_commands_near_cube_fraction": close_near_steps / max(close_steps, 1),
            "action_mean": action_mean.cpu().tolist(),
            "action_std": torch.sqrt(action_variance).cpu().tolist(),
        }
        if args_cli.capture:
            from omni.kit.viewport.utility import (
                capture_viewport_to_file,
                get_active_viewport,
            )

            capture_path = args_cli.capture.resolve()
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            viewport = get_active_viewport()
            if viewport is not None:
                capture_viewport_to_file(viewport, file_path=str(capture_path), is_hdr=False)
                for _ in range(5):
                    env.step(torch.zeros_like(actions))
                result["capture"] = str(capture_path)
        _write_output(result)
        print("DOFBOT_GRASP_EVALUATION=" + json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
