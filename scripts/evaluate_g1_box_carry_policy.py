"""Evaluate G1 bilateral 15 kg carry with a visible following camera and RecordVideo."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--task-stage", choices=("hold", "run"), default="run")
parser.add_argument("--steps", type=int, default=750)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--video-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "headless", False):
    parser.error("This project requires a visible Kit window; --headless is not allowed.")
if not 500 <= args_cli.steps <= 1000:
    parser.error("--steps must produce a 10-20 second clip at 50 FPS (500-1000 steps).")

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

from isaaclab_mcp.runtime_tasks import register_tasks
from isaaclab_mcp.runtime_tasks.g1_box_carry import PLAY_TASK_ID, RUN_PLAY_TASK_ID
from isaaclab_mcp.runtime_tasks.g1_box_carry.agents.rsl_rl_ppo_cfg import (
    G1BoxCarryPPORunnerCfg,
    G1BoxCarryRunPPORunnerCfg,
)
from isaaclab_mcp.runtime_tasks.g1_box_carry.env_cfg import G1BoxCarryEnvCfg_PLAY, G1BoxCarryRunEnvCfg_PLAY


def _write_output(payload: dict[str, object]) -> None:
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    _write_output({"state": "starting"})
    register_tasks()
    is_run = args_cli.task_stage == "run"
    task_id = RUN_PLAY_TASK_ID if is_run else PLAY_TASK_ID
    env_cfg = G1BoxCarryRunEnvCfg_PLAY() if is_run else G1BoxCarryEnvCfg_PLAY()
    agent_cfg = G1BoxCarryRunPPORunnerCfg() if is_run else G1BoxCarryPPORunnerCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42

    checkpoint = args_cli.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    video_dir = args_cli.video_dir.resolve()
    video_dir.mkdir(parents=True, exist_ok=True)

    base_env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array")
    _write_output({"state": "environment_ready", "task_id": task_id})
    base_env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=str(video_dir),
        step_trigger=lambda step: step == 0,
        video_length=args_cli.steps,
        disable_logger=True,
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    _write_output({"state": "wrapper_ready", "task_id": task_id})
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    _write_output({"state": "runner_ready", "task_id": task_id})
    runner.load(str(checkpoint))
    _write_output({"state": "checkpoint_loaded", "task_id": task_id})
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    _write_output({"state": "policy_ready", "task_id": task_id})

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    box = unwrapped.scene["box"]
    palm_ids = robot.find_bodies(["left_palm_link", "right_palm_link"], preserve_order=True)[0]
    anchor_offsets = torch.tensor(
        [[0.0, 0.17, 0.0], [0.0, -0.17, 0.0]], device=unwrapped.device, dtype=torch.float32
    )
    obs = env.get_observations()
    previous_x = robot.data.root_pos_w.torch[:, 0].clone()
    episode_distance = torch.zeros(1, device=unwrapped.device)
    max_episode_distance = torch.zeros_like(episode_distance)
    carry_streak = torch.zeros(1, dtype=torch.long, device=unwrapped.device)
    max_carry_streak = torch.zeros_like(carry_streak)
    survival_streak = torch.zeros_like(carry_streak)
    max_survival_streak = torch.zeros_like(carry_streak)
    termination_counts = {name: 0 for name in unwrapped.termination_manager.active_terms}
    total_forward_speed = 0.0
    total_command_speed = 0.0
    total_tracking_error = 0.0
    total_anchor_error = torch.zeros(2, device=unwrapped.device)
    max_anchor_error = torch.zeros(2, device=unwrapped.device)
    carried_samples = 0

    try:
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                done_mask = dones.bool()
                policy.reset(done_mask)

                for name in termination_counts:
                    termination_counts[name] += int(unwrapped.termination_manager.get_term(name).sum().item())

                root_x = robot.data.root_pos_w.torch[:, 0]
                dx = root_x - previous_x
                episode_distance += torch.where(done_mask, 0.0, torch.clamp(dx, min=0.0))
                max_episode_distance = torch.maximum(max_episode_distance, episode_distance)
                episode_distance = torch.where(done_mask, 0.0, episode_distance)
                previous_x = root_x.clone()

                survival_streak = torch.where(done_mask, 0, survival_streak + 1)
                max_survival_streak = torch.maximum(max_survival_streak, survival_streak)

                box_quat = box.data.root_quat_w.torch[:, None, :].expand(-1, 2, -1)
                offsets = anchor_offsets[None, :, :].expand(1, -1, -1)
                anchors = box.data.root_pos_w.torch[:, None, :] + quat_apply(box_quat, offsets)
                palms = robot.data.body_pos_w.torch[:, palm_ids]
                errors = torch.linalg.vector_norm(palms - anchors, dim=-1)
                carried = (errors < 0.40).all(dim=1) & (box.data.root_pos_w.torch[:, 2] > 0.30)
                carry_streak = torch.where(carried, carry_streak + 1, 0)
                max_carry_streak = torch.maximum(max_carry_streak, carry_streak)
                carried_samples += int(carried.sum().item())
                total_anchor_error += errors.sum(dim=0)
                max_anchor_error = torch.maximum(max_anchor_error, errors.max(dim=0).values)

                forward_speed = robot.data.root_lin_vel_b.torch[:, 0]
                command_speed = unwrapped.command_manager.get_command("base_velocity")[:, 0]
                total_forward_speed += float(forward_speed.sum().item())
                total_command_speed += float(command_speed.sum().item())
                total_tracking_error += float(torch.abs(forward_speed - command_speed).sum().item())

        max_episode_distance = torch.maximum(max_episode_distance, episode_distance)
        duration_s = args_cli.steps * unwrapped.step_dt
        mean_forward_speed = total_forward_speed / args_cli.steps
        carried_fraction = carried_samples / args_cli.steps
        survived_entire_clip = sum(termination_counts.values()) == 0
        speed_threshold = 0.45 if is_run else 0.10
        passed = survived_entire_clip and carried_fraction >= 0.95 and mean_forward_speed >= speed_threshold
        expected_video = video_dir / "rl-video-step-0.mp4"
        result = {
            "state": "finished",
            "passed": passed,
            "task_id": task_id,
            "checkpoint": str(checkpoint),
            "visible": True,
            "steps": args_cli.steps,
            "duration_s": duration_s,
            "box_mass_kg_readback": float(box.data.body_mass.torch[0, 0].cpu()),
            "mean_commanded_forward_speed_mps": total_command_speed / args_cli.steps,
            "mean_actual_forward_speed_mps": mean_forward_speed,
            "mean_forward_speed_absolute_error_mps": total_tracking_error / args_cli.steps,
            "maximum_episode_forward_distance_m": float(max_episode_distance.max().item()),
            "carried_sample_fraction": carried_fraction,
            "maximum_continuous_carry_duration_s": float(max_carry_streak.max().item() * unwrapped.step_dt),
            "maximum_survival_duration_s": float(max_survival_streak.max().item() * unwrapped.step_dt),
            "mean_anchor_error_m": (total_anchor_error / args_cli.steps).cpu().tolist(),
            "max_anchor_error_m": max_anchor_error.cpu().tolist(),
            "termination_counts": termination_counts,
            "video": str(expected_video),
            "video_finalized_on_environment_close": True,
            "pass_criteria": {
                "no_termination_during_clip": True,
                "carried_sample_fraction_min": 0.95,
                "mean_actual_forward_speed_mps_min": speed_threshold,
            },
        }
        _write_output(result)
        print("G1_BOX_CARRY_EVALUATION=" + json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        try:
            raise SystemExit(main())
        except Exception as exc:
            failure = {"state": "failed", "type": type(exc).__name__, "message": str(exc)}
            _write_output(failure)
            print("G1_BOX_CARRY_EVALUATION_FAILURE=" + json.dumps(failure, ensure_ascii=False), flush=True)
            raise
    finally:
        simulation_app.close()
