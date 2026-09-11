"""Validate Dofbot lift task reset, stepping, and gripper closure."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--settle-steps", type=int, default=20)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_mcp.runtime_tasks import register_tasks
from isaaclab_mcp.runtime_tasks.dofbot_cube_lift import TASK_ID
from isaaclab_mcp.runtime_tasks.dofbot_cube_lift.env_cfg import DofbotCubeLiftEnvCfg


def _finger_distance(robot) -> torch.Tensor:
    left_id = robot.find_bodies("Finger_Left_03")[0][0]
    right_id = robot.find_bodies("Finger_Right_03")[0][0]
    left = robot.data.body_pos_w.torch[:, left_id]
    right = robot.data.body_pos_w.torch[:, right_id]
    return torch.linalg.vector_norm(left - right, dim=-1)


def main() -> int:
    register_tasks()
    cfg = DofbotCubeLiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    env = gym.make(TASK_ID, cfg=cfg)
    try:
        observations, _ = env.reset()
        robot = env.unwrapped.scene["robot"]
        opened_distance = _finger_distance(robot)

        close_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        close_actions[:, -1] = -1.0
        rewards = None
        for _ in range(args_cli.settle_steps):
            observations, rewards, _, _, _ = env.step(close_actions)
        closed_distance = _finger_distance(robot)

        open_actions = torch.zeros_like(close_actions)
        open_actions[:, -1] = 1.0
        for _ in range(args_cli.settle_steps):
            observations, rewards, terminated, truncated, _ = env.step(open_actions)
        reopened_distance = _finger_distance(robot)

        policy = observations["policy"]
        ee_positions = env.unwrapped.scene["ee_frame"].data.target_pos_w
        result = {
            "task_id": TASK_ID,
            "asset": env.unwrapped.cfg.scene.robot.spawn.usd_path,
            "num_envs": args_cli.num_envs,
            "device": str(env.unwrapped.device),
            "action_shape": list(env.action_space.shape),
            "policy_observation_shape": list(policy.shape),
            "finite_observations": bool(torch.isfinite(policy).all().item()),
            "finite_rewards": bool(torch.isfinite(rewards).all().item()),
            "end_effector_frame_shape": list(ee_positions.shape),
            "finite_end_effector_frame": bool(torch.isfinite(ee_positions).all().item()),
            "opened_fingertip_distance_m": opened_distance.detach().cpu().tolist(),
            "closed_fingertip_distance_m": closed_distance.detach().cpu().tolist(),
            "reopened_fingertip_distance_m": reopened_distance.detach().cpu().tolist(),
            "gripper_closes": bool(torch.all(closed_distance < opened_distance).item()),
            "gripper_reopens": bool(torch.all(reopened_distance > closed_distance).item()),
            "terminated_count": int(terminated.sum().item()),
            "truncated_count": int(truncated.sum().item()),
        }
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if args_cli.output:
            args_cli.output.parent.mkdir(parents=True, exist_ok=True)
            args_cli.output.write_text(rendered + "\n", encoding="utf-8")
        print("DOFBOT_LIFT_VALIDATION=" + json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["gripper_closes"] and result["gripper_reopens"] else 2
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
