"""Validate TM6S task registration, reset, observation, and bounded stepping."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_mcp.runtime_tasks import register_tasks
from isaaclab_mcp.runtime_tasks.tm6s_lift_proxy import TASK_ID
from isaaclab_mcp.runtime_tasks.tm6s_lift_proxy.env_cfg import TM6SLiftProxyEnvCfg


def main() -> int:
    register_tasks()
    cfg = TM6SLiftProxyEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    env = gym.make(TASK_ID, cfg=cfg)
    try:
        observations, _ = env.reset()
        reward_min = float("inf")
        reward_max = float("-inf")
        terminated_count = 0
        truncated_count = 0
        for _ in range(args_cli.steps):
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            observations, rewards, terminated, truncated, _ = env.step(actions)
            reward_min = min(reward_min, float(rewards.min().item()))
            reward_max = max(reward_max, float(rewards.max().item()))
            terminated_count += int(terminated.sum().item())
            truncated_count += int(truncated.sum().item())

        policy = observations["policy"]
        result = {
            "task_id": TASK_ID,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "device": str(env.unwrapped.device),
            "action_shape": list(env.action_space.shape),
            "policy_observation_shape": list(policy.shape),
            "finite_observations": bool(torch.isfinite(policy).all().item()),
            "reward_min": reward_min,
            "reward_max": reward_max,
            "terminated_count": terminated_count,
            "truncated_count": truncated_count,
        }
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        if args_cli.output:
            args_cli.output.parent.mkdir(parents=True, exist_ok=True)
            args_cli.output.write_text(rendered + "\n", encoding="utf-8")
        print("TM6S_TASK_VALIDATION=" + json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
