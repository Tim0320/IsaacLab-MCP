"""Inspect the visible G1 runtime pose used to seed the bimanual carry task."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "headless", False):
    parser.error("This project requires a visible Kit window; --headless is not allowed.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main() -> int:
    cfg = parse_env_cfg("Isaac-Velocity-Flat-G1-Play-v0", device=args_cli.device, num_envs=1)
    cfg.observations.policy.enable_corruption = False
    env = gym.make("Isaac-Velocity-Flat-G1-Play-v0", cfg=cfg)
    try:
        print("G1_CARRY_POSE_STAGE=environment_created", flush=True)
        env.reset()
        unwrapped = env.unwrapped
        for _ in range(5):
            env.step(torch.zeros(unwrapped.num_envs, unwrapped.action_manager.total_action_dim, device=unwrapped.device))
        robot = unwrapped.scene["robot"]
        print("G1_CARRY_POSE_STAGE=state_sampled", flush=True)
        print("G1_BODY_NAMES=" + json.dumps(robot.body_names), flush=True)
        print("G1_JOINT_NAMES=" + json.dumps(robot.joint_names), flush=True)

        selected_names = [
            "torso_link",
            "left_palm_link",
            "right_palm_link",
        ]
        selected: dict[str, list[float]] = {}
        for name in selected_names:
            ids, names = robot.find_bodies(name)
            if ids:
                selected[names[0]] = robot.data.body_pos_w.torch[0, ids[0]].cpu().tolist()
        print("G1_SELECTED_POSITIONS=" + json.dumps(selected), flush=True)

        result = {
            "task": "Isaac-Velocity-Flat-G1-Play-v0",
            "visible": True,
            "body_names": robot.body_names,
            "joint_names": robot.joint_names,
            "selected_body_positions_w": selected,
            "root_position_w": robot.data.root_pos_w.torch[0].cpu().tolist(),
            "joint_positions": {
                name: float(robot.data.joint_pos.torch[0, index].cpu())
                for index, name in enumerate(robot.joint_names)
            },
        }
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(rendered + "\n", encoding="utf-8")
        print("G1_CARRY_POSE=" + json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
