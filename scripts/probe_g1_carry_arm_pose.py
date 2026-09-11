"""Probe several G1 bimanual carry poses in one visible Isaac Lab scene."""

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

POSES = [
    {"name": "default", "shoulder_pitch": 0.35, "elbow_pitch": 0.87},
    {"name": "positive_low", "shoulder_pitch": 0.90, "elbow_pitch": 1.30},
    {"name": "negative_low", "shoulder_pitch": -0.50, "elbow_pitch": 1.30},
    {"name": "negative_mid", "shoulder_pitch": -1.00, "elbow_pitch": 1.50},
    {"name": "positive_high", "shoulder_pitch": 0.80, "elbow_pitch": 1.80},
    {"name": "negative_high", "shoulder_pitch": -1.30, "elbow_pitch": 2.00},
]


def main() -> int:
    cfg = parse_env_cfg("Isaac-Velocity-Flat-G1-Play-v0", device=args_cli.device, num_envs=len(POSES))
    cfg.observations.policy.enable_corruption = False
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    env = gym.make("Isaac-Velocity-Flat-G1-Play-v0", cfg=cfg)
    try:
        env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        joint_pos = robot.data.default_joint_pos.torch.clone()

        for side in ("left", "right"):
            shoulder_ids, _ = robot.find_joints(f"{side}_shoulder_pitch_joint")
            elbow_ids, _ = robot.find_joints(f"{side}_elbow_pitch_joint")
            for env_id, pose in enumerate(POSES):
                joint_pos[env_id, shoulder_ids[0]] = pose["shoulder_pitch"]
                joint_pos[env_id, elbow_ids[0]] = pose["elbow_pitch"]

        env_ids = torch.arange(len(POSES), device=unwrapped.device, dtype=torch.int32)
        robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)
        robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)

        for _ in range(30):
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.step(render=True)
            unwrapped.scene.update(dt=unwrapped.physics_dt)

        left_id = robot.find_bodies("left_palm_link")[0][0]
        right_id = robot.find_bodies("right_palm_link")[0][0]
        result = []
        for env_id, pose in enumerate(POSES):
            root = robot.data.root_pos_w.torch[env_id]
            left = robot.data.body_pos_w.torch[env_id, left_id]
            right = robot.data.body_pos_w.torch[env_id, right_id]
            entry = {
                **pose,
                "left_palm_relative_w": (left - root).cpu().tolist(),
                "right_palm_relative_w": (right - root).cpu().tolist(),
                "palm_midpoint_relative_w": (0.5 * (left + right) - root).cpu().tolist(),
                "palm_separation_m": float(torch.linalg.vector_norm(left - right).cpu()),
            }
            result.append(entry)

        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(rendered + "\n", encoding="utf-8")
        print("G1_CARRY_ARM_POSES=" + json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
