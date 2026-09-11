"""Run a visible physics smoke test for the G1 15 kg bimanual carry task."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "headless", False):
    parser.error("This project requires a visible Kit window; --headless is not allowed.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_mcp.runtime_tasks import register_tasks

TASK_ID = "Isaac-Carry-Box-G1-Play-v0"


def main() -> int:
    register_tasks()
    cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1)
    print("G1_BOX_CARRY_STAGE=config_parsed", flush=True)
    try:
        env = gym.make(TASK_ID, cfg=cfg)
    except BaseException as exc:
        failure = {"stage": "gym.make", "type": type(exc).__name__, "message": str(exc)}
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("G1_BOX_CARRY_FAILURE=" + json.dumps(failure, ensure_ascii=False), flush=True)
        raise
    try:
        print("G1_BOX_CARRY_STAGE=environment_created", flush=True)
        env.reset()
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        box = unwrapped.scene["box"]
        palm_ids = robot.find_bodies(["left_palm_link", "right_palm_link"], preserve_order=True)[0]
        anchors_b = torch.tensor([[0.0, 0.17, 0.0], [0.0, -0.17, 0.0]], device=unwrapped.device)
        zero_action = torch.zeros((1, unwrapped.action_manager.total_action_dim), device=unwrapped.device)

        anchor_error_samples: list[list[float]] = []
        box_height_samples: list[float] = []
        support_force_samples: list[list[float]] = []
        first_geometry: dict[str, object] | None = None
        termination_count = 0
        for _ in range(args_cli.steps):
            _, _, terminated, truncated, _ = env.step(zero_action)
            termination_count += int((terminated | truncated).sum().cpu())
            box_quat = box.data.root_quat_w.torch[0].expand(2, -1)
            anchor_pos = box.data.root_pos_w.torch[0].unsqueeze(0) + quat_apply(box_quat, anchors_b)
            palms = robot.data.body_pos_w.torch[0, palm_ids]
            if first_geometry is None:
                first_geometry = {
                    "robot_root_position_w": robot.data.root_pos_w.torch[0].cpu().tolist(),
                    "robot_root_quaternion_w": robot.data.root_quat_w.torch[0].cpu().tolist(),
                    "palm_positions_w": palms.cpu().tolist(),
                    "box_position_w": box.data.root_pos_w.torch[0].cpu().tolist(),
                    "box_quaternion_w": box.data.root_quat_w.torch[0].cpu().tolist(),
                    "anchor_positions_w": anchor_pos.cpu().tolist(),
                }
            anchor_error_samples.append(torch.linalg.vector_norm(palms - anchor_pos, dim=-1).cpu().tolist())
            box_height_samples.append(float(box.data.root_pos_w.torch[0, 2].cpu()))
            forces = robot.permanent_wrench_composer.out_force_b.torch[0, palm_ids]
            support_force_samples.append(torch.linalg.vector_norm(forces, dim=-1).cpu().tolist())

        errors = torch.tensor(anchor_error_samples)
        support_forces = torch.tensor(support_force_samples)
        result = {
            "task": TASK_ID,
            "visible": True,
            "steps": args_cli.steps,
            "box_mass_kg_readback": float(box.data.body_mass.torch[0, 0].cpu()),
            "palm_bodies": [robot.body_names[index] for index in palm_ids],
            "action_dim": unwrapped.action_manager.total_action_dim,
            "policy_observation_dim": int(unwrapped.observation_manager.group_obs_dim["policy"][0]),
            "mean_anchor_error_m": errors.mean(dim=0).tolist(),
            "max_anchor_error_m": errors.max(dim=0).values.tolist(),
            "mean_palm_support_force_n": support_forces.mean(dim=0).tolist(),
            "minimum_box_height_m": min(box_height_samples),
            "automatic_episode_retries": termination_count,
            "first_step_geometry": first_geometry,
        }
        rendered = json.dumps(result, ensure_ascii=False, indent=2)
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(rendered + "\n", encoding="utf-8")
        print("G1_BOX_CARRY_VALIDATION=" + json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
