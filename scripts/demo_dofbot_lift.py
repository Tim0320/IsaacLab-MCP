"""Show the Dofbot cube-lift scene in an interactive Isaac Sim window."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--duration", type=float, default=300.0, help="Maximum demo duration in seconds.")
parser.add_argument(
    "--status-file",
    type=Path,
    default=Path("artifacts/dofbot_demo_status.json"),
    help="JSON status file used to verify that the visible demo is advancing.",
)
parser.add_argument(
    "--capture-dir",
    type=Path,
    help="Optional directory for four viewport captures across one motion cycle.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Keep the visible demo on one GPU. The dedicated demo experience disables RTX
# geometry streaming while retaining Fabric articulation transforms.
physics_gpu = args_cli.device.split(":", maxsplit=1)[1] if ":" in args_cli.device else "0"
demo_kit_args = [
    "--/renderer/multiGpu/enabled=false",
    f"--/physics/cudaDevice={physics_gpu}",
]
args_cli.kit_args = " ".join(filter(None, [args_cli.kit_args, *demo_kit_args]))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

from isaaclab_mcp.runtime_tasks import register_tasks
from isaaclab_mcp.runtime_tasks.dofbot_cube_lift import TASK_ID
from isaaclab_mcp.runtime_tasks.dofbot_cube_lift.env_cfg import DofbotCubeLiftEnvCfg


def _finger_distance(robot) -> float:
    left_id = robot.find_bodies("Finger_Left_03")[0][0]
    right_id = robot.find_bodies("Finger_Right_03")[0][0]
    left = robot.data.body_pos_w.torch[0, left_id]
    right = robot.data.body_pos_w.torch[0, right_id]
    return float(torch.linalg.vector_norm(left - right).item())


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    register_tasks()
    cfg = DofbotCubeLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device
    cfg.sim.use_fabric = True
    cfg.actions.arm_action.scale = 1.0
    cfg.episode_length_s = max(args_cli.duration + 30.0, 600.0)
    cfg.commands.object_pose.debug_vis = False
    cfg.viewer.eye = (1.05, 1.05, 0.72)
    cfg.viewer.lookat = (0.0, 0.0, 0.22)

    env = gym.make(TASK_ID, cfg=cfg)
    started = time.monotonic()
    step = 0
    capture_steps = {90, 190, 290, 390}
    capture_futures = []
    state = {
        "state": "starting",
        "task_id": TASK_ID,
        "asset": env.unwrapped.cfg.scene.robot.spawn.usd_path,
        "device": str(env.unwrapped.device),
        "duration_seconds": args_cli.duration,
        "camera_eye": list(cfg.viewer.eye),
        "camera_lookat": list(cfg.viewer.lookat),
    }
    _write_status(args_cli.status_file, state)

    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        env.unwrapped.sim.set_camera_view(eye=cfg.viewer.eye, target=cfg.viewer.lookat)
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        arm_joint_min = torch.full((5,), torch.inf, device=env.unwrapped.device)
        arm_joint_max = torch.full((5,), -torch.inf, device=env.unwrapped.device)
        dt = float(env.unwrapped.step_dt)
        state["state"] = "ready"
        state["action_shape"] = list(env.action_space.shape)
        state["step_dt_seconds"] = dt
        state["fingertip_distance_m"] = _finger_distance(robot)
        _write_status(args_cli.status_file, state)
        print("DOFBOT_DEMO_READY=" + json.dumps(state, ensure_ascii=False), flush=True)

        while simulation_app.is_running() and time.monotonic() - started < args_cli.duration:
            frame_started = time.monotonic()
            cycle = step * dt / 8.0
            sweep = math.sin(2.0 * math.pi * cycle)

            # Deliberately large but joint-limit-safe motion for visual inspection.
            actions.zero_()
            actions[:, 0] = 0.65 * sweep
            actions[:, 1] = 0.45 * sweep
            actions[:, 2] = -0.32 * sweep
            actions[:, 3] = 0.22 * sweep
            actions[:, 4] = 0.70 * math.sin(4.0 * math.pi * cycle)
            actions[:, -1] = 1.0 if step % 200 < 100 else -1.0
            env.step(actions)
            step += 1

            if args_cli.capture_dir and step in capture_steps:
                viewport = get_active_viewport()
                if viewport is not None:
                    args_cli.capture_dir.mkdir(parents=True, exist_ok=True)
                    capture_path = args_cli.capture_dir / f"dofbot_step_{step:04d}.png"
                    capture_futures.append(
                        capture_viewport_to_file(viewport, file_path=str(capture_path), is_hdr=False)
                    )

            arm_joint_pos = robot.data.joint_pos.torch[0, :5]
            arm_joint_min = torch.minimum(arm_joint_min, arm_joint_pos)
            arm_joint_max = torch.maximum(arm_joint_max, arm_joint_pos)

            if step % 30 == 0:
                state.update(
                    {
                        "state": "running",
                        "step": step,
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                        "gripper_command": "open" if actions[0, -1].item() > 0 else "close",
                        "fingertip_distance_m": _finger_distance(robot),
                        "arm_action_command": actions[0, :5].detach().cpu().tolist(),
                        "arm_joint_positions_rad": arm_joint_pos.detach().cpu().tolist(),
                        "observed_arm_joint_range_rad": (arm_joint_max - arm_joint_min).detach().cpu().tolist(),
                        "viewport_captures_requested": len(capture_futures),
                    }
                )
                _write_status(args_cli.status_file, state)

            remaining = dt - (time.monotonic() - frame_started)
            if remaining > 0:
                time.sleep(remaining)

        state.update(
            {
                "state": "finished",
                "step": step,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
        )
        _write_status(args_cli.status_file, state)
        return 0
    except Exception as exc:
        state.update({"state": "error", "error": repr(exc), "step": step})
        _write_status(args_cli.status_file, state)
        raise
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
