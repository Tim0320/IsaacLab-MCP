"""Replay the verified Dofbot grasp sequence in a visible Isaac Sim window."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hold-seconds", type=float, default=30.0)
parser.add_argument(
    "--status-file",
    type=Path,
    default=Path("artifacts/dofbot_grasp_demo_status.json"),
)
parser.add_argument(
    "--capture-dir",
    type=Path,
    default=Path("artifacts/dofbot_grasp_demo_captures"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

visible_viz_requested = "--viz=kit" in sys.argv or any(
    value == "--viz" and index + 1 < len(sys.argv) and sys.argv[index + 1] == "kit"
    for index, value in enumerate(sys.argv)
)
if not visible_viz_requested:
    parser.error("This demonstration requires a visible window: pass --viz kit")

physics_gpu = args_cli.device.split(":", maxsplit=1)[1] if ":" in args_cli.device else "0"
args_cli.kit_args = " ".join(
    filter(
        None,
        [
            args_cli.kit_args,
            "--/renderer/multiGpu/enabled=false",
            f"--/physics/cudaDevice={physics_gpu}",
        ],
    )
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

from isaaclab_mcp.runtime_tasks import register_tasks
from isaaclab_mcp.runtime_tasks.dofbot_cube_lift import TASK_ID
from isaaclab_mcp.runtime_tasks.dofbot_cube_lift.env_cfg import DofbotCubeLiftEnvCfg

APPROACH_ACTION = [0.0, 0.53, 0.19, -0.158, -0.759]
GRASP_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0]
LIFT_ACTION = [0.095, 0.791, 0.0, -0.696, -0.037]


def _finger_gap(robot) -> float:
    left_id = robot.find_bodies("Finger_Left_03")[0][0]
    right_id = robot.find_bodies("Finger_Right_03")[0][0]
    delta = robot.data.body_pos_w.torch[0, left_id] - robot.data.body_pos_w.torch[0, right_id]
    return float(torch.linalg.vector_norm(delta).item())


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    register_tasks()
    cfg = DofbotCubeLiftEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device
    cfg.sim.use_fabric = True
    cfg.episode_length_s = 600.0
    cfg.events.reset_object_position.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
    }
    cfg.viewer.eye = (0.58, 0.58, 0.38)
    cfg.viewer.lookat = (0.12, 0.0, 0.09)

    env = gym.make(TASK_ID, cfg=cfg)
    robot = env.unwrapped.scene["robot"]
    cube = env.unwrapped.scene["object"]
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    captures = []
    state = {
        "state": "starting",
        "task_id": TASK_ID,
        "asset": env.unwrapped.cfg.scene.robot.spawn.usd_path,
        "device": str(env.unwrapped.device),
        "cube_mass_kg": float(cube.data.body_mass.torch[0, 0].item()),
        "visible": True,
        "self_collisions_enabled": False,
    }

    def run_phase(name: str, arm_action: list[float], gripper: float, steps: int) -> None:
        actions[:, :5] = torch.tensor(arm_action, device=env.unwrapped.device)
        actions[:, -1] = gripper
        for _ in range(steps):
            env.step(actions)
        cube_height = float(cube.data.root_pos_w.torch[0, 2].item())
        capture_path = args_cli.capture_dir.resolve() / f"{len(captures):02d}_{name}.png"
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        viewport = get_active_viewport()
        if viewport is not None:
            captures.append(capture_viewport_to_file(viewport, file_path=str(capture_path), is_hdr=False))
            for _ in range(5):
                env.step(actions)
        state.update(
            {
                "state": "running",
                "phase": name,
                "cube_height_m": cube_height,
                "fingertip_gap_m": _finger_gap(robot),
                "arm_joint_positions_rad": robot.data.joint_pos.torch[0, :5].detach().cpu().tolist(),
                "capture": str(capture_path),
            }
        )
        _write_status(args_cli.status_file.resolve(), state)
        print("DOFBOT_GRASP_PHASE=" + json.dumps(state, ensure_ascii=False), flush=True)

    try:
        env.reset()
        env.unwrapped.sim.set_camera_view(eye=cfg.viewer.eye, target=cfg.viewer.lookat)
        initial_height = float(cube.data.root_pos_w.torch[0, 2].item())
        run_phase("approach", APPROACH_ACTION, 1.0, 120)
        run_phase("descend", GRASP_ACTION, 1.0, 120)
        run_phase("close", GRASP_ACTION, -1.0, 100)
        closed_height = float(cube.data.root_pos_w.torch[0, 2].item())
        run_phase("lift", LIFT_ACTION, -1.0, 160)
        maximum_height = float(cube.data.root_pos_w.torch[0, 2].item())
        lift_gain = maximum_height - closed_height
        state.update(
            {
                "state": "holding",
                "initial_cube_height_m": initial_height,
                "closed_cube_height_m": closed_height,
                "cube_height_at_lift_end_m": maximum_height,
                "lift_gain_at_lift_end_m": lift_gain,
                "lifted_over_2cm": lift_gain > 0.02,
            }
        )
        _write_status(args_cli.status_file.resolve(), state)
        print("DOFBOT_GRASP_RESULT=" + json.dumps(state, ensure_ascii=False), flush=True)

        hold_started = time.monotonic()
        while simulation_app.is_running() and time.monotonic() - hold_started < args_cli.hold_seconds:
            env.step(actions)
            time.sleep(float(env.unwrapped.step_dt))
        held_height = float(cube.data.root_pos_w.torch[0, 2].item())
        held_gain = held_height - closed_height
        state.update(
            {
                "cube_height_after_hold_m": held_height,
                "lift_gain_after_hold_m": held_gain,
                "lifted_over_2cm": held_gain > 0.02,
            }
        )
        state["state"] = "finished"
        _write_status(args_cli.status_file.resolve(), state)
        return 0 if state["lifted_over_2cm"] else 2
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
