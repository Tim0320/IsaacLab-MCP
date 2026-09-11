"""Inspect the local Isaac Lab registrations used by the G1 people RL plan."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
parser.add_argument("--hold-seconds", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "headless", False):
    parser.error("This project requires a visible Kit window; --headless is not allowed.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from isaaclab_mcp.runtime_tasks import register_tasks

TASKS = {
    "run": "Isaac-Velocity-Flat-G1-v0",
    "pick": "Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
    "chase": "Isaac-Chase-Person-G1-v0",
    "carry": "Isaac-Carry-Box-G1-v0",
    "carry_run": "Isaac-Run-Carry-Box-G1-v0",
}

register_tasks()


def inspect_task(task_id: str) -> dict[str, object]:
    """Return registration and learning-entry-point evidence for one task."""
    if task_id not in gym.registry:
        return {
            "task_id": task_id,
            "registered": False,
            "has_rewards": False,
            "rsl_rl_entry_point": None,
            "robomimic_entry_point": None,
        }

    spec = gym.spec(task_id)
    cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    return {
        "task_id": task_id,
        "registered": True,
        "entry_point": str(spec.entry_point),
        "env_cfg": spec.kwargs.get("env_cfg_entry_point"),
        "has_rewards": getattr(cfg, "rewards", None) is not None,
        "rsl_rl_entry_point": spec.kwargs.get("rsl_rl_cfg_entry_point"),
        "robomimic_entry_point": spec.kwargs.get("robomimic_bc_cfg_entry_point"),
        "num_envs": cfg.scene.num_envs,
        "episode_length_s": cfg.episode_length_s,
    }


def main() -> int:
    result = {
        "runtime": "Isaac Lab 3.0.0 / Isaac Sim 6.0.1",
        "visible": True,
        "skills": {name: inspect_task(task_id) for name, task_id in TASKS.items()},
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args_cli.output:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(rendered + "\n", encoding="utf-8")
    print("PEOPLE_RL_INSPECTION=" + json.dumps(result, ensure_ascii=False), flush=True)
    if args_cli.hold_seconds > 0:
        deadline = time.monotonic() + args_cli.hold_seconds
        while simulation_app.is_running() and time.monotonic() < deadline:
            simulation_app.update()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
