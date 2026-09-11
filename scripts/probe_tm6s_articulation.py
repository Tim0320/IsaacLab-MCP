"""Launch Isaac Lab, load TM6S, and report resolved articulation metadata."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext


def main() -> int:
    asset = args_cli.asset.resolve(strict=True)
    trace_path = args_cli.output.with_suffix(".trace.txt") if args_cli.output else None

    def trace(message: str) -> None:
        if trace_path:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")

    trace("start")
    simulation = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 120.0))
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=100.0,
                velocity_limit_sim=2.0,
                stiffness=400.0,
                damping=40.0,
            )
        },
    )
    robot = Articulation(robot_cfg)
    trace("articulation-created")
    simulation.reset()
    trace("simulation-reset")
    trace(f"app-running={simulation_app.is_running()}")

    for step in range(args_cli.steps):
        trace(f"step-{step}-before")
        robot.set_joint_position_target_index(robot.data.default_joint_pos.torch)
        robot.write_data_to_sim()
        simulation.step()
        robot.update(simulation.get_physics_dt())
        trace(f"step-{step}-after")

    tool_indices, tool_names = robot.find_bodies("tool0")
    payload = {
        "asset": str(asset),
        "device": str(robot.device),
        "num_instances": robot.num_instances,
        "joint_names": robot.joint_names,
        "body_names": robot.body_names,
        "tool0_indices": tool_indices,
        "tool0_names": tool_names,
        "joint_positions_rad": robot.data.joint_pos.torch[0].detach().cpu().tolist(),
        "tool0_position_world_m": (
            robot.data.body_pos_w.torch[0, tool_indices[0]].detach().cpu().tolist() if tool_indices else None
        ),
        "finite_state": bool(
            torch.isfinite(robot.data.joint_pos.torch).all() and torch.isfinite(robot.data.body_pos_w.torch).all()
        ),
        "steps": args_cli.steps,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args_cli.output:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(rendered + "\n", encoding="utf-8")
    trace("report-written")
    print("TM6S_PROBE_JSON=" + json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
