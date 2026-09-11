"""Launch Isaac Lab and report the default Dofbot articulation metadata."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
parser.add_argument("--steps", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

DOFBOT_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots/Yahboom/Dofbot/dofbot.usd"


def main() -> int:
    simulation = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    config = ArticulationCfg(
        prim_path="/World/Dofbot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DOFBOT_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={"joint1": 0.0, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0}
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-4]"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=10000.0,
                damping=100.0,
            )
        },
    )
    robot = Articulation(config)
    simulation.reset()

    for _ in range(args_cli.steps):
        robot.set_joint_position_target_index(robot.data.default_joint_pos.torch)
        robot.write_data_to_sim()
        simulation.step()
        robot.update(simulation.get_physics_dt())

    result = {
        "usd_path": DOFBOT_USD_PATH,
        "device": str(robot.device),
        "joint_names": robot.joint_names,
        "body_names": robot.body_names,
        "body_positions_world_m": {
            name: robot.data.body_pos_w.torch[0, index].detach().cpu().tolist()
            for index, name in enumerate(robot.body_names)
        },
        "default_joint_pos_rad": robot.data.default_joint_pos.torch[0].detach().cpu().tolist(),
        "soft_joint_pos_limits_rad": robot.data.soft_joint_pos_limits.torch[0].detach().cpu().tolist(),
        "finite_state": bool(
            torch.isfinite(robot.data.joint_pos.torch).all() and torch.isfinite(robot.data.body_pos_w.torch).all()
        ),
        "steps": args_cli.steps,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args_cli.output:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(rendered + "\n", encoding="utf-8")
    print("DOFBOT_PROBE=" + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
