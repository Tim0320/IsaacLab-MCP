"""Runtime-only moving-person command loaded after SimulationApp starts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand
from isaaclab.utils.math import quat_from_euler_xyz, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .mdp import MovingPersonCommandCfg


class MovingPersonCommand(UniformPose2dCommand):
    """Move one visual person target around each environment on a circular path."""

    cfg: MovingPersonCommandCfg

    def __init__(self, cfg: MovingPersonCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._last_dt = 0.0

    @property
    def command(self) -> torch.Tensor:
        """Convert target pose error into the velocity command expected by G1 locomotion."""
        target_xy_b = self.pos_command_b[:, :2]
        distance = torch.linalg.vector_norm(target_xy_b, dim=1)
        direction = target_xy_b / torch.clamp(distance.unsqueeze(1), min=1.0e-6)
        speed = torch.clamp(
            (distance - self.cfg.desired_distance_m) * self.cfg.distance_gain + self.cfg.closing_speed_mps,
            min=0.0,
            max=self.cfg.max_forward_speed_mps,
        )

        forward = torch.clamp(direction[:, 0] * speed, min=0.0, max=self.cfg.max_forward_speed_mps)
        lateral = torch.clamp(
            direction[:, 1] * speed,
            min=-self.cfg.max_lateral_speed_mps,
            max=self.cfg.max_lateral_speed_mps,
        )
        yaw_rate = torch.clamp(
            self.cfg.heading_control_stiffness * self.heading_command_b,
            min=-self.cfg.max_yaw_rate_rps,
            max=self.cfg.max_yaw_rate_rps,
        )
        return torch.stack([forward, lateral, yaw_rate], dim=1)

    def compute(self, dt: float):
        self._last_dt = dt
        super().compute(dt)

    def _update_metrics(self):
        """Report contact-based success while retaining distance and heading diagnostics."""
        self.metrics["error_pos"] = torch.linalg.vector_norm(
            self.pos_command_w[:, :2] - self.robot.data.root_pos_w.torch[:, :2], dim=1
        )
        self.metrics["error_heading"] = torch.abs(
            wrap_to_pi(self.heading_command_w - self.robot.data.heading_w.torch)
        )
        contact_sensor = self._env.scene.sensors[self.cfg.contact_sensor_name]
        contact_force = torch.linalg.vector_norm(contact_sensor.data.net_forces_w.torch, dim=-1)
        contact_force = contact_force.flatten(start_dim=1).amax(dim=1)
        self.metrics["contact_force"] = contact_force
        if self._track_success:
            self._succeeded |= contact_force >= self.cfg.contact_force_threshold_n

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Capture contact termination before the parent reset clears episode state."""
        if self._track_success:
            selected_env_ids = slice(None) if env_ids is None else env_ids
            contact_done = self._env.termination_manager.get_term("person_contact")
            self._succeeded[selected_env_ids] |= contact_done[selected_env_ids]
        return super().reset(env_ids=env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        count = len(env_ids)
        self._phase[env_ids] = torch.empty(count, device=self.device).uniform_(0.0, 2.0 * torch.pi)
        self._write_target_pose(env_ids)

    def _update_command(self):
        if self._last_dt > 0:
            self._phase += self.cfg.target_speed_mps / self.cfg.orbit_radius_m * self._last_dt
            self._phase %= 2.0 * torch.pi
        self._write_target_pose(slice(None))
        super()._update_command()

    def _write_target_pose(self, env_ids: Sequence[int] | slice):
        origins = self._env.scene.env_origins[env_ids]
        phase = self._phase[env_ids]
        self.pos_command_w[env_ids, 0] = origins[:, 0] + self.cfg.orbit_radius_m * torch.cos(phase)
        self.pos_command_w[env_ids, 1] = origins[:, 1] + self.cfg.orbit_radius_m * torch.sin(phase)
        self.pos_command_w[env_ids, 2] = origins[:, 2]

        robot_pos = self.robot.data.root_pos_w.torch[env_ids, :2]
        target_delta = self.pos_command_w[env_ids, :2] - robot_pos
        self.heading_command_w[env_ids] = torch.atan2(target_delta[:, 1], target_delta[:, 0])
        self._write_collision_proxy_pose(env_ids, phase)

    def _write_collision_proxy_pose(self, env_ids: Sequence[int] | slice, phase: torch.Tensor):
        """Keep the kinematic collision capsule aligned with the visual person."""
        person_proxy = self._env.scene[self.cfg.collision_proxy_asset_name]
        person_heading = wrap_to_pi(phase + 0.5 * torch.pi)
        proxy_position = self.pos_command_w[env_ids, :3].clone()
        proxy_position[:, 2] += self.cfg.collision_proxy_center_height_m
        proxy_orientation = quat_from_euler_xyz(
            torch.zeros_like(person_heading),
            torch.zeros_like(person_heading),
            person_heading,
        )
        proxy_pose = torch.cat((proxy_position, proxy_orientation), dim=1)
        if isinstance(env_ids, slice):
            person_proxy.write_root_pose_to_sim_index(root_pose=proxy_pose)
        else:
            person_proxy.write_root_pose_to_sim_index(root_pose=proxy_pose, env_ids=env_ids)

    def _debug_vis_callback(self, event):
        person_heading = wrap_to_pi(self._phase + 0.5 * torch.pi)
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(person_heading),
                torch.zeros_like(person_heading),
                person_heading,
            ),
        )
