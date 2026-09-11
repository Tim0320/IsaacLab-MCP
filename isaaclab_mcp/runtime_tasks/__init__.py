"""Isaac Lab runtime task registration callbacks."""

from __future__ import annotations

import sys


def register_tasks() -> list[str]:
    """Register project-owned Gym tasks for Isaac Lab's external callback hook."""
    from isaaclab_mcp.runtime_tasks import (  # noqa: F401
        dofbot_cube_lift,
        g1_box_carry,
        g1_people_chase,
        tm6s_lift_proxy,
    )

    return sys.argv[1:]


__all__ = ["register_tasks"]
