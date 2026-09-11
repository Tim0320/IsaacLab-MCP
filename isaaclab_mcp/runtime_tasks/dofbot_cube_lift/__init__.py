"""Gym registration for the Dofbot cube-lift task."""

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Lift-Cube-Dofbot-v0"
GRASP_TASK_ID = "Isaac-Grasp-Cube-Dofbot-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DofbotCubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DofbotCubeLiftPPORunnerCfg",
    },
)

gym.register(
    id=GRASP_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DofbotCubeGraspEnvCfg",
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:DofbotCubeGraspPPORunnerCfg"),
    },
)

__all__ = ["GRASP_TASK_ID", "TASK_ID"]
