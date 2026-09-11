"""Gym registration for the TM6S lift-proxy task."""

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Reach-TM6S-Lift-Proxy-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TM6SLiftProxyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TM6SLiftProxyPPORunnerCfg",
    },
)

__all__ = ["TASK_ID"]
