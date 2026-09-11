"""Register the Unitree G1 moving-person chase task."""

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Chase-Person-G1-v0"
PLAY_TASK_ID = "Isaac-Chase-Person-G1-Play-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1PeopleChaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PeopleChasePPORunnerCfg",
    },
)

gym.register(
    id=PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1PeopleChaseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1PeopleChasePPORunnerCfg",
    },
)

__all__ = ["PLAY_TASK_ID", "TASK_ID"]
