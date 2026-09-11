"""Register the Unitree G1 bimanual 15 kg box-carry task."""

import gymnasium as gym

from . import agents

TASK_ID = "Isaac-Carry-Box-G1-v0"
PLAY_TASK_ID = "Isaac-Carry-Box-G1-Play-v0"
RUN_TASK_ID = "Isaac-Run-Carry-Box-G1-v0"
RUN_PLAY_TASK_ID = "Isaac-Run-Carry-Box-G1-Play-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1BoxCarryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1BoxCarryPPORunnerCfg",
    },
)

gym.register(
    id=PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1BoxCarryEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1BoxCarryPPORunnerCfg",
    },
)

gym.register(
    id=RUN_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1BoxCarryRunEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1BoxCarryRunPPORunnerCfg",
    },
)

gym.register(
    id=RUN_PLAY_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1BoxCarryRunEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1BoxCarryRunPPORunnerCfg",
    },
)

__all__ = ["PLAY_TASK_ID", "RUN_PLAY_TASK_ID", "RUN_TASK_ID", "TASK_ID"]
