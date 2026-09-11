"""RSL-RL PPO configuration for the G1 moving-person chase task."""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class G1PeopleChasePPORunnerCfg(G1FlatPPORunnerCfg):
    """Reuse G1 locomotion PPO settings with a separate experiment namespace."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500
        self.save_interval = 50
        self.experiment_name = "g1_people_chase"
