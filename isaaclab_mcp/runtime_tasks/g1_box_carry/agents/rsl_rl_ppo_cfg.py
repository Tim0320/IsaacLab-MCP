"""RSL-RL PPO configuration for G1 bimanual 15 kg box carry."""

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class G1BoxCarryPPORunnerCfg(G1FlatPPORunnerCfg):
    """Keep pretrained G1 observation/action dimensions in a carry-specific run."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2000
        self.save_interval = 50
        self.experiment_name = "g1_box_carry"


@configclass
class G1BoxCarryRunPPORunnerCfg(G1BoxCarryPPORunnerCfg):
    """Store the acceleration stage separately from the hold curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_box_carry_run"
