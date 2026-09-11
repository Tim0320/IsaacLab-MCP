from pathlib import Path

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "isaaclab_mcp" / "runtime_tasks" / "g1_box_carry"


def test_g1_carry_task_has_a_physical_15kg_box_and_bilateral_coupling():
    env_source = (PACKAGE / "env_cfg.py").read_text(encoding="utf-8")
    mdp_source = (PACKAGE / "mdp.py").read_text(encoding="utf-8")

    assert "MassPropertiesCfg(mass=15.0)" in env_source
    assert 'body_names=["left_palm_link", "right_palm_link"]' in env_source
    assert "apply_bimanual_grasp_forces" in env_source
    assert "forces=-forces_on_box" in mdp_source
    assert "box_force = forces_on_box.sum" in mdp_source
    assert "both_attached" in mdp_source
    assert "bimanual_grasp_lost" in env_source
    assert '"break_distance_m": 0.40' in env_source


def test_g1_carry_preserves_locomotion_policy_dimensions_for_warm_start():
    env_source = (PACKAGE / "env_cfg.py").read_text(encoding="utf-8")

    assert "class G1BoxCarryEnvCfg(G1FlatEnvCfg)" in env_source
    assert "self.observations.policy" not in env_source.split("class G1BoxCarryEnvCfg_PLAY")[0]
    assert "joint_pos" not in env_source or "init_state.joint_pos" in env_source
    assert '".*_shoulder_.*_joint": 0.03' in env_source


def test_g1_carry_play_mode_is_visible_and_deterministic():
    env_source = (PACKAGE / "env_cfg.py").read_text(encoding="utf-8")

    assert "self.scene.num_envs = 1" in env_source
    assert 'self.viewer.origin_type = "asset_root"' in env_source
    assert 'self.viewer.asset_name = "robot"' in env_source
    assert "class G1BoxCarryRunEnvCfg(G1BoxCarryEnvCfg)" in env_source
    assert "self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.80)" in env_source
    assert "class G1BoxCarryRunEnvCfg_PLAY(G1BoxCarryRunEnvCfg)" in env_source
    assert "self.commands.base_velocity.ranges.lin_vel_x = (0.60, 0.60)" in env_source


def test_hold_and_run_curricula_have_separate_task_ids_and_logs():
    registry_source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    agent_source = (PACKAGE / "agents" / "rsl_rl_ppo_cfg.py").read_text(encoding="utf-8")

    assert 'RUN_TASK_ID = "Isaac-Run-Carry-Box-G1-v0"' in registry_source
    assert 'RUN_PLAY_TASK_ID = "Isaac-Run-Carry-Box-G1-Play-v0"' in registry_source
    assert 'self.experiment_name = "g1_box_carry_run"' in agent_source


def test_carry_evaluation_launcher_requires_visible_recording():
    launcher_source = (ROOT / "scripts" / "play_g1_carry_with_video.ps1").read_text(encoding="utf-8")

    assert "Isaac-Run-Carry-Box-G1-Play-v0" in launcher_source
    assert "[ValidateRange(500, 1000)]" in launcher_source
    assert "rejects --headless" in launcher_source
    assert "'--video'" in launcher_source


def test_policy_evaluator_measures_carry_speed_and_terminations():
    evaluator_source = (ROOT / "scripts" / "evaluate_g1_box_carry_policy.py").read_text(encoding="utf-8")

    assert "gym.wrappers.RecordVideo" in evaluator_source
    assert "root_lin_vel_b" in evaluator_source
    assert "maximum_continuous_carry_duration_s" in evaluator_source
    assert "termination_counts" in evaluator_source
    assert "rejects --headless" not in evaluator_source
    assert "--headless is not allowed" in evaluator_source
