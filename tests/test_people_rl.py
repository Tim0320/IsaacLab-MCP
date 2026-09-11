from isaaclab_mcp.people_rl import design_people_rl_program


def test_people_rl_program_separates_the_three_skills():
    program = design_people_rl_program()
    skills = {skill.name: skill for skill in program.skills}

    assert set(skills) == {"run", "pick", "chase"}
    assert skills["run"].status == "rl_ready"
    assert skills["run"].task_id == "Isaac-Velocity-Flat-G1-v0"
    assert skills["pick"].status == "imitation_ready"
    assert skills["pick"].rewards == []
    assert any("rewards=None" in blocker for blocker in skills["pick"].blockers)
    assert skills["chase"].status == "rl_ready"
    assert program.readiness == {
        "run": True,
        "pick": False,
        "chase": True,
        "integrated_autonomy": False,
    }


def test_people_rl_curriculum_starts_with_locomotion():
    program = design_people_rl_program()

    assert [phase["skill"] for phase in program.curriculum] == ["run", "pick", "chase", "integration"]
    chase = next(skill for skill in program.skills if skill.name == "chase")
    assert "person-contact success rate" in chase.success_metrics
    assert any("1 N" in rule for rule in program.safety_rules)
    assert "physical person-contact success" in program.curriculum[2]["advance_when"]


def test_chase_runtime_requires_physical_contact():
    from pathlib import Path

    package = Path(__file__).parents[1] / "isaaclab_mcp" / "runtime_tasks" / "g1_people_chase"
    env_source = (package / "env_cfg.py").read_text(encoding="utf-8")
    mdp_source = (package / "mdp.py").read_text(encoding="utf-8")
    command_source = (package / "moving_person_command.py").read_text(encoding="utf-8")

    assert "PersonContactProxy" in env_source
    assert "filter_prim_paths_expr" not in env_source
    assert "person_contact_success" in env_source
    assert "terminations.base_contact = None" in env_source
    assert "net_forces_w" in mdp_source
    assert "force_threshold_n" in mdp_source
    assert "write_root_pose_to_sim_index" in command_source
    assert 'self.metrics["contact_force"]' in command_source
    assert "contact_force_threshold_n" in command_source
    assert 'get_term("person_contact")' in command_source


def test_g1_training_launcher_rejects_incomplete_trailing_video():
    from pathlib import Path

    launcher = Path(__file__).parents[1] / "scripts" / "train_g1_run_with_video.ps1"
    launcher_source = launcher.read_text(encoding="utf-8")

    assert "[int]$VideoInterval = 1000000" in launcher_source
    assert "$lastRecordingStart" in launcher_source
    assert "incomplete trailing clip" in launcher_source
    assert "Isaac-Carry-Box-G1-v0" in launcher_source
    assert "Isaac-Run-Carry-Box-G1-v0" in launcher_source
    assert "g1_box_carry" in launcher_source
    assert "g1_box_carry_run" in launcher_source
