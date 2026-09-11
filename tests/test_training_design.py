import pytest

from isaaclab_mcp.training_design import design_lifting_training, normalize_project_name


def test_lifting_design_contains_complete_rl_information():
    design = design_lifting_training(
        project_name="Fishing Tackle Lift",
        use_case="Lift a payload and hold it at the target height.",
    )

    assert design.project_name == "fishing-tackle-lift"
    assert design.task_id == "Isaac-FishingTackleLift-v0"
    assert design.workflow == "manager_based_single_agent"
    assert {term.name for term in design.observations} >= {"lift_joint_position", "payload_to_target"}
    assert {term.name for term in design.rewards} >= {"target_distance", "success_bonus", "dropped_load"}
    assert {term.name for term in design.terminations} >= {"success", "overload", "timeout"}
    assert design.algorithm.library == "rsl_rl"
    assert design.readiness == {"design_ready": True, "asset_ready": False, "training_ready": False}
    assert len(design.missing_requirements) == 5


def test_existing_assets_and_identifiers_make_design_ready(tmp_path):
    mechanism = tmp_path / "mechanism.usd"
    payload = tmp_path / "payload.usd"
    mechanism.write_text("#usda 1.0\n", encoding="utf-8")
    payload.write_text("#usda 1.0\n", encoding="utf-8")

    design = design_lifting_training(
        project_name="ready-lift",
        use_case="Lift one payload.",
        mechanism_asset_path=str(mechanism),
        payload_asset_path=str(payload),
        lift_joint_name="winch_joint",
        end_effector_body_name="hook",
        safe_working_load_kg=2.0,
    )

    assert design.missing_requirements == []
    assert design.readiness["training_ready"] is True


def test_cable_mode_requires_a_cable_asset(tmp_path):
    mechanism = tmp_path / "mechanism.usd"
    payload = tmp_path / "payload.usd"
    mechanism.write_text("#usda 1.0\n", encoding="utf-8")
    payload.write_text("#usda 1.0\n", encoding="utf-8")

    design = design_lifting_training(
        project_name="cable-lift",
        use_case="Lift with a simulated cable.",
        mechanism_asset_path=str(mechanism),
        payload_asset_path=str(payload),
        lift_joint_name="winch_joint",
        end_effector_body_name="hook",
        safe_working_load_kg=2.0,
        attachment_mode="cable",
    )

    assert any("cable_asset_path" in item for item in design.missing_requirements)
    assert design.readiness["training_ready"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"payload_mass_kg": 0}, "greater than zero"),
        ({"initial_height_m": 1.0, "target_height_m": 0.5}, "greater than initial_height_m"),
        ({"num_envs": 0}, "between 1 and 16384"),
        ({"payload_mass_kg": 2.0, "safe_working_load_kg": 1.0}, "greater than or equal"),
    ],
)
def test_invalid_physical_values_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        design_lifting_training(project_name="invalid", use_case="test", **kwargs)


def test_project_name_rejects_path_traversal():
    with pytest.raises(ValueError, match="ASCII"):
        normalize_project_name("../..")
