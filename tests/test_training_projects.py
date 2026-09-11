import json

from isaaclab_mcp.settings import Settings
from isaaclab_mcp.training_design import design_lifting_training
from isaaclab_mcp.training_projects import create_training_project, validate_training_project


def _settings(training_root):
    return Settings(
        isaaclab_path=None,
        isaac_sim_path=None,
        training_root=training_root,
        transport="stdio",
        http_host="127.0.0.1",
        http_port=8010,
    )


def test_preview_does_not_write_files(tmp_path):
    design = design_lifting_training("preview-lift", "Lift a payload.")

    result = create_training_project(design.model_dump(mode="json"), preview=True, settings=_settings(tmp_path))

    assert result["ok"] is True
    assert result["preview"] is True
    assert not (tmp_path / "preview-lift").exists()


def test_create_and_validate_draft_project(tmp_path):
    design = design_lifting_training("draft-lift", "Lift a payload.")

    created = create_training_project(design.model_dump(mode="json"), preview=False, settings=_settings(tmp_path))
    validated = validate_training_project("draft-lift", settings=_settings(tmp_path))

    assert created["ok"] is True
    assert {path.name for path in (tmp_path / "draft-lift").iterdir()} == {
        "README.md",
        "asset_manifest.json",
        "rl_design.json",
        "training_spec.json",
    }
    assert validated["structure_valid"] is True
    assert validated["schema_valid"] is True
    assert validated["training_ready"] is False
    assert (
        json.loads((tmp_path / "draft-lift" / "training_spec.json").read_text(encoding="utf-8"))["task_id"]
        == "Isaac-DraftLift-v0"
    )


def test_existing_project_requires_explicit_overwrite(tmp_path):
    design = design_lifting_training("protected-lift", "Lift a payload.")
    settings = _settings(tmp_path)
    create_training_project(design.model_dump(mode="json"), preview=False, settings=settings)

    result = create_training_project(design.model_dump(mode="json"), preview=False, settings=settings)

    assert result["ok"] is False
    assert "overwrite=true" in result["error"]
