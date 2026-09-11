"""Create and validate non-interactive training design packets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from isaaclab_mcp.settings import Settings, load_settings
from isaaclab_mcp.training_design import LiftingTrainingDesign, asset_reference_status, normalize_project_name

_PROJECT_FILES = ("training_spec.json", "asset_manifest.json", "rl_design.json", "README.md")


def _project_path(project_name: str, settings: Settings) -> Path:
    slug = normalize_project_name(project_name)
    root = settings.training_root.resolve(strict=False)
    project = (root / slug).resolve(strict=False)
    if not project.is_relative_to(root):
        raise ValueError("project path must stay inside ISAACLAB_MCP_TRAINING_ROOT")
    return project


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _render_readme(design: LiftingTrainingDesign) -> str:
    assets = "\n".join(
        f"- `{asset.role}`: `{asset.reference_status}`; reference: `{asset.reference or 'MISSING'}`"
        for asset in design.assets
    )
    rewards = "\n".join(f"- `{term.name}`: weight `{term.weight}`; {term.purpose}" for term in design.rewards)
    missing = "\n".join(f"- {item}" for item in design.missing_requirements) or "- None"
    return f"""# {design.display_name}

Task ID: `{design.task_id}`

## Purpose

{design.use_case}

## Readiness

- Design ready: `{design.readiness["design_ready"]}`
- Asset ready: `{design.readiness["asset_ready"]}`
- Training ready: `{design.readiness["training_ready"]}`

## Assets

{assets}

## Reward and penalty terms

{rewards}

## Missing requirements

{missing}

## Files

- `training_spec.json`: canonical complete design.
- `asset_manifest.json`: asset references and validation status.
- `rl_design.json`: actions, observations, rewards, terminations, randomization, curriculum, PPO, and metrics.

This packet describes the training task. Runtime environment code and long-running training are separate gated steps.
"""


def _project_contents(design: LiftingTrainingDesign) -> dict[str, str]:
    complete = design.model_dump(mode="json")
    return {
        "training_spec.json": _json(complete),
        "asset_manifest.json": _json(
            {
                "schema_version": design.schema_version,
                "project_name": design.project_name,
                "assets": [asset.model_dump(mode="json") for asset in design.assets],
                "missing_requirements": design.missing_requirements,
            }
        ),
        "rl_design.json": _json(
            {
                "schema_version": design.schema_version,
                "project_name": design.project_name,
                "task_id": design.task_id,
                "workflow": design.workflow,
                "simulation": design.simulation,
                "control": design.control,
                "observations": [term.model_dump(mode="json") for term in design.observations],
                "rewards": [term.model_dump(mode="json") for term in design.rewards],
                "terminations": [term.model_dump(mode="json") for term in design.terminations],
                "randomization": [term.model_dump(mode="json") for term in design.randomization],
                "curriculum": design.curriculum,
                "algorithm": design.algorithm.model_dump(mode="json"),
                "evaluation_metrics": design.evaluation_metrics,
            }
        ),
        "README.md": _render_readme(design),
    }


def create_training_project(
    design_data: dict[str, Any],
    preview: bool = True,
    overwrite: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Preview or write one validated training design packet."""
    design = LiftingTrainingDesign.model_validate(design_data)
    resolved = settings or load_settings()
    project = _project_path(design.project_name, resolved)
    contents = _project_contents(design)

    if preview:
        return {
            "ok": True,
            "preview": True,
            "project_directory": str(project),
            "would_write": [str(project / name) for name in contents],
            "training_ready": design.readiness["training_ready"],
            "missing_requirements": design.missing_requirements,
        }

    if project.exists() and not overwrite:
        return {
            "ok": False,
            "preview": False,
            "error": "project already exists; set overwrite=true to replace the known design files",
            "project_directory": str(project),
        }

    project.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in contents.items():
        destination = project / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
        written.append(str(destination))

    return {
        "ok": True,
        "preview": False,
        "project_directory": str(project),
        "written": written,
        "training_ready": design.readiness["training_ready"],
        "missing_requirements": design.missing_requirements,
    }


def validate_training_project(project_name: str, settings: Settings | None = None) -> dict[str, Any]:
    """Validate structure, schema, and current asset references for one generated project."""
    resolved = settings or load_settings()
    project = _project_path(project_name, resolved)
    missing_files = [name for name in _PROJECT_FILES if not (project / name).is_file()]
    if missing_files:
        return {
            "ok": False,
            "project_directory": str(project),
            "structure_valid": False,
            "missing_files": missing_files,
        }

    try:
        data = json.loads((project / "training_spec.json").read_text(encoding="utf-8"))
        design = LiftingTrainingDesign.model_validate(data)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        return {
            "ok": False,
            "project_directory": str(project),
            "structure_valid": True,
            "schema_valid": False,
            "error": str(error),
        }

    assets = []
    for asset in design.assets:
        current_status = asset_reference_status(asset.reference, required=asset.required)
        assets.append({"role": asset.role, "reference": asset.reference, "status": current_status})

    local_required_assets_ready = all(
        asset["status"] == "local_exists"
        for asset in assets
        if next(item for item in design.assets if item.role == asset["role"]).required
    )
    identifiers_ready = bool(design.control.get("lift_joint_name") and design.control.get("end_effector_body_name"))
    load_limit_ready = design.control.get("safe_working_load_kg") is not None
    training_ready = local_required_assets_ready and identifiers_ready and load_limit_ready
    current_missing = []
    for asset in assets:
        requirement = next(item for item in design.assets if item.role == asset["role"])
        if requirement.required and asset["status"] != "local_exists":
            current_missing.append(f"{asset['role']} asset must resolve to a validated local file")
    if not design.control.get("lift_joint_name"):
        current_missing.append("lift_joint_name is required")
    if not design.control.get("end_effector_body_name"):
        current_missing.append("end_effector_body_name is required")
    if not load_limit_ready:
        current_missing.append("safe_working_load_kg is required")

    return {
        "ok": True,
        "project_directory": str(project),
        "structure_valid": True,
        "schema_valid": True,
        "asset_validation": assets,
        "design_ready": True,
        "training_ready": training_ready,
        "missing_requirements": current_missing,
    }
