"""Restricted, durable policy evaluation for completed Dofbot training runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from isaaclab_mcp.contracts import TrainingRunRecord
from isaaclab_mcp.settings import Settings, load_settings

_TRAINING_JOB_ID = re.compile(r"^dofbot-[a-f0-9]{12}$")
_EVALUATION_ID = re.compile(r"^eval-[a-f0-9]{12}$")
_SUPPORTED_TASKS = {
    "Isaac-Grasp-Cube-Dofbot-v0": "grasp",
    "Isaac-Lift-Cube-Dofbot-v0": "lift",
}
_VIDEO_LENGTH = 750


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(summary: str, next_actions: list[str]) -> dict[str, Any]:
    return {"status": "error", "summary": summary, "next_actions": next_actions, "artifacts": []}


def _source_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_project_root(),
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _evaluation_root(settings: Settings) -> Path:
    return (settings.training_root.resolve(strict=False) / "evaluation_jobs").resolve(strict=False)


def _evaluation_directory(evaluation_id: str, settings: Settings) -> Path | None:
    if not _EVALUATION_ID.fullmatch(evaluation_id):
        return None
    root = _evaluation_root(settings)
    directory = (root / evaluation_id).resolve(strict=False)
    return directory if directory.is_relative_to(root) else None


def _training_record_path(job_id: str, settings: Settings) -> Path | None:
    if not _TRAINING_JOB_ID.fullmatch(job_id):
        return None
    root = (settings.training_root.resolve(strict=False) / "runtime_jobs").resolve(strict=False)
    path = (root / job_id / "training_run_record.json").resolve(strict=False)
    return path if path.is_relative_to(root) else None


def _load_verified_training_record(job_id: str, settings: Settings) -> tuple[TrainingRunRecord, Path] | dict[str, Any]:
    record_path = _training_record_path(job_id, settings)
    if record_path is None:
        return _error("Invalid training job identifier.", ["Use a job_id returned by submit_dofbot_training_run."])
    if not record_path.is_file():
        return _error(
            "Completed TrainingRunRecord was not found for this job.",
            ["Wait for get_training_run_status to report completed before evaluation."],
        )
    try:
        record = TrainingRunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return _error(
            "TrainingRunRecord did not pass schema validation.",
            ["Inspect the training job artifacts; do not evaluate an unverified record."],
        )
    if record.run_id != job_id or record.job.job_id != job_id or record.job.status != "completed":
        return _error(
            "TrainingRunRecord is not a completed record for the requested job.",
            ["Use the exact completed job_id returned by the training runner."],
        )
    if record.task_id not in _SUPPORTED_TASKS:
        return _error(
            "Policy evaluation currently supports only the two Dofbot tasks.",
            ["Use a completed Dofbot grasp or lift training job."],
        )
    checkpoint = record.result.checkpoint if record.result is not None else None
    if checkpoint is None:
        return _error("TrainingRunRecord has no checkpoint.", ["Do not claim policy evaluation for this run."])
    checkpoint_path = Path(checkpoint.path).resolve(strict=False)
    if not checkpoint_path.is_file():
        return _error(
            "Recorded checkpoint no longer exists.", ["Restore the exact artifact or submit a new training run."]
        )
    if _sha256(checkpoint_path) != checkpoint.sha256:
        return _error(
            "Checkpoint hash does not match the completed TrainingRunRecord.",
            ["Restore the original checkpoint or submit a new training run; do not evaluate the changed file."],
        )
    return record, record_path


def evaluate_training_run(
    job_id: str,
    *,
    num_envs: int = 4,
    steps: int = 800,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Start a visible, recorded evaluation for one verified training job."""
    if not 1 <= num_envs <= 64:
        return _error("num_envs must be between 1 and 64.", ["Start with 4 environments for evaluation."])
    if not _VIDEO_LENGTH <= steps <= 10_000:
        return _error(
            f"steps must be between {_VIDEO_LENGTH} and 10000 so the mandatory 15-second clip can finish.",
            ["Use the default 800 steps unless a longer behavioral sample is required."],
        )
    resolved = settings or load_settings()
    loaded = _load_verified_training_record(job_id, resolved)
    if isinstance(loaded, dict):
        return loaded
    record, record_path = loaded
    source_commit = _source_commit()
    if source_commit is None:
        return _error(
            "Current Git commit could not be resolved; refusing to create unverifiable evaluation provenance.",
            ["Restore Git access, then submit a new evaluation."],
        )

    evaluation_id = f"eval-{uuid.uuid4().hex[:12]}"
    evaluation_directory = _evaluation_directory(evaluation_id, resolved)
    assert evaluation_directory is not None
    evaluation_directory.mkdir(parents=True, exist_ok=False)
    checkpoint = record.result.checkpoint
    assert checkpoint is not None
    manifest = {
        "schema_version": "1.0",
        "evaluation_id": evaluation_id,
        "submitted_at": _timestamp(),
        "source_commit": source_commit,
        "training_job_id": job_id,
        "training_run_id": record.run_id,
        "training_record_path": str(record_path),
        "task_id": record.task_id,
        "task_stage": _SUPPORTED_TASKS[record.task_id],
        "num_envs": num_envs,
        "steps": steps,
        "seed": 42,
        "video_length": _VIDEO_LENGTH,
        "checkpoint": checkpoint.model_dump(mode="json"),
        "environment": record.environment.model_dump(mode="json"),
    }
    manifest_path = evaluation_directory / "evaluation_manifest.json"
    status_path = evaluation_directory / "evaluation_status.json"
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        status_path,
        {
            "status": "success",
            "summary": "Visible Dofbot policy evaluation has been submitted.",
            "next_actions": ["Poll get_evaluation_status until the evaluation reaches a terminal state."],
            "artifacts": [str(manifest_path), str(evaluation_directory / "evaluation.log")],
            "evaluation": {
                "id": evaluation_id,
                "state": "running",
                "verdict": None,
                "directory": str(evaluation_directory),
                "training_job_id": job_id,
            },
        },
    )

    command = [
        sys.executable,
        "-m",
        "isaaclab_mcp.evaluation_worker",
        "--evaluation-directory",
        str(evaluation_directory),
    ]
    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NEW_CONSOLE
    try:
        with (evaluation_directory / "worker.log").open("ab") as output:
            process = subprocess.Popen(
                command,
                cwd=_project_root(),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
    except OSError:
        _atomic_json(
            status_path,
            {
                "status": "error",
                "summary": "The policy evaluation worker could not be started.",
                "next_actions": ["Check worker.log and the local Python environment before retrying."],
                "artifacts": [str(manifest_path), str(evaluation_directory / "worker.log")],
                "evaluation": {
                    "id": evaluation_id,
                    "state": "failed",
                    "verdict": None,
                    "directory": str(evaluation_directory),
                    "training_job_id": job_id,
                },
            },
        )
        return get_evaluation_status(evaluation_id, settings=resolved)

    manifest["worker_pid"] = process.pid
    _atomic_json(manifest_path, manifest)
    return get_evaluation_status(evaluation_id, settings=resolved)


def get_evaluation_status(evaluation_id: str, *, settings: Settings | None = None) -> dict[str, Any]:
    """Read one durable policy-evaluation status."""
    resolved = settings or load_settings()
    directory = _evaluation_directory(evaluation_id, resolved)
    if directory is None:
        return _error("Invalid evaluation identifier.", ["Use the evaluation_id returned by evaluate_training_run."])
    status_path = directory / "evaluation_status.json"
    if not status_path.is_file():
        return _error("Policy evaluation was not found.", ["Use a known evaluation_id."])
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _error("Policy evaluation status could not be read.", ["Inspect the evaluation directory and retry."])
    required = {"status": str, "summary": str, "next_actions": list, "artifacts": list, "evaluation": dict}
    if not isinstance(status, dict) or any(
        not isinstance(status.get(key), value_type) for key, value_type in required.items()
    ):
        return _error("Policy evaluation status is malformed.", ["Do not infer a verdict from this evaluation."])
    return status


def cancel_evaluation(evaluation_id: str, *, settings: Settings | None = None) -> dict[str, Any]:
    """Request cancellation through a worker-owned control file."""
    resolved = settings or load_settings()
    status = get_evaluation_status(evaluation_id, settings=resolved)
    if status["status"] == "error":
        return status
    state = status["evaluation"].get("state")
    if state in {"completed", "failed", "cancelled"}:
        return {
            "status": "warning",
            "summary": f"Policy evaluation is already {state}; no cancellation was requested.",
            "next_actions": ["Review its evidence or start a new evaluation if needed."],
            "artifacts": status.get("artifacts", []),
            "evaluation": status["evaluation"],
        }
    directory = _evaluation_directory(evaluation_id, resolved)
    assert directory is not None
    (directory / "cancel.request").write_text(_timestamp() + "\n", encoding="utf-8")
    status["status"] = "warning"
    status["summary"] = "Cancellation requested; the worker will stop its owned evaluation process."
    status["next_actions"] = ["Poll get_evaluation_status until state becomes cancelled or failed."]
    status["evaluation"]["state"] = "cancellation_requested"
    _atomic_json(directory / "evaluation_status.json", status)
    return status
