"""Restricted, durable job control for visible Dofbot training runs."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from isaaclab_mcp.contracts import EnvironmentContract
from isaaclab_mcp.settings import Settings, load_settings

DOFBOT_TASKS = {
    "Isaac-Grasp-Cube-Dofbot-v0": "dofbot_cube_grasp_pretrain",
    "Isaac-Lift-Cube-Dofbot-v0": "dofbot_cube_lift",
}
_JOB_ID = re.compile(r"^dofbot-[a-f0-9]{12}$")
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_VIDEO_LENGTH = 750
_VIDEO_INTERVAL = 1000
_STEPS_PER_ITERATION = 32


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _job_root(settings: Settings) -> Path:
    return (settings.training_root.resolve(strict=False) / "runtime_jobs").resolve(strict=False)


def _job_directory(job_id: str, settings: Settings) -> Path | None:
    if not _JOB_ID.fullmatch(job_id):
        return None
    root = _job_root(settings)
    directory = (root / job_id).resolve(strict=False)
    return directory if directory.is_relative_to(root) else None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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


def _has_dofbot_asset(contract: EnvironmentContract) -> bool:
    return any(
        asset.asset_path.replace("\\", "/").endswith("Robots/Yahboom/Dofbot/dofbot.usd") for asset in contract.assets
    )


def submit_dofbot_training_run(
    environment_contract: dict[str, Any],
    *,
    task_id: Literal["Isaac-Grasp-Cube-Dofbot-v0", "Isaac-Lift-Cube-Dofbot-v0"],
    run_name: str,
    num_envs: int = 4,
    max_iterations: int = 40,
    seed: int = 42,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Start only an allow-listed visible Dofbot runner and persist its input."""
    if task_id not in DOFBOT_TASKS:
        return _error("Unsupported task for the Dofbot job runner.", ["Use one of the documented Dofbot task IDs."])
    if not _RUN_NAME.fullmatch(run_name):
        return _error(
            "run_name must contain only letters, digits, underscores, or hyphens.",
            ["Choose a safe run_name without paths or spaces."],
        )
    if not 1 <= num_envs <= 64:
        return _error("num_envs must be between 1 and 64.", ["Start with 1-4 environments before a larger run."])
    minimum_iterations = math.ceil(_VIDEO_LENGTH / _STEPS_PER_ITERATION)
    if not minimum_iterations <= max_iterations <= 100_000:
        return _error(
            f"max_iterations must be between {minimum_iterations} and 100000 to produce the required recording.",
            ["Increase max_iterations or keep the default 40."],
        )
    if not 0 <= seed <= 2_147_483_647:
        return _error("seed must be a non-negative 32-bit integer.", ["Use a seed between 0 and 2147483647."])

    try:
        contract = EnvironmentContract.model_validate(environment_contract)
    except ValidationError:
        return _error(
            "EnvironmentContract did not pass schema validation.",
            ["Validate the contract first, then submit the same document without changes."],
        )
    if not _has_dofbot_asset(contract):
        return _error(
            "EnvironmentContract does not identify the allow-listed Dofbot asset.",
            ["Submit a contract whose asset list includes Robots/Yahboom/Dofbot/dofbot.usd."],
        )

    resolved = settings or load_settings()
    source_commit = _source_commit()
    if source_commit is None:
        return _error(
            "Current Git commit could not be resolved; refusing to create unverifiable provenance.",
            ["Restore Git access, then submit a new run."],
        )
    job_id = f"dofbot-{uuid.uuid4().hex[:12]}"
    job_directory = _job_directory(job_id, resolved)
    assert job_directory is not None
    job_directory.mkdir(parents=True, exist_ok=False)
    contract_fingerprint = contract.fingerprint()
    submitted_at = _timestamp()
    contract_path = job_directory / "environment_contract.json"
    manifest_path = job_directory / "job_manifest.json"
    status_path = job_directory / "job_status.json"
    contract_path.write_text(
        json.dumps(contract.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "job_id": job_id,
        "submitted_at": submitted_at,
        "source_commit": source_commit,
        "task_id": task_id,
        "experiment_name": DOFBOT_TASKS[task_id],
        "run_name": run_name,
        "num_envs": num_envs,
        "max_iterations": max_iterations,
        "seed": seed,
        "video_length": _VIDEO_LENGTH,
        "video_interval": _VIDEO_INTERVAL,
        "environment": {
            "environment_id": contract.environment_id,
            "contract_version": contract.contract_version,
            "contract_fingerprint": contract_fingerprint,
        },
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        status_path,
        {
            "status": "success",
            "summary": "Visible Dofbot training job has been submitted.",
            "next_actions": ["Poll get_training_run_status until the job reaches a terminal state."],
            "artifacts": [str(contract_path), str(manifest_path), str(job_directory / "training.log")],
            "job": {
                "id": job_id,
                "state": "running",
                "directory": str(job_directory),
                "contract_fingerprint": contract_fingerprint,
            },
        },
    )

    command = [sys.executable, "-m", "isaaclab_mcp.job_worker", "--job-directory", str(job_directory)]
    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NEW_CONSOLE
    try:
        with (job_directory / "worker.log").open("ab") as output:
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
                "summary": "The Dofbot worker could not be started.",
                "next_actions": ["Check the job worker log and the local Python environment before retrying."],
                "artifacts": [str(contract_path), str(manifest_path), str(job_directory / "worker.log")],
                "job": {
                    "id": job_id,
                    "state": "failed",
                    "directory": str(job_directory),
                    "contract_fingerprint": contract_fingerprint,
                },
            },
        )
        return get_training_run_status(job_id, settings=resolved)

    manifest["worker_pid"] = process.pid
    _atomic_json(manifest_path, manifest)
    return get_training_run_status(job_id, settings=resolved)


def get_training_run_status(job_id: str, *, settings: Settings | None = None) -> dict[str, Any]:
    """Read one durable job status without starting or controlling a process."""
    resolved = settings or load_settings()
    job_directory = _job_directory(job_id, resolved)
    if job_directory is None:
        return _error("Invalid training job identifier.", ["Use the job_id returned by submit_dofbot_training_run."])
    status_path = job_directory / "job_status.json"
    if not status_path.is_file():
        return _error("Training job was not found.", ["Use get_training_run_status with a known job_id."])
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _error(
            "Training job status could not be read.",
            ["Inspect the job directory and retry after the worker updates it."],
        )
    required = {
        "status": str,
        "summary": str,
        "next_actions": list,
        "artifacts": list,
        "job": dict,
    }
    if not isinstance(status, dict) or any(
        not isinstance(status.get(key), value_type) for key, value_type in required.items()
    ):
        return _error("Training job status is malformed.", ["Inspect the job directory; do not infer job completion."])
    return status


def cancel_training_run(job_id: str, *, settings: Settings | None = None) -> dict[str, Any]:
    """Request cancellation through a worker-owned control file, never by arbitrary PID."""
    resolved = settings or load_settings()
    status = get_training_run_status(job_id, settings=resolved)
    if status["status"] == "error":
        return status
    state = status["job"].get("state")
    if state in {"completed", "failed", "cancelled"}:
        return {
            "status": "warning",
            "summary": f"Training job is already {state}; no cancellation was requested.",
            "next_actions": ["Review its artifacts or submit a new run if another attempt is needed."],
            "artifacts": status.get("artifacts", []),
            "job": status["job"],
        }
    job_directory = _job_directory(job_id, resolved)
    assert job_directory is not None
    (job_directory / "cancel.request").write_text(_timestamp() + "\n", encoding="utf-8")
    status["status"] = "warning"
    status["summary"] = "Cancellation requested; the worker will stop its owned training process."
    status["next_actions"] = ["Poll get_training_run_status until state becomes cancelled or failed."]
    status["job"]["state"] = "cancellation_requested"
    _atomic_json(job_directory / "job_status.json", status)
    return status
