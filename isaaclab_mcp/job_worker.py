"""Subprocess worker that runs one allow-listed visible Dofbot training job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab_mcp.contracts import (
    ArtifactReference,
    EnvironmentReference,
    JobMetadata,
    Provenance,
    TrainingConfiguration,
    TrainingResult,
    TrainingRunRecord,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _artifact(kind: str, path: Path, summary: str) -> ArtifactReference:
    return ArtifactReference(kind=kind, path=str(path), sha256=_sha256(path), summary=summary)


def _status(
    job_directory: Path,
    *,
    status: str,
    summary: str,
    state: str,
    manifest: dict[str, Any],
    artifacts: list[str],
    next_actions: list[str],
) -> None:
    _atomic_json(
        job_directory / "job_status.json",
        {
            "status": status,
            "summary": summary,
            "next_actions": next_actions,
            "artifacts": artifacts,
            "job": {
                "id": manifest["job_id"],
                "state": state,
                "directory": str(job_directory),
                "contract_fingerprint": manifest["environment"]["contract_fingerprint"],
            },
        },
    )


def _cancel_owned_launcher(process: subprocess.Popen[bytes]) -> None:
    """Stop only this worker's launcher process tree after a cancellation request."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _find_run_directory(manifest: dict[str, Any], started_at: float) -> Path | None:
    log_root = _PROJECT_ROOT / "logs" / "rsl_rl" / manifest["experiment_name"]
    if not log_root.is_dir():
        return None
    candidates = [
        path
        for path in log_root.iterdir()
        if path.is_dir() and path.name.endswith(f"_{manifest['run_name']}") and path.stat().st_mtime >= started_at - 60
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _complete(job_directory: Path, manifest: dict[str, Any], run_directory: Path, training_log: Path) -> None:
    videos = sorted((run_directory / "videos" / "train").glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    checkpoints = sorted(run_directory.glob("model_*.pt"), key=lambda path: path.stat().st_mtime)
    if not videos or not checkpoints:
        _status(
            job_directory,
            status="error",
            summary="Training launcher exited successfully but required checkpoint or MP4 evidence is missing.",
            state="failed",
            manifest=manifest,
            artifacts=[str(training_log), str(run_directory)],
            next_actions=["Inspect the training log and rerun; do not claim training success."],
        )
        return

    checkpoint = _artifact("checkpoint", checkpoints[-1], "Final Dofbot checkpoint")
    video_artifacts = [_artifact("video", path, "Visible Isaac Lab RecordVideo clip") for path in videos]
    log_artifact = _artifact("log", training_log, "Dofbot training launcher output")
    record = TrainingRunRecord(
        run_id=manifest["job_id"],
        environment=EnvironmentReference(**manifest["environment"]),
        task_id=manifest["task_id"],
        config=TrainingConfiguration(algorithm="rsl_rl_ppo", seed=manifest["seed"], num_envs=manifest["num_envs"]),
        job=JobMetadata(job_id=manifest["job_id"], status="completed"),
        result=TrainingResult(checkpoint=checkpoint, metrics={"recorded_video_count": float(len(videos))}),
        artifacts=[log_artifact, *video_artifacts],
        provenance=Provenance(
            source_commit=manifest["source_commit"],
            task_version="dofbot-runtime-v1",
            created_at=datetime.now(timezone.utc),
        ),
    )
    record_path = job_directory / "training_run_record.json"
    _atomic_json(record_path, record.model_dump(mode="json"))
    _status(
        job_directory,
        status="success",
        summary="Training completed with checkpoint and visible MP4 evidence.",
        state="completed",
        manifest=manifest,
        artifacts=[str(record_path), str(checkpoints[-1]), *(str(path) for path in videos), str(training_log)],
        next_actions=["Send training_run_record.json to the Verification Agent for independent evaluation."],
    )


def run_job(job_directory: Path) -> int:
    manifest_path = job_directory / "job_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 2

    training_log = job_directory / "training.log"
    if (job_directory / "cancel.request").exists():
        _status(
            job_directory,
            status="warning",
            summary="Training job was cancelled before the launcher started.",
            state="cancelled",
            manifest=manifest,
            artifacts=[str(training_log)],
            next_actions=["Review the contract and submit a new run when ready."],
        )
        return 0

    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(_PROJECT_ROOT / "scripts" / "train_dofbot_with_video.ps1"),
        "-Task",
        manifest["task_id"],
        "-NumEnvs",
        str(manifest["num_envs"]),
        "-MaxIterations",
        str(manifest["max_iterations"]),
        "-RunName",
        manifest["run_name"],
        "-Seed",
        str(manifest["seed"]),
        "-VideoLength",
        str(manifest["video_length"]),
        "-VideoInterval",
        str(manifest["video_interval"]),
    ]
    started_at = time.time()
    with training_log.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=_PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            if (job_directory / "cancel.request").exists():
                _cancel_owned_launcher(process)
                _status(
                    job_directory,
                    status="warning",
                    summary="Training job was cancelled by the worker that owns the launcher process.",
                    state="cancelled",
                    manifest=manifest,
                    artifacts=[str(training_log)],
                    next_actions=["Review the log and submit a new run when ready."],
                )
                return 0
            time.sleep(0.5)
        exit_code = process.returncode

    if exit_code != 0:
        _status(
            job_directory,
            status="error",
            summary=f"Training launcher exited with code {exit_code}.",
            state="failed",
            manifest=manifest,
            artifacts=[str(training_log)],
            next_actions=["Inspect the training log, fix the reported issue, then submit a new run."],
        )
        return exit_code

    run_directory = _find_run_directory(manifest, started_at)
    if run_directory is None:
        _status(
            job_directory,
            status="error",
            summary="Training launcher exited successfully but its run directory could not be resolved.",
            state="failed",
            manifest=manifest,
            artifacts=[str(training_log)],
            next_actions=["Inspect the training log and do not claim a completed run."],
        )
        return 1
    _complete(job_directory, manifest, run_directory, training_log)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-directory", required=True)
    arguments = parser.parse_args()
    raise SystemExit(run_job(Path(arguments.job_directory).resolve(strict=False)))


if __name__ == "__main__":
    main()
