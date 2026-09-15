"""Worker for one visible, recorded Dofbot checkpoint evaluation."""

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
    EvidenceBundle,
    EvidenceCriterion,
    EvidenceTarget,
    Provenance,
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
    directory: Path,
    manifest: dict[str, Any],
    *,
    status: str,
    summary: str,
    state: str,
    verdict: str | None,
    artifacts: list[str],
    next_actions: list[str],
) -> None:
    _atomic_json(
        directory / "evaluation_status.json",
        {
            "status": status,
            "summary": summary,
            "next_actions": next_actions,
            "artifacts": artifacts,
            "evaluation": {
                "id": manifest["evaluation_id"],
                "state": state,
                "verdict": verdict,
                "directory": str(directory),
                "training_job_id": manifest["training_job_id"],
            },
        },
    )


def _cancel_owned_process(process: subprocess.Popen[bytes]) -> None:
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


def _criterion(criterion_id: str, passed: bool, summary: str) -> EvidenceCriterion:
    return EvidenceCriterion(criterion_id=criterion_id, result="PASS" if passed else "FAIL", summary=summary)


def _criteria(manifest: dict[str, Any], metrics: dict[str, Any]) -> list[EvidenceCriterion]:
    held_rate = float(metrics.get("held_grasp_success_rate", 0.0))
    attempts = int(metrics.get("grasp_attempt_count", 0))
    close_near = float(metrics.get("close_commands_near_cube_fraction", 0.0))
    criteria = [
        _criterion(
            "held_grasp_rate", held_rate >= 0.75, f"Held grasp success rate {held_rate:.3f}; required >= 0.750."
        ),
        _criterion("grasp_attempt", attempts > 0, f"Observed {attempts} debounced grasp attempts; required > 0."),
        _criterion(
            "close_near_cube",
            close_near >= 0.80,
            f"Close commands issued near cube {close_near:.3f}; required >= 0.800.",
        ),
    ]
    if manifest["task_stage"] == "lift":
        num_envs = max(int(metrics.get("num_envs", 0)), 1)
        retained = int(metrics.get("environments_lifted_2cm_and_retained", 0))
        retained_rate = retained / num_envs
        maximum_height = float(metrics.get("maximum_cube_height_m", 0.0))
        criteria.extend(
            [
                _criterion(
                    "lift_retained_rate",
                    retained_rate >= 0.75,
                    f"Lift-and-retain rate {retained_rate:.3f}; required >= 0.750.",
                ),
                _criterion(
                    "cube_height",
                    maximum_height >= 0.085,
                    f"Maximum cube height {maximum_height:.3f} m; required >= 0.085 m.",
                ),
            ]
        )
    return criteria


def _validate_inputs(directory: Path, manifest: dict[str, Any]) -> str | None:
    supported = {
        "Isaac-Grasp-Cube-Dofbot-v0": "grasp",
        "Isaac-Lift-Cube-Dofbot-v0": "lift",
    }
    required = {
        "evaluation_id",
        "source_commit",
        "training_job_id",
        "training_run_id",
        "training_record_path",
        "task_id",
        "task_stage",
        "num_envs",
        "steps",
        "seed",
        "video_length",
        "checkpoint",
        "environment",
    }
    if not required.issubset(manifest):
        return "Evaluation manifest is missing required fields."
    if supported.get(manifest["task_id"]) != manifest["task_stage"]:
        return "Evaluation manifest task is not allow-listed or its stage does not match."
    if not isinstance(manifest["num_envs"], int) or not 1 <= manifest["num_envs"] <= 64:
        return "Evaluation manifest num_envs is outside the allowed range."
    if not isinstance(manifest["seed"], int) or not 0 <= manifest["seed"] <= 2_147_483_647:
        return "Evaluation manifest seed is outside the allowed range."
    if (
        not isinstance(manifest["steps"], int)
        or not isinstance(manifest["video_length"], int)
        or not 500 <= manifest["video_length"] <= 1000
        or manifest["steps"] < manifest["video_length"]
    ):
        return "Evaluation manifest cannot produce the required 10-20 second video."

    training_root = directory.parent.parent.resolve(strict=False)
    expected_record = (
        training_root / "runtime_jobs" / manifest["training_job_id"] / "training_run_record.json"
    ).resolve(strict=False)
    if Path(manifest["training_record_path"]).resolve(strict=False) != expected_record or not expected_record.is_file():
        return "Evaluation manifest does not reference the expected TrainingRunRecord."
    try:
        record = TrainingRunRecord.model_validate_json(expected_record.read_text(encoding="utf-8"))
        checkpoint = ArtifactReference.model_validate(manifest["checkpoint"])
        environment = EnvironmentReference.model_validate(manifest["environment"])
    except (OSError, ValueError):
        return "Evaluation inputs did not pass protocol schema validation."
    if (
        record.job.status != "completed"
        or record.run_id != manifest["training_run_id"]
        or record.job.job_id != manifest["training_job_id"]
        or record.task_id != manifest["task_id"]
        or record.environment != environment
        or record.result is None
        or record.result.checkpoint != checkpoint
    ):
        return "Evaluation inputs do not match the completed TrainingRunRecord."
    checkpoint_path = Path(checkpoint.path).resolve(strict=False)
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != checkpoint.sha256:
        return "Checkpoint is missing or its hash changed after evaluation submission."
    return None


def _complete(
    directory: Path,
    manifest: dict[str, Any],
    metrics_path: Path,
    evaluation_log: Path,
    videos: list[Path],
) -> None:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _status(
            directory,
            manifest,
            status="error",
            summary="Evaluation exited successfully but metrics JSON is missing or malformed.",
            state="failed",
            verdict=None,
            artifacts=[str(evaluation_log), str(metrics_path)],
            next_actions=["Inspect evaluation.log and rerun; do not infer policy quality."],
        )
        return
    metrics_match_manifest = (
        metrics.get("task_id") == manifest["task_id"]
        and metrics.get("num_envs") == manifest["num_envs"]
        and metrics.get("steps") == manifest["steps"]
        and metrics.get("seed") == manifest["seed"]
        and metrics.get("video_length") == manifest["video_length"]
    )
    if metrics.get("state") != "finished" or not metrics_match_manifest or not videos:
        _status(
            directory,
            manifest,
            status="error",
            summary="Evaluation did not produce both finished metrics and mandatory MP4 evidence.",
            state="failed",
            verdict=None,
            artifacts=[str(evaluation_log), str(metrics_path), *(str(path) for path in videos)],
            next_actions=["Inspect the evaluation log and rerun; do not issue a PASS verdict."],
        )
        return

    criteria = _criteria(manifest, metrics)
    verdict = "PASS" if all(item.result == "PASS" for item in criteria) else "FIX_REQUIRED"
    metrics_artifact = _artifact("structured", metrics_path, "Independent Dofbot policy metrics")
    behavioral_artifact = _artifact("behavioral", metrics_path, "Measured grasp and lift behavior")
    video_artifacts = [_artifact("video", path, "Visible Isaac Lab policy-evaluation clip") for path in videos]
    checkpoint = ArtifactReference.model_validate(manifest["checkpoint"])
    bundle = EvidenceBundle(
        bundle_id=manifest["evaluation_id"],
        target=EvidenceTarget(
            environment=EnvironmentReference.model_validate(manifest["environment"]),
            training_run_id=manifest["training_run_id"],
            checkpoint=checkpoint,
        ),
        structured_evidence=[metrics_artifact],
        visual_evidence=video_artifacts,
        behavioral_evidence=[behavioral_artifact],
        training_evidence=[checkpoint],
        criteria=criteria,
        final_verdict=verdict,
        provenance=Provenance(
            source_commit=manifest["source_commit"],
            task_version="dofbot-policy-evaluation-v1",
            created_at=datetime.now(timezone.utc),
        ),
    )
    bundle_path = directory / "evidence_bundle.json"
    _atomic_json(bundle_path, bundle.model_dump(mode="json"))
    _status(
        directory,
        manifest,
        status="success" if verdict == "PASS" else "warning",
        summary=(
            "Policy evaluation passed every defined criterion."
            if verdict == "PASS"
            else "Policy evaluation completed, but one or more behavioral criteria failed."
        ),
        state="completed",
        verdict=verdict,
        artifacts=[str(bundle_path), str(metrics_path), *(str(path) for path in videos), str(evaluation_log)],
        next_actions=(
            ["Send the EvidenceBundle to the Audit & Knowledge Agent."]
            if verdict == "PASS"
            else ["Inspect failed criteria, revise training or environment inputs, then run a new evaluation."]
        ),
    )


def run_evaluation(directory: Path) -> int:
    manifest_path = directory / "evaluation_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 2

    evaluation_log = directory / "evaluation.log"
    metrics_path = directory / "evaluation_metrics.json"
    video_dir = directory / "videos"
    validation_error = _validate_inputs(directory, manifest)
    if validation_error is not None:
        _status(
            directory,
            manifest,
            status="error",
            summary=validation_error,
            state="failed",
            verdict=None,
            artifacts=[str(manifest_path)],
            next_actions=["Restore the original verified inputs or submit a new evaluation."],
        )
        return 2
    if (directory / "cancel.request").exists():
        _status(
            directory,
            manifest,
            status="warning",
            summary="Policy evaluation was cancelled before Isaac Sim started.",
            state="cancelled",
            verdict=None,
            artifacts=[str(evaluation_log)],
            next_actions=["Start a new evaluation when ready."],
        )
        return 0

    command = [
        r"C:\isaacsim\python.bat",
        str(_PROJECT_ROOT / "scripts" / "evaluate_dofbot_grasp_policy.py"),
        "--checkpoint",
        manifest["checkpoint"]["path"],
        "--task-stage",
        manifest["task_stage"],
        "--num-envs",
        str(manifest["num_envs"]),
        "--steps",
        str(manifest["steps"]),
        "--seed",
        str(manifest["seed"]),
        "--video-length",
        str(manifest["video_length"]),
        "--video-dir",
        str(video_dir),
        "--output",
        str(metrics_path),
        "--device",
        "cuda:0",
        "--experience",
        str(_PROJECT_ROOT / "apps" / "isaaclab.dofbot.demo.kit"),
        "--kit_args",
        "--/renderer/multiGpu/enabled=false --/physics/cudaDevice=0",
    ]
    with evaluation_log.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=_PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            if (directory / "cancel.request").exists():
                _cancel_owned_process(process)
                _status(
                    directory,
                    manifest,
                    status="warning",
                    summary="Policy evaluation was cancelled by its owning worker.",
                    state="cancelled",
                    verdict=None,
                    artifacts=[str(evaluation_log)],
                    next_actions=["Review the log and start a new evaluation when ready."],
                )
                return 0
            time.sleep(0.5)
        exit_code = process.returncode

    if exit_code != 0:
        _status(
            directory,
            manifest,
            status="error",
            summary=f"Policy evaluator exited with code {exit_code}.",
            state="failed",
            verdict=None,
            artifacts=[str(evaluation_log), str(metrics_path)],
            next_actions=["Inspect evaluation.log, fix the reported issue, then start a new evaluation."],
        )
        return exit_code

    videos = sorted(video_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    _complete(directory, manifest, metrics_path, evaluation_log, videos)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-directory", required=True)
    arguments = parser.parse_args()
    raise SystemExit(run_evaluation(Path(arguments.evaluation_directory).resolve(strict=False)))


if __name__ == "__main__":
    main()
