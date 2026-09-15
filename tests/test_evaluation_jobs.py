import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from isaaclab_mcp import evaluation_jobs, evaluation_worker
from isaaclab_mcp.contracts import (
    ArtifactReference,
    EnvironmentReference,
    EvidenceBundle,
    JobMetadata,
    Provenance,
    TrainingConfiguration,
    TrainingResult,
    TrainingRunRecord,
)
from isaaclab_mcp.settings import Settings


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings(root: Path) -> Settings:
    return Settings(
        isaaclab_path=Path(r"D:\IsaacLab"),
        isaac_sim_path=Path(r"C:\isaacsim"),
        training_root=root,
        transport="stdio",
        http_host="127.0.0.1",
        http_port=8010,
    )


def _completed_training_job(root: Path, project_root: Path) -> tuple[str, Path]:
    job_id = "dofbot-abc123def456"
    job_dir = root / "runtime_jobs" / job_id
    job_dir.mkdir(parents=True)
    run_dir = project_root / "logs" / "rsl_rl" / "dofbot_cube_grasp_pretrain" / "run"
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "model_39.pt"
    video_path = run_dir / "training.mp4"
    checkpoint_path.write_bytes(b"checkpoint")
    video_path.write_bytes(b"training-video")
    checkpoint = ArtifactReference(
        kind="checkpoint", path=str(checkpoint_path), sha256=_sha(checkpoint_path), summary="checkpoint"
    )
    record = TrainingRunRecord(
        run_id=job_id,
        environment=EnvironmentReference(environment_id="dofbot", contract_version=1, contract_fingerprint="a" * 64),
        task_id="Isaac-Grasp-Cube-Dofbot-v0",
        config=TrainingConfiguration(algorithm="rsl_rl_ppo", seed=42, num_envs=4),
        job=JobMetadata(job_id=job_id, status="completed"),
        result=TrainingResult(checkpoint=checkpoint),
        artifacts=[ArtifactReference(kind="video", path=str(video_path), sha256=_sha(video_path), summary="video")],
        provenance=Provenance(source_commit="435110b", task_version="v1", created_at=datetime.now(timezone.utc)),
    )
    record_path = job_dir / "training_run_record.json"
    record_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return job_id, record_path


class _FakeWorker:
    command: list[str] | None = None

    def __init__(self, command, **_kwargs):
        type(self).command = command
        self.pid = 12345


def test_evaluate_training_run_uses_verified_checkpoint_and_safe_worker(tmp_path, monkeypatch):
    job_id, _ = _completed_training_job(tmp_path, tmp_path)
    monkeypatch.setattr(evaluation_jobs, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(evaluation_jobs, "_source_commit", lambda: "435110b")
    monkeypatch.setattr(evaluation_jobs.subprocess, "Popen", _FakeWorker)

    result = evaluation_jobs.evaluate_training_run(job_id, num_envs=4, steps=800, settings=_settings(tmp_path))

    assert result["status"] == "success"
    assert result["evaluation"]["state"] == "running"
    assert _FakeWorker.command is not None
    assert _FakeWorker.command[1:3] == ["-m", "isaaclab_mcp.evaluation_worker"]
    assert "--headless" not in _FakeWorker.command
    manifest = json.loads(
        (Path(result["evaluation"]["directory"]) / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["training_run_id"] == job_id
    assert manifest["checkpoint"]["sha256"] == _sha(Path(manifest["checkpoint"]["path"]))


def test_evaluation_rejects_tampered_checkpoint(tmp_path, monkeypatch):
    job_id, record_path = _completed_training_job(tmp_path, tmp_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    Path(record["result"]["checkpoint"]["path"]).write_bytes(b"tampered")
    monkeypatch.setattr(evaluation_jobs, "_project_root", lambda: tmp_path)

    result = evaluation_jobs.evaluate_training_run(job_id, settings=_settings(tmp_path))

    assert result["status"] == "error"
    assert "hash" in result["summary"].lower()
    assert not (tmp_path / "evaluation_jobs").exists()


class _SuccessfulEvaluation:
    def __init__(self, command, **_kwargs):
        self.command = command
        self.returncode = 0
        output_path = Path(command[command.index("--output") + 1])
        video_dir = Path(command[command.index("--video-dir") + 1])
        video_dir.mkdir(parents=True)
        (video_dir / "evaluation.mp4").write_bytes(b"evaluation-video")
        output_path.write_text(
            json.dumps(
                {
                    "state": "finished",
                    "task_id": "Isaac-Grasp-Cube-Dofbot-v0",
                    "num_envs": 4,
                    "steps": 800,
                    "seed": 42,
                    "video_length": 750,
                    "held_grasp_success_rate": 1.0,
                    "grasp_attempt_count": 4,
                    "close_commands_near_cube_fraction": 0.95,
                    "maximum_stable_grasp_duration_s": 2.0,
                    "environments_lifted_2cm_and_retained": 0,
                }
            ),
            encoding="utf-8",
        )

    def poll(self):
        return 0


def test_evaluation_worker_builds_pass_evidence_bundle(tmp_path, monkeypatch):
    training_root = tmp_path / "training"
    job_id, record_path = _completed_training_job(training_root, tmp_path)
    record = TrainingRunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
    evaluation_dir = training_root / "evaluation_jobs" / "eval-abc123def456"
    evaluation_dir.mkdir(parents=True)
    manifest = {
        "evaluation_id": "eval-abc123def456",
        "source_commit": "435110b",
        "training_job_id": job_id,
        "training_run_id": record.run_id,
        "training_record_path": str(record_path),
        "task_id": record.task_id,
        "task_stage": "grasp",
        "num_envs": 4,
        "steps": 800,
        "seed": 42,
        "video_length": 750,
        "checkpoint": record.result.checkpoint.model_dump(mode="json"),
        "environment": record.environment.model_dump(mode="json"),
    }
    (evaluation_dir / "evaluation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(evaluation_worker, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluation_worker.subprocess, "Popen", _SuccessfulEvaluation)

    assert evaluation_worker.run_evaluation(evaluation_dir) == 0

    status = json.loads((evaluation_dir / "evaluation_status.json").read_text(encoding="utf-8"))
    bundle = EvidenceBundle.model_validate_json((evaluation_dir / "evidence_bundle.json").read_text(encoding="utf-8"))
    assert status["evaluation"]["state"] == "completed"
    assert status["evaluation"]["verdict"] == "PASS"
    assert bundle.final_verdict == "PASS"
    assert bundle.target.training_run_id == job_id
    assert all(criterion.result == "PASS" for criterion in bundle.criteria)
    assert any(artifact.kind == "video" for artifact in bundle.visual_evidence)


def test_evaluation_worker_rechecks_checkpoint_before_launch(tmp_path, monkeypatch):
    training_root = tmp_path / "training"
    job_id, record_path = _completed_training_job(training_root, tmp_path)
    record = TrainingRunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
    evaluation_dir = training_root / "evaluation_jobs" / "eval-abc123def456"
    evaluation_dir.mkdir(parents=True)
    manifest = {
        "evaluation_id": "eval-abc123def456",
        "source_commit": "435110b",
        "training_job_id": job_id,
        "training_run_id": record.run_id,
        "training_record_path": str(record_path),
        "task_id": record.task_id,
        "task_stage": "grasp",
        "num_envs": 4,
        "steps": 800,
        "seed": 42,
        "video_length": 750,
        "checkpoint": record.result.checkpoint.model_dump(mode="json"),
        "environment": record.environment.model_dump(mode="json"),
    }
    (evaluation_dir / "evaluation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    Path(record.result.checkpoint.path).write_bytes(b"changed-after-submit")

    def _unexpected_launch(*_args, **_kwargs):
        raise AssertionError("tampered checkpoint must not launch Isaac Sim")

    monkeypatch.setattr(evaluation_worker.subprocess, "Popen", _unexpected_launch)

    assert evaluation_worker.run_evaluation(evaluation_dir) == 2

    status = json.loads((evaluation_dir / "evaluation_status.json").read_text(encoding="utf-8"))
    assert status["evaluation"]["state"] == "failed"
    assert "hash" in status["summary"].lower()


def test_dofbot_evaluator_requires_visible_recordvideo():
    source = (Path(__file__).parents[1] / "scripts" / "evaluate_dofbot_grasp_policy.py").read_text(encoding="utf-8")

    assert 'parser.error("This project requires a visible Kit window; --headless is not allowed.")' in source
    assert 'render_mode="rgb_array"' in source
    assert "gym.wrappers.RecordVideo" in source
    assert 'parser.add_argument("--video-dir", type=Path, required=True)' in source
