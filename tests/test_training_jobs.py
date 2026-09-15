import json
from pathlib import Path

from isaaclab_mcp import job_worker, training_jobs
from isaaclab_mcp.contracts import TrainingRunRecord
from isaaclab_mcp.settings import Settings


def _settings(root: Path) -> Settings:
    return Settings(
        isaaclab_path=Path(r"D:\IsaacLab"),
        isaac_sim_path=Path(r"C:\isaacsim"),
        training_root=root,
        transport="stdio",
        http_host="127.0.0.1",
        http_port=8010,
    )


def _environment_contract() -> dict:
    digest = "a" * 64
    provenance = {"source_commit": "3e34db1", "task_version": "v1", "created_at": "2026-09-15T00:00:00Z"}
    artifact = {"kind": "structured", "path": "F:/evidence/scene.json", "sha256": digest, "summary": "scene read-back"}
    return {
        "environment_id": "dofbot-cube-lift",
        "contract_version": 1,
        "scene": {"usd_path": "F:/scene.usda", "sha256": digest},
        "assets": [
            {
                "role": "robot",
                "prim_path": "/World/Dofbot",
                "asset_path": "Robots/Yahboom/Dofbot/dofbot.usd",
                "sha256": digest,
            }
        ],
        "articulation_root": "/World/Dofbot",
        "joints": [
            {
                "name": "joint1",
                "joint_type": "revolute",
                "axis": "Z",
                "lower_limit": -1.0,
                "upper_limit": 1.0,
                "control_mode": "position",
            }
        ],
        "sensors": [{"name": "cube_pose", "observation": "relative pose"}],
        "physics": {"physics_dt_s": 1 / 120, "control_dt_s": 1 / 60},
        "workspace": {"frame": "world", "nodes": ["dofbot", "cube"]},
        "known_limitations": ["grasp stage only"],
        "sim_verification": [artifact],
        "provenance": provenance,
    }


class _FakePopen:
    command: list[str] | None = None

    def __init__(self, command, **_kwargs):
        type(self).command = command
        self.pid = 12345


def test_submit_creates_immutable_contract_snapshot_and_safe_worker_command(tmp_path, monkeypatch):
    monkeypatch.setattr(training_jobs.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(training_jobs, "_source_commit", lambda: "3e34db1")
    result = training_jobs.submit_dofbot_training_run(
        _environment_contract(),
        task_id="Isaac-Grasp-Cube-Dofbot-v0",
        run_name="vertical_grasp",
        num_envs=4,
        max_iterations=24,
        settings=_settings(tmp_path),
    )

    assert result["status"] == "success"
    assert result["job"]["state"] == "running"
    job_dir = Path(result["job"]["directory"])
    manifest = json.loads((job_dir / "job_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((job_dir / "environment_contract.json").read_text(encoding="utf-8"))
    assert manifest["environment"]["contract_fingerprint"] == result["job"]["contract_fingerprint"]
    assert contract["environment_id"] == "dofbot-cube-lift"
    assert _FakePopen.command is not None
    assert _FakePopen.command[1:3] == ["-m", "isaaclab_mcp.job_worker"]
    assert "--headless" not in _FakePopen.command


def test_cancel_is_a_worker_control_file_not_a_direct_process_kill(tmp_path):
    job_id = "dofbot-abc123def456"
    job_dir = tmp_path / "runtime_jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "job_status.json").write_text(
        json.dumps(
            {
                "status": "success",
                "summary": "Running.",
                "next_actions": ["Wait."],
                "artifacts": [],
                "job": {"state": "running"},
            }
        ),
        encoding="utf-8",
    )

    result = training_jobs.cancel_training_run(job_id, settings=_settings(tmp_path))

    assert result["status"] == "warning"
    assert result["job"]["state"] == "cancellation_requested"
    assert (job_dir / "cancel.request").is_file()


def test_status_rejects_job_id_path_traversal(tmp_path):
    result = training_jobs.get_training_run_status("../outside", settings=_settings(tmp_path))

    assert result["status"] == "error"
    assert result["next_actions"] == ["Use the job_id returned by submit_dofbot_training_run."]


def test_submit_refuses_unverifiable_git_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(training_jobs, "_source_commit", lambda: None)

    result = training_jobs.submit_dofbot_training_run(
        _environment_contract(),
        task_id="Isaac-Grasp-Cube-Dofbot-v0",
        run_name="vertical_grasp",
        max_iterations=24,
        settings=_settings(tmp_path),
    )

    assert result["status"] == "error"
    assert not (tmp_path / "runtime_jobs").exists()


class _SuccessfulLauncher:
    command: list[str] | None = None

    def __init__(self, command, **_kwargs):
        type(self).command = command
        self.returncode = 0

    def poll(self):
        return 0


def test_worker_writes_completed_training_run_record_with_video_and_checkpoint(tmp_path, monkeypatch):
    job_dir = tmp_path / "runtime_jobs" / "dofbot-abc123def456"
    job_dir.mkdir(parents=True)
    run_dir = tmp_path / "logs" / "rsl_rl" / "dofbot_cube_grasp_pretrain" / "2026-09-15_vertical_grasp"
    video_dir = run_dir / "videos" / "train"
    video_dir.mkdir(parents=True)
    (run_dir / "model_24.pt").write_bytes(b"checkpoint")
    (video_dir / "clip.mp4").write_bytes(b"video")
    manifest = {
        "job_id": "dofbot-abc123def456",
        "source_commit": "3e34db1",
        "task_id": "Isaac-Grasp-Cube-Dofbot-v0",
        "experiment_name": "dofbot_cube_grasp_pretrain",
        "run_name": "vertical_grasp",
        "num_envs": 4,
        "max_iterations": 24,
        "seed": 42,
        "video_length": 750,
        "video_interval": 1000,
        "environment": {
            "environment_id": "dofbot-cube-lift",
            "contract_version": 1,
            "contract_fingerprint": "a" * 64,
        },
    }
    (job_dir / "job_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(job_worker, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(job_worker.subprocess, "Popen", _SuccessfulLauncher)

    assert job_worker.run_job(job_dir) == 0

    status = json.loads((job_dir / "job_status.json").read_text(encoding="utf-8"))
    record = TrainingRunRecord.model_validate_json((job_dir / "training_run_record.json").read_text(encoding="utf-8"))
    assert status["job"]["state"] == "completed"
    assert record.environment.contract_fingerprint == "a" * 64
    assert record.result is not None and record.result.checkpoint is not None
    assert any(artifact.kind == "video" for artifact in record.artifacts)
    assert _SuccessfulLauncher.command is not None
    assert "--headless" not in _SuccessfulLauncher.command
