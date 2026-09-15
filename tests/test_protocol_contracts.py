from copy import deepcopy

import pytest
from pydantic import ValidationError

from isaaclab_mcp.contracts import (
    EnvironmentContract,
    EvidenceBundle,
    SceneChangeRequest,
    TrainingRunRecord,
    validate_environment_contract,
    validate_evidence_bundle,
    validate_scene_change_request,
)

_HASH = "a" * 64


def _environment_contract():
    return {
        "environment_id": "g1-carry-lab",
        "contract_version": 3,
        "scene": {"usd_path": "Assets/Scenes/g1_carry.usda", "sha256": _HASH},
        "assets": [
            {
                "role": "robot",
                "prim_path": "/World/Robot",
                "asset_path": "Assets/Robots/g1.usd",
                "sha256": _HASH,
            }
        ],
        "articulation_root": "/World/Robot",
        "joints": [
            {
                "name": "left_knee_joint",
                "joint_type": "revolute",
                "axis": "Y",
                "lower_limit": -1.2,
                "upper_limit": 0.3,
                "control_mode": "position",
            }
        ],
        "sensors": [{"name": "base_imu", "observation": "projected_gravity"}],
        "physics": {"physics_dt_s": 0.005, "control_dt_s": 0.02},
        "workspace": {"frame": "world", "nodes": ["start", "target"]},
        "known_limitations": ["Bimanual carry uses a compliant grasp aid."],
        "sim_verification": [
            {
                "kind": "visual",
                "path": "evidence/g1-carry-view.mp4",
                "sha256": _HASH,
                "summary": "Robot and payload move in the visible viewport.",
            }
        ],
        "provenance": {
            "source_commit": "58fca0a",
            "task_version": "g1-carry-v3",
            "created_at": "2026-09-15T12:00:00Z",
        },
    }


def _training_run(contract: EnvironmentContract):
    return {
        "run_id": "TRAIN-0042",
        "environment": {
            "environment_id": contract.environment_id,
            "contract_version": contract.contract_version,
            "contract_fingerprint": contract.fingerprint(),
        },
        "task_id": "Isaac-Run-Carry-Box-G1-v0",
        "config": {"algorithm": "PPO", "seed": 42, "num_envs": 64},
        "job": {"job_id": "JOB-0042", "status": "completed"},
        "result": {
            "checkpoint": {"kind": "checkpoint", "path": "logs/model_1489.pt", "sha256": _HASH, "summary": "best"},
            "metrics": {"reward_mean": 16.21, "carried_fraction": 1.0},
        },
        "artifacts": [{"kind": "video", "path": "logs/rl-video-step-0.mp4", "sha256": _HASH, "summary": "15 seconds"}],
        "provenance": {"source_commit": "58fca0a", "task_version": "g1-carry-v3", "created_at": "2026-09-15T12:10:00Z"},
    }


def _evidence_bundle(contract: EnvironmentContract, run: TrainingRunRecord):
    checkpoint = run.result.checkpoint.model_dump(mode="json")
    return {
        "bundle_id": "EVIDENCE-0042",
        "target": {
            "environment": {
                "environment_id": contract.environment_id,
                "contract_version": contract.contract_version,
                "contract_fingerprint": contract.fingerprint(),
            },
            "training_run_id": run.run_id,
            "checkpoint": checkpoint,
        },
        "structured_evidence": [
            {
                "kind": "structured",
                "path": "artifacts/evaluation.json",
                "sha256": _HASH,
                "summary": "mass and carry metrics",
            }
        ],
        "visual_evidence": [
            {"kind": "visual", "path": "artifacts/evaluation.mp4", "sha256": _HASH, "summary": "visible carry"}
        ],
        "behavioral_evidence": [
            {"kind": "behavioral", "path": "artifacts/behavior.json", "sha256": _HASH, "summary": "no drops"}
        ],
        "training_evidence": [{"kind": "training", "path": "logs/train.json", "sha256": _HASH, "summary": "PPO run"}],
        "criteria": [
            {"criterion_id": "C1", "result": "PASS", "summary": "15 kg read-back is correct"},
            {"criterion_id": "C2", "result": "PASS", "summary": "carry is retained for the clip"},
        ],
        "final_verdict": "PASS",
        "provenance": {"source_commit": "58fca0a", "task_version": "g1-carry-v3", "created_at": "2026-09-15T12:20:00Z"},
    }


def test_environment_contract_has_stable_fingerprint_and_requires_control_ratio():
    contract = EnvironmentContract.model_validate(_environment_contract())

    assert contract.fingerprint() == EnvironmentContract.model_validate(_environment_contract()).fingerprint()
    assert validate_environment_contract(_environment_contract())["ok"] is True

    invalid = _environment_contract()
    invalid["physics"]["control_dt_s"] = 0.017
    with pytest.raises(ValidationError, match="integer multiple"):
        EnvironmentContract.model_validate(invalid)


def test_validation_errors_do_not_echo_untrusted_input_values():
    invalid = _environment_contract()
    invalid["unexpected_secret"] = "do-not-return-this-value"

    result = validate_environment_contract(invalid)

    assert result["ok"] is False
    assert "do-not-return-this-value" not in str(result["errors"])


def test_training_run_binds_to_exact_environment_contract():
    contract = EnvironmentContract.model_validate(_environment_contract())
    run = TrainingRunRecord.model_validate(_training_run(contract))

    assert run.environment.contract_fingerprint == contract.fingerprint()
    assert run.job.status == "completed"


def test_evidence_bundle_pass_requires_all_criteria_to_pass_and_locks_target():
    contract = EnvironmentContract.model_validate(_environment_contract())
    run = TrainingRunRecord.model_validate(_training_run(contract))
    bundle = EvidenceBundle.model_validate(_evidence_bundle(contract, run))

    assert bundle.target.training_run_id == run.run_id
    assert bundle.final_verdict == "PASS"
    assert validate_evidence_bundle(_evidence_bundle(contract, run))["fingerprint"] == bundle.fingerprint()

    invalid = _evidence_bundle(contract, run)
    invalid["criteria"][1]["result"] = "FAIL"
    with pytest.raises(ValidationError, match="PASS verdict"):
        EvidenceBundle.model_validate(invalid)


def test_scene_change_request_separates_observation_hypothesis_and_capability():
    contract = EnvironmentContract.model_validate(_environment_contract())
    run = TrainingRunRecord.model_validate(_training_run(contract))
    bundle = EvidenceBundle.model_validate(_evidence_bundle(contract, run))
    request = {
        "request_id": "SCR-018",
        "source": {
            "environment": {
                "environment_id": contract.environment_id,
                "contract_version": contract.contract_version,
                "contract_fingerprint": contract.fingerprint(),
            },
            "training_run_id": run.run_id,
            "evidence_bundle_id": bundle.bundle_id,
        },
        "observed_evidence": [
            {
                "metric": "upper_limit_failures",
                "value": 0.73,
                "unit": "fraction",
                "artifact": {
                    "kind": "structured",
                    "path": "artifacts/failures.json",
                    "sha256": _HASH,
                    "summary": "failure distribution",
                },
            }
        ],
        "suspected_cause": "Workspace may be insufficient near the rail upper limit.",
        "requested_capability": "The target node must be reachable without exceeding the rail limit.",
        "priority": "high",
        "provenance": {"source_commit": "58fca0a", "task_version": "g1-carry-v3", "created_at": "2026-09-15T12:30:00Z"},
    }

    result = validate_scene_change_request(request)

    assert result["ok"] is True
    assert result["document"]["requested_capability"].startswith("The target node")

    missing_observation = deepcopy(request)
    missing_observation.pop("observed_evidence")
    with pytest.raises(ValidationError):
        SceneChangeRequest.model_validate(missing_observation)
