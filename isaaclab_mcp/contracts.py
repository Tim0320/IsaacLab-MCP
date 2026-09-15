"""Versioned data contracts for the controlled Sim-to-Lab feedback loop."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_SHA256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class ProtocolModel(BaseModel):
    """Strict serializable document with a deterministic content fingerprint."""

    model_config = ConfigDict(extra="forbid")

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=True)
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Provenance(ProtocolModel):
    """Code and task version that produced one immutable protocol document."""

    source_commit: str = Field(min_length=7, max_length=64)
    task_version: str = Field(min_length=1)
    created_at: datetime


class ArtifactReference(ProtocolModel):
    """A content-addressed evidence artifact; the path is descriptive, the hash is authoritative."""

    kind: Literal["structured", "visual", "behavioral", "training", "checkpoint", "log", "video"]
    path: str = Field(min_length=1)
    sha256: _SHA256
    summary: str = Field(min_length=1)


class SceneReference(ProtocolModel):
    usd_path: str = Field(min_length=1)
    sha256: _SHA256


class EnvironmentAsset(ProtocolModel):
    role: str = Field(min_length=1)
    prim_path: str = Field(pattern=r"^/")
    asset_path: str = Field(min_length=1)
    sha256: _SHA256


class JointSpecification(ProtocolModel):
    name: str = Field(min_length=1)
    joint_type: Literal["revolute", "prismatic", "fixed"]
    axis: Literal["X", "Y", "Z"]
    lower_limit: float
    upper_limit: float
    control_mode: Literal["position", "velocity", "effort"]

    @model_validator(mode="after")
    def validate_limits(self) -> "JointSpecification":
        if self.joint_type != "fixed" and self.lower_limit >= self.upper_limit:
            raise ValueError("joint lower_limit must be smaller than upper_limit")
        return self


class SensorSpecification(ProtocolModel):
    name: str = Field(min_length=1)
    observation: str = Field(min_length=1)


class PhysicsTiming(ProtocolModel):
    physics_dt_s: float = Field(gt=0)
    control_dt_s: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_control_ratio(self) -> "PhysicsTiming":
        ratio = self.control_dt_s / self.physics_dt_s
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("control_dt_s must be an integer multiple of physics_dt_s")
        return self


class WorkspaceSpecification(ProtocolModel):
    frame: str = Field(min_length=1)
    nodes: list[str] = Field(min_length=1)


class EnvironmentContract(ProtocolModel):
    """The only verified Sim-to-Lab input; it contains facts, not RL design guesses."""

    schema_version: Literal["1.0"] = "1.0"
    environment_id: str = Field(min_length=1)
    contract_version: int = Field(ge=1)
    scene: SceneReference
    assets: list[EnvironmentAsset] = Field(min_length=1)
    articulation_root: str = Field(pattern=r"^/")
    joints: list[JointSpecification] = Field(min_length=1)
    sensors: list[SensorSpecification] = Field(min_length=1)
    physics: PhysicsTiming
    workspace: WorkspaceSpecification
    known_limitations: list[str] = Field(min_length=1)
    sim_verification: list[ArtifactReference] = Field(min_length=1)
    provenance: Provenance

    @field_validator("sim_verification")
    @classmethod
    def require_sim_evidence(cls, value: list[ArtifactReference]) -> list[ArtifactReference]:
        if not any(item.kind in {"structured", "visual", "behavioral"} for item in value):
            raise ValueError("sim_verification requires structured, visual, or behavioral evidence")
        return value


class EnvironmentReference(ProtocolModel):
    environment_id: str = Field(min_length=1)
    contract_version: int = Field(ge=1)
    contract_fingerprint: _SHA256


class TrainingConfiguration(ProtocolModel):
    algorithm: str = Field(min_length=1)
    seed: int = Field(ge=0)
    num_envs: int = Field(gt=0)


class JobMetadata(ProtocolModel):
    job_id: str = Field(min_length=1)
    status: Literal["planned", "running", "completed", "failed", "cancelled"]


class TrainingResult(ProtocolModel):
    checkpoint: ArtifactReference | None = None
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("checkpoint")
    @classmethod
    def require_checkpoint_kind(cls, value: ArtifactReference | None) -> ArtifactReference | None:
        if value is not None and value.kind != "checkpoint":
            raise ValueError("result checkpoint must use kind='checkpoint'")
        return value


class TrainingRunRecord(ProtocolModel):
    """What the Lab side requested, ran, and produced for an exact environment version."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    environment: EnvironmentReference
    task_id: str = Field(min_length=1)
    config: TrainingConfiguration
    job: JobMetadata
    result: TrainingResult | None = None
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    provenance: Provenance

    @model_validator(mode="after")
    def validate_completed_run(self) -> "TrainingRunRecord":
        if self.job.status == "completed":
            if self.result is None or self.result.checkpoint is None:
                raise ValueError("completed training runs require a checkpoint result")
            if not any(item.kind in {"video", "visual"} for item in self.artifacts):
                raise ValueError("completed training runs require visible video evidence")
        return self


class EvidenceTarget(ProtocolModel):
    environment: EnvironmentReference
    training_run_id: str = Field(min_length=1)
    checkpoint: ArtifactReference

    @field_validator("checkpoint")
    @classmethod
    def require_checkpoint(cls, value: ArtifactReference) -> ArtifactReference:
        if value.kind != "checkpoint":
            raise ValueError("evidence target checkpoint must use kind='checkpoint'")
        return value


class EvidenceCriterion(ProtocolModel):
    criterion_id: str = Field(min_length=1)
    result: Literal["PASS", "FAIL"]
    summary: str = Field(min_length=1)


class EvidenceBundle(ProtocolModel):
    """Independent verification result tied to one environment, run, and checkpoint."""

    schema_version: Literal["1.0"] = "1.0"
    bundle_id: str = Field(min_length=1)
    target: EvidenceTarget
    structured_evidence: list[ArtifactReference] = Field(min_length=1)
    visual_evidence: list[ArtifactReference] = Field(min_length=1)
    behavioral_evidence: list[ArtifactReference] = Field(min_length=1)
    training_evidence: list[ArtifactReference] = Field(min_length=1)
    criteria: list[EvidenceCriterion] = Field(min_length=1)
    final_verdict: Literal["PASS", "FIX_REQUIRED", "RETHINK", "BLOCKED"]
    provenance: Provenance

    @model_validator(mode="after")
    def validate_verdict(self) -> "EvidenceBundle":
        has_failure = any(criterion.result == "FAIL" for criterion in self.criteria)
        if self.final_verdict == "PASS" and has_failure:
            raise ValueError("PASS verdict requires every criterion to pass")
        if self.final_verdict == "FIX_REQUIRED" and not has_failure:
            raise ValueError("FIX_REQUIRED verdict requires at least one failed criterion")
        return self


class ObservedEvidence(ProtocolModel):
    metric: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    artifact: ArtifactReference


class SceneChangeSource(ProtocolModel):
    environment: EnvironmentReference
    training_run_id: str = Field(min_length=1)
    evidence_bundle_id: str = Field(min_length=1)


class SceneChangeRequest(ProtocolModel):
    """Lab-side request for a Sim capability, deliberately separate from a scene edit."""

    schema_version: Literal["1.0"] = "1.0"
    request_id: str = Field(min_length=1)
    source: SceneChangeSource
    observed_evidence: list[ObservedEvidence] = Field(min_length=1)
    suspected_cause: str = Field(min_length=1)
    requested_capability: str = Field(min_length=1)
    priority: Literal["low", "medium", "high", "critical"]
    provenance: Provenance


def _validate(document_type: type[ProtocolModel], data: dict[str, Any]) -> dict[str, Any]:
    try:
        document = document_type.model_validate(data)
    except ValidationError as error:
        errors = [
            {key: value for key, value in item.items() if key not in {"input", "ctx"}}
            for item in error.errors(include_url=False)
        ]
        return {"ok": False, "errors": errors}
    return {"ok": True, "document": document.model_dump(mode="json"), "fingerprint": document.fingerprint()}


def validate_environment_contract(data: dict[str, Any]) -> dict[str, Any]:
    """Validate one Sim-to-Lab environment contract without touching the runtime."""
    return _validate(EnvironmentContract, data)


def validate_evidence_bundle(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a verifier bundle and its locked target identity."""
    return _validate(EvidenceBundle, data)


def validate_scene_change_request(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a Lab-to-Sim capability request without editing a scene."""
    return _validate(SceneChangeRequest, data)
