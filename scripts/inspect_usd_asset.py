"""Inspect a USD robot asset without modifying it.

Run this script with the Isaac Sim / Isaac Lab Python runtime so ``pxr`` is
available.  The JSON output is intended to be both human-readable evidence and
machine-readable input for generated Isaac Lab task configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdPhysics

JOINT_TYPES = (
    UsdPhysics.RevoluteJoint,
    UsdPhysics.PrismaticJoint,
    UsdPhysics.SphericalJoint,
    UsdPhysics.DistanceJoint,
    UsdPhysics.FixedJoint,
    UsdPhysics.Joint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _targets(relationship: Any) -> list[str]:
    return [str(path) for path in relationship.GetTargets()]


def _authored_value(attribute: Any) -> Any:
    if not attribute or not attribute.HasAuthoredValueOpinion():
        return None
    value = attribute.Get()
    if isinstance(value, Gf.Vec3f | Gf.Vec3d):
        return list(value)
    return value


def _joint_info(prim: Usd.Prim) -> dict[str, Any] | None:
    schema = None
    for joint_type in JOINT_TYPES:
        candidate = joint_type(prim)
        if candidate:
            schema = candidate
            break
    if schema is None:
        return None

    info: dict[str, Any] = {
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName(),
        "body0": _targets(schema.GetBody0Rel()),
        "body1": _targets(schema.GetBody1Rel()),
        "collision_enabled": _authored_value(schema.GetCollisionEnabledAttr()),
        "exclude_from_articulation": _authored_value(schema.GetExcludeFromArticulationAttr()),
    }
    if isinstance(schema, UsdPhysics.RevoluteJoint | UsdPhysics.PrismaticJoint):
        info["axis"] = str(schema.GetAxisAttr().Get())
        info["lower_limit"] = _authored_value(schema.GetLowerLimitAttr())
        info["upper_limit"] = _authored_value(schema.GetUpperLimitAttr())

    drives = []
    for axis in ("angular", "linear", "rotX", "rotY", "rotZ", "transX", "transY", "transZ"):
        drive = UsdPhysics.DriveAPI.Get(prim, axis)
        if drive and drive.GetPrim().HasAPI(UsdPhysics.DriveAPI, axis):
            drives.append(
                {
                    "axis": axis,
                    "type": str(drive.GetTypeAttr().Get()),
                    "target_position": _authored_value(drive.GetTargetPositionAttr()),
                    "target_velocity": _authored_value(drive.GetTargetVelocityAttr()),
                    "stiffness": _authored_value(drive.GetStiffnessAttr()),
                    "damping": _authored_value(drive.GetDampingAttr()),
                    "max_force": _authored_value(drive.GetMaxForceAttr()),
                }
            )
    info["drives"] = drives
    return info


def inspect_asset(asset_path: str | Path) -> dict[str, Any]:
    asset_identifier = str(asset_path)
    is_remote = "://" in asset_identifier
    local_path = None if is_remote else Path(asset_identifier).resolve(strict=True)
    if local_path:
        asset_identifier = str(local_path)

    stage = Usd.Stage.Open(asset_identifier, Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"USD stage could not be opened: {asset_identifier}")

    articulation_roots: list[str] = []
    rigid_bodies: list[str] = []
    collision_prims: list[str] = []
    joints: list[dict[str, Any]] = []
    unresolved_references: list[str] = []

    for prim in stage.TraverseAll():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_prims.append(str(prim.GetPath()))
        if joint := _joint_info(prim):
            joints.append(joint)

        references = prim.GetMetadata("references")
        if references:
            for item in references.GetAddedOrExplicitItems():
                if item.assetPath:
                    resolved = Sdf.ComputeAssetPathRelativeToLayer(stage.GetRootLayer(), item.assetPath)
                    if not resolved or not Ar.GetResolver().Resolve(resolved):
                        unresolved_references.append(f"{prim.GetPath()} -> {item.assetPath}")

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True)
    world_bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    minimum = list(world_bbox.GetMin())
    maximum = list(world_bbox.GetMax())

    default_prim = stage.GetDefaultPrim()
    layer = stage.GetRootLayer()
    return {
        "asset": {
            "path": asset_identifier,
            "size_bytes": local_path.stat().st_size if local_path else None,
            "sha256": _sha256(local_path) if local_path else None,
            "file_format": layer.GetFileFormat().formatId,
            "default_prim": str(default_prim.GetPath()) if default_prim else None,
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        },
        "bounds": {
            "minimum_stage_units": minimum,
            "maximum_stage_units": maximum,
            "size_stage_units": [maximum[index] - minimum[index] for index in range(3)],
        },
        "physics": {
            "articulation_roots": articulation_roots,
            "rigid_bodies": rigid_bodies,
            "collision_prim_count": len(collision_prims),
            "collision_prims": collision_prims,
            "joints": joints,
        },
        "dependencies": {
            "unresolved_references": sorted(set(unresolved_references)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = inspect_asset(args.asset)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
