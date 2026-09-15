"""Read-only MCP validation tools for the multi-agent protocol documents."""

from __future__ import annotations

from typing import Any

from isaaclab_mcp.contracts import (
    validate_environment_contract,
    validate_evidence_bundle,
    validate_scene_change_request,
)


def register_protocol_tools(mcp: Any) -> None:
    """Register strict data-contract validators without launching Kit or mutating files."""

    @mcp.tool("validate_environment_contract")
    def validate_environment_contract_tool(contract: dict[str, Any]) -> dict[str, Any]:
        """Validate the immutable, verified Sim-to-Lab environment contract."""
        return validate_environment_contract(contract)

    @mcp.tool("validate_evidence_bundle")
    def validate_evidence_bundle_tool(bundle: dict[str, Any]) -> dict[str, Any]:
        """Validate evidence that is locked to one environment version, run, and checkpoint."""
        return validate_evidence_bundle(bundle)

    @mcp.tool("validate_scene_change_request")
    def validate_scene_change_request_tool(request: dict[str, Any]) -> dict[str, Any]:
        """Validate a Lab-side capability request for Isaac Sim without changing the scene."""
        return validate_scene_change_request(request)
