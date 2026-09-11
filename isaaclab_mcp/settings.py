"""Environment-backed paths and transport settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Resolved IsaacLab-MCP settings."""

    isaaclab_path: Path | None
    isaac_sim_path: Path | None
    training_root: Path
    transport: str
    http_host: str
    http_port: int


def _optional_path(value: str | None) -> Path | None:
    if value is None or not value.strip():
        return None
    return Path(value.strip()).expanduser().resolve(strict=False)


def load_settings() -> Settings:
    """Load settings from environment variables without loading an env file."""
    default_isaaclab = r"D:\IsaacLab" if os.name == "nt" else None
    isaaclab_path = _optional_path(os.getenv("ISAACLAB_PATH", default_isaaclab))

    configured_sim = _optional_path(os.getenv("ISAAC_SIM_PATH"))
    linked_sim = isaaclab_path / "_isaac_sim" if isaaclab_path is not None else None
    isaac_sim_path = configured_sim or linked_sim
    default_training_root = Path(__file__).resolve().parents[1] / "training_projects"
    training_root = _optional_path(os.getenv("ISAACLAB_MCP_TRAINING_ROOT")) or default_training_root

    return Settings(
        isaaclab_path=isaaclab_path,
        isaac_sim_path=isaac_sim_path,
        training_root=training_root,
        transport=os.getenv("ISAACLAB_MCP_TRANSPORT", "stdio").strip().lower(),
        http_host=os.getenv("ISAACLAB_MCP_HTTP_HOST", "127.0.0.1").strip(),
        http_port=int(os.getenv("ISAACLAB_MCP_HTTP_PORT", "8010")),
    )
