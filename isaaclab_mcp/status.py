"""Read-only Isaac Lab environment inspection."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from isaaclab_mcp.settings import Settings, load_settings


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _git(root: Path, *args: str) -> str | None:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            check=False,
            stdin=subprocess.DEVNULL,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=5,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_status(root: Path) -> dict[str, Any]:
    output = _git(root, "status", "--porcelain=v2", "--branch")
    if output is None:
        return {"repository": False}

    lines = output.splitlines()
    metadata = {}
    changes = []
    for line in lines:
        if line.startswith("# branch."):
            key, _, value = line[2:].partition(" ")
            metadata[key] = value
        elif line:
            changes.append(line)

    head = metadata.get("branch.oid")
    if head == "(initial)":
        head = None
    return {
        "repository": True,
        "branch": metadata.get("branch.head"),
        "head": head,
        "dirty": bool(changes),
        "status": changes,
    }


def collect_status(settings: Settings | None = None) -> dict[str, Any]:
    """Collect read-only status for the configured Isaac Lab checkout."""
    resolved = settings or load_settings()
    root = resolved.isaaclab_path
    if root is None:
        return {
            "ok": False,
            "error": "ISAACLAB_PATH is not configured",
            "runtime_boundary": "Use isaaclab.bat -p for Isaac Lab and Kit-bound Python.",
        }

    launcher = root / "isaaclab.bat"
    version_file = root / "VERSION"
    source_root = root / "source"
    packages = sorted(path.name for path in source_root.iterdir() if path.is_dir()) if source_root.is_dir() else []

    linked_sim = root / "_isaac_sim"
    sim_path = resolved.isaac_sim_path
    sim_version = _read_text(sim_path / "VERSION") if sim_path is not None else None
    runtime_python = linked_sim / "python.bat"
    ok = root.is_dir() and launcher.is_file() and version_file.is_file() and source_root.is_dir()

    return {
        "ok": ok,
        "isaac_lab": {
            "path": str(root),
            "exists": root.is_dir(),
            "version": _read_text(version_file),
            "launcher": str(launcher),
            "launcher_exists": launcher.is_file(),
            "source_packages": packages,
        },
        "isaac_sim": {
            "path": str(sim_path) if sim_path is not None else None,
            "version": sim_version,
            "linked_from_isaac_lab": linked_sim.exists(),
            "runtime_python": str(runtime_python),
            "runtime_python_exists": runtime_python.is_file(),
        },
        "git": _git_status(root),
        "runtime_boundary": "Use isaaclab.bat -p for Isaac Lab and Kit-bound Python.",
    }
