from pathlib import Path
from unittest.mock import MagicMock

from isaaclab_mcp.settings import load_settings
from isaaclab_mcp.status import _git_status, collect_status


def _fake_isaaclab(root: Path) -> None:
    (root / "source" / "isaaclab").mkdir(parents=True)
    (root / "_isaac_sim").mkdir()
    (root / "VERSION").write_text("3.0.0\n", encoding="utf-8")
    (root / "isaaclab.bat").write_text("@echo off\n", encoding="utf-8")
    (root / "_isaac_sim" / "python.bat").write_text("@echo off\n", encoding="utf-8")


def test_environment_path_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv("ISAACLAB_PATH", str(tmp_path))

    settings = load_settings()

    assert settings.isaaclab_path == tmp_path.resolve()
    assert settings.isaac_sim_path == (tmp_path / "_isaac_sim").resolve()
    assert settings.training_root.name == "training_projects"


def test_collect_status_reports_runtime(monkeypatch, tmp_path):
    _fake_isaaclab(tmp_path)
    monkeypatch.setenv("ISAACLAB_PATH", str(tmp_path))
    monkeypatch.delenv("ISAAC_SIM_PATH", raising=False)

    status = collect_status()

    assert status["ok"] is True
    assert status["isaac_lab"]["version"] == "3.0.0"
    assert status["isaac_lab"]["source_packages"] == ["isaaclab"]
    assert status["isaac_sim"]["runtime_python_exists"] is True
    assert status["git"]["repository"] is False


def test_git_status_uses_single_porcelain_query(monkeypatch, tmp_path):
    run = MagicMock(
        return_value=MagicMock(
            returncode=0,
            stdout=("# branch.oid 0123456789abcdef\n# branch.head feature/test\n? local-file.txt\n"),
        )
    )
    monkeypatch.setattr("isaaclab_mcp.status.subprocess.run", run)

    status = _git_status(tmp_path)

    assert status == {
        "repository": True,
        "branch": "feature/test",
        "head": "0123456789abcdef",
        "dirty": True,
        "status": ["? local-file.txt"],
    }
    assert run.call_count == 1
    assert run.call_args.args[0][-3:] == ["status", "--porcelain=v2", "--branch"]
