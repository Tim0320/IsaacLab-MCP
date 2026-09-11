from pathlib import Path

import pytest

from isaaclab_mcp.discovery import discover_tasks


def _write_registration(root: Path, package: str, relative: str, task_id: str) -> None:
    path = root / "source" / package / package / relative / "__init__.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        f'import gymnasium as gym\ngym.register(id="{task_id}", entry_point="example:Env")\n', encoding="utf-8"
    )


def test_discovers_literal_task_ids(tmp_path):
    _write_registration(tmp_path, "isaaclab_tasks", "classic/cartpole", "Isaac-Cartpole-v0")
    _write_registration(tmp_path, "isaaclab_tasks_experimental", "factory", "Isaac-Factory-Experimental-v0")

    tasks = discover_tasks(tmp_path)

    assert [task["task_id"] for task in tasks] == ["Isaac-Cartpole-v0", "Isaac-Factory-Experimental-v0"]
    assert tasks[0]["package"] == "isaaclab_tasks"


def test_keyword_and_limit_are_applied(tmp_path):
    _write_registration(tmp_path, "isaaclab_tasks", "a", "Isaac-Cartpole-v0")
    _write_registration(tmp_path, "isaaclab_tasks", "b", "Isaac-Cartpole-RGB-v0")
    _write_registration(tmp_path, "isaaclab_tasks", "c", "Isaac-Ant-v0")

    tasks = discover_tasks(tmp_path, keyword="cartpole", limit=1)

    assert len(tasks) == 1
    assert "Cartpole" in tasks[0]["task_id"]


@pytest.mark.parametrize("limit", [0, 1001])
def test_limit_is_bounded(tmp_path, limit):
    with pytest.raises(ValueError, match="between 1 and 1000"):
        discover_tasks(tmp_path, limit=limit)
