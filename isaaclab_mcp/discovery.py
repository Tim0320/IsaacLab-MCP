"""Static Isaac Lab task discovery that does not launch Kit."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class TaskRecord:
    """One statically declared Gym task."""

    task_id: str
    source_file: str
    package: str


def _literal_registration_ids(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError):
        return []

    task_ids: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "register":
            continue
        for keyword in node.keywords:
            if keyword.arg == "id" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                task_ids.append(keyword.value.value)
    return task_ids


def discover_tasks(isaaclab_path: Path, keyword: str | None = None, limit: int = 200) -> list[dict[str, str]]:
    """Discover literal ``gym.register(id=...)`` declarations under Isaac Lab task packages."""
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")

    source_root = isaaclab_path / "source"
    package_names = ("isaaclab_tasks", "isaaclab_tasks_experimental")
    normalized_keyword = keyword.casefold() if keyword else None
    records: set[TaskRecord] = set()

    for package_name in package_names:
        package_root = source_root / package_name / package_name
        if not package_root.is_dir():
            continue
        for init_file in package_root.rglob("__init__.py"):
            for task_id in _literal_registration_ids(init_file):
                if normalized_keyword and normalized_keyword not in task_id.casefold():
                    continue
                records.add(
                    TaskRecord(
                        task_id=task_id,
                        source_file=init_file.relative_to(isaaclab_path).as_posix(),
                        package=package_name,
                    )
                )

    return [asdict(record) for record in sorted(records)[:limit]]
