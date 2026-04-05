"""Shared types for the self-improvement subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# TypedDicts for task data — these are plain dicts at runtime so all existing
# dict-style access (result["passed"], task["path"], etc.) continues to work.
# ---------------------------------------------------------------------------

try:
    from typing import TypedDict, Required
except ImportError:  # Python < 3.11
    from typing_extensions import TypedDict, Required  # type: ignore[assignment]


class TaskDescriptor(TypedDict):
    """Description of a benchmark task as returned by TaskRunner.discover_tasks."""

    name: str
    description: str
    timeout: int
    path: str
    instruction: str


class TaskResult(TypedDict, total=False):
    """Result of running a single benchmark task."""

    name: Required[str]
    passed: Required[bool]
    score: Required[float]
    output: Required[str]
    duration: Required[float]
    task_dir: str  # optional: set by TaskRunner so MetaAgent can write solutions


# ---------------------------------------------------------------------------
# Proposal — replaces the raw dict used by MetaAgent
# ---------------------------------------------------------------------------


@dataclass
class Proposal:
    """A concrete change proposed by MetaAgent for one failing task.

    Parameters
    ----------
    file:
        Filename to write, relative to *task_dir* (e.g. ``"solution.py"``).
    task_dir:
        Absolute path of the task directory.  When set, the file is written
        at ``{task_dir}/{file}``.  When *None*, *file* is treated as an
        absolute or project-relative path.
    description:
        Human-readable description of what the proposal does.
    content:
        Full content to write to *file*.  An empty string is valid (e.g.
        to create an empty file).
    task_name:
        Name of the task this proposal targets (for logging).
    """

    file: str
    description: str
    content: str
    task_name: str = ""
    task_dir: str | None = None

    def resolve_path(self) -> str:
        """Return the absolute path where this proposal should be written."""
        if self.task_dir:
            from pathlib import Path
            return str(Path(self.task_dir) / self.file)
        return self.file
