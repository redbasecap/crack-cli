from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortingTask:
    """A single porting task with a name and description."""

    name: str
    description: str


__all__ = ["PortingTask"]
