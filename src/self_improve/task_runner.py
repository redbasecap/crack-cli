"""Discover and execute benchmark tasks from a tasks/ directory."""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


def _parse_toml_simple(path: Path) -> dict:
    """Minimal TOML parser for simple key-value task.toml files.

    Falls back to this when neither tomllib nor tomli is available.
    Handles basic ``[section]`` headers and ``key = "value"`` / ``key = number`` pairs.
    """
    if tomllib is not None:
        with path.open("rb") as fh:
            return tomllib.load(fh)

    config: dict = {}
    current_section: dict | None = None
    with path.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section_name = line[1:-1].strip()
                config[section_name] = {}
                current_section = config[section_name]
                continue
            if "=" in line and current_section is not None:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Strip surrounding quotes.
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    current_section[key] = value[1:-1]
                else:
                    # Try int, then float, then keep as string.
                    try:
                        current_section[key] = int(value)
                    except ValueError:
                        try:
                            current_section[key] = float(value)
                        except ValueError:
                            current_section[key] = value
    return config


@dataclass
class TaskRunner:
    """Discovers and runs benchmark tasks.

    Each task lives in its own subdirectory under *tasks_dir* and must contain:
        - ``task.toml``   — configuration (name, description, timeout)
        - ``instruction.md`` — human-readable task description
        - ``tests/test.sh``  — verification script (exit 0 = pass)
    """

    tasks_dir: Path = field(default_factory=lambda: Path("tasks"))
    default_timeout: int = 120

    def discover_tasks(self) -> list[dict]:
        """Return a list of task descriptors found under *tasks_dir*.

        Each descriptor is a dict with keys: ``name``, ``description``,
        ``timeout``, ``path``, ``instruction``.
        """
        tasks: list[dict] = []
        if not self.tasks_dir.is_dir():
            return tasks

        for task_dir in sorted(self.tasks_dir.iterdir()):
            toml_path = task_dir / "task.toml"
            if not toml_path.exists():
                continue

            config = _parse_toml_simple(toml_path)

            task_section = config.get("task", {})
            name = task_section.get("name", task_dir.name)
            description = task_section.get("description", "")
            timeout = task_section.get("timeout", self.default_timeout)

            instruction_path = task_dir / "instruction.md"
            instruction = instruction_path.read_text() if instruction_path.exists() else ""

            tasks.append(
                {
                    "name": name,
                    "description": description,
                    "timeout": timeout,
                    "path": str(task_dir),
                    "instruction": instruction,
                }
            )

        return tasks

    def run_task(self, task: dict) -> dict:
        """Execute a single task and return the result.

        Returns a dict with keys: ``name``, ``passed``, ``score``,
        ``output``, ``duration``.
        """
        task_dir = Path(task["path"])
        test_script = task_dir / "tests" / "test.sh"
        timeout = task.get("timeout", self.default_timeout)

        if not test_script.exists():
            return {
                "name": task["name"],
                "passed": False,
                "score": 0.0,
                "output": f"test script not found: {test_script}",
                "duration": 0.0,
                "task_dir": str(task_dir),
            }

        start = time.monotonic()
        try:
            result = subprocess.run(
                ["bash", str(test_script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(task_dir),
            )
            elapsed = time.monotonic() - start
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            score = 1.0 if passed else 0.0
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            passed = False
            output = f"task timed out after {timeout}s"
            score = 0.0
        except Exception as exc:
            elapsed = time.monotonic() - start
            passed = False
            output = f"unexpected error: {exc}"
            score = 0.0

        return {
            "name": task["name"],
            "passed": passed,
            "score": score,
            "output": output.strip(),
            "duration": round(elapsed, 3),
            "task_dir": str(task_dir),
        }

    async def run_task_async(self, task: dict) -> dict:
        """Async version of :meth:`run_task` using ``asyncio`` subprocesses."""
        task_dir = Path(task["path"])
        test_script = task_dir / "tests" / "test.sh"
        timeout = task.get("timeout", self.default_timeout)

        if not test_script.exists():
            return {
                "name": task["name"],
                "passed": False,
                "score": 0.0,
                "output": f"test script not found: {test_script}",
                "duration": 0.0,
                "task_dir": str(task_dir),
            }

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", str(test_script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(task_dir),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            elapsed = time.monotonic() - start
            passed = proc.returncode == 0
            output = (stdout or b"").decode() + (stderr or b"").decode()
            score = 1.0 if passed else 0.0
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            passed = False
            output = f"task timed out after {timeout}s"
            score = 0.0
        except Exception as exc:
            elapsed = time.monotonic() - start
            passed = False
            output = f"unexpected error: {exc}"
            score = 0.0

        return {
            "name": task["name"],
            "passed": passed,
            "score": score,
            "output": output.strip() if isinstance(output, str) else output,
            "duration": round(elapsed, 3),
            "task_dir": str(task_dir),
        }

    async def run_tasks_async(self, tasks: list[dict]) -> list[dict]:
        """Run all *tasks* concurrently and return results in order."""
        return list(await asyncio.gather(*(self.run_task_async(t) for t in tasks)))
