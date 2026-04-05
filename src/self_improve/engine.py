"""Core experiment loop — hill-climbing over benchmark scores."""

from __future__ import annotations

import asyncio
import concurrent.futures
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .scorer import TaskScorer, ScoreHistory
from .task_runner import TaskRunner
from .meta_agent import MetaAgent
from .sandbox import MicroVMSandbox
from .types import TaskDescriptor, TaskResult


def _find_project_root() -> Path:
    """Locate the project root by walking up to the nearest .git directory."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def _current_commit_sha() -> str:
    """Return the short SHA of HEAD."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _git_commit(message: str) -> str:
    """Stage all changes and commit; return the new short SHA."""
    subprocess.run(["git", "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        check=True,
        capture_output=True,
    )
    return _current_commit_sha()


def _git_revert_last() -> None:
    """Revert the most recent commit (keeps history clean)."""
    subprocess.run(
        ["git", "revert", "HEAD", "--no-edit"],
        check=True,
        capture_output=True,
    )


def _run_async(coro: object) -> object:
    """Run *coro* safely regardless of whether an event loop is already running.

    ``asyncio.run()`` raises ``RuntimeError`` when called from inside a running
    loop (e.g. in a Jupyter notebook or an async test framework).  This helper
    detects that case and runs the coroutine in a fresh thread instead, which
    has its own event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We are already inside an event loop — delegate to a worker thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    return asyncio.run(coro)  # type: ignore[arg-type]


@dataclass
class IterationResult:
    """Outcome of a single experiment iteration."""

    iteration: int
    commit: str
    avg_score: float
    passed: int
    total: int
    kept: bool
    description: str


@dataclass
class ExperimentLoop:
    """Hill-climbing loop that keeps changes only when they improve scores.

    Parameters
    ----------
    project_root : Path | None
        Root of the project.  Auto-detected from git when *None*.
    tasks_dir : str
        Subdirectory name (relative to *project_root*) containing benchmark tasks.
    results_file : str
        Name of the TSV log file written inside *project_root*.
    """

    project_root: Path | None = None
    tasks_dir: str = "tasks"
    results_file: str = "results.tsv"
    use_sandbox: bool = True
    parallel: int = 10
    memory_mb: int = 64

    # Internal collaborators — built lazily in ``_setup``.
    _runner: TaskRunner = field(init=False, repr=False, default=None)  # type: ignore[assignment]
    _scorer: TaskScorer = field(init=False, repr=False, default=None)  # type: ignore[assignment]
    _history: ScoreHistory = field(init=False, repr=False, default=None)  # type: ignore[assignment]
    _meta: MetaAgent = field(init=False, repr=False, default=None)  # type: ignore[assignment]
    _sandbox: MicroVMSandbox | None = field(init=False, repr=False, default=None)
    _program_text: str = field(init=False, repr=False, default="")

    def _setup(self) -> None:
        if self.project_root is None:
            self.project_root = _find_project_root()

        program_path = self.project_root / "program.md"
        self._program_text = program_path.read_text() if program_path.exists() else ""

        tasks_path = self.project_root / self.tasks_dir
        results_path = self.project_root / self.results_file

        self._runner = TaskRunner(tasks_dir=tasks_path)
        self._scorer = TaskScorer()
        self._history = ScoreHistory(results_path=results_path)
        self._meta = MetaAgent(
            project_root=self.project_root,
            program_text=self._program_text,
        )
        if self.use_sandbox:
            self._sandbox = MicroVMSandbox(
                max_parallel=self.parallel,
                memory_mb=self.memory_mb,
            )
        else:
            self._sandbox = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_all_tasks(self, tasks: list[TaskDescriptor]) -> list[TaskResult]:
        """Run *tasks* using sandbox (parallel) or sequential subprocess."""
        if self._sandbox is not None:
            return _run_async(  # type: ignore[return-value]
                self._sandbox.run_tasks_parallel(tasks, runner=self._runner)
            )
        return [self._runner.run_task(t) for t in tasks]  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_benchmark(self) -> list[TaskResult]:
        """Run all benchmark tasks once and return the raw results."""
        self._setup()
        tasks = self._runner.discover_tasks()
        return self._run_all_tasks(tasks)  # type: ignore[arg-type]

    def show_scores(self) -> str:
        """Return a human-readable summary of the current score history."""
        self._setup()
        entries = self._history.read_entries()
        if not entries:
            return "No experiment results recorded yet."
        lines = ["timestamp\tcommit\tavg_score\tpassed\ttotal\tstatus\tdescription"]
        for e in entries:
            lines.append(
                f"{e['timestamp']}\t{e['commit']}\t{e['avg_score']}\t"
                f"{e['passed']}\t{e['total']}\t{e['status']}\t{e['description']}"
            )
        return "\n".join(lines)

    def run(self, max_iterations: int | None = None) -> list[IterationResult]:
        """Execute the self-improvement loop.

        Each iteration:
        1. Run benchmarks and record a baseline score.
        2. Ask the MetaAgent to diagnose failures and propose changes.
        3. Apply proposals, re-run benchmarks.
        4. Keep the changes if the score improved; revert otherwise.

        Returns the list of ``IterationResult`` for every iteration executed.
        """
        self._setup()
        results: list[IterationResult] = []
        iteration = 0

        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break

            iteration += 1

            # --- baseline ---
            tasks = self._runner.discover_tasks()
            baseline_results = self._run_all_tasks(tasks)  # type: ignore[arg-type]
            baseline_score = self._scorer.aggregate(baseline_results)
            baseline_passed = sum(1 for r in baseline_results if r["passed"])
            baseline_total = len(baseline_results)

            baseline_commit = _current_commit_sha()
            self._history.append(
                commit=baseline_commit,
                avg_score=baseline_score,
                passed=baseline_passed,
                total=baseline_total,
                status="baseline",
                description=f"iteration {iteration} baseline",
            )

            # --- diagnose + propose ---
            diagnosis = self._meta.diagnose(baseline_results)
            proposals = self._meta.propose_changes(diagnosis, results=baseline_results)

            if not proposals:
                results.append(
                    IterationResult(
                        iteration=iteration,
                        commit=baseline_commit,
                        avg_score=baseline_score,
                        passed=baseline_passed,
                        total=baseline_total,
                        kept=False,
                        description="no proposals generated — converged",
                    )
                )
                break

            # --- apply + measure ---
            self._meta.apply_changes(proposals)
            experiment_commit = _git_commit(f"experiment: iteration {iteration}")

            experiment_results = self._run_all_tasks(tasks)  # type: ignore[arg-type]
            experiment_score = self._scorer.aggregate(experiment_results)
            experiment_passed = sum(1 for r in experiment_results if r["passed"])
            experiment_total = len(experiment_results)

            change_size = len(proposals)
            keep = self._scorer.should_keep(baseline_score, experiment_score, change_size)

            if keep:
                status = "kept"
                description = (
                    f"iteration {iteration}: score {baseline_score:.3f} -> {experiment_score:.3f}"
                )
                self._history.append(
                    commit=experiment_commit,
                    avg_score=experiment_score,
                    passed=experiment_passed,
                    total=experiment_total,
                    status=status,
                    description=description,
                )
            else:
                status = "discarded"
                description = (
                    f"iteration {iteration}: score {baseline_score:.3f} -> "
                    f"{experiment_score:.3f} (reverted)"
                )
                _git_revert_last()
                self._history.append(
                    commit=_current_commit_sha(),
                    avg_score=baseline_score,
                    passed=baseline_passed,
                    total=baseline_total,
                    status=status,
                    description=description,
                )

            results.append(
                IterationResult(
                    iteration=iteration,
                    commit=experiment_commit if keep else baseline_commit,
                    avg_score=experiment_score if keep else baseline_score,
                    passed=experiment_passed if keep else baseline_passed,
                    total=experiment_total if keep else baseline_total,
                    kept=keep,
                    description=description,
                )
            )

        return results
