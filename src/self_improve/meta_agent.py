"""Meta-agent — diagnose failures and propose improvements via LLM.

When an ``ANTHROPIC_API_KEY`` is present the agent calls Claude to generate
concrete solution files for each failing benchmark task.  Without an API key
it degrades gracefully to deterministic placeholder proposals so the rest of
the self-improvement loop can still exercise its scaffold.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from .types import Proposal, TaskResult

logger = logging.getLogger(__name__)

_MODEL = "claude-opus-4-6"
_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# LLM client helpers (optional dependency)
# ---------------------------------------------------------------------------

def _anthropic_available() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def _call_llm(system: str, user: str) -> str:
    """Make a single blocking LLM call and return the text response."""
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    # Extract the first text block.
    for block in message.content:
        if hasattr(block, "text"):
            return block.text
    return ""


# ---------------------------------------------------------------------------
# MetaAgent
# ---------------------------------------------------------------------------

@dataclass
class MetaAgent:
    """Reads program directives and experiment history to propose changes.

    When ``ANTHROPIC_API_KEY`` is set in the environment, the agent uses
    Claude to generate real solution files for failing tasks.  Otherwise it
    falls back to deterministic placeholder proposals.
    """

    project_root: Path = field(default_factory=Path.cwd)
    program_text: str = ""

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------

    def diagnose(self, results: list[TaskResult]) -> str:
        """Analyse task results and return a human-readable diagnosis.

        Parameters
        ----------
        results:
            Each element must have at least ``name``, ``passed``, ``score``,
            and ``output`` keys.
        """
        if not results:
            return "No task results to diagnose."

        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        failed = total - passed
        avg_score = sum(r["score"] for r in results) / total if total else 0.0

        lines: list[str] = [
            f"Diagnosis: {passed}/{total} tasks passed (avg score {avg_score:.3f}).",
        ]

        if failed == 0:
            lines.append("All tasks passing — look for optimisation opportunities.")
            return "\n".join(lines)

        lines.append("")
        lines.append("Failed tasks:")
        for r in results:
            if not r["passed"]:
                snippet = (r.get("output") or "")[:300]
                lines.append(f"  - {r['name']}: {snippet}")

        timeout_failures = [r for r in results if "timed out" in (r.get("output") or "")]
        if timeout_failures:
            names = ", ".join(r["name"] for r in timeout_failures)
            lines.append(f"\nTimeout pattern detected in: {names}")

        import_failures = [
            r for r in results if "ModuleNotFoundError" in (r.get("output") or "")
        ]
        if import_failures:
            names = ", ".join(r["name"] for r in import_failures)
            lines.append(f"\nMissing-module pattern detected in: {names}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Proposals
    # ------------------------------------------------------------------

    def propose_changes(
        self,
        diagnosis: str,
        results: list[TaskResult] | None = None,
    ) -> list[Proposal]:
        """Generate a list of proposed modifications based on the diagnosis.

        When ``ANTHROPIC_API_KEY`` is set *and* ``results`` is provided, the
        agent calls Claude to write real solution files for each failing task.

        Returns an empty list when the diagnosis indicates full convergence
        (all tasks passing with no obvious improvements).
        """
        if "All tasks passing" in diagnosis:
            return []

        failing = [r for r in (results or []) if not r["passed"]]

        # LLM path — use Claude when credentials and task context are available.
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key and failing and _anthropic_available():
            try:
                return self._llm_proposals(failing)
            except Exception as exc:
                logger.warning("LLM proposal generation failed (%s); falling back to placeholders", exc)

        return self._placeholder_proposals(diagnosis)

    def _llm_proposals(self, failing: list[TaskResult]) -> list[Proposal]:
        """Ask Claude to write solution files for every failing task."""
        proposals: list[Proposal] = []

        for result in failing:
            task_dir = result.get("task_dir")
            if not task_dir:
                logger.debug("No task_dir in result for %s; skipping LLM proposal", result["name"])
                continue

            proposal = self._llm_proposal_for_task(result, task_dir)
            if proposal:
                proposals.append(proposal)

        return proposals

    def _llm_proposal_for_task(
        self, result: TaskResult, task_dir: str
    ) -> Proposal | None:
        """Generate a single solution-file proposal for one failing task."""
        task_path = Path(task_dir)
        name = result["name"]

        instruction = ""
        instruction_path = task_path / "instruction.md"
        if instruction_path.exists():
            instruction = instruction_path.read_text()

        test_sh = ""
        test_path = task_path / "tests" / "test.sh"
        if test_path.exists():
            test_sh = test_path.read_text()

        failure_output = (result.get("output") or "")[:1000]

        system = textwrap.dedent("""\
            You are a code-writing agent embedded in an automated benchmark harness.
            Your job: given a failing benchmark task, write the implementation file
            that will make all tests pass.

            Rules:
            - Output ONLY a JSON object — no prose, no markdown fences.
            - The JSON must have exactly two keys:
                "filename": the relative filename to create (e.g. "solution.py")
                "content":  the complete file content as a string
            - Write clean, correct, minimal Python unless the task requires otherwise.
        """)

        user_parts = [
            f"Task name: {name}",
            "",
        ]

        if self.program_text:
            user_parts += ["Program directives:", self.program_text, ""]

        if instruction:
            user_parts += ["Task instruction:", instruction, ""]

        if test_sh:
            user_parts += ["Test script (must pass):", "```bash", test_sh, "```", ""]

        if failure_output:
            user_parts += ["Current failure output:", failure_output, ""]

        user_parts.append("Write the solution JSON now.")

        user = "\n".join(user_parts)

        logger.info("Calling LLM for task %s ...", name)
        raw = _call_llm(system, user)

        return self._parse_llm_response(raw, name, task_dir)

    @staticmethod
    def _parse_llm_response(
        raw: str, task_name: str, task_dir: str
    ) -> Proposal | None:
        """Parse the LLM JSON response into a Proposal.  Returns None on failure."""
        raw = raw.strip()

        # Strip markdown code fences if the model added them despite instructions.
        if raw.startswith("```"):
            lines = raw.splitlines()
            inner = [
                ln for ln in lines[1:]
                if not ln.startswith("```")
            ]
            raw = "\n".join(inner).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse LLM response as JSON for task %s: %s", task_name, exc)
            logger.debug("Raw LLM response: %s", raw[:500])
            return None

        filename = data.get("filename")
        content = data.get("content")

        if not filename or content is None:
            logger.warning(
                "LLM response for task %s missing 'filename' or 'content' keys", task_name
            )
            return None

        return Proposal(
            file=filename,
            description=f"LLM-generated solution for {task_name}",
            content=content,
            task_name=task_name,
            task_dir=task_dir,
        )

    @staticmethod
    def _placeholder_proposals(diagnosis: str) -> list[Proposal]:
        """Return deterministic stub proposals when LLM is unavailable."""
        proposals: list[Proposal] = []

        if "Timeout pattern" in diagnosis:
            proposals.append(
                Proposal(
                    file="task_runner.py",
                    description="Increase default timeout or optimise slow tasks",
                    content="",  # no-op placeholder
                    task_name="",
                )
            )

        if "Missing-module pattern" in diagnosis:
            proposals.append(
                Proposal(
                    file="requirements.txt",
                    description="Add missing dependencies",
                    content="",  # no-op placeholder
                    task_name="",
                )
            )

        if not proposals:
            proposals.append(
                Proposal(
                    file="solution.py",
                    description="Placeholder: set ANTHROPIC_API_KEY to enable LLM-generated solutions",
                    content="# Placeholder — set ANTHROPIC_API_KEY to enable LLM-generated solutions\n",
                    task_name="",
                )
            )

        return proposals

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply_changes(self, proposals: list[Proposal]) -> None:
        """Write proposed file content to disk.

        For proposals with a ``task_dir``, the file is written inside that
        task directory.  For proposals without, the file is written relative
        to ``project_root``.

        Placeholder proposals (empty content, no task_dir) are skipped so
        they do not create spurious files.
        """
        for proposal in proposals:
            target = Path(proposal.resolve_path())

            # Skip placeholders that have no real content and no task dir.
            if not proposal.task_dir and not proposal.content.strip():
                logger.debug("Skipping placeholder proposal for %s", proposal.file)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(proposal.content)
            logger.info("Wrote %s (%d bytes)", target, len(proposal.content))
