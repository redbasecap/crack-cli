"""MicroVM sandbox — lightweight VM isolation for parallel task execution.

MicroVM is a RISC-V system emulator.  To run tasks inside a VM you need a
pre-built RISC-V Linux kernel image and a rootfs that contains the tools
your tasks need (bash, python3, etc.).

When a kernel image is available, tasks are mounted into the VM via 9P
filesystem sharing (``--share``) and executed inside the guest.

When no kernel is configured the sandbox falls back to async subprocess
isolation on the host — still parallel, just not VM-isolated.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default path where users can place a RISC-V Linux image.
_DEFAULT_KERNEL_PATH = Path.home() / ".forge" / "vm" / "Image"
_DEFAULT_ROOTFS_PATH = Path.home() / ".forge" / "vm" / "rootfs.img"


@dataclass
class MicroVMSandbox:
    """Manages MicroVM instances for isolated, parallel task execution.

    Falls back to async subprocess execution when ``microvm`` is not
    installed or no kernel image is configured.

    Parameters
    ----------
    max_parallel : int
        Maximum number of VMs to run concurrently.
    memory_mb : int
        DRAM allocation per VM in megabytes.
    vm_timeout : int
        Per-VM timeout in seconds.  The VM is killed if the task exceeds this.
    kernel_path : Path | None
        Path to a RISC-V Linux kernel image.  Overridden by ``$FORGE_VM_KERNEL``.
    rootfs_path : Path | None
        Path to a root filesystem image.  Overridden by ``$FORGE_VM_ROOTFS``.
    """

    max_parallel: int = 10
    memory_mb: int = 128
    vm_timeout: int = 120
    kernel_path: Path | None = None
    rootfs_path: Path | None = None

    _semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.max_parallel)
        # Env vars override constructor args.
        if os.environ.get("FORGE_VM_KERNEL"):
            self.kernel_path = Path(os.environ["FORGE_VM_KERNEL"])
        elif self.kernel_path is None and _DEFAULT_KERNEL_PATH.exists():
            self.kernel_path = _DEFAULT_KERNEL_PATH
        if os.environ.get("FORGE_VM_ROOTFS"):
            self.rootfs_path = Path(os.environ["FORGE_VM_ROOTFS"])
        elif self.rootfs_path is None and _DEFAULT_ROOTFS_PATH.exists():
            self.rootfs_path = _DEFAULT_ROOTFS_PATH

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return *True* if ``microvm`` binary is on ``$PATH``."""
        return shutil.which("microvm") is not None

    def vm_ready(self) -> bool:
        """Return *True* if both microvm binary AND kernel image are available."""
        return (
            self.is_available()
            and self.kernel_path is not None
            and self.kernel_path.exists()
        )

    def status(self) -> dict:
        """Return a summary dict suitable for CLI display."""
        available = self.is_available()
        vm_ok = self.vm_ready()
        return {
            "available": available,
            "binary": shutil.which("microvm") or "not found",
            "kernel": str(self.kernel_path) if self.kernel_path and self.kernel_path.exists() else "not configured",
            "rootfs": str(self.rootfs_path) if self.rootfs_path and self.rootfs_path.exists() else "not configured",
            "max_parallel": self.max_parallel,
            "memory_mb": self.memory_mb,
            "vm_timeout": self.vm_timeout,
            "mode": "microvm" if vm_ok else "subprocess (parallel, no VM isolation)",
        }

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    async def run_task(self, task: dict, *, runner: object | None = None) -> dict:
        """Run a single task inside an isolated MicroVM (or subprocess fallback)."""
        async with self._semaphore:
            if self.vm_ready():
                return await self._run_in_vm(task)
            return await self._run_in_subprocess(task, runner)

    async def run_tasks_parallel(
        self, tasks: list[dict], *, runner: object | None = None
    ) -> list[dict]:
        """Run *tasks* in parallel (up to *max_parallel* concurrently)."""
        coros = [self.run_task(t, runner=runner) for t in tasks]
        return list(await asyncio.gather(*coros))

    # ------------------------------------------------------------------
    # VM execution path — uses real microvm CLI flags
    # ------------------------------------------------------------------

    async def _run_in_vm(self, task: dict) -> dict:
        """Execute *task* inside a MicroVM via 9P filesystem sharing.

        The task directory is shared into the guest at ``/mnt`` using
        ``--share``.  The guest kernel is expected to mount 9P on boot
        and execute ``/mnt/tests/test.sh``.
        """
        task_dir = Path(task["path"])
        test_script = task_dir / "tests" / "test.sh"
        name = task["name"]

        if not test_script.exists():
            return {
                "name": name,
                "passed": False,
                "score": 0.0,
                "output": f"test script not found: {test_script}",
                "duration": 0.0,
                "task_dir": str(task_dir),
            }

        assert self.kernel_path is not None  # guaranteed by vm_ready()

        cmd = [
            "microvm", "run",
            "--kernel", str(self.kernel_path),
            "--memory", str(self.memory_mb),
            "--share", str(task_dir),
            "--timeout-secs", str(self.vm_timeout),
            "--cmdline", "console=ttyS0 earlycon=sbi init=/mnt/tests/test.sh",
        ]
        if self.rootfs_path and self.rootfs_path.exists():
            cmd.extend(["--disk", str(self.rootfs_path)])

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(task_dir),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.vm_timeout + 5
            )
            elapsed = time.monotonic() - start
            output = (stdout or b"").decode() + (stderr or b"").decode()
            passed = proc.returncode == 0
            return {
                "name": name,
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "output": output.strip(),
                "duration": round(elapsed, 3),
                "task_dir": str(task_dir),
            }
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            logger.warning("VM timed out for task %s after %ds", name, self.vm_timeout)
            return {
                "name": name,
                "passed": False,
                "score": 0.0,
                "output": f"VM timed out after {self.vm_timeout}s",
                "duration": round(elapsed, 3),
                "task_dir": str(task_dir),
            }
        except Exception as exc:
            elapsed = time.monotonic() - start
            return {
                "name": name,
                "passed": False,
                "score": 0.0,
                "output": f"VM error: {exc}",
                "duration": round(elapsed, 3),
                "task_dir": str(task_dir),
            }

    # ------------------------------------------------------------------
    # Subprocess fallback path
    # ------------------------------------------------------------------

    async def _run_in_subprocess(
        self, task: dict, runner: object | None
    ) -> dict:
        """Fall back to async subprocess execution (no VM)."""
        from pathlib import Path

        task_dir = Path(task["path"])
        test_script = task_dir / "tests" / "test.sh"
        name = task["name"]
        timeout = task.get("timeout", self.vm_timeout)

        if not test_script.exists():
            return {
                "name": name,
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
            return {
                "name": name,
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "output": output.strip(),
                "duration": round(elapsed, 3),
                "task_dir": str(task_dir),
            }
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            return {
                "name": name,
                "passed": False,
                "score": 0.0,
                "output": f"task timed out after {timeout}s",
                "duration": round(elapsed, 3),
                "task_dir": str(task_dir),
            }
        except Exception as exc:
            elapsed = time.monotonic() - start
            return {
                "name": name,
                "passed": False,
                "score": 0.0,
                "output": f"unexpected error: {exc}",
                "duration": round(elapsed, 3),
                "task_dir": str(task_dir),
            }
