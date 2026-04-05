//! MicroVM sandbox — isolated, parallel task execution.
//!
//! When a `microvm` binary and a RISC-V kernel image are available, each task
//! runs inside its own lightweight VM.  The task directory is shared into the
//! guest via 9P (``--share``), and the guest kernel mounts it at ``/mnt`` and
//! executes ``/mnt/tests/test.sh``.
//!
//! When the binary or kernel is absent the sandbox transparently falls back to
//! async subprocess execution on the host — still parallel, just not VM-isolated.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::process::Command;
use tokio::sync::Semaphore;
use tracing::{debug, warn};

use crate::task_runner::TaskRunner;
use crate::types::{TaskDescriptor, TaskResult};

// ---------------------------------------------------------------------------
// Default paths
// ---------------------------------------------------------------------------

fn default_kernel_path() -> PathBuf {
    dirs_home().join(".forge").join("vm").join("Image")
}

fn default_rootfs_path() -> PathBuf {
    dirs_home().join(".forge").join("vm").join("rootfs.img")
}

fn dirs_home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

// ---------------------------------------------------------------------------
// MicroVMSandbox
// ---------------------------------------------------------------------------

/// Manages MicroVM instances for isolated, parallel task execution.
///
/// Falls back to async subprocess execution when ``microvm`` is not installed
/// or no kernel image is configured.
#[derive(Debug, Clone)]
pub struct MicroVMSandbox {
    /// Maximum number of concurrent VMs (or processes).
    pub max_parallel: usize,
    /// DRAM per VM in megabytes.
    pub memory_mb: u64,
    /// Per-VM (or per-process) timeout in seconds.
    pub vm_timeout_secs: u64,
    /// Path to the RISC-V Linux kernel image.
    pub kernel_path: Option<PathBuf>,
    /// Optional root filesystem image.
    pub rootfs_path: Option<PathBuf>,
}

impl Default for MicroVMSandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl MicroVMSandbox {
    #[must_use]
    pub fn new() -> Self {
        let kernel_path = std::env::var_os("FORGE_VM_KERNEL")
            .map(PathBuf::from)
            .or_else(|| {
                let p = default_kernel_path();
                p.exists().then_some(p)
            });

        let rootfs_path = std::env::var_os("FORGE_VM_ROOTFS")
            .map(PathBuf::from)
            .or_else(|| {
                let p = default_rootfs_path();
                p.exists().then_some(p)
            });

        Self {
            max_parallel: 10,
            memory_mb: 128,
            vm_timeout_secs: 120,
            kernel_path,
            rootfs_path,
        }
    }

    // ------------------------------------------------------------------
    // Status helpers
    // ------------------------------------------------------------------

    /// Returns `true` if the `microvm` binary is on `$PATH`.
    #[must_use]
    pub fn binary_available(&self) -> bool {
        which::which("microvm").is_ok()
    }

    /// Returns `true` if both the binary AND a kernel image are present.
    #[must_use]
    pub fn vm_ready(&self) -> bool {
        self.binary_available()
            && self
                .kernel_path
                .as_deref()
                .map_or(false, Path::exists)
    }

    /// Return a summary dict suitable for CLI display.
    #[must_use]
    pub fn status(&self) -> SandboxStatus {
        SandboxStatus {
            binary: which::which("microvm")
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "not found".into()),
            kernel: self
                .kernel_path
                .as_deref()
                .filter(|p| p.exists())
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "not configured".into()),
            rootfs: self
                .rootfs_path
                .as_deref()
                .filter(|p| p.exists())
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "not configured".into()),
            max_parallel: self.max_parallel,
            memory_mb: self.memory_mb,
            vm_timeout_secs: self.vm_timeout_secs,
            mode: if self.vm_ready() {
                "microvm".into()
            } else {
                "subprocess (parallel, no VM isolation)".into()
            },
        }
    }

    // ------------------------------------------------------------------
    // Parallel execution
    // ------------------------------------------------------------------

    /// Run `tasks` in parallel (bounded by `max_parallel`).
    ///
    /// Uses VM isolation when available; falls back to async subprocesses.
    pub async fn run_tasks_parallel(
        &self,
        tasks: &[TaskDescriptor],
        runner: &TaskRunner,
    ) -> Vec<TaskResult> {
        let sem = Arc::new(Semaphore::new(self.max_parallel.max(1)));
        let mut handles = Vec::with_capacity(tasks.len());

        for task in tasks {
            let task = task.clone();
            let sandbox = self.clone();
            let runner = runner.clone();
            let sem = Arc::clone(&sem);

            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await;
                if sandbox.vm_ready() {
                    sandbox.run_in_vm(&task).await
                } else {
                    runner.run_task(&task).await
                }
            }));
        }

        let mut results = Vec::with_capacity(handles.len());
        for h in handles {
            match h.await {
                Ok(r) => results.push(r),
                Err(e) => results.push(TaskResult {
                    name: "unknown".into(),
                    passed: false,
                    score: 0.0,
                    output: format!("task panicked: {e}"),
                    duration_secs: 0.0,
                    task_dir: PathBuf::new(),
                }),
            }
        }
        results
    }

    // ------------------------------------------------------------------
    // VM execution path
    // ------------------------------------------------------------------

    async fn run_in_vm(&self, task: &TaskDescriptor) -> TaskResult {
        let test_script = task.path.join("tests").join("test.sh");

        if !test_script.exists() {
            return TaskResult {
                name: task.name.clone(),
                passed: false,
                score: 0.0,
                output: format!("test script not found: {}", test_script.display()),
                duration_secs: 0.0,
                task_dir: task.path.clone(),
            };
        }

        let kernel = self.kernel_path.as_ref().unwrap(); // guaranteed by vm_ready()

        let mut cmd = Command::new("microvm");
        cmd.arg("run")
            .arg("--kernel")
            .arg(kernel)
            .arg("--memory")
            .arg(self.memory_mb.to_string())
            .arg("--share")
            .arg(&task.path)
            .arg("--timeout-secs")
            .arg(self.vm_timeout_secs.to_string())
            .arg("--cmdline")
            .arg("console=ttyS0 earlycon=sbi init=/mnt/tests/test.sh")
            .current_dir(&task.path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        if let Some(rootfs) = &self.rootfs_path {
            if rootfs.exists() {
                cmd.arg("--disk").arg(rootfs);
            }
        }

        let start = Instant::now();
        let timeout = Duration::from_secs(self.vm_timeout_secs + 5);

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return self.vm_error_result(task, &format!("spawn: {e}"), start),
        };

        // `wait_with_output` takes ownership; Tokio kills the child when dropped.
        match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => {
                let duration_secs = start.elapsed().as_secs_f64();
                let combined = String::from_utf8_lossy(&output.stdout).into_owned()
                    + &String::from_utf8_lossy(&output.stderr);
                let passed = output.status.success();
                TaskResult {
                    name: task.name.clone(),
                    passed,
                    score: if passed { 1.0 } else { 0.0 },
                    output: combined.trim().to_string(),
                    duration_secs,
                    task_dir: task.path.clone(),
                }
            }
            Ok(Err(e)) => self.vm_error_result(task, &e.to_string(), start),
            Err(_) => {
                warn!(
                    "VM timed out for task {} after {}s",
                    task.name, self.vm_timeout_secs
                );
                TaskResult {
                    name: task.name.clone(),
                    passed: false,
                    score: 0.0,
                    output: format!("VM timed out after {}s", self.vm_timeout_secs),
                    duration_secs: start.elapsed().as_secs_f64(),
                    task_dir: task.path.clone(),
                }
            }
        }
    }

    fn vm_error_result(&self, task: &TaskDescriptor, msg: &str, start: Instant) -> TaskResult {
        debug!("VM error for {}: {msg}", task.name);
        TaskResult {
            name: task.name.clone(),
            passed: false,
            score: 0.0,
            output: format!("VM error: {msg}"),
            duration_secs: start.elapsed().as_secs_f64(),
            task_dir: task.path.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Status type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SandboxStatus {
    pub binary: String,
    pub kernel: String,
    pub rootfs: String,
    pub max_parallel: usize,
    pub memory_mb: u64,
    pub vm_timeout_secs: u64,
    pub mode: String,
}

impl SandboxStatus {
    pub fn display(&self) -> String {
        format!(
            "MicroVM Sandbox Status\n  Binary:       {}\n  Kernel:       {}\n  Rootfs:       {}\n  Mode:         {}\n  Max parallel: {}\n  Memory/VM:    {}MB\n  VM timeout:   {}s",
            self.binary,
            self.kernel,
            self.rootfs,
            self.mode,
            self.max_parallel,
            self.memory_mb,
            self.vm_timeout_secs,
        )
    }
}
