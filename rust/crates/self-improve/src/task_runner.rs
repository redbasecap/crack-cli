//! Task discovery and execution.
//!
//! Each benchmark task lives in its own subdirectory and must contain:
//!   - `task.toml`         — metadata (`[task]` section: name, description, timeout)
//!   - `instruction.md`    — human-readable prompt (optional but recommended)
//!   - `tests/test.sh`     — verification script; exit 0 → pass, non-zero → fail

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Deserialize;
use tokio::process::Command;
use tracing::{debug, warn};

use crate::types::{TaskDescriptor, TaskResult};

// ---------------------------------------------------------------------------
// TOML schema for task.toml
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct TaskToml {
    task: TaskSection,
}

#[derive(Debug, Deserialize)]
struct TaskSection {
    #[serde(default)]
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default = "default_timeout")]
    timeout: u64,
}

fn default_timeout() -> u64 {
    120
}

// ---------------------------------------------------------------------------
// TaskRunner
// ---------------------------------------------------------------------------

/// Discovers and runs benchmark tasks from a directory tree.
#[derive(Debug, Clone)]
pub struct TaskRunner {
    /// Root directory containing task subdirectories.
    pub tasks_dir: PathBuf,
    /// Fallback timeout (seconds) when `task.toml` does not specify one.
    pub default_timeout_secs: u64,
}

impl TaskRunner {
    #[must_use]
    pub fn new(tasks_dir: impl Into<PathBuf>) -> Self {
        Self {
            tasks_dir: tasks_dir.into(),
            default_timeout_secs: 120,
        }
    }

    // ------------------------------------------------------------------
    // Discovery
    // ------------------------------------------------------------------

    /// Return descriptors for all tasks found under `tasks_dir`.
    ///
    /// A valid task directory must contain a `task.toml` file.  Entries
    /// that cannot be read are logged and skipped — they do not abort the
    /// discovery loop.
    pub fn discover_tasks(&self) -> Vec<TaskDescriptor> {
        let mut tasks = Vec::new();

        let read_dir = match std::fs::read_dir(&self.tasks_dir) {
            Ok(rd) => rd,
            Err(e) => {
                warn!("Cannot read tasks directory {:?}: {e}", self.tasks_dir);
                return tasks;
            }
        };

        let mut entries: Vec<_> = read_dir.flatten().collect();
        entries.sort_by_key(|e| e.path());

        for entry in entries {
            let task_dir = entry.path();
            if !task_dir.is_dir() {
                continue;
            }
            let toml_path = task_dir.join("task.toml");
            if !toml_path.exists() {
                continue;
            }

            match self.parse_task_dir(&task_dir, &toml_path) {
                Ok(desc) => tasks.push(desc),
                Err(e) => warn!("Skipping task at {task_dir:?}: {e}"),
            }
        }

        tasks
    }

    fn parse_task_dir(&self, task_dir: &Path, toml_path: &Path) -> Result<TaskDescriptor> {
        let raw = std::fs::read_to_string(toml_path)
            .with_context(|| format!("reading {toml_path:?}"))?;
        let parsed: TaskToml =
            toml::from_str(&raw).with_context(|| format!("parsing {toml_path:?}"))?;

        let name = if parsed.task.name.is_empty() {
            task_dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unnamed".to_string())
        } else {
            parsed.task.name
        };

        let timeout_secs = if parsed.task.timeout == 0 {
            self.default_timeout_secs
        } else {
            parsed.task.timeout
        };

        let instruction_path = task_dir.join("instruction.md");
        let instruction = std::fs::read_to_string(&instruction_path).unwrap_or_default();

        Ok(TaskDescriptor {
            name,
            description: parsed.task.description,
            timeout_secs,
            path: task_dir.to_path_buf(),
            instruction,
        })
    }

    // ------------------------------------------------------------------
    // Execution (single task)
    // ------------------------------------------------------------------

    /// Run one task asynchronously and return its result.
    pub async fn run_task(&self, task: &TaskDescriptor) -> TaskResult {
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

        let timeout = Duration::from_secs(task.timeout_secs);
        let start = Instant::now();

        match run_script(&test_script, &task.path, timeout).await {
            Ok((exit_ok, output)) => {
                let duration_secs = start.elapsed().as_secs_f64();
                TaskResult {
                    name: task.name.clone(),
                    passed: exit_ok,
                    score: if exit_ok { 1.0 } else { 0.0 },
                    output,
                    duration_secs,
                    task_dir: task.path.clone(),
                }
            }
            Err(e) => {
                let duration_secs = start.elapsed().as_secs_f64();
                TaskResult {
                    name: task.name.clone(),
                    passed: false,
                    score: 0.0,
                    output: e.to_string(),
                    duration_secs,
                    task_dir: task.path.clone(),
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Parallel execution
    // ------------------------------------------------------------------

    /// Run all `tasks` concurrently (bounded by `max_parallel`).
    pub async fn run_tasks_parallel(
        &self,
        tasks: &[TaskDescriptor],
        max_parallel: usize,
    ) -> Vec<TaskResult> {
        use tokio::sync::Semaphore;
        use std::sync::Arc;

        let sem = Arc::new(Semaphore::new(max_parallel.max(1)));
        let mut handles = Vec::with_capacity(tasks.len());

        for task in tasks {
            let task = task.clone();
            let runner = self.clone();
            let permit = Arc::clone(&sem);
            handles.push(tokio::spawn(async move {
                let _guard = permit.acquire_owned().await;
                runner.run_task(&task).await
            }));
        }

        let mut results = Vec::with_capacity(handles.len());
        for h in handles {
            match h.await {
                Ok(r) => results.push(r),
                Err(e) => {
                    // Should not happen, but surface as a failed task.
                    results.push(TaskResult {
                        name: "unknown".into(),
                        passed: false,
                        score: 0.0,
                        output: format!("task panicked: {e}"),
                        duration_secs: 0.0,
                        task_dir: PathBuf::new(),
                    });
                }
            }
        }
        results
    }
}

// ---------------------------------------------------------------------------
// Low-level subprocess helper
// ---------------------------------------------------------------------------

/// Run `script` with `bash` in `cwd` subject to `timeout`.
///
/// Returns `(exit_ok, combined_output)` on success, or an error if the
/// process could not be spawned or timed out.
async fn run_script(
    script: &Path,
    cwd: &Path,
    timeout: Duration,
) -> Result<(bool, String)> {
    debug!("Running script: {:?}", script);

    let child = Command::new("bash")
        .arg(script)
        .current_dir(cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning bash {:?}", script))?;

    // `wait_with_output` takes ownership of `child`.  When the returned Future
    // is dropped on timeout, Tokio automatically kills the child process.
    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => {
            let combined = String::from_utf8_lossy(&output.stdout).into_owned()
                + &String::from_utf8_lossy(&output.stderr);
            Ok((output.status.success(), combined.trim().to_string()))
        }
        Ok(Err(e)) => Err(e).context("waiting for child process"),
        Err(_) => {
            let secs = timeout.as_secs();
            Err(anyhow::anyhow!("task timed out after {secs}s"))
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_task_dir(root: &Path, name: &str, timeout: u64, test_exit: i32) -> PathBuf {
        let task_dir = root.join(name);
        let tests_dir = task_dir.join("tests");
        std::fs::create_dir_all(&tests_dir).unwrap();

        std::fs::write(
            task_dir.join("task.toml"),
            format!("[task]\nname = \"{name}\"\ndescription = \"test\"\ntimeout = {timeout}\n"),
        )
        .unwrap();
        std::fs::write(task_dir.join("instruction.md"), format!("# {name}")).unwrap();

        let script_content = format!("#!/usr/bin/env bash\nexit {test_exit}\n");
        let script_path = tests_dir.join("test.sh");
        std::fs::write(&script_path, script_content).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        task_dir
    }

    #[test]
    fn discover_finds_valid_tasks() {
        let tmp = TempDir::new().unwrap();
        make_task_dir(tmp.path(), "task-a", 60, 0);
        make_task_dir(tmp.path(), "task-b", 30, 1);

        let runner = TaskRunner::new(tmp.path());
        let tasks = runner.discover_tasks();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].name, "task-a");
        assert_eq!(tasks[1].name, "task-b");
        assert_eq!(tasks[0].timeout_secs, 60);
    }

    #[test]
    fn discover_skips_dirs_without_toml() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir(tmp.path().join("no-toml")).unwrap();
        make_task_dir(tmp.path(), "valid", 60, 0);

        let runner = TaskRunner::new(tmp.path());
        assert_eq!(runner.discover_tasks().len(), 1);
    }

    #[tokio::test]
    async fn run_passing_task() {
        let tmp = TempDir::new().unwrap();
        make_task_dir(tmp.path(), "pass", 10, 0);
        let runner = TaskRunner::new(tmp.path());
        let tasks = runner.discover_tasks();
        let result = runner.run_task(&tasks[0]).await;
        assert!(result.passed);
        assert!((result.score - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn run_failing_task() {
        let tmp = TempDir::new().unwrap();
        make_task_dir(tmp.path(), "fail", 10, 1);
        let runner = TaskRunner::new(tmp.path());
        let tasks = runner.discover_tasks();
        let result = runner.run_task(&tasks[0]).await;
        assert!(!result.passed);
        assert!((result.score - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn run_tasks_parallel_collects_all() {
        let tmp = TempDir::new().unwrap();
        make_task_dir(tmp.path(), "t1", 10, 0);
        make_task_dir(tmp.path(), "t2", 10, 1);
        make_task_dir(tmp.path(), "t3", 10, 0);
        let runner = TaskRunner::new(tmp.path());
        let tasks = runner.discover_tasks();
        let results = runner.run_tasks_parallel(&tasks, 4).await;
        assert_eq!(results.len(), 3);
        let passed = results.iter().filter(|r| r.passed).count();
        assert_eq!(passed, 2);
    }
}
