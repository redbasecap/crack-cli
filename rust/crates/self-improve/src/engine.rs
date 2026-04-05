//! Hill-climbing experiment loop.
//!
//! Each iteration:
//! 1. Run all benchmark tasks → measure baseline score.
//! 2. Ask `MetaAgent` to diagnose failures + propose changes.
//! 3. Apply proposals → commit.
//! 4. Re-run benchmarks → keep if improved, revert otherwise.
//! 5. Repeat until converged or `max_iterations` reached.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use tracing::{info, warn};

use crate::meta_agent::MetaAgent;
use crate::sandbox::MicroVMSandbox;
use crate::scorer::{utc_timestamp, ScoreHistory, TaskScorer};
use crate::task_runner::TaskRunner;
use crate::types::{IterationResult, ScoreEntry, TaskDescriptor, TaskResult};

// ---------------------------------------------------------------------------
// Git helpers
// ---------------------------------------------------------------------------

fn git(args: &[&str], cwd: &Path) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .with_context(|| format!("git {}", args.join(" ")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git {} failed: {stderr}", args.join(" "));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn current_commit(cwd: &Path) -> Result<String> {
    git(&["rev-parse", "--short", "HEAD"], cwd)
}

fn git_commit(message: &str, cwd: &Path) -> Result<String> {
    git(&["add", "-A"], cwd)?;
    git(&["commit", "-m", message, "--allow-empty"], cwd)?;
    current_commit(cwd)
}

fn git_revert_last(cwd: &Path) -> Result<()> {
    git(&["revert", "HEAD", "--no-edit"], cwd)?;
    Ok(())
}

fn find_project_root() -> Result<PathBuf> {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("git rev-parse --show-toplevel")?;
    if !out.status.success() {
        bail!("not inside a git repository");
    }
    Ok(PathBuf::from(
        String::from_utf8_lossy(&out.stdout).trim().to_string(),
    ))
}

// ---------------------------------------------------------------------------
// ExperimentLoop configuration
// ---------------------------------------------------------------------------

/// Configuration for `ExperimentLoop`.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// Root of the project.  Auto-detected from git when `None`.
    pub project_root: Option<PathBuf>,
    /// Subdirectory containing benchmark tasks (relative to project root).
    pub tasks_dir: String,
    /// TSV log filename (relative to project root).
    pub results_file: String,
    /// Enable MicroVM sandbox (falls back to subprocess when VM not available).
    pub use_sandbox: bool,
    /// Maximum parallel VMs / processes.
    pub parallel: usize,
    /// DRAM per VM (megabytes).
    pub memory_mb: u64,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            project_root: None,
            tasks_dir: "tasks".into(),
            results_file: "results.tsv".into(),
            use_sandbox: true,
            parallel: 10,
            memory_mb: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// ExperimentLoop
// ---------------------------------------------------------------------------

/// Hill-climbing optimizer over benchmark task scores.
pub struct ExperimentLoop {
    config: ExperimentConfig,
    project_root: PathBuf,
    runner: TaskRunner,
    #[allow(dead_code)] // kept for future extensibility (custom scoring strategies)
    scorer: TaskScorer,
    history: ScoreHistory,
    meta: MetaAgent,
    sandbox: Option<MicroVMSandbox>,
}

impl ExperimentLoop {
    /// Build an `ExperimentLoop` from the given configuration.
    pub fn new(mut config: ExperimentConfig) -> Result<Self> {
        let project_root = match config.project_root.take() {
            Some(p) => p,
            None => find_project_root().context("locating project root")?,
        };

        let program_text = {
            let p = project_root.join("program.md");
            std::fs::read_to_string(&p).unwrap_or_default()
        };

        let tasks_path = project_root.join(&config.tasks_dir);
        let results_path = project_root.join(&config.results_file);

        let sandbox = if config.use_sandbox {
            let mut sb = MicroVMSandbox::new();
            sb.max_parallel = config.parallel;
            sb.memory_mb = config.memory_mb;
            Some(sb)
        } else {
            None
        };

        Ok(Self {
            runner: TaskRunner::new(tasks_path),
            scorer: TaskScorer,
            history: ScoreHistory::new(results_path),
            meta: MetaAgent::new(&project_root, program_text),
            sandbox,
            project_root,
            config,
        })
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    async fn run_all_tasks(&self, tasks: &[TaskDescriptor]) -> Vec<TaskResult> {
        match &self.sandbox {
            Some(sb) if sb.vm_ready() => {
                info!("Running {} tasks in MicroVM sandbox", tasks.len());
                sb.run_tasks_parallel(tasks, &self.runner).await
            }
            _ => {
                info!(
                    "Running {} tasks with subprocess (max_parallel={})",
                    tasks.len(),
                    self.config.parallel
                );
                self.runner
                    .run_tasks_parallel(tasks, self.config.parallel)
                    .await
            }
        }
    }

    fn append_score(
        &self,
        commit: &str,
        avg_score: f64,
        passed: usize,
        total: usize,
        status: &str,
        description: &str,
    ) {
        let entry = ScoreEntry {
            timestamp: utc_timestamp(),
            commit: commit.to_string(),
            avg_score,
            passed,
            total,
            status: status.to_string(),
            description: description.to_string(),
        };
        if let Err(e) = self.history.append(&entry) {
            warn!("Failed to append score entry: {e}");
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Run all benchmark tasks once and return raw results.
    pub async fn run_benchmark(&self) -> Result<Vec<TaskResult>> {
        let tasks = self.runner.discover_tasks();
        Ok(self.run_all_tasks(&tasks).await)
    }

    /// Return a formatted summary of the score history.
    pub fn show_scores(&self) -> Result<String> {
        let entries = self.history.read_entries()?;
        if entries.is_empty() {
            return Ok("No experiment results recorded yet.".into());
        }
        let mut lines =
            vec!["timestamp\tcommit\tavg_score\tpassed\ttotal\tstatus\tdescription".to_string()];
        for e in &entries {
            lines.push(format!(
                "{}\t{}\t{:.4}\t{}\t{}\t{}\t{}",
                e.timestamp, e.commit, e.avg_score, e.passed, e.total, e.status, e.description
            ));
        }
        Ok(lines.join("\n"))
    }

    /// Execute the self-improvement loop.
    ///
    /// Returns results for every iteration that was attempted.
    pub async fn run(&self, max_iterations: Option<usize>) -> Result<Vec<IterationResult>> {
        let mut results: Vec<IterationResult> = Vec::new();
        let cwd = &self.project_root;
        let mut iteration = 0_usize;

        loop {
            if max_iterations.map_or(false, |m| iteration >= m) {
                break;
            }
            iteration += 1;

            // ---- baseline ----
            let tasks = self.runner.discover_tasks();
            let baseline_results = self.run_all_tasks(&tasks).await;
            let baseline_score = TaskScorer::aggregate(&baseline_results);
            let baseline_passed = baseline_results.iter().filter(|r| r.passed).count();
            let baseline_total = baseline_results.len();
            let baseline_commit = current_commit(cwd)?;

            info!(
                "Iteration {iteration}: baseline score={baseline_score:.3} \
                 passed={baseline_passed}/{baseline_total}"
            );

            self.append_score(
                &baseline_commit,
                baseline_score,
                baseline_passed,
                baseline_total,
                "baseline",
                &format!("iteration {iteration} baseline"),
            );

            // ---- diagnose + propose ----
            let diagnosis = self.meta.diagnose(&baseline_results);
            let proposals = self.meta.propose_changes(&diagnosis, &baseline_results).await;

            if proposals.is_empty() {
                info!("Iteration {iteration}: no proposals — converged.");
                results.push(IterationResult {
                    iteration,
                    commit: baseline_commit,
                    avg_score: baseline_score,
                    passed: baseline_passed,
                    total: baseline_total,
                    kept: false,
                    description: "no proposals generated — converged".into(),
                });
                break;
            }

            // ---- apply + measure ----
            self.meta.apply_changes(&proposals)?;
            let experiment_commit =
                git_commit(&format!("experiment: iteration {iteration}"), cwd)?;

            let experiment_results = self.run_all_tasks(&tasks).await;
            let experiment_score = TaskScorer::aggregate(&experiment_results);
            let experiment_passed = experiment_results.iter().filter(|r| r.passed).count();
            let experiment_total = experiment_results.len();

            let keep = TaskScorer::should_keep(baseline_score, experiment_score, proposals.len());

            let (status, description, final_commit, final_score, final_passed, final_total) =
                if keep {
                    let desc = format!(
                        "iteration {iteration}: score {baseline_score:.3} -> {experiment_score:.3}"
                    );
                    info!("KEPT: {desc}");
                    self.append_score(
                        &experiment_commit,
                        experiment_score,
                        experiment_passed,
                        experiment_total,
                        "kept",
                        &desc,
                    );
                    (
                        "kept",
                        desc,
                        experiment_commit,
                        experiment_score,
                        experiment_passed,
                        experiment_total,
                    )
                } else {
                    let desc = format!(
                        "iteration {iteration}: score {baseline_score:.3} -> \
                         {experiment_score:.3} (reverted)"
                    );
                    warn!("DISCARDED: {desc}");
                    if let Err(e) = git_revert_last(cwd) {
                        warn!("git revert failed: {e}");
                    }
                    let revert_commit = current_commit(cwd).unwrap_or_else(|_| baseline_commit.clone());
                    self.append_score(
                        &revert_commit,
                        baseline_score,
                        baseline_passed,
                        baseline_total,
                        "discarded",
                        &desc,
                    );
                    (
                        "discarded",
                        desc,
                        revert_commit,
                        baseline_score,
                        baseline_passed,
                        baseline_total,
                    )
                };

            results.push(IterationResult {
                iteration,
                commit: final_commit,
                avg_score: final_score,
                passed: final_passed,
                total: final_total,
                kept: status == "kept",
                description,
            });
        }

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experiment_config_defaults() {
        let cfg = ExperimentConfig::default();
        assert_eq!(cfg.tasks_dir, "tasks");
        assert_eq!(cfg.results_file, "results.tsv");
        assert!(cfg.use_sandbox);
        assert_eq!(cfg.parallel, 10);
    }
}
