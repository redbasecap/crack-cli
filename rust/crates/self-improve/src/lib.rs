//! Production-grade self-improvement harness for Forge CLI.
//!
//! # Architecture
//!
//! ```text
//! ExperimentLoop
//!   ├── TaskRunner        — discovers + runs benchmark tasks
//!   ├── MicroVMSandbox    — parallel isolation (VM or subprocess fallback)
//!   ├── MetaAgent         — diagnoses failures, calls LLM for solution files
//!   └── ScoreHistory      — append-only TSV experiment log
//! ```
//!
//! # Quick start
//!
//! ```no_run
//! use self_improve::engine::{ExperimentConfig, ExperimentLoop};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let cfg = ExperimentConfig::default();
//!     let lp = ExperimentLoop::new(cfg)?;
//!     let results = lp.run(Some(3)).await?;
//!     for r in &results {
//!         println!("[{}] iter={} score={:.3}", if r.kept { "KEPT" } else { "DISC" }, r.iteration, r.avg_score);
//!     }
//!     Ok(())
//! }
//! ```

pub mod engine;
pub mod meta_agent;
pub mod sandbox;
pub mod scorer;
pub mod task_runner;
pub mod types;

// Re-export the most commonly used items.
pub use engine::{ExperimentConfig, ExperimentLoop};
pub use meta_agent::MetaAgent;
pub use sandbox::{MicroVMSandbox, SandboxStatus};
pub use scorer::{ScoreHistory, TaskScorer};
pub use task_runner::TaskRunner;
pub use types::{IterationResult, Proposal, ScoreEntry, TaskDescriptor, TaskResult};
