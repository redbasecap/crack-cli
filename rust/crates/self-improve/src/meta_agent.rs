//! Meta-agent — diagnose failures and propose improvements via LLM.
//!
//! When an `ANTHROPIC_API_KEY` (or `ANTHROPIC_AUTH_TOKEN`) is present the
//! agent calls Claude to generate concrete solution files for each failing
//! benchmark task.  Without credentials it degrades gracefully to
//! deterministic placeholder proposals so the rest of the loop still runs.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use api::{
    AnthropicClient, AuthSource, InputMessage, MessageRequest, OutputContentBlock,
};

use crate::types::{Proposal, TaskResult};

const AGENT_MODEL: &str = "claude-opus-4-6";
const AGENT_MAX_TOKENS: u32 = 4096;

// ---------------------------------------------------------------------------
// MetaAgent
// ---------------------------------------------------------------------------

/// Reads program directives and experiment history, then proposes improvements.
///
/// Uses the Anthropic API when credentials are available; falls back to
/// placeholder proposals otherwise.
pub struct MetaAgent {
    pub project_root: PathBuf,
    pub program_text: String,
    client: Option<AnthropicClient>,
}

impl MetaAgent {
    /// Create a new `MetaAgent`.  Attempts to initialise an `AnthropicClient`
    /// from environment variables; logs a warning and sets `client = None` if
    /// credentials are absent.
    #[must_use]
    pub fn new(project_root: impl Into<PathBuf>, program_text: impl Into<String>) -> Self {
        let client = match AuthSource::from_env() {
            Ok(auth) => {
                info!("MetaAgent: Anthropic credentials found — LLM proposals enabled");
                Some(AnthropicClient::from_auth(auth))
            }
            Err(e) => {
                warn!("MetaAgent: no Anthropic credentials ({e}) — using placeholder proposals");
                None
            }
        };

        Self {
            project_root: project_root.into(),
            program_text: program_text.into(),
            client,
        }
    }

    /// Returns `true` if the agent has valid API credentials.
    #[must_use]
    pub fn llm_enabled(&self) -> bool {
        self.client.is_some()
    }

    // ------------------------------------------------------------------
    // Diagnosis
    // ------------------------------------------------------------------

    /// Analyse task results and return a human-readable diagnosis string.
    #[must_use]
    pub fn diagnose(&self, results: &[TaskResult]) -> String {
        if results.is_empty() {
            return "No task results to diagnose.".to_string();
        }

        let total = results.len();
        let passed = results.iter().filter(|r| r.passed).count();
        let failed = total - passed;
        let avg_score: f64 = results.iter().map(|r| r.score).sum::<f64>() / total as f64;

        let mut lines = vec![format!(
            "Diagnosis: {passed}/{total} tasks passed (avg score {avg_score:.3})."
        )];

        if failed == 0 {
            lines.push("All tasks passing — look for optimisation opportunities.".into());
            return lines.join("\n");
        }

        lines.push(String::new());
        lines.push("Failed tasks:".into());
        for r in results.iter().filter(|r| !r.passed) {
            let snippet = r.output.chars().take(300).collect::<String>();
            lines.push(format!("  - {}: {snippet}", r.name));
        }

        let timeout_names: Vec<_> = results
            .iter()
            .filter(|r| r.output.contains("timed out"))
            .map(|r| r.name.as_str())
            .collect();
        if !timeout_names.is_empty() {
            lines.push(format!("\nTimeout pattern detected in: {}", timeout_names.join(", ")));
        }

        let import_names: Vec<_> = results
            .iter()
            .filter(|r| r.output.contains("ModuleNotFoundError"))
            .map(|r| r.name.as_str())
            .collect();
        if !import_names.is_empty() {
            lines.push(format!(
                "\nMissing-module pattern detected in: {}",
                import_names.join(", ")
            ));
        }

        lines.join("\n")
    }

    // ------------------------------------------------------------------
    // Proposals
    // ------------------------------------------------------------------

    /// Generate proposals for all failing tasks.
    ///
    /// Uses the LLM when available; returns placeholder proposals otherwise.
    pub async fn propose_changes(
        &self,
        diagnosis: &str,
        results: &[TaskResult],
    ) -> Vec<Proposal> {
        if diagnosis.contains("All tasks passing") {
            return Vec::new();
        }

        let failing: Vec<_> = results.iter().filter(|r| !r.passed).collect();

        if let Some(client) = &self.client {
            let mut proposals = Vec::new();
            for result in &failing {
                match self.llm_proposal(client, result).await {
                    Ok(Some(p)) => proposals.push(p),
                    Ok(None) => {}
                    Err(e) => {
                        warn!("LLM proposal failed for {}: {e}", result.name);
                    }
                }
            }
            if !proposals.is_empty() {
                return proposals;
            }
        }

        Self::placeholder_proposals(diagnosis)
    }

    async fn llm_proposal(
        &self,
        client: &AnthropicClient,
        result: &TaskResult,
    ) -> Result<Option<Proposal>> {
        let task_dir = &result.task_dir;
        let instruction = read_optional(task_dir.join("instruction.md"));
        let test_sh = read_optional(task_dir.join("tests").join("test.sh"));
        let failure_output: String = result.output.chars().take(1000).collect();

        let system = "\
You are a code-writing agent embedded in an automated benchmark harness. \
Your job: given a failing benchmark task, write the implementation file \
that will make all tests pass.\n\
\n\
Rules:\n\
- Output ONLY a JSON object — no prose, no markdown fences.\n\
- The JSON must have exactly two keys:\n\
    \"filename\": the relative filename to create (e.g. \"solution.py\")\n\
    \"content\":  the complete file content as a string\n\
- Write clean, correct, minimal Python unless the task requires otherwise.";

        let mut user_parts = vec![format!("Task name: {}\n", result.name)];

        if !self.program_text.is_empty() {
            user_parts.push(format!("Program directives:\n{}\n", self.program_text));
        }
        if !instruction.is_empty() {
            user_parts.push(format!("Task instruction:\n{instruction}\n"));
        }
        if !test_sh.is_empty() {
            user_parts.push(format!("Test script (must pass):\n```bash\n{test_sh}\n```\n"));
        }
        if !failure_output.is_empty() {
            user_parts.push(format!("Current failure output:\n{failure_output}\n"));
        }
        user_parts.push("Write the solution JSON now.".into());

        let user_text = user_parts.join("\n");

        let request = MessageRequest {
            model: AGENT_MODEL.to_string(),
            max_tokens: AGENT_MAX_TOKENS,
            messages: vec![InputMessage::user_text(user_text)],
            system: Some(system.to_string()),
            tools: None,
            tool_choice: None,
            stream: false,
        };

        info!("Calling LLM for task {} ...", result.name);

        let response = client
            .send_message(&request)
            .await
            .with_context(|| format!("LLM call for task {}", result.name))?;

        let raw_text = response
            .content
            .into_iter()
            .filter_map(|block| match block {
                OutputContentBlock::Text { text } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        parse_llm_response(&raw_text, &result.name, task_dir)
    }

    fn placeholder_proposals(diagnosis: &str) -> Vec<Proposal> {
        let mut proposals = Vec::new();

        if diagnosis.contains("Timeout pattern") {
            proposals.push(Proposal {
                file: "task_runner.py".into(),
                description: "Increase default timeout or optimise slow tasks".into(),
                content: String::new(),
                task_name: String::new(),
                task_dir: None,
            });
        }

        if diagnosis.contains("Missing-module pattern") {
            proposals.push(Proposal {
                file: "requirements.txt".into(),
                description: "Add missing dependencies".into(),
                content: String::new(),
                task_name: String::new(),
                task_dir: None,
            });
        }

        if proposals.is_empty() {
            proposals.push(Proposal {
                file: "solution.py".into(),
                description:
                    "Placeholder: set ANTHROPIC_API_KEY to enable LLM-generated solutions".into(),
                content:
                    "# Placeholder — set ANTHROPIC_API_KEY to enable LLM-generated solutions\n"
                        .into(),
                task_name: String::new(),
                task_dir: None,
            });
        }

        proposals
    }

    // ------------------------------------------------------------------
    // Apply
    // ------------------------------------------------------------------

    /// Write proposed file content to disk.
    ///
    /// Placeholder proposals (empty content, no `task_dir`) are skipped.
    pub fn apply_changes(&self, proposals: &[Proposal]) -> Result<()> {
        for proposal in proposals {
            if proposal.is_placeholder() {
                debug!("Skipping placeholder proposal for {}", proposal.file);
                continue;
            }

            let target = proposal.resolve_path();
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating directory {parent:?}"))?;
            }
            std::fs::write(&target, &proposal.content)
                .with_context(|| format!("writing {target:?}"))?;
            info!("Wrote {} ({} bytes)", target.display(), proposal.content.len());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn read_optional(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

fn parse_llm_response(
    raw: &str,
    task_name: &str,
    task_dir: &Path,
) -> Result<Option<Proposal>> {
    let trimmed = raw.trim();

    // Strip markdown fences if the model ignored our instructions.
    let json_str = if let Some(inner) = strip_fences(trimmed) {
        inner
    } else {
        trimmed.to_string()
    };

    let value: serde_json::Value = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => {
            warn!(
                "Cannot parse LLM response for {task_name} as JSON: {e}\n  (first 200 chars: {})",
                &json_str.chars().take(200).collect::<String>()
            );
            return Ok(None);
        }
    };

    let filename = value.get("filename").and_then(|v| v.as_str());
    let content = value.get("content").and_then(|v| v.as_str());

    match (filename, content) {
        (Some(f), Some(c)) => Ok(Some(Proposal {
            file: f.to_string(),
            description: format!("LLM-generated solution for {task_name}"),
            content: c.to_string(),
            task_name: task_name.to_string(),
            task_dir: Some(task_dir.to_path_buf()),
        })),
        _ => {
            warn!("LLM response for {task_name} missing 'filename' or 'content'");
            Ok(None)
        }
    }
}

fn strip_fences(s: &str) -> Option<String> {
    if !s.starts_with("```") {
        return None;
    }
    let lines: Vec<&str> = s.lines().collect();
    let inner: Vec<&str> = lines[1..].iter().filter(|l| !l.starts_with("```")).copied().collect();
    Some(inner.join("\n"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_result(name: &str, passed: bool, output: &str) -> TaskResult {
        TaskResult {
            name: name.to_string(),
            passed,
            score: if passed { 1.0 } else { 0.0 },
            output: output.to_string(),
            duration_secs: 0.0,
            task_dir: PathBuf::new(),
        }
    }

    #[test]
    fn diagnose_all_pass() {
        let agent = MetaAgent::new("/tmp", "");
        let results = vec![make_result("t1", true, "")];
        let diag = agent.diagnose(&results);
        assert!(diag.contains("All tasks passing"));
    }

    #[test]
    fn diagnose_failure() {
        let agent = MetaAgent::new("/tmp", "");
        let results = vec![make_result("t1", false, "some error")];
        let diag = agent.diagnose(&results);
        assert!(diag.contains("0/1 tasks passed"));
        assert!(diag.contains("t1"));
    }

    #[test]
    fn diagnose_timeout_pattern() {
        let agent = MetaAgent::new("/tmp", "");
        let results = vec![make_result("slow", false, "task timed out after 60s")];
        let diag = agent.diagnose(&results);
        assert!(diag.contains("Timeout pattern detected in: slow"));
    }

    #[test]
    fn parse_valid_llm_response() {
        let raw = r#"{"filename": "solution.py", "content": "x = 1\n"}"#;
        let proposal = parse_llm_response(raw, "t1", Path::new("/tmp/t1"))
            .unwrap()
            .unwrap();
        assert_eq!(proposal.file, "solution.py");
        assert_eq!(proposal.content, "x = 1\n");
    }

    #[test]
    fn parse_llm_response_strips_fences() {
        let raw = "```json\n{\"filename\": \"s.py\", \"content\": \"pass\"}\n```";
        let proposal = parse_llm_response(raw, "t1", Path::new("/tmp/t1"))
            .unwrap()
            .unwrap();
        assert_eq!(proposal.file, "s.py");
    }

    #[test]
    fn parse_llm_response_invalid_json() {
        let raw = "not json at all";
        let result = parse_llm_response(raw, "t1", Path::new("/tmp/t1")).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn placeholder_proposals_returned_without_key() {
        // No ANTHROPIC_API_KEY set in test environment → placeholders.
        let _agent = MetaAgent::new("/tmp", "");
        let diag = "0/1 tasks passed.";
        let proposals = MetaAgent::placeholder_proposals(diag);
        assert!(!proposals.is_empty());
    }

    #[test]
    fn apply_changes_writes_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let task_dir = tmp.path().join("task1");
        std::fs::create_dir(&task_dir).unwrap();

        let agent = MetaAgent::new(tmp.path(), "");
        let proposals = vec![Proposal {
            file: "solution.py".into(),
            description: "test".into(),
            content: "x = 42\n".into(),
            task_name: "task1".into(),
            task_dir: Some(task_dir.clone()),
        }];
        agent.apply_changes(&proposals).unwrap();

        let written = std::fs::read_to_string(task_dir.join("solution.py")).unwrap();
        assert_eq!(written, "x = 42\n");
    }
}
