//! Task scoring and append-only TSV experiment log.

use std::path::PathBuf;

use anyhow::{Context, Result};
use csv::WriterBuilder;
use tracing::debug;

use crate::types::{ScoreEntry, TaskResult};

// ---------------------------------------------------------------------------
// TaskScorer
// ---------------------------------------------------------------------------

/// Computes aggregate scores from task results and decides whether to keep
/// an experimental change.
pub struct TaskScorer;

impl TaskScorer {
    /// Arithmetic mean of all task scores.  Returns `0.0` for an empty slice.
    #[must_use]
    pub fn aggregate(results: &[TaskResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64
    }

    /// Decide whether experimental changes should be kept.
    ///
    /// - **Keep** if `new_score > old_score`.
    /// - **Keep** if scores are equal *and* `change_size < 3` (prefer simpler).
    /// - **Discard** otherwise.
    #[must_use]
    pub fn should_keep(old_score: f64, new_score: f64, change_size: usize) -> bool {
        if new_score > old_score {
            return true;
        }
        new_score >= old_score - f64::EPSILON && change_size < 3
    }
}

// ---------------------------------------------------------------------------
// ScoreHistory
// ---------------------------------------------------------------------------

/// Append-only TSV log of experiment results.
///
/// Columns (tab-separated):
/// `timestamp`, `commit`, `avg_score`, `passed`, `total`, `status`, `description`
pub struct ScoreHistory {
    pub results_path: PathBuf,
}

impl ScoreHistory {
    #[must_use]
    pub fn new(results_path: impl Into<PathBuf>) -> Self {
        Self {
            results_path: results_path.into(),
        }
    }

    // ------------------------------------------------------------------
    // Write
    // ------------------------------------------------------------------

    fn ensure_header(&self) -> Result<()> {
        if self.results_path.exists() {
            return Ok(());
        }
        if let Some(parent) = self.results_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating directory {:?}", parent))?;
        }
        let mut wtr = WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(&self.results_path)
            .with_context(|| format!("creating {:?}", self.results_path))?;
        wtr.write_record([
            "timestamp",
            "commit",
            "avg_score",
            "passed",
            "total",
            "status",
            "description",
        ])?;
        wtr.flush()?;
        Ok(())
    }

    /// Append a single result row to the log.
    pub fn append(&self, entry: &ScoreEntry) -> Result<()> {
        self.ensure_header()?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.results_path)
            .with_context(|| format!("opening {:?}", self.results_path))?;
        let mut wtr = WriterBuilder::new().delimiter(b'\t').from_writer(file);
        wtr.write_record([
            &entry.timestamp,
            &entry.commit,
            &format!("{:.4}", entry.avg_score),
            &entry.passed.to_string(),
            &entry.total.to_string(),
            &entry.status,
            &entry.description,
        ])?;
        wtr.flush()?;
        debug!("Appended score entry: {:?}", entry.description);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Read
    // ------------------------------------------------------------------

    /// Return all rows as a `Vec<ScoreEntry>`.  Returns empty on missing file.
    pub fn read_entries(&self) -> Result<Vec<ScoreEntry>> {
        if !self.results_path.exists() {
            return Ok(Vec::new());
        }
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_path(&self.results_path)
            .with_context(|| format!("reading {:?}", self.results_path))?;

        let mut entries = Vec::new();
        for record in rdr.records() {
            let record = record?;
            entries.push(ScoreEntry {
                timestamp: record.get(0).unwrap_or("").to_string(),
                commit: record.get(1).unwrap_or("").to_string(),
                avg_score: record.get(2).unwrap_or("0").parse().unwrap_or(0.0),
                passed: record.get(3).unwrap_or("0").parse().unwrap_or(0),
                total: record.get(4).unwrap_or("0").parse().unwrap_or(0),
                status: record.get(5).unwrap_or("").to_string(),
                description: record.get(6).unwrap_or("").to_string(),
            });
        }
        Ok(entries)
    }

    /// Latest average score, or `None` if the log is empty.
    pub fn latest_score(&self) -> Result<Option<f64>> {
        let entries = self.read_entries()?;
        Ok(entries.last().map(|e| e.avg_score))
    }
}

// ---------------------------------------------------------------------------
// Timestamp helper
// ---------------------------------------------------------------------------

/// Return an ISO-8601 timestamp string for the current UTC instant using only
/// `std::time` (no `chrono` dependency).
#[must_use]
pub fn utc_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Decompose seconds into date and time components.
    let sec_of_day = secs % 86_400;
    let h = sec_of_day / 3600;
    let m = (sec_of_day % 3600) / 60;
    let s = sec_of_day % 60;

    let days = (secs / 86_400) as u32;
    let (y, mo, d) = days_to_ymd(days);

    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Civil-from-days algorithm (Gregorian calendar).
fn days_to_ymd(z: u32) -> (u32, u32, u32) {
    let z = z + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use crate::types::TaskResult;

    fn fake_result(passed: bool, score: f64) -> TaskResult {
        TaskResult {
            name: "t".into(),
            passed,
            score,
            output: String::new(),
            duration_secs: 0.0,
            task_dir: PathBuf::new(),
        }
    }

    #[test]
    fn aggregate_empty() {
        assert!((TaskScorer::aggregate(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn aggregate_mixed() {
        let results = vec![fake_result(true, 1.0), fake_result(false, 0.0)];
        assert!((TaskScorer::aggregate(&results) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn should_keep_strict_improvement() {
        assert!(TaskScorer::should_keep(0.5, 0.8, 5));
    }

    #[test]
    fn should_keep_equal_small_change() {
        assert!(TaskScorer::should_keep(0.5, 0.5, 2));
    }

    #[test]
    fn should_discard_equal_large_change() {
        assert!(!TaskScorer::should_keep(0.5, 0.5, 4));
    }

    #[test]
    fn should_discard_regression() {
        assert!(!TaskScorer::should_keep(0.8, 0.5, 1));
    }

    #[test]
    fn tsv_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let history = ScoreHistory::new(tmp.path().join("results.tsv"));

        let entry = ScoreEntry {
            timestamp: "2026-01-01T00:00:00Z".into(),
            commit: "abc1234".into(),
            avg_score: 0.75,
            passed: 3,
            total: 4,
            status: "kept".into(),
            description: "test entry".into(),
        };
        history.append(&entry).unwrap();

        let read = history.read_entries().unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].commit, "abc1234");
        assert!((read[0].avg_score - 0.75).abs() < 1e-4);
        assert_eq!(read[0].passed, 3);
    }

    #[test]
    fn utc_timestamp_format() {
        let ts = utc_timestamp();
        // Should match YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(ts.len(), 20);
        assert_eq!(&ts[4..5], "-");
        assert!(ts.ends_with('Z'));
    }
}
