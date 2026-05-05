//! Factor pool for maintaining diverse alpha factors with redundancy filtering
//! and decay detection.

use crate::expr::Expr;
use crate::gp::operators::check_redundancy;
use crate::gp::types::{AdmissionResult, PoolEntry, to_parseable_string};

/// A maintained pool of diverse alpha factors with redundancy filtering and decay detection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FactorPool {
    /// Entries sorted by RankIC descending
    entries: Vec<PoolEntry>,
    /// Maximum pool capacity
    max_size: usize,
    /// Structural similarity threshold for outright rejection (default 0.95)
    reject_threshold: f64,
    /// Structural similarity threshold for flagging (default 0.80)
    flag_threshold: f64,
    /// Minimum correlations (absolute) for redundancy filtering
    correlation_threshold: f64,
}

impl FactorPool {
    /// Create a new factor pool with given max capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_size,
            reject_threshold: 0.95,
            flag_threshold: 0.80,
            correlation_threshold: 0.7,
        }
    }

    /// Number of entries currently in the pool.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get a reference to all pool entries.
    pub fn entries(&self) -> &[PoolEntry] {
        &self.entries
    }

    /// Attempt to admit a factor based on IC and structural similarity
    /// against the pool's stored expressions.
    pub fn try_admit_parsed(
        &mut self,
        expr: &Expr,
        ic: f64,
        rank_ic: f64,
        pool_expressions: &[Expr],
        now: u64,
    ) -> AdmissionResult {
        // 1. Structural redundancy check
        if let Some(max_sim) = check_redundancy(expr, pool_expressions) {
            if max_sim >= self.reject_threshold {
                return AdmissionResult::RejectedDuplicate(max_sim);
            }
            if max_sim >= self.flag_threshold {
                // Accept but flag
                self.insert_entry(expr, ic, rank_ic, now);
                return AdmissionResult::Flagged(max_sim);
            }
        }

        // 2. Pool capacity check — if full, remove worst if new is better
        if self.entries.len() >= self.max_size {
            let min_rank_ic = self
                .entries
                .last()
                .map(|e| e.rank_ic)
                .unwrap_or(f64::NEG_INFINITY);
            if rank_ic <= min_rank_ic {
                return AdmissionResult::RejectedBelowMinimum;
            }
            self.entries.pop(); // remove worst
        }

        self.insert_entry(expr, ic, rank_ic, now);
        AdmissionResult::Added
    }

    fn insert_entry(&mut self, expr: &Expr, ic: f64, rank_ic: f64, now: u64) {
        let entry = PoolEntry {
            expression: to_parseable_string(expr),
            ic,
            rank_ic,
            added_at: now,
            last_check_at: now,
            survival_rounds: 0,
        };
        // Insert in descending RankIC order
        let pos = self
            .entries
            .binary_search_by(|e| e.rank_ic.partial_cmp(&rank_ic).unwrap().reverse())
            .unwrap_or_else(|i| i);
        self.entries.insert(pos, entry);
    }

    /// Prune the pool to max_size, keeping the best by RankIC.
    pub fn prune(&mut self) {
        self.entries.truncate(self.max_size);
    }

    /// Mark surviving factors and remove decayed ones.
    /// `decayed_indices` contains indices of factors that failed re-validation.
    pub fn remove_decayed(&mut self, decayed_indices: &[usize]) {
        // Remove in reverse order to preserve indices
        let mut sorted: Vec<usize> = decayed_indices.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        for idx in sorted {
            if idx < self.entries.len() {
                self.entries.remove(idx);
            }
        }
    }

    /// Bump survival rounds for all entries (called after a validation pass).
    pub fn bump_survival(&mut self) {
        for entry in &mut self.entries {
            entry.survival_rounds += 1;
        }
    }
}
