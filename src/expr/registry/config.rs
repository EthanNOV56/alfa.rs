//! Configuration and data structures for the factor registry

use crate::expr::ast::{Expr, Frequency};
use std::collections::{HashMap, HashSet};

/// Configuration for resource limits and timeout to prevent system overload
#[derive(Debug, Clone)]
pub struct ComputeConfig {
    /// Maximum computation time in seconds (timeout)
    pub timeout_secs: u64,
    /// Maximum number of parallel threads
    pub max_workers: usize,
    /// Maximum batch size for chunked processing
    pub batch_size: usize,
    /// Maximum memory usage estimate in MB
    pub memory_limit_mb: usize,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            max_workers: 2,
            batch_size: 5000,
            memory_limit_mb: 512,
        }
    }
}

impl ComputeConfig {
    pub fn conservative() -> Self {
        Self {
            timeout_secs: 15,
            max_workers: 1,
            batch_size: 2000,
            memory_limit_mb: 256,
        }
    }

    pub fn high_performance() -> Self {
        Self {
            timeout_secs: 120,
            max_workers: 8,
            batch_size: 50000,
            memory_limit_mb: 4096,
        }
    }
}

/// Factor information
#[derive(Debug, Clone)]
pub struct FactorInfo {
    pub name: String,
    pub expression: String,
    pub parsed_expr: Expr,
    pub description: Option<String>,
    pub category: Option<String>,
}

/// Pre-built computation DAG with explicit common subexpression elimination.
///
/// Built once from all registered factors. Reusable across data batches,
/// different date ranges, and multiple evaluation calls.
#[derive(Debug, Clone)]
pub struct ComputationPlan {
    /// Topologically ordered expressions (leaves first: columns, then literals,
    /// then composite expressions, then factor roots).
    pub exprs: Vec<Expr>,
    /// Expression hash → index in `exprs` (for CSE dedup).
    pub hash_to_idx: HashMap<u64, usize>,
    /// Factor name → index in `exprs` (root expression).
    pub factor_roots: HashMap<String, usize>,
    /// All column names required by this plan (for data query construction).
    pub required_columns: HashSet<String>,
}

/// What data to fetch: frequency + column list.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub frequency: Frequency,
    /// Full prefixed column names (e.g., "1d:close", "1d:volume").
    pub columns: Vec<String>,
    /// Grouping columns: trading date + symbol.
    pub trading_date_col: String,
    pub symbol_col: String,
}

/// Factor computation result
#[derive(Debug, Clone)]
pub struct FactorResult {
    pub name: String,
    pub values: Vec<f64>,
    pub n_rows: usize,
    pub n_cols: usize,
    pub compute_time_ms: u64,
    /// Per-group (date_int, symbol_int) keys when using compact group evaluation
    pub groups: Option<Vec<(i64, i64)>>,
}

impl FactorResult {
    /// Return per-date (date_int, start_idx, end_idx) ranges for compact results.
    /// Groups must be sorted by (date, symbol) — guaranteed by `compute_batch_for_freq`.
    pub fn date_ranges(&self) -> Option<Vec<(i64, usize, usize)>> {
        let groups = self.groups.as_ref()?;
        let mut ranges = Vec::new();
        let mut i = 0;
        while i < groups.len() {
            let date = groups[i].0;
            let start = i;
            while i < groups.len() && groups[i].0 == date {
                i += 1;
            }
            ranges.push((date, start, i));
        }
        Some(ranges)
    }
}

/// Cross-sectional pipeline result for one factor.
/// All vecs are parallel — same index maps to the same (date, symbol) in `groups`.
///
/// Intermediate stages (`raw`, `winsored`, `zscored`, `cap_neued`) are only
/// available in debug builds.
#[derive(Debug, Clone)]
pub struct CsResult {
    /// (date_int, symbol_idx) keys, sorted by date then symbol
    pub groups: Vec<(i64, i64)>,
    #[cfg(debug_assertions)]
    pub raw: Vec<f64>,
    #[cfg(debug_assertions)]
    pub winsored: Vec<f64>,
    #[cfg(debug_assertions)]
    pub zscored: Vec<f64>,
    pub cap_neued: Vec<f64>,
    /// None for symbols with NaN factor values
    pub qcut: Vec<Option<i32>>,
}

impl CsResult {
    /// Write to a CSV file. Overwrites if exists.
    /// In debug builds, includes all intermediate stages; in release, only `qcut`.
    pub fn write_csv<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        symbol_list: &[String],
    ) -> csv::Result<usize> {
        let mut wtr = csv::Writer::from_path(&path)?;
        Self::write_header(&mut wtr);
        self.write_rows(&mut wtr, symbol_list)
    }

    /// Write to an existing CSV writer (for appending/streaming).
    /// Does not write a header — call `write_header` first if needed.
    pub fn write_to<W: std::io::Write>(
        &self,
        wtr: &mut csv::Writer<W>,
        symbol_list: &[String],
    ) -> csv::Result<usize> {
        self.write_rows(wtr, symbol_list)
    }

    /// Write the CSV header row to a writer.
    pub fn write_header<W: std::io::Write>(wtr: &mut csv::Writer<W>) {
        #[cfg(debug_assertions)]
        wtr.write_record(&[
            "symbol", "trading_date", "raw", "winsored", "zscored", "cap_neued", "qcut",
        ])
        .ok();
        #[cfg(not(debug_assertions))]
        wtr.write_record(&["symbol", "trading_date", "cap_neued", "qcut"]).ok();
    }

    fn write_rows<W: std::io::Write>(
        &self,
        wtr: &mut csv::Writer<W>,
        symbol_list: &[String],
    ) -> csv::Result<usize> {
        for i in 0..self.groups.len() {
            let (date, sym_idx) = self.groups[i];
            let yr = date / 10000;
            let mo = (date % 10000) / 100;
            let dy = date % 100;
            let date_str = format!("{:04}-{:02}-{:02}", yr, mo, dy);
            let sym = symbol_list
                .get(sym_idx as usize)
                .map(|s| s.as_str())
                .unwrap_or("");
            let q = self.qcut[i].map(|v| v.to_string()).unwrap_or_default();

            #[cfg(debug_assertions)]
            wtr.write_record(&[
                sym,
                &date_str,
                &self.raw[i].to_string(),
                &self.winsored[i].to_string(),
                &self.zscored[i].to_string(),
                &self.cap_neued[i].to_string(),
                &q,
            ])?;

            #[cfg(not(debug_assertions))]
            wtr.write_record(&[
                sym, &date_str, &self.cap_neued[i].to_string(), &q,
            ])?;
        }
        wtr.flush()?;
        Ok(self.groups.len())
    }
}

/// Column metadata
#[derive(Debug, Clone)]
pub struct ColumnMeta {
    pub name: String,
    pub data_type: String,
}
