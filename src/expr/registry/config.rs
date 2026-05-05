//! Configuration and data structures for the factor registry

use crate::expr::ast::Expr;
use std::sync::Arc;

/// Number of years to process in parallel during calc().
///
/// Benchmarked on WCR 2010-2025 (16 cores, 32GB RAM, single CH node):
///   threads=1:  84.1s  calc=78.0s  peak=6GB
///   threads=3:  42.8s  calc=35.4s  peak=9.5GB
///   threads=4:  36.9s  calc=29.4s  peak=10.4GB
///   threads=5:  31.3s  calc=23.8s  peak=17.0GB  ← best
///   threads=6:  37.5s  calc=29.4s  peak=18.6GB  (CH contention)
///
/// 5 is the sweet spot: maximum throughput before ClickHouse server
/// contention dominates. Peak memory ~17GB with single-factor workload;
/// multi-factor workloads add ~45MB/year/factor of retained FactorSlice data.
pub const CALC_PARALLEL_YEARS: usize = 5;

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

/// Pre-built computation DAG with common subexpression elimination and reference counting.
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
    /// Groups must be sorted by (date, symbol) — guaranteed by `compute`.
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

/// Cross-sectional result for one factor × one year.
///
/// Self-contained: carries its own name and symbol list.
/// All vecs are parallel — same index → same (date, symbol) in `groups`.
#[derive(Debug, Clone)]
pub struct FactorSlice {
    pub factor_name: String,
    /// (date_int, symbol_idx) keys, sorted by date then symbol
    pub groups: Arc<Vec<(i64, i64)>>,
    /// Symbol strings at each index position
    pub symbols: Arc<Vec<String>>,
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

impl FactorSlice {
    /// Write to a CSV file. Overwrites if exists.
    pub fn write_csv<P: AsRef<std::path::Path>>(&self, path: P) -> csv::Result<()> {
        let mut wtr = csv::Writer::from_path(&path)?;
        Self::write_header(&mut wtr);
        self.write_rows(&mut wtr)?;
        wtr.flush().map_err(csv::Error::from)
    }

    /// Write to an existing CSV writer (for appending).
    pub fn write_to<W: std::io::Write>(&self, wtr: &mut csv::Writer<W>) -> csv::Result<usize> {
        self.write_rows(wtr)
    }

    /// Write the CSV header row to a writer.
    pub fn write_header<W: std::io::Write>(wtr: &mut csv::Writer<W>) {
        #[cfg(debug_assertions)]
        wtr.write_record(&[
            "factor",
            "symbol",
            "date",
            "raw",
            "winsored",
            "zscored",
            "cap_neued",
            "qcut",
        ])
        .ok();
        #[cfg(not(debug_assertions))]
        wtr.write_record(&["factor", "symbol", "date", "cap_neued", "qcut"])
            .ok();
    }

    fn write_rows<W: std::io::Write>(&self, wtr: &mut csv::Writer<W>) -> csv::Result<usize> {
        for i in 0..self.groups.len() {
            let (date, sym_idx) = self.groups[i];
            let yr = date / 10000;
            let mo = (date % 10000) / 100;
            let dy = date % 100;
            let date_str = format!("{:04}-{:02}-{:02}", yr, mo, dy);
            let sym = self
                .symbols
                .get(sym_idx as usize)
                .map(|s| s.as_str())
                .unwrap_or("");
            let q = self.qcut[i].map(|v| v.to_string()).unwrap_or_default();

            #[cfg(debug_assertions)]
            wtr.write_record(&[
                &self.factor_name,
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
                &self.factor_name,
                sym,
                &date_str,
                &self.cap_neued[i].to_string(),
                &q,
            ])?;
        }
        wtr.flush()?;
        Ok(self.groups.len())
    }
}

/// Multi-factor × multi-year panel data, ready for backtest or CSV output.
///
/// Returned by `FactorRegistry::calc()`.
pub struct FactorPanel {
    pub slices: Vec<FactorSlice>,
    pub factor_names: Vec<String>,
}

impl FactorPanel {
    /// Write all factor values (cap_neued + qcut) to CSV.
    pub fn to_csv<P: AsRef<std::path::Path>>(&self, path: P) -> csv::Result<()> {
        let mut wtr = csv::Writer::from_path(&path)?;
        FactorSlice::write_header(&mut wtr);
        for s in &self.slices {
            s.write_to(&mut wtr)?;
        }
        wtr.flush().map_err(csv::Error::from)
    }

    /// Build a (n_dates × n_symbols) factor matrix aligned to the given PriceMatrix.
    pub fn build_factor_matrix(
        &self,
        prices: &crate::data::layer::PriceMatrix,
    ) -> ndarray::Array2<f64> {
        prices.build_factor_matrix(&self.slices)
    }

    pub fn total_records(&self) -> usize {
        self.slices.iter().map(|s| s.groups.len()).sum()
    }
}

/// Column metadata
#[derive(Debug, Clone)]
pub struct ColumnMeta {
    pub name: String,
    pub data_type: String,
}
