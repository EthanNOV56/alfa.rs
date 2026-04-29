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

/// Pre-built computation DAG with common subexpression elimination and reference counting.
///
/// Built once per evaluate call from registered factors. Intermediate results are
/// dropped when no longer referenced, minimizing memory.
#[derive(Debug, Clone)]
pub struct ComputationPlan {
    /// Topologically ordered nodes (leaves first, roots last).
    pub nodes: Vec<PlanNode>,
    /// Factor name → index in `nodes` (root expression).
    pub factor_roots: HashMap<String, usize>,
}

/// One node in the computation DAG.
#[derive(Debug, Clone)]
pub struct PlanNode {
    pub expr: Expr,
    /// Unique hash for CSE deduplication.
    pub hash: u64,
    /// Number of remaining consumers. When drops to 0 after evaluation,
    /// this intermediate result can be freed.
    pub ref_count: usize,
}

impl ComputationPlan {
    /// Build a DAG from all registered factor expressions.
    pub fn build(factors: &HashMap<String, FactorInfo>) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_expr(expr: &Expr) -> u64 {
            let mut h = DefaultHasher::new();
            hash_impl(expr, &mut h);
            h.finish()
        }

        fn hash_impl(expr: &Expr, h: &mut DefaultHasher) {
            match expr {
                Expr::Column(name) => { 0u8.hash(h); name.hash(h); }
                Expr::Literal(lit) => {
                    1u8.hash(h);
                    match lit {
                        crate::expr::ast::Literal::Boolean(b) => { 0u8.hash(h); b.hash(h); }
                        crate::expr::ast::Literal::Integer(i) => { 1u8.hash(h); i.hash(h); }
                        crate::expr::ast::Literal::Float(f) => { 2u8.hash(h); f.to_bits().hash(h); }
                        crate::expr::ast::Literal::String(s) => { 3u8.hash(h); s.hash(h); }
                        crate::expr::ast::Literal::Null => { 4u8.hash(h); }
                    }
                }
                Expr::BinaryExpr { left, right, op } => {
                    2u8.hash(h); op.hash(h);
                    hash_impl(left, h); hash_impl(right, h);
                }
                Expr::UnaryExpr { op, expr: e } => {
                    3u8.hash(h); op.hash(h); hash_impl(e, h);
                }
                Expr::FunctionCall { name, args, freq } => {
                    4u8.hash(h); name.hash(h); freq.hash(h);
                    for a in args { hash_impl(a, h); }
                }
                _ => {}
            }
        }

        // Phase 1: collect unique subexpressions and count references (pre-order DFS)
        let mut unique: Vec<(Expr, u64)> = Vec::new();
        let mut seen: HashSet<u64> = HashSet::new();
        let mut ref_counts: HashMap<u64, usize> = HashMap::new();

        for info in factors.values() {
            collect(&info.parsed_expr, &mut unique, &mut seen, &mut ref_counts);
        }

        fn collect(
            expr: &Expr,
            unique: &mut Vec<(Expr, u64)>,
            seen: &mut HashSet<u64>,
            ref_counts: &mut HashMap<u64, usize>,
        ) -> u64 {
            let h = hash_expr(expr);
            if seen.insert(h) {
                unique.push((expr.clone(), h));
                // Recurse into children
                match expr {
                    Expr::BinaryExpr { left, right, .. } => {
                        collect(left, unique, seen, ref_counts);
                        collect(right, unique, seen, ref_counts);
                    }
                    Expr::UnaryExpr { expr: e, .. } => {
                        collect(e, unique, seen, ref_counts);
                    }
                    Expr::FunctionCall { args, .. } => {
                        for a in args { collect(a, unique, seen, ref_counts); }
                    }
                    _ => {}
                }
            }
            *ref_counts.entry(h).or_insert(0) += 1;
            h
        }

        // Phase 2: topological sort via Kahn's algorithm
        let hash_to_idx: HashMap<u64, usize> = unique.iter().enumerate()
            .map(|(i, (_, h))| (*h, i)).collect();
        let n = unique.len();
        let mut indegree = vec![0u32; n];
        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (i, (expr, _)) in unique.iter().enumerate() {
            let child_hashes = match expr {
                Expr::BinaryExpr { left, right, .. } => vec![hash_expr(left), hash_expr(right)],
                Expr::UnaryExpr { expr: e, .. } => vec![hash_expr(e)],
                Expr::FunctionCall { args, .. } => args.iter().map(|a| hash_expr(a)).collect(),
                _ => vec![],
            };
            for ch in child_hashes {
                if let Some(&ci) = hash_to_idx.get(&ch) {
                    edges[ci].push(i);
                    indegree[i] += 1;
                }
            }
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(i) = queue.pop() {
            order.push(i);
            for &c in &edges[i] {
                indegree[c] -= 1;
                if indegree[c] == 0 { queue.push(c); }
            }
        }

        let nodes: Vec<PlanNode> = order.iter().map(|&orig_idx| {
            let (ref expr, hash) = unique[orig_idx];
            let rc = *ref_counts.get(&hash).unwrap_or(&0);
            PlanNode { expr: expr.clone(), hash, ref_count: rc }
        }).collect();

        let mut factor_roots = HashMap::new();
        for (name, info) in factors.iter() {
            let h = hash_expr(&info.parsed_expr);
            if let Some(&orig_idx) = hash_to_idx.get(&h) {
                if let Some(sorted_pos) = order.iter().position(|&i| i == orig_idx) {
                    factor_roots.insert(name.clone(), sorted_pos);
                }
            }
        }

        ComputationPlan { nodes, factor_roots }
    }
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
    pub groups: Vec<(i64, i64)>,
    /// Symbol strings at each index position
    pub symbols: Vec<String>,
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
            "factor", "symbol", "date", "raw", "winsored", "zscored", "cap_neued", "qcut",
        ])
        .ok();
        #[cfg(not(debug_assertions))]
        wtr.write_record(&["factor", "symbol", "date", "cap_neued", "qcut"]).ok();
    }

    fn write_rows<W: std::io::Write>(&self, wtr: &mut csv::Writer<W>) -> csv::Result<usize> {
        for i in 0..self.groups.len() {
            let (date, sym_idx) = self.groups[i];
            let yr = date / 10000;
            let mo = (date % 10000) / 100;
            let dy = date % 100;
            let date_str = format!("{:04}-{:02}-{:02}", yr, mo, dy);
            let sym = self.symbols
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
                &self.factor_name, sym, &date_str,
                &self.cap_neued[i].to_string(), &q,
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
    pub fn build_factor_matrix(&self, prices: &crate::data::layer::PriceMatrix) -> ndarray::Array2<f64> {
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
