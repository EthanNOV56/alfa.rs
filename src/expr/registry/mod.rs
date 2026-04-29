//! Factor Registry System
//!
//! A system for registering factor expressions, parsing them into AST,
//! generating computation plans, and executing them efficiently with
//! timeout protection to prevent system overload.

pub mod config;
pub mod functions;
pub mod parser;
pub mod timeseries;

pub use config::{
    ColumnMeta, ComputeConfig, FactorInfo, FactorPanel, FactorResult, FactorSlice,
};
pub use parser::parse_expression;

use crate::data::layer::DataLayer;
use crate::expr::ast::{Expr, Frequency};
use ndarray::Array1;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::time::Instant;

pub use functions::{
    collect_frequencies, collect_unique_subexpressions, eval_expr_memoized, eval_expr_vectorized,
    eval_function_memoized, eval_function_vectorized, eval_ts_function_memoized,
    eval_ts_function_vectorized, expr_hash, extract_columns,
};

/// Factor Registry
#[derive(Clone)]
pub struct FactorRegistry {
    factors: HashMap<String, FactorInfo>,
    config: ComputeConfig,
    available_columns: HashMap<String, ColumnMeta>,
    required_columns: HashSet<String>,
}

impl Default for FactorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FactorRegistry {
    pub fn new() -> Self {
        Self {
            factors: HashMap::new(),
            config: ComputeConfig::default(),
            available_columns: HashMap::new(),
            required_columns: HashSet::new(),
        }
    }

    pub fn with_config(config: ComputeConfig) -> Self {
        Self {
            factors: HashMap::new(),
            config,
            available_columns: HashMap::new(),
            required_columns: HashSet::new(),
        }
    }

    pub fn set_columns(&mut self, columns: Vec<String>) {
        for col in columns {
            self.available_columns.insert(
                col.clone(),
                ColumnMeta {
                    name: col,
                    data_type: "float64".to_string(),
                },
            );
        }
    }

    pub fn columns(&self) -> Vec<String> {
        self.available_columns.keys().cloned().collect()
    }

    pub fn register(&mut self, name: &str, expression: &str) -> Result<String, String> {
        let expr = parse_expression(expression)?;

        // Add required columns to the tracked set
        let used_cols = extract_columns(&expr);
        for col in &used_cols {
            self.required_columns.insert(col.clone());
        }

        let info = FactorInfo {
            name: name.to_string(),
            expression: expression.to_string(),
            parsed_expr: expr,
            description: None,
            category: None,
        };

        self.factors.insert(name.to_string(), info);
        Ok(name.to_string())
    }

    pub fn compute(
        &self,
        name: &str,
        data: &HashMap<String, Vec<f64>>,
    ) -> Result<FactorResult, String> {
        let info = self
            .factors
            .get(name)
            .ok_or_else(|| format!("Factor '{}' not found", name))?;

        let n_rows = data.values().next().map(|v| v.len()).unwrap_or(0);
        if n_rows == 0 {
            return Err("Empty data".to_string());
        }

        let start = Instant::now();

        // Convert Vec data to Array1
        let arr_data: HashMap<String, Array1<f64>> = data
            .iter()
            .map(|(k, v)| (k.clone(), Array1::from_vec(v.clone())))
            .collect();

        let mut cache: HashMap<u64, Array1<f64>> = HashMap::new();
        for (col_name, vals) in &arr_data {
            let mut hasher = DefaultHasher::new();
            0u8.hash(&mut hasher);
            col_name.hash(&mut hasher);
            cache.insert(hasher.finish(), vals.clone());
        }

        let result = functions::eval_expr_memoized(&info.parsed_expr, &arr_data, n_rows, &mut cache)?;
        let elapsed = start.elapsed().as_millis() as u64;

        Ok(FactorResult {
            name: name.to_string(),
            values: result.to_vec(),
            n_rows,
            n_cols: 1,
            compute_time_ms: elapsed,
            groups: None,
        })
    }

    pub fn compute_batch(
        &self,
        names: &[&str],
        data: &HashMap<String, Vec<f64>>,
        parallel: bool,
    ) -> Result<HashMap<String, FactorResult>, String> {
        if names.is_empty() {
            return Ok(HashMap::new());
        }

        let n_rows = data.values().next().map(|v| v.len()).unwrap_or(0);
        if n_rows == 0 {
            return Err("Empty data".to_string());
        }

        // Convert Vec data to Array1
        let arr_data: HashMap<String, Array1<f64>> = data
            .iter()
            .map(|(k, v)| (k.clone(), Array1::from_vec(v.clone())))
            .collect();

        let start = Instant::now();

        let mut unique_exprs: Vec<Expr> = Vec::new();
        let mut expr_hash_set: HashSet<u64> = HashSet::new();
        let mut factor_exprs_owned: Vec<Expr> = Vec::new();
        let mut skipped_names: Vec<String> = Vec::new();
        let mut factor_exprs: Vec<(String, &Expr)> = Vec::new();

        for name in names {
            let info = match self.factors.get(*name) {
                Some(info) => info,
                None => { continue; }
            };
            let expr = info.parsed_expr.clone();
            collect_unique_subexpressions(&expr, &mut unique_exprs, &mut expr_hash_set);
            factor_exprs_owned.push(expr);
        }

        let mut seen_names: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut idx = 0;
        for name in names {
            if skipped_names.contains(&name.to_string()) { continue; }
            if seen_names.contains(name) { continue; }
            if idx < factor_exprs_owned.len() {
                seen_names.insert(name);
                factor_exprs.push((name.to_string(), &factor_exprs_owned[idx]));
                idx += 1;
            }
        }

        let mut cache: HashMap<u64, Array1<f64>> = HashMap::new();

        for (name, vals) in &arr_data {
            let mut hasher = DefaultHasher::new();
            0u8.hash(&mut hasher);
            name.hash(&mut hasher);
            cache.insert(hasher.finish(), vals.clone());
        }

        // Step 3: Compute final factors using memoized evaluation
        // This will automatically cache and reuse intermediate results
        let mut results: HashMap<String, FactorResult> = HashMap::new();

        if parallel && factor_exprs.len() > 1 {
            // Parallel computation using rayon
            use rayon::prelude::*;

            // Collect errors separately using a mutex
            use std::sync::Mutex;
            let failed_names: Mutex<Vec<String>> = Mutex::new(Vec::new());

            let results_vec: Vec<(String, FactorResult)> = factor_exprs
                .par_iter()
                .filter_map(|(name, expr)| {
                    let mut thread_cache = cache.clone();
                    let arr_data_clone = arr_data.clone();
                    match eval_expr_memoized(expr, &arr_data_clone, n_rows, &mut thread_cache) {
                        Ok(result) => Some((
                            name.clone(),
                            FactorResult {
                                name: name.clone(),
                                values: result.to_vec(),
                                n_rows,
                                n_cols: 1,
                                compute_time_ms: 0,
                                groups: None,
                            },
                        )),
                        Err(e) => {
                            if failed_names.lock().unwrap().len() < 20 {
                                failed_names
                                    .lock()
                                    .unwrap()
                                    .push(format!("{}: {}", name, e));
                            }
                            None
                        }
                    }
                })
                .collect();

            let failed = failed_names.lock().unwrap();
            if !failed.is_empty() {
                eprintln!(
                    "Warning: {} factors failed to compute: {:?}",
                    failed.len(),
                    failed
                );
            }
            drop(failed);

            for (name, result) in results_vec {
                results.insert(name, result);
            }
        } else {
            // Sequential computation
            let n_total = factor_exprs.len();
            let mut failed_names: Vec<String> = Vec::new();
            for (name, expr) in factor_exprs.iter() {
                match eval_expr_memoized(expr, &arr_data, n_rows, &mut cache) {
                    Ok(result) => {
                        results.insert(
                            name.clone(),
                            FactorResult {
                                name: name.clone(),
                                values: result.to_vec(),
                                n_rows,
                                n_cols: 1,
                                compute_time_ms: 0,
                                groups: None,
                            },
                        );
                    }
                    Err(e) => {
                        if failed_names.len() < 20 {
                            failed_names.push(format!("{}: {}", name, e));
                        }
                    }
                }
            }
            if !failed_names.is_empty() {
                eprintln!(
                    "Warning: {} factors failed to compute: {:?}",
                    failed_names.len(),
                    failed_names
                );
            }
        }

        let elapsed = start.elapsed().as_millis() as u64;

        // Update compute time
        for result in results.values_mut() {
            result.compute_time_ms = elapsed;
        }

        Ok(results)
    }

    /// Batch compute multiple factors with vectorized (SIMD) evaluation
    ///
    /// This method uses ndarray::Array1 for SIMD-optimized operations.
    /// It automatically fetches data based on the frequency prefix in factor names.
    ///
    /// # Arguments
    /// * `data_layer` - DataLayer for fetching data
    /// * `parallel` - Whether to use parallel computation
    ///
    /// # Returns
    /// HashMap from factor name to (FactorResult, Frequency) tuple
    pub fn compute_batch_vectorized(
        &self,
        data_layer: &mut DataLayer,
        parallel: bool,
    ) -> Result<HashMap<String, (FactorResult, Frequency)>, String> {
        // Step 1: Get all registered factor names
        let factor_names: Vec<String> = self.factors.keys().cloned().collect();
        if factor_names.is_empty() {
            return Ok(HashMap::new());
        }

        // Step 2: Extract required fields from all expressions
        let mut all_fields: Vec<String> = Vec::new();
        let mut freq_per_factor: HashMap<String, Frequency> = HashMap::new();

        for name in &factor_names {
            let info = self
                .factors
                .get(name)
                .ok_or_else(|| format!("Factor '{}' not found", name))?;
            let expr = info.parsed_expr.clone();

            let columns = extract_columns(&expr);
            for col in &columns {
                if !all_fields.contains(col) {
                    all_fields.push(col.clone());
                }
            }

            // Determine the finest frequency needed for this factor
            let mut finest = Frequency::Daily;
            for col in &columns {
                if let Some(colon_pos) = col.find(':') {
                    let freq_str = &col[..colon_pos];
                    if let Some(freq) = Frequency::parse(freq_str) {
                        if freq.period_days() < finest.period_days() {
                            finest = freq;
                        }
                    }
                }
            }
            freq_per_factor.insert(name.clone(), finest);
        }

        // Collect all group-by frequencies from FunctionCall nodes in all expressions,
        // then add {finest_freq}:trading_date and {finest_freq}:symbol as grouping columns.
        // Grouping columns must match the data granularity (finest frequency), not the
        // output frequency, because the values being aggregated are at the finest level.
        {
            let mut all_group_freqs: Vec<Frequency> = Vec::new();
            let mut finest_col_freq = Frequency::Daily;
            for name in &factor_names {
                let info = self.factors.get(name).unwrap();
                let expr = info.parsed_expr.clone();
                collect_frequencies(&expr, &mut all_group_freqs);

                // Find finest column frequency
                let cols = extract_columns(&expr);
                for col in &cols {
                    if let Some(colon_pos) = col.find(':') {
                        let freq_str = &col[..colon_pos];
                        if let Some(freq) = Frequency::parse(freq_str) {
                            if freq.period_days() < finest_col_freq.period_days() {
                                finest_col_freq = freq;
                            }
                        }
                    }
                }
            }

            // Use finest column frequency for grouping columns (aligns with data row count)
            let prefix = finest_col_freq.as_str();
            for meta_col in &["trading_date", "symbol"] {
                let field = format!("{}:{}", prefix, meta_col);
                if !all_fields.contains(&field) {
                    all_fields.push(field);
                }
            }
        }

        // Step 3: Query data from DataLayer
        let data = data_layer
            .query(all_fields)
            .map_err(|e| format!("DataLayer query error: {:?}", e))?;

        let n_rows = data.values().next().map(|arr| arr.len()).unwrap_or(0);
        if n_rows == 0 {
            return Err("No data fetched".to_string());
        }

        // Step 4: Compute each factor
        let mut results: HashMap<String, (FactorResult, Frequency)> = HashMap::new();

        for name in &factor_names {
            let info = self
                .factors
                .get(name)
                .ok_or_else(|| format!("Factor '{}' not found", name))?;
            let expr = info.parsed_expr.clone();

            let freq = freq_per_factor
                .get(name)
                .cloned()
                .unwrap_or(Frequency::Daily);

            // Use the internal computation method
            let result = self.compute_single_factor(&expr, &data, n_rows, parallel)?;

            results.insert(
                name.clone(),
                (
                    FactorResult {
                        name: name.clone(),
                        values: result,
                        n_rows,
                        n_cols: 1,
                        compute_time_ms: 0,
                        groups: None,
                    },
                    freq,
                ),
            );
        }

        Ok(results)
    }

    /// Build cache from data columns for expression evaluation
    fn build_cache(_data: &HashMap<String, Array1<f64>>) -> HashMap<u64, Array1<f64>> {
        // Column lookups go through `data` directly (Expr::Column handler),
        // not through cache. Cache is only for intermediate expression results.
        HashMap::new()
    }

    /// Compute a single factor expression
    fn compute_single_factor(
        &self,
        expr: &Expr,
        data: &HashMap<String, Array1<f64>>,
        _n_rows: usize,
        _parallel: bool,
    ) -> Result<Vec<f64>, String> {
        use crate::expr::registry::functions::eval_expr_vectorized;

        let mut cache = Self::build_cache(data);
        let arr = eval_expr_vectorized(expr, data, &mut cache)?;
        Ok(arr.to_vec())
    }

    /// Compute factors using pre-loaded data (no DataLayer dependency)
    ///
    /// When `compact` is true, group-aggregation expressions return per-group results
    /// instead of expanding back to n_rows. FactorResult.groups will be populated with
    /// (date_int, symbol_int) group keys, sorted by date then symbol.
    ///
    /// Suitable for callers that manage their own data loading (e.g., server binaries).
    pub fn compute_batch_for_freq(
        &self,
        names: &[&str],
        data: &HashMap<String, Array1<f64>>,
        parallel: bool,
        compact: bool,
    ) -> Result<HashMap<String, FactorResult>, String> {
        let start = Instant::now();

        // Step 1: Build computation graph
        let mut unique_exprs: Vec<Expr> = Vec::new();
        let mut expr_hash_set: HashSet<u64> = HashSet::new();
        let mut factor_exprs_owned: Vec<Expr> = Vec::new();
        let mut factor_exprs: Vec<(String, &Expr)> = Vec::new();

        for name in names {
            let info = self
                .factors
                .get(*name)
                .ok_or_else(|| format!("Factor '{}' not found", name))?;

            let expr = info.parsed_expr.clone();

            collect_unique_subexpressions(&expr, &mut unique_exprs, &mut expr_hash_set);
            factor_exprs_owned.push(expr);
        }

        for (i, name) in names.iter().enumerate() {
            factor_exprs.push((name.to_string(), &factor_exprs_owned[i]));
        }

        // Step 2: Build cache
        let mut cache = Self::build_cache(data);

        // Step 3: Compute factors
        let mut results: HashMap<String, FactorResult> = HashMap::new();

        if compact {
            use crate::expr::registry::functions::eval_expr_compact;
            // Compact evaluation: returns per-group values + group keys
            // No parallel path needed — compact results are small (n_groups << n_rows)
            for (name, expr) in factor_exprs {
                let (arr, groups) = eval_expr_compact(expr, data, &mut cache)?;
                results.insert(
                    name.clone(),
                    FactorResult {
                        name,
                        values: arr.to_vec(),
                        n_rows: arr.len(),
                        n_cols: 1,
                        compute_time_ms: 0,
                        groups: if groups.is_empty() {
                            None
                        } else {
                            Some(groups)
                        },
                    },
                );
            }
        } else {
            let n_rows = data.values().next().map(|arr| arr.len()).unwrap_or(0);

            if parallel && factor_exprs.len() > 1 {
                let use_parallel = n_rows < 1_000_000;

                if use_parallel {
                    use rayon::prelude::*;

                    let results_vec: Vec<(String, FactorResult)> = factor_exprs
                        .par_iter()
                        .filter_map(|(name, expr)| {
                            let mut thread_cache = cache.clone();
                            match eval_expr_vectorized(expr, data, &mut thread_cache) {
                                Ok(arr) => Some((
                                    name.clone(),
                                    FactorResult {
                                        name: name.clone(),
                                        values: arr.to_vec(),
                                        n_rows,
                                        n_cols: 1,
                                        compute_time_ms: 0,
                                        groups: None,
                                    },
                                )),
                                Err(e) => {
                                    eprintln!("Error computing {}: {}", name, e);
                                    None
                                }
                            }
                        })
                        .collect();

                    for (name, result) in results_vec {
                        results.insert(name, result);
                    }
                } else {
                    for (name, expr) in factor_exprs {
                        let arr = eval_expr_vectorized(expr, data, &mut cache)?;
                        results.insert(
                            name.clone(),
                            FactorResult {
                                name,
                                values: arr.to_vec(),
                                n_rows,
                                n_cols: 1,
                                compute_time_ms: 0,
                                groups: None,
                            },
                        );
                    }
                }
            } else {
                for (name, expr) in factor_exprs {
                    let arr = eval_expr_vectorized(expr, data, &mut cache)?;
                    results.insert(
                        name.clone(),
                        FactorResult {
                            name,
                            values: arr.to_vec(),
                            n_rows,
                            n_cols: 1,
                            compute_time_ms: 0,
                            groups: None,
                        },
                    );
                }
            }
        }

        let elapsed = start.elapsed().as_millis() as u64;

        for result in results.values_mut() {
            result.compute_time_ms = elapsed;
        }

        Ok(results)
    }

    /// One-stop compute: factor evaluation + cross-sectional pipeline.
    ///
    /// Queries 5m data, computes factors (compact mode), builds free_float_cap map,
    /// and applies winsor → zscore → cap_neu → qcut per date.
    pub fn compute_cs_pipeline(
        &self,
        data_layer: &mut DataLayer,
    ) -> Result<HashMap<String, FactorSlice>, String> {
        use crate::expr::registry::timeseries::{cap_neu, qcut, winsor, zscore};
        use rayon::prelude::*;

        // Collect columns by frequency
        let mut cols_1d: Vec<String> = Vec::new();
        let mut cols_5m: Vec<String> = Vec::new();
        for info in self.factors.values() {
            for col in extract_columns(&info.parsed_expr) {
                if let Some(stripped) = col.strip_prefix("5m:") {
                    let c = stripped.to_string();
                    if !cols_5m.contains(&c) { cols_5m.push(c); }
                } else if let Some(stripped) = col.strip_prefix("1d:") {
                    let c = stripped.to_string();
                    if !cols_1d.contains(&c) { cols_1d.push(c); }
                }
            }
        }

        let has_5m = !cols_5m.is_empty();
        if has_5m {
            // ── 5m path (existing) ──
            let mut query_fields = vec!["5m:trading_date".to_string(), "5m:symbol".to_string()];
            for c in &cols_5m { query_fields.push(format!("5m:{}", c)); }
            let data = data_layer.query(query_fields)
                .map_err(|e| format!("DataLayer query error: {:?}", e))?;
            data_layer.clear_cache_keep_symbols();

            let factor_names: Vec<&str> = self.factors.keys().map(|s| s.as_str()).collect();
            let results = self.compute_batch_for_freq(&factor_names, &data, false, true)?;
            let mktcap_map = data_layer.build_free_float_cap_map()
                .map_err(|e| format!("build_free_float_cap_map: {:?}", e))?;
            let symbol_list = data_layer.get_symbols_5m().to_vec();

            self.build_slices(&results, &symbol_list, &mktcap_map)
        } else {
            // ── 1d path ──
            let mut query_fields = vec!["1d:trading_date".to_string(), "1d:symbol".to_string()];
            for c in &cols_1d { query_fields.push(format!("1d:{}", c)); }
            let data = data_layer.query(query_fields)
                .map_err(|e| format!("DataLayer query error: {:?}", e))?;
            data_layer.clear_cache_keep_symbols();

            let factor_names: Vec<&str> = self.factors.keys().map(|s| s.as_str()).collect();
            // compact=false: each row is already a daily observation (no group-by needed)
            let results = self.compute_batch_for_freq(&factor_names, &data, false, false)?;

            // Build per-date groups from 1d trading_date and symbol columns
            let dates = data.get("1d:trading_date")
                .ok_or("1d:trading_date missing")?;
            let symbols = data.get("1d:symbol")
                .ok_or("1d:symbol missing")?;
            let n = dates.len();
            let mut groups: Vec<(i64, i64)> = Vec::with_capacity(n);
            for i in 0..n {
                groups.push((dates[i] as i64, symbols[i] as i64));
            }
            let symbol_list = data_layer.get_symbols_5m().to_vec();
            let mktcap_map = data_layer.build_free_float_cap_map()
                .map_err(|e| format!("build_free_float_cap_map: {:?}", e))?;

            // Wrap results with groups
            let mut grouped_results: HashMap<String, FactorResult> = HashMap::new();
            for (name, fr) in results {
                let values = crate::expr::registry::FactorResult {
                    groups: Some(groups.clone()),
                    ..fr
                };
                grouped_results.insert(name, values);
            }
            self.build_slices(&grouped_results, &symbol_list, &mktcap_map)
        }
    }

    fn build_slices(
        &self,
        results: &HashMap<String, FactorResult>,
        symbol_list: &[String],
        mktcap_map: &HashMap<(i64, usize), f64>,
    ) -> Result<HashMap<String, FactorSlice>, String> {
        use crate::expr::registry::timeseries::{cap_neu, qcut, winsor, zscore};
        use rayon::prelude::*;
        let mut cs_results: HashMap<String, FactorSlice> = HashMap::new();

        for (name, _info) in &self.factors {
            let result = results
                .get(name)
                .ok_or_else(|| format!("Factor '{}' missing from results", name))?;
            let groups = result.groups.as_ref().ok_or("Group keys missing")?;
            let values = &result.values;

            let mut slice = FactorSlice {
                factor_name: name.clone(),
                groups: groups.clone(),
                symbols: symbol_list.to_vec(),
                #[cfg(debug_assertions)]
                raw: Vec::with_capacity(values.len()),
                #[cfg(debug_assertions)]
                winsored: Vec::with_capacity(values.len()),
                #[cfg(debug_assertions)]
                zscored: Vec::with_capacity(values.len()),
                cap_neued: Vec::with_capacity(values.len()),
                qcut: Vec::with_capacity(values.len()),
            };

            // Collect per-date ranges, then process in parallel
            let mut date_ranges: Vec<(i64, usize, usize)> = Vec::new();
            let mut i = 0;
            while i < groups.len() {
                let date_int = groups[i].0;
                let start = i;
                while i < groups.len() && groups[i].0 == date_int {
                    i += 1;
                }
                date_ranges.push((date_int, start, i));
            }

            // Process dates in parallel with rayon
            let date_results: Vec<(usize, Vec<f64>, Vec<Option<i32>>)> = date_ranges
                .par_iter()
                .map(|&(date_int, start, end)| {
                    let n = end - start;
                    let mut cap = Vec::with_capacity(n);
                    let mut qc = Vec::with_capacity(n);
                    if n < 2 {
                        for _ in 0..n { cap.push(f64::NAN); qc.push(None); }
                        return (start, cap, qc);
                    }
                    let vals = Array1::from_vec(values[start..end].to_vec());
                    let syms: Vec<usize> = groups[start..end].iter().map(|g| g.1 as usize).collect();
                    let mktcaps: Array1<f64> = syms.iter()
                        .map(|&s| mktcap_map.get(&(date_int, s)).copied().unwrap_or(f64::NAN))
                        .collect();
                    let ws = winsor(&vals, n);
                    let zs = zscore(&ws, n);
                    let cn = cap_neu(&zs, &mktcaps, n);
                    let qc_vals = qcut(&cn, 10);
                    cap.extend_from_slice(cn.as_slice().unwrap());
                    qc.extend(qc_vals);
                    (start, cap, qc)
                })
                .collect();

            // Merge parallel results in order
            for (_, cap, qc) in date_results {
                slice.cap_neued.extend(cap);
                slice.qcut.extend(qc);
            }
            // Debug columns (raw/winsored/zscored) are not parallelized
            #[cfg(debug_assertions)]
            {
                for &(_, start, end) in &date_ranges {
                    for k in start..end {
                        slice.raw.push(values[k]);
                        slice.winsored.push(f64::NAN);
                        slice.zscored.push(f64::NAN);
                    }
                }
            }

            cs_results.insert(name.clone(), slice);
        }
        Ok(cs_results)
    }

    /// Compute all registered factors across a year range.
    ///
    /// Processes years sequentially to avoid OOM from parallel 5m data loads.
    /// Results can be streamed to CSV via `AlfarsLab::run()`.
    pub fn calc(
        &self,
        dl: &DataLayer,
        start_year: i32,
        end_year: i32,
    ) -> Result<FactorPanel, String> {
        let base_filter = dl.pre_filter().to_string();
        let mut all_slices = Vec::new();

        for year in start_year..=end_year {
            let start = format!("{}-01-01", year);
            let end = format!("{}-01-01", year + 1);
            let pre_filter = format!("{}:{} {}", start, end, base_filter);

            let mut year_dl = DataLayer::new(dl.source().clone());
            year_dl.set_pre_filter(&pre_filter);

            let slices: Vec<FactorSlice> = self
                .compute_cs_pipeline(&mut year_dl)?
                .into_values()
                .collect();

            all_slices.extend(slices);
        }

        Ok(FactorPanel {
            factor_names: self.list(),
            slices: all_slices,
        })
    }

    /// Get factor information by name
    pub fn get(&self, name: &str) -> Option<&FactorInfo> {
        self.factors.get(name)
    }

    /// List all registered factors
    pub fn list(&self) -> Vec<String> {
        self.factors.keys().cloned().collect()
    }

    /// Unregister a factor by name
    pub fn unregister(&mut self, name: &str) -> bool {
        if let Some(info) = self.factors.remove(name) {
            // Remove columns used by this factor from required_columns
            let cols = extract_columns(&info.parsed_expr);
            for col in cols {
                self.required_columns.remove(&col);
            }
            true
        } else {
            false
        }
    }

    /// Clear all registered factors
    pub fn clear(&mut self) {
        self.factors.clear();
        self.required_columns.clear();
    }

    /// Get the configuration
    pub fn config(&self) -> &ComputeConfig {
        &self.config
    }

    /// Get all columns required by registered factors (deduplicated)
    pub fn required_columns(&self) -> &HashSet<String> {
        &self.required_columns
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ComputeConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== FactorRegistry Tests ====================

    #[test]
    fn test_registry_new() {
        let registry = FactorRegistry::new();
        assert!(registry.columns().is_empty());
    }

    #[test]
    fn test_registry_set_columns() {
        let mut registry = FactorRegistry::new();
        registry.set_columns(vec![
            "close".to_string(),
            "open".to_string(),
            "volume".to_string(),
        ]);

        let columns = registry.columns();
        assert_eq!(columns.len(), 3);
        assert!(columns.contains(&"close".to_string()));
        assert!(columns.contains(&"open".to_string()));
        assert!(columns.contains(&"volume".to_string()));
    }

    #[test]
    fn test_registry_register() {
        let mut registry = FactorRegistry::new();
        registry.set_columns(vec![
            "close".to_string(),
            "open".to_string(),
            "volume".to_string(),
        ]);

        let result = registry.register("alpha_001", "close + open");
        assert!(result.is_ok());

        let factor_id = result.unwrap();
        assert_eq!(factor_id, "alpha_001");
    }

    #[test]
    fn test_registry_register_with_missing_column() {
        let mut registry = FactorRegistry::new();
        registry.set_columns(vec!["close".to_string()]);

        // Registration now succeeds even if column is not in available_columns
        // The required column is tracked for later validation at compute time
        let result = registry.register("alpha_001", "close + volume");
        assert!(result.is_ok());

        // Verify the required column is tracked
        let required = registry.required_columns();
        assert!(required.contains("close"));
        assert!(required.contains("volume"));
    }

    #[test]
    fn test_registry_get() {
        let mut registry = FactorRegistry::new();
        registry.set_columns(vec!["close".to_string()]);
        registry.register("alpha_001", "close * 2").unwrap();

        let info = registry.get("alpha_001");
        assert!(info.is_some());
        assert_eq!(info.unwrap().name, "alpha_001");
    }

    #[test]
    fn test_registry_list() {
        let mut registry = FactorRegistry::new();
        registry.set_columns(vec!["close".to_string()]);
        registry.register("alpha_001", "close * 2").unwrap();
        registry.register("alpha_002", "close + 1").unwrap();

        let factors = registry.list();
        assert_eq!(factors.len(), 2);
        assert!(factors.iter().any(|f| f == "alpha_001"));
        assert!(factors.iter().any(|f| f == "alpha_002"));
    }

    #[test]
    fn test_registry_compute() {
        let mut registry = FactorRegistry::new();
        registry.set_columns(vec!["close".to_string()]);

        registry.register("alpha_001", "close * 2").unwrap();

        let mut data: HashMap<String, Vec<f64>> = HashMap::new();
        data.insert("close".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = registry.compute("alpha_001", &data);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.values, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_registry_compute_batch() {
        let mut registry = FactorRegistry::new();
        registry.set_columns(vec!["close".to_string(), "volume".to_string()]);

        registry.register("alpha_001", "close * 2").unwrap();
        registry.register("alpha_002", "close + volume").unwrap();

        let mut data: HashMap<String, Vec<f64>> = HashMap::new();
        data.insert("close".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        data.insert("volume".to_string(), vec![10.0, 20.0, 30.0, 40.0, 50.0]);

        let result = registry.compute_batch(&["alpha_001", "alpha_002"], &data, false);
        assert!(result.is_ok());

        let results = result.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(
            results.get("alpha_001").unwrap().values,
            vec![2.0, 4.0, 6.0, 8.0, 10.0]
        );
        assert_eq!(
            results.get("alpha_002").unwrap().values,
            vec![11.0, 22.0, 33.0, 44.0, 55.0]
        );
    }

    #[test]
    fn test_registry_compute_ts_function() {
        let mut registry = FactorRegistry::new();
        registry.set_columns(vec!["close".to_string()]);

        registry.register("alpha_001", "ts_mean(close, 3)").unwrap();

        let mut data: HashMap<String, Vec<f64>> = HashMap::new();
        data.insert("close".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = registry.compute("alpha_001", &data);
        assert!(result.is_ok());

        let result = result.unwrap();
        // ts_mean with window 3: [1, 1.5, 2, 3, 4]
        assert_eq!(result.values, vec![1.0, 1.5, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_registry_compute_nested_function() {
        let mut registry = FactorRegistry::new();
        registry.set_columns(vec!["close".to_string()]);

        registry
            .register("alpha_001", "rank(ts_mean(close, 3))")
            .unwrap();

        let mut data: HashMap<String, Vec<f64>> = HashMap::new();
        data.insert("close".to_string(), vec![1.0, 5.0, 3.0, 2.0, 4.0]);

        let result = registry.compute("alpha_001", &data);
        assert!(result.is_ok());

        let result = result.unwrap();
        // ts_mean(close, 3): [1, 3, 3, 10/3, 3]
        // rank of [1, 3, 3, 3.33, 3] = [0, 0.5, 0.5, 1, 0.5]
        assert!(!result.values.is_empty());
    }

    #[test]
    fn test_registry_compute_empty() {
        let registry = FactorRegistry::new();
        let data: HashMap<String, Vec<f64>> = HashMap::new();

        let result = registry.compute("nonexistent", &data);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_default() {
        let config = ComputeConfig::default();
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_workers, 2);
    }

    #[test]
    fn test_config_conservative() {
        let config = ComputeConfig::conservative();
        assert_eq!(config.timeout_secs, 15);
        assert_eq!(config.max_workers, 1);
    }

    #[test]
    fn test_config_high_performance() {
        let config = ComputeConfig::high_performance();
        assert_eq!(config.timeout_secs, 120);
        assert_eq!(config.max_workers, 8);
    }

    #[test]
    fn test_required_columns() {
        let mut registry = FactorRegistry::new();
        registry.set_columns(vec![
            "close".to_string(),
            "open".to_string(),
            "volume".to_string(),
            "high".to_string(),
            "low".to_string(),
        ]);

        // Register factors
        registry.register("alpha_001", "close + open").unwrap();
        registry
            .register("alpha_002", "rank(ts_mean(close, 20))")
            .unwrap();
        registry.register("alpha_003", "volume / close").unwrap();

        // Get required columns
        let cols = registry.required_columns();

        // Should have close, open, volume (deduplicated)
        assert!(cols.contains("close"));
        assert!(cols.contains("open"));
        assert!(cols.contains("volume"));
        // high and low not needed by any factor
        assert!(!cols.contains("high"));
        assert!(!cols.contains("low"));
    }

    #[test]
    fn test_required_columns_empty() {
        let registry = FactorRegistry::new();
        let cols = registry.required_columns();
        assert!(cols.is_empty());
    }

    // ==================== Time Series Function Tests ====================

    #[test]
    fn test_ts_mean() {
        let vals = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = timeseries::ts_mean(&vals, 3);
        assert_eq!(result.to_vec(), vec![1.0, 1.5, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ts_sum() {
        let vals = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = timeseries::ts_sum(&vals, 3);
        assert_eq!(result.to_vec(), vec![1.0, 3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_ts_max() {
        let vals = Array1::from_vec(vec![1.0, 5.0, 3.0, 2.0, 4.0]);
        let result = timeseries::ts_max(&vals, 3);
        assert_eq!(result.to_vec(), vec![1.0, 5.0, 5.0, 5.0, 4.0]);
    }

    #[test]
    fn test_ts_min() {
        let vals = Array1::from_vec(vec![3.0, 1.0, 5.0, 2.0, 4.0]);
        let result = timeseries::ts_min(&vals, 3);
        assert_eq!(result.to_vec(), vec![3.0, 1.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn test_rank() {
        let vals = Array1::from_vec(vec![3.0, 1.0, 2.0, 5.0, 4.0]);
        let result = timeseries::rank(&vals);
        assert_eq!(result.to_vec(), vec![0.4, 0.0, 0.2, 0.8, 0.6]);
    }

    #[test]
    fn test_delay() {
        let vals = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = timeseries::delay(&vals, 2);
        assert_eq!(result.to_vec(), vec![0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_scale() {
        let vals = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = timeseries::scale(&vals);
        let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
        assert!((mean - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sign() {
        let vals = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = timeseries::sign(&vals);
        assert_eq!(result.to_vec(), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_delta() {
        let vals = Array1::from_vec(vec![10.0, 12.0, 11.0, 15.0, 14.0]);
        let result = timeseries::ts_delta(&vals, 1);
        assert_eq!(result.to_vec(), vec![0.0, 2.0, -1.0, 4.0, -1.0]);
    }

    #[test]
    fn test_winsor() {
        let vals = Array1::from_vec(vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            10.0, 20.0, 30.0, 40.0, 50.0,
        ]);
        let result = timeseries::winsor(&vals, 5);
        assert_eq!(result.to_vec(), vals.to_vec());
    }

    #[test]
    fn test_winsor_with_outliers() {
        let vals = Array1::from_vec(vec![1.0, 2.0, 100.0, 4.0, 5.0]);
        let result = timeseries::winsor(&vals, 5);
        assert_eq!(result.to_vec(), vals.to_vec());
    }

    #[test]
    fn test_zscore() {
        let vals = Array1::from_vec(vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            2.0, 4.0, 6.0, 8.0, 10.0,
        ]);
        let result = timeseries::zscore(&vals, 5);

        let date0 = &result.as_slice().unwrap()[0..5];
        let mean0: f64 = date0.iter().sum::<f64>() / 5.0;
        let std0_sample = (date0.iter().map(|x| x.powi(2)).sum::<f64>() / 4.0).sqrt();
        assert!((mean0 - 0.0).abs() < 1e-10);
        assert!((std0_sample - 1.0).abs() < 0.01);

        let date1 = &result.as_slice().unwrap()[5..10];
        let mean1: f64 = date1.iter().sum::<f64>() / 5.0;
        let std1_sample = (date1.iter().map(|x| x.powi(2)).sum::<f64>() / 4.0).sqrt();
        assert!((mean1 - 0.0).abs() < 1e-10);
        assert!((std1_sample - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cap_neu() {
        let n = 12;
        let mut vals_vec = Vec::with_capacity(n);
        let mut mktcap_vec = Vec::with_capacity(n);
        for i in 0..n {
            vals_vec.push((i + 1) as f64 * 10.0);
            mktcap_vec.push((i + 1) as f64);
        }
        let vals = Array1::from_vec(vals_vec);
        let market_cap = Array1::from_vec(mktcap_vec);
        let result = timeseries::cap_neu(&vals, &market_cap, n);

        let mean: f64 = result.iter().sum::<f64>() / n as f64;
        assert!((mean - 0.0).abs() < 1e-9);
        assert!(result.iter().any(|x| x.abs() > 0.01));
    }
}
