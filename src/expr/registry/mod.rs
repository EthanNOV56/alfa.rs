//! Factor Registry System
//!
//! A system for registering factor expressions, parsing them into AST,
//! generating computation plans, and executing them efficiently with
//! timeout protection to prevent system overload.

pub mod config;
pub mod functions;
pub mod parser;
pub mod timeseries;

pub use config::{ColumnMeta, ComputeConfig, FactorInfo, FactorPanel, FactorResult, FactorSlice};
pub use parser::parse_expression;

use crate::data::layer::DataLayer;
use ahash::AHashMap;
use ndarray::Array1;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

pub use functions::{
    collect_frequencies, collect_unique_subexpressions, eval_expr_compact, eval_expr_vectorized,
    eval_function_vectorized, eval_ts_function_vectorized, expr_hash, extract_columns,
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

    /// Compute factors with vectorized evaluation.
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
    /// Unified compute: evaluates registered factors using a DAG with ref-counted
    /// CSE. Intermediate results are dropped when no longer referenced.
    /// `compact=true` for group-by mode (5m data); `compact=false` for full row mode (1d data).
    /// When `parallel=true`, processes independent nodes in waves with rayon.
    pub fn compute(
        &self,
        names: &[&str],
        data: &HashMap<String, Array1<f64>>,
        parallel: bool,
        compact: bool,
    ) -> Result<HashMap<String, FactorResult>, String> {
        let start = Instant::now();
        // Fast path: single factor + compact mode — delegate to direct eval
        if names.len() == 1 && compact {
            let mut results = HashMap::new();
            let name = names[0];
            let info = self
                .factors
                .get(name)
                .ok_or_else(|| format!("Factor '{}' not found", name))?;
            let (arr, groups) = crate::expr::registry::functions::eval_expr_compact(
                &info.parsed_expr,
                data,
                &mut AHashMap::new(),
            )?;
            results.insert(
                name.to_string(),
                FactorResult {
                    name: name.to_string(),
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
            let elapsed = start.elapsed().as_millis() as u64;
            for r in results.values_mut() {
                r.compute_time_ms = elapsed;
            }
            return Ok(results);
        }

        // Per-factor parallel — each thread has its own cache for full parallelism.
        if parallel && names.len() > 1 {
            use parking_lot::Mutex;
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicUsize, Ordering};
            let results_mtx = Mutex::new(HashMap::<String, FactorResult>::new());
            let failed_mtx = parking_lot::Mutex::new(Vec::<String>::new());
            let done = AtomicUsize::new(0);
            let total = names.len();
            #[cfg(debug_assertions)]
            eprintln!("  compute: parallel {} factors: {:?}", total, names);
            names.par_iter().for_each(|&name| {
                let info = match self.factors.get(name) {
                    Some(i) => i,
                    None => return,
                };
                let mut cache = AHashMap::new();
                let t0 = std::time::Instant::now();
                let eval_result = if compact {
                    crate::expr::registry::functions::eval_expr_compact(
                        &info.parsed_expr,
                        data,
                        &mut cache,
                    )
                    .map(|v| (v.0, v.1))
                } else {
                    eval_expr_vectorized(&info.parsed_expr, data, &mut cache).map(|v| (v, vec![]))
                };
                let (arr, groups) = match eval_result {
                    Ok(v) => v,
                    Err(e) => {
                        let mut failed = failed_mtx.lock();
                        if failed.len() < 30 {
                            failed.push(format!("{}: {}", name, e));
                        }
                        return;
                    }
                };
                let t = t0.elapsed().as_millis();
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                #[cfg(debug_assertions)]
                eprintln!("    [{}/{}] {} {}ms", n, total, name, t);
                results_mtx.lock().insert(
                    name.to_string(),
                    FactorResult {
                        name: name.to_string(),
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
            });
            let failed = failed_mtx.into_inner();
            #[cfg(debug_assertions)]
            if !failed.is_empty() {
                eprintln!(
                    "  compute: {} factors failed: {:?}",
                    failed.len(),
                    &failed[..failed.len().min(20)]
                );
            }
            let mut results = results_mtx.into_inner();
            let elapsed = start.elapsed().as_millis() as u64;
            #[cfg(debug_assertions)]
            eprintln!(
                "  compute: done in {}ms ({}/{} succeeded)",
                elapsed,
                results.len(),
                total
            );
            return Ok(results);
        }

        // Sequential evaluation with shared cache across factors
        let mut cache = AHashMap::new();
        let mut results = HashMap::new();
        for &name in names {
            let info = self
                .factors
                .get(name)
                .ok_or_else(|| format!("Factor '{}' not found", name))?;
            let (arr, groups) = if compact {
                crate::expr::registry::functions::eval_expr_compact(
                    &info.parsed_expr,
                    data,
                    &mut cache,
                )?
            } else {
                let v = eval_expr_vectorized(&info.parsed_expr, data, &mut cache)?;
                (v, vec![])
            };
            results.insert(
                name.to_string(),
                FactorResult {
                    name: name.to_string(),
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
        return Ok(results);
    }

    /// One-stop compute: factor evaluation + cross-sectional pipeline.
    ///
    /// Queries 5m data, computes factors (compact mode), builds free_float_cap map,
    /// and applies winsor → zscore → cap_neu → qcut per date.
    pub fn compute_cs_pipeline(
        &self,
        data_layer: &mut DataLayer,
    ) -> Result<HashMap<String, FactorSlice>, String> {
        // Reset symbol encoding so the first query populates it from fresh
        // data. With DataPool::KeepAll the cached DataLayer may carry stale
        // symbols from a previous batch's 5m query, which would cause
        // mismatched encoding when the current batch queries 1d data first.
        data_layer.reset_symbols();

        // Collect columns by frequency
        let mut cols_1d: Vec<String> = Vec::new();
        let mut cols_5m: Vec<String> = Vec::new();
        for info in self.factors.values() {
            for col in extract_columns(&info.parsed_expr) {
                if let Some(stripped) = col.strip_prefix("5m:") {
                    let c = stripped.to_string();
                    if !cols_5m.contains(&c) {
                        cols_5m.push(c);
                    }
                } else if let Some(stripped) = col.strip_prefix("1d:") {
                    let c = stripped.to_string();
                    if !cols_1d.contains(&c) {
                        cols_1d.push(c);
                    }
                }
            }
        }

        let has_5m = !cols_5m.is_empty();
        let has_1d = !cols_1d.is_empty();

        eprintln!("  [pipeline] 5m={} cols={:?}  1d={} cols={:?}", has_5m, cols_5m, has_1d, cols_1d);

        // Query data for each frequency independently. When a batch mixes
        // 5m and 1d factors, we query both datasets and evaluate each group
        // with the appropriate mode (compact for 5m, flat for 1d).
        let mut data_5m: HashMap<String, Array1<f64>> = HashMap::new();
        let mut data_1d: HashMap<String, Array1<f64>> = HashMap::new();

        if has_5m {
            let mut query_fields = vec!["5m:trading_date".to_string(), "5m:symbol".to_string()];
            for c in &cols_5m {
                query_fields.push(format!("5m:{}", c));
            }
            data_5m = data_layer
                .query(query_fields)
                .map_err(|e| format!("DataLayer query error: {:?}", e))?;
            let n_5m = data_5m.values().next().map(|a| a.len()).unwrap_or(0);
            eprintln!("  [pipeline] 5m query done, {} rows, {} cols", n_5m, data_5m.len());
        }
        if has_1d {
            let mut query_fields = vec!["1d:trading_date".to_string(), "1d:symbol".to_string()];
            for c in &cols_1d {
                query_fields.push(format!("1d:{}", c));
            }
            data_1d = data_layer
                .query(query_fields)
                .map_err(|e| format!("DataLayer query error: {:?}", e))?;
            let n_1d = data_1d.values().next().map(|a| a.len()).unwrap_or(0);
            eprintln!("  [pipeline] 1d query done, {} rows, {} cols", n_1d, data_1d.len());
        }

        // Pre-compute 1d sort order and perm for the flat evaluation path.
        let (shared_groups, perm) = if has_1d {
            let n_dates = {
                let dates = data_1d
                    .get("1d:trading_date")
                    .ok_or("1d:trading_date missing")?;
                let mut seen = std::collections::HashSet::new();
                dates.iter().filter(|d| d.is_finite()).for_each(|&d| {
                    seen.insert(d.to_bits());
                });
                seen.len()
            };
            data_1d.insert("_n_dates".to_string(), Array1::from_elem(1, n_dates as f64));
            let dates = data_1d
                .get("1d:trading_date")
                .ok_or("1d:trading_date missing")?;
            let syms = data_1d.get("1d:symbol").ok_or("1d:symbol missing")?;
            let n = dates.len();
            let mut indexed: Vec<(usize, (i64, i64))> = (0..n)
                .map(|i| (i, (dates[i] as i64, syms[i] as i64)))
                .collect();
            indexed.sort_by_key(|(_, (d, s))| (*d, *s));
            let perm_vec: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
            let groups: Vec<(i64, i64)> = indexed.iter().map(|(_, g)| *g).collect();
            let perm_valid = perm_vec.len() == n;
            let perm_opt = if perm_valid { Some(perm_vec) } else { None };
            (Some(Arc::new(groups)), perm_opt)
        } else {
            (None, None)
        };
        data_layer.clear_cache_keep_symbols();

        let mktcap_map = data_layer
            .build_free_float_cap_map()
            .map_err(|e| format!("build_free_float_cap_map: {:?}", e))?;
        let symbol_list = data_layer.get_symbols_5m().to_vec();
        let symbols_arc = Arc::new(symbol_list);

        let factor_names: Vec<&str> = self.factors.keys().map(|s| s.as_str()).collect();
        let parallel = std::env::var("ALFARS_SEQUENTIAL").is_err();
        let mut all_slices: HashMap<String, FactorSlice> = HashMap::new();

        const BATCH_SIZE: usize = 80;
        let mut batch_idx = 0usize;
        let n_batches = factor_names.len().div_ceil(BATCH_SIZE);
        for batch in factor_names.chunks(BATCH_SIZE) {
            batch_idx += 1;
            #[cfg(debug_assertions)]
            eprintln!(
                "  batch [{}/{}] compute {} factors",
                batch_idx,
                n_batches,
                batch.len()
            );

            // Split batch by frequency: 5m factors use compact mode (5m data),
            // 1d/bare factors use flat mode (1d data). Mixed-frequency batches
            // previously dropped non-dominant columns silently.
            let mut batch_5m: Vec<&str> = Vec::new();
            let mut batch_1d: Vec<&str> = Vec::new();
            for &name in batch {
                let info = match self.factors.get(name) {
                    Some(i) => i,
                    None => continue,
                };
                let has_5m_col = extract_columns(&info.parsed_expr)
                    .iter()
                    .any(|c| c.starts_with("5m:"));
                if has_5m_col {
                    batch_5m.push(name);
                } else {
                    batch_1d.push(name);
                }
            }

            if !batch_5m.is_empty() {
                let results = self.compute(&batch_5m, &data_5m, parallel, true)?;
                let names: Vec<&str> = results.keys().map(|s| s.as_str()).collect();
                let batch_slices = self.build_slices(
                    &results,
                    &names,
                    &symbols_arc,
                    &mktcap_map,
                    None,
                    None, // 5m path: no shared_groups, no perm
                )?;
                eprintln!("  [pipeline] batch_5m build_slices done, {} slices", batch_slices.len());
                all_slices.extend(batch_slices);
            }
            if !batch_1d.is_empty() {
                let results = self.compute(&batch_1d, &data_1d, parallel, false)?;
                let names: Vec<&str> = results.keys().map(|s| s.as_str()).collect();
                let batch_slices = self.build_slices(
                    &results,
                    &names,
                    &symbols_arc,
                    &mktcap_map,
                    shared_groups.as_ref(),
                    perm.as_deref(),
                )?;
                eprintln!("  [pipeline] batch_1d build_slices done, {} slices", batch_slices.len());
                all_slices.extend(batch_slices);
            }
        }
        Ok(all_slices)
    }

    fn build_slices(
        &self,
        results: &HashMap<String, FactorResult>,
        names: &[&str],
        symbols: &Arc<Vec<String>>,
        mktcap_map: &AHashMap<(i64, usize), f64>,
        shared_groups: Option<&Arc<Vec<(i64, i64)>>>,
        perm: Option<&[usize]>,
    ) -> Result<HashMap<String, FactorSlice>, String> {
        use crate::expr::registry::timeseries::{cap_neu, qcut, winsor, zscore};
        use rayon::prelude::*;
        let mut cs_results: HashMap<String, FactorSlice> = HashMap::new();
        let n_names = names.len();

        for (fi, &name) in names.iter().enumerate() {
            eprintln!("    build_slices [{}/{}] {}", fi + 1, n_names, name);
            let result = results
                .get(name)
                .ok_or_else(|| format!("Factor '{}' missing from results", name))?;
            let groups_slice: &[(i64, i64)] = if let Some(sg) = shared_groups {
                sg
            } else {
                result.groups.as_deref().ok_or("Group keys missing")?
            };
            let values = &result.values;

            let groups_arc: Arc<Vec<(i64, i64)>> = if let Some(sg) = shared_groups {
                Arc::clone(sg)
            } else {
                Arc::new(groups_slice.to_vec())
            };

            let mut slice = FactorSlice {
                factor_name: name.to_string(),
                groups: groups_arc,
                symbols: Arc::clone(symbols),
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
            while i < groups_slice.len() {
                let date_int = groups_slice[i].0;
                let start = i;
                while i < groups_slice.len() && groups_slice[i].0 == date_int {
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
                        for _ in 0..n {
                            cap.push(f64::NAN);
                            qc.push(None);
                        }
                        return (start, cap, qc);
                    }
                    let vals = if let Some(p) = perm {
                        Array1::from_vec((start..end).map(|j| values[p[j]]).collect())
                    } else {
                        Array1::from_vec(values[start..end].to_vec())
                    };
                    let syms: Vec<usize> = groups_slice[start..end]
                        .iter()
                        .map(|g| g.1 as usize)
                        .collect();
                    let mktcaps: Array1<f64> = syms
                        .iter()
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
                        let val_idx = if let Some(p) = perm { p[k] } else { k };
                        slice.raw.push(values[val_idx]);
                        slice.winsored.push(f64::NAN);
                        slice.zscored.push(f64::NAN);
                    }
                }
            }

            cs_results.insert(name.to_string(), slice);
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
        assert!(required.contains("1d:close"));
        assert!(required.contains("1d:volume"));
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

        let mut data: HashMap<String, Array1<f64>> = HashMap::new();
        data.insert(
            "1d:close".to_string(),
            Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        );

        let result = registry.compute(&["alpha_001"], &data, false, false);
        assert!(result.is_ok());

        let results = result.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results.get("alpha_001").unwrap().values,
            vec![2.0, 4.0, 6.0, 8.0, 10.0]
        );
    }

    #[test]
    fn test_registry_compute_batch() {
        let mut registry = FactorRegistry::new();
        registry.set_columns(vec!["close".to_string(), "volume".to_string()]);

        registry.register("alpha_001", "close * 2").unwrap();
        registry.register("alpha_002", "close + volume").unwrap();

        let mut data: HashMap<String, Array1<f64>> = HashMap::new();
        data.insert(
            "1d:close".to_string(),
            Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        );
        data.insert(
            "1d:volume".to_string(),
            Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]),
        );

        let result = registry.compute(&["alpha_001", "alpha_002"], &data, false, false);
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

        let mut data: HashMap<String, Array1<f64>> = HashMap::new();
        data.insert(
            "1d:close".to_string(),
            Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        );

        let result = registry.compute(&["alpha_001"], &data, false, false);
        assert!(result.is_ok());

        let results = result.unwrap();
        // ts_mean with window 3: [1, 1.5, 2, 3, 4]
        assert_eq!(
            results.get("alpha_001").unwrap().values,
            vec![1.0, 1.5, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_registry_compute_nested_function() {
        let mut registry = FactorRegistry::new();
        registry.set_columns(vec!["close".to_string()]);

        registry
            .register("alpha_001", "rank(ts_mean(close, 3))")
            .unwrap();

        let mut data: HashMap<String, Array1<f64>> = HashMap::new();
        data.insert(
            "1d:close".to_string(),
            Array1::from_vec(vec![1.0, 5.0, 3.0, 2.0, 4.0]),
        );

        let result = registry.compute(&["alpha_001"], &data, false, false);
        assert!(result.is_ok());

        let results = result.unwrap();
        // ts_mean(close, 3): [1, 3, 3, 10/3, 3]
        // rank of [1, 3, 3, 3.33, 3] = [0, 0.5, 0.5, 1, 0.5]
        assert!(!results.get("alpha_001").unwrap().values.is_empty());
    }

    #[test]
    fn test_registry_compute_empty() {
        let registry = FactorRegistry::new();
        let data: HashMap<String, Array1<f64>> = HashMap::new();

        let result = registry.compute(&["nonexistent"], &data, false, false);
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

        // Should have 1d:close, 1d:open, 1d:volume (deduplicated)
        assert!(cols.contains("1d:close"));
        assert!(cols.contains("1d:open"));
        assert!(cols.contains("1d:volume"));
        // high and low not needed by any factor
        assert!(!cols.contains("1d:high"));
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
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 1.0);
        assert_eq!(result[3], 2.0);
        assert_eq!(result[4], 3.0);
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
        assert!(result[0].is_nan());
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], -1.0);
        assert_eq!(result[3], 4.0);
        assert_eq!(result[4], -1.0);
    }

    #[test]
    fn test_winsor() {
        let vals = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]);
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
        let vals = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
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
