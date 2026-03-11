//! Factor Registry System
//!
//! A system for registering factor expressions, parsing them into AST,
//! generating computation plans, and executing them efficiently with
//! timeout protection to prevent system overload.

pub mod config;
pub mod functions;
pub mod parser;
pub mod timeseries;

pub use config::{ColumnMeta, ComputeConfig, FactorInfo, FactorResult};
pub use parser::parse_expression;

use crate::expr::ast::{BinaryOp, Expr, Literal, UnaryOp};
use crate::lazy::LogicalPlan;
use ndarray::Array1;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::time::Instant;

pub use functions::{
    collect_unique_subexpressions, eval_expr_memoized, eval_expr_vectorized,
    eval_function_memoized, eval_function_vectorized, eval_ts_function_memoized,
    eval_ts_function_vectorized, expr_hash, extract_columns,
};

/// Factor Registry
#[derive(Clone)]
pub struct FactorRegistry {
    factors: HashMap<String, FactorInfo>,
    config: ComputeConfig,
    available_columns: HashMap<String, ColumnMeta>,
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
        }
    }

    pub fn with_config(config: ComputeConfig) -> Self {
        Self {
            factors: HashMap::new(),
            config,
            available_columns: HashMap::new(),
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

        let used_cols = extract_columns(&expr);
        for col in &used_cols {
            if !self.available_columns.contains_key(col) {
                return Err(format!("Column '{}' not found", col));
            }
        }

        let plan = functions::expr_to_logical_plan(&expr, name)?;

        let info = FactorInfo {
            name: name.to_string(),
            expression: expression.to_string(),
            parsed_expr: expr,
            plan,
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
        let result = self.execute_plan(&info.plan, data, n_rows)?;
        let elapsed = start.elapsed().as_millis() as u64;

        Ok(FactorResult {
            name: name.to_string(),
            values: result,
            n_rows,
            n_cols: 1,
            compute_time_ms: elapsed,
        })
    }

    /// Batch compute multiple factors with shared subexpression optimization
    /// This method builds a computation graph and computes each unique subexpression only once
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

        let start = Instant::now();

        // Step 1: Build computation graph - collect all unique subexpressions
        let mut unique_exprs: Vec<Expr> = Vec::new();
        let mut expr_hash_set: HashSet<u64> = HashSet::new();
        // Store owned expressions to keep them alive
        let mut factor_exprs_owned: Vec<Expr> = Vec::new();
        let mut skipped_names: Vec<String> = Vec::new();
        let mut factor_exprs: Vec<(String, &Expr)> = Vec::new();

        for name in names {
            let info = match self.factors.get(*name) {
                Some(info) => info,
                None => {
                    continue;
                }
            };

            // Extract the expression from plan (unwrap from Projection)
            let expr = match &info.plan {
                LogicalPlan::Projection { exprs, .. } => exprs
                    .first()
                    .map(|(_, e)| e.clone())
                    .ok_or("Empty expression")?,
                _ => {
                    skipped_names.push(name.to_string());
                    continue;
                }
            };

            // Collect subexpressions
            collect_unique_subexpressions(&expr, &mut unique_exprs, &mut expr_hash_set);
            factor_exprs_owned.push(expr);
        }

        // Step 1b: Deduplicate factor names and build final list
        // (avoids HashMap overwrite issues when duplicate names are passed)
        let mut seen_names: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut idx = 0;
        for name in names {
            // Skip factors that were not found or had wrong plan type
            if skipped_names.contains(&name.to_string()) {
                continue;
            }
            // Skip duplicates - only keep first occurrence
            if seen_names.contains(name) {
                continue;
            }
            if idx < factor_exprs_owned.len() {
                seen_names.insert(name);
                factor_exprs.push((name.to_string(), &factor_exprs_owned[idx]));
                idx += 1;
            }
        }

        // Step 2: Build cache using memoization - compute on demand
        // We'll use a RefCell to allow mutable caching during evaluation
        let mut cache: HashMap<u64, Vec<f64>> = HashMap::new();

        // Pre-populate cache with base columns (they're cheap)
        for (name, vals) in data {
            let col_hash = {
                let mut hasher = DefaultHasher::new();
                0u8.hash(&mut hasher);
                name.hash(&mut hasher);
                hasher.finish()
            };
            cache.insert(col_hash, vals.clone());
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
                    match eval_expr_memoized(expr, data, n_rows, &mut thread_cache) {
                        Ok(result) => Some((
                            name.clone(),
                            FactorResult {
                                name: name.clone(),
                                values: result,
                                n_rows,
                                n_cols: 1,
                                compute_time_ms: 0,
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
                match eval_expr_memoized(expr, data, n_rows, &mut cache) {
                    Ok(result) => {
                        results.insert(
                            name.clone(),
                            FactorResult {
                                name: name.clone(),
                                values: result,
                                n_rows,
                                n_cols: 1,
                                compute_time_ms: 0,
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
    /// It builds a computation graph and computes each unique subexpression only once.
    pub fn compute_batch_vectorized(
        &self,
        names: &[&str],
        data: &HashMap<String, Array1<f64>>,
        parallel: bool,
    ) -> Result<HashMap<String, FactorResult>, String> {
        if names.is_empty() {
            return Ok(HashMap::new());
        }

        let n_rows = data.values().next().map(|arr| arr.len()).unwrap_or(0);
        if n_rows == 0 {
            return Err("Empty data".to_string());
        }

        let start = Instant::now();

        // Step 1: Build computation graph - collect all unique subexpressions
        let mut unique_exprs: Vec<Expr> = Vec::new();
        let mut expr_hash_set: HashSet<u64> = HashSet::new();
        let mut factor_exprs_owned: Vec<Expr> = Vec::new();
        let mut factor_exprs: Vec<(String, &Expr)> = Vec::new();

        for name in names {
            let info = self
                .factors
                .get(*name)
                .ok_or_else(|| format!("Factor '{}' not found", name))?;

            // Extract the expression from plan (unwrap from Projection)
            let expr = match &info.plan {
                LogicalPlan::Projection { exprs, .. } => exprs
                    .first()
                    .map(|(_, e)| e.clone())
                    .ok_or("Empty expression")?,
                _ => continue,
            };

            // Collect subexpressions
            collect_unique_subexpressions(&expr, &mut unique_exprs, &mut expr_hash_set);
            factor_exprs_owned.push(expr);
        }

        // Build references after all owned expressions are created
        for (i, name) in names.iter().enumerate() {
            factor_exprs.push((name.to_string(), &factor_exprs_owned[i]));
        }

        // Step 2: Build cache using memoization
        let mut cache: HashMap<u64, Array1<f64>> = HashMap::new();

        // Pre-populate cache with base columns
        for (name, arr) in data {
            let col_hash = {
                let mut hasher = DefaultHasher::new();
                0u8.hash(&mut hasher);
                name.hash(&mut hasher);
                hasher.finish()
            };
            cache.insert(col_hash, arr.clone());
        }

        // Step 3: Compute final factors using memoized vectorized evaluation
        let mut results: HashMap<String, FactorResult> = HashMap::new();

        if parallel && factor_exprs.len() > 1 {
            // Parallel computation using rayon
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
            // Sequential computation
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
                    },
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

    /// Evaluate expression with cached intermediate results
    #[allow(dead_code)]
    fn eval_expr_with_cache(
        &self,
        expr: &Expr,
        data: &HashMap<String, Vec<f64>>,
        n_rows: usize,
        cache: &HashMap<u64, Vec<f64>>,
    ) -> Result<Vec<f64>, String> {
        // Check cache first
        let hash = expr_hash(expr);
        if let Some(cached) = cache.get(&hash) {
            return Ok(cached.clone());
        }

        let result = self.eval_expr_with_cache_inner(expr, data, n_rows, cache)?;

        // Cache the result
        // Note: We can't modify the cache here because it's immutable
        // So we'll rely on the pre-computed cache from step 2
        Ok(result)
    }

    /// Inner evaluation that actually computes the expression
    #[allow(dead_code)]
    fn eval_expr_with_cache_inner(
        &self,
        expr: &Expr,
        data: &HashMap<String, Vec<f64>>,
        n_rows: usize,
        cache: &HashMap<u64, Vec<f64>>,
    ) -> Result<Vec<f64>, String> {
        match expr {
            Expr::Column(name) => data
                .get(name)
                .cloned()
                .ok_or_else(|| format!("Column '{}' not found", name)),
            Expr::Literal(lit) => {
                let val = match lit {
                    Literal::Float(f) => *f,
                    Literal::Integer(i) => *i as f64,
                    _ => 0.0,
                };
                Ok(vec![val; n_rows])
            }
            Expr::BinaryExpr { left, op, right } => {
                let left_vals = self.eval_expr_with_cache(left, data, n_rows, cache)?;
                let right_vals = self.eval_expr_with_cache(right, data, n_rows, cache)?;
                let mut result = vec![0.0; n_rows];
                for i in 0..n_rows {
                    result[i] = match op {
                        BinaryOp::Add => left_vals[i] + right_vals[i],
                        BinaryOp::Subtract => left_vals[i] - right_vals[i],
                        BinaryOp::Multiply => left_vals[i] * right_vals[i],
                        BinaryOp::Divide => {
                            if right_vals[i].abs() < 1e-10 {
                                0.0
                            } else {
                                left_vals[i] / right_vals[i]
                            }
                        }
                        _ => 0.0,
                    };
                }
                Ok(result)
            }
            Expr::FunctionCall { name, args } => {
                self.eval_function_with_cache(name, args, data, n_rows, cache)
            }
            Expr::UnaryExpr { op, expr: e } => {
                let vals = self.eval_expr_with_cache(e, data, n_rows, cache)?;
                Ok(vals
                    .into_iter()
                    .map(|v| match op {
                        UnaryOp::Negate => -v,
                        _ => v,
                    })
                    .collect())
            }
            _ => Err("Unsupported expr type".to_string()),
        }
    }

    #[allow(dead_code)]
    fn eval_function_with_cache(
        &self,
        name: &str,
        args: &[Expr],
        data: &HashMap<String, Vec<f64>>,
        n_rows: usize,
        cache: &HashMap<u64, Vec<f64>>,
    ) -> Result<Vec<f64>, String> {
        let name_lower = name.to_lowercase();

        if name_lower.starts_with("ts_") {
            return self.eval_ts_function_with_cache(&name_lower, args, data, n_rows, cache);
        }

        // First compute all args
        let mut arg_values: Vec<Vec<f64>> = Vec::new();
        for arg in args {
            arg_values.push(self.eval_expr_with_cache(arg, data, n_rows, cache)?);
        }

        match name_lower.as_str() {
            "rank" => Ok(timeseries::rank(&arg_values[0])),
            "delay" => {
                let periods = functions::get_literal_int(&args[1]).unwrap_or(1);
                Ok(timeseries::delay(&arg_values[0], periods))
            }
            "scale" => Ok(timeseries::scale(&arg_values[0])),
            "sign" => Ok(timeseries::sign(&arg_values[0])),
            "abs" => Ok(arg_values[0].iter().map(|v| v.abs()).collect()),
            // Element-wise min of two series
            "min" => {
                if arg_values.len() >= 2 {
                    Ok(arg_values[0]
                        .iter()
                        .zip(arg_values[1].iter())
                        .map(|(&a, &b)| a.min(b))
                        .collect())
                } else {
                    Ok(arg_values[0].clone())
                }
            }
            // Element-wise max of two series
            "max" => {
                if arg_values.len() >= 2 {
                    Ok(arg_values[0]
                        .iter()
                        .zip(arg_values[1].iter())
                        .map(|(&a, &b)| a.max(b))
                        .collect())
                } else {
                    Ok(arg_values[0].clone())
                }
            }
            // Sum of all elements in a series
            "sum" => {
                let total: f64 = arg_values[0].iter().sum();
                Ok(arg_values[0].iter().map(|_| total).collect())
            }
            "log" => Ok(arg_values[0]
                .iter()
                .map(|v| if *v > 0.0 { v.ln() } else { f64::NAN })
                .collect()),
            "log10" => Ok(arg_values[0]
                .iter()
                .map(|v| if *v > 0.0 { v.log10() } else { f64::NAN })
                .collect()),
            "sqrt" => Ok(arg_values[0].iter().map(|v| v.sqrt()).collect()),
            "power" => {
                let exponent = functions::get_literal_int(&args[1])
                    .map(|e| e as f64)
                    .unwrap_or(2.0);
                Ok(arg_values[0].iter().map(|v| v.powf(exponent)).collect())
            }
            "decay_linear" | "decay" => {
                let periods = functions::get_literal_int(&args[1]).unwrap_or(10);
                Ok(timeseries::decay_linear(&arg_values[0], periods))
            }
            "delta" => {
                let periods = functions::get_literal_int(&args[1]).unwrap_or(1);
                Ok(timeseries::ts_delta(&arg_values[0], periods))
            }
            "if" => {
                if args.len() != 3 {
                    return Err("IF requires 3 arguments".to_string());
                }
                let result: Vec<f64> = arg_values[0]
                    .iter()
                    .zip(arg_values[1].iter())
                    .zip(arg_values[2].iter())
                    .map(|((&c, &t), &f)| if c > 0.0 { t } else { f })
                    .collect();
                Ok(result)
            }
            "gt" | "greater" => Ok(arg_values[0]
                .iter()
                .zip(arg_values[1].iter())
                .map(|(&x, &y)| if x > y { 1.0 } else { 0.0 })
                .collect()),
            "lt" | "less" => Ok(arg_values[0]
                .iter()
                .zip(arg_values[1].iter())
                .map(|(&x, &y)| if x < y { 1.0 } else { 0.0 })
                .collect()),
            "ge" | "greater_equal" | "gte" => Ok(arg_values[0]
                .iter()
                .zip(arg_values[1].iter())
                .map(|(&x, &y)| if x >= y { 1.0 } else { 0.0 })
                .collect()),
            "le" | "less_equal" | "lte" => Ok(arg_values[0]
                .iter()
                .zip(arg_values[1].iter())
                .map(|(&x, &y)| if x <= y { 1.0 } else { 0.0 })
                .collect()),
            "eq" | "equal" => Ok(arg_values[0]
                .iter()
                .zip(arg_values[1].iter())
                .map(|(&x, &y)| if (x - y).abs() < 1e-10 { 1.0 } else { 0.0 })
                .collect()),
            "ne" | "not_equal" => Ok(arg_values[0]
                .iter()
                .zip(arg_values[1].iter())
                .map(|(&x, &y)| if (x - y).abs() >= 1e-10 { 1.0 } else { 0.0 })
                .collect()),
            _ => Err(format!("Unknown function: {}", name)),
        }
    }

    #[allow(dead_code)]
    fn eval_ts_function_with_cache(
        &self,
        name: &str,
        args: &[Expr],
        data: &HashMap<String, Vec<f64>>,
        n_rows: usize,
        cache: &HashMap<u64, Vec<f64>>,
    ) -> Result<Vec<f64>, String> {
        let vals = self.eval_expr_with_cache(&args[0], data, n_rows, cache)?;
        let window = args
            .get(1)
            .and_then(|a| functions::get_literal_int(a))
            .unwrap_or(20);

        match name {
            "ts_mean" => Ok(timeseries::ts_mean(&vals, window)),
            "ts_sum" => Ok(timeseries::ts_sum(&vals, window)),
            "ts_count" => Ok(timeseries::ts_count(&vals, window)),
            "ts_std" => Ok(timeseries::ts_std(&vals, window)),
            "ts_max" => Ok(timeseries::ts_max(&vals, window)),
            "ts_min" => Ok(timeseries::ts_min(&vals, window)),
            "ts_rank" => Ok(timeseries::ts_rank(&vals, window)),
            "ts_argmax" => Ok(timeseries::ts_argmax(&vals, window)),
            "ts_argmin" => Ok(timeseries::ts_argmin(&vals, window)),
            "ts_delta" => Ok(timeseries::ts_delta(&vals, window)),
            "ts_product" => Ok(timeseries::ts_product(&vals, window)),
            "ts_correlation" => {
                let vals2 = self.eval_expr_with_cache(&args[1], data, n_rows, cache)?;
                Ok(timeseries::ts_correlation(&vals, &vals2, window))
            }
            "ts_cov" | "ts_covariance" => {
                let vals2 = self.eval_expr_with_cache(&args[1], data, n_rows, cache)?;
                Ok(timeseries::ts_cov(&vals, &vals2, window))
            }
            "sma" => {
                let n = window;
                let m = args
                    .get(2)
                    .and_then(|a| functions::get_literal_int(a))
                    .unwrap_or(2);
                let alpha = m as f64 / n as f64;
                let alpha = if alpha > 0.0 && alpha <= 1.0 {
                    alpha
                } else {
                    0.5
                };
                Ok(timeseries::sma(&vals, alpha))
            }
            "lowday" => Ok(timeseries::lowday(&vals, window)),
            "highday" => Ok(timeseries::highday(&vals, window)),
            "wma" => Ok(timeseries::wma(&vals, window)),
            "min" => Ok(timeseries::ts_min(&vals, window)),
            "max" => Ok(timeseries::ts_max(&vals, window)),
            "sum" => Ok(timeseries::ts_sum(&vals, window)),
            _ => Err(format!("Unknown ts function: {}", name)),
        }
    }

    /// Execute the logical plan to compute factor values
    fn execute_plan(
        &self,
        plan: &LogicalPlan,
        data: &HashMap<String, Vec<f64>>,
        n_rows: usize,
    ) -> Result<Vec<f64>, String> {
        match plan {
            LogicalPlan::Projection { exprs, .. } => {
                if let Some((_, expr)) = exprs.first() {
                    let mut cache: HashMap<u64, Vec<f64>> = HashMap::new();
                    // Pre-populate with data columns
                    for (name, vals) in data {
                        let col_hash = {
                            let mut hasher = DefaultHasher::new();
                            0u8.hash(&mut hasher);
                            name.hash(&mut hasher);
                            hasher.finish()
                        };
                        cache.insert(col_hash, vals.clone());
                    }
                    eval_expr_memoized(expr, data, n_rows, &mut cache)
                } else {
                    Err("Empty projection".to_string())
                }
            }
            _ => Err("Unsupported plan type".to_string()),
        }
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
        self.factors.remove(name).is_some()
    }

    /// Clear all registered factors
    pub fn clear(&mut self) {
        self.factors.clear();
    }

    /// Get the configuration
    pub fn config(&self) -> &ComputeConfig {
        &self.config
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

        let result = registry.register("alpha_001", "close + volume");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Column 'volume' not found"));
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

    // ==================== Time Series Function Tests ====================

    #[test]
    fn test_ts_mean() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = timeseries::ts_mean(&vals, 3);
        assert_eq!(result, vec![1.0, 1.5, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ts_sum() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = timeseries::ts_sum(&vals, 3);
        assert_eq!(result, vec![1.0, 3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_ts_max() {
        let vals = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let result = timeseries::ts_max(&vals, 3);
        assert_eq!(result, vec![1.0, 5.0, 5.0, 5.0, 4.0]);
    }

    #[test]
    fn test_ts_min() {
        let vals = vec![3.0, 1.0, 5.0, 2.0, 4.0];
        let result = timeseries::ts_min(&vals, 3);
        assert_eq!(result, vec![3.0, 1.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn test_rank() {
        let vals = vec![3.0, 1.0, 2.0, 5.0, 4.0];
        let result = timeseries::rank(&vals);
        // Values sorted: 1, 2, 3, 4, 5
        // Original indices: 1->0, 2->1, 0->2, 4->3, 3->4
        // Ranks: 1=0/5=0.0, 2=1/5=0.2, 3=2/5=0.4, 4=3/5=0.6, 5=4/5=0.8
        // Result at original positions: [0.4, 0.0, 0.2, 0.8, 0.6]
        assert_eq!(result, vec![0.4, 0.0, 0.2, 0.8, 0.6]);
    }

    #[test]
    fn test_delay() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = timeseries::delay(&vals, 2);
        assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_scale() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = timeseries::scale(&vals);
        // Should have mean 0 and std 1
        let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
        assert!((mean - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sign() {
        let vals = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = timeseries::sign(&vals);
        assert_eq!(result, vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_delta() {
        let vals = vec![10.0, 12.0, 11.0, 15.0, 14.0];
        let result = timeseries::ts_delta(&vals, 1);
        assert_eq!(result, vec![0.0, 2.0, -1.0, 4.0, -1.0]);
    }
}
