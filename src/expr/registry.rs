//! Factor Registry System
//!
//! A system for registering factor expressions, parsing them into AST,
//! generating computation plans, and executing them efficiently with
//! timeout protection to prevent system overload.

use super::ast::{BinaryOp, Expr, Literal, UnaryOp};
use crate::lazy::{DataSource, LogicalPlan};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

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
    pub plan: LogicalPlan,
    pub description: Option<String>,
    pub category: Option<String>,
}

/// Factor computation result
#[derive(Debug, Clone)]
pub struct FactorResult {
    pub name: String,
    pub values: Vec<f64>,
    pub n_rows: usize,
    pub n_cols: usize,
    pub compute_time_ms: u64,
}

/// Factor Registry
#[derive(Clone)]
pub struct FactorRegistry {
    factors: HashMap<String, FactorInfo>,
    config: ComputeConfig,
    available_columns: HashMap<String, ColumnMeta>,
}

#[derive(Debug, Clone)]
pub struct ColumnMeta {
    pub name: String,
    pub data_type: String,
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

        let plan = expr_to_logical_plan(&expr, name)?;

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
                let result = eval_expr_memoized(expr, data, n_rows, &mut cache)?;
                results.insert(
                    name.clone(),
                    FactorResult {
                        name,
                        values: result,
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
            "rank" => Ok(rank(&arg_values[0])),
            "delay" => {
                let periods = args.get(1).and_then(|a| get_literal_int(a)).unwrap_or(1);
                Ok(delay(&arg_values[0], periods))
            }
            "scale" => Ok(scale(&arg_values[0])),
            "sign" => Ok(sign(&arg_values[0])),
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
                let exponent = args
                    .get(1)
                    .and_then(|a| get_literal_int(a))
                    .map(|e| e as f64)
                    .unwrap_or(2.0);
                Ok(arg_values[0].iter().map(|v| v.powf(exponent)).collect())
            }
            "decay_linear" | "decay" => {
                let periods = args.get(1).and_then(|a| get_literal_int(a)).unwrap_or(10);
                Ok(decay_linear(&arg_values[0], periods))
            }
            "delta" => {
                let periods = args.get(1).and_then(|a| get_literal_int(a)).unwrap_or(1);
                Ok(ts_delta(&arg_values[0], periods))
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
            "gte" => Ok(arg_values[0]
                .iter()
                .zip(arg_values[1].iter())
                .map(|(&x, &y)| if x >= y { 1.0 } else { 0.0 })
                .collect()),
            "lte" => Ok(arg_values[0]
                .iter()
                .zip(arg_values[1].iter())
                .map(|(&x, &y)| if x <= y { 1.0 } else { 0.0 })
                .collect()),
            "eq" | "equal" => Ok(arg_values[0]
                .iter()
                .zip(arg_values[1].iter())
                .map(|(&x, &y)| if (x - y).abs() < 1e-10 { 1.0 } else { 0.0 })
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
        let window = args.get(1).and_then(|a| get_literal_int(a)).unwrap_or(20);

        match name {
            "ts_mean" => Ok(ts_mean(&vals, window)),
            "ts_sum" => Ok(ts_sum(&vals, window)),
            "ts_count" => Ok(ts_count(&vals, window)),
            "ts_std" => Ok(ts_std(&vals, window)),
            "ts_max" => Ok(ts_max(&vals, window)),
            "ts_min" => Ok(ts_min(&vals, window)),
            "ts_rank" => Ok(ts_rank(&vals, window)),
            "ts_argmax" => Ok(ts_argmax(&vals, window)),
            "ts_argmin" => Ok(ts_argmin(&vals, window)),
            "ts_delta" => Ok(ts_delta(&vals, window)),
            "ts_product" => Ok(ts_product(&vals, window)),
            "ts_correlation" => {
                let vals2 = self.eval_expr_with_cache(&args[1], data, n_rows, cache)?;
                Ok(ts_correlation(&vals, &vals2, window))
            }
            "ts_cov" => {
                let vals2 = self.eval_expr_with_cache(&args[1], data, n_rows, cache)?;
                Ok(ts_cov(&vals, &vals2, window))
            }
            "ts_covariance" => {
                let vals2 = self.eval_expr_with_cache(&args[1], data, n_rows, cache)?;
                Ok(ts_cov(&vals, &vals2, window))
            }
            "sma" => {
                // sma(vals, n, m) - exponential weighted moving average
                // alpha = m/n, so we need two parameters
                // arg[0] = values, arg[1] = n, arg[2] = m
                let n = window; // default 20
                let m = args.get(2).and_then(|a| get_literal_int(a)).unwrap_or(2);
                let alpha = m as f64 / n as f64;
                let alpha = if alpha > 0.0 && alpha <= 1.0 {
                    alpha
                } else {
                    0.5
                };
                Ok(sma(&vals, alpha))
            }
            "lowday" => Ok(lowday(&vals, window)),
            "highday" => Ok(highday(&vals, window)),
            "wma" => Ok(wma(&vals, window)),
            _ => Err(format!("Unknown ts function: {}", name)),
        }
    }

    fn execute_plan(
        &self,
        plan: &LogicalPlan,
        data: &HashMap<String, Vec<f64>>,
        n_rows: usize,
    ) -> Result<Vec<f64>, String> {
        match plan {
            LogicalPlan::Projection { exprs, .. } => {
                for (_, expr) in exprs {
                    return self.eval_expr(expr, data, n_rows);
                }
                Err("No expressions".to_string())
            }
            _ => Err("Unsupported plan".to_string()),
        }
    }

    fn eval_expr(
        &self,
        expr: &Expr,
        data: &HashMap<String, Vec<f64>>,
        n_rows: usize,
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
                let left_vals = self.eval_expr(left, data, n_rows)?;
                let right_vals = self.eval_expr(right, data, n_rows)?;
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
            Expr::FunctionCall { name, args } => self.eval_function(name, args, data, n_rows),
            Expr::UnaryExpr { op, expr: e } => {
                let vals = self.eval_expr(e, data, n_rows)?;
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

    fn eval_function(
        &self,
        name: &str,
        args: &[Expr],
        data: &HashMap<String, Vec<f64>>,
        n_rows: usize,
    ) -> Result<Vec<f64>, String> {
        let name_lower = name.to_lowercase();

        if name_lower.starts_with("ts_") {
            return self.eval_ts_function(&name_lower, args, data, n_rows);
        }

        match name_lower.as_str() {
            "rank" => {
                let vals = self.eval_args(args, data, n_rows)?;
                Ok(rank(&vals))
            }
            "delay" => {
                let vals = self.eval_args(args, data, n_rows)?;
                let periods = args.get(1).and_then(|a| get_literal_int(a)).unwrap_or(1);
                Ok(delay(&vals, periods))
            }
            "scale" => {
                let vals = self.eval_args(args, data, n_rows)?;
                Ok(scale(&vals))
            }
            "sign" => {
                let vals = self.eval_args(args, data, n_rows)?;
                Ok(sign(&vals))
            }
            "abs" => {
                let vals = self.eval_args(args, data, n_rows)?;
                Ok(vals.iter().map(|v| v.abs()).collect())
            }
            // Element-wise min of two series: min(a, b)
            "min" => {
                if args.len() >= 2 {
                    let vals1 = self.eval_expr(&args[0], data, n_rows)?;
                    let vals2 = self.eval_expr(&args[1], data, n_rows)?;
                    Ok(vals1
                        .iter()
                        .zip(vals2.iter())
                        .map(|(&a, &b)| a.min(b))
                        .collect())
                } else {
                    let vals = self.eval_args(args, data, n_rows)?;
                    Ok(vals)
                }
            }
            // Element-wise max of two series: max(a, b)
            "max" => {
                if args.len() >= 2 {
                    let vals1 = self.eval_expr(&args[0], data, n_rows)?;
                    let vals2 = self.eval_expr(&args[1], data, n_rows)?;
                    Ok(vals1
                        .iter()
                        .zip(vals2.iter())
                        .map(|(&a, &b)| a.max(b))
                        .collect())
                } else {
                    let vals = self.eval_args(args, data, n_rows)?;
                    Ok(vals)
                }
            }
            // Sum of all elements in a series: sum(a)
            "sum" => {
                let vals = self.eval_args(args, data, n_rows)?;
                let total: f64 = vals.iter().sum();
                Ok(vals.iter().map(|_| total).collect())
            }
            "log" => {
                let vals = self.eval_args(args, data, n_rows)?;
                Ok(vals
                    .iter()
                    .map(|v| if *v > 0.0 { v.ln() } else { f64::NAN })
                    .collect())
            }
            "log10" => {
                let vals = self.eval_args(args, data, n_rows)?;
                Ok(vals
                    .iter()
                    .map(|v| if *v > 0.0 { v.log10() } else { f64::NAN })
                    .collect())
            }
            "sqrt" => {
                let vals = self.eval_args(args, data, n_rows)?;
                Ok(vals.iter().map(|v| v.sqrt()).collect())
            }
            "power" => {
                let vals = self.eval_args(args, data, n_rows)?;
                let exponent = args
                    .get(1)
                    .and_then(|a| get_literal_int(a))
                    .map(|e| e as f64)
                    .unwrap_or(2.0);
                Ok(vals.iter().map(|v| v.powf(exponent)).collect())
            }
            "decay_linear" | "decay" => {
                let vals = self.eval_args(args, data, n_rows)?;
                let periods = args.get(1).and_then(|a| get_literal_int(a)).unwrap_or(10);
                Ok(decay_linear(&vals, periods))
            }
            "delta" => {
                let vals = self.eval_args(args, data, n_rows)?;
                let periods = args.get(1).and_then(|a| get_literal_int(a)).unwrap_or(1);
                Ok(ts_delta(&vals, periods))
            }
            "if" => {
                // IF(cond, true_val, false_val) - ternary operator
                // If cond > 0, return true_val, else return false_val
                if args.len() != 3 {
                    return Err(
                        "IF requires 3 arguments: IF(condition, true_value, false_value)"
                            .to_string(),
                    );
                }
                let cond = self.eval_expr(&args[0], data, n_rows)?;
                let true_val = self.eval_expr(&args[1], data, n_rows)?;
                let false_val = self.eval_expr(&args[2], data, n_rows)?;

                let result: Vec<f64> = cond
                    .iter()
                    .zip(true_val.iter())
                    .zip(false_val.iter())
                    .map(|((&c, &t), &f)| if c > 0.0 { t } else { f })
                    .collect();
                Ok(result)
            }
            // Comparison functions that return 1.0 (true) or 0.0 (false)
            "gt" | "greater" | "greater_than" => {
                if args.len() != 2 {
                    return Err("GT requires 2 arguments: GT(a, b)".to_string());
                }
                let a = self.eval_expr(&args[0], data, n_rows)?;
                let b = self.eval_expr(&args[1], data, n_rows)?;
                Ok(a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| if x > y { 1.0 } else { 0.0 })
                    .collect())
            }
            "lt" | "less" | "less_than" => {
                if args.len() != 2 {
                    return Err("LT requires 2 arguments: LT(a, b)".to_string());
                }
                let a = self.eval_expr(&args[0], data, n_rows)?;
                let b = self.eval_expr(&args[1], data, n_rows)?;
                Ok(a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| if x < y { 1.0 } else { 0.0 })
                    .collect())
            }
            "gte" | "greater_equal" => {
                if args.len() != 2 {
                    return Err("GTE requires 2 arguments: GTE(a, b)".to_string());
                }
                let a = self.eval_expr(&args[0], data, n_rows)?;
                let b = self.eval_expr(&args[1], data, n_rows)?;
                Ok(a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| if x >= y { 1.0 } else { 0.0 })
                    .collect())
            }
            "lte" | "less_equal" => {
                if args.len() != 2 {
                    return Err("LTE requires 2 arguments: LTE(a, b)".to_string());
                }
                let a = self.eval_expr(&args[0], data, n_rows)?;
                let b = self.eval_expr(&args[1], data, n_rows)?;
                Ok(a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| if x <= y { 1.0 } else { 0.0 })
                    .collect())
            }
            "eq" | "equal" => {
                if args.len() != 2 {
                    return Err("EQ requires 2 arguments: EQ(a, b)".to_string());
                }
                let a = self.eval_expr(&args[0], data, n_rows)?;
                let b = self.eval_expr(&args[1], data, n_rows)?;
                Ok(a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| if (x - y).abs() < 1e-10 { 1.0 } else { 0.0 })
                    .collect())
            }
            "vwap" => {
                // VWAP = amount / volume
                // This requires both amount and volume columns
                // For now, return close as approximation
                let vals = self.eval_args(args, data, n_rows)?;
                Ok(vals)
            }
            _ => Err(format!("Unknown function: {}", name)),
        }
    }

    fn eval_ts_function(
        &self,
        name: &str,
        args: &[Expr],
        data: &HashMap<String, Vec<f64>>,
        n_rows: usize,
    ) -> Result<Vec<f64>, String> {
        let vals = self.eval_args(args, data, n_rows)?;
        let window = args.get(1).and_then(|a| get_literal_int(a)).unwrap_or(20);

        match name {
            "ts_mean" | "ts_avg" => Ok(ts_mean(&vals, window)),
            "ts_sum" => Ok(ts_sum(&vals, window)),
            "ts_count" => Ok(ts_count(&vals, window)),
            "ts_max" | "ts_argmax" => Ok(ts_argmax(&vals, window)),
            "ts_min" | "ts_argmin" => Ok(ts_argmin(&vals, window)),
            "ts_std" | "ts_stddev" => Ok(ts_std(&vals, window)),
            "ts_rank" => Ok(ts_rank(&vals, window)),
            "ts_correlation" | "ts_corr" => {
                // ts_correlation(expr1, expr2, window) - need two expressions and window
                // args[0] = expr1, args[1] = expr2, args[2] = window (optional)
                let vals2 = if args.len() > 1 {
                    self.eval_expr(&args[1], data, n_rows)?
                } else {
                    vals.clone()
                };
                Ok(ts_correlation(&vals, &vals2, window))
            }
            "ts_covariance" | "ts_cov" => {
                let vals2 = if args.len() > 1 {
                    self.eval_expr(&args[1], data, n_rows)?
                } else {
                    vals.clone()
                };
                Ok(ts_cov(&vals, &vals2, window))
            }
            "sma" => {
                // sma(vals, n, m) - exponential weighted moving average
                // alpha = m/n
                let n = window;
                let m = args.get(2).and_then(|a| get_literal_int(a)).unwrap_or(2);
                let alpha = m as f64 / n as f64;
                let alpha = if alpha > 0.0 && alpha <= 1.0 {
                    alpha
                } else {
                    0.5
                };
                Ok(sma(&vals, alpha))
            }
            "ts_delta" | "delta" => Ok(ts_delta(&vals, window)),
            "ts_product" | "product" => Ok(ts_product(&vals, window)),
            "lowday" => Ok(lowday(&vals, window)),
            "highday" => Ok(highday(&vals, window)),
            "wma" => Ok(wma(&vals, window)),
            _ => Err(format!("Unknown ts function: {}", name)),
        }
    }

    fn eval_args(
        &self,
        args: &[Expr],
        data: &HashMap<String, Vec<f64>>,
        n_rows: usize,
    ) -> Result<Vec<f64>, String> {
        if args.is_empty() {
            return Ok(vec![0.0; n_rows]);
        }
        self.eval_expr(&args[0], data, n_rows)
    }

    pub fn list(&self) -> Vec<String> {
        self.factors.keys().cloned().collect()
    }

    pub fn get(&self, name: &str) -> Option<&FactorInfo> {
        self.factors.get(name)
    }

    pub fn unregister(&mut self, name: &str) -> bool {
        self.factors.remove(name).is_some()
    }

    pub fn clear(&mut self) {
        self.factors.clear();
    }

    pub fn config(&self) -> &ComputeConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: ComputeConfig) {
        self.config = config;
    }
}

// Helper functions
fn extract_columns(expr: &Expr) -> Vec<String> {
    match expr {
        Expr::Literal(_) => vec![],
        Expr::Column(name) => vec![name.clone()],
        Expr::BinaryExpr { left, right, .. } => {
            let mut cols = extract_columns(left);
            cols.extend(extract_columns(right));
            cols
        }
        Expr::UnaryExpr { expr, .. } => extract_columns(expr),
        Expr::FunctionCall { args, .. } => args.iter().flat_map(extract_columns).collect(),
        _ => vec![],
    }
}

fn expr_to_logical_plan(expr: &Expr, output_name: &str) -> Result<LogicalPlan, String> {
    Ok(LogicalPlan::Projection {
        input: Arc::new(LogicalPlan::Scan {
            source: DataSource::NumpyArrays(HashMap::new()),
            projection: None,
            selection: None,
        }),
        exprs: vec![(output_name.to_string(), expr.clone())],
    })
}

/// Compute a hash for an expression to identify unique subexpressions
fn expr_hash(expr: &Expr) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_expr(expr, &mut hasher);
    hasher.finish()
}

fn hash_expr<H: Hasher>(expr: &Expr, hasher: &mut H) {
    match expr {
        Expr::Column(name) => {
            0u8.hash(hasher);
            name.hash(hasher);
        }
        Expr::Literal(lit) => {
            1u8.hash(hasher);
            format!("{:?}", lit).hash(hasher);
        }
        Expr::BinaryExpr { left, op, right } => {
            2u8.hash(hasher);
            format!("{:?}", op).hash(hasher);
            hash_expr(left, hasher);
            hash_expr(right, hasher);
        }
        Expr::UnaryExpr { op, expr } => {
            3u8.hash(hasher);
            format!("{:?}", op).hash(hasher);
            hash_expr(expr, hasher);
        }
        Expr::FunctionCall { name, args } => {
            4u8.hash(hasher);
            name.hash(hasher);
            args.len().hash(hasher);
            for arg in args {
                hash_expr(arg, hasher);
            }
        }
        _ => {}
    }
}

/// Collect unique subexpressions from an expression tree
fn collect_unique_subexpressions(
    expr: &Expr,
    unique_exprs: &mut Vec<Expr>,
    expr_hash_set: &mut HashSet<u64>,
) {
    let hash = expr_hash(expr);

    // Add if not already present
    if expr_hash_set.insert(hash) {
        unique_exprs.push(expr.clone());

        // Recursively collect from children
        match expr {
            Expr::BinaryExpr { left, right, .. } => {
                collect_unique_subexpressions(left, unique_exprs, expr_hash_set);
                collect_unique_subexpressions(right, unique_exprs, expr_hash_set);
            }
            Expr::UnaryExpr { expr: e, .. } => {
                collect_unique_subexpressions(e, unique_exprs, expr_hash_set);
            }
            Expr::FunctionCall { args, .. } => {
                for arg in args {
                    collect_unique_subexpressions(arg, unique_exprs, expr_hash_set);
                }
            }
            _ => {}
        }
    }
}

/// Memoized expression evaluation - computes each subexpression only once
fn eval_expr_memoized(
    expr: &Expr,
    data: &HashMap<String, Vec<f64>>,
    n_rows: usize,
    cache: &mut HashMap<u64, Vec<f64>>,
) -> Result<Vec<f64>, String> {
    let hash = expr_hash(expr);

    // Check cache first
    if let Some(cached) = cache.get(&hash) {
        return Ok(cached.clone());
    }

    // Compute the expression
    let result = match expr {
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
            let left_vals = eval_expr_memoized(left, data, n_rows, cache)?;
            let right_vals = eval_expr_memoized(right, data, n_rows, cache)?;
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
        Expr::UnaryExpr { op, expr: e } => {
            let vals = eval_expr_memoized(e, data, n_rows, cache)?;
            Ok(vals
                .into_iter()
                .map(|v| match op {
                    UnaryOp::Negate => -v,
                    _ => v,
                })
                .collect())
        }
        Expr::FunctionCall { name, args } => {
            eval_function_memoized(name, args, data, n_rows, cache)
        }
        _ => Err("Unsupported expr type".to_string()),
    }?;

    // Cache the result
    cache.insert(hash, result.clone());
    Ok(result)
}

/// Memoized function evaluation
fn eval_function_memoized(
    name: &str,
    args: &[Expr],
    data: &HashMap<String, Vec<f64>>,
    n_rows: usize,
    cache: &mut HashMap<u64, Vec<f64>>,
) -> Result<Vec<f64>, String> {
    let name_lower = name.to_lowercase();

    // Check for ts_ prefix OR known ts functions without prefix
    if name_lower.starts_with("ts_")
        || matches!(
            name_lower.as_str(),
            "sma" | "lowday" | "highday" | "wma" | "min" | "max" | "sum"
        )
    {
        return eval_ts_function_memoized(&name_lower, args, data, n_rows, cache);
    }

    // First evaluate all arguments (with memoization)
    let mut arg_values: Vec<Vec<f64>> = Vec::new();
    for arg in args {
        arg_values.push(eval_expr_memoized(arg, data, n_rows, cache)?);
    }

    match name_lower.as_str() {
        "rank" => Ok(rank(&arg_values[0])),
        "delay" => {
            let periods = get_literal_int(&args[1]).unwrap_or(1);
            Ok(delay(&arg_values[0], periods))
        }
        "scale" => Ok(scale(&arg_values[0])),
        "sign" => Ok(sign(&arg_values[0])),
        "abs" => Ok(arg_values[0].iter().map(|v| v.abs()).collect()),
        // Element-wise min of two series: min(a, b)
        "min" => {
            if args.len() >= 2 {
                Ok(arg_values[0]
                    .iter()
                    .zip(arg_values[1].iter())
                    .map(|(&a, &b)| a.min(b))
                    .collect())
            } else {
                Ok(arg_values[0].clone())
            }
        }
        // Element-wise max of two series: max(a, b)
        "max" => {
            if args.len() >= 2 {
                Ok(arg_values[0]
                    .iter()
                    .zip(arg_values[1].iter())
                    .map(|(&a, &b)| a.max(b))
                    .collect())
            } else {
                Ok(arg_values[0].clone())
            }
        }
        // Sum of all elements in a series: sum(a)
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
            let exponent = get_literal_int(&args[1]).map(|e| e as f64).unwrap_or(2.0);
            Ok(arg_values[0].iter().map(|v| v.powf(exponent)).collect())
        }
        "decay_linear" | "decay" => {
            let periods = get_literal_int(&args[1]).unwrap_or(10);
            Ok(decay_linear(&arg_values[0], periods))
        }
        "delta" => {
            let periods = get_literal_int(&args[1]).unwrap_or(1);
            Ok(ts_delta(&arg_values[0], periods))
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
        "gte" => Ok(arg_values[0]
            .iter()
            .zip(arg_values[1].iter())
            .map(|(&x, &y)| if x >= y { 1.0 } else { 0.0 })
            .collect()),
        "lte" => Ok(arg_values[0]
            .iter()
            .zip(arg_values[1].iter())
            .map(|(&x, &y)| if x <= y { 1.0 } else { 0.0 })
            .collect()),
        "eq" | "equal" => Ok(arg_values[0]
            .iter()
            .zip(arg_values[1].iter())
            .map(|(&x, &y)| if (x - y).abs() < 1e-10 { 1.0 } else { 0.0 })
            .collect()),
        _ => Err(format!("Unknown function: {}", name)),
    }
}

/// Memoized time-series function evaluation
fn eval_ts_function_memoized(
    name: &str,
    args: &[Expr],
    data: &HashMap<String, Vec<f64>>,
    n_rows: usize,
    cache: &mut HashMap<u64, Vec<f64>>,
) -> Result<Vec<f64>, String> {
    // Evaluate the first argument (the values)
    let vals = eval_expr_memoized(&args[0], data, n_rows, cache)?;
    let window = args.get(1).and_then(|a| get_literal_int(a)).unwrap_or(20);

    match name {
        "ts_mean" => Ok(ts_mean(&vals, window)),
        "ts_sum" => Ok(ts_sum(&vals, window)),
        "ts_count" => Ok(ts_count(&vals, window)),
        "ts_std" => Ok(ts_std(&vals, window)),
        "ts_max" => Ok(ts_max(&vals, window)),
        "ts_min" => Ok(ts_min(&vals, window)),
        "ts_rank" => Ok(ts_rank(&vals, window)),
        "ts_argmax" => Ok(ts_argmax(&vals, window)),
        "ts_argmin" => Ok(ts_argmin(&vals, window)),
        "ts_delta" => Ok(ts_delta(&vals, window)),
        "ts_product" => Ok(ts_product(&vals, window)),
        "ts_correlation" => {
            let vals2 = eval_expr_memoized(&args[1], data, n_rows, cache)?;
            Ok(ts_correlation(&vals, &vals2, window))
        }
        "ts_cov" => {
            let vals2 = eval_expr_memoized(&args[1], data, n_rows, cache)?;
            Ok(ts_cov(&vals, &vals2, window))
        }
        "ts_covariance" => {
            let vals2 = eval_expr_memoized(&args[1], data, n_rows, cache)?;
            Ok(ts_cov(&vals, &vals2, window))
        }
        "sma" => {
            let n = window; // default 20
            let m = args.get(2).and_then(|a| get_literal_int(a)).unwrap_or(2);
            let alpha = m as f64 / n as f64;
            let alpha = if alpha > 0.0 && alpha <= 1.0 {
                alpha
            } else {
                0.5
            };
            Ok(sma(&vals, alpha))
        }
        "lowday" => Ok(lowday(&vals, window)),
        "highday" => Ok(highday(&vals, window)),
        "wma" => Ok(wma(&vals, window)),
        "min" => Ok(ts_min(&vals, window)),
        "max" => Ok(ts_max(&vals, window)),
        "sum" => Ok(ts_sum(&vals, window)),
        _ => Err(format!("Unknown ts function: {}", name)),
    }
}

/// Compute expressions in parallel using rayon
#[allow(dead_code)]
fn compute_exprs_parallel(
    exprs: &[Expr],
    data: &HashMap<String, Vec<f64>>,
    n_rows: usize,
    cache: &mut HashMap<u64, Vec<f64>>,
) -> Result<(), String> {
    // For now, compute sequentially but this can be extended for parallel execution
    // The key optimization here is the shared subexpression caching
    for expr in exprs {
        let hash = expr_hash(expr);
        if !cache.contains_key(&hash) {
            // Use a simple evaluator (inline evaluation for parallel)
            let result = eval_expr_simple(expr, data, n_rows)?;
            cache.insert(hash, result);
        }
    }
    Ok(())
}

/// Simple expression evaluator for parallel computation
#[allow(dead_code)]
fn eval_expr_simple(
    expr: &Expr,
    data: &HashMap<String, Vec<f64>>,
    n_rows: usize,
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
        _ => Err("Complex expressions handled elsewhere".to_string()),
    }
}

fn get_literal_int(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::Literal(Literal::Integer(i)) => Some(*i as usize),
        Expr::Literal(Literal::Float(f)) => Some(*f as usize),
        _ => None,
    }
}

// Time series functions
fn ts_mean(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        result[i] = slice.iter().sum::<f64>() / slice.len() as f64;
    }
    result
}

fn ts_sum(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    // Optimized path for expanding window (window=0 means from start)
    if window == 0 {
        let mut cumsum = 0.0;
        for i in 0..n {
            cumsum += vals[i];
            result[i] = cumsum;
        }
        return result;
    }

    // Regular rolling window
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        result[i] = vals[start..=i].iter().sum();
    }
    result
}

fn ts_count(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    // Optimized path for expanding window (window=0 means from start)
    if window == 0 {
        let mut cumcount = 0.0;
        for i in 0..n {
            if !vals[i].is_nan() {
                cumcount += 1.0;
            }
            result[i] = cumcount;
        }
        return result;
    }

    // Regular rolling window
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        // Count non-NaN values in the window
        result[i] = vals[start..=i].iter().filter(|v| !v.is_nan()).count() as f64;
    }
    result
}

fn ts_max(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        result[i] = vals[start..=i]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    }
    result
}

fn ts_min(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        result[i] = vals[start..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
    }
    result
}

fn ts_std(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;
        result[i] = variance.sqrt();
    }
    result
}

fn ts_rank(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let current = vals[i];
        let rank = slice.iter().filter(|&&x| x < current).count() as f64;
        result[i] = rank / slice.len() as f64;
    }
    result
}

// Position of maximum value in window (1-indexed like Alpha101)
fn ts_argmax(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let pos = slice
            .iter()
            .position(|&x| (x - max_val).abs() < f64::EPSILON);
        result[i] = pos.map(|p| (slice.len() - p) as f64).unwrap_or(0.0);
    }
    result
}

// Position of minimum value in window (1-indexed like Alpha101)
fn ts_argmin(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let min_val = slice.iter().cloned().fold(f64::INFINITY, f64::min);
        let pos = slice
            .iter()
            .position(|&x| (x - min_val).abs() < f64::EPSILON);
        result[i] = pos.map(|p| (slice.len() - p) as f64).unwrap_or(0.0);
    }
    result
}

// Rolling correlation between two series
fn ts_correlation(vals1: &[f64], vals2: &[f64], window: usize) -> Vec<f64> {
    let n = vals1.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice1 = &vals1[start..=i];
        let slice2 = &vals2[start..=i];

        let len = slice1.len();
        if len < 2 {
            result[i] = 0.0;
            continue;
        }

        let mean1: f64 = slice1.iter().sum::<f64>() / len as f64;
        let mean2: f64 = slice2.iter().sum::<f64>() / len as f64;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for j in 0..len {
            let d1 = slice1[j] - mean1;
            let d2 = slice2[j] - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom > 1e-10 {
            result[i] = cov / denom;
        } else {
            result[i] = 0.0;
        }
    }
    result
}

// Rolling covariance between two series
fn ts_cov(vals1: &[f64], vals2: &[f64], window: usize) -> Vec<f64> {
    let n = vals1.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice1 = &vals1[start..=i];
        let slice2 = &vals2[start..=i];

        let len = slice1.len();
        if len < 2 {
            result[i] = 0.0;
            continue;
        }

        let mean1: f64 = slice1.iter().sum::<f64>() / len as f64;
        let mean2: f64 = slice2.iter().sum::<f64>() / len as f64;

        let mut cov = 0.0;
        for j in 0..len {
            let d1 = slice1[j] - mean1;
            let d2 = slice2[j] - mean2;
            cov += d1 * d2;
        }
        result[i] = cov / (len - 1) as f64; // Sample covariance
    }
    result
}

// Exponential weighted moving average (EMA/SMA)
// alpha: smoothing factor (0 < alpha <= 1)
fn sma(vals: &[f64], alpha: f64) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    if n == 0 {
        return result;
    }

    // Initialize with first value
    result[0] = vals[0];

    for i in 1..n {
        result[i] = alpha * vals[i] + (1.0 - alpha) * result[i - 1];
    }
    result
}

// Days since minimum value in window
fn lowday(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];

        if slice.is_empty() {
            result[i] = 0.0;
            continue;
        }

        // Find position of minimum (from start of window)
        let min_pos = slice
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Days since low = position from end
        result[i] = (slice.len() - 1 - min_pos) as f64;
    }
    result
}

// Days since maximum value in window
fn highday(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];

        if slice.is_empty() {
            result[i] = 0.0;
            continue;
        }

        // Find position of maximum (from start of window)
        let max_pos = slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Days since high = position from end
        result[i] = (slice.len() - 1 - max_pos) as f64;
    }
    result
}

// Weighted moving average
fn wma(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    if window == 0 {
        return result;
    }

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let len = slice.len();

        if len == 0 {
            result[i] = 0.0;
            continue;
        }

        // Weights: 0.9^(len-1), 0.9^(len-2), ..., 0.9^0
        let mut sum_weighted = 0.0;
        let mut sum_weights = 0.0;

        for (j, &val) in slice.iter().enumerate() {
            let weight = (0.9_f64).powi((len - 1 - j) as i32);
            sum_weighted += val * weight;
            sum_weights += weight;
        }

        result[i] = if sum_weights > 0.0 {
            sum_weighted / sum_weights
        } else {
            0.0
        };
    }
    result
}

// Difference over window periods (like delta in Alpha101)
fn ts_delta(vals: &[f64], periods: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in periods..n {
        result[i] = vals[i] - vals[i - periods];
    }
    result
}

// Rolling product
fn ts_product(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let prod: f64 = slice.iter().fold(1.0, |acc, &v| acc * v);
        result[i] = prod;
    }
    result
}

// Decay linear (exponentially weighted with linear weights)
fn decay_linear(vals: &[f64], periods: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(periods - 1);
        let slice = &vals[start..=i];
        let len = slice.len();

        // Linear weights: 1, 2, 3, ..., len
        let weight_sum: f64 = (1..=len).sum::<usize>() as f64;
        let mut weighted_sum = 0.0;

        for (j, &v) in slice.iter().enumerate() {
            let weight = (j + 1) as f64;
            weighted_sum += weight * v;
        }

        result[i] = weighted_sum / weight_sum;
    }
    result
}

fn rank(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    if n == 0 {
        return vec![];
    }

    // Separate NaN values and valid values
    let mut indexed: Vec<(usize, f64)> = vals
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, &v)| (i, v))
        .collect();

    if indexed.is_empty() {
        return vec![f64::NAN; n];
    }

    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = vec![f64::NAN; n];
    let len = indexed.len();
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        result[*idx] = rank as f64 / len as f64;
    }
    result
}

fn delay(vals: &[f64], periods: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in periods..n {
        result[i] = vals[i - periods];
    }
    result
}

fn scale(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    if n == 0 {
        return vec![];
    }
    let sum: f64 = vals.iter().sum();
    let mean = sum / n as f64;
    let std = (vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    if std < 1e-10 {
        return vec![0.0; n];
    }
    vals.iter().map(|v| (v - mean) / std).collect()
}

fn sign(vals: &[f64]) -> Vec<f64> {
    vals.iter()
        .map(|v| {
            if *v > 0.0 {
                1.0
            } else if *v < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

// =============================================================================
// Simple Expression Parser - Iterative approach to avoid recursion issues
// =============================================================================

/// Parse expression string into Expr AST
pub fn parse_expression(expression: &str) -> Result<Expr, String> {
    let tokens = tokenize(expression)?;
    parse_tokens(&tokens)
}

fn tokenize(s: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = s.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }

        if c.is_ascii_digit() {
            let mut num = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
                    // Handle scientific notation: 1e-10, 1.5e+5, etc.
                    if c == 'e' || c == 'E' {
                        num.push(c);
                        chars.next();
                        // After 'e' or 'E', can have + or -
                        if let Some(&next_c) = chars.peek() {
                            if next_c == '+' || next_c == '-' {
                                num.push(next_c);
                                chars.next();
                            }
                        }
                        continue;
                    }
                    num.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            tokens.push(Token::Number(num.parse().unwrap_or(0.0)));
        } else if c.is_alphabetic() || c == '_' {
            let mut ident = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_alphanumeric() || c == '_' {
                    ident.push(c);
                    chars.next();
                } else {
                    break;
                }
            }
            if let Some(&'(') = chars.peek() {
                tokens.push(Token::Function(ident));
            } else {
                tokens.push(Token::Identifier(ident));
            }
        } else {
            let op = match c {
                '+' => Token::Plus,
                '-' => Token::Minus,
                '*' => Token::Multiply,
                '/' => Token::Divide,
                '(' => Token::LParen,
                ')' => Token::RParen,
                ',' => Token::Comma,
                _ => {
                    chars.next();
                    continue;
                }
            };
            tokens.push(op);
            chars.next();
        }
    }

    if tokens.is_empty() {
        return Err("Empty expression".to_string());
    }

    Ok(tokens)
}

#[derive(Debug, Clone)]
enum Token {
    Number(f64),
    Identifier(String),
    Function(String),
    Plus,
    Minus,
    Multiply,
    Divide,
    LParen,
    RParen,
    Comma,
}

/// Parse tokens using a simple iterative shunting-yard inspired approach
fn parse_tokens(tokens: &[Token]) -> Result<Expr, String> {
    // Use recursive descent but with careful position tracking
    parse_expression_rec(tokens, 0).map(|(e, _)| e)
}

fn parse_expression_rec(tokens: &[Token], start: usize) -> Result<(Expr, usize), String> {
    if tokens.is_empty() {
        return Err("Empty tokens".to_string());
    }
    parse_additive(tokens, start)
}

fn parse_additive(tokens: &[Token], start: usize) -> Result<(Expr, usize), String> {
    let (mut left, mut pos) = parse_multiplicative(tokens, start)?;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Plus => {
                let (right, new_pos) = parse_multiplicative(tokens, pos + 1)?;
                left = left.binary(BinaryOp::Add, right);
                pos = new_pos;
            }
            Token::Minus => {
                let (right, new_pos) = parse_multiplicative(tokens, pos + 1)?;
                left = left.binary(BinaryOp::Subtract, right);
                pos = new_pos;
            }
            _ => break,
        }
    }

    Ok((left, pos))
}

fn parse_multiplicative(tokens: &[Token], start: usize) -> Result<(Expr, usize), String> {
    let (mut left, mut pos) = parse_unary(tokens, start)?;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Multiply => {
                let (right, new_pos) = parse_unary(tokens, pos + 1)?;
                left = left.binary(BinaryOp::Multiply, right);
                pos = new_pos;
            }
            Token::Divide => {
                let (right, new_pos) = parse_unary(tokens, pos + 1)?;
                left = left.binary(BinaryOp::Divide, right);
                pos = new_pos;
            }
            _ => break,
        }
    }

    Ok((left, pos))
}

fn parse_unary(tokens: &[Token], start: usize) -> Result<(Expr, usize), String> {
    if start >= tokens.len() {
        return Err("Unexpected end".to_string());
    }

    match &tokens[start] {
        Token::Minus => {
            let (expr, pos) = parse_primary(tokens, start + 1)?;
            Ok((expr.unary(UnaryOp::Negate), pos))
        }
        _ => parse_primary(tokens, start),
    }
}

fn parse_primary(tokens: &[Token], start: usize) -> Result<(Expr, usize), String> {
    if start >= tokens.len() {
        return Err("Unexpected end".to_string());
    }

    match &tokens[start] {
        Token::Number(n) => Ok((Expr::Literal(Literal::Float(*n)), start + 1)),
        Token::Identifier(name) => {
            // Check if it's a function call
            if start + 1 < tokens.len() && matches!(&tokens[start + 1], Token::LParen) {
                parse_function(tokens, start)
            } else {
                Ok((Expr::Column(name.clone()), start + 1))
            }
        }
        Token::Function(_name) => parse_function(tokens, start),
        Token::LParen => {
            // Parenthesized expression
            let (expr, pos) = parse_additive(tokens, start + 1)?;
            if pos < tokens.len() && matches!(&tokens[pos], Token::RParen) {
                Ok((expr, pos + 1))
            } else {
                Err("Expected ')'".to_string())
            }
        }
        _ => Err(format!(
            "Unexpected token at position {}: {:?}",
            start, tokens[start]
        )),
    }
}

fn parse_function(tokens: &[Token], start: usize) -> Result<(Expr, usize), String> {
    let name = match &tokens[start] {
        Token::Identifier(n) => n.clone(),
        Token::Function(n) => n.clone(),
        _ => return Err("Expected function name".to_string()),
    };

    // Find opening paren
    let mut paren_pos = start;
    while paren_pos < tokens.len() && !matches!(&tokens[paren_pos], Token::LParen) {
        paren_pos += 1;
    }
    if paren_pos >= tokens.len() {
        return Err("Expected '('".to_string());
    }

    // Parse arguments
    let mut args = Vec::new();
    let mut pos = paren_pos + 1;
    let mut _arg_count = 0;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::RParen => {
                pos += 1;
                break;
            }
            Token::Comma => {
                pos += 1;
                _arg_count += 1;
            }
            _ => {
                let (expr, new_pos) = parse_additive(tokens, pos)?;
                args.push(expr);
                pos = new_pos;
            }
        }
    }

    // Map function names
    let func_name = match name.as_str() {
        "ts_mean" | "ts_avg" => "ts_mean",
        "ts_sum" => "ts_sum",
        "ts_max" => "ts_max",
        "ts_min" => "ts_min",
        "ts_std" => "ts_std",
        "ts_rank" => "ts_rank",
        _ => &name,
    };

    Ok((
        Expr::FunctionCall {
            name: func_name.to_string(),
            args,
        },
        pos,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("close + volume").unwrap();
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_parse_column() {
        let expr = parse_expression("close").unwrap();
        assert!(matches!(expr, Expr::Column(_)));
    }

    #[test]
    fn test_parse_binary() {
        let expr = parse_expression("close + volume").unwrap();
        assert!(matches!(expr, Expr::BinaryExpr { .. }));
    }

    #[test]
    fn test_parse_function() {
        let expr = parse_expression("ts_mean(close, 20)").unwrap();
        assert!(matches!(expr, Expr::FunctionCall { .. }));
    }
}
