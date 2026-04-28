//! Helper functions for factor expression evaluation

use crate::expr::ast::{BinaryOp, Expr, Frequency, Literal, UnaryOp};
use ahash::AHashMap;
use ndarray::Array1;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

pub use crate::expr::registry::timeseries::{
    cap_neu, decay_linear, delay, highday, lowday, quesval, quesval2, rank, scale, sign, sma,
    ts_argmax, ts_argmin, ts_correlation, ts_count, ts_cov, ts_delta, ts_max, ts_mean, ts_min,
    ts_product, ts_quantile, ts_rank, ts_resi, ts_rsquare, ts_slope, ts_std, ts_sum, winsor,
    wma, zscore,
};

/// Extract column names from an expression
pub fn extract_columns(expr: &Expr) -> Vec<String> {
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

/// Compute a hash for an expression to identify unique subexpressions
pub fn expr_hash(expr: &Expr) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_expr(expr, &mut hasher);
    hasher.finish()
}

pub fn hash_expr<H: Hasher>(expr: &Expr, hasher: &mut H) {
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
        Expr::FunctionCall { name, args, freq } => {
            4u8.hash(hasher);
            name.hash(hasher);
            args.len().hash(hasher);
            freq.hash(hasher);
            for arg in args {
                hash_expr(arg, hasher);
            }
        }
        _ => {}
    }
}

/// Collect unique subexpressions from an expression tree
pub fn collect_unique_subexpressions(
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

/// Collect all group-by frequency levels used in an expression tree
/// Walks FunctionCall nodes with `freq: Some(...)` and collects their frequencies
pub fn collect_frequencies(expr: &Expr, freqs: &mut Vec<Frequency>) {
    match expr {
        Expr::FunctionCall { freq, args, .. } => {
            if let Some(f) = freq {
                if !freqs.contains(f) {
                    freqs.push(f.clone());
                }
            }
            for arg in args {
                collect_frequencies(arg, freqs);
            }
        }
        Expr::BinaryExpr { left, right, .. } => {
            collect_frequencies(left, freqs);
            collect_frequencies(right, freqs);
        }
        Expr::UnaryExpr { expr: e, .. } => {
            collect_frequencies(e, freqs);
        }
        _ => {}
    }
}

/// Memoized expression evaluation - computes each subexpression only once
pub fn eval_expr_memoized(
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
        Expr::FunctionCall { name, args, freq: _ } => {
            eval_function_memoized(name, args, data, n_rows, cache)
        }
        _ => Err("Unsupported expr type".to_string()),
    }?;

    // Cache the result
    cache.insert(hash, result.clone());
    Ok(result)
}

/// Memoized function evaluation
pub fn eval_function_memoized(
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
        "cs_rank" => Ok(rank(&arg_values[0])),
        "cs_scale" => Ok(scale(&arg_values[0])),
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
        "winsor" => {
            // winsor(alpha, n_symbols) - clip to [mean - 3*std, mean + 3*std] per date
            let n_symbols = args
                .get(1)
                .and_then(|a| get_literal_int(a))
                .unwrap_or(100); // default 100 symbols
            Ok(winsor(&arg_values[0], n_symbols))
        }
        "zscore" => {
            // zscore(alpha, n_symbols) - (x - mean) / std per date
            let n_symbols = args
                .get(1)
                .and_then(|a| get_literal_int(a))
                .unwrap_or(100); // default 100 symbols
            Ok(zscore(&arg_values[0], n_symbols))
        }
        "cap_neu" => {
            // cap_neu(alpha, market_cap, n_symbols) - regress on log(market_cap), return standardized residuals
            if args.len() < 2 {
                return Err("cap_neu requires 2 arguments: alpha and market_cap".to_string());
            }
            let n_symbols = args
                .get(2)
                .and_then(|a| get_literal_int(a))
                .unwrap_or(100); // default 100 symbols
            Ok(cap_neu(&arg_values[0], &arg_values[1], n_symbols))
        }
        "quesval" => {
            if args.len() != 4 {
                return Err("quesval requires 4 arguments".to_string());
            }
            let threshold = get_literal_float(&args[0]).unwrap_or(0.0);
            let result: Vec<f64> = arg_values[1]
                .iter()
                .zip(arg_values[2].iter())
                .zip(arg_values[3].iter())
                .map(|((&a, &b), &c)| if a > threshold { b } else { c })
                .collect();
            Ok(result)
        }
        "quesval2" => {
            // quesval2(a, b, c, d): if a > b, c else d
            if args.len() != 4 {
                return Err("quesval2 requires 4 arguments".to_string());
            }
            let result: Vec<f64> = arg_values[0]
                .iter()
                .zip(arg_values[1].iter())
                .zip(arg_values[2].iter())
                .zip(arg_values[3].iter())
                .map(|(((&a, &b), &c), &d)| if a > b { c } else { d })
                .collect();
            Ok(result)
        }
        "returns" => {
            let delayed = delay(&arg_values[0], 1);
            Ok(arg_values[0]
                .iter()
                .zip(delayed.iter())
                .map(|(&c, &d)| if d != 0.0 { c / d - 1.0 } else { 0.0 })
                .collect())
        }
        _ => Err(format!("Unknown function: {}", name)),
    }
}

/// Memoized time-series function evaluation
pub fn eval_ts_function_memoized(
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
        "ts_covariance" => {
            let vals2 = eval_expr_memoized(&args[1], data, n_rows, cache)?;
            Ok(ts_cov(&vals, &vals2, window))
        }
        "ts_cov" => {
            let vals2 = eval_expr_memoized(&args[1], data, n_rows, cache)?;
            Ok(ts_cov(&vals, &vals2, window))
        }
        "ts_delay" => Ok(delay(&vals, window)),
        "ts_decay_linear" => Ok(decay_linear(&vals, window)),
        "ts_sma" => {
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
        "ts_quantile" => {
            let q = args
                .get(2)
                .and_then(|a| get_literal_float(a))
                .unwrap_or(0.5);
            Ok(ts_quantile(&vals, window, q))
        }
        "ts_slope" => Ok(ts_slope(&vals, window)),
        "ts_rsquare" => Ok(ts_rsquare(&vals, window)),
        "ts_resi" => Ok(ts_resi(&vals, window)),
        "ts_lowday" => Ok(lowday(&vals, window)),
        "ts_highday" => Ok(highday(&vals, window)),
        "ts_wma" => Ok(wma(&vals, window)),
        // Backward-compat bare names
        "sma" => {
            let n = window;
            let m = args.get(2).and_then(|a| get_literal_int(a)).unwrap_or(2);
            let alpha = m as f64 / n as f64;
            let alpha = if alpha > 0.0 && alpha <= 1.0 { alpha } else { 0.5 };
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
pub fn compute_exprs_parallel(
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
pub fn eval_expr_simple(
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

/// Extract integer literal from expression
pub fn get_literal_int(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::Literal(Literal::Integer(i)) => Some(*i as usize),
        Expr::Literal(Literal::Float(f)) => Some(*f as usize),
        _ => None,
    }
}

/// Extract float literal from expression
pub fn get_literal_float(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Literal(Literal::Float(f)) => Some(*f),
        Expr::Literal(Literal::Integer(i)) => Some(*i as f64),
        _ => None,
    }
}

// ============================================================================
// Vectorized Evaluation Functions (SIMD-optimized)
// ============================================================================

/// Vectorized memoized expression evaluation using Array1<f64> for SIMD
pub fn eval_expr_vectorized(
    expr: &Expr,
    data: &HashMap<String, Array1<f64>>,
    cache: &mut HashMap<u64, Array1<f64>>,
) -> Result<Array1<f64>, String> {
    eval_expr_impl(expr, data, cache, true, &mut None)
}

/// Compact evaluation: returns per-group values + group keys (date_int, symbol_int)
/// Used for frequency-grouped expressions where expand=false avoids OOM.
pub fn eval_expr_compact(
    expr: &Expr,
    data: &HashMap<String, Array1<f64>>,
    cache: &mut HashMap<u64, Array1<f64>>,
) -> Result<(Array1<f64>, Vec<(i64, i64)>), String> {
    let mut group_keys: Option<Vec<(i64, i64)>> = None;
    let values = eval_expr_impl(expr, data, cache, false, &mut group_keys)?;
    Ok((values, group_keys.unwrap_or_default()))
}

fn eval_expr_impl(
    expr: &Expr,
    data: &HashMap<String, Array1<f64>>,
    cache: &mut HashMap<u64, Array1<f64>>,
    expand: bool,
    group_keys: &mut Option<Vec<(i64, i64)>>,
) -> Result<Array1<f64>, String> {
    let hash = expr_hash(expr);

    // Check cache first - clone to avoid returning reference to cached value
    if let Some(cached) = cache.get(&hash) {
        return Ok(cached.clone());
    }

    // Compute the expression
    let result = match expr {
        Expr::Column(name) => data
            .get(name)
            .cloned()
            .ok_or_else(|| format!("Column '{}' not found", name))?,
        Expr::Literal(lit) => {
            let n_rows = data.values().next().map(|arr| arr.len()).unwrap_or(0);
            let val = match lit {
                Literal::Float(f) => *f,
                Literal::Integer(i) => *i as f64,
                _ => 0.0,
            };
            Array1::from_elem(n_rows, val)
        }
        Expr::BinaryExpr { left, op, right } => {
            // Special case: both sides are group aggregations with same frequency
            // Compute both first, then apply the binary operation
            let left_is_group = match left.as_ref() {
                Expr::FunctionCall { name, args, freq: Some(f) } => Some((name.clone(), args.as_slice(), f.clone())),
                _ => None,
            };
            let right_is_group = match right.as_ref() {
                Expr::FunctionCall { name, args, freq: Some(f) } => Some((name.clone(), args.as_slice(), f.clone())),
                _ => None,
            };

            if let (Some((left_name, left_args, left_freq)), Some((right_name, right_args, right_freq))) =
                (left_is_group.clone(), right_is_group.clone())
            {
                if left_freq == right_freq {
                    // Both are group aggregations with same frequency.
                    // When expand=false, both produce compact results with matching group keys.
                    let left_result = eval_group_function_vectorized(
                        &left_name,
                        left_args,
                        data,
                        cache,
                        left_freq,
                        expand,
                        group_keys,
                    )?;
                    let right_result = eval_group_function_vectorized(
                        &right_name,
                        right_args,
                        data,
                        cache,
                        right_freq,
                        expand,
                        group_keys,
                    )?;

                    // Apply the binary operation element-wise
                    match op {
                        BinaryOp::Divide => {
                            let mut result = left_result.clone();
                            for i in 0..result.len() {
                                if right_result[i].abs() < 1e-10 {
                                    result[i] = f64::NAN;
                                } else {
                                    result[i] = left_result[i] / right_result[i];
                                }
                            }
                            return Ok(result);
                        }
                        BinaryOp::Multiply => {
                            return Ok(&left_result * &right_result);
                        }
                        BinaryOp::Add => {
                            return Ok(&left_result + &right_result);
                        }
                        BinaryOp::Subtract => {
                            return Ok(&left_result - &right_result);
                        }
                        _ => {}
                    }
                }
            }

            // Fallback: evaluate normally
            let left_vals = eval_expr_impl(left, data, cache, expand, group_keys)?;
            let right_vals = eval_expr_impl(right, data, cache, expand, group_keys)?;
            // Use ndarray's vectorized operations for SIMD
            match op {
                BinaryOp::Add => &left_vals + &right_vals,
                BinaryOp::Subtract => &left_vals - &right_vals,
                BinaryOp::Multiply => &left_vals * &right_vals,
                BinaryOp::Divide => {
                    // Handle division by zero - use element-wise safe division
                    let mut result = left_vals.clone();
                    for i in 0..result.len() {
                        if right_vals[i].abs() < 1e-10 {
                            result[i] = 0.0;
                        } else {
                            result[i] = left_vals[i] / right_vals[i];
                        }
                    }
                    result
                }
                _ => Array1::from_elem(left_vals.len(), 0.0),
            }
        }
        Expr::UnaryExpr { op, expr: e } => {
            let vals = eval_expr_impl(e, data, cache, expand, group_keys)?;
            match op {
                UnaryOp::Negate => -&vals,
                _ => vals,
            }
        }
        Expr::FunctionCall { name, args, freq } => {
            if let Some(group_freq) = freq {
                eval_group_function_vectorized(name, args, data, cache, group_freq.clone(), expand, group_keys)?
            } else {
                eval_function_vectorized(name, args, data, cache)?
            }
        }
        _ => {
            let n_rows = data.values().next().map(|arr| arr.len()).unwrap_or(0);
            Array1::from_elem(n_rows, 0.0)
        }
    };

    // Cache the result only if it's not too large (avoid OOM)
    // Group aggregation results are n_rows which can be millions of elements
    const CACHE_SIZE_LIMIT: usize = 100_000; // ~800KB for f64
    if result.len() <= CACHE_SIZE_LIMIT {
        cache.insert(hash, result.clone());
    }
    Ok(result)
}

/// Vectorized function evaluation using Array1<f64>
pub fn eval_function_vectorized(
    name: &str,
    args: &[Expr],
    data: &HashMap<String, Array1<f64>>,
    cache: &mut HashMap<u64, Array1<f64>>,
) -> Result<Array1<f64>, String> {
    let name_lower = name.to_lowercase();

    // Check for ts_ prefix OR known ts functions without prefix
    if name_lower.starts_with("ts_")
        || matches!(
            name_lower.as_str(),
            "sma" | "lowday" | "highday" | "wma" | "min" | "max" | "sum"
        )
    {
        return eval_ts_function_vectorized(&name_lower, args, data, cache);
    }

    // First evaluate all arguments (with memoization)
    let mut arg_values: Vec<Array1<f64>> = Vec::new();
    for arg in args {
        arg_values.push(eval_expr_vectorized(arg, data, cache)?);
    }

    match name_lower.as_str() {
        "cs_rank" => Ok(rank(arg_values[0].as_slice().unwrap()).into()),
        "cs_scale" => Ok(scale(arg_values[0].as_slice().unwrap()).into()),
        "sign" => Ok(sign(arg_values[0].as_slice().unwrap()).into()),
        "abs" => Ok(arg_values[0].mapv(f64::abs)),
        // Element-wise min of two series
        "min" => {
            if args.len() >= 2 {
                let mut result = arg_values[0].clone();
                for i in 0..result.len() {
                    result[i] = result[i].min(arg_values[1][i]);
                }
                Ok(result)
            } else {
                Ok(arg_values[0].clone())
            }
        }
        // Element-wise max of two series
        "max" => {
            if args.len() >= 2 {
                let mut result = arg_values[0].clone();
                for i in 0..result.len() {
                    result[i] = result[i].max(arg_values[1][i]);
                }
                Ok(result)
            } else {
                Ok(arg_values[0].clone())
            }
        }
        // Sum of all elements in a series
        "sum" => {
            let total: f64 = arg_values[0].iter().sum();
            let n = arg_values[0].len();
            Ok(Array1::from_elem(n, total))
        }
        "log" => Ok(arg_values[0].mapv(|v| if v > 0.0 { v.ln() } else { f64::NAN })),
        "log10" => Ok(arg_values[0].mapv(|v| if v > 0.0 { v.log10() } else { f64::NAN })),
        "sqrt" => Ok(arg_values[0].mapv(f64::sqrt)),
        "power" => {
            let exponent = get_literal_int(&args[1]).map(|e| e as f64).unwrap_or(2.0);
            Ok(arg_values[0].mapv(|v| v.powf(exponent)))
        }
        "delta" => {
            let periods = get_literal_int(&args[1]).unwrap_or(1);
            Ok(ts_delta(arg_values[0].as_slice().unwrap(), periods).into())
        }
        "if" => {
            if args.len() != 3 {
                return Err("IF requires 3 arguments".to_string());
            }
            let mut result = arg_values[0].clone();
            for i in 0..result.len() {
                result[i] = if arg_values[0][i] > 0.0 {
                    arg_values[1][i]
                } else {
                    arg_values[2][i]
                };
            }
            Ok(result)
        }
        "gt" | "greater" => {
            let mut result = arg_values[0].clone();
            for i in 0..result.len() {
                result[i] = if arg_values[0][i] > arg_values[1][i] {
                    1.0
                } else {
                    0.0
                };
            }
            Ok(result)
        }
        "lt" | "less" => {
            let mut result = arg_values[0].clone();
            for i in 0..result.len() {
                result[i] = if arg_values[0][i] < arg_values[1][i] {
                    1.0
                } else {
                    0.0
                };
            }
            Ok(result)
        }
        "ge" | "greater_equal" | "gte" => {
            let mut result = arg_values[0].clone();
            for i in 0..result.len() {
                result[i] = if arg_values[0][i] >= arg_values[1][i] {
                    1.0
                } else {
                    0.0
                };
            }
            Ok(result)
        }
        "le" | "less_equal" | "lte" => {
            let mut result = arg_values[0].clone();
            for i in 0..result.len() {
                result[i] = if arg_values[0][i] <= arg_values[1][i] {
                    1.0
                } else {
                    0.0
                };
            }
            Ok(result)
        }
        "eq" | "equal" => {
            let mut result = arg_values[0].clone();
            for i in 0..result.len() {
                result[i] = if (arg_values[0][i] - arg_values[1][i]).abs() < 1e-10 {
                    1.0
                } else {
                    0.0
                };
            }
            Ok(result)
        }
        "ne" | "not_equal" => {
            let mut result = arg_values[0].clone();
            for i in 0..result.len() {
                result[i] = if (arg_values[0][i] - arg_values[1][i]).abs() >= 1e-10 {
                    1.0
                } else {
                    0.0
                };
            }
            Ok(result)
        }
        "winsor" => {
            // winsor(alpha, n_symbols) - clip to [mean - 3*std, mean + 3*std] per date
            let n_symbols = args
                .get(1)
                .and_then(|a| get_literal_int(a))
                .unwrap_or(100); // default 100 symbols
            let result = winsor(arg_values[0].as_slice().unwrap(), n_symbols);
            Ok(result.into())
        }
        "zscore" => {
            // zscore(alpha, n_symbols) - (x - mean) / std per date
            let n_symbols = args
                .get(1)
                .and_then(|a| get_literal_int(a))
                .unwrap_or(100); // default 100 symbols
            let result = zscore(arg_values[0].as_slice().unwrap(), n_symbols);
            Ok(result.into())
        }
        "cap_neu" => {
            // cap_neu(alpha, market_cap, n_symbols) - regress on log(market_cap), return standardized residuals
            if args.len() < 2 {
                return Err("cap_neu requires 2 arguments: alpha and market_cap".to_string());
            }
            let market_cap = eval_expr_vectorized(&args[1], data, cache)?;
            let n_symbols = args
                .get(2)
                .and_then(|a| get_literal_int(a))
                .unwrap_or(100); // default 100 symbols
            let result = cap_neu(
                arg_values[0].as_slice().unwrap(),
                market_cap.as_slice().unwrap(),
                n_symbols,
            );
            Ok(result.into())
        }
        "quesval" => {
            if args.len() != 4 {
                return Err("quesval requires 4 arguments".to_string());
            }
            let threshold = get_literal_float(&args[0]).unwrap_or(0.0);
            let mut result = arg_values[1].clone();
            for i in 0..result.len() {
                result[i] = if arg_values[1][i] > threshold {
                    arg_values[2][i]
                } else {
                    arg_values[3][i]
                };
            }
            Ok(result)
        }
        "quesval2" => {
            if args.len() != 4 {
                return Err("quesval2 requires 4 arguments".to_string());
            }
            let mut result = arg_values[0].clone();
            for i in 0..result.len() {
                result[i] = if arg_values[0][i] > arg_values[1][i] {
                    arg_values[2][i]
                } else {
                    arg_values[3][i]
                };
            }
            Ok(result)
        }
        "returns" => {
            let delayed = delay(arg_values[0].as_slice().unwrap(), 1);
            let mut result = arg_values[0].clone();
            for i in 0..result.len() {
                if delayed[i] != 0.0 {
                    result[i] = result[i] / delayed[i] - 1.0;
                } else {
                    result[i] = 0.0;
                }
            }
            Ok(result)
        }
        _ => Err(format!("Unknown function: {}", name)),
    }
}

/// Vectorized group-by aggregation function evaluation
/// When freq is Some, this function performs group-by aggregation instead of rolling window
///
/// Memory optimization: Uses HashMap with pre-reserved capacity to reduce reallocations.
/// Expands to n_rows only when expand=true. Intermediate structures are dropped
/// as soon as possible to free memory.
///
/// When expand=false, populates `group_keys` with sorted (date_int, symbol_int) pairs.
pub fn eval_group_function_vectorized(
    name: &str,
    args: &[Expr],
    data: &HashMap<String, Array1<f64>>,
    cache: &mut HashMap<u64, Array1<f64>>,
    group_freq: Frequency,
    expand: bool,
    group_keys: &mut Option<Vec<(i64, i64)>>,
) -> Result<Array1<f64>, String> {
    // Determine the column prefix based on frequency.
    // Fall back to finer-frequency columns for alignment: the grouping keys
    // must match the data row count, which is at the finest column frequency.
    let prefix = match group_freq {
        Frequency::Minute5 => "5m",
        Frequency::Minute1 => "1m",
        Frequency::Daily => "1d",
        _ => return Err(format!("Unsupported group frequency: {:?}", group_freq)),
    };

    let trading_date_col = format!("{}:trading_date", prefix);
    let symbol_col = format!("{}:symbol", prefix);

    // Try the group-frequency column first; if missing, fall back to finer frequencies
    // (e.g. Daily grouping over 5m data needs 5m:trading_date for row alignment)
    let trading_dates = data
        .get(&trading_date_col)
        .or_else(|| data.get("5m:trading_date"))
        .or_else(|| data.get("1m:trading_date"))
        .ok_or_else(|| format!("Column '{}' (or finer) not found for grouping", trading_date_col))?;
    let symbols = data
        .get(&symbol_col)
        .or_else(|| data.get("5m:symbol"))
        .or_else(|| data.get("1m:symbol"))
        .ok_or_else(|| format!("Column '{}' (or finer) not found for grouping", symbol_col))?;

    let values = eval_expr_vectorized(&args[0], data, cache)?;
    let n_rows = values.len();
    if n_rows == 0 {
        return Ok(Array1::from_elem(0, f64::NAN));
    }

    let name_lower = name.to_lowercase();
    // Strip ts_ prefix if present (parser aliases sum→ts_sum, mean→ts_mean, etc.)
    let name_key = name_lower.strip_prefix("ts_").unwrap_or(&name_lower);

    // Pre-estimate number of groups (assuming ~1 row per group as upper bound)
    // Will shrink_to_fit after aggregation
    let estimated_groups = n_rows.min(10_000_000);

    // Threshold for parallel processing
    let parallel_threshold = 500_000;

    match name_key {
        "sum" | "add" => {
            let aggregation: AHashMap<(i64, i64), f64> = if n_rows > parallel_threshold {
                // Parallel chunk-based aggregation
                (0..n_rows)
                    .into_par_iter()
                    .filter(|&i| values[i].is_finite())
                    .fold(
                        || AHashMap::with_capacity(estimated_groups / rayon::current_num_threads()),
                        |mut acc, i| {
                            let key = (trading_dates[i] as i64, symbols[i] as i64);
                            *acc.entry(key).or_insert(0.0) += values[i];
                            acc
                        },
                    )
                    .reduce(
                        || AHashMap::new(),
                        |mut acc, local| {
                            for (k, v) in local {
                                *acc.entry(k).or_insert(0.0) += v;
                            }
                            acc
                        },
                    )
            } else {
                let mut agg = AHashMap::with_capacity(estimated_groups);
                for i in 0..n_rows {
                    let v = values[i];
                    if v.is_finite() {
                        *agg.entry((trading_dates[i] as i64, symbols[i] as i64)).or_insert(0.0) += v;
                    }
                }
                agg
            };

            let mut unique_groups: Vec<(i64, i64)> = aggregation.keys().copied().collect();
            unique_groups.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            let n_groups = unique_groups.len();

            let group_index: AHashMap<(i64, i64), usize> = unique_groups.iter()
                .enumerate()
                .map(|(i, g)| (*g, i))
                .collect();
            let result: Vec<f64> = unique_groups.iter()
                .map(|g| *aggregation.get(g).unwrap_or(&0.0))
                .collect();

            drop(aggregation);

            if expand {
                let mut expanded_result: Vec<f64> = Vec::with_capacity(n_rows);
                for i in 0..n_rows {
                    let key = (trading_dates[i] as i64, symbols[i] as i64);
                    expanded_result.push(
                        group_index
                            .get(&key)
                            .map(|&idx| result[idx])
                            .unwrap_or(f64::NAN),
                    );
                }
                Ok(Array1::from_vec(expanded_result))
            } else {
                if group_keys.is_none() {
                    *group_keys = Some(unique_groups);
                }
                Ok(Array1::from_vec(result))
            }
        }
        "mean" | "avg" | "average" => {
            let (agg_sum, agg_count): (AHashMap<(i64, i64), f64>, AHashMap<(i64, i64), usize>) =
                if n_rows > parallel_threshold {
                    (0..n_rows)
                        .into_par_iter()
                        .filter(|&i| values[i].is_finite())
                        .fold(
                            || (AHashMap::new(), AHashMap::new()),
                            |(mut s, mut c), i| {
                                let key = (trading_dates[i] as i64, symbols[i] as i64);
                                *s.entry(key).or_insert(0.0) += values[i];
                                *c.entry(key).or_insert(0) += 1;
                                (s, c)
                            },
                        )
                        .reduce(
                            || (AHashMap::new(), AHashMap::new()),
                            |(mut s1, mut c1), (s2, c2)| {
                                for (k, v) in s2 {
                                    *s1.entry(k).or_insert(0.0) += v;
                                }
                                for (k, v) in c2 {
                                    *c1.entry(k).or_insert(0) += v;
                                }
                                (s1, c1)
                            },
                        )
                } else {
                    let mut s = AHashMap::with_capacity(estimated_groups);
                    let mut c = AHashMap::with_capacity(estimated_groups);
                    for i in 0..n_rows {
                        let v = values[i];
                        if v.is_finite() {
                            let key = (trading_dates[i] as i64, symbols[i] as i64);
                            *s.entry(key).or_insert(0.0) += v;
                            *c.entry(key).or_insert(0) += 1;
                        }
                    }
                    (s, c)
                };

            let mut unique_groups: Vec<(i64, i64)> = agg_sum.keys().copied().collect();
            unique_groups.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

            let means: Vec<f64> = unique_groups.iter().map(|g| {
                let sum = *agg_sum.get(g).unwrap_or(&0.0);
                let count = *agg_count.get(g).unwrap_or(&0);
                if count > 0 { sum / count as f64 } else { f64::NAN }
            }).collect();

            drop(agg_sum);
            drop(agg_count);

            if expand {
                let group_index: AHashMap<(i64, i64), usize> = unique_groups.iter()
                    .enumerate()
                    .map(|(i, g)| (*g, i))
                    .collect();

                let expanded_result: Vec<f64> = (0..n_rows)
                    .map(|i| {
                        let key = (trading_dates[i] as i64, symbols[i] as i64);
                        group_index.get(&key).map(|&idx| means[idx]).unwrap_or(f64::NAN)
                    })
                    .collect();
                Ok(Array1::from_vec(expanded_result))
            } else {
                if group_keys.is_none() {
                    *group_keys = Some(unique_groups);
                }
                Ok(Array1::from_vec(means))
            }
        }
        "count" | "cnt" => {
            let agg_count: AHashMap<(i64, i64), usize> = if n_rows > parallel_threshold {
                (0..n_rows)
                    .into_par_iter()
                    .fold(
                        || AHashMap::new(),
                        |mut acc, i| {
                            let key = (trading_dates[i] as i64, symbols[i] as i64);
                            *acc.entry(key).or_insert(0) += 1;
                            acc
                        },
                    )
                    .reduce(
                        || AHashMap::new(),
                        |mut acc, local| {
                            for (k, v) in local {
                                *acc.entry(k).or_insert(0) += v;
                            }
                            acc
                        },
                    )
            } else {
                let mut agg = AHashMap::with_capacity(estimated_groups);
                for i in 0..n_rows {
                    *agg.entry((trading_dates[i] as i64, symbols[i] as i64)).or_insert(0) += 1;
                }
                agg
            };

            let mut unique_groups: Vec<(i64, i64)> = agg_count.keys().copied().collect();
            unique_groups.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

            let counts: Vec<f64> = unique_groups.iter()
                .map(|g| *agg_count.get(g).unwrap_or(&0) as f64)
                .collect();
            drop(agg_count);

            if expand {
                let group_index: HashMap<(i64, i64), usize> = unique_groups.iter()
                    .enumerate()
                    .map(|(i, g)| (*g, i))
                    .collect();

                let mut expanded_result: Vec<f64> = Vec::with_capacity(n_rows);
                for i in 0..n_rows {
                    let key = (trading_dates[i] as i64, symbols[i] as i64);
                    expanded_result.push(group_index.get(&key).map(|idx| counts[*idx]).unwrap_or(f64::NAN));
                }
                Ok(Array1::from_vec(expanded_result))
            } else {
                if group_keys.is_none() {
                    *group_keys = Some(unique_groups);
                }
                Ok(Array1::from_vec(counts))
            }
        }
        "min" => {
            let mut agg_min: HashMap<(i64, i64), f64> = HashMap::with_capacity(estimated_groups);
            for i in 0..n_rows {
                let date_int = trading_dates[i] as i64;
                let symbol_int = symbols[i] as i64;
                let v = values[i];
                if v.is_finite() {
                    agg_min.entry((date_int, symbol_int)).or_insert(f64::INFINITY);
                    if v < *agg_min.get(&(date_int, symbol_int)).unwrap() {
                        *agg_min.get_mut(&(date_int, symbol_int)).unwrap() = v;
                    }
                }
            }

            let mut unique_groups: Vec<(i64, i64)> = agg_min.keys().cloned().collect();
            unique_groups.sort_by(|a, b| {
                let date_cmp = a.0.cmp(&b.0);
                if date_cmp != std::cmp::Ordering::Equal {
                    date_cmp
                } else {
                    a.1.cmp(&b.1)
                }
            });

            let mut mins: Vec<f64> = Vec::with_capacity(unique_groups.len());
            for g in &unique_groups {
                mins.push(*agg_min.get(g).unwrap_or(&f64::NAN));
            }
            drop(agg_min);

            if expand {
                let group_index: HashMap<(i64, i64), usize> = unique_groups.iter()
                    .enumerate()
                    .map(|(i, g)| (*g, i))
                    .collect();

                let mut expanded_result: Vec<f64> = Vec::with_capacity(n_rows);
                for i in 0..n_rows {
                    let key = (trading_dates[i] as i64, symbols[i] as i64);
                    expanded_result.push(group_index.get(&key).map(|idx| mins[*idx]).unwrap_or(f64::NAN));
                }
                Ok(Array1::from_vec(expanded_result))
            } else {
                if group_keys.is_none() {
                    *group_keys = Some(unique_groups);
                }
                Ok(Array1::from_vec(mins))
            }
        }
        "max" => {
            let mut agg_max: HashMap<(i64, i64), f64> = HashMap::with_capacity(estimated_groups);
            for i in 0..n_rows {
                let date_int = trading_dates[i] as i64;
                let symbol_int = symbols[i] as i64;
                let v = values[i];
                if v.is_finite() {
                    agg_max.entry((date_int, symbol_int)).or_insert(f64::NEG_INFINITY);
                    if v > *agg_max.get(&(date_int, symbol_int)).unwrap() {
                        *agg_max.get_mut(&(date_int, symbol_int)).unwrap() = v;
                    }
                }
            }

            let mut unique_groups: Vec<(i64, i64)> = agg_max.keys().cloned().collect();
            unique_groups.sort_by(|a, b| {
                let date_cmp = a.0.cmp(&b.0);
                if date_cmp != std::cmp::Ordering::Equal {
                    date_cmp
                } else {
                    a.1.cmp(&b.1)
                }
            });

            let mut maxs: Vec<f64> = Vec::with_capacity(unique_groups.len());
            for g in &unique_groups {
                maxs.push(*agg_max.get(g).unwrap_or(&f64::NAN));
            }
            drop(agg_max);

            if expand {
                let group_index: HashMap<(i64, i64), usize> = unique_groups.iter()
                    .enumerate()
                    .map(|(i, g)| (*g, i))
                    .collect();

                let mut expanded_result: Vec<f64> = Vec::with_capacity(n_rows);
                for i in 0..n_rows {
                    let key = (trading_dates[i] as i64, symbols[i] as i64);
                    expanded_result.push(group_index.get(&key).map(|idx| maxs[*idx]).unwrap_or(f64::NAN));
                }
                Ok(Array1::from_vec(expanded_result))
            } else {
                if group_keys.is_none() {
                    *group_keys = Some(unique_groups);
                }
                Ok(Array1::from_vec(maxs))
            }
        }
        "std" | "stdev" => {
            // Two-pass: first compute means, then variance
            let mut agg_sum: HashMap<(i64, i64), f64> = HashMap::with_capacity(estimated_groups);
            let mut agg_count: HashMap<(i64, i64), usize> = HashMap::with_capacity(estimated_groups);

            for i in 0..n_rows {
                let date_int = trading_dates[i] as i64;
                let symbol_int = symbols[i] as i64;
                let v = values[i];
                if v.is_finite() {
                    *agg_sum.entry((date_int, symbol_int)).or_insert(0.0) += v;
                    *agg_count.entry((date_int, symbol_int)).or_insert(0) += 1;
                }
            }

            let mut unique_groups: Vec<(i64, i64)> = agg_sum.keys().cloned().collect();
            unique_groups.sort_by(|a, b| {
                let date_cmp = a.0.cmp(&b.0);
                if date_cmp != std::cmp::Ordering::Equal {
                    date_cmp
                } else {
                    a.1.cmp(&b.1)
                }
            });

            // Compute means
            let means: Vec<f64> = unique_groups.iter()
                .map(|g| {
                    let sum = *agg_sum.get(g).unwrap_or(&0.0);
                    let count = *agg_count.get(g).unwrap_or(&0);
                    if count > 0 { sum / count as f64 } else { f64::NAN }
                })
                .collect();

            // Second pass: compute variance
            let mut agg_var: HashMap<(i64, i64), (f64, usize)> = HashMap::with_capacity(estimated_groups);
            for i in 0..n_rows {
                let date_int = trading_dates[i] as i64;
                let symbol_int = symbols[i] as i64;
                let v = values[i];
                if v.is_finite() {
                    let key = (date_int, symbol_int);
                    let mean_val = means[unique_groups.iter().position(|g| *g == key).unwrap_or(0)];
                    let entry = agg_var.entry(key).or_insert((0.0, 0));
                    let diff = v - mean_val;
                    entry.0 += diff * diff;
                    entry.1 += 1;
                }
            }

            let mut stds: Vec<f64> = Vec::with_capacity(unique_groups.len());
            for g in &unique_groups {
                let &(var_sum, count) = agg_var.get(g).unwrap_or(&(0.0, 0));
                stds.push(if count > 1 { (var_sum / (count - 1) as f64).sqrt() } else { f64::NAN });
            }
            drop(agg_sum);
            drop(agg_count);
            drop(agg_var);

            if expand {
                let group_index: HashMap<(i64, i64), usize> = unique_groups.iter()
                    .enumerate()
                    .map(|(i, g)| (*g, i))
                    .collect();

                let mut expanded_result: Vec<f64> = Vec::with_capacity(n_rows);
                for i in 0..n_rows {
                    let key = (trading_dates[i] as i64, symbols[i] as i64);
                    expanded_result.push(group_index.get(&key).map(|idx| stds[*idx]).unwrap_or(f64::NAN));
                }
                Ok(Array1::from_vec(expanded_result))
            } else {
                if group_keys.is_none() {
                    *group_keys = Some(unique_groups);
                }
                Ok(Array1::from_vec(stds))
            }
        }
        _ => {
            // Default to sum
            let mut aggregation: HashMap<(i64, i64), f64> = HashMap::with_capacity(estimated_groups);
            for i in 0..n_rows {
                let date_int = trading_dates[i] as i64;
                let symbol_int = symbols[i] as i64;
                let v = values[i];
                if v.is_finite() {
                    *aggregation.entry((date_int, symbol_int)).or_insert(0.0) += v;
                }
            }

            let mut unique_groups: Vec<(i64, i64)> = aggregation.keys().cloned().collect();
            unique_groups.sort_by(|a, b| {
                let date_cmp = a.0.cmp(&b.0);
                if date_cmp != std::cmp::Ordering::Equal {
                    date_cmp
                } else {
                    a.1.cmp(&b.1)
                }
            });
            let n_groups = unique_groups.len();

            let mut result: Vec<f64> = Vec::with_capacity(n_groups);
            for g in &unique_groups {
                result.push(*aggregation.get(g).unwrap_or(&0.0));
            }

            drop(aggregation);

            if expand {
                let group_index: HashMap<(i64, i64), usize> = unique_groups.iter()
                    .enumerate()
                    .map(|(i, g)| (*g, i))
                    .collect();

                let mut expanded_result: Vec<f64> = Vec::with_capacity(n_rows);
                for i in 0..n_rows {
                    let key = (trading_dates[i] as i64, symbols[i] as i64);
                    expanded_result.push(group_index.get(&key).map(|idx| result[*idx]).unwrap_or(f64::NAN));
                }
                Ok(Array1::from_vec(expanded_result))
            } else {
                if group_keys.is_none() {
                    *group_keys = Some(unique_groups);
                }
                Ok(Array1::from_vec(result))
            }
        }
    }
}

/// Vectorized time-series function evaluation
pub fn eval_ts_function_vectorized(
    name: &str,
    args: &[Expr],
    data: &HashMap<String, Array1<f64>>,
    cache: &mut HashMap<u64, Array1<f64>>,
) -> Result<Array1<f64>, String> {
    // Evaluate the first argument
    let vals = eval_expr_vectorized(&args[0], data, cache)?;
    let vals_slice = vals.as_slice().unwrap();
    let window = args.get(1).and_then(|a| get_literal_int(a)).unwrap_or(20);

    match name {
        "ts_mean" => Ok(ts_mean(vals_slice, window).into()),
        "ts_sum" => Ok(ts_sum(vals_slice, window).into()),
        "ts_count" => Ok(ts_count(vals_slice, window).into()),
        "ts_std" => Ok(ts_std(vals_slice, window).into()),
        "ts_max" => Ok(ts_max(vals_slice, window).into()),
        "ts_min" => Ok(ts_min(vals_slice, window).into()),
        "ts_rank" => Ok(ts_rank(vals_slice, window).into()),
        "ts_argmax" => Ok(ts_argmax(vals_slice, window).into()),
        "ts_argmin" => Ok(ts_argmin(vals_slice, window).into()),
        "ts_delta" => Ok(ts_delta(vals_slice, window).into()),
        "ts_product" => Ok(ts_product(vals_slice, window).into()),
        "ts_correlation" => {
            let vals2 = eval_expr_vectorized(&args[1], data, cache)?;
            Ok(ts_correlation(vals_slice, vals2.as_slice().unwrap(), window).into())
        }
        "ts_covariance" | "ts_cov" => {
            let vals2 = eval_expr_vectorized(&args[1], data, cache)?;
            Ok(ts_cov(vals_slice, vals2.as_slice().unwrap(), window).into())
        }
        "ts_delay" => Ok(delay(vals_slice, window).into()),
        "ts_decay_linear" => Ok(decay_linear(vals_slice, window).into()),
        "ts_sma" => {
            let m = args.get(2).and_then(|a| get_literal_int(a)).unwrap_or(2);
            let alpha = m as f64 / window as f64;
            let alpha = if alpha > 0.0 && alpha <= 1.0 { alpha } else { 0.5 };
            Ok(sma(vals_slice, alpha).into())
        }
        "ts_quantile" => {
            let q = args
                .get(2)
                .and_then(|a| get_literal_float(a))
                .unwrap_or(0.5);
            Ok(ts_quantile(vals_slice, window, q).into())
        }
        "ts_slope" => Ok(ts_slope(vals_slice, window).into()),
        "ts_rsquare" => Ok(ts_rsquare(vals_slice, window).into()),
        "ts_resi" => Ok(ts_resi(vals_slice, window).into()),
        "ts_lowday" => Ok(lowday(vals_slice, window).into()),
        "ts_highday" => Ok(highday(vals_slice, window).into()),
        "ts_wma" => Ok(wma(vals_slice, window).into()),
        // Backward-compat bare names
        "sma" => {
            let m = args.get(2).and_then(|a| get_literal_int(a)).unwrap_or(2);
            let alpha = m as f64 / window as f64;
            let alpha = if alpha > 0.0 && alpha <= 1.0 { alpha } else { 0.5 };
            Ok(sma(vals_slice, alpha).into())
        }
        "lowday" => Ok(lowday(vals_slice, window).into()),
        "highday" => Ok(highday(vals_slice, window).into()),
        "wma" => Ok(wma(vals_slice, window).into()),
        "min" => Ok(ts_min(vals_slice, window).into()),
        "max" => Ok(ts_max(vals_slice, window).into()),
        "sum" => Ok(ts_sum(vals_slice, window).into()),
        _ => Err(format!("Unknown ts function: {}", name)),
    }
}
