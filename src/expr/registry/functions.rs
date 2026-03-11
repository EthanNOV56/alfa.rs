//! Helper functions for factor expression evaluation

use crate::expr::ast::{BinaryOp, Expr, Literal, UnaryOp};
use crate::lazy::{DataSource, LogicalPlan};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

pub use crate::expr::registry::timeseries::{
    decay_linear, delay, highday, lowday, rank, scale, sign, sma, ts_argmax, ts_argmin, ts_correlation,
    ts_count, ts_cov, ts_delta, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_std, ts_sum, wma,
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

/// Convert expression to logical plan
pub fn expr_to_logical_plan(expr: &Expr, output_name: &str) -> Result<LogicalPlan, String> {
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
        "ge" | "greater_equal" | "gte" => Ok(arg_values[0]
            .iter()
            .zip(arg_values[1].iter())
            .map(|(&x, &y)| if x >= y { 1.0 } else { 0.0 })
            .collect()),
        "le" | "less_equal" | "lte" => Ok(arg_values[0]
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
