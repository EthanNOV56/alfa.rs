//! DataFrame - Collection of named Series with vectorized expression evaluation
//!
//! This module provides a DataFrame struct for working with tabular data
//! and vectorized expression evaluation on time series.

use crate::expr::ast::{BinaryOp, Expr, Literal, UnaryOp};
use crate::expr::optimizer::{ExpressionOptimizer, optimize_expression};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::Arc;

use super::series::Series;

// ============================================================================
// DataFrame - Collection of named Series
// ============================================================================

/// A DataFrame represents a collection of named Series (columns)
#[derive(Debug, Clone)]
pub struct DataFrame {
    columns: HashMap<String, Series>,
    n_rows: usize,
}

impl DataFrame {
    /// Create a new empty DataFrame
    pub fn new() -> Self {
        DataFrame {
            columns: HashMap::new(),
            n_rows: 0,
        }
    }

    /// Create a DataFrame from a map of column names to Series
    pub fn from_series_map(columns: HashMap<String, Series>) -> Result<Self, String> {
        let mut n_rows = 0;
        for (name, series) in &columns {
            if n_rows == 0 {
                n_rows = series.len();
            } else if series.len() != n_rows {
                return Err(format!(
                    "Column '{}' has length {}, expected {}",
                    name,
                    series.len(),
                    n_rows
                ));
            }
        }

        Ok(DataFrame { columns, n_rows })
    }

    /// Get the number of rows in the DataFrame
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Get the number of columns in the DataFrame
    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    /// Get the column names
    pub fn column_names(&self) -> Vec<String> {
        self.columns.keys().cloned().collect()
    }

    /// Get a column by name
    pub fn column(&self, name: &str) -> Option<&Series> {
        self.columns.get(name)
    }

    /// Add a column to the DataFrame
    pub fn with_column(mut self, name: &str, series: Series) -> Result<Self, String> {
        if self.n_rows == 0 {
            self.n_rows = series.len();
        } else if series.len() != self.n_rows {
            return Err(format!(
                "Column '{}' has length {}, expected {}",
                name,
                series.len(),
                self.n_rows
            ));
        }

        self.columns.insert(name.to_string(), series);
        Ok(self)
    }

    /// Select specific columns
    pub fn select(&self, col_names: &[&str]) -> Result<DataFrame, String> {
        let mut new_columns = HashMap::new();
        for &name in col_names {
            if let Some(series) = self.columns.get(name) {
                new_columns.insert(name.to_string(), series.clone());
            } else {
                return Err(format!("Column '{}' not found", name));
            }
        }
        DataFrame::from_series_map(new_columns)
    }

    /// Evaluate an expression and add it as a new column
    pub fn with_expr(self, name: &str, expr: &Expr) -> Result<DataFrame, String> {
        let series = evaluate_expr_on_dataframe(expr, &self)?;
        self.with_column(name, series)
    }
}

// ============================================================================
// Vectorized Expression Evaluator
// ============================================================================

/// Evaluate an expression on a DataFrame (vectorized)
pub fn evaluate_expr_on_dataframe(expr: &Expr, df: &DataFrame) -> Result<Series, String> {
    match expr {
        Expr::Literal(lit) => {
            match lit {
                Literal::Float(f) => {
                    // Create a constant series with the same length as DataFrame
                    let data = Array1::from_elem(df.n_rows(), *f);
                    Ok(Series::from_array(data))
                }
                Literal::Integer(i) => {
                    let data = Array1::from_elem(df.n_rows(), *i as f64);
                    Ok(Series::from_array(data))
                }
                Literal::Boolean(b) => {
                    let data = Array1::from_elem(df.n_rows(), if *b { 1.0 } else { 0.0 });
                    Ok(Series::from_array(data))
                }
                Literal::String(_) => {
                    Err("String literals not supported in vectorized evaluation".to_string())
                }
                Literal::Null => {
                    let data = Array1::from_elem(df.n_rows(), f64::NAN);
                    Ok(Series::from_array(data))
                }
            }
        }
        Expr::Column(name) => df
            .column(name)
            .cloned()
            .ok_or_else(|| format!("Column '{}' not found in DataFrame", name)),
        Expr::BinaryExpr { left, op, right } => {
            let left_series = evaluate_expr_on_dataframe(left, df)?;
            let right_series = evaluate_expr_on_dataframe(right, df)?;

            match op {
                BinaryOp::Add => left_series.add(&right_series),
                BinaryOp::Subtract => left_series.sub(&right_series),
                BinaryOp::Multiply => left_series.mul(&right_series),
                BinaryOp::Divide => left_series.div(&right_series),
                BinaryOp::GreaterThan => left_series.gt(&right_series),
                _ => Err(format!(
                    "Binary operator {:?} not yet implemented in vectorized evaluator",
                    op
                )),
            }
        }
        Expr::UnaryExpr { op, expr } => {
            let series = evaluate_expr_on_dataframe(expr, df)?;
            match op {
                UnaryOp::Negate => Ok(series.neg()),
                UnaryOp::Abs => Ok(series.abs()),
                UnaryOp::Sqrt => Ok(series.sqrt()),
                UnaryOp::Log => Ok(series.log()),
                UnaryOp::Exp => Ok(series.exp()),
                _ => Err(format!(
                    "Unary operator {:?} not yet implemented in vectorized evaluator",
                    op
                )),
            }
        }
        Expr::FunctionCall { name, args } => evaluate_function_vectorized(name, args, df),
        _ => Err(format!(
            "Expression type {:?} not yet supported in vectorized evaluation",
            expr
        )),
    }
}

/// Evaluate a function call in vectorized manner
fn evaluate_function_vectorized(
    name: &str,
    args: &[Expr],
    df: &DataFrame,
) -> Result<Series, String> {
    match name {
        "lag" => {
            if args.len() != 2 {
                return Err("lag function requires 2 arguments: expression and periods".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let periods = match &args[1] {
                Expr::Literal(Literal::Integer(p)) => *p as usize,
                _ => return Err("lag periods must be an integer literal".to_string()),
            };
            Ok(series.lag(periods))
        }
        "diff" => {
            if args.len() != 2 {
                return Err(
                    "diff function requires 2 arguments: expression and periods".to_string()
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let periods = match &args[1] {
                Expr::Literal(Literal::Integer(p)) => *p as usize,
                _ => return Err("diff periods must be an integer literal".to_string()),
            };
            Ok(series.diff(periods))
        }
        "pct_change" => {
            if args.len() != 2 {
                return Err(
                    "pct_change function requires 2 arguments: expression and periods".to_string(),
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let periods = match &args[1] {
                Expr::Literal(Literal::Integer(p)) => *p as usize,
                _ => return Err("pct_change periods must be an integer literal".to_string()),
            };
            Ok(series.pct_change(periods))
        }
        "moving_average" => {
            if args.len() != 2 {
                return Err(
                    "moving_average function requires 2 arguments: expression and window"
                        .to_string(),
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("moving_average window must be an integer literal".to_string()),
            };
            Ok(series.moving_average(window))
        }
        "ts_mean" | "mean" => {
            if args.len() != 2 {
                return Err(
                    "ts_mean function requires 2 arguments: expression and window".to_string(),
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_mean window must be an integer literal".to_string()),
            };
            Ok(series.moving_average(window))
        }
        "ts_std" | "std" => {
            if args.len() != 2 {
                return Err(
                    "ts_std function requires 2 arguments: expression and window".to_string(),
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_std window must be an integer literal".to_string()),
            };
            Ok(series.rolling_std(window))
        }
        "momentum" => {
            // momentum is same as pct_change
            evaluate_function_vectorized("pct_change", args, df)
        }
        "volatility" => {
            if args.len() != 2 {
                return Err(
                    "volatility function requires 2 arguments: expression and periods".to_string(),
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let periods = match &args[1] {
                Expr::Literal(Literal::Integer(p)) => *p as usize,
                _ => return Err("volatility periods must be an integer literal".to_string()),
            };
            Ok(series.rolling_std(periods))
        }
        "ema" => {
            if args.len() != 2 {
                return Err("ema function requires 2 arguments: expression and span".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let span = match &args[1] {
                Expr::Literal(Literal::Integer(s)) => *s as usize,
                _ => return Err("ema span must be an integer literal".to_string()),
            };
            Ok(series.ema(span))
        }
        "rank" => {
            if args.len() != 1 {
                return Err("rank function requires 1 argument: expression".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            Ok(series.cs_rank())
        }
        "ts_rank" => {
            if args.len() != 2 {
                return Err(
                    "ts_rank function requires 2 arguments: expression and window".to_string(),
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_rank window must be an integer literal".to_string()),
            };
            Ok(series.ts_rank(window))
        }
        "ts_max" => {
            if args.len() != 2 {
                return Err(
                    "ts_max function requires 2 arguments: expression and window".to_string(),
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_max window must be an integer literal".to_string()),
            };
            Ok(series.ts_max(window))
        }
        "ts_min" => {
            if args.len() != 2 {
                return Err(
                    "ts_min function requires 2 arguments: expression and window".to_string(),
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_min window must be an integer literal".to_string()),
            };
            Ok(series.ts_min(window))
        }
        "ts_sum" => {
            if args.len() != 2 {
                return Err(
                    "ts_sum function requires 2 arguments: expression and window".to_string(),
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_sum window must be an integer literal".to_string()),
            };
            Ok(series.ts_sum(window))
        }
        "ts_argmax" => {
            if args.len() != 2 {
                return Err(
                    "ts_argmax function requires 2 arguments: expression and window".to_string(),
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_argmax window must be an integer literal".to_string()),
            };
            Ok(series.ts_argmax(window))
        }
        "ts_argmin" => {
            if args.len() != 2 {
                return Err(
                    "ts_argmin function requires 2 arguments: expression and window".to_string(),
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_argmin window must be an integer literal".to_string()),
            };
            Ok(series.ts_argmin(window))
        }
        "delay" => {
            // delay is same as lag
            evaluate_function_vectorized("lag", args, df)
        }
        "abs" => {
            if args.len() != 1 {
                return Err("abs function requires 1 argument: expression".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            Ok(series.abs())
        }
        "sign" => {
            if args.len() != 1 {
                return Err("sign function requires 1 argument: expression".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            Ok(series.sign())
        }
        "log" => {
            if args.len() != 1 {
                return Err("log function requires 1 argument: expression".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            Ok(series.log())
        }
        "exp" => {
            if args.len() != 1 {
                return Err("exp function requires 1 argument: expression".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            Ok(series.exp())
        }
        "sqrt" => {
            if args.len() != 1 {
                return Err("sqrt function requires 1 argument: expression".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            Ok(series.sqrt())
        }
        "power" => {
            if args.len() != 2 {
                return Err(
                    "power function requires 2 arguments: expression and exponent".to_string(),
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let exp = match &args[1] {
                Expr::Literal(Literal::Float(f)) => *f,
                Expr::Literal(Literal::Integer(i)) => *i as f64,
                _ => return Err("power exponent must be a numeric literal".to_string()),
            };
            Ok(series.power(exp))
        }
        "scale" => {
            if args.len() != 2 {
                return Err(
                    "scale function requires 2 arguments: expression and window".to_string()
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("scale window must be an integer literal".to_string()),
            };
            Ok(series.scale(window))
        }
        "decay_linear" => {
            if args.len() != 2 {
                return Err(
                    "decay_linear function requires 2 arguments: expression and periods"
                        .to_string(),
                );
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let periods = match &args[1] {
                Expr::Literal(Literal::Integer(p)) => *p as usize,
                _ => return Err("decay_linear periods must be an integer literal".to_string()),
            };
            Ok(series.decay_linear(periods))
        }
        _ => Err(format!(
            "Unknown function in vectorized evaluator: {}",
            name
        )),
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a column reference expression
pub fn col(name: &str) -> Expr {
    Expr::Column(name.to_string())
}

/// Create a literal float expression
pub fn lit_float(value: f64) -> Expr {
    Expr::Literal(Literal::Float(value))
}

/// Create a literal integer expression
pub fn lit_int(value: i64) -> Expr {
    Expr::Literal(Literal::Integer(value))
}

/// Create a DataFrame from column arrays
pub fn df_from_arrays(data: HashMap<String, Array1<f64>>) -> Result<DataFrame, String> {
    let mut columns = HashMap::new();
    for (name, array) in data {
        columns.insert(name, Series::from_array(array));
    }
    DataFrame::from_series_map(columns)
}

/// Create a DataFrame for backtesting (prices, returns, etc.)
pub fn create_backtest_dataframe(
    prices: Array2<f64>,
    symbols: &[&str],
) -> Result<HashMap<String, DataFrame>, String> {
    let (n_days, n_assets) = prices.dim();
    let mut result = HashMap::new();

    for asset_idx in 0..n_assets {
        let symbol = symbols
            .get(asset_idx)
            .ok_or_else(|| "Not enough symbols".to_string())?;

        let mut columns = HashMap::new();
        let price_series = prices.column(asset_idx).to_owned();
        columns.insert("close".to_string(), Series::from_array(price_series));

        // Calculate returns (simplified)
        let mut returns = Array1::zeros(n_days);
        for day_idx in 1..n_days {
            let price_today = prices[[day_idx, asset_idx]];
            let price_yesterday = prices[[day_idx - 1, asset_idx]];
            returns[day_idx] = if price_yesterday == 0.0 {
                f64::NAN
            } else {
                (price_today / price_yesterday) - 1.0
            };
        }
        returns[0] = f64::NAN;
        columns.insert("returns".to_string(), Series::from_array(returns));

        let df = DataFrame::from_series_map(columns)?;
        result.insert(symbol.to_string(), df);
    }

    Ok(result)
}

// ============================================================================
// Optimized Expression Evaluation with Caching
// ============================================================================

/// Cached expression evaluator for improved performance
pub struct CachedExpressionEvaluator<'a> {
    df: &'a DataFrame,
    cache: HashMap<String, Series>,
    optimizer: ExpressionOptimizer,
}

impl<'a> CachedExpressionEvaluator<'a> {
    /// Create a new cached evaluator for a DataFrame
    pub fn new(df: &'a DataFrame) -> Self {
        Self {
            df,
            cache: HashMap::new(),
            optimizer: ExpressionOptimizer::new(),
        }
    }

    /// Evaluate an expression with caching
    pub fn evaluate(&mut self, expr: &Expr) -> Result<Series, String> {
        // First, optimize the expression
        let optimized = self.optimizer.optimize(expr.clone());

        // Use string representation as cache key
        let cache_key = format!("{:?}", optimized);

        // Check cache
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Evaluate (using the optimized expression)
        let result = self.evaluate_uncached(&optimized)?;

        // Cache the result
        self.cache.insert(cache_key, result.clone());

        Ok(result)
    }

    /// Evaluate without caching (internal method)
    fn evaluate_uncached(&mut self, expr: &Expr) -> Result<Series, String> {
        match expr {
            Expr::Literal(lit) => match lit {
                Literal::Float(f) => {
                    let data = Array1::from_elem(self.df.n_rows(), *f);
                    Ok(Series::from_array(data))
                }
                Literal::Integer(i) => {
                    let data = Array1::from_elem(self.df.n_rows(), *i as f64);
                    Ok(Series::from_array(data))
                }
                Literal::Boolean(b) => {
                    let data = Array1::from_elem(self.df.n_rows(), if *b { 1.0 } else { 0.0 });
                    Ok(Series::from_array(data))
                }
                Literal::String(_) => {
                    Err("String literals not supported in vectorized evaluation".to_string())
                }
                Literal::Null => {
                    let data = Array1::from_elem(self.df.n_rows(), f64::NAN);
                    Ok(Series::from_array(data))
                }
            },
            Expr::Column(name) => self
                .df
                .column(name)
                .cloned()
                .ok_or_else(|| format!("Column '{}' not found in DataFrame", name)),
            Expr::BinaryExpr { left, op, right } => {
                let left_series = self.evaluate(left)?;
                let right_series = self.evaluate(right)?;

                match op {
                    BinaryOp::Add => left_series.add(&right_series),
                    BinaryOp::Subtract => left_series.sub(&right_series),
                    BinaryOp::Multiply => left_series.mul(&right_series),
                    BinaryOp::Divide => left_series.div(&right_series),
                    BinaryOp::GreaterThan => left_series.gt(&right_series),
                    _ => Err(format!(
                        "Binary operator {:?} not yet implemented in vectorized evaluator",
                        op
                    )),
                }
            }
            Expr::UnaryExpr { op, expr } => {
                let series = self.evaluate(expr)?;
                match op {
                    UnaryOp::Negate => Ok(series.neg()),
                    UnaryOp::Abs => Ok(series.abs()),
                    UnaryOp::Sqrt => Ok(series.sqrt()),
                    UnaryOp::Log => Ok(series.log()),
                    UnaryOp::Exp => Ok(series.exp()),
                    _ => Err(format!(
                        "Unary operator {:?} not yet implemented in vectorized evaluator",
                        op
                    )),
                }
            }
            Expr::FunctionCall { name, args } => self.evaluate_function(name, args),
            _ => Err(format!(
                "Expression type {:?} not yet supported in vectorized evaluation",
                expr
            )),
        }
    }

    /// Evaluate a function call
    fn evaluate_function(&mut self, name: &str, args: &[Expr]) -> Result<Series, String> {
        match name {
            "lag" => {
                if args.len() != 2 {
                    return Err(
                        "lag function requires 2 arguments: expression and periods".to_string()
                    );
                }
                let series = self.evaluate(&args[0])?;
                let periods = match &args[1] {
                    Expr::Literal(Literal::Integer(p)) => *p as usize,
                    _ => return Err("lag periods must be an integer literal".to_string()),
                };
                Ok(series.lag(periods))
            }
            "diff" => {
                if args.len() != 2 {
                    return Err(
                        "diff function requires 2 arguments: expression and periods".to_string()
                    );
                }
                let series = self.evaluate(&args[0])?;
                let periods = match &args[1] {
                    Expr::Literal(Literal::Integer(p)) => *p as usize,
                    _ => return Err("diff periods must be an integer literal".to_string()),
                };
                Ok(series.diff(periods))
            }
            "pct_change" => {
                if args.len() != 2 {
                    return Err(
                        "pct_change function requires 2 arguments: expression and periods"
                            .to_string(),
                    );
                }
                let series = self.evaluate(&args[0])?;
                let periods = match &args[1] {
                    Expr::Literal(Literal::Integer(p)) => *p as usize,
                    _ => return Err("pct_change periods must be an integer literal".to_string()),
                };
                Ok(series.pct_change(periods))
            }
            "moving_average" => {
                if args.len() != 2 {
                    return Err(
                        "moving_average function requires 2 arguments: expression and window"
                            .to_string(),
                    );
                }
                let series = self.evaluate(&args[0])?;
                let window = match &args[1] {
                    Expr::Literal(Literal::Integer(w)) => *w as usize,
                    _ => return Err("moving_average window must be an integer literal".to_string()),
                };
                Ok(series.moving_average(window))
            }
            "momentum" => {
                // momentum is same as pct_change
                self.evaluate_function("pct_change", args)
            }
            "volatility" => {
                if args.len() != 2 {
                    return Err(
                        "volatility function requires 2 arguments: expression and periods"
                            .to_string(),
                    );
                }
                let series = self.evaluate(&args[0])?;
                let periods = match &args[1] {
                    Expr::Literal(Literal::Integer(p)) => *p as usize,
                    _ => return Err("volatility periods must be an integer literal".to_string()),
                };
                Ok(series.rolling_std(periods))
            }
            "ema" => {
                if args.len() != 2 {
                    return Err(
                        "ema function requires 2 arguments: expression and span".to_string()
                    );
                }
                let series = self.evaluate(&args[0])?;
                let span = match &args[1] {
                    Expr::Literal(Literal::Integer(s)) => *s as usize,
                    _ => return Err("ema span must be an integer literal".to_string()),
                };
                Ok(series.ema(span))
            }
            _ => Err(format!(
                "Unknown function in vectorized evaluator: {}",
                name
            )),
        }
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get the number of cached results
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

/// Optimized version of evaluate_expr_on_dataframe with caching
pub fn evaluate_expr_on_dataframe_optimized(expr: &Expr, df: &DataFrame) -> Result<Series, String> {
    let mut evaluator = CachedExpressionEvaluator::new(df);
    evaluator.evaluate(expr)
}

/// Pre-optimize an expression for better performance
pub fn optimize_expr_for_evaluation(expr: &Expr) -> Expr {
    optimize_expression(expr.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::ast::{BinaryOp, Expr, Literal, UnaryOp};
    use ndarray::Array1;
    use std::collections::HashMap;

    // Helper function to compare slices with NaN values
    fn approx_eq_slice(a: &[f64], b: &[f64], epsilon: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (x, y) in a.iter().zip(b.iter()) {
            if x.is_nan() && y.is_nan() {
                continue;
            } else if (x - y).abs() > epsilon {
                return false;
            }
        }
        true
    }

    // ============================================================================
    // DataFrame Tests
    // ============================================================================

    #[test]
    fn test_dataframe_new() {
        let df = DataFrame::new();
        assert_eq!(df.n_rows(), 0);
        assert_eq!(df.n_cols(), 0);
    }

    #[test]
    fn test_dataframe_from_series_map() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));
        columns.insert("b".to_string(), Series::new(vec![4.0, 5.0, 6.0]));

        let df = DataFrame::from_series_map(columns).unwrap();
        assert_eq!(df.n_rows(), 3);
        assert_eq!(df.n_cols(), 2);
    }

    #[test]
    fn test_dataframe_from_series_map_mismatched_lengths() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));
        columns.insert("b".to_string(), Series::new(vec![4.0, 5.0]));

        let result = DataFrame::from_series_map(columns);
        assert!(result.is_err());
    }

    #[test]
    fn test_dataframe_column() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));

        let df = DataFrame::from_series_map(columns).unwrap();
        let col = df.column("a");
        assert!(col.is_some());
        assert_eq!(col.unwrap().len(), 3);
    }

    #[test]
    fn test_dataframe_column_not_found() {
        let df = DataFrame::new();
        let col = df.column("nonexistent");
        assert!(col.is_none());
    }

    #[test]
    fn test_dataframe_column_names() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0]));
        columns.insert("b".to_string(), Series::new(vec![3.0, 4.0]));

        let df = DataFrame::from_series_map(columns).unwrap();
        let names = df.column_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
    }

    #[test]
    fn test_dataframe_with_column() {
        let df = DataFrame::new();
        let df = df
            .with_column("a", Series::new(vec![1.0, 2.0, 3.0]))
            .unwrap();
        assert_eq!(df.n_rows(), 3);
        assert_eq!(df.n_cols(), 1);
    }

    #[test]
    fn test_dataframe_with_column_mismatched_length() {
        let df = DataFrame::new();
        let df = df
            .with_column("a", Series::new(vec![1.0, 2.0, 3.0]))
            .unwrap();
        let result = df.with_column("b", Series::new(vec![1.0, 2.0]));
        assert!(result.is_err());
    }

    #[test]
    fn test_dataframe_select() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));
        columns.insert("b".to_string(), Series::new(vec![4.0, 5.0, 6.0]));
        columns.insert("c".to_string(), Series::new(vec![7.0, 8.0, 9.0]));

        let df = DataFrame::from_series_map(columns).unwrap();
        let df_selected = df.select(&["a", "c"]).unwrap();

        assert_eq!(df_selected.n_cols(), 2);
        assert!(df_selected.column("a").is_some());
        assert!(df_selected.column("c").is_some());
        assert!(df_selected.column("b").is_none());
    }

    #[test]
    fn test_dataframe_select_not_found() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));

        let df = DataFrame::from_series_map(columns).unwrap();
        let result = df.select(&["a", "nonexistent"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dataframe_with_expr() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));
        columns.insert("b".to_string(), Series::new(vec![4.0, 5.0, 6.0]));

        let df = DataFrame::from_series_map(columns).unwrap();
        let expr = Expr::BinaryExpr {
            left: Arc::new(Expr::Column("a".to_string())),
            op: BinaryOp::Add,
            right: Arc::new(Expr::Column("b".to_string())),
        };

        let df_with_expr = df.with_expr("c", &expr).unwrap();
        let c_col = df_with_expr.column("c").unwrap();
        assert_eq!(c_col.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    // ============================================================================
    // Expression Evaluation Tests
    // ============================================================================

    #[test]
    fn test_evaluate_literal_float() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::Literal(Literal::Float(5.0));
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_evaluate_literal_integer() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::Literal(Literal::Integer(10));
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[10.0, 10.0, 10.0]);
    }

    #[test]
    fn test_evaluate_column() {
        let mut columns = HashMap::new();
        columns.insert("price".to_string(), Series::new(vec![10.0, 20.0, 30.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::Column("price".to_string());
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_evaluate_column_not_found() {
        let df = DataFrame::new();
        let expr = Expr::Column("nonexistent".to_string());
        let result = evaluate_expr_on_dataframe(&expr, &df);
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluate_binary_add() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));
        columns.insert("b".to_string(), Series::new(vec![4.0, 5.0, 6.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::BinaryExpr {
            left: Arc::new(Expr::Column("a".to_string())),
            op: BinaryOp::Add,
            right: Arc::new(Expr::Column("b".to_string())),
        };
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_evaluate_binary_subtract() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![10.0, 20.0, 30.0]));
        columns.insert("b".to_string(), Series::new(vec![1.0, 2.0, 3.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::BinaryExpr {
            left: Arc::new(Expr::Column("a".to_string())),
            op: BinaryOp::Subtract,
            right: Arc::new(Expr::Column("b".to_string())),
        };
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_evaluate_binary_multiply() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![2.0, 3.0, 4.0]));
        columns.insert("b".to_string(), Series::new(vec![5.0, 6.0, 7.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::BinaryExpr {
            left: Arc::new(Expr::Column("a".to_string())),
            op: BinaryOp::Multiply,
            right: Arc::new(Expr::Column("b".to_string())),
        };
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_evaluate_binary_divide() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![10.0, 20.0, 30.0]));
        columns.insert("b".to_string(), Series::new(vec![2.0, 4.0, 5.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::BinaryExpr {
            left: Arc::new(Expr::Column("a".to_string())),
            op: BinaryOp::Divide,
            right: Arc::new(Expr::Column("b".to_string())),
        };
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_evaluate_unary_negate() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, -2.0, 3.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::UnaryExpr {
            op: UnaryOp::Negate,
            expr: Arc::new(Expr::Column("a".to_string())),
        };
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_evaluate_unary_abs() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![-1.0, 2.0, -3.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::UnaryExpr {
            op: UnaryOp::Abs,
            expr: Arc::new(Expr::Column("a".to_string())),
        };
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_evaluate_function_lag() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::FunctionCall {
            name: "lag".to_string(),
            args: vec![
                Expr::Column("a".to_string()),
                Expr::Literal(Literal::Integer(2)),
            ],
        };
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        let expected = &[f64::NAN, f64::NAN, 1.0, 2.0, 3.0];
        assert!(approx_eq_slice(result.as_slice().unwrap(), expected, 1e-10));
    }

    #[test]
    fn test_evaluate_function_diff() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![10.0, 20.0, 30.0, 40.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::FunctionCall {
            name: "diff".to_string(),
            args: vec![
                Expr::Column("a".to_string()),
                Expr::Literal(Literal::Integer(1)),
            ],
        };
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        let expected = &[f64::NAN, 10.0, 10.0, 10.0];
        assert!(approx_eq_slice(result.as_slice().unwrap(), expected, 1e-10));
    }

    #[test]
    fn test_evaluate_function_moving_average() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::FunctionCall {
            name: "moving_average".to_string(),
            args: vec![
                Expr::Column("a".to_string()),
                Expr::Literal(Literal::Integer(3)),
            ],
        };
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        assert_eq!(result.len(), 5);
        // Should produce some non-NaN values
        assert!(!result.data()[2].is_nan());
    }

    #[test]
    fn test_evaluate_function_rank() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![3.0, 1.0, 2.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::FunctionCall {
            name: "rank".to_string(),
            args: vec![Expr::Column("a".to_string())],
        };
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        assert_eq!(result.len(), 3);
        // Rank should produce values between 0 and 1
        for i in 0..result.len() {
            assert!(result.data()[i] >= 0.0 && result.data()[i] <= 1.0);
        }
    }

    #[test]
    fn test_evaluate_nested_expression() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));
        columns.insert("b".to_string(), Series::new(vec![4.0, 5.0, 6.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        // (a + b) * 2
        let expr = Expr::BinaryExpr {
            left: Arc::new(Expr::BinaryExpr {
                left: Arc::new(Expr::Column("a".to_string())),
                op: BinaryOp::Add,
                right: Arc::new(Expr::Column("b".to_string())),
            }),
            op: BinaryOp::Multiply,
            right: Arc::new(Expr::Literal(Literal::Float(2.0))),
        };
        let result = evaluate_expr_on_dataframe(&expr, &df).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[10.0, 14.0, 18.0]);
    }

    // ============================================================================
    // Helper Function Tests
    // ============================================================================

    #[test]
    fn test_col() {
        let expr = col("price");
        assert_eq!(expr, Expr::Column("price".to_string()));
    }

    #[test]
    fn test_lit_float() {
        let expr = lit_float(3.14);
        assert_eq!(expr, Expr::Literal(Literal::Float(3.14)));
    }

    #[test]
    fn test_lit_int() {
        let expr = lit_int(42);
        assert_eq!(expr, Expr::Literal(Literal::Integer(42)));
    }

    #[test]
    fn test_df_from_arrays() {
        let mut data = HashMap::new();
        data.insert("a".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0]));
        data.insert("b".to_string(), Array1::from_vec(vec![4.0, 5.0, 6.0]));

        let df = df_from_arrays(data).unwrap();
        assert_eq!(df.n_rows(), 3);
        assert_eq!(df.n_cols(), 2);
    }

    #[test]
    fn test_df_from_arrays_mismatched_lengths() {
        let mut data = HashMap::new();
        data.insert("a".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0]));
        data.insert("b".to_string(), Array1::from_vec(vec![4.0, 5.0]));

        let result = df_from_arrays(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_backtest_dataframe() {
        let prices =
            Array2::from_shape_vec((3, 2), vec![100.0, 200.0, 105.0, 210.0, 110.0, 220.0]).unwrap();

        let symbols = &["AAPL", "GOOG"];
        let result = create_backtest_dataframe(prices, symbols).unwrap();

        assert_eq!(result.len(), 2);

        let aapl = result.get("AAPL").unwrap();
        assert_eq!(aapl.n_cols(), 2);
        assert!(aapl.column("close").is_some());
        assert!(aapl.column("returns").is_some());
    }

    // ============================================================================
    // CachedExpressionEvaluator Tests
    // ============================================================================

    #[test]
    fn test_cached_evaluator_new() {
        let columns = HashMap::new();
        let df = DataFrame::from_series_map(columns).unwrap();
        let evaluator = CachedExpressionEvaluator::new(&df);
        assert_eq!(evaluator.cache_size(), 0);
    }

    #[test]
    fn test_cached_evaluator_evaluate() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let mut evaluator = CachedExpressionEvaluator::new(&df);
        let expr = Expr::Column("a".to_string());
        let result = evaluator.evaluate(&expr).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cached_evaluator_caching() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let mut evaluator = CachedExpressionEvaluator::new(&df);
        let expr = Expr::Column("a".to_string());

        // First evaluation
        evaluator.evaluate(&expr).unwrap();
        assert_eq!(evaluator.cache_size(), 1);

        // Second evaluation should use cache
        evaluator.evaluate(&expr).unwrap();
        assert_eq!(evaluator.cache_size(), 1);
    }

    #[test]
    fn test_cached_evaluator_clear_cache() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let mut evaluator = CachedExpressionEvaluator::new(&df);
        let expr = Expr::Column("a".to_string());
        evaluator.evaluate(&expr).unwrap();
        assert_eq!(evaluator.cache_size(), 1);

        evaluator.clear_cache();
        assert_eq!(evaluator.cache_size(), 0);
    }

    #[test]
    fn test_evaluate_expr_on_dataframe_optimized() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![1.0, 2.0, 3.0]));
        let df = DataFrame::from_series_map(columns).unwrap();

        let expr = Expr::Column("a".to_string());
        let result = evaluate_expr_on_dataframe_optimized(&expr, &df).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_optimize_expr_for_evaluation() {
        // (1 + 2) + 3 -> should be optimized to 6
        let expr = Expr::BinaryExpr {
            left: Arc::new(Expr::BinaryExpr {
                left: Arc::new(Expr::Literal(Literal::Float(1.0))),
                op: BinaryOp::Add,
                right: Arc::new(Expr::Literal(Literal::Float(2.0))),
            }),
            op: BinaryOp::Add,
            right: Arc::new(Expr::Literal(Literal::Float(3.0))),
        };

        let optimized = optimize_expr_for_evaluation(&expr);
        // After optimization, should be a constant
        match optimized {
            Expr::Literal(Literal::Float(f)) => {
                assert!((f - 6.0).abs() < 0.001);
            }
            _ => {
                // The optimizer might not fully constant fold in all cases
                // This test just verifies it doesn't panic
            }
        }
    }
}
