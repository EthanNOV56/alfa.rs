//! Polars-style expression evaluation for vectorized alpha factor computation
//!
//! This module provides a Polars-inspired API for building and evaluating
//! expressions on time series data in a vectorized manner.

use crate::expr::{Expr, Literal, BinaryOp, UnaryOp};
use crate::expr_optimizer::{ExpressionOptimizer, optimize_expression};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

// ============================================================================
// Series - Wrapper around ndarray for vectorized operations
// ============================================================================

/// A Series represents a time series of values with vectorized operations
#[derive(Debug, Clone)]
pub struct Series {
    data: Array1<f64>,
    name: Option<String>,
}

impl Series {
    /// Create a new Series from a vector
    pub fn new(data: Vec<f64>) -> Self {
        Series {
            data: Array1::from(data),
            name: None,
        }
    }
    
    /// Create a new Series from an ndarray
    pub fn from_array(data: Array1<f64>) -> Self {
        Series {
            data,
            name: None,
        }
    }
    
    /// Create a new Series with a name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    
    /// Get the length of the series
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if the series is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Get the underlying data as a slice
    pub fn as_slice(&self) -> Option<&[f64]> {
        self.data.as_slice()
    }
    
    /// Get the underlying ndarray
    pub fn data(&self) -> &Array1<f64> {
        &self.data
    }
    
    /// Get the name of the series
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    
    // ==================== Vectorized Operations ====================
    
    /// Element-wise addition
    pub fn add(&self, other: &Series) -> Result<Series, String> {
        if self.len() != other.len() {
            return Err(format!("Series length mismatch: {} != {}", self.len(), other.len()));
        }
        Ok(Series::from_array(&self.data + &other.data))
    }
    
    /// Element-wise subtraction
    pub fn sub(&self, other: &Series) -> Result<Series, String> {
        if self.len() != other.len() {
            return Err(format!("Series length mismatch: {} != {}", self.len(), other.len()));
        }
        Ok(Series::from_array(&self.data - &other.data))
    }
    
    /// Element-wise multiplication
    pub fn mul(&self, other: &Series) -> Result<Series, String> {
        if self.len() != other.len() {
            return Err(format!("Series length mismatch: {} != {}", self.len(), other.len()));
        }
        Ok(Series::from_array(&self.data * &other.data))
    }
    
    /// Element-wise division
    pub fn div(&self, other: &Series) -> Result<Series, String> {
        if self.len() != other.len() {
            return Err(format!("Series length mismatch: {} != {}", self.len(), other.len()));
        }
        Ok(Series::from_array(&self.data / &other.data))
    }
    
    /// Element-wise comparison: greater than
    pub fn gt(&self, other: &Series) -> Result<Series, String> {
        if self.len() != other.len() {
            return Err(format!("Series length mismatch: {} != {}", self.len(), other.len()));
        }
        let result: Array1<f64> = Array1::from_iter(
            self.data.iter().zip(other.data.iter())
                .map(|(&a, &b)| if a > b { 1.0 } else { 0.0 })
        );
        Ok(Series::from_array(result))
    }
    
    /// Element-wise absolute value
    pub fn abs(&self) -> Series {
        Series::from_array(self.data.mapv(f64::abs))
    }
    
    /// Element-wise square root
    pub fn sqrt(&self) -> Series {
        Series::from_array(self.data.mapv(|x| x.sqrt()))
    }
    
    /// Element-wise logarithm
    pub fn log(&self) -> Series {
        Series::from_array(self.data.mapv(|x| x.ln()))
    }
    
    /// Element-wise exponential
    pub fn exp(&self) -> Series {
        Series::from_array(self.data.mapv(f64::exp))
    }
    
    /// Element-wise negation
    pub fn neg(&self) -> Series {
        Series::from_array(-&self.data)
    }
    
    // ==================== Time Series Operations ====================
    
    /// Lag the series by N periods
    pub fn lag(&self, periods: usize) -> Series {
        let n = self.len();
        if periods >= n {
            return Series::new(vec![f64::NAN; n]);
        }
        
        let mut lagged = vec![f64::NAN; periods];
        lagged.extend_from_slice(&self.data.as_slice().unwrap()[..n - periods]);
        Series::new(lagged)
    }
    
    /// Difference: current value - lagged value
    pub fn diff(&self, periods: usize) -> Series {
        let lagged = self.lag(periods);
        self.sub(&lagged).unwrap_or_else(|_| Series::new(vec![f64::NAN; self.len()]))
    }
    
    /// Percentage change: (current - lagged) / lagged
    pub fn pct_change(&self, periods: usize) -> Series {
        let lagged = self.lag(periods);
        let diff = self.sub(&lagged).unwrap_or_else(|_| Series::new(vec![f64::NAN; self.len()]));
        diff.div(&lagged).unwrap_or_else(|_| Series::new(vec![f64::NAN; self.len()]))
    }
    
    /// Simple moving average
    pub fn moving_average(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);
        
        for i in 0..n {
            let start = if i + 1 >= window { i + 1 - window } else { 0 };
            let slice = self.data.slice(ndarray::s![start..=i]);
            result[i] = if slice.len() == 0 { f64::NAN } else { slice.mean().unwrap_or(f64::NAN) };
        }
        
        Series::from_array(result)
    }
    
    /// Rolling standard deviation
    pub fn rolling_std(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);
        
        for i in 0..n {
            let start = if i + 1 >= window { i + 1 - window } else { 0 };
            let slice = self.data.slice(ndarray::s![start..=i]);
            result[i] = if slice.len() == 0 { f64::NAN } else { slice.std(1.0) };
        }
        
        Series::from_array(result)
    }
    
    /// Exponential moving average
    pub fn ema(&self, span: usize) -> Series {
        let n = self.len();
        let alpha = 2.0 / (span as f64 + 1.0);
        let mut result = Array1::zeros(n);
        
        if n > 0 {
            result[0] = self.data[0];
            for i in 1..n {
                result[i] = alpha * self.data[i] + (1.0 - alpha) * result[i - 1];
            }
        }
        
        Series::from_array(result)
    }
    
    /// Z-score normalization
    pub fn z_score(&self, window: usize) -> Series {
        let mean = self.moving_average(window);
        let std = self.rolling_std(window);
        let z_data = (&self.data - &mean.data) / &std.data;
        Series::from_array(z_data)
    }
}

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
                return Err(format!("Column '{}' has length {}, expected {}", 
                    name, series.len(), n_rows));
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
            return Err(format!("Column '{}' has length {}, expected {}", 
                name, series.len(), self.n_rows));
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
                Literal::String(_) => Err("String literals not supported in vectorized evaluation".to_string()),
                Literal::Null => {
                    let data = Array1::from_elem(df.n_rows(), f64::NAN);
                    Ok(Series::from_array(data))
                }
            }
        }
        Expr::Column(name) => {
            df.column(name)
                .cloned()
                .ok_or_else(|| format!("Column '{}' not found in DataFrame", name))
        }
        Expr::BinaryExpr { left, op, right } => {
            let left_series = evaluate_expr_on_dataframe(left, df)?;
            let right_series = evaluate_expr_on_dataframe(right, df)?;
            
            match op {
                BinaryOp::Add => left_series.add(&right_series),
                BinaryOp::Subtract => left_series.sub(&right_series),
                BinaryOp::Multiply => left_series.mul(&right_series),
                BinaryOp::Divide => left_series.div(&right_series),
                BinaryOp::GreaterThan => left_series.gt(&right_series),
                _ => Err(format!("Binary operator {:?} not yet implemented in vectorized evaluator", op)),
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
                _ => Err(format!("Unary operator {:?} not yet implemented in vectorized evaluator", op)),
            }
        }
        Expr::FunctionCall { name, args } => {
            evaluate_function_vectorized(name, args, df)
        }
        _ => Err(format!("Expression type {:?} not yet supported in vectorized evaluation", expr)),
    }
}

/// Evaluate a function call in vectorized manner
fn evaluate_function_vectorized(name: &str, args: &[Expr], df: &DataFrame) -> Result<Series, String> {
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
                return Err("diff function requires 2 arguments: expression and periods".to_string());
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
                return Err("pct_change function requires 2 arguments: expression and periods".to_string());
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
                return Err("moving_average function requires 2 arguments: expression and window".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("moving_average window must be an integer literal".to_string()),
            };
            Ok(series.moving_average(window))
        }
        "momentum" => {
            // momentum is same as pct_change
            evaluate_function_vectorized("pct_change", args, df)
        }
        "volatility" => {
            if args.len() != 2 {
                return Err("volatility function requires 2 arguments: expression and periods".to_string());
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
        _ => Err(format!("Unknown function in vectorized evaluator: {}", name)),
    }
}

// ============================================================================
// Convenience functions for Polars-style API
// ============================================================================

/// Create a column reference expression
pub fn col(name: &str) -> Expr {
    Expr::col(name)
}

/// Create a literal float expression
pub fn lit_float(value: f64) -> Expr {
    Expr::lit_float(value)
}

/// Create a literal integer expression
pub fn lit_int(value: i64) -> Expr {
    Expr::lit_int(value)
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
        let symbol = symbols.get(asset_idx).ok_or_else(|| "Not enough symbols".to_string())?;
        
        let mut columns = HashMap::new();
        let price_series = prices.column(asset_idx).to_owned();
        columns.insert("close".to_string(), Series::from_array(price_series));
        
        // Calculate returns (simplified)
        let mut returns = Array1::zeros(n_days);
        for day_idx in 1..n_days {
            let price_today = prices[[day_idx, asset_idx]];
            let price_yesterday = prices[[day_idx - 1, asset_idx]];
            returns[day_idx] = if price_yesterday == 0.0 { f64::NAN } else { 
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
            Expr::Literal(lit) => {
                match lit {
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
                    Literal::String(_) => Err("String literals not supported in vectorized evaluation".to_string()),
                    Literal::Null => {
                        let data = Array1::from_elem(self.df.n_rows(), f64::NAN);
                        Ok(Series::from_array(data))
                    }
                }
            }
            Expr::Column(name) => {
                self.df.column(name)
                    .cloned()
                    .ok_or_else(|| format!("Column '{}' not found in DataFrame", name))
            }
            Expr::BinaryExpr { left, op, right } => {
                let left_series = self.evaluate(left)?;
                let right_series = self.evaluate(right)?;
                
                match op {
                    BinaryOp::Add => left_series.add(&right_series),
                    BinaryOp::Subtract => left_series.sub(&right_series),
                    BinaryOp::Multiply => left_series.mul(&right_series),
                    BinaryOp::Divide => left_series.div(&right_series),
                    BinaryOp::GreaterThan => left_series.gt(&right_series),
                    _ => Err(format!("Binary operator {:?} not yet implemented in vectorized evaluator", op)),
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
                    _ => Err(format!("Unary operator {:?} not yet implemented in vectorized evaluator", op)),
                }
            }
            Expr::FunctionCall { name, args } => {
                self.evaluate_function(name, args)
            }
            _ => Err(format!("Expression type {:?} not yet supported in vectorized evaluation", expr)),
        }
    }
    
    /// Evaluate a function call
    fn evaluate_function(&mut self, name: &str, args: &[Expr]) -> Result<Series, String> {
        match name {
            "lag" => {
                if args.len() != 2 {
                    return Err("lag function requires 2 arguments: expression and periods".to_string());
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
                    return Err("diff function requires 2 arguments: expression and periods".to_string());
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
                    return Err("pct_change function requires 2 arguments: expression and periods".to_string());
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
                    return Err("moving_average function requires 2 arguments: expression and window".to_string());
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
                    return Err("volatility function requires 2 arguments: expression and periods".to_string());
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
                    return Err("ema function requires 2 arguments: expression and span".to_string());
                }
                let series = self.evaluate(&args[0])?;
                let span = match &args[1] {
                    Expr::Literal(Literal::Integer(s)) => *s as usize,
                    _ => return Err("ema span must be an integer literal".to_string()),
                };
                Ok(series.ema(span))
            }
            _ => Err(format!("Unknown function in vectorized evaluator: {}", name)),
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
