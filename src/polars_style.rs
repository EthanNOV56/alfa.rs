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

    // ==================== Alpha101 Functions ====================

    /// Time series rank over rolling window (ts_rank)
    pub fn ts_rank(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let slice = self.data.slice(ndarray::s![start..=i]);
                let mut vals: Vec<f64> = slice.iter().cloned().collect();
                // Sort and get rank of last element
                let last_val = vals[vals.len() - 1];
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                if let Some(pos) = vals.iter().position(|&x| x == last_val) {
                    result[i] = (pos + 1) as f64 / vals.len() as f64;
                } else {
                    result[i] = f64::NAN;
                }
            }
        }

        Series::from_array(result)
    }

    /// Time series argmax over rolling window (ts_argmax)
    pub fn ts_argmax(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let slice = self.data.slice(ndarray::s![start..=i]);
                // Find position of maximum (1-indexed)
                let max_pos = slice
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(pos, _)| pos + 1);
                result[i] = max_pos.unwrap_or(0) as f64;
            }
        }

        Series::from_array(result)
    }

    /// Time series argmin over rolling window (ts_argmin)
    pub fn ts_argmin(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let slice = self.data.slice(ndarray::s![start..=i]);
                // Find position of minimum (1-indexed)
                let min_pos = slice
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(pos, _)| pos + 1);
                result[i] = min_pos.unwrap_or(0) as f64;
            }
        }

        Series::from_array(result)
    }

    /// Cross-sectional rank (rank)
    pub fn cs_rank(&self) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        let mut vals: Vec<(usize, f64)> = self.data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        vals.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (orig_idx, _) in &vals {
            result[*orig_idx] = (orig_idx + 1) as f64 / n as f64;
        }

        Series::from_array(result)
    }

    /// Time series correlation over rolling window (ts_corr)
    pub fn ts_corr(&self, other: &Series, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let self_slice = self.data.slice(ndarray::s![start..=i]);
                let other_slice = other.data.slice(ndarray::s![start..=i]);

                // Calculate correlation
                let self_mean = self_slice.iter().sum::<f64>() / self_slice.len() as f64;
                let other_mean = other_slice.iter().sum::<f64>() / other_slice.len() as f64;

                let mut cov = 0.0;
                let mut self_var = 0.0;
                let mut other_var = 0.0;

                for j in 0..self_slice.len() {
                    let self_diff = self_slice[j] - self_mean;
                    let other_diff = other_slice[j] - other_mean;
                    cov += self_diff * other_diff;
                    self_var += self_diff * self_diff;
                    other_var += other_diff * other_diff;
                }

                let denom = (self_var * other_var).sqrt();
                if denom > 0.0 {
                    result[i] = cov / denom;
                } else {
                    result[i] = f64::NAN;
                }
            }
        }

        Series::from_array(result)
    }

    /// Time series covariance over rolling window (ts_cov)
    pub fn ts_cov(&self, other: &Series, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let self_slice = self.data.slice(ndarray::s![start..=i]);
                let other_slice = other.data.slice(ndarray::s![start..=i]);

                let self_mean = self_slice.iter().sum::<f64>() / self_slice.len() as f64;
                let other_mean = other_slice.iter().sum::<f64>() / other_slice.len() as f64;

                let mut cov = 0.0;
                for j in 0..self_slice.len() {
                    let self_diff = self_slice[j] - self_mean;
                    let other_diff = other_slice[j] - other_mean;
                    cov += self_diff * other_diff;
                }

                result[i] = cov / window as f64;
            }
        }

        Series::from_array(result)
    }

    /// Scale to [-1, 1] using rolling window (scale)
    pub fn scale(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let slice = self.data.slice(ndarray::s![start..=i]);
                let mean = slice.iter().sum::<f64>() / slice.len() as f64;
                let std = (slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64).sqrt();

                if std > 0.0 {
                    result[i] = (self.data[i] - mean) / std;
                } else {
                    result[i] = 0.0;
                }
            }
        }

        Series::from_array(result)
    }

    /// Linear decay weighted average (decay_linear)
    pub fn decay_linear(&self, periods: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        // Weights: 1, 2, 3, ..., periods
        let total_weight = (periods * (periods + 1)) as f64 / 2.0;

        for i in 0..n {
            if i + 1 < periods {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - periods;
                let slice = self.data.slice(ndarray::s![start..=i]);

                let mut weighted_sum = 0.0;
                for (j, &val) in slice.iter().enumerate() {
                    let weight = (j + 1) as f64;
                    weighted_sum += val * weight;
                }

                result[i] = weighted_sum / total_weight;
            }
        }

        Series::from_array(result)
    }

    /// Sign function (-1, 0, 1)
    pub fn sign(&self) -> Series {
        Series::from_array(self.data.mapv(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }))
    }

    /// Power function (x^exp)
    pub fn power(&self, exp: f64) -> Series {
        Series::from_array(self.data.mapv(|x| x.powf(exp)))
    }

    /// Rolling sum (ts_sum)
    /// window=0 means expanding window (from start to current)
    pub fn ts_sum(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        if window == 0 {
            // Expanding window: sum from start to current
            let mut cumsum = 0.0;
            for i in 0..n {
                cumsum += self.data[i];
                result[i] = cumsum;
            }
        } else {
            // Rolling window
            for i in 0..n {
                if i + 1 < window {
                    result[i] = f64::NAN;
                } else {
                    let start = i + 1 - window;
                    let slice = self.data.slice(ndarray::s![start..=i]);
                    result[i] = slice.iter().sum();
                }
            }
        }

        Series::from_array(result)
    }

    /// Rolling count (ts_count)
    /// window=0 means expanding window (count from start to current)
    pub fn ts_count(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        if window == 0 {
            // Expanding window: count from start to current
            for i in 0..n {
                result[i] = (i + 1) as f64;
            }
        } else {
            // Rolling window: count non-NaN values
            for i in 0..n {
                if i + 1 < window {
                    result[i] = f64::NAN;
                } else {
                    let start = i + 1 - window;
                    let slice = self.data.slice(ndarray::s![start..=i]);
                    let count = slice.iter().filter(|x| !x.is_nan()).count();
                    result[i] = count as f64;
                }
            }
        }

        Series::from_array(result)
    }

    /// Rolling max (ts_max)
    pub fn ts_max(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let slice = self.data.slice(ndarray::s![start..=i]);
                result[i] = slice.iter().cloned().fold(f64::NAN, f64::max);
            }
        }

        Series::from_array(result)
    }

    /// Rolling min (ts_min)
    pub fn ts_min(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let slice = self.data.slice(ndarray::s![start..=i]);
                result[i] = slice.iter().cloned().fold(f64::NAN, f64::min);
            }
        }

        Series::from_array(result)
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
        // Alpha101 Functions
        "ts_rank" => {
            if args.len() != 2 {
                return Err("ts_rank function requires 2 arguments: expression and window".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_rank window must be an integer literal".to_string()),
            };
            Ok(series.ts_rank(window))
        }
        "ts_argmax" => {
            if args.len() != 2 {
                return Err("ts_argmax function requires 2 arguments: expression and window".to_string());
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
                return Err("ts_argmin function requires 2 arguments: expression and window".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_argmin window must be an integer literal".to_string()),
            };
            Ok(series.ts_argmin(window))
        }
        "rank" | "cs_rank" => {
            // Cross-sectional rank - uses single argument
            if args.len() != 1 {
                return Err("rank function requires 1 argument: expression".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            Ok(series.cs_rank())
        }
        "ts_corr" => {
            if args.len() != 3 {
                return Err("ts_corr function requires 3 arguments: expr1, expr2, window".to_string());
            }
            let series1 = evaluate_expr_on_dataframe(&args[0], df)?;
            let series2 = evaluate_expr_on_dataframe(&args[1], df)?;
            let window = match &args[2] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_corr window must be an integer literal".to_string()),
            };
            Ok(series1.ts_corr(&series2, window))
        }
        "ts_cov" => {
            if args.len() != 3 {
                return Err("ts_cov function requires 3 arguments: expr1, expr2, window".to_string());
            }
            let series1 = evaluate_expr_on_dataframe(&args[0], df)?;
            let series2 = evaluate_expr_on_dataframe(&args[1], df)?;
            let window = match &args[2] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_cov window must be an integer literal".to_string()),
            };
            Ok(series1.ts_cov(&series2, window))
        }
        "scale" => {
            if args.len() != 2 {
                return Err("scale function requires 2 arguments: expression and window".to_string());
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
                return Err("decay_linear function requires 2 arguments: expression and periods".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let periods = match &args[1] {
                Expr::Literal(Literal::Integer(p)) => *p as usize,
                _ => return Err("decay_linear periods must be an integer literal".to_string()),
            };
            Ok(series.decay_linear(periods))
        }
        "sign" => {
            if args.len() != 1 {
                return Err("sign function requires 1 argument: expression".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            Ok(series.sign())
        }
        "power" => {
            if args.len() != 2 {
                return Err("power function requires 2 arguments: expression and exponent".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let exp = match &args[1] {
                Expr::Literal(Literal::Float(e)) => *e,
                Expr::Literal(Literal::Integer(e)) => *e as f64,
                _ => return Err("power exponent must be a numeric literal".to_string()),
            };
            Ok(series.power(exp))
        }
        "ts_sum" => {
            if args.len() != 2 {
                return Err("ts_sum function requires 2 arguments: expression and window".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_sum window must be an integer literal".to_string()),
            };
            Ok(series.ts_sum(window))
        }
        "ts_count" => {
            if args.len() != 2 {
                return Err("ts_count function requires 2 arguments: expression and window".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_count window must be an integer literal".to_string()),
            };
            Ok(series.ts_count(window))
        }
        "ts_max" => {
            if args.len() != 2 {
                return Err("ts_max function requires 2 arguments: expression and window".to_string());
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
                return Err("ts_min function requires 2 arguments: expression and window".to_string());
            }
            let series = evaluate_expr_on_dataframe(&args[0], df)?;
            let window = match &args[1] {
                Expr::Literal(Literal::Integer(w)) => *w as usize,
                _ => return Err("ts_min window must be an integer literal".to_string()),
            };
            Ok(series.ts_min(window))
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
