//! Alpha expression evaluation for time series data
//! 
//! This module provides functionality to evaluate alpha expressions
//! on time series data matrices for use in backtesting.

use crate::expr::{Expr, Literal, BinaryOp, UnaryOp};
use crate::evaluation::EvaluationContext;
use ndarray::{Array2, Array1, ArrayView1, s};
use std::collections::HashMap;

// ============================================================================
// Alpha Expression Evaluator (New implementation)
// ============================================================================

/// Alpha expression evaluator for time series data
#[derive(Clone)]
pub struct AlphaExpressionEvaluator {
    /// Column name to time series data mapping
    data: HashMap<String, Array1<f64>>,
    /// Current position in time series (for recursive evaluation)
    current_idx: usize,
}

impl AlphaExpressionEvaluator {
    /// Create a new evaluator with time series data
    pub fn new(data: HashMap<String, Array1<f64>>) -> Self {
        Self {
            data,
            current_idx: 0,
        }
    }
    
    /// Set the current time index for evaluation
    pub fn set_current_idx(&mut self, idx: usize) {
        self.current_idx = idx;
    }
    
    /// Evaluate an expression at the current time index
    pub fn evaluate(&self, expr: &Expr) -> Result<f64, String> {
        match expr {
            Expr::Literal(lit) => {
                match lit {
                    Literal::Float(f) => Ok(*f),
                    Literal::Integer(i) => Ok(*i as f64),
                    Literal::Boolean(b) => Ok(if *b { 1.0 } else { 0.0 }),
                    Literal::String(_) => Err("String literals not supported in alpha expressions".to_string()),
                    Literal::Null => Ok(f64::NAN),
                }
            }
            Expr::Column(name) => {
                self.get_column_value(name, self.current_idx)
            }
            Expr::BinaryExpr { left, op, right } => {
                let left_val = self.evaluate(left)?;
                let right_val = self.evaluate(right)?;
                eval_binary_op_f64(left_val, *op, right_val)
            }
            Expr::UnaryExpr { op, expr } => {
                let val = self.evaluate(expr)?;
                eval_unary_op_f64(val, *op)
            }
            Expr::FunctionCall { name, args } => {
                self.evaluate_function(name, args)
            }
            _ => Err(format!("Expression type not supported in alpha evaluation: {:?}", expr)),
        }
    }
    
    /// Evaluate a function call
    fn evaluate_function(&self, name: &str, args: &[Expr]) -> Result<f64, String> {
        match name {
            "lag" => self.evaluate_lag(args),
            "diff" => self.evaluate_diff(args),
            "pct_change" => self.evaluate_pct_change(args),
            "moving_average" => self.evaluate_moving_average(args),
            "momentum" => self.evaluate_momentum(args),
            "volatility" => self.evaluate_volatility(args),
            "rolling_mean" => self.evaluate_rolling_mean(args),
            "rolling_std" => self.evaluate_rolling_std(args),
            "ema" => self.evaluate_ema(args),
            _ => Err(format!("Unknown alpha function: {}", name)),
        }
    }
    
    /// Get column value at specific index
    fn get_column_value(&self, column_name: &str, idx: usize) -> Result<f64, String> {
        self.data.get(column_name)
            .and_then(|series| series.get(idx).copied())
            .ok_or_else(|| format!("Column '{}' not found or index {} out of bounds", column_name, idx))
    }
    
    /// Evaluate lag function: lag(expr, periods)
    fn evaluate_lag(&self, args: &[Expr]) -> Result<f64, String> {
        if args.len() != 2 {
            return Err("lag function requires 2 arguments: expression and periods".to_string());
        }
        
        let expr = &args[0];
        let periods = match &args[1] {
            Expr::Literal(Literal::Integer(p)) => *p as usize,
            _ => return Err("lag periods must be an integer literal".to_string()),
        };
        
        if self.current_idx < periods {
            return Ok(f64::NAN);
        }
        
        let mut evaluator = self.clone();
        evaluator.set_current_idx(self.current_idx - periods);
        evaluator.evaluate(expr)
    }
    
    /// Evaluate diff function: diff(expr, periods) = expr - lag(expr, periods)
    fn evaluate_diff(&self, args: &[Expr]) -> Result<f64, String> {
        if args.len() != 2 {
            return Err("diff function requires 2 arguments: expression and periods".to_string());
        }
        
        let expr = &args[0];
        let current = self.evaluate(expr)?;
        
        let periods = match &args[1] {
            Expr::Literal(Literal::Integer(p)) => *p as usize,
            _ => return Err("diff periods must be an integer literal".to_string()),
        };
        
        if self.current_idx < periods {
            return Ok(f64::NAN);
        }
        
        let mut evaluator = self.clone();
        evaluator.set_current_idx(self.current_idx - periods);
        let lagged = evaluator.evaluate(expr)?;
        
        Ok(current - lagged)
    }
    
    /// Evaluate percentage change function: pct_change(expr, periods)
    fn evaluate_pct_change(&self, args: &[Expr]) -> Result<f64, String> {
        if args.len() != 2 {
            return Err("pct_change function requires 2 arguments: expression and periods".to_string());
        }
        
        let expr = &args[0];
        let current = self.evaluate(expr)?;
        
        let periods = match &args[1] {
            Expr::Literal(Literal::Integer(p)) => *p as usize,
            _ => return Err("pct_change periods must be an integer literal".to_string()),
        };
        
        if self.current_idx < periods {
            return Ok(f64::NAN);
        }
        
        let mut evaluator = self.clone();
        evaluator.set_current_idx(self.current_idx - periods);
        let lagged = evaluator.evaluate(expr)?;
        
        if lagged == 0.0 || lagged.is_nan() {
            Ok(f64::NAN)
        } else {
            Ok((current - lagged) / lagged)
        }
    }
    
    /// Evaluate moving average function: moving_average(expr, window)
    fn evaluate_moving_average(&self, args: &[Expr]) -> Result<f64, String> {
        if args.len() != 2 {
            return Err("moving_average function requires 2 arguments: expression and window".to_string());
        }
        
        let expr = &args[0];
        let window = match &args[1] {
            Expr::Literal(Literal::Integer(w)) => *w as usize,
            _ => return Err("moving_average window must be an integer literal".to_string()),
        };
        
        if self.current_idx + 1 < window {
            return Ok(f64::NAN);
        }
        
        let start = self.current_idx + 1 - window;
        let end = self.current_idx + 1;
        
        let mut sum = 0.0;
        let mut count = 0;
        
        for idx in start..end {
            let mut evaluator = self.clone();
            evaluator.set_current_idx(idx);
            match evaluator.evaluate(expr) {
                Ok(val) if !val.is_nan() => {
                    sum += val;
                    count += 1;
                }
                _ => {}
            }
        }
        
        if count == 0 {
            Ok(f64::NAN)
        } else {
            Ok(sum / count as f64)
        }
    }
    
    /// Evaluate momentum function: momentum(expr, periods) = pct_change(expr, periods)
    fn evaluate_momentum(&self, args: &[Expr]) -> Result<f64, String> {
        // momentum is essentially pct_change
        self.evaluate_pct_change(args)
    }
    
    /// Evaluate volatility function: volatility(expr, periods) = standard deviation over periods
    fn evaluate_volatility(&self, args: &[Expr]) -> Result<f64, String> {
        if args.len() != 2 {
            return Err("volatility function requires 2 arguments: expression and periods".to_string());
        }
        
        let expr = &args[0];
        let periods = match &args[1] {
            Expr::Literal(Literal::Integer(p)) => *p as usize,
            _ => return Err("volatility periods must be an integer literal".to_string()),
        };
        
        if self.current_idx + 1 < periods {
            return Ok(f64::NAN);
        }
        
        let start = self.current_idx + 1 - periods;
        let end = self.current_idx + 1;
        
        let mut values = Vec::with_capacity(periods);
        for idx in start..end {
            let mut evaluator = self.clone();
            evaluator.set_current_idx(idx);
            match evaluator.evaluate(expr) {
                Ok(val) if !val.is_nan() => values.push(val),
                _ => {}
            }
        }
        
        if values.len() < 2 {
            return Ok(f64::NAN);
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        Ok(variance.sqrt())
    }
    
    /// Evaluate rolling mean function: rolling_mean(expr, window)
    fn evaluate_rolling_mean(&self, args: &[Expr]) -> Result<f64, String> {
        // rolling_mean is same as moving_average
        self.evaluate_moving_average(args)
    }
    
    /// Evaluate rolling standard deviation function: rolling_std(expr, window)
    fn evaluate_rolling_std(&self, args: &[Expr]) -> Result<f64, String> {
        // rolling_std is same as volatility but with window instead of periods
        self.evaluate_volatility(args)
    }
    
    /// Evaluate exponential moving average: ema(expr, span)
    fn evaluate_ema(&self, args: &[Expr]) -> Result<f64, String> {
        if args.len() != 2 {
            return Err("ema function requires 2 arguments: expression and span".to_string());
        }
        
        let expr = &args[0];
        let span = match &args[1] {
            Expr::Literal(Literal::Integer(s)) => *s as f64,
            _ => return Err("ema span must be an integer literal".to_string()),
        };
        
        let alpha = 2.0 / (span + 1.0);
        
        if self.current_idx == 0 {
            // First value is just the current value
            return self.evaluate(expr);
        }
        
        // Get previous EMA value
        let mut prev_evaluator = self.clone();
        prev_evaluator.set_current_idx(self.current_idx - 1);
        let prev_ema = prev_evaluator.evaluate_ema(args)?;
        
        // Get current value
        let current = self.evaluate(expr)?;
        
        if prev_ema.is_nan() {
            Ok(current)
        } else if current.is_nan() {
            Ok(prev_ema)
        } else {
            Ok(alpha * current + (1.0 - alpha) * prev_ema)
        }
    }
}

/// Evaluate binary operation on f64 values
fn eval_binary_op_f64(left: f64, op: BinaryOp, right: f64) -> Result<f64, String> {
    match op {
        BinaryOp::Add => Ok(left + right),
        BinaryOp::Subtract => Ok(left - right),
        BinaryOp::Multiply => Ok(left * right),
        BinaryOp::Divide => {
            if right == 0.0 {
                Ok(f64::NAN)
            } else {
                Ok(left / right)
            }
        }
        BinaryOp::Modulo => {
            if right == 0.0 {
                Ok(f64::NAN)
            } else {
                Ok(left % right)
            }
        }
        BinaryOp::Equal => Ok(if left == right { 1.0 } else { 0.0 }),
        BinaryOp::NotEqual => Ok(if left != right { 1.0 } else { 0.0 }),
        BinaryOp::GreaterThan => Ok(if left > right { 1.0 } else { 0.0 }),
        BinaryOp::GreaterThanOrEqual => Ok(if left >= right { 1.0 } else { 0.0 }),
        BinaryOp::LessThan => Ok(if left < right { 1.0 } else { 0.0 }),
        BinaryOp::LessThanOrEqual => Ok(if left <= right { 1.0 } else { 0.0 }),
        // Logical operations treat non-zero as true
        BinaryOp::And => Ok(if left != 0.0 && right != 0.0 { 1.0 } else { 0.0 }),
        BinaryOp::Or => Ok(if left != 0.0 || right != 0.0 { 1.0 } else { 0.0 }),
        _ => Err(format!("Binary operator {:?} not supported for f64", op)),
    }
}

/// Evaluate unary operation on f64 values
fn eval_unary_op_f64(val: f64, op: UnaryOp) -> Result<f64, String> {
    match op {
        UnaryOp::Negate => Ok(-val),
        UnaryOp::Not => Ok(if val == 0.0 { 1.0 } else { 0.0 }),
        UnaryOp::Abs => Ok(val.abs()),
        UnaryOp::Sqrt => {
            if val < 0.0 {
                Ok(f64::NAN)
            } else {
                Ok(val.sqrt())
            }
        }
        UnaryOp::Log => {
            if val <= 0.0 {
                Ok(f64::NAN)
            } else {
                Ok(val.ln())
            }
        }
        UnaryOp::Exp => Ok(val.exp()),
        _ => Err(format!("Unary operator {:?} not supported for f64", op)),
    }
}

/// Evaluate an alpha expression on a full time series
pub fn evaluate_alpha_expr_on_series_full(
    expr: &Expr,
    series_data: &HashMap<String, Array1<f64>>,
) -> Result<Array1<f64>, String> {
    let n = series_data.values().next()
        .map(|v| v.len())
        .unwrap_or(0);
    
    let mut result = Array1::<f64>::zeros(n);
    let mut evaluator = AlphaExpressionEvaluator::new(series_data.clone());
    
    for i in 0..n {
        evaluator.set_current_idx(i);
        match evaluator.evaluate(expr) {
            Ok(val) => result[i] = val,
            Err(e) => return Err(format!("Error at index {}: {}", i, e)),
        }
    }
    
    Ok(result)
}

/// Evaluate an alpha expression on a data matrix (multiple assets)
pub fn evaluate_alpha_expr_on_matrix_full(
    expr: &Expr,
    data_matrices: &HashMap<String, Array2<f64>>,
) -> Result<Array2<f64>, String> {
    let (n_days, n_assets) = data_matrices.values().next()
        .map(|m| m.dim())
        .unwrap_or((0, 0));
    
    let mut result = Array2::<f64>::zeros((n_days, n_assets));
    
    for asset_idx in 0..n_assets {
        // Extract series for this asset
        let mut series_data = HashMap::new();
        for (col_name, matrix) in data_matrices {
            let series = matrix.column(asset_idx).to_owned();
            series_data.insert(col_name.clone(), series);
        }
        
        // Evaluate expression for this asset
        let asset_result = evaluate_alpha_expr_on_series_full(expr, &series_data)?;
        
        // Copy result to output matrix
        for day_idx in 0..n_days {
            result[[day_idx, asset_idx]] = asset_result[day_idx];
        }
    }
    
    Ok(result)
}

// ============================================================================
// Legacy Alpha Factor Calculator (For backward compatibility)
// ============================================================================

/// Simple alpha factor calculator for common expressions
pub struct AlphaFactorCalculator {
    /// Column name for price data
    price_column: String,
    /// Column name for volume data (optional)
    volume_column: Option<String>,
}

impl AlphaFactorCalculator {
    /// Create a new calculator with price column name
    pub fn new(price_column: &str) -> Self {
        Self {
            price_column: price_column.to_string(),
            volume_column: None,
        }
    }
    
    /// Set volume column name
    pub fn with_volume(mut self, volume_column: &str) -> Self {
        self.volume_column = Some(volume_column.to_string());
        self
    }
    
    /// Calculate momentum factor (close / lag(close, n) - 1)
    pub fn momentum(&self, prices: &Array2<f64>, period: i64) -> Result<Array2<f64>, String> {
        let (n_days, n_assets) = prices.dim();
        let mut result = Array2::<f64>::zeros((n_days, n_assets));
        
        for asset_idx in 0..n_assets {
            let series = prices.column(asset_idx);
            
            for day_idx in period as usize..n_days {
                let current = series[day_idx];
                let past = series[day_idx - period as usize];
                
                if current.is_nan() || past.is_nan() || past == 0.0 {
                    result[[day_idx, asset_idx]] = f64::NAN;
                } else {
                    result[[day_idx, asset_idx]] = (current / past) - 1.0;
                }
            }
        }
        
        Ok(result)
    }
    
    /// Calculate moving average factor (close / SMA(close, window))
    pub fn moving_average_ratio(&self, prices: &Array2<f64>, window: usize) -> Result<Array2<f64>, String> {
        let (n_days, n_assets) = prices.dim();
        let mut result = Array2::<f64>::zeros((n_days, n_assets));
        
        for asset_idx in 0..n_assets {
            let series = prices.column(asset_idx);
            
            for day_idx in 0..n_days {
                if day_idx < window - 1 {
                    result[[day_idx, asset_idx]] = f64::NAN;
                    continue;
                }
                
                let start = day_idx - window + 1;
                let end = day_idx + 1;
                let window_slice = &series.slice(ndarray::s![start..end]);
                
                let sum: f64 = window_slice.iter().filter(|&&v| !v.is_nan()).sum();
                let count = window_slice.iter().filter(|&&v| !v.is_nan()).count();
                
                if count == 0 || sum == 0.0 {
                    result[[day_idx, asset_idx]] = f64::NAN;
                } else {
                    let sma = sum / count as f64;
                    let current = series[day_idx];
                    result[[day_idx, asset_idx]] = current / sma;
                }
            }
        }
        
        Ok(result)
    }
    
    /// Calculate WCR (Volume Weighted Close Ratio) factor
    /// WCR = VWAP / SMA(close, window)
    /// where VWAP = ∑(close * volume) / ∑volume
    pub fn wcr(&self, prices: &Array2<f64>, volumes: &Array2<f64>, window: usize) -> Result<Array2<f64>, String> {
        let (n_days, n_assets) = prices.dim();
        
        // First calculate VWAP
        let mut vwap = Array2::<f64>::zeros((n_days, n_assets));
        for asset_idx in 0..n_assets {
            let price_series = prices.column(asset_idx);
            let volume_series = volumes.column(asset_idx);
            
            for day_idx in 0..n_days {
                let price = price_series[day_idx];
                let volume = volume_series[day_idx];
                
                if price.is_nan() || volume.is_nan() || volume == 0.0 {
                    vwap[[day_idx, asset_idx]] = f64::NAN;
                } else {
                    vwap[[day_idx, asset_idx]] = price; // Daily VWAP = close price
                    // Note: For proper VWAP, we would need intraday data
                    // Using close price as approximation
                }
            }
        }
        
        // Calculate WCR = VWAP / SMA(close, window)
        self.moving_average_ratio(&vwap, window)
    }
}

/// Extension trait for easier alpha factor calculation on data matrices
pub trait AlphaFactorExt {
    /// Calculate momentum factor
    fn momentum(&self, period: i64) -> Result<Array2<f64>, String>;
    
    /// Calculate moving average ratio factor
    fn moving_average_ratio(&self, window: usize) -> Result<Array2<f64>, String>;
    
    /// Calculate WCR factor (requires volume data)
    fn wcr(&self, volumes: &Array2<f64>, window: usize) -> Result<Array2<f64>, String>;
}

impl AlphaFactorExt for Array2<f64> {
    fn momentum(&self, period: i64) -> Result<Array2<f64>, String> {
        let calculator = AlphaFactorCalculator::new("close");
        calculator.momentum(self, period)
    }
    
    fn moving_average_ratio(&self, window: usize) -> Result<Array2<f64>, String> {
        let calculator = AlphaFactorCalculator::new("close");
        calculator.moving_average_ratio(self, window)
    }
    
    fn wcr(&self, volumes: &Array2<f64>, window: usize) -> Result<Array2<f64>, String> {
        let calculator = AlphaFactorCalculator::new("close");
        calculator.wcr(self, volumes, window)
    }
}