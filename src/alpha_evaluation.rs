//! Alpha expression evaluation for time series data
//! 
//! This module provides functionality to evaluate alpha expressions
//! on time series data matrices for use in backtesting.

use crate::expr::{Expr, Literal};
use crate::evaluation::EvaluationContext;
use ndarray::{Array2, Array1, ArrayView1};

/// Evaluate an alpha expression on time series data
/// 
/// # Arguments
/// * `expr` - Alpha expression to evaluate
/// * `data` - Data matrix of shape (n_days, n_assets)
/// * `column_name` - Name of the column in the expression that corresponds to the data
/// 
/// # Returns
/// Factor matrix of same shape as input data
pub fn evaluate_alpha_expr_on_matrix(
    expr: &Expr,
    data: &Array2<f64>,
    column_name: &str,
) -> Result<Array2<f64>, String> {
    let (n_days, n_assets) = data.dim();
    let mut result = Array2::<f64>::zeros((n_days, n_assets));
    
    // For each asset (column), evaluate the expression on its time series
    for asset_idx in 0..n_assets {
        let asset_series = data.column(asset_idx);
        let factor_series = evaluate_alpha_expr_on_series(expr, asset_series, column_name)?;
        
        for day_idx in 0..n_days {
            result[[day_idx, asset_idx]] = factor_series[day_idx];
        }
    }
    
    Ok(result)
}

/// Evaluate an alpha expression on a single time series
pub fn evaluate_alpha_expr_on_series(
    expr: &Expr,
    series: ArrayView1<f64>,
    column_name: &str,
) -> Result<Vec<f64>, String> {
    let n_days = series.len();
    let mut result = Vec::with_capacity(n_days);
    
    // Create a sliding window context for time-dependent expressions
    // For now, use a simple per-day evaluation
    for day_idx in 0..n_days {
        let value = series[day_idx];
        
        // Create evaluation context with current value
        let mut ctx = EvaluationContext::new();
        ctx.set_column(column_name.to_string(), Literal::Float(value));
        
        // Add lagged values if needed (simplified - would need proper time series support)
        // For now, just evaluate the expression
        match ctx.evaluate(expr) {
            Ok(Literal::Float(f)) => result.push(f),
            Ok(Literal::Integer(i)) => result.push(i as f64),
            Ok(other) => return Err(format!("Expected numeric result, got {:?}", other)),
            Err(e) => return Err(format!("Evaluation error: {}", e)),
        }
    }
    
    Ok(result)
}

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