//! Time series operations for alpha factor computation
//!
//! This module provides efficient time series operations using ndarray,
//! including window operations, lags, and statistical calculations.

use ndarray::Array1;
use ndarray_stats::{QuantileExt, SummaryStatisticsExt};

/// Time series data structure
#[derive(Debug, Clone)]
pub struct TimeSeries {
    /// Values stored as a 1D array
    data: Array1<f64>,
    /// Optional timestamps (not yet used)
    timestamps: Option<Array1<i64>>,
}

impl TimeSeries {
    /// Create a new time series from a vector
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            data: Array1::from(data),
            timestamps: None,
        }
    }
    
    /// Create a new time series from an ndarray
    pub fn from_array(data: Array1<f64>) -> Self {
        Self {
            data,
            timestamps: None,
        }
    }
    
    /// Get the length of the time series
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if the time series is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Get the underlying data as a slice
    pub fn data(&self) -> &Array1<f64> {
        &self.data
    }
    
    /// Compute lagged series (shift forward by n periods)
    pub fn lag(&self, periods: usize) -> TimeSeries {
        let n = self.data.len();
        if periods >= n {
            // Return series of NaN values
            return TimeSeries::new(vec![f64::NAN; n]);
        }
        
        let mut lagged = vec![f64::NAN; periods];
        lagged.extend_from_slice(&self.data.as_slice().unwrap()[..n - periods]);
        TimeSeries::new(lagged)
    }
    
    /// Compute difference (current value - lagged value)
    pub fn diff(&self, periods: usize) -> TimeSeries {
        let lagged = self.lag(periods);
        let diff_data = &self.data - &lagged.data;
        TimeSeries::from_array(diff_data)
    }
    
    /// Compute percentage change
    pub fn pct_change(&self, periods: usize) -> TimeSeries {
        let lagged = self.lag(periods);
        let diff_data = &self.data - &lagged.data;
        let pct_data = &diff_data / &lagged.data * 100.0;
        TimeSeries::from_array(pct_data)
    }
    
    /// Compute simple moving average
    pub fn moving_average(&self, window: usize) -> TimeSeries {
        let n = self.data.len();
        let mut result = Array1::zeros(n);
        
        for i in 0..n {
            let start = if i + 1 >= window { i + 1 - window } else { 0 };
            let slice = self.data.slice(ndarray::s![start..=i]);
            result[i] = if slice.len() == 0 { f64::NAN } else { slice.mean().unwrap_or(f64::NAN) };
        }
        
        TimeSeries::from_array(result)
    }
    
    /// Compute exponential moving average
    pub fn exponential_moving_average(&self, span: usize) -> TimeSeries {
        let n = self.data.len();
        let alpha = 2.0 / (span as f64 + 1.0);
        let mut result = Array1::zeros(n);
        
        if n > 0 {
            result[0] = self.data[0];
            for i in 1..n {
                result[i] = alpha * self.data[i] + (1.0 - alpha) * result[i - 1];
            }
        }
        
        TimeSeries::from_array(result)
    }
    
    /// Compute rolling standard deviation (volatility)
    pub fn rolling_std(&self, window: usize) -> TimeSeries {
        let n = self.data.len();
        let mut result = Array1::zeros(n);
        
        for i in 0..n {
            let start = if i + 1 >= window { i + 1 - window } else { 0 };
            let slice = self.data.slice(ndarray::s![start..=i]);
            result[i] = if slice.len() == 0 { f64::NAN } else { slice.std(1.0) };
        }
        
        TimeSeries::from_array(result)
    }
    
    /// Compute rolling maximum
    pub fn rolling_max(&self, window: usize) -> TimeSeries {
        let n = self.data.len();
        let mut result = Array1::zeros(n);
        
        for i in 0..n {
            let start = if i + 1 >= window { i + 1 - window } else { 0 };
            let slice = self.data.slice(ndarray::s![start..=i]);
            result[i] = *slice.max().unwrap_or(&f64::NAN);
        }
        
        TimeSeries::from_array(result)
    }
    
    /// Compute rolling minimum
    pub fn rolling_min(&self, window: usize) -> TimeSeries {
        let n = self.data.len();
        let mut result = Array1::zeros(n);
        
        for i in 0..n {
            let start = if i + 1 >= window { i + 1 - window } else { 0 };
            let slice = self.data.slice(ndarray::s![start..=i]);
            result[i] = *slice.min().unwrap_or(&f64::NAN);
        }
        
        TimeSeries::from_array(result)
    }
    
    /// Compute Sharpe ratio (annualized)
    pub fn sharpe_ratio(&self, risk_free_rate: f64, window: usize) -> TimeSeries {
        let returns = self.pct_change(1); // Daily returns
        let excess_returns_data = &returns.data - risk_free_rate / 252.0; // Annual to daily
        let excess_returns = TimeSeries::from_array(excess_returns_data);
        
        let mean_return = excess_returns.moving_average(window);
        let std_return = excess_returns.rolling_std(window);
        
        // Annualize: multiply by sqrt(252) for daily data
        let sharpe_data = &mean_return.data * 252.0 / &std_return.data * 252.0_f64.sqrt();
        TimeSeries::from_array(sharpe_data)
    }
    
    /// Compute correlation with another time series
    pub fn correlation(&self, other: &TimeSeries, window: usize) -> TimeSeries {
        assert_eq!(self.len(), other.len(), "Time series must have same length");
        
        let n = self.data.len();
        let mut result = Array1::zeros(n);
        
        for i in 0..n {
            let start = if i + 1 >= window { i + 1 - window } else { 0 };
            let slice1 = self.data.slice(ndarray::s![start..=i]);
            let slice2 = other.data.slice(ndarray::s![start..=i]);
            
            if slice1.len() < 2 {
                result[i] = f64::NAN;
                continue;
            }
            
            let mean1 = slice1.mean().unwrap_or(0.0);
            let mean2 = slice2.mean().unwrap_or(0.0);
            
            let cov = (&slice1 - mean1).dot(&(&slice2 - mean2)) / (slice1.len() as f64 - 1.0);
            let std1 = slice1.std(1.0);
            let std2 = slice2.std(1.0);
            
            result[i] = cov / (std1 * std2);
        }
        
        TimeSeries::from_array(result)
    }
    
    /// Compute beta (relative to market)
    pub fn beta(&self, market: &TimeSeries, window: usize) -> TimeSeries {
        assert_eq!(self.len(), market.len(), "Time series must have same length");
        
        let self_returns = self.pct_change(1);
        let market_returns = market.pct_change(1);
        
        let n = self_returns.data.len();
        let mut result = Array1::zeros(n);
        
        for i in 0..n {
            let start = if i + 1 >= window { i + 1 - window } else { 0 };
            let slice_self = self_returns.data.slice(ndarray::s![start..=i]);
            let slice_market = market_returns.data.slice(ndarray::s![start..=i]);
            
            if slice_self.len() < 2 {
                result[i] = f64::NAN;
                continue;
            }
            
            let mean_self = slice_self.mean().unwrap_or(0.0);
            let mean_market = slice_market.mean().unwrap_or(0.0);
            
            let cov = (&slice_self - mean_self).dot(&(&slice_market - mean_market)) 
                / (slice_self.len() as f64 - 1.0);
            let var_market = (&slice_market - mean_market).dot(&(&slice_market - mean_market))
                / (slice_self.len() as f64 - 1.0);
            
            result[i] = cov / var_market;
        }
        
        TimeSeries::from_array(result)
    }
    
    /// Compute maximum drawdown
    pub fn max_drawdown(&self, window: usize) -> TimeSeries {
        let n = self.data.len();
        let mut result = Array1::zeros(n);
        
        for i in 0..n {
            let start = if i + 1 >= window { i + 1 - window } else { 0 };
            let slice = self.data.slice(ndarray::s![start..=i]);
            
            if slice.is_empty() {
                result[i] = f64::NAN;
                continue;
            }
            
            let peak = *slice.max().unwrap();
            let current = slice[slice.len() - 1];
            result[i] = (current - peak) / peak.abs();
        }
        
        TimeSeries::from_array(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_timeseries_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(data.clone());
        assert_eq!(ts.len(), 5);
        assert!(!ts.is_empty());
    }
    
    #[test]
    fn test_lag() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(data);
        let lagged = ts.lag(1);
        
        assert!(lagged.data()[0].is_nan());
        assert_abs_diff_eq!(lagged.data()[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(lagged.data()[2], 2.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_diff() {
        let data = vec![1.0, 2.0, 4.0, 7.0, 11.0];
        let ts = TimeSeries::new(data);
        let diff = ts.diff(1);
        
        assert!(diff.data()[0].is_nan());
        assert_abs_diff_eq!(diff.data()[1], 1.0, epsilon = 1e-10); // 2-1
        assert_abs_diff_eq!(diff.data()[2], 2.0, epsilon = 1e-10); // 4-2
        assert_abs_diff_eq!(diff.data()[3], 3.0, epsilon = 1e-10); // 7-4
        assert_abs_diff_eq!(diff.data()[4], 4.0, epsilon = 1e-10); // 11-7
    }
    
    #[test]
    fn test_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(data);
        let ma = ts.moving_average(3);
        
        // First two values should be partial window averages
        assert_abs_diff_eq!(ma.data()[0], 1.0, epsilon = 1e-10); // [1]
        assert_abs_diff_eq!(ma.data()[1], 1.5, epsilon = 1e-10); // [1,2]
        assert_abs_diff_eq!(ma.data()[2], 2.0, epsilon = 1e-10); // [1,2,3]
        assert_abs_diff_eq!(ma.data()[3], 3.0, epsilon = 1e-10); // [2,3,4]
        assert_abs_diff_eq!(ma.data()[4], 4.0, epsilon = 1e-10); // [3,4,5]
    }
    
    #[test]
    fn test_exponential_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(data);
        let ema = ts.exponential_moving_average(3);
        
        // Span=3 gives alpha=0.5
        // EMA1 = 1.0
        // EMA2 = 0.5*2 + 0.5*1 = 1.5
        // EMA3 = 0.5*3 + 0.5*1.5 = 2.25
        // EMA4 = 0.5*4 + 0.5*2.25 = 3.125
        // EMA5 = 0.5*5 + 0.5*3.125 = 4.0625
        assert_abs_diff_eq!(ema.data()[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ema.data()[1], 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(ema.data()[2], 2.25, epsilon = 1e-10);
        assert_abs_diff_eq!(ema.data()[3], 3.125, epsilon = 1e-10);
        assert_abs_diff_eq!(ema.data()[4], 4.0625, epsilon = 1e-10);
    }
    
    #[test]
    fn test_rolling_std() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(data);
        let std = ts.rolling_std(3);
        
        // Std of [1,2,3] = sqrt(((1-2)²+(2-2)²+(3-2)²)/2) = sqrt((1+0+1)/2) = sqrt(1) = 1
        assert_abs_diff_eq!(std.data()[2], 1.0, epsilon = 1e-10);
    }
}