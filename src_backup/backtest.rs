//! Quantile backtesting for alpha factor evaluation
//! 
//! This module implements qcut(N) grouping and long-short portfolio backtesting
//! for evaluating alpha factor performance.

use ndarray::{Array2, Array1};
use rayon::prelude::*;
use std::f64::NAN;

/// Weight allocation method for portfolio construction
#[derive(Debug, Clone, Copy)]
pub enum WeightMethod {
    /// Equal weight within each group
    Equal,
    /// Weighted by external weights (e.g., market cap)
    Weighted,
}

/// Backtest result containing performance metrics
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Daily returns for each quantile group (days × groups)
    pub group_returns: Array2<f64>,
    /// Cumulative returns for each group
    pub group_cum_returns: Array2<f64>,
    /// Daily long-short portfolio return
    pub long_short_returns: Array1<f64>,
    /// Cumulative long-short return
    pub long_short_cum_return: f64,
    /// Information coefficient (IC) series
    pub ic_series: Array1<f64>,
    /// Mean IC
    pub ic_mean: f64,
    /// IC information ratio (mean / std)
    pub ic_ir: f64,
    /// Turnover series (placeholder for future implementation)
    pub turnover: Array1<f64>,
}

/// Backtest engine for evaluating alpha factors
pub struct BacktestEngine {
    factor: Array2<f64>,          // days × assets
    returns: Array2<f64>,         // days × assets
    weights: Option<Array2<f64>>, // days × assets (optional)
    quantiles: usize,             // number of quantile groups
    weight_method: WeightMethod,
    long_top_n: usize,            // number of top groups to long
    short_top_n: usize,           // number of bottom groups to short
    commission_rate: f64,         // one-way commission
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(factor: Array2<f64>, returns: Array2<f64>) -> Self {
        assert_eq!(factor.shape(), returns.shape(), "Factor and returns must have same shape");
        let (n_days, n_assets) = factor.dim();
        assert!(n_days > 0 && n_assets > 0, "Data must have positive dimensions");
        
        Self {
            factor,
            returns,
            weights: None,
            quantiles: 10,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            commission_rate: 0.0,
        }
    }
    
    /// Set external weights for weighted portfolio construction
    pub fn set_weights(&mut self, weights: Array2<f64>) -> &mut Self {
        assert_eq!(weights.shape(), self.factor.shape(), "Weights must have same shape as factor");
        self.weights = Some(weights);
        self
    }
    
    /// Set number of quantile groups (default: 10)
    pub fn set_quantiles(&mut self, quantiles: usize) -> &mut Self {
        assert!(quantiles >= 2, "Quantiles must be at least 2");
        self.quantiles = quantiles;
        self
    }
    
    /// Set weight allocation method (default: Equal)
    pub fn set_weight_method(&mut self, method: WeightMethod) -> &mut Self {
        self.weight_method = method;
        self
    }
    
    /// Set number of top groups to long (default: 1)
    pub fn set_long_top_n(&mut self, n: usize) -> &mut Self {
        assert!(n > 0 && n <= self.quantiles, "long_top_n must be in 1..quantiles");
        self.long_top_n = n;
        self
    }
    
    /// Set number of bottom groups to short (default: 1)
    pub fn set_short_top_n(&mut self, n: usize) -> &mut Self {
        assert!(n > 0 && n <= self.quantiles, "short_top_n must be in 1..quantiles");
        self.short_top_n = n;
        self
    }
    
    /// Set commission rate (default: 0.0)
    pub fn set_commission_rate(&mut self, rate: f64) -> &mut Self {
        assert!(rate >= 0.0, "Commission rate must be non-negative");
        self.commission_rate = rate;
        self
    }
    
    /// Run the backtest and return performance metrics
    pub fn run(&self) -> Result<BacktestResult, String> {
        let (n_days, _n_assets) = self.factor.dim();
        
        // 1. Quantile grouping for each day
        let group_labels = self.compute_quantile_groups()?; // days × assets
        
        // 2. Compute group weights and returns
        let (_group_weights, group_returns) = self.compute_group_returns(&group_labels)?;
        
        // 3. Compute long-short portfolio
        let long_short_returns = self.compute_long_short_returns(&group_returns);
        
        // 4. Compute cumulative returns
        let group_cum_returns = self.compute_cumulative_returns(&group_returns);
        let long_short_cum_return = (1.0 + &long_short_returns).fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0;
        
        // 5. Compute IC series
        let (ic_series, ic_mean, ic_ir) = self.compute_ic_series()?;
        
        // 6. Compute turnover (placeholder)
        let turnover = Array1::zeros(n_days - 1);
        
        Ok(BacktestResult {
            group_returns,
            group_cum_returns,
            long_short_returns,
            long_short_cum_return,
            ic_series,
            ic_mean,
            ic_ir,
            turnover,
        })
    }
    
    /// Compute quantile groups for each cross-section
    fn compute_quantile_groups(&self) -> Result<Array2<usize>, String> {
        let (n_days, n_assets) = self.factor.dim();
        let mut groups = Array2::<usize>::zeros((n_days, n_assets));
        
        // Process each day in parallel
        let results: Vec<Array1<usize>> = (0..n_days).into_par_iter()
            .map(|day| {
                let factor_row = self.factor.row(day);
                
                // Collect non-NaN values with indices
                let mut valid_data: Vec<(usize, f64)> = factor_row
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_nan())
                    .map(|(i, v)| (i, *v))
                    .collect();
                
                if valid_data.len() < self.quantiles {
                    // Not enough data for quantile grouping
                    return Array1::from_elem(n_assets, 0);
                }
                
                // Sort by factor value
                valid_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                
                // Assign quantile labels (1..quantiles)
                let mut result = Array1::<usize>::zeros(n_assets);
                let n_valid = valid_data.len();
                let group_size = n_valid / self.quantiles;
                
                for (group_idx, &(asset_idx, _)) in valid_data.iter().enumerate() {
                    let quantile = (group_idx / group_size).min(self.quantiles - 1) + 1;
                    result[asset_idx] = quantile;
                }
                
                result
            })
            .collect();
        
        // Combine results
        for (day, row) in results.into_iter().enumerate() {
            groups.row_mut(day).assign(&row);
        }
        
        Ok(groups)
    }
    
    /// Compute group returns and weights
    fn compute_group_returns(
        &self,
        group_labels: &Array2<usize>
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let (n_days, n_assets) = self.factor.dim();
        let mut group_weights = Array2::<f64>::zeros((n_days, self.quantiles));
        let mut group_returns = Array2::<f64>::zeros((n_days - 1, self.quantiles));
        
        for day in 0..(n_days - 1) {
            let labels_today = group_labels.row(day);
            let returns_tomorrow = self.returns.row(day + 1);
            
            for group in 1..=self.quantiles {
                // Get indices of assets in this group
                let mut asset_indices = Vec::new();
                for asset in 0..n_assets {
                    if labels_today[asset] == group {
                        asset_indices.push(asset);
                    }
                }
                
                if asset_indices.is_empty() {
                    continue;
                }
                
                // Compute weights
                let weights = match self.weight_method {
                    WeightMethod::Equal => {
                        let w = 1.0 / asset_indices.len() as f64;
                        vec![w; asset_indices.len()]
                    }
                    WeightMethod::Weighted => {
                        if let Some(ref weight_data) = self.weights {
                            let total_weight: f64 = asset_indices.iter()
                                .map(|&idx| weight_data[[day, idx]])
                                .filter(|&w| !w.is_nan())
                                .sum();
                            if total_weight == 0.0 {
                                vec![0.0; asset_indices.len()]
                            } else {
                                asset_indices.iter()
                                    .map(|&idx| weight_data[[day, idx]] / total_weight)
                                    .collect()
                            }
                        } else {
                            return Err("Weighted method requires weight data".to_string());
                        }
                    }
                };
                
                // Store group weight (sum of weights within group)
                group_weights[[day, group - 1]] = weights.iter().sum();
                
                // Compute weighted return
                let weighted_return: f64 = asset_indices.iter()
                    .zip(weights.iter())
                    .map(|(&idx, &w)| {
                        let ret = returns_tomorrow[idx];
                        if ret.is_nan() { 0.0 } else { w * ret }
                    })
                    .sum();
                
                group_returns[[day, group - 1]] = weighted_return;
            }
        }
        
        Ok((group_weights, group_returns))
    }
    
    /// Compute long-short portfolio returns
    fn compute_long_short_returns(&self, group_returns: &Array2<f64>) -> Array1<f64> {
        let n_days = group_returns.dim().0;
        let mut long_short = Array1::<f64>::zeros(n_days);
        
        // Long: top groups, Short: bottom groups
        let long_groups: Vec<usize> = (self.quantiles - self.long_top_n + 1..=self.quantiles).collect();
        let short_groups: Vec<usize> = (1..=self.short_top_n).collect();
        
        for day in 0..n_days {
            let long_return: f64 = long_groups.iter()
                .map(|&g| group_returns[[day, g - 1]])
                .sum::<f64>() / long_groups.len() as f64;
            
            let short_return: f64 = short_groups.iter()
                .map(|&g| group_returns[[day, g - 1]])
                .sum::<f64>() / short_groups.len() as f64;
            
            long_short[day] = long_return - short_return - self.commission_rate;
        }
        
        long_short
    }
    
    /// Compute cumulative returns from daily returns
    fn compute_cumulative_returns(&self, daily_returns: &Array2<f64>) -> Array2<f64> {
        let (n_days, n_groups) = daily_returns.dim();
        let mut cum_returns = Array2::<f64>::zeros((n_days, n_groups));
        
        for g in 0..n_groups {
            let mut cum = 1.0;
            for d in 0..n_days {
                cum *= 1.0 + daily_returns[[d, g]];
                cum_returns[[d, g]] = cum - 1.0;
            }
        }
        
        cum_returns
    }
    
    /// Compute Information Coefficient (IC) series
    fn compute_ic_series(&self) -> Result<(Array1<f64>, f64, f64), String> {
        let (n_days, n_assets) = self.factor.dim();
        let mut ic_series = Array1::<f64>::zeros(n_days - 1);
        
        for day in 0..(n_days - 1) {
            let factor_today = self.factor.row(day);
            let returns_tomorrow = self.returns.row(day + 1);
            
            // Collect valid pairs
            let mut factor_vals = Vec::new();
            let mut return_vals = Vec::new();
            
            for asset in 0..n_assets {
                let f = factor_today[asset];
                let r = returns_tomorrow[asset];
                if !f.is_nan() && !r.is_nan() {
                    factor_vals.push(f);
                    return_vals.push(r);
                }
            }
            
            if factor_vals.len() < 2 {
                ic_series[day] = NAN;
                continue;
            }
            
            // Compute Pearson correlation
            let correlation = self.pearson_correlation(&factor_vals, &return_vals);
            ic_series[day] = correlation;
        }
        
        // Remove NaN values for statistics
        let valid_ic: Vec<f64> = ic_series.iter()
            .filter(|&&v| !v.is_nan())
            .cloned()
            .collect();
        
        if valid_ic.is_empty() {
            return Err("No valid IC values computed".to_string());
        }
        
        let ic_mean = valid_ic.mean();
        let ic_std = valid_ic.std_dev();
        let ic_ir = if ic_std == 0.0 { NAN } else { ic_mean / ic_std };
        
        Ok((ic_series, ic_mean, ic_ir))
    }
    
    /// Compute Pearson correlation coefficient
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y).map(|(&a, &b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|&a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|&b| b * b).sum();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

/// Extension trait for basic statistics on Vec<f64>
trait StatsExt {
    fn mean(&self) -> f64;
    fn std_dev(&self) -> f64;
}

impl StatsExt for Vec<f64> {
    fn mean(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        self.iter().sum::<f64>() / self.len() as f64
    }
    
    fn std_dev(&self) -> f64 {
        if self.len() <= 1 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = self.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>() / (self.len() - 1) as f64;
        variance.sqrt()
    }
}

/// Convenience function for running quantile backtest
pub fn quantile_backtest(
    factor: Array2<f64>,
    returns: Array2<f64>,
    quantiles: usize,
    weight_method: WeightMethod,
    long_top_n: usize,
    short_top_n: usize,
    commission_rate: f64,
    weights: Option<Array2<f64>>,
) -> Result<BacktestResult, String> {
    let mut engine = BacktestEngine::new(factor, returns);
    engine
        .set_quantiles(quantiles)
        .set_weight_method(weight_method)
        .set_long_top_n(long_top_n)
        .set_short_top_n(short_top_n)
        .set_commission_rate(commission_rate);
    
    if let Some(w) = weights {
        engine.set_weights(w);
    }
    
    engine.run()
}

/// Compute information coefficient (IC) statistics between factor and returns
pub fn compute_information_coefficient(
    factor: &Array2<f64>,
    returns: &Array2<f64>,
) -> (f64, f64) {
    let (n_days, n_assets) = factor.dim();
    let mut ic_vals = Vec::new();
    
    for day in 0..(n_days - 1) {
        let factor_today = factor.row(day);
        let returns_tomorrow = returns.row(day + 1);
        
        let mut factor_vals = Vec::new();
        let mut return_vals = Vec::new();
        
        for asset in 0..n_assets {
            let f = factor_today[asset];
            let r = returns_tomorrow[asset];
            if !f.is_nan() && !r.is_nan() {
                factor_vals.push(f);
                return_vals.push(r);
            }
        }
        
        if factor_vals.len() >= 2 {
            let ic = compute_pearson_correlation(&factor_vals, &return_vals);
            if !ic.is_nan() {
                ic_vals.push(ic);
            }
        }
    }
    
    if ic_vals.is_empty() {
        return (0.0, 0.0);
    }
    
    let ic_mean = ic_vals.iter().sum::<f64>() / ic_vals.len() as f64;
    let ic_std = (ic_vals.iter()
        .map(|&x| (x - ic_mean).powi(2))
        .sum::<f64>() / (ic_vals.len() - 1) as f64).sqrt();
    let ic_ir = if ic_std == 0.0 { 0.0 } else { ic_mean / ic_std };
    
    (ic_mean, ic_ir)
}

/// Helper function to compute Pearson correlation
fn compute_pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y).map(|(&a, &b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|&a| a * a).sum();
    let sum_y2: f64 = y.iter().map(|&b| b * b).sum();
    
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}