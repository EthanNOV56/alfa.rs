//! Enhanced backtest engine module
//!
//! Provides advanced backtesting capabilities including:
//! - Adjusted prices (前复权) support
//! - Custom fees and volume-based slippage
//! - Configurable long/short position ratios
//! - Additional performance metrics (Sharpe, max drawdown, turnover)

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::f64::NAN;

use crate::WeightMethod;

/// Volume-based slippage configuration
#[derive(Debug, Clone)]
pub struct SlippageConfig {
    /// Threshold for large volume trades
    pub large_volume_threshold: f64,
    /// Slippage rate for large volume trades
    pub large_slippage_rate: f64,
    /// Normal slippage rate
    pub normal_slippage_rate: f64,
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            large_volume_threshold: 1_000_000.0,
            large_slippage_rate: 0.001,
            normal_slippage_rate: 0.0005,
        }
    }
}

/// Fee configuration
#[derive(Debug, Clone)]
pub struct FeeConfig {
    /// Commission rate (e.g., 0.0003 for 0.03%)
    pub commission_rate: f64,
    /// Slippage configuration
    pub slippage: SlippageConfig,
    /// Minimum commission per trade
    pub min_commission: f64,
}

impl Default for FeeConfig {
    fn default() -> Self {
        Self {
            commission_rate: 0.0003,
            slippage: SlippageConfig::default(),
            min_commission: 5.0,
        }
    }
}

/// Position configuration for long/short portfolios
#[derive(Debug, Clone)]
pub struct PositionConfig {
    /// Long position ratio (e.g., 1.0 for 100%)
    pub long_ratio: f64,
    /// Short position ratio (e.g., 0.5 for 50% - spot shorting)
    pub short_ratio: f64,
    /// Whether to use market neutral strategy (long - short)
    pub market_neutral: bool,
}

impl Default for PositionConfig {
    fn default() -> Self {
        Self {
            long_ratio: 1.0,
            short_ratio: 1.0,
            market_neutral: true,  // Default to market neutral long-short strategy
        }
    }
}

/// Enhanced backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Group returns (quantile-based)
    pub group_returns: Array2<f64>,
    /// Group cumulative returns
    pub group_cum_returns: Array2<f64>,
    /// Long-short daily returns
    pub long_short_returns: Array1<f64>,
    /// Long-short cumulative return
    pub long_short_cum_return: f64,
    /// IC series
    pub ic_series: Array1<f64>,
    /// IC mean
    pub ic_mean: f64,
    /// IC IR (Information Ratio)
    pub ic_ir: f64,
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Turnover rate
    pub turnover: f64,
    /// Long-only returns
    pub long_returns: Array1<f64>,
    /// Short-only returns
    pub short_returns: Array1<f64>,
}

/// Backtest engine with enhanced features
pub struct BacktestEngine {
    factor: Array2<f64>,
    returns: Array2<f64>,
    /// Adjustment factors for pre-adjusted prices (前复权)
    adj_factor: Option<Array2<f64>>,
    /// Volume data for slippage calculation
    volume: Option<Array2<f64>>,
    weights: Option<Array2<f64>>,
    quantiles: usize,
    weight_method: WeightMethod,
    long_top_n: usize,
    short_top_n: usize,
    /// Fee configuration
    fee_config: FeeConfig,
    /// Position configuration
    position_config: PositionConfig,
}

impl BacktestEngine {
    /// Create a new enhanced backtest engine
    pub fn new(
        factor: Array2<f64>,
        returns: Array2<f64>,
        quantiles: usize,
        weight_method: WeightMethod,
        long_top_n: usize,
        short_top_n: usize,
        fee_config: FeeConfig,
        position_config: PositionConfig,
        weights: Option<Array2<f64>>,
        adj_factor: Option<Array2<f64>>,
        volume: Option<Array2<f64>>,
    ) -> Self {
        assert_eq!(factor.shape(), returns.shape());
        if let Some(ref w) = weights {
            assert_eq!(w.shape(), factor.shape());
        }
        if let Some(ref adj) = adj_factor {
            assert_eq!(adj.shape(), factor.shape());
        }
        if let Some(ref vol) = volume {
            assert_eq!(vol.shape(), factor.shape());
        }

        Self {
            factor,
            returns,
            adj_factor,
            volume,
            weights,
            quantiles,
            weight_method,
            long_top_n,
            short_top_n,
            fee_config,
            position_config,
        }
    }

    /// Create a new backtest engine with default configs (for backward compatibility)
    pub fn new_simple(
        factor: Array2<f64>,
        returns: Array2<f64>,
        quantiles: usize,
        weight_method: WeightMethod,
        long_top_n: usize,
        short_top_n: usize,
        commission_rate: f64,
        weights: Option<Array2<f64>>,
    ) -> Self {
        let fee_config = FeeConfig {
            commission_rate,
            ..Default::default()
        };

        Self::new(
            factor,
            returns,
            quantiles,
            weight_method,
            long_top_n,
            short_top_n,
            fee_config,
            PositionConfig::default(),
            weights,
            None,
            None,
        )
    }

    /// Run the backtest and return enhanced results
    pub fn run(&self) -> Result<BacktestResult, String> {
        let (n_days, n_assets) = self.factor.dim();

        // Compute quantile groups
        let group_labels = self.compute_quantile_groups()?;

        // Compute group returns with adjusted prices
        let (_, group_returns) = self.compute_group_returns(&group_labels)?;

        // Compute long-short returns with enhanced position config
        let (long_returns, short_returns, long_short_returns) =
            self.compute_long_short_returns(&group_returns);

        // Apply fees (commission + slippage)
        let long_short_returns = self.apply_fees(&long_short_returns, &group_returns);

        // Compute cumulative returns (use log returns for numerical stability)
        let group_cum_returns = self.compute_cumulative_returns(&group_returns);
        let long_short_cum_return = Self::compute_total_return_log(&long_short_returns);

        // Compute IC series
        let (ic_series, ic_mean, ic_ir) = self.compute_ic_series()?;

        // Compute additional metrics
        let total_return = long_short_cum_return;
        let annualized_return = self.compute_annualized_return(total_return, n_days);
        let sharpe_ratio = self.compute_sharpe_ratio(&long_short_returns, n_days);
        let max_drawdown = self.compute_max_drawdown(&long_short_returns);
        let turnover = self.compute_turnover(&group_labels);

        Ok(BacktestResult {
            group_returns,
            group_cum_returns,
            long_short_returns,
            long_short_cum_return,
            ic_series,
            ic_mean,
            ic_ir,
            total_return,
            annualized_return,
            sharpe_ratio,
            max_drawdown,
            turnover,
            long_returns,
            short_returns,
        })
    }

    fn compute_quantile_groups(&self) -> Result<Array2<usize>, String> {
        let (n_days, n_assets) = self.factor.dim();
        let mut groups = Array2::<usize>::zeros((n_days, n_assets));

        for day in 0..n_days {
            let factor_row = self.factor.row(day);
            let mut valid_data: Vec<(usize, f64)> = factor_row
                .iter()
                .enumerate()
                .filter(|&(_, &v)| !v.is_nan())
                .map(|(i, &v)| (i, v))
                .collect();

            if valid_data.len() < self.quantiles {
                continue;
            }

            valid_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let n_valid = valid_data.len();
            let group_size = n_valid / self.quantiles;

            for (group_idx, &(asset_idx, _)) in valid_data.iter().enumerate() {
                let quantile = (group_idx / group_size).min(self.quantiles - 1) + 1;
                groups[[day, asset_idx]] = quantile;
            }
        }

        Ok(groups)
    }

    fn compute_group_returns(
        &self,
        group_labels: &Array2<usize>
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let (n_days, n_assets) = self.factor.dim();
        let mut group_weights = Array2::<f64>::zeros((n_days, self.quantiles));
        let mut group_returns = Array2::<f64>::zeros((n_days - 1, self.quantiles));

        // Get adjusted factor if available
        let adj_factor = self.adj_factor.as_ref();

        for day in 0..(n_days - 1) {
            let labels_today = group_labels.row(day);
            let returns_today = self.returns.row(day);

            for group in 1..=self.quantiles {
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

                // Store group weight
                group_weights[[day, group - 1]] = weights.iter().sum();

                // Apply adjustment factor if available
                let weighted_return: f64 = asset_indices.iter()
                    .zip(weights.iter())
                    .map(|(&idx, &w)| {
                        let ret = returns_today[idx];
                        let adj = adj_factor.map_or(1.0, |adj| adj[[day + 1, idx]]);
                        let adjusted_ret = if adj.is_nan() { ret } else { ret * adj };
                        if adjusted_ret.is_nan() { 0.0 } else { w * adjusted_ret }
                    })
                    .sum();

                group_returns[[day, group - 1]] = weighted_return;
            }
        }

        Ok((group_weights, group_returns))
    }

    fn compute_long_short_returns(&self, group_returns: &Array2<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let n_days = group_returns.dim().0;
        let mut long_returns = Array1::<f64>::zeros(n_days);
        let mut short_returns = Array1::<f64>::zeros(n_days);
        let mut long_short = Array1::<f64>::zeros(n_days);

        let long_groups: Vec<usize> = (self.quantiles - self.long_top_n + 1..=self.quantiles).collect();
        let short_groups: Vec<usize> = (1..=self.short_top_n).collect();

        let long_ratio = self.position_config.long_ratio;
        let short_ratio = self.position_config.short_ratio;
        let market_neutral = self.position_config.market_neutral;

        for day in 0..n_days {
            let long_return: f64 = long_groups.iter()
                .map(|&g| group_returns[[day, g - 1]])
                .sum::<f64>() / long_groups.len() as f64;

            let short_return: f64 = short_groups.iter()
                .map(|&g| group_returns[[day, g - 1]])
                .sum::<f64>() / short_groups.len() as f64;

            // Apply position ratios
            let long_position = long_return * long_ratio;
            let short_position = short_return * short_ratio;

            long_returns[day] = long_position;
            short_returns[day] = -short_position; // Short returns are negative

            if market_neutral {
                // Market neutral: long - short
                long_short[day] = long_position - short_position;
            } else {
                // Directional: long position only
                long_short[day] = long_position;
            }
        }

        (long_returns, short_returns, long_short)
    }

    fn apply_fees(&self, long_short_returns: &Array1<f64>, group_returns: &Array2<f64>) -> Array1<f64> {
        let commission_rate = self.fee_config.commission_rate;
        let min_commission = self.fee_config.min_commission;
        let slippage_config = &self.fee_config.slippage;

        let mut result = long_short_returns.clone();

        // Simple fee calculation: subtract commission from each day
        // In real implementation, this would calculate actual position changes
        for i in 0..result.len() {
            // Apply commission
            let commission = commission_rate.abs();
            result[i] -= commission;

            // Apply slippage if volume is available
            if let Some(ref volume) = self.volume {
                // Get average volume for this day (approximation)
                let day_volumes: Vec<f64> = volume.row(i).iter()
                    .filter(|&&v| !v.is_nan())
                    .cloned()
                    .collect();

                if !day_volumes.is_empty() {
                    let avg_volume: f64 = day_volumes.iter().sum::<f64>() / day_volumes.len() as f64;
                    let slippage_rate = if avg_volume > slippage_config.large_volume_threshold {
                        slippage_config.large_slippage_rate
                    } else {
                        slippage_config.normal_slippage_rate
                    };
                    // Apply slippage (symmetric for long and short)
                    result[i] -= slippage_rate;
                }
            }

            // Apply minimum commission floor
            if result[i] < -min_commission {
                // This would only affect if we track actual trade values
            }
        }

        result
    }

    fn compute_cumulative_returns(&self, daily_returns: &Array2<f64>) -> Array2<f64> {
        let (n_days, n_groups) = daily_returns.dim();
        let mut cum_returns = Array2::<f64>::zeros((n_days, n_groups));

        for g in 0..n_groups {
            // Use log returns for numerical stability
            let mut log_cum = 0.0;
            for d in 0..n_days {
                let r = daily_returns[[d, g]];
                if r.is_nan() {
                    log_cum = f64::NAN;
                } else {
                    // Add small epsilon to avoid log(0)
                    let r_adj = 1.0 + r;
                    if r_adj > 0.0 {
                        log_cum += r_adj.ln();
                    }
                }
                // Convert back from log returns
                cum_returns[[d, g]] = if log_cum.is_nan() { f64::NAN } else { log_cum.exp() - 1.0 };
            }
        }

        cum_returns
    }

    fn compute_ic_series(&self) -> Result<(Array1<f64>, f64, f64), String> {
        let (n_days, n_assets) = self.factor.dim();

        // Parallel computation of IC for each day
        let ic_vec: Vec<f64> = (0..(n_days - 1)).into_par_iter().map(|day| {
            let factor_today = self.factor.row(day);
            let returns_today = self.returns.row(day);

            let mut factor_vals = Vec::new();
            let mut return_vals = Vec::new();

            for asset in 0..n_assets {
                let f = factor_today[asset];
                let r = returns_today[asset];
                if !f.is_nan() && !r.is_nan() {
                    factor_vals.push(f);
                    return_vals.push(r);
                }
            }

            if factor_vals.len() < 2 {
                return NAN;
            }

            Self::pearson_correlation(&factor_vals, &return_vals)
        }).collect();

        let ic_series = Array1::from_vec(ic_vec);

        let valid_ic: Vec<f64> = ic_series.iter()
            .filter(|&&v| !v.is_nan())
            .cloned()
            .collect();

        if valid_ic.is_empty() {
            return Err("No valid IC values".to_string());
        }

        let ic_mean = valid_ic.mean();
        let ic_std = valid_ic.std_dev();
        let ic_ir = if ic_std == 0.0 { NAN } else { ic_mean / ic_std };

        Ok((ic_series, ic_mean, ic_ir))
    }

    fn compute_annualized_return(&self, total_return: f64, n_days: usize) -> f64 {
        if n_days <= 1 {
            return 0.0;
        }
        // Assume 252 trading days per year
        let years = n_days as f64 / 252.0;
        if years <= 0.0 {
            return 0.0;
        }
        (1.0 + total_return).powf(1.0 / years) - 1.0
    }

    fn compute_sharpe_ratio(&self, returns: &Array1<f64>, n_days: usize) -> f64 {
        let valid_returns: Vec<f64> = returns.iter()
            .filter(|&&r| !r.is_nan())
            .cloned()
            .collect();

        if valid_returns.len() < 2 {
            return 0.0;
        }

        let mean = valid_returns.mean();
        let std = valid_returns.std_dev();

        if std == 0.0 {
            return 0.0;
        }

        // Annualize: assume 252 trading days
        let annualized_std = std * (252.0_f64).sqrt();
        mean / std * (252.0_f64).sqrt()
    }

    fn compute_max_drawdown(&self, returns: &Array1<f64>) -> f64 {
        let mut cum = 1.0;
        let mut max_cum = 1.0;
        let mut max_drawdown = 0.0;

        for &r in returns.iter() {
            if r.is_nan() {
                continue;
            }
            cum *= 1.0 + r;
            if cum > max_cum {
                max_cum = cum;
            }
            let drawdown = (max_cum - cum) / max_cum;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    /// Compute total return using log returns for numerical stability
    fn compute_total_return_log(returns: &Array1<f64>) -> f64 {
        let mut log_sum = 0.0;
        for &r in returns.iter() {
            if r.is_nan() {
                continue;
            }
            let r_adj = 1.0 + r;
            if r_adj > 0.0 {
                log_sum += r_adj.ln();
            }
        }
        log_sum.exp() - 1.0
    }

    fn compute_turnover(&self, group_labels: &Array2<usize>) -> f64 {
        // Simplified turnover calculation based on group changes
        let (n_days, n_assets) = group_labels.dim();
        let mut total_turnover = 0.0;
        let mut count = 0;

        for day in 1..n_days {
            let prev_labels = group_labels.row(day - 1);
            let curr_labels = group_labels.row(day);

            for asset in 0..n_assets {
                if prev_labels[asset] != curr_labels[asset] {
                    total_turnover += 1.0;
                }
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        // Annualize (assume 252 trading days)
        total_turnover / count as f64 * 252.0
    }

    fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y).map(|(&a, &b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|&a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|&b| b * b).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 { 0.0 } else { numerator / denominator }
    }
}

/// Stats extension for Vec<f64>
trait StatsExt {
    fn mean(&self) -> f64;
    fn std_dev(&self) -> f64;
}

impl StatsExt for Vec<f64> {
    fn mean(&self) -> f64 {
        if self.is_empty() { 0.0 } else { self.iter().sum::<f64>() / self.len() as f64 }
    }

    fn std_dev(&self) -> f64 {
        if self.len() <= 1 { 0.0 } else {
            let mean = self.mean();
            let variance = self.iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<f64>() / (self.len() - 1) as f64;
            variance.sqrt()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_config_default() {
        let config = SlippageConfig::default();
        assert_eq!(config.large_volume_threshold, 1_000_000.0);
        assert_eq!(config.large_slippage_rate, 0.001);
        assert_eq!(config.normal_slippage_rate, 0.0005);
    }

    #[test]
    fn test_fee_config_default() {
        let config = FeeConfig::default();
        assert_eq!(config.commission_rate, 0.0003);
        assert_eq!(config.min_commission, 5.0);
    }

    #[test]
    fn test_position_config_default() {
        let config = PositionConfig::default();
        assert_eq!(config.long_ratio, 1.0);
        assert_eq!(config.short_ratio, 0.5);
        assert!(!config.market_neutral);
    }

    #[test]
    fn test_backtest_engine_simple() {
        let factor = Array2::from_shape_vec((3, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            4.0, 3.0, 2.0, 1.0,
            2.0, 3.0, 4.0, 1.0,
        ]).unwrap();

        let returns = Array2::from_shape_vec((3, 4), vec![
            0.01, 0.02, 0.03, 0.04,
            0.04, 0.03, 0.02, 0.01,
            0.02, 0.01, 0.03, 0.02,
        ]).unwrap();

        let engine = BacktestEngine::new_simple(
            factor,
            returns,
            4,
            WeightMethod::Equal,
            1,
            1,
            0.001,
            None,
        );

        let result = engine.run().unwrap();
        assert!(result.long_short_cum_return.is_finite());
    }
}
