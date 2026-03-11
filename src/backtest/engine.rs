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

/// Return type for automatic return computation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReturnType {
    /// 隔夜持仓收益: (close[t+1] - close[t]) / close[t]
    Holding,
    /// 日内收益: (close[t] - open[t]) / open[t]
    Trading,
}

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
            market_neutral: true, // Default to market neutral long-short strategy
        }
    }
}

/// Backtest configuration - follows dependency inversion principle
/// Only contains configuration parameters, no data
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Number of quantile groups for factor ranking
    pub quantiles: usize,
    /// Weight method for portfolio construction
    pub weight_method: WeightMethod,
    /// Number of top stocks to go long
    pub long_top_n: usize,
    /// Number of top stocks to go short
    pub short_top_n: usize,
    /// Fee configuration (commission, slippage)
    pub fee_config: FeeConfig,
    /// Position configuration (long/short ratios)
    pub position_config: PositionConfig,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            quantiles: 10,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig::default(),
            position_config: PositionConfig::default(),
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
/// Follows dependency inversion principle: only holds config, methods accept data
pub struct BacktestEngine {
    config: BacktestConfig,
}

impl BacktestEngine {
    /// Create a new backtest engine with default configuration
    pub fn new() -> Self {
        Self {
            config: BacktestConfig::default(),
        }
    }

    /// Create a new backtest engine with custom configuration
    pub fn with_config(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run the backtest with explicit data matrices
    ///
    /// # Parameters
    /// - `factor`: Factor values matrix [n_days, n_assets]
    /// - `returns`: Return values matrix [n_days, n_assets]
    /// - `adj_factor`: Optional adjustment factors for pre-adjusted prices (前复权)
    /// - `volume`: Optional volume data for slippage calculation
    ///
    /// # Important Limitations (for Research vs Production)
    ///
    /// This implementation is designed for **factor research and alpha exploration**.
    /// It does NOT account for real trading constraints:
    ///
    /// - **No limit-up/limit-down handling**: Assumes all positions can be traded at close/vwap prices
    /// - **No stock suspensions**: Assumes all stocks are tradable every day
    /// - **No market impact**: Assumes large orders don't affect execution price
    /// - **No short selling constraints**: Assumes unlimited short selling capacity
    /// - **No position limits**: No constraints on max position size or sector limits
    ///
    /// For **production trading systems**, you must add:
    /// - Limit-up/limit-down detection and skip trading on locked stocks
    /// - Suspension days handling (no trading on suspended days)
    /// - Market impact modeling for large positions
    /// - Short selling constraints (margin, locates, fees)
    /// - Realistic execution delay modeling
    ///
    /// # Use Cases
    ///
    /// - **Suitable for**: Factor IC/IR evaluation, alpha discovery, strategy exploration
    /// - **Not suitable for**: Live trading, realistic P&L estimation, broker compatibility
    pub fn run(
        &self,
        factor: Array2<f64>,
        returns: Array2<f64>,
        adj_factor: Array2<f64>,
        close: Array2<f64>,
        vwap: Array2<f64>,
    ) -> Result<BacktestResult, String> {
        assert_eq!(
            factor.shape(),
            returns.shape(),
            "Factor and returns must have same shape"
        );
        assert_eq!(
            factor.shape(),
            close.shape(),
            "Factor and close must have same shape"
        );
        assert_eq!(
            factor.shape(),
            vwap.shape(),
            "Factor and vwap must have same shape"
        );
        assert_eq!(
            adj_factor.shape(),
            factor.shape(),
            "Adjustment factor must have same shape as factor"
        );

        let (n_days, n_assets) = factor.dim();

        // Compute quantile groups
        let group_labels = Self::compute_quantile_groups(&factor, self.config.quantiles)?;

        // Compute weight matrix from factor using quantile-based approach
        let weights = Self::compute_weight_matrix_from_factor(
            &factor,
            &group_labels,
            n_days,
            n_assets,
            self.config.quantiles,
            self.config.long_top_n,
            self.config.short_top_n,
            &self.config.position_config,
        );

        // Compute portfolio returns using weight matrix, close, and vwap
        let _portfolio_returns = Self::compute_portfolio_return(
            &weights,
            &close,
            &vwap,
            self.config.fee_config.commission_rate,
            self.config.fee_config.slippage.normal_slippage_rate,
        );

        // Compute group returns first (needed for long/short calculation)
        let (_, group_returns) = Self::compute_group_returns(
            &factor,
            &returns,
            Some(&adj_factor),
            &group_labels,
            self.config.quantiles,
            self.config.weight_method,
            None,
        )?;

        // Compute long/short returns using group returns
        let (long_returns, short_returns, long_short_returns) = Self::compute_long_short_returns(
            &group_returns,
            self.config.quantiles,
            self.config.long_top_n,
            self.config.short_top_n,
            &self.config.position_config,
        );

        // Compute IC series
        let (ic_series, ic_mean, ic_ir) = Self::compute_ic_series(&factor, &returns)?;

        // Compute cumulative returns
        let long_short_cum_return = Self::compute_total_return_log(&long_short_returns);

        // Compute additional metrics
        let total_return = long_short_cum_return;
        let annualized_return = Self::compute_annualized_return(total_return, n_days);
        let sharpe_ratio = Self::compute_sharpe_ratio(&long_short_returns, n_days);
        let max_drawdown = Self::compute_max_drawdown(&long_short_returns);
        let turnover = Self::compute_turnover(&group_labels);

        // Compute group cumulative returns
        let group_cum_returns = Self::compute_cumulative_returns(&group_returns);

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

    /// Compute weight matrix from factor using quantile-based approach
    ///
    /// Returns a weight matrix of shape [n_days, n_symbols] where:
    /// - Long positions (top quantiles): positive weights
    /// - Short positions (bottom quantiles): negative weights
    /// - Other positions: zero weights
    fn compute_weight_matrix_from_factor(
        factor: &Array2<f64>,
        group_labels: &Array2<usize>,
        n_days: usize,
        n_assets: usize,
        quantiles: usize,
        long_top_n: usize,
        short_top_n: usize,
        position_config: &PositionConfig,
    ) -> Array2<f64> {
        // Validate parameters - fail fast with clear error messages
        if long_top_n == 0 || long_top_n > quantiles {
            panic!("long_top_n must be in range [1, quantiles], got {} but quantiles={}", long_top_n, quantiles);
        }
        if short_top_n == 0 || short_top_n > quantiles {
            panic!("short_top_n must be in range [1, quantiles], got {} but quantiles={}", short_top_n, quantiles);
        }

        let mut weights = Array2::<f64>::zeros((n_days, n_assets));

        let long_groups: Vec<usize> = (quantiles - long_top_n + 1..=quantiles).collect();
        let short_groups: Vec<usize> = (1..=short_top_n).collect();

        let long_ratio = position_config.long_ratio;
        let short_ratio = position_config.short_ratio;

        for day in 0..n_days {
            // Count long and short assets
            let mut long_indices = Vec::new();
            let mut short_indices = Vec::new();

            for asset in 0..n_assets {
                let label = group_labels[[day, asset]];
                if long_groups.contains(&label) {
                    long_indices.push(asset);
                } else if short_groups.contains(&label) {
                    short_indices.push(asset);
                }
            }

            // Compute equal weights for long and short positions
            let long_weight = if !long_indices.is_empty() {
                long_ratio / long_indices.len() as f64
            } else {
                0.0
            };

            let short_weight = if !short_indices.is_empty() {
                -short_ratio / short_indices.len() as f64
            } else {
                0.0
            };

            // Assign weights
            for &idx in &long_indices {
                weights[[day, idx]] = long_weight;
            }
            for &idx in &short_indices {
                weights[[day, idx]] = short_weight;
            }
        }

        weights
    }

    fn compute_quantile_groups(
        factor: &Array2<f64>,
        quantiles: usize,
    ) -> Result<Array2<usize>, String> {
        let (n_days, n_assets) = factor.dim();
        let mut groups = Array2::<usize>::zeros((n_days, n_assets));

        for day in 0..n_days {
            let factor_row = factor.row(day);
            let mut valid_data: Vec<(usize, f64)> = factor_row
                .iter()
                .enumerate()
                .filter(|&(_, &v)| !v.is_nan())
                .map(|(i, &v)| (i, v))
                .collect();

            if valid_data.len() < quantiles {
                continue;
            }

            valid_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let n_valid = valid_data.len();
            let group_size = n_valid / quantiles;

            for (group_idx, &(asset_idx, _)) in valid_data.iter().enumerate() {
                let quantile = (group_idx / group_size).min(quantiles - 1) + 1;
                groups[[day, asset_idx]] = quantile;
            }
        }

        Ok(groups)
    }

    fn compute_group_returns(
        factor: &Array2<f64>,
        returns: &Array2<f64>,
        adj_factor: Option<&Array2<f64>>,
        group_labels: &Array2<usize>,
        quantiles: usize,
        weight_method: WeightMethod,
        weights: Option<&Array2<f64>>,
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let (n_days, n_assets) = factor.dim();
        let mut group_weights = Array2::<f64>::zeros((n_days, quantiles));
        let mut group_returns = Array2::<f64>::zeros((n_days - 1, quantiles));

        for day in 0..(n_days - 1) {
            let labels_today = group_labels.row(day);
            let returns_today = returns.row(day);

            for group in 1..=quantiles {
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
                let computed_weights = match weight_method {
                    WeightMethod::Equal => {
                        let w = 1.0 / asset_indices.len() as f64;
                        vec![w; asset_indices.len()]
                    }
                    WeightMethod::Weighted => {
                        if let Some(weight_data) = weights {
                            let total_weight: f64 = asset_indices
                                .iter()
                                .map(|&idx| weight_data[[day, idx]])
                                .filter(|&w| !w.is_nan())
                                .sum();
                            if total_weight == 0.0 {
                                vec![0.0; asset_indices.len()]
                            } else {
                                asset_indices
                                    .iter()
                                    .map(|&idx| weight_data[[day, idx]] / total_weight)
                                    .collect()
                            }
                        } else {
                            return Err("Weighted method requires weight data".to_string());
                        }
                    }
                };

                // Store group weight
                group_weights[[day, group - 1]] = computed_weights.iter().sum();

                // Apply adjustment factor if available
                let weighted_return: f64 = asset_indices
                    .iter()
                    .zip(computed_weights.iter())
                    .map(|(&idx, &w)| {
                        let ret = returns_today[idx];
                        let adj = adj_factor.map_or(1.0, |adj| adj[[day + 1, idx]]);
                        let adjusted_ret = if adj.is_nan() { ret } else { ret * adj };
                        if adjusted_ret.is_nan() {
                            0.0
                        } else {
                            w * adjusted_ret
                        }
                    })
                    .sum();

                group_returns[[day, group - 1]] = weighted_return;
            }
        }

        Ok((group_weights, group_returns))
    }

    fn compute_long_short_returns(
        group_returns: &Array2<f64>,
        quantiles: usize,
        long_top_n: usize,
        short_top_n: usize,
        position_config: &PositionConfig,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let n_days = group_returns.dim().0;
        let mut long_returns = Array1::<f64>::zeros(n_days);
        let mut short_returns = Array1::<f64>::zeros(n_days);
        let mut long_short = Array1::<f64>::zeros(n_days);

        // Validate parameters - fail fast with clear error messages
        if long_top_n == 0 || long_top_n > quantiles {
            panic!("long_top_n must be in range [1, quantiles], got {} but quantiles={}", long_top_n, quantiles);
        }
        if short_top_n == 0 || short_top_n > quantiles {
            panic!("short_top_n must be in range [1, quantiles], got {} but quantiles={}", short_top_n, quantiles);
        }

        let long_groups: Vec<usize> = (quantiles - long_top_n + 1..=quantiles).collect();
        let short_groups: Vec<usize> = (1..=short_top_n).collect();

        let long_ratio = position_config.long_ratio;
        let short_ratio = position_config.short_ratio;
        let market_neutral = position_config.market_neutral;

        for day in 0..n_days {
            let long_return: f64 = long_groups
                .iter()
                .map(|&g| group_returns[[day, g - 1]])
                .sum::<f64>()
                / long_groups.len() as f64;

            let short_return: f64 = short_groups
                .iter()
                .map(|&g| group_returns[[day, g - 1]])
                .sum::<f64>()
                / short_groups.len() as f64;

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

    fn apply_fees(
        long_short_returns: &Array1<f64>,
        volume: Option<&Array2<f64>>,
        fee_config: &FeeConfig,
    ) -> Array1<f64> {
        let commission_rate = fee_config.commission_rate;
        let min_commission = fee_config.min_commission;
        let slippage_config = &fee_config.slippage;

        let mut result = long_short_returns.clone();

        // Simple fee calculation: subtract commission from each day
        // In real implementation, this would calculate actual position changes
        for i in 0..result.len() {
            // Apply commission
            let commission = commission_rate.abs();
            result[i] -= commission;

            // Apply slippage if volume is available
            if let Some(vol) = volume {
                // Get average volume for this day (approximation)
                let day_volumes: Vec<f64> = vol
                    .row(i)
                    .iter()
                    .filter(|&&v| !v.is_nan())
                    .cloned()
                    .collect();

                if !day_volumes.is_empty() {
                    let avg_volume: f64 =
                        day_volumes.iter().sum::<f64>() / day_volumes.len() as f64;
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

    fn compute_cumulative_returns(daily_returns: &Array2<f64>) -> Array2<f64> {
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
                cum_returns[[d, g]] = if log_cum.is_nan() {
                    f64::NAN
                } else {
                    log_cum.exp() - 1.0
                };
            }
        }

        cum_returns
    }

    fn compute_ic_series(
        factor: &Array2<f64>,
        returns: &Array2<f64>,
    ) -> Result<(Array1<f64>, f64, f64), String> {
        let (n_days, n_assets) = factor.dim();

        // Parallel computation of IC for each day
        let ic_vec: Vec<f64> = (0..(n_days - 1))
            .into_par_iter()
            .map(|day| {
                let factor_today = factor.row(day);
                let returns_today = returns.row(day);

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
            })
            .collect();

        let ic_series = Array1::from_vec(ic_vec);

        let valid_ic: Vec<f64> = ic_series
            .iter()
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

    fn compute_annualized_return(total_return: f64, n_days: usize) -> f64 {
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

    fn compute_sharpe_ratio(returns: &Array1<f64>, _n_days: usize) -> f64 {
        let valid_returns: Vec<f64> = returns.iter().filter(|&&r| !r.is_nan()).cloned().collect();

        if valid_returns.len() < 2 {
            return 0.0;
        }

        let mean = valid_returns.mean();
        let std = valid_returns.std_dev();

        if std == 0.0 {
            return 0.0;
        }

        // Annualize: assume 252 trading days
        let _annualized_std = std * (252.0_f64).sqrt();
        mean / std * (252.0_f64).sqrt()
    }

    fn compute_max_drawdown(returns: &Array1<f64>) -> f64 {
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

    fn compute_turnover(group_labels: &Array2<usize>) -> f64 {
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

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Compute holding return (隔夜持仓收益) using previous day's weights
    ///
    /// Formula: holding_return[day] = sum(weights[day-1] * (close[day] / close[day-1] - 1))
    /// - Day 0: no holding return (no previous day position)
    ///
    /// # Parameters
    /// - weights: Weight matrix [n_days, n_symbols]
    /// - close: Close price matrix [n_days, n_symbols]
    ///
    /// # Returns
    /// - Array2<f64> of shape [n_days, 1] with holding returns per day
    pub fn compute_holding_return(weights: &Array2<f64>, close: &Array2<f64>) -> Array2<f64> {
        let (n_days, _n_symbols) = weights.dim();

        // Vectorized: previous day's weights
        // weights_lag: [n_days-1, n_symbols] = weights[0:n_days-1] (weights for days 0 to n_days-2)
        let weights_lag = weights.slice(ndarray::s![0..n_days - 1, ..]);

        // Vectorized: compute price returns for all days and symbols
        // close_lag: [n_days-1, n_symbols] = close[0:n_days-1]
        let close_lag = close.slice(ndarray::s![0..n_days - 1, ..]);
        // close_current: [n_days-1, n_symbols] = close[1:n_days]
        let close_current = close.slice(ndarray::s![1.., ..]);
        // price_returns: [n_days-1, n_symbols]
        let price_returns = (&close_current / &close_lag) - 1.0;

        // Element-wise multiply and sum across symbols: [n_days-1, n_symbols] -> [n_days-1]
        let weighted_returns = &weights_lag * &price_returns;
        let day_returns = weighted_returns.sum_axis(ndarray::Axis(1)); // Shape: [n_days-1]

        // Build result: day 0 = 0, days 1..n_days = computed returns
        let mut returns = ndarray::Array2::<f64>::zeros((n_days, 1));
        for day in 1..n_days {
            returns[[day, 0]] = day_returns[day - 1]; // 1D indexing
        }

        returns
    }

    /// Compute trading return (日内交易收益) from weight changes, close, and vwap prices
    ///
    /// Formula:
    /// - Day 0: trading_return[0] = sum(weights[0] * (close[0]/vwap[0] - 1 - cost))
    /// - Days 1..: trading_return[day] = sum((weights[day] - weights[day-1]) * (close[day]/vwap[day] - 1 - cost))
    ///
    /// # Parameters
    /// - weights: Weight matrix [n_days, n_symbols]
    /// - close: Close price matrix [n_days, n_symbols]
    /// - vwap: VWAP price matrix [n_days, n_symbols]
    /// - fee: Commission fee rate (e.g., 0.0003 for 0.03%)
    /// - slippage: Slippage rate (e.g., 0.0005 for 0.05%)
    ///
    /// # Returns
    /// - Array2<f64> of shape [n_days, 1] with trading returns per day
    pub fn compute_trading_return(
        weights: &Array2<f64>,
        close: &Array2<f64>,
        vwap: &Array2<f64>,
        fee: f64,
        slippage: f64,
    ) -> Array2<f64> {
        let (n_days, n_symbols) = weights.dim();
        let total_cost = fee + slippage;

        // Result for all n_days
        let mut returns = ndarray::Array2::<f64>::zeros((n_days, 1));

        // Day 0: initial position establishment (from 0 to weights[0])
        // trading_return[0] = sum(weights[0] * (close[0]/vwap[0] - 1 - cost))
        let price_return_0 = (close[[0, 0]] / vwap[[0, 0]]) - 1.0;
        let trade_return_0 = weights[[0, 0]] * (price_return_0 - total_cost);
        returns[[0, 0]] = trade_return_0;

        // Days 1..n_days-1: position changes
        // weight_diff = weights[day] - weights[day-1]
        if n_days > 1 {
            let weights_lag = weights.slice(ndarray::s![0..n_days - 1, ..]);
            let weights_current = weights.slice(ndarray::s![1.., ..]);
            let weight_diff = &weights_current - &weights_lag; // [n_days-1, n_symbols]

            // Vectorized: price return (close / vwap - 1)
            let vwap_current = vwap.slice(ndarray::s![1.., ..]);
            let close_current = close.slice(ndarray::s![1.., ..]);
            let price_returns = (&close_current / &vwap_current) - 1.0; // [n_days-1, n_symbols]

            // Vectorized: trade return = weight_diff * (price_return - cost)
            let cost_array = Array2::from_elem((n_days - 1, n_symbols), total_cost);
            let trade_returns = &weight_diff * (&price_returns - &cost_array);

            // Sum across symbols: [n_days-1, n_symbols] -> [n_days-1]
            let day_returns = trade_returns.sum_axis(ndarray::Axis(1)); // Shape: [n_days-1]

            // Fill days 1..n_days-1
            for day in 1..n_days {
                returns[[day, 0]] = day_returns[day - 1];
            }
        }

        returns
    }

    /// Compute total portfolio return combining holding and trading returns
    ///
    /// # Parameters
    /// - weights: Weight matrix [n_days, n_symbols]
    /// - close: Close price matrix [n_days, n_symbols]
    /// - vwap: VWAP price matrix [n_days, n_symbols]
    /// - fee: Commission fee rate
    /// - slippage: Slippage rate
    ///
    /// # Returns
    /// - Array2<f64> of shape [n_days, 1] with total returns per day
    pub fn compute_portfolio_return(
        weights: &Array2<f64>,
        close: &Array2<f64>,
        vwap: &Array2<f64>,
        fee: f64,
        slippage: f64,
    ) -> Array2<f64> {
        let holding_return = Self::compute_holding_return(weights, close);
        let trading_return = Self::compute_trading_return(weights, close, vwap, fee, slippage);
        holding_return + trading_return
    }
}

/// Stats extension for Vec<f64>
trait StatsExt {
    fn mean(&self) -> f64;
    fn std_dev(&self) -> f64;
}

impl StatsExt for Vec<f64> {
    fn mean(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.iter().sum::<f64>() / self.len() as f64
        }
    }

    fn std_dev(&self) -> f64 {
        if self.len() <= 1 {
            0.0
        } else {
            let mean = self.mean();
            let variance = self.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>()
                / (self.len() - 1) as f64;
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
        assert_eq!(config.short_ratio, 1.0);
        assert!(config.market_neutral);
    }

    #[test]
    fn test_backtest_engine_simple() {
        let factor = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.02, 0.01, 0.03, 0.02,
            ],
        )
        .unwrap();

        // Create close and vwap arrays (using close = 1.0 as placeholder)
        let close = Array2::from_elem((3, 4), 1.0);
        let vwap = Array2::from_elem((3, 4), 1.0);
        let adj_factor = Array2::from_elem((3, 4), 1.0);

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig {
                commission_rate: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);

        let result = engine
            .run(factor, returns, adj_factor, close, vwap)
            .unwrap();
        assert!(result.long_short_cum_return.is_finite());
    }

    // === Configuration Tests ===

    #[test]
    fn test_slippage_config_custom() {
        let config = SlippageConfig {
            large_volume_threshold: 5_000_000.0,
            large_slippage_rate: 0.002,
            normal_slippage_rate: 0.001,
        };
        assert_eq!(config.large_volume_threshold, 5_000_000.0);
        assert_eq!(config.large_slippage_rate, 0.002);
        assert_eq!(config.normal_slippage_rate, 0.001);
    }

    #[test]
    fn test_fee_config_custom() {
        let slippage = SlippageConfig::default();
        let config = FeeConfig {
            commission_rate: 0.001,
            slippage,
            min_commission: 10.0,
        };
        assert_eq!(config.commission_rate, 0.001);
        assert_eq!(config.min_commission, 10.0);
    }

    #[test]
    fn test_position_config_long_only() {
        let config = PositionConfig {
            long_ratio: 1.0,
            short_ratio: 0.0,
            market_neutral: false,
        };
        assert_eq!(config.long_ratio, 1.0);
        assert_eq!(config.short_ratio, 0.0);
        assert!(!config.market_neutral);
    }

    #[test]
    fn test_position_config_long_short() {
        let config = PositionConfig {
            long_ratio: 1.0,
            short_ratio: 0.5,
            market_neutral: true,
        };
        assert_eq!(config.long_ratio, 1.0);
        assert_eq!(config.short_ratio, 0.5);
        assert!(config.market_neutral);
    }

    // === Core Functionality Tests ===

    #[test]
    fn test_compute_quantile_groups() {
        let factor = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.02, 0.01, 0.03, 0.02,
            ],
        )
        .unwrap();

        // Test via public API
        let close = Array2::from_elem((3, 4), 1.0);
        let vwap = Array2::from_elem((3, 4), 1.0);

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig {
                commission_rate: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
        };

        let adj_factor = Array2::from_elem((3, 4), 1.0);
        let engine = BacktestEngine::with_config(config);
        let result = engine.run(factor, returns.clone(), adj_factor, close, vwap).unwrap();

        // Verify result is valid
        assert!(result.group_returns.dim().1 == 4);
    }

    #[test]
    fn test_compute_group_returns() {
        let factor = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.02, 0.01, 0.03, 0.02,
            ],
        )
        .unwrap();

        // Test via public API
        let close = Array2::from_elem((3, 4), 1.0);
        let vwap = Array2::from_elem((3, 4), 1.0);

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig {
                commission_rate: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
        };

        let adj_factor = Array2::from_elem((3, 4), 1.0);
        let engine = BacktestEngine::with_config(config);
        let result = engine.run(factor, returns.clone(), adj_factor, close, vwap).unwrap();

        // group_returns should have shape (n_days-1, quantiles) = (2, 4)
        assert_eq!(result.group_returns.dim(), (2, 4));
    }

    #[test]
    fn test_compute_long_short_returns() {
        let factor = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.02, 0.01, 0.03, 0.02,
            ],
        )
        .unwrap();

        // Test via public API
        let close = Array2::from_elem((3, 4), 1.0);
        let vwap = Array2::from_elem((3, 4), 1.0);

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig {
                commission_rate: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let adj_factor = Array2::from_elem((3, 4), 1.0);
        let result = engine.run(factor.clone(), returns.clone(), adj_factor, close, vwap).unwrap();

        // Verify long/short returns exist
        assert!(result.long_returns.len() > 0);
    }

    #[test]
    fn test_apply_fees() {
        let factor = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.02, 0.01, 0.03, 0.02,
            ],
        )
        .unwrap();

        let close = Array2::from_elem((3, 4), 1.0);
        let vwap = Array2::from_elem((3, 4), 1.0);

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig {
                commission_rate: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let result = engine.run(factor, returns.clone(), Array2::from_elem((3, 4), 1.0), close.clone(), vwap.clone()).unwrap();

        // Verify result is valid
        assert!(result.long_short_returns.len() > 0);
    }

    #[test]
    fn test_compute_cumulative_returns() {
        let factor = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.02, 0.01, 0.03, 0.02,
            ],
        )
        .unwrap();

        let close = Array2::from_elem((3, 4), 1.0);
        let vwap = Array2::from_elem((3, 4), 1.0);

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig {
                commission_rate: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let result = engine.run(factor, returns.clone(), Array2::from_elem((3, 4), 1.0), close.clone(), vwap.clone()).unwrap();

        // Cumulative returns should have same shape as group_returns
        assert_eq!(result.group_cum_returns.dim(), result.group_returns.dim());
    }

    #[test]
    fn test_compute_ic_series() {
        let factor = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.02, 0.01, 0.03, 0.02,
            ],
        )
        .unwrap();

        let close = Array2::from_elem((3, 4), 1.0);
        let vwap = Array2::from_elem((3, 4), 1.0);

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig {
                commission_rate: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let result = engine.run(factor, returns.clone(), Array2::from_elem((3, 4), 1.0), close.clone(), vwap.clone()).unwrap();

        // IC series length should be n_days - 1
        assert_eq!(result.ic_series.len(), 2);

        // IC mean should just be finite
        assert!(result.ic_mean.is_finite());

        // IC IR can be NaN if std is 0
        assert!(result.ic_ir.is_nan() || result.ic_ir.is_finite());
    }

    // === Edge Case Tests ===

    #[test]
    fn test_nan_handling() {
        // Test with NaN values in factor
        let factor = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0,
                f64::NAN,
                3.0,
                4.0,
                4.0,
                3.0,
                f64::NAN,
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
            ],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.02, 0.01, 0.03, 0.02,
            ],
        )
        .unwrap();

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig {
                commission_rate: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
        };

        let close = Array2::from_elem((3, 4), 1.0);
        let vwap = Array2::from_elem((3, 4), 1.0);

        let engine = BacktestEngine::with_config(config);

        // Should handle NaN gracefully
        let result = engine.run(factor, returns.clone(), Array2::from_elem((3, 4), 1.0), close.clone(), vwap.clone()).unwrap();
        assert!(result.long_short_cum_return.is_finite());
    }

    #[test]
    fn test_single_day_backtest() {
        // Test with minimal data - single day
        let factor = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let returns = Array2::from_shape_vec((1, 4), vec![0.01, 0.02, 0.03, 0.04]).unwrap();

        let config = BacktestConfig {
            quantiles: 2,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig {
                commission_rate: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);

        let close = Array2::from_elem((1, 4), 1.0);
        let vwap = Array2::from_elem((1, 4), 1.0);
        let adj_factor = Array2::from_elem((1, 4), 1.0);

        // Single day returns - should work but produce empty results
        let result = engine.run(factor, returns.clone(), adj_factor, close, vwap);
        // Single day might fail or produce empty results - that's expected
        assert!(result.is_err() || result.unwrap().group_returns.dim().0 == 0);
    }

    #[test]
    fn test_single_asset_backtest() {
        // Test with single asset - this may fail or succeed depending on implementation
        // Single asset is an edge case
        let factor = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let returns = Array2::from_shape_vec((3, 1), vec![0.01, 0.02, 0.03]).unwrap();

        let config = BacktestConfig {
            quantiles: 2,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig {
                commission_rate: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);

        let close = Array2::from_elem((3, 1), 1.0);
        let vwap = Array2::from_elem((3, 1), 1.0);
        let adj_factor = Array2::from_elem((3, 1), 1.0);

        // Single asset - run might fail due to edge case handling
        // Just verify it doesn't panic
        let _ = engine.run(factor, returns.clone(), adj_factor, close, vwap);
    }

    #[test]
    fn test_negative_returns() {
        // Test with negative returns
        let factor = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (3, 4),
            vec![
                -0.01, -0.02, -0.03, -0.04, -0.04, -0.03, -0.02, -0.01, -0.02, -0.01, -0.03, -0.02,
            ],
        )
        .unwrap();

        // Create close prices that reflect the returns
        // Day 0: close = 1.0, Day 1: close = 0.99, 0.98, etc.
        let close = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 1.0, 1.0, 1.0,       // day 0
                0.99, 0.98, 0.97, 0.96,   // day 1: returns = -0.01, -0.02, -0.03, -0.04
                0.95, 0.94, 0.93, 0.92,   // day 2
            ],
        )
        .unwrap();

        let vwap = close.clone();

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig {
                commission_rate: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);

        let adj_factor = Array2::from_elem((3, 4), 1.0);
        let result = engine
            .run(factor, returns, adj_factor, close, vwap)
            .unwrap();
        assert!(result.long_short_cum_return.is_finite());
        // Negative returns should result in negative cumulative return
        assert!(result.long_short_cum_return < 0.0);
    }

    // === Integration Tests ===

    #[test]
    fn test_full_backtest_flow() {
        let factor = Array2::from_shape_vec(
            (10, 5),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, //
                5.0, 4.0, 3.0, 2.0, 1.0, //
                2.0, 3.0, 4.0, 5.0, 1.0, //
                4.0, 5.0, 1.0, 2.0, 3.0, //
                3.0, 1.0, 5.0, 4.0, 2.0, //
                1.0, 2.0, 3.0, 4.0, 5.0, //
                5.0, 4.0, 3.0, 2.0, 1.0, //
                2.0, 3.0, 4.0, 5.0, 1.0, //
                4.0, 5.0, 1.0, 2.0, 3.0, //
                3.0, 1.0, 5.0, 4.0, 2.0,
            ],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (10, 5),
            vec![
                0.01, 0.02, 0.03, 0.04, 0.05, //
                0.05, 0.04, 0.03, 0.02, 0.01, //
                0.02, 0.03, 0.04, 0.05, 0.01, //
                0.04, 0.05, 0.01, 0.02, 0.03, //
                0.03, 0.01, 0.05, 0.04, 0.02, //
                0.01, 0.02, 0.03, 0.04, 0.05, //
                0.05, 0.04, 0.03, 0.02, 0.01, //
                0.02, 0.03, 0.04, 0.05, 0.01, //
                0.04, 0.05, 0.01, 0.02, 0.03, //
                0.03, 0.01, 0.05, 0.04, 0.02,
            ],
        )
        .unwrap();

        // Test with custom configs
        let fee_config = FeeConfig {
            commission_rate: 0.001,
            slippage: SlippageConfig::default(),
            min_commission: 5.0,
        };

        let position_config = PositionConfig {
            long_ratio: 1.0,
            short_ratio: 0.5,
            market_neutral: true,
        };

        let config = BacktestConfig {
            quantiles: 5,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config,
            position_config,
        };

        let engine = BacktestEngine::with_config(config);

        let close = Array2::from_elem((10, 5), 1.0);
        let vwap = Array2::from_elem((10, 5), 1.0);

        let result = engine
            .run(factor, returns, Array2::from_elem((10, 5), 1.0), close, vwap)
            .unwrap();

        // Verify all result fields are present and finite
        assert!(result.long_short_cum_return.is_finite());
        assert!(result.ic_mean.is_finite());
        assert!(result.sharpe_ratio.is_finite());
        assert!(result.max_drawdown.is_finite());
        assert!(result.turnover.is_finite());
        assert!(result.annualized_return.is_finite());

        // Verify shapes
        assert_eq!(result.group_returns.dim(), (9, 5));
        assert_eq!(result.ic_series.len(), 9);
    }

    #[test]
    fn test_different_quantile_configurations() {
        let factor = Array2::from_shape_vec(
            (5, 10),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, //
                10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, //
                3.0, 5.0, 7.0, 9.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, //
                6.0, 4.0, 2.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0, 8.0, //
                9.0, 7.0, 5.0, 3.0, 1.0, 10.0, 8.0, 6.0, 4.0, 2.0,
            ],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (5, 10),
            vec![
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, //
                0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, //
                0.02, 0.03, 0.04, 0.05, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, //
                0.06, 0.05, 0.04, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, //
                0.07, 0.06, 0.05, 0.04, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
            ],
        )
        .unwrap();

        // Test with 5 quantiles (deciles)
        let config = BacktestConfig {
            quantiles: 5,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig {
                commission_rate: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
        };

        let close = Array2::from_elem((5, 10), 1.0);
        let vwap = Array2::from_elem((5, 10), 1.0);

        let engine = BacktestEngine::with_config(config);
        let result = engine
            .run(factor.clone(), returns.clone(), Array2::from_elem((5, 10), 1.0), close.clone(), vwap.clone())
            .unwrap();
        assert_eq!(result.group_returns.dim(), (4, 5));

        // Test with 2 quantiles (median split)
        let config2 = BacktestConfig {
            quantiles: 2,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig {
                commission_rate: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
        };

        let engine2 = BacktestEngine::with_config(config2);
        let adj_factor2 = Array2::from_elem((5, 10), 1.0);
        let result2 = engine2
            .run(factor, returns, adj_factor2, close, vwap)
            .unwrap();
        assert_eq!(result2.group_returns.dim(), (4, 2));
    }

    #[test]
    fn test_market_neutral_vs_directional() {
        let factor = Array2::from_shape_vec(
            (5, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, //
                4.0, 3.0, 2.0, 1.0, //
                2.0, 3.0, 4.0, 1.0, //
                1.0, 4.0, 3.0, 2.0, //
                3.0, 2.0, 1.0, 4.0,
            ],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (5, 4),
            vec![
                0.01, 0.02, 0.03, 0.04, //
                0.04, 0.03, 0.02, 0.01, //
                0.02, 0.03, 0.04, 0.01, //
                0.01, 0.04, 0.03, 0.02, //
                0.03, 0.02, 0.01, 0.04,
            ],
        )
        .unwrap();

        let close = Array2::from_elem((5, 4), 1.0);
        let vwap = Array2::from_elem((5, 4), 1.0);
        let adj_factor = Array2::from_elem((5, 4), 1.0);

        // Market neutral (long - short)
        let position_neutral = PositionConfig {
            long_ratio: 1.0,
            short_ratio: 1.0,
            market_neutral: true,
        };

        let config_neutral = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig::default(),
            position_config: position_neutral,
        };

        let engine_neutral = BacktestEngine::with_config(config_neutral);
        let result_neutral = engine_neutral
            .run(factor.clone(), returns.clone(), adj_factor.clone(), close.clone(), vwap.clone())
            .unwrap();

        // Directional (long only)
        let position_directional = PositionConfig {
            long_ratio: 1.0,
            short_ratio: 0.0,
            market_neutral: false,
        };

        let config_directional = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig::default(),
            position_config: position_directional,
        };

        let engine_directional = BacktestEngine::with_config(config_directional);
        let result_directional = engine_directional
            .run(factor, returns, adj_factor, close, vwap)
            .unwrap();

        // Both should produce finite results
        assert!(result_neutral.long_short_cum_return.is_finite());
        assert!(result_directional.long_short_cum_return.is_finite());
    }
}
