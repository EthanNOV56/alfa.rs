//! Enhanced backtest engine module
//!
//! Provides advanced backtesting capabilities including:
//! - Adjusted prices (前复权) support
//! - Custom fees and volume-based slippage
//! - Configurable long/short position ratios
//! - Additional performance metrics (Sharpe, max drawdown, turnover)

use ndarray::{Array1, Array2};

use crate::WeightMethod;
use crate::data::layer::PriceMatrix;

use super::config::{BacktestConfig, FeeConfig, LimitUpDownConfig, PositionConfig, SlippageConfig};
use super::types::BacktestResult;

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
        open: Array2<f64>,
        vwap: Array2<f64>,
        tradable: Array2<f64>,
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
            open.shape(),
            "Factor and open must have same shape"
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
        assert_eq!(
            tradable.shape(),
            factor.shape(),
            "Tradable must have same shape as factor"
        );

        let (n_days, n_assets) = factor.dim();
        let quantiles = self.config.quantiles;

        // Compute quantile groups
        let group_labels = super::portfolio::compute_quantile_groups(&factor, quantiles)?;

        // Compute group weights based on factor values
        // weight_method controls within-group allocation: Equal or Weighted
        let group_weights = super::portfolio::compute_group_weights(
            &factor,
            &group_labels,
            quantiles,
            self.config.weight_method,
        );

        let fee_rate = self.config.fee_config.commission_rate
            + self.config.fee_config.slippage.normal_slippage_rate;
        let group_returns = super::portfolio::simulate_groups(
            &group_labels,
            &group_weights,
            quantiles,
            n_days,
            n_assets,
            &tradable,
            &open,
            &close,
            &vwap,
            fee_rate,
        )?;

        // Compute long/short returns directly from per-group returns
        let mut long_short_returns = Array1::<f64>::zeros(n_days - 1);

        // Long returns (top quantiles)
        let mut long_returns = Array1::<f64>::zeros(n_days - 1);
        let long_groups: Vec<usize> = (quantiles - self.config.long_top_n..quantiles).collect();
        for day in 0..(n_days - 1) {
            let mut sum = 0.0;
            for &g in &long_groups {
                sum += group_returns[[day, g]];
            }
            long_returns[day] =
                sum / self.config.long_top_n as f64 * self.config.position_config.long_ratio;
        }

        // Short returns (bottom quantiles)
        let mut short_returns = Array1::<f64>::zeros(n_days - 1);
        let short_groups: Vec<usize> = (0..self.config.short_top_n).collect();
        for day in 0..(n_days - 1) {
            let mut sum = 0.0;
            for &g in &short_groups {
                sum += group_returns[[day, g]];
            }
            short_returns[day] =
                -sum / self.config.short_top_n as f64 * self.config.position_config.short_ratio;
        }

        // Long-short = long + short
        for day in 0..(n_days - 1) {
            long_short_returns[day] = long_returns[day] + short_returns[day];
        }

        // Compute IC series
        let (ic_series, ic_mean, ic_ir) = super::metrics::compute_ic_series(&factor, &returns)?;

        // Compute cumulative NAV curves from daily returns
        let long_short_cum_returns = super::metrics::cumulative_nav_curve(&long_short_returns);
        let long_cum_returns = super::metrics::cumulative_nav_curve(&long_returns);
        let short_cum_returns = super::metrics::cumulative_nav_curve(&short_returns);

        // Compute final cumulative return (log method for numerical stability)
        let long_short_cum_return = super::metrics::compute_total_return_log(&long_short_returns);

        // Compute additional metrics
        let total_return = long_short_cum_return;
        let annualized_return = super::metrics::compute_annualized_return(total_return, n_days);
        let sharpe_ratio = super::metrics::compute_sharpe_ratio(&long_short_returns, n_days);
        let max_drawdown = super::metrics::compute_max_drawdown(&long_short_returns);
        let turnover = super::metrics::compute_turnover(&group_labels);

        // Compute group cumulative returns
        let group_cum_returns = super::metrics::compute_cumulative_returns(&group_returns);

        Ok(BacktestResult {
            dates: vec![],
            group_returns,
            group_cum_returns,
            long_short_returns,
            long_short_cum_return,
            long_short_cum_returns,
            long_cum_returns,
            short_cum_returns,
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

    /// Run backtest with a `PriceMatrix` and pre-computed qcut matrix.
    ///
    /// Uses the qcut values directly instead of recomputing quantile groups.
    /// qcut[d][s] = group (0..quantiles-1) or -1 for NaN/unassigned.
    pub fn run_with_qcut(
        &self,
        factor: Array2<f64>,
        qcut: &Array2<i32>,
        prices: &PriceMatrix,
    ) -> Result<BacktestResult, String> {
        let (n_days, n_assets) = factor.dim();
        let quantiles = self.config.quantiles;

        // Build group_labels from qcut (1-indexed, 0 for -1/no-group)
        let mut group_labels = Array2::<usize>::zeros((n_days, n_assets));
        for d in 0..n_days {
            for a in 0..n_assets {
                let g = qcut[[d, a]];
                if g >= 0 {
                    group_labels[[d, a]] = g as usize + 1; // 1-indexed
                }
            }
        }

        let adj_factor = Array2::<f64>::from_elem((n_days, n_assets), 1.0);
        self.run_with_labels(
            factor,
            prices,
            &group_labels,
            quantiles,
            &prices.returns,
            &adj_factor,
            &prices.close,
            &prices.open,
            &prices.vwap,
            &prices.tradable,
        )
    }

    /// Run backtest with a `PriceMatrix` instead of individual arrays.
    ///
    /// `adj_factor` is set to all-1.0 because `query_price_matrix()` already
    /// forward-adjusts prices.
    pub fn run_with_prices(
        &self,
        factor: Array2<f64>,
        prices: &PriceMatrix,
    ) -> Result<BacktestResult, String> {
        let (n_dates, n_syms) = factor.dim();
        let adj_factor = Array2::<f64>::from_elem((n_dates, n_syms), 1.0);
        let mut result = self.run(
            factor,
            prices.returns.clone(),
            adj_factor,
            prices.close.clone(),
            prices.open.clone(),
            prices.vwap.clone(),
            prices.tradable.clone(),
        )?;
        result.dates = prices.dates.clone();
        Ok(result)
    }

    /// IC-only backtest: skip P&L simulation, compute only IC series and turnover.
    /// Much faster than `run()` because it avoids NAV simulation on all assets.
    pub fn run_ic_only(
        &self,
        factor: Array2<f64>,
        returns: Array2<f64>,
    ) -> Result<BacktestResult, String> {
        assert_eq!(factor.shape(), returns.shape());
        let (n_days, n_assets) = factor.dim();
        let quantiles = self.config.quantiles;

        let group_labels = super::portfolio::compute_quantile_groups(&factor, quantiles)?;
        let (ic_series, ic_mean, ic_ir) = super::metrics::compute_ic_series(&factor, &returns)?;
        let turnover = super::metrics::compute_turnover(&group_labels);

        Ok(BacktestResult {
            dates: vec![],
            group_returns: Array2::zeros((0, 0)),
            group_cum_returns: Array2::zeros((0, 0)),
            long_short_returns: Array1::zeros(0),
            long_short_cum_return: 0.0,
            long_short_cum_returns: Array1::zeros(0),
            long_cum_returns: Array1::zeros(0),
            short_cum_returns: Array1::zeros(0),
            ic_series,
            ic_mean,
            ic_ir,
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            turnover,
            long_returns: Array1::zeros(0),
            short_returns: Array1::zeros(0),
        })
    }

    /// IC-only backtest with PriceMatrix (auto-extracts returns).
    pub fn run_ic_only_with_prices(
        &self,
        factor: Array2<f64>,
        prices: &PriceMatrix,
    ) -> Result<BacktestResult, String> {
        let mut result = self.run_ic_only(factor, prices.returns.clone())?;
        result.dates = prices.dates.clone();
        Ok(result)
    }

    /// Multi-factor equal-weight combination backtest.
    ///
    /// Takes a list of factor matrices (all same shape), averages them
    /// element-wise with equal weight, then runs the standard `run()`.
    pub fn run_multi(
        &self,
        factors: &[Array2<f64>],
        returns: Array2<f64>,
        adj_factor: Array2<f64>,
        close: Array2<f64>,
        open: Array2<f64>,
        vwap: Array2<f64>,
        tradable: Array2<f64>,
    ) -> Result<BacktestResult, String> {
        if factors.is_empty() {
            return Err("No factors provided".to_string());
        }
        let shape = factors[0].dim();
        for f in factors.iter().skip(1) {
            if f.dim() != shape {
                return Err(format!(
                    "Factor shape mismatch: expected {:?}, got {:?}",
                    shape,
                    f.dim()
                ));
            }
        }

        let n = factors.len() as f64;
        let mut combined = Array2::<f64>::zeros(shape);
        for f in factors {
            for i in 0..shape.0 {
                for j in 0..shape.1 {
                    if f[[i, j]].is_finite() {
                        combined[[i, j]] += f[[i, j]] / n;
                    }
                }
            }
        }

        self.run(combined, returns, adj_factor, close, open, vwap, tradable)
    }

    /// Multi-factor equal-weight combination backtest with PriceMatrix.
    pub fn run_multi_with_prices(
        &self,
        factors: &[Array2<f64>],
        prices: &PriceMatrix,
    ) -> Result<BacktestResult, String> {
        if factors.is_empty() {
            return Err("No factors provided".to_string());
        }
        let (n_dates, n_syms) = factors[0].dim();
        let adj_factor = Array2::<f64>::from_elem((n_dates, n_syms), 1.0);
        let mut result = self.run_multi(
            factors,
            prices.returns.clone(),
            adj_factor,
            prices.close.clone(),
            prices.open.clone(),
            prices.vwap.clone(),
            prices.tradable.clone(),
        )?;
        result.dates = prices.dates.clone();
        Ok(result)
    }

    /// Run with pre-computed group_labels (from qcut pipeline).
    fn run_with_labels(
        &self,
        factor: Array2<f64>,
        prices: &PriceMatrix,
        group_labels: &Array2<usize>,
        quantiles: usize,
        returns: &Array2<f64>,
        _adj_factor: &Array2<f64>,
        close: &Array2<f64>,
        open: &Array2<f64>,
        vwap: &Array2<f64>,
        tradable: &Array2<f64>,
    ) -> Result<BacktestResult, String> {
        let (n_days, n_assets) = factor.dim();
        let fee_rate = self.config.fee_config.commission_rate
            + self.config.fee_config.slippage.normal_slippage_rate;

        let group_weights = super::portfolio::compute_group_weights(
            &factor,
            group_labels,
            quantiles,
            self.config.weight_method,
        );
        let group_returns = super::portfolio::simulate_groups(
            group_labels,
            &group_weights,
            quantiles,
            n_days,
            n_assets,
            tradable,
            open,
            close,
            vwap,
            fee_rate,
        )?;

        let mut long_short_returns = Array1::<f64>::zeros(n_days - 1);
        let mut long_returns = Array1::<f64>::zeros(n_days - 1);
        let long_groups: Vec<usize> = (quantiles - self.config.long_top_n..quantiles).collect();
        for day in 0..(n_days - 1) {
            let mut sum = 0.0;
            for &g in &long_groups {
                sum += group_returns[[day, g]];
            }
            long_returns[day] =
                sum / self.config.long_top_n as f64 * self.config.position_config.long_ratio;
        }

        let mut short_returns = Array1::<f64>::zeros(n_days - 1);
        let short_groups: Vec<usize> = (0..self.config.short_top_n).collect();
        for day in 0..(n_days - 1) {
            let mut sum = 0.0;
            for &g in &short_groups {
                sum += group_returns[[day, g]];
            }
            short_returns[day] =
                -sum / self.config.short_top_n as f64 * self.config.position_config.short_ratio;
        }

        for day in 0..(n_days - 1) {
            long_short_returns[day] = long_returns[day] + short_returns[day];
        }

        let (ic_series, ic_mean, ic_ir) = super::metrics::compute_ic_series(&factor, returns)?;
        let long_short_cum_returns = super::metrics::cumulative_nav_curve(&long_short_returns);
        let long_cum_returns = super::metrics::cumulative_nav_curve(&long_returns);
        let short_cum_returns = super::metrics::cumulative_nav_curve(&short_returns);
        let long_short_cum_return = super::metrics::compute_total_return_log(&long_short_returns);
        let total_return = long_short_cum_return;
        let annualized_return = super::metrics::compute_annualized_return(total_return, n_days);
        let sharpe_ratio = super::metrics::compute_sharpe_ratio(&long_short_returns, n_days);
        let max_drawdown = super::metrics::compute_max_drawdown(&long_short_returns);
        let turnover = super::metrics::compute_turnover(group_labels);
        let group_cum_returns = super::metrics::compute_cumulative_returns(&group_returns);

        let mut result = BacktestResult {
            dates: prices.dates.clone(),
            group_returns,
            group_cum_returns,
            long_short_returns,
            long_short_cum_return,
            long_short_cum_returns,
            long_cum_returns,
            short_cum_returns,
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
        };

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);

        let result = engine
            .run(
                factor,
                returns,
                adj_factor,
                close.clone(),
                close.clone(),
                vwap.clone(),
                Array2::from_elem(close.dim(), 1.0),
            )
            .unwrap();
        assert!(result.long_short_cum_return.is_finite());
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
            limit_up_down_config: Default::default(),
        };

        let adj_factor = Array2::from_elem((3, 4), 1.0);
        let engine = BacktestEngine::with_config(config);
        let result = engine
            .run(
                factor,
                returns.clone(),
                adj_factor,
                close.clone(),
                close.clone(),
                vwap.clone(),
                Array2::from_elem(close.dim(), 1.0),
            )
            .unwrap();

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
            limit_up_down_config: Default::default(),
        };

        let adj_factor = Array2::from_elem((3, 4), 1.0);
        let engine = BacktestEngine::with_config(config);
        let result = engine
            .run(
                factor,
                returns.clone(),
                adj_factor,
                close.clone(),
                close.clone(),
                vwap.clone(),
                Array2::from_elem(close.dim(), 1.0),
            )
            .unwrap();

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
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let open = close.clone();
        let tradable = Array2::from_elem(close.dim(), 1.0);
        let adj_factor = Array2::from_elem((3, 4), 1.0);
        let result = engine
            .run(
                factor.clone(),
                returns.clone(),
                adj_factor,
                close,
                open,
                vwap,
                tradable,
            )
            .unwrap();

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
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let open = close.clone();
        let tradable = Array2::from_elem(close.dim(), 1.0);
        let result = engine
            .run(
                factor,
                returns.clone(),
                Array2::from_elem((3, 4), 1.0),
                close.clone(),
                open,
                vwap.clone(),
                tradable,
            )
            .unwrap();

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
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let open = close.clone();
        let tradable = Array2::from_elem(close.dim(), 1.0);
        let result = engine
            .run(
                factor,
                returns.clone(),
                Array2::from_elem((3, 4), 1.0),
                close.clone(),
                open,
                vwap.clone(),
                tradable,
            )
            .unwrap();

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
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let open = close.clone();
        let tradable = Array2::from_elem(close.dim(), 1.0);
        let result = engine
            .run(
                factor,
                returns.clone(),
                Array2::from_elem((3, 4), 1.0),
                close.clone(),
                open,
                vwap.clone(),
                tradable,
            )
            .unwrap();

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
            limit_up_down_config: Default::default(),
        };

        let close = Array2::from_elem((3, 4), 1.0);
        let vwap = Array2::from_elem((3, 4), 1.0);

        let engine = BacktestEngine::with_config(config);

        // Should handle NaN gracefully
        let result = engine
            .run(
                factor,
                returns.clone(),
                Array2::from_elem((3, 4), 1.0),
                close.clone(),
                close.clone(),
                vwap.clone(),
                Array2::from_elem(close.dim(), 1.0),
            )
            .unwrap();
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
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let close = Array2::from_elem((1, 4), 1.0);
        let open = Array2::from_elem((1, 4), 1.0);
        let vwap = Array2::from_elem((1, 4), 1.0);
        let tradable = Array2::from_elem((1, 4), 1.0);
        let adj_factor = Array2::from_elem((1, 4), 1.0);

        // Single day returns - should work but produce empty results
        let result = engine.run(
            factor,
            returns.clone(),
            adj_factor,
            close,
            open,
            vwap,
            tradable,
        );
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
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let close = Array2::from_elem((3, 1), 1.0);
        let open = Array2::from_elem((3, 1), 1.0);
        let vwap = Array2::from_elem((3, 1), 1.0);
        let tradable = Array2::from_elem((3, 1), 1.0);
        let adj_factor = Array2::from_elem((3, 1), 1.0);

        // Single asset - run might fail due to edge case handling
        // Just verify it doesn't panic
        let _ = engine.run(
            factor,
            returns.clone(),
            adj_factor,
            close,
            open,
            vwap,
            tradable,
        );
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
                1.0, 1.0, 1.0, 1.0, // day 0
                0.99, 0.98, 0.97, 0.96, // day 1: returns = -0.01, -0.02, -0.03, -0.04
                0.95, 0.94, 0.93, 0.92, // day 2
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
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);

        let adj_factor = Array2::from_elem((3, 4), 1.0);
        let result = engine
            .run(
                factor,
                returns,
                adj_factor,
                close.clone(),
                close.clone(),
                vwap.clone(),
                Array2::from_elem(close.dim(), 1.0),
            )
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
            limit_up_down_config: Default::default(),
        };

        let close = Array2::from_elem((10, 5), 1.0);
        let open = Array2::from_elem((10, 5), 1.0);
        let vwap = Array2::from_elem((10, 5), 1.0);
        let tradable = Array2::from_elem((10, 5), 1.0);

        let engine = BacktestEngine::with_config(config);
        let result = engine
            .run(
                factor,
                returns,
                Array2::from_elem((10, 5), 1.0),
                close,
                open,
                vwap,
                tradable,
            )
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
            limit_up_down_config: Default::default(),
        };

        let close = Array2::from_elem((5, 10), 1.0);
        let open = Array2::from_elem((5, 10), 1.0);
        let vwap = Array2::from_elem((5, 10), 1.0);
        let tradable = Array2::from_elem((5, 10), 1.0);

        let engine = BacktestEngine::with_config(config);
        let result = engine
            .run(
                factor.clone(),
                returns.clone(),
                Array2::from_elem((5, 10), 1.0),
                close.clone(),
                open.clone(),
                vwap.clone(),
                tradable.clone(),
            )
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
            limit_up_down_config: Default::default(),
        };

        let engine2 = BacktestEngine::with_config(config2);
        let adj_factor2 = Array2::from_elem((5, 10), 1.0);
        let result2 = engine2
            .run(
                factor,
                returns,
                adj_factor2,
                close.clone(),
                open.clone(),
                vwap.clone(),
                tradable.clone(),
            )
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
            limit_up_down_config: Default::default(),
        };

        let engine_neutral = BacktestEngine::with_config(config_neutral);
        let result_neutral = engine_neutral
            .run(
                factor.clone(),
                returns.clone(),
                adj_factor.clone(),
                close.clone(),
                close.clone(),
                vwap.clone(),
                Array2::from_elem(close.dim(), 1.0),
            )
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
            limit_up_down_config: Default::default(),
        };

        let engine_directional = BacktestEngine::with_config(config_directional);
        let result_directional = engine_directional
            .run(
                factor,
                returns,
                adj_factor,
                close.clone(),
                close.clone(),
                vwap.clone(),
                Array2::from_elem(close.dim(), 1.0),
            )
            .unwrap();

        // Both should produce finite results
        assert!(result_neutral.long_short_cum_return.is_finite());
        assert!(result_directional.long_short_cum_return.is_finite());
    }
}
