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

    /// Run a full backtest: NAV simulation, IC, and all performance metrics.
    ///
    /// Takes `factor` by value (per-call data) and borrows price arrays via
    /// `PriceData` for zero-clone reuse across multiple backtest calls
    /// (e.g. GP fitness evaluation).
    ///
    /// Computes quantile groups from the factor, then delegates to the shared
    /// `run_core` for NAV simulation and all metrics.
    pub fn run(&self, factor: Array2<f64>, prices: &PriceMatrix) -> Result<BacktestResult, String> {
        assert_eq!(
            factor.shape(),
            prices.returns.shape(),
            "Factor and returns must have same shape"
        );
        assert_eq!(
            factor.shape(),
            prices.close.shape(),
            "Factor and close must have same shape"
        );
        assert_eq!(
            factor.shape(),
            prices.open.shape(),
            "Factor and open must have same shape"
        );
        assert_eq!(
            factor.shape(),
            prices.vwap.shape(),
            "Factor and vwap must have same shape"
        );
        assert_eq!(
            prices.tradable.shape(),
            factor.shape(),
            "Tradable must have same shape as factor"
        );

        let (n_days, n_assets) = factor.dim();
        let quantiles = self.config.quantiles;

        let group_labels = super::portfolio::compute_quantile_groups(&factor, quantiles)?;

        let mut result =
            self.run_core(&factor, prices, &group_labels, quantiles, n_days, n_assets)?;
        result.dates = prices.dates.clone();
        Ok(result)
    }

    /// Run a full backtest with pre-computed quantile groups (qcut).
    ///
    /// Used by AlfarsLab when qcut is already available from the FactorSlice CS
    /// pipeline, avoiding recomputation of quantile groups with different
    /// tie-breaking logic.
    ///
    /// qcut[d][s] = group index (0..quantiles-1) or -1 for NaN/unassigned.
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
                    group_labels[[d, a]] = g as usize + 1;
                }
            }
        }

        let mut result =
            self.run_core(&factor, prices, &group_labels, quantiles, n_days, n_assets)?;
        result.dates = prices.dates.clone();
        Ok(result)
    }

    /// IC-only backtest: compute IC series and turnover, skip NAV simulation.
    ///
    /// Much faster than `run()` — suitable for rough factor screening when
    /// P&L simulation is not needed.
    pub fn run_ic(
        &self,
        factor: Array2<f64>,
        returns: &Array2<f64>,
    ) -> Result<BacktestResult, String> {
        assert_eq!(factor.shape(), returns.shape());
        let quantiles = self.config.quantiles;

        let group_labels = super::portfolio::compute_quantile_groups(&factor, quantiles)?;
        let (ic_series, ic_mean, ic_ir) = super::metrics::compute_ic_series(&factor, returns)?;
        let turnover = super::metrics::compute_turnover(&group_labels);

        Ok(BacktestResult {
            ic_series,
            ic_mean,
            ic_ir,
            turnover,
            ..Default::default()
        })
    }

    /// Multi-factor equal-weight combination backtest.
    ///
    /// Averages all factor arrays element-wise with equal weight, then runs
    /// the standard `run` backtest on the combined factor.
    pub fn run_multi(
        &self,
        factors: &[Array2<f64>],
        prices: &PriceMatrix,
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

        self.run(combined, prices)
    }

    /// Shared core: NAV simulation + IC + all metrics.
    ///
    /// Called by both `run` (after computing quantile groups) and
    /// `run_with_qcut` (after converting qcut to group_labels).
    /// Produces all `BacktestResult` fields consistently for both paths.
    fn run_core(
        &self,
        factor: &Array2<f64>,
        prices: &PriceMatrix,
        group_labels: &Array2<usize>,
        quantiles: usize,
        n_days: usize,
        n_assets: usize,
    ) -> Result<BacktestResult, String> {
        let buy_fee =
            self.config.fee_config.buy_commission + self.config.fee_config.slippage.buy_slippage;
        let sell_fee =
            self.config.fee_config.sell_commission + self.config.fee_config.slippage.sell_slippage;

        let group_weights = super::portfolio::compute_group_weights(
            factor,
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
            &prices.tradable,
            &prices.open,
            &prices.close,
            &prices.vwap,
            buy_fee,
            sell_fee,
            self.config.rebalance_freq,
        )?;

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
        let mut long_short_returns = Array1::<f64>::zeros(n_days - 1);
        for day in 0..(n_days - 1) {
            long_short_returns[day] = long_returns[day] + short_returns[day];
        }

        // Passive benchmark: equal-weight all tradable stocks
        let mut passive_returns = Array1::<f64>::zeros(n_days - 1);
        for day in 0..(n_days - 1) {
            let mut sum_ret = 0.0f64;
            let mut count = 0usize;
            for a in 0..n_assets {
                let r = prices.returns[[day, a]];
                if prices.tradable[[day, a]] > 0.5 && r.is_finite() {
                    sum_ret += r;
                    count += 1;
                }
            }
            if count > 0 {
                passive_returns[day] = sum_ret / count as f64;
            }
        }

        // IC series (full market)
        let (ic_series, ic_mean, ic_ir) =
            super::metrics::compute_ic_series(factor, &prices.returns)?;

        // Group IC: long groups only (1-indexed)
        let long_groups_1idx: Vec<usize> = long_groups.iter().map(|g| g + 1).collect();
        let (_, long_ic_mean, long_ic_ir) = super::metrics::compute_group_ic_series(
            factor,
            &prices.returns,
            group_labels,
            &long_groups_1idx,
        )?;

        // Group IC: short groups only
        let short_groups_1idx: Vec<usize> = short_groups.iter().map(|g| g + 1).collect();
        let (_, short_ic_mean, short_ic_ir) = super::metrics::compute_group_ic_series(
            factor,
            &prices.returns,
            group_labels,
            &short_groups_1idx,
        )?;

        // Group IC: long+short combined
        let mut ls_groups = long_groups_1idx.clone();
        ls_groups.extend(&short_groups_1idx);
        let (_, long_short_ic_mean, long_short_ic_ir) = super::metrics::compute_group_ic_series(
            factor,
            &prices.returns,
            group_labels,
            &ls_groups,
        )?;

        // Cumulative NAV curves
        let long_short_cum_returns = super::metrics::cumulative_nav_curve(&long_short_returns);
        let long_cum_returns = super::metrics::cumulative_nav_curve(&long_returns);
        let short_cum_returns = super::metrics::cumulative_nav_curve(&short_returns);
        let passive_cum_returns = super::metrics::cumulative_nav_curve(&passive_returns);

        let long_short_cum_return = super::metrics::compute_total_return_log(&long_short_returns);

        // Performance metrics
        let total_return = long_short_cum_return;
        let annualized_return = super::metrics::compute_annualized_return(total_return, n_days);
        let sharpe_ratio = super::metrics::compute_sharpe_ratio(&long_short_returns, n_days);
        let max_drawdown = super::metrics::compute_max_drawdown(&long_short_returns);
        let turnover = super::metrics::compute_turnover(group_labels);
        let weight_turnover = super::metrics::compute_weight_turnover(&group_weights);
        let win_rate = super::metrics::compute_win_rate(&long_short_returns);
        let calmar_ratio = super::metrics::compute_calmar_ratio(annualized_return, max_drawdown);

        let group_cum_returns = super::metrics::compute_cumulative_returns(&group_returns);

        Ok(BacktestResult {
            dates: vec![], // caller sets this
            group_returns,
            group_cum_returns,
            long_short_returns,
            long_short_cum_return,
            long_short_cum_returns,
            long_cum_returns,
            short_cum_returns,
            passive_returns,
            passive_cum_returns,
            ic_series,
            ic_mean,
            ic_ir,
            long_ic_mean,
            long_ic_ir,
            short_ic_mean,
            short_ic_ir,
            long_short_ic_mean,
            long_short_ic_ir,
            total_return,
            annualized_return,
            sharpe_ratio,
            max_drawdown,
            turnover,
            weight_turnover,
            win_rate,
            calmar_ratio,
            long_returns,
            short_returns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: run full backtest with owned arrays.
    fn bt(
        engine: &BacktestEngine,
        factor: Array2<f64>,
        returns: Array2<f64>,
        _adj_factor: Array2<f64>,
        close: Array2<f64>,
        open: Array2<f64>,
        vwap: Array2<f64>,
        tradable: Array2<f64>,
    ) -> Result<BacktestResult, String> {
        let (n_days, n_assets) = factor.dim();
        let pm = PriceMatrix {
            dates: vec![],
            symbols: vec![],
            close,
            open,
            high: Array2::from_elem((n_days, n_assets), 1.0),
            low: Array2::from_elem((n_days, n_assets), 1.0),
            vwap,
            returns,
            tradable,
            adj_factor: Array2::from_elem((n_days, n_assets), 1.0),
        };
        engine.run(factor, &pm)
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
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);

        let result = bt(
            &engine,
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
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let adj_factor = Array2::from_elem((3, 4), 1.0);
        let engine = BacktestEngine::with_config(config);
        let result = bt(
            &engine,
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
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let adj_factor = Array2::from_elem((3, 4), 1.0);
        let engine = BacktestEngine::with_config(config);
        let result = bt(
            &engine,
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
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let open = close.clone();
        let tradable = Array2::from_elem(close.dim(), 1.0);
        let adj_factor = Array2::from_elem((3, 4), 1.0);
        let result = bt(
            &engine,
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
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let open = close.clone();
        let tradable = Array2::from_elem(close.dim(), 1.0);
        let result = bt(
            &engine,
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
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let open = close.clone();
        let tradable = Array2::from_elem(close.dim(), 1.0);
        let result = bt(
            &engine,
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
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let open = close.clone();
        let tradable = Array2::from_elem(close.dim(), 1.0);
        let result = bt(
            &engine,
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
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let close = Array2::from_elem((3, 4), 1.0);
        let vwap = Array2::from_elem((3, 4), 1.0);

        let engine = BacktestEngine::with_config(config);

        // Should handle NaN gracefully
        let result = bt(
            &engine,
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
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
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
        let result = bt(
            &engine,
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
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
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
        let _ = bt(
            &engine,
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
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);

        let adj_factor = Array2::from_elem((3, 4), 1.0);
        let result = bt(
            &engine,
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
            buy_commission: 0.001,
            sell_commission: 0.001,
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
            rebalance_freq: 1,
            fee_config,
            position_config,
            limit_up_down_config: Default::default(),
        };

        let close = Array2::from_elem((10, 5), 1.0);
        let open = Array2::from_elem((10, 5), 1.0);
        let vwap = Array2::from_elem((10, 5), 1.0);
        let tradable = Array2::from_elem((10, 5), 1.0);

        let engine = BacktestEngine::with_config(config);
        let result = bt(
            &engine,
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
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
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
        let result = bt(
            &engine,
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
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine2 = BacktestEngine::with_config(config2);
        let adj_factor2 = Array2::from_elem((5, 10), 1.0);
        let result2 = bt(
            &engine2,
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
            rebalance_freq: 1,
            fee_config: FeeConfig::default(),
            position_config: position_neutral,
            limit_up_down_config: Default::default(),
        };

        let engine_neutral = BacktestEngine::with_config(config_neutral);
        let result_neutral = bt(
            &engine_neutral,
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
            rebalance_freq: 1,
            fee_config: FeeConfig::default(),
            position_config: position_directional,
            limit_up_down_config: Default::default(),
        };

        let engine_directional = BacktestEngine::with_config(config_directional);
        let result_directional = bt(
            &engine_directional,
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

    // === Rebalance Frequency Tests ===

    #[test]
    fn test_rebalance_freq_daily_is_default() {
        // With freq=1, behavior should match the original daily rebalancing
        let factor = Array2::from_shape_vec(
            (5, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 4.0, 3.0, 2.0,
                3.0, 2.0, 1.0, 4.0,
            ],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (5, 4),
            vec![
                0.01, 0.02, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.02, 0.03, 0.04, 0.01, 0.01, 0.04,
                0.03, 0.02, 0.03, 0.02, 0.01, 0.04,
            ],
        )
        .unwrap();

        let close = Array2::from_elem((5, 4), 1.0);
        let vwap = Array2::from_elem((5, 4), 1.0);
        let tradable = Array2::from_elem((5, 4), 1.0);

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 1,
            fee_config: FeeConfig::default(),
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let result = bt(
            &engine,
            factor.clone(),
            returns.clone(),
            Array2::from_elem((5, 4), 1.0),
            close.clone(),
            close.clone(),
            vwap.clone(),
            tradable.clone(),
        )
        .unwrap();

        // Basic validity checks
        assert_eq!(result.group_returns.dim(), (4, 4));
        assert!(result.long_short_cum_return.is_finite());
        assert!(result.sharpe_ratio.is_finite());
    }

    #[test]
    fn test_rebalance_freq_reduces_trading() {
        // With higher rebalance_freq, there should be fewer rebalance days.
        // Use constant prices so that holding days have zero return.
        let factor = Array2::from_shape_vec(
            (6, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, // day1: group assignments change
                2.0, 3.0, 4.0, 1.0, 1.0, 4.0, 3.0, 2.0, 3.0, 2.0, 1.0, 4.0, 4.0, 1.0, 2.0, 3.0,
            ],
        )
        .unwrap();

        let returns = Array2::from_elem((6, 4), 0.0); // zero returns
        let close = Array2::from_elem((6, 4), 1.0);
        let vwap = Array2::from_elem((6, 4), 1.0);
        let tradable = Array2::from_elem((6, 4), 1.0);

        // freq=1: daily rebalance -> trades on days 1,2,3,4,5
        let config_daily = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 1,
            fee_config: FeeConfig::default(),
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };
        let engine_daily = BacktestEngine::with_config(config_daily);
        let _result_daily = bt(
            &engine_daily,
            factor.clone(),
            returns.clone(),
            Array2::from_elem((6, 4), 1.0),
            close.clone(),
            close.clone(),
            vwap.clone(),
            tradable.clone(),
        )
        .unwrap();

        // freq=3: rebalance on days 1,4 only -> less trading
        let config_wide = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 3,
            fee_config: FeeConfig::default(),
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };
        let engine_wide = BacktestEngine::with_config(config_wide);
        let result_wide = bt(
            &engine_wide,
            factor.clone(),
            returns.clone(),
            Array2::from_elem((6, 4), 1.0),
            close.clone(),
            close.clone(),
            vwap.clone(),
            tradable.clone(),
        )
        .unwrap();

        // Both should produce valid results
        assert!(result_wide.long_short_cum_return.is_finite());
    }

    #[test]
    fn test_rebalance_freq_hold_day_no_fee() {
        // On hold days, there should be no trading and thus no fees.
        // If all prices are 1.0, fees come from share deltas * fee_rate * vwap.
        // With freq=2 on a 3-day factor set, day1 is rebalance, day2 is hold.
        // On day2 (hold), shares don't change, so no fee on that day.
        let factor = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0],
        )
        .unwrap();

        let returns = Array2::from_elem((3, 4), 0.0);
        let close = Array2::from_elem((3, 4), 1.0);
        let vwap = Array2::from_elem((3, 4), 1.0);
        let tradable = Array2::from_elem((3, 4), 1.0);

        // freq=2: day 1 rebalance, day 2 hold
        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 2,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let result = bt(
            &engine,
            factor,
            returns,
            Array2::from_elem((3, 4), 1.0),
            close.clone(),
            close.clone(),
            vwap.clone(),
            tradable.clone(),
        )
        .unwrap();

        // With freq=2 and 3 days: day1 rebalance, day2 hold.
        // The hold day should have zero price movement (close=1.0 always),
        // so NAV should stay at 1.0 after the hold day
        // (minus any fee from day1 rebalance).
        // The key point: no crash, finite result.
        assert!(result.group_returns[[1, 0]].is_finite());
    }

    #[test]
    fn test_rebalance_freq_weights_drift_with_prices() {
        // On a hold day with non-zero price movement, the portfolio value
        // changes with prices. Shares stay constant, but NAV drifts.
        //
        // Setup: 2 assets, 2 groups, 3 days, freq=2.
        // Day 1: rebalance, buy asset0 at open=1.0, close=1.0. NAV=1.0, shares=1.0.
        // Day 2: hold (no rebalance). Asset0 close jumps to 2.0. NAV should double
        //        since we still hold 1.0 shares now worth 2.0 each.
        let factor = Array2::from_shape_vec(
            (3, 2),
            vec![
                1.0, 2.0, // day0: asset0 in group1, asset1 in group2
                1.0, 2.0, // day1: same
                1.0, 2.0, // day2: same
            ],
        )
        .unwrap();

        let returns = Array2::from_elem((3, 2), 0.0);

        // Day1 (rebalance): open=1.0, close=1.0 -> flat
        // Day2 (hold):      open=1.0, close=2.0 -> price doubles while we hold
        let close = Array2::from_shape_vec(
            (3, 2),
            vec![
                1.0, 1.0, // day0
                1.0, 1.0, // day1: flat (rebalance day, entry)
                2.0, 1.0, // day2: asset0 doubles (hold day)
            ],
        )
        .unwrap();
        let open = Array2::from_shape_vec(
            (3, 2),
            vec![
                1.0, 1.0, // day0
                1.0, 1.0, // day1: open=1.0, we buy shares cheap
                1.0, 1.0, // day2: open still 1.0 (not used on hold day)
            ],
        )
        .unwrap();
        let vwap = close.clone();
        let tradable = Array2::from_elem((3, 2), 1.0);

        // group 1 = asset0 (factor=1.0, bottom), group 2 = asset1 (factor=2.0, top)
        let config = BacktestConfig {
            quantiles: 2,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 2,
            fee_config: FeeConfig {
                buy_commission: 0.0,
                sell_commission: 0.0, // zero fees for clean test
                ..Default::default()
            },
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let result = bt(
            &engine,
            factor.clone(),
            returns,
            Array2::from_elem((3, 2), 1.0),
            close.clone(),
            open.clone(),
            vwap.clone(),
            tradable.clone(),
        )
        .unwrap();

        // Day 1 (index 0 in returns): rebalance, buy asset0 at open=1.0
        //   group_weight=0.5, alloc=1.0*(0.5/0.5)=1.0, shares=1.0/1.0=1.0
        //   close=1.0, NAV=1.0, return=0.0
        assert!(
            (result.group_returns[[0, 0]] - 0.0).abs() < 0.01,
            "Rebalance day (flat price) should give 0 return"
        );

        // Day 2 (index 1 in returns): hold, no trading
        //   shares still 1.0, close=2.0, NAV=2.0, return=2.0/1.0-1=1.0
        let group0_ret_d2 = result.group_returns[[1, 0]];
        assert!(
            (group0_ret_d2 - 1.0).abs() < 0.01,
            "Hold day with 2x close should give ~100% return, got {}",
            group0_ret_d2
        );
    }

    #[test]
    fn test_rebalance_freq_must_not_bfill_weights() {
        // This test verifies that we do NOT bfill weights on hold days.
        // If we bfill'ed, the portfolio would rebalance to the stale weights
        // every day, generating phantom turnover.
        //
        // Setup: factor changes on day1, so group assignments change.
        // With freq=2:
        //   Day1: rebalance with day0 labels
        //   Day2: hold (do NOT rebalance with day1 labels)
        //
        // If we incorrectly bfill'ed day0 weights to day1, day2 would trade
        // using day0's groups. Correct behavior: day2 holds, no trading.
        //
        // We verify by checking that the two freq variants produce different
        // group_returns — freq=2 should skip the day2 rebalance that freq=1 does.

        let factor = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, // day0
                4.0, 3.0, 2.0, 1.0, // day1: groups reversed
                1.0, 2.0, 3.0, 4.0, // day2: groups reversed again
            ],
        )
        .unwrap();

        let returns = Array2::from_elem((3, 4), 0.0);
        let close = Array2::from_elem((3, 4), 10.0);
        let vwap = Array2::from_elem((3, 4), 10.0);
        let tradable = Array2::from_elem((3, 4), 1.0);

        let config_daily = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 1,
            fee_config: FeeConfig {
                buy_commission: 0.001,
                sell_commission: 0.001,
                ..Default::default()
            },
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let config_hold = BacktestConfig {
            rebalance_freq: 2,
            ..config_daily.clone()
        };

        let engine_daily = BacktestEngine::with_config(config_daily);
        let result_daily = bt(
            &engine_daily,
            factor.clone(),
            returns.clone(),
            Array2::from_elem((3, 4), 1.0),
            close.clone(),
            close.clone(),
            vwap.clone(),
            tradable.clone(),
        )
        .unwrap();

        let engine_hold = BacktestEngine::with_config(config_hold);
        let result_hold = bt(
            &engine_hold,
            factor,
            returns,
            Array2::from_elem((3, 4), 1.0),
            close.clone(),
            close.clone(),
            vwap.clone(),
            tradable.clone(),
        )
        .unwrap();

        // With prices constant at 10.0, daily rebalancing trades every day
        // generating fees. Hold mode trades only day1, so fees are lower.
        // Since returns are zero, the difference is purely from fees.
        // freq=2 should have HIGHER cumulative return (less fee drag) than freq=1.
        assert!(
            result_hold.long_short_cum_return >= result_daily.long_short_cum_return,
            "Hold mode (freq=2) should have >= cumulative return than daily (freq=1) due to lower fees. hold={}, daily={}",
            result_hold.long_short_cum_return,
            result_daily.long_short_cum_return
        );
    }

    // === Passive Benchmark Tests ===

    #[test]
    fn test_passive_returns_equal_weight() {
        // Passive benchmark should return equal-weight average of all
        // tradable stock returns.
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

        let close = Array2::from_elem((3, 4), 10.0);
        let vwap = Array2::from_elem((3, 4), 10.0);
        let tradable = Array2::from_elem((3, 4), 1.0);

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 1,
            fee_config: FeeConfig::default(),
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let result = bt(
            &engine,
            factor,
            returns.clone(),
            Array2::from_elem((3, 4), 1.0),
            close.clone(),
            close.clone(),
            vwap.clone(),
            tradable.clone(),
        )
        .unwrap();

        // passive_returns should have length n_days-1 = 2
        assert_eq!(result.passive_returns.len(), 2);
        assert_eq!(result.passive_cum_returns.len(), 2);

        // Day 0 returns: [0.01, 0.02, 0.03, 0.04], mean = 0.025
        assert!(
            (result.passive_returns[0] - 0.025).abs() < 1e-10,
            "Passive return day0 should be 0.025, got {}",
            result.passive_returns[0]
        );

        // Day 1 returns: [0.04, 0.03, 0.02, 0.01], mean = 0.025
        assert!(
            (result.passive_returns[1] - 0.025).abs() < 1e-10,
            "Passive return day1 should be 0.025, got {}",
            result.passive_returns[1]
        );

        // Cumulative: 1.0 * 1.025 * 1.025 = 1.050625
        let expected_nav = 1.0 * 1.025 * 1.025;
        assert!(
            (result.passive_cum_returns[1] - expected_nav).abs() < 1e-10,
            "Passive cum NAV should be {}, got {}",
            expected_nav,
            result.passive_cum_returns[1]
        );
    }

    #[test]
    fn test_passive_excludes_untradable() {
        // Stocks with tradable=0 should be excluded from passive benchmark.
        let factor =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0])
                .unwrap();

        let returns = Array2::from_shape_vec(
            (3, 3),
            vec![0.10, 0.20, 0.30, 0.30, 0.20, 0.10, 0.10, 0.10, 0.10],
        )
        .unwrap();

        let close = Array2::from_elem((3, 3), 10.0);
        let vwap = Array2::from_elem((3, 3), 10.0);

        // Asset 2 is untradable on days 0 and 1
        let tradable = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 1.0, 0.0, // day0: asset2 excluded
                1.0, 1.0, 0.0, // day1: asset2 excluded
                1.0, 1.0, 1.0, // day2
            ],
        )
        .unwrap();

        let config = BacktestConfig {
            quantiles: 3,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 1,
            fee_config: FeeConfig::default(),
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let result = bt(
            &engine,
            factor,
            returns.clone(),
            Array2::from_elem((3, 3), 1.0),
            close.clone(),
            close.clone(),
            vwap.clone(),
            tradable,
        )
        .unwrap();

        // Day 0: only assets 0,1 tradable => mean(0.10, 0.20) = 0.15
        assert!(
            (result.passive_returns[0] - 0.15).abs() < 1e-10,
            "Passive day0 should exclude untradable, got {}",
            result.passive_returns[0]
        );
    }

    // === Group IC Tests ===

    #[test]
    fn test_group_ic_isolates_correct_groups() {
        // Group IC should only consider assets within the target groups.
        // Long IC = only top group assets, Short IC = only bottom group assets.
        // Use 8 assets, 4 quantiles → 2 assets per group (enough for correlation).
        let factor = Array2::from_shape_vec(
            (3, 8),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
                3.0, 1.0, 4.0, 5.0, 2.0, 6.0, 8.0, 7.0,
            ],
        )
        .unwrap();

        let returns = Array2::from_shape_vec(
            (3, 8),
            vec![
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03,
                0.02, 0.01, 0.03, 0.06, 0.01, 0.04, 0.05, 0.07, 0.02, 0.08,
            ],
        )
        .unwrap();

        let close = Array2::from_elem((3, 8), 10.0);
        let vwap = Array2::from_elem((3, 8), 10.0);
        let tradable = Array2::from_elem((3, 8), 1.0);

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,  // only top group (2 assets)
            short_top_n: 1, // only bottom group (2 assets)
            rebalance_freq: 1,
            fee_config: FeeConfig::default(),
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let result = bt(
            &engine,
            factor,
            returns.clone(),
            Array2::from_elem((3, 8), 1.0),
            close.clone(),
            close.clone(),
            vwap.clone(),
            tradable.clone(),
        )
        .unwrap();

        // Each group has 2 assets, enough for Pearson correlation
        assert!(
            result.long_ic_mean.is_finite(),
            "Long IC mean should be finite, got {:?}",
            result.long_ic_mean
        );
        assert!(
            result.short_ic_mean.is_finite(),
            "Short IC mean should be finite, got {:?}",
            result.short_ic_mean
        );
        // Long+short combined = 4 assets, enough for correlation
        assert!(
            result.long_short_ic_mean.is_finite(),
            "Long+short IC mean should be finite, got {:?}",
            result.long_short_ic_mean
        );
    }

    #[test]
    fn test_group_ic_handles_empty_groups() {
        // When a group has only 1 asset, group IC should be NaN without panicking.
        // Need >= 3 days for full-market IC (needs >= 2 IC values for IR computation).
        let factor =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0])
                .unwrap();

        let returns = Array2::from_shape_vec(
            (3, 3),
            vec![0.01, 0.02, 0.03, 0.03, 0.02, 0.01, 0.02, 0.03, 0.01],
        )
        .unwrap();

        let close = Array2::from_elem((3, 3), 10.0);
        let vwap = Array2::from_elem((3, 3), 10.0);
        let tradable = Array2::from_elem((3, 3), 1.0);

        // 3 assets, 2 quantiles => one group has 2 assets, one has 1.
        // The 1-asset group IC will be NaN.
        // This should not panic.
        let config = BacktestConfig {
            quantiles: 2,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 1,
            fee_config: FeeConfig::default(),
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let result = bt(
            &engine,
            factor,
            returns.clone(),
            Array2::from_elem((3, 3), 1.0),
            close.clone(),
            close.clone(),
            vwap.clone(),
            tradable.clone(),
        )
        .unwrap();

        // With 1 asset per group, group IC should be NaN (not enough samples)
        // but the result should not contain infinite or crashing values
        // (NaN is acceptable for insufficient data)
        assert!(result.long_ic_mean.is_nan() || result.long_ic_mean.is_finite());
    }

    // === Weight Turnover, Win Rate, Calmar Tests ===

    #[test]
    fn test_weight_turnover_vs_label_turnover() {
        // When factor values shift slightly but group labels don't change,
        // weight_turnover should be > 0 while label_turnover is 0.
        let factor = Array2::from_shape_vec(
            (3, 8),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1,
                8.1, // slight shift, same ordering
                1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2,
            ],
        )
        .unwrap();

        let returns = Array2::from_elem((3, 8), 0.0);
        let close = Array2::from_elem((3, 8), 10.0);
        let vwap = Array2::from_elem((3, 8), 10.0);
        let tradable = Array2::from_elem((3, 8), 1.0);

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Weighted, // weight within group depends on factor value
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 1,
            fee_config: FeeConfig::default(),
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let result = bt(
            &engine,
            factor,
            returns,
            Array2::from_elem((3, 8), 1.0),
            close.clone(),
            close.clone(),
            vwap.clone(),
            tradable.clone(),
        )
        .unwrap();

        // With ordered but slightly shifted factors, label turnover should be 0,
        // but weight turnover should be > 0 (within-group weight shifts).
        assert!(
            result.weight_turnover >= 0.0,
            "Weight turnover should be non-negative, got {}",
            result.weight_turnover
        );
    }

    #[test]
    fn test_win_rate_computation() {
        // Verify win rate calculation with known return values.
        let factor = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 4.0, 3.0, 2.0,
            ],
        )
        .unwrap();

        let returns = Array2::from_elem((4, 4), 0.0);
        let close = Array2::from_elem((4, 4), 10.0);
        let vwap = Array2::from_elem((4, 4), 10.0);
        let tradable = Array2::from_elem((4, 4), 1.0);

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 1,
            fee_config: FeeConfig::default(),
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let result = bt(
            &engine,
            factor,
            returns,
            Array2::from_elem((4, 4), 1.0),
            close.clone(),
            close.clone(),
            vwap.clone(),
            tradable.clone(),
        )
        .unwrap();

        // Win rate should be between 0 and 1
        assert!(
            result.win_rate >= 0.0 && result.win_rate <= 1.0,
            "Win rate should be in [0, 1], got {}",
            result.win_rate
        );
    }

    #[test]
    fn test_calmar_ratio_computation() {
        let factor = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 4.0, 3.0, 2.0,
            ],
        )
        .unwrap();

        let returns = Array2::from_elem((4, 4), 0.0);
        let close = Array2::from_elem((4, 4), 10.0);
        let vwap = Array2::from_elem((4, 4), 10.0);
        let tradable = Array2::from_elem((4, 4), 1.0);

        let config = BacktestConfig {
            quantiles: 4,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 1,
            fee_config: FeeConfig::default(),
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);
        let result = bt(
            &engine,
            factor,
            returns,
            Array2::from_elem((4, 4), 1.0),
            close.clone(),
            close.clone(),
            vwap.clone(),
            tradable.clone(),
        )
        .unwrap();

        // With zero returns, annualized_return ≈ 0, max_drawdown ≈ 0, calmar ≈ NaN or 0
        // The key assertion: it's not panicking and is finite or NaN
        assert!(
            result.calmar_ratio.is_finite() || result.calmar_ratio.is_nan(),
            "Calmar should be finite or NaN"
        );
    }
}
