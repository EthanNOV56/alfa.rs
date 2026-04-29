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

use crate::data::layer::PriceMatrix;
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

/// Limit up/down handling configuration
#[derive(Debug, Clone)]
pub struct LimitUpDownConfig {
    /// Whether to enable limit up/down handling
    pub enabled: bool,
}

impl Default for LimitUpDownConfig {
    fn default() -> Self {
        Self {
            enabled: false,
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
    /// Limit up/down handling configuration
    pub limit_up_down_config: LimitUpDownConfig,
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
            limit_up_down_config: LimitUpDownConfig::default(),
        }
    }
}

/// Enhanced backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Trading dates (YYYYMMDD), length n_days
    pub dates: Vec<i64>,
    /// Group returns (quantile-based)
    pub group_returns: Array2<f64>,
    /// Group cumulative returns
    pub group_cum_returns: Array2<f64>,
    /// Long-short daily returns
    pub long_short_returns: Array1<f64>,
    /// Long-short cumulative return (final scalar)
    pub long_short_cum_return: f64,
    /// Long-short cumulative NAV curve [n_days-1], starts at 1.0
    pub long_short_cum_returns: Array1<f64>,
    /// Long leg cumulative NAV curve [n_days-1]
    pub long_cum_returns: Array1<f64>,
    /// Short leg cumulative NAV curve [n_days-1]
    pub short_cum_returns: Array1<f64>,
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

impl BacktestResult {
    /// Write NAV curves to CSV file (date,nv,group).
    pub fn to_csv<P: AsRef<std::path::Path>>(&self, path: P) -> csv::Result<()> {
        let mut wtr = csv::Writer::from_path(&path)?;
        self.write_nav_csv(&mut wtr)
    }

    /// Write group NAV curves to CSV writer.
    pub fn write_nav_csv<W: std::io::Write>(
        &self,
        wtr: &mut csv::Writer<W>,
    ) -> csv::Result<()> {
        let dates = &self.dates;
        wtr.write_record(&["date", "nv", "group"])?;
        let fmt_date = |d: i64| -> String {
            let yr = d / 10000;
            let mo = (d % 10000) / 100;
            let dy = d % 100;
            format!("{:04}-{:02}-{:02}", yr, mo, dy)
        };
        for g in 0..self.group_returns.ncols() {
            wtr.write_record(&[&fmt_date(dates[0]), "1.0", &g.to_string()])?;
            for t in 0..self.group_returns.nrows() {
                let nv = 1.0 + self.group_cum_returns[[t, g]];
                let date_idx = t + 1;
                if date_idx < dates.len() {
                    wtr.write_record(&[
                        &fmt_date(dates[date_idx]),
                        &nv.to_string(),
                        &g.to_string(),
                    ])?;
                }
            }
        }
        // Write long-short NAV curve
        wtr.write_record(&[&fmt_date(dates[0]), "1.0", "long_short"])?;
        for t in 0..self.long_short_cum_returns.len() {
            let date_idx = t + 1;
            if date_idx < dates.len() {
                wtr.write_record(&[
                    &fmt_date(dates[date_idx]),
                    &self.long_short_cum_returns[t].to_string(),
                    "long_short",
                ])?;
            }
        }
        Ok(())
    }
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
        let group_labels = Self::compute_quantile_groups(&factor, quantiles)?;

        // Compute group weights based on factor values
        let group_weights = Self::compute_group_weights(
            &factor, &group_labels, quantiles, self.config.weight_method,
        );

        let fee_rate = self.config.fee_config.commission_rate
            + self.config.fee_config.slippage.normal_slippage_rate;
        let group_returns = Self::simulate_groups(
            &group_labels, quantiles, n_days, n_assets,
            &tradable, &open, &close, &vwap, fee_rate,
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
        let (ic_series, ic_mean, ic_ir) = Self::compute_ic_series(&factor, &returns)?;

        // Compute cumulative NAV curves from daily returns
        let long_short_cum_returns = Self::cumulative_nav_curve(&long_short_returns);
        let long_cum_returns = Self::cumulative_nav_curve(&long_returns);
        let short_cum_returns = Self::cumulative_nav_curve(&short_returns);

        // Compute final cumulative return (log method for numerical stability)
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
            factor, prices, &group_labels, quantiles,
            &prices.returns, &adj_factor,
            &prices.close, &prices.open,
            &prices.vwap, &prices.tradable,
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

    // /// Compute weight matrix from factor using quantile-based approach
    // ///
    // /// Returns a weight matrix of shape [n_days, n_symbols] where:
    // /// - Long positions (top quantiles): positive weights
    // /// - Short positions (bottom quantiles): negative weights
    // /// - Other positions: zero weights
    // fn compute_weight_matrix_from_factor(
    //     factor: &Array2<f64>,
    //     group_labels: &Array2<usize>,
    //     n_days: usize,
    //     n_assets: usize,
    //     quantiles: usize,
    //     long_top_n: usize,
    //     short_top_n: usize,
    //     position_config: &PositionConfig,
    // ) -> Array2<f64> {
    //     // Validate parameters - fail fast with clear error messages
    //     if long_top_n == 0 || long_top_n > quantiles {
    //         panic!(
    //             "long_top_n must be in range [1, quantiles], got {} but quantiles={}",
    //             long_top_n, quantiles
    //         );
    //     }
    //     if short_top_n == 0 || short_top_n > quantiles {
    //         panic!(
    //             "short_top_n must be in range [1, quantiles], got {} but quantiles={}",
    //             short_top_n, quantiles
    //         );
    //     }

    //     let mut weights = Array2::<f64>::zeros((n_days, n_assets));

    //     let long_groups: Vec<usize> = (quantiles - long_top_n + 1..=quantiles).collect();
    //     let short_groups: Vec<usize> = (1..=short_top_n).collect();

    //     let long_ratio = position_config.long_ratio;
    //     let short_ratio = position_config.short_ratio;

    //     for day in 0..n_days {
    //         // Count long and short assets
    //         let mut long_indices = Vec::new();
    //         let mut short_indices = Vec::new();

    //         for asset in 0..n_assets {
    //             let label = group_labels[[day, asset]];
    //             if long_groups.contains(&label) {
    //                 long_indices.push(asset);
    //             } else if short_groups.contains(&label) {
    //                 short_indices.push(asset);
    //             }
    //         }

    //         // Compute equal weights for long and short positions
    //         let long_weight = if !long_indices.is_empty() {
    //             long_ratio / long_indices.len() as f64
    //         } else {
    //             0.0
    //         };

    //         let short_weight = if !short_indices.is_empty() {
    //             -short_ratio / short_indices.len() as f64
    //         } else {
    //             0.0
    //         };

    //         // Assign weights
    //         for &idx in &long_indices {
    //             weights[[day, idx]] = long_weight;
    //         }
    //         for &idx in &short_indices {
    //             weights[[day, idx]] = short_weight;
    //         }
    //     }

    //     weights
    // }

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

            let n_valid = valid_data.len() as f64;
            let bins = quantiles as f64;

            // Use average-rank tie-breaking, matching Python:
            //   rank = cap_neued.rank(method="average")
            //   qcut = floor((rank - 1) * bins / n_valid).clip(0, bins-1)
            let mut i = 0usize;
            while i < valid_data.len() {
                // Find range of tied values
                let mut j = i + 1;
                while j < valid_data.len() && valid_data[j].1 == valid_data[i].1 {
                    j += 1;
                }
                // Average 1-based rank for tied values: (first + last) / 2
                let avg_rank = (i + 1 + j) as f64 / 2.0;
                let q = ((avg_rank - 1.0) * bins / n_valid).floor() as usize;
                let q = q.min(quantiles - 1) + 1; // 1-based group label
                for k in i..j {
                    let (asset_idx, _) = valid_data[k];
                    groups[[day, asset_idx]] = q;
                }
                i = j;
            }
        }

        Ok(groups)
    }

    /// Simulate per-group NAV using pre-computed group labels.
    ///
    /// Shared between `run()` (auto-computed labels) and `run_with_labels()`
    /// (pre-computed qcut).
    fn simulate_groups(
        group_labels: &Array2<usize>,
        quantiles: usize,
        n_days: usize,
        n_assets: usize,
        tradable: &Array2<f64>,
        open: &Array2<f64>,
        close: &Array2<f64>,
        vwap: &Array2<f64>,
        fee_rate: f64,
    ) -> Result<Array2<f64>, String> {
        let mut group_returns = Array2::<f64>::zeros((n_days - 1, quantiles));

        for group in 0..quantiles {
            let mut nv = 1.0f64;
            let mut prev_shares: Vec<f64> = vec![0.0f64; n_assets];

            for day in 1..n_days {
                let mut pool_count = 0usize;
                let mut in_pool = vec![false; n_assets];
                for a in 0..n_assets {
                    if group_labels[[day - 1, a]] == group + 1
                        && tradable[[day, a]] > 0.5
                    {
                        in_pool[a] = true;
                        pool_count += 1;
                    }
                }

                let mut asset = 0.0f64;
                for a in 0..n_assets {
                    if tradable[[day, a]] > 0.5 {
                        let s = prev_shares[a];
                        if s.is_finite() && s != 0.0 {
                            asset += s * open[[day, a]];
                        }
                    }
                }
                if asset <= 0.0 {
                    asset = nv;
                }

                if pool_count == 0 {
                    // No pool: carry forward ALL positions, compute close value
                    let mut asset_close = 0.0f64;
                    for a in 0..n_assets {
                        let cl = close[[day, a]];
                        if cl.is_finite() && prev_shares[a].is_finite() {
                            asset_close += prev_shares[a] * cl;
                        }
                    }
                    let new_nv = asset_close.max(0.0);
                    group_returns[[day - 1, group]] = new_nv / nv - 1.0;
                    nv = new_nv;
                    continue;
                }

                let per_stock = asset / pool_count as f64;

                let mut new_shares: Vec<f64> = vec![0.0f64; n_assets];
                let mut asset_close = 0.0f64;
                let mut fee_dollars = 0.0f64;

                for a in 0..n_assets {
                    if in_pool[a] {
                        let op = open[[day, a]];
                        if op.is_finite() && op > 0.0 {
                            new_shares[a] = per_stock / op;
                            let cl = close[[day, a]];
                            if cl.is_finite() {
                                asset_close += new_shares[a] * cl;
                            }
                        }
                    } else if tradable[[day, a]] <= 0.5 {
                        new_shares[a] = prev_shares[a];
                        let cl = close[[day, a]];
                        if cl.is_finite() {
                            asset_close += new_shares[a] * cl;
                        }
                    }

                    let delta = new_shares[a] - prev_shares[a];
                    let vp = vwap[[day, a]];
                    if delta.abs() > 1e-15 && vp.is_finite() && vp > 0.0 {
                        fee_dollars += delta.abs() * fee_rate * vp;
                    }
                }

                let new_nv = (asset_close - fee_dollars).max(0.0);
                group_returns[[day - 1, group]] = new_nv / nv - 1.0;

                nv = new_nv;
                prev_shares = new_shares;
            }
        }

        Ok(group_returns)
    }

    /// Run with pre-computed group_labels (from qcut pipeline).
    fn run_with_labels(
        &self,
        factor: Array2<f64>,
        prices: &PriceMatrix,
        group_labels: &Array2<usize>,
        quantiles: usize,
        returns: &Array2<f64>,
        adj_factor: &Array2<f64>,
        close: &Array2<f64>,
        open: &Array2<f64>,
        vwap: &Array2<f64>,
        tradable: &Array2<f64>,
    ) -> Result<BacktestResult, String> {
        let (n_days, n_assets) = factor.dim();
        let fee_rate = self.config.fee_config.commission_rate
            + self.config.fee_config.slippage.normal_slippage_rate;

        let group_returns = Self::simulate_groups(
            group_labels, quantiles, n_days, n_assets,
            tradable, open, close, vwap, fee_rate,
        )?;

        let mut long_short_returns = Array1::<f64>::zeros(n_days - 1);
        let mut long_returns = Array1::<f64>::zeros(n_days - 1);
        let long_groups: Vec<usize> = (quantiles - self.config.long_top_n..quantiles).collect();
        for day in 0..(n_days - 1) {
            let mut sum = 0.0;
            for &g in &long_groups { sum += group_returns[[day, g]]; }
            long_returns[day] = sum / self.config.long_top_n as f64 * self.config.position_config.long_ratio;
        }

        let mut short_returns = Array1::<f64>::zeros(n_days - 1);
        let short_groups: Vec<usize> = (0..self.config.short_top_n).collect();
        for day in 0..(n_days - 1) {
            let mut sum = 0.0;
            for &g in &short_groups { sum += group_returns[[day, g]]; }
            short_returns[day] = -sum / self.config.short_top_n as f64 * self.config.position_config.short_ratio;
        }

        for day in 0..(n_days - 1) {
            long_short_returns[day] = long_returns[day] + short_returns[day];
        }

        let (ic_series, ic_mean, ic_ir) = Self::compute_ic_series(&factor, returns)?;
        let long_short_cum_returns = Self::cumulative_nav_curve(&long_short_returns);
        let long_cum_returns = Self::cumulative_nav_curve(&long_returns);
        let short_cum_returns = Self::cumulative_nav_curve(&short_returns);
        let long_short_cum_return = Self::compute_total_return_log(&long_short_returns);
        let total_return = long_short_cum_return;
        let annualized_return = Self::compute_annualized_return(total_return, n_days);
        let sharpe_ratio = Self::compute_sharpe_ratio(&long_short_returns, n_days);
        let max_drawdown = Self::compute_max_drawdown(&long_short_returns);
        let turnover = Self::compute_turnover(group_labels);
        let group_cum_returns = Self::compute_cumulative_returns(&group_returns);

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

    /// Compute group weights based on factor values
    ///
    /// For each day, sorts assets by factor value (descending) and assigns to quantiles.
    /// Then computes weights within each group based on the weight method.
    ///
    /// # Parameters
    /// - factor: Factor matrix [n_days, n_symbols]
    /// - group_labels: Group labels from compute_quantile_groups [n_days, n_symbols]
    /// - quantiles: Number of quantile groups
    /// - weight_method: Weight allocation method (Equal or Weighted)
    ///
    /// # Returns
    /// - Array2<f64> of shape [n_days, n_symbols] with group weights
    fn compute_group_weights(
        factor: &Array2<f64>,
        group_labels: &Array2<usize>,
        quantiles: usize,
        weight_method: WeightMethod,
    ) -> Array2<f64> {
        let (n_days, n_assets) = factor.dim();
        let mut group_weights = Array2::<f64>::zeros((n_days, n_assets));

        for day in 0..n_days {
            let labels_today = group_labels.row(day);
            let factor_today = factor.row(day);

            // Collect valid assets for each group
            let mut group_assets: Vec<Vec<usize>> = vec![Vec::new(); quantiles];
            for asset in 0..n_assets {
                let label = labels_today[asset];
                if label > 0 && label <= quantiles {
                    group_assets[label - 1].push(asset);
                }
            }

            // Compute weights for each group
            for group in 0..quantiles {
                let assets = &group_assets[group];
                if assets.is_empty() {
                    continue;
                }

                let group_weight = 1.0 / quantiles as f64; // Each group gets equal total weight

                let weights: Vec<f64> = match weight_method {
                    WeightMethod::Equal => {
                        let w = group_weight / assets.len() as f64;
                        vec![w; assets.len()]
                    }
                    WeightMethod::Weighted => {
                        // Use factor values as weights
                        let total_factor: f64 = assets
                            .iter()
                            .map(|&idx| factor_today[idx])
                            .filter(|&v| !v.is_nan())
                            .sum();
                        if total_factor == 0.0 {
                            vec![0.0; assets.len()]
                        } else {
                            assets
                                .iter()
                                .map(|&idx| {
                                    let f = factor_today[idx];
                                    if f.is_nan() {
                                        0.0
                                    } else {
                                        group_weight * f / total_factor
                                    }
                                })
                                .collect()
                        }
                    }
                };

                // Assign weights
                for (&asset, &weight) in assets.iter().zip(weights.iter()) {
                    group_weights[[day, asset]] = weight;
                }
            }
        }

        group_weights
    }

    /// Compute weights for a specific subgroup (extract weights for target group, zero out others)
    ///
    /// # Parameters
    /// - group_weights: Full group weights [n_days, n_symbols]
    /// - group_labels: Group labels [n_days, n_symbols]
    /// - target_group: Target group index (0-based)
    ///
    /// # Returns
    /// - Array2<f64> of shape [n_days, n_symbols] with weights for target group only
    fn compute_subgroup_weights(
        group_weights: &Array2<f64>,
        group_labels: &Array2<usize>,
        target_group: usize,
    ) -> Array2<f64> {
        let (n_days, n_assets) = group_weights.dim();
        let mut subgroup_weights = Array2::<f64>::zeros((n_days, n_assets));

        for day in 0..n_days {
            for asset in 0..n_assets {
                // Labels are 1-based, convert to 0-based for comparison
                if group_labels[[day, asset]] == target_group + 1 {
                    subgroup_weights[[day, asset]] = group_weights[[day, asset]];
                }
            }
        }

        subgroup_weights
    }

    /// Compute long/short weights from group weights
    ///
    /// Long positions: highest factor groups (quantiles-1, quantiles-2, ...)
    /// Short positions: lowest factor groups (0, 1, ...)
    ///
    /// # Parameters
    /// - group_weights: Full group weights [n_days, n_symbols]
    /// - group_labels: Group labels [n_days, n_symbols]
    /// - quantiles: Number of quantile groups
    /// - long_top_n: Number of top groups to go long
    /// - short_top_n: Number of bottom groups to go short
    /// - position_config: Position configuration
    ///
    /// # Returns
    /// - Array2<f64> of shape [n_days, n_symbols] with long/short weights
    fn compute_long_short_weights(
        group_weights: &Array2<f64>,
        group_labels: &Array2<usize>,
        quantiles: usize,
        long_top_n: usize,
        short_top_n: usize,
        position_config: &PositionConfig,
    ) -> Array2<f64> {
        let (n_days, n_assets) = group_weights.dim();
        let mut long_short_weights = Array2::<f64>::zeros((n_days, n_assets));

        let long_groups: Vec<usize> = (quantiles - long_top_n..quantiles).collect();
        let short_groups: Vec<usize> = (0..short_top_n).collect();

        let long_ratio = position_config.long_ratio;
        let short_ratio = position_config.short_ratio;

        for day in 0..n_days {
            for asset in 0..n_assets {
                let label = group_labels[[day, asset]]; // 1-based
                let weight = group_weights[[day, asset]];

                if weight == 0.0 {
                    continue;
                }

                // Check if this asset is in a long group (highest factor groups)
                // Labels: 1 = lowest, quantiles = highest
                if long_groups.contains(&(label - 1)) {
                    long_short_weights[[day, asset]] = weight * long_ratio;
                }
                // Check if this asset is in a short group (lowest factor groups)
                else if short_groups.contains(&(label - 1)) {
                    long_short_weights[[day, asset]] = -weight * short_ratio;
                }
            }
        }

        long_short_weights
    }

    // fn compute_long_short_returns(
    //     group_returns: &Array2<f64>,
    //     quantiles: usize,
    //     long_top_n: usize,
    //     short_top_n: usize,
    //     position_config: &PositionConfig,
    // ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    //     let n_days = group_returns.dim().0;
    //     let mut long_returns = Array1::<f64>::zeros(n_days);
    //     let mut short_returns = Array1::<f64>::zeros(n_days);
    //     let mut long_short = Array1::<f64>::zeros(n_days);

    //     // Validate parameters - fail fast with clear error messages
    //     if long_top_n == 0 || long_top_n > quantiles {
    //         panic!(
    //             "long_top_n must be in range [1, quantiles], got {} but quantiles={}",
    //             long_top_n, quantiles
    //         );
    //     }
    //     if short_top_n == 0 || short_top_n > quantiles {
    //         panic!(
    //             "short_top_n must be in range [1, quantiles], got {} but quantiles={}",
    //             short_top_n, quantiles
    //         );
    //     }

    //     let long_groups: Vec<usize> = (quantiles - long_top_n + 1..=quantiles).collect();
    //     let short_groups: Vec<usize> = (1..=short_top_n).collect();

    //     let long_ratio = position_config.long_ratio;
    //     let short_ratio = position_config.short_ratio;
    //     let market_neutral = position_config.market_neutral;

    //     for day in 0..n_days {
    //         let long_return: f64 = long_groups
    //             .iter()
    //             .map(|&g| group_returns[[day, g - 1]])
    //             .sum::<f64>()
    //             / long_groups.len() as f64;

    //         let short_return: f64 = short_groups
    //             .iter()
    //             .map(|&g| group_returns[[day, g - 1]])
    //             .sum::<f64>()
    //             / short_groups.len() as f64;

    //         // Apply position ratios
    //         let long_position = long_return * long_ratio;
    //         let short_position = short_return * short_ratio;

    //         long_returns[day] = long_position;
    //         short_returns[day] = -short_position; // Short returns are negative

    //         if market_neutral {
    //             // Market neutral: long - short
    //             long_short[day] = long_position - short_position;
    //         } else {
    //             // Directional: long position only
    //             long_short[day] = long_position;
    //         }
    //     }

    //     (long_returns, short_returns, long_short)
    // }

    // fn apply_fees(
    //     long_short_returns: &Array1<f64>,
    //     volume: Option<&Array2<f64>>,
    //     fee_config: &FeeConfig,
    // ) -> Array1<f64> {
    //     let commission_rate = fee_config.commission_rate;
    //     let min_commission = fee_config.min_commission;
    //     let slippage_config = &fee_config.slippage;

    //     let mut result = long_short_returns.clone();

    //     // Simple fee calculation: subtract commission from each day
    //     // In real implementation, this would calculate actual position changes
    //     for i in 0..result.len() {
    //         // Apply commission
    //         let commission = commission_rate.abs();
    //         result[i] -= commission;

    //         // Apply slippage if volume is available
    //         if let Some(vol) = volume {
    //             // Get average volume for this day (approximation)
    //             let day_volumes: Vec<f64> = vol
    //                 .row(i)
    //                 .iter()
    //                 .filter(|&&v| !v.is_nan())
    //                 .cloned()
    //                 .collect();

    //             if !day_volumes.is_empty() {
    //                 let avg_volume: f64 =
    //                     day_volumes.iter().sum::<f64>() / day_volumes.len() as f64;
    //                 let slippage_rate = if avg_volume > slippage_config.large_volume_threshold {
    //                     slippage_config.large_slippage_rate
    //                 } else {
    //                     slippage_config.normal_slippage_rate
    //                 };
    //                 // Apply slippage (symmetric for long and short)
    //                 result[i] -= slippage_rate;
    //             }
    //         }

    //         // Apply minimum commission floor
    //         if result[i] < -min_commission {
    //             // This would only affect if we track actual trade values
    //         }
    //     }

    //     result
    // }

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

    /// Build cumulative NAV curve from daily returns.
    ///
    /// Each element at index `t` equals the cumulative product of `(1 + r)` from
    /// day 0 to day `t`, starting at 1.0. NaN/infinite returns are skipped (NAV
    /// stays flat for that day).
    fn cumulative_nav_curve(returns: &Array1<f64>) -> Array1<f64> {
        let n = returns.len();
        let mut curve = Array1::zeros(n);
        let mut cum = 1.0;
        for (i, &r) in returns.iter().enumerate() {
            if r.is_finite() {
                cum *= 1.0 + r;
            }
            curve[i] = cum;
        }
        curve
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

    /// Compute holding return (隔夜持仓收益) using adjusted prices
    ///
    /// Formula:
    /// - adj_close[t] = close[t] * adj_factor[t] / adj_factor[last]
    /// - holding_return[t] = (adj_close[t+1] / adj_close[t]) - 1
    ///
    /// # Parameters
    /// - weights: Weight matrix [n_days, n_symbols]
    /// - close: Close price matrix [n_days, n_symbols]
    /// - adj_factor: Adjustment factor matrix [n_days, n_symbols]
    ///
    /// # Returns
    /// - Array2<f64> of shape [n_days, 1] with holding returns per day
    pub fn compute_holding_return(
        weights: &Array2<f64>,
        close: &Array2<f64>,
        adj_factor: &Array2<f64>,
    ) -> Array2<f64> {
        let (n_days, n_symbols) = weights.dim();
        let (_, n_symbols_check) = close.dim();
        assert_eq!(
            n_symbols, n_symbols_check,
            "Weights and close must have same number of symbols"
        );

        // Compute adjusted close prices: adj_close[t] = close[t] * adj_factor[t] / adj_factor[last]
        let last_adj_factor = adj_factor.row(n_days - 1); // [n_symbols]
        let mut adj_close = Array2::<f64>::zeros((n_days, n_symbols));
        for day in 0..n_days {
            let adj_factors = adj_factor.row(day);
            for symbol in 0..n_symbols {
                let adj = adj_factors[symbol];
                let last_adj = last_adj_factor[symbol];
                if !adj.is_nan() && !last_adj.is_nan() && last_adj != 0.0 {
                    adj_close[[day, symbol]] = close[[day, symbol]] * adj / last_adj;
                } else {
                    adj_close[[day, symbol]] = close[[day, symbol]];
                }
            }
        }

        // Vectorized: previous day's weights
        // weights_lag: [n_days-1, n_symbols] = weights[0:n_days-1] (weights for days 0 to n_days-2)
        let weights_lag = weights.slice(ndarray::s![0..n_days - 1, ..]);

        // Vectorized: compute price returns for all days and symbols
        // close_lag: [n_days-1, n_symbols] = close[0:n_days-1]
        let adj_close_lag = adj_close.slice(ndarray::s![0..n_days - 1, ..]);
        // close_current: [n_days-1, n_symbols] = close[1:n_days]
        let adj_close_current = adj_close.slice(ndarray::s![1.., ..]);
        // price_returns: [n_days-1, n_symbols]
        let price_returns = (&adj_close_current / &adj_close_lag) - 1.0;

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

    /// Compute trading return (日内交易收益) using adjusted prices with limit-up/down handling
    ///
    /// Position calculation uses open price, actual execution uses vwap.
    /// If stock is not tradable (limit-up/down), position is carried forward.
    ///
    /// # Parameters
    /// - weights: Weight matrix [n_days, n_symbols]
    /// - close: Close price matrix [n_days, n_symbols] - used for mark-to-market
    /// - open: Open price matrix [n_days, n_symbols] - used for position sizing
    /// - vwap: VWAP price matrix [n_days, n_symbols] - used for fee calculation
    /// - adj_factor: Adjustment factor matrix [n_days, n_symbols]
    /// - fee: Commission fee rate (e.g., 0.0003 for 0.03%)
    /// - slippage: Slippage rate (e.g., 0.0005 for 0.05%)
    /// - tradable: Tradable mask [n_days, n_symbols] - 1.0 = can trade, 0.0 = locked
    ///
    /// # Returns
    /// - Array2<f64> of shape [n_days, 1] with trading returns per day
    pub fn compute_trading_return(
        weights: &Array2<f64>,
        close: &Array2<f64>,
        open: &Array2<f64>,
        vwap: &Array2<f64>,
        adj_factor: &Array2<f64>,
        fee: f64,
        slippage: f64,
        tradable: &Array2<f64>,
    ) -> Array2<f64> {
        let (n_days, n_symbols) = weights.dim();
        let total_cost = fee + slippage;

        // Compute adjusted prices using SIMD-friendly row-wise operations
        let last_adj_factor = adj_factor.row(n_days - 1);
        let mut adj_close = Array2::<f64>::zeros((n_days, n_symbols));
        let mut adj_open = Array2::<f64>::zeros((n_days, n_symbols));
        let mut adj_vwap = Array2::<f64>::zeros((n_days, n_symbols));

        for day in 0..n_days {
            let adj_factors = adj_factor.row(day);
            for symbol in 0..n_symbols {
                let adj = adj_factors[symbol];
                let last_adj = last_adj_factor[symbol];
                if !adj.is_nan() && !last_adj.is_nan() && last_adj != 0.0 {
                    adj_close[[day, symbol]] = close[[day, symbol]] * adj / last_adj;
                    adj_open[[day, symbol]] = open[[day, symbol]] * adj / last_adj;
                    adj_vwap[[day, symbol]] = vwap[[day, symbol]] * adj / last_adj;
                } else {
                    adj_close[[day, symbol]] = close[[day, symbol]];
                    adj_open[[day, symbol]] = open[[day, symbol]];
                    adj_vwap[[day, symbol]] = vwap[[day, symbol]];
                }
            }
        }

        // Result for all n_days
        let mut returns = ndarray::Array2::<f64>::zeros((n_days, 1));

        // Day 0: initial position establishment using open price
        // Position = weight / open (shares per unit of weight)
        // Asset at close = sum(weight[i] * close[i] for tradable[i])
        // Fee = sum(|weight[i]| * vwap[i] * cost) for all positions
        let tradable_0 = tradable.row(0);
        let adj_open_0 = adj_open.row(0);
        let adj_close_0 = adj_close.row(0);
        let adj_vwap_0 = adj_vwap.row(0);
        let weights_0 = weights.row(0);

        let mut asset_0 = 0.0;
        let mut fee_0 = 0.0;
        for symbol in 0..n_symbols {
            let w = weights_0[symbol];
            if w != 0.0 && tradable_0[symbol] > 0.5 {
                let shares = w / adj_open_0[symbol];
                asset_0 += shares * adj_close_0[symbol];
                fee_0 += w.abs() * total_cost;
            }
        }
        let nav_0 = if asset_0 > 0.0 { asset_0 - fee_0 } else { 1.0 };
        returns[[0, 0]] = 0.0; // Day 0 has no previous day to compare

        // Days 1..n_days-1: position changes with tradable constraint
        if n_days > 1 {
            let weights_lag = weights.slice(ndarray::s![0..n_days - 1, ..]);
            let weights_current = weights.slice(ndarray::s![1.., ..]);
            let adj_open_current = adj_open.slice(ndarray::s![1.., ..]);
            let adj_close_current = adj_close.slice(ndarray::s![1.., ..]);
            let adj_vwap_current = adj_vwap.slice(ndarray::s![1.., ..]);
            let tradable_current = tradable.slice(ndarray::s![1.., ..]);

            for day_idx in 0..(n_days - 1) {
                let day = day_idx + 1;
                let prev_day = day - 1;

                // Previous day's weights and tradable
                let prev_weights = weights_lag.row(day_idx);
                let prev_tradable = tradable.slice(ndarray::s![prev_day, ..]);

                // Current day's weights, tradable, prices
                let curr_weights = weights_current.row(day_idx);
                let curr_tradable = tradable_current.row(day_idx);
                let curr_open = adj_open_current.row(day_idx);
                let curr_close = adj_close_current.row(day_idx);
                let curr_vwap = adj_vwap_current.row(day_idx);

                // Calculate asset at end of current day using close
                let mut curr_asset = 0.0;
                let mut prev_asset = 0.0;
                let mut fee = 0.0;

                for symbol in 0..n_symbols {
                    let pt = prev_tradable[symbol];
                    let ct = curr_tradable[symbol];
                    let pw = prev_weights[symbol];
                    let cw = curr_weights[symbol];
                    let co = curr_open[symbol];
                    let ccl = curr_close[symbol];
                    let cv = curr_vwap[symbol];

                    // If not tradable, carry forward previous position
                    let effective_weight = if ct <= 0.5 { pw } else { cw };

                    // Previous day's asset using close price
                    if pt > 0.5 && pw != 0.0 {
                        let prev_shares = pw / adj_close.row(prev_day)[symbol];
                        prev_asset += prev_shares * ccl;
                    }

                    // Current day's asset using close
                    if ct > 0.5 && effective_weight != 0.0 {
                        let curr_shares = effective_weight / co;
                        curr_asset += curr_shares * ccl;
                    }

                    // Fee calculation following reference:
                    // delta = weight_today - weight_prev (both in dollars)
                    // fee_rate = BUY_FEE + BUY_SLPG for buys, SELL_FEE + SELL_SLPG for sells
                    // fee_dollars = delta * fee_rate (still in dollars)
                    // actual_fee = fee_dollars * vwap (converts to actual cost)
                    let weight_change = effective_weight - pw;
                    if weight_change.abs() > 1e-10 {
                        // Apply buy or sell fee rate
                        let fee_rate = if weight_change > 0.0 {
                            (fee + slippage)  // Buy: commission + slippage
                        } else {
                            (fee + slippage)  // Sell: same rates
                        };
                        fee += weight_change.abs() * fee_rate;
                    }
                }

                // Return = (current_asset - fee) / previous_asset - 1
                let nav = if prev_asset > 0.0 { curr_asset - fee } else { curr_asset };
                let day_return = if prev_asset > 0.0 { nav / prev_asset - 1.0 } else { 0.0 };
                returns[[day, 0]] = day_return;
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
    /// - adj_factor: Adjustment factor matrix [n_days, n_symbols]
    /// - fee: Commission fee rate
    /// - slippage: Slippage rate
    ///
    /// # Returns
    /// - Array2<f64> of shape [n_days, 1] with total returns per day
    pub fn compute_portfolio_return(
        weights: &Array2<f64>,
        close: &Array2<f64>,
        open: &Array2<f64>,
        vwap: &Array2<f64>,
        adj_factor: &Array2<f64>,
        fee: f64,
        slippage: f64,
        tradable: &Array2<f64>,
    ) -> Array2<f64> {
        let holding_return = Self::compute_holding_return(weights, close, adj_factor);
        let trading_return =
            Self::compute_trading_return(weights, close, open, vwap, adj_factor, fee, slippage, tradable);
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
            limit_up_down_config: Default::default(),
        };

        let engine = BacktestEngine::with_config(config);

        let result = engine
            .run(factor, returns, adj_factor, close.clone(), close.clone(), vwap.clone(), Array2::from_elem(close.dim(), 1.0))
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
            limit_up_down_config: Default::default(),
        };

        let adj_factor = Array2::from_elem((3, 4), 1.0);
        let engine = BacktestEngine::with_config(config);
        let result = engine
            .run(factor, returns.clone(), adj_factor, close.clone(), close.clone(), vwap.clone(), Array2::from_elem(close.dim(), 1.0))
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
            .run(factor, returns.clone(), adj_factor, close.clone(), close.clone(), vwap.clone(), Array2::from_elem(close.dim(), 1.0))
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
            .run(factor.clone(), returns.clone(), adj_factor, close, open, vwap, tradable)
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
        let result = engine.run(factor, returns.clone(), adj_factor, close, open, vwap, tradable);
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
        let _ = engine.run(factor, returns.clone(), adj_factor, close, open, vwap, tradable);
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
            .run(factor, returns, adj_factor, close.clone(), close.clone(), vwap.clone(), Array2::from_elem(close.dim(), 1.0))
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
            .run(factor, returns, adj_factor2, close.clone(), open.clone(), vwap.clone(), tradable.clone())
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
            .run(factor, returns, adj_factor, close.clone(), close.clone(), vwap.clone(), Array2::from_elem(close.dim(), 1.0))
            .unwrap();

        // Both should produce finite results
        assert!(result_neutral.long_short_cum_return.is_finite());
        assert!(result_directional.long_short_cum_return.is_finite());
    }
}
