//! Optimizer — portfolio weight optimization module.
//!
//! Transforms strategy signals into position weight matrices, sitting between
//! the strategy layer and the backtest engine.
//!
//! ```text
//! signal (Array2<f64>)
//!         │
//!         ▼
//!    Optimizer::optimize_day()  × N_days  (rayon parallel)
//!         │
//!         ▼
//!    weights (Array2<f64>, each row sums to 1 or 0)
//! ```
//!
//! Supports two usage modes:
//! 1. **Single-factor group backtest** (compatible with existing flow):
//!    factor → quantile groups → group weights → backtest
//! 2. **Multi-factor direct-weight backtest** (new):
//!    Strategy::combine() → Optimizer → BacktestEngine::run_weights()
//!
//! Research basis: `docs/opt.md` (18 reports across performance attribution,
//! turnover/cost optimization, and portfolio weight optimization).

pub mod alpha;
pub mod attribution;
pub mod black_litterman;
pub mod constraints;
pub mod cost_model;
pub mod covariance;
pub mod equal;
pub mod mvo;
pub mod risk_parity;
pub mod solver;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

// ── Optimizer trait ──────────────────────────────────────────────────────────

/// Trait for portfolio optimizers: maps a single-day signal vector to
/// position weights.
pub trait Optimizer: Send + Sync {
    /// Optimize weights for a single cross-section (one trading day).
    ///
    /// # Parameters
    /// - `signal`: signal/alpha vector for the current day `(n_assets,)`
    /// - `covariance`: covariance matrix `(n_assets × n_assets)`, estimated
    ///   externally from historical returns
    /// - `prev_weights`: previous day's weights, for turnover control
    ///   (`None` on the first day)
    /// - `constraints`: optimization constraints
    /// - `cost_model`: optional transaction cost model
    ///
    /// # Returns
    /// Weight vector `(n_assets,)` satisfying `Σw = 1` or `Σw = 0`
    /// (market_neutral).
    fn optimize_day(
        &self,
        signal: &Array1<f64>,
        covariance: &Array2<f64>,
        prev_weights: Option<&Array1<f64>>,
        constraints: &OptimizerConstraints,
        cost_model: Option<&dyn CostModel>,
    ) -> Result<Array1<f64>, String>;
}

// ── Config types ─────────────────────────────────────────────────────────────

/// Optimizer configuration (serde-serializable, TOML-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Optimization method
    pub method: OptimizerMethod,
    /// Constraint set
    pub constraints: OptimizerConstraints,
    /// Signal → Alpha mapping pipeline
    #[serde(default)]
    pub alpha_pipeline: AlphaPipeline,
    /// Covariance estimator type
    #[serde(default)]
    pub cov_estimator: CovEstimatorType,
    /// Covariance estimation lookback window (trading days)
    #[serde(default = "default_cov_lookback")]
    pub cov_lookback: usize,
    /// Transaction cost model config (`None` = ignore costs)
    #[serde(default)]
    pub cost_model: Option<CostModelConfig>,
    /// Turnover penalty coefficient λ (`None` = no penalty).
    ///
    /// Objective: `max wᵀα - λ·||w - w_prev||₂² - cost(w, w_prev)`
    #[serde(default)]
    pub turnover_penalty: Option<f64>,
}

fn default_cov_lookback() -> usize {
    60
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            method: OptimizerMethod::EqualWeight,
            constraints: OptimizerConstraints::default(),
            alpha_pipeline: AlphaPipeline::default(),
            cov_estimator: CovEstimatorType::Sample,
            cov_lookback: 60,
            cost_model: None,
            turnover_penalty: None,
        }
    }
}

/// Optimization method.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizerMethod {
    /// 1/K for top-K assets by signal
    EqualWeight,
    /// Weight proportional to signal value (positive-only)
    SignalProportional,
    /// Weight proportional to inverse volatility
    VolatilityInverse,
    /// max wᵀα / √(wᵀΣw)
    MaxSharpe,
    /// min wᵀΣw
    MinVariance,
    /// max wᵀα / √(wᵀΣw) with factor exposure constraints
    MaxIR,
    /// Equal risk contribution: RC_i = RC_j for all i, j
    RiskParity,
    /// max (wᵀσ) / √(wᵀΣw)
    MaxDiversification,
    /// Black-Litterman Bayesian fusion
    BlackLitterman(BlackLittermanConfig),
}

/// Black-Litterman configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlackLittermanConfig {
    /// Prior weight (typically 1/T where T is the number of observations)
    pub tau: f64,
    /// Risk aversion coefficient (typically 2.5–3.0)
    pub delta: f64,
    /// Market-cap equilibrium weights
    pub equilibrium_weights: Vec<f64>,
    /// View portfolios (K × N pick matrix, stored row-major)
    pub view_pick: Vec<Vec<f64>>,
    /// View expected returns (K × 1)
    pub view_returns: Vec<f64>,
    /// View confidences c_k ∈ [0, 1] (K × 1)
    pub view_confidences: Option<Vec<f64>>,
}

impl Default for BlackLittermanConfig {
    fn default() -> Self {
        Self {
            tau: 0.05,
            delta: 3.0,
            equilibrium_weights: vec![],
            view_pick: vec![],
            view_returns: vec![],
            view_confidences: None,
        }
    }
}

// ── Batch scheduling ─────────────────────────────────────────────────────────

/// Run optimizer across all trading days, producing a weight matrix.
///
/// Uses rayon for parallel per-day optimization. Covariance is estimated
/// via a rolling window on the returns matrix.
///
/// # Parameters
/// - `optimizer`: optimizer instance
/// - `signals`: per-day signal matrix `(n_days × n_assets)`
/// - `cov_estimator`: covariance estimator
/// - `returns`: return matrix `(n_days × n_assets)` for covariance estimation
/// - `constraints`: optimization constraints
/// - `config`: optimizer configuration
///
/// # Returns
/// Per-day weight matrix `(n_days × n_assets)`.
pub fn optimize(
    optimizer: &dyn Optimizer,
    signals: &Array2<f64>,
    cov_estimator: &dyn CovEstimator,
    returns: &Array2<f64>,
    constraints: &OptimizerConstraints,
    config: &OptimizerConfig,
) -> Result<Array2<f64>, String> {
    let (n_days, n_assets) = signals.dim();

    if returns.ncols() != n_assets {
        return Err(format!(
            "returns.ncols ({}) != signals.ncols ({})",
            returns.ncols(),
            n_assets
        ));
    }

    let mut weights = Array2::<f64>::zeros((n_days, n_assets));
    let cost_model: Option<Box<dyn CostModel>> =
        config.cost_model.as_ref().map(|c| c.build()).transpose()?;

    // Sequential loop — each day depends on the previous day's weights.
    // Per-day optimization is self-contained; the rolling covariance
    // estimation is the bottleneck.
    for day in 0..n_days {
        let start = if day >= config.cov_lookback {
            day - config.cov_lookback
        } else {
            0
        };
        let _window_size = day - start + 1;

        let ret_window = returns.slice(ndarray::s![start..=day, ..]);
        let ret_mat = ret_window.to_owned();

        let cov = match cov_estimator.estimate(&ret_mat) {
            Ok(c) => c,
            Err(_) if ret_mat.nrows() < 2 => {
                // Not enough data yet — use identity
                Array2::eye(ret_mat.ncols())
            }
            Err(e) => return Err(e),
        };

        let alpha = config
            .alpha_pipeline
            .transform(&signals.row(day).to_owned());

        let prev = if day > 0 {
            Some(weights.row(day - 1).to_owned())
        } else {
            None
        };

        let w_day = optimizer.optimize_day(
            &alpha,
            &cov,
            prev.as_ref(),
            constraints,
            cost_model.as_deref(),
        )?;

        weights.row_mut(day).assign(&w_day);
    }

    Ok(weights)
}

/// Convenience: auto-build the estimator from `config.cov_estimator`
/// and wrap the full optimization pipeline.
pub fn optimize_from_config(
    optimizer: &dyn Optimizer,
    signals: &Array2<f64>,
    returns: &Array2<f64>,
    config: &OptimizerConfig,
) -> Result<Array2<f64>, String> {
    let cov_estimator = config.cov_estimator.build()?;
    optimize(
        optimizer,
        signals,
        cov_estimator.as_ref(),
        returns,
        &config.constraints,
        config,
    )
}

// ── Re-exports ───────────────────────────────────────────────────────────────

// These modules are declared above but their public types are imported here
// to avoid circular references. The actual `use` statements are at the end
// to allow all types to be defined first.
pub use alpha::{AlphaMapping, AlphaPipeline};
pub use constraints::{
    FactorExposureConstraint, GroupConstraint, OptimizerConstraints, check_feasibility,
};
pub use cost_model::{CostModel, CostModelConfig, LinearCost, NoCost, QuadraticImpact};
pub use covariance::{CovEstimator, CovEstimatorType};
pub use equal::{EqualWeight, SignalProportional, VolatilityInverse};

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::covariance::SampleCov;
    use crate::opt::mvo::MaxSharpe;

    /// [SYNTHETIC] End-to-end: synthetic data → covariance → optimize → weights.
    #[test]
    fn test_optimize_e2e() {
        let n_days = 30;
        let n_assets = 5;

        // Generate random signals and returns
        let mut signals = Array2::<f64>::zeros((n_days, n_assets));
        let mut returns = Array2::<f64>::zeros((n_days, n_assets));
        for d in 0..n_days {
            for i in 0..n_assets {
                signals[[d, i]] = rand::random::<f64>() * 0.1;
                returns[[d, i]] = rand::random::<f64>() * 0.02;
            }
        }

        let cov_est = SampleCov::default();
        let optimizer = MaxSharpe::default();
        let constraints = OptimizerConstraints {
            long_only: true,
            full_investment: true,
            ..Default::default()
        };
        let config = OptimizerConfig {
            method: OptimizerMethod::MaxSharpe,
            cov_lookback: 5,
            ..Default::default()
        };

        let weights = optimize(
            &optimizer,
            &signals,
            &cov_est,
            &returns,
            &constraints,
            &config,
        )
        .unwrap();

        // Check shape
        assert_eq!(weights.dim(), (n_days, n_assets));

        // Each day from cov_lookback onward: sum ≈ 1, all non-negative
        // (early days may have insufficient data for covariance)
        for d in config.cov_lookback..n_days {
            let row = weights.row(d);
            let s: f64 = row.sum();
            assert!((s - 1.0).abs() < 1e-8, "day {d}: sum = {s}");
            for i in 0..n_assets {
                assert!(row[i] >= -1e-12, "day {d} asset {i}: w = {}", row[i]);
            }
        }
    }

    /// [SYNTHETIC] TOML roundtrip: serialize → deserialize → verify equality.
    #[test]
    fn test_config_toml_roundtrip() {
        let config = OptimizerConfig {
            method: OptimizerMethod::MaxSharpe,
            constraints: OptimizerConstraints {
                long_only: true,
                max_position: Some(0.1),
                max_assets: Some(100),
                full_investment: true,
                ..Default::default()
            },
            cov_lookback: 60,
            turnover_penalty: Some(0.1),
            ..Default::default()
        };

        let toml_str = toml::to_string(&config).unwrap();
        let restored: OptimizerConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(restored.cov_lookback, config.cov_lookback);
        assert_eq!(restored.constraints.long_only, config.constraints.long_only);
        assert_eq!(
            restored.constraints.max_position,
            config.constraints.max_position
        );
        assert_eq!(restored.turnover_penalty, config.turnover_penalty);
    }
}
