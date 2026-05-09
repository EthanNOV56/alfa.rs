//! Risk management module.
//!
//! Provides portfolio risk analysis and factor-based risk attribution.
//! Consumes backtest results (weights) and price data (returns) to
//! produce a [`RiskReport`].

mod attribution;
mod barra;
mod cov_estimation;
mod simple;
mod stress;

pub use attribution::{BrinsonAttributor, PerformanceAttribution, SectorDetail};
pub use barra::BarraRiskModel;
pub use cov_estimation::{
    CovEstimator, EWMACovEstimator, LedoitWolfEstimator, LedoitWolfTarget, NeweyWestEstimator,
    SampleCovEstimator,
};
pub use simple::SimpleRiskModel;
pub use stress::{StressScenario, StressTestResult, run_stress_test};

use ndarray::Array2;
use rayon::prelude::*;
use std::collections::HashMap;

/// Risk attribution report.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct RiskReport {
    /// Total risk (annualized volatility).
    pub total_risk: f64,
    /// Systematic (factor-driven) risk — NaN when no factor decomposition is performed.
    pub systematic_risk: f64,
    /// Specific (idiosyncratic) risk — NaN when no factor decomposition is performed.
    pub specific_risk: f64,
    /// Net exposure to each style factor (weighted average of asset-level exposures).
    pub factor_exposures: HashMap<String, f64>,
    /// Fraction of total risk contributed by each factor (sums to ~1.0 for Barra model).
    pub factor_contributions: HashMap<String, f64>,
    /// R²: fraction of total variance explained by style factors.
    pub r_squared: f64,
}

impl RiskReport {
    /// Format a single-line summary of key risk metrics.
    pub fn summary(&self) -> String {
        let fmt_factor = |map: &HashMap<String, f64>| -> String {
            if map.is_empty() {
                return "n/a".into();
            }
            let mut items: Vec<_> = map.iter().collect();
            items.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
            items
                .iter()
                .take(5)
                .map(|(k, v)| format!("{}={:.4}", k, v))
                .collect::<Vec<_>>()
                .join(", ")
        };

        format!(
            "RiskReport | total={:.4} sys={:.4} spec={:.4} R²={:.4} | exposures: [{}] | contribs: [{}]",
            self.total_risk,
            self.systematic_risk,
            self.specific_risk,
            self.r_squared,
            fmt_factor(&self.factor_exposures),
            fmt_factor(&self.factor_contributions),
        )
    }
}

/// Error type for risk analysis operations.
#[derive(Debug, Clone)]
pub enum RiskError {
    /// Input dimensions are inconsistent or insufficient.
    InvalidInput(String),
    /// Numerical failure (singular matrix, non-convergence, etc.).
    ComputationFailed(String),
}

impl std::fmt::Display for RiskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskError::InvalidInput(msg) => write!(f, "Risk model invalid input: {}", msg),
            RiskError::ComputationFailed(msg) => {
                write!(f, "Risk model computation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for RiskError {}

/// Risk model trait — any implementation must be `Send + Sync` so it can be shared
/// across threads (e.g. in a Python multi-threaded context).
pub trait RiskModel: Send + Sync {
    /// Analyze portfolio risk given allocation weights, asset returns, and (optional)
    /// style-factor exposure matrices.
    ///
    /// # Parameters
    /// - `weights`: shape `(n_days, n_assets)` — daily portfolio weights
    /// - `returns`: shape `(n_days, n_assets)` — daily asset returns
    /// - `style_exposures`: slice of `K` matrices, each shape `(n_days, n_assets)`.
    ///   One matrix per style factor. May be empty for simple models.
    fn analyze(
        &self,
        weights: &Array2<f64>,
        returns: &Array2<f64>,
        style_exposures: &[Array2<f64>],
    ) -> Result<RiskReport, RiskError>;

    /// Rolling-window risk analysis over time.
    ///
    /// Slides a window of `window_size` days across the data with step `step`,
    /// calling [`analyze`] for each window. Returns one [`RiskReport`] per window.
    ///
    /// # Parameters
    /// - `window_size`: number of days per analysis window (e.g. 126 = half-year)
    /// - `step`: number of days between window starts (e.g. 21 = monthly)
    ///
    /// # Returns
    /// `Vec<RiskReport>` with length `⌈(n_days - window_size) / step⌉ + 1`.
    /// Empty if window_size > n_days.
    fn analyze_rolling(
        &self,
        weights: &Array2<f64>,
        returns: &Array2<f64>,
        style_exposures: &[Array2<f64>],
        window_size: usize,
        step: usize,
    ) -> Result<Vec<RiskReport>, RiskError> {
        let (n_days, n_assets) = returns.dim();
        if window_size > n_days || window_size == 0 || step == 0 {
            return Ok(Vec::new());
        }

        let n_windows = (n_days - window_size) / step + 1;
        let windows: Vec<(usize, usize)> = (0..n_windows)
            .map(|w| (w * step, w * step + window_size))
            .collect();

        let results: Result<Vec<RiskReport>, RiskError> = windows
            .into_par_iter()
            .map(|(start, end)| {
                let w_slice = weights.slice(ndarray::s![start..end, ..]).to_owned();
                let r_slice = returns.slice(ndarray::s![start..end, ..]).to_owned();
                let exp_slices: Vec<Array2<f64>> = style_exposures
                    .iter()
                    .map(|e| e.slice(ndarray::s![start..end, ..]).to_owned())
                    .collect();
                self.analyze(&w_slice, &r_slice, &exp_slices)
            })
            .collect();

        results
    }
}
