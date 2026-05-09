//! Strategy layer: combine multiple factors into a single signal matrix.
//!
//! Sits between factor computation (FactorRegistry) and the optimizer.
//!
//! ```text
//! factor_1 ... factor_N (Array2<f64>, n_days × n_assets)
//!         │
//!         ▼
//!    Strategy::combine()
//!         │
//!         ▼
//!      signal (Array2<f64>)
//! ```

pub mod compression;
pub mod equal;
pub mod ic_based;
pub mod pca;
pub mod regression;
pub mod ridge;
pub mod state_aware;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Unified error type matching the rest of the codebase.
pub type Result<T> = std::result::Result<T, String>;

// ═══════════════════════════════════════════════════════════════════
//  Strategy trait
// ═══════════════════════════════════════════════════════════════════

/// Combines multiple factor matrices into a single signal matrix.
///
/// # Design constraints
///
/// - `fit()` must be called exactly once before any `combine()` call.
/// - Implementations must NOT leak `forward_returns` information from
///   `fit()` into any state that `combine()` could access.
/// - L0 (parameter-free) strategies have an empty `fit()`.
pub trait Strategy: Send + Sync {
    /// Fit the strategy using training-period data.
    ///
    /// * `factors` — slice of factor matrices, each (n_days × n_assets).
    /// * `forward_returns` — forward returns (n_days × n_assets), where
    ///   `returns[t][a]` is the return from t to t+1 for asset a.
    fn fit(&mut self, factors: &[Array2<f64>], forward_returns: &Array2<f64>) -> Result<()>;

    /// Combine factors into a signal matrix (n_days × n_assets).
    ///
    /// Positive values indicate long-tilt, negative short-tilt.
    /// Absolute magnitude represents conviction.
    /// Must be callable without prior `fit()` for L0 strategies.
    fn combine(&self, factors: &[Array2<f64>]) -> Result<Array2<f64>>;

    /// Human-readable name.
    fn name(&self) -> &str;
}

// ═══════════════════════════════════════════════════════════════════
//  StrategyConfig
// ═══════════════════════════════════════════════════════════════════

/// Serializable strategy configuration.
///
/// Uses `#[serde(tag = "type")]` for self-describing JSON:
/// ```json
/// {"type": "ridge_combine", "alpha": 0.1}
/// {"type": "factor_zoo_compress", "n_components": 5, "rotate": true}
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StrategyConfig {
    // ── L0 Static ──
    #[serde(rename = "equal_weight")]
    EqualWeight,
    #[serde(rename = "rank_average")]
    RankAverage,
    #[serde(rename = "signal_weighted")]
    SignalWeighted,

    // ── L1 History-driven ──
    #[serde(rename = "ic_weighted")]
    ICWeighted {
        /// Lookback window in trading days for rolling IC estimation.
        /// `None` uses the full training period.
        #[serde(default)]
        lookback_days: Option<usize>,
    },
    #[serde(rename = "ic_ir_weighted")]
    ICIRWeighted {
        #[serde(default)]
        lookback_days: Option<usize>,
    },
    #[serde(rename = "factor_zoo_compress")]
    FactorZooCompress {
        /// Number of principal components to retain.
        n_components: usize,
        /// Whether to apply varimax rotation for interpretability.
        #[serde(default)]
        rotate: bool,
    },

    // ── L2 Supervised learning ──
    #[serde(rename = "ridge_combine")]
    RidgeCombine {
        /// Regularization strength λ.
        #[serde(default = "default_alpha")]
        alpha: f64,
    },

    // ── L3 Dynamic timing ──
    #[serde(rename = "state_aware")]
    StateAware {
        /// Number of market states (regimes) to identify.
        #[serde(default = "default_n_states")]
        n_states: usize,
        /// Lookback days for state-conditional IC estimation.
        #[serde(default = "default_ic_lookback")]
        ic_lookback: usize,
    },
    #[serde(rename = "factor_comfort_zone")]
    FactorComfortZone {
        #[serde(default = "default_n_states")]
        n_states: usize,
        /// Percentile threshold: a factor is active if its IC in the
        /// current state exceeds this percentile of its IC distribution
        /// across all states.
        #[serde(default = "default_activity_pctile")]
        activity_pctile: f64,
    },
}

fn default_alpha() -> f64 {
    1.0
}
fn default_n_states() -> usize {
    3
}
fn default_ic_lookback() -> usize {
    252
}
fn default_activity_pctile() -> f64 {
    0.7
}

impl StrategyConfig {
    /// Build the corresponding `Strategy` trait object.
    pub fn build(&self) -> Box<dyn Strategy> {
        match self {
            Self::EqualWeight => Box::new(equal::EqualWeight),
            Self::RankAverage => Box::new(equal::RankAverage),
            Self::SignalWeighted => Box::new(equal::SignalWeighted),
            Self::ICWeighted { lookback_days } => {
                Box::new(ic_based::ICWeighted::new(*lookback_days))
            }
            Self::ICIRWeighted { lookback_days } => {
                Box::new(ic_based::ICIRWeighted::new(*lookback_days))
            }
            Self::FactorZooCompress {
                n_components,
                rotate,
            } => Box::new(compression::FactorZooCompress::new(*n_components, *rotate)),
            Self::RidgeCombine { alpha } => Box::new(regression::RidgeCombine::new(*alpha)),
            Self::StateAware {
                n_states,
                ic_lookback,
            } => Box::new(state_aware::StateAware::new(*n_states, *ic_lookback)),
            Self::FactorComfortZone {
                n_states,
                activity_pctile,
            } => Box::new(state_aware::FactorComfortZone::new(
                *n_states,
                *activity_pctile,
            )),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Helper types
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
//  Shared helper functions
// ═══════════════════════════════════════════════════════════════════

/// Cross-sectionally z-score each row (day).
///
/// For each row, computes mean and std using only finite values.
/// Cells with NaN input or zero std produce NaN output.
pub(crate) fn zscore_rows(mat: &Array2<f64>) -> Array2<f64> {
    let (n_rows, n_cols) = mat.dim();
    let mut out = Array2::zeros((n_rows, n_cols));
    for t in 0..n_rows {
        let row = mat.row(t);
        let finite: Vec<f64> = row.iter().filter(|&&x| x.is_finite()).copied().collect();
        if finite.len() < 2 {
            out.row_mut(t).fill(f64::NAN);
            continue;
        }
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        let var =
            finite.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (finite.len() - 1) as f64;
        let std = var.sqrt();
        if std < 1e-15 {
            out.row_mut(t).fill(0.0);
            continue;
        }
        for a in 0..n_cols {
            let v = mat[[t, a]];
            out[[t, a]] = if v.is_finite() {
                (v - mean) / std
            } else {
                f64::NAN
            };
        }
    }
    out
}

/// Z-score each column (asset) across the time axis.
pub(crate) fn zscore_cols(mat: &Array2<f64>) -> Array2<f64> {
    let (n_rows, n_cols) = mat.dim();
    let mut out = Array2::zeros((n_rows, n_cols));
    for a in 0..n_cols {
        let col = mat.column(a);
        let finite: Vec<f64> = col.iter().filter(|&&x| x.is_finite()).copied().collect();
        if finite.len() < 2 {
            out.column_mut(a).fill(f64::NAN);
            continue;
        }
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        let var =
            finite.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (finite.len() - 1) as f64;
        let std = var.sqrt();
        if std < 1e-15 {
            out.column_mut(a).fill(0.0);
            continue;
        }
        for t in 0..n_rows {
            let v = mat[[t, a]];
            out[[t, a]] = if v.is_finite() {
                (v - mean) / std
            } else {
                f64::NAN
            };
        }
    }
    out
}

/// Convert values to cross-sectional percentiles [0, 1] using
/// average-rank tie-breaking.
pub(crate) fn to_percentiles(mat: &Array2<f64>) -> Array2<f64> {
    let (n_rows, n_cols) = mat.dim();
    let mut out = Array2::zeros((n_rows, n_cols));
    for t in 0..n_rows {
        let mut pairs: Vec<(usize, f64)> = (0..n_cols)
            .filter(|&a| mat[[t, a]].is_finite())
            .map(|a| (a, mat[[t, a]]))
            .collect();
        if pairs.is_empty() {
            out.row_mut(t).fill(f64::NAN);
            continue;
        }
        pairs.sort_by(|a, b| a.1.total_cmp(&b.1));
        let n = pairs.len();
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < n && pairs[j].1 == pairs[i].1 {
                j += 1;
            }
            let avg_rank = ((i + j - 1) as f64) / 2.0 / (n - 1) as f64;
            for k in i..j {
                let a = pairs[k].0;
                out[[t, a]] = avg_rank;
            }
            i = j;
        }
    }
    out
}

/// Compute IC for a single factor against forward returns.
///
/// Delegates to `backtest::metrics::compute_ic_series`, which returns
/// `(ic_series: Array1<f64>, ic_mean: f64, ic_ir: f64)`.
pub(crate) fn compute_factor_ic(
    factor: &Array2<f64>,
    forward_returns: &Array2<f64>,
) -> Result<(ndarray::Array1<f64>, f64, f64)> {
    crate::backtest::metrics::compute_ic_series(factor, forward_returns)
}

/// Validate inputs to `fit()`.
pub(crate) fn validate_fit_input(
    factors: &[Array2<f64>],
    forward_returns: &Array2<f64>,
) -> Result<()> {
    if factors.is_empty() {
        return Err("no factors".into());
    }
    let shape = factors[0].dim();
    for (i, f) in factors.iter().enumerate().skip(1) {
        if f.dim() != shape {
            return Err(format!(
                "factor shape mismatch: factor[0]={:?}, factor[{}]={:?}",
                shape,
                i,
                f.dim()
            ));
        }
    }
    if shape != forward_returns.dim() {
        return Err(format!(
            "factor/return shape mismatch: factor={:?}, returns={:?}",
            shape,
            forward_returns.dim()
        ));
    }
    if shape.0 < 2 {
        return Err("insufficient days: need >= 2 rows".into());
    }
    Ok(())
}

/// Validate inputs to `combine()`.
pub(crate) fn validate_combine_input(factors: &[Array2<f64>]) -> Result<()> {
    if factors.is_empty() {
        return Err("no factors".into());
    }
    let shape = factors[0].dim();
    for (i, f) in factors.iter().enumerate().skip(1) {
        if f.dim() != shape {
            return Err(format!(
                "factor shape mismatch: factor[0]={:?}, factor[{}]={:?}",
                shape,
                i,
                f.dim()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    // SYNTHETIC DATA: all test data is hand-constructed ndarray matrices,
    // no external data sources.

    #[test]
    fn syn_zscore_rows_basic() {
        let mat = arr2(&[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]);
        let z = zscore_rows(&mat);
        // Row 0: mean=2, std=1 -> [−1, 0, 1]
        assert!((z[[0, 0]] + 1.0).abs() < 1e-10);
        assert!(z[[0, 1]].abs() < 1e-10);
        assert!((z[[0, 2]] - 1.0).abs() < 1e-10);
        // Row 1: std=0 → all zeros
        assert!(z[[1, 0]].abs() < 1e-10);
        assert!(z[[1, 1]].abs() < 1e-10);
        assert!(z[[1, 2]].abs() < 1e-10);
    }

    #[test]
    fn syn_zscore_rows_with_nan() {
        let mat = arr2(&[[1.0, f64::NAN, 3.0], [0.0, 2.0, 4.0]]);
        let z = zscore_rows(&mat);
        // Row 0: mean=2, std=√2 → [−1/√2, NaN, 1/√2]
        assert!((z[[0, 0]] + 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);
        assert!(z[[0, 1]].is_nan());
        assert!((z[[0, 2]] - 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn syn_zscore_cols_basic() {
        let mat = arr2(&[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]);
        let z = zscore_cols(&mat);
        // Col 0: mean=2, std=1 → [−1, 0, 1]
        assert!((z[[0, 0]] + 1.0).abs() < 1e-10);
        assert!(z[[1, 0]].abs() < 1e-10);
        assert!((z[[2, 0]] - 1.0).abs() < 1e-10);
        // Col 1: mean=20, std=10 → [−1, 0, 1]
        assert!((z[[0, 1]] + 1.0).abs() < 1e-10);
        assert!(z[[1, 1]].abs() < 1e-10);
        assert!((z[[2, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn syn_percentiles_ordering() {
        let mat = arr2(&[[3.0, 1.0, 2.0]]);
        let p = to_percentiles(&mat);
        // 1.0→0.0, 2.0→0.5, 3.0→1.0 (with n=3)
        assert!((p[[0, 0]] - 1.0).abs() < 1e-10);
        assert!(p[[0, 1]].abs() < 1e-10);
        assert!((p[[0, 2]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn syn_percentiles_ties() {
        let mat = arr2(&[[2.0, 2.0, 1.0]]);
        let p = to_percentiles(&mat);
        // 1.0→0.0, 2.0 and 2.0 both → 0.75 (avg rank of positions 1 and 2)
        assert!(p[[0, 2]].abs() < 1e-10);
        assert!((p[[0, 0]] - 0.75).abs() < 1e-10);
        assert!((p[[0, 1]] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn syn_strategy_config_serialization() {
        let config = StrategyConfig::RidgeCombine { alpha: 0.5 };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"type\":\"ridge_combine\""));
        assert!(json.contains("\"alpha\":0.5"));

        let parsed: StrategyConfig = serde_json::from_str(&json).unwrap();
        match parsed {
            StrategyConfig::RidgeCombine { alpha } => assert!((alpha - 0.5).abs() < 1e-10),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn syn_validate_fit_empty() {
        let ret = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        assert!(validate_fit_input(&[], &ret).is_err());
    }

    #[test]
    fn syn_validate_fit_shape_mismatch() {
        let f1 = Array2::zeros((2, 3));
        let f2 = Array2::zeros((2, 4));
        let ret = Array2::zeros((2, 3));
        assert!(validate_fit_input(&[f1, f2], &ret).is_err());
    }

    #[test]
    fn syn_validate_combine_ok() {
        let f1 = Array2::zeros((2, 3));
        let f2 = Array2::zeros((2, 3));
        assert!(validate_combine_input(&[f1, f2]).is_ok());
    }
}
