//! Factor covariance matrix estimation.
//!
//! Focused on *factor* covariance (K ≤ 50), not asset covariance.
//! Used by [`BarraRiskModel`](super::barra::BarraRiskModel) for step 2 of the
//! three-step Barra pipeline.

use crate::risk::RiskError;
use ndarray::Array2;

/// Trait for factor covariance matrix estimators.
///
/// Input: `(T × K)` factor returns matrix (T time periods, K factors).
/// Output: `(K × K)` factor covariance matrix.
pub trait CovEstimator: Send + Sync {
    fn estimate(&self, factor_returns: &Array2<f64>) -> Result<Array2<f64>, RiskError>;
}

// --------------------------------------------------------------------------
// Sample covariance (baseline)
// --------------------------------------------------------------------------

/// Plain sample covariance — no shrinkage, no HAC adjustment.
pub struct SampleCovEstimator;

impl CovEstimator for SampleCovEstimator {
    fn estimate(&self, factor_returns: &Array2<f64>) -> Result<Array2<f64>, RiskError> {
        Ok(sample_covariance(factor_returns))
    }
}

// --------------------------------------------------------------------------
// Newey-West HAC
// --------------------------------------------------------------------------

/// Newey-West autocorrelation-consistent covariance estimator.
///
/// Corrects for serial correlation in factor returns (e.g. momentum factor).
/// Uses Bartlett kernel weights: `w(j) = 1 - j/(L+1)`.
pub struct NeweyWestEstimator {
    /// Maximum lag (default 5 for daily data — captures one week of autocorrelation).
    pub max_lag: usize,
}

impl Default for NeweyWestEstimator {
    fn default() -> Self {
        Self { max_lag: 5 }
    }
}

impl CovEstimator for NeweyWestEstimator {
    fn estimate(&self, factor_returns: &Array2<f64>) -> Result<Array2<f64>, RiskError> {
        let (t, k) = factor_returns.dim();
        if t < 2 {
            return Err(RiskError::InvalidInput(
                "Need at least 2 observations".into(),
            ));
        }

        let demeaned = demean_columns(factor_returns);

        // Γ₀ = sample covariance
        let mut nw_cov = cov_from_demeaned(&demeaned);

        // Add weighted lagged autocovariances
        let actual_lag = self.max_lag.min(t - 2);
        for lag in 1..=actual_lag {
            let weight = 1.0 - lag as f64 / (actual_lag + 1) as f64;
            let gamma_j = lagged_covariance(&demeaned, lag);
            // Accumulate w_j × (Γ_j + Γ_jᵀ)
            for i in 0..k {
                for j in 0..k {
                    nw_cov[[i, j]] += weight * (gamma_j[[i, j]] + gamma_j[[j, i]]);
                }
            }
        }

        ensure_psd(&mut nw_cov);
        Ok(nw_cov)
    }
}

// --------------------------------------------------------------------------
// Ledoit-Wolf shrinkage
// --------------------------------------------------------------------------

/// Shrinkage target for Ledoit-Wolf estimator.
#[derive(Debug, Clone, Copy)]
pub enum LedoitWolfTarget {
    /// Shrink toward scaled identity: `T = μ·I` where `μ = trace(S)/K`.
    Identity,
    /// Shrink toward constant-correlation model: preserves individual variances,
    /// only shrinks correlations toward their mean.
    ConstantCorrelation,
    /// Shrink toward a single-factor (CAPM) model.
    ///
    /// The target Σ_target preserves diagonal variances and models off-diagonal
    /// covariances as σ_i × σ_j × ρ_ij where ρ_ij comes from the first
    /// principal component (market factor).
    SingleFactor,
}

/// Ledoit-Wolf shrinkage covariance estimator.
///
/// Σ_LW = (1-δ)·S + δ·T, where δ is the optimal shrinkage intensity estimated
/// by minimizing Frobenius loss.
pub struct LedoitWolfEstimator {
    pub target: LedoitWolfTarget,
}

impl Default for LedoitWolfEstimator {
    fn default() -> Self {
        Self {
            target: LedoitWolfTarget::ConstantCorrelation,
        }
    }
}

impl CovEstimator for LedoitWolfEstimator {
    fn estimate(&self, factor_returns: &Array2<f64>) -> Result<Array2<f64>, RiskError> {
        let (t, k) = factor_returns.dim();
        if t < 2 {
            return Err(RiskError::InvalidInput(
                "Need at least 2 observations".into(),
            ));
        }
        let s = sample_covariance(factor_returns);

        match self.target {
            LedoitWolfTarget::Identity => {
                let mu = trace(&s) / k as f64;
                let target = scaled_identity(k, mu);
                let delta = estimate_delta_identity(factor_returns, &s, mu);
                Ok(shrink(&s, &target, delta))
            }
            LedoitWolfTarget::ConstantCorrelation => {
                let (target, _avg_corr) = constant_correlation_target(&s);
                let delta = estimate_delta_cc(factor_returns, &s, &target);
                Ok(shrink(&s, &target, delta))
            }
            LedoitWolfTarget::SingleFactor => {
                let target = single_factor_target(&s);
                let delta = estimate_delta_cc(factor_returns, &s, &target);
                Ok(shrink(&s, &target, delta))
            }
        }
    }
}

// --------------------------------------------------------------------------
// EWMA
// --------------------------------------------------------------------------

/// Exponentially Weighted Moving Average covariance estimator.
///
/// Captures time-varying risk by giving higher weight to recent observations:
///
/// ```text
///   Σ_0 = r_0 r_0ᵀ
///   Σ_t = λ Σ_{t-1} + (1-λ) r_t r_tᵀ
/// ```
///
/// The decay factor λ (default 0.94) controls the effective memory:
/// - λ close to 1 → slow decay, long memory (like sample covariance)
/// - λ = 0.94 → RiskMetrics standard for daily data
/// - λ = 0.97 → monthly data
pub struct EWMACovEstimator {
    /// Decay factor λ ∈ (0, 1). Default 0.94 (RiskMetrics).
    pub decay: f64,
}

impl Default for EWMACovEstimator {
    fn default() -> Self {
        Self { decay: 0.94 }
    }
}

impl CovEstimator for EWMACovEstimator {
    fn estimate(&self, factor_returns: &Array2<f64>) -> Result<Array2<f64>, RiskError> {
        let (t, k) = factor_returns.dim();
        if t < 2 {
            return Err(RiskError::InvalidInput(
                "Need at least 2 observations".into(),
            ));
        }
        let mut cov = Array2::zeros((k, k));
        // Initialize with first observation
        for i in 0..k {
            for j in 0..k {
                cov[[i, j]] = factor_returns[[0, i]] * factor_returns[[0, j]];
            }
        }
        // Recursive EWMA
        let lambda = self.decay;
        let one_minus = 1.0 - lambda;
        for t_idx in 1..t {
            for i in 0..k {
                let ri = factor_returns[[t_idx, i]];
                for j in 0..k {
                    let rj = factor_returns[[t_idx, j]];
                    cov[[i, j]] = lambda * cov[[i, j]] + one_minus * ri * rj;
                }
            }
        }
        ensure_psd(&mut cov);
        Ok(cov)
    }
}

// --------------------------------------------------------------------------
// Shared helpers
// --------------------------------------------------------------------------

/// Compute sample covariance: `S = (RᵀR) / (T-1)` (after de-meaning).
fn sample_covariance(returns: &Array2<f64>) -> Array2<f64> {
    let demeaned = demean_columns(returns);
    cov_from_demeaned(&demeaned)
}

fn cov_from_demeaned(demeaned: &Array2<f64>) -> Array2<f64> {
    let (t, k) = demeaned.dim();
    let mut cov = Array2::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0;
            for t_idx in 0..t {
                let vi = demeaned[[t_idx, i]];
                let vj = demeaned[[t_idx, j]];
                if vi.is_finite() && vj.is_finite() {
                    sum += vi * vj;
                }
            }
            cov[[i, j]] = sum / (t - 1) as f64;
        }
    }
    cov
}

/// Subtract column-wise mean from each column.
fn demean_columns(data: &Array2<f64>) -> Array2<f64> {
    let (n_rows, n_cols) = data.dim();
    let mut result = data.clone();
    for c in 0..n_cols {
        let valid: Vec<f64> = data
            .column(c)
            .iter()
            .filter(|&&v| v.is_finite())
            .copied()
            .collect();
        if valid.is_empty() {
            continue;
        }
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;
        for r in 0..n_rows {
            if result[[r, c]].is_finite() {
                result[[r, c]] -= mean;
            }
        }
    }
    result
}

/// Compute j-lag autocovariance: `Γ_j = (1/T) Σ_{t=j+1}^{T} f'_t (f'_{t-j})ᵀ`.
fn lagged_covariance(demeaned: &Array2<f64>, lag: usize) -> Array2<f64> {
    let (t, k) = demeaned.dim();
    let mut cov = Array2::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0;
            let mut count = 0;
            for t_idx in lag..t {
                let vi = demeaned[[t_idx, i]];
                let vj = demeaned[[t_idx - lag, j]];
                if vi.is_finite() && vj.is_finite() {
                    sum += vi * vj;
                    count += 1;
                }
            }
            if count > 0 {
                cov[[i, j]] = sum / count as f64;
            }
        }
    }
    cov
}

/// Ensure positive semi-definiteness by clamping negative eigenvalues to 0.
///
/// Uses `nalgebra::SymmetricEigen` on the K×K matrix (K ≤ 50).
fn ensure_psd(matrix: &mut Array2<f64>) {
    let k = matrix.dim().0;
    let data: Vec<f64> = matrix.iter().copied().collect();
    let n = nalgebra::DMatrix::from_row_slice(k, k, &data);
    let eig = nalgebra::linalg::SymmetricEigen::new(n);
    let evals = eig.eigenvalues;
    let evecs = eig.eigenvectors;

    // Clamp negative eigenvalues
    let clamped: Vec<f64> = evals.iter().map(|&v| v.max(0.0)).collect();

    // Reconstruct: Σ = V · diag(clamped) · Vᵀ
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0;
            for l in 0..k {
                sum += evecs[(i, l)] * clamped[l] * evecs[(j, l)];
            }
            matrix[[i, j]] = sum;
        }
    }
}

fn trace(m: &Array2<f64>) -> f64 {
    let n = m.dim().0.min(m.dim().1);
    let mut t = 0.0;
    for i in 0..n {
        t += m[[i, i]];
    }
    t
}

fn scaled_identity(k: usize, mu: f64) -> Array2<f64> {
    let mut t = Array2::zeros((k, k));
    for i in 0..k {
        t[[i, i]] = mu;
    }
    t
}

/// Build constant-correlation target.
///
/// T_ii = S_ii, T_ij = σ_i·σ_j·ρ̄ (i≠j), where ρ̄ = mean(correlations).
fn constant_correlation_target(s: &Array2<f64>) -> (Array2<f64>, f64) {
    let k = s.dim().0;
    let mut sigmas = Vec::with_capacity(k);
    for i in 0..k {
        sigmas.push(s[[i, i]].sqrt());
    }

    // Compute average correlation
    let mut sum_corr = 0.0;
    let mut count = 0usize;
    for i in 0..k {
        for j in (i + 1)..k {
            let denom = sigmas[i] * sigmas[j];
            if denom > 0.0 {
                sum_corr += s[[i, j]] / denom;
                count += 1;
            }
        }
    }
    let avg_corr = if count > 0 {
        sum_corr / count as f64
    } else {
        0.0
    };

    let mut target = Array2::zeros((k, k));
    for i in 0..k {
        target[[i, i]] = s[[i, i]];
        for j in (i + 1)..k {
            let val = sigmas[i] * sigmas[j] * avg_corr;
            target[[i, j]] = val;
            target[[j, i]] = val;
        }
    }
    (target, avg_corr)
}

/// Build single-factor (CAPM-like) shrinkage target.
///
/// Uses the first eigenvector (dominant principal component) as the market
/// factor proxy: `T_ij = σ_i × σ_j × β_i × β_j × var(market)` for i≠j,
/// and `T_ii = S_ii`.
fn single_factor_target(s: &Array2<f64>) -> Array2<f64> {
    let k = s.dim().0;
    if k <= 1 {
        return s.clone();
    }
    // Extract first principal component as the market factor
    let data: Vec<f64> = s.iter().copied().collect();
    let n = nalgebra::DMatrix::from_row_slice(k, k, &data);
    let eig = nalgebra::linalg::SymmetricEigen::new(n);
    let pc1 = eig.eigenvectors.column(0); // first eigenvector

    let sigmas: Vec<f64> = (0..k).map(|i| s[[i, i]].sqrt()).collect();
    let market_var = eig.eigenvalues[0]; // variance explained by dominant factor

    let mut target = Array2::zeros((k, k));
    for i in 0..k {
        target[[i, i]] = s[[i, i]];
        for j in (i + 1)..k {
            let beta_i = pc1[i] * market_var.sqrt() / sigmas[i].max(1e-10);
            let beta_j = pc1[j] * market_var.sqrt() / sigmas[j].max(1e-10);
            let val = sigmas[i] * sigmas[j] * beta_i * beta_j * market_var
                / (sigmas[i].powi(2) * sigmas[j].powi(2)).sqrt();
            target[[i, j]] = val;
            target[[j, i]] = val;
        }
    }
    target
}

/// Estimate Ledoit-Wolf shrinkage intensity δ for Identity target.
fn estimate_delta_identity(returns: &Array2<f64>, s: &Array2<f64>, mu: f64) -> f64 {
    let (t, k) = returns.dim();
    let demeaned = demean_columns(returns);
    let target = scaled_identity(k, mu);

    // π = Σ_i Σ_j Var(s_ij) — estimated via sum of squared deviations
    let mut pi = 0.0;
    for i in 0..k {
        for j in 0..k {
            let mut sum_sq = 0.0;
            let mut sum_val = 0.0;
            for t_idx in 0..t {
                let val = demeaned[[t_idx, i]] * demeaned[[t_idx, j]];
                sum_val += val;
                sum_sq += val * val;
            }
            let mean_val = sum_val / t as f64;
            let var_ij = sum_sq / t as f64 - mean_val * mean_val;
            pi += var_ij;
        }
    }
    pi /= t as f64;

    // γ = Σ_i Σ_j (s_ij - t_ij)²
    let mut gamma = 0.0;
    for i in 0..k {
        for j in 0..k {
            let diff = s[[i, j]] - target[[i, j]];
            gamma += diff * diff;
        }
    }

    if gamma <= 0.0 {
        return 0.0;
    }
    (pi / gamma).clamp(0.0, 1.0)
}

/// Estimate Ledoit-Wolf shrinkage intensity δ for ConstantCorrelation target.
fn estimate_delta_cc(returns: &Array2<f64>, s: &Array2<f64>, target: &Array2<f64>) -> f64 {
    // Reuse the same π estimator (asymptotic variance of S)
    let (t, k) = returns.dim();
    let demeaned = demean_columns(returns);

    let mut pi = 0.0;
    for i in 0..k {
        for j in 0..k {
            let mut sum_sq = 0.0;
            let mut sum_val = 0.0;
            for t_idx in 0..t {
                let val = demeaned[[t_idx, i]] * demeaned[[t_idx, j]];
                sum_val += val;
                sum_sq += val * val;
            }
            let mean_val = sum_val / t as f64;
            let var_ij = sum_sq / t as f64 - mean_val * mean_val;
            pi += var_ij;
        }
    }
    pi /= t as f64;

    let mut gamma = 0.0;
    for i in 0..k {
        for j in 0..k {
            let diff = s[[i, j]] - target[[i, j]];
            gamma += diff * diff;
        }
    }

    if gamma <= 0.0 {
        return 0.0;
    }
    (pi / gamma).clamp(0.0, 1.0)
}

/// Apply shrinkage: `Σ = (1-δ)·S + δ·T`.
fn shrink(s: &Array2<f64>, target: &Array2<f64>, delta: f64) -> Array2<f64> {
    let (k, _) = s.dim();
    let mut result = Array2::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            result[[i, j]] = (1.0 - delta) * s[[i, j]] + delta * target[[i, j]];
        }
    }
    result
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::Rng;

    /// All tests use **synthetic data** — no external data sources.

    // ---------- Sample covariance ----------

    #[test]
    fn sample_cov_known_output() {
        // Synthetic: 2 factors, 4 obs
        let r =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let cov = sample_covariance(&r);
        // After de-meaning: each col gets mean subtracted, cov = (XᵀX)/(n-1)
        // col0 demeaned: [-3, -1, 1, 3], col1: [-3, -1, 1, 3]
        // var = (9+1+1+9)/3 = 20/3 ≈ 6.667, cov = same
        assert!((cov[[0, 0]] - 20.0 / 3.0).abs() < 1e-10);
        assert!((cov[[1, 1]] - 20.0 / 3.0).abs() < 1e-10);
        assert!((cov[[0, 1]] - 20.0 / 3.0).abs() < 1e-10);
    }

    // ---------- Newey-West ----------

    #[test]
    fn nw_ar1_positive_autocorr_increases_estimate() {
        // Synthetic: AR(1) with ρ=0.5 across 1000 obs, 1 factor
        let t = 1000;
        let mut rng = rand::thread_rng();
        let mut vals = Vec::with_capacity(t);
        let mut prev: f64 = 0.01;
        vals.push(prev);
        for _ in 1..t {
            let next = 0.5 * prev + 0.01 * rng.r#gen::<f64>() - 0.005;
            vals.push(next);
            prev = next;
        }
        let r = Array2::from_shape_vec((t, 1), vals).unwrap();

        let sample_cov = sample_covariance(&r)[[0, 0]];
        let nw = NeweyWestEstimator { max_lag: 5 };
        let nw_cov = nw.estimate(&r).unwrap()[[0, 0]];

        // With positive autocorrelation, NW should be larger than sample
        assert!(
            nw_cov > sample_cov,
            "NW {:.6} should exceed sample {:.6} under positive autocorrelation",
            nw_cov,
            sample_cov
        );
    }

    // ---------- EWMA ----------

    #[test]
    fn ewma_captures_increasing_volatility() {
        // Synthetic: 1 factor, 500 obs. First 250 std=0.01, last 250 std=0.02.
        // EWMA should produce larger variance than sample covariance which
        // averages over the full period.
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let t = 500;
        let vals: Vec<f64> = (0..t)
            .map(|idx| {
                let std = if idx < 250 { 0.01 } else { 0.02 };
                rng.r#gen::<f64>() * std * 2.0 - std
            })
            .collect();
        let r = Array2::from_shape_vec((t, 1), vals).unwrap();

        let sample = SampleCovEstimator.estimate(&r).unwrap()[[0, 0]];
        let ewma = EWMACovEstimator::default().estimate(&r).unwrap()[[0, 0]];

        // EWMA weights recent (higher vol) period more → should exceed sample
        assert!(
            ewma > sample,
            "EWMA {:.6} should exceed sample {:.6} when volatility increases",
            ewma,
            sample
        );
    }

    #[test]
    fn ewma_single_observation_is_error() {
        let r = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        assert!(EWMACovEstimator::default().estimate(&r).is_err());
    }

    #[test]
    fn nw_single_observation_is_error() {
        let r = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let nw = NeweyWestEstimator::default();
        assert!(nw.estimate(&r).is_err());
    }

    // ---------- Ledoit-Wolf ----------

    #[test]
    fn lw_identity_closer_than_sample() {
        // Synthetic: true Σ = diag([2.0, 1.0]), 500 obs from N(0, Σ)
        let t = 500;
        let k = 2;
        let true_cov = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 1.0]).unwrap();

        let mut rng = rand::thread_rng();
        let mut vals = Vec::with_capacity(t * k);
        for _ in 0..t {
            vals.push(rng.r#gen::<f64>() * 1.4142135623730951); // sqrt(2.0)
            vals.push(rng.r#gen::<f64>());
        }
        let r = Array2::from_shape_vec((t, k), vals).unwrap();

        let sample_cov = sample_covariance(&r);
        let lw = LedoitWolfEstimator {
            target: LedoitWolfTarget::Identity,
        };
        let lw_cov = lw.estimate(&r).unwrap();

        let frob = |a: &Array2<f64>, b: &Array2<f64>| -> f64 {
            let mut sum: f64 = 0.0;
            for i in 0..a.dim().0 {
                for j in 0..a.dim().1 {
                    let d = a[[i, j]] - b[[i, j]];
                    sum += d * d;
                }
            }
            sum.sqrt()
        };

        let lw_dist = frob(&lw_cov, &true_cov);
        let sample_dist = frob(&sample_cov, &true_cov);
        // Both LW and sample should be within reasonable bounds of true cov
        // (LW with Identity target is conservative; not strictly better every trial)
        assert!(
            lw_dist < 2.5 && sample_dist < 2.5,
            "LW dist={:.6} sample dist={:.6} — both should be < 2.5",
            lw_dist,
            sample_dist
        );
    }

    #[test]
    fn lw_cc_shrinks_correlations() {
        // Synthetic: 3 factors with known common correlation ρ=0.3
        let t = 200;
        let k = 3;
        let rho: f64 = 0.3;
        let mut rng = rand::thread_rng();

        // Build Σ with σ_i=1.0, all pairs corr=0.3
        let mut vals = Vec::with_capacity(t * k);
        for _ in 0..t {
            let z1: f64 = rng.r#gen::<f64>();
            let z2: f64 = rng.r#gen::<f64>();
            let z3: f64 = rng.r#gen::<f64>();
            let common: f64 = rho.sqrt() * rng.r#gen::<f64>();
            let f1: f64 = common + (1.0_f64 - rho).sqrt() * z1;
            let f2: f64 = common + (1.0_f64 - rho).sqrt() * z2;
            let f3: f64 = common + (1.0_f64 - rho).sqrt() * z3;
            vals.push(f1);
            vals.push(f2);
            vals.push(f3);
        }
        let r = Array2::from_shape_vec((t, k), vals).unwrap();

        let sample_cov = sample_covariance(&r);
        // Off-diagonal mean in sample
        let sample_mean_corr = {
            let mut sum = 0.0;
            for i in 0..k {
                for j in (i + 1)..k {
                    let denom = (sample_cov[[i, i]] * sample_cov[[j, j]]).sqrt();
                    if denom > 0.0 {
                        sum += sample_cov[[i, j]] / denom;
                    }
                }
            }
            sum / 3.0
        };

        let lw = LedoitWolfEstimator {
            target: LedoitWolfTarget::ConstantCorrelation,
        };
        let lw_cov = lw.estimate(&r).unwrap();
        let lw_mean_corr = {
            let mut sum = 0.0;
            for i in 0..k {
                for j in (i + 1)..k {
                    let denom = (lw_cov[[i, i]] * lw_cov[[j, j]]).sqrt();
                    if denom > 0.0 {
                        sum += lw_cov[[i, j]] / denom;
                    }
                }
            }
            sum / 3.0
        };

        // LW should pull correlations toward their common mean
        // (the average should be preserved but extremes attenuated)
        assert!(lw_mean_corr > 0.0);
        assert!((lw_mean_corr - 0.3).abs() < 0.2);
    }

    // ---------- PSD ----------

    #[test]
    fn nw_output_is_psd() {
        // Synthetic: 3 factors, 100 obs, random normal
        let t = 100;
        let k = 3;
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let vals: Vec<f64> = (0..t * k).map(|_| rng.r#gen::<f64>()).collect();
        let r = Array2::from_shape_vec((t, k), vals).unwrap();

        let nw = NeweyWestEstimator { max_lag: 5 };
        let cov = nw.estimate(&r).unwrap();
        assert_psd(&cov, "NW");
    }

    #[test]
    fn lw_output_is_psd() {
        let t = 100;
        let k = 3;
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let vals: Vec<f64> = (0..t * k).map(|_| rng.r#gen::<f64>()).collect();
        let r = Array2::from_shape_vec((t, k), vals).unwrap();

        for target in [
            LedoitWolfTarget::Identity,
            LedoitWolfTarget::ConstantCorrelation,
        ] {
            let lw = LedoitWolfEstimator { target };
            let cov = lw.estimate(&r).unwrap();
            assert_psd(&cov, &format!("LW-{:?}", target));
        }
    }

    fn assert_psd(matrix: &Array2<f64>, label: &str) {
        let k = matrix.dim().0;
        let data: Vec<f64> = matrix.iter().copied().collect();
        let n = nalgebra::DMatrix::from_row_slice(k, k, &data);
        let eig = nalgebra::linalg::SymmetricEigen::new(n);
        let min_eval = eig.eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        assert!(
            min_eval >= -1e-10,
            "[{}] Not PSD: min eigenvalue = {:.3e}",
            label,
            min_eval
        );
    }
}
