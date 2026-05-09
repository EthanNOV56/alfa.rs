//! Simple risk metrics — volatility, VaR, CVaR without factor decomposition.
//!
//! Provides [`SimpleRiskModel`] that operates on portfolio returns directly,
//! ignoring style-factor exposures.

use crate::risk::{RiskError, RiskModel, RiskReport};
use ndarray::{Array1, Array2};
use statrs::distribution::Continuous;
use statrs::distribution::ContinuousCDF;
use std::collections::HashMap;

/// VaR calculation method.
pub enum SimpleVaRMethod {
    /// Non-parametric — sort returns, take quantile.
    Historical,
    /// Assumes normal distribution.
    Parametric,
    /// Assumes Student's t distribution (better for fat tails).
    ParametricT,
}

/// Simple risk model — no factor decomposition.
pub struct SimpleRiskModel;

impl RiskModel for SimpleRiskModel {
    fn analyze(
        &self,
        weights: &Array2<f64>,
        returns: &Array2<f64>,
        _style_exposures: &[Array2<f64>],
    ) -> Result<RiskReport, RiskError> {
        if weights.dim() != returns.dim() {
            return Err(RiskError::InvalidInput(format!(
                "Shape mismatch: weights {:?} vs returns {:?}",
                weights.dim(),
                returns.dim()
            )));
        }

        let portfolio_returns = compute_portfolio_returns(weights, returns);
        let total_risk = annualized_volatility(&portfolio_returns);

        Ok(RiskReport {
            total_risk,
            systematic_risk: f64::NAN,
            specific_risk: f64::NAN,
            factor_exposures: HashMap::new(),
            factor_contributions: HashMap::new(),
            r_squared: f64::NAN,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute portfolio daily returns: `portfolio_return[d] = Σ_a weights[d,a] × returns[d,a]`.
pub(crate) fn compute_portfolio_returns(
    weights: &Array2<f64>,
    returns: &Array2<f64>,
) -> Array1<f64> {
    let (n_days, n_assets) = weights.dim();
    let mut port_returns = Array1::zeros(n_days);
    for d in 0..n_days {
        let mut sum = 0.0;
        for a in 0..n_assets {
            sum += weights[[d, a]] * returns[[d, a]];
        }
        port_returns[d] = sum;
    }
    port_returns
}

/// Annualized volatility from daily returns: `std(daily_returns) × √252`.
pub(crate) fn annualized_volatility(daily_returns: &Array1<f64>) -> f64 {
    let valid: Vec<f64> = daily_returns
        .iter()
        .filter(|&&r| r.is_finite())
        .copied()
        .collect();
    if valid.len() < 2 {
        return f64::NAN;
    }
    let n = valid.len() as f64;
    let mean = valid.iter().sum::<f64>() / n;
    let var = valid.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt() * (252.0_f64).sqrt()
}

/// Historical VaR: sort returns ascending, take `⌊n×(1-confidence)⌋`-th value.
pub(crate) fn historical_var(daily_returns: &Array1<f64>, confidence: f64) -> f64 {
    let mut valid: Vec<f64> = daily_returns
        .iter()
        .filter(|&&r| r.is_finite())
        .copied()
        .collect();
    if valid.len() < 2 {
        return f64::NAN;
    }
    valid.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((1.0 - confidence) * valid.len() as f64).floor() as usize;
    -valid[idx.min(valid.len() - 1)]
}

/// Parametric VaR (normal): `VaR = -(μ + σ × Φ⁻¹(1-α))`.
pub(crate) fn parametric_var(daily_returns: &Array1<f64>, confidence: f64) -> f64 {
    let valid: Vec<f64> = daily_returns
        .iter()
        .filter(|&&r| r.is_finite())
        .copied()
        .collect();
    if valid.len() < 2 {
        return f64::NAN;
    }
    let n = valid.len() as f64;
    let mu = valid.iter().sum::<f64>() / n;
    let var = valid.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0);
    let sigma = var.sqrt();
    if sigma == 0.0 {
        return 0.0;
    }
    let z = statrs::distribution::Normal::new(0.0, 1.0)
        .unwrap()
        .inverse_cdf(1.0 - confidence);
    -(mu + sigma * z)
}

/// Parametric VaR (Student's t): better for fat-tailed returns.
pub(crate) fn parametric_t_var(daily_returns: &Array1<f64>, confidence: f64) -> f64 {
    let valid: Vec<f64> = daily_returns
        .iter()
        .filter(|&&r| r.is_finite())
        .copied()
        .collect();
    if valid.len() < 3 {
        return f64::NAN;
    }
    let n = valid.len() as f64;
    let mu = valid.iter().sum::<f64>() / n;
    let var = valid.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0);
    let sigma = var.sqrt();
    if sigma == 0.0 {
        return 0.0;
    }
    let df = n - 1.0;
    let t_inv = statrs::distribution::StudentsT::new(0.0, 1.0, df)
        .unwrap()
        .inverse_cdf(1.0 - confidence);
    -(mu + sigma * t_inv)
}

/// Monte Carlo VaR: fit a normal distribution, simulate n_simulations returns,
/// then apply historical VaR to the simulated sample.
///
/// More flexible than parametric (can use any fitted distribution) and smoother
/// than pure historical VaR.
pub(crate) fn monte_carlo_var(
    daily_returns: &Array1<f64>,
    confidence: f64,
    n_simulations: usize,
) -> f64 {
    let valid: Vec<f64> = daily_returns
        .iter()
        .filter(|&&r| r.is_finite())
        .copied()
        .collect();
    if valid.len() < 2 {
        return f64::NAN;
    }
    let n = valid.len() as f64;
    let mu = valid.iter().sum::<f64>() / n;
    let var = valid.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0);
    let sigma = var.sqrt();

    let mut rng = rand::thread_rng();
    use rand::Rng;
    let simulated: Vec<f64> = (0..n_simulations)
        .map(|_| {
            // Box-Muller transform for Normal(0,1) → scale to N(mu, sigma)
            let u1: f64 = rng.r#gen::<f64>().max(1e-15);
            let u2: f64 = rng.r#gen::<f64>();
            let z = (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            mu + sigma * z
        })
        .collect();

    let sim_array = Array1::from_vec(simulated);
    historical_var(&sim_array, confidence)
}

/// Parametric CVaR (normal distribution closed form).
///
/// CVaR_α = μ + σ × φ(Φ⁻¹(α)) / (1-α)
///
/// where φ is the standard normal PDF. This is the expected loss given that
/// the loss exceeds VaR under the normality assumption.
pub(crate) fn parametric_cvar(daily_returns: &Array1<f64>, confidence: f64) -> f64 {
    let valid: Vec<f64> = daily_returns
        .iter()
        .filter(|&&r| r.is_finite())
        .copied()
        .collect();
    if valid.len() < 2 {
        return f64::NAN;
    }
    let n = valid.len() as f64;
    let mu = valid.iter().sum::<f64>() / n;
    let var = valid.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0);
    let sigma = var.sqrt();
    if sigma == 0.0 {
        return 0.0;
    }
    let alpha = 1.0 - confidence;
    let dist = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
    let z = dist.inverse_cdf(alpha);
    let phi_z = dist.pdf(z);
    -mu + sigma * phi_z / alpha
}

/// Historical CVaR / Expected Shortfall: mean of losses beyond VaR.
pub(crate) fn historical_cvar(daily_returns: &Array1<f64>, confidence: f64) -> f64 {
    let var = historical_var(daily_returns, confidence);
    if var.is_nan() {
        return f64::NAN;
    }
    let beyond: Vec<f64> = daily_returns
        .iter()
        .filter(|&&r| r.is_finite() && -r > var)
        .copied()
        .collect();
    if beyond.is_empty() {
        return var;
    }
    let sum: f64 = beyond.iter().map(|&r| -r).sum();
    sum / beyond.len() as f64
}

/// Drawdown analysis result.
pub struct DrawdownAnalysis {
    /// Maximum drawdown as a positive fraction (e.g. 0.25 = 25%).
    pub max_drawdown: f64,
    /// Index (day) when the peak before the max drawdown occurred.
    pub peak_idx: usize,
    /// Index (day) when the trough of the max drawdown occurred.
    pub trough_idx: usize,
    /// Index (day) when NAV recovered to the previous peak (None if never recovered).
    pub recovery_idx: Option<usize>,
    /// Number of days from peak to recovery (None if never recovered).
    pub recovery_days: Option<usize>,
}

/// Compute detailed drawdown analysis from a cumulative NAV curve.
///
/// # Parameters
/// - `cum_nav`: cumulative NAV curve (starts at 1.0, can be from [`cumulative_nav_curve`]).
pub fn max_drawdown_analysis(cum_nav: &Array1<f64>) -> Option<DrawdownAnalysis> {
    let n = cum_nav.len();
    if n < 2 {
        return None;
    }

    let mut peak = cum_nav[0];
    let mut peak_idx = 0usize;
    let mut max_dd = 0.0_f64;
    let mut dd_peak_idx = 0usize;
    let mut dd_trough_idx = 0usize;

    for i in 0..n {
        if cum_nav[i] > peak {
            peak = cum_nav[i];
            peak_idx = i;
        }
        let dd = (peak - cum_nav[i]) / peak;
        if dd > max_dd {
            max_dd = dd;
            dd_peak_idx = peak_idx;
            dd_trough_idx = i;
        }
    }

    if max_dd < 1e-15 {
        return Some(DrawdownAnalysis {
            max_drawdown: 0.0,
            peak_idx: 0,
            trough_idx: 0,
            recovery_idx: Some(0),
            recovery_days: Some(0),
        });
    }

    // Find recovery: first day after trough where NAV ≥ peak
    let peak_val = cum_nav[dd_peak_idx];
    let mut recovery_idx = None;
    let mut recovery_days = None;
    for i in dd_trough_idx..n {
        if cum_nav[i] >= peak_val {
            recovery_idx = Some(i);
            recovery_days = Some(i - dd_peak_idx);
            break;
        }
    }

    Some(DrawdownAnalysis {
        max_drawdown: max_dd,
        peak_idx: dd_peak_idx,
        trough_idx: dd_trough_idx,
        recovery_idx,
        recovery_days,
    })
}

/// Component volatility: per-asset contribution to portfolio volatility.
///
/// `comp_vol[i] = w[i] × (Σ[i,:] · w) / σ_p`, and `Σ_i comp_vol[i] = σ_p`.
pub(crate) fn component_volatility(weights: &Array1<f64>, returns: &Array2<f64>) -> Array1<f64> {
    let (n_days, n_assets) = returns.dim();
    // Sample covariance matrix
    let cov = compute_covariance_matrix(returns);
    // Portfolio variance
    let mut port_var = 0.0;
    for i in 0..n_assets {
        for j in 0..n_assets {
            port_var += weights[i] * weights[j] * cov[[i, j]];
        }
    }
    let port_vol = port_var.sqrt();
    if port_vol < 1e-15 {
        return Array1::zeros(n_assets);
    }
    let mut comp = Array1::zeros(n_assets);
    for i in 0..n_assets {
        let mut marginal = 0.0;
        for j in 0..n_assets {
            marginal += cov[[i, j]] * weights[j];
        }
        comp[i] = weights[i] * marginal / port_vol;
    }
    comp
}

fn compute_covariance_matrix(returns: &Array2<f64>) -> Array2<f64> {
    let (n_days, n_assets) = returns.dim();
    // De-mean
    let mut demeaned = returns.clone();
    for a in 0..n_assets {
        let valid: Vec<f64> = returns.column(a).iter().filter(|&&v| v.is_finite()).copied().collect();
        if valid.len() < 2 {
            continue;
        }
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;
        for d in 0..n_days {
            if demeaned[[d, a]].is_finite() {
                demeaned[[d, a]] -= mean;
            }
        }
    }
    let mut cov = Array2::zeros((n_assets, n_assets));
    for i in 0..n_assets {
        for j in 0..n_assets {
            let mut sum = 0.0;
            let mut count = 0;
            for d in 0..n_days {
                let vi = demeaned[[d, i]];
                let vj = demeaned[[d, j]];
                if vi.is_finite() && vj.is_finite() {
                    sum += vi * vj;
                    count += 1;
                }
            }
            if count > 1 {
                cov[[i, j]] = sum / (count - 1) as f64;
            }
        }
    }
    cov
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// All tests use **synthetic data** — no external data sources.

    // ---------- compute_portfolio_returns ----------

    #[test]
    fn portfolio_returns_equal_weight() {
        // Synthetic: 4 assets, 5 days, equal weights → portfolio return = mean return per day
        let (n_days, n_assets) = (5, 4);
        let weights = Array2::from_elem((n_days, n_assets), 1.0 / n_assets as f64);
        let returns = Array2::from_shape_vec((n_days, n_assets), vec![
            0.01, 0.02, 0.03, 0.04,
            0.01, 0.01, 0.01, 0.01,
            -0.01, 0.00, 0.01, 0.02,
            0.02, -0.01, 0.03, 0.00,
            0.00, 0.00, 0.00, 0.00,
        ]).unwrap();

        let pr = compute_portfolio_returns(&weights, &returns);
        assert_eq!(pr.len(), n_days);
        // Day 0: (0.01+0.02+0.03+0.04)/4 = 0.025
        assert!((pr[0] - 0.025).abs() < 1e-12);
        // Day 1: (0.01*4)/4 = 0.01
        assert!((pr[1] - 0.01).abs() < 1e-12);
        // Day 2: (-0.01+0.00+0.01+0.02)/4 = 0.005
        assert!((pr[2] - 0.005).abs() < 1e-12);
        // Day 3: (0.02-0.01+0.03+0.00)/4 = 0.01
        assert!((pr[3] - 0.01).abs() < 1e-12);
        // Day 4: 0.0
        assert!((pr[4] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn portfolio_returns_concentrated_weight() {
        // Synthetic: 2 assets, 3 days. 100% in asset 0 → should mirror asset 0 returns.
        let (n_days, n_assets) = (3, 2);
        let mut weights = Array2::zeros((n_days, n_assets));
        weights[[0, 0]] = 1.0;
        weights[[1, 0]] = 1.0;
        weights[[2, 0]] = 1.0;
        let returns = Array2::from_shape_vec((n_days, n_assets), vec![
            0.05, 0.99,
            -0.02, -0.99,
            0.03, 0.50,
        ]).unwrap();

        let pr = compute_portfolio_returns(&weights, &returns);
        assert!((pr[0] - 0.05).abs() < 1e-12);
        assert!((pr[1] + 0.02).abs() < 1e-12);
        assert!((pr[2] - 0.03).abs() < 1e-12);
    }

    // ---------- annualized_volatility ----------

    #[test]
    fn annualized_vol_known_value() {
        // Synthetic: returns with known sample std = 0.01.
        // Use scaled values so the sample std falls exactly at 0.01.
        // 4 values with mean=0: scale [1,-1,1,-1] to std=0.01.
        // var = (4 × s²) / 3 → s = 0.01 × √(3/4). Then annualized = 0.01 × √252.
        let s = 0.01 * (3.0_f64 / 4.0_f64).sqrt();
        let daily = Array1::from_vec(vec![s, -s, s, -s]);
        let vol = annualized_volatility(&daily);
        let expected = 0.01 * (252.0_f64).sqrt();
        assert!((vol - expected).abs() < 1e-4, "vol={:.6} expected={:.6}", vol, expected);
    }

    #[test]
    fn annualized_vol_insufficient_data() {
        let daily = Array1::from_vec(vec![0.01]);
        assert!(annualized_volatility(&daily).is_nan());
    }

    // ---------- VaR ----------

    #[test]
    fn historical_var_95() {
        // Synthetic: 100 returns from N(0, 0.01). 95% VaR ≈ 1.645 × 0.01 ≈ 0.01645
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let vals: Vec<f64> = (0..100).map(|_| rng.r#gen::<f64>() * 0.02 - 0.01).collect();
        let daily = Array1::from_vec(vals);
        let var = historical_var(&daily, 0.95);
        // Should be in reasonable range for N(0, 0.01)
        assert!(var > 0.005 && var < 0.04, "Historical VaR={:.6} out of expected range", var);
    }

    #[test]
    fn historical_var_sorted_data() {
        // Synthetic: sorted returns [-10, -5, -2, -1, 0, 1, 2, 5, 10], n=9
        // 95% VaR → idx = floor(9 × 0.05) = 0 → -(-10) = 10? Wait:
        // Sorted ascending: [-10, -5, -2, -1, 0, 1, 2, 5, 10]
        // idx = floor(9 × (1-0.95)) = floor(0.45) = 0
        // VaR = -sorted[0] = -(-10) = 10
        let daily = Array1::from_vec(vec![-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]);
        let var = historical_var(&daily, 0.95);
        assert!((var - 10.0).abs() < 1e-10, "95% Historical VaR should be 10, got {}", var);
    }

    #[test]
    fn parametric_var_matches_theory() {
        // Synthetic: fixed returns with known μ and σ
        // mean=0, std=0.01 → Φ⁻¹(0.05) = -1.64485... → VaR = 0.0164485
        let daily = Array1::from_vec(vec![0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01]);
        let var = parametric_var(&daily, 0.95);
        // σ ≈ 0.0110 (sample std), so VaR ≈ 0.0110 × 1.6449 ≈ 0.0181
        assert!(var > 0.01 && var < 0.03, "Parametric VaR={:.6} out of expected range", var);
    }

    #[test]
    fn parametric_t_var_wider_than_normal() {
        // Synthetic: 5 returns — small sample so t-distribution has wider tails.
        // t-VaR should be > normal-VaR for same confidence.
        let daily = Array1::from_vec(vec![0.01, -0.02, 0.015, -0.01, 0.02]);
        let var_normal = parametric_var(&daily, 0.95);
        let var_t = parametric_t_var(&daily, 0.95);
        assert!(var_t > var_normal * 1.1,
            "t-VaR ({:.6}) should be > normal-VaR ({:.6}) due to thicker tails",
            var_t, var_normal);
    }

    // ---------- Monte Carlo VaR ----------

    #[test]
    fn mc_var_close_to_theoretical() {
        // Synthetic: 500 returns from N(0, 0.01). 10000 simulations.
        // MC VaR(95%) ≈ 1.645 × 0.01 ≈ 0.01645
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let vals: Vec<f64> = (0..500).map(|_| rng.r#gen::<f64>() * 0.02 - 0.01).collect();
        let daily = Array1::from_vec(vals);
        let mc = monte_carlo_var(&daily, 0.95, 10000);
        assert!(mc > 0.008 && mc < 0.03, "MC VaR={:.6} should be near 0.0165", mc);
    }

    // ---------- CVaR ----------

    #[test]
    fn parametric_cvar_larger_than_var() {
        // Synthetic: known-mean returns. Parametric CVaR must be ≥ Parametric VaR.
        let vals: Vec<f64> = (0..100)
            .map(|i| (i as f64 - 50.0) * 0.001)
            .collect();
        let daily = Array1::from_vec(vals);
        let var = parametric_var(&daily, 0.95);
        let cvar = parametric_cvar(&daily, 0.95);
        assert!(cvar >= var - 1e-12,
            "Parametric CVaR ({:.6}) should be ≥ VaR ({:.6})", cvar, var);
    }

    #[test]
    fn cvar_gte_var() {
        // Synthetic: random returns. CVaR should always be ≥ VaR.
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let vals: Vec<f64> = (0..200).map(|_| rng.r#gen::<f64>() * 0.04 - 0.02).collect();
        let daily = Array1::from_vec(vals);
        let var = historical_var(&daily, 0.95);
        let cvar = historical_cvar(&daily, 0.95);
        assert!(cvar >= var - 1e-12,
            "CVaR ({:.6}) should be ≥ VaR ({:.6})", cvar, var);
    }

    // ---------- Drawdown analysis ----------

    #[test]
    fn drawdown_analysis_known_shape() {
        // Synthetic: NAV = [1.0, 1.1, 0.9, 0.95, 1.15] — peak at idx 4, trough at idx 2
        let nav = Array1::from_vec(vec![1.0, 1.1, 0.9, 0.95, 1.15]);
        let dd = max_drawdown_analysis(&nav).unwrap();
        // Peak before max DD: idx 1 (NAV=1.1), trough: idx 2 (NAV=0.9)
        // DD = (1.1-0.9)/1.1 ≈ 0.1818
        assert!((dd.max_drawdown - 0.181818).abs() < 0.001);
        assert_eq!(dd.peak_idx, 1);
        assert_eq!(dd.trough_idx, 2);
        // Recovery at idx 4: NAV 1.15 > 1.1
        assert_eq!(dd.recovery_idx, Some(4));
        assert_eq!(dd.recovery_days, Some(3));
    }

    #[test]
    fn drawdown_no_drawdown() {
        // Synthetic: NAV always rising
        let nav = Array1::from_vec(vec![1.0, 1.05, 1.10, 1.15]);
        let dd = max_drawdown_analysis(&nav).unwrap();
        assert!((dd.max_drawdown - 0.0).abs() < 1e-10);
    }

    #[test]
    fn drawdown_never_recovers() {
        // Synthetic: NAV drops and never recovers to peak
        let nav = Array1::from_vec(vec![1.0, 1.5, 1.2, 1.1, 1.05]);
        let dd = max_drawdown_analysis(&nav).unwrap();
        assert_eq!(dd.peak_idx, 1);
        assert_eq!(dd.trough_idx, 4);
        assert!(dd.recovery_idx.is_none());
    }

    // ---------- SimpleRiskModel ----------

    #[test]
    fn simple_risk_model_returns_volatility() {
        // Synthetic: equal-weight portfolio, verify total_risk > 0
        let (n_days, n_assets) = (50, 4);
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let vals: Vec<f64> = (0..n_days * n_assets).map(|_| rng.r#gen::<f64>() * 0.02).collect();
        let returns = Array2::from_shape_vec((n_days, n_assets), vals).unwrap();
        let weights = Array2::from_elem((n_days, n_assets), 1.0 / n_assets as f64);

        let model = SimpleRiskModel;
        let report = model.analyze(&weights, &returns, &[]).unwrap();
        assert!(report.total_risk > 0.0);
        assert!(report.total_risk.is_finite());
        assert!(report.systematic_risk.is_nan());
        assert!(report.specific_risk.is_nan());
        assert!(report.r_squared.is_nan());
        assert!(report.factor_exposures.is_empty());
        assert!(report.factor_contributions.is_empty());
    }

    #[test]
    fn simple_risk_model_shape_mismatch() {
        let w = Array2::from_elem((10, 3), 1.0);
        let r = Array2::from_elem((10, 4), 0.01);
        assert!(SimpleRiskModel.analyze(&w, &r, &[]).is_err());
    }

    // ---------- component_volatility ----------

    #[test]
    fn component_vol_sum_equals_total() {
        // Synthetic: 3 assets, 200 days, random returns, long-only weights.
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let n_days = 200;
        let n_assets = 3;
        let vals: Vec<f64> = (0..n_days * n_assets).map(|_| rng.r#gen::<f64>() * 0.04 - 0.02).collect();
        let returns = Array2::from_shape_vec((n_days, n_assets), vals).unwrap();
        let weights = Array1::from_vec(vec![0.5, 0.3, 0.2]);

        let comp = component_volatility(&weights, &returns);
        let sum: f64 = comp.iter().sum();

        // Recompute total vol directly
        let cov = compute_covariance_matrix(&returns);
        let mut port_var = 0.0;
        for i in 0..n_assets {
            for j in 0..n_assets {
                port_var += weights[i] * weights[j] * cov[[i, j]];
            }
        }
        let port_vol = port_var.sqrt();

        assert!(
            (sum - port_vol).abs() < 1e-12,
            "Component volatilities should sum to portfolio vol: sum={:.6}, vol={:.6}",
            sum,
            port_vol
        );
    }
}
