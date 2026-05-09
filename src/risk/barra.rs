//! Barra-style multi-factor risk attribution.
//!
//! Implements the three-step Barra pipeline:
//! 1. Cross-sectional regression (OLS/WLS) — estimate factor returns
//! 2. Factor covariance estimation (via [`CovEstimator`])
//! 3. Portfolio risk decomposition
//!
//! All factor definitions are injectable via `style_exposures` — the model
//! does not hard-code any specific factor set.

use crate::risk::cov_estimation::CovEstimator;
use crate::risk::{RiskError, RiskModel, RiskReport};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::collections::HashMap;

/// Regression method for cross-sectional estimation.
pub enum BarraRegressionMethod {
    OLS,
    /// Weighted least squares with `sqrt(weights)` as the per-observation scale.
    /// Weights are provided per-day via `BarraRiskModel::analyze` as an
    /// `(n_days × n_assets)` matrix (e.g. `sqrt(market_cap)`).
    WLS,
    /// Huber robust regression via IRLS (Iteratively Reweighted Least Squares).
    /// Down-weights outliers automatically. Tuning constant c = 1.345.
    Huber,
}

/// Barra-style risk model — multi-factor attribution engine.
pub struct BarraRiskModel {
    /// Factor covariance estimator (Newey-West or Ledoit-Wolf).
    pub cov_estimator: Box<dyn CovEstimator>,
    /// Regression method (OLS or WLS).
    pub regression_method: BarraRegressionMethod,
    /// Names for the K style factors (used for labelling contributions).
    pub factor_names: Vec<String>,
    /// Whether to orthogonalize factor exposures before regression
    /// (Gram-Schmidt by business-logic order). Default: off.
    pub orthogonalize: bool,
}

impl BarraRiskModel {
    pub fn new(cov_estimator: Box<dyn CovEstimator>, factor_names: Vec<String>) -> Self {
        Self {
            cov_estimator,
            regression_method: BarraRegressionMethod::OLS,
            factor_names,
            orthogonalize: false,
        }
    }
}

impl RiskModel for BarraRiskModel {
    fn analyze(
        &self,
        weights: &Array2<f64>,
        returns: &Array2<f64>,
        style_exposures: &[Array2<f64>],
    ) -> Result<RiskReport, RiskError> {
        let (n_days, n_assets) = returns.dim();
        let k_factors = style_exposures.len();

        if k_factors == 0 {
            return Err(RiskError::InvalidInput(
                "BarraRiskModel requires at least one style exposure matrix".into(),
            ));
        }

        // Validate dimensions
        if weights.dim() != (n_days, n_assets) {
            return Err(RiskError::InvalidInput(format!(
                "Shape mismatch: weights {:?} vs returns {:?}",
                weights.dim(),
                returns.dim()
            )));
        }
        for (i, exp) in style_exposures.iter().enumerate() {
            if exp.dim() != (n_days, n_assets) {
                return Err(RiskError::InvalidInput(format!(
                    "Shape mismatch: exposure[{}] {:?} vs returns {:?}",
                    i,
                    exp.dim(),
                    returns.dim()
                )));
            }
        }

        // Build regression weights for WLS if needed
        let reg_weights: Option<Array2<f64>> = match &self.regression_method {
            BarraRegressionMethod::OLS | BarraRegressionMethod::Huber => None,
            BarraRegressionMethod::WLS => {
                None // external weights must be injected — see analyze_with_weights
            }
        };

        // ---- Step 1: Cross-sectional regression per day ----
        let (factor_returns, specific_returns) =
            estimate_factor_returns(returns, style_exposures, &self.regression_method, reg_weights.as_ref(), self.orthogonalize)?;

        // ---- Step 2: Factor covariance estimation ----
        let factor_cov = self.cov_estimator.estimate(&factor_returns)?;

        // ---- Step 3: Risk decomposition ----
        let avg_weights = average_weights(weights);
        let last_day = n_days - 1;
        let last_exposures = build_exposure_matrix(style_exposures, last_day);
        let specific_vars = estimate_specific_variances(&specific_returns);

        decompose_risk(
            &avg_weights,
            &last_exposures,
            &factor_cov,
            &specific_vars,
            n_assets,
            k_factors,
            &self.factor_names,
        )
    }
}

// --------------------------------------------------------------------------
// Step 1: Cross-sectional regression
// --------------------------------------------------------------------------

/// Estimate factor returns and specific returns from daily cross-sectional regressions.
///
/// Uses rayon to parallelize over days.
fn estimate_factor_returns(
    returns: &Array2<f64>,
    exposures: &[Array2<f64>],
    method: &BarraRegressionMethod,
    reg_weights: Option<&Array2<f64>>,
    orthogonalize: bool,
) -> Result<(Array2<f64>, Array2<f64>), RiskError> {
    let (n_days, n_assets) = returns.dim();
    let k_factors = exposures.len();

    // Build per-day results in parallel
    let day_results: Vec<Result<(Vec<f64>, Vec<f64>), RiskError>> = (0..n_days)
        .into_par_iter()
        .map(|d| {
            // Build exposure matrix for day d: (n_assets × K)
            let mut x_data = Vec::with_capacity(n_assets * k_factors);
            for a in 0..n_assets {
                for k in 0..k_factors {
                    x_data.push(exposures[k][[d, a]]);
                }
            }
            let x_orig = ndarray::ArrayView2::from_shape((n_assets, k_factors), &x_data)
                .map_err(|e| RiskError::ComputationFailed(e.to_string()))?;

            // Optional factor orthogonalization
            let x_owned = if orthogonalize {
                orthogonalize_exposures(&x_orig.to_owned())
            } else {
                x_orig.to_owned()
            };
            let x = x_owned.view();

            let y_data: Vec<f64> = (0..n_assets).map(|a| returns[[d, a]]).collect();
            let y = Array1::from_vec(y_data);

            match method {
                BarraRegressionMethod::OLS => {
                    ols_regression(&x, &y)
                }
                BarraRegressionMethod::WLS => {
                    let sqrt_w: Array1<f64> = match reg_weights {
                        Some(w) => w.row(d).mapv(|v| v.sqrt()),
                        None => Array1::ones(n_assets),
                    };
                    wls_regression(&x, &y, &sqrt_w)
                }
                BarraRegressionMethod::Huber => {
                    huber_regression(&x, &y)
                }
            }
            .map(|(f, eps)| (f.to_vec(), eps.to_vec()))
        })
        .collect();

    // Collect into output matrices
    let mut factor_returns = Array2::zeros((n_days, k_factors));
    let mut specific_returns = Array2::zeros((n_days, n_assets));
    for (d, result) in day_results.into_iter().enumerate() {
        let (f, eps) = result?;
        for k in 0..k_factors {
            factor_returns[[d, k]] = f[k];
        }
        for a in 0..n_assets {
            specific_returns[[d, a]] = eps[a];
        }
    }

    Ok((factor_returns, specific_returns))
}

/// Ordinary least squares: r = X·f + ε → solve (XᵀX)f = Xᵀy via Cholesky.
///
/// Falls back to SVD if Cholesky fails (near-singular XᵀX).
fn ols_regression(
    x: &ndarray::ArrayView2<f64>,
    y: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>), RiskError> {
    let (n, k) = x.dim();

    // Filter valid rows (no NaN in X or y)
    let mut valid_x = Vec::new();
    let mut valid_y = Vec::with_capacity(n);
    for i in 0..n {
        let row_valid = (0..k).all(|j| x[[i, j]].is_finite());
        if row_valid && y[i].is_finite() {
            valid_x.push(i);
            valid_y.push(y[i]);
        }
    }

    let n_valid = valid_x.len();
    if n_valid < k + 2 {
        return Err(RiskError::ComputationFailed(format!(
            "Insufficient valid observations: {} need ≥ {}",
            n_valid,
            k + 2
        )));
    }

    // Build XtX (K × K) and Xty (K,)
    let mut xtx = Array2::zeros((k, k));
    let mut xty = Array1::zeros(k);
    for (&idx, y_val) in valid_x.iter().zip(valid_y.iter()) {
        for p in 0..k {
            let xip = x[[idx, p]];
            xty[p] += xip * y_val;
            for q in 0..k {
                xtx[[p, q]] += xip * x[[idx, q]];
            }
        }
    }

    // Ridge regularization
    let ridge = 1e-10;
    for p in 0..k {
        xtx[[p, p]] += ridge;
    }

    // Convert to nalgebra for Cholesky
    let xtx_data: Vec<f64> = xtx.iter().copied().collect();
    let xtx_n = nalgebra::DMatrix::from_row_slice(k, k, &xtx_data);
    let xty_data: Vec<f64> = xty.iter().copied().collect();
    let xty_n = nalgebra::DVector::from_row_slice(&xty_data);

    let f_n = match nalgebra::linalg::Cholesky::new(xtx_n.clone()) {
        Some(chol) => chol.solve(&xty_n),
        None => {
            let svd = nalgebra::linalg::SVD::new(xtx_n, true, true);
            svd.solve(&xty_n, 1e-10)
                .map_err(|e| RiskError::ComputationFailed(format!("SVD solve failed: {}", e)))?
        }
    };

    // Compute residuals: ε_i = y_i - Σ_k x_ik · f_k
    let mut f = Array1::zeros(k);
    for p in 0..k {
        f[p] = f_n[p];
    }

    let mut specific = Array1::zeros(n);
    for i in 0..n {
        let mut predicted = 0.0;
        for p in 0..k {
            predicted += x[[i, p]] * f[p];
        }
        specific[i] = y[i] - predicted;
    }

    Ok((f, specific))
}

/// Weighted least squares: pre-multiply X and y by sqrt(weights), then call OLS.
fn wls_regression(
    x: &ndarray::ArrayView2<f64>,
    y: &Array1<f64>,
    sqrt_weights: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>), RiskError> {
    let (n, k) = x.dim();
    let mut x_w = Array2::zeros((n, k));
    let mut y_w = Array1::zeros(n);
    for i in 0..n {
        let sw = sqrt_weights[i];
        if !sw.is_finite() || sw <= 0.0 {
            continue;
        }
        y_w[i] = y[i] * sw;
        for p in 0..k {
            x_w[[i, p]] = x[[i, p]] * sw;
        }
    }
    ols_regression(&x_w.view(), &y_w)
}

/// Huber robust regression via IRLS.
///
/// Algorithm:
/// 1. Start with OLS estimate β₀
/// 2. Compute residuals ε = y − Xβ, robust scale σ̂ = 1.4826 × MAD
/// 3. Huber weights: w_i = 1 if |ε_i/σ̂| ≤ c, else c / |ε_i/σ̂| (c = 1.345)
/// 4. Re-solve weighted normal equations: (XᵀWX)β = XᵀWy
/// 5. Repeat steps 2-4 until convergence (max 20 iterations, tol 1e-6)
fn huber_regression(
    x: &ndarray::ArrayView2<f64>,
    y: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>), RiskError> {
    let (n, k) = x.dim();
    const C: f64 = 1.345;       // Huber tuning constant (95% efficiency)
    const MAX_ITER: usize = 20;
    const TOL: f64 = 1e-6;

    // Start with OLS
    let (mut beta, _) = ols_regression(x, y)?;

    for _ in 0..MAX_ITER {
        // Compute residuals
        let mut residuals = Vec::with_capacity(n);
        for i in 0..n {
            let mut pred = 0.0;
            for p in 0..k {
                pred += x[[i, p]] * beta[p];
            }
            residuals.push(y[i] - pred);
        }

        // Robust scale: σ̂ = 1.4826 × MAD
        let valid_res: Vec<f64> = residuals.iter().filter(|&&r| r.is_finite()).copied().collect();
        if valid_res.len() < 2 {
            break;
        }
        let median = {
            let mut sorted = valid_res.clone();
            sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        };
        let abs_devs: Vec<f64> = valid_res.iter().map(|&r| (r - median).abs()).collect();
        let mut sorted_abs = abs_devs.clone();
        sorted_abs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = {
            let mid = sorted_abs.len() / 2;
            if sorted_abs.len() % 2 == 0 {
                (sorted_abs[mid - 1] + sorted_abs[mid]) / 2.0
            } else {
                sorted_abs[mid]
            }
        };
        let scale = 1.4826 * mad.max(1e-10);

        // Huber weights
        let weights: Vec<f64> = residuals
            .iter()
            .map(|&r| {
                let z = (r / scale).abs();
                if z <= C { 1.0 } else { C / z }
            })
            .collect();

        // Weighted normal equations: (XᵀWX)β_new = XᵀWy
        let mut xtwx = Array2::zeros((k, k));
        let mut xtwy = Array1::zeros(k);
        for i in 0..n {
            let w = weights[i];
            if !w.is_finite() || w <= 0.0 {
                continue;
            }
            for p in 0..k {
                let xip = x[[i, p]];
                xtwy[p] += xip * w * y[i];
                for q in 0..k {
                    xtwx[[p, q]] += xip * w * x[[i, q]];
                }
            }
        }

        // Ridge
        for p in 0..k {
            xtwx[[p, p]] += 1e-10;
        }

        // Solve via Cholesky / SVD
        let xtwx_data: Vec<f64> = xtwx.iter().copied().collect();
        let xtwx_n = nalgebra::DMatrix::from_row_slice(k, k, &xtwx_data);
        let xtwy_data: Vec<f64> = xtwy.iter().copied().collect();
        let xtwy_n = nalgebra::DVector::from_row_slice(&xtwy_data);

        let beta_new_n = match nalgebra::linalg::Cholesky::new(xtwx_n.clone()) {
            Some(chol) => chol.solve(&xtwy_n),
            None => {
                let svd = nalgebra::linalg::SVD::new(xtwx_n, true, true);
                svd.solve(&xtwy_n, 1e-10)
                    .map_err(|e| RiskError::ComputationFailed(format!("SVD failed: {}", e)))?
            }
        };

        let mut beta_new = Array1::zeros(k);
        for p in 0..k {
            beta_new[p] = beta_new_n[p];
        }

        // Check convergence
        let change: f64 = (0..k)
            .map(|p| (beta_new[p] - beta[p]).abs())
            .fold(0.0_f64, |a, b| a.max(b));
        beta = beta_new;
        if change < TOL {
            break;
        }
    }

    // Compute final residuals
    let mut specific = Array1::zeros(n);
    for i in 0..n {
        let mut pred = 0.0;
        for p in 0..k {
            pred += x[[i, p]] * beta[p];
        }
        specific[i] = y[i] - pred;
    }

    Ok((beta, specific))
}

// --------------------------------------------------------------------------
// Factor orthogonalization (Gram-Schmidt)
// --------------------------------------------------------------------------

/// Orthogonalize factor exposures column-by-column using sequential regression.
///
/// Algorithm: for each column k from 1..K, replace X[:,k] with the residual
/// of regressing X[:,k] on X[:,0..k-1]. Column 0 is left unchanged.
///
/// This ensures no two columns share overlapping linear information,
/// giving stable and additive factor contributions.
///
/// **Economic interpretation changes**: after orthogonalization, "Size"
/// means "the part of Size not explained by Beta". Enable only when
/// factor collinearity (e.g. Size ↔ NonLinearSize) causes regression
/// instability.
fn orthogonalize_exposures(x: &Array2<f64>) -> Array2<f64> {
    let (n, k) = x.dim();
    if k <= 1 {
        return x.clone();
    }
    let mut result = x.clone();
    for col in 1..k {
        // Regress column `col` on all previous columns and keep the residual
        let y_col = result.column(col).to_owned();
        let x_prev = result.slice(ndarray::s![.., 0..col]).to_owned();
        let x_view = x_prev.view();

        match ols_regression(&x_view, &y_col) {
            Ok((_f, residuals)) => {
                for i in 0..n {
                    result[[i, col]] = residuals[i];
                }
            }
            Err(_) => {
                // If regression fails (not enough valid data), leave column unchanged
            }
        }
    }
    result
}

// --------------------------------------------------------------------------
// Step 2: Factor covariance (delegated to CovEstimator in cov_estimation.rs)
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Step 3: Risk decomposition
// --------------------------------------------------------------------------

/// Build exposure matrix for a single day: (n_assets × K).
fn build_exposure_matrix(exposures: &[Array2<f64>], day: usize) -> Array2<f64> {
    let k = exposures.len();
    let n_assets = exposures[0].dim().1;
    let mut b = Array2::zeros((n_assets, k));
    for a in 0..n_assets {
        for k_idx in 0..k {
            b[[a, k_idx]] = exposures[k_idx][[day, a]];
        }
    }
    b
}

/// Time-averaged weights: `w̄[a] = mean(weights[:, a])`.
fn average_weights(weights: &Array2<f64>) -> Array1<f64> {
    let (n_days, n_assets) = weights.dim();
    let mut avg = Array1::zeros(n_assets);
    for a in 0..n_assets {
        let mut sum = 0.0;
        let mut count = 0;
        for d in 0..n_days {
            let w = weights[[d, a]];
            if w.is_finite() {
                sum += w;
                count += 1;
            }
        }
        if count > 0 {
            avg[a] = sum / count as f64;
        }
    }
    avg
}

/// Estimate per-asset specific variance from specific return time series.
fn estimate_specific_variances(specific_returns: &Array2<f64>) -> Array1<f64> {
    let (n_days, n_assets) = specific_returns.dim();
    let mut vars = Array1::zeros(n_assets);
    for a in 0..n_assets {
        let mut valid_vals = Vec::with_capacity(n_days);
        for d in 0..n_days {
            let v = specific_returns[[d, a]];
            if v.is_finite() {
                valid_vals.push(v);
            }
        }
        if valid_vals.len() < 2 {
            vars[a] = f64::NAN;
            continue;
        }
        let n = valid_vals.len() as f64;
        let mean = valid_vals.iter().sum::<f64>() / n;
        let var = valid_vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        vars[a] = var;
    }
    vars
}

/// Decompose portfolio risk into systematic, specific, and factor contributions.
fn decompose_risk(
    avg_weights: &Array1<f64>,
    last_exposures: &Array2<f64>,
    factor_cov: &Array2<f64>,
    specific_vars: &Array1<f64>,
    n_assets: usize,
    k_factors: usize,
    factor_names: &[String],
) -> Result<RiskReport, RiskError> {
    // Weighted exposure: e = Bᵀ × w̄  (K,)
    let mut e = Array1::zeros(k_factors);
    for k in 0..k_factors {
        let mut sum = 0.0;
        for a in 0..n_assets {
            sum += avg_weights[a] * last_exposures[[a, k]];
        }
        e[k] = sum;
    }

    // Systematic variance: eᵀ Σ_f e
    let mut sys_var = 0.0;
    for p in 0..k_factors {
        let mut row_sum = 0.0;
        for q in 0..k_factors {
            row_sum += factor_cov[[p, q]] * e[q];
        }
        sys_var += e[p] * row_sum;
    }

    // Specific variance: Σ w_i² σ²_ε,i
    let mut spec_var = 0.0;
    for a in 0..n_assets {
        let sv = specific_vars[a];
        if sv.is_finite() && sv > 0.0 {
            spec_var += avg_weights[a] * avg_weights[a] * sv;
        }
    }

    let total_var = sys_var + spec_var;
    if total_var <= 0.0 {
        return Err(RiskError::ComputationFailed(
            "Total variance ≤ 0".into(),
        ));
    }

    let annualize = |var: f64| var.sqrt() * (252.0_f64).sqrt();

    let total_risk = annualize(total_var);
    let systematic_risk = annualize(sys_var);
    let specific_risk = annualize(spec_var);
    let r_squared = sys_var / total_var;

    // Factor exposures
    let mut factor_exposures = HashMap::new();
    for k in 0..k_factors {
        let name = factor_names
            .get(k)
            .cloned()
            .unwrap_or_else(|| format!("factor_{}", k));
        factor_exposures.insert(name, e[k]);
    }

    // Factor contributions
    let mut factor_contributions = HashMap::new();
    for k in 0..k_factors {
        let contrib = e[k] * e[k] * factor_cov[[k, k]] / total_var;
        let name = factor_names
            .get(k)
            .cloned()
            .unwrap_or_else(|| format!("factor_{}", k));
        factor_contributions.insert(name, contrib);
    }

    Ok(RiskReport {
        total_risk,
        systematic_risk,
        specific_risk,
        factor_exposures,
        factor_contributions,
        r_squared,
    })
}

// --------------------------------------------------------------------------
// Portfolio return decomposition
// --------------------------------------------------------------------------

/// Decompose portfolio return into factor-driven and specific components.
///
/// Given estimated factor returns `f` (n_days × K) and asset-level specific
/// returns `ε` (n_days × n_assets), plus portfolio weights `w` (n_assets,):
///
/// ```text
///   r_port[d]  = Σ_a w[a] × r[d,a]
///   r_factor[d] = Σ_a w[a] × Σ_k B[a,k] × f[d,k]
///   r_specific[d] = Σ_a w[a] × ε[d,a]
/// ```
///
/// Returns `(factor_contribution, specific_contribution)` each as (n_days,).
pub fn decompose_portfolio_return(
    weights: &Array1<f64>,
    factor_returns: &Array2<f64>,
    specific_returns: &Array2<f64>,
    exposures: &[Array2<f64>],
) -> Result<(Array1<f64>, Array1<f64>), RiskError> {
    let (n_days, n_assets) = specific_returns.dim();
    let k = factor_returns.dim().1;

    let mut r_factor = Array1::zeros(n_days);
    let mut r_specific = Array1::zeros(n_days);

    for d in 0..n_days {
        // Build exposure matrix for day d
        let b = build_exposure_matrix(exposures, d);

        // Factor-driven return: wᵀ · B · f_d
        let mut factor_contrib = 0.0;
        for a in 0..n_assets {
            let mut exposure_return = 0.0;
            for k_idx in 0..k {
                exposure_return += b[[a, k_idx]] * factor_returns[[d, k_idx]];
            }
            factor_contrib += weights[a] * exposure_return;
        }
        r_factor[d] = factor_contrib;

        // Specific return: wᵀ · ε_d
        let mut spec_contrib = 0.0;
        for a in 0..n_assets {
            spec_contrib += weights[a] * specific_returns[[d, a]];
        }
        r_specific[d] = spec_contrib;
    }

    Ok((r_factor, r_specific))
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk::cov_estimation::{NeweyWestEstimator, SampleCovEstimator};

    /// All tests use **synthetic data** — no external data sources.

    // ---------- OLS regression ----------

    #[test]
    fn ols_exact_fit() {
        // Synthetic: 4 assets, 2 factors (need ≥ K+2 = 4 rows)
        // y = X · [1.0, 1.0]
        let x_data = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.2, 0.8];
        let x = ndarray::ArrayView2::from_shape((4, 2), &x_data).unwrap();
        let y = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);

        let (f, _eps) = ols_regression(&x, &y).unwrap();
        assert!((f[0] - 1.0).abs() < 1e-10, "f0 should be 1.0, got {}", f[0]);
        assert!((f[1] - 1.0).abs() < 1e-10, "f1 should be 1.0, got {}", f[1]);
    }

    #[test]
    fn ols_with_noise() {
        // Synthetic: y = X·[2.0, -1.0] + ε where ε ~ N(0, 0.01)
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let n = 30;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);
        for _ in 0..n {
            let x1 = rng.r#gen::<f64>() * 2.0 - 1.0;
            let x2 = rng.r#gen::<f64>() * 2.0 - 1.0;
            x_data.push(x1);
            x_data.push(x2);
            let y_true = 2.0 * x1 - 1.0 * x2;
            let noise = rng.r#gen::<f64>() * 0.02 - 0.01;
            y_data.push(y_true + noise);
        }
        let x = ndarray::ArrayView2::from_shape((n, 2), &x_data).unwrap();
        let y = Array1::from_vec(y_data);

        let (f, _eps) = ols_regression(&x, &y).unwrap();
        assert!((f[0] - 2.0).abs() < 0.1, "f0≈2.0, got {}", f[0]);
        assert!((f[1] + 1.0).abs() < 0.1, "f1≈-1.0, got {}", f[1]);
    }

    #[test]
    fn ols_insufficient_data() {
        // Only 2 valid rows for 3 factors → should fail
        let x_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let x = ndarray::ArrayView2::from_shape((2, 3), &x_data).unwrap();
        let y = Array1::from_vec(vec![1.0, 1.0]);
        assert!(ols_regression(&x, &y).is_err());
    }

    // ---------- WLS regression ----------

    #[test]
    fn wls_weights_large_obs() {
        // Synthetic: 4 assets, 1 factor.
        // Asset 0 has weight ≈100 (dominates), so regression should match its return closely.
        let x_data = vec![1.0, 1.0, 1.0, 1.0];
        let x = ndarray::ArrayView2::from_shape((4, 1), &x_data).unwrap();
        // True: y = 2.0 * x + noise. But asset 0 is exact, assets 1-3 are noise.
        let y = Array1::from_vec(vec![2.0, 0.0, 0.0, 0.0]);
        let sqrt_w = Array1::from_vec(vec![10.0, 0.01, 0.01, 0.01]);

        let (f, _eps) = wls_regression(&x, &y, &sqrt_w).unwrap();
        // Asset 0 has 10^6× the weight of others, so f ≈ 2.0
        assert!((f[0] - 2.0).abs() < 0.01, "WLS should match high-weight obs, got {}", f[0]);
    }

    // ---------- Huber robust regression ----------

    #[test]
    fn huber_downweights_outlier() {
        // Synthetic: 10 assets, 1 factor. Asset 0 is an outlier with return 10× others.
        // OLS should be pulled toward the outlier; Huber should stay close to the clean data.
        let n = 10;
        let k = 1;
        let x_data: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
        let x = ndarray::ArrayView2::from_shape((n, k), &x_data).unwrap();
        // Clean model: y ≈ 2.0 × x. Asset 0 has outlier return = 20.0.
        let mut y_vec = Vec::with_capacity(n);
        for i in 0..n {
            let base = 2.0 * x[[i, 0]];
            y_vec.push(if i == 0 { 20.0 } else { base + 0.01 * (i as f64 - 5.0) });
        }
        let y = Array1::from_vec(y_vec);

        let (f_ols, _) = ols_regression(&x, &y).unwrap();
        let (f_huber, _) = huber_regression(&x, &y).unwrap();

        // Huber should be closer to the true β = 2.0
        assert!(
            (f_huber[0] - 2.0).abs() < (f_ols[0] - 2.0).abs(),
            "Huber {} should be closer to 2.0 than OLS {}",
            f_huber[0],
            f_ols[0]
        );
    }

    // ---------- Factor orthogonalization ----------

    #[test]
    fn orthogonalize_reduces_variance() {
        // Synthetic: 10 assets, 2 factors where factor_1 ≈ 0.5 * factor_0 + small noise.
        // After orthogonalization, column 1 variance should drop substantially.
        let n = 10;
        let mut x = Array2::zeros((n, 2));
        for i in 0..n {
            x[[i, 0]] = (i + 1) as f64;
            x[[i, 1]] = 0.5 * x[[i, 0]] + 0.001 * (i as f64 - 5.0);
        }
        let orig_var_c1 = {
            let c = x.column(1);
            let m = c.mean().unwrap();
            c.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / (n - 1) as f64
        };

        let result = orthogonalize_exposures(&x);
        let res_var_c1 = {
            let c = result.column(1);
            let m = c.mean().unwrap();
            c.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / (n - 1) as f64
        };

        // Residual should have much smaller variance than the original column
        assert!(
            res_var_c1 < orig_var_c1 * 0.5,
            "Orthogonalization should reduce variance of col1: orig={:.6} residual={:.6}",
            orig_var_c1,
            res_var_c1
        );
        // Column 0 should be unchanged
        for i in 0..n {
            assert!((result[[i, 0]] - x[[i, 0]]).abs() < 1e-12);
        }
    }

    // ---------- Full Barra pipeline ----------

    #[test]
    fn barra_known_system() {
        // Synthetic: 4 stocks, 2 factors, 60 days
        let n_days = 60;
        let n_assets = 4;
        let k = 2;

        let mut rng = rand::thread_rng();
        use rand::Rng;

        // Generate true factor returns
        let mut factor_returns_true = Array2::zeros((n_days, k));
        for d in 0..n_days {
            factor_returns_true[[d, 0]] = 0.02 + 0.03 * rng.r#gen::<f64>();
            factor_returns_true[[d, 1]] = 0.01 + 0.02 * rng.r#gen::<f64>();
        }

        // Build exposures: A=[1,0], B=[0,1], C=[0.5,0.5], D=[0.3,0.7]
        let mut exp0 = Array2::zeros((n_days, n_assets));
        let mut exp1 = Array2::zeros((n_days, n_assets));
        for d in 0..n_days {
            exp0[[d, 0]] = 1.0;
            exp0[[d, 2]] = 0.5;
            exp0[[d, 3]] = 0.3;
            exp1[[d, 1]] = 1.0;
            exp1[[d, 2]] = 0.5;
            exp1[[d, 3]] = 0.7;
        }

        let mut returns = Array2::zeros((n_days, n_assets));
        for d in 0..n_days {
            for a in 0..n_assets {
                let predicted = exp0[[d, a]] * factor_returns_true[[d, 0]]
                    + exp1[[d, a]] * factor_returns_true[[d, 1]];
                let noise = 0.01 * rng.r#gen::<f64>();
                returns[[d, a]] = predicted + noise;
            }
        }

        let mut weights = Array2::zeros((n_days, n_assets));
        for d in 0..n_days {
            weights[[d, 0]] = 0.3;
            weights[[d, 1]] = 0.3;
            weights[[d, 2]] = 0.2;
            weights[[d, 3]] = 0.2;
        }

        let model = BarraRiskModel::new(
            Box::new(SampleCovEstimator),
            vec!["size".to_string(), "momentum".to_string()],
        );

        let report = model
            .analyze(&weights, &returns, &[exp0, exp1])
            .unwrap();

        assert!(report.total_risk > 0.0);
        assert!(report.total_risk.is_finite());
        assert!(report.systematic_risk.is_finite());
        assert!(report.specific_risk.is_finite());
        assert!(report.r_squared > 0.0 && report.r_squared <= 1.0);

        // e[0] = 0.3*1.0 + 0.3*0.0 + 0.2*0.5 + 0.2*0.3 = 0.46
        assert!(
            (report.factor_exposures["size"] - 0.46).abs() < 0.02,
            "Expected size exposure ≈ 0.46, got {:?}",
            report.factor_exposures.get("size")
        );
        // e[1] = 0.3*0.0 + 0.3*1.0 + 0.2*0.5 + 0.2*0.7 = 0.54
        assert!(
            (report.factor_exposures["momentum"] - 0.54).abs() < 0.02,
            "Expected momentum exposure ≈ 0.54, got {:?}",
            report.factor_exposures.get("momentum")
        );

        let total_contrib: f64 = report.factor_contributions.values().sum();
        assert!(total_contrib > 0.0 && total_contrib < 1.5);
    }

    #[test]
    fn barra_with_wls_and_orthogonalization() {
        // Synthetic: 4 stocks, 2 factors, 30 days. Use WLS and orthogonalization.
        let n_days = 30;
        let n_assets = 4;
        let k = 2;

        let mut rng = rand::thread_rng();
        use rand::Rng;

        let mut exp0 = Array2::zeros((n_days, n_assets));
        let mut exp1 = Array2::zeros((n_days, n_assets));
        for d in 0..n_days {
            exp0[[d, 0]] = 1.0;
            exp0[[d, 1]] = 0.0;
            exp0[[d, 2]] = 0.5;
            exp0[[d, 3]] = 0.5;
            exp1[[d, 0]] = 0.0;
            exp1[[d, 1]] = 1.0;
            exp1[[d, 2]] = 0.5;
            exp1[[d, 3]] = 0.5;
        }

        let mut returns = Array2::zeros((n_days, n_assets));
        for d in 0..n_days {
            for a in 0..n_assets {
                returns[[d, a]] = rng.r#gen::<f64>() * 0.05;
            }
        }

        let weights = Array2::from_elem((n_days, n_assets), 1.0 / n_assets as f64);

        let mut model = BarraRiskModel::new(
            Box::new(NeweyWestEstimator::default()),
            vec!["f0".to_string(), "f1".to_string()],
        );
        model.regression_method = BarraRegressionMethod::WLS;
        model.orthogonalize = true;

        let report = model.analyze(&weights, &returns, &[exp0, exp1]).unwrap();
        assert!(report.total_risk > 0.0);
        assert!(report.total_risk.is_finite());
        assert!(report.systematic_risk.is_finite());
        assert!(report.r_squared > 0.0 && report.r_squared <= 1.0);
    }

    // ---------- Return decomposition ----------

    #[test]
    fn decompose_return_reconstructs_portfolio_return() {
        // Synthetic: 4 stocks, 2 factors, 20 days. Reconstruct portfolio return.
        let n_days = 20;
        let n_assets = 4;
        let k = 2;

        let mut rng = rand::thread_rng();
        use rand::Rng;

        let mut exp0 = Array2::zeros((n_days, n_assets));
        let mut exp1 = Array2::zeros((n_days, n_assets));
        for d in 0..n_days {
            exp0[[d, 0]] = 1.0;
            exp1[[d, 1]] = 1.0;
        }

        let mut returns = Array2::zeros((n_days, n_assets));
        for d in 0..n_days {
            for a in 0..n_assets {
                returns[[d, a]] = rng.r#gen::<f64>() * 0.05;
            }
        }

        // Run full Barra to get factor_returns and specific_returns
        let model = BarraRiskModel::new(
            Box::new(SampleCovEstimator),
            vec!["f0".to_string(), "f1".to_string()],
        );
        let weights = Array2::from_elem((n_days, n_assets), 1.0 / n_assets as f64);
        let report = model.analyze(&weights, &returns, &[exp0.clone(), exp1.clone()]).unwrap();
        assert!(report.total_risk.is_finite());

        // Now decompose. We need to re-run the regression to get factor_returns + specific_returns.
        let (factor_ret, specific_ret) =
            estimate_factor_returns(&returns, &[exp0.clone(), exp1.clone()], &BarraRegressionMethod::OLS, None, false)
            .unwrap();

        let avg_w = average_weights(&weights);
        let (r_factor, r_specific) =
            decompose_portfolio_return(&avg_w, &factor_ret, &specific_ret, &[exp0.clone(), exp1.clone()]).unwrap();

        // Reconstruct: r_port = r_factor + r_specific
        let r_port = crate::risk::simple::compute_portfolio_returns(&weights, &returns);
        for d in 0..n_days {
            let reconstructed = r_factor[d] + r_specific[d];
            assert!(
                (reconstructed - r_port[d]).abs() < 1e-12,
                "Day {}: reconstructed {:.6} ≠ portfolio return {:.6}",
                d,
                reconstructed,
                r_port[d]
            );
        }
    }

    #[test]
    fn barra_handles_nan_in_returns() {
        let n_days = 20;
        let n_assets = 5;
        let k = 2;

        let mut rng = rand::thread_rng();
        use rand::Rng;

        let mut exp0 = Array2::zeros((n_days, n_assets));
        let mut exp1 = Array2::zeros((n_days, n_assets));
        for d in 0..n_days {
            exp0[[d, 0]] = 1.0;
            exp1[[d, 1]] = 1.0;
        }

        let mut returns = Array2::zeros((n_days, n_assets));
        for d in 0..n_days {
            for a in 0..n_assets {
                returns[[d, a]] = rng.r#gen::<f64>() * 0.02;
            }
        }
        returns[[0, 0]] = f64::NAN;

        let weights = Array2::from_elem((n_days, n_assets), 1.0 / n_assets as f64);

        let model = BarraRiskModel::new(
            Box::new(SampleCovEstimator),
            vec!["f0".to_string(), "f1".to_string()],
        );

        let report = model.analyze(&weights, &returns, &[exp0, exp1]).unwrap();
        assert!(report.total_risk.is_finite());
    }

    // ---------- Rolling analysis ----------

    #[test]
    fn rolling_analysis_correct_count() {
        // Synthetic: 60 days, 4 assets, window=30, step=10 → 4 windows
        let n_days = 60;
        let n_assets = 4;
        let k = 2;

        let mut rng = rand::thread_rng();
        use rand::Rng;
        let vals: Vec<f64> = (0..n_days * n_assets).map(|_| rng.r#gen::<f64>() * 0.02).collect();
        let returns = Array2::from_shape_vec((n_days, n_assets), vals).unwrap();
        let weights = Array2::from_elem((n_days, n_assets), 1.0 / n_assets as f64);
        let exp0 = Array2::from_elem((n_days, n_assets), 1.0);
        let exp1 = Array2::from_elem((n_days, n_assets), 0.5);

        let model = BarraRiskModel::new(
            Box::new(SampleCovEstimator),
            vec!["f0".to_string(), "f1".to_string()],
        );

        let reports = model
            .analyze_rolling(&weights, &returns, &[exp0, exp1], 30, 10)
            .unwrap();
        // windows: 0-29, 10-39, 20-49, 30-59 = 4
        assert_eq!(reports.len(), 4, "Expected 4 rolling windows, got {}", reports.len());
        for r in &reports {
            assert!(r.total_risk.is_finite());
            assert!(r.total_risk > 0.0);
        }
    }

    #[test]
    fn rolling_window_too_large_returns_empty() {
        let n_days = 10;
        let n_assets = 3;
        let returns = Array2::from_elem((n_days, n_assets), 0.01);
        let weights = Array2::from_elem((n_days, n_assets), 1.0 / n_assets as f64);

        let model = BarraRiskModel::new(
            Box::new(SampleCovEstimator),
            vec!["f0".to_string()],
        );
        let reports = model
            .analyze_rolling(&weights, &returns, &[Array2::from_elem((n_days, n_assets), 1.0)], 20, 5)
            .unwrap();
        assert!(reports.is_empty());
    }

    #[test]
    fn barra_empty_exposures_is_error() {
        let model = BarraRiskModel::new(
            Box::new(SampleCovEstimator),
            vec![],
        );
        let data = Array2::from_elem((10, 3), 0.01);
        assert!(model.analyze(&data, &data, &[]).is_err());
    }
}
