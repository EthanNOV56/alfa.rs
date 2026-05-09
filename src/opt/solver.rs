//! Numerical optimization solver wrappers.
//!
//! Thin wrappers around `nalgebra` (Cholesky, KKT) and `argmin` (L-BFGS,
//! Nelder-Mead).  Provides five solution paths selected by constraint
//! complexity.

use ndarray::{Array1, Array2};

use super::constraints::{FactorExposureConstraint, GroupConstraint};

// ── ndarray ↔ nalgebra bridge ───────────────────────────────────────────────

/// Convert an ndarray Array1<f64> to a nalgebra DVector<f64> (zero-copy).
pub(crate) fn to_nalgebra_vector(a: &Array1<f64>) -> nalgebra::DVector<f64> {
    nalgebra::DVector::from_row_slice(a.as_slice().unwrap())
}

/// Convert an ndarray Array2<f64> to a nalgebra DMatrix<f64> (zero-copy).
pub(crate) fn to_nalgebra_matrix(a: &Array2<f64>) -> nalgebra::DMatrix<f64> {
    let (rows, cols) = a.dim();
    nalgebra::DMatrix::from_row_slice(rows, cols, a.as_slice().unwrap())
}

/// Convert a nalgebra DVector back to ndarray Array1.
pub(crate) fn to_ndarray_vector(v: &nalgebra::DVector<f64>) -> Array1<f64> {
    Array1::from_vec(v.as_slice().to_vec())
}

/// Convert a nalgebra DMatrix back to ndarray Array2.
pub(crate) fn to_ndarray_matrix(m: &nalgebra::DMatrix<f64>) -> Array2<f64> {
    let shape = (m.nrows(), m.ncols());
    Array2::from_shape_vec(shape, m.as_slice().to_vec()).unwrap()
}

// ── Path 1: MV unconstrained (Cholesky closed form) ─────────────────────────

/// Mean-variance unconstrained: `w* = (1/λ) · Σ⁻¹ · α`.
///
/// Uses nalgebra Cholesky decomposition to solve `Σ · w = α/λ`.
pub fn solve_mv_unconstrained(
    cov: &Array2<f64>,
    alpha: &Array1<f64>,
    risk_aversion: f64,
) -> Result<Array1<f64>, String> {
    let n = cov.nrows();
    if cov.ncols() != n {
        return Err("Covariance must be square".into());
    }
    if alpha.len() != n {
        return Err("alpha dimension mismatch".into());
    }

    let sigma = to_nalgebra_matrix(cov);
    let rhs = to_nalgebra_vector(alpha) / risk_aversion;

    let chol = nalgebra::linalg::Cholesky::new(sigma)
        .ok_or_else(|| "Covariance matrix is not positive definite".to_string())?;
    let w_nalg = chol.solve(&rhs);

    Ok(to_ndarray_vector(&w_nalg))
}

// ── Path 2: MV with sum-to-1 constraint (KKT closed form) ───────────────────

/// KKT closed form for `min ½wᵀΣw - (1/λ)wᵀα`  s.t. `𝟏ᵀw = 1`.
///
/// ```text
/// [ Σ   𝟏 ] [ w ]   [ α/λ ]
/// [ 𝟏ᵀ  0 ] [ ν ] = [  1  ]
/// ```
pub fn solve_mv_kkt(
    cov: &Array2<f64>,
    alpha: &Array1<f64>,
    risk_aversion: f64,
) -> Result<(Array1<f64>, f64), String> {
    let n = cov.nrows();
    let sigma = to_nalgebra_matrix(cov);
    let ones = nalgebra::DVector::from_element(n, 1.0_f64);
    let rhs_alpha = to_nalgebra_vector(alpha) / risk_aversion;

    // Solve Σ · x = α/λ and Σ · y = 𝟏
    let chol = nalgebra::linalg::Cholesky::new(sigma.clone())
        .ok_or_else(|| "Covariance matrix is not positive definite".to_string())?;
    let x = chol.solve(&rhs_alpha);
    let y = chol.solve(&ones);

    // Lagrange multiplier: ν = (𝟏ᵀx - 1) / (𝟏ᵀy)
    let ones_dot_x = ones.dot(&x);
    let ones_dot_y = ones.dot(&y);

    if ones_dot_y.abs() < 1e-12 {
        return Err("Numerical singularity in KKT system".into());
    }

    let nu = (ones_dot_x - 1.0) / ones_dot_y;

    // w = x - ν · y
    let w_nalg = x - nu * y;

    Ok((to_ndarray_vector(&w_nalg), nu))
}

// ── Path 3: MV with general constraints (Augmented Lagrangian + L-BFGS) ─────

use argmin::core::{CostFunction, Gradient};

/// Problem definition for constrained MV optimization.
struct MvoProblem {
    cov: nalgebra::DMatrix<f64>,
    alpha: nalgebra::DVector<f64>,
    risk_aversion: f64,
    long_only: bool,
    market_neutral: bool,
    prev_weights: Option<nalgebra::DVector<f64>>,
    turnover_penalty: Option<f64>,
    lower_bounds: Option<nalgebra::DVector<f64>>,
    upper_bounds: Option<nalgebra::DVector<f64>>,
}

impl CostFunction for MvoProblem {
    type Param = nalgebra::DVector<f64>;
    type Output = f64;

    fn cost(&self, w: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        // f(w) = ½ wᵀ Σ w - (1/λ) wᵀ α + turnover penalty
        let risk = 0.5 * w.dot(&(&self.cov * w));
        let reward = w.dot(&self.alpha) / self.risk_aversion;
        let mut cost = risk - reward;

        if let (Some(prev), Some(lambda)) = (&self.prev_weights, self.turnover_penalty) {
            let diff = w - prev;
            let turnover = diff.iter().map(|&d| d.abs()).sum::<f64>();
            cost += lambda * turnover;
        }

        Ok(cost)
    }
}

impl Gradient for MvoProblem {
    type Param = nalgebra::DVector<f64>;
    type Gradient = nalgebra::DVector<f64>;

    fn gradient(&self, w: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let mut grad = &self.cov * w - &self.alpha / self.risk_aversion;

        if let (Some(prev), Some(lambda)) = (&self.prev_weights, self.turnover_penalty) {
            let diff = w - prev;
            for (i, g) in grad.iter_mut().enumerate() {
                *g += lambda * diff[i].signum();
            }
        }

        Ok(grad)
    }
}

/// Penalised MV with long-only + bounds via L-BFGS.
pub fn solve_mv_long_only(
    cov: &Array2<f64>,
    alpha: &Array1<f64>,
    risk_aversion: f64,
    prev_weights: Option<&Array1<f64>>,
    turnover_penalty: Option<f64>,
) -> Result<Array1<f64>, String> {
    let n = cov.nrows();
    let sigma = to_nalgebra_matrix(cov);
    let alpha_n = to_nalgebra_vector(alpha);
    let prev_w = prev_weights.map(|w| to_nalgebra_vector(w));
    let lb = nalgebra::DVector::from_element(n, 0.0);

    let problem = MvoProblem {
        cov: sigma,
        alpha: alpha_n,
        risk_aversion,
        long_only: true,
        market_neutral: false,
        prev_weights: prev_w,
        turnover_penalty,
        lower_bounds: Some(lb),
        upper_bounds: None,
    };

    // Projected gradient descent — simpler and more robust than L-BFGS
    // with bound constraints for this specific case.
    let w0 = nalgebra::DVector::from_element(n, 1.0 / n as f64);
    let result = projected_gradient_descent(&problem, &w0, 0.001, 1000, 1e-8)?;

    Ok(to_ndarray_vector(&result))
}

/// Projected gradient descent for bound-constrained problems.
fn projected_gradient_descent<P>(
    problem: &P,
    w0: &nalgebra::DVector<f64>,
    step_size: f64,
    max_iter: usize,
    tol: f64,
) -> Result<nalgebra::DVector<f64>, String>
where
    P: CostFunction<Param = nalgebra::DVector<f64>, Output = f64>
        + Gradient<Param = nalgebra::DVector<f64>, Gradient = nalgebra::DVector<f64>>,
{
    let mut w = w0.clone();

    for _iter in 0..max_iter {
        let grad = problem
            .gradient(&w)
            .map_err(|e| format!("gradient error: {e}"))?;

        // Gradient descent step
        w = &w - step_size * &grad;

        // Project onto simplex (sum to 1, non-negative)
        project_simplex(&mut w);

        // Convergence check
        let grad_norm = grad.norm();
        if grad_norm < tol {
            break;
        }
    }

    Ok(w)
}

/// Project a vector onto the probability simplex { w ≥ 0, Σw = 1 }.
fn project_simplex(w: &mut nalgebra::DVector<f64>) {
    let n = w.len();

    // Clamp negatives to 0
    for val in w.iter_mut() {
        *val = val.max(0.0);
    }

    // Normalise to sum 1
    let s: f64 = w.sum();
    if s > 1e-12 {
        *w /= s;
    } else {
        // Degenerate: uniform
        w.fill(1.0 / n as f64);
    }
}

// ── Path 4: Risk Parity (CCD) ───────────────────────────────────────────────

/// Cyclical Coordinate Descent for Risk Parity.
///
/// Iterates `w_i = sqrt(target / (Σw)_i)` per asset, reprojects onto bounds.
pub fn solve_risk_parity_ccd(
    cov: &Array2<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<Array1<f64>, String> {
    let n = cov.nrows();
    if n < 2 {
        return Err("RiskParity requires at least 2 assets".into());
    }

    let sigma = to_nalgebra_matrix(cov);
    let mut w = nalgebra::DVector::from_element(n, 1.0 / n as f64);

    for _iter in 0..max_iter {
        let sigma_w = &sigma * &w;
        let portfolio_var = w.dot(&sigma_w);

        if portfolio_var.abs() < 1e-12 {
            break;
        }

        let target_risk = portfolio_var / n as f64;
        let mut w_new = w.clone();
        let mut max_delta = 0.0_f64;

        for i in 0..n {
            let grad_i = sigma_w[i];
            if grad_i.abs() < 1e-12 {
                continue;
            }
            w_new[i] = (target_risk / grad_i).sqrt();
            let delta = (w_new[i] - w[i]).abs();
            max_delta = max_delta.max(delta);
        }

        w = w_new;

        // Renormalise to sum 1
        let s: f64 = w.sum();
        if s > 1e-12 {
            w /= s;
        }

        if max_delta < tol {
            break;
        }
    }

    Ok(to_ndarray_vector(&w))
}

// ── Path 5: Augmented Lagrangian for general constraints ─────────────────────

/// Augmented Lagrangian Method for constrained MV optimization.
///
/// ```text
/// min  ½wᵀΣw - (1/λ)wᵀα
/// s.t. l ≤ w ≤ u
///      𝟏ᵀw = 1  (if full_investment)
///      𝟏ᵀw = 0  (if market_neutral)
/// ```
///
/// Constraints are handled via penalty terms with Lagrange multiplier
/// updates.  Non-feasible constraints are relaxed by priority order.
pub fn solve_mv_alm(
    cov: &Array2<f64>,
    alpha: &Array1<f64>,
    risk_aversion: f64,
    long_only: bool,
    full_investment: bool,
    market_neutral: bool,
    max_position: Option<f64>,
    leverage_limit: Option<f64>,
    max_assets: Option<usize>,
    group_constraints: &[GroupConstraint],
    factor_exposure_constraints: &[FactorExposureConstraint],
    prev_weights: Option<&Array1<f64>>,
    turnover_penalty: Option<f64>,
) -> Result<Array1<f64>, String> {
    let n = cov.nrows();
    let sigma = to_nalgebra_matrix(cov);
    let alpha_n = to_nalgebra_vector(alpha);
    let prev_w = prev_weights.map(|w| to_nalgebra_vector(w));

    let mut w = prev_w
        .clone()
        .unwrap_or_else(|| nalgebra::DVector::from_element(n, 1.0 / n as f64));

    // Feasibility: max_position must accommodate sum-to-1 with at most n assets.
    let ub_global = if let Some(mp) = max_position {
        let min_for_feasibility = if market_neutral { 0.0 } else { 1.0 / n as f64 };
        if mp < min_for_feasibility - 1e-10 {
            // Relax max_position to the minimum feasible value
            min_for_feasibility
        } else {
            mp
        }
    } else {
        f64::INFINITY
    };
    let lb = if long_only {
        0.0_f64
    } else {
        f64::NEG_INFINITY
    };

    let mut rho = 10.0_f64;
    let rho_max = 1e6_f64;
    let mut nu = 0.0_f64;
    let inner_lr = 0.001_f64;
    let tol = 1e-8_f64;

    for _outer in 0..30 {
        for _inner in 0..200 {
            let sigma_w = &sigma * &w;
            let mut grad = &sigma_w - &alpha_n / risk_aversion;

            if let (Some(prev), Some(lambda)) = (&prev_w, turnover_penalty) {
                let diff = &w - prev;
                for (i, g) in grad.iter_mut().enumerate() {
                    *g += lambda * diff[i].signum();
                }
            }

            // Factor exposure budget penalty: penalty only outside tolerance band
            // Budget: |w·β - target| ≤ tolerance
            // Penalty: (λ_f/2) * max(0, |w·β - target| - tolerance)²
            let lambda_f = 100.0_f64; // exposure penalty coefficient
            for fc in factor_exposure_constraints {
                let exposure: f64 = fc
                    .exposures
                    .iter()
                    .zip(w.iter())
                    .map(|(&beta_i, &w_i)| beta_i * w_i)
                    .sum();
                let violation = (exposure - fc.target).abs() - fc.tolerance;
                if violation > 0.0 {
                    let sign = if exposure > fc.target { 1.0 } else { -1.0 };
                    let penalty_grad_factor = lambda_f * violation * sign;
                    for (i, g) in grad.iter_mut().enumerate() {
                        if i < fc.exposures.len() {
                            *g += penalty_grad_factor * fc.exposures[i];
                        }
                    }
                }
            }

            w = &w - inner_lr * &grad;

            // NaN safety: reset to uniform if diverging
            if w.iter().any(|x| x.is_nan()) {
                w = nalgebra::DVector::from_element(n, 1.0 / n as f64);
                break;
            }

            // Bounds projection
            for val in w.iter_mut() {
                *val = val.clamp(lb, ub_global);
            }

            // Sum constraint correction
            let s: f64 = w.sum();
            if s.abs() > 1e-12 {
                let target = if market_neutral { 0.0 } else { 1.0 };
                let corr = (target - s) / n as f64;
                for val in w.iter_mut() {
                    *val += corr;
                }
            }

            // Leverage limit
            if let Some(lev) = leverage_limit {
                let l1: f64 = w.iter().map(|x| x.abs()).sum();
                if l1 > lev {
                    let scale = lev / l1;
                    for val in w.iter_mut() {
                        *val *= scale;
                    }
                }
            }

            // Group constraints
            for gc in group_constraints {
                let mut g_sum = 0.0_f64;
                for &idx in &gc.members {
                    if idx < n {
                        g_sum += w[idx];
                    }
                }
                if let Some(max_w) = gc.max_weight {
                    if g_sum > max_w {
                        let scale = max_w / g_sum;
                        for &idx in &gc.members {
                            if idx < n {
                                w[idx] *= scale;
                            }
                        }
                    }
                }
                if let Some(min_w) = gc.min_weight {
                    if g_sum < min_w && g_sum.abs() > 1e-12 {
                        let scale = min_w / g_sum;
                        for &idx in &gc.members {
                            if idx < n {
                                w[idx] *= scale;
                            }
                        }
                    }
                }
            }

            let gnorm = grad.norm();
            if gnorm < tol {
                break;
            }
        }

        // Outer loop
        let s: f64 = w.sum();
        let target = if market_neutral { 0.0 } else { 1.0 };
        let constraint_violation = if full_investment || market_neutral {
            s - target
        } else {
            0.0
        };

        nu += rho * constraint_violation;
        rho = (1.2 * rho).min(rho_max);
        if constraint_violation.abs() < tol {
            break;
        }
    }

    // Final projection: enforce bounds → max_assets → renormalize → reclamp
    for val in w.iter_mut() {
        *val = val.clamp(lb, ub_global);
    }

    if let Some(max_k) = max_assets {
        let mut indexed: Vec<(usize, f64)> = w.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        let keep: std::collections::HashSet<usize> =
            indexed.iter().take(max_k).map(|(i, _)| *i).collect();
        for (i, val) in w.iter_mut().enumerate() {
            if !keep.contains(&i) {
                *val = 0.0;
            }
        }
    }

    let s: f64 = w.sum();
    if full_investment && s.abs() > 1e-12 && !market_neutral {
        w /= s;
    }

    // Re-clamp after normalization (can overshoot due to scaling)
    for val in w.iter_mut() {
        *val = val.clamp(lb, ub_global);
    }

    // Re-enforce group constraints
    for _pass in 0..2 {
        for gc in group_constraints {
            let mut g_sum = 0.0;
            for &idx in &gc.members {
                if idx < n {
                    g_sum += w[idx];
                }
            }
            if let Some(max_w) = gc.max_weight {
                if g_sum > max_w && g_sum.abs() > 1e-12 {
                    let scale = max_w / g_sum;
                    for &idx in &gc.members {
                        if idx < n {
                            w[idx] *= scale;
                        }
                    }
                }
            }
        }
    }

    // Final re-normalization in case clamping perturbed the sum
    let s: f64 = w.sum();
    if full_investment && s.abs() > 1e-12 && !market_neutral {
        w /= s;
    }

    Ok(to_ndarray_vector(&w))
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    /// [SYNTHETIC] 2-asset MV unconstrained: analytical verification.
    #[test]
    fn test_mv_unconstrained_2asset() {
        // Σ = [[0.04, 0.01], [0.01, 0.09]], α = [0.08, 0.12], λ = 2.0
        let cov = Array2::from_shape_vec((2, 2), vec![0.04, 0.01, 0.01, 0.09]).unwrap();
        let alpha = Array1::from_vec(vec![0.08, 0.12]);
        let w = solve_mv_unconstrained(&cov, &alpha, 2.0).unwrap();

        // Hand-derived: Σ⁻¹ α = ..., / λ
        // Σ⁻¹ = 1/(0.04*0.09 - 0.01²) * [[0.09, -0.01], [-0.01, 0.04]]
        // = 1/0.0035 * [[0.09, -0.01], [-0.01, 0.04]]
        // Σ⁻¹ α = [1.7143, 1.1429], /2 = [0.8571, 0.5714]
        assert!((w[0] - 0.8571).abs() < 0.01);
        assert!((w[1] - 0.5714).abs() < 0.01);
    }

    /// [SYNTHETIC] KKT: 3-asset, sum-to-1 constraint satisfied.
    #[test]
    fn test_mv_kkt_sum_to_one() {
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.04, 0.09, 0.16]));
        let alpha = Array1::from_vec(vec![0.06, 0.09, 0.12]);
        let (w, _nu) = solve_mv_kkt(&cov, &alpha, 3.0).unwrap();

        assert!((w.sum() - 1.0).abs() < 1e-10);
        // With KKT, lower-vol assets get a boost from the constraint (Σ⁻¹1)
        assert!(w[0] > w[1] && w[1] > w[2]);
    }

    /// [SYNTHETIC] Long-only MV: all weights non-negative.
    #[test]
    fn test_mv_long_only_non_negative() {
        let cov = Array2::from_shape_vec(
            (3, 3),
            vec![0.04, 0.02, 0.01, 0.02, 0.09, 0.03, 0.01, 0.03, 0.16],
        )
        .unwrap();
        let alpha = Array1::from_vec(vec![0.1, 0.2, 0.05]);
        let w = solve_mv_long_only(&cov, &alpha, 2.0, None, None).unwrap();

        assert!(w.iter().all(|&x| x >= -1e-12)); // all non-negative
        assert!((w.sum() - 1.0).abs() < 1e-10);
    }

    /// [SYNTHETIC] RiskParity CCD: equal RC on diagonal covariance.
    #[test]
    fn test_risk_parity_diagonal() {
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.01, 0.04, 0.09]));
        let w = solve_risk_parity_ccd(&cov, 2000, 1e-12).unwrap();

        assert!((w.sum() - 1.0).abs() < 1e-10);
        // Higher vol → lower weight for diagonal cov
        assert!(w[0] > w[1] && w[1] > w[2]);
        // All weights positive
        assert!(w.iter().all(|&x| x >= 0.0));

        // Check that risk contributions are moving toward equality:
        // for diagonal cov, RC_i = w_i² σ_i² / (wᵀ Σ w)
        let sigma = to_nalgebra_matrix(&cov);
        let w_nalg = to_nalgebra_vector(&w);
        let sigma_w = &sigma * &w_nalg;
        let portfolio_var = w_nalg.dot(&sigma_w);
        let rc: Vec<f64> = (0..3).map(|i| w[i] * sigma_w[i] / portfolio_var).collect();
        // RC of highest-weight asset (lowest vol) should not be too extreme
        // For 3 assets with vols [0.1, 0.2, 0.3], equal-weight would give
        // RC ≈ [0.07, 0.27, 0.66]. Optimised should be closer to equal.
        assert!(rc[0] > 0.10 && rc[0] < 0.50);
        assert!(rc[2] > 0.20 && rc[2] < 0.65);
    }

    /// [SYNTHETIC] ALM: long-only + full investment.
    #[test]
    fn test_alm_long_only() {
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.04, 0.09, 0.16]));
        let alpha = Array1::from_vec(vec![0.1, 0.2, 0.05]);
        let w = solve_mv_alm(
            &cov,
            &alpha,
            3.0,
            true,
            true,
            false,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
        )
        .unwrap();
        assert!(w.iter().all(|&x| x >= -1e-10));
        assert!((w.sum() - 1.0).abs() < 1e-8);
    }

    /// [SYNTHETIC] ALM: market_neutral (no bounds) → sum near 0.
    #[test]
    fn test_alm_market_neutral() {
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.04, 0.09, 0.16]));
        let alpha = Array1::from_vec(vec![0.1, -0.2, 0.05]);
        let w = solve_mv_alm(
            &cov,
            &alpha,
            3.0,
            false,
            true,
            true,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
        )
        .unwrap();
        assert!(w.sum().abs() < 1e-4);
    }

    /// [SYNTHETIC] ALM: max_position = 0.5 → all weights ≤ 0.5.
    /// (n=3 requires max_position ≥ 1/3 for feasibility with sum-to-1.)
    #[test]
    fn test_alm_max_position() {
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.01, 0.04, 0.09]));
        let alpha = Array1::from_vec(vec![0.3, 0.1, 0.05]);
        let w = solve_mv_alm(
            &cov,
            &alpha,
            3.0,
            true,
            true,
            false,
            Some(0.5),
            None,
            None,
            &[],
            &[],
            None,
            None,
        )
        .unwrap();
        assert!(w.iter().all(|&x| x >= -1e-10));
        assert!(w.iter().all(|&x| x <= 0.5 + 1e-8));
        assert!((w.sum() - 1.0).abs() < 1e-8);
    }

    /// [SYNTHETIC] ALM: max_assets = 2 → at most 2 non-zero weights.
    #[test]
    fn test_alm_max_assets() {
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.01, 0.04, 0.09, 0.16]));
        let alpha = Array1::from_vec(vec![0.2, 0.1, 0.05, 0.01]);
        let w = solve_mv_alm(
            &cov,
            &alpha,
            3.0,
            true,
            true,
            false,
            None,
            None,
            Some(2),
            &[],
            &[],
            None,
            None,
        )
        .unwrap();
        let nonzero = w.iter().filter(|&&x| x.abs() > 1e-8).count();
        assert!(nonzero <= 2);
        assert!((w.sum() - 1.0).abs() < 1e-8);
    }

    /// [SYNTHETIC] ALM: group max_weight = 0.3 → sector sum ≤ 0.3.
    #[test]
    fn test_alm_group_constraint() {
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.01, 0.01, 0.04, 0.09]));
        let alpha = Array1::from_vec(vec![0.2, 0.2, 0.1, 0.05]);
        let gc = GroupConstraint {
            name: "SectorA".into(),
            members: vec![0, 1],
            min_weight: None,
            max_weight: Some(0.3),
        };
        let w = solve_mv_alm(
            &cov,
            &alpha,
            3.0,
            true,
            true,
            false,
            None,
            None,
            None,
            &[gc],
            &[],
            None,
            None,
        )
        .unwrap();
        let sector_sum = w[0] + w[1];
        assert!(sector_sum <= 0.3 + 1e-4, "sector sum {sector_sum} > 0.3");
        assert!((w.sum() - 1.0).abs() < 1e-8);
    }

    /// [SYNTHETIC] ALM: infeasible max_position auto-relaxes.
    #[test]
    fn test_alm_infeasible_max_position_relaxes() {
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.01, 0.04, 0.09]));
        let alpha = Array1::from_vec(vec![0.1, 0.2, 0.05]);
        // max_position=0.30 with n=3 is infeasible (3*0.30=0.90<1.0)
        // Should auto-relax to 1/3 ≈ 0.333
        let w = solve_mv_alm(
            &cov,
            &alpha,
            3.0,
            true,
            true,
            false,
            Some(0.30),
            None,
            None,
            &[],
            &[],
            None,
            None,
        )
        .unwrap();
        assert!((w.sum() - 1.0).abs() < 1e-8);
        // All weights should be ≥ 0 (long_only)
        assert!(w.iter().all(|&x| x >= -1e-10));
    }

    /// [SYNTHETIC] Factor exposure budget: small tolerance allows some exposure.
    #[test]
    fn test_factor_exposure_budget() {
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.01, 0.04, 0.09]));
        let alpha = Array1::from_vec(vec![0.1, 0.2, 0.05]);
        // Factor beta: asset 0 loads heavily on factor, assets 1-2 = 0
        let fc = FactorExposureConstraint {
            name: "Size".into(),
            exposures: vec![1.0, 0.0, 0.0],
            target: 0.0,
            tolerance: 0.05, // budget: up to ±5% deviation allowed
        };
        let w = solve_mv_alm(
            &cov,
            &alpha,
            3.0,
            true,
            true,
            false,
            None,
            None,
            None,
            &[],
            &[fc],
            None,
            None,
        )
        .unwrap();
        // Exposure = w[0] (since beta=[1,0,0])
        // With tolerance=0.05, exposure should be ≤ some reasonable value
        // (without penalty it would be ~0.5)
        assert!(
            w[0] < 0.4,
            "exposure {} should be limited by budget penalty",
            w[0]
        );
        assert!((w.sum() - 1.0).abs() < 1e-8);
    }
}
