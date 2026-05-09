//! Risk budgeting methods: RiskParity, MaxDiversification.

use ndarray::{Array1, Array2};

use super::solver;
use super::{CostModel, Optimizer, OptimizerConstraints};

// ── RiskParity ───────────────────────────────────────────────────────────────

/// Equal risk contribution: RC_i = RC_j for all i, j.
pub struct RiskParity {
    pub max_iter: usize,
    pub tolerance: f64,
}

impl Default for RiskParity {
    fn default() -> Self {
        Self {
            max_iter: 2000,
            tolerance: 1e-10,
        }
    }
}

impl Optimizer for RiskParity {
    fn optimize_day(
        &self,
        _signal: &Array1<f64>,
        covariance: &Array2<f64>,
        _prev_weights: Option<&Array1<f64>>,
        constraints: &OptimizerConstraints,
        _cost_model: Option<&dyn CostModel>,
    ) -> Result<Array1<f64>, String> {
        let mut w = solver::solve_risk_parity_ccd(covariance, self.max_iter, self.tolerance)?;

        // Apply bounds
        if constraints.long_only {
            for val in w.iter_mut() {
                *val = val.max(0.0);
            }
            let s: f64 = w.sum();
            if s > 1e-12 {
                w /= s;
            }
        }

        Ok(w)
    }
}

// ── MaxDiversification ──────────────────────────────────────────────────────

/// Maximise diversification ratio: max (wᵀσ) / √(wᵀΣw).
pub struct MaxDiversification {
    pub max_iter: usize,
    pub tolerance: f64,
}

impl Default for MaxDiversification {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tolerance: 1e-8,
        }
    }
}

impl Optimizer for MaxDiversification {
    fn optimize_day(
        &self,
        _signal: &Array1<f64>,
        covariance: &Array2<f64>,
        _prev_weights: Option<&Array1<f64>>,
        constraints: &OptimizerConstraints,
        _cost_model: Option<&dyn CostModel>,
    ) -> Result<Array1<f64>, String> {
        let n = covariance.nrows();
        let vols: Array1<f64> = covariance.diag().map(|v| v.sqrt());

        // Gradient descent on DR(w) = (wᵀσ) / √(wᵀΣw)
        let mut w = Array1::<f64>::from_elem(n, 1.0 / n as f64);
        let lr = 0.001;

        for _iter in 0..self.max_iter {
            let sigma_nalg = super::solver::to_nalgebra_matrix(covariance);
            let w_nalg = super::solver::to_nalgebra_vector(&w);
            let sigma_w = &sigma_nalg * &w_nalg;

            use super::solver::to_ndarray_vector;
            let port_vol = w_nalg.dot(&sigma_w).sqrt();
            if port_vol < 1e-12 {
                break;
            }
            let vol_sum = w.dot(&vols);

            // Gradient: ∂DR/∂w_i = σ_i / port_vol - DR · (Σw)_i / port_vol²
            let dr = vol_sum / port_vol;
            let grad_w = &vols / port_vol
                - &to_ndarray_vector(&sigma_w) * (dr / (port_vol * port_vol));

            w = &w + lr * &grad_w;

            // Project: long-only, sum to 1
            if constraints.long_only {
                for val in w.iter_mut() {
                    *val = val.max(0.0);
                }
            }
            let s: f64 = w.sum();
            if s > 1e-12 {
                w /= s;
            }

            let grad_norm: f64 = grad_w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if grad_norm < self.tolerance {
                break;
            }
        }

        Ok(w)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// [SYNTHETIC] RiskParity: higher vol → lower weight.
    #[test]
    fn test_risk_parity_vol_ranking() {
        let opt = RiskParity::default();
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.01, 0.04, 0.09]));
        let constraints = OptimizerConstraints {
            long_only: true,
            full_investment: true,
            ..Default::default()
        };
        let w = opt
            .optimize_day(
                &Array1::zeros(3),
                &cov,
                None,
                &constraints,
                None,
            )
            .unwrap();
        assert!(w[0] > w[1] && w[1] > w[2]);
        assert!((w.sum() - 1.0).abs() < 1e-10);
    }

    /// [SYNTHETIC] MaxDiversification: produces valid weights.
    #[test]
    fn test_max_diversification() {
        let opt = MaxDiversification::default();
        let cov = Array2::from_shape_vec(
            (3, 3),
            vec![0.04, 0.02, 0.01, 0.02, 0.09, 0.03, 0.01, 0.03, 0.16],
        )
        .unwrap();
        let constraints = OptimizerConstraints {
            long_only: true,
            full_investment: true,
            ..Default::default()
        };
        let w = opt
            .optimize_day(
                &Array1::zeros(3),
                &cov,
                None,
                &constraints,
                None,
            )
            .unwrap();
        assert!(w.iter().all(|&x| x >= -1e-12));
        assert!((w.sum() - 1.0).abs() < 1e-10);
    }
}
