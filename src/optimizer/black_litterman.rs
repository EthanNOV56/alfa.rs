//! Black-Litterman Bayesian fusion.

use ndarray::{Array1, Array2};

use super::solver;
use super::BlackLittermanConfig;
use super::{CostModel, Optimizer, OptimizerConstraints};

pub struct BlackLitterman;

impl BlackLitterman {
    /// Implied equilibrium returns: π = δ · Σ · w_eq.
    pub fn implied_returns(
        delta: f64,
        cov: &Array2<f64>,
        w_eq: &Array1<f64>,
    ) -> Array1<f64> {
        let sigma = solver::to_nalgebra_matrix(cov);
        let w = solver::to_nalgebra_vector(w_eq);
        let pi = delta * (&sigma * &w);
        solver::to_ndarray_vector(&pi)
    }

    /// Posterior returns via Bayesian fusion.
    ///
    /// `E[r|views] = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹π + PᵀΩ⁻¹q]`
    pub fn posterior_returns(
        pi: &Array1<f64>,
        tau: f64,
        cov: &Array2<f64>,
        pick: &Array2<f64>,
        views: &Array1<f64>,
        omega_inv: &Array2<f64>,
    ) -> Result<Array1<f64>, String> {
        let n = cov.nrows();
        let k = pick.nrows();

        let sigma = solver::to_nalgebra_matrix(cov);
        let sigma_inv = nalgebra::linalg::Cholesky::new(sigma.clone())
            .ok_or("Covariance not positive-definite")?
            .inverse();
        let tau_sigma_inv = &sigma_inv / tau;

        let p = solver::to_nalgebra_matrix(pick);  // (k × n)
        let omega_inv_n = solver::to_nalgebra_matrix(omega_inv);  // (k × k)
        let pt_omega_inv = p.transpose() * &omega_inv_n;  // (n × k)
        let posterior_precision = &tau_sigma_inv + &pt_omega_inv * &p;

        let pi_n = solver::to_nalgebra_vector(pi);
        let q_n = solver::to_nalgebra_vector(views);
        let rhs = &tau_sigma_inv * &pi_n + &pt_omega_inv * &q_n;

        let posterior_cov = nalgebra::linalg::Cholesky::new(posterior_precision.clone())
            .ok_or("Posterior precision not positive-definite")?
            .inverse();

        let posterior_mean = &posterior_cov * rhs;
        Ok(solver::to_ndarray_vector(&posterior_mean))
    }
}

impl Optimizer for BlackLitterman {
    fn optimize_day(
        &self,
        signal: &Array1<f64>,
        covariance: &Array2<f64>,
        prev_weights: Option<&Array1<f64>>,
        constraints: &OptimizerConstraints,
        cost_model: Option<&dyn CostModel>,
    ) -> Result<Array1<f64>, String> {
        // When called as an Optimizer, delegate to MaxSharpe.
        // The BlackLitterman transformation should be applied upstream.
        super::mvo::MaxSharpe::default()
            .optimize_day(signal, covariance, prev_weights, constraints, cost_model)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// [SYNTHETIC] Implied returns: π = δ · Σ · w_eq.
    #[test]
    fn test_implied_returns() {
        let cov = Array2::from_shape_vec(
            (3, 3),
            vec![0.04, 0.02, 0.01, 0.02, 0.09, 0.03, 0.01, 0.03, 0.16],
        )
        .unwrap();
        let w_eq = Array1::from_vec(vec![0.3, 0.3, 0.4]);
        let pi = BlackLitterman::implied_returns(3.0, &cov, &w_eq);
        assert_eq!(pi.len(), 3);
        // π = 3 * Σ * w_eq
        // Σ * w_eq = [0.04*0.3 + 0.02*0.3 + 0.01*0.4, ...]
        // = [0.012+0.006+0.004, 0.006+0.027+0.012, 0.003+0.009+0.064]
        // = [0.022, 0.045, 0.076], *3 = [0.066, 0.135, 0.228]
        assert!((pi[0] - 0.066).abs() < 0.001);
        assert!((pi[1] - 0.135).abs() < 0.001);
        assert!((pi[2] - 0.228).abs() < 0.001);
    }

    /// [SYNTHETIC] Posterior returns: no views → posterior ≈ prior.
    #[test]
    fn test_posterior_no_views() {
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.04, 0.09, 0.16]));
        let w_eq = Array1::from_vec(vec![0.3, 0.3, 0.4]);
        let pi = BlackLitterman::implied_returns(3.0, &cov, &w_eq);

        // No views: P = 0, omega⁻¹ = 0
        let pick = Array2::<f64>::zeros((1, 3));
        let views = Array1::<f64>::zeros(1);
        let omega_inv = Array2::<f64>::zeros((1, 1));

        let posterior = BlackLitterman::posterior_returns(
            &pi, 0.05, &cov, &pick, &views, &omega_inv,
        )
        .unwrap();

        // With no views, posterior should approximately equal prior
        for (a, b) in pi.iter().zip(posterior.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }
}
