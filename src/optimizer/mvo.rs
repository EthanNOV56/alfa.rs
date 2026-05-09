//! Mean-variance optimisation implementations.

use ndarray::{Array1, Array2};

use super::solver;
use super::{CostModel, Optimizer, OptimizerConstraints};

// ── MaxSharpe ────────────────────────────────────────────────────────────────

/// Maximise Sharpe ratio: max wᵀα / √(wᵀΣw).
pub struct MaxSharpe {
    pub risk_aversion: f64,
}

impl Default for MaxSharpe {
    fn default() -> Self {
        Self { risk_aversion: 3.0 }
    }
}

impl Optimizer for MaxSharpe {
    fn optimize_day(
        &self,
        signal: &Array1<f64>,
        covariance: &Array2<f64>,
        prev_weights: Option<&Array1<f64>>,
        constraints: &OptimizerConstraints,
        cost_model: Option<&dyn CostModel>,
    ) -> Result<Array1<f64>, String> {
        // Compute turnover penalty from cost_model if present
        let turnover_penalty = cost_model.map(|cm| {
            // Use linear cost rate as rough penalty coefficient
            0.001_f64
        });

        solver::solve_mv_alm(
            covariance,
            signal,
            self.risk_aversion,
            constraints.long_only,
            constraints.full_investment,
            constraints.market_neutral,
            constraints.max_position,
            constraints.leverage_limit,
            constraints.max_assets,
            &constraints.group_constraints,
            &constraints.factor_exposure_constraints,
            prev_weights,
            turnover_penalty.or(constraints.turnover_limit),
        )
    }
}

// ── MinVariance ──────────────────────────────────────────────────────────────

/// Minimise portfolio variance: min wᵀΣw.
pub struct MinVariance;

impl Optimizer for MinVariance {
    fn optimize_day(
        &self,
        signal: &Array1<f64>,
        covariance: &Array2<f64>,
        _prev_weights: Option<&Array1<f64>>,
        constraints: &OptimizerConstraints,
        _cost_model: Option<&dyn CostModel>,
    ) -> Result<Array1<f64>, String> {
        // Min variance with a tiny alpha (flat signal) to use MV machinery
        let flat_alpha = Array1::<f64>::ones(signal.len()) * 1e-6;
        let risk_aversion = 1e6;

        if constraints.long_only {
            solver::solve_mv_long_only(covariance, &flat_alpha, risk_aversion, None, None)
        } else if constraints.full_investment
            && !constraints.market_neutral
            && constraints.max_position.is_none()
        {
            let (w, _) = solver::solve_mv_kkt(covariance, &flat_alpha, risk_aversion)?;
            Ok(w)
        } else {
            solver::solve_mv_unconstrained(covariance, &flat_alpha, risk_aversion)
        }
    }
}

// ── MaxIR ───────────────────────────────────────────────────────────────────

/// Maximise Information Ratio with factor exposure constraints.
///
/// Currently delegates to MaxSharpe — factor constraints are enforced
/// via the penalty term in the augmented Lagrangian path.
pub struct MaxIR {
    pub risk_aversion: f64,
}

impl Default for MaxIR {
    fn default() -> Self {
        Self { risk_aversion: 3.0 }
    }
}

impl Optimizer for MaxIR {
    fn optimize_day(
        &self,
        signal: &Array1<f64>,
        covariance: &Array2<f64>,
        prev_weights: Option<&Array1<f64>>,
        constraints: &OptimizerConstraints,
        cost_model: Option<&dyn CostModel>,
    ) -> Result<Array1<f64>, String> {
        // Same ALM path as MaxSharpe — factor exposure budget constraints
        // are enforced directly in the solver via budget penalty
        let turnover_penalty = cost_model.map(|_cm| 0.001_f64);

        solver::solve_mv_alm(
            covariance,
            signal,
            self.risk_aversion,
            constraints.long_only,
            constraints.full_investment,
            constraints.market_neutral,
            constraints.max_position,
            constraints.leverage_limit,
            constraints.max_assets,
            &constraints.group_constraints,
            &constraints.factor_exposure_constraints,
            prev_weights,
            turnover_penalty.or(constraints.turnover_limit),
        )
    }
}

// ── TurnoverAware ───────────────────────────────────────────────────────────

/// Mean-variance with turnover penalty.
///
/// Objective: `max wᵀα - (λ/2) wᵀΣw - γ · ‖w - w_prev‖₁`.
pub struct TurnoverAware {
    pub risk_aversion: f64,
    pub penalty_coefficient: f64,
}

impl Default for TurnoverAware {
    fn default() -> Self {
        Self {
            risk_aversion: 3.0,
            penalty_coefficient: 0.1,
        }
    }
}

impl Optimizer for TurnoverAware {
    fn optimize_day(
        &self,
        signal: &Array1<f64>,
        covariance: &Array2<f64>,
        prev_weights: Option<&Array1<f64>>,
        constraints: &OptimizerConstraints,
        _cost_model: Option<&dyn CostModel>,
    ) -> Result<Array1<f64>, String> {
        solver::solve_mv_long_only(
            covariance,
            signal,
            self.risk_aversion,
            prev_weights,
            Some(self.penalty_coefficient),
        )
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// [SYNTHETIC] MaxSharpe long-only: valid weights.
    #[test]
    fn test_max_sharpe_long_only() {
        let opt = MaxSharpe::default();
        let signal = Array1::from_vec(vec![0.1, 0.2, 0.05]);
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.04, 0.09, 0.16]));
        let constraints = OptimizerConstraints {
            long_only: true,
            full_investment: true,
            ..Default::default()
        };
        let w = opt
            .optimize_day(&signal, &cov, None, &constraints, None)
            .unwrap();
        assert!(w.iter().all(|&x| x >= -1e-12));
        assert!((w.sum() - 1.0).abs() < 1e-10);
    }

    /// [SYNTHETIC] MinVariance: low-vol assets get higher weight.
    #[test]
    fn test_min_variance() {
        let opt = MinVariance;
        let signal = Array1::from_vec(vec![0.0, 0.0, 0.0]); // flat signal
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.01, 0.04, 0.09]));
        let constraints = OptimizerConstraints {
            long_only: true,
            full_investment: true,
            ..Default::default()
        };
        let w = opt
            .optimize_day(&signal, &cov, None, &constraints, None)
            .unwrap();
        // Asset 0 has lowest variance, should get highest weight
        assert!(w[0] > w[1] && w[1] > w[2]);
    }

    /// [SYNTHETIC] TurnoverAware: penalty reduces weight change.
    #[test]
    fn test_turnover_aware() {
        let opt = TurnoverAware {
            risk_aversion: 3.0,
            penalty_coefficient: 10.0, // large penalty
        };
        let signal = Array1::from_vec(vec![0.5, 0.2, 0.3]);
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.04, 0.09, 0.16]));
        let prev = Array1::from_vec(vec![0.33, 0.33, 0.34]);
        let constraints = OptimizerConstraints {
            long_only: true,
            full_investment: true,
            ..Default::default()
        };
        let w = opt
            .optimize_day(&signal, &cov, Some(&prev), &constraints, None)
            .unwrap();
        // With large penalty, weights should stay close to prev
        let turnover = w
            .iter()
            .zip(prev.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>()
            / 2.0;
        assert!(turnover < 0.3);
    }
}
