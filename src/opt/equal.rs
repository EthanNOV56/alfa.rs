//! Simple closed-form optimizers (no solver required).
//!
//! These methods produce weights directly from signal or volatility data
//! without numerical optimization.

use ndarray::{Array1, Array2};

use super::{CostModel, Optimizer, OptimizerConstraints};

// ── EqualWeight ──────────────────────────────────────────────────────────────

/// Weight = 1/K for the top-K assets by signal.
pub struct EqualWeight {
    pub top_k: usize,
}

impl EqualWeight {
    pub fn new(top_k: usize) -> Self {
        Self { top_k }
    }
}

impl Optimizer for EqualWeight {
    fn optimize_day(
        &self,
        signal: &Array1<f64>,
        _covariance: &Array2<f64>,
        _prev_weights: Option<&Array1<f64>>,
        constraints: &OptimizerConstraints,
        _cost_model: Option<&dyn CostModel>,
    ) -> Result<Array1<f64>, String> {
        let n = signal.len();
        let mut weights = Array1::<f64>::zeros(n);

        // Collect valid signals
        let mut indexed: Vec<(usize, f64)> = signal
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, v)| (i, *v))
            .collect();

        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_k = self.top_k.min(indexed.len());
        let wt = if constraints.market_neutral {
            // For market-neutral: long top K/2, short bottom K/2
            let half = top_k / 2;
            for &(i, _) in &indexed[..half] {
                weights[i] = 1.0 / half as f64;
            }
            for &(i, _) in &indexed[indexed.len() - half..] {
                weights[i] = -1.0 / half as f64;
            }
            return Ok(weights);
        } else {
            1.0 / top_k.max(1) as f64
        };

        for &(i, _) in &indexed[..top_k] {
            weights[i] = wt;
        }

        if constraints.long_only {
            for w in weights.iter_mut() {
                *w = w.max(0.0);
            }
        }

        Ok(weights)
    }
}

// ── SignalProportional ───────────────────────────────────────────────────────

/// Weight proportional to signal value (positive signals only).
pub struct SignalProportional;

impl Optimizer for SignalProportional {
    fn optimize_day(
        &self,
        signal: &Array1<f64>,
        _covariance: &Array2<f64>,
        _prev_weights: Option<&Array1<f64>>,
        constraints: &OptimizerConstraints,
        _cost_model: Option<&dyn CostModel>,
    ) -> Result<Array1<f64>, String> {
        let mut weights = Array1::<f64>::zeros(signal.len());

        let positive_sum: f64 = signal.iter().filter(|&&v| v > 0.0 && v.is_finite()).sum();

        if positive_sum <= 0.0 {
            return Err("No positive signals for SignalProportional".into());
        }

        for (w, &s) in weights.iter_mut().zip(signal.iter()) {
            if s > 0.0 && s.is_finite() {
                *w = s / positive_sum;
            }
        }

        if constraints.long_only {
            for w in weights.iter_mut() {
                *w = w.max(0.0);
            }
        }

        Ok(weights)
    }
}

// ── VolatilityInverse ────────────────────────────────────────────────────────

/// Weight proportional to inverse volatility: w_i ∝ 1/σ_i.
///
/// Volatility is derived from the diagonal of the covariance matrix.
pub struct VolatilityInverse;

impl Optimizer for VolatilityInverse {
    fn optimize_day(
        &self,
        _signal: &Array1<f64>,
        covariance: &Array2<f64>,
        _prev_weights: Option<&Array1<f64>>,
        constraints: &OptimizerConstraints,
        _cost_model: Option<&dyn CostModel>,
    ) -> Result<Array1<f64>, String> {
        let n = covariance.nrows();
        let mut weights = Array1::<f64>::zeros(n);

        let vol_sum: f64 = covariance
            .diag()
            .iter()
            .filter(|&&v| v > 0.0 && v.is_finite())
            .map(|v| 1.0 / v.sqrt())
            .sum();

        if vol_sum <= 0.0 {
            return Err("All assets have zero or negative variance".into());
        }

        for (i, w) in weights.iter_mut().enumerate() {
            let var = covariance[[i, i]];
            if var > 0.0 && var.is_finite() {
                *w = (1.0 / var.sqrt()) / vol_sum;
            }
        }

        if constraints.long_only {
            for w in weights.iter_mut() {
                *w = w.max(0.0);
            }
        }

        Ok(weights)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// [SYNTHETIC] EqualWeight: top-1 assigns 100%.
    #[test]
    fn test_equal_weight_top1() {
        let opt = EqualWeight::new(1);
        let signal = Array1::from_vec(vec![0.5, 0.3, 0.2]);
        let cov = Array2::eye(3);
        let c = OptimizerConstraints::default();
        let w = opt.optimize_day(&signal, &cov, None, &c, None).unwrap();
        assert!(((w.sum()) - (1.0)).abs() < 1e-10);
        assert!(w[0] > 0.9);
    }

    /// [SYNTHETIC] SignalProportional: positive signals only.
    #[test]
    fn test_signal_proportional() {
        let opt = SignalProportional;
        let signal = Array1::from_vec(vec![0.6, 0.4, -0.1]);
        let cov = Array2::eye(3);
        let c = OptimizerConstraints::default();
        let w = opt.optimize_day(&signal, &cov, None, &c, None).unwrap();
        assert!(((w.sum()) - (1.0)).abs() < 1e-10);
        assert!(w[2] < 1e-10);
        assert!(w[0] > w[1]);
    }

    /// [SYNTHETIC] VolatilityInverse: higher vol → lower weight.
    #[test]
    fn test_volatility_inverse() {
        let opt = VolatilityInverse;
        let signal = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let cov = Array2::from_diag(&Array1::from_vec(vec![0.01, 0.04, 0.09]));
        let c = OptimizerConstraints::default();
        let w = opt.optimize_day(&signal, &cov, None, &c, None).unwrap();
        assert!(((w.sum()) - (1.0)).abs() < 1e-10);
        // Asset 0 has lowest vol (0.1), should get highest weight
        assert!(w[0] > w[1]);
        assert!(w[1] > w[2]);
    }
}
