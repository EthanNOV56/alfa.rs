//! L1 history-driven combiners: ICWeighted, ICIRWeighted.
//!
//! Both use historical IC (information coefficient) to determine
//! per-factor weights. IC is computed via `backtest::metrics::compute_ic_series`,
//! which returns daily cross-sectional Pearson r values.

use ndarray::Array2;
use crate::strategy::{compute_factor_ic, Result, Strategy, zscore_rows};

// ═══════════════════════════════════════════════════════════════════
//  ICWeighted
// ═══════════════════════════════════════════════════════════════════

pub struct ICWeighted {
    lookback_days: Option<usize>,
    weights: Vec<f64>,
    fallback_uniform: bool,
}

impl ICWeighted {
    pub fn new(lookback_days: Option<usize>) -> Self {
        Self {
            lookback_days,
            weights: Vec::new(),
            fallback_uniform: true,
        }
    }
}

impl Strategy for ICWeighted {
    fn fit(
        &mut self,
        factors: &[Array2<f64>],
        forward_returns: &Array2<f64>,
    ) -> Result<()> {
        crate::strategy::validate_fit_input(factors, forward_returns)?;
        let n = factors.len();

        let mut ic_means = Vec::with_capacity(n);
        for f in factors {
            let (_ic_series, ic_mean, _ic_ir) = compute_factor_ic(f, forward_returns)?;
            // Apply lookback window if specified
            let effective_mean = match self.lookback_days {
                Some(w) => {
                    let start = _ic_series.len().saturating_sub(w);
                    let win: Vec<f64> = _ic_series.as_slice().unwrap()[start..]
                        .iter().filter(|&&x| x.is_finite()).copied().collect();
                    if win.is_empty() { ic_mean } else { win.iter().sum::<f64>() / win.len() as f64 }
                }
                None => ic_mean,
            };
            ic_means.push(effective_mean);
        }

        // weight_i = max(0, ic_mean_i), normalized
        let sum: f64 = ic_means.iter().map(|&m| m.max(0.0)).sum();
        if sum > 1e-12 {
            self.weights = ic_means.iter().map(|&m| m.max(0.0) / sum).collect();
            self.fallback_uniform = false;
        } else {
            self.weights = vec![1.0 / n as f64; n];
            self.fallback_uniform = true;
        }
        Ok(())
    }

    fn combine(&self, factors: &[Array2<f64>]) -> Result<Array2<f64>> {
        crate::strategy::validate_combine_input(factors)?;
        let (n_days, n_assets) = factors[0].dim();
        let n_factors = factors.len();

        if self.weights.len() != n_factors {
            return Err("ICWeighted not fitted: weight count mismatch".into());
        }

        let z_mats: Vec<Array2<f64>> = factors.iter().map(|f| zscore_rows(f)).collect();
        let mut signal = Array2::zeros((n_days, n_assets));

        for t in 0..n_days {
            for a in 0..n_assets {
                let mut sum = 0.0;
                let mut den = 0.0;
                for (f_idx, z) in z_mats.iter().enumerate() {
                    let v = z[[t, a]];
                    if v.is_finite() {
                        sum += v * self.weights[f_idx];
                        den += self.weights[f_idx];
                    }
                }
                signal[[t, a]] = if den > 0.0 { sum / den } else { f64::NAN };
            }
        }
        Ok(signal)
    }

    fn name(&self) -> &str {
        "ICWeighted"
    }
}

// ═══════════════════════════════════════════════════════════════════
//  ICIRWeighted
// ═══════════════════════════════════════════════════════════════════

pub struct ICIRWeighted {
    lookback_days: Option<usize>,
    weights: Vec<f64>,
    fallback_uniform: bool,
}

impl ICIRWeighted {
    pub fn new(lookback_days: Option<usize>) -> Self {
        Self {
            lookback_days,
            weights: Vec::new(),
            fallback_uniform: true,
        }
    }
}

impl Strategy for ICIRWeighted {
    fn fit(
        &mut self,
        factors: &[Array2<f64>],
        forward_returns: &Array2<f64>,
    ) -> Result<()> {
        crate::strategy::validate_fit_input(factors, forward_returns)?;
        let n = factors.len();

        let mut ic_irs = Vec::with_capacity(n);
        for f in factors {
            let (_ic_series, _ic_mean, ic_ir) = compute_factor_ic(f, forward_returns)?;
            let effective_ir = match self.lookback_days {
                Some(w) => {
                    let start = _ic_series.len().saturating_sub(w);
                    let win: Vec<f64> = _ic_series.as_slice().unwrap()[start..]
                        .iter().filter(|&&x| x.is_finite()).copied().collect();
                    if win.len() >= 2 {
                        let m = win.iter().sum::<f64>() / win.len() as f64;
                        let v = win.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (win.len() - 1) as f64;
                        if v > 1e-15 { m / v.sqrt() } else { 0.0 }
                    } else { ic_ir }
                }
                None => ic_ir,
            };
            ic_irs.push(effective_ir);
        }

        let sum: f64 = ic_irs.iter().map(|&m| m.max(0.0)).sum();
        if sum > 1e-12 {
            self.weights = ic_irs.iter().map(|&m| m.max(0.0) / sum).collect();
            self.fallback_uniform = false;
        } else {
            self.weights = vec![1.0 / n as f64; n];
            self.fallback_uniform = true;
        }
        Ok(())
    }

    fn combine(&self, factors: &[Array2<f64>]) -> Result<Array2<f64>> {
        crate::strategy::validate_combine_input(factors)?;
        let (n_days, n_assets) = factors[0].dim();
        let n_factors = factors.len();

        if self.weights.len() != n_factors {
            return Err("ICIRWeighted not fitted: weight count mismatch".into());
        }

        let z_mats: Vec<Array2<f64>> = factors.iter().map(|f| zscore_rows(f)).collect();
        let mut signal = Array2::zeros((n_days, n_assets));

        for t in 0..n_days {
            for a in 0..n_assets {
                let mut sum = 0.0;
                let mut den = 0.0;
                for (f_idx, z) in z_mats.iter().enumerate() {
                    let v = z[[t, a]];
                    if v.is_finite() {
                        sum += v * self.weights[f_idx];
                        den += self.weights[f_idx];
                    }
                }
                signal[[t, a]] = if den > 0.0 { sum / den } else { f64::NAN };
            }
        }
        Ok(signal)
    }

    fn name(&self) -> &str {
        "ICIRWeighted"
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    // SYNTHETIC DATA: factor and returns matrices hand-constructed to
    // produce known IC values.

    #[test]
    fn syn_ic_positive_weight() {
        // Factor perfectly correlated with returns → positive IC → positive weight
        let ret = arr2(&[[0.01, -0.01], [0.02, -0.02], [0.03, -0.03]]);
        let f = ret.clone(); // IC = 1.0
        let mut s = ICWeighted::new(None);
        s.fit(&[f], &ret).unwrap();
        assert!(!s.weights.is_empty());
        assert!(s.weights[0] > 0.0);
    }

    #[test]
    fn syn_ic_negative_weight() {
        // Factor negatively correlated → zero weight
        let ret = arr2(&[[0.01, -0.01], [0.02, -0.02], [0.03, -0.03]]);
        let f = -&ret; // IC = −1.0
        let mut s = ICWeighted::new(None);
        s.fit(&[f], &ret).unwrap();
        // Fallback to uniform since all weights are zero
        assert!(s.fallback_uniform);
    }

    #[test]
    fn syn_icir_penalizes_noise() {
        // Two factors: one stable (high IR), one noisy (low IR)
        let ret = arr2(&[
            [0.01, -0.01, 0.0],
            [0.02, -0.02, 0.0],
            [0.03, -0.03, 0.0],
            [0.01, -0.01, 0.0],
            [0.02, -0.02, 0.0],
            [0.03, -0.03, 0.0],
        ]);
        let f_stable = &ret * 0.1;     // IC consistent
        let mut f_noisy = &ret * 0.1;
        f_noisy[[0, 0]] += 100.0; // inject huge outlier → IC noisy
        f_noisy[[3, 1]] -= 200.0;

        let mut s = ICIRWeighted::new(None);
        s.fit(&[f_stable.clone(), f_noisy], &ret).unwrap();
        // Stable factor should get higher IR weight
        assert!(s.weights[0] > s.weights[1]);
    }

    #[test]
    fn syn_lookback_window() {
        let ret = arr2(&[
            [0.01, -0.01],
            [0.02, -0.02],
            [0.03, -0.03],
            [0.01, -0.01],
            [0.02, -0.02],
        ]);
        let f = &ret * 0.1;
        let mut s = ICWeighted::new(Some(2)); // only last 2 days
        s.fit(&[f], &ret).unwrap();
        assert!(!s.weights.is_empty());
        assert!(s.weights[0] > 0.0);
    }

    #[test]
    fn syn_combine_returns_signal() {
        let ret = arr2(&[[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]);
        let f1 = &ret * 2.0;
        let f2 = &ret * 3.0;
        let mut s = ICWeighted::new(None);
        s.fit(&[f1.clone(), f2.clone()], &ret).unwrap();
        let signal = s.combine(&[f1, f2]).unwrap();
        assert_eq!(signal.dim(), ret.dim());
    }

    // ── Corner cases ──

    #[test]
    fn syn_ic_single_factor() {
        let ret = arr2(&[[0.01, -0.01], [0.02, -0.02], [0.03, -0.03]]);
        let f = ret.clone();
        let mut s = ICWeighted::new(None);
        s.fit(&[f.clone()], &ret).unwrap();
        let signal = s.combine(&[f]).unwrap();
        assert_eq!(signal.dim(), ret.dim());
        assert!(signal.iter().any(|v| v.is_finite()));
    }

    #[test]
    fn syn_ic_single_factor_few_assets() {
        // IC needs at least 2 assets per day + 3 days for valid IC series (2 IC values)
        let ret = arr2(&[[0.01, -0.01], [0.02, -0.02], [0.03, -0.03]]);
        let f = ret.clone();
        let mut s = ICWeighted::new(None);
        s.fit(&[f.clone()], &ret).unwrap();
        let signal = s.combine(&[f]).unwrap();
        assert_eq!(signal.dim(), ret.dim());
    }

    #[test]
    fn syn_icir_not_fitted_errors() {
        let f = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let s = ICIRWeighted::new(None);
        assert!(s.combine(&[f]).is_err());
    }

    #[test]
    fn syn_ic_zero_lookback() {
        // lookback_days=Some(0) should still work (saturate)
        let ret = arr2(&[[0.01, -0.01], [0.02, -0.02], [0.03, -0.03]]);
        let f = ret.clone();
        let mut s = ICWeighted::new(Some(0));
        s.fit(&[f.clone()], &ret).unwrap();
        let signal = s.combine(&[f]).unwrap();
        assert_eq!(signal.dim(), ret.dim());
    }
}
