//! L2 RidgeCombine: supervised factor combination via ridge regression.

use ndarray::{Array1, Array2};
use crate::strategy::{Result, Strategy};

pub struct RidgeCombine {
    alpha: f64,
    weights: Option<Array1<f64>>,
}

impl RidgeCombine {
    pub fn new(alpha: f64) -> Self {
        Self { alpha, weights: None }
    }
}

impl Strategy for RidgeCombine {
    fn fit(
        &mut self,
        factors: &[Array2<f64>],
        forward_returns: &Array2<f64>,
    ) -> Result<()> {
        crate::strategy::validate_fit_input(factors, forward_returns)?;

        let n_factors = factors.len();
        let (n_days, n_assets) = factors[0].dim();

        // Flatten: each row is (day, asset)
        let n_samples = n_days * n_assets;
        let mut x = Array2::zeros((n_samples, n_factors));
        let mut y = Array1::zeros(n_samples);

        for f_idx in 0..n_factors {
            let f = &factors[f_idx];
            for t in 0..n_days {
                for a in 0..n_assets {
                    x[[t * n_assets + a, f_idx]] = f[[t, a]];
                }
            }
        }
        for t in 0..n_days {
            for a in 0..n_assets {
                y[t * n_assets + a] = forward_returns[[t, a]];
            }
        }

        // Drop rows with NaN in X or y
        let mask: Vec<bool> = (0..n_samples)
            .map(|i| {
                x.row(i).iter().all(|v| v.is_finite()) && y[i].is_finite()
            })
            .collect();
        let clean_idx: Vec<usize> = mask.iter().enumerate()
            .filter(|(_, ok)| **ok)
            .map(|(i, _)| i)
            .collect();

        if clean_idx.len() < n_factors {
            return Err("insufficient clean samples for ridge regression".into());
        }

        let mut x_clean = Array2::zeros((clean_idx.len(), n_factors));
        let mut y_clean = Array1::zeros(clean_idx.len());
        for (new_i, &old_i) in clean_idx.iter().enumerate() {
            x_clean.row_mut(new_i).assign(&x.row(old_i));
            y_clean[new_i] = y[old_i];
        }

        let w = super::ridge::ridge_solve(&x_clean, &y_clean, self.alpha)?;
        self.weights = Some(w);
        Ok(())
    }

    fn combine(&self, factors: &[Array2<f64>]) -> Result<Array2<f64>> {
        crate::strategy::validate_combine_input(factors)?;
        let w = self.weights.as_ref().ok_or("RidgeCombine not fitted")?;

        let n_factors = factors.len();
        let (n_days, n_assets) = factors[0].dim();

        if w.len() != n_factors {
            return Err(format!(
                "weight dimension mismatch: {} vs {} factors",
                w.len(),
                n_factors
            ));
        }

        let mut signal = Array2::zeros((n_days, n_assets));

        for t in 0..n_days {
            for a in 0..n_assets {
                let mut s = 0.0;
                let mut all_finite = true;
                for f_idx in 0..n_factors {
                    let v = factors[f_idx][[t, a]];
                    if v.is_finite() {
                        s += v * w[f_idx];
                    } else {
                        all_finite = false;
                        break;
                    }
                }
                signal[[t, a]] = if all_finite { s } else { f64::NAN };
            }
        }

        Ok(signal)
    }

    fn name(&self) -> &str {
        "RidgeCombine"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    // SYNTHETIC DATA: factors constructed from known linear combination of returns.

    #[test]
    fn syn_ridge_combine_recover_weights() {
        // Create two factors: f1 = 2*ret + noise, f2 = 3*ret with noise
        // RidgeCombine should give more weight to f2
        let ret = arr2(&[
            [0.01, -0.02, 0.03],
            [-0.01, 0.02, 0.01],
            [0.02, 0.01, -0.01],
            [0.00, 0.03, 0.02],
        ]);
        let f1 = &ret * 2.0;
        let f2 = &ret * 3.0;

        let mut s = RidgeCombine::new(0.1);
        s.fit(&[f1.clone(), f2.clone()], &ret).unwrap();
        let signal = s.combine(&[f1, f2]).unwrap();
        assert_eq!(signal.dim(), ret.dim());
    }

    #[test]
    fn syn_ridge_combine_nan_handling() {
        let ret = arr2(&[[0.01, 0.02], [0.03, 0.04]]);
        let mut f1 = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let f2 = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        f1[[0, 0]] = f64::NAN; // insert NaN

        let mut s = RidgeCombine::new(1.0);
        s.fit(&[f1.clone(), f2.clone()], &ret).unwrap();
        let signal = s.combine(&[f1, f2]).unwrap();
        assert!(signal[[0, 0]].is_nan());
        assert!(signal[[0, 1]].is_finite());
    }

    // ── Corner cases ──

    #[test]
    fn syn_ridge_combine_not_fitted_errors() {
        let f = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let s = RidgeCombine::new(1.0);
        assert!(s.combine(&[f]).is_err());
    }

    #[test]
    fn syn_ridge_combine_few_assets() {
        // RidgeCombine works with small dimensions
        let ret = arr2(&[[0.01, -0.01], [0.02, -0.02], [0.03, -0.03], [0.01, -0.01]]);
        let f = &ret * 0.5;
        let mut s = RidgeCombine::new(1.0);
        s.fit(&[f.clone()], &ret).unwrap();
        let signal = s.combine(&[f]).unwrap();
        assert_eq!(signal.dim(), ret.dim());
    }

    #[test]
    fn syn_ridge_combine_zero_alpha() {
        let ret = arr2(&[[0.01, -0.01], [0.02, -0.02], [0.03, -0.03], [0.01, -0.01]]);
        let f = &ret * 0.5;
        let mut s = RidgeCombine::new(0.0); // no regularization
        s.fit(&[f.clone()], &ret).unwrap();
        let signal = s.combine(&[f]).unwrap();
        assert_eq!(signal.dim(), ret.dim());
    }
}
