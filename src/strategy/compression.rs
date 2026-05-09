//! FactorZooCompress: PCA-based factor de-correlation.

use crate::strategy::{Result, Strategy};
use ndarray::Array2;

pub struct FactorZooCompress {
    n_components: usize,
    rotate: bool,
    projection: Option<Array2<f64>>,
    component_weights: Option<Vec<f64>>,
}

impl FactorZooCompress {
    pub fn new(n_components: usize, rotate: bool) -> Self {
        Self {
            n_components,
            rotate,
            projection: None,
            component_weights: None,
        }
    }
}

impl Strategy for FactorZooCompress {
    fn fit(&mut self, factors: &[Array2<f64>], _forward_returns: &Array2<f64>) -> Result<()> {
        crate::strategy::validate_fit_input(factors, _forward_returns)?;

        let n_assets = factors[0].ncols();
        let n_factors = factors.len();
        let n_days = factors[0].nrows();

        // Flatten: each row is (day, asset), each column is a factor
        let n_samples = n_days * n_assets;
        let mut x = Array2::zeros((n_samples, n_factors));
        for f_idx in 0..n_factors {
            let f = &factors[f_idx];
            for t in 0..n_days {
                for a in 0..n_assets {
                    x[[t * n_assets + a, f_idx]] = f[[t, a]];
                }
            }
        }

        // Drop rows with NaN
        let mask: Vec<bool> = (0..n_samples)
            .map(|i| x.row(i).iter().all(|v| v.is_finite()))
            .collect();
        let clean_rows: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, ok)| **ok)
            .map(|(i, _)| i)
            .collect();
        let n_clean = clean_rows.len();
        if n_clean < 2 {
            return Err("insufficient clean samples for PCA".into());
        }
        let mut x_clean = Array2::zeros((n_clean, n_factors));
        for (new_i, &old_i) in clean_rows.iter().enumerate() {
            x_clean.row_mut(new_i).assign(&x.row(old_i));
        }

        let k = self.n_components.min(n_factors);
        let (_, components, explained) = super::pca::pca_fit_transform(&x_clean, k)?;

        self.projection = Some(components.clone());
        self.component_weights = Some(explained);

        // Varimax rotation if requested
        if self.rotate {
            let rotated = varimax_rotate(&components, 50, 1e-6);
            self.projection = Some(rotated);
        }

        Ok(())
    }

    fn combine(&self, factors: &[Array2<f64>]) -> Result<Array2<f64>> {
        crate::strategy::validate_combine_input(factors)?;
        let proj = self
            .projection
            .as_ref()
            .ok_or("FactorZooCompress not fitted")?;
        let cw = self
            .component_weights
            .as_ref()
            .ok_or("FactorZooCompress not fitted")?;

        let n_assets = factors[0].ncols();
        let n_factors = factors.len();
        let n_days = factors[0].nrows();
        let n_samples = n_days * n_assets;
        let k = proj.ncols();

        // Flatten test factors
        let mut x_test = Array2::zeros((n_samples, n_factors));
        for f_idx in 0..n_factors {
            let f = &factors[f_idx];
            for t in 0..n_days {
                for a in 0..n_assets {
                    x_test[[t * n_assets + a, f_idx]] = f[[t, a]];
                }
            }
        }

        // Project to PC space
        let z = x_test.dot(proj); // (n_samples, k)

        // Weighted average of PCs
        let weights_sum: f64 = cw.iter().sum();
        let signal_flat: Vec<f64> = (0..n_samples)
            .map(|i| {
                let has_nan = x_test.row(i).iter().any(|v| !v.is_finite());
                if has_nan {
                    f64::NAN
                } else {
                    let mut s = 0.0;
                    for j in 0..k {
                        s += z[[i, j]] * cw[j] / weights_sum;
                    }
                    s
                }
            })
            .collect();

        Ok(Array2::from_shape_vec((n_days, n_assets), signal_flat).unwrap())
    }

    fn name(&self) -> &str {
        "FactorZooCompress"
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Varimax rotation (Kaiser, 1958)
// ═══════════════════════════════════════════════════════════════════

/// Apply varimax orthogonal rotation to the loadings matrix in-place.
///
/// Maximizes the sum of variances of squared loadings, producing a
/// sparser, more interpretable factor structure.
///
/// # Arguments
/// * `loadings` — (n_variables × n_components) PCA loadings.
/// * `max_iter` — maximum number of full sweep iterations.
/// * `tol` — convergence tolerance on the maximum rotation angle.
///
/// Returns the rotated loadings matrix.
fn varimax_rotate(
    loadings: &ndarray::Array2<f64>,
    max_iter: usize,
    tol: f64,
) -> ndarray::Array2<f64> {
    let (p, k) = loadings.dim();
    if k < 2 {
        return loadings.clone();
    }

    let mut l = loadings.clone();

    // Row-wise communalities for normalization
    let mut h = ndarray::Array1::zeros(p);
    for i in 0..p {
        let mut sum_sq = 0.0;
        for j in 0..k {
            sum_sq += l[[i, j]] * l[[i, j]];
        }
        h[i] = sum_sq.sqrt().max(1e-15);
    }

    for _sweep in 0..max_iter {
        let mut max_angle = 0.0_f64;

        for j in 0..(k - 1) {
            for m in (j + 1)..k {
                // Build u_i = x_i^2 - y_i^2  and  v_i = 2*x_i*y_i
                // where x_i = l[i,j]/h[i],  y_i = l[i,m]/h[i]
                let mut a = 0.0_f64; // Σ u_i
                let mut b = 0.0_f64; // Σ v_i
                let mut c = 0.0_f64; // Σ (u_i^2 - v_i^2)
                let mut d = 0.0_f64; // Σ (2*u_i*v_i)

                for i in 0..p {
                    let x = l[[i, j]] / h[i];
                    let y = l[[i, m]] / h[i];
                    let u = x * x - y * y;
                    let v = 2.0 * x * y;
                    a += u;
                    b += v;
                    c += u * u - v * v;
                    d += 2.0 * u * v;
                }

                let p_f = p as f64;
                let num = d - 2.0 * a * b / p_f;
                let den = c - (a * a - b * b) / p_f;

                let phi = if den.abs() > 1e-15 {
                    0.25 * (num / den).atan()
                } else {
                    // If denominator is (near) zero, use π/4 = 45°
                    std::f64::consts::FRAC_PI_4
                };

                if phi.abs() > max_angle {
                    max_angle = phi.abs();
                }

                let cos = phi.cos();
                let sin = phi.sin();

                // Rotate columns j and m
                for i in 0..p {
                    let lj = l[[i, j]];
                    let lm = l[[i, m]];
                    l[[i, j]] = lj * cos + lm * sin;
                    l[[i, m]] = -lj * sin + lm * cos;
                }
            }
        }

        if max_angle < tol {
            break;
        }
    }

    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    // SYNTHETIC DATA: highly correlated factors generated from known structure.

    #[test]
    fn syn_varimax_preserves_orthogonality() {
        // SYNTHETIC DATA: loadings from a known 2-component PCA
        let loadings = arr2(&[[0.8, 0.0], [0.6, 0.4], [0.0, 0.9], [0.3, 0.7]]);
        let rotated = varimax_rotate(&loadings, 50, 1e-6);
        assert_eq!(rotated.dim(), loadings.dim());
        // Columns should still be approximately orthogonal after rotation
        let n_rows = rotated.nrows() as f64;
        let mut dot = 0.0;
        for i in 0..rotated.nrows() {
            dot += rotated[[i, 0]] * rotated[[i, 1]];
        }
        // The dot product of orthogonal columns should be near 0 (scaled by n)
        assert!(
            (dot / n_rows).abs() < 0.3,
            "columns should be roughly orthogonal after varimax, got dot/n={}",
            dot / n_rows
        );
    }

    #[test]
    fn syn_compress_reduces_dim() {
        // 3 factors, compress to 1 component
        let c = FactorZooCompress::new(1, false);
        let factors = vec![
            arr2(&[[1.0, 2.0], [3.0, 4.0]]),
            arr2(&[[1.1, 2.1], [3.1, 4.1]]),
            arr2(&[[0.9, 1.9], [2.9, 3.9]]),
        ];
        let ret = arr2(&[[0.01, 0.02], [0.03, 0.04]]);
        let mut s = FactorZooCompress::new(1, false);
        s.fit(&factors, &ret).unwrap();
        let signal = s.combine(&factors).unwrap();
        assert_eq!(signal.dim(), (2, 2));
    }

    #[test]
    fn syn_compress_shape_consistent() {
        // 5 factors × 3 days × 4 assets
        let mut factors = Vec::new();
        for _ in 0..5 {
            factors.push(Array2::zeros((3, 4)));
        }
        let ret = Array2::zeros((3, 4));
        let mut s = FactorZooCompress::new(3, false);
        s.fit(&factors, &ret).unwrap();
        let signal = s.combine(&factors).unwrap();
        assert_eq!(signal.dim(), (3, 4));
    }

    // ── Corner cases ──

    #[test]
    fn syn_compress_not_fitted_errors() {
        let f = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let s = FactorZooCompress::new(1, false);
        assert!(s.combine(&[f]).is_err());
    }

    #[test]
    fn syn_compress_single_factor() {
        // PCA with 1 factor → 1 component (identity-like)
        let factors = vec![arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])];
        let ret = arr2(&[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]);
        let mut s = FactorZooCompress::new(1, false);
        s.fit(&factors, &ret).unwrap();
        let signal = s.combine(&factors).unwrap();
        assert_eq!(signal.dim(), (2, 3));
        assert!(signal.iter().any(|v| v.is_finite()));
    }

    #[test]
    fn syn_compress_all_nan_row() {
        let mut f1 = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mut f2 = arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        f1[[1, 0]] = f64::NAN; // one cell NaN in second row
        f2[[1, 0]] = f64::NAN;
        let ret = arr2(&[[0.01, 0.02], [0.03, 0.04]]);
        let mut s = FactorZooCompress::new(2, false);
        s.fit(&[f1.clone(), f2.clone()], &ret).unwrap();
        let signal = s.combine(&[f1, f2]).unwrap();
        assert_eq!(signal.dim(), (2, 2));
    }
}
