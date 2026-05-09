//! Ridge regression solver (ndarray → nalgebra → ndarray).

use ndarray::{Array1, Array2};
use crate::strategy::Result;

/// Solve ridge regression: minimize ||Xw - y||² + α||w||².
///
/// Uses normal equations with Cholesky decomposition:
///   w = (XᵀX + αI)⁻¹ Xᵀy
pub(crate) fn ridge_solve(
    x: &Array2<f64>,
    y: &Array1<f64>,
    alpha: f64,
) -> Result<Array1<f64>> {
    let (n, p) = x.dim();
    if y.len() != n {
        return Err(format!("dim mismatch: X={:?}, y={}", x.dim(), y.len()));
    }
    if alpha < 0.0 {
        return Err("alpha must be non-negative".into());
    }
    let eps = 1e-10;

    // Build XᵀX (p × p) and Xᵀy (p × 1)
    let mut xtx = Array2::zeros((p, p));
    let mut xty = Array1::zeros(p);

    for i in 0..p {
        for j in i..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += x[[k, i]] * x[[k, j]];
            }
            xtx[[i, j]] = sum;
            xtx[[j, i]] = sum;
        }
    }
    for i in 0..p {
        let mut sum = 0.0;
        for k in 0..n {
            if y[k].is_finite() {
                sum += x[[k, i]] * y[k];
            }
        }
        xty[i] = sum;
    }

    // Add regularization: A = XᵀX + (α + ε) I
    for i in 0..p {
        xtx[[i, i]] += alpha + eps;
    }

    // Convert to nalgebra, solve via Cholesky
    let (xtx_slice, _) = xtx.into_raw_vec_and_offset();
    let a_na = nalgebra::DMatrix::from_vec(p, p, xtx_slice);
    let b_na = nalgebra::DVector::from_vec(xty.to_vec());

    let cholesky = nalgebra::linalg::Cholesky::new(a_na)
        .ok_or("Cholesky: matrix is not positive definite")?;
    let w_na = cholesky.solve(&b_na);

    Ok(Array1::from_vec(w_na.as_slice().to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    // SYNTHETIC DATA: small linear systems with known solutions.

    #[test]
    fn syn_ridge_perfect_fit() {
        // y = 2*x1 + 3*x2, no noise → should recover [2, 3] with alpha=0
        let x = arr2(&[[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]]);
        let y = arr1(&[2.0, 3.0, 7.0]); // 2*1+3*0=2, 2*0+3*1=3, 2*2+3*1=7
        let w = ridge_solve(&x, &y, 0.0).unwrap();
        assert!((w[0] - 2.0).abs() < 1e-6);
        assert!((w[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn syn_ridge_shrinkage() {
        // y = 5*x1, alpha=1 → w should be < 5
        let x = arr2(&[[1.0], [2.0]]);
        let y = arr1(&[5.0, 10.0]);
        let w0 = ridge_solve(&x, &y, 0.0).unwrap();
        let w1 = ridge_solve(&x, &y, 10.0).unwrap();
        assert!(w0[0] > w1[0].abs()); // regularization shrinks
    }

    #[test]
    fn syn_ridge_known_solution() {
        // X = I(2), y = [2, 3] → w = [2/(1+α), 3/(1+α)]
        let x = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let y = arr1(&[2.0, 3.0]);
        let alpha = 0.5;
        let w = ridge_solve(&x, &y, alpha).unwrap();
        assert!((w[0] - 2.0 / 1.5).abs() < 1e-6);
        assert!((w[1] - 3.0 / 1.5).abs() < 1e-6);
    }

    #[test]
    fn syn_ridge_negative_alpha() {
        let x = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let y = arr1(&[1.0, 2.0]);
        assert!(ridge_solve(&x, &y, -1.0).is_err());
    }

    // ── Corner cases ──

    #[test]
    fn syn_ridge_dimension_mismatch() {
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let y = arr1(&[1.0]); // wrong length
        assert!(ridge_solve(&x, &y, 1.0).is_err());
    }

    #[test]
    fn syn_ridge_single_feature() {
        // y = 3*x, alpha=1 → w = Xᵀy / (XᵀX + α + ε)
        let x = arr2(&[[1.0], [2.0], [3.0]]);
        let y = arr1(&[3.0, 6.0, 9.0]);
        let w = ridge_solve(&x, &y, 0.0).unwrap();
        assert!((w[0] - 3.0).abs() < 1e-6); // exact recovery
    }

    #[test]
    fn syn_ridge_underdetermined() {
        // Fewer samples than features (2 × 3) → still works with regularization
        let x = arr2(&[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]);
        let y = arr1(&[10.0, 16.0]);
        let w = ridge_solve(&x, &y, 1.0).unwrap();
        assert_eq!(w.len(), 3);
        assert!(w.iter().all(|v| v.is_finite()));
    }
}
