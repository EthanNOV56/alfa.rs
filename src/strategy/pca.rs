//! PCA dimensionality reduction (ndarray → nalgebra → ndarray).

use crate::strategy::Result;
use ndarray::Array2;

pub(crate) fn covariance_matrix(x: &Array2<f64>) -> Array2<f64> {
    let (n, p) = x.dim();
    let mean = x.mean_axis(ndarray::Axis(0)).unwrap();
    let mut cov = Array2::zeros((p, p));
    for k in 0..n {
        let row = x.row(k).to_owned() - &mean;
        for i in 0..p {
            for j in i..p {
                cov[[i, j]] += row[i] * row[j];
            }
        }
    }
    for i in 0..p {
        for j in 0..i {
            cov[[i, j]] = cov[[j, i]];
        }
    }
    cov / (n as f64 - 1.0)
}

pub(crate) fn pca_fit_transform(
    x: &Array2<f64>,
    n_components: usize,
) -> Result<(Array2<f64>, Array2<f64>, Vec<f64>)> {
    let p = x.ncols();
    let k = n_components.min(p);
    if k == 0 {
        return Err("n_components must be >= 1".into());
    }

    let cov = covariance_matrix(x);

    // Convert to nalgebra for eigenvalue decomposition
    let (cov_slice, _) = cov.clone().into_raw_vec_and_offset();
    let cov_na = nalgebra::DMatrix::from_vec(p, p, cov_slice);
    let eigen = nalgebra::linalg::SymmetricEigen::new(cov_na);

    // Extract eigenvalues and eigenvectors, sort descending
    let eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;

    let mut pairs: Vec<(f64, Vec<f64>)> = (0..p)
        .map(|i| {
            let ev = eigenvalues[i];
            let vec: Vec<f64> = eigenvectors.column(i).iter().copied().collect();
            (ev, vec)
        })
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

    let components_flat: Vec<f64> = pairs.iter().take(k).flat_map(|(_, v)| v.clone()).collect();
    let components = Array2::from_shape_vec((p, k), components_flat).unwrap();

    let explained_var_ratio: Vec<f64> = pairs.iter().take(k).map(|(v, _)| v.abs()).collect();
    let total_var: f64 = pairs.iter().map(|(v, _)| v.abs()).sum();
    let explained_var_ratio: Vec<f64> = explained_var_ratio
        .iter()
        .map(|&v| {
            if total_var > 1e-12 {
                v / total_var
            } else {
                0.0
            }
        })
        .collect();

    let centered = x - &x.mean_axis(ndarray::Axis(0)).unwrap();
    let transformed = centered.dot(&components);

    Ok((transformed, components, explained_var_ratio))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    // SYNTHETIC DATA: small hand-constructed matrices with known eigendecomposition.

    #[test]
    fn syn_cov_identity() {
        // [1,0; 0,1]: mean=[0.5,0.5], centered=[[0.5,-0.5],[-0.5,0.5]]
        // cov = [[0.5, -0.5], [-0.5, 0.5]] / (2-1) = [[0.5, -0.5], [-0.5, 0.5]]
        let x = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let cov = covariance_matrix(&x);
        assert!((cov[[0, 0]] - 0.5).abs() < 1e-10);
        assert!((cov[[1, 1]] - 0.5).abs() < 1e-10);
        assert!((cov[[0, 1]] + 0.5).abs() < 1e-10);
    }

    #[test]
    fn syn_pca_reduces_dim() {
        // 5 samples, 3 features → reduce to 2
        let data = arr2(&[
            [1.0, 2.0, 0.5],
            [2.0, 3.0, 1.0],
            [3.0, 1.0, 1.5],
            [4.0, 4.0, 2.0],
            [5.0, 5.0, 3.0],
        ]);
        let (transformed, components, ratio) = pca_fit_transform(&data, 2).unwrap();
        assert_eq!(transformed.ncols(), 2);
        assert_eq!(transformed.nrows(), 5);
        assert_eq!(components.dim(), (3, 2));
        assert_eq!(ratio.len(), 2);
        // Ratios should sum to approximately 1 (allow small tolerance due to FP)
        let sum: f64 = ratio.iter().sum();
        assert!((sum - 1.0).abs() < 1e-2, "sum={}", sum);
    }

    #[test]
    fn syn_pca_variance_preserved() {
        // k = n_features → all variance preserved
        let data = arr2(&[[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]]);
        let (transformed, _, ratio) = pca_fit_transform(&data, 2).unwrap();
        assert_eq!(transformed.ncols(), 2);
        assert_eq!(ratio.len(), 2);
        let sum: f64 = ratio.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // ── Corner cases ──

    #[test]
    fn syn_pca_n_components_zero() {
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        assert!(pca_fit_transform(&data, 0).is_err());
    }

    #[test]
    fn syn_pca_clamp_n_components() {
        let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        // n_components=10 > n_features=2 → clamped to 2
        let (transformed, components, ratio) = pca_fit_transform(&data, 10).unwrap();
        assert_eq!(transformed.ncols(), 2);
        assert_eq!(components.ncols(), 2);
        assert_eq!(ratio.len(), 2);
    }

    #[test]
    fn syn_pca_single_feature() {
        let data = arr2(&[[1.0], [2.0], [3.0]]);
        let (transformed, components, ratio) = pca_fit_transform(&data, 1).unwrap();
        assert_eq!(transformed.dim(), (3, 1));
        assert_eq!(components.dim(), (1, 1));
        assert_eq!(ratio.len(), 1);
    }
}
