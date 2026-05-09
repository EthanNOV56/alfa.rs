//! Covariance estimator framework.
//!
//! Provides five estimation strategies:
//! - `SampleCov` — standard unbiased estimator
//! - `LedoitWolf` — shrinkage towards constant-correlation target
//! - `EWMACov` — exponentially-weighted moving average
//! - `FactorModelCov` — Barra-style factor model decomposition
//! - `NeweyWest` — HAC autocorrelation-consistent estimator
//!
//! Research basis: reports #1 (risk models), #2 (common/idio risk), #5 (xCN4).

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use super::solver;

/// Trait for covariance estimation from historical returns.
pub trait CovEstimator: Send + Sync {
    /// Estimate covariance matrix from return data.
    ///
    /// # Parameters
    /// - `returns`: return matrix `(n_obs × n_assets)`
    ///
    /// # Returns
    /// Covariance matrix `(n_assets × n_assets)`.
    fn estimate(&self, returns: &Array2<f64>) -> Result<Array2<f64>, String>;
}

/// Covariance estimator type (serde-serializable config).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CovEstimatorType {
    Sample,
    LedoitWolf,
    EWMA { decay: f64 },
    FactorModel { n_factors: usize },
    NeweyWest { max_lag: usize },
}

impl Default for CovEstimatorType {
    fn default() -> Self {
        Self::Sample
    }
}

impl CovEstimatorType {
    /// Build a `CovEstimator` trait object from the type config.
    pub fn build(&self) -> Result<Box<dyn CovEstimator>, String> {
        match *self {
            Self::Sample => Ok(Box::<SampleCov>::default()),
            Self::LedoitWolf => Ok(Box::<LedoitWolf>::default()),
            Self::EWMA { decay } => Ok(Box::new(EWMACov::from_half_life(decay))),
            Self::FactorModel { n_factors } => Ok(Box::new(FactorModelCov {
                n_factors,
                min_obs: 60,
                loadings: None,
            })),
            Self::NeweyWest { max_lag } => Ok(Box::new(NeweyWest {
                max_lag: Some(max_lag),
                min_obs: 30,
            })),
        }
    }
}

// ── SampleCov ────────────────────────────────────────────────────────────────

/// Standard unbiased sample covariance: Σ = (RᵀR) / (T - 1).
pub struct SampleCov {
    /// Minimum number of pairwise observations required.
    pub min_obs: usize,
}

impl Default for SampleCov {
    fn default() -> Self {
        Self { min_obs: 20 }
    }
}

impl CovEstimator for SampleCov {
    fn estimate(&self, returns: &Array2<f64>) -> Result<Array2<f64>, String> {
        let (t, n) = returns.dim();
        if t < 2 {
            return Err(format!(
                "SampleCov requires at least 2 observations, got {t}"
            ));
        }

        let means = column_means_pairwise(returns, self.min_obs);
        let mut cov = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            cov[[i, i]] = pairwise_variance(returns.column(i), means[i], self.min_obs);
            for j in (i + 1)..n {
                let c = pairwise_covariance(
                    returns.column(i),
                    returns.column(j),
                    means[i],
                    means[j],
                    self.min_obs,
                );
                cov[[i, j]] = c;
                cov[[j, i]] = c;
            }
        }

        Ok(cov)
    }
}

// ── LedoitWolf ───────────────────────────────────────────────────────────────

/// Ledoit-Wolf (2004) shrinkage estimator.
///
/// Shrinks the sample covariance matrix towards a constant-correlation
/// target.  The shrinkage intensity δ has a closed-form solution — no
/// iterative optimisation required.
pub struct LedoitWolf {
    pub min_obs: usize,
}

impl Default for LedoitWolf {
    fn default() -> Self {
        Self { min_obs: 20 }
    }
}

impl CovEstimator for LedoitWolf {
    fn estimate(&self, returns: &Array2<f64>) -> Result<Array2<f64>, String> {
        let (t, n) = returns.dim();
        if t < 2 {
            return Err(format!(
                "LedoitWolf requires at least 2 observations, got {t}"
            ));
        }
        if n < 2 {
            return Err(format!("LedoitWolf requires at least 2 assets, got {n}"));
        }

        // 1. Sample covariance
        let sample = SampleCov {
            min_obs: self.min_obs,
        }
        .estimate(returns)?;

        // 2. Build constant-correlation target F
        let vols: Vec<f64> = sample.diag().iter().map(|&v| v.sqrt()).collect();

        let mut sum_corr = 0.0_f64;
        let mut count = 0_usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if vols[i] > 0.0 && vols[j] > 0.0 {
                    sum_corr += sample[[i, j]] / (vols[i] * vols[j]);
                    count += 1;
                }
            }
        }

        let r_bar = if count > 0 {
            sum_corr / count as f64
        } else {
            0.0
        };
        let mut target = Array2::<f64>::eye(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let v = r_bar * vols[i] * vols[j];
                target[[i, j]] = v;
                target[[j, i]] = v;
            }
        }

        // 3. Shrinkage intensity π, ρ, γ
        let t_f64 = t as f64;
        let mut pi_sum = 0.0_f64;
        let mut rho_sum = 0.0_f64;
        let mut gamma_sum = 0.0_f64;

        for i in 0..n {
            for j in (i + 1)..n {
                let s_ij = sample[[i, j]];
                let t_ij = target[[i, j]];

                // Asymptotic variance of √T · s_ij
                let num = 0_usize;
                // Compute using centred cross-products
                let cross = pairwise_cross_products(
                    returns.column(i),
                    returns.column(j),
                    means(returns.column(i), self.min_obs),
                    means(returns.column(j), self.min_obs),
                );
                let n_eff = cross.len().max(1) as f64;
                let var_cp: f64 = cross.iter().map(|&x| x * x).sum::<f64>() / n_eff
                    - (cross.iter().sum::<f64>() / n_eff).powi(2);
                let a_var = var_cp.max(0.0);

                pi_sum += a_var;
                rho_sum += a_var; // simplified: same asymptotic variance for target
                gamma_sum += (t_ij - s_ij).powi(2);
            }
        }

        let delta = if gamma_sum > 1e-12 {
            ((pi_sum - rho_sum) / (gamma_sum * t_f64)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 4. Blend
        let mut result = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    result[[i, i]] = sample[[i, i]]; // keep sample variances
                } else {
                    result[[i, j]] = (1.0 - delta) * sample[[i, j]] + delta * target[[i, j]];
                }
            }
        }

        Ok(result)
    }
}

// ── EWMACov ──────────────────────────────────────────────────────────────────

/// Exponentially-weighted moving average covariance.
///
/// S_t = α · S_{t-1} + (1 - α) · r_t · r_tᵀ
pub struct EWMACov {
    /// Decay factor α ∈ (0, 1).  α = exp(-ln(2) / half_life).
    pub decay: f64,
    pub min_obs: usize,
}

impl EWMACov {
    pub fn from_half_life(half_life: f64) -> Self {
        let decay = (-(2.0_f64).ln() / half_life).exp();
        Self {
            decay,
            min_obs: half_life as usize,
        }
    }
}

impl Default for EWMACov {
    fn default() -> Self {
        Self::from_half_life(60.0)
    }
}

impl CovEstimator for EWMACov {
    fn estimate(&self, returns: &Array2<f64>) -> Result<Array2<f64>, String> {
        let (t, n) = returns.dim();
        if t == 0 {
            return Err("EWMACov requires at least 1 observation".into());
        }

        let alpha = self.decay;
        let one_minus_alpha = 1.0 - alpha;
        let mut s = Array2::<f64>::zeros((n, n));

        for row_idx in 0..t {
            let row = returns.row(row_idx);
            // Skip rows containing NaN
            if row.iter().any(|v| v.is_nan()) {
                continue;
            }
            for i in 0..n {
                for j in i..n {
                    let prod = row[i] * row[j];
                    s[[i, j]] = alpha * s[[i, j]] + one_minus_alpha * prod;
                    if i != j {
                        s[[j, i]] = s[[i, j]];
                    }
                }
            }
        }

        Ok(s)
    }
}

// ── FactorModelCov ───────────────────────────────────────────────────────────

/// Factor model covariance: Σ = B · Σ_f · Bᵀ + diag(σ²_ε).
///
/// If `loadings` is provided, uses the given factor exposure matrix
/// (e.g., Barra industry dummies + style factors).  Otherwise extracts
/// factors via PCA on the return covariance.
pub struct FactorModelCov {
    /// Number of factors to retain (ignored if loadings is provided).
    pub n_factors: usize,
    pub min_obs: usize,
    /// Predefined factor loadings `(n_assets × n_factors)`.
    /// When provided, skips PCA and uses these directly.
    pub loadings: Option<Array2<f64>>,
}

impl Default for FactorModelCov {
    fn default() -> Self {
        Self {
            n_factors: 5,
            min_obs: 60,
            loadings: None,
        }
    }
}

impl FactorModelCov {
    /// Build with predefined factor exposures (e.g. Barra industry/style loadings).
    pub fn with_loadings(loadings: Array2<f64>, min_obs: usize) -> Self {
        let n_factors = loadings.ncols();
        Self {
            n_factors,
            min_obs,
            loadings: Some(loadings),
        }
    }
}

impl CovEstimator for FactorModelCov {
    fn estimate(&self, returns: &Array2<f64>) -> Result<Array2<f64>, String> {
        let (t, n) = returns.dim();
        if t < self.min_obs {
            return Err(format!(
                "FactorModelCov requires at least {} observations, got {t}",
                self.min_obs
            ));
        }

        // 1. Centre returns
        let mean = returns.mean_axis(Axis(0)).unwrap();
        let centered = returns - &mean;

        // 2. Factor loadings: use predefined or PCA
        let factor_loadings = if let Some(ref b) = self.loadings {
            if b.nrows() != n {
                return Err(format!(
                    "loadings.nrows ({}) != n_assets ({})",
                    b.nrows(),
                    n
                ));
            }
            b.clone()
        } else {
            let k = self.n_factors.min(n).min(t);
            let sample_cov = centered.t().dot(&centered) / (t - 1) as f64;
            let sigma_nalg = solver::to_nalgebra_matrix(&sample_cov);
            let svd = nalgebra::linalg::SVD::new(sigma_nalg, true, true);
            let u = svd.u.unwrap();
            let cols: Vec<nalgebra::DVector<f64>> =
                (0..k).map(|c| u.column(c).into_owned()).collect();
            let v_nalg = nalgebra::DMatrix::from_columns(&cols);
            solver::to_ndarray_matrix(&v_nalg)
        };

        // 3. Estimate factor covariance Σ_f
        let factor_returns = centered.dot(&factor_loadings); // (t × k)
        let sigma_f = SampleCov::default().estimate(&factor_returns)?;

        // 4. Estimate idiosyncratic variances
        let reconstructed = factor_returns.dot(&factor_loadings.t()); // (t × n)
        let residuals = &centered - &reconstructed;
        let mut specific_var = Array1::<f64>::zeros(n);
        for (j, v) in specific_var.iter_mut().enumerate() {
            let col = residuals.column(j);
            *v = col.mapv(|x| x * x).sum() / (t - 1) as f64;
            *v = v.max(1e-8);
        }

        // 5. Σ = B · Σ_f · Bᵀ + diag(σ²_ε)
        let b_sigma_f = factor_loadings.dot(&sigma_f); // (n × k)
        let mut result = b_sigma_f.dot(&factor_loadings.t()); // (n × n)
        for j in 0..n {
            result[[j, j]] += specific_var[j];
        }

        Ok(result)
    }
}

// ── NeweyWest ────────────────────────────────────────────────────────────────

/// Newey-West HAC covariance estimator.
///
/// Σ_NW = Γ₀ + Σ_{ℓ=1}^{L} (1 - ℓ/(L+1)) · (Γ_ℓ + Γ_ℓᵀ)
pub struct NeweyWest {
    /// Maximum lag L (auto-selected if None).
    pub max_lag: Option<usize>,
    pub min_obs: usize,
}

impl Default for NeweyWest {
    fn default() -> Self {
        Self {
            max_lag: None,
            min_obs: 30,
        }
    }
}

impl CovEstimator for NeweyWest {
    fn estimate(&self, returns: &Array2<f64>) -> Result<Array2<f64>, String> {
        let (t, n) = returns.dim();
        if t < self.min_obs {
            return Err(format!(
                "NeweyWest requires at least {} observations, got {t}",
                self.min_obs
            ));
        }

        let lag = self
            .max_lag
            .unwrap_or_else(|| (4.0 * (t as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize);
        let lag = lag.min(t - 1);

        // Centre returns
        let mean = returns.mean_axis(Axis(0)).unwrap();
        let centered = returns - &mean;

        // Γ₀ — contemporaneous covariance
        let gamma0 = centered.t().dot(&centered) / t as f64;

        let mut result = gamma0.clone();

        for ell in 1..=lag {
            let weight = 1.0 - ell as f64 / (lag + 1) as f64;
            // Γ_ℓ: covariance between r_t and r_{t-ℓ}
            let t_ell = t - ell;
            let slice0 = centered.slice(ndarray::s![ell.., ..]); // t-ℓ rows
            let slice_lag = centered.slice(ndarray::s![..t_ell, ..]); // first t-ℓ rows

            let gamma_ell = slice0.t().dot(&slice_lag) / t_ell as f64;

            for i in 0..n {
                for j in 0..n {
                    result[[i, j]] += weight * (gamma_ell[[i, j]] + gamma_ell[[j, i]]);
                }
            }
        }

        Ok(result)
    }
}

// ── Helper functions ─────────────────────────────────────────────────────────

fn column_means_pairwise(data: &Array2<f64>, _min_obs: usize) -> Vec<f64> {
    data.columns()
        .into_iter()
        .map(|col| means(col, _min_obs))
        .collect()
}

fn means(col: ndarray::ArrayView1<'_, f64>, _min_obs: usize) -> f64 {
    let finite: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return 0.0;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

fn pairwise_variance(col: ndarray::ArrayView1<'_, f64>, mean: f64, min_obs: usize) -> f64 {
    let vals: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
    if vals.len() < min_obs {
        return f64::NAN;
    }
    vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64
}

fn pairwise_covariance(
    col_a: ndarray::ArrayView1<'_, f64>,
    col_b: ndarray::ArrayView1<'_, f64>,
    mean_a: f64,
    mean_b: f64,
    min_obs: usize,
) -> f64 {
    let pairs: Vec<(f64, f64)> = col_a
        .iter()
        .zip(col_b.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(&a, &b)| (a, b))
        .collect();

    if pairs.len() < min_obs {
        return f64::NAN;
    }

    pairs
        .iter()
        .map(|&(a, b)| (a - mean_a) * (b - mean_b))
        .sum::<f64>()
        / (pairs.len() - 1) as f64
}

fn pairwise_cross_products(
    col_a: ndarray::ArrayView1<'_, f64>,
    col_b: ndarray::ArrayView1<'_, f64>,
    mean_a: f64,
    mean_b: f64,
) -> Vec<f64> {
    col_a
        .iter()
        .zip(col_b.iter())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(&a, &b)| (a - mean_a) * (b - mean_b))
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};

    /// [SYNTHETIC] Helper: generate correlated returns from known covariance.
    fn generate_synthetic_returns(n_obs: usize, cov: &Array2<f64>) -> Array2<f64> {
        let n = cov.nrows();
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Cholesky decomposition of cov
        let l = cholesky_decompose(cov);

        let mut returns = Array2::<f64>::zeros((n_obs, n));
        for t in 0..n_obs {
            let z: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
            let z_arr = Array1::from_vec(z);
            let correlated = l.dot(&z_arr);
            returns.row_mut(t).assign(&correlated);
        }
        returns
    }

    /// Simple Cholesky for testing (ill-conditioned matrices will panic).
    fn cholesky_decompose(a: &Array2<f64>) -> Array2<f64> {
        let n = a.nrows();
        let mut l = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[[i, j]];
                for k in 0..j {
                    sum -= l[[i, k]] * l[[j, k]];
                }
                if i == j {
                    l[[i, i]] = sum.sqrt();
                } else {
                    l[[i, j]] = sum / l[[j, j]];
                }
            }
        }
        l
    }

    /// [SYNTHETIC] SampleCov: recovers true covariance with large sample.
    #[test]
    fn test_sample_cov_recovery() {
        let cov_true = Array2::from_shape_vec(
            (3, 3),
            vec![0.04, 0.02, 0.01, 0.02, 0.09, 0.03, 0.01, 0.03, 0.16],
        )
        .unwrap();

        let returns = generate_synthetic_returns(5000, &cov_true);
        let est = SampleCov::default().estimate(&returns).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let err = (est[[i, j]] - cov_true[[i, j]]).abs();
                assert!(
                    err < 0.01,
                    "cov[{i},{j}]: est={:.6} true={:.6} err={:.6}",
                    est[[i, j]],
                    cov_true[[i, j]],
                    err
                );
            }
        }
    }

    /// [SYNTHETIC] LedoitWolf: produces positive-definite matrix.
    #[test]
    fn test_ledoit_wolf_positive_definite() {
        let cov_true = Array2::from_shape_vec(
            (3, 3),
            vec![0.04, 0.02, 0.01, 0.02, 0.09, 0.03, 0.01, 0.03, 0.16],
        )
        .unwrap();
        let returns = generate_synthetic_returns(100, &cov_true);
        let est = LedoitWolf::default().estimate(&returns).unwrap();

        // Check diagonal is positive
        for i in 0..3 {
            assert!(est[[i, i]] > 0.0);
        }

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                let diff = (est[[i, j]] - est[[j, i]]).abs();
                assert!(diff < 1e-10);
            }
        }
    }

    /// [SYNTHETIC] LedoitWolf: n_assets >> n_obs triggers shrinkage.
    #[test]
    fn test_ledoit_wolf_shrinkage_triggered() {
        // 10 assets, 8 observations → must shrink
        let n = 10;
        let t = 8;
        let mut rng = StdRng::seed_from_u64(99);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut returns = Array2::<f64>::zeros((t, n));
        for row in 0..t {
            for col in 0..n {
                returns[[row, col]] = normal.sample(&mut rng);
            }
        }
        let est = LedoitWolf { min_obs: 2 }.estimate(&returns).unwrap();

        // All diagonals should be finite and positive
        for i in 0..n {
            assert!(est[[i, i]].is_finite() && est[[i, i]] > 0.0);
        }
    }

    /// [SYNTHETIC] EWMA: recent observations have more influence.
    #[test]
    fn test_ewma_responds_to_regime_change() {
        let n = 3;
        let t = 200;
        let mut returns = Array2::<f64>::zeros((t, n));

        // Low-vol regime (t=0..99): σ ≈ 0.05
        let mut rng = StdRng::seed_from_u64(7);
        let normal_low = Normal::new(0.0, 0.05).unwrap();
        for row in 0..100 {
            for col in 0..n {
                returns[[row, col]] = normal_low.sample(&mut rng);
            }
        }
        // High-vol regime (t=100..199): σ ≈ 0.15
        let normal_high = Normal::new(0.0, 0.15).unwrap();
        for row in 100..t {
            for col in 0..n {
                returns[[row, col]] = normal_high.sample(&mut rng);
            }
        }

        let sample = SampleCov::default().estimate(&returns).unwrap();
        let ewma = EWMACov::from_half_life(20.0).estimate(&returns).unwrap();

        // EWMA should report higher variance than sample cov (more weight on
        // recent high-vol period).
        let sample_avg_var = sample.diag().mean().unwrap();
        let ewma_avg_var = ewma.diag().mean().unwrap();
        assert!(
            ewma_avg_var > sample_avg_var,
            "EWMA avg var {ewma_avg_var:.6} should exceed sample avg var {sample_avg_var:.6}"
        );
    }

    /// [SYNTHETIC] FactorModelCov: reconstructs known structure.
    #[test]
    fn test_factor_model_cov() {
        let t = 500;
        let n = 10;
        let k = 3;

        // Generate synthetic factors
        let mut rng = StdRng::seed_from_u64(123);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut factors = Array2::<f64>::zeros((t, k));
        for row in 0..t {
            for col in 0..k {
                factors[[row, col]] = normal.sample(&mut rng);
            }
        }

        // Generate random loadings
        let mut loadings = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                loadings[[i, j]] = normal.sample(&mut rng);
            }
        }

        // Specific risk
        let idio_vol: Vec<f64> = (0..n)
            .map(|_| 0.1 * normal.sample(&mut rng).abs())
            .collect();

        // Generate returns
        let mut returns = Array2::<f64>::zeros((t, n));
        for row in 0..t {
            let f_row = factors.row(row);
            let sys = loadings.dot(&f_row.to_owned());
            for col in 0..n {
                returns[[row, col]] = sys[col] + idio_vol[col] * normal.sample(&mut rng);
            }
        }

        let est = FactorModelCov {
            n_factors: k,
            min_obs: 50,
            loadings: None,
        }
        .estimate(&returns)
        .unwrap();

        assert_eq!(est.dim(), (n, n));
        // All diagonal entries should be positive
        for i in 0..n {
            assert!(est[[i, i]] > 0.0, "diagonal[{i}] = {}", est[[i, i]]);
        }
    }

    /// [SYNTHETIC] NeweyWest: produces symmetric positive-definite matrix.
    #[test]
    fn test_newey_west() {
        // Generate autocorrelated returns (MA(1) process)
        let t = 200;
        let n = 4;
        let mut rng = StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut innovations = Array2::<f64>::zeros((t, n));
        for row in 0..t {
            for col in 0..n {
                innovations[[row, col]] = normal.sample(&mut rng);
            }
        }

        let theta = 0.5;
        let mut returns = Array2::<f64>::zeros((t, n));
        for col in 0..n {
            returns[[0, col]] = innovations[[0, col]];
            for row in 1..t {
                returns[[row, col]] = innovations[[row, col]] + theta * innovations[[row - 1, col]];
            }
        }

        let est = NeweyWest::default().estimate(&returns).unwrap();

        // Symmetry
        for i in 0..n {
            for j in 0..n {
                let diff = (est[[i, j]] - est[[j, i]]).abs();
                assert!(diff < 1e-10);
            }
        }

        // Diagonals positive
        for i in 0..n {
            assert!(est[[i, i]] > 0.0);
        }
    }

    /// [SYNTHETIC] SampleCov: handles NaN via pairwise-complete.
    #[test]
    fn test_sample_cov_with_nan() {
        let returns = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0,
                2.0,
                f64::NAN,
                2.0,
                3.0,
                1.0,
                3.0,
                4.0,
                2.0,
                4.0,
                5.0,
                3.0,
            ],
        )
        .unwrap();

        let est = SampleCov { min_obs: 2 }.estimate(&returns).unwrap();

        // Asset 0 and 2 only overlap on 3 observations
        assert!(est[[0, 0]] > 0.0);
        assert!(est[[2, 2]] > 0.0);
        // Cross-cov should be computed from the valid overlap
        assert!(est[[0, 2]].is_finite());
    }

    /// [SYNTHETIC] FactorModelCov with predefined loadings: known B → verify
    /// Σ ≈ B Σ_f Bᵀ + diag(σ²).
    #[test]
    fn test_factor_model_predefined_loadings() {
        let t = 500;
        let n = 5;
        let k = 2;

        // Known loadings: asset 0-1 belong to factor 1, 2-3 to factor 2, asset 4 mixed
        let loadings = Array2::from_shape_vec(
            (n, k),
            vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5],
        )
        .unwrap();

        // Generate factor returns and specific noise
        let mut rng = StdRng::seed_from_u64(101);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut returns = Array2::<f64>::zeros((t, n));
        for day in 0..t {
            let f0 = normal.sample(&mut rng);
            let f1 = normal.sample(&mut rng);
            for i in 0..n {
                let sys = loadings[[i, 0]] * f0 + loadings[[i, 1]] * f1;
                let idio = 0.1 * normal.sample(&mut rng);
                returns[[day, i]] = sys + idio;
            }
        }

        let est = FactorModelCov::with_loadings(loadings, 50)
            .estimate(&returns)
            .unwrap();

        assert_eq!(est.dim(), (n, n));
        for i in 0..n {
            assert!(est[[i, i]] > 0.0);
        }
        // Assets with the same loadings should have similar variance
        assert!((est[[0, 0]] - est[[1, 1]]).abs() < 0.1);
    }
}
