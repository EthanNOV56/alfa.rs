//! Signal → Alpha mapping pipeline.
//!
//! Transforms raw strategy signals into expected return estimates (α)
//! before they are fed into the portfolio optimizer.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// A single signal → alpha transformation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlphaMapping {
    /// Linear scaling: α = signal × scale
    Linear { scale: f64 },
    /// Rank mapping: α = rank(signal) / (n - 1) (more robust than raw)
    Rank { normalize: bool },
    /// Quantile mapping: α = quantile(signal, N)
    Quantile { n_groups: usize },
    /// Z-Score: α = (signal - μ) / σ, with optional winsorization
    ZScore { winsorize_sigma: Option<f64> },
    /// Box-Cox transformation
    BoxCox { lambda: f64 },
}

impl AlphaMapping {
    fn apply(&self, signal: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Linear { scale } => signal.mapv(|x| x * scale),
            Self::Rank { normalize } => Self::apply_rank(signal, *normalize),
            Self::Quantile { n_groups } => Self::apply_quantile(signal, *n_groups),
            Self::ZScore { winsorize_sigma } => Self::apply_zscore(signal, *winsorize_sigma),
            Self::BoxCox { lambda } => Self::apply_boxcox(signal, *lambda),
        }
    }

    fn apply_rank(signal: &Array1<f64>, normalize: bool) -> Array1<f64> {
        let n = signal.len();
        if n == 0 {
            return Array1::zeros(0);
        }

        // Collect (index, value) pairs, skipping NaN
        let mut indexed: Vec<(usize, f64)> = signal
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, v)| (i, *v))
            .collect();

        indexed.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut result = Array1::from_elem(n, f64::NAN);
        let m = indexed.len();
        if m == 0 {
            return result;
        }

        // Handle ties by assigning average rank
        let mut i = 0;
        while i < m {
            let mut j = i + 1;
            while j < m && indexed[j].1 == indexed[i].1 {
                j += 1;
            }
            let avg_rank = (i + j - 1) as f64 / 2.0;
            let rank_val = if normalize && m > 1 {
                avg_rank / (m - 1) as f64
            } else {
                avg_rank
            };
            for k in i..j {
                result[indexed[k].0] = rank_val;
            }
            i = j;
        }

        result
    }

    fn apply_quantile(signal: &Array1<f64>, n_groups: usize) -> Array1<f64> {
        if n_groups == 0 {
            return Array1::zeros(signal.len());
        }
        let ranked = Self::apply_rank(signal, true);
        ranked.mapv(|r| {
            if r.is_nan() {
                f64::NAN
            } else {
                (r * n_groups as f64).floor() / n_groups.max(1) as f64
            }
        })
    }

    fn apply_zscore(signal: &Array1<f64>, winsorize_sigma: Option<f64>) -> Array1<f64> {
        let valid: Vec<f64> = signal.iter().copied().filter(|v| v.is_finite()).collect();
        if valid.len() < 2 {
            return Array1::from_elem(signal.len(), 0.0);
        }

        let n = valid.len() as f64;
        let mean = valid.iter().sum::<f64>() / n;
        let var = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();

        if std < 1e-12 {
            return Array1::from_elem(signal.len(), 0.0);
        }

        let mut result = signal.mapv(|x| {
            if x.is_finite() {
                (x - mean) / std
            } else {
                0.0
            }
        });

        if let Some(sigma) = winsorize_sigma {
            result.mapv_inplace(|x| x.clamp(-sigma, sigma));
        }

        result
    }

    fn apply_boxcox(signal: &Array1<f64>, lambda: f64) -> Array1<f64> {
        if (lambda - 0.0).abs() < 1e-10 {
            // λ ≈ 0 → natural log
            signal.mapv(|x| {
                if x.is_finite() && x > 0.0 {
                    x.ln()
                } else {
                    f64::NAN
                }
            })
        } else {
            signal.mapv(|x| {
                if x.is_finite() && x > 0.0 {
                    (x.powf(lambda) - 1.0) / lambda
                } else {
                    f64::NAN
                }
            })
        }
    }
}

/// Ordered sequence of alpha transformations.
///
/// Each step's output feeds into the next step's input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaPipeline {
    pub steps: Vec<AlphaMapping>,
}

impl Default for AlphaPipeline {
    fn default() -> Self {
        Self {
            steps: vec![AlphaMapping::Linear { scale: 1.0 }],
        }
    }
}

impl AlphaPipeline {
    /// Apply the full pipeline: raw signal → alpha vector.
    pub fn transform(&self, signal: &Array1<f64>) -> Array1<f64> {
        let mut current = signal.clone();
        for step in &self.steps {
            current = step.apply(&current);
        }
        current
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// [SYNTHETIC] Linear: scale applies correctly.
    #[test]
    fn test_linear() {
        let s = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mapping = AlphaMapping::Linear { scale: 2.0 };
        let a = mapping.apply(&s);
        assert_eq!(a.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    /// [SYNTHETIC] Rank: normalized rank in [0, 1].
    #[test]
    fn test_rank_normalized() {
        let s = Array1::from_vec(vec![1.0, 5.0, 3.0]);
        let mapping = AlphaMapping::Rank { normalize: true };
        let a = mapping.apply(&s);
        assert!(((a[0]) - (0.0)).abs() < 1e-10); // smallest
        assert!(((a[1]) - (1.0)).abs() < 1e-10); // largest
        let mid = a[2]; // middle
        assert!(mid > 0.0 && mid < 1.0);
    }

    /// [SYNTHETIC] Rank with ties: tied values share average rank.
    #[test]
    fn test_rank_ties_normalized() {
        let s = Array1::from_vec(vec![3.0, 1.0, 3.0]);
        let mapping = AlphaMapping::Rank { normalize: true };
        let a = mapping.apply(&s);
        // Indices 0 and 2 are tied at 3.0, index 1 is smallest
        assert!(((a[0]) - (a[2])).abs() < 1e-10);
        assert!(((a[1]) - (0.0)).abs() < 1e-10);
    }

    /// [SYNTHETIC] ZScore: mean ≈ 0, std ≈ 1.
    #[test]
    fn test_zscore() {
        let s = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mapping = AlphaMapping::ZScore {
            winsorize_sigma: None,
        };
        let a = mapping.apply(&s);
        let mean = a.iter().sum::<f64>() / a.len() as f64;
        let var = a.iter().map(|v| v.powi(2)).sum::<f64>() / a.len() as f64;
        assert!(((mean) - (0.0)).abs() < 1e-10);
        assert!(((var) - (1.0)).abs() < 1e-10);
    }

    /// [SYNTHETIC] ZScore with winsorize: no value exceeds ±sigma.
    #[test]
    fn test_zscore_winsorize() {
        let s = Array1::from_vec(vec![0.1, 0.2, 0.3, 10.0, -5.0]);
        let mapping = AlphaMapping::ZScore {
            winsorize_sigma: Some(2.0),
        };
        let a = mapping.apply(&s);
        assert!(a.iter().all(|&x| x >= -2.0 && x <= 2.0));
    }

    /// [SYNTHETIC] BoxCox λ=0: equivalent to ln.
    #[test]
    fn test_boxcox_log() {
        let s = Array1::from_vec(vec![1.0, 2.0, std::f64::consts::E]);
        let mapping = AlphaMapping::BoxCox { lambda: 0.0 };
        let a = mapping.apply(&s);
        assert!(((a[0]) - (0.0)).abs() < 1e-10);
        assert!(((a[2]) - (1.0)).abs() < 1e-10);
    }

    /// [SYNTHETIC] BoxCox λ=1: identity (shifted).
    #[test]
    fn test_boxcox_identity() {
        let s = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let mapping = AlphaMapping::BoxCox { lambda: 1.0 };
        let a = mapping.apply(&s);
        for (expected, actual) in s.iter().zip(a.iter()) {
            assert!(((actual) - (expected - 1.0)).abs() < 1e-10);
        }
    }

    /// [SYNTHETIC] Quantile: values binned to n_groups levels.
    #[test]
    fn test_quantile() {
        let n = 100;
        let s = Array1::from_vec((0..n).map(|i| i as f64).collect());
        let mapping = AlphaMapping::Quantile { n_groups: 4 };
        let a = mapping.apply(&s);
        // With n_groups=4, ranks range across [0, 1], ceil gives up to n_groups+1=5
        // unique scaled values (0, 1, 2, 3, 4 when multiplied by n_groups).
        let unique: std::collections::BTreeSet<usize> =
            a.iter().filter(|x| x.is_finite()).map(|x| (x * 4.0).round() as usize).collect();
        assert!(unique.len() <= 5);
        assert!(a.iter().all(|&x| x.is_nan() || (x >= 0.0 && x <= 1.0)));
    }

    /// [SYNTHETIC] Pipeline: two steps compose.
    #[test]
    fn test_pipeline_compose() {
        let pipeline = AlphaPipeline {
            steps: vec![
                AlphaMapping::Rank { normalize: true },
                AlphaMapping::ZScore {
                    winsorize_sigma: None,
                },
            ],
        };
        let s = Array1::from_vec((0..20).map(|i| i as f64).collect());
        let a = pipeline.transform(&s);
        let mean = a.iter().sum::<f64>() / a.len() as f64;
        assert!(((mean) - (0.0)).abs() < 1e-10);
    }
}
