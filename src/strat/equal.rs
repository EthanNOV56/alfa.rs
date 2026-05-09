//! L0 static combiners: EqualWeight, RankAverage, SignalWeighted.
//!
//! All L0 strategies have an empty `fit()` — they are pure mathematical
//! combiners that require no training data.

use crate::strat::{Result, Strategy, to_percentiles, zscore_rows};
use ndarray::Array2;

// ═══════════════════════════════════════════════════════════════════
//  EqualWeight
// ═══════════════════════════════════════════════════════════════════

/// Averages all factor values cell-wise, ignoring NaN.
pub struct EqualWeight;

impl Strategy for EqualWeight {
    fn fit(&mut self, _factors: &[Array2<f64>], _forward_returns: &Array2<f64>) -> Result<()> {
        Ok(())
    }

    fn combine(&self, factors: &[Array2<f64>]) -> Result<Array2<f64>> {
        crate::strat::validate_combine_input(factors)?;
        let (n_days, n_assets) = factors[0].dim();
        let mut signal = Array2::zeros((n_days, n_assets));
        for t in 0..n_days {
            for a in 0..n_assets {
                let mut sum = 0.0;
                let mut count = 0u32;
                for f in factors {
                    let v = f[[t, a]];
                    if v.is_finite() {
                        sum += v;
                        count += 1;
                    }
                }
                signal[[t, a]] = if count > 0 {
                    sum / count as f64
                } else {
                    f64::NAN
                };
            }
        }
        Ok(signal)
    }

    fn name(&self) -> &str {
        "EqualWeight"
    }
}

// ═══════════════════════════════════════════════════════════════════
//  RankAverage
// ═══════════════════════════════════════════════════════════════════

/// Converts each factor to cross-sectional percentiles, then averages.
pub struct RankAverage;

impl Strategy for RankAverage {
    fn fit(&mut self, _factors: &[Array2<f64>], _forward_returns: &Array2<f64>) -> Result<()> {
        Ok(())
    }

    fn combine(&self, factors: &[Array2<f64>]) -> Result<Array2<f64>> {
        crate::strat::validate_combine_input(factors)?;
        let (n_days, n_assets) = factors[0].dim();
        let n_factors = factors.len();

        // Convert each factor to percentiles
        let pct_mats: Vec<Array2<f64>> = factors.iter().map(|f| to_percentiles(f)).collect();

        // Average and center
        let mut signal = Array2::zeros((n_days, n_assets));
        for t in 0..n_days {
            for a in 0..n_assets {
                let mut sum = 0.0;
                let mut count = 0u32;
                for p in &pct_mats {
                    let v = p[[t, a]];
                    if v.is_finite() {
                        sum += v;
                        count += 1;
                    }
                }
                signal[[t, a]] = if count > 0 {
                    sum / count as f64 - 0.5
                } else {
                    f64::NAN
                };
            }
        }
        Ok(signal)
    }

    fn name(&self) -> &str {
        "RankAverage"
    }
}

// ═══════════════════════════════════════════════════════════════════
//  SignalWeighted
// ═══════════════════════════════════════════════════════════════════

/// Self-weighted combination: stronger signals get more influence.
pub struct SignalWeighted;

impl Strategy for SignalWeighted {
    fn fit(&mut self, _factors: &[Array2<f64>], _forward_returns: &Array2<f64>) -> Result<()> {
        Ok(())
    }

    fn combine(&self, factors: &[Array2<f64>]) -> Result<Array2<f64>> {
        crate::strat::validate_combine_input(factors)?;
        let (n_days, n_assets) = factors[0].dim();

        // Z-score each factor cross-sectionally
        let z_mats: Vec<Array2<f64>> = factors.iter().map(|f| zscore_rows(f)).collect();

        let mut signal = Array2::zeros((n_days, n_assets));
        for t in 0..n_days {
            for a in 0..n_assets {
                let mut num = 0.0;
                let mut den = 0.0;
                for z in &z_mats {
                    let v = z[[t, a]];
                    if v.is_finite() {
                        let w = v.abs();
                        num += v * w;
                        den += w;
                    }
                }
                signal[[t, a]] = if den > 0.0 { num / den } else { f64::NAN };
            }
        }
        Ok(signal)
    }

    fn name(&self) -> &str {
        "SignalWeighted"
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    // SYNTHETIC DATA: all test matrices are hand-constructed ndarray arrays.

    #[test]
    fn syn_equal_identical_matrices() {
        let f1 = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let f2 = f1.clone();
        let f3 = f1.clone();
        let s = EqualWeight;
        let signal = s.combine(&[f1, f2, f3]).unwrap();
        assert!((signal[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((signal[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((signal[[1, 0]] - 3.0).abs() < 1e-10);
        assert!((signal[[1, 1]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn syn_equal_with_nan() {
        let mut f1 = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let f2 = arr2(&[[10.0, 20.0], [30.0, 40.0]]);
        f1[[0, 1]] = f64::NAN; // cell with NaN → should only use f2
        let s = EqualWeight;
        let signal = s.combine(&[f1, f2]).unwrap();
        assert!((signal[[0, 0]] - 5.5).abs() < 1e-10); // (1 + 10) / 2
        assert!((signal[[0, 1]] - 20.0).abs() < 1e-10); // only f2
    }

    #[test]
    fn syn_equal_all_nan() {
        let mut f1 = arr2(&[[1.0, f64::NAN], [3.0, 4.0]]);
        let mut f2 = arr2(&[[f64::NAN, f64::NAN], [30.0, 40.0]]);
        f1[[0, 0]] = f64::NAN;
        f2[[0, 1]] = f64::NAN;
        let s = EqualWeight;
        let signal = s.combine(&[f1, f2]).unwrap();
        assert!(signal[[0, 0]].is_nan()); // both NaN
        assert!(signal[[0, 1]].is_nan()); // both NaN
    }

    #[test]
    fn syn_rank_ordering() {
        let f = arr2(&[[1.0, 2.0, 3.0]]);
        let s = RankAverage;
        let signal = s.combine(&[f]).unwrap();
        // 1.0 → rank 0.0 → signal −0.5
        // 2.0 → rank 0.5 → signal 0.0
        // 3.0 → rank 1.0 → signal 0.5
        assert!((signal[[0, 0]] + 0.5).abs() < 1e-10);
        assert!(signal[[0, 1]].abs() < 1e-10);
        assert!((signal[[0, 2]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn syn_signal_weighted_single() {
        let f = arr2(&[[1.0, 2.0, 3.0]]);
        let s = SignalWeighted;
        let signal = s.combine(&[f]).unwrap();
        // Single factor with self-weighting:
        // z=[−1, 0, 1], w=[1, 0, 1],
        // signal[0] = −1*1/1 = −1, signal[1] = NaN (zero weight), signal[2] = 1
        assert!((signal[[0, 0]] + 1.0).abs() < 1e-10);
        assert!(signal[[0, 1]].is_nan());
        assert!((signal[[0, 2]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn syn_fit_noop() {
        let mut s = EqualWeight;
        let f = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let ret = f.clone();
        assert!(s.fit(&[f], &ret).is_ok());
    }

    // ── Corner cases ──

    #[test]
    fn syn_equal_single_factor() {
        let f = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let signal = EqualWeight.combine(&[f.clone()]).unwrap();
        // Single factor → signal equals factor itself
        assert!((signal[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((signal[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((signal[[1, 0]] - 3.0).abs() < 1e-10);
        assert!((signal[[1, 1]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn syn_equal_single_asset() {
        let f = arr2(&[[1.0], [3.0]]);
        let signal = EqualWeight.combine(&[f]).unwrap();
        assert_eq!(signal.dim(), (2, 1));
        assert!((signal[[0, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn syn_equal_all_nan_column() {
        let mut f1 = arr2(&[[1.0, f64::NAN], [3.0, f64::NAN]]);
        let mut f2 = arr2(&[[5.0, f64::NAN], [7.0, f64::NAN]]);
        // Column 1 is all NaN → should remain NaN in output
        let signal = EqualWeight.combine(&[f1, f2]).unwrap();
        assert!((signal[[0, 0]] - 3.0).abs() < 1e-10); // (1+5)/2
        assert!(signal[[0, 1]].is_nan());
        assert!((signal[[1, 0]] - 5.0).abs() < 1e-10); // (3+7)/2
        assert!(signal[[1, 1]].is_nan());
    }

    #[test]
    fn syn_rank_single_day() {
        let f = arr2(&[[5.0, 1.0, 3.0]]);
        let signal = RankAverage.combine(&[f]).unwrap();
        assert_eq!(signal.dim(), (1, 3));
        // percentiles: 1→0, 3→0.5, 5→1; minus 0.5 → −0.5, 0, 0.5
        assert!((signal[[0, 0]] - 0.5).abs() < 1e-10); // 5.0
        assert!((signal[[0, 1]] + 0.5).abs() < 1e-10); // 1.0
        assert!(signal[[0, 2]].abs() < 1e-10); // 3.0
    }

    #[test]
    fn syn_empty_factors_combine() {
        assert!(EqualWeight.combine(&[]).is_err());
        assert!(RankAverage.combine(&[]).is_err());
        assert!(SignalWeighted.combine(&[]).is_err());
    }
}
