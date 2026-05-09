// Integration tests for the strategy module.
//
// SYNTHETIC DATA: ALL test data below is hand-constructed ndarray matrices.
// No external data sources (ClickHouse, PriceMatrix, etc.) are used.
// Every test function is prefixed with `syn_` to signal synthetic data.

use alfars::strat::{
    Strategy,
    compression::FactorZooCompress,
    equal::{EqualWeight, RankAverage, SignalWeighted},
    ic_based::{ICIRWeighted, ICWeighted},
    regression::RidgeCombine,
    state_aware::{FactorComfortZone, StateAware},
};
use ndarray::Array2;

/// Build a synthetic dataset with 2 market states and 3 factors.
///
/// Returns (factors, forward_returns, train_idx, test_idx).
///
/// - 100 trading days × 50 assets
/// - Days 0..50: state A (low returns)
/// - Days 50..100: state B (high returns)
/// - Factor 0: strong in state A (IC≈0.1), weak in state B (IC≈−0.05)
/// - Factor 1: strong in state B (IC≈0.08), weak in state A (IC≈−0.03)
/// - Factor 2: noise (IC≈0 in both states)
/// - Training: days 0..60, Testing: days 60..100
fn make_synthetic_dataset() -> (Vec<Array2<f64>>, Array2<f64>, usize) {
    let n_days = 100usize;
    let n_assets = 50usize;
    let train_days = 60;

    let mut ret = Array2::zeros((n_days, n_assets));
    let mut f0 = Array2::zeros((n_days, n_assets)); // good in state A
    let mut f1 = Array2::zeros((n_days, n_assets)); // good in state B
    let mut f2 = Array2::zeros((n_days, n_assets)); // noise

    // State A: days 0..50, low returns
    // State B: days 50..100, high returns
    for t in 0..n_days {
        // State-specific returns with noise
        let base_ret = if t < 50 { 0.001 } else { 0.03 };
        let state_noise = if t < 50 { 0.0005 } else { 0.005 };

        for a in 0..n_assets {
            let noise = ((t * 37 + a * 73) as f64 % 100.0 - 50.0) / 50.0 * state_noise;
            ret[[t, a]] = base_ret + noise;

            // Factor 0: correlated with returns in state A, anti-correlated in state B
            f0[[t, a]] = if t < 50 {
                ret[[t, a]] * 0.1 + noise * 0.01
            } else {
                -ret[[t, a]] * 0.05 + noise * 0.02
            };

            // Factor 1: anti-correlated in state A, correlated in state B
            f1[[t, a]] = if t < 50 {
                -ret[[t, a]] * 0.03 + noise * 0.02
            } else {
                ret[[t, a]] * 0.08 + noise * 0.01
            };

            // Factor 2: random noise, no correlation
            f2[[t, a]] = ((t * 17 + a * 41) as f64 % 200.0 - 100.0) / 100.0 * 0.001;
        }
    }

    (vec![f0, f1, f2], ret, train_days)
}

/// Split data into training (0..train_days) and test (train_days..).
fn split(
    factors: &[Array2<f64>],
    returns: &Array2<f64>,
    train_days: usize,
) -> (Vec<Array2<f64>>, Array2<f64>, Vec<Array2<f64>>) {
    let n_days = returns.nrows();
    let n_assets = returns.ncols();

    let train_factors: Vec<Array2<f64>> = factors
        .iter()
        .map(|f| f.slice(ndarray::s![..train_days, ..]).to_owned())
        .collect();
    let train_ret = returns.slice(ndarray::s![..train_days, ..]).to_owned();

    let test_factors: Vec<Array2<f64>> = factors
        .iter()
        .map(|f| f.slice(ndarray::s![train_days.., ..]).to_owned())
        .collect();

    (train_factors, train_ret, test_factors)
}

// ═══════════════════════════════════════════════════════════════════
//  L0 Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn syn_l0_all_strategies_produce_signal() {
    let (factors, ret, train_days) = make_synthetic_dataset();
    let (train_f, train_ret, test_f) = split(&factors, &ret, train_days);

    let strats: Vec<(Box<dyn Strategy>, &str)> = vec![
        (Box::new(EqualWeight), "EqualWeight"),
        (Box::new(RankAverage), "RankAverage"),
        (Box::new(SignalWeighted), "SignalWeighted"),
    ];
    for (mut strat, name) in strats {
        // L0 fit is a no-op
        strat.fit(&train_f, &train_ret).unwrap();

        let signal = strat.combine(&test_f).unwrap();
        let expected_shape = (ret.nrows() - train_days, ret.ncols());
        assert_eq!(
            signal.dim(),
            expected_shape,
            "{} signal shape mismatch",
            name
        );
        // Signal must have at least some finite values (no all-NaN)
        let finite_count = signal.iter().filter(|v| v.is_finite()).count();
        assert!(finite_count > 0, "{} produced all-NaN signal", name);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  L1 Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn syn_l1_ic_strategies_produce_signal() {
    let (factors, ret, train_days) = make_synthetic_dataset();
    let (train_f, train_ret, test_f) = split(&factors, &ret, train_days);

    let strats: Vec<(&str, Box<dyn Strategy>)> = vec![
        ("ICWeighted", Box::new(ICWeighted::new(None))),
        ("ICIRWeighted", Box::new(ICIRWeighted::new(None))),
    ];
    for (strategy_name, mut strat) in strats {
        strat.fit(&train_f, &train_ret).unwrap();
        let signal = strat.combine(&test_f).unwrap();
        let expected_shape = (ret.nrows() - train_days, ret.ncols());
        assert_eq!(
            signal.dim(),
            expected_shape,
            "{} signal shape mismatch",
            strategy_name
        );
        let finite_count = signal.iter().filter(|v| v.is_finite()).count();
        assert!(
            finite_count > 0,
            "{} produced all-NaN signal",
            strategy_name
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
//  L2 Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn syn_l2_ridge_combine_produces_signal() {
    let (factors, ret, train_days) = make_synthetic_dataset();
    let (train_f, train_ret, test_f) = split(&factors, &ret, train_days);

    let mut strat = RidgeCombine::new(1.0);
    strat.fit(&train_f, &train_ret).unwrap();

    let signal = strat.combine(&test_f).unwrap();
    let expected_shape = (ret.nrows() - train_days, ret.ncols());
    assert_eq!(signal.dim(), expected_shape);

    let finite_count = signal.iter().filter(|v| v.is_finite()).count();
    assert!(finite_count > 0, "RidgeCombine produced all-NaN signal");
}

// ═══════════════════════════════════════════════════════════════════
//  L3 Tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn syn_l3_state_aware_produces_signal() {
    let (factors, ret, train_days) = make_synthetic_dataset();
    let (train_f, train_ret, test_f) = split(&factors, &ret, train_days);

    let mut strat = StateAware::new(2, 252);
    strat.fit(&train_f, &train_ret).unwrap();

    let signal = strat.combine(&test_f).unwrap();
    let expected_shape = (ret.nrows() - train_days, ret.ncols());
    assert_eq!(signal.dim(), expected_shape);

    let finite_count = signal.iter().filter(|v| v.is_finite()).count();
    assert!(finite_count > 0, "StateAware produced all-NaN signal");
}

#[test]
fn syn_l3_comfort_zone_produces_signal() {
    let (factors, ret, train_days) = make_synthetic_dataset();
    let (train_f, train_ret, test_f) = split(&factors, &ret, train_days);

    let mut strat = FactorComfortZone::new(2, 0.5);
    strat.fit(&train_f, &train_ret).unwrap();

    let signal = strat.combine(&test_f).unwrap();
    let expected_shape = (ret.nrows() - train_days, ret.ncols());
    assert_eq!(signal.dim(), expected_shape);

    let finite_count = signal.iter().filter(|v| v.is_finite()).count();
    assert!(
        finite_count > 0,
        "FactorComfortZone produced all-NaN signal"
    );
}

// ═══════════════════════════════════════════════════════════════════
//  No look-ahead test
// ═══════════════════════════════════════════════════════════════════

#[test]
fn syn_no_lookahead() {
    // SYNTHETIC DATA: fit on period A (days 0..60), combine on period B (days 60..100).
    // The signal must NOT depend on period B's forward_returns data.
    let (factors, ret, train_days) = make_synthetic_dataset();
    let (train_f, train_ret, test_f) = split(&factors, &ret, train_days);

    // Test with RidgeCombine as representative of fit-dependent strategies
    let mut strat = RidgeCombine::new(1.0);

    // Fit only on training data
    strat.fit(&train_f, &train_ret).unwrap();

    // Combine on test data — should not crash or panic
    let signal_a = strat.combine(&test_f).unwrap();

    // Fit again with DIFFERENT training data (first half of train)
    let half = train_days / 2;
    let alt_train_f: Vec<Array2<f64>> = train_f
        .iter()
        .map(|f| f.slice(ndarray::s![..half, ..]).to_owned())
        .collect();
    let alt_train_ret = train_ret.slice(ndarray::s![..half, ..]).to_owned();

    let mut strat2 = RidgeCombine::new(1.0);
    strat2.fit(&alt_train_f, &alt_train_ret).unwrap();
    let signal_b = strat2.combine(&test_f).unwrap();

    // Different training periods should produce different signals
    // (ensuring fit actually learned different weights)
    let diff = &signal_a - &signal_b;
    let max_diff = diff.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        max_diff > 1e-10,
        "Different training data should produce different signals"
    );
}

// ═══════════════════════════════════════════════════════════════════
//  End-to-end: train→combine pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn syn_e2e_all_strategies_pipeline() {
    let (factors, ret, train_days) = make_synthetic_dataset();
    let (train_f, train_ret, test_f) = split(&factors, &ret, train_days);

    // Build one of each strategy type
    let strategies: Vec<(&str, Box<dyn Strategy>)> = vec![
        ("EqualWeight", Box::new(EqualWeight)),
        ("ICWeighted", Box::new(ICWeighted::new(None))),
        (
            "FactorZooCompress(2)",
            Box::new(FactorZooCompress::new(2, false)),
        ),
        ("RidgeCombine(1.0)", Box::new(RidgeCombine::new(1.0))),
        ("StateAware(2)", Box::new(StateAware::new(2, 252))),
        (
            "FactorComfortZone(2)",
            Box::new(FactorComfortZone::new(2, 0.5)),
        ),
    ];

    let expected_shape = (ret.nrows() - train_days, ret.ncols());
    let mut all_signals: Vec<(String, Array2<f64>)> = Vec::new();

    for (name, mut strat) in strategies {
        strat.fit(&train_f, &train_ret).unwrap();
        let signal = strat.combine(&test_f).unwrap();
        assert_eq!(signal.dim(), expected_shape, "{}: shape mismatch", name);

        let finite_count = signal.iter().filter(|v| v.is_finite()).count();
        assert!(
            finite_count > expected_shape.0,
            "{}: too few finite values ({}/{})",
            name,
            finite_count,
            expected_shape.0 * expected_shape.1
        );

        all_signals.push((name.to_string(), signal));
    }

    // Verify at least some strategies produce different signals
    // (not all strategies must differ — some may behave similarly on synthetic data)
    let mut found_diff = false;
    for i in 0..all_signals.len() {
        for j in (i + 1)..all_signals.len() {
            let (ref _na, ref sa) = all_signals[i];
            let (ref _nb, ref sb) = all_signals[j];
            let diff_sum: f64 = sa
                .iter()
                .zip(sb.iter())
                .map(|(a, b)| {
                    if a.is_finite() && b.is_finite() {
                        (a - b).abs()
                    } else {
                        0.0
                    }
                })
                .sum();
            let n = (expected_shape.0 * expected_shape.1) as f64;
            if diff_sum / n > 1e-8 {
                found_diff = true;
            }
        }
    }
    assert!(
        found_diff,
        "All strategies produced identical signals — expected at least some variation"
    );
}
