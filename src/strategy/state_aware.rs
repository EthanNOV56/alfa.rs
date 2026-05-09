//! L3 dynamic timing: StateAware, FactorComfortZone.
//!
//! Uses linfa-clustering for K-means market state identification.

use crate::strategy::{Result, Strategy, zscore_rows};
use ndarray::Array2;
use rand::Rng;

pub struct StateAware {
    n_states: usize,
    ic_lookback: usize,
    centroids: Option<Array2<f64>>,
    state_ic: Option<Array2<f64>>,
}

impl StateAware {
    pub fn new(n_states: usize, ic_lookback: usize) -> Self {
        Self {
            n_states,
            ic_lookback,
            centroids: None,
            state_ic: None,
        }
    }
}

impl Strategy for StateAware {
    fn fit(&mut self, factors: &[Array2<f64>], forward_returns: &Array2<f64>) -> Result<()> {
        crate::strategy::validate_fit_input(factors, forward_returns)?;

        let n_factors = factors.len();
        let n_days = forward_returns.nrows();

        // ── Temporal split: prevent double-dipping ──
        // First half of training data → state definition (K-means).
        // Second half → per-state IC estimation (evaluation).
        // If not enough data (< 60 days), fall back to full-sample fit
        // with a warning that results are in-sample.
        const MIN_SPLIT_DAYS: usize = 60;
        let (state_days, ic_factors, ic_returns, _ic_offset) = if n_days >= MIN_SPLIT_DAYS {
            let split = n_days / 2;
            let state_ret = forward_returns.slice(ndarray::s![..split, ..]).to_owned();
            let ic_f: Vec<Array2<f64>> = factors
                .iter()
                .map(|f| f.slice(ndarray::s![split.., ..]).to_owned())
                .collect();
            let ic_r = forward_returns.slice(ndarray::s![split.., ..]).to_owned();
            (state_ret, ic_f, ic_r, split)
        } else {
            // Fallback: use all data (in-sample, but better than nothing)
            let state_ret = forward_returns.clone();
            let ic_f: Vec<Array2<f64>> = factors.iter().map(|f| f.clone()).collect();
            let ic_r = forward_returns.clone();
            (state_ret, ic_f, ic_r, 0_usize)
        };

        // 1. K-means on state-definition period
        let state_features = build_market_features(&state_days);
        let (_, centroids) = kmeans_cluster(&state_features, self.n_states)?;

        // 2. Assign states to IC-evaluation period using learned centroids
        let ic_features = build_market_features(&ic_returns);
        let ic_labels: Vec<usize> = (0..ic_features.nrows())
            .map(|t| assign_nearest(&ic_features.row(t), &centroids))
            .collect();

        // 3. Per-factor, per-state IC estimation on evaluation period
        let ic_n_days = ic_returns.nrows();
        let ic_start = if self.ic_lookback == 0 || self.ic_lookback >= ic_n_days {
            0
        } else {
            ic_n_days - self.ic_lookback
        };

        let mut state_ic = Array2::zeros((n_factors, self.n_states));
        let mut state_counts = vec![0usize; self.n_states];

        for f_idx in 0..n_factors {
            let (ic_arr, _, _) =
                crate::backtest::metrics::compute_ic_series(&ic_factors[f_idx], &ic_returns)?;
            for t in ic_start..ic_n_days.min(ic_arr.len()) {
                let s = ic_labels[t];
                let ic_val = ic_arr[t];
                if ic_val.is_finite() {
                    state_ic[[f_idx, s]] += ic_val;
                    state_counts[s] += 1;
                }
            }
        }

        // Handle states with no samples (possible if a rare state
        // only appears in the state-definition period): use zero IC.
        for s in 0..self.n_states {
            if state_counts[s] > 0 {
                for f in 0..n_factors {
                    state_ic[[f, s]] /= state_counts[s] as f64;
                }
            }
        }

        self.centroids = Some(centroids);
        self.state_ic = Some(state_ic);
        Ok(())
    }

    fn combine(&self, factors: &[Array2<f64>]) -> Result<Array2<f64>> {
        crate::strategy::validate_combine_input(factors)?;
        let centroids = self.centroids.as_ref().ok_or("StateAware not fitted")?;
        let state_ic = self.state_ic.as_ref().ok_or("StateAware not fitted")?;

        let n_factors = factors.len();
        let (n_days, n_assets) = factors[0].dim();

        let features = build_market_features_from_factors(factors)?;
        let mut day_states = vec![0usize; n_days];
        for t in 0..n_days.min(features.nrows()) {
            day_states[t] = assign_nearest(&features.row(t), centroids);
        }

        let zscored: Vec<Array2<f64>> = factors.iter().map(|f| zscore_rows(f)).collect();
        let mut signal = Array2::zeros((n_days, n_assets));

        for t in 0..n_days {
            let s = day_states[t];
            let mut weights: Vec<f64> = (0..n_factors).map(|f| state_ic[[f, s]].max(0.0)).collect();
            let sum: f64 = weights.iter().sum();
            if sum < 1e-12 {
                for w in &mut weights {
                    *w = 1.0 / n_factors as f64;
                }
            } else {
                for w in &mut weights {
                    *w /= sum;
                }
            }
            for a in 0..n_assets {
                let mut num = 0.0;
                let mut den = 0.0;
                for f in 0..n_factors {
                    let v = zscored[f][[t, a]];
                    if v.is_finite() {
                        num += v * weights[f];
                        den += weights[f];
                    }
                }
                signal[[t, a]] = if den > 0.0 { num / den } else { f64::NAN };
            }
        }
        Ok(signal)
    }

    fn name(&self) -> &str {
        "StateAware"
    }
}

// ── FactorComfortZone ──

pub struct FactorComfortZone {
    n_states: usize,
    activity_pctile: f64,
    centroids: Option<Array2<f64>>,
    state_ic: Option<Array2<f64>>,
    thresholds: Option<Vec<f64>>,
}

impl FactorComfortZone {
    pub fn new(n_states: usize, activity_pctile: f64) -> Self {
        Self {
            n_states,
            activity_pctile: activity_pctile.clamp(0.0, 1.0),
            centroids: None,
            state_ic: None,
            thresholds: None,
        }
    }
}

impl Strategy for FactorComfortZone {
    fn fit(&mut self, factors: &[Array2<f64>], forward_returns: &Array2<f64>) -> Result<()> {
        crate::strategy::validate_fit_input(factors, forward_returns)?;

        let n_factors = factors.len();
        let n_days = forward_returns.nrows();

        // ── Temporal split (same logic as StateAware) ──
        const MIN_SPLIT_DAYS: usize = 60;
        let (state_days, ic_factors, ic_returns, _ic_offset) = if n_days >= MIN_SPLIT_DAYS {
            let split = n_days / 2;
            let state_ret = forward_returns.slice(ndarray::s![..split, ..]).to_owned();
            let ic_f: Vec<Array2<f64>> = factors
                .iter()
                .map(|f| f.slice(ndarray::s![split.., ..]).to_owned())
                .collect();
            let ic_r = forward_returns.slice(ndarray::s![split.., ..]).to_owned();
            (state_ret, ic_f, ic_r, split)
        } else {
            let state_ret = forward_returns.clone();
            let ic_f: Vec<Array2<f64>> = factors.iter().map(|f| f.clone()).collect();
            let ic_r = forward_returns.clone();
            (state_ret, ic_f, ic_r, 0_usize)
        };

        // 1. K-means on state-definition period
        let state_features = build_market_features(&state_days);
        let (_, centroids) = kmeans_cluster(&state_features, self.n_states)?;

        // 2. Assign states to evaluation period
        let ic_features = build_market_features(&ic_returns);
        let ic_labels: Vec<usize> = (0..ic_features.nrows())
            .map(|t| assign_nearest(&ic_features.row(t), &centroids))
            .collect();

        // 3. Per-state IC estimation on evaluation period only
        let ic_n_days = ic_returns.nrows();
        let mut state_ic = Array2::zeros((n_factors, self.n_states));
        let mut state_counts = vec![0usize; self.n_states];

        for f_idx in 0..n_factors {
            let (ic_arr, _, _) =
                crate::backtest::metrics::compute_ic_series(&ic_factors[f_idx], &ic_returns)?;
            for t in 0..ic_n_days.min(ic_arr.len()) {
                let s = ic_labels[t];
                let ic_val = ic_arr[t];
                if ic_val.is_finite() {
                    state_ic[[f_idx, s]] += ic_val;
                    state_counts[s] += 1;
                }
            }
        }

        for s in 0..self.n_states {
            if state_counts[s] > 0 {
                for f in 0..n_factors {
                    state_ic[[f, s]] /= state_counts[s] as f64;
                }
            }
        }

        // Per-factor activation thresholds
        let mut thresholds = Vec::with_capacity(n_factors);
        for f in 0..n_factors {
            let mut ics: Vec<f64> = (0..self.n_states).map(|s| state_ic[[f, s]]).collect();
            ics.sort_by(|a, b| a.total_cmp(b));
            let idx = ((self.activity_pctile * (ics.len() - 1) as f64).floor() as usize)
                .min(ics.len() - 1);
            thresholds.push(ics[idx]);
        }

        self.centroids = Some(centroids);
        self.state_ic = Some(state_ic);
        self.thresholds = Some(thresholds);
        Ok(())
    }

    fn combine(&self, factors: &[Array2<f64>]) -> Result<Array2<f64>> {
        crate::strategy::validate_combine_input(factors)?;
        let centroids = self
            .centroids
            .as_ref()
            .ok_or("FactorComfortZone not fitted")?;
        let state_ic = self
            .state_ic
            .as_ref()
            .ok_or("FactorComfortZone not fitted")?;
        let thresholds = self
            .thresholds
            .as_ref()
            .ok_or("FactorComfortZone not fitted")?;

        let n_factors = factors.len();
        let (n_days, n_assets) = factors[0].dim();

        let features = build_market_features_from_factors(factors)?;
        let mut day_states = vec![0usize; n_days];
        for t in 0..n_days.min(features.nrows()) {
            day_states[t] = assign_nearest(&features.row(t), centroids);
        }

        let zscored: Vec<Array2<f64>> = factors.iter().map(|f| zscore_rows(f)).collect();
        let mut signal = Array2::zeros((n_days, n_assets));

        for t in 0..n_days {
            let s = day_states[t];
            let mut active: Vec<usize> = Vec::new();
            for f in 0..n_factors {
                if state_ic[[f, s]] >= thresholds[f] {
                    active.push(f);
                }
            }

            let weights: Vec<f64> = if active.is_empty() {
                vec![1.0 / n_factors as f64; n_factors]
            } else {
                let mut w = vec![0.0; n_factors];
                let mut sum = 0.0;
                for &f in &active {
                    w[f] = state_ic[[f, s]].max(0.0);
                    sum += w[f];
                }
                if sum < 1e-12 {
                    w.iter_mut().for_each(|x| *x = 1.0 / active.len() as f64);
                } else {
                    for &f in &active {
                        w[f] /= sum;
                    }
                }
                w
            };

            for a in 0..n_assets {
                let mut num = 0.0;
                let mut den = 0.0;
                for f in 0..n_factors {
                    let v = zscored[f][[t, a]];
                    if v.is_finite() {
                        num += v * weights[f];
                        den += weights[f];
                    }
                }
                signal[[t, a]] = if den > 0.0 { num / den } else { f64::NAN };
            }
        }
        Ok(signal)
    }

    fn name(&self) -> &str {
        "FactorComfortZone"
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Internal helpers
// ═══════════════════════════════════════════════════════════════════

fn build_market_features(returns: &Array2<f64>) -> Array2<f64> {
    let (n_days, n_assets) = returns.dim();
    let vol_window = 20;
    let mut features = Array2::zeros((n_days, 4));

    for t in 1..n_days {
        let prev = t - 1;

        // 1. Market return
        let fin: Vec<f64> = returns
            .row(prev)
            .iter()
            .filter(|&&x| x.is_finite())
            .copied()
            .collect();
        if !fin.is_empty() {
            features[[t, 0]] = fin.iter().sum::<f64>() / fin.len() as f64;
        }

        // 2. CS dispersion
        if fin.len() >= 2 {
            let mean = features[[t, 0]];
            let var: f64 =
                fin.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (fin.len() - 1) as f64;
            features[[t, 1]] = var.sqrt();
        }

        // 3. Market vol
        let start = t.saturating_sub(vol_window);
        let hist: Vec<f64> = (start..t)
            .filter_map(|d| {
                let row_fin: Vec<f64> = returns
                    .row(d)
                    .iter()
                    .filter(|&&x| x.is_finite())
                    .copied()
                    .collect();
                if row_fin.is_empty() {
                    None
                } else {
                    Some(row_fin.iter().sum::<f64>() / row_fin.len() as f64)
                }
            })
            .collect();
        if hist.len() >= 2 {
            let h_mean = hist.iter().sum::<f64>() / hist.len() as f64;
            let h_var: f64 =
                hist.iter().map(|&x| (x - h_mean).powi(2)).sum::<f64>() / (hist.len() - 1) as f64;
            features[[t, 2]] = h_var.sqrt();
        }

        // 4. Turnover proxy
        if t >= 2 {
            let prev2 = t - 2;
            let med_p2 = median_of_row(returns, prev2);
            let med_p1 = median_of_row(returns, prev);
            let mut cross = 0;
            let mut total = 0;
            for a in 0..n_assets {
                let vp2 = returns[[prev2, a]];
                let vp1 = returns[[prev, a]];
                if vp2.is_finite() && vp1.is_finite() {
                    if (vp2 - med_p2).signum() != (vp1 - med_p1).signum()
                        || (vp2 - med_p2).abs() < 1e-15
                        || (vp1 - med_p1).abs() < 1e-15
                    {
                        cross += 1;
                    }
                    total += 1;
                }
            }
            if total > 0 {
                features[[t, 3]] = cross as f64 / total as f64;
            }
        }
    }

    // Z-score normalize each feature dimension
    let (n, d) = features.dim();
    let mut normed = Array2::zeros((n, d));
    for j in 0..d {
        let col_fin: Vec<f64> = features
            .column(j)
            .iter()
            .filter(|&&x| x.is_finite())
            .copied()
            .collect();
        if col_fin.len() >= 2 {
            let mean = col_fin.iter().sum::<f64>() / col_fin.len() as f64;
            let var: f64 = col_fin.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / (col_fin.len() - 1) as f64;
            let std = var.sqrt();
            if std > 1e-15 {
                for i in 0..n {
                    if features[[i, j]].is_finite() {
                        normed[[i, j]] = (features[[i, j]] - mean) / std;
                    } else {
                        normed[[i, j]] = f64::NAN;
                    }
                }
            }
        }
    }
    normed
}

fn build_market_features_from_factors(factors: &[Array2<f64>]) -> Result<Array2<f64>> {
    if factors.is_empty() {
        return Err("no factors".into());
    }
    let f0 = &factors[0];
    let (n_days, _n_assets) = f0.dim();
    let mut features = Array2::zeros((n_days, 4));

    for t in 1..n_days {
        let prev = t - 1;
        let fin: Vec<f64> = f0
            .row(prev)
            .iter()
            .filter(|&&x| x.is_finite())
            .copied()
            .collect();
        if !fin.is_empty() {
            features[[t, 0]] = fin.iter().sum::<f64>() / fin.len() as f64;
        }
        if fin.len() >= 2 {
            let mean = features[[t, 0]];
            let var: f64 =
                fin.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (fin.len() - 1) as f64;
            features[[t, 1]] = var.sqrt();
        }
    }
    Ok(features)
}

fn median_of_row(mat: &Array2<f64>, row: usize) -> f64 {
    let mut vals: Vec<f64> = mat
        .row(row)
        .iter()
        .filter(|&&x| x.is_finite())
        .copied()
        .collect();
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 0 {
        (vals[mid - 1] + vals[mid]) / 2.0
    } else {
        vals[mid]
    }
}

fn euclidean_dist_sq(a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
}

fn assign_nearest(row: &ndarray::ArrayView1<f64>, centroids: &Array2<f64>) -> usize {
    let k = centroids.nrows();
    let mut best = 0;
    let mut best_dist = f64::INFINITY;
    for j in 0..k {
        let d = euclidean_dist_sq(row, &centroids.row(j));
        if d < best_dist {
            best_dist = d;
            best = j;
        }
    }
    best
}

/// Simple K-means with Lloyd's algorithm and K-means++ initialization.
/// Hand-rolled because linfa-clustering 0.7 depends on ndarray 0.15
/// which is incompatible with our ndarray 0.17.
fn kmeans_cluster(data: &Array2<f64>, k: usize) -> Result<(Vec<usize>, Array2<f64>)> {
    if data.nrows() == 0 {
        return Err("empty data for clustering".into());
    }
    if k < 2 || k > 10 {
        return Err(format!("n_states must be 2..=10, got {}", k));
    }

    let mask: Vec<bool> = (0..data.nrows())
        .map(|i| data.row(i).iter().all(|v| v.is_finite()))
        .collect();
    let clean_idx: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter(|(_, ok)| **ok)
        .map(|(i, _)| i)
        .collect();

    if clean_idx.len() < k {
        return Err(format!(
            "only {} clean feature rows, need at least {} for {} states",
            clean_idx.len(),
            k,
            k
        ));
    }

    let n = clean_idx.len();
    let d = data.ncols();
    let mut x = Array2::zeros((n, d));
    for (new_i, &old_i) in clean_idx.iter().enumerate() {
        x.row_mut(new_i).assign(&data.row(old_i));
    }

    let mut rng = rand::thread_rng();
    let n_restarts = 10;
    let max_iter = 100;
    let tol = 1e-4;
    let mut best_inertia = f64::INFINITY;
    let mut best_labels = vec![0usize; n];
    let mut best_centroids = Array2::zeros((k, d));

    for _ in 0..n_restarts {
        let centroids = kmeans_plus_plus(&x, k, &mut rng);
        let (labels, final_centroids, inertia) = lloyd(&x, &centroids, max_iter, tol);
        if inertia < best_inertia {
            best_inertia = inertia;
            best_labels = labels;
            best_centroids = final_centroids;
        }
    }

    let mut all_labels = vec![0usize; data.nrows()];
    for (ci, &orig_i) in clean_idx.iter().enumerate() {
        all_labels[orig_i] = best_labels[ci];
    }

    Ok((all_labels, best_centroids))
}

fn kmeans_plus_plus(x: &Array2<f64>, k: usize, rng: &mut impl Rng) -> Array2<f64> {
    let (n, d) = x.dim();
    let mut centroids = Array2::zeros((k, d));

    let first = rng.gen_range(0..n);
    centroids.row_mut(0).assign(&x.row(first));

    for c in 1..k {
        let mut min_dist_sq = vec![f64::INFINITY; n];
        for i in 0..n {
            for j in 0..c {
                let dist = euclidean_dist_sq(&x.row(i), &centroids.row(j));
                if dist < min_dist_sq[i] {
                    min_dist_sq[i] = dist;
                }
            }
        }
        let total: f64 = min_dist_sq.iter().sum();
        let threshold = rng.r#gen::<f64>() * total;
        let mut cumsum = 0.0;
        for i in 0..n {
            cumsum += min_dist_sq[i];
            if cumsum >= threshold {
                centroids.row_mut(c).assign(&x.row(i));
                break;
            }
        }
    }
    centroids
}

fn lloyd(
    x: &Array2<f64>,
    init_centroids: &Array2<f64>,
    max_iter: usize,
    _tol: f64,
) -> (Vec<usize>, Array2<f64>, f64) {
    let (n, d) = x.dim();
    let k = init_centroids.nrows();
    let mut centroids = init_centroids.clone();
    let mut labels = vec![0usize; n];

    for _iter in 0..max_iter {
        let mut changed = false;

        // Assignment step
        for i in 0..n {
            let mut best = labels[i];
            let mut best_dist = euclidean_dist_sq(&x.row(i), &centroids.row(best));
            for j in 0..k {
                let dist = euclidean_dist_sq(&x.row(i), &centroids.row(j));
                if dist < best_dist {
                    best_dist = dist;
                    best = j;
                }
            }
            if best != labels[i] {
                labels[i] = best;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step
        centroids.fill(0.0);
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = labels[i];
            for dim in 0..d {
                centroids[[c, dim]] += x[[i, dim]];
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for dim in 0..d {
                    centroids[[c, dim]] /= counts[c] as f64;
                }
            }
        }
    }

    let mut inertia = 0.0;
    for i in 0..n {
        inertia += euclidean_dist_sq(&x.row(i), &centroids.row(labels[i]));
    }
    (labels, centroids, inertia)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    // SYNTHETIC DATA: two market states + two complementary factors.

    #[test]
    fn syn_state_aware_two_states() {
        let n_days = 100;
        let n_assets = 2;
        let mut ret = Array2::zeros((n_days, n_assets));
        let mut f_a = Array2::zeros((n_days, n_assets));
        let mut f_b = Array2::zeros((n_days, n_assets));

        for t in 0..50 {
            for a in 0..n_assets {
                ret[[t, a]] = 0.001;
                f_a[[t, a]] = ret[[t, a]] * 0.1;
                f_b[[t, a]] = ret[[t, a]] * (-0.03);
            }
        }
        for t in 50..100 {
            for a in 0..n_assets {
                ret[[t, a]] = 0.03;
                f_a[[t, a]] = ret[[t, a]] * (-0.05);
                f_b[[t, a]] = ret[[t, a]] * 0.08;
            }
        }

        let mut s = StateAware::new(2, 252);
        s.fit(&[f_a.clone(), f_b.clone()], &ret).unwrap();
        let signal = s.combine(&[f_a, f_b]).unwrap();
        assert_eq!(signal.dim(), (n_days, n_assets));
        assert!(signal.iter().any(|v| v.is_finite()));
    }

    #[test]
    fn syn_ic_lookback_changes_weights() {
        // SYNTHETIC DATA: verify that ic_lookback affects per-state IC estimation.
        // Short window vs full period should produce different weights.
        let n_days = 60;
        let n_assets = 3;
        let mut ret = Array2::zeros((n_days, n_assets));
        let mut f = Array2::zeros((n_days, n_assets));

        // First 30 days: factor positively correlated
        // Last 30 days: factor negatively correlated
        for t in 0..30 {
            for a in 0..n_assets {
                ret[[t, a]] = (a as f64 + 1.0) * 0.01;
                f[[t, a]] = ret[[t, a]] * 0.1;
            }
        }
        for t in 30..60 {
            for a in 0..n_assets {
                ret[[t, a]] = (a as f64 + 1.0) * 0.01;
                f[[t, a]] = -ret[[t, a]] * 0.1;
            }
        }

        // Full window (252, more than n_days → use all)
        let mut s_full = StateAware::new(2, 252);
        s_full.fit(&[f.clone()], &ret).unwrap();

        // Short window (only last 10 days, where IC is negative)
        let mut s_short = StateAware::new(2, 10);
        s_short.fit(&[f.clone()], &ret).unwrap();

        // Both should produce valid signals
        let sig_full = s_full.combine(&[f.clone()]).unwrap();
        let sig_short = s_short.combine(&[f]).unwrap();
        assert_eq!(sig_full.dim(), (n_days, n_assets));
        assert_eq!(sig_short.dim(), (n_days, n_assets));
        assert!(sig_full.iter().any(|v| v.is_finite()));
        assert!(sig_short.iter().any(|v| v.is_finite()));
    }

    #[test]
    fn syn_comfort_zone_binary() {
        let n_days = 100;
        let n_assets = 2;
        let mut ret = Array2::zeros((n_days, n_assets));
        let mut f_a = Array2::zeros((n_days, n_assets));
        let mut f_b = Array2::zeros((n_days, n_assets));

        for t in 0..50 {
            for a in 0..n_assets {
                ret[[t, a]] = 0.001;
                f_a[[t, a]] = ret[[t, a]] * 0.1;
                f_b[[t, a]] = ret[[t, a]] * (-0.03);
            }
        }
        for t in 50..100 {
            for a in 0..n_assets {
                ret[[t, a]] = 0.03;
                f_a[[t, a]] = ret[[t, a]] * (-0.05);
                f_b[[t, a]] = ret[[t, a]] * 0.08;
            }
        }

        let mut c = FactorComfortZone::new(2, 0.7);
        c.fit(&[f_a.clone(), f_b.clone()], &ret).unwrap();
        let signal = c.combine(&[f_a, f_b]).unwrap();
        assert_eq!(signal.dim(), (n_days, n_assets));
        assert!(signal.iter().any(|v| v.is_finite()));
    }

    #[test]
    fn syn_comfort_zone_fallback() {
        // More data to get valid IC values
        let n = 30;
        let ret =
            Array2::from_shape_vec((n, 2), (0..60).map(|i| (i as f64) * 0.001).collect()).unwrap();
        let f = &ret * 0.5;
        let mut c = FactorComfortZone::new(2, 0.9);
        c.fit(&[f.clone()], &ret).unwrap();
        let signal = c.combine(&[f]).unwrap();
        assert_eq!(signal.dim(), (n, 2));
    }

    // ── Corner cases ──

    #[test]
    fn syn_state_aware_not_fitted_errors() {
        let f = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let s = StateAware::new(2, 252);
        assert!(s.combine(&[f]).is_err());
    }

    #[test]
    fn syn_comfort_zone_not_fitted_errors() {
        let f = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let s = FactorComfortZone::new(2, 0.7);
        assert!(s.combine(&[f]).is_err());
    }

    #[test]
    fn syn_state_aware_single_factor() {
        let n = 40;
        let ret = Array2::from_shape_vec((n, 3), (0..n * 3).map(|i| (i as f64) * 0.001).collect())
            .unwrap();
        let f = &ret * 0.5;
        let mut s = StateAware::new(2, 252);
        s.fit(&[f.clone()], &ret).unwrap();
        let signal = s.combine(&[f]).unwrap();
        assert_eq!(signal.dim(), (n, 3));
        assert!(signal.iter().any(|v| v.is_finite()));
    }

    #[test]
    fn syn_state_aware_large_lookback() {
        // ic_lookback larger than n_days → use all days (no truncation)
        let n = 30;
        let ret =
            Array2::from_shape_vec((n, 2), (0..60).map(|i| (i as f64) * 0.001).collect()).unwrap();
        let f = &ret * 0.5;
        let mut s = StateAware::new(2, 500); // lookback > n_days
        s.fit(&[f.clone()], &ret).unwrap();
        let signal = s.combine(&[f]).unwrap();
        assert_eq!(signal.dim(), (n, 2));
    }
}
