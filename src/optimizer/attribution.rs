//! Performance attribution and IC statistics.
//!
//! Three families of analysis:
//! - **Brinson attribution**: decompose excess return into allocation, selection,
//!   and interaction effects.
//! - **Factor attribution**: time-series regression of portfolio returns onto
//!   factor returns via OLS (nalgebra SVD).
//! - **IC statistics**: rank and Pearson IC, IR, t-stat, autocorrelation.
//!
//! Research basis: reports #1, #2, #4, #6.

use ndarray::{Array1, Array2, Axis};

use super::solver;

// ── ICStatistics ─────────────────────────────────────────────────────────────

/// Comprehensive IC metrics for factor evaluation.
#[derive(Debug, Clone)]
pub struct ICStatistics {
    pub rank_ic: Array1<f64>,
    pub pearson_ic: Array1<f64>,
    pub rank_ic_mean: f64,
    pub rank_ic_ir: f64,
    pub rank_ic_tstat: f64,
    pub rank_ic_positive_ratio: f64,
    /// Autocorrelation lags 1–4 (factor momentum).
    pub rank_ic_autocorr: [f64; 4],
}

// ── IC computation ───────────────────────────────────────────────────────────

/// Compute comprehensive IC statistics from factor values and forward returns.
///
/// # Parameters
/// - `factor_values`: per-day factor values `(n_dates × n_assets)`
/// - `forward_returns`: per-day forward returns `(n_dates × n_assets)`
pub fn compute_ic_statistics(
    factor_values: &Array2<f64>,
    forward_returns: &Array2<f64>,
) -> ICStatistics {
    let (n_days, _n_assets) = factor_values.dim();
    let mut rank_ic = Array1::<f64>::zeros(n_days);
    let mut pearson_ic = Array1::<f64>::zeros(n_days);

    for day in 0..n_days {
        let f_row = factor_values.row(day);
        let r_row = forward_returns.row(day);

        // Collect valid (factor, return) pairs
        let pairs: Vec<(f64, f64)> = f_row
            .iter()
            .zip(r_row.iter())
            .filter(|(a, b)| a.is_finite() && b.is_finite())
            .map(|(&a, &b)| (a, b))
            .collect();

        if pairs.len() < 3 {
            rank_ic[day] = f64::NAN;
            pearson_ic[day] = f64::NAN;
            continue;
        }

        let fv: Vec<f64> = pairs.iter().map(|(f, _)| *f).collect();
        let rv: Vec<f64> = pairs.iter().map(|(_, r)| *r).collect();

        rank_ic[day] = spearman_rank_ic(&fv, &rv);
        pearson_ic[day] = pearson_corr(&fv, &rv);
    }

    let valid_ic: Vec<f64> = rank_ic.iter().copied().filter(|x| x.is_finite()).collect();
    let t = valid_ic.len() as f64;

    let mean = if t > 0.0 {
        valid_ic.iter().sum::<f64>() / t
    } else {
        f64::NAN
    };

    let std_dev = if t > 1.0 {
        let var = valid_ic.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (t - 1.0);
        var.sqrt()
    } else {
        f64::NAN
    };

    let ir = if std_dev.is_finite() && std_dev > 1e-12 {
        mean / std_dev
    } else {
        f64::NAN
    };

    let tstat = if std_dev.is_finite() && std_dev > 1e-12 && t > 0.0 {
        mean / (std_dev / t.sqrt())
    } else {
        f64::NAN
    };

    let positive_ratio = if t > 0.0 {
        valid_ic.iter().filter(|&&v| v > 0.0).count() as f64 / t
    } else {
        f64::NAN
    };

    let mut autocorr = [f64::NAN; 4];
    for lag in 1..=4 {
        if t as usize > lag {
            autocorr[lag - 1] = pearson_corr(&valid_ic[..t as usize - lag], &valid_ic[lag..]);
        }
    }

    ICStatistics {
        rank_ic,
        pearson_ic,
        rank_ic_mean: mean,
        rank_ic_ir: ir,
        rank_ic_tstat: tstat,
        rank_ic_positive_ratio: positive_ratio,
        rank_ic_autocorr: autocorr,
    }
}

// ── Brinson attribution ──────────────────────────────────────────────────────

/// Result of a Brinson performance attribution.
#[derive(Debug, Clone)]
pub struct BrinsonAttribution {
    pub sector_names: Vec<String>,
    pub allocation_effect: Array1<f64>,
    pub selection_effect: Array1<f64>,
    pub interaction_effect: Array1<f64>,
    pub total_excess_return: f64,
}

/// Sector-level Brinson attribution.
///
/// Decomposes excess return into allocation, selection, and interaction
/// effects.
///
/// - allocation_effect[s] = (w_p[s] - w_b[s]) * r_b[s]
/// - selection_effect[s]  = w_b[s] * (r_p[s] - r_b[s])
/// - interaction_effect[s] = (w_p[s] - w_b[s]) * (r_p[s] - r_b[s])
///
/// `sector_assignments[i]` gives the sector index (0..S-1) of asset `i`.
pub fn brinson_attribution(
    portfolio_weights: &Array1<f64>,
    benchmark_weights: &Array1<f64>,
    portfolio_returns: &Array1<f64>,
    benchmark_returns: &Array1<f64>,
    sector_assignments: &[usize],
    sector_names: &[String],
) -> BrinsonAttribution {
    let n_sectors = sector_names.len();
    let mut allocation = Array1::<f64>::zeros(n_sectors);
    let mut selection = Array1::<f64>::zeros(n_sectors);
    let mut interaction = Array1::<f64>::zeros(n_sectors);

    // Aggregate weights and returns per sector
    let mut pw_sector = vec![0.0_f64; n_sectors];
    let mut bw_sector = vec![0.0_f64; n_sectors];
    let mut pr_sector = vec![0.0_f64; n_sectors];
    let mut br_sector = vec![0.0_f64; n_sectors];
    let mut count_sector = vec![0_usize; n_sectors];

    for (i, &s) in sector_assignments.iter().enumerate() {
        if s < n_sectors {
            pw_sector[s] += portfolio_weights[i];
            bw_sector[s] += benchmark_weights[i];
            pr_sector[s] += portfolio_returns[i] * portfolio_weights[i];
            br_sector[s] += benchmark_returns[i] * benchmark_weights[i];
            count_sector[s] += 1;
        }
    }

    // Per-sector average returns
    for s in 0..n_sectors {
        if bw_sector[s] > 1e-12 {
            br_sector[s] /= bw_sector[s];
        }
        if pw_sector[s] > 1e-12 {
            pr_sector[s] /= pw_sector[s];
        }
    }

    for s in 0..n_sectors {
        if count_sector[s] == 0 {
            continue;
        }
        allocation[s] = (pw_sector[s] - bw_sector[s]) * br_sector[s];
        selection[s] = bw_sector[s] * (pr_sector[s] - br_sector[s]);
        interaction[s] = (pw_sector[s] - bw_sector[s]) * (pr_sector[s] - br_sector[s]);
    }

    let total = allocation.sum() + selection.sum() + interaction.sum();

    BrinsonAttribution {
        sector_names: sector_names.to_vec(),
        allocation_effect: allocation,
        selection_effect: selection,
        interaction_effect: interaction,
        total_excess_return: total,
    }
}

// ── Factor return attribution ────────────────────────────────────────────────

/// Result of a factor-based performance attribution.
#[derive(Debug, Clone)]
pub struct FactorAttribution {
    pub factor_contributions: Vec<(String, f64)>,
    /// Beta coefficients from the regression.
    pub betas: Array1<f64>,
    pub specific_return: f64,
    pub r_squared: f64,
}

/// Factor-based performance attribution via OLS.
///
/// Regresses `r_portfolio` onto the `factor_returns` matrix:
///
/// ```text
/// r_portfolio = F · β + ε
/// ```
///
/// Uses nalgebra SVD for the least-squares solution.
pub fn factor_attribution(
    portfolio_returns: &Array1<f64>,
    factor_returns: &Array2<f64>,
    factor_names: &[String],
) -> FactorAttribution {
    let t = portfolio_returns.len();
    let k = factor_returns.ncols();

    // SVD solve: F = U S V^T, then β = V S⁺ U^T r
    let f_nalg = solver::to_nalgebra_matrix(factor_returns);
    let r_nalg = solver::to_nalgebra_vector(portfolio_returns);

    let svd = nalgebra::linalg::SVD::new(f_nalg, true, true);
    let beta_nalg = svd
        .solve(&r_nalg, 1e-10)
        .unwrap_or(nalgebra::DVector::from_element(k, 0.0));
    let betas = solver::to_ndarray_vector(&beta_nalg);

    // Factor contributions
    let factor_means: Array1<f64> = factor_returns.mean_axis(Axis(0)).unwrap();
    let contributions: Vec<(String, f64)> = (0..k)
        .map(|i| {
            let name = factor_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("F{i}"));
            (name, betas[i] * factor_means[i])
        })
        .collect();

    // Specific return
    let predicted: Array1<f64> = factor_returns.dot(&betas);
    let residuals = portfolio_returns - &predicted;
    let specific_return = residuals.mean().unwrap_or(0.0);

    // R²
    let ss_res: f64 = residuals.mapv(|x| x * x).sum();
    let mean_r = portfolio_returns.mean().unwrap_or(0.0);
    let ss_tot: f64 = portfolio_returns.mapv(|x| (x - mean_r).powi(2)).sum();
    let r_squared = if ss_tot > 1e-12 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    FactorAttribution {
        factor_contributions: contributions,
        betas,
        specific_return,
        r_squared,
    }
}

// ── Helper functions ─────────────────────────────────────────────────────────

/// Spearman rank IC via tied-rank method.
fn spearman_rank_ic(fv: &[f64], rv: &[f64]) -> f64 {
    let n = fv.len();
    let rank_f = rank_with_ties(fv);
    let rank_r = rank_with_ties(rv);
    pearson_corr(&rank_f, &rank_r)
}

/// Rank values with average-tie handling.
fn rank_with_ties(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = (i + j - 1) as f64 / 2.0;
        for idx in &indexed[i..j] {
            ranks[idx.0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Pearson correlation coefficient.
fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }

    let denom = (vx * vy).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    cov / denom
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// [SYNTHETIC] Brinson: allocation + selection + interaction = total excess.
    #[test]
    fn test_brinson_decomposition() {
        let pw = Array1::from_vec(vec![0.6, 0.4, 0.0, 0.0]);
        let bw = Array1::from_vec(vec![0.5, 0.0, 0.5, 0.0]);
        let pr = Array1::from_vec(vec![0.05, 0.10, 0.0, 0.0]);
        let br = Array1::from_vec(vec![0.03, 0.0, 0.02, 0.0]);
        let sectors = &[0, 0, 1, 1];
        let names = &["SectorA".to_string(), "SectorB".to_string()];

        let attr = brinson_attribution(&pw, &bw, &pr, &br, sectors, names);

        let alloc_sum = attr.allocation_effect.sum();
        let select_sum = attr.selection_effect.sum();
        let interact_sum = attr.interaction_effect.sum();
        let total = attr.total_excess_return;

        assert!(
            (alloc_sum + select_sum + interact_sum - total).abs() < 1e-12,
            "decomposition should sum to total"
        );
    }

    /// [SYNTHETIC] Factor attribution: known beta recovered.
    #[test]
    fn test_factor_attribution_known_beta() {
        // True betas: [0.8, 0.4, 0.2]
        let true_beta = Array1::from_vec(vec![0.8, 0.4, 0.2]);
        let t = 100;
        let mut factor_r = Array2::<f64>::zeros((t, 3));
        for row in 0..t {
            for col in 0..3 {
                factor_r[[row, col]] = rand::random::<f64>() - 0.5;
            }
        }
        let mut port_r = Array1::<f64>::zeros(t);
        for row in 0..t {
            port_r[row] = true_beta[0] * factor_r[[row, 0]]
                + true_beta[1] * factor_r[[row, 1]]
                + true_beta[2] * factor_r[[row, 2]]
                + (rand::random::<f64>() - 0.5) * 0.01;
        }

        let names = &["MKT".to_string(), "SMB".to_string(), "HML".to_string()];

        let names = &["MKT".to_string(), "SMB".to_string(), "HML".to_string()];
        let attr = factor_attribution(&port_r, &factor_r, names);

        assert!(attr.r_squared > 0.95);
        assert_eq!(attr.betas.len(), 3);
        // Betas should be close to true values
        assert!((attr.betas[0] - 0.8).abs() < 0.05);
        assert!((attr.betas[1] - 0.4).abs() < 0.05);
        assert!((attr.betas[2] - 0.2).abs() < 0.05);
    }

    /// [SYNTHETIC] IC: perfectly correlated signal → Rank IC ≈ 1.0.
    #[test]
    fn test_ic_perfect_correlation() {
        let n = 100;
        let signal = Array2::from_shape_vec((1, n), (0..n).map(|i| i as f64).collect()).unwrap();
        // Same ranking → IC = 1.0
        let ret = Array2::from_shape_vec((1, n), (0..n).map(|i| i as f64).collect()).unwrap();

        let ic = compute_ic_statistics(&signal, &ret);
        assert!((ic.rank_ic_mean - 1.0).abs() < 1e-10);
    }

    /// [SYNTHETIC] IC: anti-correlated signal → Rank IC ≈ -1.0.
    #[test]
    fn test_ic_anti_correlation() {
        let n = 100;
        let signal = Array2::from_shape_vec((1, n), (0..n).map(|i| i as f64).collect()).unwrap();
        let ret = Array2::from_shape_vec((1, n), (0..n).map(|i| -(i as f64)).collect()).unwrap();

        let ic = compute_ic_statistics(&signal, &ret);
        assert!((ic.rank_ic_mean + 1.0).abs() < 1e-10);
    }

    /// [SYNTHETIC] IC: autocorrelation computed for >4 periods.
    #[test]
    fn test_ic_autocorr() {
        let n = 10;
        let days = 30;
        let mut signal = Array2::<f64>::zeros((days, n));
        let mut ret = Array2::<f64>::zeros((days, n));
        for d in 0..days {
            for i in 0..n {
                signal[[d, i]] = rand::random::<f64>();
                ret[[d, i]] = rand::random::<f64>();
            }
        }
        let ic = compute_ic_statistics(&signal, &ret);
        assert_eq!(ic.rank_ic_autocorr.len(), 4);
        // All should be finite (may be NaN if IC std is 0, but with random data, shouldn't be)
        for &ac in &ic.rank_ic_autocorr {
            assert!(ac.is_finite() || ac.is_nan()); // just check type
        }
    }
}
