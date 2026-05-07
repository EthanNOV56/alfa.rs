//! Performance metrics and statistics functions.

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::f64::NAN;

pub(crate) fn compute_cumulative_returns(daily_returns: &Array2<f64>) -> Array2<f64> {
    let (n_days, n_groups) = daily_returns.dim();
    let mut cum_returns = Array2::<f64>::zeros((n_days, n_groups));

    for g in 0..n_groups {
        let mut log_cum = 0.0;
        for d in 0..n_days {
            let r = daily_returns[[d, g]];
            if r.is_nan() {
                log_cum = f64::NAN;
            } else {
                let r_adj = 1.0 + r;
                if r_adj > 0.0 {
                    log_cum += r_adj.ln();
                }
            }
            cum_returns[[d, g]] = if log_cum.is_nan() {
                f64::NAN
            } else {
                log_cum.exp() - 1.0
            };
        }
    }

    cum_returns
}

pub(crate) fn compute_ic_series(
    factor: &Array2<f64>,
    returns: &Array2<f64>,
) -> Result<(Array1<f64>, f64, f64), String> {
    let (n_days, n_assets) = factor.dim();

    let ic_vec: Vec<f64> = (0..(n_days - 1))
        .into_par_iter()
        .map(|day| {
            let factor_today = factor.row(day);
            let forward_returns = returns.row(day);

            let mut factor_vals = Vec::new();
            let mut return_vals = Vec::new();

            for asset in 0..n_assets {
                let f = factor_today[asset];
                let r = forward_returns[asset];
                if !f.is_nan() && !r.is_nan() {
                    factor_vals.push(f);
                    return_vals.push(r);
                }
            }

            if factor_vals.len() < 2 {
                return NAN;
            }

            pearson_correlation(&factor_vals, &return_vals)
        })
        .collect();

    let ic_series = Array1::from_vec(ic_vec);

    let valid_n = ic_series.iter().filter(|&&v| !v.is_nan()).count();
    if valid_n < 2 {
        return Err("No valid IC values".to_string());
    }

    let ic_mean = ic_series.iter().filter(|v| !v.is_nan()).sum::<f64>() / valid_n as f64;
    let ic_std = {
        let variance: f64 = ic_series
            .iter()
            .filter(|v| !v.is_nan())
            .map(|&x| (x - ic_mean) * (x - ic_mean))
            .sum::<f64>()
            / (valid_n - 1) as f64;
        variance.sqrt()
    };
    let ic_ir = if ic_std == 0.0 { NAN } else { ic_mean / ic_std };

    Ok((ic_series, ic_mean, ic_ir))
}

pub(crate) fn compute_annualized_return(total_return: f64, n_days: usize) -> f64 {
    if n_days <= 1 {
        return 0.0;
    }
    let years = n_days as f64 / 252.0;
    if years <= 0.0 {
        return 0.0;
    }
    (1.0 + total_return).powf(1.0 / years) - 1.0
}

pub(crate) fn compute_sharpe_ratio(returns: &Array1<f64>, _n_days: usize) -> f64 {
    let valid_n = returns.iter().filter(|&&r| !r.is_nan()).count();
    if valid_n < 2 {
        return 0.0;
    }

    let mean = returns.iter().filter(|v| !v.is_nan()).sum::<f64>() / valid_n as f64;
    let variance: f64 = returns
        .iter()
        .filter(|v| !v.is_nan())
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f64>()
        / (valid_n - 1) as f64;
    let std = variance.sqrt();

    if std == 0.0 {
        return 0.0;
    }

    let _annualized_std = std * (252.0_f64).sqrt();
    mean / std * (252.0_f64).sqrt()
}

pub(crate) fn compute_max_drawdown(returns: &Array1<f64>) -> f64 {
    let mut cum = 1.0;
    let mut max_cum = 1.0;
    let mut max_drawdown = 0.0;

    for &r in returns.iter() {
        if r.is_nan() {
            continue;
        }
        cum *= 1.0 + r;
        if cum > max_cum {
            max_cum = cum;
        }
        let drawdown = (max_cum - cum) / max_cum;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    max_drawdown
}

/// Build cumulative NAV curve from daily returns.
pub(crate) fn cumulative_nav_curve(returns: &Array1<f64>) -> Array1<f64> {
    let n = returns.len();
    let mut curve = Array1::zeros(n);
    let mut cum = 1.0;
    for (i, &r) in returns.iter().enumerate() {
        if r.is_finite() {
            cum *= 1.0 + r;
        }
        curve[i] = cum;
    }
    curve
}

/// Compute total return using log returns for numerical stability
pub(crate) fn compute_total_return_log(returns: &Array1<f64>) -> f64 {
    let mut log_sum = 0.0;
    for &r in returns.iter() {
        if r.is_nan() {
            continue;
        }
        let r_adj = 1.0 + r;
        if r_adj > 0.0 {
            log_sum += r_adj.ln();
        }
    }
    log_sum.exp() - 1.0
}

pub(crate) fn compute_turnover(group_labels: &Array2<usize>) -> f64 {
    let (n_days, n_assets) = group_labels.dim();
    let mut total_turnover = 0.0;
    let mut count = 0;

    for day in 1..n_days {
        let prev_labels = group_labels.row(day - 1);
        let curr_labels = group_labels.row(day);

        for asset in 0..n_assets {
            if prev_labels[asset] != curr_labels[asset] {
                total_turnover += 1.0;
            }
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    total_turnover / count as f64
}

/// Compute IC series within a subset of assets filtered by group membership.
///
/// For each day, only assets whose `group_labels[day]` belongs to `target_groups`
/// are included in the Pearson correlation between factor and forward returns.
pub(crate) fn compute_group_ic_series(
    factor: &Array2<f64>,
    returns: &Array2<f64>,
    group_labels: &Array2<usize>,
    target_groups: &[usize],
) -> Result<(Array1<f64>, f64, f64), String> {
    let (n_days, n_assets) = factor.dim();
    let target_set: std::collections::HashSet<usize> = target_groups.iter().copied().collect();

    let ic_vec: Vec<f64> = (0..(n_days - 1))
        .into_par_iter()
        .map(|day| {
            let factor_today = factor.row(day);
            let forward_returns = returns.row(day);
            let labels_today = group_labels.row(day);

            let mut factor_vals = Vec::new();
            let mut return_vals = Vec::new();

            for asset in 0..n_assets {
                let label = labels_today[asset];
                if label == 0 || !target_set.contains(&label) {
                    continue;
                }
                let f = factor_today[asset];
                let r = forward_returns[asset];
                if !f.is_nan() && !r.is_nan() {
                    factor_vals.push(f);
                    return_vals.push(r);
                }
            }

            if factor_vals.len() < 2 {
                return std::f64::NAN;
            }

            pearson_correlation(&factor_vals, &return_vals)
        })
        .collect();

    let ic_series = Array1::from_vec(ic_vec);

    let valid_n = ic_series.iter().filter(|&&v| !v.is_nan()).count();
    if valid_n < 2 {
        return Ok((ic_series, std::f64::NAN, std::f64::NAN));
    }

    let ic_mean = ic_series.iter().filter(|v| !v.is_nan()).sum::<f64>() / valid_n as f64;
    let ic_std = {
        let variance: f64 = ic_series
            .iter()
            .filter(|v| !v.is_nan())
            .map(|&x| (x - ic_mean) * (x - ic_mean))
            .sum::<f64>()
            / (valid_n - 1) as f64;
        variance.sqrt()
    };
    let ic_ir = if ic_std == 0.0 {
        std::f64::NAN
    } else {
        ic_mean / ic_std
    };

    Ok((ic_series, ic_mean, ic_ir))
}

/// Compute weight-based turnover: average daily sum of absolute weight changes.
/// Normalized by 2: turnover=0 means no change, turnover=1 means complete rebalance.
pub(crate) fn compute_weight_turnover(group_weights: &Array2<f64>) -> f64 {
    let (n_days, n_assets) = group_weights.dim();
    if n_days < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for day in 1..n_days {
        let prev_w = group_weights.row(day - 1);
        let curr_w = group_weights.row(day);
        let mut day_sum = 0.0;
        for a in 0..n_assets {
            let pw = prev_w[a];
            let cw = curr_w[a];
            if pw.is_finite() && cw.is_finite() {
                day_sum += (cw - pw).abs();
            }
        }
        total += day_sum / 2.0; // normalize: full rebalance = 1.0
    }
    total / (n_days - 1) as f64
}

/// Compute win rate: fraction of periods with positive return.
pub(crate) fn compute_win_rate(returns: &Array1<f64>) -> f64 {
    let valid: Vec<f64> = returns
        .iter()
        .filter(|&&r| r.is_finite())
        .copied()
        .collect();
    if valid.is_empty() {
        return 0.0;
    }
    let wins = valid.iter().filter(|&&r| r > 0.0).count();
    wins as f64 / valid.len() as f64
}

/// Compute Calmar ratio: annualized_return / max_drawdown.
/// Returns NAN if max_drawdown is 0.
pub(crate) fn compute_calmar_ratio(annualized_return: f64, max_drawdown: f64) -> f64 {
    if max_drawdown <= 0.0 || max_drawdown.is_nan() {
        return std::f64::NAN;
    }
    annualized_return / max_drawdown
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y).map(|(&a, &b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|&a| a * a).sum();
    let sum_y2: f64 = y.iter().map(|&b| b * b).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}
