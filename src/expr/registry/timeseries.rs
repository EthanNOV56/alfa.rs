//! Time series functions for factor computation

use ndarray::Array1;

/// Winsorize: clip values to [mean - 3*std, mean + 3*std] per cross-section (per date)
/// Data is flattened as (n_dates * n_symbols), n_symbols is the number of assets per date
/// NaN values are excluded from mean/std computation but remain as NaN in output
pub fn winsor(vals: &Array1<f64>, n_symbols: usize) -> Array1<f64> {
    let n = vals.len();
    if n == 0 || n_symbols == 0 {
        return Array1::zeros(0);
    }
    let n_dates = n / n_symbols;
    let mut result = Array1::from_elem(n, f64::NAN);

    for d in 0..n_dates {
        let start = d * n_symbols;
        let end = start + n_symbols;
        let slice = &vals.as_slice().unwrap()[start..end];

        // Pass 1: count + sum + sum_sq (no allocation)
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0usize;
        for &v in slice {
            if v.is_finite() {
                sum += v;
                sum_sq += v * v;
                count += 1;
            }
        }

        if count == 0 {
            for (j, &v) in slice.iter().enumerate() {
                result[start + j] = v;
            }
            continue;
        }

        let mean = sum / count as f64;
        let variance = if count > 1 {
            // Var = (Σx² - (Σx)²/n) / (n-1) — single-pass using sum + sum_sq
            let ss = sum_sq - sum * sum / count as f64;
            if ss > 0.0 { ss / (count - 1) as f64 } else { 0.0 }
        } else {
            0.0
        };
        let std = variance.sqrt();

        let lower = mean - 3.0 * std;
        let upper = mean + 3.0 * std;

        for (j, &v) in slice.iter().enumerate() {
            result[start + j] = if v.is_finite() {
                v.clamp(lower, upper)
            } else {
                v
            };
        }
    }
    result
}

/// Zscore: (x - mean) / std per cross-section (per date)
/// Data is flattened as (n_dates * n_symbols), n_symbols is the number of assets per date
/// NaN values are excluded from mean/std computation but remain as NaN in output
pub fn zscore(vals: &Array1<f64>, n_symbols: usize) -> Array1<f64> {
    let n = vals.len();
    if n == 0 || n_symbols == 0 {
        return Array1::zeros(0);
    }
    let n_dates = n / n_symbols;
    let mut result = Array1::from_elem(n, f64::NAN);

    for d in 0..n_dates {
        let start = d * n_symbols;
        let end = start + n_symbols;
        let slice = &vals.as_slice().unwrap()[start..end];

        // Pass 1: count + sum + sum_sq
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0usize;
        for &v in slice {
            if v.is_finite() {
                sum += v;
                sum_sq += v * v;
                count += 1;
            }
        }

        if count == 0 {
            for (j, &v) in slice.iter().enumerate() {
                result[start + j] = v;
            }
            continue;
        }

        let mean = sum / count as f64;
        let variance = if count > 1 {
            let ss = sum_sq - sum * sum / count as f64;
            if ss > 0.0 { ss / (count - 1) as f64 } else { 0.0 }
        } else {
            0.0
        };
        let std = variance.sqrt();

        if std < 1e-10 {
            for (j, _) in slice.iter().enumerate() {
                result[start + j] = f64::NAN;
            }
        } else {
            for (j, &v) in slice.iter().enumerate() {
                result[start + j] = if v.is_finite() { (v - mean) / std } else { v };
            }
        }
    }
    result
}

/// Market cap neutralization: regress alpha on log(market_cap), return standardized residuals
/// Data is flattened as (n_dates * n_symbols), n_symbols is the number of assets per date
/// market_cap: market cap values per asset (used across all dates)
/// NaN values in alpha are excluded from regression but remain as NaN in output
pub fn cap_neu(vals: &Array1<f64>, market_cap: &Array1<f64>, n_symbols: usize) -> Array1<f64> {
    let n = vals.len();
    if n == 0 || n_symbols == 0 || market_cap.len() != n_symbols {
        return Array1::zeros(0);
    }
    let n_dates = n / n_symbols;
    let mut result = Array1::from_elem(n, f64::NAN);

    // Pre-compute log(market_cap)
    let log_mktcap: Array1<f64> =
        market_cap.mapv(|v| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY });

    for d in 0..n_dates {
        let start = d * n_symbols;
        let end = start + n_symbols;
        let alpha_slice = &vals.as_slice().unwrap()[start..end];

        let valid_indices: Vec<usize> = (0..n_symbols)
            .filter(|&i| alpha_slice[i].is_finite() && log_mktcap[i].is_finite())
            .collect();

        if valid_indices.len() < 10 {
            for (j, &v) in alpha_slice.iter().enumerate() {
                result[start + j] = v;
            }
            continue;
        }

        let n_valid = valid_indices.len() as f64;

        let alpha_sum: f64 = valid_indices.iter().map(|&i| alpha_slice[i]).sum();
        let alpha_mean = alpha_sum / n_valid;

        let log_mktcap_sum: f64 = valid_indices.iter().map(|&i| log_mktcap[i]).sum();
        let log_mktcap_mean = log_mktcap_sum / n_valid;

        let cov: f64 = valid_indices
            .iter()
            .map(|&i| (alpha_slice[i] - alpha_mean) * (log_mktcap[i] - log_mktcap_mean))
            .sum::<f64>()
            / n_valid;

        let var_log_mktcap: f64 = valid_indices
            .iter()
            .map(|&i| (log_mktcap[i] - log_mktcap_mean).powi(2))
            .sum::<f64>()
            / n_valid;

        let beta = if var_log_mktcap > 1e-10 {
            cov / var_log_mktcap
        } else {
            0.0
        };

        let residuals: Array1<f64> = alpha_slice
            .iter()
            .zip(log_mktcap.iter())
            .map(|(&a, &l)| a - alpha_mean - beta * (l - log_mktcap_mean))
            .collect();

        let valid_residuals: Array1<f64> = valid_indices.iter().map(|&i| residuals[i]).collect();
        let residual_sum: f64 = valid_residuals.iter().sum();
        let residual_mean = residual_sum / n_valid;
        let residual_var = valid_residuals
            .iter()
            .map(|&r| (r - residual_mean).powi(2))
            .sum::<f64>()
            / n_valid;
        let residual_std = residual_var.sqrt();

        if residual_std > 1e-10 {
            for i in 0..n_symbols {
                if alpha_slice[i].is_finite() && log_mktcap[i].is_finite() {
                    result[start + i] = (residuals[i] - residual_mean) / residual_std;
                } else {
                    result[start + i] = f64::NAN;
                }
            }
        } else {
            for (j, &v) in alpha_slice.iter().enumerate() {
                result[start + j] = v;
            }
        }
    }
    result
}

/// Rolling mean (simple moving average). NaN values in the window are excluded.
pub fn ts_mean(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];
        let mut sum = 0.0;
        let mut count = 0usize;
        for &v in slice {
            if v.is_finite() {
                sum += v;
                count += 1;
            }
        }
        result[i] = if count > 0 { sum / count as f64 } else { f64::NAN };
    }
    result
}

/// Rolling sum
pub fn ts_sum(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);

    if window == 0 {
        let mut cumsum = 0.0;
        for i in 0..n {
            // Skip NaN — otherwise NaN propagates to all subsequent cumsum.
            if vals[i].is_finite() {
                cumsum += vals[i];
            }
            result[i] = cumsum;
        }
        return result;
    }

    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];
        result[i] = slice.iter().filter(|v| v.is_finite()).sum();
    }
    result
}

/// Rolling count of non-NaN values
pub fn ts_count(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);

    if window == 0 {
        let mut cumcount = 0.0;
        for i in 0..n {
            if !vals[i].is_nan() {
                cumcount += 1.0;
            }
            result[i] = cumcount;
        }
        return result;
    }

    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        result[i] = vals.as_slice().unwrap()[start..=i]
            .iter()
            .filter(|v| !v.is_nan())
            .count() as f64;
    }
    result
}

/// Rolling maximum. Returns NaN if the window contains no finite values.
pub fn ts_max(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];
        let max_val = slice.iter().filter(|v| v.is_finite()).fold(f64::NAN, |a, &b| {
            if a.is_nan() { b } else { a.max(b) }
        });
        result[i] = max_val;
    }
    result
}

/// Rolling minimum. Returns NaN if the window contains no finite values.
pub fn ts_min(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];
        let min_val = slice.iter().filter(|v| v.is_finite()).fold(f64::NAN, |a, &b| {
            if a.is_nan() { b } else { a.min(b) }
        });
        result[i] = min_val;
    }
    result
}

/// Rolling standard deviation. NaN values in the window are excluded.
pub fn ts_std(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];
        // Pass 1: count + sum
        let mut sum = 0.0;
        let mut count = 0usize;
        for &v in slice {
            if v.is_finite() {
                sum += v;
                count += 1;
            }
        }
        if count < 2 {
            result[i] = f64::NAN;
            continue;
        }
        let mean = sum / count as f64;
        // Pass 2: variance
        let mut sum_sq = 0.0;
        for &v in slice {
            if v.is_finite() {
                let d = v - mean;
                sum_sq += d * d;
            }
        }
        result[i] = (sum_sq / count as f64).sqrt();
    }
    result
}

/// Rolling rank (percentile rank of current value in window)
pub fn ts_rank(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];
        let current = vals[i];
        let rank = slice.iter().filter(|&&x| x < current).count() as f64;
        result[i] = rank / slice.len() as f64;
    }
    result
}

/// Position of maximum value in window (1-indexed). NaN if no finite values.
pub fn ts_argmax(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];
        let max_val = slice.iter().filter(|v| v.is_finite()).fold(f64::NAN, |a, &b| {
            if a.is_nan() { b } else { a.max(b) }
        });
        if max_val.is_nan() {
            result[i] = f64::NAN;
            continue;
        }
        let pos = slice.iter().position(|&x| (x - max_val).abs() < f64::EPSILON);
        result[i] = pos.map(|p| (slice.len() - p) as f64).unwrap_or(f64::NAN);
    }
    result
}

/// Position of minimum value in window (1-indexed). NaN if no finite values.
pub fn ts_argmin(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];
        let min_val = slice.iter().filter(|v| v.is_finite()).fold(f64::NAN, |a, &b| {
            if a.is_nan() { b } else { a.min(b) }
        });
        if min_val.is_nan() {
            result[i] = f64::NAN;
            continue;
        }
        let pos = slice.iter().position(|&x| (x - min_val).abs() < f64::EPSILON);
        result[i] = pos.map(|p| (slice.len() - p) as f64).unwrap_or(f64::NAN);
    }
    result
}

/// Rolling correlation between two series. Pairs where either value is NaN are excluded.
pub fn ts_correlation(vals1: &Array1<f64>, vals2: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals1.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice1 = &vals1.as_slice().unwrap()[start..=i];
        let slice2 = &vals2.as_slice().unwrap()[start..=i];

        // Pass 1: count + sums
        let mut sum1 = 0.0;
        let mut sum2 = 0.0;
        let mut count = 0usize;
        for (a, b) in slice1.iter().zip(slice2.iter()) {
            if a.is_finite() && b.is_finite() {
                sum1 += a;
                sum2 += b;
                count += 1;
            }
        }
        if count < 2 {
            result[i] = f64::NAN;
            continue;
        }
        let mean1 = sum1 / count as f64;
        let mean2 = sum2 / count as f64;

        // Pass 2: covariance + variances
        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;
        for (a, b) in slice1.iter().zip(slice2.iter()) {
            if a.is_finite() && b.is_finite() {
                let d1 = a - mean1;
                let d2 = b - mean2;
                cov += d1 * d2;
                var1 += d1 * d1;
                var2 += d2 * d2;
            }
        }

        let denom = (var1 * var2).sqrt();
        result[i] = if denom.is_finite() && denom > 1e-10 { cov / denom } else { f64::NAN };
    }
    result
}

/// Rolling covariance between two series. Pairs where either value is NaN are excluded.
pub fn ts_cov(vals1: &Array1<f64>, vals2: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals1.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice1 = &vals1.as_slice().unwrap()[start..=i];
        let slice2 = &vals2.as_slice().unwrap()[start..=i];

        // Pass 1: count + sums
        let mut sum1 = 0.0;
        let mut sum2 = 0.0;
        let mut count = 0usize;
        for (a, b) in slice1.iter().zip(slice2.iter()) {
            if a.is_finite() && b.is_finite() {
                sum1 += a;
                sum2 += b;
                count += 1;
            }
        }
        if count < 2 {
            result[i] = f64::NAN;
            continue;
        }
        let mean1 = sum1 / count as f64;
        let mean2 = sum2 / count as f64;

        // Pass 2: covariance
        let mut cov = 0.0;
        for (a, b) in slice1.iter().zip(slice2.iter()) {
            if a.is_finite() && b.is_finite() {
                cov += (a - mean1) * (b - mean2);
            }
        }
        result[i] = cov / (count - 1) as f64;
    }
    result
}

/// Exponential weighted moving average (EMA/SMA)
/// alpha: smoothing factor (0 < alpha <= 1)
pub fn sma(vals: &Array1<f64>, alpha: f64) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);
    if n == 0 {
        return result;
    }

    // Skip leading NaN — otherwise EMA recurrence propagates NaN forever.
    let mut start = 0;
    while start < n && vals[start].is_nan() {
        result[start] = f64::NAN;
        start += 1;
    }
    if start >= n {
        return result;
    }

    result[start] = vals[start];

    for i in (start + 1)..n {
        if vals[i].is_nan() {
            result[i] = result[i - 1];
        } else {
            result[i] = alpha * vals[i] + (1.0 - alpha) * result[i - 1];
        }
    }
    result
}

/// Days since minimum value in window. NaN if no finite values in window.
pub fn lowday(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];

        if slice.is_empty() {
            result[i] = f64::NAN;
            continue;
        }

        let finite: Vec<(usize, &f64)> = slice.iter().enumerate().filter(|(_, v)| v.is_finite()).collect();
        if finite.is_empty() {
            result[i] = f64::NAN;
            continue;
        }

        let min_pos = finite
            .iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| *idx)
            .unwrap_or(0);

        result[i] = (slice.len() - 1 - min_pos) as f64;
    }
    result
}

/// Days since maximum value in window. NaN if no finite values in window.
pub fn highday(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];

        if slice.is_empty() {
            result[i] = f64::NAN;
            continue;
        }

        let finite: Vec<(usize, &f64)> = slice.iter().enumerate().filter(|(_, v)| v.is_finite()).collect();
        if finite.is_empty() {
            result[i] = f64::NAN;
            continue;
        }

        let max_pos = finite
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| *idx)
            .unwrap_or(0);

        result[i] = (slice.len() - 1 - max_pos) as f64;
    }
    result
}

/// Weighted moving average. NaN values are excluded from the weighted sum.
pub fn wma(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);

    if window == 0 {
        return result;
    }

    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];
        let len = slice.len();

        if len == 0 {
            result[i] = f64::NAN;
            continue;
        }

        let mut sum_weighted = 0.0;
        let mut sum_weights = 0.0;

        for (j, &val) in slice.iter().enumerate() {
            if val.is_finite() {
                let weight = (0.9_f64).powi((len - 1 - j) as i32);
                sum_weighted += val * weight;
                sum_weights += weight;
            }
        }

        result[i] = if sum_weights > 0.0 {
            sum_weighted / sum_weights
        } else {
            f64::NAN
        };
    }
    result
}

/// Rolling product. NaN values in the window are excluded.
pub fn ts_product(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];
        let mut prod = 1.0;
        let mut any = false;
        for &v in slice {
            if v.is_finite() {
                prod *= v;
                any = true;
            }
        }
        result[i] = if any { prod } else { f64::NAN };
    }
    result
}

/// Decay linear (linearly weighted moving average). NaN values are excluded.
pub fn decay_linear(vals: &Array1<f64>, periods: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(periods.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];
        let len = slice.len();

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (j, &v) in slice.iter().enumerate() {
            if v.is_finite() {
                let weight = (j + 1) as f64;
                weighted_sum += weight * v;
                weight_sum += weight;
            }
        }

        result[i] = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            f64::NAN
        };
    }
    result
}

/// Cross-sectional rank (average tie-breaking, matching Polars `rank(method="average")`).
///
/// Returns fractional ranks in [0, 1), where tied values receive the same average rank.
pub fn rank(vals: &Array1<f64>) -> Array1<f64> {
    let n = vals.len();
    if n == 0 {
        return Array1::zeros(0);
    }

    // Separate NaN values and valid values
    let mut indexed: Vec<(usize, f64)> = vals
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, &v)| (i, v))
        .collect();

    if indexed.is_empty() {
        return Array1::from_elem(n, f64::NAN);
    }

    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = Array1::from_elem(n, f64::NAN);
    let len = indexed.len() as f64;

    let mut i = 0usize;
    while i < indexed.len() {
        let mut j = i + 1;
        while j < indexed.len() && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        // Average 0-based position for tied values: (i + (j-1)) / 2 = (i + j - 1) / 2
        let avg_pos = (i + j - 1) as f64 / 2.0;
        let pct = avg_pos / len;
        for k in i..j {
            result[indexed[k].0] = pct;
        }
        i = j;
    }
    result
}

/// Cross-sectional quantile cut
pub fn qcut(vals: &Array1<f64>, n_bins: i32) -> Vec<Option<i32>> {
    let ranks = rank(vals);
    ranks
        .iter()
        .map(|&r| {
            if r.is_finite() {
                let q = (r * n_bins as f64) as i32;
                Some(q.clamp(0, n_bins - 1))
            } else {
                None
            }
        })
        .collect()
}

/// Shift values by periods. First `periods` positions are NaN.
pub fn delay(vals: &Array1<f64>, periods: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::from_elem(n, f64::NAN);
    for i in periods..n {
        result[i] = vals[i - periods];
    }
    result
}

/// Difference over window periods. First `periods` positions are NaN.
pub fn ts_delta(vals: &Array1<f64>, periods: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::from_elem(n, f64::NAN);
    for i in periods..n {
        result[i] = vals[i] - vals[i - periods];
    }
    result
}

/// Scale values to have mean 0 and std 1. NaN values are left as NaN.
pub fn scale(vals: &Array1<f64>) -> Array1<f64> {
    let n = vals.len();
    if n == 0 {
        return Array1::zeros(0);
    }
    // Compute mean/std on finite values only — a single NaN would otherwise
    // make mean=NaN, std=NaN, and mapv would turn ALL output to NaN.
    let valid: Vec<f64> = vals.iter().filter(|v| v.is_finite()).copied().collect();
    let n_valid = valid.len();
    if n_valid < 2 {
        return Array1::from_elem(n, f64::NAN);
    }
    let sum: f64 = valid.iter().sum();
    let mean = sum / n_valid as f64;
    let variance = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_valid as f64;
    let std = variance.sqrt();
    if std < 1e-10 {
        return Array1::from_elem(n, f64::NAN);
    }
    vals.mapv(|v| if v.is_finite() { (v - mean) / std } else { v })
}

/// Sign of values
pub fn sign(vals: &Array1<f64>) -> Array1<f64> {
    vals.mapv(|v| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

/// Conditional: if a > threshold, return b, else return c
pub fn quesval(threshold: f64, a: &Array1<f64>, b: &Array1<f64>, c: &Array1<f64>) -> Array1<f64> {
    let n = a.len();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        result[i] = if a[i] > threshold { b[i] } else { c[i] };
    }
    result
}

/// Conditional: if a > b, return c, else return d
pub fn quesval2(a: &Array1<f64>, b: &Array1<f64>, c: &Array1<f64>, d: &Array1<f64>) -> Array1<f64> {
    let n = a.len();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        result[i] = if a[i] > b[i] { c[i] } else { d[i] };
    }
    result
}

/// Rolling quantile (linear interpolation)
pub fn ts_quantile(vals: &Array1<f64>, window: usize, quantile: f64) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);
    let q = quantile.clamp(0.0, 1.0);

    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];
        let len = slice.len();

        if len == 0 {
            result[i] = 0.0;
            continue;
        }

        let mut sorted: Vec<f64> = slice.iter().filter(|v| v.is_finite()).copied().collect();
        if sorted.is_empty() {
            result[i] = f64::NAN;
            continue;
        }
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let pos = q * (sorted.len() - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi || hi >= sorted.len() {
            result[i] = sorted[lo.min(sorted.len() - 1)];
        } else {
            let frac = pos - lo as f64;
            result[i] = sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
        }
    }
    result
}

/// Rolling linear regression slope: β = Cov(y, t) / Var(t).
/// Pairs where y is NaN are excluded.
pub fn ts_slope(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];

        // Pass 1: count + sums
        let mut sum_t = 0.0;
        let mut sum_y = 0.0;
        let mut count = 0usize;
        for (j, &y) in slice.iter().enumerate() {
            if y.is_finite() {
                sum_t += j as f64;
                sum_y += y;
                count += 1;
            }
        }
        if count < 2 {
            result[i] = f64::NAN;
            continue;
        }
        let t_mean = sum_t / count as f64;
        let y_mean = sum_y / count as f64;

        // Pass 2: cov + var_t
        let mut cov = 0.0;
        let mut var_t = 0.0;
        for (j, &y) in slice.iter().enumerate() {
            if y.is_finite() {
                let dt = j as f64 - t_mean;
                cov += dt * (y - y_mean);
                var_t += dt * dt;
            }
        }
        result[i] = if var_t.is_finite() && var_t > 1e-12 {
            cov / var_t
        } else {
            f64::NAN
        };
    }
    result
}

/// Rolling regression R² = 1 - SS_res / SS_tot
pub fn ts_rsquare(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];
        let len = slice.len();

        if len < 2 {
            result[i] = 0.0;
            continue;
        }

        let t_mean = (len - 1) as f64 / 2.0;
        let y_mean = slice.iter().sum::<f64>() / len as f64;

        let mut cov = 0.0;
        let mut var_t = 0.0;
        for (j, &y) in slice.iter().enumerate() {
            let dt = j as f64 - t_mean;
            cov += dt * (y - y_mean);
            var_t += dt * dt;
        }

        let beta = if var_t > 1e-12 { cov / var_t } else { 0.0 };
        let intercept = y_mean - beta * t_mean;

        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for (j, &y) in slice.iter().enumerate() {
            let y_pred = intercept + beta * j as f64;
            ss_res += (y - y_pred).powi(2);
            ss_tot += (y - y_mean).powi(2);
        }

        result[i] = if ss_tot.is_finite() && ss_tot > 1e-12 {
            1.0 - ss_res / ss_tot
        } else {
            f64::NAN
        };
    }
    result
}

/// Rolling regression residual. NaN values are excluded from the regression.
pub fn ts_resi(vals: &Array1<f64>, window: usize) -> Array1<f64> {
    let n = vals.len();
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(window.saturating_sub(1));
        let slice = &vals.as_slice().unwrap()[start..=i];

        let last_val = slice[slice.len() - 1];
        if !last_val.is_finite() {
            result[i] = f64::NAN;
            continue;
        }

        // Pass 1: count + sums
        let mut sum_t = 0.0;
        let mut sum_y = 0.0;
        let mut count = 0usize;
        for (j, &y) in slice.iter().enumerate() {
            if y.is_finite() {
                sum_t += j as f64;
                sum_y += y;
                count += 1;
            }
        }
        if count < 2 {
            result[i] = f64::NAN;
            continue;
        }
        let t_mean = sum_t / count as f64;
        let y_mean = sum_y / count as f64;

        // Pass 2: cov + var_t
        let mut cov = 0.0;
        let mut var_t = 0.0;
        for (j, &y) in slice.iter().enumerate() {
            if y.is_finite() {
                let dt = j as f64 - t_mean;
                cov += dt * (y - y_mean);
                var_t += dt * dt;
            }
        }

        let beta = if var_t.is_finite() && var_t > 1e-12 { cov / var_t } else { f64::NAN };
        if beta.is_nan() {
            result[i] = f64::NAN;
            continue;
        }
        let intercept = y_mean - beta * t_mean;
        let y_pred = intercept + beta * (slice.len() - 1) as f64;

        result[i] = last_val - y_pred;
    }
    result
}

/// Apply a per-symbol time-series function to data ordered as (symbol, date).
///
/// Each symbol has `n_dates` consecutive values in the flat array. The closure
/// receives each symbol's contiguous time series and returns the result for
/// that symbol. Results are reassembled into the same flat layout.
pub fn per_symbol_ts<F>(vals: &Array1<f64>, n_dates: usize, f: F) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let n_symbols = vals.len() / n_dates;
    let mut result = Array1::zeros(vals.len());
    for s in 0..n_symbols {
        let start = s * n_dates;
        let end = start + n_dates;
        let chunk = vals.slice(ndarray::s![start..end]).to_owned();
        let chunk_result = f(&chunk);
        let src = chunk_result.as_slice().unwrap();
        result.as_slice_mut().unwrap()[start..end].copy_from_slice(src);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ts_mean() {
        let vals = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = ts_mean(&vals, 3);
        assert_eq!(result.to_vec(), vec![1.0, 1.5, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ts_sum() {
        let vals = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = ts_sum(&vals, 3);
        assert_eq!(result.to_vec(), vec![1.0, 3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_ts_max() {
        let vals = Array1::from_vec(vec![1.0, 5.0, 3.0, 2.0, 4.0]);
        let result = ts_max(&vals, 3);
        assert_eq!(result.to_vec(), vec![1.0, 5.0, 5.0, 5.0, 4.0]);
    }

    #[test]
    fn test_ts_min() {
        let vals = Array1::from_vec(vec![3.0, 1.0, 5.0, 2.0, 4.0]);
        let result = ts_min(&vals, 3);
        assert_eq!(result.to_vec(), vec![3.0, 1.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn test_rank() {
        let vals = Array1::from_vec(vec![3.0, 1.0, 2.0, 5.0, 4.0]);
        let result = rank(&vals);
        assert_eq!(result.to_vec(), vec![0.4, 0.0, 0.2, 0.8, 0.6]);
    }

    #[test]
    fn test_delay() {
        let vals = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = delay(&vals, 2);
        assert_eq!(result.to_vec(), vec![0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_scale() {
        let vals = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = scale(&vals);
        let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
        assert!((mean - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_sign() {
        let vals = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = sign(&vals);
        assert_eq!(result.to_vec(), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_delta() {
        let vals = Array1::from_vec(vec![10.0, 12.0, 11.0, 15.0, 14.0]);
        let result = ts_delta(&vals, 1);
        assert_eq!(result.to_vec(), vec![0.0, 2.0, -1.0, 4.0, -1.0]);
    }

    #[test]
    fn test_winsor() {
        let vals = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]);
        let result = winsor(&vals, 5);
        assert_eq!(result.to_vec(), vals.to_vec());
    }

    #[test]
    fn test_winsor_with_outliers() {
        let vals = Array1::from_vec(vec![1.0, 2.0, 100.0, 4.0, 5.0]);
        let result = winsor(&vals, 5);
        assert_eq!(result.to_vec(), vals.to_vec());
    }

    #[test]
    fn test_zscore() {
        let vals = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        let result = zscore(&vals, 5);

        let date0 = &result.as_slice().unwrap()[0..5];
        let mean0: f64 = date0.iter().sum::<f64>() / 5.0;
        let std0_sample = (date0.iter().map(|x| x.powi(2)).sum::<f64>() / 4.0).sqrt();
        assert!((mean0 - 0.0).abs() < 1e-10);
        assert!((std0_sample - 1.0).abs() < 0.01);

        let date1 = &result.as_slice().unwrap()[5..10];
        let mean1: f64 = date1.iter().sum::<f64>() / 5.0;
        let std1_sample = (date1.iter().map(|x| x.powi(2)).sum::<f64>() / 4.0).sqrt();
        assert!((mean1 - 0.0).abs() < 1e-10);
        assert!((std1_sample - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cap_neu() {
        // Need >= 10 values for OLS (matches Python min_samples=10)
        let n = 12;
        let mut vals_vec = Vec::with_capacity(n);
        let mut mktcap_vec = Vec::with_capacity(n);
        for i in 0..n {
            vals_vec.push((i + 1) as f64 * 10.0);
            mktcap_vec.push((i + 1) as f64);
        }
        let vals = Array1::from_vec(vals_vec);
        let market_cap = Array1::from_vec(mktcap_vec);
        let result = cap_neu(&vals, &market_cap, n);

        let mean: f64 = result.iter().sum::<f64>() / n as f64;
        assert!((mean - 0.0).abs() < 1e-9);
        assert!(result.iter().any(|x| x.abs() > 0.01));
    }
}
