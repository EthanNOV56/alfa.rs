//! Time series functions for factor computation

/// Winsorize: clip values to [mean - 3*std, mean + 3*std] per cross-section (per date)
/// Data is flattened as (n_dates * n_symbols), n_symbols is the number of assets per date
/// NaN values are excluded from mean/std computation but remain as NaN in output
pub fn winsor(vals: &[f64], n_symbols: usize) -> Vec<f64> {
    let n = vals.len();
    if n == 0 || n_symbols == 0 {
        return vec![];
    }
    let n_dates = n / n_symbols;
    let mut result = Vec::with_capacity(n);

    for d in 0..n_dates {
        let start = d * n_symbols;
        let end = start + n_symbols;
        let slice = &vals[start..end];

        // Filter out NaN values for computing mean and std
        let valid_vals: Vec<f64> = slice.iter().filter(|v| v.is_finite()).copied().collect();
        let n_valid = valid_vals.len();

        if n_valid == 0 {
            // No valid values, keep all as NaN
            for &v in slice {
                result.push(v); // NaN stays NaN
            }
            continue;
        }

        // Compute mean and std on valid values (sample std, ddof=1, matches Polars default)
        let sum: f64 = valid_vals.iter().sum();
        let mean = sum / n_valid as f64;
        let variance = if n_valid > 1 {
            valid_vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n_valid - 1) as f64
        } else {
            0.0
        };
        let std = variance.sqrt();

        let lower = mean - 3.0 * std;
        let upper = mean + 3.0 * std;

        // Clip each value (NaN stays as NaN)
        for &v in slice {
            if v.is_finite() {
                result.push(v.clamp(lower, upper));
            } else {
                result.push(v);
            }
        }
    }
    result
}

/// Zscore: (x - mean) / std per cross-section (per date)
/// Data is flattened as (n_dates * n_symbols), n_symbols is the number of assets per date
/// NaN values are excluded from mean/std computation but remain as NaN in output
pub fn zscore(vals: &[f64], n_symbols: usize) -> Vec<f64> {
    let n = vals.len();
    if n == 0 || n_symbols == 0 {
        return vec![];
    }
    let n_dates = n / n_symbols;
    let mut result = Vec::with_capacity(n);

    for d in 0..n_dates {
        let start = d * n_symbols;
        let end = start + n_symbols;
        let slice = &vals[start..end];

        // Filter out NaN values for computing mean and std
        let valid_vals: Vec<f64> = slice.iter().filter(|v| v.is_finite()).copied().collect();
        let n_valid = valid_vals.len();

        if n_valid == 0 {
            // No valid values, keep all as NaN
            for &v in slice {
                result.push(v); // NaN stays NaN
            }
            continue;
        }

        // Compute mean and std on valid values (sample std, ddof=1, matches Polars default)
        let sum: f64 = valid_vals.iter().sum();
        let mean = sum / n_valid as f64;
        let variance = if n_valid > 1 {
            valid_vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n_valid - 1) as f64
        } else {
            0.0
        };
        let std = variance.sqrt();

        if std < 1e-10 {
            // If std is near zero, return NaN for all (or zeros)
            for &v in slice {
                result.push(f64::NAN);
            }
        } else {
            // Zscore each value (NaN stays as NaN)
            for &v in slice {
                if v.is_finite() {
                    result.push((v - mean) / std);
                } else {
                    result.push(v);
                }
            }
        }
    }
    result
}

/// Market cap neutralization: regress alpha on log(market_cap), return standardized residuals
/// Data is flattened as (n_dates * n_symbols), n_symbols is the number of assets per date
/// market_cap: market cap values per asset (used across all dates)
/// NaN values in alpha are excluded from regression but remain as NaN in output
pub fn cap_neu(vals: &[f64], market_cap: &[f64], n_symbols: usize) -> Vec<f64> {
    let n = vals.len();
    if n == 0 || n_symbols == 0 || market_cap.len() != n_symbols {
        return vec![];
    }
    let n_dates = n / n_symbols;
    let mut result = Vec::with_capacity(n);

    // Pre-compute log(market_cap)
    // Use NEG_INFINITY for non-positive values so they are filtered out by is_finite check,
    // matching Python's behavior: np.log(negative/zero) = NaN/-inf, both not finite.
    let log_mktcap: Vec<f64> = market_cap
        .iter()
        .map(|&v| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY })
        .collect();

    for d in 0..n_dates {
        let start = d * n_symbols;
        let end = start + n_symbols;
        let alpha_slice = &vals[start..end];

        // Filter to indices where BOTH alpha and log_mktcap are finite
        // Matches Python: np.isfinite(factor_values) & np.isfinite(log_market_cap)
        let valid_indices: Vec<usize> = (0..n_symbols)
            .filter(|&i| alpha_slice[i].is_finite() && log_mktcap[i].is_finite())
            .collect();

        if valid_indices.len() < 2 {
            // Not enough valid values for regression, keep all as NaN
            for &v in alpha_slice {
                result.push(v); // NaN stays NaN
            }
            continue;
        }

        let n_valid = valid_indices.len() as f64;

        // Compute means on valid values only
        let alpha_sum: f64 = valid_indices.iter().map(|&i| alpha_slice[i]).sum();
        let alpha_mean = alpha_sum / n_valid;

        let log_mktcap_sum: f64 = valid_indices.iter().map(|&i| log_mktcap[i]).sum();
        let log_mktcap_mean = log_mktcap_sum / n_valid;

        // Covariance: E[(alpha - alpha_mean) * (log_mktcap - log_mktcap_mean)] over valid set
        let cov: f64 = valid_indices
            .iter()
            .map(|&i| (alpha_slice[i] - alpha_mean) * (log_mktcap[i] - log_mktcap_mean))
            .sum::<f64>() / n_valid;

        // Variance of log_mktcap over the SAME valid set
        let var_log_mktcap: f64 = valid_indices
            .iter()
            .map(|&i| (log_mktcap[i] - log_mktcap_mean).powi(2))
            .sum::<f64>() / n_valid;

        let beta = if var_log_mktcap > 1e-10 {
            cov / var_log_mktcap
        } else {
            0.0
        };

        // Compute residuals: alpha - (alpha_mean + beta * (log_mktcap - log_mktcap_mean))
        // This is equivalent to OLS with intercept: alpha = b0 + b1*log_mktcap + residual
        // where b0 = alpha_mean - beta * log_mktcap_mean
        let residuals: Vec<f64> = alpha_slice
            .iter()
            .zip(log_mktcap.iter())
            .map(|(&a, &l)| a - alpha_mean - beta * (l - log_mktcap_mean))
            .collect();

        // Standardize residuals on valid set only (matches numpy .std(ddof=0))
        let valid_residuals: Vec<f64> = valid_indices.iter().map(|&i| residuals[i]).collect();
        let residual_sum: f64 = valid_residuals.iter().sum();
        let residual_mean = residual_sum / n_valid;
        let residual_var = valid_residuals
            .iter()
            .map(|&r| (r - residual_mean).powi(2))
            .sum::<f64>() / n_valid;
        let residual_std = residual_var.sqrt();

        // Standardized residuals (NaN values stay as NaN)
        if residual_std > 1e-10 {
            for i in 0..n_symbols {
                if alpha_slice[i].is_finite() && log_mktcap[i].is_finite() {
                    result.push((residuals[i] - residual_mean) / residual_std);
                } else {
                    result.push(f64::NAN);
                }
            }
        } else {
            for &v in alpha_slice {
                result.push(v); // Keep original (NaN stays NaN)
            }
        }
    }
    result
}

/// Rolling mean (simple moving average)
pub fn ts_mean(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        result[i] = slice.iter().sum::<f64>() / slice.len() as f64;
    }
    result
}

/// Rolling sum
pub fn ts_sum(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    // Optimized path for expanding window (window=0 means from start)
    if window == 0 {
        let mut cumsum = 0.0;
        for i in 0..n {
            cumsum += vals[i];
            result[i] = cumsum;
        }
        return result;
    }

    // Regular rolling window
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        result[i] = vals[start..=i].iter().sum();
    }
    result
}

/// Rolling count of non-NaN values
pub fn ts_count(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    // Optimized path for expanding window (window=0 means from start)
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

    // Regular rolling window
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        // Count non-NaN values in the window
        result[i] = vals[start..=i].iter().filter(|v| !v.is_nan()).count() as f64;
    }
    result
}

/// Rolling maximum
pub fn ts_max(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        result[i] = vals[start..=i]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    }
    result
}

/// Rolling minimum
pub fn ts_min(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        result[i] = vals[start..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
    }
    result
}

/// Rolling standard deviation
pub fn ts_std(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;
        result[i] = variance.sqrt();
    }
    result
}

/// Rolling rank (percentile rank of current value in window)
pub fn ts_rank(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let current = vals[i];
        let rank = slice.iter().filter(|&&x| x < current).count() as f64;
        result[i] = rank / slice.len() as f64;
    }
    result
}

/// Position of maximum value in window (1-indexed like Alpha101)
pub fn ts_argmax(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let pos = slice
            .iter()
            .position(|&x| (x - max_val).abs() < f64::EPSILON);
        result[i] = pos.map(|p| (slice.len() - p) as f64).unwrap_or(0.0);
    }
    result
}

/// Position of minimum value in window (1-indexed like Alpha101)
pub fn ts_argmin(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let min_val = slice.iter().cloned().fold(f64::INFINITY, f64::min);
        let pos = slice
            .iter()
            .position(|&x| (x - min_val).abs() < f64::EPSILON);
        result[i] = pos.map(|p| (slice.len() - p) as f64).unwrap_or(0.0);
    }
    result
}

/// Rolling correlation between two series
pub fn ts_correlation(vals1: &[f64], vals2: &[f64], window: usize) -> Vec<f64> {
    let n = vals1.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice1 = &vals1[start..=i];
        let slice2 = &vals2[start..=i];

        let len = slice1.len();
        if len < 2 {
            result[i] = 0.0;
            continue;
        }

        let mean1: f64 = slice1.iter().sum::<f64>() / len as f64;
        let mean2: f64 = slice2.iter().sum::<f64>() / len as f64;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for j in 0..len {
            let d1 = slice1[j] - mean1;
            let d2 = slice2[j] - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom > 1e-10 {
            result[i] = cov / denom;
        } else {
            result[i] = 0.0;
        }
    }
    result
}

/// Rolling covariance between two series
pub fn ts_cov(vals1: &[f64], vals2: &[f64], window: usize) -> Vec<f64> {
    let n = vals1.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice1 = &vals1[start..=i];
        let slice2 = &vals2[start..=i];

        let len = slice1.len();
        if len < 2 {
            result[i] = 0.0;
            continue;
        }

        let mean1: f64 = slice1.iter().sum::<f64>() / len as f64;
        let mean2: f64 = slice2.iter().sum::<f64>() / len as f64;

        let mut cov = 0.0;
        for j in 0..len {
            let d1 = slice1[j] - mean1;
            let d2 = slice2[j] - mean2;
            cov += d1 * d2;
        }
        result[i] = cov / (len - 1) as f64; // Sample covariance
    }
    result
}

/// Exponential weighted moving average (EMA/SMA)
/// alpha: smoothing factor (0 < alpha <= 1)
pub fn sma(vals: &[f64], alpha: f64) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    if n == 0 {
        return result;
    }

    // Initialize with first value
    result[0] = vals[0];

    for i in 1..n {
        result[i] = alpha * vals[i] + (1.0 - alpha) * result[i - 1];
    }
    result
}

/// Days since minimum value in window
pub fn lowday(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];

        if slice.is_empty() {
            result[i] = 0.0;
            continue;
        }

        // Find position of minimum (from start of window)
        let min_pos = slice
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Days since low = position from end
        result[i] = (slice.len() - 1 - min_pos) as f64;
    }
    result
}

/// Days since maximum value in window
pub fn highday(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];

        if slice.is_empty() {
            result[i] = 0.0;
            continue;
        }

        // Find position of maximum (from start of window)
        let max_pos = slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Days since high = position from end
        result[i] = (slice.len() - 1 - max_pos) as f64;
    }
    result
}

/// Weighted moving average
pub fn wma(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    if window == 0 {
        return result;
    }

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let len = slice.len();

        if len == 0 {
            result[i] = 0.0;
            continue;
        }

        // Weights: 0.9^(len-1), 0.9^(len-2), ..., 0.9^0
        let mut sum_weighted = 0.0;
        let mut sum_weights = 0.0;

        for (j, &val) in slice.iter().enumerate() {
            let weight = (0.9_f64).powi((len - 1 - j) as i32);
            sum_weighted += val * weight;
            sum_weights += weight;
        }

        result[i] = if sum_weights > 0.0 {
            sum_weighted / sum_weights
        } else {
            0.0
        };
    }
    result
}

/// Difference over window periods (like delta in Alpha101)
pub fn ts_delta(vals: &[f64], periods: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in periods..n {
        result[i] = vals[i] - vals[i - periods];
    }
    result
}

/// Rolling product
pub fn ts_product(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let prod: f64 = slice.iter().fold(1.0, |acc, &v| acc * v);
        result[i] = prod;
    }
    result
}

/// Decay linear (exponentially weighted with linear weights)
pub fn decay_linear(vals: &[f64], periods: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(periods - 1);
        let slice = &vals[start..=i];
        let len = slice.len();

        // Linear weights: 1, 2, 3, ..., len
        let weight_sum: f64 = (1..=len).sum::<usize>() as f64;
        let mut weighted_sum = 0.0;

        for (j, &v) in slice.iter().enumerate() {
            let weight = (j + 1) as f64;
            weighted_sum += weight * v;
        }

        result[i] = weighted_sum / weight_sum;
    }
    result
}

/// Cross-sectional rank
pub fn rank(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    if n == 0 {
        return vec![];
    }

    // Separate NaN values and valid values
    let mut indexed: Vec<(usize, f64)> = vals
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .map(|(i, &v)| (i, v))
        .collect();

    if indexed.is_empty() {
        return vec![f64::NAN; n];
    }

    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = vec![f64::NAN; n];
    let len = indexed.len();
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        result[*idx] = rank as f64 / len as f64;
    }
    result
}

/// Cross-sectional quantile cut
pub fn qcut(vals: &[f64], n_bins: i32) -> Vec<Option<i32>> {
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

/// Shift values by periods
pub fn delay(vals: &[f64], periods: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    for i in periods..n {
        result[i] = vals[i - periods];
    }
    result
}

/// Scale values to have mean 0 and std 1
pub fn scale(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    if n == 0 {
        return vec![];
    }
    let sum: f64 = vals.iter().sum();
    let mean = sum / n as f64;
    let std = (vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    if std < 1e-10 {
        return vec![0.0; n];
    }
    vals.iter().map(|v| (v - mean) / std).collect()
}

/// Sign of values
pub fn sign(vals: &[f64]) -> Vec<f64> {
    vals.iter()
        .map(|v| {
            if *v > 0.0 {
                1.0
            } else if *v < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Conditional: if a > threshold, return b, else return c
pub fn quesval(threshold: f64, a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        result[i] = if a[i] > threshold { b[i] } else { c[i] };
    }
    result
}

/// Conditional: if a > b, return c, else return d
pub fn quesval2(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        result[i] = if a[i] > b[i] { c[i] } else { d[i] };
    }
    result
}

/// Rolling quantile (linear interpolation)
pub fn ts_quantile(vals: &[f64], window: usize, quantile: f64) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];
    let q = quantile.clamp(0.0, 1.0);

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let len = slice.len();

        if len == 0 {
            result[i] = 0.0;
            continue;
        }

        // Collect and sort valid values
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

/// Rolling linear regression slope: β = Cov(y, t) / Var(t)
/// where t = 0..window-1 are the time indices
pub fn ts_slope(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
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
            let t = j as f64;
            let dt = t - t_mean;
            cov += dt * (y - y_mean);
            var_t += dt * dt;
        }

        result[i] = if var_t > 1e-12 { cov / var_t } else { 0.0 };
    }
    result
}

/// Rolling regression R² = 1 - SS_res / SS_tot
pub fn ts_rsquare(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
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

        result[i] = if ss_tot > 1e-12 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };
    }
    result
}

/// Rolling regression residual: y_last - (intercept + slope * t_last)
pub fn ts_resi(vals: &[f64], window: usize) -> Vec<f64> {
    let n = vals.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let start = i.saturating_sub(window - 1);
        let slice = &vals[start..=i];
        let len = slice.len();

        if len < 2 {
            result[i] = 0.0;
            continue;
        }

        let last_val = slice[len - 1];
        let t_last = (len - 1) as f64;

        let t_mean = t_last / 2.0;
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
        let y_pred = intercept + beta * t_last;

        result[i] = last_val - y_pred;
    }
    result
}
