//! Time series functions for factor computation

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
