//! Series - Vectorized time series wrapper
//!
//! This module provides a Series struct for vectorized operations on time series data.

use ndarray::Array1;

// ============================================================================
// Series - Wrapper around ndarray for vectorized operations
// ============================================================================

/// A Series represents a time series of values with vectorized operations
#[derive(Debug, Clone)]
pub struct Series {
    data: Array1<f64>,
    name: Option<String>,
}

impl Series {
    /// Create a new Series from a vector
    pub fn new(data: Vec<f64>) -> Self {
        Series {
            data: Array1::from(data),
            name: None,
        }
    }

    /// Create a new Series from an ndarray
    pub fn from_array(data: Array1<f64>) -> Self {
        Series { data, name: None }
    }

    /// Create a new Series with a name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Get the length of the series
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the series is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the underlying data as a slice
    pub fn as_slice(&self) -> Option<&[f64]> {
        self.data.as_slice()
    }

    /// Get the underlying ndarray
    pub fn data(&self) -> &Array1<f64> {
        &self.data
    }

    /// Get the name of the series
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    // ==================== Vectorized Operations ====================

    /// Element-wise addition
    pub fn add(&self, other: &Series) -> Result<Series, String> {
        if self.len() != other.len() {
            return Err(format!(
                "Series length mismatch: {} != {}",
                self.len(),
                other.len()
            ));
        }
        Ok(Series::from_array(&self.data + &other.data))
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Series) -> Result<Series, String> {
        if self.len() != other.len() {
            return Err(format!(
                "Series length mismatch: {} != {}",
                self.len(),
                other.len()
            ));
        }
        Ok(Series::from_array(&self.data - &other.data))
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Series) -> Result<Series, String> {
        if self.len() != other.len() {
            return Err(format!(
                "Series length mismatch: {} != {}",
                self.len(),
                other.len()
            ));
        }
        Ok(Series::from_array(&self.data * &other.data))
    }

    /// Element-wise division
    pub fn div(&self, other: &Series) -> Result<Series, String> {
        if self.len() != other.len() {
            return Err(format!(
                "Series length mismatch: {} != {}",
                self.len(),
                other.len()
            ));
        }
        Ok(Series::from_array(&self.data / &other.data))
    }

    /// Element-wise comparison: greater than
    pub fn gt(&self, other: &Series) -> Result<Series, String> {
        if self.len() != other.len() {
            return Err(format!(
                "Series length mismatch: {} != {}",
                self.len(),
                other.len()
            ));
        }
        let result: Array1<f64> = Array1::from_iter(self.data.iter().zip(other.data.iter()).map(
            |(&a, &b)| {
                if a > b { 1.0 } else { 0.0 }
            },
        ));
        Ok(Series::from_array(result))
    }

    /// Element-wise absolute value
    pub fn abs(&self) -> Series {
        Series::from_array(self.data.mapv(f64::abs))
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Series {
        Series::from_array(self.data.mapv(|x| x.sqrt()))
    }

    /// Element-wise logarithm
    pub fn log(&self) -> Series {
        Series::from_array(self.data.mapv(|x| x.ln()))
    }

    /// Element-wise exponential
    pub fn exp(&self) -> Series {
        Series::from_array(self.data.mapv(f64::exp))
    }

    /// Element-wise negation
    pub fn neg(&self) -> Series {
        Series::from_array(-&self.data)
    }

    // ==================== Time Series Operations ====================

    /// Lag the series by N periods
    pub fn lag(&self, periods: usize) -> Series {
        let n = self.len();
        if periods >= n {
            return Series::new(vec![f64::NAN; n]);
        }

        let mut lagged = vec![f64::NAN; periods];
        lagged.extend_from_slice(&self.data.as_slice().unwrap()[..n - periods]);
        Series::new(lagged)
    }

    /// Difference: current value - lagged value
    pub fn diff(&self, periods: usize) -> Series {
        let lagged = self.lag(periods);
        self.sub(&lagged)
            .unwrap_or_else(|_| Series::new(vec![f64::NAN; self.len()]))
    }

    /// Percentage change: (current - lagged) / lagged
    pub fn pct_change(&self, periods: usize) -> Series {
        let lagged = self.lag(periods);
        let diff = self
            .sub(&lagged)
            .unwrap_or_else(|_| Series::new(vec![f64::NAN; self.len()]));
        diff.div(&lagged)
            .unwrap_or_else(|_| Series::new(vec![f64::NAN; self.len()]))
    }

    /// Simple moving average
    pub fn moving_average(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            let start = if i + 1 >= window { i + 1 - window } else { 0 };
            let slice = self.data.slice(ndarray::s![start..=i]);
            result[i] = if slice.len() == 0 {
                f64::NAN
            } else {
                slice.mean().unwrap_or(f64::NAN)
            };
        }

        Series::from_array(result)
    }

    /// Rolling standard deviation
    pub fn rolling_std(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            let start = if i + 1 >= window { i + 1 - window } else { 0 };
            let slice = self.data.slice(ndarray::s![start..=i]);
            result[i] = if slice.len() == 0 {
                f64::NAN
            } else {
                slice.std(1.0)
            };
        }

        Series::from_array(result)
    }

    /// Exponential moving average
    pub fn ema(&self, span: usize) -> Series {
        let n = self.len();
        let alpha = 2.0 / (span as f64 + 1.0);
        let mut result = Array1::zeros(n);

        if n > 0 {
            result[0] = self.data[0];
            for i in 1..n {
                result[i] = alpha * self.data[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        Series::from_array(result)
    }

    /// Z-score normalization
    pub fn z_score(&self, window: usize) -> Series {
        let mean = self.moving_average(window);
        let std = self.rolling_std(window);
        let z_data = (&self.data - &mean.data) / &std.data;
        Series::from_array(z_data)
    }

    // ==================== Alpha101 Functions ====================

    /// Time series rank over rolling window (ts_rank)
    pub fn ts_rank(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let slice = self.data.slice(ndarray::s![start..=i]);
                let mut vals: Vec<f64> = slice.iter().cloned().collect();
                // Sort and get rank of last element
                let last_val = vals[vals.len() - 1];
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                if let Some(pos) = vals.iter().position(|&x| x == last_val) {
                    result[i] = (pos + 1) as f64 / vals.len() as f64;
                } else {
                    result[i] = f64::NAN;
                }
            }
        }

        Series::from_array(result)
    }

    /// Time series argmax over rolling window (ts_argmax)
    pub fn ts_argmax(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let slice = self.data.slice(ndarray::s![start..=i]);
                // Find position of maximum (1-indexed)
                let max_pos = slice
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(pos, _)| pos + 1);
                result[i] = max_pos.unwrap_or(0) as f64;
            }
        }

        Series::from_array(result)
    }

    /// Time series argmin over rolling window (ts_argmin)
    pub fn ts_argmin(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let slice = self.data.slice(ndarray::s![start..=i]);
                // Find position of minimum (1-indexed)
                let min_pos = slice
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(pos, _)| pos + 1);
                result[i] = min_pos.unwrap_or(0) as f64;
            }
        }

        Series::from_array(result)
    }

    /// Cross-sectional rank (rank)
    pub fn cs_rank(&self) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        let mut vals: Vec<(usize, f64)> =
            self.data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        vals.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (orig_idx, _) in &vals {
            result[*orig_idx] = (orig_idx + 1) as f64 / n as f64;
        }

        Series::from_array(result)
    }

    /// Time series correlation over rolling window (ts_corr)
    pub fn ts_corr(&self, other: &Series, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let self_slice = self.data.slice(ndarray::s![start..=i]);
                let other_slice = other.data.slice(ndarray::s![start..=i]);

                // Calculate correlation
                let self_mean = self_slice.iter().sum::<f64>() / self_slice.len() as f64;
                let other_mean = other_slice.iter().sum::<f64>() / other_slice.len() as f64;

                let mut cov = 0.0;
                let mut self_var = 0.0;
                let mut other_var = 0.0;

                for j in 0..self_slice.len() {
                    let self_diff = self_slice[j] - self_mean;
                    let other_diff = other_slice[j] - other_mean;
                    cov += self_diff * other_diff;
                    self_var += self_diff * self_diff;
                    other_var += other_diff * other_diff;
                }

                let denom = (self_var * other_var).sqrt();
                if denom > 0.0 {
                    result[i] = cov / denom;
                } else {
                    result[i] = f64::NAN;
                }
            }
        }

        Series::from_array(result)
    }

    /// Time series covariance over rolling window (ts_cov)
    pub fn ts_cov(&self, other: &Series, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let self_slice = self.data.slice(ndarray::s![start..=i]);
                let other_slice = other.data.slice(ndarray::s![start..=i]);

                let self_mean = self_slice.iter().sum::<f64>() / self_slice.len() as f64;
                let other_mean = other_slice.iter().sum::<f64>() / other_slice.len() as f64;

                let mut cov = 0.0;
                for j in 0..self_slice.len() {
                    let self_diff = self_slice[j] - self_mean;
                    let other_diff = other_slice[j] - other_mean;
                    cov += self_diff * other_diff;
                }

                result[i] = cov / window as f64;
            }
        }

        Series::from_array(result)
    }

    /// Scale to [-1, 1] using rolling window (scale)
    pub fn scale(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let slice = self.data.slice(ndarray::s![start..=i]);
                let mean = slice.iter().sum::<f64>() / slice.len() as f64;
                let std = (slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / slice.len() as f64)
                    .sqrt();

                if std > 0.0 {
                    result[i] = (self.data[i] - mean) / std;
                } else {
                    result[i] = 0.0;
                }
            }
        }

        Series::from_array(result)
    }

    /// Linear decay weighted average (decay_linear)
    pub fn decay_linear(&self, periods: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        // Weights: 1, 2, 3, ..., periods
        let total_weight = (periods * (periods + 1)) as f64 / 2.0;

        for i in 0..n {
            if i + 1 < periods {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - periods;
                let slice = self.data.slice(ndarray::s![start..=i]);

                let mut weighted_sum = 0.0;
                for (j, &val) in slice.iter().enumerate() {
                    let weight = (j + 1) as f64;
                    weighted_sum += val * weight;
                }

                result[i] = weighted_sum / total_weight;
            }
        }

        Series::from_array(result)
    }

    /// Sign function (-1, 0, 1)
    pub fn sign(&self) -> Series {
        Series::from_array(self.data.mapv(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }))
    }

    /// Power function (x^exp)
    pub fn power(&self, exp: f64) -> Series {
        Series::from_array(self.data.mapv(|x| x.powf(exp)))
    }

    /// Rolling sum (ts_sum)
    /// window=0 means expanding window (from start to current)
    pub fn ts_sum(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        if window == 0 {
            // Expanding window: sum from start to current
            let mut cumsum = 0.0;
            for i in 0..n {
                cumsum += self.data[i];
                result[i] = cumsum;
            }
        } else {
            // Rolling window
            for i in 0..n {
                if i + 1 < window {
                    result[i] = f64::NAN;
                } else {
                    let start = i + 1 - window;
                    let slice = self.data.slice(ndarray::s![start..=i]);
                    result[i] = slice.iter().sum();
                }
            }
        }

        Series::from_array(result)
    }

    /// Rolling count (ts_count)
    /// window=0 means expanding window (count from start to current)
    pub fn ts_count(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        if window == 0 {
            // Expanding window: count from start to current
            for i in 0..n {
                result[i] = (i + 1) as f64;
            }
        } else {
            // Rolling window: count non-NaN values
            for i in 0..n {
                if i + 1 < window {
                    result[i] = f64::NAN;
                } else {
                    let start = i + 1 - window;
                    let slice = self.data.slice(ndarray::s![start..=i]);
                    let count = slice.iter().filter(|x| !x.is_nan()).count();
                    result[i] = count as f64;
                }
            }
        }

        Series::from_array(result)
    }

    /// Rolling max (ts_max)
    pub fn ts_max(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let slice = self.data.slice(ndarray::s![start..=i]);
                result[i] = slice.iter().cloned().fold(f64::NAN, f64::max);
            }
        }

        Series::from_array(result)
    }

    /// Rolling min (ts_min)
    pub fn ts_min(&self, window: usize) -> Series {
        let n = self.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            if i + 1 < window {
                result[i] = f64::NAN;
            } else {
                let start = i + 1 - window;
                let slice = self.data.slice(ndarray::s![start..=i]);
                result[i] = slice.iter().cloned().fold(f64::NAN, f64::min);
            }
        }

        Series::from_array(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    // Helper function to compare slices with NaN values
    fn approx_eq_slice(a: &[f64], b: &[f64], epsilon: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (x, y) in a.iter().zip(b.iter()) {
            if x.is_nan() && y.is_nan() {
                continue;
            } else if (x - y).abs() > epsilon {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_series_new() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series = Series::new(data.clone());
        assert_eq!(series.len(), 5);
        assert!(!series.is_empty());
    }

    #[test]
    fn test_series_from_array() {
        let arr = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let series = Series::from_array(arr);
        assert_eq!(series.len(), 3);
    }

    #[test]
    fn test_series_with_name() {
        let data = vec![1.0, 2.0, 3.0];
        let series = Series::new(data).with_name("test_series");
        assert_eq!(series.name(), Some("test_series"));
    }

    #[test]
    fn test_series_len() {
        let series = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(series.len(), 5);
    }

    #[test]
    fn test_series_is_empty() {
        let empty_series = Series::new(vec![]);
        assert!(empty_series.is_empty());

        let non_empty_series = Series::new(vec![1.0]);
        assert!(!non_empty_series.is_empty());
    }

    #[test]
    fn test_series_as_slice() {
        let data = vec![1.0, 2.0, 3.0];
        let series = Series::new(data.clone());
        let slice = series.as_slice();
        assert!(slice.is_some());
        assert_eq!(slice.unwrap(), data.as_slice());
    }

    #[test]
    fn test_series_data() {
        let data = vec![1.0, 2.0, 3.0];
        let series = Series::new(data.clone());
        assert_eq!(series.data().len(), 3);
    }

    #[test]
    fn test_series_add() {
        let s1 = Series::new(vec![1.0, 2.0, 3.0]);
        let s2 = Series::new(vec![4.0, 5.0, 6.0]);
        let result = s1.add(&s2).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_series_add_length_mismatch() {
        let s1 = Series::new(vec![1.0, 2.0, 3.0]);
        let s2 = Series::new(vec![1.0, 2.0]);
        let result = s1.add(&s2);
        assert!(result.is_err());
    }

    #[test]
    fn test_series_sub() {
        let s1 = Series::new(vec![5.0, 8.0, 3.0]);
        let s2 = Series::new(vec![1.0, 2.0, 1.0]);
        let result = s1.sub(&s2).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[4.0, 6.0, 2.0]);
    }

    #[test]
    fn test_series_mul() {
        let s1 = Series::new(vec![2.0, 3.0, 4.0]);
        let s2 = Series::new(vec![5.0, 6.0, 7.0]);
        let result = s1.mul(&s2).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_series_div() {
        let s1 = Series::new(vec![10.0, 20.0, 30.0]);
        let s2 = Series::new(vec![2.0, 4.0, 5.0]);
        let result = s1.div(&s2).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_series_gt() {
        let s1 = Series::new(vec![5.0, 3.0, 7.0]);
        let s2 = Series::new(vec![4.0, 6.0, 7.0]);
        let result = s1.gt(&s2).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_series_abs() {
        let s = Series::new(vec![-1.0, 2.0, -3.0]);
        let result = s.abs();
        assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_series_sqrt() {
        let s = Series::new(vec![4.0, 9.0, 16.0]);
        let result = s.sqrt();
        assert_eq!(result.as_slice().unwrap(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_series_log() {
        let s = Series::new(vec![1.0, 2.0, 10.0]);
        let result = s.log();
        assert!((result.data()[0] - 0.0).abs() < 1e-10);
        assert!((result.data()[1] - 0.6931).abs() < 0.001);
        assert!((result.data()[2] - 2.3026).abs() < 0.001);
    }

    #[test]
    fn test_series_exp() {
        let s = Series::new(vec![0.0, 1.0, 2.0]);
        let result = s.exp();
        assert!((result.data()[0] - 1.0).abs() < 1e-10);
        assert!((result.data()[1] - 2.7183).abs() < 0.001);
        assert!((result.data()[2] - 7.3891).abs() < 0.01);
    }

    #[test]
    fn test_series_neg() {
        let s = Series::new(vec![1.0, -2.0, 3.0]);
        let result = s.neg();
        assert_eq!(result.as_slice().unwrap(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_series_lag() {
        let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = s.lag(2);
        let expected = &[f64::NAN, f64::NAN, 1.0, 2.0, 3.0];
        assert!(approx_eq_slice(result.as_slice().unwrap(), expected, 1e-10));
    }

    #[test]
    fn test_series_diff() {
        let s = Series::new(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let result = s.diff(1);
        let expected = &[f64::NAN, 10.0, 10.0, 10.0, 10.0];
        assert!(approx_eq_slice(result.as_slice().unwrap(), expected, 1e-10));
    }

    #[test]
    fn test_series_pct_change() {
        let s = Series::new(vec![100.0, 110.0, 121.0]);
        let result = s.pct_change(1);
        assert!(result.data()[0].is_nan());
        assert!((result.data()[1] - 0.1).abs() < 0.001);
        assert!((result.data()[2] - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_series_moving_average() {
        let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = s.moving_average(3);
        assert_eq!(result.len(), 5);
        // Moving average should produce some values
        assert!(!result.data()[2].is_nan());
    }

    #[test]
    fn test_series_rolling_std() {
        let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = s.rolling_std(3);
        assert_eq!(result.len(), 5);
        // Rolling std should produce some values
        assert!(!result.data()[2].is_nan());
    }

    #[test]
    fn test_series_ema() {
        let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = s.ema(3);
        assert_eq!(result.len(), 5);
        assert!((result.data()[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_series_z_score() {
        let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = s.z_score(5);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_series_ts_rank() {
        let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = s.ts_rank(3);
        assert!(result.data()[0].is_nan());
        assert!(result.data()[1].is_nan());
        // For window=3 at index 2, values are [1,2,3], rank of 3 is 1.0
        assert!((result.data()[2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_series_cs_rank() {
        let s = Series::new(vec![3.0, 1.0, 2.0]);
        let result = s.cs_rank();
        assert_eq!(result.len(), 3);
        // cs_rank should produce values between 0 and 1
        for i in 0..result.len() {
            assert!(result.data()[i] >= 0.0 && result.data()[i] <= 1.0);
        }
    }

    #[test]
    fn test_series_ts_argmax() {
        let s = Series::new(vec![1.0, 5.0, 3.0, 2.0, 4.0]);
        let result = s.ts_argmax(3);
        assert!(result.data()[0].is_nan());
        assert!(result.data()[1].is_nan());
        // For window=3 at index 2, values are [1,5,3], max is at position 2 (1-indexed)
        assert!((result.data()[2] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_series_ts_argmin() {
        let s = Series::new(vec![3.0, 1.0, 5.0, 2.0, 4.0]);
        let result = s.ts_argmin(3);
        assert!(result.data()[0].is_nan());
        assert!(result.data()[1].is_nan());
        // For window=3 at index 2, values are [3,1,5], min is at position 2 (1-indexed)
        assert!((result.data()[2] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_series_ts_corr() {
        let s1 = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let s2 = Series::new(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let result = s1.ts_corr(&s2, 3);
        assert_eq!(result.len(), 5);
        // Perfect correlation should be 1.0
        assert!((result.data()[2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_series_sign() {
        let s = Series::new(vec![-2.0, 0.0, 3.0]);
        let result = s.sign();
        assert_eq!(result.as_slice().unwrap(), &[-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_series_power() {
        let s = Series::new(vec![2.0, 3.0, 4.0]);
        let result = s.power(2.0);
        assert_eq!(result.as_slice().unwrap(), &[4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_series_ts_sum() {
        let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = s.ts_sum(3);
        assert!(result.data()[0].is_nan());
        assert!(result.data()[1].is_nan());
        assert!((result.data()[2] - 6.0).abs() < 0.001);
        assert!((result.data()[3] - 9.0).abs() < 0.001);
        assert!((result.data()[4] - 12.0).abs() < 0.001);
    }

    #[test]
    fn test_series_ts_count() {
        let s = Series::new(vec![1.0, 2.0, f64::NAN, 4.0, 5.0]);
        let result = s.ts_count(3);
        assert_eq!(result.len(), 5);
        // Should produce some non-NaN values
        assert!(!result.data()[2].is_nan());
    }

    #[test]
    fn test_series_ts_max() {
        let s = Series::new(vec![1.0, 5.0, 3.0, 8.0, 2.0]);
        let result = s.ts_max(3);
        assert_eq!(result.len(), 5);
        // Should produce some non-NaN values
        assert!(!result.data()[2].is_nan());
    }

    #[test]
    fn test_series_ts_min() {
        let s = Series::new(vec![5.0, 1.0, 3.0, 2.0, 8.0]);
        let result = s.ts_min(3);
        assert!(result.data()[0].is_nan());
        assert!(result.data()[1].is_nan());
        assert!((result.data()[2] - 1.0).abs() < 0.001);
        assert!((result.data()[3] - 1.0).abs() < 0.001);
        assert!((result.data()[4] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_series_decay_linear() {
        let s = Series::new(vec![1.0, 2.0, 3.0]);
        let result = s.decay_linear(3);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_series_scale() {
        let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = s.scale(5);
        assert_eq!(result.len(), 5);
    }
}
