//! exprs core library: high-performance factor backtesting and expression evaluation
//! Exposed as Python extension via PyO3

// Core modules
mod expr;
mod expr_optimizer;
mod lazy;
mod polars_style;
mod gp;
mod persistence;
mod metalearning;

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::types::{PyDict, PyList, PyTuple};
use ndarray::{Array2, Array1};
use numpy::{PyArray2, PyArray1, PyArrayMethods, PyUntypedArrayMethods, IntoPyArray};
use std::f64::NAN;
use std::sync::Arc;

// Re-exports for internal use
use crate::expr::{Expr, Literal, BinaryOp, UnaryOp};
use crate::polars_style::{DataFrame, Series, evaluate_expr_on_dataframe};
use crate::lazy::{LazyFrame, DataSource, JoinType};

/// Weight allocation method
#[derive(Debug, Clone, Copy)]
pub enum WeightMethod {
    Equal,
    Weighted,
}

/// Backtest result (Rust internal representation)
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub group_returns: Array2<f64>,
    pub group_cum_returns: Array2<f64>,
    pub long_short_returns: Array1<f64>,
    pub long_short_cum_return: f64,
    pub ic_series: Array1<f64>,
    pub ic_mean: f64,
    pub ic_ir: f64,
}

/// Backtest engine (core Rust implementation)
pub struct BacktestEngine {
    factor: Array2<f64>,
    returns: Array2<f64>,
    weights: Option<Array2<f64>>,
    quantiles: usize,
    weight_method: WeightMethod,
    long_top_n: usize,
    short_top_n: usize,
    commission_rate: f64,
}

impl BacktestEngine {
    pub fn new(
        factor: Array2<f64>,
        returns: Array2<f64>,
        quantiles: usize,
        weight_method: WeightMethod,
        long_top_n: usize,
        short_top_n: usize,
        commission_rate: f64,
        weights: Option<Array2<f64>>,
    ) -> Self {
        assert_eq!(factor.shape(), returns.shape());
        if let Some(ref w) = weights {
            assert_eq!(w.shape(), factor.shape());
        }
        
        Self {
            factor,
            returns,
            weights,
            quantiles,
            weight_method,
            long_top_n,
            short_top_n,
            commission_rate,
        }
    }
    
    pub fn run(&self) -> Result<BacktestResult, String> {
        // Implementation from backtest_rs
        // For brevity, including core logic only
        let (n_days, n_assets) = self.factor.dim();
        
        // Compute quantile groups
        let group_labels = self.compute_quantile_groups()?;
        
        // Compute group returns
        let (_, group_returns) = self.compute_group_returns(&group_labels)?;
        
        // Compute long-short returns
        let long_short_returns = self.compute_long_short_returns(&group_returns);
        
        // Compute cumulative returns
        let group_cum_returns = self.compute_cumulative_returns(&group_returns);
        let long_short_cum_return = (1.0 + &long_short_returns).fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0;
        
        // Compute IC series
        let (ic_series, ic_mean, ic_ir) = self.compute_ic_series()?;
        
        Ok(BacktestResult {
            group_returns,
            group_cum_returns,
            long_short_returns,
            long_short_cum_return,
            ic_series,
            ic_mean,
            ic_ir,
        })
    }
    
    fn compute_quantile_groups(&self) -> Result<Array2<usize>, String> {
        let (n_days, n_assets) = self.factor.dim();
        let mut groups = Array2::<usize>::zeros((n_days, n_assets));
        
        // Parallel processing (simplified)
        for day in 0..n_days {
            let factor_row = self.factor.row(day);
            let mut valid_data: Vec<(usize, f64)> = factor_row
                .iter()
                .enumerate()
                .filter(|&(_, &v)| !v.is_nan())
                .map(|(i, &v)| (i, v))
                .collect();
            
            if valid_data.len() < self.quantiles {
                continue;
            }
            
            valid_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            let n_valid = valid_data.len();
            let group_size = n_valid / self.quantiles;
            
            for (group_idx, &(asset_idx, _)) in valid_data.iter().enumerate() {
                let quantile = (group_idx / group_size).min(self.quantiles - 1) + 1;
                groups[[day, asset_idx]] = quantile;
            }
        }
        
        Ok(groups)
    }
    
    fn compute_group_returns(
        &self,
        group_labels: &Array2<usize>
    ) -> Result<(Array2<f64>, Array2<f64>), String> {
        let (n_days, n_assets) = self.factor.dim();
        let mut group_weights = Array2::<f64>::zeros((n_days, self.quantiles));
        let mut group_returns = Array2::<f64>::zeros((n_days - 1, self.quantiles));
        
        for day in 0..(n_days - 1) {
            let labels_today = group_labels.row(day);
            let returns_tomorrow = self.returns.row(day + 1);
            
            for group in 1..=self.quantiles {
                // Get indices of assets in this group
                let mut asset_indices = Vec::new();
                for asset in 0..n_assets {
                    if labels_today[asset] == group {
                        asset_indices.push(asset);
                    }
                }
                
                if asset_indices.is_empty() {
                    continue;
                }
                
                // Compute weights
                let weights = match self.weight_method {
                    WeightMethod::Equal => {
                        let w = 1.0 / asset_indices.len() as f64;
                        vec![w; asset_indices.len()]
                    }
                    WeightMethod::Weighted => {
                        if let Some(ref weight_data) = self.weights {
                            let total_weight: f64 = asset_indices.iter()
                                .map(|&idx| weight_data[[day, idx]])
                                .filter(|&w| !w.is_nan())
                                .sum();
                            if total_weight == 0.0 {
                                vec![0.0; asset_indices.len()]
                            } else {
                                asset_indices.iter()
                                    .map(|&idx| weight_data[[day, idx]] / total_weight)
                                    .collect()
                            }
                        } else {
                            return Err("Weighted method requires weight data".to_string());
                        }
                    }
                };
                
                // Store group weight (sum of weights within group)
                group_weights[[day, group - 1]] = weights.iter().sum();
                
                // Compute weighted return
                let weighted_return: f64 = asset_indices.iter()
                    .zip(weights.iter())
                    .map(|(&idx, &w)| {
                        let ret = returns_tomorrow[idx];
                        if ret.is_nan() { 0.0 } else { w * ret }
                    })
                    .sum();
                
                group_returns[[day, group - 1]] = weighted_return;
            }
        }
        
        Ok((group_weights, group_returns))
    }
    
    fn compute_long_short_returns(&self, group_returns: &Array2<f64>) -> Array1<f64> {
        let n_days = group_returns.dim().0;
        let mut long_short = Array1::<f64>::zeros(n_days);
        
        let long_groups: Vec<usize> = (self.quantiles - self.long_top_n + 1..=self.quantiles).collect();
        let short_groups: Vec<usize> = (1..=self.short_top_n).collect();
        
        for day in 0..n_days {
            let long_return: f64 = long_groups.iter()
                .map(|&g| group_returns[[day, g - 1]])
                .sum::<f64>() / long_groups.len() as f64;
            
            let short_return: f64 = short_groups.iter()
                .map(|&g| group_returns[[day, g - 1]])
                .sum::<f64>() / short_groups.len() as f64;
            
            long_short[day] = long_return - short_return - self.commission_rate;
        }
        
        long_short
    }
    
    fn compute_cumulative_returns(&self, daily_returns: &Array2<f64>) -> Array2<f64> {
        let (n_days, n_groups) = daily_returns.dim();
        let mut cum_returns = Array2::<f64>::zeros((n_days, n_groups));
        
        for g in 0..n_groups {
            let mut cum = 1.0;
            for d in 0..n_days {
                cum *= 1.0 + daily_returns[[d, g]];
                cum_returns[[d, g]] = cum - 1.0;
            }
        }
        
        cum_returns
    }
    
    fn compute_ic_series(&self) -> Result<(Array1<f64>, f64, f64), String> {
        let (n_days, n_assets) = self.factor.dim();
        let mut ic_series = Array1::<f64>::zeros(n_days - 1);
        
        for day in 0..(n_days - 1) {
            let factor_today = self.factor.row(day);
            let returns_tomorrow = self.returns.row(day + 1);
            
            let mut factor_vals = Vec::new();
            let mut return_vals = Vec::new();
            
            for asset in 0..n_assets {
                let f = factor_today[asset];
                let r = returns_tomorrow[asset];
                if !f.is_nan() && !r.is_nan() {
                    factor_vals.push(f);
                    return_vals.push(r);
                }
            }
            
            if factor_vals.len() < 2 {
                ic_series[day] = NAN;
                continue;
            }
            
            ic_series[day] = self.pearson_correlation(&factor_vals, &return_vals);
        }
        
        let valid_ic: Vec<f64> = ic_series.iter()
            .filter(|&&v| !v.is_nan())
            .cloned()
            .collect();
        
        if valid_ic.is_empty() {
            return Err("No valid IC values".to_string());
        }
        
        let ic_mean = valid_ic.mean();
        let ic_std = valid_ic.std_dev();
        let ic_ir = if ic_std == 0.0 { NAN } else { ic_mean / ic_std };
        
        Ok((ic_series, ic_mean, ic_ir))
    }
    
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y).map(|(&a, &b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|&a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|&b| b * b).sum();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator == 0.0 { 0.0 } else { numerator / denominator }
    }
}

trait StatsExt {
    fn mean(&self) -> f64;
    fn std_dev(&self) -> f64;
}

impl StatsExt for Vec<f64> {
    fn mean(&self) -> f64 {
        if self.is_empty() { 0.0 } else { self.iter().sum::<f64>() / self.len() as f64 }
    }
    
    fn std_dev(&self) -> f64 {
        if self.len() <= 1 { 0.0 } else {
            let mean = self.mean();
            let variance = self.iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<f64>() / (self.len() - 1) as f64;
            variance.sqrt()
        }
    }
}

// ============================================================================
// Expression System Python Bindings
// ============================================================================

/// Python-exposed expression
#[pyclass(name = "Expr")]
#[derive(Clone)]
pub struct PyExpr {
    inner: Expr,
}

#[pymethods]
impl PyExpr {
    #[new]
    fn new() -> Self {
        PyExpr {
            inner: Expr::Literal(Literal::Float(0.0)),
        }
    }

    /// Create a column expression
    #[staticmethod]
    fn col(name: &str) -> Self {
        PyExpr {
            inner: Expr::Column(name.to_string()),
        }
    }

    /// Create a literal integer expression
    #[staticmethod]
    fn lit_int(value: i64) -> Self {
        PyExpr {
            inner: Expr::Literal(Literal::Integer(value)),
        }
    }

    /// Create a literal float expression
    #[staticmethod]
    fn lit_float(value: f64) -> Self {
        PyExpr {
            inner: Expr::Literal(Literal::Float(value)),
        }
    }

    /// Create a literal boolean expression
    #[staticmethod]
    fn lit_bool(value: bool) -> Self {
        PyExpr {
            inner: Expr::Literal(Literal::Boolean(value)),
        }
    }

    /// Add two expressions
    fn add(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Arc::new(self.inner.clone()),
                op: BinaryOp::Add,
                right: Arc::new(other.inner.clone()),
            },
        }
    }

    /// Python operator overload: +
    #[pyo3(name = "__add__")]
    fn python_add(&self, other: &PyExpr) -> Self {
        self.add(other)
    }

    /// Subtract two expressions
    fn sub(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Arc::new(self.inner.clone()),
                op: BinaryOp::Subtract,
                right: Arc::new(other.inner.clone()),
            },
        }
    }

    /// Python operator overload: -
    #[pyo3(name = "__sub__")]
    fn python_sub(&self, other: &PyExpr) -> Self {
        self.sub(other)
    }

    /// Multiply two expressions
    fn mul(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Arc::new(self.inner.clone()),
                op: BinaryOp::Multiply,
                right: Arc::new(other.inner.clone()),
            },
        }
    }

    /// Python operator overload: *
    #[pyo3(name = "__mul__")]
    fn python_mul(&self, other: &PyExpr) -> Self {
        self.mul(other)
    }

    /// Divide two expressions
    fn div(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Arc::new(self.inner.clone()),
                op: BinaryOp::Divide,
                right: Arc::new(other.inner.clone()),
            },
        }
    }

    /// Python operator overload: /
    #[pyo3(name = "__truediv__")]
    fn python_div(&self, other: &PyExpr) -> Self {
        self.div(other)
    }

    /// Negate expression
    fn neg(&self) -> Self {
        PyExpr {
            inner: Expr::UnaryExpr {
                op: UnaryOp::Negate,
                expr: Arc::new(self.inner.clone()),
            },
        }
    }

    /// Absolute value
    fn abs(&self) -> Self {
        PyExpr {
            inner: Expr::UnaryExpr {
                op: UnaryOp::Abs,
                expr: Arc::new(self.inner.clone()),
            },
        }
    }

    /// Square root
    fn sqrt(&self) -> Self {
        PyExpr {
            inner: Expr::UnaryExpr {
                op: UnaryOp::Sqrt,
                expr: Arc::new(self.inner.clone()),
            },
        }
    }

    /// Natural logarithm
    fn log(&self) -> Self {
        PyExpr {
            inner: Expr::UnaryExpr {
                op: UnaryOp::Log,
                expr: Arc::new(self.inner.clone()),
            },
        }
    }

    /// Exponential
    fn exp(&self) -> Self {
        PyExpr {
            inner: Expr::UnaryExpr {
                op: UnaryOp::Exp,
                expr: Arc::new(self.inner.clone()),
            },
        }
    }

    /// Greater than comparison
    fn gt(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Arc::new(self.inner.clone()),
                op: BinaryOp::GreaterThan,
                right: Arc::new(other.inner.clone()),
            },
        }
    }

    /// Less than comparison
    fn lt(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Arc::new(self.inner.clone()),
                op: BinaryOp::LessThan,
                right: Arc::new(other.inner.clone()),
            },
        }
    }

    /// Equal comparison
    fn eq(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Arc::new(self.inner.clone()),
                op: BinaryOp::Equal,
                right: Arc::new(other.inner.clone()),
            },
        }
    }

    /// Not equal comparison
    fn ne(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Arc::new(self.inner.clone()),
                op: BinaryOp::NotEqual,
                right: Arc::new(other.inner.clone()),
            },
        }
    }

    /// Get string representation
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    /// Get string representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python-exposed Series for vectorized operations
#[pyclass(name = "Series")]
#[derive(Clone)]
pub struct PySeries {
    inner: Series,
}

#[pymethods]
impl PySeries {
    #[new]
    fn new(data: Vec<f64>) -> Self {
        PySeries {
            inner: Series::new(data),
        }
    }

    /// Get length
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Get data as Python list
    fn to_list(&self) -> Vec<f64> {
        self.inner.data().to_vec()
    }

    fn __repr__(&self) -> String {
        format!("Series(len={})", self.len())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python-exposed DataFrame
#[pyclass(name = "DataFrame")]
pub struct PyDataFrame {
    inner: DataFrame,
}

#[pymethods]
impl PyDataFrame {
    #[new]
    fn new(py: Python<'_>, columns: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        if let Some(cols) = columns {
            let mut inner_columns = std::collections::HashMap::new();

            for (key, value) in cols.iter() {
                let col_name: String = key.extract()?;

                if let Ok(py_series) = value.extract::<PySeries>() {
                    inner_columns.insert(col_name, py_series.inner.clone());
                } else if let Ok(list) = value.extract::<Vec<f64>>() {
                    inner_columns.insert(col_name, Series::new(list));
                } else {
                    return Err(PyValueError::new_err(
                        format!("Column '{}' must be a Series or list of floats", col_name)
                    ));
                }
            }

            // Use from_series_map which returns Result
            match DataFrame::from_series_map(inner_columns) {
                Ok(df) => Ok(PyDataFrame { inner: df }),
                Err(e) => Err(PyValueError::new_err(e)),
            }
        } else {
            Ok(PyDataFrame {
                inner: DataFrame::new(), // Empty dataframe
            })
        }
    }

    /// Get number of rows
    fn n_rows(&self) -> usize {
        self.inner.n_rows()
    }

    /// Get number of columns
    fn n_cols(&self) -> usize {
        self.inner.n_cols()
    }

    /// Get column names
    fn column_names(&self) -> Vec<String> {
        self.inner.column_names()
    }

    /// Evaluate an expression on this DataFrame
    fn evaluate(&self, expr: &PyExpr) -> PyResult<PySeries> {
        evaluate_expr_on_dataframe(&expr.inner, &self.inner)
            .map(|series| PySeries { inner: series })
            .map_err(|e| PyValueError::new_err(e))
    }

    fn __repr__(&self) -> String {
        format!("DataFrame(rows={}, cols={})", self.n_rows(), self.n_cols())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Evaluate expression on multi-asset data (returns factor matrix)
#[pyfunction]
fn evaluate_expression(
    py: Python<'_>,
    expr: &PyExpr,
    data: Bound<'_, PyDict>,
    n_days: usize,
    n_assets: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    use numpy::PyArray2;

    // Validate input dimensions
    let mut column_arrays = std::collections::HashMap::new();

    for (key, value) in data.iter() {
        let col_name: String = key.extract()?;

        // Try to extract as numpy array
        if let Ok(arr) = value.extract::<Bound<'_, PyArray2<f64>>>() {
            let array = arr.readonly();
            let shape = array.shape();
            if shape[0] != n_days || shape[1] != n_assets {
                return Err(PyValueError::new_err(
                    format!("Column '{}' has shape {:?}, expected ({}, {})",
                           col_name, shape, n_days, n_assets)
                ));
            }
            column_arrays.insert(col_name, array.as_array().to_owned());
        } else {
            return Err(PyValueError::new_err(
                    format!("Column '{}' must be a 2D numpy array", col_name)
                ));
            }
        }
        
        // Create result array
        let mut result = Array2::<f64>::zeros((n_days, n_assets));
        
        // Process each asset in parallel using Rayon's parallel iterator
        use rayon::prelude::*;
        
        // Clone necessary data for parallel processing
        let column_arrays_clone = column_arrays.clone();
        let expr_inner = expr.inner.clone();
        
        // Process assets in parallel and collect results
        let asset_results: Vec<_> = (0..n_assets)
            .into_par_iter()
            .map(|asset_idx| {
                // Create DataFrame for this asset
                let mut columns = std::collections::HashMap::new();
                
                for (col_name, array) in &column_arrays_clone {
                    let column_data = array.column(asset_idx).to_owned();
                    columns.insert(col_name.clone(), Series::new(column_data.to_vec()));
                }
                
                // Evaluate expression for this asset
                if let Ok(df) = DataFrame::from_series_map(columns) {
                    if let Ok(series) = evaluate_expr_on_dataframe(&expr_inner, &df) {
                        return series.data().to_vec();
                    }
                }
                
                // Return NaN-filled vector if evaluation failed
                vec![f64::NAN; n_days]
            })
            .collect();
        
        // Assemble results into the 2D array
        for (asset_idx, asset_result) in asset_results.into_iter().enumerate() {
            for day_idx in 0..n_days {
                result[[day_idx, asset_idx]] = asset_result[day_idx];
            }
        }
        
        Ok(result.into_pyarray(py).into())
}

/// Create a lag expression
#[pyfunction]
fn lag(expr: &PyExpr, periods: usize) -> PyResult<PyExpr> {
    let lag_expr = Expr::FunctionCall {
        name: "lag".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Integer(periods as i64)),
        ],
    };
    Ok(PyExpr { inner: lag_expr })
}

/// Create a difference expression
#[pyfunction]
fn diff(expr: &PyExpr, periods: usize) -> PyResult<PyExpr> {
    let diff_expr = Expr::FunctionCall {
        name: "diff".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Integer(periods as i64)),
        ],
    };
    Ok(PyExpr { inner: diff_expr })
}

/// Create a rolling mean expression
#[pyfunction]
fn rolling_mean(expr: &PyExpr, window: usize) -> PyResult<PyExpr> {
    let ma_expr = Expr::FunctionCall {
        name: "moving_average".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Integer(window as i64)),
        ],
    };
    Ok(PyExpr { inner: ma_expr })
}

/// Create a cumulative sum expression
#[pyfunction]
fn cumsum(expr: &PyExpr) -> PyResult<PyExpr> {
    let cumsum_expr = Expr::FunctionCall {
        name: "cumsum".to_string(),
        args: vec![expr.inner.clone()],
    };
    Ok(PyExpr { inner: cumsum_expr })
}

/// Create a cumulative product expression
#[pyfunction]
fn cumprod(expr: &PyExpr) -> PyResult<PyExpr> {
    let cumprod_expr = Expr::FunctionCall {
        name: "cumprod".to_string(),
        args: vec![expr.inner.clone()],
    };
    Ok(PyExpr { inner: cumprod_expr })
}

/// Create a rolling standard deviation expression
#[pyfunction]
fn rolling_std(expr: &PyExpr, window: usize) -> PyResult<PyExpr> {
    let std_expr = Expr::FunctionCall {
        name: "rolling_std".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Integer(window as i64)),
        ],
    };
    Ok(PyExpr { inner: std_expr })
}

// ============================================================================
// Alpha101 Expression Functions
// ============================================================================

/// Time series rank - rank of the value in the last `window` time periods
#[pyfunction]
fn ts_rank(expr: &PyExpr, window: usize) -> PyResult<PyExpr> {
    let ts_rank_expr = Expr::FunctionCall {
        name: "ts_rank".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Integer(window as i64)),
        ],
    };
    Ok(PyExpr { inner: ts_rank_expr })
}

/// Time series argmax - index of maximum value in the last `window` time periods
#[pyfunction]
fn ts_argmax(expr: &PyExpr, window: usize) -> PyResult<PyExpr> {
    let ts_argmax_expr = Expr::FunctionCall {
        name: "ts_argmax".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Integer(window as i64)),
        ],
    };
    Ok(PyExpr { inner: ts_argmax_expr })
}

/// Time series argmin - index of minimum value in the last `window` time periods
#[pyfunction]
fn ts_argmin(expr: &PyExpr, window: usize) -> PyResult<PyExpr> {
    let ts_argmin_expr = Expr::FunctionCall {
        name: "ts_argmin".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Integer(window as i64)),
        ],
    };
    Ok(PyExpr { inner: ts_argmin_expr })
}

/// Cross-sectional rank - rank of the value across assets at each time point
#[pyfunction]
fn rank(expr: &PyExpr) -> PyResult<PyExpr> {
    let rank_expr = Expr::FunctionCall {
        name: "rank".to_string(),
        args: vec![expr.inner.clone()],
    };
    Ok(PyExpr { inner: rank_expr })
}

/// Time series correlation - rolling correlation between two series
#[pyfunction]
fn ts_corr(expr1: &PyExpr, expr2: &PyExpr, window: usize) -> PyResult<PyExpr> {
    let ts_corr_expr = Expr::FunctionCall {
        name: "ts_corr".to_string(),
        args: vec![
            expr1.inner.clone(),
            expr2.inner.clone(),
            Expr::Literal(Literal::Integer(window as i64)),
        ],
    };
    Ok(PyExpr { inner: ts_corr_expr })
}

/// Time series covariance - rolling covariance between two series
#[pyfunction]
fn ts_cov(expr1: &PyExpr, expr2: &PyExpr, window: usize) -> PyResult<PyExpr> {
    let ts_cov_expr = Expr::FunctionCall {
        name: "ts_cov".to_string(),
        args: vec![
            expr1.inner.clone(),
            expr2.inner.clone(),
            Expr::Literal(Literal::Integer(window as i64)),
        ],
    };
    Ok(PyExpr { inner: ts_cov_expr })
}

/// Scale - z-score normalization over `window` time periods
#[pyfunction]
fn scale(expr: &PyExpr, window: usize) -> PyResult<PyExpr> {
    let scale_expr = Expr::FunctionCall {
        name: "scale".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Integer(window as i64)),
        ],
    };
    Ok(PyExpr { inner: scale_expr })
}

/// Linear decay weighted average - exponentially decaying weights
#[pyfunction]
fn decay_linear(expr: &PyExpr, periods: usize) -> PyResult<PyExpr> {
    let decay_expr = Expr::FunctionCall {
        name: "decay_linear".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Integer(periods as i64)),
        ],
    };
    Ok(PyExpr { inner: decay_expr })
}

/// Sign function - returns -1, 0, or 1
#[pyfunction]
fn sign(expr: &PyExpr) -> PyResult<PyExpr> {
    let sign_expr = Expr::FunctionCall {
        name: "sign".to_string(),
        args: vec![expr.inner.clone()],
    };
    Ok(PyExpr { inner: sign_expr })
}

/// Power function - raises expression to a power
#[pyfunction]
fn power(expr: &PyExpr, exponent: f64) -> PyResult<PyExpr> {
    let power_expr = Expr::FunctionCall {
        name: "power".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Float(exponent)),
        ],
    };
    Ok(PyExpr { inner: power_expr })
}

/// Time series sum - rolling sum over `window` time periods
#[pyfunction]
fn ts_sum(expr: &PyExpr, window: usize) -> PyResult<PyExpr> {
    let ts_sum_expr = Expr::FunctionCall {
        name: "ts_sum".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Integer(window as i64)),
        ],
    };
    Ok(PyExpr { inner: ts_sum_expr })
}

/// Time series max - rolling maximum over `window` time periods
#[pyfunction]
fn ts_max(expr: &PyExpr, window: usize) -> PyResult<PyExpr> {
    let ts_max_expr = Expr::FunctionCall {
        name: "ts_max".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Integer(window as i64)),
        ],
    };
    Ok(PyExpr { inner: ts_max_expr })
}

/// Time series min - rolling minimum over `window` time periods
#[pyfunction]
fn ts_min(expr: &PyExpr, window: usize) -> PyResult<PyExpr> {
    let ts_min_expr = Expr::FunctionCall {
        name: "ts_min".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Integer(window as i64)),
        ],
    };
    Ok(PyExpr { inner: ts_min_expr })
}

// ============================================================================
// Main Python Module
// ============================================================================

// PyO3 bindings
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Backtest functionality
    m.add_class::<PyBacktestEngine>()?;
    m.add_function(wrap_pyfunction!(quantile_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ic, m)?)?;

    // Expression system
    m.add_class::<PyExpr>()?;
    m.add_class::<PySeries>()?;
    m.add_class::<PyDataFrame>()?;

    // Expression evaluation functions
    m.add_function(wrap_pyfunction!(evaluate_expression, m)?)?;
    m.add_function(wrap_pyfunction!(lag, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(cumprod, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std, m)?)?;

    // Alpha101 expression functions
    m.add_function(wrap_pyfunction!(ts_rank, m)?)?;
    m.add_function(wrap_pyfunction!(ts_argmax, m)?)?;
    m.add_function(wrap_pyfunction!(ts_argmin, m)?)?;
    m.add_function(wrap_pyfunction!(rank, m)?)?;
    m.add_function(wrap_pyfunction!(ts_corr, m)?)?;
    m.add_function(wrap_pyfunction!(ts_cov, m)?)?;
    m.add_function(wrap_pyfunction!(scale, m)?)?;
    m.add_function(wrap_pyfunction!(decay_linear, m)?)?;
    m.add_function(wrap_pyfunction!(sign, m)?)?;
    m.add_function(wrap_pyfunction!(power, m)?)?;
    m.add_function(wrap_pyfunction!(ts_sum, m)?)?;
    m.add_function(wrap_pyfunction!(ts_max, m)?)?;
    m.add_function(wrap_pyfunction!(ts_min, m)?)?;

    // Lazy evaluation system
    m.add_class::<PyLazyFrame>()?;
    m.add_function(wrap_pyfunction!(rolling_window, m)?)?;
    m.add_function(wrap_pyfunction!(expanding_window, m)?)?;

    // Genetic Programming system
    m.add_class::<PyGpEngine>()?;

    // Persistence system
    m.add_class::<PyPersistenceManager>()?;
    m.add_class::<PyFactorMetadata>()?;
    m.add_class::<PyGpHistoryRecord>()?;

    // Meta-learning system
    m.add_class::<PyMetaLearningAnalyzer>()?;
    m.add_class::<PyGpRecommendations>()?;

    Ok(())
}

/// Python-exposed backtest engine
#[pyclass]
struct PyBacktestEngine {
    engine: BacktestEngine,
}

#[pymethods]
impl PyBacktestEngine {
    #[new]
    fn new(
        py: Python<'_>,
        factor: Bound<'_, PyArray2<f64>>,
        returns: Bound<'_, PyArray2<f64>>,
        quantiles: usize,
        weight_method: &str,
        long_top_n: usize,
        short_top_n: usize,
        commission_rate: f64,
        weights: Option<Bound<'_, PyArray2<f64>>>,
    ) -> PyResult<Self> {
        let factor_array = factor.readonly().as_array().to_owned();
        let returns_array = returns.readonly().as_array().to_owned();
        let weights_array = weights.map(|w| w.readonly().as_array().to_owned());

        let wmethod = match weight_method {
            "equal" => WeightMethod::Equal,
            "weighted" => WeightMethod::Weighted,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                "weight_method must be 'equal' or 'weighted'"
            )),
        };

        let engine = BacktestEngine::new(
            factor_array,
            returns_array,
            quantiles,
            wmethod,
            long_top_n,
            short_top_n,
            commission_rate,
            weights_array,
        );

        Ok(Self { engine })
    }

    fn run(&self) -> PyResult<PyBacktestResult> {
        match self.engine.run() {
            Ok(result) => Ok(PyBacktestResult::from(result)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }
}

/// Python-exposed backtest result
#[pyclass]
struct PyBacktestResult {
    #[pyo3(get)]
    group_returns: Py<PyArray2<f64>>,
    #[pyo3(get)]
    group_cum_returns: Py<PyArray2<f64>>,
    #[pyo3(get)]
    long_short_returns: Py<PyArray1<f64>>,
    #[pyo3(get)]
    long_short_cum_return: f64,
    #[pyo3(get)]
    ic_series: Py<PyArray1<f64>>,
    #[pyo3(get)]
    ic_mean: f64,
    #[pyo3(get)]
    ic_ir: f64,
}

impl From<BacktestResult> for PyBacktestResult {
    fn from(result: BacktestResult) -> Self {
        pyo3::Python::try_attach(|py| {
            Self {
                group_returns: result.group_returns.into_pyarray(py).into(),
                group_cum_returns: result.group_cum_returns.into_pyarray(py).into(),
                long_short_returns: result.long_short_returns.into_pyarray(py).into(),
                long_short_cum_return: result.long_short_cum_return,
                ic_series: result.ic_series.into_pyarray(py).into(),
                ic_mean: result.ic_mean,
                ic_ir: result.ic_ir,
            }
        }).unwrap()
    }
}

/// Standalone quantile backtest function (Python interface)
#[pyfunction]
fn quantile_backtest(
    py: Python<'_>,
    factor: Bound<'_, PyArray2<f64>>,
    returns: Bound<'_, PyArray2<f64>>,
    quantiles: usize,
    weight_method: &str,
    long_top_n: usize,
    short_top_n: usize,
    commission_rate: f64,
    weights: Option<Bound<'_, PyArray2<f64>>>,
) -> PyResult<PyBacktestResult> {
    let engine = PyBacktestEngine::new(
        py,
        factor, returns, quantiles, weight_method,
        long_top_n, short_top_n, commission_rate, weights
    )?;
    engine.run()
}

/// Standalone IC computation (Python interface)
#[pyfunction]
fn compute_ic(
    py: Python<'_>,
    factor: Bound<'_, PyArray2<f64>>,
    returns: Bound<'_, PyArray2<f64>>,
) -> PyResult<(f64, f64)> {
    let factor_guard = factor.readonly();
    let factor_array = factor_guard.as_array();
    let returns_guard = returns.readonly();
    let returns_array = returns_guard.as_array();
    
    let (n_days, n_assets) = factor_array.dim();
    let mut ic_vals = Vec::new();
    
    for day in 0..(n_days - 1) {
        let mut factor_vals = Vec::new();
        let mut return_vals = Vec::new();
        
        for asset in 0..n_assets {
            let f = factor_array[[day, asset]];
            let r = returns_array[[day + 1, asset]];
            if !f.is_nan() && !r.is_nan() {
                factor_vals.push(f);
                return_vals.push(r);
            }
        }
        
        if factor_vals.len() >= 2 {
            let ic = compute_pearson(&factor_vals, &return_vals);
            ic_vals.push(ic);
        }
    }
    
    if ic_vals.is_empty() {
        return Ok((0.0, 0.0));
    }
    
    let ic_mean = ic_vals.iter().sum::<f64>() / ic_vals.len() as f64;
    let ic_std = (ic_vals.iter()
        .map(|&x| (x - ic_mean).powi(2))
        .sum::<f64>() / (ic_vals.len() - 1) as f64).sqrt();
    let ic_ir = if ic_std == 0.0 { 0.0 } else { ic_mean / ic_std };
    
    Ok((ic_mean, ic_ir))
}

fn compute_pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y).map(|(&a, &b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|&a| a * a).sum();
    let sum_y2: f64 = y.iter().map(|&b| b * b).sum();
    
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    
    if denominator == 0.0 { 0.0 } else { numerator / denominator }
}

// ============================================================================
// Lazy Module Python Bindings
// ============================================================================

/// Python-exposed LazyFrame for lazy evaluation
#[pyclass(name = "LazyFrame")]
pub struct PyLazyFrame {
    inner: Option<LazyFrame>,
}

#[pymethods]
impl PyLazyFrame {
    /// Create a new LazyFrame from numpy arrays
    #[staticmethod]
    fn scan(py: Python<'_>, data: Bound<'_, PyDict>) -> PyResult<Self> {
        use numpy::PyArray2;

        let mut arrays = std::collections::HashMap::new();

        for (key, value) in data.iter() {
            let col_name: String = key.extract()?;

            if let Ok(arr) = value.extract::<Bound<'_, PyArray2<f64>>>() {
                let array = arr.readonly().as_array().to_owned();
                arrays.insert(col_name, array);
            } else {
                return Err(PyValueError::new_err(
                    format!("Column '{}' must be a 2D numpy array", col_name)
                ));
            }
        }

        let source = DataSource::NumpyArrays(arrays);
        let lazy_frame = LazyFrame::scan(source);

        Ok(PyLazyFrame {
            inner: Some(lazy_frame),
        })
    }
    
    /// Add new columns to the LazyFrame
    fn with_columns(&self, py: Python<'_>, exprs: Bound<'_, PyList>) -> PyResult<Self> {
        let Some(ref inner) = self.inner else {
            return Err(PyValueError::new_err("LazyFrame is already consumed"));
        };

        // Convert Python expressions to Rust expressions
        let mut rust_exprs = Vec::new();
        for item in exprs.iter() {
            let tuple = item.downcast::<PyTuple>()?;
            let name: String = tuple.get_item(0)?.extract()?;
            let py_expr: PyExpr = tuple.get_item(1)?.extract()?;
            rust_exprs.push((name, py_expr.inner));
        }

        // Create new LazyFrame with columns added
        let new_lazy_frame = inner.clone().with_columns(rust_exprs);

        Ok(PyLazyFrame {
            inner: Some(new_lazy_frame),
        })
    }
    
    /// Join with another LazyFrame
    fn join(&self, other: &PyLazyFrame, on: Vec<String>, how: &str) -> PyResult<Self> {
        let Some(ref inner) = self.inner else {
            return Err(PyValueError::new_err("LazyFrame is already consumed"));
        };
        
        let Some(ref other_inner) = other.inner else {
            return Err(PyValueError::new_err("Other LazyFrame is already consumed"));
        };
        
        // Convert string join type to JoinType enum
        let join_type = match how.to_lowercase().as_str() {
            "inner" => JoinType::Inner,
            "left" => JoinType::Left,
            "right" => JoinType::Right,
            "outer" => JoinType::Outer,
            _ => return Err(PyValueError::new_err(
                "Join type must be 'inner', 'left', 'right', or 'outer'"
            )),
        };
        
        // Create new LazyFrame with join
        let new_lazy_frame = inner.clone().join(other_inner.clone(), on, join_type);
        
        Ok(PyLazyFrame {
            inner: Some(new_lazy_frame),
        })
    }
    
    /// Collect (execute) the lazy computation
    fn collect(&mut self) -> PyResult<Py<PyDict>> {
        let Some(lazy_frame) = self.inner.take() else {
            return Err(PyValueError::new_err("LazyFrame is already consumed"));
        };

        pyo3::Python::try_attach(|py| {
            match lazy_frame.collect() {
                Ok(result) => {
                    let dict = PyDict::new(py);
                    for (key, array) in result {
                        let py_array = array.into_pyarray(py);
                        dict.set_item(key, py_array)?;
                    }
                    Ok(dict.into())
                }
                Err(e) => Err(PyValueError::new_err(e)),
            }
        }).ok_or_else(|| PyRuntimeError::new_err("Failed to attach to Python"))?
    }
    
    /// Explain the logical plan
    fn explain(&self, optimized: bool) -> PyResult<String> {
        let Some(ref inner) = self.inner else {
            return Err(PyValueError::new_err("LazyFrame is already consumed"));
        };
        
        Ok(inner.explain(optimized))
    }
    
    /// Get string representation
    fn __repr__(&self) -> String {
        if self.inner.is_some() {
            "LazyFrame(active)".to_string()
        } else {
            "LazyFrame(consumed)".to_string()
        }
    }
    
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Create a rolling window specification for lazy evaluation
#[pyfunction]
fn rolling_window(size: usize, min_periods: Option<usize>) -> PyResult<Py<PyDict>> {
    pyo3::Python::try_attach(|py| {
        let window_spec = crate::lazy::rolling_window(size, min_periods);
        let dict = PyDict::new(py);
        dict.set_item("kind", "rolling")?;
        dict.set_item("size", size)?;
        dict.set_item("min_periods", min_periods.unwrap_or(1))?;
        Ok(dict.into())
    }).ok_or_else(|| PyRuntimeError::new_err("Failed to attach to Python"))?
}

/// Create an expanding window specification for lazy evaluation
#[pyfunction]
fn expanding_window(min_periods: Option<usize>) -> PyResult<Py<PyDict>> {
    pyo3::Python::try_attach(|py| {
        let window_spec = crate::lazy::expanding_window(min_periods);
        let dict = PyDict::new(py);
        dict.set_item("kind", "expanding")?;
        dict.set_item("min_periods", min_periods.unwrap_or(1))?;
        Ok(dict.into())
    }).ok_or_else(|| PyRuntimeError::new_err("Failed to attach to Python"))?
}

// ============================================================================
// Genetic Programming (GP) Module Python Bindings
// ============================================================================

use crate::gp::{GPConfig, Terminal, Function, BacktestFitnessEvaluator, run_gp};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

/// Python-exposed Genetic Programming engine for factor mining
#[pyclass(name = "GpEngine")]
pub struct PyGpEngine {
    config: GPConfig,
    terminals: Vec<Terminal>,
    functions: Vec<Function>,
    rng: StdRng,
}

#[pymethods]
impl PyGpEngine {
    #[new]
    fn new(
        population_size: usize,
        max_generations: usize,
        tournament_size: usize,
        crossover_prob: f64,
        mutation_prob: f64,
        max_depth: usize,
    ) -> Self {
        // Create default terminals (will be updated later)
        let terminals = vec![
            Terminal::Ephemeral,
            Terminal::Constant(1.0),
            Terminal::Constant(2.0),
        ];
        
        // Create default functions
        let functions = vec![
            Function::add(),
            Function::sub(),
            Function::mul(),
            Function::div(),
            Function::sqrt(),
            Function::abs(),
            Function::neg(),
        ];
        
        let config = GPConfig {
            population_size,
            max_generations,
            tournament_size,
            crossover_prob,
            mutation_prob,
            max_depth,
        };
        
        let rng = StdRng::from_entropy();
        
        PyGpEngine { config, terminals, functions, rng }
    }
    
    /// Set available columns (variables) for expression generation
    fn set_columns(&mut self, py: Python<'_>, columns: Bound<'_, PyList>) {
        // Update terminals with column variables
        let mut new_terminals = vec![
            Terminal::Ephemeral,
            Terminal::Constant(1.0),
            Terminal::Constant(2.0),
        ];

        for item in columns.iter() {
            let col: String = item.extract().unwrap_or_default();
            new_terminals.push(Terminal::Variable(col));
        }

        self.terminals = new_terminals;
    }
    
    /// Run genetic programming for factor mining
    fn mine_factors(
        &mut self,
        py: Python<'_>,
        data: Bound<'_, PyDict>,
        returns: Bound<'_, PyArray2<f64>>,
        num_factors: usize,
    ) -> PyResult<Vec<(String, f64)>> {
        use numpy::PyArray2;

        // Extract data arrays
        let mut data_arrays = HashMap::new();

        for (key, value) in data.iter() {
            let col_name: String = key.extract()?;

            if let Ok(arr) = value.extract::<Bound<'_, PyArray2<f64>>>() {
                let array = arr.readonly().as_array().to_owned();
                data_arrays.insert(col_name, array);
            } else {
                return Err(PyValueError::new_err(
                    format!("Column '{}' must be a 2D numpy array", col_name)
                ));
            }
        }

        // Extract returns array
        let returns_array = returns.readonly().as_array().to_owned();

        // Create fitness evaluator
        let evaluator = BacktestFitnessEvaluator::new(data_arrays, returns_array);

        // Run GP multiple times to get multiple factors
        let mut results = Vec::new();

        for _ in 0..num_factors {
            let (best_expr, best_fitness) = run_gp(
                &self.config,
                &evaluator,
                self.terminals.clone(),
                self.functions.clone(),
                &mut self.rng,
            );

            // Convert expression to string representation
            let expr_str = format!("{:?}", best_expr);
            results.push((expr_str, best_fitness));
        }

        Ok(results)
    }
    
    /// Run a simple test to verify GP functionality
    fn test_run(&mut self) -> PyResult<String> {
        // Create simple test data
        let mut test_data = HashMap::new();
        test_data.insert("x".to_string(), ndarray::Array2::<f64>::zeros((10, 5)));
        test_data.insert("y".to_string(), ndarray::Array2::<f64>::ones((10, 5)));
        
        let test_returns = ndarray::Array2::<f64>::zeros((10, 5));
        let evaluator = BacktestFitnessEvaluator::new(test_data, test_returns);
        
        // Set test terminals
        let mut test_terminals = self.terminals.clone();
        test_terminals.push(Terminal::Variable("x".to_string()));
        test_terminals.push(Terminal::Variable("y".to_string()));
        
        // Run GP
        let (best_expr, fitness) = run_gp(
            &self.config,
            &evaluator,
            test_terminals,
            self.functions.clone(),
            &mut self.rng,
        );
        
        Ok(format!("Best expression: {:?}, Fitness: {:.6}", best_expr, fitness))
    }
}

// ============================================================================
// Persistence Module Python Bindings
// ============================================================================

use crate::persistence::{PersistenceManager, FactorMetadata, GPHistoryRecord};
use crate::metalearning::{MetaLearningAnalyzer, GPRecommendations};

/// Python-exposed Persistence Manager for factor storage and retrieval
#[pyclass(name = "PersistenceManager")]
pub struct PyPersistenceManager {
    inner: PersistenceManager,
}

#[pymethods]
impl PyPersistenceManager {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        match PersistenceManager::new(path) {
            Ok(manager) => Ok(PyPersistenceManager { inner: manager }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create persistence manager: {}", e))),
        }
    }
    
    /// Save a factor to disk
    fn save_factor(&mut self, factor: &PyFactorMetadata) -> PyResult<()> {
        self.inner.save_factor(&factor.inner)
            .map_err(|e| PyValueError::new_err(format!("Failed to save factor: {}", e)))
    }
    
    /// Load a factor from disk
    fn load_factor(&mut self, factor_id: &str) -> PyResult<Option<PyFactorMetadata>> {
        match self.inner.load_factor(factor_id) {
            Ok(Some(factor)) => Ok(Some(PyFactorMetadata { inner: factor })),
            Ok(None) => Ok(None),
            Err(e) => Err(PyValueError::new_err(format!("Failed to load factor: {}", e))),
        }
    }
    
    /// Search factors by criteria
    #[pyo3(signature = (min_ic = None, max_complexity = None, tags = Vec::new()))]
    fn search_factors(
        &self,
        min_ic: Option<f64>,
        max_complexity: Option<f64>,
        tags: Vec<String>,
    ) -> PyResult<Vec<PyFactorMetadata>> {
        let factors = self.inner.search_factors(min_ic, None, max_complexity, &tags);
        Ok(factors.into_iter().map(|f| PyFactorMetadata { inner: f }).collect())
    }
    
    /// Load all factors from disk
    fn load_all_factors(&mut self) -> PyResult<usize> {
        self.inner.load_all_factors()
            .map_err(|e| PyValueError::new_err(format!("Failed to load factors: {}", e)))
    }
    
    /// Load all GP history records from disk
    fn load_all_history(&mut self) -> PyResult<usize> {
        self.inner.load_all_history()
            .map_err(|e| PyValueError::new_err(format!("Failed to load history: {}", e)))
    }
    
    /// Get all loaded factors
    fn get_all_factors(&self) -> Vec<PyFactorMetadata> {
        self.inner.get_all_factors()
            .into_iter()
            .map(|f| PyFactorMetadata { inner: f })
            .collect()
    }
    
    /// Get all loaded GP history records
    fn get_all_history(&self) -> Vec<PyGpHistoryRecord> {
        self.inner.get_all_history()
            .into_iter()
            .map(|r| PyGpHistoryRecord { inner: r })
            .collect()
    }
    
    /// Clear all in-memory data (but not disk)
    fn clear_memory(&mut self) {
        self.inner.clear_memory();
    }
    
    /// Get cache statistics
    fn cache_stats(&self) -> PyResult<Py<PyAny>> {
        use crate::persistence::CacheStats;
        unsafe {
            let py = Python::assume_attached();
            match self.inner.get_cache_stats() {
                Ok(stats) => {
                    let dict = PyDict::new(py);
                    dict.set_item("total_entries", stats.total_entries)?;
                    dict.set_item("total_size_bytes", stats.total_size_bytes)?;
                    dict.set_item("avg_access_count", stats.avg_access_count)?;
                    dict.set_item("max_size", stats.max_size)?;
                    Ok(dict.into())
                }
                Err(e) => Err(PyValueError::new_err(format!("Failed to get cache stats: {}", e))),
            }
        }
    }
    
    fn __repr__(&self) -> String {
        format!("PersistenceManager({} factors loaded)", self.inner.get_all_factors().len())
    }
}

/// Python wrapper for FactorMetadata
#[pyclass(name = "FactorMetadata")]
pub struct PyFactorMetadata {
    inner: FactorMetadata,
}

#[pymethods]
impl PyFactorMetadata {
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }
    
    #[getter]
    fn expression(&self) -> String {
        self.inner.expression.clone()
    }
    
    #[getter]
    fn metrics(&self) -> PyResult<Py<PyAny>> {
        unsafe {
            let py = Python::assume_attached();
            let dict = PyDict::new(py);
            dict.set_item("ic_mean", self.inner.metrics.ic_mean)?;
            dict.set_item("ic_ir", self.inner.metrics.ic_ir)?;
            dict.set_item("turnover", self.inner.metrics.turnover)?;
            dict.set_item("complexity_penalty", self.inner.metrics.complexity_penalty)?;
            dict.set_item("combined_score", self.inner.metrics.combined_score)?;
            Ok(dict.into())
        }
    }
    
    #[getter]
    fn tags(&self) -> Vec<String> {
        self.inner.tags.clone()
    }
    
    fn __repr__(&self) -> String {
        format!("FactorMetadata(id={}, ic={:.4})", self.inner.id, self.inner.metrics.ic_mean)
    }
}

/// Python wrapper for GPHistoryRecord
#[pyclass(name = "GPHistoryRecord")]
pub struct PyGpHistoryRecord {
    inner: GPHistoryRecord,
}

#[pymethods]
impl PyGpHistoryRecord {
    #[getter]
    fn run_id(&self) -> String {
        self.inner.run_id.clone()
    }
    
    #[getter]
    fn best_factor(&self) -> PyFactorMetadata {
        PyFactorMetadata { inner: self.inner.best_factor.clone() }
    }
    
    #[getter]
    fn config(&self) -> PyResult<Py<PyAny>> {
        unsafe {
            let py = Python::assume_attached();
            let dict = PyDict::new(py);
            dict.set_item("population_size", self.inner.config.population_size)?;
            dict.set_item("max_generations", self.inner.config.max_generations)?;
            dict.set_item("tournament_size", self.inner.config.tournament_size)?;
            dict.set_item("crossover_prob", self.inner.config.crossover_prob)?;
            dict.set_item("mutation_prob", self.inner.config.mutation_prob)?;
            dict.set_item("max_depth", self.inner.config.max_depth)?;
            Ok(dict.into())
        }
    }
    
    fn __repr__(&self) -> String {
        format!("GPHistoryRecord(id={})", self.inner.run_id)
    }
}

// ============================================================================
// Meta-learning Module Python Bindings
// ============================================================================

/// Python-exposed Meta-learning Analyzer for intelligent factor mining
#[pyclass(name = "MetaLearningAnalyzer")]
pub struct PyMetaLearningAnalyzer {
    inner: MetaLearningAnalyzer,
}

#[pymethods]
impl PyMetaLearningAnalyzer {
    #[new]
    fn new() -> Self {
        PyMetaLearningAnalyzer { inner: MetaLearningAnalyzer::new() }
    }
    
    /// Train the meta-learning model on historical data
    fn train(
        &mut self,
        factors: Vec<PyRef<PyFactorMetadata>>,
        gp_runs: Vec<PyRef<PyGpHistoryRecord>>,
    ) -> PyResult<()> {
        let factors_rust: Vec<FactorMetadata> = factors.into_iter().map(|f| f.inner.clone()).collect();
        let gp_runs_rust: Vec<GPHistoryRecord> = gp_runs.into_iter().map(|r| r.inner.clone()).collect();
        
        self.inner.train(&factors_rust, &gp_runs_rust)
            .map_err(|e| PyValueError::new_err(format!("Failed to train meta-learning model: {}", e)))
    }
    
    /// Get recommendations for next GP run
    fn get_recommendations(&self, target_complexity: Option<f64>) -> PyResult<PyGpRecommendations> {
        let recommendations = self.inner.get_recommendations(target_complexity);
        Ok(PyGpRecommendations { inner: recommendations })
    }
    
    /// Check if model is trained
    fn is_trained(&self) -> bool {
        self.inner.is_trained()
    }
    
    /// Get model version
    fn version(&self) -> u32 {
        self.inner.version()
    }
    
    /// Get confidence score of recommendations
    fn confidence_score(&self) -> f64 {
        let recommendations = self.inner.get_recommendations(None);
        recommendations.confidence_score
    }
    
    /// Save model to file
    fn save_model(&self, path: &str) -> PyResult<()> {
        self.inner.save_model(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to save model: {}", e)))
    }
    
    /// Load model from file
    #[staticmethod]
    fn load_model(path: &str) -> PyResult<Self> {
        match MetaLearningAnalyzer::load_model(path) {
            Ok(analyzer) => Ok(PyMetaLearningAnalyzer { inner: analyzer }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to load model: {}", e))),
        }
    }
    
    /// Get high performance threshold
    fn get_high_perf_threshold(&self) -> f64 {
        self.inner.get_high_perf_threshold()
    }
    
    /// Set high performance threshold
    fn set_high_perf_threshold(&mut self, threshold: f64) {
        self.inner.set_high_perf_threshold(threshold);
    }
    
    /// Get minimum data points required for training
    fn get_min_data_points(&self) -> u32 {
        self.inner.get_min_data_points()
    }
    
    /// Set minimum data points required for training
    fn set_min_data_points(&mut self, min_points: u32) {
        self.inner.set_min_data_points(min_points);
    }
    
    fn __repr__(&self) -> String {
        format!("MetaLearningAnalyzer(trained={}, version={})", 
                self.is_trained(), self.version())
    }
}

/// Python wrapper for GPRecommendations
#[pyclass(name = "GPRecommendations")]
pub struct PyGpRecommendations {
    inner: GPRecommendations,
}

#[pymethods]
impl PyGpRecommendations {
    #[getter]
    fn recommended_functions(&self) -> Vec<String> {
        self.inner.recommended_functions.clone()
    }
    
    #[getter]
    fn recommended_terminals(&self) -> Vec<String> {
        self.inner.recommended_terminals.clone()
    }
    
    #[getter]
    fn target_complexity(&self) -> f64 {
        self.inner.target_complexity
    }
    
    #[getter]
    fn confidence_score(&self) -> f64 {
        self.inner.confidence_score
    }
    
    #[getter]
    fn confidence_level(&self) -> &'static str {
        self.inner.confidence_level()
    }
    
    /// Convert recommendations to a GPConfig (using midpoint of ranges)
    fn to_gp_config(&self) -> PyResult<Py<PyAny>> {
        unsafe {
            let py = Python::assume_attached();
            let config = self.inner.to_gp_config();
            let dict = PyDict::new(py);
            dict.set_item("population_size", config.population_size)?;
            dict.set_item("max_generations", config.max_generations)?;
            dict.set_item("tournament_size", config.tournament_size)?;
            dict.set_item("crossover_prob", config.crossover_prob)?;
            dict.set_item("mutation_prob", config.mutation_prob)?;
            dict.set_item("max_depth", config.max_depth)?;
            Ok(dict.into())
        }
    }
    
    /// Check if recommendations are valid
    fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    fn __repr__(&self) -> String {
        format!("GPRecommendations(functions={}, confidence={:.2})",
                self.inner.recommended_functions.len(), self.inner.confidence_score)
    }
}
