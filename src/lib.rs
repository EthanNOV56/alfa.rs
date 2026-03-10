//! exprs core library: high-performance factor backtesting and expression evaluation
//! Exposed as Python extension via PyO3

// Core modules (public for binary usage)
pub mod al_parser;
pub mod backtest;
pub mod expr;
pub mod expr_optimizer;
pub mod factor;
pub mod gp;
pub mod lazy;
pub mod metalearning;
pub mod persistence;
pub mod polars_style;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::sync::Arc;

/// Set the number of threads for parallel processing
///
/// Note: Due to Rayon's design, this must be called BEFORE any parallel operations.
/// For best results, set the RAYON_NUM_THREADS environment variable before importing:
///     import os
///     os.environ['RAYON_NUM_THREADS'] = '4'
///     import alfars as al
#[pyfunction]
fn set_num_threads(n_threads: usize) -> PyResult<()> {
    if n_threads == 0 {
        return Err(PyValueError::new_err("n_threads must be > 0"));
    }
    // Rayon doesn't allow replacing the global pool at runtime
    // Recommend using RAYON_NUM_THREADS env var
    eprintln!("Warning: set_num_threads() is limited. For better control, set RAYON_NUM_THREADS environment variable before importing the module.");
    Ok(())
}

// Re-exports for internal use
use crate::backtest::{BacktestEngine, BacktestResult, FeeConfig, PositionConfig, SlippageConfig};
use crate::expr::{BinaryOp, Expr, Literal, UnaryOp};
use crate::lazy::{DataSource, JoinType, LazyFrame};
use crate::polars_style::{evaluate_expr_on_dataframe, DataFrame, Series};

/// Weight allocation method
#[derive(Debug, Clone, Copy)]
pub enum WeightMethod {
    Equal,
    Weighted,
}

// ============================================================================
// Expression System Python Bindings
// ============================================================================

/// Python-exposed expression

// ============================================================================
// Expression System Python Bindings
// ============================================================================

/// Python-exposed expression
#[pyclass(name = "Expr", from_py_object)]
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

    /// Get string representation - simplified to avoid infinite recursion
    fn __repr__(&self) -> String {
        "Expr(...)".to_string()
    }

    /// Get string representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python-exposed Series for vectorized operations
#[pyclass(name = "Series", from_py_object)]
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
    fn new(_py: Python<'_>, columns: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        if let Some(cols) = columns {
            let mut inner_columns = std::collections::HashMap::new();

            for (key, value) in cols.iter() {
                let col_name: String = key.extract()?;

                if let Ok(py_series) = value.extract::<PySeries>() {
                    inner_columns.insert(col_name, py_series.inner.clone());
                } else if let Ok(list) = value.extract::<Vec<f64>>() {
                    inner_columns.insert(col_name, Series::new(list));
                } else {
                    return Err(PyValueError::new_err(format!(
                        "Column '{}' must be a Series or list of floats",
                        col_name
                    )));
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

/// Parse expression string into Expr AST
#[pyfunction]
fn parse_expression(expression: &str) -> PyResult<PyExpr> {
    match factor::parse_expression(expression) {
        Ok(expr) => Ok(PyExpr { inner: expr }),
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

/// Evaluate expression on multi-asset data (returns factor matrix)
#[allow(unused_imports)]
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
                return Err(PyValueError::new_err(format!(
                    "Column '{}' has shape {:?}, expected ({}, {})",
                    col_name, shape, n_days, n_assets
                )));
            }
            column_arrays.insert(col_name, array.as_array().to_owned());
        } else {
            return Err(PyValueError::new_err(format!(
                "Column '{}' must be a 2D numpy array",
                col_name
            )));
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
    Ok(PyExpr {
        inner: cumprod_expr,
    })
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
    Ok(PyExpr {
        inner: ts_rank_expr,
    })
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
    Ok(PyExpr {
        inner: ts_argmax_expr,
    })
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
    Ok(PyExpr {
        inner: ts_argmin_expr,
    })
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
    Ok(PyExpr {
        inner: ts_corr_expr,
    })
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
        args: vec![expr.inner.clone(), Expr::Literal(Literal::Float(exponent))],
    };
    Ok(PyExpr { inner: power_expr })
}

/// Time series sum - rolling sum over `window` time periods
/// window=0 means expanding window (cumulative sum from start to current)
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

/// Time series count - rolling count over `window` time periods
/// window=0 means expanding window (cumulative count from start)
#[pyfunction]
fn ts_count(expr: &PyExpr, window: usize) -> PyResult<PyExpr> {
    let ts_count_expr = Expr::FunctionCall {
        name: "ts_count".to_string(),
        args: vec![
            expr.inner.clone(),
            Expr::Literal(Literal::Integer(window as i64)),
        ],
    };
    Ok(PyExpr {
        inner: ts_count_expr,
    })
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
    // Backtest configuration
    m.add_class::<PySlippageConfig>()?;
    m.add_class::<PyFeeConfig>()?;
    m.add_class::<PyPositionConfig>()?;

    // Backtest functionality
    m.add_class::<PyBacktestEngine>()?;
    m.add_function(wrap_pyfunction!(quantile_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ic, m)?)?;

    // Expression system
    m.add_class::<PyExpr>()?;
    m.add_class::<PySeries>()?;
    m.add_class::<PyDataFrame>()?;

    // Expression evaluation functions
    m.add_function(wrap_pyfunction!(parse_expression, m)?)?;
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
    m.add_function(wrap_pyfunction!(ts_count, m)?)?;
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
    m.add_class::<PyAlFactor>()?;
    m.add_function(wrap_pyfunction!(load_al_factors, m)?)?;

    // Meta-learning system
    m.add_class::<PyMetaLearningAnalyzer>()?;
    m.add_class::<PyGpRecommendations>()?;

    // Factor Registry
    m.add_class::<PyFactorRegistry>()?;
    m.add_class::<PyFactorInfo>()?;
    m.add_class::<PyFactorResult>()?;

    // Threading control
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;

    Ok(())
}

/// Volume-based slippage configuration
#[pyclass]
struct PySlippageConfig {
    #[pyo3(get, set)]
    large_volume_threshold: f64,
    #[pyo3(get, set)]
    large_slippage_rate: f64,
    #[pyo3(get, set)]
    normal_slippage_rate: f64,
}

#[pymethods]
impl PySlippageConfig {
    #[new]
    fn new(
        large_volume_threshold: f64,
        large_slippage_rate: f64,
        normal_slippage_rate: f64,
    ) -> Self {
        Self {
            large_volume_threshold,
            large_slippage_rate,
            normal_slippage_rate,
        }
    }
}

impl From<PySlippageConfig> for SlippageConfig {
    fn from(py_config: PySlippageConfig) -> Self {
        SlippageConfig {
            large_volume_threshold: py_config.large_volume_threshold,
            large_slippage_rate: py_config.large_slippage_rate,
            normal_slippage_rate: py_config.normal_slippage_rate,
        }
    }
}

/// Fee configuration
#[pyclass]
struct PyFeeConfig {
    #[pyo3(get, set)]
    commission_rate: f64,
    #[pyo3(get, set)]
    large_volume_threshold: f64,
    #[pyo3(get, set)]
    large_slippage_rate: f64,
    #[pyo3(get, set)]
    normal_slippage_rate: f64,
    #[pyo3(get, set)]
    min_commission: f64,
}

#[pymethods]
impl PyFeeConfig {
    #[new]
    fn new(
        commission_rate: f64,
        large_volume_threshold: f64,
        large_slippage_rate: f64,
        normal_slippage_rate: f64,
        min_commission: f64,
    ) -> Self {
        Self {
            commission_rate,
            large_volume_threshold,
            large_slippage_rate,
            normal_slippage_rate,
            min_commission,
        }
    }
}

impl From<PyFeeConfig> for FeeConfig {
    fn from(py_config: PyFeeConfig) -> Self {
        FeeConfig {
            commission_rate: py_config.commission_rate,
            slippage: SlippageConfig {
                large_volume_threshold: py_config.large_volume_threshold,
                large_slippage_rate: py_config.large_slippage_rate,
                normal_slippage_rate: py_config.normal_slippage_rate,
            },
            min_commission: py_config.min_commission,
        }
    }
}

/// Position configuration
#[pyclass]
struct PyPositionConfig {
    #[pyo3(get, set)]
    long_ratio: f64,
    #[pyo3(get, set)]
    short_ratio: f64,
    #[pyo3(get, set)]
    market_neutral: bool,
}

#[pymethods]
impl PyPositionConfig {
    #[new]
    fn new(long_ratio: f64, short_ratio: f64, market_neutral: bool) -> Self {
        Self {
            long_ratio,
            short_ratio,
            market_neutral,
        }
    }
}

impl From<PyPositionConfig> for PositionConfig {
    fn from(py_config: PyPositionConfig) -> Self {
        PositionConfig {
            long_ratio: py_config.long_ratio,
            short_ratio: py_config.short_ratio,
            market_neutral: py_config.market_neutral,
        }
    }
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
        _py: Python<'_>,
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
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "weight_method must be 'equal' or 'weighted'",
                ))
            }
        };

        let engine = BacktestEngine::new_simple(
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
    /// Total return
    #[pyo3(get)]
    total_return: f64,
    /// Annualized return
    #[pyo3(get)]
    annualized_return: f64,
    /// Sharpe ratio (annualized)
    #[pyo3(get)]
    sharpe_ratio: f64,
    /// Maximum drawdown
    #[pyo3(get)]
    max_drawdown: f64,
    /// Turnover rate
    #[pyo3(get)]
    turnover: f64,
    /// Long-only returns
    #[pyo3(get)]
    long_returns: Py<PyArray1<f64>>,
    /// Short-only returns
    #[pyo3(get)]
    short_returns: Py<PyArray1<f64>>,
}

impl From<BacktestResult> for PyBacktestResult {
    fn from(result: BacktestResult) -> Self {
        pyo3::Python::try_attach(|py| Self {
            group_returns: result.group_returns.into_pyarray(py).into(),
            group_cum_returns: result.group_cum_returns.into_pyarray(py).into(),
            long_short_returns: result.long_short_returns.into_pyarray(py).into(),
            long_short_cum_return: result.long_short_cum_return,
            ic_series: result.ic_series.into_pyarray(py).into(),
            ic_mean: result.ic_mean,
            ic_ir: result.ic_ir,
            total_return: result.total_return,
            annualized_return: result.annualized_return,
            sharpe_ratio: result.sharpe_ratio,
            max_drawdown: result.max_drawdown,
            turnover: result.turnover,
            long_returns: result.long_returns.into_pyarray(py).into(),
            short_returns: result.short_returns.into_pyarray(py).into(),
        })
        .unwrap()
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
        factor,
        returns,
        quantiles,
        weight_method,
        long_top_n,
        short_top_n,
        commission_rate,
        weights,
    )?;
    engine.run()
}

/// Standalone IC computation (Python interface)
#[pyfunction]
fn compute_ic(
    _py: Python<'_>,
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
    let ic_std = (ic_vals.iter().map(|&x| (x - ic_mean).powi(2)).sum::<f64>()
        / (ic_vals.len() - 1) as f64)
        .sqrt();
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

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
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
    fn scan(_py: Python<'_>, data: Bound<'_, PyDict>) -> PyResult<Self> {
        use numpy::PyArray2;

        let mut arrays = std::collections::HashMap::new();

        for (key, value) in data.iter() {
            let col_name: String = key.extract()?;

            if let Ok(arr) = value.extract::<Bound<'_, PyArray2<f64>>>() {
                let array = arr.readonly().as_array().to_owned();
                arrays.insert(col_name, array);
            } else {
                return Err(PyValueError::new_err(format!(
                    "Column '{}' must be a 2D numpy array",
                    col_name
                )));
            }
        }

        let source = DataSource::NumpyArrays(arrays);
        let lazy_frame = LazyFrame::scan(source);

        Ok(PyLazyFrame {
            inner: Some(lazy_frame),
        })
    }

    /// Add new columns to the LazyFrame
    fn with_columns(&self, _py: Python<'_>, exprs: Bound<'_, PyList>) -> PyResult<Self> {
        let Some(ref inner) = self.inner else {
            return Err(PyValueError::new_err("LazyFrame is already consumed"));
        };

        // Convert Python expressions to Rust expressions
        let mut rust_exprs = Vec::new();
        for item in exprs.iter() {
            let tuple = item.cast::<PyTuple>()?;
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
            _ => {
                return Err(PyValueError::new_err(
                    "Join type must be 'inner', 'left', 'right', or 'outer'",
                ))
            }
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

        pyo3::Python::try_attach(|py| match lazy_frame.collect() {
            Ok(result) => {
                let dict = PyDict::new(py);
                for (key, array) in result {
                    let py_array = array.into_pyarray(py);
                    dict.set_item(key, py_array)?;
                }
                Ok(dict.into())
            }
            Err(e) => Err(PyValueError::new_err(e)),
        })
        .ok_or_else(|| PyRuntimeError::new_err("Failed to attach to Python"))?
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
        let _window_spec = crate::lazy::rolling_window(size, min_periods);
        let dict = PyDict::new(py);
        dict.set_item("kind", "rolling")?;
        dict.set_item("size", size)?;
        dict.set_item("min_periods", min_periods.unwrap_or(1))?;
        Ok(dict.into())
    })
    .ok_or_else(|| PyRuntimeError::new_err("Failed to attach to Python"))?
}

/// Create an expanding window specification for lazy evaluation
#[pyfunction]
fn expanding_window(min_periods: Option<usize>) -> PyResult<Py<PyDict>> {
    pyo3::Python::try_attach(|py| {
        let _window_spec = crate::lazy::expanding_window(min_periods);
        let dict = PyDict::new(py);
        dict.set_item("kind", "expanding")?;
        dict.set_item("min_periods", min_periods.unwrap_or(1))?;
        Ok(dict.into())
    })
    .ok_or_else(|| PyRuntimeError::new_err("Failed to attach to Python"))?
}

// ============================================================================
// Genetic Programming (GP) Module Python Bindings
// ============================================================================

use crate::gp::{
    run_gp, BacktestFitnessEvaluator, DataSplitConfig, Function, GPConfig,
    RealBacktestFitnessEvaluator, Terminal,
};
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
    #[pyo3(signature = (population_size=100, max_generations=50, tournament_size=7, crossover_prob=0.8, mutation_prob=0.2, max_depth=6, allow_ephemeral=true))]
    fn new(
        population_size: usize,
        max_generations: usize,
        tournament_size: usize,
        crossover_prob: f64,
        mutation_prob: f64,
        max_depth: usize,
        allow_ephemeral: bool,
    ) -> Self {
        // Create default terminals (will be updated later)
        let mut terminals = vec![Terminal::Constant(1.0), Terminal::Constant(2.0)];

        if allow_ephemeral {
            terminals.insert(0, Terminal::Ephemeral);
        }

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

        PyGpEngine {
            config,
            terminals,
            functions,
            rng,
        }
    }

    /// Set available columns (variables) for expression generation
    #[pyo3(signature = (columns, allow_ephemeral = None))]
    fn set_columns(
        &mut self,
        _py: Python<'_>,
        columns: Bound<'_, PyList>,
        allow_ephemeral: Option<bool>,
    ) {
        // Check if ephemeral is allowed (default to true if not specified)
        let allow_ephemeral = allow_ephemeral.unwrap_or(true);

        // Update terminals with column variables
        let mut new_terminals = vec![Terminal::Constant(1.0), Terminal::Constant(2.0)];

        if allow_ephemeral {
            new_terminals.insert(0, Terminal::Ephemeral);
        }

        for item in columns.iter() {
            let col: String = item.extract().unwrap_or_default();
            new_terminals.push(Terminal::Variable(col));
        }

        self.terminals = new_terminals;
    }

    /// Run genetic programming for factor mining (backward compatible)
    fn mine_factors(
        &mut self,
        _py: Python<'_>,
        data: Bound<'_, PyDict>,
        returns: Bound<'_, PyArray2<f64>>,
        num_factors: usize,
        weight_ic: Option<f64>,
        weight_ir: Option<f64>,
        weight_turnover: Option<f64>,
        weight_complexity: Option<f64>,
    ) -> PyResult<Vec<(String, f64, f64, f64, f64, usize)>> {
        use numpy::PyArray2;

        // Extract data arrays
        let mut data_arrays = HashMap::new();

        for (key, value) in data.iter() {
            let col_name: String = key.extract()?;

            if let Ok(arr) = value.extract::<Bound<'_, PyArray2<f64>>>() {
                let array = arr.readonly().as_array().to_owned();
                data_arrays.insert(col_name, array);
            } else {
                return Err(PyValueError::new_err(format!(
                    "Column '{}' must be a 2D numpy array",
                    col_name
                )));
            }
        }

        // Extract returns array
        let returns_array = returns.readonly().as_array().to_owned();

        // Create multi-objective fitness evaluator (without split - backward compatible)
        let mut evaluator = RealBacktestFitnessEvaluator::new(data_arrays, returns_array);

        // Set weights for multi-objective optimization
        let w_ic = weight_ic.unwrap_or(0.4);
        let w_ir = weight_ir.unwrap_or(0.3);
        let w_to = weight_turnover.unwrap_or(0.15);
        let w_comp = weight_complexity.unwrap_or(0.15);

        let weights = HashMap::from([
            ("ic".to_string(), w_ic),
            ("ir".to_string(), w_ir),
            ("turnover".to_string(), w_to),
            ("complexity".to_string(), w_comp),
        ]);
        evaluator.set_weights(weights);

        // Run GP multiple times to get multiple factors
        let mut results = Vec::new();

        for _i in 0..num_factors {
            let (best_expr, best_fitness) = run_gp(
                &self.config,
                &evaluator,
                self.terminals.clone(),
                self.functions.clone(),
                &mut self.rng,
            );

            // Convert expression to string representation
            let expr_str = format!("{:?}", best_expr);

            // Get additional metrics from evaluator
            let ic = evaluator.get_last_ic();
            let ir = evaluator.get_last_ir();
            let turnover = evaluator.get_last_turnover();
            let complexity = evaluator.get_last_complexity();

            results.push((expr_str, best_fitness, ic, ir, turnover, complexity));
        }

        Ok(results)
    }

    /// Run genetic programming with train/test/validation split
    fn mine_factors_with_split(
        &mut self,
        _py: Python<'_>,
        data: Bound<'_, PyDict>,
        returns: Bound<'_, PyArray2<f64>>,
        num_factors: usize,
        train_ratio: f64,
        validation_ratio: f64,
        weight_ic: Option<f64>,
        weight_ir: Option<f64>,
        weight_turnover: Option<f64>,
        weight_complexity: Option<f64>,
    ) -> PyResult<
        Vec<(
            String,
            f64,
            f64,
            f64,
            f64,
            usize,
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
        )>,
    > {
        use numpy::PyArray2;

        // Validate split ratios
        if train_ratio <= 0.0 || validation_ratio <= 0.0 {
            return Err(PyValueError::new_err(
                "train_ratio and validation_ratio must be positive",
            ));
        }
        if train_ratio + validation_ratio >= 1.0 {
            return Err(PyValueError::new_err(
                "train_ratio + validation_ratio must be < 1.0",
            ));
        }

        // Extract data arrays
        let mut data_arrays = HashMap::new();

        for (key, value) in data.iter() {
            let col_name: String = key.extract()?;

            if let Ok(arr) = value.extract::<Bound<'_, PyArray2<f64>>>() {
                let array = arr.readonly().as_array().to_owned();
                data_arrays.insert(col_name, array);
            } else {
                return Err(PyValueError::new_err(format!(
                    "Column '{}' must be a 2D numpy array",
                    col_name
                )));
            }
        }

        // Extract returns array
        let returns_array = returns.readonly().as_array().to_owned();

        // Create split config
        let split_config = DataSplitConfig {
            train_ratio,
            validation_ratio,
            test_ratio: Some(1.0 - train_ratio - validation_ratio),
        };

        // Create evaluator with split
        let mut evaluator =
            RealBacktestFitnessEvaluator::with_split(data_arrays, returns_array, split_config);

        // Set weights for multi-objective optimization
        let w_ic = weight_ic.unwrap_or(0.4);
        let w_ir = weight_ir.unwrap_or(0.3);
        let w_to = weight_turnover.unwrap_or(0.15);
        let w_comp = weight_complexity.unwrap_or(0.15);

        let weights = HashMap::from([
            ("ic".to_string(), w_ic),
            ("ir".to_string(), w_ir),
            ("turnover".to_string(), w_to),
            ("complexity".to_string(), w_comp),
        ]);
        evaluator.set_weights(weights);

        // Run GP multiple times to get multiple factors
        let mut results = Vec::new();

        for _i in 0..num_factors {
            let (best_expr, best_fitness) = run_gp(
                &self.config,
                &evaluator,
                self.terminals.clone(),
                self.functions.clone(),
                &mut self.rng,
            );

            // Convert expression to string representation
            let expr_str = format!("{:?}", best_expr);

            // Get additional metrics from evaluator
            let ic = evaluator.get_last_ic();
            let ir = evaluator.get_last_ir();
            let turnover = evaluator.get_last_turnover();
            let complexity = evaluator.get_last_complexity();

            // Get split evaluation results
            let split_result = evaluator.get_last_split_result();
            let (train_metrics, validation_metrics, test_metrics) =
                if let Some(ref sr) = split_result {
                    (
                        vec![
                            sr.train.ic_mean,
                            sr.train.ic_ir,
                            sr.train.sharpe_ratio,
                            sr.train.max_drawdown,
                        ],
                        vec![
                            sr.validation.ic_mean,
                            sr.validation.ic_ir,
                            sr.validation.sharpe_ratio,
                            sr.validation.max_drawdown,
                        ],
                        vec![
                            sr.test.ic_mean,
                            sr.test.ic_ir,
                            sr.test.sharpe_ratio,
                            sr.test.max_drawdown,
                        ],
                    )
                } else {
                    (
                        vec![0.0, 0.0, 0.0, 0.0],
                        vec![0.0, 0.0, 0.0, 0.0],
                        vec![0.0, 0.0, 0.0, 0.0],
                    )
                };

            results.push((
                expr_str,
                best_fitness,
                ic,
                ir,
                turnover,
                complexity,
                train_metrics,
                validation_metrics,
                test_metrics,
            ));
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

        Ok(format!(
            "Best expression: {:?}, Fitness: {:.6}",
            best_expr, fitness
        ))
    }
}

// ============================================================================
// Persistence Module Python Bindings
// ============================================================================

use crate::metalearning::{GPRecommendations, MetaLearningAnalyzer};
use crate::persistence::{AlFactor, AlParser, FactorMetadata, GPHistoryRecord, PersistenceManager};

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
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to create persistence manager: {}",
                e
            ))),
        }
    }

    /// Save a factor to disk
    fn save_factor(&mut self, factor: &PyFactorMetadata) -> PyResult<()> {
        self.inner
            .save_factor(&factor.inner)
            .map_err(|e| PyValueError::new_err(format!("Failed to save factor: {}", e)))
    }

    /// Load a factor from disk
    fn load_factor(&mut self, factor_id: &str) -> PyResult<Option<PyFactorMetadata>> {
        match self.inner.load_factor(factor_id) {
            Ok(Some(factor)) => Ok(Some(PyFactorMetadata { inner: factor })),
            Ok(None) => Ok(None),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to load factor: {}",
                e
            ))),
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
        let factors = self
            .inner
            .search_factors(min_ic, None, max_complexity, &tags);
        Ok(factors
            .into_iter()
            .map(|f| PyFactorMetadata { inner: f })
            .collect())
    }

    /// Load all factors from disk
    fn load_all_factors(&mut self) -> PyResult<usize> {
        self.inner
            .load_all_factors()
            .map_err(|e| PyValueError::new_err(format!("Failed to load factors: {}", e)))
    }

    /// Load all GP history records from disk
    fn load_all_history(&mut self) -> PyResult<usize> {
        self.inner
            .load_all_history()
            .map_err(|e| PyValueError::new_err(format!("Failed to load history: {}", e)))
    }

    /// Get all loaded factors
    fn get_all_factors(&self) -> Vec<PyFactorMetadata> {
        self.inner
            .get_all_factors()
            .into_iter()
            .map(|f| PyFactorMetadata { inner: f })
            .collect()
    }

    /// Get all loaded GP history records
    fn get_all_history(&self) -> Vec<PyGpHistoryRecord> {
        self.inner
            .get_all_history()
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
                Err(e) => Err(PyValueError::new_err(format!(
                    "Failed to get cache stats: {}",
                    e
                ))),
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PersistenceManager({} factors loaded)",
            self.inner.get_all_factors().len()
        )
    }

    /// Load factors from .al files in ~/.alfars/ directory
    fn load_from_al(&mut self) -> PyResult<Vec<PyAlFactor>> {
        match self.inner.load_from_al() {
            Ok(factors) => Ok(factors
                .into_iter()
                .map(|f| PyAlFactor { inner: f })
                .collect()),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to load .al files: {}",
                e
            ))),
        }
    }

    /// Save a factor to .al file in ~/.alfars/ directory
    fn save_to_al(&self, factor: &PyAlFactor, filename: Option<String>) -> PyResult<String> {
        match self.inner.save_to_al(&factor.inner, filename.as_deref()) {
            Ok(path) => Ok(path.to_string_lossy().to_string()),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to save .al file: {}",
                e
            ))),
        }
    }
}

/// Python wrapper for AlFactor (.al file format)
#[pyclass(name = "AlFactor")]
pub struct PyAlFactor {
    inner: AlFactor,
}

#[pymethods]
impl PyAlFactor {
    #[new]
    fn new(
        name: String,
        expression: String,
        description: String,
        dimension: String,
        tags: Vec<String>,
    ) -> Self {
        PyAlFactor {
            inner: AlFactor {
                name,
                expression,
                description,
                dimension,
                tags,
                readonly: false,
            },
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn expression(&self) -> String {
        self.inner.expression.clone()
    }

    #[getter]
    fn description(&self) -> String {
        self.inner.description.clone()
    }

    #[getter]
    fn dimension(&self) -> String {
        self.inner.dimension.clone()
    }

    #[getter]
    fn tags(&self) -> Vec<String> {
        self.inner.tags.clone()
    }

    #[getter]
    fn readonly(&self) -> bool {
        self.inner.readonly
    }

    fn __repr__(&self) -> String {
        format!(
            "AlFactor(name={}, expression={})",
            self.inner.name, self.inner.expression
        )
    }

    /// Save factor to .al file in default directory (~/.alfars/)
    #[pyo3(signature = (filename = None))]
    fn save_to_al(&self, filename: Option<String>) -> PyResult<String> {
        match AlParser::save_to_default_dir(&self.inner, filename.as_deref()) {
            Ok(path) => Ok(path.to_string_lossy().to_string()),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to save .al file: {}",
                e
            ))),
        }
    }
}

/// Load all factors from default directory (~/.alfars/)
#[pyfunction]
fn load_al_factors() -> PyResult<Vec<PyAlFactor>> {
    match AlParser::load_from_default_dir() {
        Ok(factors) => Ok(factors
            .into_iter()
            .map(|f| PyAlFactor { inner: f })
            .collect()),
        Err(e) => Err(PyValueError::new_err(format!(
            "Failed to load .al files: {}",
            e
        ))),
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
        format!(
            "FactorMetadata(id={}, ic={:.4})",
            self.inner.id, self.inner.metrics.ic_mean
        )
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
        PyFactorMetadata {
            inner: self.inner.best_factor.clone(),
        }
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
        PyMetaLearningAnalyzer {
            inner: MetaLearningAnalyzer::new(),
        }
    }

    /// Train the meta-learning model on historical data
    fn train(
        &mut self,
        factors: Vec<PyRef<PyFactorMetadata>>,
        gp_runs: Vec<PyRef<PyGpHistoryRecord>>,
    ) -> PyResult<()> {
        let factors_rust: Vec<FactorMetadata> =
            factors.into_iter().map(|f| f.inner.clone()).collect();
        let gp_runs_rust: Vec<GPHistoryRecord> =
            gp_runs.into_iter().map(|r| r.inner.clone()).collect();

        self.inner.train(&factors_rust, &gp_runs_rust).map_err(|e| {
            PyValueError::new_err(format!("Failed to train meta-learning model: {}", e))
        })
    }

    /// Get recommendations for next GP run
    fn get_recommendations(&self, target_complexity: Option<f64>) -> PyResult<PyGpRecommendations> {
        let recommendations = self.inner.get_recommendations(target_complexity);
        Ok(PyGpRecommendations {
            inner: recommendations,
        })
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
        self.inner
            .save_model(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to save model: {}", e)))
    }

    /// Load model from file
    #[staticmethod]
    fn load_model(path: &str) -> PyResult<Self> {
        match MetaLearningAnalyzer::load_model(path) {
            Ok(analyzer) => Ok(PyMetaLearningAnalyzer { inner: analyzer }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to load model: {}",
                e
            ))),
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
        format!(
            "MetaLearningAnalyzer(trained={}, version={})",
            self.is_trained(),
            self.version()
        )
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
        format!(
            "GPRecommendations(functions={}, confidence={:.2})",
            self.inner.recommended_functions.len(),
            self.inner.confidence_score
        )
    }
}

// ============================================================================
// Factor Registry Python Bindings
// ============================================================================

use crate::factor::{ComputeConfig, FactorRegistry};

/// Python wrapper for FactorRegistry
#[pyclass(name = "FactorRegistry")]
pub struct PyFactorRegistry {
    inner: FactorRegistry,
}

#[pymethods]
impl PyFactorRegistry {
    /// Create a new FactorRegistry
    /// - mode: "default", "conservative", or "high_performance"
    #[new]
    fn new(mode: &str) -> Self {
        let config = match mode {
            "conservative" => ComputeConfig::conservative(),
            "high_performance" => ComputeConfig::high_performance(),
            _ => ComputeConfig::default(),
        };
        PyFactorRegistry {
            inner: FactorRegistry::with_config(config),
        }
    }

    /// Set available columns (e.g., ["close", "volume", "open"])
    fn set_columns(&mut self, columns: Vec<String>) {
        self.inner.set_columns(columns);
    }

    /// Get available column names
    fn columns(&self) -> Vec<String> {
        self.inner.columns()
    }

    /// Register a factor expression (auto-parses and generates plan)
    fn register(&mut self, name: &str, expression: &str) -> PyResult<String> {
        self.inner
            .register(name, expression)
            .map_err(|e| PyValueError::new_err(e))
    }

    /// Compute factor with data (with timeout protection)
    /// data: dict of column_name -> list of values
    /// Returns FactorResult with values array
    fn compute(
        &self,
        _py: Python<'_>,
        name: &str,
        data: Bound<'_, PyDict>,
    ) -> PyResult<PyFactorResult> {
        // Convert Python dict to HashMap
        let mut hashmap = std::collections::HashMap::new();
        for (key, value) in data.iter() {
            let key_str: String = key.extract()?;
            let values: Vec<f64> = value.extract()?;
            hashmap.insert(key_str, values);
        }

        self.inner
            .compute(name, &hashmap)
            .map(|r| PyFactorResult {
                name: r.name,
                values: r.values,
                n_rows: r.n_rows,
                n_cols: r.n_cols,
                compute_time_ms: r.compute_time_ms,
            })
            .map_err(|e| PyValueError::new_err(e))
    }

    /// Batch compute multiple factors with shared subexpression optimization
    /// names: list of factor names to compute
    /// data: dict of column_name -> list of values
    /// parallel: whether to use parallel computation
    /// Returns dict of name -> FactorResult
    fn compute_batch(
        &self,
        py: Python<'_>,
        names: Vec<String>,
        data: Bound<'_, PyDict>,
        parallel: bool,
    ) -> PyResult<Py<PyDict>> {
        // Convert Python dict to HashMap
        let mut hashmap = std::collections::HashMap::new();
        for (key, value) in data.iter() {
            let key_str: String = key.extract()?;
            let values: Vec<f64> = value.extract()?;
            hashmap.insert(key_str, values);
        }

        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

        let results = self
            .inner
            .compute_batch(&name_refs, &hashmap, parallel)
            .map_err(|e| PyValueError::new_err(e))?;

        // Convert results to Python dict
        let result_dict = PyDict::new(py);
        for (name, r) in results {
            result_dict.set_item(
                name,
                PyFactorResult {
                    name: r.name,
                    values: r.values,
                    n_rows: r.n_rows,
                    n_cols: r.n_cols,
                    compute_time_ms: r.compute_time_ms,
                },
            )?;
        }
        Ok(result_dict.into())
    }

    /// List all registered factor names
    fn list(&self) -> Vec<String> {
        self.inner.list()
    }

    /// Get factor info
    fn get(&self, name: &str) -> Option<PyFactorInfo> {
        self.inner.get(name).map(|info| PyFactorInfo {
            name: info.name.clone(),
            expression: info.expression.clone(),
            description: info.description.clone(),
            category: info.category.clone(),
        })
    }

    /// Unregister a factor
    fn unregister(&mut self, name: &str) -> bool {
        self.inner.unregister(name)
    }

    /// Clear all factors
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get compute config info
    fn get_config(&self) -> PyResult<Py<PyDict>> {
        unsafe {
            let py = Python::assume_attached();
            let dict = PyDict::new(py);
            let c = self.inner.config();
            dict.set_item("timeout_secs", c.timeout_secs)?;
            dict.set_item("max_workers", c.max_workers)?;
            dict.set_item("batch_size", c.batch_size)?;
            dict.set_item("memory_limit_mb", c.memory_limit_mb)?;
            Ok(dict.into())
        }
    }

    fn __repr__(&self) -> String {
        let n = self.inner.list().len();
        format!("FactorRegistry(factors={})", n)
    }

    fn __len__(&self) -> usize {
        self.inner.list().len()
    }
}

/// Python wrapper for factor information
#[pyclass(name = "FactorInfo")]
pub struct PyFactorInfo {
    name: String,
    expression: String,
    description: Option<String>,
    category: Option<String>,
}

#[pymethods]
impl PyFactorInfo {
    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    fn expression(&self) -> String {
        self.expression.clone()
    }

    #[getter]
    fn description(&self) -> Option<String> {
        self.description.clone()
    }

    #[getter]
    fn category(&self) -> Option<String> {
        self.category.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "FactorInfo(name={}, expression={})",
            self.name, self.expression
        )
    }
}

/// Python wrapper for factor computation result
#[pyclass(name = "FactorResult")]
pub struct PyFactorResult {
    name: String,
    values: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
    compute_time_ms: u64,
}

#[pymethods]
impl PyFactorResult {
    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    fn values(&self) -> Vec<f64> {
        self.values.clone()
    }

    #[getter]
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    #[getter]
    fn n_cols(&self) -> usize {
        self.n_cols
    }

    #[getter]
    fn compute_time_ms(&self) -> u64 {
        self.compute_time_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "FactorResult(name={}, n_rows={}, time_ms={})",
            self.name, self.n_rows, self.compute_time_ms
        )
    }
}
