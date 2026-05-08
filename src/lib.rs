//! exprs core library: high-performance factor backtesting and expression evaluation
//! Exposed as Python extension via PyO3

// Core modules (public for binary usage)
pub mod al;
pub mod backtest;
pub mod data;
pub mod expr;
pub mod gp;
pub mod lab;
pub mod persistence;

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};
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
    eprintln!(
        "Warning: set_num_threads() is limited. For better control, set RAYON_NUM_THREADS environment variable before importing the module."
    );
    Ok(())
}

// Re-exports for internal use
use crate::backtest::{BacktestEngine, BacktestResult, FeeConfig, PositionConfig, SlippageConfig};
use crate::data::clickhouse::ClickHouseSource;
use crate::data::layer::{DataLayer, PriceMatrix};
use crate::expr::registry::config::FactorSlice;
use crate::expr::{BinaryOp, Expr, Literal, UnaryOp};

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

/// Parse expression string into Expr AST
#[pyfunction]
fn parse_expression(expression: &str) -> PyResult<PyExpr> {
    match expr::parse_expression(expression) {
        Ok(expr) => Ok(PyExpr { inner: expr }),
        Err(e) => Err(PyValueError::new_err(e)),
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

    // Process assets in parallel, using eval_expr_vectorized directly on Array1<f64>
    let asset_results: Vec<_> = (0..n_assets)
        .into_par_iter()
        .map(|asset_idx| {
            use crate::expr::registry::functions::eval_expr_vectorized;
            use ndarray::Array1;
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut columns: std::collections::HashMap<String, Array1<f64>> =
                std::collections::HashMap::new();

            for (col_name, array) in &column_arrays_clone {
                let column_data = array.column(asset_idx).to_owned();
                columns.insert(col_name.clone(), column_data);
            }

            // Pre-populate cache with column hashes
            let mut cache = ahash::AHashMap::new();
            for (name, arr) in &columns {
                let mut hasher = DefaultHasher::new();
                0u8.hash(&mut hasher);
                name.hash(&mut hasher);
                cache.insert(hasher.finish(), arr.clone());
            }

            match eval_expr_vectorized(&expr_inner, &columns, &mut cache) {
                Ok(arr) => arr.to_vec(),
                Err(_) => vec![f64::NAN; n_days],
            }
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
        freq: None,
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
        freq: None,
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
        freq: None,
    };
    Ok(PyExpr { inner: ma_expr })
}

/// Create a cumulative sum expression
#[pyfunction]
fn cumsum(expr: &PyExpr) -> PyResult<PyExpr> {
    let cumsum_expr = Expr::FunctionCall {
        name: "cumsum".to_string(),
        args: vec![expr.inner.clone()],
        freq: None,
    };
    Ok(PyExpr { inner: cumsum_expr })
}

/// Create a cumulative product expression
#[pyfunction]
fn cumprod(expr: &PyExpr) -> PyResult<PyExpr> {
    let cumprod_expr = Expr::FunctionCall {
        name: "cumprod".to_string(),
        args: vec![expr.inner.clone()],
        freq: None,
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
        freq: None,
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
        freq: None,
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
        freq: None,
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
        freq: None,
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
        freq: None,
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
        freq: None,
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
        freq: None,
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
        freq: None,
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
        freq: None,
    };
    Ok(PyExpr { inner: decay_expr })
}

/// Sign function - returns -1, 0, or 1
#[pyfunction]
fn sign(expr: &PyExpr) -> PyResult<PyExpr> {
    let sign_expr = Expr::FunctionCall {
        name: "sign".to_string(),
        args: vec![expr.inner.clone()],
        freq: None,
    };
    Ok(PyExpr { inner: sign_expr })
}

/// Power function - raises expression to a power
#[pyfunction]
fn power(expr: &PyExpr, exponent: f64) -> PyResult<PyExpr> {
    let power_expr = Expr::FunctionCall {
        name: "power".to_string(),
        args: vec![expr.inner.clone(), Expr::Literal(Literal::Float(exponent))],
        freq: None,
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
        freq: None,
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
        freq: None,
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
        freq: None,
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
        freq: None,
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
    m.add_class::<PyBacktestResult>()?;
    m.add_function(wrap_pyfunction!(quantile_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ic, m)?)?;

    // Expression system
    m.add_class::<PyExpr>()?;

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

    // Factor pool and redundancy detection
    m.add_class::<PyFactorPool>()?;
    m.add_function(wrap_pyfunction!(expr_similarity, m)?)?;

    // Data source + pipeline (ClickHouse → DataLayer → FactorSlice → PriceMatrix)
    m.add_class::<PyClickHouseSource>()?;
    m.add_class::<PyDataLayer>()?;
    m.add_class::<PyPriceMatrix>()?;
    m.add_class::<PyFactorSlice>()?;

    // Factor → Position abstraction + multi-factor combination
    m.add_class::<PyFactorCombiner>()?;
    m.add_class::<PyPositionBuilder>()?;

    // Factor Registry
    m.add_class::<PyFactorRegistry>()?;
    m.add_class::<PyFactorInfo>()?;
    m.add_class::<PyFactorResult>()?;

    // Unified lab entry point
    m.add_class::<PyAlfarsLab>()?;
    m.add_class::<PyFactorPanel>()?;

    // Data pipeline configuration
    m.add_class::<PyCachePolicy>()?;
    m.add_class::<PyDataPoolConfig>()?;

    // Threading control
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;

    Ok(())
}

/// Slippage configuration
#[pyclass(name = "SlippageConfig")]
struct PySlippageConfig {
    #[pyo3(get, set)]
    large_volume_threshold: f64,
    #[pyo3(get, set)]
    buy_slippage: f64,
    #[pyo3(get, set)]
    sell_slippage: f64,
}

#[pymethods]
impl PySlippageConfig {
    #[new]
    fn new(large_volume_threshold: f64, buy_slippage: f64, sell_slippage: f64) -> Self {
        Self {
            large_volume_threshold,
            buy_slippage,
            sell_slippage,
        }
    }
}

impl From<PySlippageConfig> for SlippageConfig {
    fn from(py_config: PySlippageConfig) -> Self {
        SlippageConfig {
            large_volume_threshold: py_config.large_volume_threshold,
            buy_slippage: py_config.buy_slippage,
            sell_slippage: py_config.sell_slippage,
        }
    }
}

/// Fee configuration
#[pyclass(name = "FeeConfig")]
struct PyFeeConfig {
    #[pyo3(get, set)]
    buy_commission: f64,
    #[pyo3(get, set)]
    sell_commission: f64,
    #[pyo3(get, set)]
    large_volume_threshold: f64,
    #[pyo3(get, set)]
    buy_slippage: f64,
    #[pyo3(get, set)]
    sell_slippage: f64,
    #[pyo3(get, set)]
    min_commission: f64,
}

#[pymethods]
impl PyFeeConfig {
    #[new]
    fn new(
        buy_commission: f64,
        sell_commission: f64,
        large_volume_threshold: f64,
        buy_slippage: f64,
        sell_slippage: f64,
        min_commission: f64,
    ) -> Self {
        Self {
            buy_commission,
            sell_commission,
            large_volume_threshold,
            buy_slippage,
            sell_slippage,
            min_commission,
        }
    }
}

impl From<PyFeeConfig> for FeeConfig {
    fn from(py_config: PyFeeConfig) -> Self {
        FeeConfig {
            buy_commission: py_config.buy_commission,
            sell_commission: py_config.sell_commission,
            slippage: SlippageConfig {
                large_volume_threshold: py_config.large_volume_threshold,
                buy_slippage: py_config.buy_slippage,
                sell_slippage: py_config.sell_slippage,
            },
            min_commission: py_config.min_commission,
        }
    }
}

/// Position configuration
#[pyclass(name = "PositionConfig")]
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
#[pyclass(name = "BacktestEngine")]
struct PyBacktestEngine {
    config: backtest::BacktestConfig,
}

#[pymethods]
impl PyBacktestEngine {
    #[new]
    #[pyo3(signature = (quantiles, weight_method, long_top_n, short_top_n, buy_commission, sell_commission, rebalance_freq=1))]
    fn new(
        _py: Python<'_>,
        quantiles: usize,
        weight_method: &str,
        long_top_n: usize,
        short_top_n: usize,
        buy_commission: f64,
        sell_commission: f64,
        rebalance_freq: usize,
    ) -> PyResult<Self> {
        let wmethod = match weight_method {
            "equal" => WeightMethod::Equal,
            "weighted" => WeightMethod::Weighted,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "weight_method must be 'equal' or 'weighted'",
                ));
            }
        };

        let fee_config = backtest::FeeConfig {
            buy_commission,
            sell_commission,
            ..Default::default()
        };

        let config = backtest::BacktestConfig {
            quantiles,
            weight_method: wmethod,
            long_top_n,
            short_top_n,
            rebalance_freq,
            fee_config,
            position_config: Default::default(),
            limit_up_down_config: Default::default(),
        };

        Ok(Self { config })
    }

    fn run(
        &self,
        factor: Bound<'_, PyArray2<f64>>,
        returns: Bound<'_, PyArray2<f64>>,
        adj_factor: Bound<'_, PyArray2<f64>>,
        close: Bound<'_, PyArray2<f64>>,
        open: Bound<'_, PyArray2<f64>>,
        vwap: Bound<'_, PyArray2<f64>>,
        tradable: Bound<'_, PyArray2<f64>>,
    ) -> PyResult<PyBacktestResult> {
        let factor_array = factor.readonly().as_array().to_owned();
        let returns_array = returns.readonly().as_array().to_owned();
        let adj_factor_array = adj_factor.readonly().as_array().to_owned();
        let close_array = close.readonly().as_array().to_owned();
        let open_array = open.readonly().as_array().to_owned();
        let vwap_array = vwap.readonly().as_array().to_owned();
        let tradable_array = tradable.readonly().as_array().to_owned();

        let engine = BacktestEngine::with_config(self.config.clone());
        let (n_days, n_assets) = factor_array.dim();
        let pm = PriceMatrix {
            dates: vec![],
            symbols: vec![],
            close: close_array,
            open: open_array,
            high: Array2::from_elem((n_days, n_assets), 1.0),
            low: Array2::from_elem((n_days, n_assets), 1.0),
            vwap: vwap_array,
            returns: returns_array,
            tradable: tradable_array,
            adj_factor: adj_factor_array,
        };

        match engine.run(factor_array, &pm) {
            Ok(result) => Ok(PyBacktestResult::from(result)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    /// Multi-factor equal-weight combination backtest.
    ///
    /// Takes a list of factor arrays (all same shape), averages them
    /// element-wise with equal weight, then runs the standard backtest.
    fn run_multi(
        &self,
        factors: Vec<Bound<'_, PyArray2<f64>>>,
        returns: Bound<'_, PyArray2<f64>>,
        adj_factor: Bound<'_, PyArray2<f64>>,
        close: Bound<'_, PyArray2<f64>>,
        open: Bound<'_, PyArray2<f64>>,
        vwap: Bound<'_, PyArray2<f64>>,
        tradable: Bound<'_, PyArray2<f64>>,
    ) -> PyResult<PyBacktestResult> {
        let factor_arrays: Vec<Array2<f64>> = factors
            .iter()
            .map(|f| f.readonly().as_array().to_owned())
            .collect();
        let returns_array = returns.readonly().as_array().to_owned();
        let adj_factor_array = adj_factor.readonly().as_array().to_owned();
        let close_array = close.readonly().as_array().to_owned();
        let open_array = open.readonly().as_array().to_owned();
        let vwap_array = vwap.readonly().as_array().to_owned();
        let tradable_array = tradable.readonly().as_array().to_owned();

        let engine = BacktestEngine::with_config(self.config.clone());
        let (n_days, n_assets) = factor_arrays[0].dim();
        let pm = PriceMatrix {
            dates: vec![],
            symbols: vec![],
            close: close_array,
            open: open_array,
            high: Array2::from_elem((n_days, n_assets), 1.0),
            low: Array2::from_elem((n_days, n_assets), 1.0),
            vwap: vwap_array,
            returns: returns_array,
            tradable: tradable_array,
            adj_factor: adj_factor_array,
        };

        match engine.run_multi(&factor_arrays, &pm) {
            Ok(result) => Ok(PyBacktestResult::from(result)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    /// Run backtest with a PriceMatrix (single-factor).
    ///
    /// adj_factor comes from the database via query_price_matrix().
    fn run_with_prices(
        &self,
        factor: Bound<'_, PyArray2<f64>>,
        prices: &PyPriceMatrix,
    ) -> PyResult<PyBacktestResult> {
        let factor_array = factor.readonly().as_array().to_owned();
        let engine = BacktestEngine::with_config(self.config.clone());
        match engine.run(factor_array, &prices.inner) {
            Ok(result) => Ok(PyBacktestResult::from(result)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    /// Run multi-factor equal-weight backtest with a PriceMatrix.
    fn run_multi_with_prices(
        &self,
        factors: Vec<Bound<'_, PyArray2<f64>>>,
        prices: &PyPriceMatrix,
    ) -> PyResult<PyBacktestResult> {
        let factor_arrays: Vec<Array2<f64>> = factors
            .iter()
            .map(|f| f.readonly().as_array().to_owned())
            .collect();
        let engine = BacktestEngine::with_config(self.config.clone());
        match engine.run_multi(&factor_arrays, &prices.inner) {
            Ok(result) => Ok(PyBacktestResult::from(result)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }
}

/// Python-exposed backtest result
#[pyclass(name = "BacktestResult")]
struct PyBacktestResult {
    inner: BacktestResult,
}

impl From<BacktestResult> for PyBacktestResult {
    fn from(inner: BacktestResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyBacktestResult {
    #[getter]
    fn dates(&self) -> Vec<i64> {
        self.inner.dates.clone()
    }
    #[getter]
    fn group_returns(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.inner.group_returns.clone().into_pyarray(py).into()
    }
    #[getter]
    fn group_cum_returns(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.inner.group_cum_returns.clone().into_pyarray(py).into()
    }
    #[getter]
    fn long_short_returns(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.inner.long_short_returns.clone().into_pyarray(py).into()
    }
    #[getter]
    fn long_short_cum_return(&self) -> f64 {
        self.inner.long_short_cum_return
    }
    #[getter]
    fn long_short_cum_returns(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.inner.long_short_cum_returns.clone().into_pyarray(py).into()
    }
    #[getter]
    fn long_cum_returns(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.inner.long_cum_returns.clone().into_pyarray(py).into()
    }
    #[getter]
    fn short_cum_returns(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.inner.short_cum_returns.clone().into_pyarray(py).into()
    }
    #[getter]
    fn ic_series(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.inner.ic_series.clone().into_pyarray(py).into()
    }
    #[getter]
    fn ic_mean(&self) -> f64 { self.inner.ic_mean }
    #[getter]
    fn ic_ir(&self) -> f64 { self.inner.ic_ir }
    #[getter]
    fn long_ic_mean(&self) -> f64 { self.inner.long_ic_mean }
    #[getter]
    fn long_ic_ir(&self) -> f64 { self.inner.long_ic_ir }
    #[getter]
    fn short_ic_mean(&self) -> f64 { self.inner.short_ic_mean }
    #[getter]
    fn short_ic_ir(&self) -> f64 { self.inner.short_ic_ir }
    #[getter]
    fn long_short_ic_mean(&self) -> f64 { self.inner.long_short_ic_mean }
    #[getter]
    fn long_short_ic_ir(&self) -> f64 { self.inner.long_short_ic_ir }
    #[getter]
    fn total_return(&self) -> f64 { self.inner.total_return }
    #[getter]
    fn annualized_return(&self) -> f64 { self.inner.annualized_return }
    #[getter]
    fn sharpe_ratio(&self) -> f64 { self.inner.sharpe_ratio }
    #[getter]
    fn max_drawdown(&self) -> f64 { self.inner.max_drawdown }
    #[getter]
    fn turnover(&self) -> f64 { self.inner.turnover }
    #[getter]
    fn weight_turnover(&self) -> f64 { self.inner.weight_turnover }
    #[getter]
    fn win_rate(&self) -> f64 { self.inner.win_rate }
    #[getter]
    fn calmar_ratio(&self) -> f64 { self.inner.calmar_ratio }
    #[getter]
    fn long_returns(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.inner.long_returns.clone().into_pyarray(py).into()
    }
    #[getter]
    fn short_returns(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.inner.short_returns.clone().into_pyarray(py).into()
    }
    #[getter]
    fn passive_returns(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.inner.passive_returns.clone().into_pyarray(py).into()
    }
    #[getter]
    fn passive_cum_returns(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.inner.passive_cum_returns.clone().into_pyarray(py).into()
    }

    /// Write group NAV curves to CSV (date,nv,group).
    fn to_csv(&self, path: &str) -> PyResult<()> {
        self.inner.to_csv(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("CSV write: {}", e))
        })
    }
}

/// Standalone quantile backtest function (Python interface)
#[pyfunction]
fn quantile_backtest(
    _py: Python<'_>,
    factor: Bound<'_, PyArray2<f64>>,
    returns: Bound<'_, PyArray2<f64>>,
    quantiles: usize,
    weight_method: &str,
    long_top_n: usize,
    short_top_n: usize,
    buy_commission: f64,
    sell_commission: f64,
    adj_factor: Bound<'_, PyArray2<f64>>,
    close: Bound<'_, PyArray2<f64>>,
    open: Bound<'_, PyArray2<f64>>,
    vwap: Bound<'_, PyArray2<f64>>,
    tradable: Bound<'_, PyArray2<f64>>,
) -> PyResult<PyBacktestResult> {
    let wmethod = match weight_method {
        "equal" => WeightMethod::Equal,
        "weighted" => WeightMethod::Weighted,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "weight_method must be 'equal' or 'weighted'",
            ));
        }
    };

    let fee_config = backtest::FeeConfig {
        buy_commission,
        sell_commission,
        ..Default::default()
    };

    let config = backtest::BacktestConfig {
        quantiles,
        weight_method: wmethod,
        long_top_n,
        short_top_n,
        rebalance_freq: 1,
        fee_config,
        position_config: Default::default(),
        limit_up_down_config: Default::default(),
    };

    let engine = BacktestEngine::with_config(config);

    let factor_array = factor.readonly().as_array().to_owned();
    let returns_array = returns.readonly().as_array().to_owned();
    let adj_factor_array = adj_factor.readonly().as_array().to_owned();
    let close_array = close.readonly().as_array().to_owned();
    let open_array = open.readonly().as_array().to_owned();
    let vwap_array = vwap.readonly().as_array().to_owned();
    let tradable_array = tradable.readonly().as_array().to_owned();

    let (n_days, n_assets) = factor_array.dim();
    let pm = PriceMatrix {
        dates: vec![],
        symbols: vec![],
        close: close_array,
        open: open_array,
        high: Array2::from_elem((n_days, n_assets), 1.0),
        low: Array2::from_elem((n_days, n_assets), 1.0),
        vwap: vwap_array,
        returns: returns_array,
        tradable: tradable_array,
        adj_factor: adj_factor_array,
    };

    match engine.run(factor_array, &pm) {
        Ok(result) => Ok(PyBacktestResult::from(result)),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
    }
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
// Genetic Programming (GP) Module Python Bindings
// ============================================================================

use crate::gp::{
    BacktestFitnessEvaluator, DataSplitConfig, FactorPool, Function, GPConfig, PoolEntry,
    RealBacktestFitnessEvaluator, Terminal, check_redundancy, expr_structural_similarity, run_gp,
    to_parseable_string,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
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
    #[pyo3(signature = (population_size=100, max_generations=50, tournament_size=7, crossover_prob=0.8, mutation_prob=0.2, max_depth=6, allow_ephemeral=true, parent_diversity_penalty=0.1, use_diverse_init=false, smart_mutation_ratio=0.3, use_frequencies=false))]
    fn new(
        population_size: usize,
        max_generations: usize,
        tournament_size: usize,
        crossover_prob: f64,
        mutation_prob: f64,
        max_depth: usize,
        allow_ephemeral: bool,
        parent_diversity_penalty: f64,
        use_diverse_init: bool,
        smart_mutation_ratio: f64,
        use_frequencies: bool,
    ) -> Self {
        // Create default terminals (will be updated later)
        let mut terminals = vec![Terminal::Constant(1.0), Terminal::Constant(2.0)];

        if allow_ephemeral {
            terminals.insert(0, Terminal::Ephemeral);
        }

        let functions = vec![
            Function::add(),
            Function::sub(),
            Function::mul(),
            Function::div(),
            Function::power(),
            Function::sqrt(),
            Function::abs(),
            Function::neg(),
            Function::log(),
            Function::sign(),
            Function::exp(),
            Function::rank(),
            Function::cs_scale(),
            Function::ts_mean(),
            Function::ts_std(),
            Function::ts_max(),
            Function::ts_min(),
            Function::ts_sum(),
            Function::delay(),
            Function::ts_delta(),
            Function::ts_rank(),
            Function::decay_linear(),
            Function::correlation(),
            Function::ts_covariance(),
        ];

        let config = GPConfig {
            population_size,
            max_generations,
            tournament_size,
            crossover_prob,
            mutation_prob,
            max_depth,
            parent_diversity_penalty,
            use_diverse_init,
            smart_mutation_ratio,
            use_frequencies,
        };

        let rng = StdRng::from_entropy();

        PyGpEngine {
            config,
            terminals,
            functions,
            rng,
        }
    }

    /// When use_frequencies is enabled, bare names like "close" are expanded to
    /// all available frequency-prefixed variants (e.g. "1d:close", "5m:close", "1m:close").
    /// Names already containing a freq prefix (e.g. "1d:close") are kept as-is.
    #[pyo3(signature = (columns, allow_ephemeral = None))]
    fn set_columns(
        &mut self,
        _py: Python<'_>,
        columns: Bound<'_, PyList>,
        allow_ephemeral: Option<bool>,
    ) {
        let allow_ephemeral = allow_ephemeral.unwrap_or(true);
        let mut new_terminals = vec![Terminal::Constant(1.0), Terminal::Constant(2.0)];
        if allow_ephemeral {
            new_terminals.insert(0, Terminal::Ephemeral);
        }

        for item in columns.iter() {
            let col: String = item.extract().unwrap_or_default();
            if self.config.use_frequencies && !col.contains(':') {
                let mut expanded = false;
                for freq in crate::data::frequency::all_frequencies() {
                    if crate::data::frequency::field_at_frequency(&col, freq) {
                        new_terminals.push(Terminal::Variable(format!(
                            "{}:{}",
                            freq.as_str(),
                            col
                        )));
                        expanded = true;
                    }
                }
                if !expanded {
                    new_terminals.push(Terminal::Variable(col));
                }
            } else {
                new_terminals.push(Terminal::Variable(col));
            }
        }

        self.terminals = new_terminals;
    }

    /// Run genetic programming with train/validation/test split.
    ///
    /// Evolution uses training data only for fitness selection.
    /// Returns (expr, fitness, ic, ir, turnover, complexity, train, val, test)
    /// where train/val/test are [ic_mean, ic_ir, sharpe_ratio, max_drawdown, annualized_return].
    #[pyo3(signature = (data, returns, num_factors=3, train_ratio=0.6, validation_ratio=0.2,
                        weight_ic=0.4, weight_ir=0.3, weight_turnover=0.15, weight_complexity=0.15))]
    fn mine_factors(
        &mut self,
        _py: Python<'_>,
        data: Bound<'_, PyDict>,
        returns: Bound<'_, PyArray2<f64>>,
        num_factors: usize,
        train_ratio: f64,
        validation_ratio: f64,
        weight_ic: f64,
        weight_ir: f64,
        weight_turnover: f64,
        weight_complexity: f64,
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

        let mut data_arrays = HashMap::new();
        for (key, value) in data.iter() {
            let col_name: String = key.extract()?;
            if let Ok(arr) = value.extract::<Bound<'_, PyArray2<f64>>>() {
                data_arrays.insert(col_name, arr.readonly().as_array().to_owned());
            } else {
                return Err(PyValueError::new_err(format!(
                    "Column '{}' must be a 2D numpy array",
                    col_name
                )));
            }
        }

        let returns_array = returns.readonly().as_array().to_owned();
        let (n_days, n_assets) = returns_array.dim();

        // Build PriceMatrix from data dict + returns
        let close = data_arrays.remove("close").expect("close data required");
        let open = data_arrays.remove("open").expect("open data required");
        let high = data_arrays.remove("high").unwrap_or_else(|| close.clone());
        let low = data_arrays.remove("low").unwrap_or_else(|| close.clone());
        let vwap = data_arrays.remove("vwap").expect("vwap data required");
        let tradable = data_arrays
            .remove("tradable")
            .expect("tradable data required");

        use crate::data::layer::PriceMatrix;
        let prices = PriceMatrix {
            dates: (0..n_days as i64).collect(),
            symbols: (0..n_assets).map(|i| i.to_string()).collect(),
            close,
            open,
            high,
            low,
            vwap,
            returns: returns_array,
            tradable,
            adj_factor: Array2::from_elem((n_days, n_assets), 1.0),
        };

        let split_config = DataSplitConfig {
            train_ratio,
            validation_ratio,
            test_ratio: Some(1.0 - train_ratio - validation_ratio),
        };

        let mut evaluator =
            RealBacktestFitnessEvaluator::with_split(data_arrays, Arc::new(prices), split_config);

        let weights = HashMap::from([
            ("ic".to_string(), weight_ic),
            ("ir".to_string(), weight_ir),
            ("turnover".to_string(), weight_turnover),
            ("complexity".to_string(), weight_complexity),
        ]);
        evaluator.set_weights(weights);

        let mut results = Vec::new();
        for _ in 0..num_factors {
            let (best_expr, best_fitness) = run_gp(
                &self.config,
                &evaluator,
                self.terminals.clone(),
                self.functions.clone(),
                &mut self.rng,
            );

            let expr_str = to_parseable_string(&best_expr);
            let ic = evaluator.get_last_ic();
            let ir = evaluator.get_last_ir();
            let turnover = evaluator.get_last_turnover();
            let complexity = evaluator.get_last_complexity();

            let (train_m, val_m, test_m) = match evaluator.get_last_split_result() {
                Some(s) => (
                    vec![
                        s.train.ic_mean,
                        s.train.ic_ir,
                        s.train.sharpe_ratio,
                        s.train.max_drawdown,
                        s.train.annualized_return,
                    ],
                    vec![
                        s.validation.ic_mean,
                        s.validation.ic_ir,
                        s.validation.sharpe_ratio,
                        s.validation.max_drawdown,
                        s.validation.annualized_return,
                    ],
                    vec![
                        s.test.ic_mean,
                        s.test.ic_ir,
                        s.test.sharpe_ratio,
                        s.test.max_drawdown,
                        s.test.annualized_return,
                    ],
                ),
                None => (vec![0.0; 5], vec![0.0; 5], vec![0.0; 5]),
            };

            results.push((
                expr_str,
                best_fitness,
                ic,
                ir,
                turnover,
                complexity,
                train_m,
                val_m,
                test_m,
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
            "Best expression: {}, Fitness: {:.6}",
            best_expr, fitness
        ))
    }
}

// ============================================================================
// Factor Pool Python Bindings
// ============================================================================

/// Python-exposed factor pool for maintaining a diverse alpha zoo.
#[pyclass(name = "FactorPool")]
pub struct PyFactorPool {
    inner: FactorPool,
}

#[pymethods]
impl PyFactorPool {
    #[new]
    fn new(max_size: usize) -> Self {
        Self {
            inner: FactorPool::new(max_size),
        }
    }

    /// Number of factors currently in the pool.
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the pool is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Attempt to admit a factor expression string into the pool.
    /// Returns (status, similarity) where status is one of:
    /// "added", "rejected_duplicate", "flagged", "rejected_below_min"
    fn try_admit(
        &mut self,
        expr_str: &str,
        ic: f64,
        rank_ic: f64,
    ) -> PyResult<(String, Option<f64>)> {
        use crate::expr::registry::parser::parse_expression;

        let expr = parse_expression(expr_str)
            .map_err(|e| PyValueError::new_err(format!("Parse error: {}", e)))?;

        let pool_exprs: Vec<crate::expr::Expr> = self
            .inner
            .entries()
            .iter()
            .filter_map(|e| parse_expression(&e.expression).ok())
            .collect();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        use crate::gp::AdmissionResult;
        let result = self
            .inner
            .try_admit_parsed(&expr, ic, rank_ic, &pool_exprs, now);

        match result {
            AdmissionResult::Added => Ok(("added".to_string(), None)),
            AdmissionResult::RejectedDuplicate(sim) => {
                Ok(("rejected_duplicate".to_string(), Some(sim)))
            }
            AdmissionResult::Flagged(sim) => Ok(("flagged".to_string(), Some(sim))),
            AdmissionResult::RejectedBelowMinimum => Ok(("rejected_below_min".to_string(), None)),
        }
    }

    /// Get entry count.
    fn entry_count(&self) -> usize {
        self.inner.len()
    }

    /// Get a specific entry by index as tuple.
    fn get_entry(&self, idx: usize) -> PyResult<(String, f64, f64, u64, u64, u32)> {
        let entries = self.inner.entries();
        if idx >= entries.len() {
            return Err(PyValueError::new_err("index out of bounds"));
        }
        let e = &entries[idx];
        Ok((
            e.expression.clone(),
            e.ic,
            e.rank_ic,
            e.added_at,
            e.last_check_at,
            e.survival_rounds,
        ))
    }

    /// Bump survival rounds for all entries.
    fn bump_survival(&mut self) {
        self.inner.bump_survival();
    }

    /// Prune pool to max capacity.
    fn prune(&mut self) {
        self.inner.prune();
    }
}

/// Compute structural similarity between two expression strings.
#[pyfunction]
fn expr_similarity(a: &str, b: &str) -> PyResult<f64> {
    use crate::expr::registry::parser::parse_expression;
    let expr_a = parse_expression(a)
        .map_err(|e| PyValueError::new_err(format!("Parse error in a: {}", e)))?;
    let expr_b = parse_expression(b)
        .map_err(|e| PyValueError::new_err(format!("Parse error in b: {}", e)))?;
    Ok(expr_structural_similarity(&expr_a, &expr_b))
}

// ============================================================================
// Persistence Module Python Bindings
// ============================================================================

use crate::gp::metalearning::{GPRecommendations, MetaLearningAnalyzer};
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

use crate::expr::registry::{ComputeConfig, FactorRegistry};

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
        // Convert Python dict to HashMap<String, Array1<f64>>
        let mut arr_data = std::collections::HashMap::new();
        for (key, value) in data.iter() {
            let key_str: String = key.extract()?;
            let values: Vec<f64> = value.extract()?;
            arr_data.insert(key_str, Array1::from_vec(values));
        }

        let results = self
            .inner
            .compute(&[name], &arr_data, false, false)
            .map_err(|e| PyValueError::new_err(e))?;

        results
            .into_values()
            .next()
            .map(|r| PyFactorResult {
                name: r.name,
                values: r.values,
                n_rows: r.n_rows,
                n_cols: r.n_cols,
                compute_time_ms: r.compute_time_ms,
            })
            .ok_or_else(|| PyValueError::new_err("No result returned"))
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
        // Convert Python dict to HashMap<String, Array1<f64>>
        let mut arr_data = std::collections::HashMap::new();
        for (key, value) in data.iter() {
            let key_str: String = key.extract()?;
            let values: Vec<f64> = value.extract()?;
            arr_data.insert(key_str, Array1::from_vec(values));
        }

        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

        let results = self
            .inner
            .compute(&name_refs, &arr_data, parallel, false)
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

    /// Compute all registered factors via the cross-sectional pipeline.
    ///
    /// Takes a &mut DataLayer, queries 5m data automatically, computes factors,
    /// applies winsor→zscore→cap_neu→qcut per date, and returns FactorSlice
    /// for each registered factor.
    fn compute_cs_pipeline(
        &self,
        py: Python<'_>,
        data_layer: &mut PyDataLayer,
    ) -> PyResult<Py<PyDict>> {
        let results = self
            .inner
            .compute_cs_pipeline(&mut data_layer.inner)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let result_dict = PyDict::new(py);
        for (name, cs) in results {
            result_dict.set_item(name, PyFactorSlice { inner: cs })?;
        }
        Ok(result_dict.into())
    }

    /// Compute factors on 1d data and return 2D matrices aligned with PriceMatrix.
    ///
    /// Queries 1d data via DataLayer, computes all registered factors, reshapes
    /// to (n_dates × n_symbols) 2D arrays using the PriceMatrix alignment.
    /// Returns (factor_matrices_dict, price_matrix).
    fn compute_factor_matrices_1d(
        &self,
        py: Python<'_>,
        data_layer: &mut PyDataLayer,
    ) -> PyResult<(Py<PyDict>, PyPriceMatrix)> {
        // 1. Use FactorRegistry's built-in required_columns()
        let col_set = self.inner.required_columns();
        let names = self.inner.list();

        // 2. Build query fields with 1d: prefix
        let mut query_fields = vec!["1d:trading_date".to_string(), "1d:symbol".to_string()];
        for c in col_set {
            query_fields.push(format!("1d:{}", c));
        }

        // 3. Query data
        let raw_data = data_layer
            .inner
            .query(query_fields)
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;

        // 4. Strip 1d: prefix from keys for expression evaluation
        let mut data: std::collections::HashMap<String, ndarray::Array1<f64>> =
            std::collections::HashMap::new();
        for (key, arr) in &raw_data {
            if let Some(stripped) = key.strip_prefix("1d:") {
                data.insert(stripped.to_string(), arr.clone());
            } else {
                data.insert(key.clone(), arr.clone());
            }
        }

        // 5. Get date and symbol arrays for later 2D mapping
        let dates_arr = raw_data
            .get("1d:trading_date")
            .ok_or_else(|| PyRuntimeError::new_err("trading_date missing"))?;
        let syms_arr = raw_data
            .get("1d:symbol")
            .ok_or_else(|| PyRuntimeError::new_err("symbol missing"))?;

        // 6. Compute factors (non-compact for dense 1d data)
        let factor_names: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let results = self
            .inner
            .compute(&factor_names, &data, false, false)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        // 7. Build PriceMatrix for alignment
        let pm = data_layer
            .inner
            .query_price_matrix()
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;

        let n_dates = pm.dates.len();
        let n_symbols = pm.symbols.len();

        // Build symbol → index map
        let symbol_list = data_layer.inner.get_symbols_5m();
        let mut sym_to_idx: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for (i, s) in symbol_list.iter().enumerate() {
            if let Some(pos) = pm.symbols.iter().position(|ps| ps == s) {
                sym_to_idx.insert(i, pos);
            }
        }

        // Build date → index map
        let mut date_to_idx: std::collections::HashMap<i64, usize> =
            std::collections::HashMap::new();
        for (i, &d) in pm.dates.iter().enumerate() {
            date_to_idx.insert(d, i);
        }

        // 8. Reshape each factor result to 2D
        let result_dict = PyDict::new(py);
        for (name, fr) in &results {
            let mut mat = Array2::<f64>::from_elem((n_dates, n_symbols), f64::NAN);

            for i in 0..fr.values.len() {
                if i >= dates_arr.len() || i >= syms_arr.len() {
                    break;
                }
                let d = dates_arr[i] as i64;
                let s = syms_arr[i] as usize;

                if d < 19000101 {
                    continue;
                }

                if let (Some(&di), Some(&si)) = (date_to_idx.get(&d), sym_to_idx.get(&s)) {
                    mat[[di, si]] = fr.values[i];
                }
            }

            result_dict.set_item(name.clone(), mat.into_pyarray(py))?;
        }

        Ok((result_dict.into(), PyPriceMatrix { inner: pm }))
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

// ============================================================================
// ClickHouseSource Python Bindings
// ============================================================================

#[pyclass(name = "ClickHouseSource", skip_from_py_object)]
#[derive(Clone)]
pub struct PyClickHouseSource {
    inner: ClickHouseSource,
}

#[pymethods]
impl PyClickHouseSource {
    #[staticmethod]
    fn from_env() -> Self {
        Self {
            inner: ClickHouseSource::from_env(),
        }
    }

    #[new]
    #[pyo3(signature = (host, port, database, username, password=None))]
    fn new(
        host: &str,
        port: u16,
        database: &str,
        username: &str,
        password: Option<String>,
    ) -> Self {
        Self {
            inner: ClickHouseSource::with_units(host, port, database, username, 100, 1000),
        }
    }

    #[getter]
    fn host(&self) -> &str {
        self.inner.host()
    }

    #[getter]
    fn port(&self) -> u16 {
        self.inner.port()
    }

    #[getter]
    fn database(&self) -> &str {
        self.inner.database()
    }

    #[getter]
    fn username(&self) -> &str {
        self.inner.username()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClickHouseSource(host={}:{}, database={})",
            self.inner.host(),
            self.inner.port(),
            self.inner.database()
        )
    }
}

// ============================================================================
// DataLayer Python Bindings
// ============================================================================

#[pyclass(name = "DataLayer")]
pub struct PyDataLayer {
    inner: DataLayer,
}

#[pymethods]
impl PyDataLayer {
    #[new]
    fn new(source: &PyClickHouseSource) -> Self {
        Self {
            inner: DataLayer::new(source.inner.clone()),
        }
    }

    fn set_pre_filter(&mut self, filter: &str) {
        self.inner.set_pre_filter(filter);
    }

    fn clear_cache(&mut self) {
        self.inner.clear_cache();
    }

    #[getter]
    fn symbols_5m(&self) -> Vec<String> {
        self.inner.get_symbols_5m().to_vec()
    }

    /// Query 1d price data for backtest. Returns PriceMatrix with all OHLCV fields.
    fn query_price_matrix(&mut self) -> PyResult<PyPriceMatrix> {
        self.inner
            .query_price_matrix()
            .map(|pm| PyPriceMatrix { inner: pm })
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
    }

    fn __repr__(&self) -> String {
        format!("DataLayer(pre_filter={})", "...")
    }
}

// ============================================================================
// PriceMatrix Python Bindings
// ============================================================================

#[pyclass(name = "PriceMatrix", skip_from_py_object)]
#[derive(Clone)]
pub struct PyPriceMatrix {
    inner: PriceMatrix,
}

#[pymethods]
impl PyPriceMatrix {
    #[getter]
    fn dates(&self) -> Vec<i64> {
        self.inner.dates.clone()
    }

    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.inner.symbols.clone()
    }

    #[getter]
    fn close<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        self.inner.close.clone().into_pyarray(py).into()
    }

    #[getter]
    fn open<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        self.inner.open.clone().into_pyarray(py).into()
    }

    #[getter]
    fn high<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        self.inner.high.clone().into_pyarray(py).into()
    }

    #[getter]
    fn low<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        self.inner.low.clone().into_pyarray(py).into()
    }

    #[getter]
    fn vwap<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        self.inner.vwap.clone().into_pyarray(py).into()
    }

    #[getter]
    fn returns<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        self.inner.returns.clone().into_pyarray(py).into()
    }

    #[getter]
    fn tradable<'py>(&self, py: Python<'py>) -> Py<PyArray2<f64>> {
        self.inner.tradable.clone().into_pyarray(py).into()
    }

    #[getter]
    fn n_dates(&self) -> usize {
        self.inner.dates.len()
    }

    #[getter]
    fn n_symbols(&self) -> usize {
        self.inner.symbols.len()
    }

    /// Build a factor matrix from FactorSlices, aligned to this PriceMatrix.
    fn build_factor_matrix(
        &self,
        py: Python<'_>,
        slices: Vec<Py<PyFactorSlice>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let rust_slices: Vec<FactorSlice> =
            slices.iter().map(|s| s.borrow(py).inner.clone()).collect();
        let mat = self.inner.build_factor_matrix(&rust_slices);
        Ok(mat.into_pyarray(py).into())
    }

    fn __repr__(&self) -> String {
        format!(
            "PriceMatrix(dates={}, symbols={})",
            self.inner.dates.len(),
            self.inner.symbols.len()
        )
    }
}

// ============================================================================
// FactorSlice Python Bindings
// ============================================================================

#[pyclass(name = "FactorSlice", from_py_object)]
#[derive(Clone)]
pub struct PyFactorSlice {
    inner: FactorSlice,
}

#[pymethods]
impl PyFactorSlice {
    #[getter]
    fn factor_name(&self) -> String {
        self.inner.factor_name.clone()
    }

    #[getter]
    fn groups(&self) -> Vec<(i64, i64)> {
        (*self.inner.groups).clone()
    }

    #[getter]
    fn symbols(&self) -> Vec<String> {
        (*self.inner.symbols).clone()
    }

    #[getter]
    fn cap_neued(&self) -> Vec<f64> {
        self.inner.cap_neued.clone()
    }

    #[getter]
    fn qcut(&self) -> Vec<Option<i32>> {
        self.inner.qcut.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "FactorSlice({}, n={})",
            self.inner.factor_name,
            self.inner.groups.len()
        )
    }
}

// ============================================================================
// Factor → Position Matrix Abstraction
// ============================================================================

/// Methods for combining multiple factor matrices into one.
#[pyclass(name = "FactorCombiner")]
pub struct PyFactorCombiner;

/// Methods for building position/weight matrices from factor values.
#[pyclass(name = "PositionBuilder")]
pub struct PyPositionBuilder;

#[pymethods]
impl PyFactorCombiner {
    #[staticmethod]
    /// Equal-weight average of multiple factor matrices.
    /// NaN values are excluded from the average per cell.
    fn equal_weight(
        py: Python<'_>,
        factors: Vec<Bound<'_, PyArray2<f64>>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        if factors.is_empty() {
            return Err(PyValueError::new_err("Empty factor list"));
        }
        let arrays: Vec<Array2<f64>> = factors
            .iter()
            .map(|f| f.readonly().as_array().to_owned())
            .collect();
        let shape = arrays[0].dim();
        for a in arrays.iter().skip(1) {
            if a.dim() != shape {
                return Err(PyValueError::new_err(format!(
                    "Shape mismatch: expected {:?}, got {:?}",
                    shape,
                    a.dim()
                )));
            }
        }
        let n = arrays.len() as f64;
        let mut combined = Array2::<f64>::zeros(shape);
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                let mut sum = 0.0f64;
                let mut count = 0u32;
                for a in &arrays {
                    let v = a[[i, j]];
                    if v.is_finite() {
                        sum += v;
                        count += 1;
                    }
                }
                combined[[i, j]] = if count > 0 {
                    sum / count as f64
                } else {
                    f64::NAN
                };
            }
        }
        Ok(combined.into_pyarray(py).into())
    }

    #[staticmethod]
    /// Rank-average combination: convert each factor to CS rank, then average.
    fn rank_average(
        py: Python<'_>,
        factors: Vec<Bound<'_, PyArray2<f64>>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        if factors.is_empty() {
            return Err(PyValueError::new_err("Empty factor list"));
        }
        let arrays: Vec<Array2<f64>> = factors
            .iter()
            .map(|f| f.readonly().as_array().to_owned())
            .collect();
        let shape = arrays[0].dim();
        let n_factors = arrays.len() as f64;
        let mut combined = Array2::<f64>::zeros(shape);

        for i in 0..shape.0 {
            for factor_idx in 0..arrays.len() {
                let row = arrays[factor_idx].row(i);
                let mut valid: Vec<(usize, f64)> = row
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_finite())
                    .map(|(j, &v)| (j, v))
                    .collect();
                if valid.len() < 2 {
                    continue;
                }
                valid.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let n = valid.len() as f64;
                // Average-rank tie-breaking
                let mut k = 0;
                while k < valid.len() {
                    let mut end = k + 1;
                    while end < valid.len() && valid[end].1 == valid[k].1 {
                        end += 1;
                    }
                    let avg_rank = (k + end + 1) as f64 / 2.0;
                    let norm = (avg_rank - 1.0) / (n - 1.0);
                    for m in k..end {
                        combined[[i, valid[m].0]] += (norm - 0.5) / n_factors;
                    }
                    k = end;
                }
            }
        }
        Ok(combined.into_pyarray(py).into())
    }

    #[staticmethod]
    /// Signal-weighted combination: weight each factor by the absolute
    /// magnitude of its z-scored signal, then sum.
    fn signal_weighted(
        py: Python<'_>,
        factors: Vec<Bound<'_, PyArray2<f64>>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        if factors.is_empty() {
            return Err(PyValueError::new_err("Empty factor list"));
        }
        let arrays: Vec<Array2<f64>> = factors
            .iter()
            .map(|f| f.readonly().as_array().to_owned())
            .collect();
        let shape = arrays[0].dim();
        let mut combined = Array2::<f64>::zeros(shape);

        for i in 0..shape.0 {
            let mut total_weight = vec![0.0f64; shape.1];
            let mut weighted_sum = vec![0.0f64; shape.1];
            for arr in &arrays {
                let row = arr.row(i);
                let mut valid: Vec<(usize, f64)> = row
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_finite())
                    .map(|(j, &v)| (j, v))
                    .collect();
                if valid.is_empty() {
                    continue;
                }
                let mean = valid.iter().map(|(_, v)| *v).sum::<f64>() / valid.len() as f64;
                let std = (valid.iter().map(|(_, v)| (*v - mean).powi(2)).sum::<f64>()
                    / valid.len() as f64)
                    .sqrt();
                if std < 1e-12 {
                    continue;
                }
                for (j, v) in &valid {
                    let z = (*v - mean) / std;
                    let w = z.abs();
                    total_weight[*j] += w;
                    weighted_sum[*j] += z * w;
                }
            }
            for j in 0..shape.1 {
                if total_weight[j] > 0.0 {
                    combined[[i, j]] = weighted_sum[j] / total_weight[j];
                } else {
                    combined[[i, j]] = f64::NAN;
                }
            }
        }
        Ok(combined.into_pyarray(py).into())
    }
}

#[pymethods]
impl PyPositionBuilder {
    #[staticmethod]
    /// Convert a factor matrix to a long-short position matrix.
    ///
    /// Positions are 1 for top N quantiles, -1 for bottom N quantiles,
    /// 0 otherwise. All positions within a group are equal-weighted.
    fn from_factor(
        py: Python<'_>,
        factor: Bound<'_, PyArray2<f64>>,
        quantiles: usize,
        long_top_n: usize,
        short_top_n: usize,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let factor_arr = factor.readonly().as_array().to_owned();
        let (n_dates, n_assets) = factor_arr.dim();
        let mut positions = Array2::<f64>::zeros((n_dates, n_assets));

        for d in 0..n_dates {
            let row = factor_arr.row(d);
            let mut valid: Vec<(usize, f64)> = row
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(j, &v)| (j, v))
                .collect();

            if valid.len() < quantiles {
                continue;
            }

            valid.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let n = valid.len() as f64;
            let bins = quantiles as f64;

            // Average-rank tie-breaking
            let mut k = 0;
            while k < valid.len() {
                let mut end = k + 1;
                while end < valid.len() && valid[end].1 == valid[k].1 {
                    end += 1;
                }
                let avg_rank = (k + end + 1) as f64 / 2.0;
                let q = ((avg_rank - 1.0) * bins / n).floor() as usize;
                let q = q.min(quantiles - 1);
                for m in k..end {
                    let j = valid[m].0;
                    if q >= quantiles - long_top_n {
                        positions[[d, j]] = 1.0;
                    } else if q < short_top_n {
                        positions[[d, j]] = -1.0;
                    }
                }
                k = end;
            }
        }

        Ok(positions.into_pyarray(py).into())
    }

    fn __repr__(&self) -> String {
        "PositionBuilder()".to_string()
    }
}

impl PyFactorCombiner {
    fn __repr__(&self) -> String {
        "FactorCombiner()".to_string()
    }
}

// ── AlfarsLab ──────────────────────────────────────────────────────────

use crate::data::pool::{CachePolicy, DataPoolConfig};
use crate::expr::registry::config::FactorPanel;
use crate::lab::AlfarsLab;

// ── DataPoolConfig (Python-exposed config) ────────────────────────────

/// Cache policy for year-level DataLayers (5m data).
///
/// Controls memory/time tradeoff:
/// - DropAll: lowest memory, highest ClickHouse load
/// - KeepMostRecent: ~2 GB overhead, good for sequential years
/// - KeepAll: ~2 GB/year, zero re-queries
/// - KeepN(n): ~2 GB × n, precise tuning
#[pyclass(name = "CachePolicy")]
#[derive(Clone)]
struct PyCachePolicy {
    inner: CachePolicy,
}

#[pymethods]
impl PyCachePolicy {
    #[staticmethod]
    fn drop_all() -> Self {
        Self {
            inner: CachePolicy::DropAll,
        }
    }

    #[staticmethod]
    fn keep_most_recent() -> Self {
        Self {
            inner: CachePolicy::KeepMostRecent,
        }
    }

    #[staticmethod]
    fn keep_all() -> Self {
        Self {
            inner: CachePolicy::KeepAll,
        }
    }

    #[staticmethod]
    fn keep_n(n: usize) -> Self {
        Self {
            inner: CachePolicy::KeepN(n),
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Configuration for the data pipeline.
///
/// All hyper-parameters are documented with their memory implications.
///
/// ```python
/// config = DataPoolConfig(
///     cache_policy=CachePolicy.keep_all(),
///     calc_parallel_years=5,
///     memory_budget_bytes=24_000_000_000,
///     backtest_batch_size=20,
/// )
/// lab = AlfarsLab.from_env_with_config(config)
/// ```
#[pyclass(name = "DataPoolConfig")]
#[derive(Clone)]
struct PyDataPoolConfig {
    inner: DataPoolConfig,
}

#[pymethods]
impl PyDataPoolConfig {
    #[new]
    #[pyo3(signature = (cache_policy=None, calc_parallel_years=5, memory_budget_bytes=0, backtest_batch_size=5))]
    fn new(
        cache_policy: Option<PyCachePolicy>,
        calc_parallel_years: usize,
        memory_budget_bytes: usize,
        backtest_batch_size: usize,
    ) -> Self {
        Self {
            inner: DataPoolConfig {
                cache_policy: cache_policy.map(|p| p.inner).unwrap_or_default(),
                calc_parallel_years,
                memory_budget_bytes,
                backtest_batch_size,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Opaque FactorPanel handle returned by AlfarsLab.calc().
/// Pass to AlfarsLab.run() for backtesting.
#[pyclass(name = "FactorPanel")]
struct PyFactorPanel {
    inner: FactorPanel,
}

/// Unified factor research entry point.
///
/// ```python
/// lab = AlfarsLab()
/// lab.with_filter("symbols not like '%BJ'")
/// lab.with_years(2010, 2025)
/// lab.register("wcr", "1d:sum(5m:vol * 5m:close) / 1d:sum(5m:vol) / 1d:mean(5m:close)")
/// panel = lab.calc("output.csv")
/// result = lab.run(panel)
/// print(f"Sharpe: {result.sharpe_ratio:.4f}")
/// ```
#[pyclass(name = "AlfarsLab")]
struct PyAlfarsLab {
    inner: std::sync::Mutex<AlfarsLab>,
}

#[pymethods]
impl PyAlfarsLab {
    #[new]
    fn new() -> Self {
        // Users should call from_env() instead to load .env
        Self {
            inner: std::sync::Mutex::new(AlfarsLab::new(
                crate::data::clickhouse::ClickHouseSource::from_env(),
            )),
        }
    }

    #[staticmethod]
    fn from_env() -> Self {
        dotenv::dotenv().ok();
        Self {
            inner: std::sync::Mutex::new(AlfarsLab::new(
                crate::data::clickhouse::ClickHouseSource::from_env(),
            )),
        }
    }

    /// Create an AlfarsLab with a custom DataPoolConfig.
    ///
    /// ```python
    /// config = DataPoolConfig(
    ///     cache_policy=CachePolicy.keep_all(),
    ///     backtest_batch_size=20,
    /// )
    /// lab = AlfarsLab.from_env_with_config(config)
    /// ```
    #[staticmethod]
    fn from_env_with_config(config: &PyDataPoolConfig) -> Self {
        dotenv::dotenv().ok();
        Self {
            inner: std::sync::Mutex::new(AlfarsLab::new_with_config(
                crate::data::clickhouse::ClickHouseSource::from_env(),
                config.inner.clone(),
            )),
        }
    }

    fn with_filter(&self, filter: &str) {
        self.inner.lock().unwrap().set_filter(filter);
    }

    fn with_years(&self, start: i32, end: i32) {
        self.inner.lock().unwrap().set_years(start, end);
    }

    #[pyo3(signature = (quantiles, weight_method, long_top_n, short_top_n, buy_commission, sell_commission, rebalance_freq=1))]
    fn with_backtest_config(
        &self,
        quantiles: usize,
        weight_method: &str,
        long_top_n: usize,
        short_top_n: usize,
        buy_commission: f64,
        sell_commission: f64,
        rebalance_freq: usize,
    ) -> PyResult<()> {
        let wm = match weight_method {
            "equal" => WeightMethod::Equal,
            "weighted" => WeightMethod::Weighted,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "weight_method must be 'equal' or 'weighted'",
                ));
            }
        };
        let config = backtest::BacktestConfig {
            quantiles,
            weight_method: wm,
            long_top_n,
            short_top_n,
            rebalance_freq,
            fee_config: backtest::FeeConfig {
                buy_commission,
                sell_commission,
                ..Default::default()
            },
            ..Default::default()
        };
        self.inner.lock().unwrap().set_backtest_config(config);
        Ok(())
    }

    fn register(&self, name: &str, expression: &str) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap()
            .register(name, expression)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(())
    }

    fn calc(&self, csv_path: &str) -> PyResult<PyFactorPanel> {
        let panel = self
            .inner
            .lock()
            .unwrap()
            .calc(csv_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(PyFactorPanel { inner: panel })
    }

    fn run(&self, panel: &PyFactorPanel) -> PyResult<PyBacktestResult> {
        let result = self
            .inner
            .lock()
            .unwrap()
            .run(&panel.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(PyBacktestResult::from(result))
    }

    fn evaluate(&self, py: Python<'_>) -> PyResult<(Py<PyDict>, Py<PyAny>)> {
        py.check_signals()?;
        let (matrices, prices) = self
            .inner
            .lock()
            .unwrap()
            .evaluate()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        py.check_signals()?;
        let dict = PyDict::new(py);
        for (name, mat) in &matrices {
            let arr: Py<PyArray2<f64>> = mat.clone().into_pyarray(py).into();
            dict.set_item(name, arr)?;
        }
        let py_prices = Py::new(py, PyPriceMatrix { inner: prices })?;
        Ok((dict.into(), py_prices.into()))
    }

    /// Streaming per-factor evaluate + backtest. For large factor counts
    /// (e.g., alpha191), processes in small batches to avoid OOM.
    fn backtest_each(&self) -> PyResult<Vec<(String, PyBacktestResult)>> {
        let results = self
            .inner
            .lock()
            .unwrap()
            .evaluate_and_backtest_each()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(results
            .into_iter()
            .map(|(name, r)| (name, PyBacktestResult::from(r)))
            .collect())
    }

    fn run_multi(
        &self,
        factor_mats: Vec<Bound<'_, PyArray2<f64>>>,
        prices: &PyPriceMatrix,
    ) -> PyResult<PyBacktestResult> {
        let mats: Vec<_> = factor_mats
            .iter()
            .map(|m| m.readonly().as_array().to_owned())
            .collect();
        let result = self
            .inner
            .lock()
            .unwrap()
            .run_multi(&mats, &prices.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(PyBacktestResult::from(result))
    }

    /// Run genetic programming to discover alpha factors.
    ///
    /// Uses the lab's configured data source and filter. Returns a list of
    /// `(expression, fitness, ic, ir, turnover, complexity)` tuples.
    #[pyo3(signature = (population_size=100, max_generations=50, tournament_size=7, crossover_prob=0.8, mutation_prob=0.2, max_depth=6, use_diverse_init=true, smart_mutation_ratio=0.3, num_factors=3, max_symbols=0))]
    fn mine_factors(
        &self,
        population_size: usize,
        max_generations: usize,
        tournament_size: usize,
        crossover_prob: f64,
        mutation_prob: f64,
        max_depth: usize,
        use_diverse_init: bool,
        smart_mutation_ratio: f64,
        num_factors: usize,
        max_symbols: usize,
    ) -> PyResult<Vec<(String, f64, f64, f64, f64, usize)>> {
        let config = GPConfig {
            population_size,
            max_generations,
            tournament_size,
            crossover_prob,
            mutation_prob,
            max_depth,
            parent_diversity_penalty: 0.1,
            use_diverse_init,
            smart_mutation_ratio,
            use_frequencies: false,
        };
        let mut inner = self.inner.lock().unwrap();
        inner
            .mine_factors(config, num_factors, max_symbols)
            .map_err(|e| PyRuntimeError::new_err(e))
    }

    fn __repr__(&self) -> String {
        "AlfarsLab()".to_string()
    }
}
