//! Python bindings for alpha-expr crate using PyO3
//!
//! This module provides Python interfaces to the Rust alpha expression system,
//! including expression building, evaluation, and backtesting functionality.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, IntoPyArray, PyArrayDyn};
use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2};

use crate::expr::{Expr, Literal, BinaryOp, UnaryOp};
use crate::polars_style::{DataFrame, Series, evaluate_expr_on_dataframe};
use crate::backtest::{BacktestEngine, BacktestResult, WeightMethod};
use crate::alpha_evaluation::AlphaExpressionEvaluator;
use crate::clickhouse_provider::ClickHouseProvider;

// ============================================================================
// Python Expr wrapper
// ============================================================================

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
                left: Box::new(self.inner.clone()),
                op: BinaryOp::Add,
                right: Box::new(other.inner.clone()),
            },
        }
    }

    /// Subtract two expressions
    fn sub(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Box::new(self.inner.clone()),
                op: BinaryOp::Subtract,
                right: Box::new(other.inner.clone()),
            },
        }
    }

    /// Multiply two expressions
    fn mul(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Box::new(self.inner.clone()),
                op: BinaryOp::Multiply,
                right: Box::new(other.inner.clone()),
            },
        }
    }

    /// Divide two expressions
    fn div(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Box::new(self.inner.clone()),
                op: BinaryOp::Divide,
                right: Box::new(other.inner.clone()),
            },
        }
    }

    /// Negate expression
    fn neg(&self) -> Self {
        PyExpr {
            inner: Expr::UnaryExpr {
                op: UnaryOp::Negate,
                expr: Box::new(self.inner.clone()),
            },
        }
    }

    /// Absolute value
    fn abs(&self) -> Self {
        PyExpr {
            inner: Expr::UnaryExpr {
                op: UnaryOp::Abs,
                expr: Box::new(self.inner.clone()),
            },
        }
    }

    /// Square root
    fn sqrt(&self) -> Self {
        PyExpr {
            inner: Expr::UnaryExpr {
                op: UnaryOp::Sqrt,
                expr: Box::new(self.inner.clone()),
            },
        }
    }

    /// Natural logarithm
    fn log(&self) -> Self {
        PyExpr {
            inner: Expr::UnaryExpr {
                op: UnaryOp::Log,
                expr: Box::new(self.inner.clone()),
            },
        }
    }

    /// Exponential
    fn exp(&self) -> Self {
        PyExpr {
            inner: Expr::UnaryExpr {
                op: UnaryOp::Exp,
                expr: Box::new(self.inner.clone()),
            },
        }
    }

    /// Greater than comparison
    fn gt(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Box::new(self.inner.clone()),
                op: BinaryOp::GreaterThan,
                right: Box::new(other.inner.clone()),
            },
        }
    }

    /// Less than comparison
    fn lt(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Box::new(self.inner.clone()),
                op: BinaryOp::LessThan,
                right: Box::new(other.inner.clone()),
            },
        }
    }

    /// Equal comparison
    fn eq(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Box::new(self.inner.clone()),
                op: BinaryOp::Equal,
                right: Box::new(other.inner.clone()),
            },
        }
    }

    /// Not equal comparison
    fn ne(&self, other: &PyExpr) -> Self {
        PyExpr {
            inner: Expr::BinaryExpr {
                left: Box::new(self.inner.clone()),
                op: BinaryOp::NotEqual,
                right: Box::new(other.inner.clone()),
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

// ============================================================================
// Python Series wrapper
// ============================================================================

#[pyclass(name = "Series")]
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

    /// Create from numpy array
    #[staticmethod]
    fn from_numpy(py: Python, arr: &PyArray1<f64>) -> PyResult<Self> {
        let data = arr.readonly();
        let slice = data.as_slice().ok_or_else(|| {
            PyValueError::new_err("Cannot convert numpy array to slice")
        })?;
        
        Ok(PySeries {
            inner: Series::new(slice.to_vec()),
        })
    }

    /// Convert to numpy array
    fn to_numpy<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        let data = self.inner.data().to_vec();
        data.into_pyarray(py)
    }

    /// Get length
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Get mean
    fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }

    /// Get standard deviation
    fn std(&self, ddof: f64) -> Option<f64> {
        self.inner.std(ddof)
    }

    /// Get minimum
    fn min(&self) -> Option<f64> {
        self.inner.min()
    }

    /// Get maximum
    fn max(&self) -> Option<f64> {
        self.inner.max()
    }

    /// Lag operation
    fn lag(&self, periods: usize) -> Self {
        PySeries {
            inner: self.inner.lag(periods),
        }
    }

    /// Difference operation
    fn diff(&self, periods: usize) -> Self {
        PySeries {
            inner: self.inner.diff(periods),
        }
    }

    /// Percentage change
    fn pct_change(&self, periods: usize) -> Self {
        PySeries {
            inner: self.inner.pct_change(periods),
        }
    }

    /// Moving average
    fn moving_average(&self, window: usize) -> Self {
        PySeries {
            inner: self.inner.moving_average(window),
        }
    }

    /// Exponential moving average
    fn ema(&self, span: usize) -> Self {
        PySeries {
            inner: self.inner.ema(span),
        }
    }

    /// Rolling standard deviation
    fn rolling_std(&self, window: usize) -> Self {
        PySeries {
            inner: self.inner.rolling_std(window),
        }
    }

    /// Get data as Python list
    fn to_list(&self) -> Vec<f64> {
        self.inner.data().to_vec()
    }

    /// Get string representation
    fn __repr__(&self) -> String {
        format!("Series(len={})", self.len())
    }

    /// Get string representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// Python DataFrame wrapper
// ============================================================================

#[pyclass(name = "DataFrame")]
pub struct PyDataFrame {
    inner: DataFrame,
}

#[pymethods]
impl PyDataFrame {
    #[new]
    fn new(columns: Option<&PyDict>) -> PyResult<Self> {
        if let Some(cols) = columns {
            let mut inner_columns = std::collections::HashMap::new();
            
            for (key, value) in cols.iter() {
                let col_name = key.extract::<String>()?;
                
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
            
            Ok(PyDataFrame {
                inner: DataFrame::new(inner_columns),
            })
        } else {
            Ok(PyDataFrame {
                inner: DataFrame::new(std::collections::HashMap::new()),
            })
        }
    }

    /// Create DataFrame from dictionary of lists
    #[staticmethod]
    fn from_dict(py: Python, data: &PyDict) -> PyResult<Self> {
        let mut columns = std::collections::HashMap::new();
        
        for (key, value) in data.iter() {
            let col_name = key.extract::<String>()?;
            
            if let Ok(list) = value.extract::<Vec<f64>>() {
                columns.insert(col_name, Series::new(list));
            } else {
                return Err(PyValueError::new_err(
                    format!("Column '{}' must be a list of floats", col_name)
                ));
            }
        }
        
        Ok(PyDataFrame {
            inner: DataFrame::new(columns),
        })
    }

    /// Create DataFrame from numpy arrays
    #[staticmethod]
    fn from_numpy_arrays(py: Python, columns: &PyDict) -> PyResult<Self> {
        let mut inner_columns = std::collections::HashMap::new();
        
        for (key, value) in columns.iter() {
            let col_name = key.extract::<String>()?;
            
            if let Ok(arr) = value.extract::<&PyArray1<f64>>() {
                let data = arr.readonly();
                let slice = data.as_slice().ok_or_else(|| {
                    PyValueError::new_err("Cannot convert numpy array to slice")
                })?;
                
                inner_columns.insert(col_name, Series::new(slice.to_vec()));
            } else {
                return Err(PyValueError::new_err(
                    format!("Column '{}' must be a numpy array of floats", col_name)
                ));
            }
        }
        
        Ok(PyDataFrame {
            inner: DataFrame::new(inner_columns),
        })
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

    /// Get a column as Series
    fn column(&self, name: &str) -> PyResult<PySeries> {
        self.inner.column(name)
            .cloned()
            .map(|series| PySeries { inner: series })
            .ok_or_else(|| PyValueError::new_err(format!("Column '{}' not found", name)))
    }

    /// Add a column
    fn add_column(&mut self, name: String, series: PySeries) -> PyResult<()> {
        if self.inner.n_rows() == 0 || series.len() == self.inner.n_rows() {
            self.inner.add_column(name, series.inner);
            Ok(())
        } else {
            Err(PyValueError::new_err(
                format!("Series length {} must match DataFrame row count {}", 
                       series.len(), self.inner.n_rows())
            ))
        }
    }

    /// Evaluate an expression on this DataFrame
    fn evaluate(&self, expr: &PyExpr) -> PyResult<PySeries> {
        evaluate_expr_on_dataframe(&expr.inner, &self.inner)
            .map(|series| PySeries { inner: series })
            .map_err(|e| PyValueError::new_err(e))
    }

    /// Get string representation
    fn __repr__(&self) -> String {
        format!("DataFrame(rows={}, cols={})", self.n_rows(), self.n_cols())
    }

    /// Get string representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// Python BacktestEngine wrapper
// ============================================================================

#[pyclass(name = "BacktestEngine")]
pub struct PyBacktestEngine {
    inner: BacktestEngine,
}

#[pymethods]
impl PyBacktestEngine {
    #[new]
    fn new(
        weight_method: &str,
        commission_rate: f64,
        enable_ic: bool,
    ) -> PyResult<Self> {
        let weight_method = match weight_method {
            "equal" => WeightMethod::Equal,
            "value" => WeightMethod::Value,
            "rank" => WeightMethod::Rank,
            "zscore" => WeightMethod::ZScore,
            _ => return Err(PyValueError::new_err(
                "weight_method must be one of: 'equal', 'value', 'rank', 'zscore'"
            )),
        };
        
        Ok(PyBacktestEngine {
            inner: BacktestEngine::new(weight_method, commission_rate, enable_ic),
        })
    }

    /// Run backtest with qcut grouping
    fn qcut_backtest(
        &self,
        symbols: Vec<String>,
        dates: Vec<String>,
        factors: Vec<Vec<f64>>,
        returns: Vec<Vec<f64>>,
        n_groups: usize,
    ) -> PyResult<PyBacktestResult> {
        // Convert inputs to 2D arrays
        let n_dates = dates.len();
        let n_symbols = symbols.len();
        
        if factors.len() != n_dates {
            return Err(PyValueError::new_err(
                format!("factors length {} must match dates length {}", 
                       factors.len(), n_dates)
            ));
        }
        
        if returns.len() != n_dates {
            return Err(PyValueError::new_err(
                format!("returns length {} must match dates length {}", 
                       returns.len(), n_dates)
            ));
        }
        
        // Run backtest
        let result = self.inner.qcut_backtest(
            &symbols,
            &dates,
            &factors,
            &returns,
            n_groups,
        ).map_err(|e| PyValueError::new_err(e))?;
        
        Ok(PyBacktestResult { inner: result })
    }

    /// Get string representation
    fn __repr__(&self) -> String {
        "BacktestEngine".to_string()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// Python BacktestResult wrapper
// ============================================================================

#[pyclass(name = "BacktestResult")]
pub struct PyBacktestResult {
    inner: BacktestResult,
}

#[pymethods]
impl PyBacktestResult {
    /// Get group returns
    fn group_returns(&self) -> Vec<Vec<f64>> {
        self.inner.group_returns.clone()
    }

    /// Get long-short cumulative return
    fn long_short_cum_return(&self) -> f64 {
        self.inner.long_short_cum_return
    }

    /// Get long-short daily returns
    fn long_short_daily_returns(&self) -> Vec<f64> {
        self.inner.long_short_daily_returns.clone()
    }

    /// Get IC (Information Coefficient) if enabled
    fn ic(&self) -> Option<f64> {
        self.inner.ic
    }

    /// Get Sharpe ratio
    fn sharpe_ratio(&self) -> f64 {
        self.inner.sharpe_ratio()
    }

    /// Get maximum drawdown
    fn max_drawdown(&self) -> f64 {
        self.inner.max_drawdown()
    }

    /// Get annualized return
    fn annualized_return(&self) -> f64 {
        self.inner.annualized_return()
    }

    /// Get annualized volatility
    fn annualized_volatility(&self) -> f64 {
        self.inner.annualized_volatility()
    }

    /// Get Win rate
    fn win_rate(&self) -> f64 {
        self.inner.win_rate()
    }

    /// Get string representation
    fn __repr__(&self) -> String {
        format!("BacktestResult(sharpe={:.2}, max_dd={:.1%})", 
                self.sharpe_ratio(), self.max_drawdown())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// Python ClickHouseProvider wrapper
// ============================================================================

#[pyclass(name = "ClickHouseProvider")]
pub struct PyClickHouseProvider {
    inner: ClickHouseProvider,
}

#[pymethods]
impl PyClickHouseProvider {
    #[new]
    fn new(host: &str, port: u16, user: &str, password: &str, database: Option<&str>) -> PyResult<Self> {
        let provider = ClickHouseProvider::new(
            host,
            port,
            user,
            password,
            database.map(|s| s.to_string()),
        ).map_err(|e| PyValueError::new_err(e))?;
        
        Ok(PyClickHouseProvider { inner: provider })
    }

    /// Execute query and return results as list of dictionaries
    fn query(&self, sql: &str) -> PyResult<Vec<Py<PyDict>>> {
        let results = self.inner.exec_query_json_each(sql)
            .map_err(|e| PyValueError::new_err(e))?;
        
        Python::with_gil(|py| {
            let mut py_results = Vec::new();
            for row in results {
                let dict = PyDict::new(py);
                for (key, value) in row {
                    // Convert serde_json::Value to Python object
                    let py_value = match value {
                        serde_json::Value::Null => py.None(),
                        serde_json::Value::Bool(b) => b.into_py(py),
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                i.into_py(py)
                            } else if let Some(f) = n.as_f64() {
                                f.into_py(py)
                            } else {
                                n.to_string().into_py(py)
                            }
                        }
                        serde_json::Value::String(s) => s.into_py(py),
                        _ => value.to_string().into_py(py),
                    };
                    dict.set_item(key, py_value)?;
                }
                py_results.push(dict.into());
            }
            Ok(py_results)
        })
    }

    /// Get string representation
    fn __repr__(&self) -> String {
        "ClickHouseProvider".to_string()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// Python module definition
// ============================================================================

/// Alpha Expression System for Python
///
/// This module provides Python bindings for the alpha-expr crate,
/// including expression building, evaluation, backtesting, and
/// ClickHouse integration.
#[pymodule]
fn alpha_expr_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyExpr>()?;
    m.add_class::<PySeries>()?;
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PyBacktestEngine>()?;
    m.add_class::<PyBacktestResult>()?;
    m.add_class::<PyClickHouseProvider>()?;

    // Add helper functions
    m.add_function(wrap_pyfunction!(create_wcr_factor, m)?)?;
    m.add_function(wrap_pyfunction!(create_momentum_factor, m)?)?;
    m.add_function(wrap_pyfunction!(create_volatility_factor, m)?)?;

    // Add version constant
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

// ============================================================================
// Helper functions
// ============================================================================

/// Create a WCR (Weighted Close Ratio) factor expression
///
/// Args:
///     close_col: Name of close price column
///     volume_col: Name of volume column
///     ma_period: Moving average period (default: 10)
///
/// Returns:
///     Expression: (close * volume) / moving_average(close, ma_period)
#[pyfunction]
fn create_wcr_factor(close_col: &str, volume_col: &str, ma_period: Option<usize>) -> PyExpr {
    let period = ma_period.unwrap_or(10);
    
    let close_expr = PyExpr::col(close_col);
    let volume_expr = PyExpr::col(volume_col);
    
    // Create expression manually since alpha() method not directly exposed
    let ma_expr = PyExpr {
        inner: Expr::FunctionCall {
            name: "moving_average".to_string(),
            args: vec![
                Expr::Column(close_col.to_string()),
                Expr::Literal(Literal::Integer(period as i64)),
            ],
        },
    };
    
    (close_expr.mul(&volume_expr)).div(&ma_expr)
}

/// Create a momentum factor expression
///
/// Args:
///     price_col: Name of price column
///     period: Lookback period (default: 1)
///
/// Returns:
///     Expression: (price - lag(price, period)) / lag(price, period)
#[pyfunction]
fn create_momentum_factor(price_col: &str, period: Option<usize>) -> PyExpr {
    let period = period.unwrap_or(1);
    
    let price_expr = PyExpr::col(price_col);
    
    // Create lag expression manually
    let lag_expr = PyExpr {
        inner: Expr::FunctionCall {
            name: "lag".to_string(),
            args: vec![
                Expr::Column(price_col.to_string()),
                Expr::Literal(Literal::Integer(period as i64)),
            ],
        },
    };
    
    (price_expr.sub(&lag_expr)).div(&lag_expr)
}

/// Create a volatility factor expression
///
/// Args:
///     price_col: Name of price column
///     period: Rolling window period (default: 20)
///
/// Returns:
///     Expression: rolling_std(price, period)
#[pyfunction]
fn create_volatility_factor(price_col: &str, period: Option<usize>) -> PyExpr {
    let period = period.unwrap_or(20);
    
    PyExpr {
        inner: Expr::FunctionCall {
            name: "rolling_std".to_string(),
            args: vec![
                Expr::Column(price_col.to_string()),
                Expr::Literal(Literal::Integer(period as i64)),
            ],
        },
    }
}