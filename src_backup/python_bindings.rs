//! Python bindings for alpha-expr crate using PyO3
//!
//! Minimal implementation focusing on core functionality.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::PyValueError;

use crate::expr::{Expr, Literal};
use crate::polars_style::{DataFrame, Series};

// ============================================================================
// Python Expr wrapper (simplified)
// ============================================================================

#[pyclass(name = "Expr")]
#[derive(Clone)]
pub struct PyExpr {
    inner: Expr,
}

impl PyExpr {
    fn from_expr(expr: Expr) -> Self {
        PyExpr { inner: expr }
    }
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
        PyExpr::from_expr(Expr::col(name))
    }

    /// Create a literal integer expression
    #[staticmethod]
    fn lit_int(value: i64) -> Self {
        PyExpr::from_expr(Expr::lit_int(value))
    }

    /// Create a literal float expression
    #[staticmethod]
    fn lit_float(value: f64) -> Self {
        PyExpr::from_expr(Expr::lit_float(value))
    }

    /// Add two expressions
    fn add(&self, other: &PyExpr) -> Self {
        PyExpr::from_expr(self.inner.clone().add(other.inner.clone()))
    }

    /// Subtract two expressions
    fn sub(&self, other: &PyExpr) -> Self {
        PyExpr::from_expr(self.inner.clone().sub(other.inner.clone()))
    }

    /// Multiply two expressions
    fn mul(&self, other: &PyExpr) -> Self {
        PyExpr::from_expr(self.inner.clone().mul(other.inner.clone()))
    }

    /// Divide two expressions
    fn div(&self, other: &PyExpr) -> Self {
        PyExpr::from_expr(self.inner.clone().div(other.inner.clone()))
    }

    /// Get string representation
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// Python Series wrapper (simplified)
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

// ============================================================================
// Python DataFrame wrapper (simplified)
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
                
                if let Ok(list) = value.extract::<Vec<f64>>() {
                    inner_columns.insert(col_name, Series::new(list));
                } else {
                    return Err(PyValueError::new_err(
                        format!("Column '{}' must be a list of floats", col_name)
                    ));
                }
            }
            
            Ok(PyDataFrame {
                inner: DataFrame::from_series_map(inner_columns)
                    .map_err(|e| PyValueError::new_err(e))?,
            })
        } else {
            Ok(PyDataFrame {
                inner: DataFrame::new(),
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

    fn __repr__(&self) -> String {
        format!("DataFrame(rows={}, cols={})", self.n_rows(), self.n_cols())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ============================================================================
// Python module definition
// ============================================================================

/// Alpha Expression System for Python (Minimal)
/// This function is called by the main `#[pymodule]` in lib.rs
pub fn alpha_expr_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyExpr>()?;
    m.add_class::<PySeries>()?;
    m.add_class::<PyDataFrame>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}