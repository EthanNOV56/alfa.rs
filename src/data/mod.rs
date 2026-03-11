//! Data abstraction layer module
//!
//! This module provides data source abstractions and implementations
//! for connecting to various data backends.

pub mod clickhouse;
pub mod source;

pub use clickhouse::ClickHouseSource;
pub use source::{DataError, DataSource};

use ndarray::Array1;
use std::collections::HashMap;

/// Vectorized data structure for SIMD-friendly columnar storage
///
/// This structure stores numerical columns as ndarray::Array1<f64> for
/// SIMD-optimized vector operations, while keeping metadata separate.
#[derive(Debug, Clone)]
pub struct VectorizedData {
    /// Numerical columns for SIMD computation - stored as Array1<f64>
    pub columns: HashMap<String, Array1<f64>>,
    /// Symbol metadata (for alignment and debugging)
    pub symbols: Vec<String>,
    /// Date metadata (for alignment and debugging)
    pub dates: Vec<String>,
    /// Original row mapping: (symbol, date) pairs
    pub row_mapping: Vec<(String, String)>,
}

impl VectorizedData {
    /// Create a new empty VectorizedData
    pub fn new() -> Self {
        VectorizedData {
            columns: HashMap::new(),
            symbols: Vec::new(),
            dates: Vec::new(),
            row_mapping: Vec::new(),
        }
    }

    /// Create VectorizedData from HashMap<String, Vec<f64>>
    /// This converts the legacy format to vectorized format
    pub fn from_hashmap(data: &HashMap<String, Vec<f64>>) -> Self {
        let n_rows = data.values().next().map(|v| v.len()).unwrap_or(0);
        let mut columns = HashMap::new();

        for (name, vals) in data {
            columns.insert(name.clone(), Array1::from(vals.clone()));
        }

        VectorizedData {
            columns,
            symbols: Vec::new(),
            dates: Vec::new(),
            row_mapping: Vec::new(),
        }
    }

    /// Get the number of rows
    pub fn n_rows(&self) -> usize {
        self.columns
            .values()
            .next()
            .map(|arr| arr.len())
            .unwrap_or(0)
    }

    /// Get a column as Array1<f64>
    pub fn get_column(&self, name: &str) -> Option<&Array1<f64>> {
        self.columns.get(name)
    }

    /// Get a column as Vec<f64> (for compatibility)
    pub fn get_column_vec(&self, name: &str) -> Option<Vec<f64>> {
        self.columns.get(name).map(|arr| arr.to_vec())
    }

    /// Insert a column
    pub fn insert_column(&mut self, name: String, data: Array1<f64>) {
        self.columns.insert(name, data);
    }

    /// Convert back to HashMap<String, Vec<f64>> (for compatibility)
    pub fn to_hashmap(&self) -> HashMap<String, Vec<f64>> {
        self.columns
            .iter()
            .map(|(k, v)| (k.clone(), v.to_vec()))
            .collect()
    }
}

impl Default for VectorizedData {
    fn default() -> Self {
        Self::new()
    }
}
