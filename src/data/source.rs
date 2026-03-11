//! Data source abstraction layer
//!
//! This module provides a trait-based abstraction for data sources,
//! allowing flexible backends (ClickHouse, MySQL, CSV, etc.)

use crate::types::DataFrame;
use std::fmt;

/// Error type for data source operations
#[derive(Debug)]
pub enum DataError {
    Query(String),
    Connection(String),
    NotFound(String),
    InvalidParam(String),
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::Query(s) => write!(f, "Query error: {}", s),
            DataError::Connection(s) => write!(f, "Connection error: {}", s),
            DataError::NotFound(s) => write!(f, "Data not found: {}", s),
            DataError::InvalidParam(s) => write!(f, "Invalid parameter: {}", s),
        }
    }
}

impl std::error::Error for DataError {}

/// Trait for data source implementations
pub trait DataSource {
    /// Execute a SQL query and return results as DataFrame
    fn query(&self, sql: &str) -> Result<DataFrame, DataError>;

    /// Get factor data for a symbol within a date range
    fn get_factor_data(
        &self,
        symbol: &str,
        start_date: &str,
        end_date: &str,
    ) -> Result<DataFrame, DataError>;

    /// Check if the data source is connected
    fn is_connected(&self) -> bool;
}
