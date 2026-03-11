//! Data source abstraction layer
//!
//! This module provides a trait-based abstraction for data sources,
//! allowing flexible backends (ClickHouse, MySQL, CSV, etc.)

use crate::types::DataFrame;
use std::fmt;

/// Error type for data source operations
#[derive(Debug, Clone)]
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

/// A mock data source for testing purposes
#[derive(Debug)]
pub struct MockDataSource {
    connected: bool,
    query_result: Option<Result<crate::types::DataFrame, DataError>>,
}

impl MockDataSource {
    /// Create a new mock data source
    pub fn new() -> Self {
        MockDataSource {
            connected: false,
            query_result: None,
        }
    }

    /// Create a connected mock data source
    pub fn connected() -> Self {
        MockDataSource {
            connected: true,
            query_result: None,
        }
    }

    /// Set the query result to return
    pub fn with_query_result(mut self, result: Result<crate::types::DataFrame, DataError>) -> Self {
        self.query_result = Some(result);
        self
    }
}

impl Default for MockDataSource {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSource for MockDataSource {
    fn query(&self, _sql: &str) -> Result<crate::types::DataFrame, DataError> {
        if let Some(result) = &self.query_result {
            result.clone()
        } else {
            Err(DataError::Query("No query result set".to_string()))
        }
    }

    fn get_factor_data(
        &self,
        _symbol: &str,
        _start_date: &str,
        _end_date: &str,
    ) -> Result<crate::types::DataFrame, DataError> {
        if let Some(result) = &self.query_result {
            result.clone()
        } else {
            Err(DataError::NotFound("No factor data available".to_string()))
        }
    }

    fn is_connected(&self) -> bool {
        self.connected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DataFrame;

    #[test]
    fn test_data_error_query() {
        let err = DataError::Query("Invalid SQL".to_string());
        assert_eq!(format!("{}", err), "Query error: Invalid SQL");
    }

    #[test]
    fn test_data_error_connection() {
        let err = DataError::Connection("Connection refused".to_string());
        assert_eq!(format!("{}", err), "Connection error: Connection refused");
    }

    #[test]
    fn test_data_error_not_found() {
        let err = DataError::NotFound("Table not found".to_string());
        assert_eq!(format!("{}", err), "Data not found: Table not found");
    }

    #[test]
    fn test_data_error_invalid_param() {
        let err = DataError::InvalidParam("Invalid date format".to_string());
        assert_eq!(format!("{}", err), "Invalid parameter: Invalid date format");
    }

    #[test]
    fn test_data_error_debug() {
        let err = DataError::Query("test error".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Query"));
    }

    #[test]
    fn test_mock_data_source_default() {
        let mock = MockDataSource::new();
        assert!(!mock.is_connected());
    }

    #[test]
    fn test_mock_data_source_connected() {
        let mock = MockDataSource::connected();
        assert!(mock.is_connected());
    }

    #[test]
    fn test_mock_data_source_query_error() {
        let mock = MockDataSource::new();
        let result = mock.query("SELECT * FROM test");
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_data_source_with_query_result() {
        let df = DataFrame::new();
        let mock = MockDataSource::new().with_query_result(Ok(df));
        let result = mock.query("SELECT * FROM test");
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_data_source_get_factor_data() {
        let mock = MockDataSource::new();
        let result = mock.get_factor_data("000001.SZ", "2024-01-01", "2024-12-31");
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_data_source_with_factor_data_result() {
        let df = DataFrame::new();
        let mock = MockDataSource::new().with_query_result(Ok(df));
        let result = mock.get_factor_data("000001.SZ", "2024-01-01", "2024-12-31");
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_data_source_default_trait_object() {
        let mock: MockDataSource = Default::default();
        assert!(!mock.is_connected());
    }
}

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
