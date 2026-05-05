//! Error types and query filter for data access.

use std::fmt;

/// Generic query filter for flexible data filtering
#[derive(Debug, Clone, Default)]
pub struct QueryFilter {
    /// Columns to select
    pub columns: Vec<String>,
    /// Stock symbols to filter (None = all symbols)
    pub symbols: Option<Vec<String>>,
    /// Date range filter (start, end) - uses trading_date column
    pub date_range: Option<(String, String)>,
    /// Additional SQL conditions (e.g., "market_cap > 1000000000")
    pub conditions: Vec<String>,
}

impl QueryFilter {
    /// Create a new QueryFilter with required columns
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            symbols: None,
            date_range: None,
            conditions: Vec::new(),
        }
    }

    /// Add symbol filter
    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.symbols = Some(symbols);
        self
    }

    /// Add date range filter
    pub fn with_date_range(mut self, start: &str, end: &str) -> Self {
        self.date_range = Some((start.to_string(), end.to_string()));
        self
    }

    /// Add additional condition
    pub fn with_condition(mut self, condition: &str) -> Self {
        self.conditions.push(condition.to_string());
        self
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_query_filter_new() {
        let filter = QueryFilter::new(vec!["close".to_string(), "open".to_string()]);
        assert_eq!(filter.columns.len(), 2);
        assert!(filter.symbols.is_none());
        assert!(filter.date_range.is_none());
        assert!(filter.conditions.is_empty());
    }

    #[test]
    fn test_query_filter_with_symbols() {
        let filter = QueryFilter::new(vec!["close".to_string()])
            .with_symbols(vec!["000001.SZ".to_string(), "600000.SH".to_string()]);
        assert_eq!(filter.columns.len(), 1);
        assert!(filter.symbols.is_some());
        let symbols = filter.symbols.unwrap();
        assert_eq!(symbols.len(), 2);
    }

    #[test]
    fn test_query_filter_with_date_range() {
        let filter =
            QueryFilter::new(vec!["close".to_string()]).with_date_range("2024-01-01", "2024-12-31");
        assert!(filter.date_range.is_some());
        let (start, end) = filter.date_range.unwrap();
        assert_eq!(start, "2024-01-01");
        assert_eq!(end, "2024-12-31");
    }

    #[test]
    fn test_query_filter_with_conditions() {
        let filter = QueryFilter::new(vec!["close".to_string()])
            .with_condition("market_cap > 1000000000")
            .with_condition("pe > 0");
        assert_eq!(filter.conditions.len(), 2);
    }
}
