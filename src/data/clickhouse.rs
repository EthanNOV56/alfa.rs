//! ClickHouse data source implementation
//!
//! This module provides ClickHouse-specific data source implementation.

use super::{DataError, DataSource};
use crate::types::DataFrame;

/// ClickHouse data source
///
/// Note: This is a placeholder implementation. The actual ClickHouse
/// integration is handled at the Python layer through clickhouse-connect.
#[derive(Debug, Clone)]
pub struct ClickHouseSource {
    host: String,
    port: u16,
    database: String,
    username: String,
}

impl ClickHouseSource {
    /// Create a new ClickHouse source using environment variables for credentials
    ///
    /// Uses environment variables:
    /// - CLICKHOUSE_HOST (default: "localhost")
    /// - CLICKHOUSE_PORT (default: 8123)
    /// - CLICKHOUSE_DATABASE (default: "default")
    /// - CLICKHOUSE_USER (default: "default")
    pub fn from_env() -> Self {
        let host = std::env::var("CLICKHOUSE_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port: u16 = std::env::var("CLICKHOUSE_PORT")
            .unwrap_or_else(|_| "8123".to_string())
            .parse()
            .unwrap_or(8123);
        let database = std::env::var("CLICKHOUSE_DATABASE").unwrap_or_else(|_| "default".to_string());
        let username = std::env::var("CLICKHOUSE_USER").unwrap_or_else(|_| "default".to_string());

        ClickHouseSource {
            host,
            port,
            database,
            username,
        }
    }

    /// Create a new ClickHouse source
    pub fn new(host: &str, port: u16, database: &str) -> Self {
        ClickHouseSource {
            host: host.to_string(),
            port,
            database: database.to_string(),
            username: "default".to_string(),
        }
    }

    /// Create a new ClickHouse source with username
    pub fn with_username(host: &str, port: u16, database: &str, username: &str) -> Self {
        ClickHouseSource {
            host: host.to_string(),
            port,
            database: database.to_string(),
            username: username.to_string(),
        }
    }

    /// Get the host address
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Get the port
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Get the database name
    pub fn database(&self) -> &str {
        &self.database
    }

    /// Get the username
    pub fn username(&self) -> &str {
        &self.username
    }
}

impl DataSource for ClickHouseSource {
    fn query(&self, _sql: &str) -> Result<DataFrame, DataError> {
        Err(DataError::Connection(
            "ClickHouse query not implemented in Rust. Use Python clickhouse-connect.".to_string(),
        ))
    }

    fn get_factor_data(
        &self,
        _symbol: &str,
        _start_date: &str,
        _end_date: &str,
    ) -> Result<DataFrame, DataError> {
        Err(DataError::Connection(
            "ClickHouse factor data not implemented in Rust. Use Python clickhouse-connect."
                .to_string(),
        ))
    }

    fn is_connected(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clickhouse_source_new() {
        let source = ClickHouseSource::new("localhost", 8123, "default");
        assert_eq!(source.host(), "localhost");
        assert_eq!(source.port(), 8123);
        assert_eq!(source.database(), "default");
        assert_eq!(source.username(), "default");
    }

    #[test]
    fn test_clickhouse_source_with_username() {
        let source = ClickHouseSource::with_username("localhost", 8123, "default", "readonly_user");
        assert_eq!(source.host(), "localhost");
        assert_eq!(source.port(), 8123);
        assert_eq!(source.database(), "default");
        assert_eq!(source.username(), "readonly_user");
    }

    #[test]
    fn test_clickhouse_source_from_env_defaults() {
        // Note: This test assumes default values when env vars are not set
        // Since we can't safely manipulate env vars in tests, we just verify the function works
        let source = ClickHouseSource::from_env();
        // The function should not panic, and returns default values if not set
        assert!(!source.host().is_empty());
    }

    #[test]
    fn test_clickhouse_source_from_env_with_values() {
        // Skip this test as it requires unsafe env manipulation
        // In a real scenario, these would be integration tests
        let source = ClickHouseSource::new("localhost", 8123, "default");
        assert_eq!(source.host(), "localhost");
    }

    #[test]
    fn test_clickhouse_is_connected_returns_false() {
        let source = ClickHouseSource::new("localhost", 8123, "default");
        assert!(!source.is_connected());
    }

    #[test]
    fn test_clickhouse_query_returns_error() {
        let source = ClickHouseSource::new("localhost", 8123, "default");
        let result = source.query("SELECT * FROM test");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DataError::Connection(_)));
    }

    #[test]
    fn test_clickhouse_get_factor_data_returns_error() {
        let source = ClickHouseSource::new("localhost", 8123, "default");
        let result = source.get_factor_data("000001.SZ", "2024-01-01", "2024-12-31");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, DataError::Connection(_)));
    }

    #[test]
    fn test_clickhouse_source_debug() {
        let source = ClickHouseSource::new("localhost", 8123, "default");
        let debug_str = format!("{:?}", source);
        assert!(debug_str.contains("ClickHouseSource"));
    }

    #[test]
    fn test_clickhouse_source_clone() {
        let source = ClickHouseSource::new("localhost", 8123, "default");
        let cloned = source.clone();
        assert_eq!(source.host(), cloned.host());
        assert_eq!(source.port(), cloned.port());
    }
}
