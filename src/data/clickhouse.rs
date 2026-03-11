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
    _host: String,
    _port: u16,
    _database: String,
}

impl ClickHouseSource {
    /// Create a new ClickHouse source
    pub fn new(host: &str, port: u16, database: &str) -> Self {
        ClickHouseSource {
            _host: host.to_string(),
            _port: port,
            _database: database.to_string(),
        }
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
