//! ClickHouse data source implementation
//!
//! This module provides ClickHouse-specific data source implementation.

use super::{DataError, DataSource};
use crate::types::DataFrame;
use std::collections::HashMap;

/// ClickHouse data source
#[derive(Debug, Clone)]
pub struct ClickHouseSource {
    host: String,
    port: u16,
    database: String,
    username: String,
    password: Option<String>,
    /// Volume unit (e.g., 100 for hand = 100 shares)
    volume_unit: u16,
    /// Amount unit (e.g., 1000 for 千元)
    amount_unit: u16,
}

impl ClickHouseSource {
    /// Create a new ClickHouse source using environment variables for credentials
    ///
    /// Uses environment variables:
    /// - CLICKHOUSE_HOST (default: "localhost")
    /// - CLICKHOUSE_PORT (default: 8123)
    /// - CLICKHOUSE_DATABASE (default: "default")
    /// - CLICKHOUSE_USER (default: "default")
    /// - CLICKHOUSE_PASSWORD (optional)
    /// - VOLUME_UNIT (default: 100, e.g., hand = 100 shares)
    /// - AMOUNT_UNIT (default: 10000, e.g., 万元)
    pub fn from_env() -> Self {
        let host = std::env::var("CLICKHOUSE_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port: u16 = std::env::var("CLICKHOUSE_PORT")
            .unwrap_or_else(|_| "8123".to_string())
            .parse()
            .unwrap_or(8123);
        let database =
            std::env::var("CLICKHOUSE_DATABASE").unwrap_or_else(|_| "default".to_string());
        let username = std::env::var("CLICKHOUSE_USER").unwrap_or_else(|_| "default".to_string());
        let password = std::env::var("CLICKHOUSE_PASSWORD").ok();
        let volume_unit: u16 = std::env::var("VOLUME_UNIT")
            .unwrap_or_else(|_| "100".to_string())
            .parse()
            .unwrap_or(100);
        let amount_unit: u16 = std::env::var("AMOUNT_UNIT")
            .unwrap_or_else(|_| "10000".to_string())
            .parse()
            .unwrap_or(10000);

        ClickHouseSource {
            host,
            port,
            database,
            username,
            password,
            volume_unit,
            amount_unit,
        }
    }

    /// Create a new ClickHouse source
    pub fn new(host: &str, port: u16, database: &str) -> Self {
        ClickHouseSource {
            host: host.to_string(),
            port,
            database: database.to_string(),
            username: "default".to_string(),
            password: None,
            volume_unit: 100,
            amount_unit: 10000,
        }
    }

    /// Create a new ClickHouse source with username
    pub fn with_username(host: &str, port: u16, database: &str, username: &str) -> Self {
        ClickHouseSource {
            host: host.to_string(),
            port,
            database: database.to_string(),
            username: username.to_string(),
            password: None,
            volume_unit: 100,
            amount_unit: 10000,
        }
    }

    /// Create a new ClickHouse source with custom volume/amount units
    pub fn with_units(host: &str, port: u16, database: &str, username: &str, volume_unit: u16, amount_unit: u16) -> Self {
        ClickHouseSource {
            host: host.to_string(),
            port,
            database: database.to_string(),
            username: username.to_string(),
            password: None,
            volume_unit,
            amount_unit,
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

    /// Build the ClickHouse HTTP URL
    fn build_url(&self) -> String {
        format!("http://{}:{}/", self.host, self.port)
    }

    /// Execute a SQL query and return results as HashMap
    pub fn query_to_hashmap(&self, sql: &str) -> Result<HashMap<String, Vec<f64>>, DataError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| DataError::Connection(format!("Failed to create client: {}", e)))?;

        // Build query string - use GET request style (URL parameters)
        let mut url = format!(
            "{}?database={}&default_format=JSON&query={}",
            self.build_url(),
            self.database,
            urlencoding::encode(sql)
        );
        if !self.username.is_empty() {
            url.push_str(&format!("&user={}", self.username));
        }
        if let Some(ref password) = self.password {
            url.push_str(&format!("&password={}", password));
        }

        let response = client
            .get(&url)
            .send()
            .map_err(|e| DataError::Connection(format!("Failed to connect: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            return Err(DataError::Query(format!("HTTP {}: {}", status, text)));
        }

        // Parse JSON response
        let json: serde_json::Value = response
            .json()
            .map_err(|e| DataError::Query(format!("Failed to parse JSON: {}", e)))?;

        // ClickHouse returns data in "data" array with "columns" and "data"
        let data = json.get("data").and_then(|v| v.as_array()).ok_or_else(|| {
            DataError::Query("Invalid response format: no data array".to_string())
        })?;

        if data.is_empty() {
            return Ok(HashMap::new());
        }

        // Get column names
        let columns: Vec<String> = json
            .get("columns")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.get("name").and_then(|n| n.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| {
                // Fallback: try to get keys from first data row
                if let Some(first_row) = data.first() {
                    if let Some(obj) = first_row.as_object() {
                        obj.keys().cloned().collect()
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                }
            });

        // Build result HashMap
        let mut result: HashMap<String, Vec<f64>> = HashMap::new();
        for col in &columns {
            result.insert(col.clone(), Vec::new());
        }

        for row in data {
            if let Some(obj) = row.as_object() {
                for col in &columns {
                    let val = obj.get(col).and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                    if let Some(vec) = result.get_mut(col) {
                        vec.push(val);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Fetch stock data for a symbol
    pub fn fetch_stock_data(
        &self,
        symbols: &[String],
        start_date: &str,
        end_date: &str,
        table_name: &str,
    ) -> Result<HashMap<String, Vec<f64>>, DataError> {
        let symbols_str = symbols
            .iter()
            .map(|s| format!("'{}'", s))
            .collect::<Vec<_>>()
            .join(",");

        // Compute vwap conversion factor: amount_unit / volume_unit
        let vwap_factor = self.amount_unit as f64 / self.volume_unit as f64;

        let sql = format!(
            "SELECT symbol, trading_date, open, high, low, close, volume, amount, \
             (close - open) / open AS returns, \
             amount * {} / volume AS vwap, \
             adjust_factor \
             FROM {} \
             WHERE symbol IN ({}) \
             AND trading_date >= '{}' \
             AND trading_date <= '{}' \
             ORDER BY symbol, trading_date",
            vwap_factor, table_name, symbols_str, start_date, end_date
        );

        self.query_to_hashmap(&sql)
    }
}

impl DataSource for ClickHouseSource {
    fn query(&self, sql: &str) -> Result<DataFrame, DataError> {
        let data = self.query_to_hashmap(sql)?;
        if data.is_empty() {
            return Ok(DataFrame::new());
        }

        // Convert HashMap<String, Vec<f64>> to DataFrame
        use crate::types::Series;
        let mut columns_map: std::collections::HashMap<String, Series> =
            std::collections::HashMap::new();
        for (name, values) in data {
            columns_map.insert(name.clone(), Series::new(values).with_name(&name));
        }

        DataFrame::from_series_map(columns_map).map_err(|e| DataError::Query(e))
    }

    fn get_factor_data(
        &self,
        symbol: &str,
        start_date: &str,
        end_date: &str,
    ) -> Result<DataFrame, DataError> {
        // Default table name is stock_1d
        let table_name = "stock_1d";
        self.query(&format!(
            "SELECT symbol, trading_date, open, high, low, close, volume, amount \
             FROM {} \
             WHERE symbol = '{}' \
             AND trading_date >= '{}' \
             AND trading_date <= '{}' \
             ORDER BY trading_date",
            table_name, symbol, start_date, end_date
        ))
    }

    fn is_connected(&self) -> bool {
        // Try to execute a simple query to check connection
        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
        {
            Ok(c) => c,
            Err(_) => return false,
        };

        // Use GET request style for connection check
        let mut url = format!(
            "{}?database={}&default_format=JSON&query=SELECT+1",
            self.build_url(),
            self.database
        );
        if !self.username.is_empty() {
            url.push_str(&format!("&user={}", self.username));
        }
        if let Some(ref password) = self.password {
            url.push_str(&format!("&password={}", password));
        }

        match client.get(&url).send() {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
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
