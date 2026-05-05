//! ClickHouse data source implementation
//!
//! This module provides ClickHouse-specific data source implementation.

use super::DataError;
use std::sync::OnceLock;
use std::time::Duration;

/// ClickHouse data source
#[derive(Debug)]
pub struct ClickHouseSource {
    host: String,
    port: u16,
    database: String,
    username: String,
    password: Option<String>,
    /// Volume unit (e.g., 100 for 手 = 100 shares)
    volume_unit: u16,
    /// Amount unit (e.g., 1000 for 千元)
    amount_unit: u16,
    /// Lazily-initialized reqwest client for HTTP keep-alive connection reuse
    client: OnceLock<reqwest::blocking::Client>,
}

impl Clone for ClickHouseSource {
    fn clone(&self) -> Self {
        Self {
            host: self.host.clone(),
            port: self.port,
            database: self.database.clone(),
            username: self.username.clone(),
            password: self.password.clone(),
            volume_unit: self.volume_unit,
            amount_unit: self.amount_unit,
            client: OnceLock::new(),
        }
    }
}

impl ClickHouseSource {
    /// Create a new ClickHouse source using environment variables for credentials
    ///
    /// Uses environment variables (CLICKHOUSE_* preferred, CH_* as fallback):
    /// - CLICKHOUSE_HOST / CH_HOST (default: "localhost")
    /// - CLICKHOUSE_PORT / CH_PORT (default: 8123)
    /// - CLICKHOUSE_DATABASE / CH_DATABASE (default: "default")
    /// - CLICKHOUSE_USER / CH_USER (default: "default")
    /// - CLICKHOUSE_PASSWORD / CH_PASSWORD (optional)
    /// - VOLUME_UNIT (default: 100, e.g., hand = 100 shares)
    /// - AMOUNT_UNIT (default: 1000, e.g., 千元)
    pub fn from_env() -> Self {
        dotenv::dotenv().ok();
        let host = std::env::var("CLICKHOUSE_HOST")
            .or_else(|_| std::env::var("CH_HOST"))
            .unwrap_or_else(|_| "localhost".to_string());
        let port: u16 = std::env::var("CLICKHOUSE_PORT")
            .or_else(|_| std::env::var("CH_PORT"))
            .unwrap_or_else(|_| "8123".to_string())
            .parse()
            .unwrap_or(8123);
        let database = std::env::var("CLICKHOUSE_DATABASE")
            .or_else(|_| std::env::var("CH_DATABASE"))
            .unwrap_or_else(|_| "default".to_string());
        let username = std::env::var("CLICKHOUSE_USER")
            .or_else(|_| std::env::var("CH_USER"))
            .unwrap_or_else(|_| "default".to_string());
        let password = std::env::var("CLICKHOUSE_PASSWORD")
            .or_else(|_| std::env::var("CH_PASSWORD"))
            .ok();
        let volume_unit: u16 = std::env::var("VOLUME_UNIT")
            .unwrap_or_else(|_| "100".to_string())
            .parse()
            .unwrap_or(100);
        let amount_unit: u16 = std::env::var("AMOUNT_UNIT")
            .unwrap_or_else(|_| "1000".to_string())
            .parse()
            .unwrap_or(1000);

        ClickHouseSource {
            host,
            port,
            database,
            username,
            password,
            volume_unit,
            amount_unit,
            client: OnceLock::new(),
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
            amount_unit: 1000,
            client: OnceLock::new(),
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
            amount_unit: 1000,
            client: OnceLock::new(),
        }
    }

    /// Create a new ClickHouse source with custom volume/amount units
    pub fn with_units(
        host: &str,
        port: u16,
        database: &str,
        username: &str,
        volume_unit: u16,
        amount_unit: u16,
    ) -> Self {
        ClickHouseSource {
            host: host.to_string(),
            port,
            database: database.to_string(),
            username: username.to_string(),
            password: None,
            volume_unit,
            amount_unit,
            client: OnceLock::new(),
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

    /// Volume unit (shares per lot, default 100 for 手)
    pub fn volume_unit(&self) -> f64 {
        self.volume_unit as f64
    }

    /// Amount unit (yuan per unit, default 1000 for 千元)
    pub fn amount_unit(&self) -> f64 {
        self.amount_unit as f64
    }

    /// Build the ClickHouse HTTP URL
    fn build_url(&self) -> String {
        format!("http://{}:{}/", self.host, self.port)
    }

    /// Return a shared reqwest Client (lazily initialized, reused across queries)
    fn client(&self) -> &reqwest::blocking::Client {
        self.client.get_or_init(|| {
            reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("Failed to create reqwest client")
        })
    }

    /// Execute SQL and return raw Arrow IPC stream bytes
    ///
    /// Uses ArrowStream format (binary, columnar) for efficient data transfer.
    /// This is the primary query path — all data flows through Arrow.
    pub(crate) fn query_raw_arrow(&self, sql: &str) -> Result<Vec<u8>, DataError> {
        let base_url = format!("{}?database={}", self.build_url(), self.database);
        let url = format!("{}&default_format=ArrowStream", base_url);
        let client = self.client();

        let mut request = client.post(&url);
        if !self.username.is_empty() {
            request = request.query(&[("user", &self.username)]);
        }
        if let Some(ref password) = self.password {
            request = request.query(&[("password", password)]);
        }

        let response = request
            .body(sql.to_string())
            .send()
            .map_err(|e| DataError::Connection(format!("Failed to connect: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            return Err(DataError::Query(format!("HTTP {}: {}", status, text)));
        }

        response
            .bytes()
            .map(|b| b.to_vec())
            .map_err(|e| DataError::Query(format!("Failed to read response bytes: {}", e)))
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
        let source = ClickHouseSource::from_env();
        assert!(!source.host().is_empty());
    }

    #[test]
    fn test_clickhouse_source_from_env_with_values() {
        let source = ClickHouseSource::new("localhost", 8123, "default");
        assert_eq!(source.host(), "localhost");
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
