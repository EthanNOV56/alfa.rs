//! Configuration for the alpha expression system

#[cfg(feature = "config")]
use serde::Deserialize;

/// ClickHouse database configuration (optional)
#[cfg_attr(feature = "config", derive(Debug, Deserialize, Clone))]
#[derive(Debug, Clone)]
pub struct ClickhouseConfig {
    /// ClickHouse server URL
    pub url: Option<String>,
    /// Username for authentication
    pub user: Option<String>,
    /// Password for authentication
    pub password: Option<String>,
    /// Query timeout in seconds
    pub timeout_secs: Option<u64>,
}

impl ClickhouseConfig {
    /// Create a new ClickHouse configuration
    pub fn new(url: Option<String>) -> Self {
        Self {
            url,
            user: None,
            password: None,
            timeout_secs: None,
        }
    }
}

/// Main configuration structure
#[cfg_attr(feature = "config", derive(Debug, Deserialize, Clone))]
#[derive(Debug, Clone)]
pub struct Config {
    /// ClickHouse configuration (optional)
    pub clickhouse: Option<ClickhouseConfig>,
    /// Cache capacity for expression results
    pub cache_capacity: Option<usize>,
    /// Number of cache shards for concurrent access
    pub cache_shards: Option<usize>,
}

impl Config {
    /// Create a default configuration
    pub fn default() -> Self {
        Self {
            clickhouse: None,
            cache_capacity: Some(64),
            cache_shards: None,
        }
    }
    
    /// Load configuration from a TOML file (requires "config" feature)
    #[cfg(feature = "config")]
    pub fn from_path(path: &str) -> Result<Self, String> {
        use std::fs;
        let s = fs::read_to_string(path).map_err(|e| format!("read config: {}", e))?;
        toml::from_str(&s).map_err(|e| format!("toml parse: {}", e))
    }
    
    /// Load configuration from a TOML string (requires "config" feature)
    #[cfg(feature = "config")]
    pub fn from_toml(toml_str: &str) -> Result<Self, String> {
        toml::from_str(toml_str).map_err(|e| format!("toml parse: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.clickhouse.is_none());
        assert_eq!(config.cache_capacity, Some(64));
        assert!(config.cache_shards.is_none());
    }
    
    #[test]
    fn test_clickhouse_config() {
        let ch_config = ClickhouseConfig::new(Some("http://localhost:8123".to_string()));
        assert_eq!(ch_config.url, Some("http://localhost:8123".to_string()));
        assert!(ch_config.user.is_none());
        assert!(ch_config.password.is_none());
        assert!(ch_config.timeout_secs.is_none());
    }
}