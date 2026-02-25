//! ClickHouse data provider implementation
//!
//! This module provides a ClickHouse HTTP-backed data provider for
//! accessing time series data from ClickHouse databases.

use crate::config::ClickhouseConfig;
use crate::data_provider::{DataProvider, Series};
use once_cell::sync::Lazy;
use reqwest;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tokio;

/// Use a single shared Tokio runtime for all ClickHouse HTTP calls
static TOKIO_RT: Lazy<tokio::runtime::Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build global tokio runtime")
});

/// ClickHouse HTTP-backed provider
pub struct ClickhouseProvider {
    /// Base URL of the ClickHouse server
    pub base_url: String,
    /// Optional basic-auth credentials (username, password)
    pub auth: Option<(String, String)>,
    /// Timeout in seconds for requests
    pub timeout_secs: u64,
}

impl ClickhouseProvider {
    /// Create a new ClickHouse provider
    pub fn new(base_url: &str) -> Self {
        ClickhouseProvider {
            base_url: base_url.trim_end_matches('/').to_string(),
            auth: None,
            timeout_secs: 10,
        }
    }
    
    /// Construct ClickHouseProvider from configuration
    pub fn from_config(cfg: &ClickhouseConfig) -> Result<Self, String> {
        if let Some(ref url) = cfg.url {
            let timeout = cfg.timeout_secs.unwrap_or(10);
            let auth = match (&cfg.user, &cfg.password) {
                (Some(u), Some(p)) => Some((u.clone(), p.clone())),
                _ => None,
            };
            Ok(ClickhouseProvider {
                base_url: url.trim_end_matches('/').to_string(),
                auth,
                timeout_secs: timeout,
            })
        } else {
            Err("clickhouse.url not set in config".to_string())
        }
    }
    
    /// Enumerate distinct (symbol, trading_date) pairs from a table
    pub fn enumerate_group_keys(&self, table: &str) -> Result<Vec<(String, String)>, String> {
        let sql = format!(
            "SELECT symbol, trading_date FROM {} GROUP BY symbol, trading_date ORDER BY symbol, trading_date",
            table
        );
        let rows = self.exec_query_json_each(&sql)?;
        let mut out = Vec::new();
        for v in rows {
            let sym = v
                .get("symbol")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            let td = v
                .get("trading_date")
                .and_then(|s| s.as_str())
                .map(|s| s.to_string());
            if let (Some(sy), Some(td)) = (sym, td) {
                out.push((sy, td));
            }
        }
        Ok(out)
    }
    
    /// Execute an arbitrary SQL query and return the response body as string
    pub fn exec_query_raw(&self, sql: &str) -> Result<String, String> {
        let base = self.base_url.clone();
        let sql = sql.to_string();
        let timeout = self.timeout_secs;
        let auth = self.auth.clone();
        
        TOKIO_RT.block_on(async move {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(timeout))
                .build()
                .map_err(|e| e.to_string())?;
            
            let req = if let Some((user, pass)) = auth {
                client.post(&base).basic_auth(user, Some(pass)).body(sql)
            } else {
                client.post(&base).body(sql)
            };
            
            let resp = req.send().await.map_err(|e| e.to_string())?;
            let status = resp.status();
            if status.is_success() {
                resp.text().await.map_err(|e| e.to_string())
            } else {
                Err(format!("clickhouse error: {}", status))
            }
        })
    }
    
    /// Execute a query with FORMAT JSONEachRow and parse each line as JSON
    fn exec_query_json_each(&self, sql: &str) -> Result<Vec<Value>, String> {
        let q = format!("{} FORMAT JSONEachRow", sql.trim_end_matches(';'));
        let s = self.exec_query_raw(&q)?;
        let mut out = Vec::new();
        for line in s.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(line)
                .map_err(|e| format!("json parse: {} on line: {}", e, line))?;
            out.push(v);
        }
        Ok(out)
    }
    
    /// Query a series of numeric values for a given symbol/trading_date and column
    pub fn get_series_for_group_from_table(
        &self,
        table: &str,
        symbol: &str,
        trading_date: &str,
        col: &str,
    ) -> Result<Vec<f64>, String> {
        let sql = format!(
            "SELECT {} FROM {} WHERE symbol='{}' AND trading_date='{}' ORDER BY trading_time",
            col, table, symbol, trading_date
        );
        let rows = self.exec_query_json_each(&sql)?;
        let mut out = Vec::new();
        for v in rows {
            if let Some(val) = v.get(col) {
                if val.is_null() {
                    continue;
                }
                let n = val
                    .as_f64()
                    .ok_or_else(|| format!("non-numeric value for {}: {}", col, val))?;
                out.push(n);
            }
        }
        Ok(out)
    }
    
    /// Run a scalar SQL that returns a single numeric value
    pub fn query_scalar_f64(&self, sql: &str) -> Result<f64, String> {
        let q = format!("{} FORMAT JSON", sql.trim_end_matches(';'));
        let s = self.exec_query_raw(&q)?;
        let v: Value = serde_json::from_str(&s).map_err(|e| format!("json parse: {}", e))?;
        if let Some(arr) = v.get("data").and_then(|d| d.as_array()) {
            if arr.is_empty() {
                return Err("no rows".to_string());
            }
            let first = &arr[0];
            if let Some((_k, val)) = first.as_object().and_then(|o| o.iter().next()) {
                return val.as_f64().ok_or_else(|| "not a number".to_string());
            }
        }
        Err("unexpected clickhouse json scalar format".to_string())
    }
}

impl DataProvider for ClickhouseProvider {
    fn name(&self) -> &str {
        "clickhouse"
    }
    
    fn get_scalar(&self, col: &str) -> Option<f64> {
        // Support legacy synthetic key for rolling: "rolling:{col}:{size}"
        if let Some(rest) = col.strip_prefix("rolling:") {
            let parts: Vec<&str> = rest.split(':').collect();
            if parts.len() == 2 {
                let _c = parts[0];
                if parts[1].parse::<usize>().is_ok() {
                    // no-op fallback for legacy synthetic rolling key
                }
            }
        }
        None
    }
    
    fn get_series(&self, _col: &str) -> Option<Series> {
        None
    }
    
    fn get_series_group(
        &self,
        table: &str,
        symbol: &str,
        trading_date: &str,
        col: &str,
    ) -> Option<Series> {
        match self.get_series_for_group_from_table(table, symbol, trading_date, col) {
            Ok(v) => Some(v),
            Err(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_clickhouse_provider_creation() {
        let provider = ClickhouseProvider::new("http://localhost:8123");
        assert_eq!(provider.base_url, "http://localhost:8123");
        assert!(provider.auth.is_none());
        assert_eq!(provider.timeout_secs, 10);
    }
    
    #[test]
    fn test_clickhouse_provider_from_config() {
        let config = ClickhouseConfig::new(Some("http://localhost:8123".to_string()));
        let provider = ClickhouseProvider::from_config(&config);
        assert!(provider.is_ok());
        
        let config = ClickhouseConfig::new(None);
        let provider = ClickhouseProvider::from_config(&config);
        assert!(provider.is_err());
    }
}