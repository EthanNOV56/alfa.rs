//! Data provider abstraction for accessing scalar values and time series
//!
//! This module provides a trait for accessing data from various sources
//! (in-memory mocks, databases, etc.) and implementations for common providers.

use std::collections::HashMap;

/// A time-ordered series of numeric values
pub type Series = Vec<f64>;

/// Trait for data providers that can supply scalar values and time series
pub trait DataProvider: Send + Sync {
    /// Get the name of this provider
    fn name(&self) -> &str;
    
    /// Get a scalar value by column name
    fn get_scalar(&self, col: &str) -> Option<f64>;
    
    /// Get a time series by column name
    fn get_series(&self, col: &str) -> Option<Series>;
    
    /// Get a time series for a specific group (table, symbol, trading date)
    fn get_series_group(
        &self,
        table: &str,
        symbol: &str,
        trading_date: &str,
        col: &str,
    ) -> Option<Series>;
}

/// In-memory mock data provider for testing
pub struct MockProvider {
    scalars: HashMap<String, f64>,
    series: HashMap<String, Series>,
}

impl MockProvider {
    /// Create a new mock provider with scalar values only
    pub fn new(scalars: HashMap<String, f64>) -> Self {
        MockProvider {
            scalars,
            series: HashMap::new(),
        }
    }
    
    /// Create a new mock provider with both scalars and series
    pub fn new_with_series(
        scalars: HashMap<String, f64>,
        series: HashMap<String, Series>,
    ) -> Self {
        MockProvider { scalars, series }
    }
    
    /// Add a scalar value
    pub fn add_scalar(&mut self, key: String, value: f64) {
        self.scalars.insert(key, value);
    }
    
    /// Add a time series
    pub fn add_series(&mut self, key: String, series: Series) {
        self.series.insert(key, series);
    }
    
    /// Add a grouped series (table:symbol:date:column format)
    pub fn add_group_series(
        &mut self,
        table: &str,
        symbol: &str,
        trading_date: &str,
        col: &str,
        series: Series,
    ) {
        let key = format!("{}:{}:{}:{}", table, symbol, trading_date, col);
        self.series.insert(key, series);
    }
}

impl DataProvider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }
    
    fn get_scalar(&self, col: &str) -> Option<f64> {
        self.scalars.get(col).copied()
    }
    
    fn get_series(&self, col: &str) -> Option<Series> {
        self.series.get(col).cloned()
    }
    
    fn get_series_group(
        &self,
        table: &str,
        symbol: &str,
        trading_date: &str,
        col: &str,
    ) -> Option<Series> {
        let key = format!("{}:{}:{}:{}", table, symbol, trading_date, col);
        self.series.get(&key).cloned()
    }
}

/// A simple provider that always returns None (useful for testing)
pub struct NullProvider;

impl NullProvider {
    pub fn new() -> Self {
        NullProvider
    }
}

impl DataProvider for NullProvider {
    fn name(&self) -> &str {
        "null"
    }
    
    fn get_scalar(&self, _col: &str) -> Option<f64> {
        None
    }
    
    fn get_series(&self, _col: &str) -> Option<Series> {
        None
    }
    
    fn get_series_group(
        &self,
        _table: &str,
        _symbol: &str,
        _trading_date: &str,
        _col: &str,
    ) -> Option<Series> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mock_provider_scalar() {
        let mut scalars = HashMap::new();
        scalars.insert("price".to_string(), 100.0);
        scalars.insert("volume".to_string(), 1000.0);
        
        let provider = MockProvider::new(scalars);
        assert_eq!(provider.name(), "mock");
        assert_eq!(provider.get_scalar("price"), Some(100.0));
        assert_eq!(provider.get_scalar("volume"), Some(1000.0));
        assert_eq!(provider.get_scalar("unknown"), None);
    }
    
    #[test]
    fn test_mock_provider_series() {
        let scalars = HashMap::new();
        let mut series = HashMap::new();
        series.insert("prices".to_string(), vec![1.0, 2.0, 3.0]);
        
        let provider = MockProvider::new_with_series(scalars, series);
        assert_eq!(provider.get_series("prices"), Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(provider.get_series("unknown"), None);
    }
    
    #[test]
    fn test_mock_provider_group_series() {
        let mut provider = MockProvider::new(HashMap::new());
        provider.add_group_series(
            "stock_5m",
            "AAPL",
            "20230101",
            "close",
            vec![150.0, 151.0, 152.0],
        );
        
        let series = provider.get_series_group("stock_5m", "AAPL", "20230101", "close");
        assert_eq!(series, Some(vec![150.0, 151.0, 152.0]));
        
        // Wrong group should return None
        let series = provider.get_series_group("stock_5m", "AAPL", "20230102", "close");
        assert_eq!(series, None);
    }
    
    #[test]
    fn test_null_provider() {
        let provider = NullProvider::new();
        assert_eq!(provider.name(), "null");
        assert_eq!(provider.get_scalar("anything"), None);
        assert_eq!(provider.get_series("anything"), None);
    }
}