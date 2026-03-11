//! Data derivation module
//!
//! This module provides methods for computing derived fields from raw OHLCV data,
//! including holding returns, adjusted prices, and VWAP.

use std::collections::HashMap;

/// Data derivation for computing derived fields from raw OHLCV data
#[derive(Debug, Clone)]
pub struct DataDerivation;

impl DataDerivation {
    /// Create a new DataDerivation instance
    pub fn new() -> Self {
        Self
    }

    /// Compute holding returns (隔夜持仓收益)
    ///
    /// Holding return is calculated as:
    /// return[t] = (close[t+1] * adj[t+1] / latest_adj - close[t] * adj[t] / latest_adj)
    ///              / (close[t] * adj[t] / latest_adj)
    ///
    /// The last element will be 0.0 (no next day data available).
    ///
    /// # Parameters
    /// - close: Close price sequence (in chronological order)
    /// - adjust_factor: Optional adjustment factor sequence
    ///
    /// # Returns
    /// - Vector of holding returns with same length as input
    pub fn compute_holding_returns(
        &self,
        close: &[f64],
        adjust_factor: Option<&[f64]>,
    ) -> Vec<f64> {
        let n = close.len();
        if n == 0 {
            return Vec::new();
        }

        // Get the latest (most recent) adjustment factor
        let latest_adj = adjust_factor
            .and_then(|adj| adj.last().copied())
            .unwrap_or(1.0);

        // Ensure latest_adj is not zero to avoid division by zero
        let latest_adj = if latest_adj == 0.0 { 1.0 } else { latest_adj };

        let mut returns = vec![0.0; n];

        // Compute holding returns for each day except the last
        for i in 0..n - 1 {
            let curr_price = close[i];
            let next_price = close[i + 1];

            let curr_adj = adjust_factor
                .and_then(|adj| adj.get(i).copied())
                .unwrap_or(1.0);
            let next_adj = adjust_factor
                .and_then(|adj| adj.get(i + 1).copied())
                .unwrap_or(1.0);

            // Compute adjusted prices
            let curr_adj_price = curr_price * curr_adj / latest_adj;
            let next_adj_price = next_price * next_adj / latest_adj;

            // Compute holding return
            if curr_adj_price > 0.0 {
                returns[i] = (next_adj_price - curr_adj_price) / curr_adj_price;
            }
        }

        // Last element is 0.0 (no next day data)
        returns[n - 1] = 0.0;

        returns
    }

    /// Compute adjusted (复权) prices
    ///
    /// # Parameters
    /// - prices: Original price sequence
    /// - adjust_factor: Adjustment factor sequence
    ///
    /// # Returns
    /// - Vector of adjusted prices
    pub fn compute_adjusted_prices(&self, prices: &[f64], adjust_factor: &[f64]) -> Vec<f64> {
        if prices.is_empty() || adjust_factor.is_empty() {
            return Vec::new();
        }

        // Get the latest (most recent) adjustment factor
        let latest_adj = *adjust_factor.last().unwrap_or(&1.0);

        // Ensure latest_adj is not zero
        let latest_adj = if latest_adj == 0.0 { 1.0 } else { latest_adj };

        prices
            .iter()
            .zip(adjust_factor.iter())
            .map(|(price, adj)| price * adj / latest_adj)
            .collect()
    }

    /// Compute VWAP from amount and volume
    ///
    /// # Parameters
    /// - amount: Amount sequence
    /// - volume: Volume sequence
    /// - amount_unit: Unit for amount (e.g., 1000 for 千元, 10000 for 万元)
    /// - volume_unit: Unit for volume (e.g., 100 for 手 = 100 shares)
    ///
    /// # Returns
    /// - Vector of VWAP values (price per share)
    pub fn compute_vwap(
        &self,
        amount: &[f64],
        volume: &[f64],
        amount_unit: f64,
        volume_unit: f64,
    ) -> Vec<f64> {
        if amount.is_empty() || volume.is_empty() {
            return Vec::new();
        }

        amount
            .iter()
            .zip(volume.iter())
            .map(|(amt, vol)| {
                if *vol > 0.0 {
                    // amount is in units (e.g., 万元), volume is in units (e.g., 手)
                    // VWAP = amount * amount_unit / (volume * volume_unit)
                    amt * amount_unit / (vol * volume_unit)
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Compute simple returns (intraday return)
    ///
    /// Simple return = (close - open) / open
    ///
    /// # Parameters
    /// - close: Close price sequence
    /// - open: Open price sequence
    ///
    /// # Returns
    /// - Vector of simple returns
    pub fn compute_simple_returns(&self, close: &[f64], open: &[f64]) -> Vec<f64> {
        if close.is_empty() || open.is_empty() {
            return Vec::new();
        }

        close
            .iter()
            .zip(open.iter())
            .map(|(c, o)| if *o > 0.0 { (c - o) / o } else { f64::NAN })
            .collect()
    }

    /// Batch compute all derived fields for a single stock's data
    ///
    /// # Parameters
    /// - raw_data: HashMap containing raw OHLCV data with keys:
    ///   - "open", "high", "low", "close", "volume", "amount"
    ///   - "adjust_factor" (optional)
    /// - amount_unit: Unit for amount (e.g., 1000 for 千元)
    /// - volume_unit: Unit for volume (e.g., 100 for 手)
    ///
    /// # Returns
    /// - HashMap containing both original and derived fields
    pub fn derive_all(
        &self,
        raw_data: &HashMap<String, Vec<f64>>,
        amount_unit: f64,
        volume_unit: f64,
    ) -> HashMap<String, Vec<f64>> {
        let mut result = raw_data.clone();

        // Get raw close and volume for derivation
        let close = raw_data.get("close").map(|v| v.as_slice()).unwrap_or(&[]);
        let open = raw_data.get("open").map(|v| v.as_slice()).unwrap_or(&[]);
        let volume = raw_data.get("volume").map(|v| v.as_slice()).unwrap_or(&[]);
        let amount = raw_data.get("amount").map(|v| v.as_slice()).unwrap_or(&[]);
        let adjust_factor = raw_data.get("adjust_factor").map(|v| v.as_slice());

        // Compute holding returns (隔夜持仓收益)
        let holding_returns = self.compute_holding_returns(close, adjust_factor);
        if !holding_returns.is_empty() {
            result.insert("holding_return".to_string(), holding_returns);
        }

        // Compute simple returns (日内收益率, if not already present)
        if !result.contains_key("return") && !close.is_empty() && !open.is_empty() {
            let simple_returns = self.compute_simple_returns(close, open);
            if !simple_returns.is_empty() {
                result.insert("return".to_string(), simple_returns);
            }
        }

        // Compute adjusted close prices
        if let Some(adj) = adjust_factor {
            if !close.is_empty() && !adj.is_empty() {
                let adj_close = self.compute_adjusted_prices(close, adj);
                if !adj_close.is_empty() {
                    result.insert("close_adj".to_string(), adj_close);
                }
            }
        }

        // Compute VWAP (if not already present or we want to ensure correct calculation)
        if !amount.is_empty() && !volume.is_empty() {
            let vwap = self.compute_vwap(amount, volume, amount_unit, volume_unit);
            if !vwap.is_empty() {
                // Replace or add vwap
                result.insert("vwap".to_string(), vwap);
            }
        }

        // Compute adjusted VWAP if adjust_factor is available
        if let Some(adj) = adjust_factor {
            if !amount.is_empty() && !volume.is_empty() && !adj.is_empty() {
                let raw_vwap = self.compute_vwap(amount, volume, amount_unit, volume_unit);
                let adj_vwap = self.compute_adjusted_prices(&raw_vwap, adj);
                if !adj_vwap.is_empty() {
                    result.insert("vwap_adj".to_string(), adj_vwap);
                }
            }
        }

        result
    }
}

impl Default for DataDerivation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_holding_returns_simple() {
        let derivation = DataDerivation::new();

        // Simple case: no adjustment factor
        let close = vec![100.0, 105.0, 110.0];
        let result = derivation.compute_holding_returns(&close, None);

        assert_eq!(result.len(), 3);
        // return[0] = (105 - 100) / 100 = 0.05
        assert!((result[0] - 0.05).abs() < 1e-10);
        // return[1] = (110 - 105) / 105 = 0.0476...
        assert!((result[1] - 0.047619).abs() < 1e-6);
        // return[2] = 0.0 (no next day)
        assert!((result[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_holding_returns_with_adjustment() {
        let derivation = DataDerivation::new();

        // Case with adjustment factor
        let close = vec![100.0, 105.0, 110.0];
        let adjust_factor = vec![1.0, 1.0, 1.0];
        let result = derivation.compute_holding_returns(&close, Some(&adjust_factor));

        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_compute_holding_returns_with_real_adjustment() {
        let derivation = DataDerivation::new();

        // Real case: adjust_factor changes over time
        // Simulating a stock split where earlier prices need to be adjusted
        let close = vec![10.0, 10.5, 11.0];
        // Earlier adjustment factor is higher (pre-split)
        let adjust_factor = vec![2.0, 1.0, 1.0];

        let result = derivation.compute_holding_returns(&close, Some(&adjust_factor));

        assert_eq!(result.len(), 3);
        // Latest adj = 1.0
        // day 0: adj_price = 10 * 2 / 1 = 20, next_adj_price = 10.5 * 1 / 1 = 10.5
        // return = (10.5 - 20) / 20 = -0.475
        assert!((result[0] - (-0.475)).abs() < 1e-6);
    }

    #[test]
    fn test_compute_adjusted_prices() {
        let derivation = DataDerivation::new();

        let prices = vec![100.0, 105.0, 110.0];
        let adjust_factor = vec![1.0, 1.0, 1.0];

        let result = derivation.compute_adjusted_prices(&prices, &adjust_factor);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 100.0).abs() < 1e-10);
        assert!((result[1] - 105.0).abs() < 1e-10);
        assert!((result[2] - 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_adjusted_prices_with_split() {
        let derivation = DataDerivation::new();

        // Pre-split prices need to be adjusted up
        let prices = vec![10.0, 10.5, 11.0];
        let adjust_factor = vec![2.0, 1.0, 1.0]; // Earlier is pre-split

        let result = derivation.compute_adjusted_prices(&prices, &adjust_factor);

        assert_eq!(result.len(), 3);
        // latest_adj = 1.0
        // adj_price[0] = 10 * 2 / 1 = 20
        assert!((result[0] - 20.0).abs() < 1e-10);
        // adj_price[1] = 10.5 * 1 / 1 = 10.5
        assert!((result[1] - 10.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_vwap() {
        let derivation = DataDerivation::new();

        let amount = vec![10000.0, 20000.0, 15000.0]; // in 万元
        let volume = vec![100.0, 200.0, 150.0]; // in 手 (100 shares each)

        let result = derivation.compute_vwap(&amount, &volume, 10000.0, 100.0);

        assert_eq!(result.len(), 3);
        // VWAP = amount * amount_unit / (volume * volume_unit)
        // VWAP[0] = 10000 * 10000 / (100 * 100) = 100000000 / 10000 = 10000
        assert!((result[0] - 10000.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_simple_returns() {
        let derivation = DataDerivation::new();

        let close = vec![105.0, 110.0, 108.0];
        let open = vec![100.0, 105.0, 109.0];

        let result = derivation.compute_simple_returns(&close, &open);

        assert_eq!(result.len(), 3);
        // return[0] = (105 - 100) / 100 = 0.05
        assert!((result[0] - 0.05).abs() < 1e-10);
        // return[1] = (110 - 105) / 105 = 0.0476...
        assert!((result[1] - 0.047619).abs() < 1e-6);
        // return[2] = (108 - 109) / 109 = -0.00917...
        assert!((result[2] - (-0.009174)).abs() < 1e-6);
    }

    #[test]
    fn test_derive_all() {
        let derivation = DataDerivation::new();

        let mut raw_data = HashMap::new();
        raw_data.insert("symbol".to_string(), vec![1.0]);
        raw_data.insert("trading_date".to_string(), vec![1.0]);
        raw_data.insert("open".to_string(), vec![100.0]);
        raw_data.insert("high".to_string(), vec![110.0]);
        raw_data.insert("low".to_string(), vec![99.0]);
        raw_data.insert("close".to_string(), vec![105.0]);
        raw_data.insert("volume".to_string(), vec![1000.0]);
        raw_data.insert("amount".to_string(), vec![100000.0]);
        raw_data.insert("adjust_factor".to_string(), vec![1.0]);

        let result = derivation.derive_all(&raw_data, 10000.0, 100.0);

        // Should have original fields plus derived ones
        assert!(result.contains_key("holding_return"));
        assert!(result.contains_key("return"));
        assert!(result.contains_key("close_adj"));
        assert!(result.contains_key("vwap"));
        assert!(result.contains_key("vwap_adj"));
    }

    #[test]
    fn test_empty_input() {
        let derivation = DataDerivation::new();

        let close: Vec<f64> = vec![];
        let result = derivation.compute_holding_returns(&close, None);
        assert!(result.is_empty());

        let prices: Vec<f64> = vec![];
        let adj: Vec<f64> = vec![];
        let result = derivation.compute_adjusted_prices(&prices, &adj);
        assert!(result.is_empty());
    }
}
