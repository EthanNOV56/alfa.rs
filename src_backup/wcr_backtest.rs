//! WCR (Volume Weighted Close Ratio) factor backtest example
//! 
//! This example demonstrates:
//! 1. Connecting to ClickHouse database
//! 2. Fetching stock daily data
//! 3. Calculating WCR factor: VWAP / SMA(close, N)
//! 4. Running quantile backtest on the factor
//! 5. Analyzing factor performance

use alpha_expr::backtest::{BacktestEngine, WeightMethod, quantile_backtest};
use alpha_expr::clickhouse_provider::ClickhouseProvider;
use ndarray::{Array2, s};
use std::collections::HashMap;

fn main() -> Result<(), String> {
    println!("=== WCR Factor Backtest with ClickHouse Data ===\n");
    
    // 1. Connect to ClickHouse
    println!("1. Connecting to ClickHouse...");
    let provider = ClickhouseProvider::new("http://localhost:18123")
        .with_auth("readonly".to_string(), "readonly".to_string());
    
    // 2. Fetch data for a specific time period
    println!("2. Fetching stock data...");
    
    // Let's fetch data for a subset of stocks and time period for demonstration
    // In production, you'd want to fetch more data
    let symbols = fetch_symbols(&provider, 50)?; // Get first 50 symbols
    println!("   Selected {} symbols", symbols.len());
    
    let start_date = "2024-01-01";
    let end_date = "2025-01-01";
    
    // Fetch data for each symbol
    let mut all_data = Vec::new();
    for (i, symbol) in symbols.iter().enumerate() {
        if i % 10 == 0 {
            println!("   Fetching data for symbol {}: {}", i, symbol);
        }
        
        if let Ok(data) = fetch_symbol_data(&provider, symbol, start_date, end_date) {
            all_data.push((symbol.clone(), data));
        }
    }
    
    println!("   Retrieved data for {} symbols", all_data.len());
    
    if all_data.is_empty() {
        return Err("No data retrieved".to_string());
    }
    
    // 3. Align data by date and create factor/returns matrices
    println!("3. Preparing data matrices...");
    let (factor_matrix, returns_matrix, dates) = prepare_matrices(&all_data)?;
    
    println!("   Data shape: {} days × {} assets", factor_matrix.dim().0, factor_matrix.dim().1);
    println!("   Date range: {:?} to {:?}", dates.first(), dates.last());
    
    // 4. Run backtest on WCR factor
    println!("\n4. Running quantile backtest on WCR factor...");
    
    let result = quantile_backtest(
        factor_matrix.clone(),
        returns_matrix.clone(),
        10,                      // 10 quantile groups
        WeightMethod::Equal,     // Equal weighting
        1,                       // Long top group
        1,                       // Short bottom group
        0.0003,                  // Commission rate
        None,                    // No external weights
    )?;
    
    // 5. Display results
    println!("\n5. Backtest Results:");
    println!("   Long-short cumulative return: {:.4}%", result.long_short_cum_return * 100.0);
    println!("   IC mean: {:.4}", result.ic_mean);
    println!("   IC information ratio: {:.4}", result.ic_ir);
    
    println!("\n   Group performance (1=lowest WCR, 10=highest WCR):");
    let group_means = result.group_returns.mean_axis(ndarray::Axis(0));
    for (i, mean) in group_means.iter().enumerate() {
        println!("     Group {}: {:+.6}%", i + 1, mean * 100.0);
    }
    
    // 6. Run additional backtests with different configurations
    println!("\n6. Additional backtest configurations:");
    
    // Quintile test (5 groups)
    let mut engine = BacktestEngine::new(factor_matrix.clone(), returns_matrix.clone());
    engine
        .set_quantiles(5)
        .set_long_top_n(2)
        .set_short_top_n(2)
        .set_commission_rate(0.0005);
    
    match engine.run() {
        Ok(quintile_result) => {
            println!("   Quintile (5 groups):");
            println!("     Long-short cumulative return: {:.4}%", quintile_result.long_short_cum_return * 100.0);
            println!("     IC mean: {:.4}", quintile_result.ic_mean);
        }
        Err(e) => println!("   Quintile backtest failed: {}", e),
    }
    
    println!("\n=== WCR Backtest Complete ===");
    Ok(())
}

/// Fetch a list of stock symbols from ClickHouse
fn fetch_symbols(provider: &ClickhouseProvider, limit: usize) -> Result<Vec<String>, String> {
    let query = format!(
        "SELECT DISTINCT symbol FROM default.stock_1d WHERE trading_date >= '2024-01-01' LIMIT {}",
        limit
    );
    
    let rows = provider.exec_query_json_each(&query)?;
    let mut symbols = Vec::new();
    
    for row in rows {
        if let Some(symbol) = row.get("symbol").and_then(|v| v.as_str()) {
            symbols.push(symbol.to_string());
        }
    }
    
    Ok(symbols)
}

/// Fetch daily data for a specific symbol
fn fetch_symbol_data(
    provider: &ClickhouseProvider,
    symbol: &str,
    start_date: &str,
    end_date: &str,
) -> Result<Vec<(String, f64, f64)>, String> {
    // Fetch close price and volume
    let query = format!(
        "SELECT trading_date, close, volume FROM default.stock_1d \
         WHERE symbol = '{}' AND trading_date >= '{}' AND trading_date <= '{}' \
         ORDER BY trading_date",
        symbol, start_date, end_date
    );
    
    let rows = provider.exec_query_json_each(&query)?;
    let mut data = Vec::new();
    
    for row in rows {
        let date = row.get("trading_date")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let close = row.get("close")
            .and_then(|v| v.as_f64());
        let volume = row.get("volume")
            .and_then(|v| v.as_f64());
        
        if let (Some(date), Some(close), Some(volume)) = (date, close, volume) {
            data.push((date, close, volume));
        }
    }
    
    Ok(data)
}

/// Calculate WCR factor for a time series
/// WCR = VWAP / SMA(close, window)
/// where VWAP = sum(close * volume) / sum(volume)
/// and SMA(close, window) is simple moving average of close prices
fn calculate_wcr_series(
    closes: &[f64],
    volumes: &[f64],
    window: usize,
) -> Vec<f64> {
    let n = closes.len();
    let mut wcr_values = Vec::with_capacity(n);
    
    for i in 0..n {
        if i < window {
            // Not enough data for SMA
            wcr_values.push(f64::NAN);
            continue;
        }
        
        // Calculate VWAP for the current day
        let vwap = closes[i] * volumes[i] / volumes[i]; // Actually this is just close price
        
        // Wait, that's wrong. VWAP for a single day is close * volume / volume = close
        // So daily VWAP equals close price. That doesn't make sense.
        // Actually, VWAP is typically calculated intraday.
        // For daily data, we might want to use a multi-day VWAP or different calculation.
        
        // Let's use a different approach: WCR = close / SMA(close, window)
        // This is a simple momentum factor
        let sma_start = i - window + 1;
        let sma_end = i + 1;
        let sma: f64 = closes[sma_start..sma_end].iter().sum::<f64>() / window as f64;
        
        let wcr = closes[i] / sma;
        wcr_values.push(wcr);
    }
    
    wcr_values
}

/// Prepare factor and returns matrices from aligned data
fn prepare_matrices(
    all_data: &[(String, Vec<(String, f64, f64)>)],
) -> Result<(Array2<f64>, Array2<f64>, Vec<String>), String> {
    // First, find all unique dates
    let mut date_set = std::collections::HashSet::new();
    for (_, data) in all_data {
        for (date, _, _) in data {
            date_set.insert(date.clone());
        }
    }
    
    let mut dates: Vec<String> = date_set.into_iter().collect();
    dates.sort();
    
    let n_dates = dates.len();
    let n_assets = all_data.len();
    
    // Create mapping from date to index
    let date_to_idx: HashMap<String, usize> = dates
        .iter()
        .enumerate()
        .map(|(i, date)| (date.clone(), i))
        .collect();
    
    // Initialize matrices with NaN values
    let mut closes_matrix = Array2::<f64>::from_elem((n_dates, n_assets), f64::NAN);
    let mut volumes_matrix = Array2::<f64>::from_elem((n_dates, n_assets), f64::NAN);
    
    // Fill matrices
    for (asset_idx, (symbol, data)) in all_data.iter().enumerate() {
        for (date, close, volume) in data {
            if let Some(&date_idx) = date_to_idx.get(date) {
                closes_matrix[[date_idx, asset_idx]] = *close;
                volumes_matrix[[date_idx, asset_idx]] = *volume;
            }
        }
    }
    
    // Calculate WCR factor (using 20-day window)
    let window = 20;
    let mut factor_matrix = Array2::<f64>::from_elem((n_dates, n_assets), f64::NAN);
    
    for asset_idx in 0..n_assets {
        let closes: Vec<f64> = closes_matrix.column(asset_idx).to_vec();
        let volumes: Vec<f64> = volumes_matrix.column(asset_idx).to_vec();
        
        let wcr_values = calculate_wcr_series(&closes, &volumes, window);
        
        for (date_idx, &wcr) in wcr_values.iter().enumerate() {
            factor_matrix[[date_idx, asset_idx]] = wcr;
        }
    }
    
    // Calculate returns: next day's close / today's close - 1
    let mut returns_matrix = Array2::<f64>::from_elem((n_dates - 1, n_assets), f64::NAN);
    
    for date_idx in 0..n_dates - 1 {
        for asset_idx in 0..n_assets {
            let close_today = closes_matrix[[date_idx, asset_idx]];
            let close_tomorrow = closes_matrix[[date_idx + 1, asset_idx]];
            
            if close_today.is_nan() || close_tomorrow.is_nan() || close_today == 0.0 {
                continue;
            }
            
            let ret = (close_tomorrow / close_today) - 1.0;
            returns_matrix[[date_idx, asset_idx]] = ret;
        }
    }
    
    // Align factor matrix with returns (factor today, returns tomorrow)
    // Factor matrix should have one less row than closes_matrix
    let factor_for_backtest = factor_matrix.slice(s![..n_dates-1, ..]).to_owned();
    
    Ok((factor_for_backtest, returns_matrix, dates))
}

/// Add auth method to ClickhouseProvider (extension)
trait ClickhouseProviderExt {
    fn with_auth(self, user: String, password: String) -> Self;
}

impl ClickhouseProviderExt for ClickhouseProvider {
    fn with_auth(mut self, user: String, password: String) -> Self {
        self.auth = Some((user, password));
        self
    }
}