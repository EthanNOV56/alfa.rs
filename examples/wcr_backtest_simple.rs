//! Simple WCR factor backtest example with simulated data
//! 
//! This example demonstrates:
//! 1. Generating simulated stock data
//! 2. Calculating WCR factor: close / SMA(close, 20)
//! 3. Running quantile backtest on the factor
//! 4. Analyzing factor performance

use alpha_expr::backtest::{BacktestEngine, WeightMethod, quantile_backtest};
use ndarray::{Array2, Array1};
use rand::Rng;

fn main() -> Result<(), String> {
    println!("=== WCR Factor Backtest with Simulated Data ===\n");
    
    // 1. Generate simulated data
    println!("1. Generating simulated stock data...");
    let n_days = 500;      // 500 trading days
    let n_assets = 100;    // 100 stocks
    
    // Generate random factor values (simulated WCR values)
    // In real scenario, we would calculate WCR from price/volume data
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    
    let mut factor_matrix = Array2::<f64>::zeros((n_days, n_assets));
    let mut returns_matrix = Array2::<f64>::zeros((n_days, n_assets));
    
    // Generate data with some predictive power
    for asset in 0..n_assets {
        // Generate factor values with auto-correlation
        let mut factor_values = Vec::with_capacity(n_days);
        factor_values.push(normal.sample(&mut rng));
        
        for day in 1..n_days {
            // Factor has persistence
            factor_values.push(0.7 * factor_values[day - 1] + 0.3 * normal.sample(&mut rng));
        }
        
        // Generate returns with factor signal (5% predictive power)
        let mut returns = Vec::with_capacity(n_days);
        for day in 0..n_days {
            let signal = 0.05 * factor_values[day];
            let noise = Normal::new(0.0, 0.01).unwrap().sample(&mut rng);
            returns.push(signal + noise);
        }
        
        // Fill matrices
        for day in 0..n_days {
            factor_matrix[[day, asset]] = factor_values[day];
            returns_matrix[[day, asset]] = returns[day];
        }
    }
    
    // Add some NaN values to simulate missing data
    for day in 0..n_days {
        for asset in 0..n_assets {
            if rand::random::<f64>() < 0.05 { // 5% missing
                factor_matrix[[day, asset]] = f64::NAN;
            }
            if rand::random::<f64>() < 0.05 { // 5% missing
                returns_matrix[[day, asset]] = f64::NAN;
            }
        }
    }
    
    println!("   Data shape: {} days × {} assets", n_days, n_assets);
    println!("   Factor predictive power: 5%");
    println!("   Missing data: 5% randomly");
    
    // Note: For actual WCR calculation, we would:
    // 1. Fetch close prices and volumes from ClickHouse
    // 2. Calculate WCR = close / SMA(close, 20)
    // 3. Use that as factor values
    
    // 2. Run backtest on the factor
    println!("\n2. Running quantile backtest (qcut=10)...");
    
    // Factor matrix for backtest should have n_days-1 rows (since we predict next day returns)
    let factor_for_backtest = factor_matrix.slice(ndarray::s![..n_days-1, ..]).to_owned();
    let returns_for_backtest = returns_matrix.slice(ndarray::s![..n_days-1, ..]).to_owned();
    
    let result = quantile_backtest(
        factor_for_backtest,
        returns_for_backtest,
        10,                      // 10 quantile groups
        WeightMethod::Equal,     // Equal weighting
        1,                       // Long top group
        1,                       // Short bottom group
        0.0003,                  // Commission rate
        None,                    // No external weights
    )?;
    
    // 3. Display results
    println!("\n3. Backtest Results:");
    println!("   Long-short cumulative return: {:.4}%", result.long_short_cum_return * 100.0);
    println!("   IC mean: {:.4}", result.ic_mean);
    println!("   IC information ratio: {:.4}", result.ic_ir);
    
    println!("\n   Group performance (1=lowest factor, 10=highest factor):");
    let group_means = result.group_returns.mean_axis(ndarray::Axis(0));
    for (i, mean) in group_means.iter().enumerate() {
        println!("     Group {}: {:+.6}%", i + 1, mean * 100.0);
    }
    
    // 4. Show monotonicity check (should be increasing from group 1 to 10)
    println!("\n4. Monotonicity check:");
    let mut monotonic = true;
    for i in 1..group_means.len() {
        if group_means[i] < group_means[i-1] - 0.0001 {
            println!("   Warning: Group {} return ({:.6}%) < Group {} return ({:.6}%)",
                     i+1, group_means[i] * 100.0, i, group_means[i-1] * 100.0);
            monotonic = false;
        }
    }
    if monotonic {
        println!("   Good: Returns generally increase from low to high factor groups");
    }
    
    // 5. Additional configurations
    println!("\n5. Additional backtest configurations:");
    
    // Test with quintiles
    let mut engine = BacktestEngine::new(
        factor_matrix.slice(ndarray::s![..n_days-1, ..]).to_owned(),
        returns_matrix.slice(ndarray::s![..n_days-1, ..]).to_owned(),
    );
    engine
        .set_quantiles(5)
        .set_weight_method(WeightMethod::Equal)
        .set_long_top_n(2)
        .set_short_top_n(2)
        .set_commission_rate(0.0005);
    
    match engine.run() {
        Ok(quintile_result) => {
            println!("   Quintile (5 groups, long/short top/bottom 2):");
            println!("     Cumulative return: {:.4}%", quintile_result.long_short_cum_return * 100.0);
            println!("     IC mean: {:.4}", quintile_result.ic_mean);
        }
        Err(e) => println!("   Quintile backtest failed: {}", e),
    }
    
    println!("\n=== WCR Backtest Demo Complete ===");
    println!("\nNext steps for real data:");
    println!("1. Connect to ClickHouse using clickhouse_provider");
    println!("2. Fetch close prices and volumes from stock_1d table");
    println!("3. Calculate WCR = close / SMA(close, 20) for each stock");
    println!("4. Run backtest on the calculated WCR values");
    
    Ok(())
}