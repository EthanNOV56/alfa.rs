//! Simple backtest demonstration
//! 
//! This example verifies that the backtest engine works correctly
//! with simulated factor and return data.

use alpha_expr::backtest::{BacktestEngine, WeightMethod, quantile_backtest};
use ndarray::Array2;
use rand::Rng;

fn main() -> Result<(), String> {
    println!("=== Simple Backtest Demo ===\n");
    
    // Create simple test data
    let n_days = 100;
    let n_assets = 50;
    
    // Create factor matrix: random values with some predictive power
    let mut factor = Array2::<f64>::zeros((n_days, n_assets));
    let mut returns = Array2::<f64>::zeros((n_days, n_assets));
    
    let mut rng = rand::thread_rng();
    
    for asset in 0..n_assets {
        // Generate factor values (random walk)
        let mut f = rng.gen_range(-1.0..1.0);
        for day in 0..n_days {
            factor[[day, asset]] = f;
            // Add some persistence
            f = 0.7 * f + 0.3 * rng.gen_range(-1.0..1.0);
            
            // Generate returns with 5% factor predictive power
            returns[[day, asset]] = 0.05 * factor[[day, asset]] + 0.01 * rng.gen_range(-1.0..1.0);
        }
    }
    
    // Add some NaN values
    for day in 0..n_days {
        for asset in 0..n_assets {
            if rng.gen_bool(0.05) {
                factor[[day, asset]] = f64::NAN;
            }
            if rng.gen_bool(0.05) {
                returns[[day, asset]] = f64::NAN;
            }
        }
    }
    
    println!("Data shape: {} days × {} assets", n_days, n_assets);
    
    // Run backtest
    println!("\nRunning quantile backtest...");
    
    let result = quantile_backtest(
        factor.slice(ndarray::s![..n_days-1, ..]).to_owned(),
        returns.slice(ndarray::s![..n_days-1, ..]).to_owned(),
        10,
        WeightMethod::Equal,
        1,
        1,
        0.0003,
        None,
    )?;
    
    println!("\nResults:");
    println!("  Long-short cumulative return: {:.4}%", result.long_short_cum_return * 100.0);
    println!("  IC mean: {:.4}", result.ic_mean);
    println!("  IC IR: {:.4}", result.ic_ir);
    
    // Check group returns
    if let Some(group_means) = result.group_returns.mean_axis(ndarray::Axis(0)) {
        println!("\nGroup mean returns:");
        for (i, mean) in group_means.iter().enumerate() {
            println!("  Group {}: {:+.6}%", i + 1, mean * 100.0);
        }
        
        // Check monotonicity (should be roughly increasing)
        let mut increasing = true;
        for i in 1..group_means.len() {
            if group_means[i] < group_means[i-1] - 0.0001 {
                increasing = false;
                break;
            }
        }
        if increasing {
            println!("\n✓ Returns increase from low to high factor groups (as expected)");
        } else {
            println!("\n⚠ Returns not monotonically increasing (random data may cause this)");
        }
    }
    
    // Test with different configurations
    println!("\nTesting different configurations:");
    
    let mut engine = BacktestEngine::new(
        factor.slice(ndarray::s![..n_days-1, ..]).to_owned(),
        returns.slice(ndarray::s![..n_days-1, ..]).to_owned(),
    );
    engine
        .set_quantiles(5)
        .set_long_top_n(2)
        .set_short_top_n(2);
    
    match engine.run() {
        Ok(result) => {
            println!("  Quintile (5 groups): {:.4}% cumulative", result.long_short_cum_return * 100.0);
        }
        Err(e) => println!("  Error: {}", e),
    }
    
    println!("\n=== Demo Complete ===");
    Ok(())
}