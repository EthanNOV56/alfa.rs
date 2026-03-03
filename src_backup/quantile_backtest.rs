//! Quantile backtest example for alpha factor evaluation
//!
//! This example demonstrates how to use the backtesting module
//! to evaluate alpha factors using qcut(N) grouping and long-short portfolios.

use alpha_expr::backtest::{BacktestEngine, WeightMethod, quantile_backtest};
use ndarray::{Array2, Array1};
use rand::distributions::{Distribution, Normal};

fn main() -> Result<(), String> {
    println!("=== Alpha Expression System: Quantile Backtest Demo ===\n");
    
    // 1. Generate synthetic factor and return data
    println!("1. Generating synthetic data...");
    let (factor, returns) = generate_sample_data(100, 200, 0.3, 0.05);
    println!("   Data shape: {} days × {} assets", factor.dim().0, factor.dim().1);
    println!("   Signal strength: 0.05 (5% factor predictive power)");
    
    // 2. Run basic quantile backtest
    println!("\n2. Running quantile backtest (qcut=10)...");
    let result = quantile_backtest(
        factor.clone(),
        returns.clone(),
        10,
        WeightMethod::Equal,
        1,  // long top group
        1,  // short bottom group
        0.0003,  // commission rate
        None,  // no external weights
    )?;
    
    println!("   Long-short cumulative return: {:.4}%", result.long_short_cum_return * 100.0);
    println!("   IC mean: {:.4}", result.ic_mean);
    println!("   IC information ratio: {:.4}", result.ic_ir);
    
    // 3. Display group performance
    println!("\n3. Group performance (1=lowest, 10=highest factor values):");
    let group_means = result.group_returns.mean_axis(ndarray::Axis(0));
    for (i, mean) in group_means.iter().enumerate() {
        println!("   Group {}: {:+.6}%", i + 1, mean * 100.0);
    }
    
    // 4. Use BacktestEngine for more control
    println!("\n4. Using BacktestEngine for advanced configuration...");
    let mut engine = BacktestEngine::new(factor.clone(), returns.clone());
    engine
        .set_quantiles(5)  // quintile groups
        .set_weight_method(WeightMethod::Equal)
        .set_long_top_n(2)  // long top 2 groups
        .set_short_top_n(2)  // short bottom 2 groups
        .set_commission_rate(0.0005);
    
    let engine_result = engine.run()?;
    
    println!("   Quintile backtest results:");
    println!("   - Long-short cumulative return: {:.4}%", engine_result.long_short_cum_return * 100.0);
    println!("   - IC mean: {:.4}", engine_result.ic_mean);
    println!("   - Number of groups: {}", engine_result.group_returns.dim().1);
    
    // 5. Demonstrate weighted backtest (with mock market cap data)
    println!("\n5. Running weighted backtest (simulated market cap weighting)...");
    let market_cap = generate_market_cap_weights(factor.dim().0, factor.dim().1);
    
    let mut weighted_engine = BacktestEngine::new(factor, returns);
    weighted_engine
        .set_quantiles(10)
        .set_weight_method(WeightMethod::Weighted)
        .set_weights(market_cap)
        .set_long_top_n(1)
        .set_short_top_n(1);
    
    match weighted_engine.run() {
        Ok(weighted_result) => {
            println!("   Weighted backtest cumulative return: {:.4}%", 
                     weighted_result.long_short_cum_return * 100.0);
            println!("   Compare with equal weight: {:.4}%", 
                     result.long_short_cum_return * 100.0);
        }
        Err(e) => println!("   Weighted backtest failed: {}", e),
    }
    
    // 6. Show how to integrate with alpha expressions
    println!("\n6. Integration with alpha expression system:");
    println!("   (In practice, you would:");
    println!("    1. Compute alpha factors using alpha_expr::alpha::AlphaBuilder");
    println!("    2. Evaluate factors on historical data");
    println!("    3. Pass factor values to BacktestEngine");
    println!("    4. Analyze performance with BacktestResult)");
    
    println!("\n=== Backtest Demo Complete ===");
    Ok(())
}

/// Generate synthetic factor and return data
fn generate_sample_data(
    n_days: usize,
    n_assets: usize,
    factor_vol: f64,
    signal_strength: f64,
) -> (Array2<f64>, Array2<f64>) {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, factor_vol).unwrap();
    
    // Generate factor values (random walk with auto-correlation)
    let mut factor = Array2::<f64>::zeros((n_days, n_assets));
    for day in 0..n_days {
        for asset in 0..n_assets {
            if day == 0 {
                factor[[day, asset]] = normal.sample(&mut rng);
            } else {
                // Add persistence to factor
                factor[[day, asset]] = 0.7 * factor[[day - 1, asset]] + 0.3 * normal.sample(&mut rng);
            }
        }
    }
    
    // Generate returns with factor signal and noise
    let mut returns = Array2::<f64>::zeros((n_days, n_assets));
    for day in 0..n_days {
        for asset in 0..n_assets {
            let noise = Normal::new(0.0, 0.01).unwrap().sample(&mut rng);
            returns[[day, asset]] = signal_strength * factor[[day, asset]] + noise;
        }
    }
    
    // Add some NaN values to simulate missing data (10%)
    for day in 0..n_days {
        for asset in 0..n_assets {
            if rand::random::<f64>() < 0.1 {
                factor[[day, asset]] = f64::NAN;
            }
            if rand::random::<f64>() < 0.1 {
                returns[[day, asset]] = f64::NAN;
            }
        }
    }
    
    (factor, returns)
}

/// Generate mock market cap weights (log-normal distribution)
fn generate_market_cap_weights(n_days: usize, n_assets: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let log_normal = Normal::new(0.0, 1.0).unwrap();
    
    let mut weights = Array2::<f64>::zeros((n_days, n_assets));
    
    // Generate base market caps (constant over time for simplicity)
    let mut base_caps = Vec::new();
    for _ in 0..n_assets {
        let cap = log_normal.sample(&mut rng).exp(); // log-normal distribution
        base_caps.push(cap);
    }
    
    // Normalize and apply small random variations over time
    let total_cap: f64 = base_caps.iter().sum();
    for day in 0..n_days {
        for asset in 0..n_assets {
            let daily_variation = 1.0 + (rand::random::<f64>() - 0.5) * 0.02; // ±1% variation
            weights[[day, asset]] = base_caps[asset] / total_cap * daily_variation;
        }
    }
    
    weights
}