//! Alpha expression backtest demonstration
//! 
//! This example shows how to use alpha expressions from the alpha-expr crate
//! to compute factors and run quantile backtests.

use alpha_expr::backtest::{BacktestEngine, WeightMethod, quantile_backtest};
use alpha_expr::alpha::{AlphaExprExt, factors};
use alpha_expr::expr::Expr;
use ndarray::Array2;
use rand::Rng;

fn main() -> Result<(), String> {
    println!("=== Alpha Expression Backtest Demo ===\n");
    
    // 1. Generate simulated price data
    println!("1. Generating simulated price data...");
    let n_days = 300;
    let n_assets = 100;
    
    let prices = generate_price_data(n_days, n_assets);
    println!("   Data shape: {} days × {} assets", n_days, n_assets);
    
    // 2. Generate returns (for backtesting)
    println!("2. Generating returns data...");
    let returns = calculate_returns(&prices);
    
    // 3. Create alpha expressions
    println!("3. Creating alpha expressions...");
    
    // Using the alpha expression system
    let price_expr = Expr::col("close");
    
    // Momentum factor: price change over 20 days
    let momentum_expr = price_expr.clone().alpha().momentum(20);
    println!("   Created momentum(20) expression");
    
    // Moving average factor
    let ma_expr = price_expr.clone().alpha().moving_average(30);
    println!("   Created moving_average(30) expression");
    
    // Academic momentum factor (Jegadeesh and Titman, 1993)
    let academic_momentum = factors::momentum(price_expr.clone(), 20, 10);
    println!("   Created academic momentum factor");
    
    // 4. Calculate factors using alpha expressions
    println!("4. Calculating factor values...");
    
    // Note: In a full implementation, we would evaluate these expressions
    // on the price data. For this demo, we'll simulate factor values.
    
    // Simulate momentum factor: close / lag(close, 20) - 1
    let momentum_factor = calculate_simple_momentum(&prices, 20);
    
    // Simulate moving average factor: close / SMA(close, 30)
    let ma_factor = calculate_ma_ratio(&prices, 30);
    
    // 5. Run backtests on different factors
    println!("\n5. Running backtests...");
    
    // Backtest on momentum factor
    println!("   a) Momentum factor (20-day):");
    let momentum_result = run_backtest_on_factor(&momentum_factor, &returns)?;
    print_backtest_result(&momentum_result);
    
    // Backtest on moving average factor
    println!("   b) Moving average factor (30-day):");
    let ma_result = run_backtest_on_factor(&ma_factor, &returns)?;
    print_backtest_result(&ma_result);
    
    // 6. Show expression structure
    println!("\n6. Alpha expression structures:");
    println!("   Momentum expression: {:?}", momentum_expr);
    println!("   Moving average expression: {:?}", ma_expr);
    println!("   Academic momentum: {:?}", academic_momentum);
    
    println!("\n=== Demo Complete ===");
    println!("\nNext steps for full implementation:");
    println!("1. Implement proper evaluation of alpha expressions on time series data");
    println!("2. Connect ClickHouse data provider to alpha expression system");
    println!("3. Add more alpha factors (value, quality, volatility, etc.)");
    println!("4. Support factor combination and neutralization");
    
    Ok(())
}

/// Generate simulated price data (random walk with drift)
fn generate_price_data(n_days: usize, n_assets: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut prices = Array2::<f64>::zeros((n_days, n_assets));
    
    for asset in 0..n_assets {
        // Start with random price between 10 and 100
        let mut price = rng.gen_range(10.0..100.0);
        
        for day in 0..n_days {
            // Random walk with slight upward drift
            let change = 0.001 * rng.gen_range(-1.0..1.0);
            price *= 1.0 + change;
            prices[[day, asset]] = price;
        }
    }
    
    // Add some NaN values
    for day in 0..n_days {
        for asset in 0..n_assets {
            if rng.gen_bool(0.02) { // 2% missing
                prices[[day, asset]] = f64::NAN;
            }
        }
    }
    
    prices
}

/// Calculate returns from price data
fn calculate_returns(prices: &Array2<f64>) -> Array2<f64> {
    let (n_days, n_assets) = prices.dim();
    let mut returns = Array2::<f64>::zeros((n_days - 1, n_assets));
    
    for asset in 0..n_assets {
        for day in 0..n_days - 1 {
            let price_today = prices[[day, asset]];
            let price_tomorrow = prices[[day + 1, asset]];
            
            if price_today.is_nan() || price_tomorrow.is_nan() || price_today == 0.0 {
                returns[[day, asset]] = f64::NAN;
            } else {
                returns[[day, asset]] = (price_tomorrow / price_today) - 1.0;
            }
        }
    }
    
    returns
}

/// Calculate simple momentum factor
fn calculate_simple_momentum(prices: &Array2<f64>, period: usize) -> Array2<f64> {
    let (n_days, n_assets) = prices.dim();
    let mut factor = Array2::<f64>::zeros((n_days - 1, n_assets));
    
    for asset in 0..n_assets {
        for day in period..n_days - 1 {
            let price_today = prices[[day, asset]];
            let price_past = prices[[day - period, asset]];
            
            if price_today.is_nan() || price_past.is_nan() || price_past == 0.0 {
                factor[[day - 1, asset]] = f64::NAN;
            } else {
                factor[[day - 1, asset]] = (price_today / price_past) - 1.0;
            }
        }
    }
    
    factor
}

/// Calculate moving average ratio factor
fn calculate_ma_ratio(prices: &Array2<f64>, window: usize) -> Array2<f64> {
    let (n_days, n_assets) = prices.dim();
    let mut factor = Array2::<f64>::zeros((n_days - 1, n_assets));
    
    for asset in 0..n_assets {
        for day in window..n_days - 1 {
            let price_today = prices[[day, asset]];
            
            // Calculate SMA
            let mut sum = 0.0;
            let mut count = 0;
            
            for offset in 0..window {
                let price = prices[[day - offset, asset]];
                if !price.is_nan() {
                    sum += price;
                    count += 1;
                }
            }
            
            if count == 0 || sum == 0.0 {
                factor[[day - 1, asset]] = f64::NAN;
            } else {
                let sma = sum / count as f64;
                factor[[day - 1, asset]] = price_today / sma;
            }
        }
    }
    
    factor
}

/// Run backtest on a factor matrix
fn run_backtest_on_factor(
    factor: &Array2<f64>,
    returns: &Array2<f64>,
) -> Result<alpha_expr::backtest::BacktestResult, String> {
    quantile_backtest(
        factor.clone(),
        returns.clone(),
        10,
        WeightMethod::Equal,
        1,
        1,
        0.0003,
        None,
    )
}

/// Print backtest results
fn print_backtest_result(result: &alpha_expr::backtest::BacktestResult) {
    println!("     Cumulative return: {:.4}%", result.long_short_cum_return * 100.0);
    println!("     IC mean: {:.4}", result.ic_mean);
    println!("     IC IR: {:.4}", result.ic_ir);
}