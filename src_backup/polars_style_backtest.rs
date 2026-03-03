//! Polars-style expression backtest demonstration
//! 
//! This example shows how to use Polars-style expressions for alpha factor
//! computation and backtesting.

use alpha_expr::polars_style::{DataFrame, Series, col, lit_float, evaluate_expr_on_dataframe, df_from_arrays};
use alpha_expr::backtest::quantile_backtest;
use alpha_expr::alpha::AlphaExprExt;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

fn main() -> Result<(), String> {
    println!("=== Polars-Style Expression Backtest Demo ===\n");
    
    // 1. Generate simulated price data
    println!("1. Generating simulated price data...");
    let n_days = 200;
    let n_assets = 50;
    
    let prices = generate_price_data(n_days, n_assets);
    let returns = calculate_returns(&prices);
    
    println!("   Data shape: {} days × {} assets", n_days, n_assets);
    
    // 2. Create DataFrame for first asset
    println!("\n2. Creating DataFrame for asset 0...");
    
    let mut columns = HashMap::new();
    columns.insert("close".to_string(), prices.column(0).to_owned());
    columns.insert("returns".to_string(), returns.column(0).to_owned());
    
    let df = df_from_arrays(columns)?;
    println!("   DataFrame shape: {} rows × {} columns", df.n_rows(), df.n_cols());
    
    // 3. Polars-style expression evaluation
    println!("\n3. Polars-style expression evaluation:");
    
    // Expression: close * 2.0
    let expr1 = col("close") * lit_float(2.0);
    let result1 = evaluate_expr_on_dataframe(&expr1, &df)?;
    println!("   a) close * 2.0:");
    println!("      First 5 values: {:?}", &result1.data().as_slice().unwrap()[..5]);
    
    // Expression: lag(close, 1)
    let expr2 = col("close").alpha().momentum(1);
    let result2 = evaluate_expr_on_dataframe(&expr2, &df)?;
    println!("   b) momentum(close, 1):");
    println!("      First 5 values: {:?}", &result2.data().as_slice().unwrap()[..5]);
    
    // Expression: moving_average(close, 10)
    let expr3 = col("close").alpha().moving_average(10);
    let result3 = evaluate_expr_on_dataframe(&expr3, &df)?;
    println!("   c) moving_average(close, 10):");
    println!("      Values at indices [9, 19, 29]: {:?}", 
        [result3.data()[9], result3.data()[19], result3.data()[29]]);
    
    // Expression: close / moving_average(close, 20) - 1.0
    let ma_expr = col("close").alpha().moving_average(20);
    let expr4 = col("close") / ma_expr - lit_float(1.0);
    let result4 = evaluate_expr_on_dataframe(&expr4, &df)?;
    println!("   d) close / moving_average(close, 20) - 1.0:");
    println!("      Values at indices [19, 39, 59]: {:?}", 
        [result4.data()[19], result4.data()[39], result4.data()[59]]);
    
    // 4. Backtest using Polars-style expressions
    println!("\n4. Running backtest on multiple assets...");
    
    // Prepare factor matrix and returns matrix for backtesting
    let mut factor_matrix = Array2::<f64>::zeros((n_days, n_assets));
    let returns_matrix = returns.clone();
    
    // Calculate factor values for each asset using Polars-style expressions
    for asset_idx in 0..n_assets {
        // Create DataFrame for this asset
        let mut columns = HashMap::new();
        columns.insert("close".to_string(), prices.column(asset_idx).to_owned());
        
        let df_asset = df_from_arrays(columns)?;
        
        // Define factor expression: momentum(close, 20)
        let factor_expr = col("close").alpha().momentum(20);
        
        // Evaluate expression
        let factor_series = evaluate_expr_on_dataframe(&factor_expr, &df_asset)?;
        
        // Copy to factor matrix
        for day_idx in 0..n_days {
            factor_matrix[[day_idx, asset_idx]] = factor_series.data()[day_idx];
        }
    }
    
    // Trim first 20 days (insufficient data for momentum(20))
    let factor_trimmed = factor_matrix.slice(ndarray::s![20.., ..]).to_owned();
    let returns_trimmed = returns_matrix.slice(ndarray::s![20.., ..]).to_owned();
    
    // Run quantile backtest
    println!("   Running quantile backtest (10 groups)...");
    let result = quantile_backtest(
        factor_trimmed,
        returns_trimmed,
        10,
        alpha_expr::backtest::WeightMethod::Equal,
        1,
        1,
        0.0003,
        None,
    )?;
    
    println!("   Backtest results:");
    println!("     Cumulative return: {:.4}%", result.long_short_cum_return * 100.0);
    println!("     IC mean: {:.4}", result.ic_mean);
    println!("     IC IR: {:.4}", result.ic_ir);
    println!("     Average group returns: {:?}", 
        result.group_cum_returns.iter().map(|&r| format!("{:.4}%", r * 100.0)).collect::<Vec<_>>());
    
    // 5. Compare with traditional alpha evaluation
    println!("\n5. Comparison with traditional alpha evaluation:");
    
    // Using the new vectorized evaluator vs old row-by-row evaluator
    println!("   a) Vectorized evaluation (Polars-style):");
    println!("      - Whole series operations");
    println!("      - Better performance for large datasets");
    println!("      - More intuitive expression building");
    
    println!("   b) Row-by-row evaluation (traditional):");
    println!("      - Suitable for streaming data");
    println!("      - Simpler implementation");
    println!("      - Lower memory footprint");
    
    // 6. Advanced Polars-style operations
    println!("\n6. Advanced Polars-style operations:");
    
    // Create a DataFrame with multiple columns
    let mut advanced_cols = HashMap::new();
    advanced_cols.insert("price".to_string(), prices.column(0).to_owned());
    
    // Simulate volume data
    let volume_data = Array1::from_vec((0..n_days).map(|i| 1000.0 + i as f64 * 10.0).collect());
    advanced_cols.insert("volume".to_string(), volume_data);
    
    let df_advanced = df_from_arrays(advanced_cols)?;
    
    // Complex expression: (price * volume) / moving_average(price, 10)
    let complex_expr = (col("price") * col("volume")) / col("price").alpha().moving_average(10);
    let complex_result = evaluate_expr_on_dataframe(&complex_expr, &df_advanced)?;
    
    println!("   a) (price * volume) / moving_average(price, 10):");
    println!("      Mean: {:.2}, Std: {:.2}", 
        complex_result.data().mean().unwrap_or(f64::NAN),
        complex_result.data().std(1.0));
    
    // 7. DataFrame operations
    println!("\n7. DataFrame operations:");
    
    // Add computed column to DataFrame
    let df_with_factor = df_advanced.with_expr("factor", &complex_expr)?;
    println!("   a) DataFrame with new 'factor' column:");
    println!("      Columns: {:?}", df_with_factor.column("factor").is_some());
    
    // Select specific columns
    let df_selected = df_with_factor.select(&["price", "factor"])?;
    println!("   b) Selected columns: price, factor");
    println!("      Shape: {} rows × {} columns", df_selected.n_rows(), df_selected.n_cols());
    
    println!("\n=== Demo Complete ===");
    println!("\nSummary of Polars-style features:");
    println!("• Vectorized expression evaluation using ndarray");
    println!("• DataFrame and Series abstractions");
    println!("• Chainable expression building (col().alpha().momentum())");
    println!("• Whole-series operations (lag, diff, moving_average, etc.)");
    println!("• Integration with existing backtest engine");
    println!("• Support for complex factor expressions");
    
    Ok(())
}

/// Generate simulated price data (random walk with drift)
fn generate_price_data(n_days: usize, n_assets: usize) -> Array2<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut prices = Array2::<f64>::zeros((n_days, n_assets));
    
    for asset in 0..n_assets {
        let mut price = rng.gen_range(10.0..100.0);
        
        for day in 0..n_days {
            let change = 0.001 * rng.gen_range(-1.0..1.0);
            price *= 1.0 + change;
            prices[[day, asset]] = price;
        }
    }
    
    prices
}

/// Calculate returns from price data
fn calculate_returns(prices: &Array2<f64>) -> Array2<f64> {
    let (n_days, n_assets) = prices.dim();
    let mut returns = Array2::<f64>::zeros((n_days, n_assets));
    
    for asset in 0..n_assets {
        for day in 1..n_days {
            let price_today = prices[[day, asset]];
            let price_yesterday = prices[[day - 1, asset]];
            
            if price_yesterday == 0.0 {
                returns[[day, asset]] = f64::NAN;
            } else {
                returns[[day, asset]] = (price_today / price_yesterday) - 1.0;
            }
        }
        returns[[0, asset]] = f64::NAN;
    }
    
    returns
}