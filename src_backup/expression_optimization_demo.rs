//! Expression optimization demonstration
//! 
//! This example shows how expression optimization improves performance
//! and simplifies expressions for alpha factor computation.

use alpha_expr::expr::Expr;
use alpha_expr::alpha::AlphaExprExt;
use alpha_expr::expr_optimizer::{ExpressionOptimizer, optimize_expression};
use alpha_expr::polars_style::{col, lit_float, lit_int, df_from_arrays, evaluate_expr_on_dataframe_optimized};
use ndarray::Array1;
use std::collections::HashMap;

fn main() -> Result<(), String> {
    println!("=== Expression Optimization Demo ===\n");
    
    // 1. Create expression optimizer
    println!("1. Creating expression optimizer...");
    let optimizer = ExpressionOptimizer::new();
    
    // 2. Test constant folding
    println!("\n2. Constant folding optimizations:");
    
    // Simple arithmetic: 5 + 3 = 8
    let expr1 = lit_int(5) + lit_int(3);
    let optimized1 = optimizer.optimize(expr1);
    println!("   a) 5 + 3 = {:?}", optimized1);
    
    // Complex constant expression: (10 * 2) - (6 / 3)
    let expr2 = (lit_int(10) * lit_int(2)) - (lit_int(6) / lit_int(3));
    let optimized2 = optimizer.optimize(expr2);
    println!("   b) (10 * 2) - (6 / 3) = {:?}", optimized2);
    
    // Mixed integer/float: 5 + 2.5
    let expr3 = lit_int(5) + lit_float(2.5);
    let optimized3 = optimizer.optimize(expr3);
    println!("   c) 5 + 2.5 = {:?}", optimized3);
    
    // 3. Test algebraic simplifications
    println!("\n3. Algebraic simplifications:");
    
    // x + 0 = x
    let expr4 = col("price") + lit_int(0);
    let optimized4 = optimizer.optimize(expr4);
    println!("   a) price + 0 = {:?}", optimized4);
    
    // 0 + x = x
    let expr5 = lit_int(0) + col("volume");
    let optimized5 = optimizer.optimize(expr5);
    println!("   b) 0 + volume = {:?}", optimized5);
    
    // x * 1 = x
    let expr6 = col("close") * lit_float(1.0);
    let optimized6 = optimizer.optimize(expr6);
    println!("   c) close * 1.0 = {:?}", optimized6);
    
    // x * 0 = 0
    let expr7 = col("value") * lit_int(0);
    let optimized7 = optimizer.optimize(expr7);
    println!("   d) value * 0 = {:?}", optimized7);
    
    // x + x = 2 * x
    let expr8 = col("returns") + col("returns");
    let optimized8 = optimizer.optimize(expr8);
    println!("   e) returns + returns = {:?}", optimized8);
    
    // -(-x) = x
    let expr9 = col("price").neg().neg();
    let optimized9 = optimizer.optimize(expr9);
    println!("   f) -(-price) = {:?}", optimized9);
    
    // 4. Complex optimization example
    println!("\n4. Complex optimization example:");
    
    // Original: (price + 0) * 1 + (5 * 0) - (-(-price))
    let original_expr = (col("price") + lit_int(0)) 
        * lit_int(1) 
        + (lit_int(5) * lit_int(0)) 
        - col("price").neg().neg();
    
    let optimized_expr = optimizer.optimize(original_expr);
    println!("   Original: (price + 0) * 1 + (5 * 0) - (-(-price))");
    println!("   Optimized: {:?}", optimized_expr);
    
    // Should simplify to: price - price = 0
    match &optimized_expr {
        Expr::Literal(lit) => println!("   Result: {:?}", lit),
        _ => println!("   Not fully simplified: {:?}", optimized_expr),
    }
    
    // 5. Alpha factor expression optimization
    println!("\n5. Alpha factor expression optimization:");
    
    // Momentum factor: (close - lag(close, 1)) / lag(close, 1)
    let momentum_expr = (col("close") - col("close").alpha().momentum(1)) 
        / col("close").alpha().momentum(1);
    
    // This won't be optimized by constant folding, but we can still show the structure
    println!("   Momentum factor expression structure:");
    println!("   {:?}", momentum_expr);
    
    // 6. Performance comparison with caching
    println!("\n6. Performance with caching and optimization:");
    
    // Create a DataFrame with test data
    let n_rows = 1000;
    let mut columns = HashMap::new();
    
    // Generate price data
    let prices: Vec<f64> = (0..n_rows).map(|i| 100.0 + i as f64 * 0.1).collect();
    columns.insert("close".to_string(), Array1::from_vec(prices.clone()));
    
    // Generate volume data
    let volumes: Vec<f64> = (0..n_rows).map(|i| 1000.0 + i as f64 * 5.0).collect();
    columns.insert("volume".to_string(), Array1::from_vec(volumes));
    
    let df = df_from_arrays(columns)?;
    println!("   Created DataFrame with {} rows", df.n_rows());
    
    // Test expression: close * 2.0 + volume * 0.5
    let test_expr = col("close") * lit_float(2.0) + col("volume") * lit_float(0.5);
    
    // Time the evaluation (simplified timing)
    println!("   Evaluating expression: close * 2.0 + volume * 0.5");
    
    // First evaluation (uncached)
    let start = std::time::Instant::now();
    let result1 = evaluate_expr_on_dataframe_optimized(&test_expr, &df)?;
    let duration1 = start.elapsed();
    
    // Second evaluation (cached)
    let start = std::time::Instant::now();
    let result2 = evaluate_expr_on_dataframe_optimized(&test_expr, &df)?;
    let duration2 = start.elapsed();
    
    println!("   First evaluation: {:?}", duration1);
    println!("   Second evaluation (cached): {:?}", duration2);
    println!("   Speedup: {:.1}x", duration1.as_nanos() as f64 / (duration2.as_nanos() as f64).max(1.0));
    
    // Verify results are the same
    let data1 = result1.data();
    let data2 = result2.data();
    assert_eq!(data1.len(), data2.len(), "Results should have same length");
    println!("   Results match: {}", data1 == data2);
    
    // 7. Show optimized factor calculation
    println!("\n7. Optimized factor calculation pipeline:");
    
    // Define a complex factor: (close - SMA(close, 20)) / volatility(close, 20)
    let ma_expr = col("close").alpha().moving_average(20);
    let vol_expr = col("close").alpha().volatility(20);
    let factor_expr = (col("close") - ma_expr) / vol_expr;
    
    // Optimize the expression
    let optimized_factor = optimize_expression(factor_expr.clone());
    
    println!("   Original factor expression:");
    println!("   (close - moving_average(close, 20)) / volatility(close, 20)");
    println!("   Optimized factor expression:");
    println!("   {:?}", optimized_factor);
    
    // Evaluate both versions to ensure they produce same results
    let original_result = evaluate_expr_on_dataframe_optimized(&factor_expr, &df)?;
    let optimized_result = evaluate_expr_on_dataframe_optimized(&optimized_factor, &df)?;
    
    // Compare first few values
    println!("   First 5 values comparison:");
    let orig_data = original_result.data();
    let opt_data = optimized_result.data();
    
    for i in 0..5.min(n_rows) {
        if i < orig_data.len() && i < opt_data.len() {
            println!("     [{}] Original: {:.4}, Optimized: {:.4}", 
                i, orig_data[i], opt_data[i]);
        }
    }
    
    // 8. Memory optimization benefits
    println!("\n8. Memory optimization benefits:");
    
    // Common subexpression elimination example
    let cse_expr = (col("close") * col("volume")) + (col("close") * col("volume")) / lit_float(2.0);
    let optimized_cse = optimizer.optimize(cse_expr);
    
    println!("   Original: (close * volume) + (close * volume) / 2.0");
    println!("   Optimized: {:?}", optimized_cse);
    println!("   Note: CSE would eliminate duplicate (close * volume) computation");
    
    // 9. Real-world alpha factor optimization
    println!("\n9. Real-world alpha factor optimization:");
    
    // WCR factor: (close * volume) / SMA(close, 10)
    let wcr_expr = (col("close") * col("volume")) / col("close").alpha().moving_average(10);
    
    // Optimize
    let optimized_wcr = optimize_expression(wcr_expr.clone());
    
    println!("   WCR factor: (close * volume) / moving_average(close, 10)");
    println!("   Optimized structure: {:?}", optimized_wcr);
    
    // Evaluate to show it works
    let wcr_result = evaluate_expr_on_dataframe_optimized(&optimized_wcr, &df)?;
    println!("   WCR factor computed successfully");
    println!("   Mean WCR: {:.2}, Std: {:.2}", 
        wcr_result.data().mean().unwrap_or(f64::NAN),
        wcr_result.data().std(1.0));
    
    // 10. Summary of optimization benefits
    println!("\n10. Optimization benefits summary:");
    println!("   a) Constant folding:");
    println!("      - Reduces runtime computation");
    println!("      - Eliminates unnecessary operations");
    println!("      - Example: 5 + 3 → 8 (computed at optimization time)");
    
    println!("   b) Algebraic simplifications:");
    println!("      - x + 0 → x, x * 1 → x, x * 0 → 0");
    println!("      - x + x → 2 * x");
    println!("      - -(-x) → x");
    
    println!("   c) Caching:");
    println!("      - Reuses previously computed results");
    println!("      - Dramatic speedup for repeated evaluations");
    println!("      - Especially useful in backtesting loops");
    
    println!("   d) Common subexpression elimination:");
    println!("      - Identifies duplicate computations");
    println!("      - Computes them once, reuses results");
    println!("      - Reduces memory and computation");
    
    println!("   e) Vectorized evaluation:");
    println!("      - Whole-series operations (not row-by-row)");
    println!("      - Better cache locality");
    println!("      - Potential SIMD optimizations");
    
    println!("\n=== Demo Complete ===");
    println!("\nKey takeaways:");
    println!("1. Expression optimization happens at compile/optimization time");
    println!("2. Optimized expressions evaluate faster at runtime");
    println!("3. Caching provides additional runtime speedups");
    println!("4. Complex alpha factors benefit significantly from optimization");
    println!("5. The system automatically applies optimizations transparently");
    
    Ok(())
}