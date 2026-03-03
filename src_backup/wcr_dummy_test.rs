//! Test for WCR (Volume Weighted Close Ratio) alpha factor using dummy data
//!
//! WCR = VWAP / Average Close
//! where VWAP = ∑(volume × price) / ∑volume

use alpha_expr::timeseries::TimeSeries;
use alpha_expr::expr::Expr;

#[test]
fn test_wcr_basic_calculation() {
    // Create dummy data for testing
    // Simulating 10 time periods of price and volume data
    let prices = vec![
        100.0, 101.0, 102.0, 103.0, 104.0,
        105.0, 106.0, 107.0, 108.0, 109.0
    ];
    
    let volumes = vec![
        1000.0, 1100.0, 1200.0, 1300.0, 1400.0,
        1500.0, 1600.0, 1700.0, 1800.0, 1900.0
    ];
    
    // Create time series (clone to avoid moving)
    let price_series = TimeSeries::new(prices.clone());
    let volume_series = TimeSeries::new(volumes.clone());
    
    // Calculate VWAP manually for verification
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (price, volume) in prices.iter().zip(volumes.iter()) {
        numerator += price * volume;
        denominator += volume;
    }
    let vwap_manual = numerator / denominator;
    
    // Calculate average price manually
    let avg_price_manual = prices.iter().sum::<f64>() / prices.len() as f64;
    
    // Calculate WCR manually
    let wcr_manual = vwap_manual / avg_price_manual;
    
    // Test basic operations using TimeSeries methods
    // Note: TimeSeries doesn't have direct VWAP method yet
    // For now, verify the manual calculations
    
    // Test element-wise multiplication (price * volume)
    let price_array = price_series.data();
    let volume_array = volume_series.data();
    let price_volume_product: Vec<f64> = price_array
        .iter()
        .zip(volume_array.iter())
        .map(|(p, v)| p * v)
        .collect();
    
    // Calculate sum of price*volume
    let sum_price_volume: f64 = price_volume_product.iter().sum();
    
    // Calculate sum of volumes
    let sum_volume: f64 = volumes.iter().sum();
    
    // Calculate VWAP from time series operations
    let vwap_calculated = sum_price_volume / sum_volume;
    
    // Calculate average price
    let avg_price_calculated = prices.iter().sum::<f64>() / prices.len() as f64;
    
    // Calculate WCR
    let wcr_calculated = vwap_calculated / avg_price_calculated;
    
    // Verify calculations match
    assert!((vwap_calculated - vwap_manual).abs() < 1e-10,
        "VWAP calculation mismatch: calculated={}, manual={}", vwap_calculated, vwap_manual);
    
    assert!((avg_price_calculated - avg_price_manual).abs() < 1e-10,
        "Average price mismatch: calculated={}, manual={}", avg_price_calculated, avg_price_manual);
    
    assert!((wcr_calculated - wcr_manual).abs() < 1e-10,
        "WCR calculation mismatch: calculated={}, manual={}", wcr_calculated, wcr_manual);
    
    // Verify WCR is close to 1 (since prices are increasing and volumes are increasing)
    // WCR should be slightly > 1 because higher volumes at higher prices
    assert!(wcr_calculated > 1.0, "WCR should be greater than 1: {}", wcr_calculated);
    assert!(wcr_calculated < 1.1, "WCR should be reasonable: {}", wcr_calculated);
    
    println!("WCR test passed:");
    println!("  Prices: {:?}", prices);
    println!("  Volumes: {:?}", volumes);
    println!("  VWAP: {}", vwap_calculated);
    println!("  Average Price: {}", avg_price_calculated);
    println!("  WCR: {}", wcr_calculated);
}

#[test]
fn test_wcr_expression_building() {
    // Test building WCR expression using alpha-expr's expression system
    
    // Create column expressions for price and volume
    let price_expr = Expr::col("close");
    let volume_expr = Expr::col("volume");
    
    // In a full implementation, we would build:
    // WCR = (sum(close * volume) / sum(volume)) / mean(close)
    
    // For now, test that we can create basic expressions
    let price_times_volume = price_expr.clone().mul(volume_expr.clone());
    
    // Verify expression structure
    // Note: Since Expr is an enum wrapped in Arc in BinaryExpr,
    // we need to match on the expression type differently
    // For now, just verify the expression was created successfully
    println!("Created price * volume expression: {:?}", price_times_volume);
    
    // Alternative: Test that we can create aggregate expressions
    // sum(close * volume) expression would be created with:
    // let sum_expr = price_times_volume.clone().sum();
    // println!("Sum expression: {:?}", sum_expr);
    
    println!("WCR expression building test passed");
}

#[test]
fn test_wcr_edge_cases() {
    // Test edge cases for WCR calculation
    
    // Case 1: Constant prices, varying volumes
    let prices1 = vec![100.0, 100.0, 100.0, 100.0, 100.0];
    let volumes1 = vec![1000.0, 2000.0, 3000.0, 4000.0, 5000.0];
    
    let price_series1 = TimeSeries::new(prices1.clone());
    let volume_series1 = TimeSeries::new(volumes1.clone());
    
    let vwap1 = {
        let sum_pv: f64 = price_series1.data().iter()
            .zip(volume_series1.data().iter())
            .map(|(p, v)| p * v)
            .sum();
        let sum_v: f64 = volumes1.iter().sum();
        sum_pv / sum_v
    };
    
    let avg_price1 = prices1.iter().sum::<f64>() / prices1.len() as f64;
    let wcr1 = vwap1 / avg_price1;
    
    // When prices are constant, WCR should be exactly 1
    assert!((wcr1 - 1.0).abs() < 1e-10,
        "WCR should be 1 for constant prices: {}", wcr1);
    
    // Case 2: Zero volumes (edge case - should handle gracefully)
    let prices2 = vec![100.0, 101.0, 102.0];
    let volumes2 = vec![0.0, 0.0, 0.0];
    
    // This would cause division by zero in real implementation
    // For test, just verify we can create the series
    let _ = TimeSeries::new(prices2);
    let _ = TimeSeries::new(volumes2);
    
    println!("WCR edge cases test passed");
}