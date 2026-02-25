//! Basic example of the alpha expression system

use alpha_expr::prelude::*;
use std::collections::HashMap;

fn main() {
    println!("=== Alpha Expression System Demo ===\n");
    
    // 1. Basic expression construction
    println!("1. Basic Expression Construction:");
    let expr = Expr::col("x") + Expr::col("y") * Expr::lit_int(2);
    println!("   Expression: {}", expr);
    
    // 2. Expression evaluation
    println!("\n2. Expression Evaluation:");
    let mut ctx = EvaluationContext::new();
    ctx.set_column("x".to_string(), Literal::Integer(5));
    ctx.set_column("y".to_string(), Literal::Integer(3));
    
    match ctx.evaluate(&expr) {
        Ok(result) => println!("   Result: {:?}", result),
        Err(e) => println!("   Error: {}", e),
    }
    
    // 3. More complex expression
    println!("\n3. Complex Expression:");
    let complex_expr = (Expr::col("price") - Expr::col("cost"))
        .div(Expr::col("cost"))
        .mul(Expr::lit_float(100.0));
    println!("   Profit margin expression: {}", complex_expr);
    
    // 4. Alpha factor construction
    println!("\n4. Alpha Factor Construction:");
    let price = Expr::col("close_price");
    let momentum = price.clone().alpha().momentum(20);
    println!("   Momentum(20) expression: {}", momentum);
    
    let volatility = price.clone().alpha().volatility(30);
    println!("   Volatility(30) expression: {}", volatility);
    
    // 5. Logical plan demonstration
    println!("\n5. Logical Plan Example:");
    let scan = LogicalPlan::scan("stocks");
    let filter = LogicalPlan::filter(
        scan,
        Expr::col("volume").gt(Expr::lit_int(1000))
    );
    let projection = LogicalPlan::projection(
        filter,
        vec![
            Expr::col("symbol"),
            Expr::col("close_price"),
            (Expr::col("close_price") - Expr::col("open_price")).alias("daily_change".to_string()),
        ],
        vec![
            ("symbol".to_string(), DataType::String),
            ("close_price".to_string(), DataType::Float),
            ("daily_change".to_string(), DataType::Float),
        ],
    );
    println!("   Logical plan: {:?}", projection);
    
    println!("\n=== Demo Complete ===");
}