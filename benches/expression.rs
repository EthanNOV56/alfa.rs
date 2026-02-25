//! Benchmark expression evaluation performance

use alpha_expr::expr::{Expr, Literal};
use alpha_expr::evaluation::EvaluationContext;
use alpha_expr::data_provider::MockProvider;
use alpha_expr::executor::{EvalExecutor, Executor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

fn bench_literal_evaluation(c: &mut Criterion) {
    c.bench_function("literal_evaluation", |b| {
        let expr = Expr::Literal(Literal::Float(42.0));
        let context = EvaluationContext::new();
        
        b.iter(|| {
            let result = context.evaluate(black_box(&expr)).unwrap();
            black_box(result);
        });
    });
}

fn bench_binary_operation(c: &mut Criterion) {
    c.bench_function("binary_addition", |b| {
        let expr = Expr::Literal(Literal::Float(10.0))
            .add(Expr::Literal(Literal::Float(32.0)));
        let context = EvaluationContext::new();
        
        b.iter(|| {
            let result = context.evaluate(black_box(&expr)).unwrap();
            black_box(result);
        });
    });
    
    c.bench_function("complex_expression", |b| {
        // (10 + 20) * (30 - 5) / 2
        let expr = Expr::Literal(Literal::Float(10.0))
            .add(Expr::Literal(Literal::Float(20.0)))
            .mul(
                Expr::Literal(Literal::Float(30.0))
                    .sub(Expr::Literal(Literal::Float(5.0)))
            )
            .div(Expr::Literal(Literal::Float(2.0)));
        let context = EvaluationContext::new();
        
        b.iter(|| {
            let result = context.evaluate(black_box(&expr)).unwrap();
            black_box(result);
        });
    });
}

fn bench_column_evaluation(c: &mut Criterion) {
    let mut scalars = HashMap::new();
    scalars.insert("price".to_string(), 150.0);
    scalars.insert("volume".to_string(), 1000.0);
    scalars.insert("factor".to_string(), 1.5);
    
    let provider = MockProvider::new(scalars);
    let executor = EvalExecutor;
    
    c.bench_function("column_lookup", |b| {
        let expr = Expr::Column("price".to_string());
        
        b.iter(|| {
            let result = executor.evaluate_expr(black_box(&expr), &provider, None);
            black_box(result);
        });
    });
    
    c.bench_function("column_arithmetic", |b| {
        // price * factor
        let expr = Expr::Column("price".to_string())
            .mul(Expr::Column("factor".to_string()));
        
        b.iter(|| {
            let result = executor.evaluate_expr(black_box(&expr), &provider, None);
            black_box(result);
        });
    });
}

fn bench_function_evaluation(c: &mut Criterion) {
    c.bench_function("sqrt_function", |b| {
        let expr = Expr::function("sqrt", vec![Expr::Literal(Literal::Float(16.0))]);
        let context = EvaluationContext::new();
        
        b.iter(|| {
            let result = context.evaluate(black_box(&expr)).unwrap();
            black_box(result);
        });
    });
    
    c.bench_function("conditional_expression", |b| {
        // if price > 100 then price * 1.1 else price * 0.9
        let expr = Expr::conditional(
            Expr::Column("price".to_string()).gt(Expr::Literal(Literal::Float(100.0))),
            Expr::Column("price".to_string()).mul(Expr::Literal(Literal::Float(1.1))),
            Expr::Column("price".to_string()).mul(Expr::Literal(Literal::Float(0.9))),
        );
        
        let mut scalars = HashMap::new();
        scalars.insert("price".to_string(), 150.0);
        let provider = MockProvider::new(scalars);
        let executor = EvalExecutor;
        
        b.iter(|| {
            let result = executor.evaluate_expr(black_box(&expr), &provider, None);
            black_box(result);
        });
    });
}

fn bench_aggregate_evaluation(c: &mut Criterion) {
    let scalars = HashMap::new();
    let mut series = HashMap::new();
    
    // Create a series of 1000 prices
    let prices: Vec<f64> = (0..1000).map(|i| 100.0 + (i as f64) * 0.1).collect();
    series.insert("prices".to_string(), prices.clone());
    
    // Also add as grouped series for context-based lookup
    series.insert("stock_5m:AAPL:20230101:close".to_string(), prices);
    
    let provider = MockProvider::new_with_series(scalars, series);
    let executor = EvalExecutor;
    
    let mut ctx = HashMap::new();
    ctx.insert("table".to_string(), "stock_5m".to_string());
    ctx.insert("symbol".to_string(), "AAPL".to_string());
    ctx.insert("trading_date".to_string(), "20230101".to_string());
    
    c.bench_function("mean_aggregate", |b| {
        let expr = Expr::Column("close".to_string()).aggregate(alpha_expr::expr::AggregateOp::Mean, false);
        
        b.iter(|| {
            let result = executor.evaluate_expr(black_box(&expr), &provider, Some(&ctx));
            black_box(result);
        });
    });
    
    c.bench_function("sum_aggregate", |b| {
        let expr = Expr::Column("close".to_string()).aggregate(alpha_expr::expr::AggregateOp::Sum, false);
        
        b.iter(|| {
            let result = executor.evaluate_expr(black_box(&expr), &provider, Some(&ctx));
            black_box(result);
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(3));
    targets = bench_literal_evaluation,
              bench_binary_operation,
              bench_column_evaluation,
              bench_function_evaluation,
              bench_aggregate_evaluation
);

criterion_main!(benches);