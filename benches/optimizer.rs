//! Benchmark optimizer performance

use alpha_expr::expr::{Expr, Literal};
use alpha_expr::logical_plan::LogicalPlan;
use alpha_expr::optimizer::{
    Optimizer, ConstantFolding, PredicatePushdown, ProjectionPushdown, DimensionValidation
};
use alpha_expr::dim::{Dimension, DimensionContext};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::Arc;

fn create_test_plan() -> LogicalPlan {
    // Create a simple plan: Scan -> Projection -> Filter
    let scan = LogicalPlan::Scan {
        source_name: "test".to_string(),
        projection: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
        filters: vec![],
    };
    
    let projection = LogicalPlan::Projection {
        input: Arc::new(scan),
        expr: vec![
            Expr::Column("a".to_string()),
            Expr::Column("b".to_string()),
            Expr::Column("a".to_string()).add(Expr::Column("b".to_string())),
        ],
        schema: vec![
            ("a".to_string(), alpha_expr::expr::DataType::Float),
            ("b".to_string(), alpha_expr::expr::DataType::Float),
            ("c".to_string(), alpha_expr::expr::DataType::Float),
        ],
    };
    
    LogicalPlan::Filter {
        input: Arc::new(projection),
        predicate: Expr::Column("a".to_string()).gt(Expr::Literal(Literal::Float(10.0))),
    }
}

fn bench_constant_folding(c: &mut Criterion) {
    c.bench_function("constant_folding_simple", |b| {
        let optimizer = Optimizer::new()
            .add_rule(ConstantFolding::new());
        let plan = LogicalPlan::Expression(
            Expr::Literal(Literal::Float(1.0))
                .add(Expr::Literal(Literal::Float(2.0)))
                .mul(Expr::Literal(Literal::Float(3.0)))
        );
        
        b.iter(|| {
            let result = optimizer.optimize(black_box(plan.clone()));
            black_box(result);
        });
    });
    
    c.bench_function("constant_folding_complex", |b| {
        let optimizer = Optimizer::new()
            .add_rule(ConstantFolding::new());
        
        // (5 * 2) - 3 + (10 / 2) * 4
        let expr = Expr::Literal(Literal::Float(5.0))
            .mul(Expr::Literal(Literal::Float(2.0)))
            .sub(Expr::Literal(Literal::Float(3.0)))
            .add(
                Expr::Literal(Literal::Float(10.0))
                    .div(Expr::Literal(Literal::Float(2.0)))
                    .mul(Expr::Literal(Literal::Float(4.0)))
            );
        let plan = LogicalPlan::Expression(expr);
        
        b.iter(|| {
            let result = optimizer.optimize(black_box(plan.clone()));
            black_box(result);
        });
    });
}

fn bench_predicate_pushdown(c: &mut Criterion) {
    c.bench_function("predicate_pushdown", |b| {
        let optimizer = Optimizer::new()
            .add_rule(PredicatePushdown::new());
        let plan = create_test_plan();
        
        b.iter(|| {
            let result = optimizer.optimize(black_box(plan.clone()));
            black_box(result);
        });
    });
}

fn bench_projection_pushdown(c: &mut Criterion) {
    c.bench_function("projection_pushdown", |b| {
        let optimizer = Optimizer::new()
            .add_rule(ProjectionPushdown::new());
        let plan = create_test_plan();
        
        b.iter(|| {
            let result = optimizer.optimize(black_box(plan.clone()));
            black_box(result);
        });
    });
}

fn bench_dimension_validation(c: &mut Criterion) {
    let mut ctx = DimensionContext::new();
    ctx.push(("a".to_string(), Dimension::new_timeseries()));
    ctx.push(("b".to_string(), Dimension::new_timeseries()));
    ctx.push(("c".to_string(), Dimension::new_scalar()));
    
    c.bench_function("dimension_validation", |b| {
        let optimizer = Optimizer::new()
            .add_rule(DimensionValidation::new(ctx.clone()));
        let plan = create_test_plan();
        
        b.iter(|| {
            let result = optimizer.optimize(black_box(plan.clone()));
            black_box(result);
        });
    });
}

fn bench_full_optimizer(c: &mut Criterion) {
    let mut ctx = DimensionContext::new();
    ctx.push(("a".to_string(), Dimension::new_timeseries()));
    ctx.push(("b".to_string(), Dimension::new_timeseries()));
    ctx.push(("c".to_string(), Dimension::new_scalar()));
    
    c.bench_function("full_optimizer_pipeline", |b| {
        let optimizer = Optimizer::default()
            .add_rule(DimensionValidation::new(ctx.clone()));
        let plan = create_test_plan();
        
        b.iter(|| {
            let result = optimizer.optimize(black_box(plan.clone()));
            black_box(result);
        });
    });
}

fn bench_optimizer_iterations(c: &mut Criterion) {
    let mut ctx = DimensionContext::new();
    ctx.push(("a".to_string(), Dimension::new_timeseries()));
    ctx.push(("b".to_string(), Dimension::new_timeseries()));
    
    let mut group = c.benchmark_group("optimizer_iterations");
    
    for &iterations in &[1, 3, 5, 8] {
        group.bench_function(format!("{}_iterations", iterations), |b| {
            let mut optimizer = Optimizer::new();
            for _ in 0..iterations {
                optimizer = optimizer
                    .add_rule(ConstantFolding::new())
                    .add_rule(PredicatePushdown::new())
                    .add_rule(ProjectionPushdown::new())
                    .add_rule(DimensionValidation::new(ctx.clone()));
            }
            let plan = create_test_plan();
            
            b.iter(|| {
                let result = optimizer.optimize(black_box(plan.clone()));
                black_box(result);
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(2));
    targets = bench_constant_folding,
              bench_predicate_pushdown,
              bench_projection_pushdown,
              bench_dimension_validation,
              bench_full_optimizer,
              bench_optimizer_iterations
);

criterion_main!(benches);