use alpha_expr::expr::Expr;
use alpha_expr::logical_plan::LogicalPlan;
use alpha_expr::optimizer::{Optimizer, DimensionValidation};
use alpha_expr::dim::{Dimension, DimKind, DimensionContext};

#[test]
fn test_dimension_validation_ok() {
    // col 'a' is TimeSeries, multiply scalar 2.0 should be ok
    let e = Expr::col("a").mul(Expr::lit_float(2.0));
    let plan = LogicalPlan::projection(
        LogicalPlan::scan("s"),
        vec![e],
        vec![("x".to_string(), alpha_expr::expr::DataType::Float)],
    );
    let ctx: DimensionContext = vec![("a".to_string(), Dimension { kind: DimKind::TimeSeries, name: None })];
    let rule = DimensionValidation::new(ctx);
    let opt = Optimizer::new().add_rule(rule);
    let _out = opt.optimize(plan);
    // Should not panic
}

#[test]
#[should_panic]
fn test_dimension_validation_fail() {
    // add two incompatible dims: col 'a' TS and col 'b' Scalar -> Add should panic
    let e = Expr::col("a").add(Expr::col("b"));
    let plan = LogicalPlan::projection(
        LogicalPlan::scan("s"),
        vec![e],
        vec![("x".to_string(), alpha_expr::expr::DataType::Float)],
    );
    let ctx: DimensionContext = vec![
        ("a".to_string(), Dimension { kind: DimKind::TimeSeries, name: None }),
        ("b".to_string(), Dimension { kind: DimKind::Scalar, name: None })
    ];
    let rule = DimensionValidation::new(ctx);
    let opt = Optimizer::new().add_rule(rule);
    let _out = opt.optimize(plan);
}