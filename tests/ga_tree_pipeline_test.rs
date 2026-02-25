use alpha_expr::ga::{run_ga_exprs, GAConfig, PipelineEvaluator};
use alpha_expr::expr::Expr;
use alpha_expr::optimizer::{Optimizer, ConstantFolding, DimensionValidation};
use alpha_expr::dim::{Dimension, DimKind, DimensionContext};
use alpha_expr::data_provider::MockProvider;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::Arc;
use std::collections::HashMap;

#[test]
fn test_ga_tree_pipeline() {
    let mut rng = StdRng::seed_from_u64(55);
    let a = Expr::lit_float(1.0).add(Expr::lit_float(2.0));
    let b = Expr::col("x").mul(Expr::lit_float(3.0));
    let pop = vec![a.clone(), b.clone()];
    let cfg = GAConfig { generations: 3, tournament_size: 2, crossover_rate: 0.7, mutation_rate: 0.3, gene_max: 10 };

    let mut vals = HashMap::new(); 
    vals.insert("x".to_string(), 2.0f64);
    let provider = Arc::new(MockProvider::new(vals));
    let ctx: DimensionContext = vec![("x".to_string(), Dimension { kind: DimKind::Scalar, name: None })];
    let optimizer = Arc::new(Optimizer::new()
        .add_rule(ConstantFolding::new())
        .add_rule(DimensionValidation::new(ctx)));
    let pipe = PipelineEvaluator::new(optimizer, provider);

    // run GA on exprs but using pipeline evaluator for fitness
    let (_best, score) = run_ga_exprs(&cfg, pop, &pipe, &mut rng);
    assert!(score > -1e8);
}