use alpha_expr::ga::{subtree_crossover, subtree_mutate, run_ga_exprs, GAConfig, FitnessEvaluator};
use alpha_expr::expr::{Expr, Literal, BinaryOp};
use rand::rngs::StdRng;
use rand::SeedableRng;

struct SimpleEval;
impl SimpleEval {
    fn fitness_expr(e: &Expr) -> f64 {
        // evaluate expression if it's a literal or binary of literals; else return 1.0
        match e {
            Expr::Literal(Literal::Float(v)) => *v,
            Expr::BinaryExpr { left, right, op } => {
                if let (Expr::Literal(Literal::Float(l)), Expr::Literal(Literal::Float(r))) = (&**left, &**right) {
                    match op {
                        BinaryOp::Add => l + r,
                        BinaryOp::Subtract => l - r,
                        BinaryOp::Multiply => l * r,
                        BinaryOp::Divide => if *r != 0.0 { l / r } else { 0.0 },
                        _ => 1.0,
                    }
                } else { 1.0 }
            }
            _ => 1.0,
         }
     }
 }

impl FitnessEvaluator for SimpleEval {
    fn fitness(&self, expr: &Expr) -> f64 { Self::fitness_expr(expr) }
}

#[test]
fn test_subtree_ops_and_run() {
    let mut rng = StdRng::seed_from_u64(42);
    let a = Expr::lit_float(1.0).add(Expr::lit_float(2.0));
    let b = Expr::lit_float(3.0).mul(Expr::lit_float(4.0));
    let (na, nb) = subtree_crossover(&a, &b, &mut rng);
    // result should be valid Expr
    let _ = format!("{:?}", na);
    let _ = format!("{:?}", nb);
    let ma = subtree_mutate(&a, 2, &mut rng);
    let _ = format!("{:?}", ma);

    let pop = vec![a.clone(), b.clone(), ma.clone()];
    let cfg = GAConfig { generations: 3, tournament_size: 2, crossover_rate: 0.7, mutation_rate: 0.3, gene_max: 10 };
    let eval = SimpleEval;
    let (_best, score) = run_ga_exprs(&cfg, pop, &eval, &mut rng);
    assert!(score >= 0.0);
}