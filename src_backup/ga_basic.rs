use alpha_expr::ga::{Chromosome, crossover, mutate, evaluate_population, GenotypeDecoder, FitnessEvaluator};
use alpha_expr::expr::Expr;
use rand::rngs::StdRng;
use rand::SeedableRng;

struct SimpleDecoder;
impl GenotypeDecoder for SimpleDecoder {
    fn decode(&self, c: &Chromosome) -> Expr {
        // decode genes to a literal: sum of genes
        let s: u8 = c.genes.iter().cloned().sum();
        Expr::Literal(alpha_expr::expr::Literal::Float(s as f64))
    }
}

struct SimpleEvaluator;
impl FitnessEvaluator for SimpleEvaluator {
    fn fitness(&self, expr: &Expr) -> f64 {
        if let Expr::Literal(alpha_expr::expr::Literal::Float(v)) = expr { *v } else { 0.0 }
    }
}

#[test]
fn test_crossover_mutate_and_eval() {
    let mut rng = StdRng::seed_from_u64(42);
    let a = Chromosome { genes: vec![1,2,3,4] };
    let b = Chromosome { genes: vec![9,9,9,9] };
    let (c1, c2) = crossover(&a, &b, &mut rng);
    assert_ne!(c1.genes, a.genes);
    assert_ne!(c2.genes, b.genes);
    let mut m = c1.clone();
    mutate(&mut m, 0.5, 10, &mut rng);

    let pop = vec![a.clone(), b.clone(), c1.clone(), c2.clone(), m.clone()];
    let decoder = SimpleDecoder;
    let evaluator = SimpleEvaluator;
    let scores = evaluate_population(&decoder, &evaluator, &pop);
    assert_eq!(scores.len(), pop.len());
}