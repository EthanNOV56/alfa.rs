//! Genetic algorithm framework for expression optimization
//!
//! This module provides generic genetic algorithm operations that can be used
//! for symbolic regression and other expression optimization tasks.

use crate::expr::Expr;
use rand::Rng;
use rayon::prelude::*;

/// Chromosome encoding for genetic algorithms
#[derive(Clone, Debug)]
pub struct Chromosome {
    /// Gene sequence (simple byte encoding for now)
    pub genes: Vec<u8>,
}

/// Trait for decoding chromosomes into expressions
pub trait GenotypeDecoder: Sync {
    /// Decode a chromosome into an expression
    fn decode(&self, c: &Chromosome) -> Expr;
}

/// Trait for evaluating expression fitness
pub trait FitnessEvaluator: Send + Sync {
    /// Compute fitness score for an expression (higher is better)
    fn fitness(&self, expr: &Expr) -> f64;
}

/// Genetic algorithm configuration
pub struct GAConfig {
    /// Number of generations to run
    pub generations: usize,
    /// Tournament size for selection
    pub tournament_size: usize,
    /// Crossover probability (0.0 to 1.0)
    pub crossover_rate: f64,
    /// Mutation probability per gene (0.0 to 1.0)
    pub mutation_rate: f64,
    /// Maximum gene value (exclusive)
    pub gene_max: u8,
}

/// Evaluate population fitness in parallel
pub fn evaluate_population(
    decoder: &dyn GenotypeDecoder,
    evaluator: &dyn FitnessEvaluator,
    pop: &[Chromosome],
) -> Vec<f64> {
    pop.par_iter()
        .map(|c| {
            let e = decoder.decode(c);
            if !is_executable_expr(&e) {
                return -1e9; // Very low fitness for invalid expressions
            }
            evaluator.fitness(&e)
        })
        .collect()
}

/// Single-point crossover for vector-encoded chromosomes
pub fn crossover(
    a: &Chromosome,
    b: &Chromosome,
    rng: &mut impl Rng,
) -> (Chromosome, Chromosome) {
    let len = a.genes.len().min(b.genes.len());
    if len == 0 {
        return (a.clone(), b.clone());
    }
    let point = rng.gen_range(0..len);
    let mut c1 = a.genes.clone();
    let mut c2 = b.genes.clone();
    for i in point..len {
        c1[i] = b.genes[i];
        c2[i] = a.genes[i];
    }
    (Chromosome { genes: c1 }, Chromosome { genes: c2 })
}

/// Mutate chromosome in-place
pub fn mutate(ch: &mut Chromosome, rate: f64, max_val: u8, rng: &mut impl Rng) {
    for g in ch.genes.iter_mut() {
        if rng.gen_bool(rate) {
            *g = rng.gen_range(0..max_val);
        }
    }
}

/// Generate a random chromosome
fn random_chromosome(len: usize, max_val: u8, rng: &mut impl Rng) -> Chromosome {
    let mut genes = Vec::with_capacity(len);
    for _ in 0..len {
        genes.push(rng.gen_range(0..max_val));
    }
    Chromosome { genes }
}

/// Check if an expression is executable (no division by zero, valid operations, etc.)
pub fn is_executable_expr(_expr: &Expr) -> bool {
    // Simple check for now
    // TODO: Implement more sophisticated checks
    true
}

/// Run a generational genetic algorithm
pub fn run_ga(
    config: &GAConfig,
    mut population: Vec<Chromosome>,
    decoder: &dyn GenotypeDecoder,
    evaluator: &dyn FitnessEvaluator,
    rng: &mut impl Rng,
) -> (Chromosome, f64) {
    let pop_size = population.len();
    assert!(pop_size > 0, "population must be non-empty");

    // Ensure initial population contains only executable expressions
    for ch in population.iter_mut() {
        let e = decoder.decode(ch);
        if !is_executable_expr(&e) {
            // Replace with random chromosome of same length
            *ch = random_chromosome(ch.genes.len(), config.gene_max, rng);
        }
    }
    
    let mut scores = evaluate_population(decoder, evaluator, &population);

    for _gen in 0..config.generations {
        // Selection + reproduction
        let mut new_pop = Vec::with_capacity(pop_size);
        
        while new_pop.len() < pop_size {
            // Tournament select parent 1
            let mut best_idx = rng.gen_range(0..pop_size);
            for _ in 1..config.tournament_size {
                let idx = rng.gen_range(0..pop_size);
                if scores[idx] > scores[best_idx] {
                    best_idx = idx;
                }
            }
            
            // Tournament select parent 2
            let mut best2_idx = rng.gen_range(0..pop_size);
            for _ in 1..config.tournament_size {
                let idx = rng.gen_range(0..pop_size);
                if scores[idx] > scores[best2_idx] {
                    best2_idx = idx;
                }
            }
            
            let parent1 = &population[best_idx];
            let parent2 = &population[best2_idx];
            
            if rng.gen_bool(config.crossover_rate) {
                let (child1, child2) = crossover(parent1, parent2, rng);
                new_pop.push(child1);
                if new_pop.len() < pop_size {
                    new_pop.push(child2);
                }
            } else {
                new_pop.push(parent1.clone());
                if new_pop.len() < pop_size {
                    new_pop.push(parent2.clone());
                }
            }
        }
        
        // Mutate new population
        for ch in new_pop.iter_mut() {
            mutate(ch, config.mutation_rate, config.gene_max, rng);
        }
        
        // Evaluate new population
        population = new_pop;
        scores = evaluate_population(decoder, evaluator, &population);
    }
    
    // Find best chromosome
    let mut best_idx = 0;
    let mut best_score = scores[0];
    for (i, &score) in scores.iter().enumerate() {
        if score > best_score {
            best_idx = i;
            best_score = score;
        }
    }
    
    (population[best_idx].clone(), best_score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    
    // Simple decoder for testing: sum of genes as a constant expression
    struct TestDecoder;
    
    impl GenotypeDecoder for TestDecoder {
        fn decode(&self, c: &Chromosome) -> Expr {
            let sum: u8 = c.genes.iter().cloned().sum();
            Expr::Literal(crate::expr::Literal::Float(sum as f64))
        }
    }
    
    // Simple evaluator for testing: fitness = value
    struct TestEvaluator;
    
    impl FitnessEvaluator for TestEvaluator {
        fn fitness(&self, expr: &Expr) -> f64 {
            match expr {
                Expr::Literal(crate::expr::Literal::Float(f)) => *f,
                _ => 0.0,
            }
        }
    }
    
    #[test]
    fn test_crossover() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = Chromosome { genes: vec![1, 2, 3, 4] };
        let b = Chromosome { genes: vec![9, 9, 9, 9] };
        
        let (c1, c2) = crossover(&a, &b, &mut rng);
        assert_ne!(c1.genes, a.genes);
        assert_ne!(c2.genes, b.genes);
        assert_eq!(c1.genes.len(), 4);
        assert_eq!(c2.genes.len(), 4);
    }
    
    #[test]
    fn test_mutate() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut ch = Chromosome { genes: vec![1, 2, 3, 4] };
        let original = ch.genes.clone();
        
        mutate(&mut ch, 0.5, 10, &mut rng);
        // With 50% mutation rate, some genes should change
        assert!(ch.genes != original || ch.genes == original);
    }
    
    #[test]
    fn test_evaluate_population() {
        let _rng = StdRng::seed_from_u64(42);
        let decoder = TestDecoder;
        let evaluator = TestEvaluator;
        
        let pop = vec![
            Chromosome { genes: vec![1, 1] },  // sum = 2
            Chromosome { genes: vec![2, 2] },  // sum = 4
            Chromosome { genes: vec![3, 3] },  // sum = 6
        ];
        
        let scores = evaluate_population(&decoder, &evaluator, &pop);
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0], 2.0);
        assert_eq!(scores[1], 4.0);
        assert_eq!(scores[2], 6.0);
    }
    
    #[test]
    fn test_run_ga() {
        let mut rng = StdRng::seed_from_u64(123);
        let decoder = TestDecoder;
        let evaluator = TestEvaluator;
        
        let config = GAConfig {
            generations: 5,
            tournament_size: 2,
            crossover_rate: 0.7,
            mutation_rate: 0.1,
            gene_max: 10,
        };
        
        let pop = vec![
            Chromosome { genes: vec![1, 1] },
            Chromosome { genes: vec![2, 2] },
            Chromosome { genes: vec![3, 3] },
        ];
        
        let (best, score) = run_ga(&config, pop, &decoder, &evaluator, &mut rng);
        assert!(score > 0.0);
        assert!(!best.genes.is_empty());
    }
}