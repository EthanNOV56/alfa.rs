//! Genetic programming algorithm implementation.
//!
//! The `run_gp` function orchestrates the full evolution loop: initialization,
//! tournament selection with diversity pressure, crossover, mutation (including
//! operator-feedback-guided and smart template mutations), and fitness tracking.

use crate::expr::Expr;
use crate::gp::fitness::FitnessEvaluator;
use crate::gp::generator::ExpressionGenerator;
use crate::gp::operators::{self, apply_smart_mutation, family_hash};
use crate::gp::types::{Function, GPConfig, OperatorFeedback, Terminal};
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;

/// Run genetic programming
pub fn run_gp<R: Rng + ?Sized>(
    config: &GPConfig,
    evaluator: &dyn FitnessEvaluator,
    terminals: Vec<Terminal>,
    functions: Vec<Function>,
    rng: &mut R,
    seed_exprs: Option<&[Expr]>,
) -> (Expr, f64) {
    // Generate initial population
    let generator = ExpressionGenerator::new(config, terminals, functions);

    let mut population = if config.use_diverse_init {
        generator.generate_diverse_population(config.population_size, rng)
    } else {
        generator.generate_initial_population(config.population_size, rng)
    };

    // Inject seed expressions if provided (replace first N individuals)
    if let Some(seeds) = seed_exprs {
        let n = seeds.len().min(population.len());
        for i in 0..n {
            population[i] = seeds[i].clone();
        }
    }

    let pop_size = population.len();

    // Evaluate initial population (using batch if supported)
    let mut scores: Vec<f64> = if evaluator.supports_batch() {
        evaluator.fitness_batch(&population)
    } else {
        // Use parallel iteration for better performance
        population
            .par_iter()
            .map(|e| evaluator.fitness(e))
            .collect()
    };

    let mut best_idx = 0;
    for i in 1..pop_size {
        if scores[i] > scores[best_idx] {
            best_idx = i;
        }
    }

    // Initialize operator feedback from initial population
    let mut feedback = OperatorFeedback::default();
    for (expr, &score) in population.iter().zip(scores.iter()) {
        feedback.record(expr, score);
    }

    // Main evolution loop
    for generation in 0..config.max_generations {
        let gen_t0 = std::time::Instant::now();
        // Compute family hashes for diversity-aware selection
        let family_hashes: Vec<u64> = population.iter().map(|e| family_hash(e)).collect();
        let mut family_usage: HashMap<u64, u32> = HashMap::new();

        // Selection and reproduction
        let mut next_population = Vec::with_capacity(pop_size);
        let penalty = config.parent_diversity_penalty;

        while next_population.len() < pop_size {
            // Tournament selection with diversity pressure
            let mut best = rng.gen_range(0..pop_size);
            for _ in 1..config.tournament_size {
                let cand = rng.gen_range(0..pop_size);
                let best_family_used = *family_usage.get(&family_hashes[best]).unwrap_or(&0) as f64;
                let cand_family_used = *family_usage.get(&family_hashes[cand]).unwrap_or(&0) as f64;
                let best_eff = scores[best] / (1.0 + best_family_used * penalty);
                let cand_eff = scores[cand] / (1.0 + cand_family_used * penalty);
                if cand_eff > best_eff {
                    best = cand;
                }
            }
            *family_usage.entry(family_hashes[best]).or_insert(0) += 1;
            next_population.push(population[best].clone());
        }

        // Crossover
        for i in (0..pop_size).step_by(2) {
            if i + 1 < pop_size && rng.gen_bool(config.crossover_prob) {
                let (c1, c2) = operators::tree_ops::subtree_crossover(
                    &next_population[i],
                    &next_population[i + 1],
                    rng,
                );
                next_population[i] = c1;
                next_population[i + 1] = c2;
            }
        }

        // Mutation (with operator feedback and smart templates)
        for i in 0..pop_size {
            if rng.gen_bool(config.mutation_prob) {
                if config.smart_mutation_ratio > 0.0 && rng.gen_bool(config.smart_mutation_ratio) {
                    // Smart template mutation: use a random peer as B
                    let mut b_idx = rng.gen_range(0..pop_size);
                    while b_idx == i && pop_size > 1 {
                        b_idx = rng.gen_range(0..pop_size);
                    }
                    next_population[i] =
                        apply_smart_mutation(&next_population[i], &next_population[b_idx], rng);
                } else {
                    next_population[i] = operators::tree_ops::subtree_mutate_feedback(
                        &next_population[i],
                        &generator,
                        &feedback,
                        config.max_depth,
                        rng,
                    );
                }
            }

            // Frequency mutation (~10% chance per individual)
            if config.use_frequencies && rng.gen_bool(0.1) {
                next_population[i] =
                    operators::tree_ops::mutate_frequency(&next_population[i], rng);
            }
        }

        // Update population
        population = next_population;
        scores = if evaluator.supports_batch() {
            evaluator.fitness_batch(&population)
        } else {
            // Use parallel iteration for better performance
            population
                .par_iter()
                .map(|e| evaluator.fitness(e))
                .collect()
        };

        // Update operator feedback for next generation
        feedback.reset();
        for (expr, &score) in population.iter().zip(scores.iter()) {
            feedback.record(expr, score);
        }

        // Update best individual
        for i in 0..pop_size {
            if scores[i] > scores[best_idx] {
                best_idx = i;
            }
        }

        let gen_elapsed = gen_t0.elapsed();
        if generation % 5 == 0 {
            println!(
                "Generation {}: best fitness = {:.6}, unique families = {}, time = {:.1}s",
                generation,
                scores[best_idx],
                family_usage.len(),
                gen_elapsed.as_secs_f64()
            );
        }
    }

    (population[best_idx].clone(), scores[best_idx])
}
