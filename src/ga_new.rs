//! Genetic algorithm framework for expression optimization
//!
//! This module provides generic genetic algorithm operations that can be used
//! for symbolic regression and other expression optimization tasks.

use crate::data_provider::{DataProvider, MockProvider};
use crate::executor::{EvalExecutor, Executor};
use crate::expr::{Expr, Literal};
use crate::logical_plan::LogicalPlan;
use crate::optimizer::{Optimizer, ConstantFolding, PredicatePushdown, ProjectionPushdown, DimensionValidation};
use ahash::AHasher;
use bincode;
use lru::LruCache;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::hash::Hasher;
use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Serialize, Deserialize)]
pub struct CacheDump {
    pub version: u32,
    pub items: Vec<(String, f64)>,
}

#[derive(Clone, Debug)]
pub struct Chromosome {
    pub genes: Vec<u8>,
}

pub trait GenotypeDecoder: Sync {
    fn decode(&self, c: &Chromosome) -> Expr;
}
pub trait FitnessEvaluator: Send + Sync {
    fn fitness(&self, expr: &Expr) -> f64;
}

/// Evaluate population in parallel using rayon
pub fn evaluate_population(
    decoder: &dyn GenotypeDecoder,
    evaluator: &dyn FitnessEvaluator,
    pop: &[Chromosome],
) -> Vec<f64> {
    pop.par_iter()
        .map(|c| {
            let e = decoder.decode(c);
            if !is_executable_expr(&e) {
                return -1e9;
            }
            evaluator.fitness(&e)
        })
        .collect()
}

/// Single-point crossover for vector-encoded chromosomes
pub fn crossover(a: &Chromosome, b: &Chromosome, rng: &mut impl Rng) -> (Chromosome, Chromosome) {
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

/// Mutate chromosome in-place: each gene has `rate` chance to be replaced by a random value from 0..max_val
pub fn mutate(ch: &mut Chromosome, rate: f64, max_val: u8, rng: &mut impl Rng) {
    for g in ch.genes.iter_mut() {
        if rng.gen_bool(rate) {
            *g = rng.gen_range(0..max_val);
        }
    }
}

fn random_chromosome(len: usize, max_val: u8, rng: &mut impl Rng) -> Chromosome {
    let mut genes = Vec::with_capacity(len);
    for _ in 0..len {
        genes.push(rng.gen_range(0..max_val));
    }
    Chromosome { genes }
}

pub struct GAConfig {
    pub generations: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub gene_max: u8,
}

/// Run a simple generational GA. `initial` is used as starting population.
/// Returns best chromosome and its fitness.
pub fn run_ga(
    config: &GAConfig,
    mut population: Vec<Chromosome>,
    decoder: &dyn GenotypeDecoder,
    evaluator: &dyn FitnessEvaluator,
    rng: &mut impl Rng,
) -> (Chromosome, f64) {
    let pop_size = population.len();
    let mut scores = evaluate_population(decoder, evaluator, &population);
    let mut best_idx = 0;
    for i in 1..pop_size {
        if scores[i] > scores[best_idx] {
            best_idx = i;
        }
    }

    for gen in 0..config.generations {
        // create next generation via tournament selection + crossover + mutation
        let mut next = Vec::with_capacity(pop_size);
        while next.len() < pop_size {
            // tournament selection
            let mut best = rng.gen_range(0..pop_size);
            for _ in 1..config.tournament_size {
                let cand = rng.gen_range(0..pop_size);
                if scores[cand] > scores[best] {
                    best = cand;
                }
            }
            next.push(population[best].clone());
        }

        // crossover
        for i in (0..pop_size).step_by(2) {
            if i + 1 < pop_size && rng.gen_bool(config.crossover_rate) {
                let (c1, c2) = crossover(&next[i], &next[i + 1], rng);
                next[i] = c1;
                next[i + 1] = c2;
            }
        }

        // mutation
        for ch in next.iter_mut() {
            mutate(ch, config.mutation_rate, config.gene_max, rng);
        }

        population = next;
        scores = evaluate_population(decoder, evaluator, &population);
        for i in 0..pop_size {
            if scores[i] > scores[best_idx] {
                best_idx = i;
            }
        }
    }
    (population[best_idx].clone(), scores[best_idx])
}

/// Basic check whether Expr contains unsupported nodes (e.g., unsupported function calls)
fn is_executable_expr(e: &Expr) -> bool {
    match e {
        // In alpha-expr, all function calls are supported via evaluation context
        // We'll assume all are executable
        Expr::FunctionCall { .. } => true,
        Expr::UnaryExpr { expr, .. } => is_executable_expr(expr),
        Expr::BinaryExpr { left, right, .. } => is_executable_expr(left) && is_executable_expr(right),
        Expr::Aggregate { expr, .. } => is_executable_expr(expr),
        Expr::Conditional { condition, then_expr, else_expr } => {
            is_executable_expr(condition) && is_executable_expr(then_expr) && is_executable_expr(else_expr)
        }
        Expr::Cast { expr, .. } => is_executable_expr(expr),
        _ => true,
    }
}

// --- Tree-based GA utilities ---

/// Collect all paths to nodes in the tree. A path is a sequence of child indices as described:
/// - Unary: child 0
/// - Binary: left 0, right 1
/// - FunctionCall: args indexed 0..n-1
/// - Conditional: condition 0, then 1, else 2
/// - Aggregate: expr 0
/// - Cast: expr 0
fn collect_paths(e: &Expr, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    out.push(cur.clone());
    match e {
        Expr::UnaryExpr { expr, .. } => {
            cur.push(0);
            collect_paths(expr, cur, out);
            cur.pop();
        }
        Expr::BinaryExpr { left, right, .. } => {
            cur.push(0);
            collect_paths(left, cur, out);
            cur.pop();
            
            cur.push(1);
            collect_paths(right, cur, out);
            cur.pop();
        }
        Expr::FunctionCall { args, .. } => {
            for (i, arg) in args.iter().enumerate() {
                cur.push(i);
                collect_paths(arg, cur, out);
                cur.pop();
            }
        }
        Expr::Aggregate { expr, .. } => {
            cur.push(0);
            collect_paths(expr, cur, out);
            cur.pop();
        }
        Expr::Conditional { condition, then_expr, else_expr } => {
            cur.push(0);
            collect_paths(condition, cur, out);
            cur.pop();
            
            cur.push(1);
            collect_paths(then_expr, cur, out);
            cur.pop();
            
            cur.push(2);
            collect_paths(else_expr, cur, out);
            cur.pop();
        }
        Expr::Cast { expr, .. } => {
            cur.push(0);
            collect_paths(expr, cur, out);
            cur.pop();
        }
        _ => {}
    }
}

/// Get node at path; returns None if path doesn't match tree shape
fn get_node_at_path<'a>(e: &'a Expr, path: &[usize]) -> Option<&'a Expr> {
    let mut cur = e;
    for &idx in path {
        cur = match cur {
            Expr::UnaryExpr { expr, .. } if idx == 0 => expr,
            Expr::BinaryExpr { left, right, .. } => {
                if idx == 0 {
                    left
                } else if idx == 1 {
                    right
                } else {
                    return None;
                }
            }
            Expr::FunctionCall { args, .. } => args.get(idx)?,
            Expr::Aggregate { expr, .. } if idx == 0 => expr,
            Expr::Conditional { condition, then_expr, else_expr, .. } => {
                if idx == 0 {
                    condition
                } else if idx == 1 {
                    then_expr
                } else if idx == 2 {
                    else_expr
                } else {
                    return None;
                }
            }
            Expr::Cast { expr, .. } if idx == 0 => expr,
            _ => return None,
        };
    }
    Some(cur)
}

/// Replace node at path with new subtree; returns new tree or None if path invalid
fn replace_node_at_path(e: Expr, path: &[usize], new: Expr) -> Option<Expr> {
    if path.is_empty() {
        return Some(new);
    }
    let idx = path[0];
    let rest = &path[1..];
    match e {
        Expr::UnaryExpr { op, expr } if idx == 0 => {
            let new_expr = replace_node_at_path(*expr, rest, new)?;
            Some(Expr::UnaryExpr { op, expr: Arc::new(new_expr) })
        }
        Expr::BinaryExpr { left, op, right } => {
            if idx == 0 {
                let new_left = replace_node_at_path(*left, rest, new)?;
                Some(Expr::BinaryExpr { left: Arc::new(new_left), op, right })
            } else if idx == 1 {
                let new_right = replace_node_at_path(*right, rest, new)?;
                Some(Expr::BinaryExpr { left, op, right: Arc::new(new_right) })
            } else {
                None
            }
        }
        Expr::FunctionCall { name, args } => {
            let mut new_args = args;
            if idx < new_args.len() {
                let new_arg = replace_node_at_path(new_args[idx].clone(), rest, new)?;
                new_args[idx] = new_arg;
                Some(Expr::FunctionCall { name, args: new_args })
            } else {
                None
            }
        }
        Expr::Aggregate { op, expr, distinct } if idx == 0 => {
            let new_expr = replace_node_at_path(*expr, rest, new)?;
            Some(Expr::Aggregate { op, expr: Arc::new(new_expr), distinct })
        }
        Expr::Conditional { condition, then_expr, else_expr } => {
            let new_expr = if idx == 0 {
                let new_cond = replace_node_at_path(*condition, rest, new)?;
                Expr::Conditional { 
                    condition: Arc::new(new_cond), 
                    then_expr, 
                    else_expr 
                }
            } else if idx == 1 {
                let new_then = replace_node_at_path(*then_expr, rest, new)?;
                Expr::Conditional { 
                    condition, 
                    then_expr: Arc::new(new_then), 
                    else_expr 
                }
            } else if idx == 2 {
                let new_else = replace_node_at_path(*else_expr, rest, new)?;
                Expr::Conditional { 
                    condition, 
                    then_expr, 
                    else_expr: Arc::new(new_else) 
                }
            } else {
                return None;
            };
            Some(new_expr)
        }
        Expr::Cast { expr, data_type } if idx == 0 => {
            let new_expr = replace_node_at_path(*expr, rest, new)?;
            Some(Expr::Cast { expr: Arc::new(new_expr), data_type })
        }
        _ => None,
    }
}

/// Subtree crossover for expression trees
pub fn subtree_crossover(a: &Expr, b: &Expr, rng: &mut impl Rng) -> (Expr, Expr) {
    // collect all paths in both trees
    let mut paths_a = Vec::new();
    collect_paths(a, &mut Vec::new(), &mut paths_a);
    let mut paths_b = Vec::new();
    collect_paths(b, &mut Vec::new(), &mut paths_b);

    // filter out root path (empty) and leaf-only paths if desired
    paths_a.retain(|p| !p.is_empty());
    paths_b.retain(|p| !p.is_empty());

    if paths_a.is_empty() || paths_b.is_empty() {
        return (a.clone(), b.clone());
    }

    // pick random path in each
    let pa = paths_a[rng.gen_range(0..paths_a.len())].clone();
    let pb = paths_b[rng.gen_range(0..paths_b.len())].clone();

    // extract subtrees
    let sa = get_node_at_path(a, &pa);
    let sb = get_node_at_path(b, &pb);

    match (sa, sb) {
        (Some(sub_a), Some(sub_b)) => {
            // replace subtree in a with subtree from b, and vice versa
            let new_a = replace_node_at_path(a.clone(), &pa, sub_b.clone());
            let new_b = replace_node_at_path(b.clone(), &pb, sub_a.clone());
            (new_a.unwrap_or_else(|| a.clone()), new_b.unwrap_or_else(|| b.clone()))
        }
        _ => (a.clone(), b.clone()),
    }
}

/// Subtree mutation: replace a random subtree with a newly generated one
pub fn subtree_mutate(e: &Expr, max_depth: usize, rng: &mut impl Rng) -> Expr {
    let mut paths = Vec::new();
    collect_paths(e, &mut Vec::new(), &mut paths);
    paths.retain(|p| !p.is_empty());
    if paths.is_empty() {
        return e.clone();
    }
    let p = paths[rng.gen_range(0..paths.len())].clone();
    
    // For simplicity, generate a random literal as replacement
    // In a real implementation, you'd have a function to generate random expressions
    let new_subtree = Expr::Literal(Literal::Float(rng.gen_range(0.0..100.0)));
    
    replace_node_at_path(e.clone(), &p, new_subtree).unwrap_or_else(|| e.clone())
}

/// Run GA directly on expression trees (no chromosome encoding)
pub fn run_ga_exprs(
    config: &GAConfig,
    initial: Vec<Expr>,
    evaluator: &dyn FitnessEvaluator,
    rng: &mut impl Rng,
) -> (Expr, f64) {
    let pop_size = initial.len();
    let mut population = initial;
    let mut scores: Vec<f64> = population
        .iter()
        .map(|e| {
            if !is_executable_expr(e) {
                -1e9
            } else {
                evaluator.fitness(e)
            }
        })
        .collect();
    let mut best_idx = 0;
    for i in 1..pop_size {
        if scores[i] > scores[best_idx] {
            best_idx = i;
        }
    }

    for _gen in 0..config.generations {
        let mut next = Vec::with_capacity(pop_size);
        while next.len() < pop_size {
            // tournament selection
            let mut best = rng.gen_range(0..pop_size);
            for _ in 1..config.tournament_size {
                let cand = rng.gen_range(0..pop_size);
                if scores[cand] > scores[best] {
                    best = cand;
                }
            }
            next.push(population[best].clone());
        }

        // crossover
        for i in (0..pop_size).step_by(2) {
            if i + 1 < pop_size && rng.gen_bool(config.crossover_rate) {
                let (c1, c2) = subtree_crossover(&next[i], &next[i + 1], rng);
                next[i] = c1;
                next[i + 1] = c2;
            }
        }

        // mutation (with a low probability per individual)
        for expr in next.iter_mut() {
            if rng.gen_bool(config.mutation_rate) {
                *expr = subtree_mutate(expr, 5, rng);
            }
        }

        population = next;
        scores = population
            .iter()
            .map(|e| {
                if !is_executable_expr(e) {
                    -1e9
                } else {
                    evaluator.fitness(e)
                }
            })
            .collect();
        for i in 0..pop_size {
            if scores[i] > scores[best_idx] {
                best_idx = i;
            }
        }
    }
    (population[best_idx].clone(), scores[best_idx])
}