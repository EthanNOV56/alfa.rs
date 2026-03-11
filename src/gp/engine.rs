//! Genetic Programming module for factor mining
//!
//! This module provides genetic programming functionality specifically designed
//! for discovering alpha factors by evolving expression trees and evaluating
//! them through backtesting.
//!
//! Key features:
//! - Train/Test/Validation split for robust factor evaluation
//! - Multi-objective optimization (IC, IR, turnover, complexity)
//! - Enhanced backtest engine with fee and position configuration

use crate::WeightMethod;
use crate::backtest::{BacktestEngine, BacktestResult, FeeConfig, PositionConfig};
use crate::expr::{BinaryOp, Expr, Literal, UnaryOp};
use crate::types::{DataFrame, Series, evaluate_expr_on_dataframe};
use lru::LruCache;
use ndarray::Array2;
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, RwLock};

/// Data split configuration for train/test/validation
#[derive(Clone, Debug)]
pub struct DataSplitConfig {
    /// Training set ratio (e.g., 0.6 for 60%)
    pub train_ratio: f64,
    /// Validation set ratio (e.g., 0.2 for 20%)
    pub validation_ratio: f64,
    /// Test set ratio (e.g., 0.2 for 20%)
    /// If not specified, remaining data goes to test
    pub test_ratio: Option<f64>,
}

impl Default for DataSplitConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.6,
            validation_ratio: 0.2,
            test_ratio: Some(0.2),
        }
    }
}

impl DataSplitConfig {
    /// Validate the split configuration
    pub fn validate(&self) -> bool {
        let total = self.train_ratio + self.validation_ratio + self.test_ratio.unwrap_or(0.0);
        (total - 1.0).abs() < 1e-6
    }

    /// Get the actual test ratio (computed from remaining if not specified)
    pub fn get_test_ratio(&self) -> f64 {
        self.test_ratio
            .unwrap_or(1.0 - self.train_ratio - self.validation_ratio)
    }
}

/// Data split indices
#[derive(Clone, Debug)]
pub struct DataSplit {
    /// Training set day indices
    pub train_indices: Vec<usize>,
    /// Validation set day indices
    pub validation_indices: Vec<usize>,
    /// Test set day indices
    pub test_indices: Vec<usize>,
}

impl DataSplit {
    /// Create a new data split
    pub fn new(
        train_indices: Vec<usize>,
        validation_indices: Vec<usize>,
        test_indices: Vec<usize>,
    ) -> Self {
        Self {
            train_indices,
            validation_indices,
            test_indices,
        }
    }

    /// Create a split from config
    pub fn from_config(n_days: usize, config: &DataSplitConfig) -> Self {
        let train_end = (n_days as f64 * config.train_ratio) as usize;
        let validation_end = train_end + (n_days as f64 * config.validation_ratio) as usize;

        let train_indices: Vec<usize> = (0..train_end).collect();
        let validation_indices: Vec<usize> = (train_end..validation_end).collect();
        let test_indices: Vec<usize> = (validation_end..n_days).collect();

        Self {
            train_indices,
            validation_indices,
            test_indices,
        }
    }
}

/// Evaluation result for a factor on different splits
#[derive(Clone, Debug)]
pub struct SplitEvaluationResult {
    /// Training set metrics
    pub train: SplitMetrics,
    /// Validation set metrics
    pub validation: SplitMetrics,
    /// Test set metrics
    pub test: SplitMetrics,
}

/// Metrics for a single data split
#[derive(Clone, Debug, Default)]
pub struct SplitMetrics {
    /// IC (Information Coefficient) mean
    pub ic_mean: f64,
    /// IC IR (Information Ratio)
    pub ic_ir: f64,
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Turnover rate
    pub turnover: f64,
}

impl SplitMetrics {
    /// Create from BacktestResult
    pub fn from_backtest(result: &BacktestResult) -> Self {
        Self {
            ic_mean: result.ic_mean,
            ic_ir: result.ic_ir,
            total_return: result.total_return,
            annualized_return: result.annualized_return,
            sharpe_ratio: result.sharpe_ratio,
            max_drawdown: result.max_drawdown,
            turnover: result.turnover,
        }
    }
}

/// Fitness evaluator trait for evaluating expression fitness
pub trait FitnessEvaluator: Send + Sync {
    fn fitness(&self, expr: &Expr) -> f64;

    /// Batch evaluate multiple expressions (optional optimization)
    fn fitness_batch(&self, exprs: &[Expr]) -> Vec<f64> {
        // Default implementation: evaluate sequentially
        exprs.iter().map(|e| self.fitness(e)).collect()
    }

    /// Check if this evaluator supports batch evaluation
    fn supports_batch(&self) -> bool {
        false
    }
}

/// Configuration for genetic programming
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GPConfig {
    /// Population size
    pub population_size: usize,
    /// Maximum number of generations
    pub max_generations: usize,
    /// Tournament size for selection
    pub tournament_size: usize,
    /// Crossover probability
    pub crossover_prob: f64,
    /// Mutation probability
    pub mutation_prob: f64,
    /// Maximum tree depth
    pub max_depth: usize,
}

impl Default for GPConfig {
    fn default() -> Self {
        GPConfig {
            population_size: 100,
            max_generations: 50,
            tournament_size: 5,
            crossover_prob: 0.8,
            mutation_prob: 0.1,
            max_depth: 5,
        }
    }
}

/// Function set for generating expressions
#[derive(Clone)]
pub struct Function {
    /// Function name
    pub name: String,
    /// Number of arguments
    pub arity: usize,
    /// Builder function
    pub builder: fn(Vec<Expr>) -> Expr,
}

impl Function {
    /// Create addition function
    pub fn add() -> Self {
        Function {
            name: "add".to_string(),
            arity: 2,
            builder: |args| {
                let mut iter = args.into_iter();
                let left = iter.next().unwrap();
                let right = iter.next().unwrap();
                left.binary(BinaryOp::Add, right)
            },
        }
    }

    /// Create subtraction function
    pub fn sub() -> Self {
        Function {
            name: "sub".to_string(),
            arity: 2,
            builder: |args| {
                let mut iter = args.into_iter();
                let left = iter.next().unwrap();
                let right = iter.next().unwrap();
                left.binary(BinaryOp::Subtract, right)
            },
        }
    }

    /// Create multiplication function
    pub fn mul() -> Self {
        Function {
            name: "mul".to_string(),
            arity: 2,
            builder: |args| {
                let mut iter = args.into_iter();
                let left = iter.next().unwrap();
                let right = iter.next().unwrap();
                left.binary(BinaryOp::Multiply, right)
            },
        }
    }

    /// Create division function
    pub fn div() -> Self {
        Function {
            name: "div".to_string(),
            arity: 2,
            builder: |args| {
                let mut iter = args.into_iter();
                let left = iter.next().unwrap();
                let right = iter.next().unwrap();
                left.binary(BinaryOp::Divide, right)
            },
        }
    }

    /// Create square root function
    pub fn sqrt() -> Self {
        Function {
            name: "sqrt".to_string(),
            arity: 1,
            builder: |args| args.into_iter().next().unwrap().unary(UnaryOp::Sqrt),
        }
    }

    /// Create absolute value function
    pub fn abs() -> Self {
        Function {
            name: "abs".to_string(),
            arity: 1,
            builder: |args| args.into_iter().next().unwrap().unary(UnaryOp::Abs),
        }
    }

    /// Create negation function
    pub fn neg() -> Self {
        Function {
            name: "neg".to_string(),
            arity: 1,
            builder: |args| args.into_iter().next().unwrap().unary(UnaryOp::Negate),
        }
    }

    /// Create rank function (cross-sectional rank)
    pub fn rank() -> Self {
        Function {
            name: "rank".to_string(),
            arity: 1,
            builder: |args| {
                let expr = args.into_iter().next().unwrap();
                Expr::function("rank", vec![expr])
            },
        }
    }

    /// Create ts_mean function (time-series mean)
    pub fn ts_mean() -> Self {
        Function {
            name: "ts_mean".to_string(),
            arity: 2, // expr, window
            builder: |args| {
                let mut iter = args.into_iter();
                let expr = iter.next().unwrap();
                let window = iter.next().unwrap_or(Expr::lit_float(20.0));
                Expr::function("ts_mean", vec![expr, window])
            },
        }
    }

    /// Create ts_std function (time-series standard deviation)
    pub fn ts_std() -> Self {
        Function {
            name: "ts_std".to_string(),
            arity: 2, // expr, window
            builder: |args| {
                let mut iter = args.into_iter();
                let expr = iter.next().unwrap();
                let window = iter.next().unwrap_or(Expr::lit_float(20.0));
                Expr::function("ts_std", vec![expr, window])
            },
        }
    }

    /// Create ts_max function (time-series maximum)
    pub fn ts_max() -> Self {
        Function {
            name: "ts_max".to_string(),
            arity: 2, // expr, window
            builder: |args| {
                let mut iter = args.into_iter();
                let expr = iter.next().unwrap();
                let window = iter.next().unwrap_or(Expr::lit_float(20.0));
                Expr::function("ts_max", vec![expr, window])
            },
        }
    }

    /// Create ts_min function (time-series minimum)
    pub fn ts_min() -> Self {
        Function {
            name: "ts_min".to_string(),
            arity: 2, // expr, window
            builder: |args| {
                let mut iter = args.into_iter();
                let expr = iter.next().unwrap();
                let window = iter.next().unwrap_or(Expr::lit_float(20.0));
                Expr::function("ts_min", vec![expr, window])
            },
        }
    }

    /// Create delay function (time-series shift)
    pub fn delay() -> Self {
        Function {
            name: "delay".to_string(),
            arity: 2, // expr, periods
            builder: |args| {
                let mut iter = args.into_iter();
                let expr = iter.next().unwrap();
                let periods = iter.next().unwrap_or(Expr::lit_float(1.0));
                Expr::function("delay", vec![expr, periods])
            },
        }
    }

    /// Create log function (natural logarithm)
    pub fn log() -> Self {
        Function {
            name: "log".to_string(),
            arity: 1,
            builder: |args| {
                let expr = args.into_iter().next().unwrap();
                Expr::function("log", vec![expr])
            },
        }
    }

    /// Create sign function
    pub fn sign() -> Self {
        Function {
            name: "sign".to_string(),
            arity: 1,
            builder: |args| {
                let expr = args.into_iter().next().unwrap();
                Expr::function("sign", vec![expr])
            },
        }
    }

    /// Create ts_rank function (time-series rank)
    pub fn ts_rank() -> Self {
        Function {
            name: "ts_rank".to_string(),
            arity: 2, // expr, window
            builder: |args| {
                let mut iter = args.into_iter();
                let expr = iter.next().unwrap();
                let window = iter.next().unwrap_or(Expr::lit_float(20.0));
                Expr::function("ts_rank", vec![expr, window])
            },
        }
    }

    /// Create decay_linear function (exponential decay weighted average)
    pub fn decay_linear() -> Self {
        Function {
            name: "decay_linear".to_string(),
            arity: 2, // expr, window
            builder: |args| {
                let mut iter = args.into_iter();
                let expr = iter.next().unwrap();
                let window = iter.next().unwrap_or(Expr::lit_float(20.0));
                Expr::function("decay_linear", vec![expr, window])
            },
        }
    }

    /// Create correlation function
    pub fn correlation() -> Self {
        Function {
            name: "correlation".to_string(),
            arity: 3, // expr1, expr2, window
            builder: |args| {
                let mut iter = args.into_iter();
                let expr1 = iter.next().unwrap();
                let expr2 = iter.next().unwrap();
                let window = iter.next().unwrap_or(Expr::lit_float(20.0));
                Expr::function("correlation", vec![expr1, expr2, window])
            },
        }
    }
}

/// Terminal set (leaf nodes) for expression trees
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum Terminal {
    /// Variable (column) reference
    Variable(String),
    /// Constant value
    Constant(f64),
    /// Ephemeral random constant
    Ephemeral,
}

/// Expression generator
pub struct ExpressionGenerator<'a> {
    config: &'a GPConfig,
    terminals: Vec<Terminal>,
    functions: Vec<Function>,
}

impl<'a> ExpressionGenerator<'a> {
    /// Create a new expression generator
    pub fn new(config: &'a GPConfig, terminals: Vec<Terminal>, functions: Vec<Function>) -> Self {
        Self {
            config,
            terminals,
            functions,
        }
    }

    /// Generate a random expression
    pub fn generate_random_expr<R: Rng + ?Sized>(&self, max_depth: usize, rng: &mut R) -> Expr {
        if max_depth == 0 || (!self.functions.is_empty() && rng.gen_bool(0.3)) {
            self.generate_random_terminal(rng)
        } else {
            self.generate_random_function(max_depth - 1, rng)
        }
    }

    /// Generate a random terminal
    pub fn generate_random_terminal<R: Rng + ?Sized>(&self, rng: &mut R) -> Expr {
        let terminal = &self.terminals[rng.gen_range(0..self.terminals.len())];

        match terminal {
            Terminal::Variable(name) => Expr::Column(name.clone()),
            Terminal::Constant(value) => Expr::Literal(Literal::Float(*value)),
            Terminal::Ephemeral => Expr::Literal(Literal::Float(rng.gen_range(-10.0..10.0))),
        }
    }

    /// Generate a random function
    pub fn generate_random_function<R: Rng + ?Sized>(&self, max_depth: usize, rng: &mut R) -> Expr {
        let function_idx = rng.gen_range(0..self.functions.len());
        let arity = self.functions[function_idx].arity;
        let mut args = Vec::with_capacity(arity);

        for _ in 0..arity {
            args.push(self.generate_random_expr(max_depth, rng));
        }

        let function = &self.functions[function_idx];
        (function.builder)(args)
    }

    /// Generate initial population
    pub fn generate_initial_population<R: Rng + ?Sized>(
        &self,
        size: usize,
        rng: &mut R,
    ) -> Vec<Expr> {
        let mut population = Vec::with_capacity(size);
        for _ in 0..size {
            population.push(self.generate_random_expr(self.config.max_depth, rng));
        }
        population
    }
}

/// Tree operations for genetic programming
pub mod tree_ops {
    use super::*;

    /// Collect all paths to nodes in the expression tree
    pub fn collect_paths(e: &Expr, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
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
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
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

    /// Get node at path
    pub fn get_node_at_path<'a>(e: &'a Expr, path: &[usize]) -> Option<&'a Expr> {
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
                Expr::Conditional {
                    condition,
                    then_expr,
                    else_expr,
                    ..
                } => {
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

    /// Replace node at path with new subtree
    pub fn replace_node_at_path(e: Expr, path: &[usize], new: Expr) -> Option<Expr> {
        if path.is_empty() {
            return Some(new);
        }

        let idx = path[0];
        let rest = &path[1..];

        match e {
            Expr::UnaryExpr { op, expr } if idx == 0 => {
                let new_expr = replace_node_at_path((*expr).clone(), rest, new)?;
                Some(Expr::UnaryExpr {
                    op,
                    expr: Arc::new(new_expr),
                })
            }
            Expr::BinaryExpr { left, op, right } => {
                if idx == 0 {
                    let new_left = replace_node_at_path((*left).clone(), rest, new)?;
                    Some(Expr::BinaryExpr {
                        left: Arc::new(new_left),
                        op,
                        right,
                    })
                } else if idx == 1 {
                    let new_right = replace_node_at_path((*right).clone(), rest, new)?;
                    Some(Expr::BinaryExpr {
                        left,
                        op,
                        right: Arc::new(new_right),
                    })
                } else {
                    None
                }
            }
            Expr::FunctionCall { name, args } => {
                let mut new_args = args;
                if idx < new_args.len() {
                    let new_arg = replace_node_at_path(new_args[idx].clone(), rest, new)?;
                    new_args[idx] = new_arg;
                    Some(Expr::FunctionCall {
                        name,
                        args: new_args,
                    })
                } else {
                    None
                }
            }
            Expr::Aggregate { op, expr, distinct } if idx == 0 => {
                let new_expr = replace_node_at_path((*expr).clone(), rest, new)?;
                Some(Expr::Aggregate {
                    op,
                    expr: Arc::new(new_expr),
                    distinct,
                })
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let new_expr = if idx == 0 {
                    let new_cond = replace_node_at_path((*condition).clone(), rest, new)?;
                    Expr::Conditional {
                        condition: Arc::new(new_cond),
                        then_expr,
                        else_expr,
                    }
                } else if idx == 1 {
                    let new_then = replace_node_at_path((*then_expr).clone(), rest, new)?;
                    Expr::Conditional {
                        condition,
                        then_expr: Arc::new(new_then),
                        else_expr,
                    }
                } else if idx == 2 {
                    let new_else = replace_node_at_path((*else_expr).clone(), rest, new)?;
                    Expr::Conditional {
                        condition,
                        then_expr,
                        else_expr: Arc::new(new_else),
                    }
                } else {
                    return None;
                };
                Some(new_expr)
            }
            Expr::Cast { expr, data_type } if idx == 0 => {
                let new_expr = replace_node_at_path((*expr).clone(), rest, new)?;
                Some(Expr::Cast {
                    expr: Arc::new(new_expr),
                    data_type,
                })
            }
            _ => None,
        }
    }

    /// Subtree crossover for expression trees
    pub fn subtree_crossover<R: Rng + ?Sized>(a: &Expr, b: &Expr, rng: &mut R) -> (Expr, Expr) {
        // Collect all paths in both trees
        let mut paths_a = Vec::new();
        collect_paths(a, &mut Vec::new(), &mut paths_a);
        let mut paths_b = Vec::new();
        collect_paths(b, &mut Vec::new(), &mut paths_b);

        // Filter out root path (empty)
        paths_a.retain(|p| !p.is_empty());
        paths_b.retain(|p| !p.is_empty());

        if paths_a.is_empty() || paths_b.is_empty() {
            return (a.clone(), b.clone());
        }

        // Pick random path in each
        let pa = paths_a[rng.gen_range(0..paths_a.len())].clone();
        let pb = paths_b[rng.gen_range(0..paths_b.len())].clone();

        // Extract subtrees
        let sa = get_node_at_path(a, &pa);
        let sb = get_node_at_path(b, &pb);

        match (sa, sb) {
            (Some(sub_a), Some(sub_b)) => {
                // Replace subtree in a with subtree from b, and vice versa
                let new_a = replace_node_at_path(a.clone(), &pa, sub_b.clone());
                let new_b = replace_node_at_path(b.clone(), &pb, sub_a.clone());
                (
                    new_a.unwrap_or_else(|| a.clone()),
                    new_b.unwrap_or_else(|| b.clone()),
                )
            }
            _ => (a.clone(), b.clone()),
        }
    }

    /// Subtree mutation
    pub fn subtree_mutate<R: Rng + ?Sized>(
        e: &Expr,
        generator: &ExpressionGenerator<'_>,
        max_depth: usize,
        rng: &mut R,
    ) -> Expr {
        let mut paths = Vec::new();
        collect_paths(e, &mut Vec::new(), &mut paths);
        paths.retain(|p| !p.is_empty());

        if paths.is_empty() {
            return e.clone();
        }

        let p = paths[rng.gen_range(0..paths.len())].clone();
        let new_subtree = generator.generate_random_expr(max_depth, rng);

        replace_node_at_path(e.clone(), &p, new_subtree).unwrap_or_else(|| e.clone())
    }
}

/// Run genetic programming
pub fn run_gp<R: Rng + ?Sized>(
    config: &GPConfig,
    evaluator: &dyn FitnessEvaluator,
    terminals: Vec<Terminal>,
    functions: Vec<Function>,
    rng: &mut R,
) -> (Expr, f64) {
    // Generate initial population
    let generator = ExpressionGenerator::new(config, terminals, functions);

    let mut population = generator.generate_initial_population(config.population_size, rng);
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

    // Main evolution loop
    for generation in 0..config.max_generations {
        // Selection and reproduction
        let mut next_population = Vec::with_capacity(pop_size);

        while next_population.len() < pop_size {
            // Tournament selection
            let mut best = rng.gen_range(0..pop_size);
            for _ in 1..config.tournament_size {
                let cand = rng.gen_range(0..pop_size);
                if scores[cand] > scores[best] {
                    best = cand;
                }
            }
            next_population.push(population[best].clone());
        }

        // Crossover
        for i in (0..pop_size).step_by(2) {
            if i + 1 < pop_size && rng.gen_bool(config.crossover_prob) {
                let (c1, c2) =
                    tree_ops::subtree_crossover(&next_population[i], &next_population[i + 1], rng);
                next_population[i] = c1;
                next_population[i + 1] = c2;
            }
        }

        // Mutation
        for expr in next_population.iter_mut() {
            if rng.gen_bool(config.mutation_prob) {
                *expr = tree_ops::subtree_mutate(expr, &generator, config.max_depth, rng);
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

        // Update best individual
        for i in 0..pop_size {
            if scores[i] > scores[best_idx] {
                best_idx = i;
            }
        }

        if generation % 10 == 0 {
            println!(
                "Generation {}: best fitness = {:.6}",
                generation, scores[best_idx]
            );
        }
    }

    (population[best_idx].clone(), scores[best_idx])
}

/// Multi-objective fitness metrics for factor evaluation
#[derive(Debug, Clone)]
pub struct MultiObjectiveFitness {
    /// Information Coefficient (IC) score
    pub ic_score: f64,
    /// Information Ratio (IR) score
    pub ir_score: f64,
    /// Turnover penalty (lower is better)
    pub turnover_penalty: f64,
    /// Complexity penalty (simpler expressions preferred)
    pub complexity_penalty: f64,
    /// Combined fitness score (higher is better)
    pub combined_score: f64,
}

impl MultiObjectiveFitness {
    /// Create a new multi-objective fitness with weighted sum
    pub fn new(
        ic_score: f64,
        ir_score: f64,
        turnover: f64,
        complexity: usize,
        weights: Option<(f64, f64, f64, f64)>, // (ic_weight, ir_weight, turnover_weight, complexity_weight)
    ) -> Self {
        let (w_ic, w_ir, w_to, w_comp) = weights.unwrap_or((0.5, 0.3, 0.1, 0.1));

        // Minimum IC threshold - penalize factors with near-zero IC
        let min_ic_threshold = 0.001;
        let ic_penalty = if ic_score < min_ic_threshold {
            // Heavily penalize factors with IC below threshold
            (min_ic_threshold - ic_score) * 100.0
        } else {
            0.0
        };

        // Only count IR if IC is above threshold (avoid artificial high IR from near-zero IC)
        let effective_ir = if ic_score >= min_ic_threshold {
            ir_score
        } else {
            0.0
        };

        // Normalize turnover (exponential penalty for high turnover)
        let turnover_penalty = (-turnover / 0.1).exp(); // Decay factor

        // Complexity penalty (logarithmic scaling)
        let complexity_penalty = (complexity as f64).ln() / 10.0;

        // Combined weighted score
        let combined_score = w_ic * (ic_score - ic_penalty) + w_ir * effective_ir
            - w_to * turnover_penalty
            - w_comp * complexity_penalty;

        Self {
            ic_score,
            ir_score,
            turnover_penalty,
            complexity_penalty,
            combined_score,
        }
    }

    /// Get the main fitness value for selection
    pub fn fitness(&self) -> f64 {
        self.combined_score
    }
}

/// Fitness evaluator for factor mining based on actual backtest performance
/// with support for train/test/validation split
pub struct RealBacktestFitnessEvaluator {
    data: HashMap<String, Array2<f64>>,
    returns: Array2<f64>,
    weights: HashMap<String, f64>, // Feature weights for multi-objective optimization
    min_valid_days: usize,         // Minimum days required for valid backtest
    // Data split configuration
    data_split: Option<DataSplit>,
    // Fee configuration for backtest
    fee_config: FeeConfig,
    // Position configuration for backtest
    position_config: PositionConfig,
    // Store last computed metrics (using RwLock for thread safety)
    last_metrics: RwLock<(f64, f64, f64, usize)>, // (ic, ir, turnover, complexity)
    // Store last split evaluation result
    last_split_result: RwLock<Option<SplitEvaluationResult>>,
}

impl RealBacktestFitnessEvaluator {
    /// Create a new real backtest fitness evaluator (backward compatible)
    pub fn new(data: HashMap<String, Array2<f64>>, returns: Array2<f64>) -> Self {
        Self {
            data,
            returns,
            weights: HashMap::new(),
            min_valid_days: 50,
            data_split: None,
            fee_config: FeeConfig::default(),
            position_config: PositionConfig::default(),
            last_metrics: RwLock::new((0.0, 0.0, 0.0, 0)),
            last_split_result: RwLock::new(None),
        }
    }

    /// Create a new evaluator with train/test/validation split
    pub fn with_split(
        data: HashMap<String, Array2<f64>>,
        returns: Array2<f64>,
        split_config: DataSplitConfig,
    ) -> Self {
        let n_days = returns.shape()[0];
        let data_split = DataSplit::from_config(n_days, &split_config);

        Self {
            data,
            returns,
            weights: HashMap::new(),
            min_valid_days: 50,
            data_split: Some(data_split),
            fee_config: FeeConfig::default(),
            position_config: PositionConfig::default(),
            last_metrics: RwLock::new((0.0, 0.0, 0.0, 0)),
            last_split_result: RwLock::new(None),
        }
    }

    /// Set fee configuration
    pub fn with_fee_config(mut self, fee_config: FeeConfig) -> Self {
        self.fee_config = fee_config;
        self
    }

    /// Set position configuration
    pub fn with_position_config(mut self, position_config: PositionConfig) -> Self {
        self.position_config = position_config;
        self
    }

    /// Get last computed IC
    pub fn get_last_ic(&self) -> f64 {
        self.last_metrics.read().unwrap().0
    }

    /// Get last computed IR
    pub fn get_last_ir(&self) -> f64 {
        self.last_metrics.read().unwrap().1
    }

    /// Get last computed turnover
    pub fn get_last_turnover(&self) -> f64 {
        self.last_metrics.read().unwrap().2
    }

    /// Get last computed complexity
    pub fn get_last_complexity(&self) -> usize {
        self.last_metrics.read().unwrap().3
    }

    /// Set feature weights for multi-objective optimization
    pub fn set_weights(&mut self, weights: HashMap<String, f64>) -> &mut Self {
        self.weights = weights;
        self
    }

    /// Set minimum valid days for backtest
    pub fn set_min_valid_days(&mut self, days: usize) -> &mut Self {
        self.min_valid_days = days;
        self
    }

    /// Evaluate expression and run backtest with multiple objectives
    fn evaluate_with_backtest(&self, expr: &Expr) -> Option<MultiObjectiveFitness> {
        let n_days = self.returns.shape()[0];

        if n_days < self.min_valid_days {
            return None; // Not enough data for valid backtest
        }

        // Evaluate expression to get factor matrix (full data)
        let factor_matrix = self.evaluate_expression(expr)?;

        // Run backtest on the factor matrix (uses training data if split is configured)
        let backtest_result = self.run_backtest(&factor_matrix)?;

        // Evaluate on all splits if configured (for tracking purposes)
        if self.data_split.is_some() {
            // Evaluate on all splits and store results
            let _ = self.evaluate_on_all_splits(&factor_matrix);
        }

        // Compute turnover (simplified - based on factor value changes)
        let turnover = self.compute_turnover(&factor_matrix);

        // Compute complexity
        let complexity = self.compute_complexity(expr);

        // Store last computed metrics
        *self.last_metrics.write().unwrap() = (
            backtest_result.ic_mean.abs(),
            backtest_result.ic_ir.abs(),
            turnover,
            complexity,
        );

        // Get weights
        let w_ic = *self.weights.get("ic").unwrap_or(&0.4);
        let w_ir = *self.weights.get("ir").unwrap_or(&0.3);
        let w_to = *self.weights.get("turnover").unwrap_or(&0.15);
        let w_comp = *self.weights.get("complexity").unwrap_or(&0.15);

        // Create multi-objective fitness
        Some(MultiObjectiveFitness::new(
            backtest_result.ic_mean.abs(), // Use absolute IC (direction doesn't matter)
            backtest_result.ic_ir.abs(),   // Use absolute IR
            turnover,
            complexity,
            Some((w_ic, w_ir, w_to, w_comp)),
        ))
    }

    /// Evaluate expression to get factor matrix
    fn evaluate_expression(&self, expr: &Expr) -> Option<Array2<f64>> {
        let n_days = self.returns.shape()[0];
        let n_assets = self.returns.shape()[1];

        // Parallel evaluation across assets
        let results: Vec<Vec<f64>> = (0..n_assets)
            .into_par_iter()
            .map(|asset_idx| {
                // Create DataFrame for this asset
                let mut columns = HashMap::new();

                for (col_name, array) in &self.data {
                    let column_data = array.column(asset_idx).to_vec();
                    columns.insert(col_name.clone(), Series::new(column_data));
                }

                // Evaluate expression
                if let Ok(df) = DataFrame::from_series_map(columns) {
                    if let Ok(series) = evaluate_expr_on_dataframe(expr, &df) {
                        return series.data().to_vec();
                    }
                }
                vec![f64::NAN; n_days]
            })
            .collect();

        // Check if we have enough valid values
        let valid_count = results
            .iter()
            .flat_map(|v| v.iter())
            .filter(|&&v| !v.is_nan())
            .count();

        if valid_count < self.min_valid_days * n_assets / 2 {
            return None; // Too many NaN values
        }

        // Convert to Array2
        let mut factor_matrix = Array2::<f64>::zeros((n_days, n_assets));
        for (asset_idx, values) in results.iter().enumerate() {
            for (day_idx, &value) in values.iter().enumerate() {
                factor_matrix[[day_idx, asset_idx]] = value;
            }
        }

        Some(factor_matrix)
    }

    /// Run backtest on factor matrix (uses training data if split is configured)
    fn run_backtest(&self, factor: &Array2<f64>) -> Option<BacktestResult> {
        // If split is configured, use training data only
        if let Some(ref split) = self.data_split {
            let train_factor = self.extract_split_data(factor, &split.train_indices);
            let train_returns = self.extract_split_data(&self.returns, &split.train_indices);
            return self.run_backtest_internal(&train_factor, &train_returns);
        }

        // No split - use all data (backward compatible)
        self.run_backtest_internal(factor, &self.returns)
    }

    /// Run backtest on a specific data split
    fn run_backtest_on_split(
        &self,
        factor: &Array2<f64>,
        indices: &[usize],
    ) -> Option<BacktestResult> {
        let split_factor = self.extract_split_data(factor, indices);
        let split_returns = self.extract_split_data(&self.returns, indices);
        self.run_backtest_internal(&split_factor, &split_returns)
    }

    /// Internal backtest implementation using enhanced BacktestEngine
    fn run_backtest_internal(
        &self,
        factor: &Array2<f64>,
        returns: &Array2<f64>,
    ) -> Option<BacktestResult> {
        // Use the enhanced BacktestEngine with fee and position config
        let engine = BacktestEngine::new(
            factor.clone(),
            returns.clone(),
            10, // quantiles
            WeightMethod::Equal,
            1, // long_top_n
            1, // short_top_n
            self.fee_config.clone(),
            self.position_config.clone(),
            None, // weights
            None, // adj_factor
            None, // volume
        );

        match engine.run() {
            Ok(result) => {
                // Check for valid results
                if result.ic_mean.is_nan() || result.ic_ir.is_nan() {
                    None
                } else {
                    Some(result)
                }
            }
            Err(_) => None,
        }
    }

    /// Extract data for a specific set of indices
    fn extract_split_data(&self, data: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let n_cols = data.shape()[1];
        let mut result = Array2::<f64>::zeros((indices.len(), n_cols));

        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..n_cols {
                result[[i, j]] = data[[idx, j]];
            }
        }

        result
    }

    /// Evaluate factor on all splits and return comprehensive results
    pub fn evaluate_on_all_splits(&self, factor: &Array2<f64>) -> Option<SplitEvaluationResult> {
        let split = self.data_split.as_ref()?;

        // Evaluate on training set
        let train_result = self.run_backtest_on_split(factor, &split.train_indices)?;
        let train_metrics = SplitMetrics::from_backtest(&train_result);

        // Evaluate on validation set
        let validation_result = self.run_backtest_on_split(factor, &split.validation_indices)?;
        let validation_metrics = SplitMetrics::from_backtest(&validation_result);

        // Evaluate on test set
        let test_result = self.run_backtest_on_split(factor, &split.test_indices)?;
        let test_metrics = SplitMetrics::from_backtest(&test_result);

        let result = SplitEvaluationResult {
            train: train_metrics,
            validation: validation_metrics,
            test: test_metrics,
        };

        // Store the result
        *self.last_split_result.write().unwrap() = Some(result.clone());

        Some(result)
    }

    /// Get last split evaluation result
    pub fn get_last_split_result(&self) -> Option<SplitEvaluationResult> {
        self.last_split_result.read().unwrap().clone()
    }

    /// Compute turnover based on rank changes (percentage of assets changing position)
    /// This is a more reasonable measure that gives values between 0 and 1
    fn compute_turnover(&self, factor: &Array2<f64>) -> f64 {
        let (n_days, n_assets) = factor.dim();
        if n_days < 2 || n_assets < 2 {
            return 0.0;
        }

        let mut total_turnover = 0.0;
        let mut valid_days = 0;

        for day in 1..n_days {
            let today = factor.row(day);
            let yesterday = factor.row(day - 1);

            // Get valid assets for both days
            let mut valid_indices = Vec::new();
            for asset in 0..n_assets {
                let f_today = today[asset];
                let f_yesterday = yesterday[asset];
                if !f_today.is_nan()
                    && !f_yesterday.is_nan()
                    && !f_today.is_infinite()
                    && !f_yesterday.is_infinite()
                {
                    valid_indices.push(asset);
                }
            }

            if valid_indices.len() < 2 {
                continue;
            }

            // Compute ranks for valid assets
            let mut yesterday_ranks: Vec<(usize, f64)> =
                valid_indices.iter().map(|&i| (i, yesterday[i])).collect();
            let mut today_ranks: Vec<(usize, f64)> =
                valid_indices.iter().map(|&i| (i, today[i])).collect();

            // Sort by factor values (descending)
            yesterday_ranks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            today_ranks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Compute top 20% turnover
            let top_n = std::cmp::max(1, valid_indices.len() / 5);

            let prev_top: std::collections::HashSet<usize> = yesterday_ranks
                .iter()
                .take(top_n)
                .map(|(i, _)| *i)
                .collect();
            let curr_top: std::collections::HashSet<usize> =
                today_ranks.iter().take(top_n).map(|(i, _)| *i).collect();

            // Compute Jaccard distance: |A ∩ B| / |A ∪ B|
            // Turnover = 1 - similarity
            let intersection: usize = prev_top.intersection(&curr_top).count();
            let union = prev_top.len() + curr_top.len() - intersection;
            let similarity = if union > 0 {
                intersection as f64 / union as f64
            } else {
                1.0
            };
            let day_turnover = 1.0 - similarity;

            total_turnover += day_turnover;
            valid_days += 1;
        }

        if valid_days == 0 {
            0.0
        } else {
            total_turnover / valid_days as f64
        }
    }

    /// Compute complexity of an expression (number of nodes)
    fn compute_complexity(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Literal(_) => 1,
            Expr::Column(_) => 1,
            Expr::UnaryExpr { expr, .. } => 1 + self.compute_complexity(expr),
            Expr::BinaryExpr { left, right, .. } => {
                1 + self.compute_complexity(left) + self.compute_complexity(right)
            }
            Expr::FunctionCall { args, .. } => {
                1 + args
                    .iter()
                    .map(|arg| self.compute_complexity(arg))
                    .sum::<usize>()
            }
            Expr::Aggregate { expr, .. } => 1 + self.compute_complexity(expr),
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                1 + self.compute_complexity(condition)
                    + self.compute_complexity(then_expr)
                    + self.compute_complexity(else_expr)
            }
            Expr::Cast { expr, .. } => 1 + self.compute_complexity(expr),
        }
    }

    /// Check expression constraints (e.g., prevent division by zero, extreme values)
    fn check_constraints(&self, expr: &Expr) -> bool {
        self.check_division_by_zero(expr) && self.check_extreme_operations(expr)
    }

    /// Check for potential division by zero
    fn check_division_by_zero(&self, expr: &Expr) -> bool {
        match expr {
            Expr::BinaryExpr { op, right, .. } if *op == BinaryOp::Divide => {
                // Check if right side could be zero
                !self.could_be_zero(right)
            }
            Expr::BinaryExpr { left, right, .. } => {
                self.check_division_by_zero(left) && self.check_division_by_zero(right)
            }
            Expr::UnaryExpr { expr, .. } => self.check_division_by_zero(expr),
            Expr::FunctionCall { args, .. } => {
                args.iter().all(|arg| self.check_division_by_zero(arg))
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.check_division_by_zero(condition)
                    && self.check_division_by_zero(then_expr)
                    && self.check_division_by_zero(else_expr)
            }
            Expr::Aggregate { expr, .. } => self.check_division_by_zero(expr),
            Expr::Cast { expr, .. } => self.check_division_by_zero(expr),
            _ => true,
        }
    }

    /// Check if an expression could evaluate to zero
    fn could_be_zero(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Literal(Literal::Float(v)) => v.abs() < 1e-10,
            Expr::Literal(Literal::Integer(v)) => *v == 0,
            Expr::BinaryExpr { left, op, right } => {
                match op {
                    BinaryOp::Add | BinaryOp::Subtract => {
                        self.could_be_zero(left) && self.could_be_zero(right)
                    }
                    BinaryOp::Multiply => self.could_be_zero(left) || self.could_be_zero(right),
                    BinaryOp::Divide => {
                        // Division by something that could be zero is dangerous
                        self.could_be_zero(right)
                    }
                    _ => false,
                }
            }
            Expr::UnaryExpr {
                op: UnaryOp::Negate,
                expr,
            } => self.could_be_zero(expr),
            _ => false,
        }
    }

    /// Check for extreme operations (e.g., log of negative, sqrt of negative)
    fn check_extreme_operations(&self, expr: &Expr) -> bool {
        match expr {
            Expr::UnaryExpr { op, expr } => match op {
                UnaryOp::Sqrt | UnaryOp::Log => {
                    // Check if expression could be negative
                    !self.could_be_negative(expr)
                }
                _ => self.check_extreme_operations(expr),
            },
            Expr::BinaryExpr { left, right, .. } => {
                self.check_extreme_operations(left) && self.check_extreme_operations(right)
            }
            Expr::FunctionCall { args, .. } => {
                args.iter().all(|arg| self.check_extreme_operations(arg))
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.check_extreme_operations(condition)
                    && self.check_extreme_operations(then_expr)
                    && self.check_extreme_operations(else_expr)
            }
            Expr::Aggregate { expr, .. } => self.check_extreme_operations(expr),
            Expr::Cast { expr, .. } => self.check_extreme_operations(expr),
            _ => true,
        }
    }

    /// Check if an expression could be negative
    fn could_be_negative(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Literal(Literal::Float(v)) => *v < 0.0,
            Expr::Literal(Literal::Integer(v)) => *v < 0,
            Expr::BinaryExpr { left, op, right } => {
                match op {
                    BinaryOp::Add => self.could_be_negative(left) || self.could_be_negative(right),
                    BinaryOp::Subtract => true, // Subtraction could always result in negative
                    BinaryOp::Multiply => {
                        // Multiplication could be negative if one is negative and one positive
                        (self.could_be_negative(left) && !self.could_be_positive(right))
                            || (self.could_be_negative(right) && !self.could_be_positive(left))
                    }
                    BinaryOp::Divide => true, // Division could result in negative
                    _ => false,
                }
            }
            Expr::UnaryExpr {
                op: UnaryOp::Negate,
                expr,
            } => !self.could_be_negative(expr),
            Expr::Column(_) => true, // Column values could be negative
            _ => false,
        }
    }

    /// Check if an expression could be positive
    fn could_be_positive(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Literal(Literal::Float(v)) => *v > 0.0,
            Expr::Literal(Literal::Integer(v)) => *v > 0,
            Expr::BinaryExpr { left, op, right } => {
                match op {
                    BinaryOp::Add => self.could_be_positive(left) || self.could_be_positive(right),
                    BinaryOp::Subtract => true, // Subtraction could result in positive
                    BinaryOp::Multiply => {
                        // Multiplication could be positive if both positive or both negative
                        (self.could_be_positive(left) && self.could_be_positive(right))
                            || (self.could_be_negative(left) && self.could_be_negative(right))
                    }
                    BinaryOp::Divide => true, // Division could result in positive
                    _ => false,
                }
            }
            Expr::UnaryExpr {
                op: UnaryOp::Negate,
                expr,
            } => !self.could_be_positive(expr),
            Expr::Column(_) => true, // Column values could be positive
            _ => false,
        }
    }
}

impl FitnessEvaluator for RealBacktestFitnessEvaluator {
    fn fitness(&self, expr: &Expr) -> f64 {
        // Check constraints first
        if !self.check_constraints(expr) {
            return -1e9; // Very low fitness for invalid expressions
        }

        // Run real backtest evaluation
        match self.evaluate_with_backtest(expr) {
            Some(fitness_metrics) => {
                let base_fitness = fitness_metrics.fitness();

                // Apply penalty for extreme complexity
                let complexity = self.compute_complexity(expr);
                let complexity_penalty = (complexity as f64).powi(2) / 1000.0;

                base_fitness - complexity_penalty
            }
            None => -1e6, // Lower fitness for failed evaluation
        }
    }
}

/// Simple fitness evaluator based on expression complexity (for testing)
#[allow(dead_code)]
pub struct BacktestFitnessEvaluator {
    data: HashMap<String, Array2<f64>>,
    returns: Array2<f64>,
}

impl BacktestFitnessEvaluator {
    /// Create a new backtest fitness evaluator
    pub fn new(data: HashMap<String, Array2<f64>>, returns: Array2<f64>) -> Self {
        Self { data, returns }
    }

    /// Compute complexity of an expression (number of nodes)
    fn compute_complexity(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Literal(_) => 1,
            Expr::Column(_) => 1,
            Expr::UnaryExpr { expr, .. } => 1 + self.compute_complexity(expr),
            Expr::BinaryExpr { left, right, .. } => {
                1 + self.compute_complexity(left) + self.compute_complexity(right)
            }
            Expr::FunctionCall { args, .. } => {
                1 + args
                    .iter()
                    .map(|arg| self.compute_complexity(arg))
                    .sum::<usize>()
            }
            Expr::Aggregate { expr, .. } => 1 + self.compute_complexity(expr),
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                1 + self.compute_complexity(condition)
                    + self.compute_complexity(then_expr)
                    + self.compute_complexity(else_expr)
            }
            Expr::Cast { expr, .. } => 1 + self.compute_complexity(expr),
        }
    }
}

impl FitnessEvaluator for BacktestFitnessEvaluator {
    fn fitness(&self, expr: &Expr) -> f64 {
        // For now, return a simple fitness based on expression complexity
        // In real implementation, this would run actual backtest
        let complexity = self.compute_complexity(expr);

        // Higher fitness for simpler expressions (encourage parsimony)
        1.0 / (complexity as f64 + 1.0)
    }
}

/// Cached fitness evaluator with LRU cache for performance
pub struct CachedFitnessEvaluator<E: FitnessEvaluator> {
    evaluator: E,
    cache: Mutex<LruCache<String, f64>>,
}

impl<E: FitnessEvaluator> CachedFitnessEvaluator<E> {
    /// Create a new cached evaluator with given capacity
    pub fn new(evaluator: E, capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap_or(NonZeroUsize::new(100).unwrap());
        Self {
            evaluator,
            cache: Mutex::new(LruCache::new(cap)),
        }
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Get cache hit rate statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        // For simplicity, just return cache size and capacity
        let cache = self.cache.lock().unwrap();
        (cache.len(), cache.cap().get())
    }
}

impl<E: FitnessEvaluator> FitnessEvaluator for CachedFitnessEvaluator<E> {
    fn fitness(&self, expr: &Expr) -> f64 {
        let key = format!("{:?}", expr);

        // Check cache first
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(&cached) = cache.get(&key) {
                return cached;
            }
        }

        // Compute fitness
        let fitness = self.evaluator.fitness(expr);

        // Store in cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(key, fitness);
        }

        fitness
    }

    fn fitness_batch(&self, exprs: &[Expr]) -> Vec<f64> {
        let mut results = Vec::with_capacity(exprs.len());
        let mut to_compute = Vec::new();

        // First pass: check cache
        {
            let mut cache = self.cache.lock().unwrap();

            for (idx, expr) in exprs.iter().enumerate() {
                let key = format!("{:?}", expr);
                if let Some(&cached) = cache.get(&key) {
                    results.push(cached);
                } else {
                    results.push(0.0); // placeholder
                    to_compute.push((idx, expr.clone(), key));
                }
            }
        }

        // Compute missing fitness values
        if !to_compute.is_empty() {
            let exprs_to_compute: Vec<Expr> =
                to_compute.iter().map(|(_, expr, _)| expr.clone()).collect();
            let computed = self.evaluator.fitness_batch(&exprs_to_compute);

            // Update cache and results
            {
                let mut cache = self.cache.lock().unwrap();

                for ((idx, _, key), fitness) in to_compute.into_iter().zip(computed.into_iter()) {
                    cache.put(key, fitness);
                    results[idx] = fitness;
                }
            }
        }

        results
    }

    fn supports_batch(&self) -> bool {
        true
    }
}

/// Batch-optimized fitness evaluator for parallel processing
pub struct BatchFitnessEvaluator<E: FitnessEvaluator> {
    evaluator: E,
    batch_size: usize,
}

impl<E: FitnessEvaluator> BatchFitnessEvaluator<E> {
    /// Create a new batch evaluator
    pub fn new(evaluator: E, batch_size: usize) -> Self {
        Self {
            evaluator,
            batch_size,
        }
    }
}

impl<E: FitnessEvaluator + Clone> FitnessEvaluator for BatchFitnessEvaluator<E> {
    fn fitness(&self, expr: &Expr) -> f64 {
        self.evaluator.fitness(expr)
    }

    fn fitness_batch(&self, exprs: &[Expr]) -> Vec<f64> {
        use rayon::prelude::*;

        // Process in parallel batches
        exprs
            .par_chunks(self.batch_size)
            .flat_map(|chunk| self.evaluator.fitness_batch(chunk))
            .collect()
    }

    fn supports_batch(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_expression_generation() {
        let config = GPConfig::default();
        let mut rng = StdRng::seed_from_u64(42);

        let terminals = vec![
            Terminal::Variable("x".to_string()),
            Terminal::Constant(1.0),
            Terminal::Ephemeral,
        ];

        let functions = vec![Function::add(), Function::mul()];

        let generator = ExpressionGenerator::new(&config, terminals, functions);

        let expr = generator.generate_random_expr(3, &mut rng);
        assert!(matches!(
            expr,
            Expr::BinaryExpr { .. } | Expr::Literal(_) | Expr::Column(_)
        ));
    }

    #[test]
    fn test_tree_operations() {
        // Create a simple expression: (x + 1) * 2
        let expr = Expr::Column("x".to_string())
            .add(Expr::Literal(Literal::Float(1.0)))
            .mul(Expr::Literal(Literal::Float(2.0)));

        // Test path collection
        let mut paths = Vec::new();
        tree_ops::collect_paths(&expr, &mut Vec::new(), &mut paths);
        assert!(!paths.is_empty());

        // Test get node at path
        if let Some(node) = tree_ops::get_node_at_path(&expr, &[0]) {
            assert!(matches!(node, Expr::BinaryExpr { .. }));
        }
    }

    #[test]
    fn test_fitness_evaluator() {
        let mut data = HashMap::new();
        data.insert("x".to_string(), Array2::<f64>::zeros((10, 5)));

        let returns = Array2::<f64>::zeros((10, 5));

        let evaluator = BacktestFitnessEvaluator::new(data, returns);

        let expr = Expr::Column("x".to_string()).add(Expr::Literal(Literal::Float(1.0)));

        let fitness = evaluator.fitness(&expr);
        assert!(fitness > 0.0);
    }
}
