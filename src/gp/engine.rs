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
use crate::backtest::{BacktestConfig, BacktestEngine, BacktestResult, FeeConfig, PositionConfig};
use crate::expr::registry::functions::eval_expr_vectorized;
use crate::expr::{BinaryOp, Expr, Literal, UnaryOp};
use lru::LruCache;
use ndarray::Array2;
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::atomic;
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
    /// Diversity penalty factor for parent usage (0.0 = disabled).
    /// Effective score = raw_score / (1 + family_usage * penalty).
    /// Higher values discourage re-selecting structurally similar parents.
    #[serde(default)]
    pub parent_diversity_penalty: f64,
    /// Use diverse niche-based initialization instead of uniform random.
    #[serde(default)]
    pub use_diverse_init: bool,
    /// Ratio of smart template mutations vs random subtree mutations (0.0-1.0).
    /// Smart mutations use constrained A-B templates that preserve semantic roles.
    #[serde(default)]
    pub smart_mutation_ratio: f64,
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
            parent_diversity_penalty: 0.1,
            use_diverse_init: false,
            smart_mutation_ratio: 0.3,
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
            name: "cs_rank".to_string(),
            arity: 1,
            builder: |args| {
                let expr = args.into_iter().next().unwrap();
                Expr::function("cs_rank", vec![expr])
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
            name: "ts_delay".to_string(),
            arity: 2, // expr, periods
            builder: |args| {
                let mut iter = args.into_iter();
                let expr = iter.next().unwrap();
                let periods = iter.next().unwrap_or(Expr::lit_float(1.0));
                Expr::function("ts_delay", vec![expr, periods])
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
            name: "ts_decay_linear".to_string(),
            arity: 2, // expr, window
            builder: |args| {
                let mut iter = args.into_iter();
                let expr = iter.next().unwrap();
                let window = iter.next().unwrap_or(Expr::lit_float(20.0));
                Expr::function("ts_decay_linear", vec![expr, window])
            },
        }
    }

    /// Create correlation function
    pub fn correlation() -> Self {
        Function {
            name: "ts_correlation".to_string(),
            arity: 3, // expr1, expr2, window
            builder: |args| {
                let mut iter = args.into_iter();
                let expr1 = iter.next().unwrap();
                let expr2 = iter.next().unwrap();
                let window = iter.next().unwrap_or(Expr::lit_float(20.0));
                Expr::function("ts_correlation", vec![expr1, expr2, window])
            },
        }
    }

    /// Create ts_delta function (period-over-period difference)
    pub fn ts_delta() -> Self {
        Function {
            name: "ts_delta".to_string(),
            arity: 2,
            builder: |args| {
                let mut iter = args.into_iter();
                let expr = iter.next().unwrap();
                let window = iter.next().unwrap_or(Expr::lit_float(1.0));
                Expr::function("ts_delta", vec![expr, window])
            },
        }
    }

    /// Create ts_sum function (rolling sum)
    pub fn ts_sum() -> Self {
        Function {
            name: "ts_sum".to_string(),
            arity: 2,
            builder: |args| {
                let mut iter = args.into_iter();
                let expr = iter.next().unwrap();
                let window = iter.next().unwrap_or(Expr::lit_float(20.0));
                Expr::function("ts_sum", vec![expr, window])
            },
        }
    }

    /// Create cs_scale function (cross-sectional normalization)
    pub fn cs_scale() -> Self {
        Function {
            name: "cs_scale".to_string(),
            arity: 1,
            builder: |args| {
                let expr = args.into_iter().next().unwrap();
                Expr::function("cs_scale", vec![expr])
            },
        }
    }

    /// Create ts_covariance function
    pub fn ts_covariance() -> Self {
        Function {
            name: "ts_covariance".to_string(),
            arity: 3,
            builder: |args| {
                let mut iter = args.into_iter();
                let expr1 = iter.next().unwrap();
                let expr2 = iter.next().unwrap();
                let window = iter.next().unwrap_or(Expr::lit_float(20.0));
                Expr::function("ts_covariance", vec![expr1, expr2, window])
            },
        }
    }

    /// Create power function (x^y)
    pub fn power() -> Self {
        Function {
            name: "power".to_string(),
            arity: 2,
            builder: |args| {
                let mut iter = args.into_iter();
                let base = iter.next().unwrap();
                let exp = iter.next().unwrap_or(Expr::lit_float(2.0));
                Expr::function("power", vec![base, exp])
            },
        }
    }

    /// Create exp function (e^x)
    pub fn exp() -> Self {
        Function {
            name: "exp".to_string(),
            arity: 1,
            builder: |args| {
                let expr = args.into_iter().next().unwrap();
                Expr::function("exp", vec![expr])
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

/// Population niches for diverse initialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Niche {
    /// Price/volume-based: prefers rolling stats, value columns
    PriceVolume,
    /// Momentum/trend: prefers delay, delta, rank, correlation
    MomentumTS,
    /// Cross-sectional comparison: prefers rank, ratio, subtraction
    CrossSectional,
    /// Unbiased: uniform random (classic behaviour)
    Mixed,
}

impl Niche {
    /// All four niches
    pub const ALL: [Niche; 4] = [
        Niche::PriceVolume,
        Niche::MomentumTS,
        Niche::CrossSectional,
        Niche::Mixed,
    ];

    /// Weight for a terminal in this niche (1.0 = base).
    fn terminal_weight(&self, terminal: &Terminal) -> f64 {
        match terminal {
            Terminal::Variable(name) => {
                let lower = name.to_lowercase();
                match self {
                    Niche::PriceVolume
                        if lower.contains("close")
                            || lower.contains("open")
                            || lower.contains("high")
                            || lower.contains("low")
                            || lower.contains("price")
                            || lower.contains("vol")
                            || lower.contains("amount") =>
                    {
                        3.0
                    }
                    Niche::MomentumTS
                        if lower.contains("return")
                            || lower.contains("ret")
                            || lower.contains("close") =>
                    {
                        2.5
                    }
                    Niche::CrossSectional => 1.5, // slight pref for all columns
                    Niche::Mixed => 1.0,
                    _ => 1.0,
                }
            }
            Terminal::Constant(_) => match self {
                Niche::CrossSectional => 1.5, // consts useful in ratios
                _ => 1.0,
            },
            Terminal::Ephemeral => match self {
                Niche::MomentumTS => 0.5, // fewer free params in momentum
                _ => 1.0,
            },
        }
    }

    /// Weight for a function in this niche (1.0 = base).
    fn function_weight(&self, name: &str) -> f64 {
        match self {
            Niche::PriceVolume => match name {
                "ts_mean" | "ts_std" | "ts_max" | "ts_min" => 3.0,
                "delay" | "log" | "sqrt" | "abs" => 2.0,
                "decay_linear" | "correlation" => 1.5,
                _ => 1.0,
            },
            Niche::MomentumTS => match name {
                "delay" | "ts_delta" | "ts_rank" | "decay_linear" | "correlation" => 3.0,
                "sign" | "ts_mean" | "ts_std" => 2.0,
                _ => 1.0,
            },
            Niche::CrossSectional => match name {
                "rank" | "div" | "sub" | "abs" | "neg" => 3.0,
                "sqrt" | "add" | "mul" => 2.0,
                _ => 1.0,
            },
            Niche::Mixed => 1.0,
        }
    }
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

    /// Generate a random expression (uniform distribution).
    pub fn generate_random_expr<R: Rng + ?Sized>(&self, max_depth: usize, rng: &mut R) -> Expr {
        if max_depth == 0 || (!self.functions.is_empty() && rng.gen_bool(0.3)) {
            self.generate_random_terminal(rng)
        } else {
            self.generate_random_function(max_depth - 1, rng)
        }
    }

    /// Generate expression using operator feedback weights for function selection.
    /// Falls back to uniform random if feedback is None or all weights are zero.
    pub fn generate_feedback_expr<R: Rng + ?Sized>(
        &self,
        feedback: &OperatorFeedback,
        max_depth: usize,
        rng: &mut R,
    ) -> Expr {
        if max_depth == 0 || (!self.functions.is_empty() && rng.gen_bool(0.3)) {
            self.generate_random_terminal(rng)
        } else {
            self.generate_feedback_function(feedback, max_depth - 1, rng)
        }
    }

    /// Generate a function with operator-feedback-biased selection.
    fn generate_feedback_function<R: Rng + ?Sized>(
        &self,
        feedback: &OperatorFeedback,
        max_depth: usize,
        rng: &mut R,
    ) -> Expr {
        // Compute weights: max(avg_fitness, 0.01) to keep all operators reachable
        let weights: Vec<f64> = self
            .functions
            .iter()
            .map(|f| feedback.avg_fitness(&f.name).max(0.01))
            .collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return self.generate_random_function(max_depth, rng);
        }
        let mut r = rng.gen_range(0.0..total);
        for (i, w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return self.build_function(i, max_depth, rng);
            }
        }
        self.build_function(0, max_depth, rng)
    }

    /// Generate an expression biased toward a niche.
    pub fn generate_niche_expr<R: Rng + ?Sized>(
        &self,
        niche: Niche,
        max_depth: usize,
        rng: &mut R,
    ) -> Expr {
        if max_depth == 0 || (!self.functions.is_empty() && rng.gen_bool(0.3)) {
            self.generate_weighted_terminal(niche, rng)
        } else {
            self.generate_weighted_function(niche, max_depth - 1, rng)
        }
    }

    /// Generate a random terminal (uniform).
    pub fn generate_random_terminal<R: Rng + ?Sized>(&self, rng: &mut R) -> Expr {
        let terminal = &self.terminals[rng.gen_range(0..self.terminals.len())];
        self.build_terminal(terminal, rng)
    }

    /// Generate a terminal with niche-biased weights.
    fn generate_weighted_terminal<R: Rng + ?Sized>(&self, niche: Niche, rng: &mut R) -> Expr {
        if niche == Niche::Mixed {
            return self.generate_random_terminal(rng);
        }
        let weights: Vec<f64> = self
            .terminals
            .iter()
            .map(|t| niche.terminal_weight(t))
            .collect();
        let total: f64 = weights.iter().sum();
        let mut r = rng.gen_range(0.0..total);
        for (i, w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return self.build_terminal(&self.terminals[i], rng);
            }
        }
        self.build_terminal(&self.terminals[0], rng)
    }

    fn build_terminal<R: Rng + ?Sized>(&self, terminal: &Terminal, rng: &mut R) -> Expr {
        match terminal {
            Terminal::Variable(name) => Expr::Column(name.clone()),
            Terminal::Constant(value) => Expr::Literal(Literal::Float(*value)),
            Terminal::Ephemeral => Expr::Literal(Literal::Float(rng.gen_range(-10.0..10.0))),
        }
    }

    /// Generate a random function (uniform).
    pub fn generate_random_function<R: Rng + ?Sized>(&self, max_depth: usize, rng: &mut R) -> Expr {
        let function_idx = rng.gen_range(0..self.functions.len());
        self.build_function(function_idx, max_depth, rng)
    }

    /// Generate a function with niche-biased weights.
    fn generate_weighted_function<R: Rng + ?Sized>(
        &self,
        niche: Niche,
        max_depth: usize,
        rng: &mut R,
    ) -> Expr {
        if niche == Niche::Mixed {
            return self.generate_random_function(max_depth, rng);
        }
        let weights: Vec<f64> = self
            .functions
            .iter()
            .map(|f| niche.function_weight(&f.name))
            .collect();
        let total: f64 = weights.iter().sum();
        let mut r = rng.gen_range(0.0..total);
        for (i, w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return self.build_function(i, max_depth, rng);
            }
        }
        self.build_function(0, max_depth, rng)
    }

    fn build_function<R: Rng + ?Sized>(
        &self,
        function_idx: usize,
        max_depth: usize,
        rng: &mut R,
    ) -> Expr {
        let arity = self.functions[function_idx].arity;
        let mut args = Vec::with_capacity(arity);
        for _ in 0..arity {
            args.push(self.generate_random_expr(max_depth, rng));
        }
        let function = &self.functions[function_idx];
        (function.builder)(args)
    }

    /// Generate initial population (uniform).
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

    /// Generate a diverse initial population with equal representation across niches.
    pub fn generate_diverse_population<R: Rng + ?Sized>(
        &self,
        size: usize,
        rng: &mut R,
    ) -> Vec<Expr> {
        let n_niches = Niche::ALL.len();
        let per_niche = size / n_niches;
        let remainder = size % n_niches;
        let mut population = Vec::with_capacity(size);

        for (i, &niche) in Niche::ALL.iter().enumerate() {
            let count = if i < remainder {
                per_niche + 1
            } else {
                per_niche
            };
            for _ in 0..count {
                population.push(self.generate_niche_expr(niche, self.config.max_depth, rng));
            }
        }

        // Shuffle to mix niches
        use rand::seq::SliceRandom;
        population.shuffle(rng);

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
            Expr::FunctionCall { name, args, freq } => {
                let mut new_args = args;
                if idx < new_args.len() {
                    let new_arg = replace_node_at_path(new_args[idx].clone(), rest, new)?;
                    new_args[idx] = new_arg;
                    Some(Expr::FunctionCall {
                        name,
                        args: new_args,
                        freq,
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

    /// Subtree mutation with operator-feedback-biased subtree generation.
    pub fn subtree_mutate_feedback<R: Rng + ?Sized>(
        e: &Expr,
        generator: &ExpressionGenerator<'_>,
        feedback: &OperatorFeedback,
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
        let new_subtree = generator.generate_feedback_expr(feedback, max_depth, rng);

        replace_node_at_path(e.clone(), &p, new_subtree).unwrap_or_else(|| e.clone())
    }
}

/// Compute a family hash for structural deduplication.
/// Two expressions that differ only in numeric constants produce the same hash.
fn family_hash(expr: &Expr) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let normalized = normalize_expression(expr);
    let mut hasher = DefaultHasher::new();
    normalized.hash(&mut hasher);
    hasher.finish()
}

/// Per-generation feedback tracking operator performance during a GP run.
#[derive(Default)]
struct OperatorFeedback {
    /// (cumulative fitness, appearance count) per function name
    func_stats: HashMap<String, (f64, u32)>,
}

impl OperatorFeedback {
    /// Attribute an individual's fitness to the functions it contains.
    fn record(&mut self, expr: &Expr, fitness: f64) {
        let mut funcs = Vec::new();
        collect_function_names(expr, &mut funcs);
        for name in funcs {
            let entry = self.func_stats.entry(name).or_insert((0.0, 0));
            entry.0 += fitness;
            entry.1 += 1;
        }
    }

    /// Get average fitness per function (0.0 if unseen).
    fn avg_fitness(&self, func_name: &str) -> f64 {
        self.func_stats
            .get(func_name)
            .map(|(total, count)| {
                if *count > 0 {
                    total / *count as f64
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0)
    }

    /// Clear for next generation.
    fn reset(&mut self) {
        self.func_stats.clear();
    }
}

/// Collect function names used in an expression (DFS).
fn collect_function_names(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::FunctionCall { name, args, .. } => {
            out.push(name.clone());
            for arg in args {
                collect_function_names(arg, out);
            }
        }
        Expr::UnaryExpr { expr, .. } => collect_function_names(expr, out),
        Expr::BinaryExpr { left, right, .. } => {
            collect_function_names(left, out);
            collect_function_names(right, out);
        }
        Expr::Aggregate { expr, .. } => collect_function_names(expr, out),
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            collect_function_names(condition, out);
            collect_function_names(then_expr, out);
            collect_function_names(else_expr, out);
        }
        Expr::Cast { expr, .. } => collect_function_names(expr, out),
        _ => {}
    }
}

/// Apply a constrained smart mutation template.
/// A is the current expression (main signal), B is a random peer from the population (modifier).
/// Returns the mutated expression.
fn apply_smart_mutation<R: Rng + ?Sized>(a: &Expr, b: &Expr, rng: &mut R) -> Expr {
    match rng.gen_range(0..4) {
        // Weak gate: rank(B) > 0.55 ? A : -1
        0 => {
            let rank_b = Expr::function("rank", vec![b.clone()]);
            let cond = rank_b.gt(Expr::lit_float(0.55));
            Expr::conditional(cond, a.clone(), Expr::lit_float(-1.0))
        }
        // Regime conditional: if_else(ts_rank(B, 20) > 0.5, A, 0.5*A)
        1 => {
            let ts_rank_b = Expr::function("ts_rank", vec![b.clone(), Expr::lit_int(20)]);
            let cond = ts_rank_b.gt(Expr::lit_float(0.5));
            let damped = a.clone().mul(Expr::lit_float(0.5));
            Expr::conditional(cond, a.clone(), damped)
        }
        // Heterogeneous injection: A + 0.1 * rank(B)
        2 => {
            let rank_b = Expr::function("rank", vec![b.clone()]);
            let weak = rank_b.mul(Expr::lit_float(0.1));
            a.clone().add(weak)
        }
        // Cross-family combination: rank(A) * 0.9 + rank(B) * 0.1
        _ => {
            let rank_a = Expr::function("rank", vec![a.clone()]);
            let rank_b = Expr::function("rank", vec![b.clone()]);
            rank_a
                .mul(Expr::lit_float(0.9))
                .add(rank_b.mul(Expr::lit_float(0.1)))
        }
    }
}

/// Maximum Common Isomorphic Subtree size between two expression trees.
/// Counts the maximum number of structurally-equivalent nodes shared.
/// Constants of the same type (Float/Integer) are considered structurally equal.
fn mcis_size(a: &Expr, b: &Expr) -> usize {
    if same_structure_root(a, b) {
        match (a, b) {
            (Expr::Literal(_), Expr::Literal(_)) => 1,
            (Expr::Column(_), Expr::Column(_)) => 1,
            (Expr::UnaryExpr { expr: ae, .. }, Expr::UnaryExpr { expr: be, .. }) => {
                1 + mcis_size(ae, be)
            }
            (
                Expr::BinaryExpr {
                    left: al,
                    op: _,
                    right: ar,
                },
                Expr::BinaryExpr {
                    left: bl,
                    op: _,
                    right: br,
                },
            ) => 1 + mcis_size(al, bl) + mcis_size(ar, br),
            (
                Expr::FunctionCall {
                    name: an,
                    args: aa,
                    freq: _,
                },
                Expr::FunctionCall {
                    name: bn,
                    args: ba,
                    freq: _,
                },
            ) if an == bn && aa.len() == ba.len() => {
                1 + aa
                    .iter()
                    .zip(ba.iter())
                    .map(|(a, b)| mcis_size(a, b))
                    .sum::<usize>()
            }
            (
                Expr::Conditional {
                    condition: ac,
                    then_expr: at,
                    else_expr: ae,
                },
                Expr::Conditional {
                    condition: bc,
                    then_expr: bt,
                    else_expr: be,
                },
            ) => 1 + mcis_size(ac, bc) + mcis_size(at, bt) + mcis_size(ae, be),
            (Expr::Aggregate { expr: ae, .. }, Expr::Aggregate { expr: be, .. }) => {
                1 + mcis_size(ae, be)
            }
            (Expr::Cast { expr: ae, .. }, Expr::Cast { expr: be, .. }) => 1 + mcis_size(ae, be),
            _ => 1,
        }
    } else {
        // Roots differ — find best match among child pairs
        child_best_match(a, b)
    }
}

/// Check if two expression roots have the same structural type.
fn same_structure_root(a: &Expr, b: &Expr) -> bool {
    matches!(
        (a, b),
        (Expr::Literal(_), Expr::Literal(_))
            | (Expr::Column(_), Expr::Column(_))
            | (Expr::UnaryExpr { .. }, Expr::UnaryExpr { .. })
            | (Expr::BinaryExpr { .. }, Expr::BinaryExpr { .. })
            | (Expr::FunctionCall { .. }, Expr::FunctionCall { .. })
            | (Expr::Conditional { .. }, Expr::Conditional { .. })
            | (Expr::Aggregate { .. }, Expr::Aggregate { .. })
            | (Expr::Cast { .. }, Expr::Cast { .. })
    )
}

/// Best MCIS match among any pair of children from two expressions.
fn child_best_match(a: &Expr, b: &Expr) -> usize {
    let children_a = collect_children(a);
    let children_b = collect_children(b);
    if children_a.is_empty() || children_b.is_empty() {
        return 0;
    }
    let mut best = 0;
    for ca in &children_a {
        for cb in &children_b {
            let s = mcis_size(ca, cb);
            if s > best {
                best = s;
            }
        }
    }
    best
}

/// Collect direct children of an expression node.
fn collect_children(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::UnaryExpr { expr, .. } => vec![(**expr).clone()],
        Expr::BinaryExpr { left, right, .. } => vec![(**left).clone(), (**right).clone()],
        Expr::FunctionCall { args, .. } => args.clone(),
        Expr::Aggregate { expr, .. } => vec![(**expr).clone()],
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => vec![
            (**condition).clone(),
            (**then_expr).clone(),
            (**else_expr).clone(),
        ],
        Expr::Cast { expr, .. } => vec![(**expr).clone()],
        _ => vec![],
    }
}

/// Count total nodes in an expression.
fn expr_node_count(expr: &Expr) -> usize {
    match expr {
        Expr::Literal(_) | Expr::Column(_) => 1,
        Expr::UnaryExpr { expr, .. } => 1 + expr_node_count(expr),
        Expr::BinaryExpr { left, right, .. } => 1 + expr_node_count(left) + expr_node_count(right),
        Expr::FunctionCall { args, .. } => {
            1 + args.iter().map(|a| expr_node_count(a)).sum::<usize>()
        }
        Expr::Aggregate { expr, .. } => 1 + expr_node_count(expr),
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            1 + expr_node_count(condition) + expr_node_count(then_expr) + expr_node_count(else_expr)
        }
        Expr::Cast { expr, .. } => 1 + expr_node_count(expr),
    }
}

/// Structural similarity between two expressions (0.0–1.0).
/// Ratio of maximum common isomorphic subtree size to the larger tree size.
pub fn expr_structural_similarity(a: &Expr, b: &Expr) -> f64 {
    let mcis = mcis_size(a, b);
    let na = expr_node_count(a);
    let nb = expr_node_count(b);
    let max_n = na.max(nb);
    if max_n == 0 {
        0.0
    } else {
        mcis as f64 / max_n as f64
    }
}

/// Check a candidate expression against a pool for structural redundancy.
/// Returns the maximum similarity found, or None if the pool is empty.
pub fn check_redundancy(candidate: &Expr, pool: &[Expr]) -> Option<f64> {
    if pool.is_empty() {
        return None;
    }
    let max_sim = pool
        .iter()
        .map(|e| expr_structural_similarity(candidate, e))
        .fold(0.0_f64, f64::max);
    Some(max_sim)
}

/// A single entry in the factor pool.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PoolEntry {
    /// Human-readable expression string
    pub expression: String,
    /// Information Coefficient
    pub ic: f64,
    /// Rank IC
    pub rank_ic: f64,
    /// Date added (Unix timestamp)
    pub added_at: u64,
    /// Last validation check timestamp
    pub last_check_at: u64,
    /// Number of times this factor passed decay checks
    pub survival_rounds: u32,
}

/// Result of attempting to admit a factor to the pool.
#[derive(Debug, Clone, PartialEq)]
pub enum AdmissionResult {
    /// Factor was added to the pool
    Added,
    /// Rejected due to high structural similarity (> 0.95)
    RejectedDuplicate(f64),
    /// Accepted but flagged for structural similarity (0.8–0.95)
    Flagged(f64),
    /// Rejected because pool is full and factor IC is below the minimum
    RejectedBelowMinimum,
}

/// Convert an expression to a parseable string (unlike Debug which uses #name).
pub(crate) fn to_parseable_string(expr: &Expr) -> String {
    match expr {
        Expr::Literal(Literal::Float(v)) => format!("{}", v),
        Expr::Literal(Literal::Integer(v)) => format!("{}", v),
        Expr::Literal(Literal::Boolean(v)) => format!("{}", v),
        Expr::Literal(Literal::String(s)) => format!("{:?}", s),
        Expr::Literal(Literal::Null) => "null".to_string(),
        Expr::Column(name) => name.clone(),
        Expr::BinaryExpr { left, op, right } => {
            let op_str = match op {
                BinaryOp::Add => "+",
                BinaryOp::Subtract => "-",
                BinaryOp::Multiply => "*",
                BinaryOp::Divide => "/",
                BinaryOp::Modulo => "%",
                BinaryOp::Equal => "==",
                BinaryOp::NotEqual => "!=",
                BinaryOp::GreaterThan => ">",
                BinaryOp::GreaterThanOrEqual => ">=",
                BinaryOp::LessThan => "<",
                BinaryOp::LessThanOrEqual => "<=",
                BinaryOp::And => "&",
                BinaryOp::Or => "|",
            };
            format!(
                "({} {} {})",
                to_parseable_string(left),
                op_str,
                to_parseable_string(right)
            )
        }
        Expr::UnaryExpr { op, expr } => {
            let op_str = match op {
                UnaryOp::Negate => "-",
                UnaryOp::Not => "!",
                UnaryOp::Abs => "abs",
                UnaryOp::Sqrt => "sqrt",
                UnaryOp::Log => "log",
                UnaryOp::Exp => "exp",
            };
            format!("{}({})", op_str, to_parseable_string(expr))
        }
        Expr::FunctionCall { name, args, freq } => {
            let args_str: Vec<String> = args.iter().map(|a| to_parseable_string(a)).collect();
            let inner = format!("{}({})", name, args_str.join(", "));
            if let Some(f) = freq {
                format!("{}:{}", f.as_str(), inner)
            } else {
                inner
            }
        }
        Expr::Aggregate { op, expr, distinct } => {
            let inner = to_parseable_string(expr);
            if *distinct {
                format!("{:?}_distinct({})", op, inner)
            } else {
                format!("{:?}({})", op, inner)
            }
        }
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            format!(
                "if({}, {}, {})",
                to_parseable_string(condition),
                to_parseable_string(then_expr),
                to_parseable_string(else_expr)
            )
        }
        Expr::Cast { expr, data_type } => {
            format!("cast({} as {:?})", to_parseable_string(expr), data_type)
        }
    }
}

/// A maintained pool of diverse alpha factors with redundancy filtering and decay detection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FactorPool {
    /// Entries sorted by RankIC descending
    entries: Vec<PoolEntry>,
    /// Maximum pool capacity
    max_size: usize,
    /// Structural similarity threshold for outright rejection (default 0.95)
    reject_threshold: f64,
    /// Structural similarity threshold for flagging (default 0.80)
    flag_threshold: f64,
    /// Minimum correlations (absolute) for redundancy filtering
    correlation_threshold: f64,
}

impl FactorPool {
    /// Create a new factor pool with given max capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_size,
            reject_threshold: 0.95,
            flag_threshold: 0.80,
            correlation_threshold: 0.7,
        }
    }

    /// Number of entries currently in the pool.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get a reference to all pool entries.
    pub fn entries(&self) -> &[PoolEntry] {
        &self.entries
    }

    /// Attempt to admit a factor based on IC and structural similarity
    /// against the pool's stored expressions.
    pub fn try_admit_parsed(
        &mut self,
        expr: &Expr,
        ic: f64,
        rank_ic: f64,
        pool_expressions: &[Expr],
        now: u64,
    ) -> AdmissionResult {
        // 1. Structural redundancy check
        if let Some(max_sim) = check_redundancy(expr, pool_expressions) {
            if max_sim >= self.reject_threshold {
                return AdmissionResult::RejectedDuplicate(max_sim);
            }
            if max_sim >= self.flag_threshold {
                // Accept but flag
                self.insert_entry(expr, ic, rank_ic, now);
                return AdmissionResult::Flagged(max_sim);
            }
        }

        // 2. Pool capacity check — if full, remove worst if new is better
        if self.entries.len() >= self.max_size {
            let min_rank_ic = self
                .entries
                .last()
                .map(|e| e.rank_ic)
                .unwrap_or(f64::NEG_INFINITY);
            if rank_ic <= min_rank_ic {
                return AdmissionResult::RejectedBelowMinimum;
            }
            self.entries.pop(); // remove worst
        }

        self.insert_entry(expr, ic, rank_ic, now);
        AdmissionResult::Added
    }

    fn insert_entry(&mut self, expr: &Expr, ic: f64, rank_ic: f64, now: u64) {
        let entry = PoolEntry {
            expression: to_parseable_string(expr),
            ic,
            rank_ic,
            added_at: now,
            last_check_at: now,
            survival_rounds: 0,
        };
        // Insert in descending RankIC order
        let pos = self
            .entries
            .binary_search_by(|e| e.rank_ic.partial_cmp(&rank_ic).unwrap().reverse())
            .unwrap_or_else(|i| i);
        self.entries.insert(pos, entry);
    }

    /// Prune the pool to max_size, keeping the best by RankIC.
    pub fn prune(&mut self) {
        self.entries.truncate(self.max_size);
    }

    /// Mark surviving factors and remove decayed ones.
    /// `decayed_indices` contains indices of factors that failed re-validation.
    pub fn remove_decayed(&mut self, decayed_indices: &[usize]) {
        // Remove in reverse order to preserve indices
        let mut sorted: Vec<usize> = decayed_indices.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        for idx in sorted {
            if idx < self.entries.len() {
                self.entries.remove(idx);
            }
        }
    }

    /// Bump survival rounds for all entries (called after a validation pass).
    pub fn bump_survival(&mut self) {
        for entry in &mut self.entries {
            entry.survival_rounds += 1;
        }
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

    let mut population = if config.use_diverse_init {
        generator.generate_diverse_population(config.population_size, rng)
    } else {
        generator.generate_initial_population(config.population_size, rng)
    };
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
                let (c1, c2) =
                    tree_ops::subtree_crossover(&next_population[i], &next_population[i + 1], rng);
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
                    next_population[i] = tree_ops::subtree_mutate_feedback(
                        &next_population[i],
                        &generator,
                        &feedback,
                        config.max_depth,
                        rng,
                    );
                }
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

        if generation % 10 == 0 {
            println!(
                "Generation {}: best fitness = {:.6}, unique families = {}",
                generation,
                scores[best_idx],
                family_usage.len()
            );
        }
    }

    (population[best_idx].clone(), scores[best_idx])
}

/// Multi-factor expression complexity.
///
/// Combines structural size, parameter count, and feature diversity
/// to provide a richer overfitting signal than node count alone.
#[derive(Debug, Clone)]
pub struct Complexity {
    /// Total node count in the expression tree
    pub node_count: usize,
    /// Number of free (non-column) constants (literal numbers + ephemerals)
    pub free_param_count: usize,
    /// Number of distinct columns referenced
    pub unique_column_count: usize,
}

impl Complexity {
    /// Weighted complexity score. Default weights: (0.5, 0.3, 0.2).
    pub fn score(&self, weights: Option<(f64, f64, f64)>) -> f64 {
        let (w1, w2, w3) = weights.unwrap_or((0.5, 0.3, 0.2));
        w1 * self.node_count as f64
            + w2 * self.free_param_count as f64
            + w3 * self.unique_column_count as f64
    }
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
    /// Create a new multi-objective fitness with weighted sum.
    /// Complexity penalty uses the weighted multi-factor score.
    pub fn new(
        ic_score: f64,
        ir_score: f64,
        turnover: f64,
        complexity: &Complexity,
        weights: Option<(f64, f64, f64, f64)>, // (ic_weight, ir_weight, turnover_weight, complexity_weight)
    ) -> Self {
        let (w_ic, w_ir, w_to, w_comp) = weights.unwrap_or((0.5, 0.3, 0.1, 0.1));

        // Minimum IC threshold - penalize factors with near-zero IC
        let min_ic_threshold = 0.001;
        let ic_penalty = if ic_score < min_ic_threshold {
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
        let turnover_penalty = (-turnover / 0.1).exp();

        // Complexity penalty using multi-factor score (logarithmic on weighted score)
        let complexity_score = complexity.score(None);
        let complexity_penalty = if complexity_score > 1.0 {
            complexity_score.ln() / 10.0
        } else {
            0.0
        };

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
            complexity.node_count,
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
            &complexity,
            Some((w_ic, w_ir, w_to, w_comp)),
        ))
    }

    /// Evaluate expression to get factor matrix
    fn evaluate_expression(&self, expr: &Expr) -> Option<Array2<f64>> {
        use ndarray::Array1;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let n_days = self.returns.shape()[0];
        let n_assets = self.returns.shape()[1];

        // Parallel evaluation across assets
        let results: Vec<Vec<f64>> = (0..n_assets)
            .into_par_iter()
            .map(|asset_idx| {
                let mut columns: HashMap<String, Array1<f64>> = HashMap::new();

                for (col_name, array) in &self.data {
                    let column_data = array.column(asset_idx).to_owned();
                    columns.insert(col_name.clone(), column_data);
                }

                // Pre-populate cache with column hashes
                let mut cache: HashMap<u64, Array1<f64>> = HashMap::new();
                for (name, arr) in &columns {
                    let mut hasher = DefaultHasher::new();
                    0u8.hash(&mut hasher);
                    name.hash(&mut hasher);
                    cache.insert(hasher.finish(), arr.clone());
                }

                match eval_expr_vectorized(expr, &columns, &mut cache) {
                    Ok(arr) => arr.to_vec(),
                    Err(_) => vec![f64::NAN; n_days],
                }
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
        let config = BacktestConfig {
            quantiles: 10,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: self.fee_config.clone(),
            limit_up_down_config: Default::default(),
            position_config: self.position_config.clone(),
        };

        let engine = BacktestEngine::with_config(config);

        // Create close, open, vwap, and tradable as placeholder (using returns as proxy)
        let close = Array2::from_elem(returns.dim(), 1.0);
        let open = Array2::from_elem(returns.dim(), 1.0);
        let vwap = Array2::from_elem(returns.dim(), 1.0);
        let tradable = Array2::from_elem(returns.dim(), 1.0);
        // adj_factor is now required - use ones as default
        let adj_factor = Array2::from_elem(returns.dim(), 1.0);

        match engine.run(
            factor.clone(),
            returns.clone(),
            adj_factor,
            close,
            open,
            vwap,
            tradable,
        ) {
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

    /// Compute multi-factor complexity of an expression.
    fn compute_complexity(&self, expr: &Expr) -> Complexity {
        let mut cols = std::collections::HashSet::new();
        let (node_count, free_params) = self.compute_complexity_inner(expr, &mut cols);
        Complexity {
            node_count,
            free_param_count: free_params,
            unique_column_count: cols.len(),
        }
    }

    /// Walk the tree, counting nodes and free parameters. Collects column names.
    fn compute_complexity_inner(
        &self,
        expr: &Expr,
        cols: &mut std::collections::HashSet<String>,
    ) -> (usize, usize) {
        match expr {
            Expr::Literal(Literal::Float(_) | Literal::Integer(_)) => (1, 1),
            Expr::Literal(_) => (1, 0),
            Expr::Column(name) => {
                cols.insert(name.clone());
                (1, 0)
            }
            Expr::UnaryExpr { expr, .. } => {
                let (n, p) = self.compute_complexity_inner(expr, cols);
                (1 + n, p)
            }
            Expr::BinaryExpr { left, right, .. } => {
                let (nl, pl) = self.compute_complexity_inner(left, cols);
                let (nr, pr) = self.compute_complexity_inner(right, cols);
                (1 + nl + nr, pl + pr)
            }
            Expr::FunctionCall { args, .. } => {
                let mut total_n = 1;
                let mut total_p = 0;
                for arg in args {
                    let (n, p) = self.compute_complexity_inner(arg, cols);
                    total_n += n;
                    total_p += p;
                }
                (total_n, total_p)
            }
            Expr::Aggregate { expr, .. } => {
                let (n, p) = self.compute_complexity_inner(expr, cols);
                (1 + n, p)
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let (nc, pc) = self.compute_complexity_inner(condition, cols);
                let (nt, pt) = self.compute_complexity_inner(then_expr, cols);
                let (ne, pe) = self.compute_complexity_inner(else_expr, cols);
                (1 + nc + nt + ne, pc + pt + pe)
            }
            Expr::Cast { expr, .. } => {
                let (n, p) = self.compute_complexity_inner(expr, cols);
                (1 + n, p)
            }
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
                let complexity_penalty = (complexity.node_count as f64).powi(2) / 1000.0;

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

/// Normalize an expression for semantic deduplication.
/// Replaces all numeric constants with a placeholder to group structurally similar
/// expressions that only differ in constant values (e.g. `rank(close/5)` vs `rank(close/10)`).
fn normalize_expression(expr: &Expr) -> String {
    match expr {
        Expr::Literal(Literal::Float(_)) => "#".to_string(),
        Expr::Literal(Literal::Integer(_)) => "#".to_string(),
        Expr::Literal(lit) => format!("{:?}", lit),
        Expr::Column(name) => format!("#{}", name),
        Expr::BinaryExpr { left, op, right } => {
            format!(
                "({} {:?} {})",
                normalize_expression(left),
                op,
                normalize_expression(right)
            )
        }
        Expr::UnaryExpr { op, expr } => {
            format!("({:?} {})", op, normalize_expression(expr))
        }
        Expr::FunctionCall { name, args, freq } => {
            let args_str: Vec<String> = args.iter().map(|a| normalize_expression(a)).collect();
            if let Some(f) = freq {
                format!("{}:{}({})", f.as_str(), name, args_str.join(","))
            } else {
                format!("{}({})", name, args_str.join(","))
            }
        }
        Expr::Aggregate { op, expr, distinct } => {
            if *distinct {
                format!("{:?}_distinct({})", op, normalize_expression(expr))
            } else {
                format!("{:?}({})", op, normalize_expression(expr))
            }
        }
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            format!(
                "if {} then {} else {}",
                normalize_expression(condition),
                normalize_expression(then_expr),
                normalize_expression(else_expr)
            )
        }
        Expr::Cast { expr, data_type } => {
            format!("cast({} as {:?})", normalize_expression(expr), data_type)
        }
    }
}

/// Cached fitness evaluator with LRU cache and semantic deduplication.
pub struct CachedFitnessEvaluator<E: FitnessEvaluator> {
    evaluator: E,
    cache: Mutex<LruCache<String, f64>>,
    /// Maps normalized expression strings to fitness values for deduplication
    dedup: Mutex<HashMap<String, f64>>,
    /// Count of evaluations skipped due to semantic deduplication
    dedup_hits: atomic::AtomicU64,
    /// Count of total fitness evaluations (cache misses only)
    total_evals: atomic::AtomicU64,
}

impl<E: FitnessEvaluator> CachedFitnessEvaluator<E> {
    /// Create a new cached evaluator with given capacity
    pub fn new(evaluator: E, capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap_or(NonZeroUsize::new(100).unwrap());
        Self {
            evaluator,
            cache: Mutex::new(LruCache::new(cap)),
            dedup: Mutex::new(HashMap::new()),
            dedup_hits: atomic::AtomicU64::new(0),
            total_evals: atomic::AtomicU64::new(0),
        }
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
        self.dedup.lock().unwrap().clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Get deduplication hit count
    pub fn dedup_hits(&self) -> u64 {
        self.dedup_hits.load(atomic::Ordering::Relaxed)
    }

    /// Get total evaluations (cache misses)
    pub fn total_evals(&self) -> u64 {
        self.total_evals.load(atomic::Ordering::Relaxed)
    }

    /// Get cache hit rate statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.lock().unwrap();
        (cache.len(), cache.cap().get())
    }
}

impl<E: FitnessEvaluator> FitnessEvaluator for CachedFitnessEvaluator<E> {
    fn fitness(&self, expr: &Expr) -> f64 {
        let key = format!("{:?}", expr);

        // Check exact cache first
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(&cached) = cache.get(&key) {
                return cached;
            }
        }

        // Check semantic deduplication
        let normalized = normalize_expression(expr);
        {
            let dedup = self.dedup.lock().unwrap();
            if let Some(&cached) = dedup.get(&normalized) {
                // Found structurally equivalent expression — reuse fitness
                self.dedup_hits.fetch_add(1, atomic::Ordering::Relaxed);
                let mut cache = self.cache.lock().unwrap();
                cache.put(key, cached);
                return cached;
            }
        }

        // Compute fitness
        self.total_evals.fetch_add(1, atomic::Ordering::Relaxed);
        let fitness = self.evaluator.fitness(expr);

        // Store in both caches
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(key, fitness);
        }
        {
            let mut dedup = self.dedup.lock().unwrap();
            dedup.insert(normalized, fitness);
        }

        fitness
    }

    fn fitness_batch(&self, exprs: &[Expr]) -> Vec<f64> {
        let mut results = vec![0.0; exprs.len()];
        let mut to_compute: Vec<(usize, Expr, String, String)> = Vec::new();

        // First pass: check both caches
        {
            let mut cache = self.cache.lock().unwrap();
            let dedup = self.dedup.lock().unwrap();

            for (idx, expr) in exprs.iter().enumerate() {
                let key = format!("{:?}", expr);
                if let Some(&cached) = cache.get(&key) {
                    results[idx] = cached;
                } else {
                    let normalized = normalize_expression(expr);
                    if let Some(&cached) = dedup.get(&normalized) {
                        cache.put(key, cached);
                        self.dedup_hits.fetch_add(1, atomic::Ordering::Relaxed);
                        results[idx] = cached;
                    } else {
                        to_compute.push((idx, expr.clone(), key, normalized));
                    }
                }
            }
        }

        // Compute missing fitness values
        if !to_compute.is_empty() {
            let exprs_to_compute: Vec<Expr> = to_compute
                .iter()
                .map(|(_, expr, _, _)| expr.clone())
                .collect();
            let computed = self.evaluator.fitness_batch(&exprs_to_compute);
            let n_computed = computed.len();

            // Update caches and results
            {
                let mut cache = self.cache.lock().unwrap();
                let mut dedup = self.dedup.lock().unwrap();

                for ((idx, _, key, normalized), fitness) in
                    to_compute.into_iter().zip(computed.into_iter())
                {
                    cache.put(key, fitness);
                    dedup.insert(normalized, fitness);
                    results[idx] = fitness;
                }
            }

            self.total_evals
                .fetch_add(n_computed as u64, atomic::Ordering::Relaxed);
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
