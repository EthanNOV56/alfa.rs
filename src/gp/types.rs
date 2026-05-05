//! Core types for the GP module.
//!
//! This module collects all data types, configuration structs,
//! expression serialization helpers, and operator feedback tracking
//! used across the GP engine.

use crate::backtest::BacktestResult;
use crate::expr::{BinaryOp, Expr, Literal, UnaryOp};
use std::collections::HashMap;

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
    #[serde(default)]
    pub use_frequencies: bool,
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
            use_frequencies: false,
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
    pub(crate) fn terminal_weight(&self, terminal: &Terminal) -> f64 {
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
    pub(crate) fn function_weight(&self, name: &str) -> f64 {
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

/// Normalize an expression for semantic deduplication.
/// Replaces all numeric constants with a placeholder to group structurally similar
/// expressions that only differ in constant values (e.g. `rank(close/5)` vs `rank(close/10)`).
pub(crate) fn normalize_expression(expr: &Expr) -> String {
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

/// Per-generation feedback tracking operator performance during a GP run.
#[derive(Default)]
pub(crate) struct OperatorFeedback {
    /// (cumulative fitness, appearance count) per function name
    func_stats: HashMap<String, (f64, u32)>,
}

impl OperatorFeedback {
    /// Attribute an individual's fitness to the functions it contains.
    pub(crate) fn record(&mut self, expr: &Expr, fitness: f64) {
        let mut funcs = Vec::new();
        collect_function_names(expr, &mut funcs);
        for name in funcs {
            let entry = self.func_stats.entry(name).or_insert((0.0, 0));
            entry.0 += fitness;
            entry.1 += 1;
        }
    }

    /// Get average fitness per function (0.0 if unseen).
    pub(crate) fn avg_fitness(&self, func_name: &str) -> f64 {
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
    pub(crate) fn reset(&mut self) {
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
