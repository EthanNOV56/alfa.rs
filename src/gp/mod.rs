//! Genetic Programming module for factor mining
//!
//! This module provides GP-based factor mining functionality including:
//! - Expression tree generation and evolution
//! - Fitness evaluation through backtesting
//! - Meta-learning for intelligent GP recommendations

pub mod evolution;
pub mod fitness;
pub mod generator;
pub mod history;
pub mod metalearning;
pub mod operators;
pub mod pool;
pub mod types;

// Re-exports from types
pub(crate) use types::to_parseable_string;
pub use types::{
    AdmissionResult, DataSplit, DataSplitConfig, Function, GPConfig, MultiObjectiveFitness,
    PoolEntry, SplitEvaluationResult, SplitMetrics, Terminal,
};

// Re-exports from pool
pub use pool::FactorPool;

// Re-exports from generator
pub use generator::ExpressionGenerator;

// Re-exports from operators
pub use operators::{check_redundancy, expr_structural_similarity};

// Re-exports from fitness
pub use fitness::{
    BacktestFitnessEvaluator, BatchFitnessEvaluator, CachedFitnessEvaluator,
    RealBacktestFitnessEvaluator,
};

// Re-exports from evolution
pub use evolution::run_gp;

// Re-exports from history
pub use history::{GPHistoryRecord, PopulationStats, create_gp_history_record};

// Re-exports from metalearning
pub use metalearning::{GPRecommendations, MetaLearningAnalyzer};
