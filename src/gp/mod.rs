//! Genetic Programming module for factor mining
//!
//! This module provides GP-based factor mining functionality including:
//! - Expression tree generation and evolution
//! - Fitness evaluation through backtesting
//! - Meta-learning for intelligent GP recommendations

pub mod engine;
pub mod history;
pub mod metalearning;

// Re-exports from engine
pub use engine::{
    AdmissionResult, BacktestFitnessEvaluator, BatchFitnessEvaluator, CachedFitnessEvaluator,
    DataSplit, DataSplitConfig, ExpressionGenerator, FactorPool, Function, GPConfig,
    MultiObjectiveFitness, PoolEntry, RealBacktestFitnessEvaluator, SplitEvaluationResult,
    SplitMetrics, Terminal, check_redundancy, expr_structural_similarity, run_gp,
};

// Re-exports from history
pub use history::{GPHistoryRecord, PopulationStats, create_gp_history_record};

// Re-exports from metalearning
pub use metalearning::{GPRecommendations, MetaLearningAnalyzer};
