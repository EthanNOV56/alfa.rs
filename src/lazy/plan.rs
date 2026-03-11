//! Data types for lazy evaluation
//!
//! This module contains the core data structures used in the lazy evaluation system.

use std::collections::HashMap;
use std::sync::Arc;

use crate::expr::Expr;
use crate::types::DataFrame;
use ndarray::Array2;

// ============================================================================
// Logical Plan
// ============================================================================

/// Logical plan node for lazy evaluation
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Scan data source (e.g., from numpy arrays, CSV, etc.)
    Scan {
        source: DataSource,
        projection: Option<Vec<String>>,
        selection: Option<Expr>,
    },
    /// Project new columns (transformations)
    Projection {
        input: Arc<LogicalPlan>,
        exprs: Vec<(String, Expr)>, // (column_name, expression)
    },
    /// Filter rows based on predicate
    Filter {
        input: Arc<LogicalPlan>,
        predicate: Expr,
    },
    /// Window operations (rolling, expanding, etc.)
    Window {
        input: Arc<LogicalPlan>,
        expr: Expr,
        window_spec: WindowSpec,
        output_name: String,
    },
    /// Stateful operations (cumulative, etc.)
    Stateful {
        input: Arc<LogicalPlan>,
        expr: StatefulExpr,
        output_name: String,
    },
    /// Cache intermediate results
    Cache {
        input: Arc<LogicalPlan>,
        key: String,
    },
    /// Join multiple data sources
    Join {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        on: Vec<String>,
        how: JoinType,
    },
}

// ============================================================================
// Data Source
// ============================================================================

/// Data source for scanning
#[derive(Debug, Clone)]
pub enum DataSource {
    /// In-memory numpy arrays (n_days × n_assets)
    NumpyArrays(HashMap<String, Array2<f64>>),
    /// DataFrame collection (per-asset DataFrames)
    DataFrames(HashMap<String, DataFrame>),
    /// External source (CSV, parquet, database)
    External { path: String, format: DataFormat },
}

/// Data format for external sources
#[derive(Debug, Clone)]
pub enum DataFormat {
    Csv,
    Parquet,
    Database,
}

// ============================================================================
// Window Specification
// ============================================================================

/// Window specification
#[derive(Debug, Clone)]
pub struct WindowSpec {
    pub kind: WindowKind,
    pub size: Option<usize>,
    pub min_periods: usize,
}

/// Window kind
#[derive(Debug, Clone, Copy)]
pub enum WindowKind {
    Rolling,
    Expanding,
    RollingOffset,
}

// ============================================================================
// Stateful Expressions
// ============================================================================

/// Stateful expressions (require maintaining state across rows)
#[derive(Debug, Clone)]
pub enum StatefulExpr {
    CumSum(Expr),
    CumProd(Expr),
    CumMax(Expr),
    CumMin(Expr),
    Ema(Expr, f64), // expression, alpha
}

// ============================================================================
// Join Type
// ============================================================================

/// Join type
#[derive(Debug, Clone, Copy)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Outer,
}

// ============================================================================
// Optimization Level
// ============================================================================

/// Optimization level
#[derive(Debug, Clone, Copy)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimizations (constant folding, simple CSE)
    Basic,
    /// Default optimizations (most useful optimizations)
    Default,
    /// Aggressive optimizations (may take longer to optimize)
    Aggressive,
}
