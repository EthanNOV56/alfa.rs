//! Expression system module - AST, optimization, and factor registry
//!
//! This module provides the core expression system including:
//! - AST (Abstract Syntax Tree) for factor expressions
//! - Expression optimization (constant folding, CSE)
//! - Factor registry for managing factor definitions

pub mod ast;
pub mod optimizer;
pub mod registry;

// Re-exports from submodules
pub use ast::{BinaryOp, DataType, Dimension, Expr, Literal, UnaryOp};
pub use optimizer::{ExpressionOptimizer, optimize_expression};
pub use registry::{
    ColumnMeta, ComputeConfig, FactorInfo, FactorRegistry, FactorResult, parse_expression,
};
