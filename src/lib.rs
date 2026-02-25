//! Alpha Expression System
//!
//! A library for building and optimizing expression graphs inspired by
//! relational algebra and Polars' expression API, with extensions for
//! alpha computation and symbolic regression.

pub mod alpha;
pub mod evaluation;
pub mod expr;
pub mod logical_plan;
pub mod optimizer;
pub mod symbolic_regression;

// Re-exports for common usage
pub use expr::Expr;
pub use logical_plan::LogicalPlan;
pub use optimizer::Optimizer;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::expr::*;
    pub use crate::logical_plan::*;
    pub use crate::optimizer::*;
    pub use crate::evaluation::*;
    pub use crate::alpha::*;
    pub use crate::symbolic_regression::*;
}