//! Lazy evaluation module
//!
//! Provides lazy evaluation engine for building and optimizing
//! factor expression execution plans.

pub mod engine;
pub mod executor;
pub mod frame;
pub mod optimizer;
pub mod plan;

pub use engine::{DataSource, JoinType, LazyFrame, LogicalPlan, expanding_window, rolling_window};
