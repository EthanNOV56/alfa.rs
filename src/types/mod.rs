//! Types module - Custom data types for the library
//!
//! This module provides Polars-style Series and DataFrame wrappers
//! for vectorized operations on time series data.

pub mod polars_style;

// Re-exports
pub use polars_style::{DataFrame, Series, evaluate_expr_on_dataframe};
