//! Types module - Custom data types for the library
//!
//! This module provides Polars-style Series and DataFrame wrappers
//! for vectorized operations on time series data.

pub mod dataframe;
pub mod series;

// Re-exports
pub use dataframe::{DataFrame, evaluate_expr_on_dataframe, evaluate_expr_on_dataframe_optimized, optimize_expr_for_evaluation, CachedExpressionEvaluator, col, lit_float, lit_int, df_from_arrays, create_backtest_dataframe};
pub use series::Series;
