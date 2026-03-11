//! Types module - Custom data types for the library
//!
//! This module provides Polars-style Series and DataFrame wrappers
//! for vectorized operations on time series data.

pub mod dataframe;
pub mod series;

// Re-exports
pub use dataframe::{
    CachedExpressionEvaluator, DataFrame, col, create_backtest_dataframe, df_from_arrays,
    evaluate_expr_on_dataframe, evaluate_expr_on_dataframe_optimized, lit_float, lit_int,
    optimize_expr_for_evaluation,
};
pub use series::Series;
