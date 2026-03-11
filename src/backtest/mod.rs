//! Backtest engine module
//!
//! Provides high-performance backtesting capabilities for alpha factors.

pub mod engine;

pub use engine::{BacktestConfig, BacktestEngine, BacktestResult, FeeConfig, PositionConfig, SlippageConfig};
