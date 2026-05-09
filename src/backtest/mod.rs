//! Backtest engine module
//!
//! Provides high-performance backtesting capabilities for alpha factors.

pub mod config;
pub mod engine;
pub mod metrics;
pub mod portfolio;
pub mod types;

pub use config::{BacktestConfig, ExecConfig, FeeConfig, PositionConfig, SlippageConfig};
pub use engine::BacktestEngine;
pub use types::BacktestResult;
