//! Data abstraction layer module
//!
//! This module provides data source abstractions and implementations
//! for connecting to various data backends.

pub mod clickhouse;
pub mod derivation;
pub mod source;

pub use clickhouse::ClickHouseSource;
pub use derivation::DataDerivation;
pub use source::{DataError, DataSource};
