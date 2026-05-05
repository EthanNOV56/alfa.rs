//! Data abstraction layer module
//!
//! This module provides data source abstractions and implementations
//! for connecting to various data backends.

pub mod clickhouse;
pub mod convention;
pub mod frequency;
pub mod layer;
pub mod source;

pub use clickhouse::ClickHouseSource;
pub use convention::*;
pub use layer::{DataLayer, PriceMatrix};
pub use source::{DataError, DataSource, QueryFilter};
