//! Data abstraction layer module
//!
//! This module provides data source abstractions and implementations
//! for connecting to various data backends.

pub mod clickhouse;
pub mod convention;
pub mod frequency;
pub mod layer;
pub mod pool;
pub mod source;

pub use clickhouse::ClickHouseSource;
pub use convention::*;
pub use layer::{DataLayer, PriceMatrix};
pub use pool::{CachePolicy, DataPool, DataPoolConfig};
pub use source::{DataError, QueryFilter};
