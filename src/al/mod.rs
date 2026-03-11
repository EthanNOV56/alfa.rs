//! Alpha parser and factor store module
//!
//! This module provides:
//! - .al file parsing for alpha factor definitions
//! - Factor storage and persistence

pub mod parser;
pub mod store;

pub use parser::AlFactor;
pub use store::FactorStore;
