//! Configuration and data structures for the factor registry

use crate::lazy::LogicalPlan;
use crate::expr::ast::Expr;

/// Configuration for resource limits and timeout to prevent system overload
#[derive(Debug, Clone)]
pub struct ComputeConfig {
    /// Maximum computation time in seconds (timeout)
    pub timeout_secs: u64,
    /// Maximum number of parallel threads
    pub max_workers: usize,
    /// Maximum batch size for chunked processing
    pub batch_size: usize,
    /// Maximum memory usage estimate in MB
    pub memory_limit_mb: usize,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            max_workers: 2,
            batch_size: 5000,
            memory_limit_mb: 512,
        }
    }
}

impl ComputeConfig {
    pub fn conservative() -> Self {
        Self {
            timeout_secs: 15,
            max_workers: 1,
            batch_size: 2000,
            memory_limit_mb: 256,
        }
    }

    pub fn high_performance() -> Self {
        Self {
            timeout_secs: 120,
            max_workers: 8,
            batch_size: 50000,
            memory_limit_mb: 4096,
        }
    }
}

/// Factor information
#[derive(Debug, Clone)]
pub struct FactorInfo {
    pub name: String,
    pub expression: String,
    pub parsed_expr: Expr,
    pub plan: LogicalPlan,
    pub description: Option<String>,
    pub category: Option<String>,
}

/// Factor computation result
#[derive(Debug, Clone)]
pub struct FactorResult {
    pub name: String,
    pub values: Vec<f64>,
    pub n_rows: usize,
    pub n_cols: usize,
    pub compute_time_ms: u64,
}

/// Column metadata
#[derive(Debug, Clone)]
pub struct ColumnMeta {
    pub name: String,
    pub data_type: String,
}
