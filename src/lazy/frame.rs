//! LazyFrame and LazyFrameBuilder
//!
//! Provides the main API for building and executing lazy computation plans.

use std::sync::Arc;

use ndarray::Array2;

use crate::expr::Expr;

use super::plan::*;
use super::optimizer::LazyOptimizer;
use super::executor::LazyExecutor;

// ============================================================================
// LazyFrame
// ============================================================================

/// LazyFrame builder (similar to Polars LazyFrame)
#[derive(Clone)]
pub struct LazyFrame {
    pub(crate) logical_plan: Arc<LogicalPlan>,
    pub(crate) optimization_level: OptimizationLevel,
}

impl LazyFrame {
    /// Create a new LazyFrame from a data source
    pub fn scan(source: DataSource) -> Self {
        LazyFrame {
            logical_plan: Arc::new(LogicalPlan::Scan {
                source,
                projection: None,
                selection: None,
            }),
            optimization_level: OptimizationLevel::Default,
        }
    }

    /// Add new columns (projections)
    pub fn with_columns<I, S>(self, exprs: I) -> Self
    where
        I: IntoIterator<Item = (S, Expr)>,
        S: Into<String>,
    {
        let exprs_vec: Vec<(String, Expr)> = exprs
            .into_iter()
            .map(|(name, expr)| (name.into(), expr))
            .collect();

        LazyFrame {
            logical_plan: Arc::new(LogicalPlan::Projection {
                input: self.logical_plan,
                exprs: exprs_vec,
            }),
            optimization_level: self.optimization_level,
        }
    }

    /// Filter rows
    pub fn filter(self, predicate: Expr) -> Self {
        LazyFrame {
            logical_plan: Arc::new(LogicalPlan::Filter {
                input: self.logical_plan,
                predicate,
            }),
            optimization_level: self.optimization_level,
        }
    }

    /// Add a window operation
    pub fn with_window(self, expr: Expr, window_spec: WindowSpec, output_name: &str) -> Self {
        LazyFrame {
            logical_plan: Arc::new(LogicalPlan::Window {
                input: self.logical_plan,
                expr,
                window_spec,
                output_name: output_name.to_string(),
            }),
            optimization_level: self.optimization_level,
        }
    }

    /// Add a stateful operation
    pub fn with_stateful(self, expr: StatefulExpr, output_name: &str) -> Self {
        LazyFrame {
            logical_plan: Arc::new(LogicalPlan::Stateful {
                input: self.logical_plan,
                expr,
                output_name: output_name.to_string(),
            }),
            optimization_level: self.optimization_level,
        }
    }

    /// Cache intermediate result
    pub fn cache(self, key: Option<&str>) -> Self {
        let cache_key = key.map(|k| k.to_string()).unwrap_or_else(|| {
            format!(
                "cache_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            )
        });

        LazyFrame {
            logical_plan: Arc::new(LogicalPlan::Cache {
                input: self.logical_plan,
                key: cache_key,
            }),
            optimization_level: self.optimization_level,
        }
    }

    /// Join with another LazyFrame
    pub fn join(self, other: LazyFrame, on: Vec<String>, how: JoinType) -> Self {
        LazyFrame {
            logical_plan: Arc::new(LogicalPlan::Join {
                left: self.logical_plan,
                right: other.logical_plan,
                on,
                how,
            }),
            optimization_level: self.optimization_level,
        }
    }

    /// Set optimization level
    pub fn optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Explain the logical plan
    pub fn explain(&self, optimized: bool) -> String {
        if optimized {
            let optimizer = LazyOptimizer::new(self.optimization_level);
            let optimized_plan = optimizer.optimize(self.logical_plan.as_ref().clone());
            format!("{:?}", optimized_plan)
        } else {
            format!("{:?}", self.logical_plan)
        }
    }

    /// Collect (execute) the lazy computation
    pub fn collect(self) -> Result<std::collections::HashMap<String, Array2<f64>>, String> {
        let optimizer = LazyOptimizer::new(self.optimization_level);
        let optimized_plan = optimizer.optimize(self.logical_plan.as_ref().clone());

        let mut executor = LazyExecutor::new();
        executor.execute(&optimized_plan)
    }

    /// Collect and return as single factor matrix
    pub fn collect_factor(self, output_column: &str) -> Result<Array2<f64>, String> {
        let result = self.collect()?;
        result
            .get(output_column)
            .cloned()
            .ok_or_else(|| format!("Output column '{}' not found", output_column))
    }
}

// ============================================================================
// LazyFrameBuilder
// ============================================================================

/// Builder for LazyFrame with fluent API
pub struct LazyFrameBuilder {
    pub(crate) lazy_frame: Option<LazyFrame>,
}

impl LazyFrameBuilder {
    /// Start building from a scan
    pub fn scan(source: DataSource) -> Self {
        Self {
            lazy_frame: Some(LazyFrame::scan(source)),
        }
    }

    /// Add columns
    pub fn with_columns<I, S>(mut self, exprs: I) -> Self
    where
        I: IntoIterator<Item = (S, Expr)>,
        S: Into<String>,
    {
        if let Some(lf) = self.lazy_frame.take() {
            self.lazy_frame = Some(lf.with_columns(exprs));
        }
        self
    }

    /// Add filter
    pub fn filter(mut self, predicate: Expr) -> Self {
        if let Some(lf) = self.lazy_frame.take() {
            self.lazy_frame = Some(lf.filter(predicate));
        }
        self
    }

    /// Add window operation
    pub fn with_window(mut self, expr: Expr, window_spec: WindowSpec, output_name: &str) -> Self {
        if let Some(lf) = self.lazy_frame.take() {
            self.lazy_frame = Some(lf.with_window(expr, window_spec, output_name));
        }
        self
    }

    /// Add stateful operation
    pub fn with_stateful(mut self, expr: StatefulExpr, output_name: &str) -> Self {
        if let Some(lf) = self.lazy_frame.take() {
            self.lazy_frame = Some(lf.with_stateful(expr, output_name));
        }
        self
    }

    /// Add cache
    pub fn cache(mut self, key: Option<&str>) -> Self {
        if let Some(lf) = self.lazy_frame.take() {
            self.lazy_frame = Some(lf.cache(key));
        }
        self
    }

    /// Set optimization level
    pub fn optimization_level(mut self, level: OptimizationLevel) -> Self {
        if let Some(lf) = self.lazy_frame.take() {
            self.lazy_frame = Some(lf.optimization_level(level));
        }
        self
    }

    /// Join with another LazyFrame
    pub fn join(mut self, other: LazyFrame, on: Vec<String>, how: JoinType) -> Self {
        if let Some(lf) = self.lazy_frame.take() {
            self.lazy_frame = Some(lf.join(other, on, how));
        }
        self
    }

    /// Build the LazyFrame
    pub fn build(self) -> Option<LazyFrame> {
        self.lazy_frame
    }

    /// Build and collect
    pub fn collect(self) -> Result<std::collections::HashMap<String, Array2<f64>>, String> {
        if let Some(lf) = self.lazy_frame {
            lf.collect()
        } else {
            Err("No LazyFrame to collect".to_string())
        }
    }

    /// Build and collect as factor matrix
    pub fn collect_factor(self, output_column: &str) -> Result<Array2<f64>, String> {
        if let Some(lf) = self.lazy_frame {
            lf.collect_factor(output_column)
        } else {
            Err("No LazyFrame to collect".to_string())
        }
    }
}
