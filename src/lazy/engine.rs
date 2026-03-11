//! Lazy evaluation system inspired by Polars' lazy API
//!
//! This module provides a lazy evaluation system for building complex factor expressions
//! with automatic optimization and efficient execution.

use std::collections::HashMap;
use std::sync::Arc;

use crate::expr::Expr;
use crate::types::DataFrame;
use ndarray::Array2;

// ============================================================================
// Logical Plan
// ============================================================================

/// Logical plan node for lazy evaluation
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Scan data source (e.g., from numpy arrays, CSV, etc.)
    Scan {
        source: DataSource,
        projection: Option<Vec<String>>,
        selection: Option<Expr>,
    },
    /// Project new columns (transformations)
    Projection {
        input: Arc<LogicalPlan>,
        exprs: Vec<(String, Expr)>, // (column_name, expression)
    },
    /// Filter rows based on predicate
    Filter {
        input: Arc<LogicalPlan>,
        predicate: Expr,
    },
    /// Window operations (rolling, expanding, etc.)
    Window {
        input: Arc<LogicalPlan>,
        expr: Expr,
        window_spec: WindowSpec,
        output_name: String,
    },
    /// Stateful operations (cumulative, etc.)
    Stateful {
        input: Arc<LogicalPlan>,
        expr: StatefulExpr,
        output_name: String,
    },
    /// Cache intermediate results
    Cache {
        input: Arc<LogicalPlan>,
        key: String,
    },
    /// Join multiple data sources
    Join {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        on: Vec<String>,
        how: JoinType,
    },
}

/// Data source for scanning
#[derive(Debug, Clone)]
pub enum DataSource {
    /// In-memory numpy arrays (n_days × n_assets)
    NumpyArrays(HashMap<String, Array2<f64>>),
    /// DataFrame collection (per-asset DataFrames)
    DataFrames(HashMap<String, DataFrame>),
    /// External source (CSV, parquet, database)
    External { path: String, format: DataFormat },
}

/// Data format for external sources
#[derive(Debug, Clone)]
pub enum DataFormat {
    Csv,
    Parquet,
    Database,
}

/// Window specification
#[derive(Debug, Clone)]
pub struct WindowSpec {
    pub kind: WindowKind,
    pub size: Option<usize>,
    pub min_periods: usize,
}

/// Window kind
#[derive(Debug, Clone, Copy)]
pub enum WindowKind {
    Rolling,
    Expanding,
    RollingOffset,
}

/// Stateful expressions (require maintaining state across rows)
#[derive(Debug, Clone)]
pub enum StatefulExpr {
    CumSum(Expr),
    CumProd(Expr),
    CumMax(Expr),
    CumMin(Expr),
    Ema(Expr, f64), // expression, alpha
}

/// Join type
#[derive(Debug, Clone, Copy)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Outer,
}

// ============================================================================
// LazyFrame
// ============================================================================

/// LazyFrame builder (similar to Polars LazyFrame)
#[derive(Clone)]
pub struct LazyFrame {
    logical_plan: Arc<LogicalPlan>,
    optimization_level: OptimizationLevel,
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
    pub fn collect(self) -> Result<HashMap<String, Array2<f64>>, String> {
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

/// Optimization level
#[derive(Debug, Clone, Copy)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimizations (constant folding, simple CSE)
    Basic,
    /// Default optimizations (most useful optimizations)
    Default,
    /// Aggressive optimizations (may take longer to optimize)
    Aggressive,
}

// ============================================================================
// Lazy Optimizer
// ============================================================================

/// Optimizer for logical plans
pub struct LazyOptimizer {
    level: OptimizationLevel,
}

impl LazyOptimizer {
    pub fn new(level: OptimizationLevel) -> Self {
        Self { level }
    }

    pub fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        match self.level {
            OptimizationLevel::None => plan,
            OptimizationLevel::Basic => self.optimize_basic(plan),
            OptimizationLevel::Default => self.optimize_default(plan),
            OptimizationLevel::Aggressive => self.optimize_aggressive(plan),
        }
    }

    fn optimize_basic(&self, plan: LogicalPlan) -> LogicalPlan {
        // Basic optimizations: constant folding and simple CSE
        let plan = self.constant_folding(plan);
        let plan = self.simple_cse(plan);
        plan
    }

    fn optimize_default(&self, plan: LogicalPlan) -> LogicalPlan {
        // Default optimizations: basic + predicate pushdown + projection pruning
        let plan = self.optimize_basic(plan);
        let plan = self.predicate_pushdown(plan);
        let plan = self.projection_pruning(plan);
        plan
    }

    fn optimize_aggressive(&self, plan: LogicalPlan) -> LogicalPlan {
        // Aggressive optimizations: default + join reordering + advanced CSE
        let plan = self.optimize_default(plan);
        let plan = self.join_reordering(plan);
        let plan = self.advanced_cse(plan);
        plan
    }

    /// Constant folding: evaluate constant expressions at optimization time
    fn constant_folding(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement constant folding
        // For now, just pass through
        plan
    }

    /// Simple common subexpression elimination
    fn simple_cse(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement simple CSE
        // For now, just pass through
        plan
    }

    /// Predicate pushdown: move filters closer to data sources
    fn predicate_pushdown(&self, plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Filter { input, predicate } => {
                let optimized_input = self.predicate_pushdown(input.as_ref().clone());

                // Try to push predicate through projection
                if let LogicalPlan::Projection {
                    input: proj_input,
                    exprs,
                } = &optimized_input
                {
                    // Check if predicate only uses columns that are preserved in projection
                    let used_columns = self.extract_column_references(&predicate);
                    let projected_columns: std::collections::HashSet<_> =
                        exprs.iter().map(|(name, _)| name.clone()).collect();

                    // Also include columns from input that are passed through unchanged
                    // For now, we assume all columns used must be in projection
                    if used_columns.is_subset(&projected_columns) {
                        // Push filter below projection
                        return LogicalPlan::Projection {
                            input: Arc::new(LogicalPlan::Filter {
                                input: proj_input.clone(),
                                predicate,
                            }),
                            exprs: exprs.clone(),
                        };
                    }
                }

                LogicalPlan::Filter {
                    input: Arc::new(optimized_input),
                    predicate,
                }
            }
            LogicalPlan::Projection { input, exprs } => LogicalPlan::Projection {
                input: Arc::new(self.predicate_pushdown(input.as_ref().clone())),
                exprs,
            },
            LogicalPlan::Window {
                input,
                expr,
                window_spec,
                output_name,
            } => LogicalPlan::Window {
                input: Arc::new(self.predicate_pushdown(input.as_ref().clone())),
                expr,
                window_spec,
                output_name,
            },
            LogicalPlan::Stateful {
                input,
                expr,
                output_name,
            } => LogicalPlan::Stateful {
                input: Arc::new(self.predicate_pushdown(input.as_ref().clone())),
                expr,
                output_name,
            },
            LogicalPlan::Cache { input, key } => LogicalPlan::Cache {
                input: Arc::new(self.predicate_pushdown(input.as_ref().clone())),
                key,
            },
            LogicalPlan::Join {
                left,
                right,
                on,
                how,
            } => LogicalPlan::Join {
                left: Arc::new(self.predicate_pushdown(left.as_ref().clone())),
                right: Arc::new(self.predicate_pushdown(right.as_ref().clone())),
                on,
                how,
            },
            LogicalPlan::Scan { .. } => plan,
        }
    }

    /// Extract column references from an expression
    fn extract_column_references(&self, expr: &Expr) -> std::collections::HashSet<String> {
        use crate::expr::Expr as E;
        let mut columns = std::collections::HashSet::new();

        match expr {
            E::Column(name) => {
                columns.insert(name.clone());
            }
            E::BinaryExpr { left, right, .. } => {
                columns.extend(self.extract_column_references(left));
                columns.extend(self.extract_column_references(right));
            }
            E::UnaryExpr { expr, .. } => {
                columns.extend(self.extract_column_references(expr));
            }
            E::FunctionCall { args, .. } => {
                for arg in args {
                    columns.extend(self.extract_column_references(arg));
                }
            }
            E::Aggregate { expr, .. } => {
                columns.extend(self.extract_column_references(expr));
            }
            E::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                columns.extend(self.extract_column_references(condition));
                columns.extend(self.extract_column_references(then_expr));
                columns.extend(self.extract_column_references(else_expr));
            }
            E::Cast { expr, .. } => {
                columns.extend(self.extract_column_references(expr));
            }
            E::Literal(_) => {}
        }

        columns
    }

    /// Projection pruning: remove unused columns from projections
    fn projection_pruning(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement projection pruning
        // For now, just pass through
        plan
    }

    /// Join reordering
    fn join_reordering(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement join reordering
        // For now, just pass through
        plan
    }

    /// Advanced common subexpression elimination
    fn advanced_cse(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement advanced CSE
        // For now, just pass through
        plan
    }
}

// ============================================================================
// Lazy Executor
// ============================================================================

/// Executor for logical plans
pub struct LazyExecutor {
    cache: HashMap<String, HashMap<String, Array2<f64>>>,
}

impl LazyExecutor {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn execute(&mut self, plan: &LogicalPlan) -> Result<HashMap<String, Array2<f64>>, String> {
        match plan {
            LogicalPlan::Scan {
                source,
                projection,
                selection,
            } => self.execute_scan(source, projection.as_ref(), selection.as_ref()),
            LogicalPlan::Projection { input, exprs } => self.execute_projection(input, exprs),
            LogicalPlan::Filter { input, predicate } => self.execute_filter(input, predicate),
            LogicalPlan::Window {
                input,
                expr,
                window_spec,
                output_name,
            } => self.execute_window(input, expr, window_spec, output_name),
            LogicalPlan::Stateful {
                input,
                expr,
                output_name,
            } => self.execute_stateful(input, expr, output_name),
            LogicalPlan::Cache { input, key } => self.execute_cache(input, key),
            LogicalPlan::Join {
                left,
                right,
                on,
                how,
            } => self.execute_join(left, right, on, *how),
        }
    }

    fn execute_scan(
        &self,
        source: &DataSource,
        projection: Option<&Vec<String>>,
        _selection: Option<&Expr>,
    ) -> Result<HashMap<String, Array2<f64>>, String> {
        match source {
            DataSource::NumpyArrays(arrays) => {
                // Apply projection if specified
                let mut result = HashMap::new();
                if let Some(proj_cols) = projection {
                    for col in proj_cols {
                        if let Some(array) = arrays.get(col) {
                            result.insert(col.clone(), array.clone());
                        } else {
                            return Err(format!("Column '{}' not found in data source", col));
                        }
                    }
                } else {
                    // Include all columns
                    result.extend(arrays.clone());
                }

                // TODO: Apply selection (filtering)
                // This would require evaluating the predicate on the data

                Ok(result)
            }
            DataSource::DataFrames(_) => {
                // TODO: Implement DataFrame source execution
                Err("DataFrame source not yet implemented".to_string())
            }
            DataSource::External { .. } => Err("External sources not yet implemented".to_string()),
        }
    }

    fn execute_projection(
        &mut self,
        input: &Arc<LogicalPlan>,
        exprs: &[(String, Expr)],
    ) -> Result<HashMap<String, Array2<f64>>, String> {
        // Execute input plan
        let input_data = self.execute(input)?;

        // Get dimensions from first array
        let (n_days, n_assets) = if let Some(first_array) = input_data.values().next() {
            first_array.dim()
        } else {
            return Err("No input data for projection".to_string());
        };

        // Create result map starting with input data
        let mut result = input_data.clone();

        // For each new column to compute
        for (col_name, expr) in exprs {
            // Create result array for this column
            let mut column_result = Array2::<f64>::zeros((n_days, n_assets));

            // Process each asset in parallel
            use rayon::prelude::*;

            // Clone necessary data for parallel processing
            let input_data_clone = input_data.clone();
            let expr_clone = expr.clone();

            // Process assets in parallel
            let asset_results: Vec<_> = (0..n_assets)
                .into_par_iter()
                .map(|asset_idx| {
                    // Create DataFrame for this asset
                    let mut columns = std::collections::HashMap::new();

                    for (col_name, array) in &input_data_clone {
                        let column_data = array.column(asset_idx).to_owned();
                        columns.insert(
                            col_name.clone(),
                            crate::types::Series::new(column_data.to_vec()),
                        );
                    }

                    // Create DataFrame and evaluate expression
                    if let Ok(df) = crate::types::DataFrame::from_series_map(columns) {
                        if let Ok(series) =
                            crate::types::evaluate_expr_on_dataframe(&expr_clone, &df)
                        {
                            return series.data().to_vec();
                        }
                    }

                    // Return NaN-filled vector if evaluation failed
                    vec![f64::NAN; n_days]
                })
                .collect();

            // Assemble results into the 2D array
            for (asset_idx, asset_result) in asset_results.into_iter().enumerate() {
                for day_idx in 0..n_days {
                    column_result[[day_idx, asset_idx]] = asset_result[day_idx];
                }
            }

            // Add computed column to result
            result.insert(col_name.clone(), column_result);
        }

        Ok(result)
    }

    fn execute_filter(
        &mut self,
        input: &Arc<LogicalPlan>,
        predicate: &Expr,
    ) -> Result<HashMap<String, Array2<f64>>, String> {
        // Execute input plan
        let input_data = self.execute(input)?;

        // Get dimensions from first array
        let (n_days, n_assets) = if let Some(first_array) = input_data.values().next() {
            first_array.dim()
        } else {
            return Err("No input data for filtering".to_string());
        };

        // Create result array for the boolean mask
        let mut mask_array = Array2::<f64>::zeros((n_days, n_assets));

        // Process each asset in parallel to compute the predicate
        use rayon::prelude::*;

        // Clone necessary data for parallel processing
        let input_data_clone = input_data.clone();
        let predicate_clone = predicate.clone();

        // Process assets in parallel
        let asset_masks: Vec<_> = (0..n_assets)
            .into_par_iter()
            .map(|asset_idx| {
                // Create DataFrame for this asset
                let mut columns = std::collections::HashMap::new();

                for (col_name, array) in &input_data_clone {
                    let column_data = array.column(asset_idx).to_owned();
                    columns.insert(
                        col_name.clone(),
                        crate::types::Series::new(column_data.to_vec()),
                    );
                }

                // Create DataFrame and evaluate predicate
                if let Ok(df) = crate::types::DataFrame::from_series_map(columns) {
                    if let Ok(series) =
                        crate::types::evaluate_expr_on_dataframe(&predicate_clone, &df)
                    {
                        // Convert boolean series to mask (1.0 for true, NaN for false)
                        let data = series.data();
                        let mut mask = Vec::with_capacity(n_days);
                        for &val in data.iter() {
                            // In our expression system, true is 1.0, false is 0.0
                            mask.push(if val != 0.0 { 1.0 } else { f64::NAN });
                        }
                        return mask;
                    }
                }

                // If evaluation failed, return all NaN (no filtering)
                vec![f64::NAN; n_days]
            })
            .collect();

        // Assemble masks into the 2D array
        for (asset_idx, asset_mask) in asset_masks.into_iter().enumerate() {
            for day_idx in 0..n_days {
                mask_array[[day_idx, asset_idx]] = asset_mask[day_idx];
            }
        }

        // Apply mask to all input columns
        let mut result = HashMap::new();
        for (col_name, array) in input_data {
            // Multiply by mask: values become NaN where mask is NaN
            let masked_array = array * &mask_array;
            result.insert(col_name, masked_array);
        }

        Ok(result)
    }

    fn execute_window(
        &mut self,
        input: &Arc<LogicalPlan>,
        expr: &Expr,
        window_spec: &WindowSpec,
        output_name: &str,
    ) -> Result<HashMap<String, Array2<f64>>, String> {
        // Execute input plan
        let input_data = self.execute(input)?;

        // Get dimensions from first array
        let (n_days, n_assets) = if let Some(first_array) = input_data.values().next() {
            first_array.dim()
        } else {
            return Err("No input data for window operation".to_string());
        };

        // Create result array for the window operation
        let mut result_array = Array2::<f64>::zeros((n_days, n_assets));

        // Process each asset in parallel
        use rayon::prelude::*;

        // Clone necessary data for parallel processing
        let input_data_clone = input_data.clone();
        let expr_clone = expr.clone();
        let window_spec_clone = window_spec.clone();

        // Process assets in parallel
        let asset_results: Vec<_> = (0..n_assets)
            .into_par_iter()
            .map(|asset_idx| {
                // Create DataFrame for this asset
                let mut columns = std::collections::HashMap::new();

                for (col_name, array) in &input_data_clone {
                    let column_data = array.column(asset_idx).to_owned();
                    columns.insert(
                        col_name.clone(),
                        crate::types::Series::new(column_data.to_vec()),
                    );
                }

                // Create DataFrame and evaluate expression
                if let Ok(df) = crate::types::DataFrame::from_series_map(columns) {
                    if let Ok(series) = crate::types::evaluate_expr_on_dataframe(&expr_clone, &df) {
                        // Apply window operation based on window specification
                        let windowed_series = apply_window_to_series(&series, &window_spec_clone);
                        return windowed_series.data().to_vec();
                    }
                }

                // Return NaN-filled vector if evaluation failed
                vec![f64::NAN; n_days]
            })
            .collect();

        // Assemble results into the 2D array
        for (asset_idx, asset_result) in asset_results.into_iter().enumerate() {
            for day_idx in 0..n_days {
                result_array[[day_idx, asset_idx]] = asset_result[day_idx];
            }
        }

        // Create result map with all input data plus the new window column
        let mut result = input_data;
        result.insert(output_name.to_string(), result_array);

        Ok(result)
    }

    fn execute_stateful(
        &mut self,
        input: &Arc<LogicalPlan>,
        expr: &StatefulExpr,
        output_name: &str,
    ) -> Result<HashMap<String, Array2<f64>>, String> {
        // Execute input plan
        let input_data = self.execute(input)?;

        // Get dimensions from first array
        let (n_days, n_assets) = if let Some(first_array) = input_data.values().next() {
            first_array.dim()
        } else {
            return Err("No input data for stateful operation".to_string());
        };

        // Create result array for the stateful operation
        let mut result_array = Array2::<f64>::zeros((n_days, n_assets));

        // Process each asset in parallel
        use rayon::prelude::*;

        // Clone necessary data for parallel processing
        let input_data_clone = input_data.clone();
        let expr_clone = expr.clone();

        // Process assets in parallel
        let asset_results: Vec<_> = (0..n_assets)
            .into_par_iter()
            .map(|asset_idx| {
                // Create DataFrame for this asset
                let mut columns = std::collections::HashMap::new();

                for (col_name, array) in &input_data_clone {
                    let column_data = array.column(asset_idx).to_owned();
                    columns.insert(
                        col_name.clone(),
                        crate::types::Series::new(column_data.to_vec()),
                    );
                }

                // Create DataFrame and evaluate the inner expression
                if let Ok(df) = crate::types::DataFrame::from_series_map(columns) {
                    // Extract the inner expression from StatefulExpr
                    let inner_expr = match &expr_clone {
                        StatefulExpr::CumSum(expr) => expr,
                        StatefulExpr::CumProd(expr) => expr,
                        StatefulExpr::CumMax(expr) => expr,
                        StatefulExpr::CumMin(expr) => expr,
                        StatefulExpr::Ema(expr, _) => expr,
                    };

                    if let Ok(series) = crate::types::evaluate_expr_on_dataframe(inner_expr, &df) {
                        // Apply stateful operation
                        let stateful_series = apply_stateful_to_series(&series, &expr_clone);
                        return stateful_series.data().to_vec();
                    }
                }

                // Return NaN-filled vector if evaluation failed
                vec![f64::NAN; n_days]
            })
            .collect();

        // Assemble results into the 2D array
        for (asset_idx, asset_result) in asset_results.into_iter().enumerate() {
            for day_idx in 0..n_days {
                result_array[[day_idx, asset_idx]] = asset_result[day_idx];
            }
        }

        // Create result map with all input data plus the new stateful column
        let mut result = input_data;
        result.insert(output_name.to_string(), result_array);

        Ok(result)
    }

    fn execute_cache(
        &mut self,
        input: &Arc<LogicalPlan>,
        key: &str,
    ) -> Result<HashMap<String, Array2<f64>>, String> {
        // Check cache first
        if let Some(cached) = self.cache.get(key) {
            return Ok(cached.clone());
        }

        // Execute input and cache result
        let result = self.execute(input)?;
        self.cache.insert(key.to_string(), result.clone());

        Ok(result)
    }

    fn execute_join(
        &mut self,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        _on: &[String],
        how: JoinType,
    ) -> Result<HashMap<String, Array2<f64>>, String> {
        // Execute both input plans
        let left_data = self.execute(left)?;
        let right_data = self.execute(right)?;

        // For factor data, we assume all arrays have the same dimensions (n_days × n_assets)
        // Join simply merges columns from both sides
        let mut result = left_data.clone();

        match how {
            JoinType::Inner => {
                // Inner join: keep only rows/columns present in both sides
                // For factor data with same dimensions, this is just merging columns
                // but we need to check dimensions match
                let (left_days, left_assets) = if let Some(first_array) = left_data.values().next()
                {
                    first_array.dim()
                } else {
                    return Err("Left side of join has no data".to_string());
                };

                let (right_days, right_assets) =
                    if let Some(first_array) = right_data.values().next() {
                        first_array.dim()
                    } else {
                        return Err("Right side of join has no data".to_string());
                    };

                if left_days != right_days || left_assets != right_assets {
                    return Err(format!(
                        "Join dimensions mismatch: left ({}, {}), right ({}, {})",
                        left_days, left_assets, right_days, right_assets
                    ));
                }

                // Merge columns from right side
                for (col_name, array) in right_data {
                    // Check for column name conflicts
                    if result.contains_key(&col_name) {
                        return Err(format!("Duplicate column name in join: {}", col_name));
                    }
                    result.insert(col_name, array);
                }

                Ok(result)
            }
            JoinType::Left => {
                // Left join: keep all from left, add matching from right
                // For factor data, this is similar to inner join but we fill missing with NaN
                let (left_days, left_assets) = if let Some(first_array) = left_data.values().next()
                {
                    first_array.dim()
                } else {
                    return Err("Left side of join has no data".to_string());
                };

                for (col_name, right_array) in right_data {
                    let (right_days, right_assets) = right_array.dim();

                    if left_days != right_days || left_assets != right_assets {
                        // Create NaN-filled array with left dimensions
                        let mut filled_array =
                            Array2::<f64>::from_elem((left_days, left_assets), f64::NAN);

                        // Copy overlapping region if any
                        let days_overlap = left_days.min(right_days);
                        let assets_overlap = left_assets.min(right_assets);

                        for day in 0..days_overlap {
                            for asset in 0..assets_overlap {
                                filled_array[[day, asset]] = right_array[[day, asset]];
                            }
                        }

                        result.insert(col_name, filled_array);
                    } else {
                        // Dimensions match, just add the column
                        if result.contains_key(&col_name) {
                            return Err(format!("Duplicate column name in join: {}", col_name));
                        }
                        result.insert(col_name, right_array);
                    }
                }

                Ok(result)
            }
            JoinType::Right => {
                // Right join: keep all from right, add matching from left
                // Similar to left join but reversed
                let mut result = right_data.clone();
                let (right_days, right_assets) =
                    if let Some(first_array) = right_data.values().next() {
                        first_array.dim()
                    } else {
                        return Err("Right side of join has no data".to_string());
                    };

                for (col_name, left_array) in left_data {
                    if !result.contains_key(&col_name) {
                        let (left_days, left_assets) = left_array.dim();

                        if right_days != left_days || right_assets != left_assets {
                            // Create NaN-filled array with right dimensions
                            let mut filled_array =
                                Array2::<f64>::from_elem((right_days, right_assets), f64::NAN);

                            // Copy overlapping region if any
                            let days_overlap = right_days.min(left_days);
                            let assets_overlap = right_assets.min(left_assets);

                            for day in 0..days_overlap {
                                for asset in 0..assets_overlap {
                                    filled_array[[day, asset]] = left_array[[day, asset]];
                                }
                            }

                            result.insert(col_name, filled_array);
                        } else {
                            result.insert(col_name, left_array);
                        }
                    }
                }

                Ok(result)
            }
            JoinType::Outer => {
                // Full outer join: union of all columns, fill missing with NaN
                // Determine maximum dimensions
                let (mut max_days, mut max_assets) = (0, 0);

                for array in left_data.values().chain(right_data.values()) {
                    let (days, assets) = array.dim();
                    max_days = max_days.max(days);
                    max_assets = max_assets.max(assets);
                }

                // Create result with all columns, padded to max dimensions
                let mut result = HashMap::new();

                // Process left columns
                for (col_name, array) in left_data {
                    let (days, assets) = array.dim();
                    if days == max_days && assets == max_assets {
                        result.insert(col_name, array);
                    } else {
                        // Pad with NaN
                        let mut padded = Array2::<f64>::from_elem((max_days, max_assets), f64::NAN);
                        for day in 0..days.min(max_days) {
                            for asset in 0..assets.min(max_assets) {
                                padded[[day, asset]] = array[[day, asset]];
                            }
                        }
                        result.insert(col_name, padded);
                    }
                }

                // Process right columns (skip duplicates)
                for (col_name, array) in right_data {
                    if !result.contains_key(&col_name) {
                        let (days, assets) = array.dim();
                        if days == max_days && assets == max_assets {
                            result.insert(col_name, array);
                        } else {
                            // Pad with NaN
                            let mut padded =
                                Array2::<f64>::from_elem((max_days, max_assets), f64::NAN);
                            for day in 0..days.min(max_days) {
                                for asset in 0..assets.min(max_assets) {
                                    padded[[day, asset]] = array[[day, asset]];
                                }
                            }
                            result.insert(col_name, padded);
                        }
                    }
                }

                Ok(result)
            }
        }
    }
}

// ============================================================================
// Builder Pattern for Easy Expression Building
// ============================================================================

/// Builder for LazyFrame with fluent API
pub struct LazyFrameBuilder {
    lazy_frame: Option<LazyFrame>,
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
    pub fn collect(self) -> Result<HashMap<String, Array2<f64>>, String> {
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

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a rolling window specification
pub fn rolling_window(size: usize, min_periods: Option<usize>) -> WindowSpec {
    WindowSpec {
        kind: WindowKind::Rolling,
        size: Some(size),
        min_periods: min_periods.unwrap_or(1),
    }
}

/// Create an expanding window specification
pub fn expanding_window(min_periods: Option<usize>) -> WindowSpec {
    WindowSpec {
        kind: WindowKind::Expanding,
        size: None,
        min_periods: min_periods.unwrap_or(1),
    }
}

/// Create a cumulative sum expression
pub fn cumsum(expr: Expr) -> StatefulExpr {
    StatefulExpr::CumSum(expr)
}

/// Create a cumulative product expression
pub fn cumprod(expr: Expr) -> StatefulExpr {
    StatefulExpr::CumProd(expr)
}

/// Create an exponential moving average expression
pub fn ema(expr: Expr, span: usize) -> StatefulExpr {
    let alpha = 2.0 / (span as f64 + 1.0);
    StatefulExpr::Ema(expr, alpha)
}

/// Apply window operation to a series based on window specification
fn apply_window_to_series(
    series: &crate::types::Series,
    window_spec: &WindowSpec,
) -> crate::types::Series {
    use crate::types::Series;

    match window_spec.kind {
        WindowKind::Rolling => {
            if let Some(window_size) = window_spec.size {
                // For now, implement rolling mean
                // TODO: Support other aggregation types (sum, std, etc.)
                series.moving_average(window_size)
            } else {
                // No window size specified, return original series
                series.clone()
            }
        }
        WindowKind::Expanding => {
            // Expanding window mean
            let n = series.len();
            let mut result = Vec::with_capacity(n);
            let mut sum = 0.0;
            let mut count = 0;

            for i in 0..n {
                let val = series.data()[i];
                if !val.is_nan() {
                    sum += val;
                    count += 1;
                }
                if count >= window_spec.min_periods && count > 0 {
                    result.push(sum / count as f64);
                } else {
                    result.push(f64::NAN);
                }
            }

            Series::new(result)
        }
        WindowKind::RollingOffset => {
            // Similar to rolling but with offset
            // For now, treat as rolling window
            if let Some(window_size) = window_spec.size {
                series.moving_average(window_size)
            } else {
                series.clone()
            }
        }
    }
}

/// Apply stateful operation to a series
fn apply_stateful_to_series(
    series: &crate::types::Series,
    expr: &StatefulExpr,
) -> crate::types::Series {
    use crate::types::Series;

    match expr {
        StatefulExpr::CumSum(_) => {
            let data = series.data();
            let mut result = Vec::with_capacity(data.len());
            let mut sum = 0.0;

            for &val in data.iter() {
                if !val.is_nan() {
                    sum += val;
                }
                result.push(sum);
            }

            Series::new(result)
        }
        StatefulExpr::CumProd(_) => {
            let data = series.data();
            let mut result = Vec::with_capacity(data.len());
            let mut prod = 1.0;

            for &val in data.iter() {
                if !val.is_nan() {
                    prod *= val;
                }
                result.push(prod);
            }

            Series::new(result)
        }
        StatefulExpr::CumMax(_) => {
            let data = series.data();
            let mut result = Vec::with_capacity(data.len());
            let mut max = f64::NEG_INFINITY;

            for &val in data.iter() {
                if !val.is_nan() && val > max {
                    max = val;
                }
                result.push(max);
            }

            Series::new(result)
        }
        StatefulExpr::CumMin(_) => {
            let data = series.data();
            let mut result = Vec::with_capacity(data.len());
            let mut min = f64::INFINITY;

            for &val in data.iter() {
                if !val.is_nan() && val < min {
                    min = val;
                }
                result.push(min);
            }

            Series::new(result)
        }
        StatefulExpr::Ema(_, alpha) => {
            // Use the Series.ema method, but we need to convert alpha to span
            // EMA formula: alpha = 2/(span+1) => span = 2/alpha - 1
            let span = ((2.0 / alpha) - 1.0) as usize;
            series.ema(span)
        }
    }
}
