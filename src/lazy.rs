//! Lazy evaluation system inspired by Polars' lazy API
//!
//! This module provides a lazy evaluation system for building complex factor expressions
//! with automatic optimization and efficient execution.

use std::collections::HashMap;
use std::sync::Arc;

use crate::expr::Expr;
use crate::polars_style::DataFrame;
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
        exprs: Vec<(String, Expr)>,  // (column_name, expression)
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
    External {
        path: String,
        format: DataFormat,
    },
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
    Ema(Expr, f64),  // expression, alpha
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
        let cache_key = key.map(|k| k.to_string())
            .unwrap_or_else(|| format!("cache_{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()));
        
        LazyFrame {
            logical_plan: Arc::new(LogicalPlan::Cache {
                input: self.logical_plan,
                key: cache_key,
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
        result.get(output_column)
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
            _ => self.apply_optimizations(plan),
        }
    }
    
    fn apply_optimizations(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement actual optimizations
        // - Predicate pushdown
        // - Projection pruning
        // - Common subexpression elimination
        // - Constant folding
        // - Join reordering
        
        // For now, just return the plan unchanged
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
            LogicalPlan::Scan { source, projection, selection } => {
                self.execute_scan(source, projection.as_ref(), selection.as_ref())
            }
            LogicalPlan::Projection { input, exprs } => {
                self.execute_projection(input, exprs)
            }
            LogicalPlan::Filter { input, predicate } => {
                self.execute_filter(input, predicate)
            }
            LogicalPlan::Window { input, expr, window_spec, output_name } => {
                self.execute_window(input, expr, window_spec, output_name)
            }
            LogicalPlan::Stateful { input, expr, output_name } => {
                self.execute_stateful(input, expr, output_name)
            }
            LogicalPlan::Cache { input, key } => {
                self.execute_cache(input, key)
            }
            LogicalPlan::Join { left, right, on, how } => {
                self.execute_join(left, right, on, *how)
            }
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
            DataSource::External { .. } => {
                Err("External sources not yet implemented".to_string())
            }
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
                        columns.insert(col_name.clone(), crate::polars_style::Series::new(column_data.to_vec()));
                    }
                    
                    // Create DataFrame and evaluate expression
                    if let Ok(df) = crate::polars_style::DataFrame::from_series_map(columns) {
                        if let Ok(series) = crate::polars_style::evaluate_expr_on_dataframe(&expr_clone, &df) {
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
        _input: &Arc<LogicalPlan>,
        _predicate: &Expr,
    ) -> Result<HashMap<String, Array2<f64>>, String> {
        // TODO: Implement filtering
        Err("Filter execution not yet implemented".to_string())
    }
    
    fn execute_window(
        &mut self,
        _input: &Arc<LogicalPlan>,
        _expr: &Expr,
        _window_spec: &WindowSpec,
        _output_name: &str,
    ) -> Result<HashMap<String, Array2<f64>>, String> {
        // TODO: Implement window operations
        Err("Window execution not yet implemented".to_string())
    }
    
    fn execute_stateful(
        &mut self,
        _input: &Arc<LogicalPlan>,
        _expr: &StatefulExpr,
        _output_name: &str,
    ) -> Result<HashMap<String, Array2<f64>>, String> {
        // TODO: Implement stateful operations
        Err("Stateful execution not yet implemented".to_string())
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
        _left: &Arc<LogicalPlan>,
        _right: &Arc<LogicalPlan>,
        _on: &[String],
        _how: JoinType,
    ) -> Result<HashMap<String, Array2<f64>>, String> {
        // TODO: Implement joins
        Err("Join execution not yet implemented".to_string())
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