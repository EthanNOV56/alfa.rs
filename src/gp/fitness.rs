//! Fitness evaluators for factor evaluation.
//!
//! Provides the `FitnessEvaluator` trait and several implementations:
//! - `RealBacktestFitnessEvaluator` — full backtest-based evaluation with
//!   train/test/validation split support
//! - `BacktestFitnessEvaluator` — simple complexity-based evaluator for testing
//! - `CachedFitnessEvaluator` — LRU-cached wrapper with semantic deduplication
//! - `BatchFitnessEvaluator` — parallel batch processing wrapper

use crate::WeightMethod;
use crate::backtest::{BacktestConfig, BacktestEngine, BacktestResult, FeeConfig, PositionConfig};
use crate::data::layer::PriceMatrix;
use crate::expr::registry::functions::eval_expr_vectorized;
use crate::expr::{BinaryOp, Expr, Literal, UnaryOp};
use crate::gp::types::{
    Complexity, DataSplit, DataSplitConfig, MultiObjectiveFitness, SplitEvaluationResult,
    SplitMetrics, normalize_expression,
};
use lru::LruCache;
use ndarray::Array2;
use rayon::prelude::*;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::atomic;
use std::sync::{Mutex, RwLock};

/// Fitness evaluator trait for evaluating expression fitness
pub trait FitnessEvaluator: Send + Sync {
    fn fitness(&self, expr: &Expr) -> f64;

    /// Batch evaluate multiple expressions (optional optimization)
    fn fitness_batch(&self, exprs: &[Expr]) -> Vec<f64> {
        // Default implementation: evaluate sequentially
        exprs.iter().map(|e| self.fitness(e)).collect()
    }

    /// Check if this evaluator supports batch evaluation
    fn supports_batch(&self) -> bool {
        false
    }
}

/// Fitness evaluator for factor mining based on actual backtest performance
/// with support for train/test/validation split
pub struct RealBacktestFitnessEvaluator {
    /// Pre-built per-asset columns: asset_columns[asset_idx][col_name] = time series
    asset_columns: Vec<HashMap<String, ndarray::Array1<f64>>>,
    prices: PriceMatrix,
    weights: HashMap<String, f64>, // Feature weights for multi-objective optimization
    min_valid_days: usize,         // Minimum days required for valid backtest
    // Data split configuration
    data_split: Option<DataSplit>,
    // Fee configuration for backtest
    fee_config: FeeConfig,
    // Position configuration for backtest
    position_config: PositionConfig,
    // Store last computed metrics (using RwLock for thread safety)
    last_metrics: RwLock<(f64, f64, f64, usize)>, // (ic, ir, turnover, complexity)
    // Store last split evaluation result
    last_split_result: RwLock<Option<SplitEvaluationResult>>,
}

impl RealBacktestFitnessEvaluator {
    /// Create a new evaluator with a PriceMatrix for backtesting.
    pub fn new(data: HashMap<String, Array2<f64>>, prices: PriceMatrix) -> Self {
        let asset_columns = Self::build_asset_columns(&data);
        Self {
            asset_columns,
            prices,
            weights: HashMap::new(),
            min_valid_days: 50,
            data_split: None,
            fee_config: FeeConfig::default(),
            position_config: PositionConfig::default(),
            last_metrics: RwLock::new((0.0, 0.0, 0.0, 0)),
            last_split_result: RwLock::new(None),
        }
    }

    /// Create a new evaluator with train/test/validation split.
    pub fn with_split(
        data: HashMap<String, Array2<f64>>,
        prices: PriceMatrix,
        split_config: DataSplitConfig,
    ) -> Self {
        let n_days = prices.returns.shape()[0];
        let data_split = DataSplit::from_config(n_days, &split_config);
        let asset_columns = Self::build_asset_columns(&data);

        Self {
            asset_columns,
            prices,
            weights: HashMap::new(),
            min_valid_days: 50,
            data_split: Some(data_split),
            fee_config: FeeConfig::default(),
            position_config: PositionConfig::default(),
            last_metrics: RwLock::new((0.0, 0.0, 0.0, 0)),
            last_split_result: RwLock::new(None),
        }
    }

    /// Pre-build per-asset column slices (done once at construction).
    fn build_asset_columns(
        data: &HashMap<String, Array2<f64>>,
    ) -> Vec<HashMap<String, ndarray::Array1<f64>>> {
        let first = data.values().next();
        let n_assets = first.map(|a| a.shape()[1]).unwrap_or(0);
        let mut result = Vec::with_capacity(n_assets);
        for asset_idx in 0..n_assets {
            let mut cols = HashMap::new();
            for (name, array) in data {
                cols.insert(name.clone(), array.column(asset_idx).to_owned());
            }
            result.push(cols);
        }
        result
    }

    /// Set fee configuration
    pub fn with_fee_config(mut self, fee_config: FeeConfig) -> Self {
        self.fee_config = fee_config;
        self
    }

    /// Set position configuration
    pub fn with_position_config(mut self, position_config: PositionConfig) -> Self {
        self.position_config = position_config;
        self
    }

    /// Get last computed IC
    pub fn get_last_ic(&self) -> f64 {
        self.last_metrics.read().unwrap().0
    }

    /// Get last computed IR
    pub fn get_last_ir(&self) -> f64 {
        self.last_metrics.read().unwrap().1
    }

    /// Get last computed turnover
    pub fn get_last_turnover(&self) -> f64 {
        self.last_metrics.read().unwrap().2
    }

    /// Get last computed complexity
    pub fn get_last_complexity(&self) -> usize {
        self.last_metrics.read().unwrap().3
    }

    /// Set feature weights for multi-objective optimization
    pub fn set_weights(&mut self, weights: HashMap<String, f64>) -> &mut Self {
        self.weights = weights;
        self
    }

    /// Set minimum valid days for backtest
    pub fn set_min_valid_days(&mut self, days: usize) -> &mut Self {
        self.min_valid_days = days;
        self
    }

    /// Evaluate expression and run backtest (one full backtest, slice for train/val/test).
    fn evaluate_with_backtest(&self, expr: &Expr) -> Option<MultiObjectiveFitness> {
        let n_days = self.prices.returns.shape()[0];

        if n_days < self.min_valid_days {
            return None;
        }

        use std::time::Instant;
        let t0 = Instant::now();
        let factor_matrix = self.evaluate_expression(expr)?;
        let t_eval = t0.elapsed();

        let result = self.run_backtest(&factor_matrix)?;
        let t_bt = t0.elapsed();

        let turnover = self.compute_turnover(&factor_matrix);
        let complexity = self.compute_complexity(expr);

        // Slice IC from train indices for fitness (if split configured)
        let (train_ic, train_ir) = if let Some(ref split) = self.data_split {
            let metrics = self.compute_split_metrics(&result.ic_series, &split.train_indices);
            (metrics.ic_mean, metrics.ic_ir)
        } else {
            (result.ic_mean, result.ic_ir)
        };

        // Store last computed metrics (train only for fitness)
        *self.last_metrics.write().unwrap() = (
            train_ic.abs(),
            train_ir.abs(),
            turnover,
            complexity.node_count,
        );

        // Compute split metrics from the same full backtest result
        if let Some(split_result) = self.evaluate_split_metrics(&result) {
            *self.last_split_result.write().unwrap() = Some(split_result);
        }

        let total = t0.elapsed();
        if total.as_millis() > 100 {
            println!(
                "[perf] fitness eval={:.0}ms bt={:.0}ms total={:.0}ms",
                t_eval.as_millis(),
                (t_bt - t_eval).as_millis(),
                total.as_millis()
            );
        }

        // Get weights
        let w_ic = *self.weights.get("ic").unwrap_or(&0.4);
        let w_ir = *self.weights.get("ir").unwrap_or(&0.3);
        let w_to = *self.weights.get("turnover").unwrap_or(&0.15);
        let w_comp = *self.weights.get("complexity").unwrap_or(&0.15);

        // Create multi-objective fitness (uses train-sliced IC/IR)
        Some(MultiObjectiveFitness::new(
            train_ic.abs(),
            train_ir.abs(),
            turnover,
            &complexity,
            Some((w_ic, w_ir, w_to, w_comp)),
        ))
    }

    /// Evaluate expression to get factor matrix.
    ///
    /// Uses pre-built per-asset columns and caches. Sequential loop per asset
    /// avoids nested parallelism (outer `par_iter` already saturates threads).
    fn evaluate_expression(&self, expr: &Expr) -> Option<Array2<f64>> {
        let n_days = self.prices.returns.shape()[0];
        let n_assets = self.asset_columns.len();
        let mut factor_matrix = Array2::<f64>::zeros((n_days, n_assets));

        for (asset_idx, columns) in self.asset_columns.iter().enumerate() {
            let mut cache = HashMap::new();

            let values = match eval_expr_vectorized(expr, columns, &mut cache) {
                Ok(arr) => arr,
                Err(_) => continue,
            };

            for (day_idx, &v) in values.iter().enumerate() {
                factor_matrix[[day_idx, asset_idx]] = v;
            }
        }

        let nan_count = factor_matrix.iter().filter(|&&v| v.is_nan()).count();
        let total = n_days * n_assets;
        if total - nan_count < self.min_valid_days * n_assets / 2 {
            return None;
        }

        Some(factor_matrix)
    }

    /// Run IC-only backtest on the full factor matrix (skips P&L simulation).
    fn run_backtest(&self, factor: &Array2<f64>) -> Option<BacktestResult> {
        let config = BacktestConfig {
            quantiles: 10,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: self.fee_config.clone(),
            limit_up_down_config: Default::default(),
            position_config: self.position_config.clone(),
        };

        let engine = BacktestEngine::with_config(config);

        match engine.run_ic_only_with_prices(factor.clone(), &self.prices) {
            Ok(result) => {
                if result.ic_mean.is_nan() || result.ic_ir.is_nan() {
                    None
                } else {
                    Some(result)
                }
            }
            Err(_) => None,
        }
    }

    /// Compute IC mean and IR from ic_series sliced by indices.
    fn compute_split_metrics(
        &self,
        ic_series: &ndarray::Array1<f64>,
        indices: &[usize],
    ) -> SplitMetrics {
        let n = indices.len() as f64;
        if n < 2.0 {
            return SplitMetrics::default();
        }
        let sum: f64 = indices.iter().map(|&i| ic_series[i]).sum();
        let mean = sum / n;
        let variance: f64 = indices
            .iter()
            .map(|&i| {
                let diff = ic_series[i] - mean;
                diff * diff
            })
            .sum::<f64>()
            / (n - 1.0);
        let ir = if variance > 0.0 {
            mean / variance.sqrt()
        } else {
            0.0
        };
        SplitMetrics {
            ic_mean: mean,
            ic_ir: ir,
            ..Default::default()
        }
    }

    /// Evaluate factor on all splits (from a single full backtest result).
    pub fn evaluate_split_metrics(&self, result: &BacktestResult) -> Option<SplitEvaluationResult> {
        let split = self.data_split.as_ref()?;
        Some(SplitEvaluationResult {
            train: self.compute_split_metrics(&result.ic_series, &split.train_indices),
            validation: self.compute_split_metrics(&result.ic_series, &split.validation_indices),
            test: self.compute_split_metrics(&result.ic_series, &split.test_indices),
        })
    }

    /// Get last split evaluation result
    pub fn get_last_split_result(&self) -> Option<SplitEvaluationResult> {
        self.last_split_result.read().unwrap().clone()
    }

    /// Compute turnover based on rank changes (percentage of assets changing position)
    /// This is a more reasonable measure that gives values between 0 and 1
    fn compute_turnover(&self, factor: &Array2<f64>) -> f64 {
        let (n_days, n_assets) = factor.dim();
        if n_days < 2 || n_assets < 2 {
            return 0.0;
        }

        let mut total_turnover = 0.0;
        let mut valid_days = 0;

        for day in 1..n_days {
            let today = factor.row(day);
            let yesterday = factor.row(day - 1);

            // Get valid assets for both days
            let mut valid_indices = Vec::new();
            for asset in 0..n_assets {
                let f_today = today[asset];
                let f_yesterday = yesterday[asset];
                if !f_today.is_nan()
                    && !f_yesterday.is_nan()
                    && !f_today.is_infinite()
                    && !f_yesterday.is_infinite()
                {
                    valid_indices.push(asset);
                }
            }

            if valid_indices.len() < 2 {
                continue;
            }

            // Compute ranks for valid assets
            let mut yesterday_ranks: Vec<(usize, f64)> =
                valid_indices.iter().map(|&i| (i, yesterday[i])).collect();
            let mut today_ranks: Vec<(usize, f64)> =
                valid_indices.iter().map(|&i| (i, today[i])).collect();

            // Sort by factor values (descending)
            yesterday_ranks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            today_ranks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Compute top 20% turnover
            let top_n = std::cmp::max(1, valid_indices.len() / 5);

            let prev_top: std::collections::HashSet<usize> = yesterday_ranks
                .iter()
                .take(top_n)
                .map(|(i, _)| *i)
                .collect();
            let curr_top: std::collections::HashSet<usize> =
                today_ranks.iter().take(top_n).map(|(i, _)| *i).collect();

            // Compute Jaccard distance: |A ∩ B| / |A ∪ B|
            // Turnover = 1 - similarity
            let intersection: usize = prev_top.intersection(&curr_top).count();
            let union = prev_top.len() + curr_top.len() - intersection;
            let similarity = if union > 0 {
                intersection as f64 / union as f64
            } else {
                1.0
            };
            let day_turnover = 1.0 - similarity;

            total_turnover += day_turnover;
            valid_days += 1;
        }

        if valid_days == 0 {
            0.0
        } else {
            total_turnover / valid_days as f64
        }
    }

    /// Compute multi-factor complexity of an expression.
    fn compute_complexity(&self, expr: &Expr) -> Complexity {
        let mut cols = std::collections::HashSet::new();
        let (node_count, free_params) = self.compute_complexity_inner(expr, &mut cols);
        Complexity {
            node_count,
            free_param_count: free_params,
            unique_column_count: cols.len(),
        }
    }

    /// Walk the tree, counting nodes and free parameters. Collects column names.
    fn compute_complexity_inner(
        &self,
        expr: &Expr,
        cols: &mut std::collections::HashSet<String>,
    ) -> (usize, usize) {
        match expr {
            Expr::Literal(Literal::Float(_) | Literal::Integer(_)) => (1, 1),
            Expr::Literal(_) => (1, 0),
            Expr::Column(name) => {
                cols.insert(name.clone());
                (1, 0)
            }
            Expr::UnaryExpr { expr, .. } => {
                let (n, p) = self.compute_complexity_inner(expr, cols);
                (1 + n, p)
            }
            Expr::BinaryExpr { left, right, .. } => {
                let (nl, pl) = self.compute_complexity_inner(left, cols);
                let (nr, pr) = self.compute_complexity_inner(right, cols);
                (1 + nl + nr, pl + pr)
            }
            Expr::FunctionCall { args, .. } => {
                let mut total_n = 1;
                let mut total_p = 0;
                for arg in args {
                    let (n, p) = self.compute_complexity_inner(arg, cols);
                    total_n += n;
                    total_p += p;
                }
                (total_n, total_p)
            }
            Expr::Aggregate { expr, .. } => {
                let (n, p) = self.compute_complexity_inner(expr, cols);
                (1 + n, p)
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let (nc, pc) = self.compute_complexity_inner(condition, cols);
                let (nt, pt) = self.compute_complexity_inner(then_expr, cols);
                let (ne, pe) = self.compute_complexity_inner(else_expr, cols);
                (1 + nc + nt + ne, pc + pt + pe)
            }
            Expr::Cast { expr, .. } => {
                let (n, p) = self.compute_complexity_inner(expr, cols);
                (1 + n, p)
            }
        }
    }

    /// Check expression constraints (e.g., prevent division by zero, extreme values)
    fn check_constraints(&self, expr: &Expr) -> bool {
        self.check_division_by_zero(expr)
            && self.check_extreme_operations(expr)
            && self.check_valid_windows(expr)
    }

    /// Reject expressions with invalid window/period values in time-series functions.
    /// Window must be >= 2 for rolling stats, >= 1 for delay/delta.
    fn check_valid_windows(&self, expr: &Expr) -> bool {
        match expr {
            Expr::FunctionCall { name, args, .. } if name.starts_with("ts_") || name == "delay" => {
                // Second argument is the window/periods parameter
                if args.len() >= 2 {
                    let window_expr = &args[1];
                    if self.could_be_window_zero_or_one(window_expr) {
                        return false;
                    }
                }
                args.iter().all(|a| self.check_valid_windows(a))
            }
            Expr::FunctionCall { args, .. } => args.iter().all(|a| self.check_valid_windows(a)),
            Expr::UnaryExpr { expr, .. } => self.check_valid_windows(expr),
            Expr::BinaryExpr { left, right, .. } => {
                self.check_valid_windows(left) && self.check_valid_windows(right)
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.check_valid_windows(condition)
                    && self.check_valid_windows(then_expr)
                    && self.check_valid_windows(else_expr)
            }
            Expr::Aggregate { expr, .. } => self.check_valid_windows(expr),
            Expr::Cast { expr, .. } => self.check_valid_windows(expr),
            _ => true,
        }
    }

    /// Check if the window expression could evaluate to 0 or 1 (invalid for rolling stats).
    fn could_be_window_zero_or_one(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Literal(Literal::Float(v)) => *v < 1.5,
            Expr::Literal(Literal::Integer(v)) => *v < 2,
            _ => false, // unknown — allow it through
        }
    }

    /// Check for potential division by zero
    fn check_division_by_zero(&self, expr: &Expr) -> bool {
        match expr {
            Expr::BinaryExpr { op, right, .. } if *op == BinaryOp::Divide => {
                // Check if right side could be zero
                !self.could_be_zero(right)
            }
            Expr::BinaryExpr { left, right, .. } => {
                self.check_division_by_zero(left) && self.check_division_by_zero(right)
            }
            Expr::UnaryExpr { expr, .. } => self.check_division_by_zero(expr),
            Expr::FunctionCall { args, .. } => {
                args.iter().all(|arg| self.check_division_by_zero(arg))
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.check_division_by_zero(condition)
                    && self.check_division_by_zero(then_expr)
                    && self.check_division_by_zero(else_expr)
            }
            Expr::Aggregate { expr, .. } => self.check_division_by_zero(expr),
            Expr::Cast { expr, .. } => self.check_division_by_zero(expr),
            _ => true,
        }
    }

    /// Check if an expression could evaluate to zero
    fn could_be_zero(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Literal(Literal::Float(v)) => v.abs() < 1e-10,
            Expr::Literal(Literal::Integer(v)) => *v == 0,
            Expr::BinaryExpr { left, op, right } => {
                match op {
                    BinaryOp::Add | BinaryOp::Subtract => {
                        self.could_be_zero(left) && self.could_be_zero(right)
                    }
                    BinaryOp::Multiply => self.could_be_zero(left) || self.could_be_zero(right),
                    BinaryOp::Divide => {
                        // Division by something that could be zero is dangerous
                        self.could_be_zero(right)
                    }
                    _ => false,
                }
            }
            Expr::UnaryExpr {
                op: UnaryOp::Negate,
                expr,
            } => self.could_be_zero(expr),
            _ => false,
        }
    }

    /// Check for extreme operations (e.g., log of negative, sqrt of negative)
    fn check_extreme_operations(&self, expr: &Expr) -> bool {
        match expr {
            Expr::UnaryExpr { op, expr } => match op {
                UnaryOp::Sqrt | UnaryOp::Log => {
                    // Check if expression could be negative
                    !self.could_be_negative(expr)
                }
                _ => self.check_extreme_operations(expr),
            },
            Expr::BinaryExpr { left, right, .. } => {
                self.check_extreme_operations(left) && self.check_extreme_operations(right)
            }
            Expr::FunctionCall { args, .. } => {
                args.iter().all(|arg| self.check_extreme_operations(arg))
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.check_extreme_operations(condition)
                    && self.check_extreme_operations(then_expr)
                    && self.check_extreme_operations(else_expr)
            }
            Expr::Aggregate { expr, .. } => self.check_extreme_operations(expr),
            Expr::Cast { expr, .. } => self.check_extreme_operations(expr),
            _ => true,
        }
    }

    /// Check if an expression could be negative
    fn could_be_negative(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Literal(Literal::Float(v)) => *v < 0.0,
            Expr::Literal(Literal::Integer(v)) => *v < 0,
            Expr::BinaryExpr { left, op, right } => {
                match op {
                    BinaryOp::Add => self.could_be_negative(left) || self.could_be_negative(right),
                    BinaryOp::Subtract => true, // Subtraction could always result in negative
                    BinaryOp::Multiply => {
                        // Multiplication could be negative if one is negative and one positive
                        (self.could_be_negative(left) && !self.could_be_positive(right))
                            || (self.could_be_negative(right) && !self.could_be_positive(left))
                    }
                    BinaryOp::Divide => true, // Division could result in negative
                    _ => false,
                }
            }
            Expr::UnaryExpr {
                op: UnaryOp::Negate,
                expr,
            } => !self.could_be_negative(expr),
            Expr::Column(_) => true, // Column values could be negative
            _ => false,
        }
    }

    /// Check if an expression could be positive
    fn could_be_positive(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Literal(Literal::Float(v)) => *v > 0.0,
            Expr::Literal(Literal::Integer(v)) => *v > 0,
            Expr::BinaryExpr { left, op, right } => {
                match op {
                    BinaryOp::Add => self.could_be_positive(left) || self.could_be_positive(right),
                    BinaryOp::Subtract => true, // Subtraction could result in positive
                    BinaryOp::Multiply => {
                        // Multiplication could be positive if both positive or both negative
                        (self.could_be_positive(left) && self.could_be_positive(right))
                            || (self.could_be_negative(left) && self.could_be_negative(right))
                    }
                    BinaryOp::Divide => true, // Division could result in positive
                    _ => false,
                }
            }
            Expr::UnaryExpr {
                op: UnaryOp::Negate,
                expr,
            } => !self.could_be_positive(expr),
            Expr::Column(_) => true, // Column values could be positive
            _ => false,
        }
    }
}

impl FitnessEvaluator for RealBacktestFitnessEvaluator {
    fn fitness(&self, expr: &Expr) -> f64 {
        // Check constraints first
        if !self.check_constraints(expr) {
            return -1e9; // Very low fitness for invalid expressions
        }

        // Run real backtest evaluation
        match self.evaluate_with_backtest(expr) {
            Some(fitness_metrics) => {
                let base_fitness = fitness_metrics.fitness();

                // Apply penalty for extreme complexity
                let complexity = self.compute_complexity(expr);
                let complexity_penalty = (complexity.node_count as f64).powi(2) / 1000.0;

                base_fitness - complexity_penalty
            }
            None => -1e6, // Lower fitness for failed evaluation
        }
    }
}

/// Simple fitness evaluator based on expression complexity (for testing)
#[allow(dead_code)]
pub struct BacktestFitnessEvaluator {
    data: HashMap<String, Array2<f64>>,
    returns: Array2<f64>,
}

impl BacktestFitnessEvaluator {
    /// Create a new backtest fitness evaluator
    pub fn new(data: HashMap<String, Array2<f64>>, returns: Array2<f64>) -> Self {
        Self { data, returns }
    }

    /// Compute complexity of an expression (number of nodes)
    fn compute_complexity(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Literal(_) => 1,
            Expr::Column(_) => 1,
            Expr::UnaryExpr { expr, .. } => 1 + self.compute_complexity(expr),
            Expr::BinaryExpr { left, right, .. } => {
                1 + self.compute_complexity(left) + self.compute_complexity(right)
            }
            Expr::FunctionCall { args, .. } => {
                1 + args
                    .iter()
                    .map(|arg| self.compute_complexity(arg))
                    .sum::<usize>()
            }
            Expr::Aggregate { expr, .. } => 1 + self.compute_complexity(expr),
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                1 + self.compute_complexity(condition)
                    + self.compute_complexity(then_expr)
                    + self.compute_complexity(else_expr)
            }
            Expr::Cast { expr, .. } => 1 + self.compute_complexity(expr),
        }
    }
}

impl FitnessEvaluator for BacktestFitnessEvaluator {
    fn fitness(&self, expr: &Expr) -> f64 {
        // For now, return a simple fitness based on expression complexity
        // In real implementation, this would run actual backtest
        let complexity = self.compute_complexity(expr);

        // Higher fitness for simpler expressions (encourage parsimony)
        1.0 / (complexity as f64 + 1.0)
    }
}

/// Cached fitness evaluator with LRU cache and semantic deduplication.
pub struct CachedFitnessEvaluator<E: FitnessEvaluator> {
    evaluator: E,
    cache: Mutex<LruCache<String, f64>>,
    /// Maps normalized expression strings to fitness values for deduplication
    dedup: Mutex<HashMap<String, f64>>,
    /// Count of evaluations skipped due to semantic deduplication
    dedup_hits: atomic::AtomicU64,
    /// Count of total fitness evaluations (cache misses only)
    total_evals: atomic::AtomicU64,
}

impl<E: FitnessEvaluator> CachedFitnessEvaluator<E> {
    /// Create a new cached evaluator with given capacity
    pub fn new(evaluator: E, capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap_or(NonZeroUsize::new(100).unwrap());
        Self {
            evaluator,
            cache: Mutex::new(LruCache::new(cap)),
            dedup: Mutex::new(HashMap::new()),
            dedup_hits: atomic::AtomicU64::new(0),
            total_evals: atomic::AtomicU64::new(0),
        }
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
        self.dedup.lock().unwrap().clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Get deduplication hit count
    pub fn dedup_hits(&self) -> u64 {
        self.dedup_hits.load(atomic::Ordering::Relaxed)
    }

    /// Get total evaluations (cache misses)
    pub fn total_evals(&self) -> u64 {
        self.total_evals.load(atomic::Ordering::Relaxed)
    }

    /// Get cache hit rate statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.lock().unwrap();
        (cache.len(), cache.cap().get())
    }
}

impl<E: FitnessEvaluator> FitnessEvaluator for CachedFitnessEvaluator<E> {
    fn fitness(&self, expr: &Expr) -> f64 {
        let key = format!("{:?}", expr);

        // Check exact cache first
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(&cached) = cache.get(&key) {
                return cached;
            }
        }

        // Check semantic deduplication
        let normalized = normalize_expression(expr);
        {
            let dedup = self.dedup.lock().unwrap();
            if let Some(&cached) = dedup.get(&normalized) {
                // Found structurally equivalent expression — reuse fitness
                self.dedup_hits.fetch_add(1, atomic::Ordering::Relaxed);
                let mut cache = self.cache.lock().unwrap();
                cache.put(key, cached);
                return cached;
            }
        }

        // Compute fitness
        self.total_evals.fetch_add(1, atomic::Ordering::Relaxed);
        let fitness = self.evaluator.fitness(expr);

        // Store in both caches
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(key, fitness);
        }
        {
            let mut dedup = self.dedup.lock().unwrap();
            dedup.insert(normalized, fitness);
        }

        fitness
    }

    fn fitness_batch(&self, exprs: &[Expr]) -> Vec<f64> {
        let mut results = vec![0.0; exprs.len()];
        let mut to_compute: Vec<(usize, Expr, String, String)> = Vec::new();

        // First pass: check both caches
        {
            let mut cache = self.cache.lock().unwrap();
            let dedup = self.dedup.lock().unwrap();

            for (idx, expr) in exprs.iter().enumerate() {
                let key = format!("{:?}", expr);
                if let Some(&cached) = cache.get(&key) {
                    results[idx] = cached;
                } else {
                    let normalized = normalize_expression(expr);
                    if let Some(&cached) = dedup.get(&normalized) {
                        cache.put(key, cached);
                        self.dedup_hits.fetch_add(1, atomic::Ordering::Relaxed);
                        results[idx] = cached;
                    } else {
                        to_compute.push((idx, expr.clone(), key, normalized));
                    }
                }
            }
        }

        // Compute missing fitness values
        if !to_compute.is_empty() {
            let exprs_to_compute: Vec<Expr> = to_compute
                .iter()
                .map(|(_, expr, _, _)| expr.clone())
                .collect();
            let computed = self.evaluator.fitness_batch(&exprs_to_compute);
            let n_computed = computed.len();

            // Update caches and results
            {
                let mut cache = self.cache.lock().unwrap();
                let mut dedup = self.dedup.lock().unwrap();

                for ((idx, _, key, normalized), fitness) in
                    to_compute.into_iter().zip(computed.into_iter())
                {
                    cache.put(key, fitness);
                    dedup.insert(normalized, fitness);
                    results[idx] = fitness;
                }
            }

            self.total_evals
                .fetch_add(n_computed as u64, atomic::Ordering::Relaxed);
        }

        results
    }

    fn supports_batch(&self) -> bool {
        true
    }
}

/// Batch-optimized fitness evaluator for parallel processing
pub struct BatchFitnessEvaluator<E: FitnessEvaluator> {
    evaluator: E,
    batch_size: usize,
}

impl<E: FitnessEvaluator> BatchFitnessEvaluator<E> {
    /// Create a new batch evaluator
    pub fn new(evaluator: E, batch_size: usize) -> Self {
        Self {
            evaluator,
            batch_size,
        }
    }
}

impl<E: FitnessEvaluator + Clone> FitnessEvaluator for BatchFitnessEvaluator<E> {
    fn fitness(&self, expr: &Expr) -> f64 {
        self.evaluator.fitness(expr)
    }

    fn fitness_batch(&self, exprs: &[Expr]) -> Vec<f64> {
        use rayon::prelude::*;

        // Process in parallel batches
        exprs
            .par_chunks(self.batch_size)
            .flat_map(|chunk| self.evaluator.fitness_batch(chunk))
            .collect()
    }

    fn supports_batch(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::collections::HashMap;

    #[test]
    fn test_fitness_evaluator() {
        let mut data = HashMap::new();
        data.insert("x".to_string(), Array2::<f64>::zeros((10, 5)));

        let returns = Array2::<f64>::zeros((10, 5));

        let evaluator = BacktestFitnessEvaluator::new(data, returns);

        let expr = Expr::Column("x".to_string()).add(Expr::Literal(Literal::Float(1.0)));

        let fitness = evaluator.fitness(&expr);
        assert!(fitness > 0.0);
    }
}
