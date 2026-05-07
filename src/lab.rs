//! AlfarsLab — unified entry point for factor research workflows.
//!
//! Encapsulates ClickHouseSource, DataPool, and FactorRegistry behind
//! a simple API: register → calc → run (or evaluate_and_backtest_each
//! for streaming multi-factor).
//!
//! ```ignore
//! let mut lab = AlfarsLab::new(ClickHouseSource::from_env())
//!     .with_filter("symbols not like '%BJ'")
//!     .with_years(2010, 2025);
//!
//! lab.register("wcr", "1d:sum(5m:vol * 5m:close) / 1d:sum(5m:vol) / 1d:mean(5m:close)")?;
//!
//! let panel = lab.calc("output.csv")?;
//! let result = lab.run(&panel)?;
//! println!("Sharpe: {:.4}", result.sharpe_ratio);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::backtest::{BacktestConfig, BacktestEngine, BacktestResult};
use crate::data::clickhouse::ClickHouseSource;
use crate::data::layer::{DataLayer, PriceMatrix};
use crate::data::pool::{DataPool, DataPoolConfig};
use crate::expr::registry::FactorRegistry;
use crate::expr::registry::config::{FactorPanel, FactorSlice};
use crate::gp::evolution::run_gp;
use crate::gp::fitness::RealBacktestFitnessEvaluator;
use crate::gp::types::{Function, GPConfig, Terminal, to_parseable_string};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub struct AlfarsLab {
    pool: DataPool,
    registry: FactorRegistry,
    backtest_config: BacktestConfig,
    start_year: Option<i32>,
    end_year: Option<i32>,
}

impl AlfarsLab {
    pub fn new(source: ClickHouseSource) -> Self {
        Self {
            pool: DataPool::new(source, String::new(), DataPoolConfig::default()),
            registry: FactorRegistry::new(),
            backtest_config: BacktestConfig::default(),
            start_year: None,
            end_year: None,
        }
    }

    pub fn new_with_config(source: ClickHouseSource, pool_config: DataPoolConfig) -> Self {
        Self {
            pool: DataPool::new(source, String::new(), pool_config),
            registry: FactorRegistry::new(),
            backtest_config: BacktestConfig::default(),
            start_year: None,
            end_year: None,
        }
    }

    pub fn with_filter(mut self, filter: &str) -> Self {
        self.pool.set_pre_filter(filter);
        self
    }

    pub fn with_years(mut self, start: i32, end: i32) -> Self {
        self.start_year = Some(start);
        self.end_year = Some(end);
        self
    }

    pub fn with_backtest_config(mut self, config: BacktestConfig) -> Self {
        self.backtest_config = config;
        self
    }

    pub fn register(&mut self, name: &str, expression: &str) -> Result<&mut Self, String> {
        self.registry.register(name, expression)?;
        Ok(self)
    }

    /// Evaluate all registered factors (raw values + CS pipeline), returning
    /// factor matrices aligned to a common PriceMatrix. For multi-factor backtest.
    ///
    /// NOTE: For large factor sets (100+), consider `evaluate_and_backtest_each()`
    /// which processes factors in configurable batches to avoid OOM.
    pub fn evaluate(
        &self,
    ) -> Result<
        (
            HashMap<String, ndarray::Array2<f64>>,
            crate::data::layer::PriceMatrix,
        ),
        String,
    > {
        let (start_year, end_year) = self.years()?;
        let base_filter = self.pool.pre_filter().to_string();
        let mut all_slices = Vec::new();

        for year in start_year..=end_year {
            let start = format!("{}-01-01", year);
            let end = format!("{}-01-01", year + 1);
            let mut year_dl = self.pool.borrow_year(year, &start, &end);
            let results = self
                .registry
                .compute_cs_pipeline(&mut year_dl)
                .map_err(|e| format!("Year {}: {}", year, e))?;
            all_slices.extend(results.into_values());
            self.pool.return_year(year, year_dl);
        }

        let start_full = format!("{}-01-01", start_year);
        let end_full = format!("{}-01-01", end_year + 1);
        let prices = self
            .pool
            .get_prices(&start_full, &end_full)
            .map_err(|e| format!("Price query: {:?}", e))?;

        let mut matrices = HashMap::new();
        for name in self.registry.list() {
            let factor_slices: Vec<_> = all_slices
                .iter()
                .filter(|s| s.factor_name == name)
                .cloned()
                .collect();
            if factor_slices.is_empty() {
                continue;
            }
            matrices.insert(name, prices.build_factor_matrix(&factor_slices));
        }
        // Extract owned PriceMatrix from Arc (clone) for return
        Ok((matrices, (*prices).clone()))
    }

    /// Streaming per-factor evaluate + backtest — avoids OOM with many factors.
    ///
    /// Processes factors in configurable batches (`DataPoolConfig.backtest_batch_size`).
    /// For each batch: accumulates FactorSlices across all years using DataPool
    /// (with configurable caching), then backtests each factor independently.
    /// Matrices and slices are released per-factor, bounding peak memory.
    ///
    /// Returns per-factor results ordered by factor name.
    pub fn evaluate_and_backtest_each(&self) -> Result<Vec<(String, BacktestResult)>, String> {
        let (start_year, end_year) = self.years()?;
        let pool_config = self.pool.config();
        let batch_size = pool_config.backtest_batch_size;

        // Query full PriceMatrix once — shared Arc across all factors
        let start_full = format!("{}-01-01", start_year);
        let end_full = format!("{}-01-01", end_year + 1);
        let prices = self
            .pool
            .get_prices(&start_full, &end_full)
            .map_err(|e| format!("Price query: {:?}", e))?;

        let factor_names = self.registry.list();
        if factor_names.is_empty() {
            return Err("No factors registered".to_string());
        }

        // Collect expressions from the main registry
        let expr_map: HashMap<String, String> = factor_names
            .iter()
            .filter_map(|name| {
                self.registry
                    .get(name)
                    .map(|info| (name.clone(), info.expression.clone()))
            })
            .collect();

        let mut results: Vec<(String, BacktestResult)> = Vec::with_capacity(factor_names.len());

        let n_batches = factor_names.len().div_ceil(batch_size);
        let mut batch_no = 0usize;
        for batch in factor_names.chunks(batch_size) {
            batch_no += 1;
            eprintln!("[batch {}/{}] {} factors: {:?}", batch_no, n_batches, batch.len(), batch);
            // Accumulate per-factor slices across years
            let mut batch_slices: HashMap<String, Vec<FactorSlice>> = batch
                .iter()
                .map(|n| {
                    (
                        n.clone(),
                        Vec::with_capacity((end_year - start_year + 1) as usize),
                    )
                })
                .collect();

            for year in start_year..=end_year {
                eprintln!("[batch {}/{}] year {}", batch_no, n_batches, year);
                let start = format!("{}-01-01", year);
                let end = format!("{}-01-01", year + 1);

                // Temp registry with just this batch's factors
                let mut batch_reg = FactorRegistry::new();
                for name in batch {
                    if let Some(expr) = expr_map.get(name) {
                        batch_reg
                            .register(name, expr)
                            .map_err(|e| format!("Year {}: {}", year, e))?;
                    }
                }

                let mut year_dl = self.pool.borrow_year(year, &start, &end);
                let year_results = batch_reg
                    .compute_cs_pipeline(&mut year_dl)
                    .map_err(|e| format!("Year {}: {}", year, e))?;

                for (name, slice) in year_results {
                    if let Some(vec) = batch_slices.get_mut(&name) {
                        vec.push(slice);
                    }
                }
                self.pool.return_year(year, year_dl);
            }

            // All years collected for this batch — backtest each factor
            // Use run_with_qcut to match the calc()+run() path: reuses pre-computed
            // qcut from FactorSlice (via build_qcut_matrix) instead of recomputing
            // quantile groups (which uses a different tie-breaking algorithm).
            let engine = BacktestEngine::with_config(self.backtest_config.clone());
            for name in batch {
                if let Some(slices) = batch_slices.remove(name) {
                    let mat = prices.build_factor_matrix(&slices);
                    let qcut_mat = prices.build_qcut_matrix(&slices);
                    match engine.run_with_qcut(mat, &qcut_mat, &prices) {
                        Ok(result) => results.push((name.clone(), result)),
                        Err(e) => eprintln!("  [warn] backtest {}: {}", name, e),
                    }
                    // mat and slices dropped here
                }
            }
        }

        results.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(results)
    }

    /// Multi-factor equal-weight combination backtest.
    pub fn run_multi(
        &self,
        factor_mats: &[ndarray::Array2<f64>],
        prices: &crate::data::layer::PriceMatrix,
    ) -> Result<BacktestResult, String> {
        BacktestEngine::with_config(self.backtest_config.clone())
            .run_multi_with_prices(factor_mats, prices)
    }

    pub fn set_filter(&mut self, filter: &str) {
        self.pool.set_pre_filter(filter);
    }

    pub fn set_years(&mut self, start: i32, end: i32) {
        self.start_year = Some(start);
        self.end_year = Some(end);
    }

    pub fn set_backtest_config(&mut self, config: BacktestConfig) {
        self.backtest_config = config;
    }

    fn years(&self) -> Result<(i32, i32), String> {
        match (self.start_year, self.end_year) {
            (Some(s), Some(e)) if s <= e => Ok((s, e)),
            _ => Err("call with_years(start, end) before calc/run".into()),
        }
    }

    /// Compute all registered factors across the configured year range,
    /// writing results to `csv_path`. Returns the FactorPanel for subsequent
    /// backtest calls.
    ///
    /// Uses `DataPoolConfig.calc_parallel_years` for parallel year processing.
    pub fn calc(&self, csv_path: &str) -> Result<FactorPanel, String> {
        let (start_year, end_year) = self.years()?;
        let pool_config = self.pool.config();
        let parallel_years = pool_config.calc_parallel_years;

        let years: Vec<i32> = (start_year..=end_year).collect();

        // Bounded parallel: each year uses pool.borrow_year/return_year.
        // Peak memory depends on DataPoolConfig.cache_policy and calc_parallel_years.
        let pool_ref = &self.pool;
        let registry_ref = &self.registry;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(parallel_years.min(years.len()))
            .build()
            .map_err(|e| format!("rayon pool: {}", e))?;

        let year_results: Result<Vec<_>, _> = pool.install(|| {
            use rayon::prelude::*;
            years
                .par_iter()
                .map(|&year| {
                    let start = format!("{}-01-01", year);
                    let end = format!("{}-01-01", year + 1);
                    let mut year_dl = pool_ref.borrow_year(year, &start, &end);

                    let results = registry_ref
                        .compute_cs_pipeline(&mut year_dl)
                        .map_err(|e| format!("Year {}: {}", year, e))?;

                    let slices: Vec<FactorSlice> = results.into_values().collect();
                    pool_ref.return_year(year, year_dl);
                    Ok::<_, String>((year, slices))
                })
                .collect::<Result<Vec<_>, _>>()
        });
        let mut year_results = year_results?;
        year_results.sort_by_key(|(year, _)| *year);

        // Sequential CSV write
        let mut wtr = csv::Writer::from_path(csv_path).map_err(|e| format!("CSV: {}", e))?;
        FactorSlice::write_header(&mut wtr);
        let mut all_slices: Vec<FactorSlice> = Vec::new();
        for (year, slices) in year_results {
            for slice in &slices {
                slice
                    .write_to(&mut wtr)
                    .map_err(|e| format!("Year {} CSV: {}", year, e))?;
            }
            all_slices.extend(slices);
        }
        wtr.flush().map_err(|e| format!("CSV flush: {}", e))?;

        let panel = FactorPanel {
            factor_names: self.registry.list(),
            slices: all_slices,
        };
        Ok(panel)
    }

    /// Run genetic programming to discover alpha factors.
    ///
    /// Queries price data for the configured year range (once, shared via Arc),
    /// runs GP evolution with real backtest-based fitness evaluation, and
    /// returns discovered expressions with their metrics.
    pub fn mine_factors(
        &self,
        gp_config: GPConfig,
        num_factors: usize,
        max_symbols: usize,
    ) -> Result<Vec<(String, f64, f64, f64, f64, usize)>, String> {
        let (start_year, end_year) = self.years()?;
        let base_filter = self.pool.pre_filter().to_string();

        // Use pool's shared PriceMatrix
        let start_full = format!("{}-01-01", start_year);
        let end_full = format!("{}-01-01", end_year + 1);
        let prices_arc = self
            .pool
            .get_prices(&start_full, &end_full)
            .map_err(|e| format!("Price query: {:?}", e))?;

        // Slice to top N symbols if needed
        let prices_arc = if max_symbols > 0 && prices_arc.symbols.len() > max_symbols {
            Arc::new(Self::slice_price_matrix_top(&prices_arc, max_symbols))
        } else {
            prices_arc
        };

        // Build data HashMap from Arc (one-time clone for evaluator construction)
        let mut data = HashMap::new();
        if prices_arc.close.dim() == prices_arc.returns.dim() {
            data.insert("close".to_string(), prices_arc.close.clone());
        }
        if prices_arc.open.dim() == prices_arc.returns.dim() {
            data.insert("open".to_string(), prices_arc.open.clone());
        }
        if prices_arc.high.dim() == prices_arc.returns.dim() {
            data.insert("high".to_string(), prices_arc.high.clone());
        }
        if prices_arc.low.dim() == prices_arc.returns.dim() {
            data.insert("low".to_string(), prices_arc.low.clone());
        }
        if prices_arc.vwap.dim() == prices_arc.returns.dim() {
            data.insert("vwap".to_string(), prices_arc.vwap.clone());
        }

        let evaluator = RealBacktestFitnessEvaluator::new(data, prices_arc);

        let terminals = vec![
            Terminal::Ephemeral,
            Terminal::Constant(1.0),
            Terminal::Constant(2.0),
            Terminal::Variable("close".to_string()),
            Terminal::Variable("open".to_string()),
            Terminal::Variable("high".to_string()),
            Terminal::Variable("low".to_string()),
            Terminal::Variable("vwap".to_string()),
        ];

        let functions = vec![
            Function::add(),
            Function::sub(),
            Function::mul(),
            Function::div(),
            Function::power(),
            Function::sqrt(),
            Function::abs(),
            Function::neg(),
            Function::log(),
            Function::sign(),
            Function::exp(),
            Function::rank(),
            Function::cs_scale(),
            Function::ts_mean(),
            Function::ts_std(),
            Function::ts_max(),
            Function::ts_min(),
            Function::ts_sum(),
            Function::delay(),
            Function::ts_delta(),
            Function::ts_rank(),
            Function::decay_linear(),
            Function::correlation(),
            Function::ts_covariance(),
        ];

        let mut rng = StdRng::from_entropy();
        let mut results = Vec::with_capacity(num_factors);

        for _ in 0..num_factors {
            let (best_expr, best_fitness) = run_gp(
                &gp_config,
                &evaluator,
                terminals.clone(),
                functions.clone(),
                &mut rng,
            );

            let expr_str = to_parseable_string(&best_expr);
            let ic = evaluator.get_last_ic();
            let ir = evaluator.get_last_ir();
            let turnover = evaluator.get_last_turnover();
            let complexity = evaluator.get_last_complexity();

            results.push((expr_str, best_fitness, ic, ir, turnover, complexity));
        }

        Ok(results)
    }

    /// Slice a PriceMatrix to the top N symbols by average volume (close × volume proxy).
    fn slice_price_matrix_top(prices: &PriceMatrix, n: usize) -> PriceMatrix {
        let n_assets = prices.symbols.len();
        let n_limit = n.min(n_assets);
        let col_indices: Vec<usize> = (0..n_limit).collect();
        PriceMatrix {
            dates: prices.dates.clone(),
            symbols: prices.symbols[..n_limit].to_vec(),
            close: slice_columns(&prices.close, &col_indices),
            open: slice_columns(&prices.open, &col_indices),
            high: slice_columns(&prices.high, &col_indices),
            low: slice_columns(&prices.low, &col_indices),
            vwap: slice_columns(&prices.vwap, &col_indices),
            returns: slice_columns(&prices.returns, &col_indices),
            tradable: slice_columns(&prices.tradable, &col_indices),
        }
    }

    /// Run backtest on a pre-computed FactorPanel.
    ///
    /// Queries price data for the configured year range, builds aligned
    /// factor/qcut matrices, and runs the backtest engine.
    pub fn run(&self, panel: &FactorPanel) -> Result<BacktestResult, String> {
        let (start_year, end_year) = self.years()?;
        let start_full = format!("{}-01-01", start_year);
        let end_full = format!("{}-01-01", end_year + 1);
        let prices = self
            .pool
            .get_prices(&start_full, &end_full)
            .map_err(|e| format!("Price query: {:?}", e))?;

        let factor_mat = panel.build_factor_matrix(&prices);
        let qcut_mat = prices.build_qcut_matrix(&panel.slices);
        BacktestEngine::with_config(self.backtest_config.clone())
            .run_with_qcut(factor_mat, &qcut_mat, &prices)
    }
}

fn slice_columns(data: &Array2<f64>, col_indices: &[usize]) -> Array2<f64> {
    let n_rows = data.shape()[0];
    let n_cols = col_indices.len();
    let mut result = Array2::<f64>::zeros((n_rows, n_cols));
    for (j, &col) in col_indices.iter().enumerate() {
        for i in 0..n_rows {
            result[[i, j]] = data[[i, col]];
        }
    }
    result
}
