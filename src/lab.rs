//! AlfarsLab — unified entry point for factor research workflows.
//!
//! Encapsulates ClickHouseSource, DataPool, and FactorRegistry behind
//! a simple API: register → calc → run (or evaluate_and_backtest_each
//! for streaming multi-factor).
//!
//! ```ignore
//! let mut lab = AlfarsLab::new(ClickHouseSource::from_env());
//! lab.set_pool("symbols not like '%BJ'");
//! lab.set_duration(2010, 2025);
//!
//! lab.register("wcr", "1d:sum(5m:vol * 5m:close) / 1d:sum(5m:vol) / 1d:mean(5m:close)")?;
//!
//! let panel = lab.calc("output.csv")?;
//! let result = lab.run(&panel)?;
//! println!("Sharpe: {:.4}", result.sharpe_ratio);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::backtest::{BacktestConfig, BacktestEngine, BacktestResult, ExecConfig};
use crate::data::clickhouse::ClickHouseSource;
use crate::data::layer::{DataLayer, PriceMatrix};
use crate::data::pool::{DataPool, DataPoolConfig};
use crate::expr;
use crate::expr::registry::FactorRegistry;
use crate::expr::registry::config::{FactorPanel, FactorSlice};
use crate::gp::evolution::run_gp as gp_evolve;
use crate::gp::fitness::RealBacktestFitnessEvaluator;
use crate::gp::types::{Function, GPConfig, Terminal, to_parseable_string};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub struct AlfarsLab {
    pool: DataPool,
    registry: FactorRegistry,
    backtest_config: BacktestConfig,
    exec_config: Option<ExecConfig>,
    start_year: Option<i32>,
    end_year: Option<i32>,
    last_panel: Mutex<Option<FactorPanel>>,
    last_prices: Mutex<Option<Arc<PriceMatrix>>>,
    gp_fields: Mutex<Option<Vec<String>>>,
    gp_ops: Mutex<Option<Vec<String>>>,
    gp_seeds: Mutex<Option<Vec<expr::Expr>>>,
}

impl AlfarsLab {
    pub fn new(source: ClickHouseSource) -> Self {
        Self {
            pool: DataPool::new(source, String::new(), DataPoolConfig::default()),
            registry: FactorRegistry::new(),
            backtest_config: BacktestConfig::default(),
            exec_config: None,
            start_year: None,
            end_year: None,
            last_panel: Mutex::new(None),
            last_prices: Mutex::new(None),
            gp_fields: Mutex::new(None),
            gp_ops: Mutex::new(None),
            gp_seeds: Mutex::new(None),
        }
    }

    pub fn new_with_config(source: ClickHouseSource, pool_config: DataPoolConfig) -> Self {
        Self {
            pool: DataPool::new(source, String::new(), pool_config),
            registry: FactorRegistry::new(),
            backtest_config: BacktestConfig::default(),
            exec_config: None,
            start_year: None,
            end_year: None,
            last_panel: Mutex::new(None),
            last_prices: Mutex::new(None),
            gp_fields: Mutex::new(None),
            gp_ops: Mutex::new(None),
            gp_seeds: Mutex::new(None),
        }
    }

    pub fn register(&mut self, name: &str, expression: &str) -> Result<&mut Self, String> {
        self.registry.register(name, expression)?;
        Ok(self)
    }

    /// All canonical field names available at daily frequency.
    pub fn avail_fields() -> Vec<String> {
        crate::data::frequency::avail_fields_1d()
    }

    /// All GP operator names (24 total).
    pub fn avail_ops() -> Vec<String> {
        vec![
            "add".into(),
            "sub".into(),
            "mul".into(),
            "div".into(),
            "power".into(),
            "sqrt".into(),
            "abs".into(),
            "neg".into(),
            "log".into(),
            "sign".into(),
            "exp".into(),
            "rank".into(),
            "cs_scale".into(),
            "ts_mean".into(),
            "ts_std".into(),
            "ts_max".into(),
            "ts_min".into(),
            "ts_sum".into(),
            "delay".into(),
            "ts_delta".into(),
            "ts_rank".into(),
            "decay_linear".into(),
            "correlation".into(),
            "ts_covariance".into(),
        ]
    }

    /// Set GP terminal fields. If not called, defaults to close/open/high/low/vwap.
    pub fn set_gp_fields(&self, fields: Vec<String>) {
        *self.gp_fields.lock().unwrap() = Some(fields);
    }

    /// Set GP operators. If not called, defaults to all 24 operators.
    pub fn set_gp_ops(&self, ops: Vec<String>) {
        *self.gp_ops.lock().unwrap() = Some(ops);
    }

    /// Set seed expressions for initial GP population.
    /// Expressions are parsed via `crate::expr::parse_expression`.
    pub fn set_gp_seed(&self, seeds: Vec<String>) -> Result<(), String> {
        let parsed: Result<Vec<_>, _> = seeds
            .iter()
            .map(|s| crate::expr::parse_expression(s))
            .collect();
        *self.gp_seeds.lock().unwrap() = Some(parsed?);
        Ok(())
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
        self.evaluate_and_backtest_with_config(&self.backtest_config.clone())
    }

    /// Internal helper: evaluate and backtest each factor, using the given config.
    fn evaluate_and_backtest_with_config(
        &self,
        config: &BacktestConfig,
    ) -> Result<Vec<(String, BacktestResult)>, String> {
        let (start_year, end_year) = self.years()?;
        let pool_config = self.pool.config();
        let batch_size = pool_config.backtest_batch_size;

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
            eprintln!(
                "[batch {}/{}] {} factors: {:?}",
                batch_no,
                n_batches,
                batch.len(),
                batch
            );
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

                let mut batch_reg = FactorRegistry::new();
                for name in batch {
                    if let Some(expr) = expr_map.get(name) {
                        batch_reg
                            .register(name, expr)
                            .map_err(|e| format!("Year {}: {}", year, e))?;
                    }
                }

                let mut year_dl = self.pool.borrow_year(year, &start, &end);
                let year_results = match batch_reg.compute_cs_pipeline(&mut year_dl) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!(
                            "[warn] batch {}/{} year {} failed: {}, skipping",
                            batch_no, n_batches, year, e
                        );
                        drop(year_dl);
                        continue;
                    }
                };

                for (name, slice) in year_results {
                    if let Some(vec) = batch_slices.get_mut(&name) {
                        vec.push(slice);
                    }
                }
                self.pool.return_year(year, year_dl);
            }

            let engine = BacktestEngine::with_config(config.clone());
            for name in batch {
                if let Some(slices) = batch_slices.remove(name) {
                    let mat = prices.build_factor_matrix(&slices);
                    let qcut_mat = prices.build_qcut_matrix(&slices);
                    match engine.run_with_qcut(mat, &qcut_mat, &prices) {
                        Ok(result) => results.push((name.clone(), result)),
                        Err(e) => eprintln!("  [warn] backtest {}: {}", name, e),
                    }
                }
            }
        }

        results.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(results)
    }

    /// Compute all registered factors and backtest each one.
    ///
    /// Returns `Vec<Factor>` with expression and performance per factor.
    /// If `factor_config` is provided, its parameters override the current
    /// backtest config for this call only (not persisted).
    pub fn run_factors(&self, factor_config: Option<&FactorConfig>) -> Result<Vec<Factor>, String> {
        let config = if let Some(fc) = factor_config {
            let mut c = self.backtest_config.clone();
            fc.apply_to(&mut c);
            c
        } else {
            self.backtest_config.clone()
        };

        let results = self.evaluate_and_backtest_with_config(&config)?;

        Ok(results
            .into_iter()
            .map(|(name, result)| Factor {
                expression: self
                    .registry
                    .get(&name)
                    .map(|info| info.expression.clone())
                    .unwrap_or_default(),
                perf: ExprPerf { result },
                name,
            })
            .collect())
    }

    /// Multi-factor equal-weight combination backtest.
    pub fn run_multi(
        &self,
        factor_mats: &[ndarray::Array2<f64>],
        prices: &crate::data::layer::PriceMatrix,
    ) -> Result<BacktestResult, String> {
        BacktestEngine::with_config(self.backtest_config.clone()).run_multi(factor_mats, prices)
    }

    pub fn set_pool(&mut self, filter: &str) {
        self.pool.set_pre_filter(filter);
    }

    pub fn set_duration(&mut self, start: i32, end: i32) {
        self.start_year = Some(start);
        self.end_year = Some(end);
    }

    pub fn set_exec_cfg(&mut self, config: ExecConfig) {
        config.apply_to_fee_config(&mut self.backtest_config.fee_config);
        self.exec_config = Some(config);
    }

    pub fn set_backtest_config(&mut self, config: BacktestConfig) {
        if let Some(ref ec) = self.exec_config {
            let mut merged = config;
            ec.apply_to_fee_config(&mut merged.fee_config);
            self.backtest_config = merged;
        } else {
            self.backtest_config = config;
        }
    }

    fn years(&self) -> Result<(i32, i32), String> {
        match (self.start_year, self.end_year) {
            (Some(s), Some(e)) if s <= e => Ok((s, e)),
            _ => Err("call set_duration(start, end) before calc/run".into()),
        }
    }

    /// Compute all registered factors across the configured year range.
    ///
    /// If `csv_path` is provided, writes factor values to CSV.
    /// Returns the FactorPanel and caches it internally for subsequent `run_bt()`.
    ///
    /// Uses `DataPoolConfig.calc_parallel_years` for parallel year processing.
    pub fn calc(&self, csv_path: Option<&str>) -> Result<FactorPanel, String> {
        let (start_year, end_year) = self.years()?;
        let pool_config = self.pool.config();
        let parallel_years = pool_config.calc_parallel_years;

        let years: Vec<i32> = (start_year..=end_year).collect();

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

        let mut all_slices: Vec<FactorSlice> = Vec::new();

        // Write CSV if path provided
        if let Some(path) = csv_path {
            let mut wtr = csv::Writer::from_path(path).map_err(|e| format!("CSV: {}", e))?;
            FactorSlice::write_header(&mut wtr);
            for (year, slices) in &year_results {
                for slice in slices {
                    slice
                        .write_to(&mut wtr)
                        .map_err(|e| format!("Year {} CSV: {}", year, e))?;
                }
            }
            wtr.flush().map_err(|e| format!("CSV flush: {}", e))?;
        }

        for (_, slices) in year_results {
            all_slices.extend(slices);
        }

        let panel = FactorPanel {
            factor_names: self.registry.list(),
            slices: all_slices,
        };

        // Cache prices for subsequent run_bt
        let start_full = format!("{}-01-01", start_year);
        let end_full = format!("{}-01-01", end_year + 1);
        let prices = self
            .pool
            .get_prices(&start_full, &end_full)
            .map_err(|e| format!("Price query: {:?}", e))?;
        *self.last_prices.lock().unwrap() = Some(prices);
        *self.last_panel.lock().unwrap() = Some(panel.clone());

        Ok(panel)
    }

    /// Run backtest on the last computed FactorPanel (from `calc()`).
    ///
    /// Uses cached panel and prices — no re-query to ClickHouse.
    /// Returns an error if `calc()` hasn't been called yet.
    pub fn run_bt(&self) -> Result<BacktestResult, String> {
        let panel_guard = self.last_panel.lock().unwrap();
        let prices_guard = self.last_prices.lock().unwrap();
        let panel = panel_guard
            .as_ref()
            .ok_or_else(|| "No panel computed — call calc() first".to_string())?;
        let prices = prices_guard
            .as_ref()
            .ok_or_else(|| "No prices cached — call calc() first".to_string())?;

        let factor_mat = panel.build_factor_matrix(prices);
        let qcut_mat = prices.build_qcut_matrix(&panel.slices);
        BacktestEngine::with_config(self.backtest_config.clone())
            .run_with_qcut(factor_mat, &qcut_mat, prices)
    }

    /// Run genetic programming and return discovered factors with backtest results.
    ///
    /// Each discovered factor's full `BacktestResult` is retrieved from the
    /// fitness evaluator (cached during GP evaluation — no re-computation).
    pub fn run_gp(
        &self,
        gp_config: GPConfig,
        num_factors: usize,
        max_symbols: usize,
    ) -> Result<Vec<(expr::Expr, BacktestResult)>, String> {
        let (start_year, end_year) = self.years()?;

        let start_full = format!("{}-01-01", start_year);
        let end_full = format!("{}-01-01", end_year + 1);
        let prices_arc = self
            .pool
            .get_prices(&start_full, &end_full)
            .map_err(|e| format!("Price query: {:?}", e))?;

        let prices_arc = if max_symbols > 0 && prices_arc.symbols.len() > max_symbols {
            Arc::new(Self::slice_price_matrix_top(&prices_arc, max_symbols))
        } else {
            prices_arc
        };

        let mut data = HashMap::new();
        let insert_if_valid = |data: &mut HashMap<_, _>, name: &str, arr: &Array2<f64>| {
            if arr.dim() == prices_arc.returns.dim() {
                data.insert(name.to_string(), arr.clone());
            }
        };
        insert_if_valid(&mut data, "close", &prices_arc.close);
        insert_if_valid(&mut data, "open", &prices_arc.open);
        insert_if_valid(&mut data, "high", &prices_arc.high);
        insert_if_valid(&mut data, "low", &prices_arc.low);
        insert_if_valid(&mut data, "vwap", &prices_arc.vwap);

        let evaluator = RealBacktestFitnessEvaluator::new(data, prices_arc);

        let terminals = build_gp_terminals(&self.gp_fields.lock().unwrap());
        let functions = build_gp_functions(&self.gp_ops.lock().unwrap());
        let seed_exprs = self.gp_seeds.lock().unwrap().clone();

        let mut rng = StdRng::from_entropy();
        let mut results = Vec::with_capacity(num_factors);

        for _ in 0..num_factors {
            let (best_expr, _best_fitness) = gp_evolve(
                &gp_config,
                &evaluator,
                terminals.clone(),
                functions.clone(),
                &mut rng,
                seed_exprs.as_deref(),
            );

            if let Some(bt) = evaluator.get_last_backtest() {
                results.push((best_expr, bt));
            }
        }

        Ok(results)
    }

    /// Legacy: discover factors returning scalar metrics only.
    /// Prefer `run_gp()` which also returns full backtest results.
    pub fn mine_factors(
        &self,
        gp_config: GPConfig,
        num_factors: usize,
        max_symbols: usize,
    ) -> Result<Vec<(String, f64, f64, f64, f64, usize)>, String> {
        let (start_year, end_year) = self.years()?;

        let start_full = format!("{}-01-01", start_year);
        let end_full = format!("{}-01-01", end_year + 1);
        let prices_arc = self
            .pool
            .get_prices(&start_full, &end_full)
            .map_err(|e| format!("Price query: {:?}", e))?;

        let prices_arc = if max_symbols > 0 && prices_arc.symbols.len() > max_symbols {
            Arc::new(Self::slice_price_matrix_top(&prices_arc, max_symbols))
        } else {
            prices_arc
        };

        let mut data = HashMap::new();
        let insert_if_valid = |data: &mut HashMap<_, _>, name: &str, arr: &Array2<f64>| {
            if arr.dim() == prices_arc.returns.dim() {
                data.insert(name.to_string(), arr.clone());
            }
        };
        insert_if_valid(&mut data, "close", &prices_arc.close);
        insert_if_valid(&mut data, "open", &prices_arc.open);
        insert_if_valid(&mut data, "high", &prices_arc.high);
        insert_if_valid(&mut data, "low", &prices_arc.low);
        insert_if_valid(&mut data, "vwap", &prices_arc.vwap);

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
            let (best_expr, best_fitness) = gp_evolve(
                &gp_config,
                &evaluator,
                terminals.clone(),
                functions.clone(),
                &mut rng,
                None,
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
            adj_factor: slice_columns(&prices.adj_factor, &col_indices),
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

fn build_gp_terminals(fields: &Option<Vec<String>>) -> Vec<Terminal> {
    let mut t = vec![
        Terminal::Ephemeral,
        Terminal::Constant(1.0),
        Terminal::Constant(2.0),
    ];
    match fields {
        Some(fields) => {
            for f in fields {
                t.push(Terminal::Variable(f.clone()));
            }
        }
        None => {
            for f in &["close", "open", "high", "low", "vwap"] {
                t.push(Terminal::Variable(f.to_string()));
            }
        }
    }
    t
}

fn build_gp_functions(ops: &Option<Vec<String>>) -> Vec<Function> {
    let all: Vec<Function> = vec![
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
    match ops {
        Some(names) => {
            let name_set: std::collections::HashSet<&str> =
                names.iter().map(|s| s.as_str()).collect();
            all.into_iter()
                .filter(|f| name_set.contains(f.name.as_str()))
                .collect()
        }
        None => all,
    }
}

/// Factor evaluation config — parameters for factor construction and backtest.
#[derive(Debug, Clone)]
pub struct FactorConfig {
    pub quantiles: usize,
    pub weight_method: String,
    pub long_top_n: usize,
    pub short_top_n: usize,
    pub rebalance_freq: usize,
}

impl Default for FactorConfig {
    fn default() -> Self {
        Self {
            quantiles: 10,
            weight_method: "equal".into(),
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 1,
        }
    }
}

impl FactorConfig {
    pub fn apply_to(&self, config: &mut BacktestConfig) {
        config.quantiles = self.quantiles;
        config.long_top_n = self.long_top_n;
        config.short_top_n = self.short_top_n;
        config.rebalance_freq = self.rebalance_freq;
        config.weight_method = match self.weight_method.as_str() {
            "equal" => crate::WeightMethod::Equal,
            "weighted" => crate::WeightMethod::Weighted,
            other => {
                eprintln!("[warn] unknown weight_method '{}', using Equal", other);
                crate::WeightMethod::Equal
            }
        };
    }
}

/// Per-factor backtest performance.
#[derive(Debug, Clone)]
pub struct ExprPerf {
    pub result: BacktestResult,
}

/// A factor with its expression and performance.
#[derive(Debug, Clone)]
pub struct Factor {
    pub name: String,
    pub expression: String,
    pub perf: ExprPerf,
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
