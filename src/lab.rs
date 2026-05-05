//! AlfarsLab — unified entry point for factor research workflows.
//!
//! Encapsulates ClickHouseSource, DataLayer, and FactorRegistry behind
//! a simple two-phase API: register → calc → run.
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

use crate::backtest::{BacktestConfig, BacktestEngine, BacktestResult};
use crate::data::clickhouse::ClickHouseSource;
use crate::data::layer::{DataLayer, PriceMatrix};
use crate::expr::registry::FactorRegistry;
use crate::expr::registry::config::{FactorPanel, FactorSlice};
use crate::gp::evolution::run_gp;
use crate::gp::fitness::RealBacktestFitnessEvaluator;
use crate::gp::types::{Function, GPConfig, Terminal, to_parseable_string};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub struct AlfarsLab {
    source: ClickHouseSource,
    dl: DataLayer,
    registry: FactorRegistry,
    backtest_config: BacktestConfig,
    start_year: Option<i32>,
    end_year: Option<i32>,
}

impl AlfarsLab {
    pub fn new(source: ClickHouseSource) -> Self {
        Self {
            dl: DataLayer::new(source.clone()),
            source,
            registry: FactorRegistry::new(),
            backtest_config: BacktestConfig::default(),
            start_year: None,
            end_year: None,
        }
    }

    pub fn with_filter(mut self, filter: &str) -> Self {
        self.dl.set_pre_filter(filter);
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
        let base_filter = self.dl.pre_filter().to_string();
        let mut all_slices = Vec::new();

        for year in start_year..=end_year {
            let start = format!("{}-01-01", year);
            let end = format!("{}-01-01", year + 1);
            let mut year_dl = DataLayer::new(self.source.clone());
            year_dl.set_pre_filter(&format!("{}:{} {}", start, end, base_filter));
            let results = self
                .registry
                .compute_cs_pipeline(&mut year_dl)
                .map_err(|e| format!("Year {}: {}", year, e))?;
            all_slices.extend(results.into_values());
        }

        let start_full = format!("{}-01-01", start_year);
        let end_full = format!("{}-01-01", end_year + 1);
        let mut dl = DataLayer::new(self.source.clone());
        dl.set_pre_filter(&format!("{}:{} {}", start_full, end_full, base_filter));
        let prices = dl
            .query_price_matrix()
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
        Ok((matrices, prices))
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
        self.dl.set_pre_filter(filter);
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
    pub fn calc(&self, csv_path: &str) -> Result<FactorPanel, String> {
        let (start_year, end_year) = self.years()?;
        let base_filter = self.dl.pre_filter().to_string();
        let mut all_slices: Vec<FactorSlice> = Vec::new();

        let mut wtr = csv::Writer::from_path(csv_path).map_err(|e| format!("CSV: {}", e))?;
        FactorSlice::write_header(&mut wtr);

        for year in start_year..=end_year {
            let start = format!("{}-01-01", year);
            let end = format!("{}-01-01", year + 1);
            let mut year_dl = DataLayer::new(self.source.clone());
            year_dl.set_pre_filter(&format!("{}:{} {}", start, end, base_filter));

            let results = self
                .registry
                .compute_cs_pipeline(&mut year_dl)
                .map_err(|e| format!("Year {}: {}", year, e))?;

            for (_name, slice) in &results {
                slice
                    .write_to(&mut wtr)
                    .map_err(|e| format!("Year {} CSV: {}", year, e))?;
            }
            all_slices.extend(results.into_values());
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
    /// Queries price data for the configured year range, runs GP evolution
    /// with real backtest-based fitness evaluation, and returns discovered
    /// expressions with their metrics.
    pub fn mine_factors(
        &self,
        gp_config: GPConfig,
        num_factors: usize,
    ) -> Result<Vec<(String, f64, f64, f64, f64, usize)>, String> {
        let (start_year, end_year) = self.years()?;
        let base_filter = self.dl.pre_filter().to_string();
        let start_full = format!("{}-01-01", start_year);
        let end_full = format!("{}-01-01", end_year + 1);

        let mut dl = DataLayer::new(self.source.clone());
        dl.set_pre_filter(&format!("{}:{} {}", start_full, end_full, base_filter));
        let prices = dl
            .query_price_matrix()
            .map_err(|e| format!("Price query: {:?}", e))?;

        let mut data = HashMap::new();
        if prices.close.dim() == prices.returns.dim() {
            data.insert("close".to_string(), prices.close.clone());
        }
        if prices.open.dim() == prices.returns.dim() {
            data.insert("open".to_string(), prices.open.clone());
        }
        if prices.high.dim() == prices.returns.dim() {
            data.insert("high".to_string(), prices.high.clone());
        }
        if prices.low.dim() == prices.returns.dim() {
            data.insert("low".to_string(), prices.low.clone());
        }
        if prices.vwap.dim() == prices.returns.dim() {
            data.insert("vwap".to_string(), prices.vwap.clone());
        }

        let evaluator = RealBacktestFitnessEvaluator::new(data, prices);

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

    /// Run backtest on a pre-computed FactorPanel.
    ///
    /// Queries price data for the configured year range, builds aligned
    /// factor/qcut matrices, and runs the backtest engine.
    pub fn run(&self, panel: &FactorPanel) -> Result<BacktestResult, String> {
        let (start_year, end_year) = self.years()?;
        let base_filter = self.dl.pre_filter().to_string();
        let start_full = format!("{}-01-01", start_year);
        let end_full = format!("{}-01-01", end_year + 1);

        let mut dl = DataLayer::new(self.source.clone());
        dl.set_pre_filter(&format!("{}:{} {}", start_full, end_full, base_filter));
        let prices = dl
            .query_price_matrix()
            .map_err(|e| format!("Price query: {:?}", e))?;

        let factor_mat = panel.build_factor_matrix(&prices);
        let qcut_mat = prices.build_qcut_matrix(&panel.slices);
        BacktestEngine::with_config(self.backtest_config.clone())
            .run_with_qcut(factor_mat, &qcut_mat, &prices)
    }
}
