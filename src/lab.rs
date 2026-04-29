//! AlfarsLab — unified entry point for factor research workflows.
//!
//! Encapsulates ClickHouseSource, DataLayer, and FactorRegistry behind
//! a simple API: register → calc → backtest.

use crate::backtest::{BacktestConfig, BacktestEngine, BacktestResult};
use crate::data::clickhouse::ClickHouseSource;
use crate::data::layer::DataLayer;
use crate::expr::registry::config::FactorPanel;
use crate::expr::registry::FactorRegistry;

/// Convenience wrapper bundling data source, data layer, and factor registry.
///
/// ```ignore
/// let mut lab = AlfarsLab::new(ClickHouseSource::from_env())
///     .with_filter("symbols not like '%BJ'");
///
/// lab.register("wcr", "1d:sum(5m:vol * 5m:close) / 1d:sum(5m:vol) / 1d:mean(5m:close)")?;
///
/// let panel = lab.calc(2010, 2025)?;
/// panel.to_csv("output.csv")?;
///
/// let result = lab.backtest(&panel)?;
/// println!("Sharpe: {:.4}", result.sharpe_ratio);
/// ```
pub struct AlfarsLab {
    source: ClickHouseSource,
    dl: DataLayer,
    registry: FactorRegistry,
    backtest_config: BacktestConfig,
}

impl AlfarsLab {
    pub fn new(source: ClickHouseSource) -> Self {
        Self {
            dl: DataLayer::new(source.clone()),
            source,
            registry: FactorRegistry::new(),
            backtest_config: BacktestConfig::default(),
        }
    }

    /// Set the data filter (symbol exclusion, conditions, etc.).
    pub fn with_filter(mut self, filter: &str) -> Self {
        self.dl.set_pre_filter(filter);
        self
    }

    /// Set backtest configuration.
    pub fn with_backtest_config(mut self, config: BacktestConfig) -> Self {
        self.backtest_config = config;
        self
    }

    /// Register a factor expression.
    pub fn register(&mut self, name: &str, expression: &str) -> Result<&mut Self, String> {
        self.registry.register(name, expression)?;
        Ok(self)
    }

    /// Compute all registered factors across a year range.
    pub fn calc(&self, start_year: i32, end_year: i32) -> Result<FactorPanel, String> {
        self.registry.calc(&self.dl, start_year, end_year)
    }

    /// Compute + write CSV + backtest in one pass.
    ///
    /// Streams CSV per-year (avoids OOM), then runs backtest.
    pub fn run(
        &mut self,
        start_year: i32,
        end_year: i32,
        csv_path: &str,
    ) -> Result<BacktestResult, String> {
        use crate::expr::registry::config::FactorSlice;
        use std::time::Instant;

        let base_filter = self.dl.pre_filter().to_string();
        let mut all_slices: Vec<FactorSlice> = Vec::new();

        // Open CSV writer, write header once
        let mut wtr = csv::Writer::from_path(csv_path)
            .map_err(|e| format!("CSV: {}", e))?;
        FactorSlice::write_header(&mut wtr);

        let t0 = Instant::now();
        let mut total_query_ms = 0u64;
        let mut total_eval_ms = 0u64;
        let mut total_cs_ms = 0u64;
        let mut total_csv_ms = 0u64;
        let mut total_mktcap_ms = 0u64;
        for year in start_year..=end_year {
            let start = format!("{}-01-01", year);
            let end = format!("{}-01-01", year + 1);
            let pre_filter = format!("{}:{} {}", start, end, base_filter);

            let mut year_dl = DataLayer::new(self.source.clone());
            year_dl.set_pre_filter(&pre_filter);

            let t_q0 = Instant::now();
            let results = self.registry.compute_cs_pipeline(&mut year_dl)
                .map_err(|e| format!("Year {}: {}", year, e))?;
            // Timing breakdown is logged inside compute_cs_pipeline

            for (_name, slice) in &results {
                slice.write_to(&mut wtr)
                    .map_err(|e| format!("Year {} CSV: {}", year, e))?;
            }

            all_slices.extend(results.into_values());
        }
        wtr.flush().map_err(|e| format!("CSV flush: {}", e))?;
        let compute_secs = t0.elapsed().as_secs_f64();

        eprintln!("Compute: {:.1}s over {} years", compute_secs, end_year - start_year + 1);

        // Backtest — set full date range on dl first
        let start_full = format!("{}-01-01", start_year);
        let end_full = format!("{}-01-01", end_year + 1);
        self.dl.set_pre_filter(&format!("{}:{} {}", start_full, end_full, base_filter));

        let t1 = Instant::now();
        let prices = self.dl.query_price_matrix()
            .map_err(|e| format!("Price query: {:?}", e))?;
        let panel = FactorPanel {
            factor_names: self.registry.list(),
            slices: all_slices,
        };
        let factor_mat = panel.build_factor_matrix(&prices);
        let qcut_mat = prices.build_qcut_matrix(&panel.slices);
        let result = BacktestEngine::with_config(self.backtest_config.clone())
            .run_with_qcut(factor_mat, &qcut_mat, &prices)?;
        let bt_secs = t1.elapsed().as_secs_f64();

        eprintln!(
            "Backtest: {:.1}s (price query + factor build + engine)",
            bt_secs
        );
        eprintln!("Total: {:.1}s", compute_secs + bt_secs);
        Ok(result)
    }

    /// Run backtest on a computed panel.
    ///
    /// The DataLayer filter MUST already include the date range before calling.
    pub fn backtest(&mut self, panel: &FactorPanel) -> Result<BacktestResult, String> {
        let prices = self
            .dl
            .query_price_matrix()
            .map_err(|e| format!("{:?}", e))?;
        let factor_mat = panel.build_factor_matrix(&prices);
        BacktestEngine::with_config(self.backtest_config.clone())
            .run_with_prices(factor_mat, &prices)
    }
}
