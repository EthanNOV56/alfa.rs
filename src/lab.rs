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

use crate::backtest::{BacktestConfig, BacktestEngine, BacktestResult};
use crate::data::clickhouse::ClickHouseSource;
use crate::data::layer::DataLayer;
use crate::expr::registry::config::{FactorPanel, FactorSlice};
use crate::expr::registry::FactorRegistry;

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

        let mut wtr = csv::Writer::from_path(csv_path)
            .map_err(|e| format!("CSV: {}", e))?;
        FactorSlice::write_header(&mut wtr);

        for year in start_year..=end_year {
            let start = format!("{}-01-01", year);
            let end = format!("{}-01-01", year + 1);
            let mut year_dl = DataLayer::new(self.source.clone());
            year_dl.set_pre_filter(&format!("{}:{} {}", start, end, base_filter));

            let results = self.registry.compute_cs_pipeline(&mut year_dl)
                .map_err(|e| format!("Year {}: {}", year, e))?;

            for (_name, slice) in &results {
                slice.write_to(&mut wtr)
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
        let prices = dl.query_price_matrix()
            .map_err(|e| format!("Price query: {:?}", e))?;

        let factor_mat = panel.build_factor_matrix(&prices);
        let qcut_mat = prices.build_qcut_matrix(&panel.slices);
        BacktestEngine::with_config(self.backtest_config.clone())
            .run_with_qcut(factor_mat, &qcut_mat, &prices)
    }
}
