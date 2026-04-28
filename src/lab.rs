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

    /// Run backtest on a computed panel.
    ///
    /// Queries prices internally via DataLayer, builds the factor matrix,
    /// and runs the backtest engine.
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
