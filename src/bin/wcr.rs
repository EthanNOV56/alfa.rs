//! WCR (Weighted Close Rate) factor — pipeline + backtest binary.
//!
//! register → calc → backtest — no glue code.

use alfars::WeightMethod;
use alfars::backtest::{BacktestConfig, FeeConfig};
use alfars::data::clickhouse::ClickHouseSource;
use alfars::lab::AlfarsLab;

const WCR_EXPR: &str = "1d:sum(5m:vol * 5m:close) / 1d:sum(5m:vol) / 1d:mean(5m:close)";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    let start_year: i32 = std::env::var("START_YEAR")
        .unwrap_or_else(|_| "2010".into())
        .parse()
        .unwrap_or(2010);
    let end_year: i32 = std::env::var("END_YEAR")
        .unwrap_or_else(|_| "2025".into())
        .parse()
        .unwrap_or(2025);

    let mut lab = AlfarsLab::new(ClickHouseSource::from_env())
        .with_filter("symbols not like '%BJ'")
        .with_backtest_config(BacktestConfig {
            quantiles: 10,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig::default(),
            ..Default::default()
        });

    lab.register("wcr", WCR_EXPR)?;
    let panel = lab.calc(start_year, end_year)?;
    panel.to_csv(".tests/wcr_output.csv")?;

    let result = lab.backtest(&panel)?;
    println!(
        "IC mean: {:.6}  Sharpe: {:.4}  Turnover: {:.4}",
        result.ic_mean, result.sharpe_ratio, result.turnover
    );

    Ok(())
}
