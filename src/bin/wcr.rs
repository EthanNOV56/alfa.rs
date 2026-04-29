//! WCR (Weighted Close Rate) factor — pipeline + backtest binary.

use alfars::WeightMethod;
use alfars::backtest::{BacktestConfig, FeeConfig};
use alfars::data::clickhouse::ClickHouseSource;
use alfars::lab::AlfarsLab;

const WCR_EXPR: &str = "1d:sum(5m:vol * 5m:close) / 1d:sum(5m:vol) / 1d:mean(5m:close)";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    // Configure rayon thread pool size
    let num_threads: usize = std::env::var("RAYON_NUM_THREADS")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .unwrap_or(0);
    if num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .map_err(|e| format!("rayon init: {}", e))?;
    }
    let threads = rayon::current_num_threads();
    eprintln!("rayon threads: {}", threads);

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
        .with_years(start_year, end_year)
        .with_backtest_config(BacktestConfig {
            quantiles: 10,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            fee_config: FeeConfig::default(),
            ..Default::default()
        });

    lab.register("wcr", WCR_EXPR)?;

    let panel = lab.calc(".tests/wcr_output.csv")?;
    let result = lab.run(&panel)?;
    result.to_csv(".tests/backtest_nav_rs.csv")?;

    println!(
        "IC mean: {:.6}  IC IR: {:.4}  Sharpe: {:.4}  MaxDD: {:.4}  Turnover: {:.4}",
        result.ic_mean, result.ic_ir, result.sharpe_ratio, result.max_drawdown, result.turnover
    );
    Ok(())
}
