//! WCR (Weighted Close Rate) factor — pipeline + backtest binary
//!
//! Thin binary: register factor → compute_cs_pipeline → backtest.
//! All pipeline logic lives in `FactorRegistry::compute_cs_pipeline`.

use alfars::backtest::{BacktestConfig, BacktestEngine, FeeConfig};
use alfars::data::clickhouse::ClickHouseSource;
use alfars::data::layer::DataLayer;
use alfars::expr::registry::FactorRegistry;
use alfars::expr::registry::config::CsResult;

const DEFAULT_EXCLUDE_SYMBOL_PATTERN: &str = "%BJ";

fn main() {
    dotenv::dotenv().ok();

    if let Ok(threads) = std::env::var("RAYON_NUM_THREADS") {
        if let Ok(n) = threads.parse() {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global()
                .ok();
        }
    }

    let ch = ClickHouseSource::from_env();

    let start_year = std::env::var("START_YEAR")
        .unwrap_or_else(|_| "2010".to_string())
        .parse()
        .unwrap_or(2010);
    let end_year = std::env::var("END_YEAR")
        .unwrap_or_else(|_| "2025".to_string())
        .parse()
        .unwrap_or(2025);
    let output_file =
        std::env::var("OUTPUT_FILE").unwrap_or_else(|_| ".tests/wcr_output.csv".to_string());

    println!("WCR Factor + Backtest");
    println!("=====================");
    println!("Host: {}:{}/{}", ch.host(), ch.port(), ch.database());
    println!("Year range: {} - {}", start_year, end_year);

    let mut wtr = csv::Writer::from_path(&output_file).expect("Failed to create CSV writer");
    CsResult::write_header(&mut wtr);

    let mut total_records = 0usize;
    let mut all_cs: Vec<(CsResult, Vec<String>)> = Vec::new();

    let compute_start = std::time::Instant::now();

    for year in start_year..=end_year {
        let start_date = format!("{}-01-01", year);
        let end_date = format!("{}-01-01", year + 1);

        let exclude_pattern = std::env::var("EXCLUDE_SYMBOL_PATTERN")
            .unwrap_or_else(|_| DEFAULT_EXCLUDE_SYMBOL_PATTERN.to_string());
        let pre_filter = format!(
            "{}:{} symbols not like '{}'",
            start_date, end_date, exclude_pattern
        );

        println!("\n--- Year {} ---", year);
        println!("Filter: {}", pre_filter);

        let mut data_layer = DataLayer::new(ch.clone());
        data_layer.set_pre_filter(&pre_filter);

        let mut registry = FactorRegistry::new();
        registry
            .register(
                "wcr",
                "1d:sum(5m:vol * 5m:close) / 1d:sum(5m:vol) / 1d:mean(5m:close)",
            )
            .expect("Failed to register WCR");

        let cs_results = registry
            .compute_cs_pipeline(&mut data_layer)
            .expect("Failed to compute cs_ pipeline");

        let wcr = cs_results.get("wcr").expect("WCR result missing");
        println!("Computed {} WCR values", wcr.groups.len());

        let n = wcr
            .write_to(&mut wtr, data_layer.get_symbols_5m())
            .expect("Failed to write CSV");
        total_records += n;

        let year_symbols = data_layer.get_symbols_5m().to_vec();
        all_cs.push((wcr.clone(), year_symbols));
        data_layer.clear_cache();
    }

    wtr.flush().expect("Failed to flush writer");
    let compute_elapsed = compute_start.elapsed();
    println!(
        "\nFactor computation: {:.1}s, {} records",
        compute_elapsed.as_secs_f64(),
        total_records
    );

    // ── Backtest ──
    println!("\nRunning backtest...");
    let bt_start = std::time::Instant::now();

    // Query price data for the full date range
    let start_date_full = format!("{}-01-01", start_year);
    let end_date_full = format!("{}-01-01", end_year + 1);
    let pre_filter_full = format!(
        "{}:{} symbols not like '{}'",
        start_date_full,
        end_date_full,
        std::env::var("EXCLUDE_SYMBOL_PATTERN")
            .unwrap_or_else(|_| DEFAULT_EXCLUDE_SYMBOL_PATTERN.to_string())
    );

    let price_query_start = std::time::Instant::now();
    let mut data_layer = DataLayer::new(ch.clone());
    data_layer.set_pre_filter(&pre_filter_full);
    let prices = data_layer
        .query_price_matrix()
        .expect("Failed to query prices");
    let price_query_elapsed = price_query_start.elapsed();
    println!(
        "  Price query: {:.1}s, {} dates × {} symbols",
        price_query_elapsed.as_secs_f64(),
        prices.dates.len(),
        prices.symbols.len()
    );

    let factor_build_start = std::time::Instant::now();
    let cs_refs: Vec<(CsResult, &[String])> =
        all_cs.iter().map(|(cs, syms)| (cs.clone(), syms.as_slice())).collect();
    let factor_mat = prices.build_factor_matrix(&cs_refs);
    let factor_build_elapsed = factor_build_start.elapsed();

    let valid_factor = factor_mat.iter().filter(|v| v.is_finite()).count();
    let valid_pct = valid_factor as f64 / factor_mat.len() as f64 * 100.0;
    println!(
        "  Factor matrix: {:.1}s, {} / {} values valid ({:.1}%)",
        factor_build_elapsed.as_secs_f64(),
        valid_factor,
        factor_mat.len(),
        valid_pct
    );

    let config = BacktestConfig {
        quantiles: 10,
        weight_method: alfars::WeightMethod::Equal,
        long_top_n: 1,
        short_top_n: 1,
        fee_config: FeeConfig::default(),
        position_config: Default::default(),
        limit_up_down_config: Default::default(),
    };

    let engine = BacktestEngine::with_config(config);

    let run_start = std::time::Instant::now();
    match engine.run_with_prices(factor_mat, &prices) {
        Ok(result) => {
            let run_elapsed = run_start.elapsed();
            let bt_elapsed = bt_start.elapsed();
            println!(
                "  engine.run(): {:.1}s (total backtest: {:.1}s)",
                run_elapsed.as_secs_f64(),
                bt_elapsed.as_secs_f64()
            );
            println!("  IC mean: {:.6}", result.ic_mean);
            println!("  IC IR: {:.4}", result.ic_ir);
            println!("  Total return: {:.4}", result.total_return);
            println!("  Sharpe: {:.4}", result.sharpe_ratio);
            println!("  Max drawdown: {:.4}", result.max_drawdown);
            println!("  Turnover: {:.4}", result.turnover);

            // Write group NAVs
            let nav_path = std::env::var("NAV_OUTPUT")
                .unwrap_or_else(|_| ".tests/backtest_nav_rs.csv".to_string());
            let mut nav_wtr =
                csv::Writer::from_path(&nav_path).expect("Failed to create NAV writer");
            nav_wtr.write_record(&["date", "nv", "group"]).expect("header");
            result.write_nav_csv(&mut nav_wtr, &prices.dates).expect("NAV write");
            nav_wtr.flush().expect("Failed to flush NAV writer");
            println!("  NAV curves written to {}", nav_path);
        }
        Err(e) => {
            eprintln!("Backtest failed: {}", e);
        }
    }

    println!("\nFactor CSV: {}", output_file);
    println!("NAV curves: {}", std::env::var("NAV_OUTPUT")
        .unwrap_or_else(|_| ".tests/backtest_nav_rs.csv".to_string()));
    println!("Compare with: .tests/backtest_nav_py.csv (Python reference)");
    println!("Done!");
}
