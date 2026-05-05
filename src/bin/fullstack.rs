//! Local test for backtest flow with real ClickHouse data

use alfars::backtest::{BacktestConfig, BacktestEngine, FeeConfig};
use alfars::expr::registry::FactorRegistry;
use ndarray::Array1;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::collections::HashMap;

const BUILTIN_ALPHAS: &[(&str, &str, &str, &str, &str, &str)] = &[
    // Alpha001-003
    (
        "alpha001",
        "Alpha001",
        "rank(ts_argmax(power(returns, 2), 5)) - 0.5",
        "Time series rank of max power returns over 5 days",
        "momentum",
        "Alpha101 factor #001",
    ),
    (
        "alpha002",
        "Alpha002",
        "-1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)",
        "Volume change correlation with returns",
        "volume",
        "Alpha101 factor #002",
    ),
    (
        "alpha003",
        "Alpha003",
        "-1 * correlation(rank(open), rank(volume), 10)",
        "Open-volume rank correlation",
        "volume",
        "Alpha101 factor #003",
    ),
    // Alpha101-106
    (
        "alpha101",
        "Alpha101",
        "(close - open) / ((high - low) + 0.001)",
        "Price range ratio",
        "volatility",
        "Alpha101 factor #101",
    ),
    (
        "alpha102",
        "Alpha102",
        "-1 * ts_rank(rank(low), 9)",
        "Low price rank",
        "value",
        "Alpha101 factor #102",
    ),
    (
        "alpha103",
        "Alpha103",
        "-1 * correlation(rank(open), rank(volume), 10)",
        "Open-volume correlation",
        "volume",
        "Alpha101 factor #103",
    ),
    (
        "alpha104",
        "Alpha104",
        "sign(close - delay(close, 1))",
        "Price direction",
        "momentum",
        "Alpha101 factor #104",
    ),
    (
        "alpha105",
        "Alpha105",
        "-1 * ts_rank(rank(volume), 5)",
        "Volume rank",
        "volume",
        "Alpha101 factor #105",
    ),
];

/// ClickHouse config from environment variables
fn get_ch_config() -> (String, u16, String, String, String) {
    (
        std::env::var("CH_HOST").unwrap_or_else(|_| "localhost".to_string()),
        std::env::var("CH_PORT")
            .unwrap_or_else(|_| "8123".to_string())
            .parse()
            .unwrap_or(8123),
        std::env::var("CH_DATABASE").unwrap_or_else(|_| "default".to_string()),
        std::env::var("CH_USER").unwrap_or_else(|_| "readonly_user".to_string()),
        std::env::var("CH_PASSWORD").unwrap_or_default(),
    )
}

#[derive(Debug, Deserialize)]
struct ChRow {
    trading_date: String,
    symbol: String,
    close: f64,
    open: f64,
    high: f64,
    low: f64,
    volume: f64,
}

fn main() {
    let (ch_host, ch_port, ch_database, ch_user, ch_password) = get_ch_config();

    println!("Loading data from ClickHouse...");
    println!("Host: {}:{}/{}", ch_host, ch_port, ch_database);

    // Query ClickHouse
    let client = Client::builder()
        .build()
        .expect("Failed to create HTTP client");

    // Use smaller subset for testing - get top 100 symbols with most data
    let query = r#"
        SELECT
            toDate(trading_date) as trading_date,
            symbol,
            anyLast(close) as close,
            any(open) as open,
            max(high) as high,
            min(low) as low,
            sum(volume) as volume
        FROM default.stock_1d
        WHERE trading_date >= '2024-01-01'
          AND trading_date <= '2024-12-31'
          AND symbol IN (
              SELECT symbol FROM default.stock_1d
              WHERE trading_date >= '2024-01-01' AND trading_date <= '2024-12-31'
              GROUP BY symbol
              ORDER BY count() DESC
              LIMIT 100
          )
        GROUP BY trading_date, symbol
        ORDER BY trading_date, symbol
    "#;

    let url = format!("http://{}:{}/?database={}", ch_host, ch_port, ch_database);

    let response = client
        .get(&url)
        .query(&[("default_format", "JSONEachRow"), ("query", &query)])
        .basic_auth(ch_user, Some(ch_password))
        .send()
        .expect("Failed to send query");

    let body = response.text().expect("Failed to get response body");

    // Parse rows
    let mut rows: Vec<ChRow> = Vec::new();
    for line in body.lines() {
        if let Ok(row) = serde_json::from_str::<ChRow>(line) {
            rows.push(row);
        }
    }

    println!("Loaded {} rows", rows.len());

    if rows.is_empty() {
        eprintln!("No data loaded!");
        return;
    }

    // Get unique dates and symbols
    let mut dates: Vec<String> = rows.iter().map(|r| r.trading_date.clone()).collect();
    dates.sort();
    dates.dedup();

    let mut symbols: Vec<String> = rows.iter().map(|r| r.symbol.clone()).collect();
    symbols.sort();
    symbols.dedup();

    let n_dates = dates.len();
    let n_symbols = symbols.len();

    println!("Dates: {}, Symbols: {}", n_dates, n_symbols);

    // Create date and symbol index maps
    let date_to_idx: HashMap<String, usize> = dates
        .iter()
        .enumerate()
        .map(|(i, d)| (d.clone(), i))
        .collect();

    let symbol_to_idx: HashMap<String, usize> = symbols
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();

    // Create matrices
    let mut close: Vec<Vec<f64>> = vec![vec![0.0; n_symbols]; n_dates];
    let mut open: Vec<Vec<f64>> = vec![vec![0.0; n_symbols]; n_dates];
    let mut high: Vec<Vec<f64>> = vec![vec![0.0; n_symbols]; n_dates];
    let mut low: Vec<Vec<f64>> = vec![vec![0.0; n_symbols]; n_dates];
    let mut volume: Vec<Vec<f64>> = vec![vec![0.0; n_symbols]; n_dates];

    for row in &rows {
        if let (Some(d_idx), Some(s_idx)) = (
            date_to_idx.get(&row.trading_date),
            symbol_to_idx.get(&row.symbol),
        ) {
            close[*d_idx][*s_idx] = row.close;
            open[*d_idx][*s_idx] = row.open;
            high[*d_idx][*s_idx] = row.high;
            low[*d_idx][*s_idx] = row.low;
            volume[*d_idx][*s_idx] = row.volume;
        }
    }

    // Compute returns
    let mut returns: Vec<Vec<f64>> = vec![vec![0.0; n_symbols]; n_dates];
    for s in 0..n_symbols {
        for d in 1..n_dates {
            let curr_close = close[d][s];
            let prev_close = close[d - 1][s];
            if prev_close != 0.0
                && curr_close != 0.0
                && curr_close.is_finite()
                && prev_close.is_finite()
            {
                returns[d][s] = curr_close / prev_close - 1.0;
            }
        }
    }

    // Flatten data
    let mut close_flat: Vec<f64> = Vec::with_capacity(n_dates * n_symbols);
    let mut open_flat: Vec<f64> = Vec::with_capacity(n_dates * n_symbols);
    let mut high_flat: Vec<f64> = Vec::with_capacity(n_dates * n_symbols);
    let mut low_flat: Vec<f64> = Vec::with_capacity(n_dates * n_symbols);
    let mut volume_flat: Vec<f64> = Vec::with_capacity(n_dates * n_symbols);
    let mut returns_flat: Vec<f64> = Vec::with_capacity(n_dates * n_symbols);

    for d in 0..n_dates {
        for s in 0..n_symbols {
            close_flat.push(close[d][s]);
            open_flat.push(open[d][s]);
            high_flat.push(high[d][s]);
            low_flat.push(low[d][s]);
            volume_flat.push(volume[d][s]);
            returns_flat.push(returns[d][s]);
        }
    }

    // Check data stats
    let close_nan = close_flat.iter().filter(|v| v.is_nan()).count();
    let close_zero = close_flat
        .iter()
        .filter(|v| (*v - 0.0).abs() < 1e-10)
        .count();
    let returns_nan = returns_flat.iter().filter(|v| v.is_nan()).count();
    let returns_zero = returns_flat
        .iter()
        .filter(|v| (*v - 0.0).abs() < 1e-10)
        .count();
    println!("Data stats:");
    println!("  close: nan={}, zero={}", close_nan, close_zero);
    println!("  returns: nan={}, zero={}", returns_nan, returns_zero);
    println!("  close first 5: {:?}", &close_flat[..5]);
    println!("  returns first 5: {:?}", &returns_flat[..5]);

    // Create data map for FactorRegistry
    let mut data: HashMap<String, Vec<f64>> = HashMap::new();
    data.insert("close".to_string(), close_flat.clone());
    data.insert("open".to_string(), open_flat.clone());
    data.insert("high".to_string(), high_flat.clone());
    data.insert("low".to_string(), low_flat.clone());
    data.insert("volume".to_string(), volume_flat.clone());
    data.insert("returns".to_string(), returns_flat.clone());

    // Convert to Array1 format
    let mut data_array1: HashMap<String, Array1<f64>> = HashMap::new();
    for (key, values) in data.iter() {
        data_array1.insert(key.clone(), Array1::from_vec(values.clone()));
    }

    println!("\nTesting {} builtin alphas\n", BUILTIN_ALPHAS.len());

    for (id, _name, expr, _desc, _category, _full_name) in BUILTIN_ALPHAS {
        println!("=== Testing {} ===", id);
        println!("Expression: {}", expr);

        // Register factor and compute
        let mut registry = FactorRegistry::new();

        if let Err(e) = registry.register("factor", expr) {
            eprintln!("  Failed to register: {}", e);
            continue;
        }

        let required_cols = registry.required_columns();
        println!("  Required columns: {:?}", required_cols);

        // Compute factor
        let compute_start = std::time::Instant::now();
        let results = match registry.compute(&["factor"], &data_array1, true, false) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  Failed to compute: {}", e);
                continue;
            }
        };
        let compute_time = compute_start.elapsed();

        let result = match results.get("factor") {
            Some(r) => r,
            None => {
                eprintln!("  Factor not found in results");
                continue;
            }
        };

        // Check for NaN/Inf
        let valid_count = result.values.iter().filter(|v| v.is_finite()).count();
        let nan_count = result.values.iter().filter(|v| v.is_nan()).count();
        let inf_count = result.values.iter().filter(|v| v.is_infinite()).count();
        println!(
            "  Factor: valid={}, nan={}, inf={}, compute_time={}ms",
            valid_count,
            nan_count,
            inf_count,
            compute_time.as_millis()
        );

        if nan_count > 0 || inf_count > 0 {
            eprintln!("  SKIPPING backtest due to invalid values");
            continue;
        }

        // Reshape factor to (n_dates, n_symbols)
        let mut factor: Vec<Vec<f64>> = Vec::with_capacity(n_dates);
        for d in 0..n_dates {
            let mut day_factor: Vec<f64> = Vec::with_capacity(n_symbols);
            for s in 0..n_symbols {
                let idx = d * n_symbols + s;
                day_factor.push(result.values[idx]);
            }
            factor.push(day_factor);
        }

        // Run backtest
        let factor_array = ndarray::Array2::from_shape_vec(
            (n_dates, n_symbols),
            factor.into_iter().flatten().collect(),
        )
        .expect("Invalid factor shape");

        let returns_array =
            ndarray::Array2::from_shape_vec((n_dates, n_symbols), returns_flat.clone())
                .expect("Invalid returns shape");

        let close_array = ndarray::Array2::from_shape_vec((n_dates, n_symbols), close_flat.clone())
            .expect("Invalid close shape");

        let vwap_array = ndarray::Array2::from_shape_vec(
            (n_dates, n_symbols),
            high_flat
                .iter()
                .zip(low_flat.iter())
                .zip(close_flat.iter())
                .map(|((h, l), c)| (h + l + c) / 3.0)
                .collect(),
        )
        .expect("Invalid vwap shape");

        let adj_factor = ndarray::Array2::from_elem((n_dates, n_symbols), 1.0);

        let open_array = ndarray::Array2::from_shape_vec((n_dates, n_symbols), open_flat.clone())
            .expect("Invalid open shape");

        // Tradable: high > low means the stock can be traded (had a real transaction)
        let tradable_array = ndarray::Array2::from_shape_vec(
            (n_dates, n_symbols),
            high_flat
                .iter()
                .zip(low_flat.iter())
                .map(|(&h, &l)| if h > l { 1.0 } else { 0.0 })
                .collect(),
        )
        .expect("Invalid tradable shape");

        // Debug: check arrays before backtest
        println!("  Backtest input:");
        println!(
            "    factor_array: {}x{}",
            factor_array.dim().0,
            factor_array.dim().1
        );
        println!(
            "    returns_array: {}x{}",
            returns_array.dim().0,
            returns_array.dim().1
        );
        println!(
            "    close_array: {}x{}",
            close_array.dim().0,
            close_array.dim().1
        );
        println!(
            "    factor sample [0,:5]: {:?}",
            factor_array.row(0).slice(ndarray::s![..5])
        );
        println!(
            "    returns sample [0,:5]: {:?}",
            returns_array.row(0).slice(ndarray::s![..5])
        );
        println!(
            "    returns sample [1,:5]: {:?}",
            returns_array.row(1).slice(ndarray::s![..5])
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

        let backtest_result = match engine.run(
            factor_array,
            returns_array,
            adj_factor,
            close_array,
            open_array,
            vwap_array,
            tradable_array,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  Backtest failed: {}", e);
                continue;
            }
        };

        // Check results
        let first_col = backtest_result.group_cum_returns.column(0);
        let sample: Vec<f64> = first_col.iter().take(5).cloned().collect();
        let has_nan = sample.iter().any(|v| v.is_nan());
        let all_zero = sample.iter().all(|v| v.abs() < 1e-10);

        println!("  group_cum_returns first 5: {:?}", sample);
        println!(
            "  Metrics: total_return={:.4}, ic_mean={:.4}, ic_ir={:.4}",
            backtest_result.total_return, backtest_result.ic_mean, backtest_result.ic_ir
        );

        if has_nan {
            eprintln!("  WARNING: NaN in results!");
        } else if all_zero {
            eprintln!("  WARNING: All zeros!");
        } else {
            println!("  OK");
        }
        println!();
    }
}
