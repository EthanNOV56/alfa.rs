//! Alfa.rs HTTP Server
//!
//! Run with: cargo run --release --bin alfars-server

use alfars::WeightMethod;
use alfars::backtest::{BacktestConfig, BacktestEngine, BacktestResult, FeeConfig};
use alfars::data::clickhouse::ClickHouseSource;
use alfars::data::layer::PriceMatrix;
use alfars::gp::GPConfig;
use alfars::lab::AlfarsLab;
use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
};
use ndarray::Array2;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

/// Check if a TCP port is already in use
fn is_port_in_use(port: u16) -> bool {
    std::net::TcpListener::bind(("0.0.0.0", port)).is_err()
}

/// Built-in Alpha101/Alpha191 factors (readonly)
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
    (
        "alpha106",
        "Alpha106",
        "-1 * rank(((returns < 0) ? stddev(returns, 20) : close))",
        "Volatility under negative returns",
        "volatility",
        "Alpha101 factor #106",
    ),
    // Alpha191
    (
        "alpha191",
        "Alpha191",
        "rank(correlation(ts_sum(close, 7) / 7, ts_sum(close, 63) / 63, 250)) * rank(correlation(ts_rank(close, 60), ts_rank(adv30, 30), 4))",
        "Multi-timeframe correlation factor",
        "composite",
        "Alpha191 factor",
    ),
];

/// Request model for backtest endpoint
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct BacktestRequest {
    /// Factor values (optional if data_source or cache_id is provided)
    factor: Option<Vec<Vec<f64>>>,
    /// Returns values (optional if data_source or cache_id is provided)
    returns: Option<Vec<Vec<f64>>>,
    dates: Option<Vec<String>>,
    /// Cache ID from compute_factor to retrieve pre-computed factor
    cache_id: Option<String>,
    quantiles: Option<usize>,
    weight_method: Option<String>,
    long_top_n: Option<usize>,
    short_top_n: Option<usize>,
    buy_commission: Option<f64>,
    sell_commission: Option<f64>,
    /// Data source config for on-demand loading from database
    data_source: Option<DataSourceConfig>,
    /// Factor ID to compute (when using data_source)
    factor_id: Option<String>,
}

/// NAV data for chart visualization
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct NavData {
    dates: Vec<String>,
    quantiles: Vec<Vec<f64>>,
    long_short: Vec<f64>,
    benchmark: Vec<f64>,
    ic_series: Vec<f64>,
    metrics: Metrics,
}

/// Performance metrics
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Metrics {
    long_short_cum_return: f64,
    total_return: f64,
    annualized_return: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    turnover: f64,
    ic_mean: f64,
    ic_ir: f64,
}

/// Health check response
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
}

/// Factor info
#[derive(Debug, Serialize)]
struct FactorInfo {
    id: String,
    name: String,
    expression: String,
    description: String,
}

/// Factor list response
#[derive(Debug, Serialize)]
struct FactorListResponse {
    factors: Vec<FactorInfo>,
}

/// Factor compute request
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct FactorComputeRequest {
    #[serde(rename = "factorId")]
    factor_id: String,
    #[serde(rename = "nDays")]
    n_days: Option<usize>,
    #[serde(rename = "nAssets")]
    n_assets: Option<usize>,
    /// Data source config for on-demand loading from database
    #[serde(rename = "dataSource")]
    data_source: Option<DataSourceConfig>,
}

/// Factor compute response
#[derive(Debug, Serialize)]
struct FactorComputeResponse {
    #[serde(rename = "factor_id")]
    factor_id: String,
    factor: Vec<Vec<f64>>,
    returns: Vec<Vec<f64>>,
    dates: Vec<String>,
    #[serde(rename = "cache_id", skip_serializing_if = "Option::is_none")]
    cache_id: Option<String>,
}

/// GP run request
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
struct GpRunRequest {
    population_size: Option<usize>,
    max_generations: Option<usize>,
    tournament_size: Option<usize>,
    crossover_prob: Option<f64>,
    mutation_prob: Option<f64>,
    max_depth: Option<usize>,
    use_diverse_init: Option<bool>,
    smart_mutation_ratio: Option<f64>,
    num_factors: Option<usize>,
    max_symbols: Option<usize>,
    /// GP terminal fields (e.g., ["close", "vol"])
    fields: Option<Vec<String>>,
    /// GP operators (e.g., ["ts_mean", "rank"])
    ops: Option<Vec<String>>,
    /// Seed expression for initial GP population
    seed_expression: Option<String>,
    /// Data source config for on-demand loading from database
    data_source: Option<DataSourceConfig>,
}

/// GP factor
#[derive(Debug, Clone, Serialize)]
struct GpFactor {
    id: String,
    name: String,
    expression: String,
    ic_mean: f64,
    ic_ir: f64,
    fitness: f64,
    sharpe_ratio: f64,
    total_return: f64,
    annualized_return: f64,
    max_drawdown: f64,
    turnover: f64,
    calmar_ratio: f64,
    win_rate: f64,
}

/// GP run response
#[derive(Debug, Serialize)]
struct GpRunResponse {
    factors: Vec<GpFactor>,
    best_factor: GpFactor,
    generations: usize,
    elapsed_time: f64,
}

/// Alpha factor (from .al file or built-in)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Alpha {
    name: String,
    expression: String,
    description: String,
    dimension: String,
    tags: Vec<String>,
    readonly: bool,
    #[serde(default)]
    builtin: bool,
}

/// Alpha list response
#[derive(Debug, Serialize)]
struct AlphaListResponse {
    alphas: Vec<Alpha>,
}

/// Save alpha request
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveAlphaRequest {
    name: String,
    expression: String,
    description: Option<String>,
    dimension: Option<String>,
    tags: Option<Vec<String>>,
}

/// Save alpha response
#[derive(Debug, Serialize)]
struct SaveAlphaResponse {
    success: bool,
    path: String,
    message: String,
}

/// Root response
#[derive(Debug, Serialize)]
struct RootResponse {
    name: String,
    version: String,
    docs: String,
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DbConfig {
    host: String,
    port: u16,
    database: String,
    username: String,
    password: String,
    /// Whether the connection is active
    connected: bool,
}

impl Default for DbConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 8123,
            database: "default".to_string(),
            username: "default".to_string(),
            password: String::new(),
            connected: false,
        }
    }
}

/// Get config directory path (~/.alfars)
fn get_config_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".alfars")
}

/// Load saved database config from file
fn load_saved_db_config() -> DbConfig {
    let config_path = get_config_dir().join("db_config.json");
    if config_path.exists() {
        match fs::read_to_string(&config_path) {
            Ok(contents) => {
                match serde_json::from_str::<DbConfig>(&contents) {
                    Ok(mut config) => {
                        // Mark as not connected on startup - user needs to reconnect
                        config.connected = false;
                        eprintln!("Loaded saved database config from {:?}", config_path);
                        return config;
                    }
                    Err(e) => {
                        eprintln!("Failed to parse saved config: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to read saved config: {}", e);
            }
        }
    }
    DbConfig::default()
}

/// Save database config to file (without password for security, except connection status)
fn save_db_config(config: &DbConfig) {
    let config_dir = get_config_dir();
    if !config_dir.exists()
        && let Err(e) = fs::create_dir_all(&config_dir)
    {
        eprintln!("Failed to create config directory: {}", e);
        return;
    }

    let config_path = config_dir.join("db_config.json");
    // Save without password for security
    let config_to_save = serde_json::json!({
        "host": config.host,
        "port": config.port,
        "database": config.database,
        "username": config.username,
        "password": "", // Don't save password
        "connected": config.connected,
    });

    match serde_json::to_string_pretty(&config_to_save) {
        Ok(contents) => {
            if let Err(e) = fs::write(&config_path, contents) {
                eprintln!("Failed to save config: {}", e);
            } else {
                eprintln!("Saved database config to {:?}", config_path);
            }
        }
        Err(e) => {
            eprintln!("Failed to serialize config: {}", e);
        }
    }
}

/// Load saved column mappings from file
fn load_saved_column_mappings() -> TableColumnMappings {
    let config_path = get_config_dir().join("column_mappings.json");
    if config_path.exists() {
        match fs::read_to_string(&config_path) {
            Ok(contents) => match serde_json::from_str::<TableColumnMappings>(&contents) {
                Ok(mappings) => {
                    eprintln!("Loaded saved column mappings from {:?}", config_path);
                    return mappings;
                }
                Err(e) => {
                    eprintln!("Failed to parse saved column mappings: {}", e);
                }
            },
            Err(e) => {
                eprintln!("Failed to read saved column mappings: {}", e);
            }
        }
    }
    TableColumnMappings::default()
}

/// Save column mappings to file
fn save_column_mappings(mappings: &TableColumnMappings) {
    let config_dir = get_config_dir();
    if !config_dir.exists()
        && let Err(e) = fs::create_dir_all(&config_dir)
    {
        eprintln!("Failed to create config directory: {}", e);
        return;
    }

    let config_path = config_dir.join("column_mappings.json");

    match serde_json::to_string_pretty(mappings) {
        Ok(contents) => {
            if let Err(e) = fs::write(&config_path, contents) {
                eprintln!("Failed to save column mappings: {}", e);
            } else {
                eprintln!("Saved column mappings to {:?}", config_path);
            }
        }
        Err(e) => {
            eprintln!("Failed to serialize column mappings: {}", e);
        }
    }
}

/// Request to set database configuration
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SetDbConfigRequest {
    host: String,
    port: Option<u16>,
    database: Option<String>,
    username: Option<String>,
    password: Option<String>,
}

/// Symbol info
#[derive(Debug, Serialize)]
struct SymbolInfo {
    symbol: String,
    name: String,
}

/// Date range info
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DateRange {
    min_date: String,
    max_date: String,
}

/// Column info
#[derive(Debug, Serialize)]
struct ColumnInfo {
    name: String,
    column_type: String,
}

/// Column mapping configuration for a single table
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ColumnMapping {
    /// Close price column name in user's database
    pub close: Option<String>,
    /// Open price column name in user's database
    pub open: Option<String>,
    /// High price column name in user's database
    pub high: Option<String>,
    /// Low price column name in user's database
    pub low: Option<String>,
    /// Volume column name in user's database
    pub volume: Option<String>,
    /// Symbol column name (optional)
    pub symbol: Option<String>,
    /// Trading date column name (optional)
    pub trading_date: Option<String>,
    /// PE column for filtering (optional)
    pub pe: Option<String>,
    /// ROE column for filtering (optional)
    pub roe: Option<String>,
    /// Market cap column for filtering (optional)
    pub market_cap: Option<String>,
    /// Adjustment factor column for 复权 (optional)
    pub adj_factor: Option<String>,
}

/// Per-table column mappings storage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct TableColumnMappings {
    /// Map of table name to its column mapping
    mappings: std::collections::HashMap<String, ColumnMapping>,
}

impl TableColumnMappings {
    /// Get mapping for a specific table, returns empty mapping if not found
    pub fn get(&self, table: &str) -> ColumnMapping {
        self.mappings.get(table).cloned().unwrap_or_default()
    }

    /// Set mapping for a specific table
    pub fn set(&mut self, table: &str, mapping: ColumnMapping) {
        self.mappings.insert(table.to_string(), mapping);
    }
}

/// Table name mapping for different data frequencies
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct TableMapping {
    /// Daily table name (required)
    stock_1day: String,
    /// 5-minute table name (optional)
    stock_5min: Option<String>,
    /// 1-minute table name (optional)
    stock_1min: Option<String>,
}

/// Filter condition for stock pool filtering
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct FilterCondition {
    column: String,
    operator: String,
    value: String,
}

/// Data source configuration for on-demand loading
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DataSourceConfig {
    table: String,
    start_date: String,
    end_date: String,
    /// Filter conditions for stock pool (AND logic)
    filters: Option<Vec<FilterCondition>>,
    /// Column mapping (optional - uses saved mapping if not provided)
    column_mapping: Option<ColumnMapping>,
}

/// Data load request
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LoadDataRequest {
    symbols: Vec<String>,
    start_date: String,
    end_date: String,
    /// Table name (required - use /api/data/tables to get available tables)
    table: Option<String>,
    /// Filter conditions for stock pool (AND logic)
    filters: Option<Vec<FilterCondition>>,
}

/// Get available tables request
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GetTablesRequest {
    /// Database name (optional, uses configured database if not provided)
    database: Option<String>,
}

/// Data load response with price matrix
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct LoadDataResponse {
    dates: Vec<String>,
    symbols: Vec<String>,
    close: Vec<Vec<f64>>,
    open: Vec<Vec<f64>>,
    high: Vec<Vec<f64>>,
    low: Vec<Vec<f64>>,
    volume: Vec<Vec<f64>>,
    returns: Vec<Vec<f64>>,
    adj_factor: Vec<Vec<f64>>,
}

/// Cached factor computation result
#[derive(Clone)]
struct CachedFactor {
    factor: Vec<Vec<f64>>,
    returns: Vec<Vec<f64>>,
    dates: Vec<String>,
    close: Vec<Vec<f64>>,
    open: Vec<Vec<f64>>,
    high: Vec<Vec<f64>>,
    low: Vec<Vec<f64>>,
    vwap: Vec<Vec<f64>>,
    adj_factor: Vec<Vec<f64>>,
}

#[derive(Clone)]
struct AppState {
    db_config: Arc<RwLock<DbConfig>>,
    column_mappings: Arc<RwLock<TableColumnMappings>>,
    table_mapping: Arc<RwLock<TableMapping>>,
    factor_cache: Arc<RwLock<HashMap<String, CachedFactor>>>,
}

fn build_backtest_config_from_request(req: &BacktestRequest) -> BacktestConfig {
    let weight_method = match req.weight_method.as_deref() {
        Some("weighted") => WeightMethod::Weighted,
        _ => WeightMethod::Equal,
    };
    BacktestConfig {
        quantiles: req.quantiles.unwrap_or(10),
        weight_method,
        long_top_n: req.long_top_n.unwrap_or(1),
        short_top_n: req.short_top_n.unwrap_or(1),
        rebalance_freq: 1,
        fee_config: FeeConfig {
            buy_commission: req.buy_commission.unwrap_or(0.0005),
            sell_commission: req.sell_commission.unwrap_or(0.0015),
            ..Default::default()
        },
        position_config: Default::default(),
        limit_up_down_config: Default::default(),
    }
}

/// Run backtest from pre-computed factor/price data (non-lab paths).
fn run_backtest_direct(
    factor: Vec<Vec<f64>>,
    returns: Vec<Vec<f64>>,
    close: Vec<Vec<f64>>,
    open: Vec<Vec<f64>>,
    high: Vec<Vec<f64>>,
    low: Vec<Vec<f64>>,
    vwap: Vec<Vec<f64>>,
    adj_factor: Vec<Vec<f64>>,
    config: &BacktestConfig,
) -> Result<BacktestResult, String> {
    let n_days = factor.len();
    let n_assets = factor[0].len();

    let to_array2 = |data: Vec<Vec<f64>>| -> Result<Array2<f64>, String> {
        Array2::from_shape_vec((n_days, n_assets), data.into_iter().flatten().collect())
            .map_err(|e| format!("Invalid shape: {}", e))
    };

    let tradable_flat: Vec<f64> = high
        .iter()
        .zip(low.iter())
        .flat_map(|(h, l)| {
            h.iter()
                .zip(l.iter())
                .map(|(&a, &b)| if a > b { 1.0 } else { 0.0 })
        })
        .collect();
    let tradable_array = Array2::from_shape_vec((n_days, n_assets), tradable_flat)
        .map_err(|e| format!("Invalid tradable: {}", e))?;

    let pm = PriceMatrix {
        dates: vec![],
        symbols: vec![],
        close: to_array2(close)?,
        open: to_array2(open)?,
        high: to_array2(high)?,
        low: to_array2(low)?,
        vwap: to_array2(vwap)?,
        returns: to_array2(returns)?,
        tradable: tradable_array,
        adj_factor: to_array2(adj_factor)?,
    };

    BacktestEngine::with_config(config.clone()).run(to_array2(factor)?, &pm)
}

/// Run backtest and return NAV data
async fn run_backtest(
    State(state): State<AppState>,
    Json(req): Json<BacktestRequest>,
) -> Result<Json<NavData>, (StatusCode, String)> {
    let config = build_backtest_config_from_request(&req);

    // Path 1: Pre-computed factor + returns + dates (no DB needed)
    if let (Some(factor), Some(returns), Some(dates)) = (&req.factor, &req.returns, &req.dates) {
        let n_days = factor.len();
        let n_assets = factor.first().map(|r| r.len()).unwrap_or(0);
        let ones = vec![vec![1.0; n_assets]; n_days];
        let result = run_backtest_direct(
            factor.clone(),
            returns.clone(),
            ones.clone(),
            ones.clone(),
            ones.clone(),
            ones.clone(),
            ones.clone(),
            ones,
            &config,
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        return Ok(Json(convert_to_nav_data(result, Some(dates.clone()))));
    }

    // Path 2: Cached factor from compute_factor endpoint
    if let Some(cache_id) = &req.cache_id {
        let cache = state.factor_cache.read().await;
        let cached = cache
            .get(cache_id)
            .ok_or_else(|| {
                (
                    StatusCode::NOT_FOUND,
                    format!("Cache not found: {}", cache_id),
                )
            })?
            .clone();
        drop(cache);

        let dates = Some(cached.dates.clone());
        let result = run_backtest_direct(
            cached.factor,
            cached.returns,
            cached.close,
            cached.open,
            cached.high,
            cached.low,
            cached.vwap,
            cached.adj_factor,
            &config,
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

        // Clear cache after successful backtest
        state.factor_cache.write().await.remove(cache_id);

        return Ok(Json(convert_to_nav_data(result, dates)));
    }

    // Path 3: Data source — delegate to AlfarsLab
    let data_source = req.data_source.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "Either provide factor/returns/dates, cache_id, or configure data_source".to_string(),
        )
    })?;

    let db_config = state.db_config.read().await;
    if !db_config.connected {
        return Err((
            StatusCode::BAD_REQUEST,
            "Database not connected.".to_string(),
        ));
    }

    let table_mapping = state.table_mapping.read().await;
    let source = build_source_from_state(&db_config, &table_mapping);

    let mut lab = build_lab_from_state(source, data_source, config)?;

    let factor_expr = req.factor_id.as_deref().unwrap_or("cs_rank(close)");
    lab.register("factor", factor_expr).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Failed to register factor: {}", e),
        )
    })?;

    let bt_result = tokio::task::spawn_blocking(move || {
        lab.calc(None)?;
        lab.run_bt()
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Task panicked: {}", e),
        )
    })?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(convert_to_nav_data(bt_result, None)))
}

/// i64 YYYYMMDD → "YYYY-MM-DD"
fn fmt_date_ymd(d: i64) -> String {
    format!("{:04}-{:02}-{:02}", d / 10000, (d % 10000) / 100, d % 100)
}

fn array2_to_vec2d(arr: &Array2<f64>) -> Vec<Vec<f64>> {
    arr.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn convert_to_nav_data(result: BacktestResult, dates_override: Option<Vec<String>>) -> NavData {
    let dates: Vec<String> =
        dates_override.unwrap_or_else(|| result.dates.iter().map(|&d| fmt_date_ymd(d)).collect());

    let group_cum_returns = result.group_cum_returns;
    let n_quantile_days = group_cum_returns.nrows();
    let n_quantiles = group_cum_returns.ncols();

    let quantiles_nav: Vec<Vec<f64>> = (0..n_quantiles)
        .map(|i| {
            group_cum_returns
                .column(i)
                .iter()
                .map(|&v| 1.0 + v)
                .collect()
        })
        .collect();

    let nav_dates = if dates.len() > n_quantile_days {
        dates[1..n_quantile_days + 1].to_vec()
    } else {
        dates
    };

    NavData {
        dates: nav_dates,
        quantiles: quantiles_nav,
        long_short: result.long_short_cum_returns.to_vec(),
        benchmark: result.passive_cum_returns.to_vec(),
        ic_series: result.ic_series.to_vec(),
        metrics: Metrics {
            long_short_cum_return: result.long_short_cum_return,
            total_return: result.total_return,
            annualized_return: result.annualized_return,
            sharpe_ratio: result.sharpe_ratio,
            max_drawdown: result.max_drawdown,
            turnover: result.turnover,
            ic_mean: result.ic_mean,
            ic_ir: result.ic_ir,
        },
    }
}

async fn root() -> Json<RootResponse> {
    Json(RootResponse {
        name: "Alfa.rs Backtest API".to_string(),
        version: "0.1.0".to_string(),
        docs: "/docs".to_string(),
    })
}

async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
    })
}

/// List all predefined factors
async fn list_factors() -> Json<FactorListResponse> {
    let factors: Vec<FactorInfo> = BUILTIN_ALPHAS
        .iter()
        .map(|(id, name, expr, desc, ..)| FactorInfo {
            id: id.to_string(),
            name: name.to_string(),
            expression: expr.to_string(),
            description: desc.to_string(),
        })
        .collect();

    Json(FactorListResponse { factors })
}

/// Compute factor values
async fn compute_factor(
    State(state): State<AppState>,
    Json(req): Json<FactorComputeRequest>,
) -> Result<Json<FactorComputeResponse>, (StatusCode, String)> {
    let data_source = req.data_source.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "Data source not configured. Please configure data source in Data Source page first."
                .to_string(),
        )
    })?;

    let db_config = state.db_config.read().await;
    if !db_config.connected {
        return Err((
            StatusCode::BAD_REQUEST,
            "Database not connected.".to_string(),
        ));
    }

    let table_mapping = state.table_mapping.read().await;
    let source = build_source_from_state(&db_config, &table_mapping);

    let mut lab = build_lab_from_state(source, data_source, BacktestConfig::default())?;
    lab.register("factor", &req.factor_id).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Failed to register factor '{}': {}", req.factor_id, e),
        )
    })?;

    eprintln!(
        "[compute_factor] Computing factor '{}' via lab.evaluate()",
        req.factor_id
    );

    let (matrices, price_matrix) = tokio::task::spawn_blocking(move || lab.evaluate())
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Task panicked: {}", e),
            )
        })?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let factor_matrix = matrices.get("factor").ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Factor not found in evaluation results".to_string(),
        )
    })?;

    let cache_id = uuid::Uuid::new_v4().to_string();
    let cached = CachedFactor {
        factor: array2_to_vec2d(factor_matrix),
        returns: array2_to_vec2d(&price_matrix.returns),
        dates: price_matrix
            .dates
            .iter()
            .map(|&d| fmt_date_ymd(d))
            .collect(),
        close: array2_to_vec2d(&price_matrix.close),
        open: array2_to_vec2d(&price_matrix.open),
        high: array2_to_vec2d(&price_matrix.high),
        low: array2_to_vec2d(&price_matrix.low),
        vwap: array2_to_vec2d(&price_matrix.vwap),
        adj_factor: array2_to_vec2d(&price_matrix.adj_factor),
    };
    state
        .factor_cache
        .write()
        .await
        .insert(cache_id.clone(), cached);

    eprintln!(
        "[compute_factor] Cached factor '{}' with id {}",
        req.factor_id, cache_id
    );

    Ok(Json(FactorComputeResponse {
        factor_id: req.factor_id,
        factor: vec![],
        returns: vec![],
        dates: vec![],
        cache_id: Some(cache_id),
    }))
}

/// List all alpha factors: built-in Alpha101/Alpha191 + .al files from ~/.alfars/ and ~/.alfars/user/
async fn list_alphas() -> Result<Json<AlphaListResponse>, (StatusCode, String)> {
    // First, add built-in alphas
    let mut alphas: Vec<Alpha> = BUILTIN_ALPHAS
        .iter()
        .map(|(_id, name, expr, desc, tags, category)| Alpha {
            name: name.to_string(),
            expression: expr.to_string(),
            description: desc.to_string(),
            dimension: "dimensionless".to_string(),
            tags: vec![tags.to_string(), category.to_string()],
            readonly: true,
            builtin: true,
        })
        .collect();

    // Then, add user-defined alphas from .al files
    let user_factors = alfars::al::parser::AlParser::load_all()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let user_alphas: Vec<Alpha> = user_factors
        .into_iter()
        .map(|f| Alpha {
            name: f.name,
            expression: f.expression,
            description: f.description,
            dimension: f.dimension,
            tags: f.tags,
            readonly: f.readonly,
            builtin: false,
        })
        .collect();

    alphas.extend(user_alphas);

    Ok(Json(AlphaListResponse { alphas }))
}

/// Save a new alpha factor to ~/.alfars/user/
async fn save_alpha(
    State(_state): State<AppState>,
    Json(req): Json<SaveAlphaRequest>,
) -> Result<Json<SaveAlphaResponse>, (StatusCode, String)> {
    let factor = alfars::al::parser::AlFactor {
        name: req.name.clone(),
        expression: req.expression,
        description: req.description.unwrap_or_default(),
        dimension: req.dimension.unwrap_or_else(|| "dimensionless".to_string()),
        tags: req.tags.unwrap_or_default(),
        readonly: false,
    };

    let path = alfars::al::parser::AlParser::save_to_user_dir(&factor, None)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(SaveAlphaResponse {
        success: true,
        path: path.display().to_string(),
        message: format!("Alpha '{}' saved successfully", req.name),
    }))
}

/// Get database configuration
async fn get_db_config(State(state): State<AppState>) -> Json<DbConfig> {
    let config = state.db_config.read().await;
    Json(config.clone())
}

/// Set database configuration and test connection
async fn set_db_config(
    State(state): State<AppState>,
    Json(req): Json<SetDbConfigRequest>,
) -> Result<Json<DbConfig>, (StatusCode, String)> {
    let mut config = state.db_config.write().await;

    config.host = req.host;
    config.port = req.port.unwrap_or(8123);
    config.database = req.database.unwrap_or_else(|| "default".to_string());
    config.username = req.username.unwrap_or_else(|| "default".to_string());
    config.password = req.password.unwrap_or_default();

    // Test connection using reqwest - execute a real query to verify authentication
    let client = HttpClient::new();
    let url = format!(
        "http://{}:{}/?query=SELECT+1+AS+test&database={}",
        config.host, config.port, config.database
    );

    match client
        .get(&url)
        .basic_auth(&config.username, Some(&config.password))
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                // Verify we can actually read data from the response
                match response.text().await {
                    Ok(text) => {
                        if text.contains("test") || text.contains("1") {
                            config.connected = true;
                            // Save config to file for persistence across restarts
                            save_db_config(&config);
                            Ok(Json(config.clone()))
                        } else {
                            config.connected = false;
                            Err((
                                StatusCode::BAD_REQUEST,
                                "Connected but failed to execute query. Please check permissions."
                                    .to_string(),
                            ))
                        }
                    }
                    Err(e) => {
                        config.connected = false;
                        Err((
                            StatusCode::BAD_REQUEST,
                            format!("Failed to read response: {}", e),
                        ))
                    }
                }
            } else {
                config.connected = false;
                Err((
                    StatusCode::BAD_REQUEST,
                    format!("ClickHouse returned error: {}", response.status()),
                ))
            }
        }
        Err(e) => {
            config.connected = false;
            Err((
                StatusCode::BAD_REQUEST,
                format!("Failed to connect to ClickHouse: {}", e),
            ))
        }
    }
}

/// Get available tables in the database
async fn get_tables(
    State(state): State<AppState>,
    Json(req): Json<GetTablesRequest>,
) -> Result<Json<Vec<String>>, (StatusCode, String)> {
    let config = state.db_config.read().await;

    if !config.connected {
        return Err((
            StatusCode::BAD_REQUEST,
            "Database not connected. Please configure database connection first.".to_string(),
        ));
    }

    let client = HttpClient::new();
    let database = req.database.unwrap_or_else(|| config.database.clone());

    // Query for all tables in the database
    let query = format!(
        "SELECT table FROM system.tables WHERE database = '{}' ORDER BY table",
        database
    );

    let url = format!("http://{}:{}/", config.host, config.port);

    let response = client
        .get(&url)
        .query(&[("default_format", "JSONEachRow"), ("query", &query)])
        .basic_auth(&config.username, Some(&config.password))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Request failed: {}", e),
            )
        })?;

    let body = response.text().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read response: {}", e),
        )
    })?;

    let mut tables: Vec<String> = Vec::new();
    for line in body.lines() {
        if let Ok(row) = serde_json::from_str::<serde_json::Value>(line)
            && let Some(table) = row.get("table").and_then(|v| v.as_str())
        {
            tables.push(table.to_string());
        }
    }

    Ok(Json(tables))
}

/// Check if default tables exist and return available tables for selection
#[derive(Serialize)]
struct TableValidationResult {
    /// Whether stock_1day table exists
    stock_1day_exists: bool,
    /// Whether stock_5min table exists
    stock_5min_exists: bool,
    /// Whether stock_1min table exists
    stock_1min_exists: bool,
    /// Available tables in the database
    available_tables: Vec<String>,
    /// Current table mapping
    current_mapping: TableMapping,
}

/// Validate table existence and get available tables
async fn validate_tables(
    State(state): State<AppState>,
) -> Result<Json<TableValidationResult>, (StatusCode, String)> {
    let config = state.db_config.read().await;

    if !config.connected {
        return Err((
            StatusCode::BAD_REQUEST,
            "Database not connected. Please configure database connection first.".to_string(),
        ));
    }

    let client = HttpClient::new();
    let database = config.database.clone();

    // Get available tables
    let query = format!(
        "SELECT table FROM system.tables WHERE database = '{}' ORDER BY table",
        database
    );
    let url = format!("http://{}:{}/", config.host, config.port);

    let response = client
        .get(&url)
        .query(&[("default_format", "JSONEachRow"), ("query", &query)])
        .basic_auth(&config.username, Some(&config.password))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Request failed: {}", e),
            )
        })?;

    let body = response.text().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read response: {}", e),
        )
    })?;

    let mut available_tables: Vec<String> = Vec::new();
    for line in body.lines() {
        if let Ok(row) = serde_json::from_str::<serde_json::Value>(line)
            && let Some(table) = row.get("table").and_then(|v| v.as_str())
        {
            available_tables.push(table.to_string());
        }
    }

    // Check existence of default tables
    let table_mapping = state.table_mapping.read().await;
    let stock_1day = table_mapping.stock_1day.clone();
    let stock_5min = table_mapping.stock_5min.clone();
    let stock_1min = table_mapping.stock_1min.clone();
    drop(table_mapping);

    let stock_1day_exists = available_tables.contains(&stock_1day);
    let stock_5min_exists = stock_5min
        .as_ref()
        .map(|t| available_tables.contains(t))
        .unwrap_or(false);
    let stock_1min_exists = stock_1min
        .as_ref()
        .map(|t| available_tables.contains(t))
        .unwrap_or(false);

    Ok(Json(TableValidationResult {
        stock_1day_exists,
        stock_5min_exists,
        stock_1min_exists,
        available_tables,
        current_mapping: TableMapping {
            stock_1day,
            stock_5min,
            stock_1min,
        },
    }))
}

/// Column validation result
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ColumnValidationResult {
    valid: bool,
    available_columns: Vec<String>,
    current_mapping: ColumnMapping,
    missing_columns: Vec<String>,
}

/// Validate column existence and get available columns for table mapping
async fn validate_columns(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<ColumnValidationResult>, (StatusCode, String)> {
    let config = state.db_config.read().await;

    if !config.connected {
        return Err((
            StatusCode::BAD_REQUEST,
            "Database not connected. Please configure database connection first.".to_string(),
        ));
    }

    let client = HttpClient::new();

    // Get table from query param or use default from table mapping
    let table = if let Some(t) = params.get("table") {
        t.clone()
    } else {
        let table_mapping = state.table_mapping.read().await;
        table_mapping.stock_1day.clone()
    };

    // Get available columns from the table
    let query = format!(
        "SELECT name FROM system.columns WHERE database = '{}' AND table = '{}' ORDER BY name",
        config.database, table
    );
    let url = format!("http://{}:{}/", config.host, config.port);

    let response = client
        .get(&url)
        .query(&[("default_format", "JSONEachRow"), ("query", &query)])
        .basic_auth(&config.username, Some(&config.password))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Request failed: {}", e),
            )
        })?;

    let body = response.text().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read response: {}", e),
        )
    })?;

    let mut available_columns: Vec<String> = Vec::new();
    for line in body.lines() {
        if let Ok(row) = serde_json::from_str::<serde_json::Value>(line)
            && let Some(name) = row.get("name").and_then(|v| v.as_str())
        {
            available_columns.push(name.to_string());
        }
    }

    // Get current column mapping for this table
    let column_mappings = state.column_mappings.read().await;
    let current_mapping = column_mappings.get(&table);

    // Check which columns from mapping exist
    let mapped_columns = [
        current_mapping.close.as_ref(),
        current_mapping.open.as_ref(),
        current_mapping.high.as_ref(),
        current_mapping.low.as_ref(),
        current_mapping.volume.as_ref(),
        current_mapping.symbol.as_ref(),
        current_mapping.trading_date.as_ref(),
    ];

    let mut missing_columns: Vec<String> = Vec::new();
    for col in mapped_columns.iter().flatten() {
        if !available_columns.contains(col) {
            missing_columns.push((*col).clone());
        }
    }

    let valid = missing_columns.is_empty();

    Ok(Json(ColumnValidationResult {
        valid,
        available_columns,
        current_mapping,
        missing_columns,
    }))
}

/// Get available symbols from database
async fn get_symbols(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<Vec<SymbolInfo>>, (StatusCode, String)> {
    let config = state.db_config.read().await;

    if !config.connected {
        return Err((
            StatusCode::BAD_REQUEST,
            "Database not connected. Please configure database connection first.".to_string(),
        ));
    }

    let client = HttpClient::new();
    let table = params.get("table").cloned().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "Table parameter is required. Use /api/data/tables to get available tables."
                .to_string(),
        )
    })?;

    let query = format!(
        "SELECT DISTINCT symbol, any(name) as name FROM {} GROUP BY symbol ORDER BY symbol LIMIT 100",
        table
    );
    let url = format!(
        "http://{}:{}/?database={}",
        config.host, config.port, config.database
    );

    let response = client
        .get(&url)
        .query(&[("default_format", "JSONEachRow"), ("query", &query)])
        .basic_auth(&config.username, Some(&config.password))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Request failed: {}", e),
            )
        })?;

    let body = response.text().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read response: {}", e),
        )
    })?;

    let mut symbols: Vec<SymbolInfo> = Vec::new();
    for line in body.lines() {
        if let Ok(row) = serde_json::from_str::<serde_json::Value>(line) {
            let symbol = row
                .get("symbol")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name = row
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            symbols.push(SymbolInfo { symbol, name });
        }
    }

    Ok(Json(symbols))
}

/// Get date range for available data
async fn get_date_range(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<DateRange>, (StatusCode, String)> {
    let config = state.db_config.read().await;

    if !config.connected {
        return Err((
            StatusCode::BAD_REQUEST,
            "Database not connected. Please configure database connection first.".to_string(),
        ));
    }

    let client = HttpClient::new();
    let table = params.get("table").cloned().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "Table parameter is required. Use /api/data/tables to get available tables."
                .to_string(),
        )
    })?;

    // Use trading_date by default
    let date_column = "trading_date";

    let query = format!(
        "SELECT min({}) as min_date, max({}) as max_date FROM {}",
        date_column, date_column, table
    );
    let url = format!(
        "http://{}:{}/?database={}",
        config.host, config.port, config.database
    );

    let response = client
        .get(&url)
        .query(&[("default_format", "JSONEachRow"), ("query", &query)])
        .basic_auth(&config.username, Some(&config.password))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Request failed: {}", e),
            )
        })?;

    let body = response.text().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read response: {}", e),
        )
    })?;

    let mut min_date = String::new();
    let mut max_date = String::new();

    for line in body.lines() {
        if let Ok(row) = serde_json::from_str::<serde_json::Value>(line) {
            min_date = row
                .get("min_date")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            max_date = row
                .get("max_date")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
        }
    }

    Ok(Json(DateRange { min_date, max_date }))
}

/// Load data from database using DataSourceConfig
async fn load_data(
    State(state): State<AppState>,
    Json(req): Json<LoadDataRequest>,
) -> Result<Json<LoadDataResponse>, (StatusCode, String)> {
    let config = state.db_config.read().await;
    let mappings = state.column_mappings.read().await;

    if !config.connected {
        return Err((
            StatusCode::BAD_REQUEST,
            "Database not connected. Please configure database connection first.".to_string(),
        ));
    }

    let client = HttpClient::new();
    let table = req.table.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "Table parameter is required. Use /api/data/tables to get available tables."
                .to_string(),
        )
    })?;

    // Get column mapping for this table
    let mapping = mappings.get(&table);

    // Build query with symbol list
    let symbols_list = req
        .symbols
        .iter()
        .map(|s| format!("'{}'", s.replace("'", "''")))
        .collect::<Vec<_>>()
        .join(",");

    // Use column mapping for column names
    let date_column = mapping.trading_date.as_deref().unwrap_or("trading_date");
    let symbol_column = mapping.symbol.as_deref().unwrap_or("symbol");

    // Build WHERE clauses - start with date range
    let mut where_clauses = vec![
        format!("{} >= '{}'", date_column, req.start_date),
        format!("{} <= '{}'", date_column, req.end_date),
    ];

    // Add symbol filter if symbols are provided
    if !symbols_list.is_empty() {
        where_clauses.push(format!("{} IN ({})", symbol_column, symbols_list));
    }

    // Add filter conditions
    if let Some(ref filters) = req.filters {
        for filter in filters {
            let clause = match filter.operator.to_uppercase().as_str() {
                "LIKE" => format!("{} LIKE '{}'", filter.column, filter.value),
                "NOT LIKE" => format!("{} NOT LIKE '{}'", filter.column, filter.value),
                "=" => format!("{} = '{}'", filter.column, filter.value),
                "!=" => format!("{} != '{}'", filter.column, filter.value),
                ">" | ">=" | "<" | "<=" => {
                    format!("{} {} {}", filter.column, filter.operator, filter.value)
                }
                _ => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!("Unsupported operator: {}", filter.operator),
                    ));
                }
            };
            where_clauses.push(clause);
        }
    }

    let adj_factor_col = mapping
        .adj_factor
        .as_deref()
        .unwrap_or("_adj_factor_default");
    let adj_factor_select = if mapping.adj_factor.is_some() {
        format!("anyLast({})", adj_factor_col)
    } else {
        "1.0".to_string()
    };

    let where_str = where_clauses.join(" AND ");

    let query = format!(
        r#"SELECT
            {} as trading_date,
            {} as symbol,
            anyLast({}) as close,
            any({}) as open,
            max({}) as high,
            min({}) as low,
            sum({}) as volume,
            {} as adj_factor
        FROM {}
        WHERE {}
        GROUP BY {}, {}
        ORDER BY {}, {}"#,
        date_column,
        symbol_column,
        mapping.close.as_deref().unwrap_or("close"),
        mapping.open.as_deref().unwrap_or("open"),
        mapping.high.as_deref().unwrap_or("high"),
        mapping.low.as_deref().unwrap_or("low"),
        mapping.volume.as_deref().unwrap_or("volume"),
        adj_factor_select,
        table,
        where_str,
        date_column,
        symbol_column,
        date_column,
        symbol_column
    );

    let url = format!(
        "http://{}:{}/?database={}",
        config.host, config.port, config.database
    );

    let response = client
        .get(&url)
        .query(&[("default_format", "JSONEachRow"), ("query", &query)])
        .basic_auth(&config.username, Some(&config.password))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Request failed: {}", e),
            )
        })?;

    let body = response.text().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read response: {}", e),
        )
    })?;

    // Get all unique dates and symbols
    let mut date_symbol_data: std::collections::HashMap<
        (String, String),
        (f64, f64, f64, f64, f64, f64),
    > = std::collections::HashMap::new();
    let mut dates_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut symbols_set: std::collections::HashSet<String> = std::collections::HashSet::new();

    for line in body.lines() {
        if let Ok(row) = serde_json::from_str::<serde_json::Value>(line) {
            let trading_date = row
                .get("trading_date")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let symbol = row
                .get("symbol")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let close = row.get("close").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let open = row.get("open").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let high = row.get("high").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let low = row.get("low").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let volume = row.get("volume").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let adj_factor = row
                .get("adj_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);

            dates_set.insert(trading_date.clone());
            symbols_set.insert(symbol.clone());
            date_symbol_data.insert(
                (trading_date, symbol),
                (close, open, high, low, volume, adj_factor),
            );
        }
    }

    // Sort dates and symbols
    let mut dates: Vec<String> = dates_set.into_iter().collect();
    dates.sort();
    let symbols: Vec<String> = symbols_set.into_iter().collect();

    // Build matrices
    let n_dates = dates.len();
    let n_symbols = symbols.len();

    let mut close_matrix: Vec<Vec<f64>> = vec![vec![0.0; n_symbols]; n_dates];
    let mut open_matrix: Vec<Vec<f64>> = vec![vec![0.0; n_symbols]; n_dates];
    let mut high_matrix: Vec<Vec<f64>> = vec![vec![0.0; n_symbols]; n_dates];
    let mut low_matrix: Vec<Vec<f64>> = vec![vec![0.0; n_symbols]; n_dates];
    let mut volume_matrix: Vec<Vec<f64>> = vec![vec![0.0; n_symbols]; n_dates];
    let mut adj_factor_matrix: Vec<Vec<f64>> = vec![vec![1.0; n_symbols]; n_dates];

    for (d_idx, date) in dates.iter().enumerate() {
        for (s_idx, symbol) in symbols.iter().enumerate() {
            if let Some((close, open, high, low, volume, adj_factor)) =
                date_symbol_data.get(&(date.clone(), symbol.clone()))
            {
                close_matrix[d_idx][s_idx] = *close;
                open_matrix[d_idx][s_idx] = *open;
                high_matrix[d_idx][s_idx] = *high;
                low_matrix[d_idx][s_idx] = *low;
                volume_matrix[d_idx][s_idx] = *volume;
                adj_factor_matrix[d_idx][s_idx] = *adj_factor;
            }
        }
    }

    // Calculate returns: (close_t / close_t-1) - 1
    let mut returns_matrix: Vec<Vec<f64>> = vec![vec![0.0; n_symbols]; n_dates];
    for s_idx in 0..n_symbols {
        for d_idx in 1..n_dates {
            let curr_close = close_matrix[d_idx][s_idx];
            let prev_close = close_matrix[d_idx - 1][s_idx];
            if prev_close != 0.0 && curr_close != 0.0 {
                returns_matrix[d_idx][s_idx] = curr_close / prev_close - 1.0;
            }
        }
    }

    Ok(Json(LoadDataResponse {
        dates,
        symbols,
        close: close_matrix,
        open: open_matrix,
        high: high_matrix,
        low: low_matrix,
        volume: volume_matrix,
        returns: returns_matrix,
        adj_factor: adj_factor_matrix,
    }))
}

/// Get column information for a table
async fn get_columns(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<Vec<ColumnInfo>>, (StatusCode, String)> {
    let config = state.db_config.read().await;

    if !config.connected {
        return Err((
            StatusCode::BAD_REQUEST,
            "Database not connected. Please configure database connection first.".to_string(),
        ));
    }

    let client = HttpClient::new();
    let table = params.get("table").cloned().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "Table parameter is required.".to_string(),
        )
    })?;

    // Query for column information
    let query = format!(
        "SELECT name, type FROM system.columns WHERE database = '{}' AND table = '{}' ORDER BY name",
        config.database, table
    );

    let url = format!("http://{}:{}/", config.host, config.port);

    let response = client
        .get(&url)
        .query(&[("default_format", "JSONEachRow"), ("query", &query)])
        .basic_auth(&config.username, Some(&config.password))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Request failed: {}", e),
            )
        })?;

    let body = response.text().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to read response: {}", e),
        )
    })?;

    let mut columns: Vec<ColumnInfo> = Vec::new();
    for line in body.lines() {
        if let Ok(row) = serde_json::from_str::<serde_json::Value>(line) {
            let name = row
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let column_type = row
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            columns.push(ColumnInfo { name, column_type });
        }
    }

    Ok(Json(columns))
}

/// Request to set column mapping for a specific table
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SetColumnMappingRequest {
    /// Table name (e.g., stock_1d, stock_1m, stock_5m)
    table: String,
    /// Column mapping for this table
    mapping: ColumnMapping,
}

/// Get column mapping configuration for a specific table
async fn get_column_mapping(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Json<ColumnMapping> {
    let mappings = state.column_mappings.read().await;
    let table = params
        .get("table")
        .cloned()
        .unwrap_or_else(|| "stock_1d".to_string());
    Json(mappings.get(&table))
}

/// Set column mapping configuration for a specific table
async fn set_column_mapping(
    State(state): State<AppState>,
    Json(req): Json<SetColumnMappingRequest>,
) -> Result<Json<ColumnMapping>, (StatusCode, String)> {
    // Validate required columns exist
    if req.mapping.close.is_none()
        || req.mapping.open.is_none()
        || req.mapping.high.is_none()
        || req.mapping.low.is_none()
        || req.mapping.volume.is_none()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "Required columns (close, open, high, low, volume) must be specified.".to_string(),
        ));
    }

    let mut mappings = state.column_mappings.write().await;
    mappings.set(&req.table, req.mapping.clone());

    // Persist to file
    save_column_mappings(&mappings);

    Ok(Json(req.mapping))
}

/// Get table mapping configuration
async fn get_table_mapping(State(state): State<AppState>) -> Json<TableMapping> {
    let mapping = state.table_mapping.read().await;
    Json(mapping.clone())
}

/// Set table mapping configuration
async fn set_table_mapping(
    State(state): State<AppState>,
    Json(req): Json<TableMapping>,
) -> Result<Json<TableMapping>, (StatusCode, String)> {
    // Validate stock_1day is provided
    if req.stock_1day.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "stock_1day (daily table) is required.".to_string(),
        ));
    }

    // Check if the database is connected
    let config = state.db_config.read().await;
    if !config.connected {
        return Err((
            StatusCode::BAD_REQUEST,
            "Database not connected. Please configure database connection first.".to_string(),
        ));
    }

    // Validate that stock_1day table exists
    let client = HttpClient::new();
    let query = format!(
        "SELECT 1 FROM system.tables WHERE database = '{}' AND table = '{}' LIMIT 1",
        config.database, req.stock_1day
    );
    let url = format!("http://{}:{}/", config.host, config.port);

    let response = client
        .get(&url)
        .query(&[("default_format", "JSONEachRow"), ("query", &query)])
        .basic_auth(&config.username, Some(&config.password))
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Request failed: {}", e),
            )
        })?;

    if !response.status().is_success() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Database query failed: {}", response.status()),
        ));
    }

    let body = response.text().await.unwrap_or_default();
    if body.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Table '{}' does not exist in database '{}'. Please check the table name.",
                req.stock_1day, config.database
            ),
        ));
    }

    // Optionally check if optional tables exist (warn only)
    if let Some(ref table_5min) = req.stock_5min
        && !table_5min.is_empty()
    {
        let query = format!(
            "SELECT 1 FROM system.tables WHERE database = '{}' AND table = '{}' LIMIT 1",
            config.database, table_5min
        );
        let response = client
            .get(&url)
            .query(&[("default_format", "JSONEachRow"), ("query", &query)])
            .basic_auth(&config.username, Some(&config.password))
            .send()
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Request failed: {}", e),
                )
            })?;

        let body = response.text().await.unwrap_or_default();
        if body.trim().is_empty() {
            eprintln!(
                "Warning: Optional table '{}' does not exist in database '{}'",
                table_5min, config.database
            );
        }
    }

    if let Some(ref table_1min) = req.stock_1min
        && !table_1min.is_empty()
    {
        let query = format!(
            "SELECT 1 FROM system.tables WHERE database = '{}' AND table = '{}' LIMIT 1",
            config.database, table_1min
        );
        let response = client
            .get(&url)
            .query(&[("default_format", "JSONEachRow"), ("query", &query)])
            .basic_auth(&config.username, Some(&config.password))
            .send()
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Request failed: {}", e),
                )
            })?;

        let body = response.text().await.unwrap_or_default();
        if body.trim().is_empty() {
            eprintln!(
                "Warning: Optional table '{}' does not exist in database '{}'",
                table_1min, config.database
            );
        }
    }

    // Save the mapping
    let mut mapping = state.table_mapping.write().await;
    *mapping = req.clone();

    Ok(Json(req))
}

/// Get available filter options (columns that can be used for filtering)
async fn get_filter_options(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<Vec<ColumnInfo>>, (StatusCode, String)> {
    // Reuse get_columns - filter options are just the columns from the table
    get_columns(State(state), axum::extract::Query(params)).await
}

/// Run GP factor mining via AlfarsLab (uses lab.run_gp)
async fn run_gp_factors(
    State(state): State<AppState>,
    Json(req): Json<GpRunRequest>,
) -> Result<Json<GpRunResponse>, (StatusCode, String)> {
    let start = std::time::Instant::now();

    let data_source = req.data_source.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "Data source not configured.".to_string(),
        )
    })?;

    let db_config = state.db_config.read().await;
    if !db_config.connected {
        return Err((
            StatusCode::BAD_REQUEST,
            "Database not connected.".to_string(),
        ));
    }

    // Build source from state
    let table_mapping = state.table_mapping.read().await;
    let source = build_source_from_state(&db_config, &table_mapping);
    let max_symbols = req.max_symbols.unwrap_or(0);

    // Build lab from source + config
    let backtest_config = BacktestConfig::default();
    let lab = build_lab_from_state(source, data_source, backtest_config)?;

    // Apply GP field/operator/seed configuration before run
    if let Some(ref fields) = req.fields {
        lab.set_gp_fields(fields.clone());
    }
    if let Some(ref ops) = req.ops {
        lab.set_gp_ops(ops.clone());
    }
    if let Some(ref seed) = req.seed_expression {
        lab.set_gp_seed(vec![seed.clone()])
            .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    }

    let gp_config = GPConfig {
        population_size: req.population_size.unwrap_or(50),
        max_generations: req.max_generations.unwrap_or(10),
        tournament_size: req.tournament_size.unwrap_or(5),
        crossover_prob: req.crossover_prob.unwrap_or(0.8),
        mutation_prob: req.mutation_prob.unwrap_or(0.15),
        max_depth: req.max_depth.unwrap_or(5),
        parent_diversity_penalty: 0.1,
        use_diverse_init: req.use_diverse_init.unwrap_or(true),
        smart_mutation_ratio: req.smart_mutation_ratio.unwrap_or(0.3),
        use_frequencies: false,
    };

    let max_gens = gp_config.max_generations;
    let num_factors = req.num_factors.unwrap_or(3);
    let results = lab
        .run_gp(gp_config, num_factors, max_symbols)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let mut factors = Vec::new();
    let mut best_idx = 0;
    let mut best_fitness = f64::NEG_INFINITY;
    for (i, (expr, bt)) in results.iter().enumerate() {
        let expr_str = alfars::gp::types::to_parseable_string(expr);
        let fitness = bt.ic_mean.max(0.0) + bt.ic_ir * 0.3;
        factors.push(GpFactor {
            id: format!("gp-factor-{}", i + 1),
            name: format!("GP Factor {}", i + 1),
            expression: expr_str,
            ic_mean: bt.ic_mean,
            ic_ir: bt.ic_ir,
            fitness,
            sharpe_ratio: bt.sharpe_ratio,
            total_return: bt.total_return,
            annualized_return: bt.annualized_return,
            max_drawdown: bt.max_drawdown,
            turnover: bt.turnover,
            calmar_ratio: bt.calmar_ratio,
            win_rate: bt.win_rate,
        });
        if fitness > best_fitness {
            best_fitness = fitness;
            best_idx = i;
        }
    }

    if factors.is_empty() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "No factors discovered".to_string(),
        ));
    }

    let best_factor = factors[best_idx].clone();
    let elapsed = start.elapsed().as_secs_f64();

    Ok(Json(GpRunResponse {
        factors,
        best_factor,
        generations: max_gens,
        elapsed_time: elapsed,
    }))
}

/// Get available GP terminal fields
async fn get_gp_fields() -> Json<Vec<String>> {
    Json(alfars::lab::AlfarsLab::avail_fields())
}

/// Get available GP operators
async fn get_gp_ops() -> Json<Vec<String>> {
    Json(alfars::lab::AlfarsLab::avail_ops())
}

/// Build a pre_filter string from DataSourceConfig.
fn build_lab_filter(data_source: &DataSourceConfig) -> String {
    let mut parts = vec![format!(
        "{}:{}",
        data_source.start_date, data_source.end_date
    )];
    if let Some(ref filters) = data_source.filters {
        for f in filters {
            parts.push(format!(
                "{} {} '{}'",
                f.column,
                f.operator.to_lowercase(),
                f.value
            ));
        }
    }
    parts.join(" ")
}

/// Build a ClickHouseSource from the server's current config state.
fn build_source_from_state(db_config: &DbConfig, table_mapping: &TableMapping) -> ClickHouseSource {
    if db_config.username.is_empty() {
        ClickHouseSource::new(&db_config.host, db_config.port, &db_config.database)
    } else {
        ClickHouseSource::with_username(
            &db_config.host,
            db_config.port,
            &db_config.database,
            &db_config.username,
        )
    }
    .with_password(if db_config.password.is_empty() {
        None
    } else {
        Some(db_config.password.clone())
    })
    .with_tables(
        &table_mapping.stock_1day,
        table_mapping.stock_5min.as_deref().unwrap_or("stock_5m"),
    )
}

/// Build an AlfarsLab configured from state + DataSourceConfig.
fn build_lab_from_state(
    source: ClickHouseSource,
    data_source: &DataSourceConfig,
    backtest_config: BacktestConfig,
) -> Result<AlfarsLab, (StatusCode, String)> {
    let filter = build_lab_filter(data_source);
    let start_year: i32 = data_source.start_date[..4]
        .parse()
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid start_date".to_string()))?;
    let end_year: i32 = data_source.end_date[..4]
        .parse()
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid end_date".to_string()))?;

    let mut lab = AlfarsLab::new(source);
    lab.set_pool(&filter);
    lab.set_duration(start_year, end_year);
    lab.set_backtest_config(backtest_config);
    Ok(lab)
}

#[tokio::main]
async fn main() {
    // Load saved database config if available
    let saved_db_config = load_saved_db_config();

    // Load saved column mappings if available
    let saved_column_mappings = load_saved_column_mappings();

    // Initialize application state with database config
    let state = AppState {
        db_config: Arc::new(RwLock::new(saved_db_config)),
        column_mappings: Arc::new(RwLock::new(saved_column_mappings)),
        table_mapping: Arc::new(RwLock::new(TableMapping {
            stock_1day: "stock_1day".to_string(),
            stock_5min: Some("stock_5min".to_string()),
            stock_1min: Some("stock_1min".to_string()),
        })),
        factor_cache: Arc::new(RwLock::new(HashMap::new())),
    };

    // Build the application
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(root))
        .route("/api/health", get(health_check))
        .route("/api/config", get(get_db_config))
        .route("/api/config", post(set_db_config))
        .route("/api/config/column-mapping", get(get_column_mapping))
        .route("/api/config/column-mapping", post(set_column_mapping))
        .route("/api/config/table-mapping", get(get_table_mapping))
        .route("/api/config/table-mapping", post(set_table_mapping))
        .route("/api/data/symbols", get(get_symbols))
        .route("/api/data/range", get(get_date_range))
        .route("/api/data/tables", post(get_tables))
        .route("/api/data/validate-tables", get(validate_tables))
        .route("/api/data/validate-columns", get(validate_columns))
        .route("/api/data/columns", get(get_columns))
        .route("/api/data/filter-options", get(get_filter_options))
        .route("/api/data/load", post(load_data))
        .route("/api/backtest", post(run_backtest))
        .route("/api/factors", get(list_factors))
        .route("/api/factors/compute", post(compute_factor))
        .route("/api/alphas", get(list_alphas))
        .route("/api/alphas", post(save_alpha))
        .route("/api/gp/run", post(run_gp_factors))
        .route("/api/gp/fields", get(get_gp_fields))
        .route("/api/gp/ops", get(get_gp_ops))
        .layer(cors)
        .with_state(state);

    // Run the server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8000));

    // Check if port is already in use
    if is_port_in_use(8000) {
        eprintln!("Warning: Port 8000 is already in use. Is another server running?");
        eprintln!(
            "Run 'pkill -f alfars-server' to stop any existing server, or use a different port."
        );
    }

    println!("Starting Alfa.rs server on http://{}", addr);
    println!("API Documentation: http://{}/docs", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.expect(
        "Failed to bind to port 8000. Is another instance already running? Try: pkill -f alfars-server",
    );
    axum::serve(listener, app)
        .await
        .expect("Server failed to run");
}
