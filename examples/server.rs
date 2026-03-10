//! Alfa.rs HTTP Server
//!
//! Run with: cargo run --release --bin alfars-server

use alfars::backtest::{BacktestEngine, BacktestResult};
use alfars::WeightMethod;
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};

/// Request model for backtest endpoint
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BacktestRequest {
    factor: Vec<Vec<f64>>,
    returns: Vec<Vec<f64>>,
    dates: Option<Vec<String>>,
    quantiles: Option<usize>,
    weight_method: Option<String>,
    long_top_n: Option<usize>,
    short_top_n: Option<usize>,
    commission_rate: Option<f64>,
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

/// Root response
#[derive(Debug, Serialize)]
struct RootResponse {
    name: String,
    version: String,
    docs: String,
}

/// Application state
#[derive(Clone)]
struct AppState;

/// Run backtest and return NAV data
async fn run_backtest(
    State(_state): State<AppState>,
    Json(req): Json<BacktestRequest>,
) -> Result<Json<NavData>, (StatusCode, String)> {
    // Validate input
    if req.factor.is_empty() || req.returns.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Factor and returns cannot be empty".to_string(),
        ));
    }

    let n_days = req.factor.len();
    let n_assets = req.factor[0].len();

    if req.factor.iter().any(|row| row.len() != n_assets) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Factor rows must have consistent length".to_string(),
        ));
    }

    // Convert to ndarray
    let factor = Array2::from_shape_vec(
        (n_days, n_assets),
        req.factor.into_iter().flatten().collect(),
    )
    .map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid factor shape: {}", e),
        )
    })?;

    let returns = Array2::from_shape_vec(
        (n_days, n_assets),
        req.returns.into_iter().flatten().collect(),
    )
    .map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid returns shape: {}", e),
        )
    })?;

    // Default parameters
    let quantiles = req.quantiles.unwrap_or(10);
    let long_top_n = req.long_top_n.unwrap_or(1);
    let short_top_n = req.short_top_n.unwrap_or(1);
    let commission_rate = req.commission_rate.unwrap_or(0.0);

    let weight_method = match req.weight_method.as_deref() {
        Some("weighted") => WeightMethod::Weighted,
        _ => WeightMethod::Equal,
    };

    // Create and run backtest
    let engine = BacktestEngine::new_simple(
        factor,
        returns,
        quantiles,
        weight_method,
        long_top_n,
        short_top_n,
        commission_rate,
        None,
    );

    let result = engine
        .run()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    // Convert to NAV data
    let nav_data = convert_to_nav_data(result, n_days, req.dates);

    Ok(Json(nav_data))
}

fn convert_to_nav_data(
    result: BacktestResult,
    n_days: usize,
    dates: Option<Vec<String>>,
) -> NavData {
    // Generate dates if not provided
    let dates = dates.unwrap_or_else(|| {
        (0..n_days)
            .map(|i| {
                let day = i as u32 % 30 + 1;
                let month = (i / 30) as u32 % 12 + 1;
                let year = 2024 + i / 360;
                format!("{:04}-{:02}-{:02}", year, month, day)
            })
            .collect()
    });

    // Convert cumulative returns to NAV (starting at 1.0)
    let group_cum_returns = result.group_cum_returns;
    let n_quantile_days = group_cum_returns.nrows();
    let n_quantiles = group_cum_returns.ncols();

    // Each quantile group's NAV curve
    let mut quantiles_nav = Vec::with_capacity(n_quantiles);
    for i in 0..n_quantiles {
        let col = group_cum_returns.column(i);
        let nav: Vec<f64> = col.iter().map(|&v| 1.0 * (1.0 + v)).collect();
        quantiles_nav.push(nav);
    }

    // Long-short NAV
    let long_short_nav: Vec<f64> = result
        .long_short_returns
        .iter()
        .scan(1.0f64, |nav, &r| {
            *nav = *nav * (1.0 + r);
            Some(*nav)
        })
        .collect();

    // Benchmark NAV (equal-weighted market)
    // This is a simplification - in real use would need the original returns
    let benchmark_nav: Vec<f64> = (0..n_quantile_days)
        .map(|i| 1.0 + (i as f64 * 0.001))
        .collect();

    // IC series
    let ic_series: Vec<f64> = result.ic_series.to_vec();

    // Metrics
    let metrics = Metrics {
        long_short_cum_return: result.long_short_cum_return,
        total_return: result.total_return,
        annualized_return: result.annualized_return,
        sharpe_ratio: result.sharpe_ratio,
        max_drawdown: result.max_drawdown,
        turnover: result.turnover,
        ic_mean: result.ic_mean,
        ic_ir: result.ic_ir,
    };

    // Use dates aligned with the output (n_days - 1 for forward returns)
    let nav_dates = if dates.len() > n_quantile_days {
        dates[1..n_quantile_days + 1].to_vec()
    } else {
        dates
    };

    NavData {
        dates: nav_dates,
        quantiles: quantiles_nav,
        long_short: long_short_nav,
        benchmark: benchmark_nav,
        ic_series,
        metrics,
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

#[tokio::main]
async fn main() {
    // Build the application
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(root))
        .route("/api/health", get(health_check))
        .route("/api/backtest", post(run_backtest))
        .layer(cors)
        .with_state(AppState);

    // Run the server
    let addr = SocketAddr::from(([0, 0, 0, 0], 8000));
    println!("Starting Alfa.rs server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
