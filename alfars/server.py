"""
FastAPI server for factor backtest visualization.

This module provides an HTTP API for running backtests and returning
NAV (Net Asset Value) data for interactive visualization.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np

app = FastAPI(
    title="Alfa.rs Backtest API",
    description="API for factor backtesting and NAV visualization",
    version="0.1.0",
)


class BacktestRequest(BaseModel):
    """Request model for backtest endpoint."""

    factor: List[List[float]] = Field(
        ..., description="Factor values, shape (n_days, n_assets)"
    )
    returns: List[List[float]] = Field(
        ..., description="Forward returns, shape (n_days, n_assets)"
    )
    dates: Optional[List[str]] = Field(
        default=None, description="Date labels for each day"
    )
    quantiles: int = Field(default=10, description="Number of quantile groups")
    weight_method: str = Field(
        default="equal", description="Weight method: 'equal' or 'weighted'"
    )
    long_top_n: int = Field(
        default=1, description="Number of top quantile groups to long"
    )
    short_top_n: int = Field(
        default=1, description="Number of bottom quantile groups to short"
    )
    commission_rate: float = Field(default=0.0, description="One-way commission rate")


class NavData(BaseModel):
    """NAV data for chart visualization."""

    dates: List[str]
    quantiles: List[List[float]]  # 各分位数净值曲线
    long_short: List[float]  # 多空曲线
    benchmark: List[float]  # 基准曲线 (equal-weighted market)
    ic_series: List[float]  # IC序列
    metrics: Dict[str, float]  # 回测指标


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": "Alfa.rs Backtest API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.post("/api/backtest", response_model=NavData)
def run_backtest(req: BacktestRequest):
    """
    Run backtest and return NAV data for visualization.

    Parameters
    ----------
    req : BacktestRequest
        Backtest request with factor and returns data

    Returns
    -------
    NavData
        NAV data including quantile curves, long-short curve, and metrics
    """
    try:
        import alfars as al

        # Convert to numpy arrays
        factor = np.array(req.factor, dtype=np.float64)
        returns = np.array(req.returns, dtype=np.float64)

        # Validate shapes
        if factor.shape != returns.shape:
            raise HTTPException(
                status_code=400,
                detail=f"Factor shape {factor.shape} must match returns shape {returns.shape}",
            )

        if factor.ndim != 2:
            raise HTTPException(
                status_code=400, detail="Factor and returns must be 2D arrays"
            )

        n_days, n_assets = factor.shape

        # Generate dates if not provided
        dates = req.dates
        if dates is None:
            dates = [
                f"2024-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}" for i in range(n_days)
            ]

        # Run backtest
        result = al.quantile_backtest(
            factor=factor,
            returns=returns,
            quantiles=req.quantiles,
            weight_method=req.weight_method,
            long_top_n=req.long_top_n,
            short_top_n=req.short_top_n,
            commission_rate=req.commission_rate,
        )

        # Prepare NAV data
        # Convert cumulative returns to NAV (starting at 1.0)
        group_cum_returns = result.group_cum_returns  # shape: (n_days-1, n_quantiles)
        n_quantile_days = group_cum_returns.shape[0]

        # Each quantile group's NAV curve
        quantiles_nav = []
        for i in range(req.quantiles):
            nav = 1.0 * (1 + group_cum_returns[:, i])
            quantiles_nav.append(nav.tolist())

        # Long-short NAV (starts at 1.0)
        long_short_nav = 1.0 * (1 + np.cumsum(result.long_short_returns))
        if len(long_short_nav) > 0:
            long_short_nav = long_short_nav.tolist()
        else:
            long_short_nav = []

        # Benchmark NAV (equal-weighted market)
        # Use mean return across all assets each day
        mean_returns = np.nanmean(returns, axis=1)
        benchmark_nav = 1.0 * np.cumprod(1 + mean_returns)
        if len(benchmark_nav) > 0:
            benchmark_nav = benchmark_nav.tolist()
        else:
            benchmark_nav = []

        # IC series
        ic_series = result.ic_series.tolist() if len(result.ic_series) > 0 else []

        # Metrics
        metrics = {
            "long_short_cum_return": float(result.long_short_cum_return),
            "total_return": float(result.total_return),
            "annualized_return": float(result.annualized_return),
            "sharpe_ratio": float(result.sharpe_ratio),
            "max_drawdown": float(result.max_drawdown),
            "turnover": float(result.turnover),
            "ic_mean": float(result.ic_mean),
            "ic_ir": float(result.ic_ir),
        }

        # Use dates aligned with the output (n_days - 1 for forward returns)
        nav_dates = (
            dates[1 : n_quantile_days + 1] if len(dates) > n_quantile_days else dates
        )

        return NavData(
            dates=nav_dates,
            quantiles=quantiles_nav,
            long_short=long_short_nav,
            benchmark=benchmark_nav[1:] if len(benchmark_nav) > 1 else benchmark_nav,
            ic_series=ic_series,
            metrics=metrics,
        )

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="alfars package not properly installed. Run 'maturin develop' first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
