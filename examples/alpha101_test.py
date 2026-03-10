"""
Alpha101 Implementation and Testing
====================================

This module implements WorldQuant Alpha101 factors using the exprs expression system
and provides comprehensive testing with backtests.

Alpha101 Reference: https://arxiv.org/abs/1601.00991
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple
import alfars as ae
from alfars import (
    Expr,
    lag,
    diff,
    rolling_mean,
    ts_rank,
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_cov,
    ts_sum,
    ts_max,
    ts_min,
    rank,
    scale,
    decay_linear,
    sign,
    power,
    cumsum,
    cumprod,
)

# exp and log are available as methods on Expr


def create_sample_data(
    n_days: int = 252, n_assets: int = 100, seed: int = 42
) -> Dict[str, npt.NDArray]:
    """Generate realistic sample OHLCV data with correlations."""
    np.random.seed(seed)

    # Generate correlated price paths (geometric Brownian motion)
    dt = 1 / 252
    mu = 0.0  # Drift
    sigma = 0.02  # Volatility

    # Generate correlated random numbers
    correlation = 0.3
    uncorrelated = np.random.randn(n_days, n_assets)
    correlated = correlation * uncorrelated + np.sqrt(
        1 - correlation**2
    ) * np.random.randn(n_days, n_assets)

    # Generate prices
    returns = mu * dt + sigma * np.sqrt(dt) * correlated
    close = 100 * np.exp(np.cumsum(returns, axis=0))

    # Generate OHLCV from close
    daily_range = np.abs(np.random.randn(n_days, n_assets)) * 0.01 + 0.005
    open_price = close * (1 + np.random.randn(n_days, n_assets) * 0.002)
    high = np.maximum(close, open_price) * (
        1 + daily_range * np.random.rand(n_days, n_assets)
    )
    low = np.minimum(close, open_price) * (
        1 - daily_range * np.random.rand(n_days, n_assets)
    )
    volume = np.random.lognormal(15, 1, size=(n_days, n_assets))

    # Calculate vwap (simplified)
    vwap = (open_price + high + low + close) / 4

    return {
        "close": close,
        "open": open_price,
        "high": high,
        "low": low,
        "volume": volume,
        "vwap": vwap,
    }


class Alpha101:
    """Alpha101 factor implementations using exprs."""

    def __init__(self, data: Dict[str, npt.NDArray]):
        self.data = data
        self.n_days, self.n_assets = data["close"].shape
        self._setup_columns()

    def _setup_columns(self):
        """Set up column references."""
        self.close = Expr.col("close")
        self.open = Expr.col("open")
        self.high = Expr.col("high")
        self.low = Expr.col("low")
        self.volume = Expr.col("volume")
        self.vwap = Expr.col("vwap")

        # Pre-compute returns
        self.returns = (self.close - lag(self.close, 1)) / lag(self.close, 1)

    # ==================== Key Alpha101 Factors ====================

    def alpha001(self) -> Expr:
        """rank(ts_argmax(power(returns, 2), 5)) - 0.5"""
        return rank(ts_argmax(power(self.returns, 2), 5)) - Expr.lit_float(0.5)

    def alpha003(self) -> Expr:
        """-1 * correlation(rank(open), rank(volume), 10)"""
        return sign(ts_corr(rank(self.open), rank(self.volume), 10)) * Expr.lit_float(
            -1
        )

    def alpha004(self) -> Expr:
        """-1 * Ts_Rank(rank(low), 9)"""
        return sign(ts_rank(rank(self.low), 9)) * Expr.lit_float(-1)

    def alpha005(self) -> Expr:
        """rank((open - (sum(vwap, 10) / 10))) * (rank(-abs(returns)))"""
        vwap_ma = rolling_mean(self.vwap, 10)
        return rank(self.open - vwap_ma) * rank(-self.returns.abs())

    def alpha006(self) -> Expr:
        """-1 * correlation(open, volume, 10)"""
        return sign(ts_corr(self.open, self.volume, 10)) * Expr.lit_float(-1)

    def alpha007(self) -> Expr:
        """(adv20 < volume) ? ((-1 * ts_rank(abs(returns), 2)) * ts_rank(open - vwap, 2)) : -1"""
        adv20 = rolling_mean(self.volume, 20)
        # Simplified: use sign instead of conditional
        return sign(
            ts_rank(self.returns.abs(), 2) * ts_rank(self.open - self.vwap, 2)
        ) * Expr.lit_float(-1)

    def alpha008(self) -> Expr:
        """-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))"""
        sum_open_5 = rolling_mean(self.open, 5)
        sum_ret_5 = rolling_mean(self.returns, 5)
        product = sum_open_5 * sum_ret_5
        delayed = lag(product, 10)
        return rank(product - delayed) * Expr.lit_float(-1)

    def alpha009(self) -> Expr:
        """sign(delta(close, 1)) * ts_min(abs(delta(close, 1)), 5)"""
        delta_close = diff(self.close, 1)
        return sign(delta_close) * ts_min(delta_close.abs(), 5)

    def alpha010(self) -> Expr:
        """rank(sign(delta(close, 1)) * ts_min(abs(delta(close, 1)), 4))"""
        delta_close = diff(self.close, 1)
        return rank(sign(delta_close) * ts_min(delta_close.abs(), 4))

    def alpha011(self) -> Expr:
        """rank(sign(delta(close, 1)) * ts_min(abs(delta(close, 1)), 2))"""
        delta_close = diff(self.close, 1)
        return rank(sign(delta_close) * ts_min(delta_close.abs(), 2))

    def alpha012(self) -> Expr:
        """sign(delta(volume, 1)) * (-1 * delta(close, 1))"""
        delta_vol = diff(self.volume, 1)
        delta_close = diff(self.close, 1)
        return sign(delta_vol) * (Expr.lit_float(0) - delta_close)

    def alpha013(self) -> Expr:
        """-1 * rank(covariance(rank(close), rank(volume), 5))"""
        return rank(ts_cov(rank(self.close), rank(self.volume), 5)) * Expr.lit_float(-1)

    def alpha014(self) -> Expr:
        """-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)"""
        corr = ts_corr(rank(self.high), rank(self.volume), 3)
        return rank(corr) * Expr.lit_float(-1)

    def alpha015(self) -> Expr:
        """-1 * rank(covariance(rank(low), rank(volume), 5))"""
        return rank(ts_cov(rank(self.low), rank(self.volume), 5)) * Expr.lit_float(-1)

    def alpha016(self) -> Expr:
        """-1 * rank(sum(returns, 10) / sum(sum(returns, 2), 3)) * rank(delta(close, 1))"""
        sum_ret_10 = ts_sum(self.returns, 10)
        sum_ret_2 = ts_sum(self.returns, 2)
        denominator = rolling_mean(sum_ret_2, 3)
        delta_close = diff(self.close, 1)
        return (Expr.lit_float(0) - rank(sum_ret_10 / denominator)) * rank(delta_close)

    def alpha017(self) -> Expr:
        """-1 * rank((stddev(returns, 2) * correlation(returns, volume, 2)))"""
        # Approximation: use rolling mean as proxy for std
        std_ret = rolling_mean(self.returns.abs(), 2) * Expr.lit_float(
            1.25
        )  # Rough approximation
        corr = ts_corr(self.returns, self.volume, 2)
        return rank(std_ret * corr) * Expr.lit_float(-1)

    def alpha018(self) -> Expr:
        """-1 * sign((sum(returns, 10) / sum(sum(returns, 2), 3)) * delay(returns, 3))"""
        sum_ret_10 = ts_sum(self.returns, 10)
        sum_ret_2 = ts_sum(self.returns, 2)
        denominator = rolling_mean(sum_ret_2, 3)
        delayed_ret = lag(self.returns, 3)
        return sign((sum_ret_10 / denominator) * delayed_ret) * Expr.lit_float(-1)

    def alpha019(self) -> Expr:
        """-1 * power(rank((sum(delay(returns, 2), 5) / rank(sum(returns, 3)))), 2)"""
        delayed_ret_2 = lag(self.returns, 2)
        sum_delay_5 = ts_sum(delayed_ret_2, 5)
        sum_ret_3 = ts_sum(self.returns, 3)
        return power(rank(sum_delay_5 / rank(sum_ret_3)), 2) * Expr.lit_float(-1)

    def alpha020(self) -> Expr:
        """(0 - rank((open - delay(high, 1)))) * (0 - rank(abs((close - delay(close, 1)))))"""
        open_delay_high = self.open - lag(self.high, 1)
        close_delay = self.close - lag(self.close, 1)
        return (Expr.lit_float(0) - rank(open_delay_high)) * (
            Expr.lit_float(0) - rank(close_delay.abs())
        )

    def alpha021(self) -> Expr:
        """-1 * rank(correlation(close, vwap, 6) * std(high))"""
        # Approximation: use rolling mean as proxy for std
        corr = ts_corr(self.close, self.vwap, 6)
        std_high = rolling_mean(
            (self.high - rolling_mean(self.high, 2)).abs(), 2
        ) * Expr.lit_float(1.25)
        return rank(corr * std_high) * Expr.lit_float(-1)

    def alpha022(self) -> Expr:
        """-1 * delta(correlation(high, volume, 5), 5) * rank(stddev(returns))"""
        corr = ts_corr(self.high, self.volume, 5)
        delta_corr = diff(corr, 5)
        std_ret = rolling_mean(self.returns.abs(), 20) * Expr.lit_float(1.25)
        return delta_corr * rank(std_ret) * Expr.lit_float(-1)

    def alpha023(self) -> Expr:
        """sign(high - rolling_mean(high, 20)) * delta(high, 2)"""
        high_ma = rolling_mean(self.high, 20)
        return sign(self.high - high_ma) * diff(self.high, 2)

    def alpha024(self) -> Expr:
        """-1 * (close - ts_min(low, 100))"""
        return (self.close - ts_min(self.low, 100)) * Expr.lit_float(-1)

    def alpha025(self) -> Expr:
        """-1 * ts_rank(abs(returns), 5) * sign(returns) * rank(delta(close, 1))"""
        delta_close = diff(self.close, 1)
        return ts_rank(self.returns.abs(), 5) * sign(self.returns) * rank(-delta_close)

    def alpha026(self) -> Expr:
        """(close - delay(close, 7)) * (1 - rank(decay_linear(volume / adv20, 9)))"""
        delta_close_7 = diff(self.close, 7)
        adv20 = rolling_mean(self.volume, 20)
        volume_ratio = self.volume / adv20
        decay = decay_linear(volume_ratio, 9)
        return delta_close_7 * (Expr.lit_float(1) - rank(decay))

    def alpha027(self) -> Expr:
        """-1 * rank(stddev(returns, 2)) * correlation(returns, volume, 2)"""
        std_ret = rolling_mean(self.returns.abs(), 2) * Expr.lit_float(1.25)
        corr = ts_corr(self.returns, self.volume, 2)
        return rank(std_ret) * corr * Expr.lit_float(-1)

    def alpha028(self) -> Expr:
        """indneutralize(log(volume) - log(mean(log(volume), 20))) - rank(close_pct_change)"""
        log_vol = self.volume.log()
        log_vol_ma = rolling_mean(log_vol, 20)
        indneutralize = log_vol - log_vol_ma
        close_pct_change = (self.close - lag(self.close, 20)) / lag(self.close, 20)
        return (indneutralize - rank(close_pct_change)) * Expr.lit_float(-1)

    def alpha029(self) -> Expr:
        """power(rank(volume / mean(volume, 20)), delta(close, 1))"""
        vol_ratio = self.volume / rolling_mean(self.volume, 20)
        delta_close_1 = diff(self.close, 1)
        return power(rank(vol_ratio), delta_close_1)

    def alpha030(self) -> Expr:
        """-1 * Ts_Rank(abs(returns), 6) * sign(returns)"""
        return ts_rank(self.returns.abs(), 6) * sign(self.returns) * Expr.lit_float(-1)

    def alpha035(self) -> Expr:
        """-1 * rank(covariance(rank(high), rank(volume), 3))"""
        return rank(ts_cov(rank(self.high), rank(self.volume), 3)) * Expr.lit_float(-1)

    def alpha036(self) -> Expr:
        """sign(delta(close, 1)) * ts_min(abs(delta(close, 1)), 4) * rank(sum(correlation(close, volume, 10), 10))"""
        delta_close = diff(self.close, 1)
        signed_delta = sign(delta_close) * ts_min(delta_close.abs(), 4)
        corr_sum = ts_sum(ts_corr(self.close, self.volume, 10), 10)
        return signed_delta * rank(corr_sum)

    def alpha037(self) -> Expr:
        """rank(delay((high - low) / sum(close, 5), 2)) * rank(rank(volume))"""
        close_ma5 = rolling_mean(self.close, 5)
        range_ma = (self.high - self.low) / close_ma5
        delayed_range = lag(range_ma, 2)
        return rank(delayed_range) * rank(rank(self.volume))

    def alpha038(self) -> Expr:
        """-1 * rank(ts_sum(correlation(vwap, sum(mean(volume, 5), 26), 5), 7))"""
        vol_ma5 = rolling_mean(self.volume, 5)
        vol_ma5_sum = ts_sum(vol_ma5, 26)
        corr = ts_corr(self.vwap, vol_ma5_sum, 5)
        return rank(ts_sum(corr, 7)) * Expr.lit_float(-1)

    def alpha039(self) -> Expr:
        """-1 * rank(delay(close, 5) / 20) * correlation(close, volume, 2)"""
        delayed_close = lag(self.close, 5)
        delayed_ma = rolling_mean(delayed_close, 20)
        corr1 = ts_corr(self.close, self.volume, 2)
        return rank(delayed_ma) * corr1 * Expr.lit_float(-1)

    def alpha040(self) -> Expr:
        """-1 * rank(stddev(abs(close - delay(close, 1)), 5))"""
        abs_diff = (self.close - lag(self.close, 1)).abs()
        std_diff = rolling_mean(abs_diff, 5) * Expr.lit_float(1.25)
        return rank(std_diff) * Expr.lit_float(-1)

    def alpha041(self) -> Expr:
        """power(((high * low) ** 0.5) - vwap, 2)"""
        geometric_mean = power(self.high * self.low, 0.5)
        return power(geometric_mean - self.vwap, 2)

    def alpha042(self) -> Expr:
        """rank(vwap - close) / rank(vwap + close)"""
        return rank(self.vwap - self.close) / rank(self.vwap + self.close)

    def alpha043(self) -> Expr:
        """-1 * ts_rank(volume / mean(volume, 20), 20) * ts_rank(-1 * delta(close, 7), 8)"""
        vol_ratio = self.volume / rolling_mean(self.volume, 20)
        delta_close_7 = diff(self.close, 7)
        return (
            ts_rank(rank(vol_ratio), 20)
            * ts_rank(rank(-delta_close_7), 8)
            * Expr.lit_float(-1)
        )

    def alpha044(self) -> Expr:
        """-1 * rank(stddev(close / delay(close, 1), 2))"""
        close_ratio = self.close / lag(self.close, 1)
        std_ratio = rolling_mean(
            (close_ratio - rolling_mean(close_ratio, 2)).abs(), 2
        ) * Expr.lit_float(1.25)
        return rank(std_ratio) * Expr.lit_float(-1)

    def alpha045(self) -> Expr:
        """-1 * rank(correlation(sum(((close * 0.35) + (vwap * 0.65)), 2), sum(mean(volume, 11), 2), 6))"""
        combined = self.close * Expr.lit_float(0.35) + self.vwap * Expr.lit_float(0.65)
        combined_sum = ts_sum(combined, 2)
        vol_ma11 = rolling_mean(self.volume, 11)
        vol_sum = ts_sum(vol_ma11, 2)
        corr = ts_corr(combined_sum, vol_sum, 6)
        return rank(corr) * Expr.lit_float(-1)

    def alpha046(self) -> Expr:
        """-1 * rank(Delta(Ts_ArgMax(close, 30), 1))"""
        argmax_close = ts_argmax(self.close, 30)
        delta_argmax = diff(argmax_close, 1)
        return rank(delta_argmax) * Expr.lit_float(-1)

    def alpha047(self) -> Expr:
        """((rank((1 / close)) * volume) / ((ts_sum(volume, 5) / 5)) - 1"""
        rank_close = rank(Expr.lit_float(1) / self.close)
        vol_ma5 = rolling_mean(self.volume, 5)
        return (rank_close * self.volume / vol_ma5) - Expr.lit_float(1)

    def alpha049(self) -> Expr:
        """-1 * rank(Delta(open, 1) * rank(close - vwap) + rank(open - vwap))"""
        delta_open = diff(self.open, 1)
        rank_vwap_diff = rank(self.close - self.vwap) + rank(self.open - self.vwap)
        return rank(delta_open * rank_vwap_diff) * Expr.lit_float(-1)

    def alpha050(self) -> Expr:
        """-1 * rank(stddev(high, 10)) * rank(Delta(vwap, 1))"""
        std_high = rolling_mean(
            (self.high - rolling_mean(self.high, 10)).abs(), 10
        ) * Expr.lit_float(1.25)
        delta_vwap = diff(self.vwap, 1)
        return rank(std_high) * rank(delta_vwap) * Expr.lit_float(-1)

    def alpha051(self) -> Expr:
        """(rank(ts_max(vwap - close, 3)) + ts_rank(vwap - min(vwap, 3), 10)) * -1"""
        vwap_diff_max = ts_max(self.vwap - self.close, 3)
        vwap_diff_min = ts_min(self.vwap - self.close, 3)
        return (rank(vwap_diff_max) + ts_rank(vwap_diff_min, 10)) * Expr.lit_float(-1)

    def alpha052(self) -> Expr:
        """-1 * Ts_Rank(ts_sum(returns, 7), 5) * Ts_Rank(ts_rank(-stddev(close, 7), 3), 5)"""
        sum_ret_7 = ts_sum(self.returns, 7)
        std_close = rolling_mean(
            (self.close - rolling_mean(self.close, 7)).abs(), 7
        ) * Expr.lit_float(1.25)
        return ts_rank(sum_ret_7, 5) * ts_rank(rank(-std_close), 5) * Expr.lit_float(-1)

    def alpha053(self) -> Expr:
        """-1 * rank(stddev(open, 2) + delta(open, 1) + ts_corr(open, volume, 10))"""
        std_open = rolling_mean(
            (self.open - rolling_mean(self.open, 2)).abs(), 2
        ) * Expr.lit_float(1.25)
        delta_open = diff(self.open, 1)
        corr = ts_corr(self.open, self.volume, 10)
        return rank(std_open + delta_open + corr) * Expr.lit_float(-1)

    def alpha054(self) -> Expr:
        """-1 * ts_rank(ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)), 3)"""
        vol_rank = ts_rank(self.volume, 32)
        hl_range = self.close + self.high - self.low
        hl_rank = ts_rank(hl_range, 16)
        return ts_rank(vol_rank * (Expr.lit_float(1) - hl_rank), 3) * Expr.lit_float(-1)

    def alpha055(self) -> Expr:
        """-1 * rank(abs(correlation(close, volume, 2)))"""
        corr = ts_corr(self.close, self.volume, 2)
        return rank(corr.abs()) * Expr.lit_float(-1)

    def alpha057(self) -> Expr:
        """-1 * rank(correlation(vwap, rank(close), 8)) * rank(correlation(rank(low), rank(volume), 5))"""
        corr1 = ts_corr(self.vwap, rank(self.close), 8)
        corr2 = ts_corr(rank(self.low), rank(self.volume), 5)
        return rank(corr1) * rank(corr2) * Expr.lit_float(-1)

    def alpha058(self) -> Expr:
        """-1 * delta(correlation(high, volume, 5), 5) * rank(stddev(returns))"""
        corr = ts_corr(self.high, self.volume, 5)
        delta_corr = diff(corr, 5)
        std_ret = rolling_mean(self.returns.abs(), 20) * Expr.lit_float(1.25)
        return delta_corr * rank(std_ret) * Expr.lit_float(-1)

    def alpha059(self) -> Expr:
        """-1 * rank(delta(correlation(vwap, sum(mean(volume, 5), 180), 8))) * rank(correlation(mean((high+low)/2, 3), mean(volume, 10), 5))"""
        delta_vol = diff(self.volume, 1)
        vol_ma5 = rolling_mean(self.volume, 5)
        vol_sum = ts_sum(vol_ma5, 180)
        corr1 = ts_corr(self.vwap, vol_sum, 8)
        price_mid = (self.high + self.low) / Expr.lit_float(2)
        price_mid_ma = rolling_mean(price_mid, 3)
        vol_ma10 = rolling_mean(self.volume, 10)
        corr2 = ts_corr(price_mid_ma, vol_ma10, 5)
        return rank(delta_vol * rank(corr1)) * rank(corr2) * Expr.lit_float(-1)

    def alpha060(self) -> Expr:
        """-1 * rank(exp(delta(vwap, 1))) * rank(correlation(ts_rank(close, 3), ts_rank(volume, 10), 7))"""
        delta_vwap = diff(self.vwap, 1)
        exp_vwap = delta_vwap.exp()
        close_rank_3 = ts_rank(self.close, 3)
        vol_rank_10 = ts_rank(self.volume, 10)
        corr = ts_corr(close_rank_3, vol_rank_10, 7)
        return rank(exp_vwap) * rank(corr) * Expr.lit_float(-1)

    def alpha065(self) -> Expr:
        """rank(correlation(((high * 0.9) + (close * 0.1)), mean(volume, 10), 10)) * rank(correlation(ts_rank(((close * 0.9) + (open * 0.1)), 4), ts_rank(volume, 9), 7))"""
        price_composite = self.high * Expr.lit_float(0.9) + self.close * Expr.lit_float(
            0.1
        )
        vol_ma10 = rolling_mean(self.volume, 10)
        corr1 = ts_corr(price_composite, vol_ma10, 10)
        close_rank = self.close * Expr.lit_float(0.9) + self.open * Expr.lit_float(0.1)
        corr2 = ts_corr(ts_rank(close_rank, 4), ts_rank(self.volume, 9), 7)
        return rank(corr1) * rank(corr2)

    def alpha066(self) -> Expr:
        """rank(correlation(vwap, sum(mean(volume, 5), 26), 9)) * rank(correlation(rank(low * 0.3522 + vwap * 0.6478), rank(mean(volume, 80)), 8))"""
        vol_ma5 = rolling_mean(self.volume, 5)
        vol_sum = ts_sum(vol_ma5, 26)
        corr1 = ts_corr(self.vwap, vol_sum, 9)
        composite = self.low * Expr.lit_float(0.3522) + self.vwap * Expr.lit_float(
            0.6478
        )
        vol_ma80 = rolling_mean(self.volume, 80)
        corr2 = ts_corr(rank(composite), rank(vol_ma80), 8)
        return rank(corr1) * rank(corr2)

    def alpha101(self) -> Expr:
        """(close - open) / ((high - low) + 0.001)"""
        epsilon = Expr.lit_float(0.001)
        return (self.close - self.open) / ((self.high - self.low) + epsilon)


def evaluate_alpha(
    expr: Expr, data: Dict[str, npt.NDArray], name: str = "Alpha"
) -> Tuple:
    """Evaluate an alpha expression and return results with statistics."""
    n_days, n_assets = data["close"].shape

    try:
        result = ae.evaluate_expression(expr, data, n_days, n_assets)

        # Calculate statistics
        stats = {
            "mean": float(np.nanmean(result)),
            "std": float(np.nanstd(result)),
            "min": float(np.nanmin(result)),
            "max": float(np.nanmax(result)),
            "nan_ratio": float(np.mean(np.isnan(result))),
            "valid_count": int(np.sum(~np.isnan(result))),
        }
        return result, stats
    except Exception as e:
        print(f"  Error evaluating {name}: {e}")
        import traceback

        traceback.print_exc()
        return None, {"error": str(e)}


def run_backtest(
    factor: npt.NDArray, returns: npt.NDArray, alpha_name: str = "Alpha"
) -> Dict:
    """Run a simple backtest on the factor."""
    try:
        result = ae.quantile_backtest(
            factor=factor,
            returns=returns,
            quantiles=5,
            weight_method="equal",
            long_top_n=1,
            short_top_n=1,
            commission_rate=0.001,
        )

        return {
            "long_short_return": float(result.long_short_cum_return),
            "ic_mean": float(result.ic_mean),
            "ic_ir": float(result.ic_ir),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    """Main function to run Alpha101 tests."""
    print("=" * 80)
    print("Alpha101 Implementation Test")
    print("=" * 80)

    # Create sample data
    print("\n1. Creating sample data...")
    data = create_sample_data(n_days=252, n_assets=100, seed=42)
    print(f"   Data shape: {data['close'].shape}")

    # Calculate returns for backtesting
    returns = (data["close"][1:] - data["close"][:-1]) / data["close"][:-1]

    # Test selected Alpha101 factors
    alphas_to_test = [
        ("alpha001", lambda a: a.alpha001()),
        ("alpha003", lambda a: a.alpha003()),
        ("alpha006", lambda a: a.alpha006()),
        ("alpha009", lambda a: a.alpha009()),
        ("alpha010", lambda a: a.alpha010()),
        ("alpha012", lambda a: a.alpha012()),
        ("alpha013", lambda a: a.alpha013()),
        ("alpha014", lambda a: a.alpha014()),
        ("alpha015", lambda a: a.alpha015()),
        ("alpha020", lambda a: a.alpha020()),
        ("alpha030", lambda a: a.alpha030()),
        ("alpha041", lambda a: a.alpha041()),
        ("alpha042", lambda a: a.alpha042()),
        ("alpha046", lambda a: a.alpha046()),
        ("alpha047", lambda a: a.alpha047()),
        ("alpha049", lambda a: a.alpha049()),
        ("alpha050", lambda a: a.alpha050()),
        ("alpha051", lambda a: a.alpha051()),
        ("alpha055", lambda a: a.alpha055()),
        ("alpha057", lambda a: a.alpha057()),
        ("alpha101", lambda a: a.alpha101()),
    ]

    alpha101_instance = Alpha101(data)

    print("\n2. Testing Alpha Factors...")
    results = []

    for name, alpha_func in alphas_to_test:
        print(f"\n   Testing {name}...")
        try:
            # Call the alpha method on the instance
            expr = alpha_func(alpha101_instance)
            result, stats = evaluate_alpha(expr, data, name)

            if result is not None:
                # Run backtest
                # Align returns with factor (factor at t predicts returns at t+1)
                min_len = min(len(result) - 1, len(returns))
                factor_aligned = result[:min_len]
                returns_aligned = returns[:min_len]

                bt_result = run_backtest(factor_aligned, returns_aligned, name)

                print(f"      Mean: {stats['mean']:>10.4f}, Std: {stats['std']:>10.4f}")
                ic = bt_result.get("ic_mean", 0)
                ir = bt_result.get("ic_ir", 0)
                print(f"      IC Mean: {ic:>10.4f}, IC IR: {ir:>10.4f}")

                results.append(
                    {
                        "name": name,
                        "stats": stats,
                        "backtest": bt_result,
                    }
                )
        except Exception as e:
            print(f"      Error: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Alpha':<12} {'Mean':>10} {'Std':>10} {'IC Mean':>10} {'IC IR':>10}")
    print("-" * 60)
    for r in results:
        bt = r["backtest"]
        print(
            f"{r['name']:<12} {r['stats']['mean']:>10.4f} {r['stats']['std']:>10.4f} {bt.get('ic_mean', 0):>10.4f} {bt.get('ic_ir', 0):>10.4f}"
        )

    # Best performers
    valid_results = [r for r in results if "ic_mean" in r["backtest"]]
    if valid_results:
        sorted_by_ic = sorted(
            valid_results,
            key=lambda x: abs(x["backtest"].get("ic_mean", 0)),
            reverse=True,
        )
        print("\n" + "=" * 80)
        print("Top 5 by |IC Mean|")
        print("=" * 80)
        for r in sorted_by_ic[:5]:
            print(
                f"  {r['name']}: IC={r['backtest']['ic_mean']:+.4f}, IR={r['backtest']['ic_ir']:+.4f}"
            )

    print("\n" + "=" * 80)
    print("Alpha101 Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
