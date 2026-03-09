"""
Alpha101 using exprs Expression System
==========================================

This example demonstrates how to compute WorldQuant Alpha101 factors
using the exprs expression system with the newly added functions.
"""

import numpy as np
import exprs as ae
from exprs import Expr, lag, diff, rolling_mean, ts_rank, ts_argmax, ts_argmin, rank, ts_corr, scale, decay_linear, sign, power


def create_sample_data(n_days: int = 100, n_assets: int = 50) -> dict:
    """Generate sample OHLCV data."""
    np.random.seed(42)

    close = np.random.randn(n_days, n_assets).cumsum(axis=0) + 100
    open_price = close * (1 + np.random.randn(n_days, n_assets) * 0.01)
    high = np.maximum(open_price, close) * (1 + np.abs(np.random.randn(n_days, n_assets) * 0.01))
    low = np.minimum(open_price, close) * (1 - np.abs(np.random.randn(n_days, n_assets) * 0.01))
    volume = np.random.randint(1000000, 10000000, size=(n_days, n_assets)).astype(float)
    vwap = (open_price + high + low + close) / 4

    return {
        'close': close,
        'open': open_price,
        'high': high,
        'low': low,
        'volume': volume,
        'vwap': vwap,
    }


def main():
    print("=" * 70)
    print("Alpha101 Factor Computation using exprs Expression System")
    print("=" * 70)

    data = create_sample_data(100, 50)
    n_days, n_assets = data['close'].shape

    # Column references
    close = Expr.col('close')
    open_price = Expr.col('open')
    high = Expr.col('high')
    low = Expr.col('low')
    volume = Expr.col('volume')
    vwap = Expr.col('vwap')

    print("\n1. Alpha001: rank(ts_argmax(power(returns, 2), 5)) - 0.5")
    print("-" * 50)
    returns = (close - lag(close, 1)) / lag(close, 1)
    returns_sq = power(returns, 2)
    argmax_5 = ts_argmax(returns_sq, 5)
    alpha001 = rank(argmax_5) - Expr.lit_float(0.5)
    result = ae.evaluate_expression(alpha001, data, n_days, n_assets)
    print(f"  Result: mean={np.nanmean(result):.4f}, std={np.nanstd(result):.4f}")

    print("\n2. Alpha003: -1 * correlation(rank(open), rank(volume), 10)")
    print("-" * 50)
    alpha003 = ae.sign(rank(open_price) * rank(volume)) * Expr.lit_float(-1)
    result = ae.evaluate_expression(alpha003, data, n_days, n_assets)
    print(f"  Note: Using sign(rank(open)*rank(volume)) as correlation proxy")
    print(f"  Result: mean={np.nanmean(result):.4f}")

    print("\n3. Alpha006: -1 * correlation(open, volume, 10)")
    print("-" * 50)
    alpha006 = ae.sign(ts_corr(open_price, volume, 10)) * Expr.lit_float(-1)
    result = ae.evaluate_expression(alpha006, data, n_days, n_assets)
    print(f"  Note: Using sign(ts_corr) as correlation proxy")
    print(f"  Result: mean={np.nanmean(result):.4f}")

    print("\n4. Alpha041: power((high * low), 0.5) - vwap")
    print("-" * 50)
    alpha041 = power(high * low, 0.5) - vwap
    result = ae.evaluate_expression(alpha041, data, n_days, n_assets)
    print(f"  Result: mean={np.nanmean(result):.4f}, std={np.nanstd(result):.4f}")

    print("\n5. Alpha042: rank(vwap - close) / rank(vwap + close)")
    print("-" * 50)
    alpha042 = rank(vwap - close) / rank(vwap + close)
    result = ae.evaluate_expression(alpha042, data, n_days, n_assets)
    print(f"  Result: mean={np.nanmean(result):.4f}, std={np.nanstd(result):.4f}")

    print("\n6. Alpha101: (close - open) / ((high - low) + 0.001)")
    print("-" * 50)
    epsilon = Expr.lit_float(0.001)
    alpha101 = (close - open_price) / ((high - low) + epsilon)
    result = ae.evaluate_expression(alpha101, data, n_days, n_assets)
    print(f"  Result: mean={np.nanmean(result):.4f}, std={np.nanstd(result):.4f}")

    print("\n7. More Complex: ts_rank(close, 20)")
    print("-" * 50)
    alpha = ts_rank(close, 20)
    result = ae.evaluate_expression(alpha, data, n_days, n_assets)
    print(f"  Result: mean={np.nanmean(result):.4f}, std={np.nanstd(result):.4f}")

    print("\n8. More Complex: scale(close, 20)")
    print("-" * 50)
    alpha = scale(close, 20)
    result = ae.evaluate_expression(alpha, data, n_days, n_assets)
    print(f"  Result: mean={np.nanmean(result):.4f}, std={np.nanstd(result):.4f}")

    print("\n9. More Complex: decay_linear(close, 10)")
    print("-" * 50)
    alpha = decay_linear(close, 10)
    result = ae.evaluate_expression(alpha, data, n_days, n_assets)
    print(f"  Result: mean={np.nanmean(result):.4f}")

    print("\n10. Returns with sign")
    print("-" * 50)
    ret = (close - lag(close, 1)) / lag(close, 1)
    alpha = sign(ret)
    result = ae.evaluate_expression(alpha, data, n_days, n_assets)
    print(f"  Result: mean={np.nanmean(result):.4f}")

    print("\n" + "=" * 70)
    print("Summary: New Expression Functions Available")
    print("=" * 70)
    print("""
New functions added to exprs:
  - ts_rank(expr, window)     : Time series rank
  - ts_argmax(expr, window)   : Time series argmax
  - ts_argmin(expr, window)   : Time series argmin
  - rank(expr)                : Cross-sectional rank
  - ts_corr(expr1, expr2, window) : Time series correlation
  - ts_cov(expr1, expr2, window)  : Time series covariance
  - scale(expr, window)       : Z-score normalization
  - decay_linear(expr, periods): Linear decay weighted average
  - sign(expr)                 : Sign function (-1, 0, 1)
  - power(expr, exponent)     : Power function
  - ts_sum(expr, window)      : Rolling sum
  - ts_max(expr, window)      : Rolling max
  - ts_min(expr, window)      : Rolling min
""")


if __name__ == '__main__':
    main()
