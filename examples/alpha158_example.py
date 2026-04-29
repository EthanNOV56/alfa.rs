#!/usr/bin/env python3
"""
Alpha158 Technical Factor Example.

Computes 158 technical factors from daily OHLCV data using the full Rust pipeline:
ClickHouseSource → DataLayer → FactorRegistry.compute_factor_matrices_1d →
FactorCombiner → BacktestEngine.run_with_prices.
"""

import alfars as al


def build_alphas() -> dict[str, str]:
    """Return dict of factor_name → alfars expression using ts_/cs_ canonical names."""
    a: dict[str, str] = {}

    # 9 basic features
    a["kmid"] = "(close - open) / open"
    a["klen"] = "(high - low) / open"
    a["kmid_2"] = "(close - open) / (high - low + 1e-12)"
    a["kup"] = "(high - gt(open, close)) / open"
    a["kup_2"] = "(high - gt(open, close)) / (high - low + 1e-12)"
    a["klow"] = "(lt(open, close) - low) / open"
    a["klow_2"] = "(lt(open, close) - low) / (high - low + 1e-12)"
    a["ksft"] = "(close * 2 - high - low) / open"
    a["ksft_2"] = "(close * 2 - high - low) / (high - low + 1e-12)"

    # 4 price ratio features
    for field in ["open", "high", "low", "vwap"]:
        a[f"{field}_0"] = f"{field} / close"

    # 5 windows × time-series features
    for w in [5, 10, 20, 30, 60]:
        a[f"roc_{w}"] = f"ts_delay(close, {w}) / close"
        a[f"ma_{w}"] = f"ts_mean(close, {w}) / close"
        a[f"std_{w}"] = f"ts_std(close, {w}) / close"
        a[f"beta_{w}"] = f"ts_slope(close, {w}) / close"
        a[f"rsqr_{w}"] = f"ts_rsquare(close, {w})"
        a[f"resi_{w}"] = f"ts_resi(close, {w}) / close"
        a[f"max_{w}"] = f"ts_max(high, {w}) / close"
        a[f"min_{w}"] = f"ts_min(low, {w}) / close"
        a[f"qtlu_{w}"] = f"ts_quantile(close, {w}, 0.8) / close"
        a[f"qtld_{w}"] = f"ts_quantile(close, {w}, 0.2) / close"
        a[f"rank_{w}"] = f"ts_rank(close, {w})"
        a[f"rsv_{w}"] = (
            f"(close - ts_min(low, {w})) "
            f"/ (ts_max(high, {w}) - ts_min(low, {w}) + 1e-12)"
        )
        a[f"imax_{w}"] = f"ts_argmax(high, {w}) / {w}"
        a[f"imin_{w}"] = f"ts_argmin(low, {w}) / {w}"
        a[f"imxd_{w}"] = (
            f"(ts_argmax(high, {w}) - ts_argmin(low, {w})) / {w}"
        )
        a[f"corr_{w}"] = f"ts_correlation(close, log(volume + 1), {w})"
        a[f"cord_{w}"] = (
            f"ts_correlation(close / ts_delay(close, 1), "
            f"log(volume / ts_delay(volume, 1) + 1), {w})"
        )
        a[f"cntp_{w}"] = f"ts_mean(gt(close, ts_delay(close, 1)), {w})"
        a[f"cntn_{w}"] = f"ts_mean(lt(close, ts_delay(close, 1)), {w})"
        a[f"cntd_{w}"] = (
            f"ts_mean(gt(close, ts_delay(close, 1)), {w}) "
            f"- ts_mean(lt(close, ts_delay(close, 1)), {w})"
        )
        a[f"sump_{w}"] = (
            f"ts_sum(gt(close - ts_delay(close, 1), 0), {w}) "
            f"/ (ts_sum(abs(close - ts_delay(close, 1)), {w}) + 1e-12)"
        )
        a[f"sumn_{w}"] = (
            f"ts_sum(gt(ts_delay(close, 1) - close, 0), {w}) "
            f"/ (ts_sum(abs(close - ts_delay(close, 1)), {w}) + 1e-12)"
        )
        a[f"sumd_{w}"] = (
            f"(ts_sum(gt(close - ts_delay(close, 1), 0), {w}) "
            f"- ts_sum(gt(ts_delay(close, 1) - close, 0), {w})) "
            f"/ (ts_sum(abs(close - ts_delay(close, 1)), {w}) + 1e-12)"
        )
        a[f"vma_{w}"] = f"ts_mean(volume, {w}) / (volume + 1e-12)"
        a[f"vstd_{w}"] = f"ts_std(volume, {w}) / (volume + 1e-12)"
        a[f"wvma_{w}"] = (
            f"ts_std(abs(close / ts_delay(close, 1) - 1) * volume, {w}) "
            f"/ (ts_mean(abs(close / ts_delay(close, 1) - 1) * volume, {w}) + 1e-12)"
        )
        a[f"vsump_{w}"] = (
            f"ts_sum(gt(volume - ts_delay(volume, 1), 0), {w}) "
            f"/ (ts_sum(abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)"
        )
        a[f"vsumn_{w}"] = (
            f"ts_sum(gt(ts_delay(volume, 1) - volume, 0), {w}) "
            f"/ (ts_sum(abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)"
        )
        a[f"vsumd_{w}"] = (
            f"(ts_sum(gt(volume - ts_delay(volume, 1), 0), {w}) "
            f"- ts_sum(gt(ts_delay(volume, 1) - volume, 0), {w})) "
            f"/ (ts_sum(abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)"
        )

    return a


def main():
    alphas = build_alphas()
    print(f"Alphas defined: {len(alphas)}")

    lab = al.AlfarsLab.from_env()
    lab.with_filter("symbols not like '%BJ'")
    lab.with_years(2024, 2024)

    for name, expr in alphas.items():
        try:
            lab.register(name, expr)
        except Exception as e:
            print(f"  SKIP {name}: {e}")

    print(f"Registered {len(alphas)} factors, computing...")
    matrices, prices = lab.evaluate()
    valid_mats = [matrices[n] for n in matrices]
    print(f"Valid factors: {len(valid_mats)}")

    result = lab.run_multi(valid_mats, prices)

    print(f"\n── Backtest Results ({len(valid_mats)} factors) ──")
    print(f"  IC mean:   {result.ic_mean:.6f}")
    print(f"  IC IR:     {result.ic_ir:.4f}")
    print(f"  Total ret: {result.total_return:.4f}")
    print(f"  Sharpe:    {result.sharpe_ratio:.4f}")
    print(f"  Max DD:    {result.max_drawdown:.4f}")
    print(f"  Turnover:  {result.turnover:.4f}")


if __name__ == "__main__":
    main()
