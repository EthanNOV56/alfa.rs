#!/usr/bin/env python3
"""
GP Engine Benchmark — Performance and feature demonstration using real market data.

Reads OHLCV data via ClickHouseSource → DataLayer → PriceMatrix, then runs
genetic programming across multiple configurations and compares results.

Usage:
  python examples/gp_benchmark.py              # quick run (1 year, ~50 stocks)
  python examples/gp_benchmark.py --years 3     # 3 years of data
  python examples/gp_benchmark.py --full        # full benchmark suite
  python examples/gp_benchmark.py --profile     # single config with verbose output
"""

import argparse
import time
import sys

import numpy as np
import alfars as al


# ── Data Loading ──────────────────────────────────────────────────────────

def load_market_data(years: int = 1, max_symbols: int = 50):
    """Load real OHLCV data from ClickHouse via DataLayer.

    Returns (data_dict, returns_array) in the format GpEngine expects.
    """
    source = al.ClickHouseSource.from_env()
    layer = al.DataLayer(source)

    # Filter: date range + exclude new listings
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    date_range = f"{start.strftime('%Y-%m-%d')}:{end.strftime('%Y-%m-%d')}"

    layer.set_pre_filter(
        f"{date_range} symbols not like '%BJ' symbols not like '%688%'"
    )

    pm = layer.query_price_matrix()

    symbols = pm.symbols
    if len(symbols) > max_symbols:
        # Pick most liquid symbols (by average volume proxy)
        symbols = symbols[:max_symbols]

    # Build column data dict from PriceMatrix
    data = {
        "close": np.asarray(pm.close),
        "open": np.asarray(pm.open),
        "high": np.asarray(pm.high),
        "low": np.asarray(pm.low),
        "vwap": np.asarray(pm.vwap),
    }
    returns = np.asarray(pm.returns)

    n_days, n_assets = returns.shape
    print(f"  Loaded: {n_days} days × {n_assets} assets "
          f"({years}y, {len(pm.dates)} dates, {len(pm.symbols)} symbols)")
    return data, returns


# ── Configuration Presets ─────────────────────────────────────────────────

def quick_config():
    """Minimal config for fast correctness check."""
    return dict(
        population_size=20, max_generations=5, tournament_size=3,
        crossover_prob=0.8, mutation_prob=0.1, max_depth=4,
        parent_diversity_penalty=0.0, use_diverse_init=False,
        smart_mutation_ratio=0.0,
    )


def enhanced_config():
    return dict(
        population_size=20, max_generations=5, tournament_size=3,
        crossover_prob=0.8, mutation_prob=0.1, max_depth=4,
        parent_diversity_penalty=0.1, use_diverse_init=True,
        smart_mutation_ratio=0.3,
    )


# ── Benchmark Runner ──────────────────────────────────────────────────────

def run_config(name, config, data, returns, num_factors=2):
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")
    for k, v in config.items():
        print(f"    {k}: {v}")

    gp = al.GpEngine(**config)
    gp.set_columns(list(data.keys()))

    t0 = time.perf_counter()
    results = gp.mine_factors(data, returns, num_factors=num_factors)
    elapsed = time.perf_counter() - t0

    print(f"\n  Time: {elapsed:.3f}s ({elapsed/num_factors:.3f}s/factor)")
    header = (f"  {'Expr':<45} {'Fit':>6} {'IC':>6} {'IR':>6} {'Cmplx':>5}"
              f"  {'T-IC':>6} {'V-IC':>6} {'E-IC':>6}")
    print(header)
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*5}  {'-'*6} {'-'*6} {'-'*6}")

    stats = {"name": name, "time_total": elapsed,
             "time_per_factor": elapsed / num_factors}
    fits, ics, irs, cmplxs = [], [], [], []
    for r in results:
        expr_str, fitness, ic, ir, turnover, complexity = r[0:6]
        train_m, val_m, test_m = r[6], r[7], r[8]
        short = expr_str[:42] + "..." if len(expr_str) > 45 else expr_str
        print(f"  {short:<45} {fitness:>6.3f} {ic:>6.4f} {ir:>6.4f} "
              f"{complexity:>5}  {train_m[0]:>6.4f} {val_m[0]:>6.4f} {test_m[0]:>6.4f}")
        fits.append(fitness); ics.append(ic); irs.append(ir); cmplxs.append(complexity)

    stats.update(fitness_mean=float(np.mean(fits)),
                 fitness_best=float(max(fits)),
                 ic_mean=float(np.mean(ics)),
                 ir_mean=float(np.mean(irs)),
                 complexity_mean=float(np.mean(cmplxs)),
                 expressions=[r[0] for r in results])
    return stats


# ── Feature Demos ─────────────────────────────────────────────────────────

def demo_structural_similarity():
    print(f"\n{'='*60}")
    print(f"  Structural Similarity (MCIS)")
    print(f"{'='*60}")

    pairs = [
        ("rank(close / 5)", "rank(close / 10)"),
        ("ts_mean(close, 20)", "ts_mean(close, 60)"),
        ("rank(close)", "ts_mean(volume, 20)"),
        ("close / open", "(close - open) / close"),
    ]
    for a, b in pairs:
        sim = al.expr_similarity(a, b)
        print(f"  sim({a}, {b}) = {sim:.4f}")


def demo_factor_pool(data, returns):
    print(f"\n{'='*60}")
    print(f"  FactorPool — Redundancy Filtering")
    print(f"{'='*60}")

    pool = al.FactorPool(max_size=5)
    gp = al.GpEngine(**quick_config())
    gp.set_columns(list(data.keys()))

    admitted = rejected_dup = flagged = rejected_min = 0
    for batch in range(1):
        results = gp.mine_factors(data, returns, num_factors=2)
        for r in results:
            expr_str, fitness, ic, ir, _, _ = r[0:6]
            rank_ic = ic * ir
            status, similarity = pool.try_admit(expr_str, ic, rank_ic)
            if status == "added":           admitted += 1
            elif status == "rejected_duplicate": rejected_dup += 1
            elif status == "flagged":        flagged += 1
            elif status == "rejected_below_min": rejected_min += 1

    print(f"  Pool size: {pool.entry_count()}/10")
    print(f"  Added: {admitted}, Flagged: {flagged}, "
          f"Rejected(dup): {rejected_dup}, Rejected(min): {rejected_min}")

    print(f"\n  Final pool:")
    for i in range(pool.entry_count()):
        expr, ic, rk_ic, _, _, surv = pool.get_entry(i)
        short = expr[:65] + "..." if len(expr) > 65 else expr
        print(f"    [{i}] IC={ic:.4f} RankIC={rk_ic:.4f} surv={surv}  {short}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GP Engine Benchmark")
    parser.add_argument("--years", type=int, default=1,
                        help="Years of data to load (default: 1)")
    parser.add_argument("--full", action="store_true",
                        help="Full benchmark (3 years, more factors)")
    parser.add_argument("--profile", action="store_true",
                        help="Single config with verbose output")
    parser.add_argument("--max-symbols", type=int, default=20,
                        help="Max symbols to load (default: 20)")
    args = parser.parse_args()

    years = args.years
    max_sym = min(args.max_symbols, 30)
    num_factors = 2

    print(f"GP Engine Benchmark")
    print(f"  Years: {years}, Max symbols: {max_sym}, "
          f"Factors per config: {num_factors}")

    try:
        data, returns = load_market_data(years, max_sym)
    except Exception as e:
        print(f"\n  ClickHouse unavailable: {e}")
        sys.exit(1)

    # Warmup
    gp = al.GpEngine(**quick_config())
    gp.set_columns(list(data.keys()))
    _ = gp.mine_factors(data, returns, num_factors=1)

    # Run 2 configs
    configs = [("quick", quick_config()), ("enhanced", enhanced_config())]
    if args.profile:
        configs = [("enhanced", enhanced_config())]

    all_stats = []
    for name, cfg in configs:
        all_stats.append(run_config(name, cfg, data, returns, num_factors))

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary ({years}y data, {num_factors} factors each)")
    print(f"{'='*60}")
    hdr = f"  {'Config':<20} {'Fitness':>8} {'Best':>8} {'IC':>8} {'Cmplx':>6} {'Time/f':>8}"
    print(hdr)
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    for s in all_stats:
        print(f"  {s['name']:<20} {s['fitness_mean']:>8.4f} {s['fitness_best']:>8.4f} "
              f"{s['ic_mean']:>8.4f} {s['complexity_mean']:>6.1f} "
              f"{s['time_per_factor']:>7.3f}s")

    print(f"\n  Best expressions:")
    for s in all_stats:
        for expr in s["expressions"][:2]:
            print(f"    [{s['name']}] {expr}")

    # Feature demos
    demo_structural_similarity()
    demo_factor_pool(data, returns)

    print(f"\n{'='*60}")
    print(f"  Done. --full for larger benchmark, --profile for verbose.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
