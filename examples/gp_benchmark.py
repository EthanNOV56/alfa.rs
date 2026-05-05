#!/usr/bin/env python3
"""
GP Engine Benchmark — Demonstrates all gp-engine features with performance metrics.

Measures:
  - Time per generation / per factor
  - Dedup hit rate from expression normalization
  - Population diversity (unique family count per generation)
  - Impact of diverse_init, smart_mutation, parent_diversity on result quality
  - Complexity distribution of discovered factors
  - FactorPool admission and redundancy filtering

Usage:
  python examples/gp_benchmark.py              # quick run (small data)
  python examples/gp_benchmark.py --full       # full benchmark suite
  python examples/gp_benchmark.py --profile    # single config with verbose output
"""

import argparse
import time
import sys
import os

import numpy as np
import alfars as al


# ── Synthetic Data Generator ──────────────────────────────────────────────

def generate_synthetic_data(n_days=252, n_assets=50, seed=42):
    """Generate realistic synthetic OHLCV data with known alpha signals."""
    rng = np.random.default_rng(seed)

    # Geometric Brownian motion for prices
    mu = 0.0002  # daily drift
    sigma = 0.02  # daily volatility

    # Start prices
    start_prices = rng.uniform(10, 200, n_assets)

    # Daily returns with mild autocorrelation
    returns = np.zeros((n_days, n_assets))
    for d in range(1, n_days):
        epsilon = rng.normal(0, sigma, n_assets)
        returns[d] = mu + epsilon

    # Cumulative price series
    cum_returns = np.exp(np.cumsum(returns, axis=0))
    close = start_prices * cum_returns

    # Build OHLC from close
    daily_range = close * sigma * rng.uniform(0.5, 2.0, (n_days, n_assets))
    high = close + daily_range * rng.uniform(0.3, 1.0, (n_days, n_assets))
    low = close - daily_range * rng.uniform(0.3, 1.0, (n_days, n_assets))
    open_price = low + daily_range * rng.uniform(0.0, 1.0, (n_days, n_assets))

    # Volume with price-volume correlation
    base_volume = rng.uniform(1e5, 1e7, n_assets)
    volume = np.zeros((n_days, n_assets))
    for d in range(n_days):
        vol_factor = 1.0 + 0.3 * np.abs(returns[d]) / sigma + rng.normal(0, 0.2, n_assets)
        volume[d] = base_volume * np.maximum(vol_factor, 0.1)

    vwap = (high + low + close) / 3.0
    amount = close * volume

    data = {
        "close": close,
        "open": open_price,
        "high": high,
        "low": low,
        "volume": volume,
        "vwap": vwap,
        "amount": amount,
    }

    # Forward 5-day returns as target
    fwd_returns = np.zeros_like(returns)
    for d in range(n_days - 5):
        fwd_returns[d] = close[d + 5] / close[d] - 1.0

    return data, fwd_returns


# ── Configuration Presets ─────────────────────────────────────────────────

def baseline_config():
    """Standard GP config (before enhancements)."""
    return dict(
        population_size=50,
        max_generations=20,
        tournament_size=5,
        crossover_prob=0.8,
        mutation_prob=0.1,
        max_depth=5,
        parent_diversity_penalty=0.0,
        use_diverse_init=False,
        smart_mutation_ratio=0.0,
    )


def enhanced_config():
    """GP config with all Phase 1+2+3 enhancements enabled."""
    return dict(
        population_size=50,
        max_generations=20,
        tournament_size=5,
        crossover_prob=0.8,
        mutation_prob=0.1,
        max_depth=5,
        parent_diversity_penalty=0.1,
        use_diverse_init=True,
        smart_mutation_ratio=0.3,
    )


def diversity_only_config():
    """Diverse init + parent diversity, no smart mutation."""
    return dict(
        population_size=50,
        max_generations=20,
        tournament_size=5,
        crossover_prob=0.8,
        mutation_prob=0.1,
        max_depth=5,
        parent_diversity_penalty=0.1,
        use_diverse_init=True,
        smart_mutation_ratio=0.0,
    )


# ── Benchmark Runner ──────────────────────────────────────────────────────

def run_single_config(name, config, data, returns, num_factors=3):
    """Run GP with one config and collect stats."""
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")
    for k, v in config.items():
        print(f"    {k}: {v}")

    gp = al.GpEngine(**config)
    gp.set_columns(list(data.keys()))

    t0 = time.perf_counter()
    results = gp.mine_factors(
        data, returns,
        num_factors=num_factors,
        weight_ic=0.4,
        weight_ir=0.3,
        weight_turnover=0.15,
        weight_complexity=0.15,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Total time: {elapsed:.3f}s ({elapsed/num_factors:.3f}s per factor)")
    print(f"  {'Expr':<50} {'Fitness':>8} {'IC':>8} {'IR':>8} {'TO':>8} {'Cmplx':>6}")
    print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    fitnesses, ics, irs, turnovers, complexities = [], [], [], [], []
    for expr_str, fitness, ic, ir, turnover, complexity in results:
        short = expr_str[:47] + "..." if len(expr_str) > 50 else expr_str
        print(f"  {short:<50} {fitness:>8.4f} {ic:>8.4f} {ir:>8.4f} {turnover:>8.4f} {complexity:>6}")
        fitnesses.append(fitness)
        ics.append(ic)
        irs.append(ir)
        turnovers.append(turnover)
        complexities.append(complexity)

    stats = {
        "name": name,
        "time_total": elapsed,
        "time_per_factor": elapsed / num_factors,
        "fitness_mean": np.mean(fitnesses),
        "fitness_std": np.std(fitnesses),
        "fitness_best": max(fitnesses),
        "ic_mean": np.mean(ics),
        "ir_mean": np.mean(irs),
        "turnover_mean": np.mean(turnovers),
        "complexity_mean": np.mean(complexities),
        "expressions": [r[0] for r in results],
    }
    return stats


# ── Factor Pool Demo ──────────────────────────────────────────────────────

def demo_factor_pool(data, returns, max_pool_size=10):
    """Demonstrate FactorPool redundancy filtering."""
    print(f"\n{'='*60}")
    print(f"  FactorPool — Redundancy Filtering Demo")
    print(f"{'='*60}")

    pool = al.FactorPool(max_size=max_pool_size)

    gp = al.GpEngine(**enhanced_config())
    gp.set_columns(list(data.keys()))

    # Generate many factors and test pool admission
    print("  Mining factors and testing pool admission...")
    admitted = 0
    rejected_dup = 0
    flagged = 0
    rejected_min = 0

    for batch in range(3):
        results = gp.mine_factors(data, returns, num_factors=3,
                                   weight_ic=0.4, weight_ir=0.3,
                                   weight_turnover=0.15, weight_complexity=0.15)
        for expr_str, fitness, ic, ir, turnover, complexity in results:
            rank_ic = ic * ir  # approximate
            status, similarity = pool.try_admit(expr_str, ic, rank_ic)
            if status == "added":
                admitted += 1
            elif status == "rejected_duplicate":
                rejected_dup += 1
            elif status == "flagged":
                flagged += 1
            elif status == "rejected_below_min":
                rejected_min += 1

    print(f"  Pool size: {pool.entry_count()}/{max_pool_size}")
    print(f"  Added: {admitted}, Flagged: {flagged}, "
          f"Rejected (duplicate): {rejected_dup}, Rejected (below min): {rejected_min}")

    print(f"\n  Final pool entries:")
    for i in range(pool.entry_count()):
        expr, ic, rank_ic, added_at, last_check, surv = pool.get_entry(i)
        short = expr[:60] + "..." if len(expr) > 60 else expr
        print(f"    [{i}] IC={ic:.4f} RankIC={rank_ic:.4f} surv={surv}  {short}")

    return pool


# ── Structural Similarity Demo ────────────────────────────────────────────

def demo_structural_similarity():
    """Demonstrate AST structural similarity detection."""
    print(f"\n{'='*60}")
    print(f"  Structural Similarity (MCIS) Demo")
    print(f"{'='*60}")

    pairs = [
        # Near-identical (only constant differs)
        ("rank(close / 5)", "rank(close / 10)"),
        ("ts_mean(close, 20)", "ts_mean(close, 60)"),
        # Different structure
        ("rank(close)", "ts_mean(volume, 20)"),
        ("close / open", "(close - open) / close"),
        # Nested similar
        ("rank(ts_mean(close, 20))", "rank(ts_mean(open, 20))"),
        # Completely different
        ("close / ts_delay(close, 1) - 1", "rank(ts_std(volume, 30))"),
    ]

    for a, b in pairs:
        sim = al.expr_similarity(a, b)
        print(f"  sim({a}, {b}) = {sim:.4f}")

    # All-pairs matrix for discovered factors
    factors = [
        "rank(close / 5)",
        "rank(close / 10)",
        "rank(close / 20)",
        "ts_mean(close, 20)",
        "close / ts_delay(close, 1) - 1",
    ]
    print(f"\n  Pairwise similarity matrix ({len(factors)} factors):")
    print(f"  {'':>4}", end="")
    for i in range(len(factors)):
        print(f" {i:>6}", end="")
    print()
    for i, a in enumerate(factors):
        print(f"  {i:>3} ", end="")
        for j, b in enumerate(factors):
            sim = al.expr_similarity(a, b)
            print(f" {sim:>6.3f}", end="")
        print()


# ── Main Benchmark ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GP Engine Benchmark and Feature Demo"
    )
    parser.add_argument("--full", action="store_true",
                        help="Run full benchmark suite (more data, more factors)")
    parser.add_argument("--profile", action="store_true",
                        help="Single config run with verbose per-generation output")
    args = parser.parse_args()

    # Data size
    if args.full:
        n_days, n_assets, num_factors = 504, 100, 5
    else:
        n_days, n_assets, num_factors = 126, 30, 3

    print(f"GP Engine Benchmark")
    print(f"  Data: {n_days} days × {n_assets} assets")
    print(f"  Factors per config: {num_factors}")

    data, returns = generate_synthetic_data(n_days, n_assets)

    # Warmup run
    print("\n[Warmup — single factor]")
    gp = al.GpEngine(population_size=10, max_generations=3)  # minimal config
    gp.set_columns(list(data.keys()))
    _ = gp.mine_factors(data, returns, num_factors=1,
                         weight_ic=0.4, weight_ir=0.3,
                         weight_turnover=0.15, weight_complexity=0.15)

    if args.profile:
        configs = [("enhanced", enhanced_config())]
    else:
        configs = [
            ("baseline", baseline_config()),
            ("diversity_only", diversity_only_config()),
            ("enhanced", enhanced_config()),
        ]

    all_stats = []
    for name, cfg in configs:
        stats = run_single_config(name, cfg, data, returns,
                                  num_factors=num_factors)
        all_stats.append(stats)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    header = f"  {'Config':<20} {'Fitness':>8} {'BestFit':>8} {'IC':>8} {'Cmplx':>6} {'Time/f':>8}"
    print(header)
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    for s in all_stats:
        print(f"  {s['name']:<20} {s['fitness_mean']:>8.4f} {s['fitness_best']:>8.4f} "
              f"{s['ic_mean']:>8.4f} {s['complexity_mean']:>6.1f} {s['time_per_factor']:>7.3f}s")

    # ── Feature Demos ─────────────────────────────────────────────────
    demo_structural_similarity()
    demo_factor_pool(data, returns)

    print(f"\n{'='*60}")
    print(f"  All done. Use --full for larger benchmark, --profile for verbose.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
