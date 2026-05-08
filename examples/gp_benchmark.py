#!/usr/bin/env python3
"""
GP Engine Benchmark — demonstrate factor mining via AlfarsLab.

Usage:
  python examples/gp_benchmark.py              # quick run (1 year)
  python examples/gp_benchmark.py --years 3     # 3 years of data
  python examples/gp_benchmark.py --profile     # single config with verbose output
"""

import argparse
import sys
import time

import numpy as np

import alfars as al

# ── Configuration Presets ─────────────────────────────────────────────────


def quick_config():
    return dict(
        population_size=50,
        max_generations=10,
        tournament_size=5,
        crossover_prob=0.8,
        mutation_prob=0.15,
        max_depth=5,
        use_diverse_init=True,
        smart_mutation_ratio=0.3,
    )


def enhanced_config():
    return dict(
        population_size=100,
        max_generations=30,
        tournament_size=7,
        crossover_prob=0.8,
        mutation_prob=0.2,
        max_depth=6,
        use_diverse_init=True,
        smart_mutation_ratio=0.3,
    )


# ── Benchmark Runner ──────────────────────────────────────────────────────


def run_config(name, config, lab, num_factors=3, max_symbols=0):
    print(f"\n{'=' * 60}")
    print(f"  Config: {name}")
    print(f"{'=' * 60}")
    for k, v in config.items():
        print(f"    {k}: {v}")

    t0 = time.perf_counter()
    results = lab.mine_factors(num_factors=num_factors, max_symbols=max_symbols, **config)
    elapsed = time.perf_counter() - t0

    print(f"\n  Time: {elapsed:.3f}s ({elapsed / num_factors:.3f}s/factor)")
    header = f"  {'Expr':<45} {'Fit':>6} {'IC':>6} {'IR':>6} {'Cmplx':>5}"
    print(header)
    print(f"  {'-' * 45} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 5}")

    fits, ics, irs, cmplxs = [], [], [], []
    for r in results:
        expr_str, fitness, ic, ir, turnover, complexity = r
        short = expr_str[:42] + "..." if len(expr_str) > 45 else expr_str
        print(f"  {short:<45} {fitness:>6.3f} {ic:>6.4f} {ir:>6.4f} {complexity:>5}")
        fits.append(fitness)
        ics.append(ic)
        irs.append(ir)
        cmplxs.append(complexity)

    stats = dict(
        name=name,
        time_total=elapsed,
        time_per_factor=elapsed / num_factors,
        fitness_mean=float(np.mean(fits)),
        fitness_best=float(max(fits)),
        ic_mean=float(np.mean(ics)),
        ir_mean=float(np.mean(irs)),
        complexity_mean=float(np.mean(cmplxs)),
        expressions=[r[0] for r in results],
    )
    return stats


# ── Feature Demos ─────────────────────────────────────────────────────────


def demo_structural_similarity():
    print(f"\n{'=' * 60}")
    print(f"  Structural Similarity (MCIS)")
    print(f"{'=' * 60}")

    pairs = [
        ("rank(close / 5)", "rank(close / 10)"),
        ("ts_mean(close, 20)", "ts_mean(close, 60)"),
        ("rank(close)", "ts_mean(volume, 20)"),
        ("close / open", "(close - open) / close"),
    ]
    for a, b in pairs:
        sim = al.expr_similarity(a, b)
        print(f"  sim({a}, {b}) = {sim:.4f}")


def demo_factor_pool(lab, max_symbols=0):
    print(f"\n{'=' * 60}")
    print(f"  FactorPool — Redundancy Filtering")
    print(f"{'=' * 60}")

    pool = al.FactorPool(max_size=5)

    results = lab.mine_factors(num_factors=3, max_symbols=max_symbols, **quick_config())
    admitted = rejected_dup = flagged = rejected_min = 0
    for r in results:
        expr_str, fitness, ic, ir, _, _ = r
        rank_ic = ic * ir
        status, similarity = pool.try_admit(expr_str, ic, rank_ic)
        if status == "added":
            admitted += 1
        elif status == "rejected_duplicate":
            rejected_dup += 1
        elif status == "flagged":
            flagged += 1
        elif status == "rejected_below_min":
            rejected_min += 1

    print(f"  Pool size: {pool.entry_count()}/5")
    print(
        f"  Added: {admitted}, Flagged: {flagged}, "
        f"Rejected(dup): {rejected_dup}, Rejected(min): {rejected_min}"
    )

    print(f"\n  Final pool:")
    for i in range(pool.entry_count()):
        expr, ic, rk_ic, _, _, surv = pool.get_entry(i)
        short = expr[:65] + "..." if len(expr) > 65 else expr
        print(f"    [{i}] IC={ic:.4f} RankIC={rk_ic:.4f} surv={surv}  {short}")


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="GP Engine Benchmark")
    parser.add_argument(
        "--years", type=int, default=1, help="Years of data to load (default: 1)"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Single config with verbose output"
    )
    parser.add_argument(
        "--max-symbols", type=int, default=100, help="Max symbols to load (default: 100, 0=all)"
    )
    args = parser.parse_args()

    years = args.years
    num_factors = 3
    max_symbols = args.max_symbols

    print(f"GP Engine Benchmark via AlfarsLab")
    print(f"  Years: {years}, Factors per config: {num_factors}")

    from datetime import datetime

    this_year = 2025
    lab = al.AlfarsLab.from_env()
    filter = "symbols not like '%BJ'"
    lab.with_filter(filter)
    lab.with_years(this_year - years, this_year)
    lab.with_backtest_config(10, "equal", 1, 1, 0.0005, 0.0015)

    print(f"  Filter: {this_year - years}–{this_year}, {filter}, max_symbols={max_symbols or 'all'}")

    # Warmup
    _ = lab.mine_factors(num_factors=1, max_symbols=max_symbols, **quick_config())

    # Run 2 configs
    configs = [("quick", quick_config()), ("enhanced", enhanced_config())]
    if args.profile:
        configs = [("enhanced", enhanced_config())]

    all_stats = []
    for name, cfg in configs:
        all_stats.append(run_config(name, cfg, lab, num_factors, max_symbols))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Summary ({years}y data, {num_factors} factors each)")
    print(f"{'=' * 60}")
    hdr = f"  {'Config':<20} {'Fitness':>8} {'Best':>8} {'IC':>8} {'Cmplx':>6} {'Time/f':>8}"
    print(hdr)
    print(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 8}")
    for s in all_stats:
        print(
            f"  {s['name']:<20} {s['fitness_mean']:>8.4f} {s['fitness_best']:>8.4f} "
            f"{s['ic_mean']:>8.4f} {s['complexity_mean']:>6.1f} "
            f"{s['time_per_factor']:>7.3f}s"
        )

    print(f"\n  Best expressions:")
    for s in all_stats:
        for expr in s["expressions"][:2]:
            print(f"    [{s['name']}] {expr}")

    # Feature demos
    demo_structural_similarity()
    demo_factor_pool(lab, max_symbols)

    print(f"\n{'=' * 60}")
    print(f"  Done. --profile for verbose output.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
