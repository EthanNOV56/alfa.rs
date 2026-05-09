"""Stress-test-like: 10 factors from example files, batch_size=5, verify correctness.

Matches the stress_test architecture exactly (same source files, naming, config)
but with only 10 factors and 1 year for quick verification.
"""

import csv
import os
import sys
import time
from datetime import datetime

import numpy as np

import alfars as al

# ── Config ──────────────────────────────────────────────────────────────
START_YEAR = 2010
END_YEAR = 2025
BACKTEST_BATCH_SIZE = (
    5  # factors per batch (5*16y*20MB ≈ 1.6GB slices, KeepAll DL ~2.4GB)
)

OUTDIR = f"/tmp/stress_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTDIR, exist_ok=True)
print(f"Output dir: {OUTDIR}")


# ── Helpers ─────────────────────────────────────────────────────────────
def rss_mb():
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return -1


def int_to_date(d: int) -> str:
    return f"{d // 10000:04d}-{(d % 10000) // 100:02d}-{d % 100:02d}"


timing_log = []


def log_phase(name, start_t):
    elapsed = time.perf_counter() - start_t
    rss = rss_mb()
    timing_log.append((name, elapsed, rss))
    print(f"  [{name}] {elapsed:.1f}s  RSS={rss:.0f}MB")


# ── Load alphas ─────────────────────────────────────────────────────────
print("Loading alphas...")
all_alphas = []
for prefix, path in [
    ("a101", "examples/alpha101_example.py"),
    ("a158", "examples/alpha158_example.py"),
    ("a191", "examples/alpha191_example.py"),
]:
    env = {}
    with open(path) as f:
        code = f.read().split("def main():")[0]
    exec(code, env)
    alphas = env["build_alphas"]()
    for name, expr in alphas.items():
        all_alphas.append((prefix, name, expr))
    print(f"  {path}: {len(alphas)} alphas")

all_alphas = all_alphas[:10]
print(f"Total: {len(all_alphas)} alphas, years={START_YEAR}-{END_YEAR}")

# ── Build lab ───────────────────────────────────────────────────────────
t_total = time.perf_counter()
t0 = t_total
print(f"\nSetup: batch={BACKTEST_BATCH_SIZE}, years={START_YEAR}-{END_YEAR}")

config = al.DataPoolConfig(
    cache_policy=al.CachePolicy.keep_all(),
    backtest_batch_size=BACKTEST_BATCH_SIZE,
    calc_parallel_years=5,
    memory_budget_bytes=0,
)
lab = al.AlfarsLab.from_env_with_config(config)
lab.set_pool("symbols not like '%BJ'")
lab.set_duration(START_YEAR, END_YEAR)
lab.set_backtest_config(10, "equal", 1, 1)

lab.set_exec_cfg({"buy_commission": 0.0005, "sell_commission": 0.0015})

log_phase("lab_setup", t0)

# ── Register ────────────────────────────────────────────────────────────
t0 = time.perf_counter()
registered = 0
register_fail = []
for prefix, name, expr in all_alphas:
    try:
        lab.register(f"{prefix}_{name}", expr)
        registered += 1
    except Exception as e:
        register_fail.append((f"{prefix}_{name}", str(e)[:80]))

print(f"Registered: {registered}, failures: {len(register_fail)}")
for pf, err in register_fail[:5]:
    print(f"  {pf}: {err}")
log_phase("register", t0)

# ── Compute + Backtest ──────────────────────────────────────────────────
print(f"\nStreaming backtest...")
t0 = time.perf_counter()

results = lab.backtest_each()

log_phase("backtest_each", t0)

# ── Dump factor values by evaluating separately ────────────────────────
# The streaming backtest doesn't keep factor matrices, so we re-evaluate
# in batches to dump values.
print(f"\nDumping factor values + NAV to {OUTDIR}...")
t0 = time.perf_counter()

# Open summary CSV
summary_path = os.path.join(OUTDIR, "summary.csv")
with open(summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "factor",
            "ic_mean",
            "ic_ir",
            "sharpe",
            "ann_ret",
            "max_dd",
            "turnover",
            "rss_mb",
        ]
    )

    for name, r in results:
        ic = r.ic_mean
        w.writerow(
            [
                name,
                f"{ic:.6f}",
                f"{r.ic_ir:.4f}",
                f"{r.sharpe_ratio:.4f}",
                f"{r.annualized_return:.4f}",
                f"{r.max_drawdown:.4f}",
                f"{r.turnover:.4f}",
                f"{rss_mb():.0f}",
            ]
        )

        # Dump NAV curves
        nav_path = os.path.join(OUTDIR, f"{name}_nav.csv")
        dates = [int_to_date(int(d)) for d in r.dates]
        ls_ret = np.array(r.long_short_returns)
        ls_nav = np.cumprod(1 + ls_ret)
        with open(nav_path, "w", newline="") as nf:
            nw = csv.writer(nf)
            nw.writerow(["date", "long_short_ret", "long_short_nav"])
            # returns start at day 1
            for i, d in enumerate(dates[1:]):
                if i < len(ls_ret):
                    nw.writerow([d, f"{ls_ret[i]:.8f}", f"{ls_nav[i]:.8f}"])

log_phase("dump", t0)

# calc() for full factor values would OOM with 416 factors × 16 years
# (FactorPanel holds all slices: ~300GB). Skip; NAV curves already saved.
log_phase("dump_done", t0)

# ── Report ──────────────────────────────────────────────────────────────
total = time.perf_counter() - t_total
print(f"\n{'=' * 60}")
print(
    f"Completed {len(results)}/{len(all_alphas)} factors in {total:.1f}s ({total / 60:.1f}min)"
)
print(f"Avg per factor: {total / len(results):.1f}s" if results else "No results")
print(f"Final RSS: {rss_mb():.0f} MB")
print(f"Output: {OUTDIR}/")

# Timing breakdown
print(f"\nPhase timing:")
for name, elapsed, rss in timing_log:
    print(f"  {name:<20s} {elapsed:8.1f}s  RSS={rss:.0f}MB")

# Score ranking
scored = []
for name, r in results:
    ic = r.ic_mean
    sharpe = r.sharpe_ratio
    scored.append(
        (name, abs(ic), sharpe, r.annualized_return, r.max_drawdown, r.turnover)
    )

scored.sort(key=lambda x: -x[1])

print(f"\nTop 20 by |IC|:")
print(
    f"{'Factor':<32s} {'|IC|':>8s} {'Sharpe':>8s} {'AnnRet':>8s} {'MaxDD':>8s} {'TO':>8s}"
)
print("-" * 72)
for name, ic, sharpe, ann_ret, mdd, turnover in scored[:20]:
    print(
        f"{name:<32s} {ic:8.4f} {sharpe:8.3f} {ann_ret:8.3f} {mdd:8.3f} {turnover:8.3f}"
    )

ics = [abs(r.ic_mean) for _, r in results]
sharpes = [r.sharpe_ratio for _, r in results if np.isfinite(r.sharpe_ratio)]
print(
    f"\n|IC| stats:  mean={np.mean(ics):.4f}  median={np.median(ics):.4f}  max={np.max(ics):.4f}"
)
print(
    f"Sharpe stats: mean={np.mean(sharpes):.4f}  median={np.median(sharpes):.4f}  "
    f"max={np.max(sharpes):.4f}  >1:{sum(1 for s in sharpes if s > 1)}"
)

# Write ranking to file
rank_path = os.path.join(OUTDIR, "ranking.csv")
with open(rank_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "factor", "abs_ic", "sharpe", "ann_ret", "max_dd", "turnover"])
    for rank, (name, ic, sharpe, ann_ret, mdd, turnover) in enumerate(scored, 1):
        w.writerow(
            [
                rank,
                name,
                f"{ic:.6f}",
                f"{sharpe:.4f}",
                f"{ann_ret:.4f}",
                f"{mdd:.4f}",
                f"{turnover:.4f}",
            ]
        )

print(f"\nRanking: {rank_path}")
print(f"Summary: {summary_path}")
print(f"DONE — results in {OUTDIR}/")
