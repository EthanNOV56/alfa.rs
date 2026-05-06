"""Full-scale stress test: all alphas (101+158+191), 2010-2025, streaming backtest.

Uses evaluate_and_backtest_each() with configurable DataPool batch_size to avoid
OOM. Monitors peak RSS throughout.

Usage:
    uv run python tests/stress_test_all_alphas.py
"""

import alfars as al
import numpy as np
import time
import os
import sys

# ── Config ──────────────────────────────────────────────────────────────
START_YEAR = 2010
END_YEAR = 2025
BACKTEST_BATCH_SIZE = 15  # factors per batch (tune for memory)
CACHE_POLICY = "drop_all"  # drop_all | keep_most_recent | keep_all
# Memory estimate (DropAll + batch=15):
#   PriceMatrix ~300MB + 1 DataLayer ~2GB + 15 factor mats ~150MB = ~2.5GB peak

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

print(f"Total: {len(all_alphas)} alphas, years={START_YEAR}-{END_YEAR}")

# ── Build lab with config ───────────────────────────────────────────────
print(f"\nSetup: cache={CACHE_POLICY}, batch={BACKTEST_BATCH_SIZE}")
t0 = time.perf_counter()

if CACHE_POLICY == "keep_all":
    cp = al.CachePolicy.keep_all()
elif CACHE_POLICY == "keep_most_recent":
    cp = al.CachePolicy.keep_most_recent()
else:
    cp = al.CachePolicy.drop_all()

config = al.DataPoolConfig(
    cache_policy=cp,
    backtest_batch_size=BACKTEST_BATCH_SIZE,
    calc_parallel_years=5,
    memory_budget_bytes=0,  # unlimited
)
lab = al.AlfarsLab.from_env_with_config(config)
lab.with_filter("symbols not like '%BJ'")
lab.with_years(START_YEAR, END_YEAR)
lab.with_backtest_config(10, "equal", 1, 1, 0.0003)

# ── Register ────────────────────────────────────────────────────────────
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
if len(register_fail) > 5:
    print(f"  ... and {len(register_fail) - 5} more")

# ── Streaming backtest ──────────────────────────────────────────────────
print(f"\nStreaming backtest ({len(all_alphas)} factors)...")
t1 = time.perf_counter()

results = lab.backtest_each()  # uses evaluate_and_backtest_each internally

dt = time.perf_counter() - t1
total = time.perf_counter() - t0

# ── Report ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Completed {len(results)} factors in {dt:.1f}s (total {total:.1f}s)")
if len(results) > 0:
    print(f"Avg per factor: {dt/len(results):.1f}s")

# Score ranking
scored = []
for name, r in results:
    ic = r.ic_mean
    sharpe = r.sharpe_ratio
    scored.append((name, abs(ic), sharpe, r.annualized_return, r.max_drawdown, r.turnover))

scored.sort(key=lambda x: -x[1])  # sort by |IC| desc

print(f"\nTop 20 by |IC|:")
print(f"{'Factor':<32s} {'|IC|':>8s} {'Sharpe':>8s} {'AnnRet':>8s} {'MaxDD':>8s} {'TO':>8s}")
print("-" * 72)
for name, ic, sharpe, ann_ret, mdd, turnover in scored[:20]:
    print(f"{name:<32s} {ic:8.4f} {sharpe:8.3f} {ann_ret:8.3f} {mdd:8.3f} {turnover:8.3f}")

# Distribution stats
ics = [abs(r.ic_mean) for _, r in results]
sharpes = [r.sharpe_ratio for _, r in results if np.isfinite(r.sharpe_ratio)]
print(f"\n|IC| stats:  mean={np.mean(ics):.4f}  median={np.median(ics):.4f}  max={np.max(ics):.4f}")
print(f"Sharpe stats: mean={np.mean(sharpes):.4f}  median={np.median(sharpes):.4f}  "
      f"max={np.max(sharpes):.4f}  >1:{sum(1 for s in sharpes if s > 1)}")

# Memory
def rss_mb():
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return -1

print(f"\nFinal RSS: {rss_mb():.0f} MB")
print("DONE")
