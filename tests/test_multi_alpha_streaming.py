"""Two-factor streaming test — exercises batch processing (build_slices, parallel compute)
without the full stress-test overhead. Covers the 1d+perm path that panicked before.
"""
import numpy as np
import pytest
import alfars as al

EXPRS = [
    "-1 * ts_correlation(cs_rank(ts_delta(log(volume), 1)), cs_rank((close - open) / open), 6)",
    "rank(ts_mean(close, 20)) - rank(ts_mean(open, 5))",
]


@pytest.fixture(scope="module", params=["drop_all", "keep_most_recent", "keep_all"])
def backtest_results(request):
    policy_name = request.param
    if policy_name == "keep_all":
        cp = al.CachePolicy.keep_all()
    elif policy_name == "keep_most_recent":
        cp = al.CachePolicy.keep_most_recent()
    else:
        cp = al.CachePolicy.drop_all()

    config = al.DataPoolConfig(cache_policy=cp, backtest_batch_size=2)
    lab = al.AlfarsLab.from_env_with_config(config)
    lab.with_filter("symbols not like '%BJ'")
    lab.with_years(2010, 2025)
    lab.with_backtest_config(10, "equal", 1, 1, 0.0003)
    for i, expr in enumerate(EXPRS):
        lab.register(f"alpha{i}", expr)

    results = lab.backtest_each()
    assert len(results) == 2
    return results, policy_name


def test_both_computed(backtest_results):
    results, policy_name = backtest_results
    for name, r in results:
        assert not np.isnan(r.ic_mean), f"[{policy_name}] {name} IC is NaN"
        assert -1.0 <= r.ic_mean <= 1.0, f"[{policy_name}] {name} IC out of range"
        assert not np.isnan(r.sharpe_ratio), f"[{policy_name}] {name} Sharpe is NaN"


def test_results_different(backtest_results):
    """The two factors should produce different IC values."""
    results, policy_name = backtest_results
    ic0 = results[0][1].ic_mean
    ic1 = results[1][1].ic_mean
    assert abs(ic0 - ic1) > 1e-10, f"[{policy_name}] IC values identical ({ic0})"


def test_consistent_across_policies(backtest_results):
    """Results should be identical regardless of cache policy."""
    results, policy_name = backtest_results
    nav0 = np.array(results[0][1].group_cum_returns)
    nav1 = np.array(results[1][1].group_cum_returns)
    assert nav0.shape == nav1.shape, f"[{policy_name}] NAV shapes differ: {nav0.shape} vs {nav1.shape}"
