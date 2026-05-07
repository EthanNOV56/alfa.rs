"""Verify a single 1d alpha factor works via streaming architecture.

Alpha factors use bare (1d-default) column names, which exercise the flat
evaluation path with perm — the path that was panicking before fixes.
"""
import numpy as np
import pytest
import alfars as al

EXPR = "-1 * ts_correlation(cs_rank(ts_delta(log(volume), 1)), cs_rank((close - open) / open), 6)"


@pytest.fixture(scope="module", params=["drop_all", "keep_most_recent", "keep_all"])
def backtest_result(request):
    policy_name = request.param
    if policy_name == "keep_all":
        cp = al.CachePolicy.keep_all()
    elif policy_name == "keep_most_recent":
        cp = al.CachePolicy.keep_most_recent()
    else:
        cp = al.CachePolicy.drop_all()

    config = al.DataPoolConfig(cache_policy=cp, backtest_batch_size=1)
    lab = al.AlfarsLab.from_env_with_config(config)
    lab.with_filter("symbols not like '%BJ'")
    lab.with_years(2010, 2025)
    lab.with_backtest_config(10, "equal", 1, 1, 0.0003)
    lab.register("test_alpha", EXPR)

    results = lab.backtest_each()
    assert len(results) == 1
    name, result = results[0]
    assert name == "test_alpha"
    return result, policy_name


def test_no_panic(backtest_result):
    """Factor computed and backtest completed without panic."""
    result, policy_name = backtest_result
    assert not np.isnan(result.ic_mean), f"[{policy_name}] IC is NaN"
    assert -1.0 <= result.ic_mean <= 1.0, f"[{policy_name}] IC out of range"
    assert not np.isnan(result.sharpe_ratio), f"[{policy_name}] Sharpe is NaN"


def test_factor_matrix_valid(backtest_result):
    """NAV and returns are finite."""
    result, policy_name = backtest_result
    ls = np.array(result.long_short_returns)
    assert np.isfinite(ls).sum() > 0, f"[{policy_name}] all LS returns non-finite"
    nav = np.array(result.group_cum_returns)
    assert nav.size > 0, f"[{policy_name}] empty NAV"
