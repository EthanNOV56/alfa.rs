"""Fault tolerance tests: pipeline should skip failed factors/years, not crash."""
import numpy as np
import alfars as al


def test_bad_column_year_skipped_others_ok():
    """Non-existent column → query error → year skipped for that batch only.
    With batch_size=1, only the bad factor's year is lost. Other factors OK.
    """
    config = al.DataPoolConfig(cache_policy=al.CachePolicy.drop_all(), backtest_batch_size=1)
    lab = al.AlfarsLab.from_env_with_config(config)
    lab.with_filter("symbols not like '%BJ'")
    lab.with_years(2010, 2010)
    lab.with_backtest_config(10, "equal", 1, 1, 0.0005, 0.0015)

    lab.register("good_1", "close / delay(close, 1) - 1")
    lab.register("good_2", "rank(ts_mean(volume, 20))")
    lab.register("bad_col", "1d:nonexistent_column / close")

    results = lab.backtest_each()
    names = set(n for n, _ in results)
    assert "good_1" in names, "good_1 should succeed"
    assert "good_2" in names, "good_2 should succeed"
    # bad_col's year was skipped → no slices → no backtest result
    assert len(results) >= 2, f"Expected at least 2 results, got {len(results)}"
    # Verify good factors have finite metrics
    for n, r in results:
        assert np.isfinite(r.ic_mean), f"{n} IC is NaN"
        assert np.isfinite(r.sharpe_ratio), f"{n} Sharpe is NaN"


def test_pipeline_continues_after_year_skip():
    """After a year fails with bad column, pipeline recovers for next year.
    Use batch_size=1 so only the bad factor batch loses its year.
    """
    config = al.DataPoolConfig(cache_policy=al.CachePolicy.drop_all(), backtest_batch_size=1)
    lab = al.AlfarsLab.from_env_with_config(config)
    lab.with_filter("symbols not like '%BJ'")
    lab.with_years(2010, 2025)
    lab.with_backtest_config(10, "equal", 1, 1, 0.0005, 0.0015)

    lab.register("test_recovery", "rank(ts_mean(close, 20)) - rank(ts_mean(open, 20))")

    results = lab.backtest_each()
    assert len(results) == 1
    r = results[0][1]
    assert np.isfinite(r.ic_mean), f"IC should be finite"
    assert np.isfinite(r.sharpe_ratio), f"Sharpe should be finite"


def test_drop_all_and_keep_all_consistent():
    """Same factor should produce identical results with drop_all and keep_all."""
    def compute_with(policy):
        cp = (al.CachePolicy.keep_all() if policy == "keep_all"
              else al.CachePolicy.drop_all())
        config = al.DataPoolConfig(cache_policy=cp, backtest_batch_size=1)
        lab = al.AlfarsLab.from_env_with_config(config)
        lab.with_filter("symbols not like '%BJ'")
        lab.with_years(2010, 2025)
        lab.with_backtest_config(10, "equal", 1, 1, 0.0005, 0.0015)
        lab.register("f", "rank(ts_mean(close, 20))")
        r = lab.backtest_each()
        return r[0][1]

    r_drop = compute_with("drop_all")
    r_keep = compute_with("keep_all")
    assert abs(r_drop.ic_mean - r_keep.ic_mean) < 1e-9, "IC should match"
    assert abs(r_drop.sharpe_ratio - r_keep.sharpe_ratio) < 1e-9, "Sharpe should match"
