"""WCR backtest via streaming architecture (evaluate_and_backtest_each).

Verifies the high-throughput pipeline produces identical results to the
reference calc()+run() path.
"""

import numpy as np
import polars as pl
import pytest
import alfars as al

WCR_EXPR = "1d:sum(5m:vol * 5m:close) / 1d:sum(5m:vol) / 1d:mean(5m:close)"
REF_CSV = ".tests/backtest_nav_py.csv"
TOLERANCE = 1e-9


def _int_to_date(d: int) -> str:
    return f"{d // 10000:04d}-{(d % 10000) // 100:02d}-{d % 100:02d}"


@pytest.fixture(scope="module", params=["drop_all", "keep_most_recent", "keep_all"])
def backtest_result(request):
    policy_name = request.param
    if policy_name == "keep_all":
        cp = al.CachePolicy.keep_all()
    elif policy_name == "keep_most_recent":
        cp = al.CachePolicy.keep_most_recent()
    else:
        cp = al.CachePolicy.drop_all()

    config = al.DataPoolConfig(
        cache_policy=cp,
        backtest_batch_size=5,
        calc_parallel_years=5,
    )
    lab = al.AlfarsLab.from_env_with_config(config)
    lab.with_filter("symbols not like '%BJ'")
    lab.with_years(2010, 2025)
    lab.with_backtest_config(10, "equal", 1, 1, 0.0005, 0.0015)
    lab.register("wcr", WCR_EXPR)

    results = lab.backtest_each()
    assert len(results) == 1
    name, result = results[0]
    assert name == "wcr"
    return result, policy_name


@pytest.fixture(scope="module")
def reference_nav() -> pl.DataFrame:
    return pl.read_csv(REF_CSV)


def test_nav_curves_match(backtest_result, reference_nav):
    result, policy_name = backtest_result
    dates = result.dates
    cum_ret = np.array(result.group_cum_returns)
    n_groups = cum_ret.shape[1]

    rows = []
    for g in range(n_groups):
        rows.append({"date": _int_to_date(int(dates[0])), "nv": 1.0, "group": g})
        for t in range(cum_ret.shape[0]):
            rows.append({
                "date": _int_to_date(int(dates[t + 1])),
                "nv": 1.0 + float(cum_ret[t, g]),
                "group": g,
            })

    actual = pl.DataFrame(rows).with_columns(pl.col("group").cast(pl.Int32))
    ref = reference_nav.with_columns(pl.col("group").cast(pl.Int32))
    joined = actual.join(ref, on=["date", "group"], how="inner", suffix="_ref")
    assert len(joined) > 0, f"[{policy_name}] no overlapping (date, group) rows"

    diff = (joined["nv"] - joined["nv_ref"]).abs()
    bad = diff > TOLERANCE
    assert bad.sum() == 0, (
        f"[{policy_name}] {(bad.sum())} NAV values exceed {TOLERANCE:.0e}\n"
        + "\n".join(
            f"  {joined.filter(bad).row(i, named=True)['date']} "
            f"group={joined.filter(bad).row(i, named=True)['group']}: "
            f"diff={diff.filter(bad)[i]:.2e}"
            for i in range(min(5, bad.sum()))
        )
    )


def test_statistics_smoke(backtest_result):
    result, policy_name = backtest_result
    assert -1.0 <= result.ic_mean <= 1.0, f"[{policy_name}] IC out of range"
    assert result.turnover >= 0.0, f"[{policy_name}] negative turnover"
    assert not np.isnan(result.sharpe_ratio), f"[{policy_name}] Sharpe is NaN"
