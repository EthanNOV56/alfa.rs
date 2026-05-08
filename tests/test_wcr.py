import os
import numpy as np
import polars as pl
import pytest
import alfars as al

WCR_EXPR = "1d:sum(5m:vol * 5m:close) / 1d:sum(5m:vol) / 1d:mean(5m:close)"
REF_CSV = ".tests/backtest_nav_py.csv"
DUMP_CSV = ".tests/backtest_nav_rs.csv"
TOLERANCE = 1e-9


def _int_to_date(d: int) -> str:
    """20100104 -> '2010-01-04'"""
    return f"{d // 10000:04d}-{(d % 10000) // 100:02d}-{d % 100:02d}"


@pytest.fixture(scope="module")
def backtest_result():
    lab = al.AlfarsLab.from_env()
    lab.with_filter("symbols not like '%BJ'")
    lab.with_years(2010, 2025)
    lab.with_backtest_config(10, "equal", 1, 1, 0.0005, 0.0015)
    lab.register("wcr", WCR_EXPR)
    panel = lab.calc()
    panel.to_csv("/tmp/wcr_panel.csv")
    result = lab.run_bt()
    result.to_csv(DUMP_CSV)
    return result


@pytest.fixture(scope="module")
def reference_nav() -> pl.DataFrame:
    return pl.read_csv(REF_CSV)


def test_nav_curves_match(backtest_result, reference_nav):
    dates = backtest_result.dates
    cum_ret = np.array(backtest_result.group_cum_returns)
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
    assert len(joined) > 0, "no overlapping (date, group) rows"

    diff = (joined["nv"] - joined["nv_ref"]).abs()
    bad = diff > TOLERANCE
    assert bad.sum() == 0, (
        f"{(bad.sum())} NAV values exceed {TOLERANCE:.0e}\n"
        + "\n".join(
            f"  {joined.filter(bad).row(i, named=True)['date']} "
            f"group={joined.filter(bad).row(i, named=True)['group']}: "
            f"diff={diff.filter(bad)[i]:.2e}"
            for i in range(min(5, bad.sum()))
        )
    )


def test_statistics_smoke(backtest_result):
    assert -1.0 <= backtest_result.ic_mean <= 1.0
    assert backtest_result.turnover >= 0.0
    assert not np.isnan(backtest_result.sharpe_ratio)


def test_register_dict():
    lab = al.AlfarsLab.from_env()
    lab.with_filter("symbols not like '%BJ'")
    lab.with_years(2010, 2010)  # single year, fast
    lab.with_backtest_config(10, "equal", 1, 1, 0.0005, 0.0015)
    lab.register({"test_factor": WCR_EXPR})
    panel = lab.calc()
    assert panel is not None


def test_to_af_and_dump():
    expr = al.parse_expression(WCR_EXPR)
    factor = expr.to_af("test_wcr")
    assert factor.name == "test_wcr"
    assert "sum" in factor.expression
    path = factor.dump()
    assert os.path.exists(path)
    os.remove(path)
