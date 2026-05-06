"""Debug alpha135/alpha152 NaN issues."""
import numpy as np
import alfars as al

# alpha135: ts_sma(ts_delay(close / ts_delay(close, 20), 1), 20, 1)
ALPHA135 = "ts_sma(ts_delay(close / ts_delay(close, 20), 1), 20, 1)"
# alpha152: complex multi-term expression
ALPHA152 = (
    "ts_sma(ts_mean(ts_delay(ts_sma(ts_delay(close / ts_delay(close, 9), 1), 9, 1), 1), 12) "
    "- ts_mean(ts_delay(ts_sma(ts_delay(close / ts_delay(close, 9), 1), 9, 1), 1), 26), 9, 1)"
)

for name, expr in [("alpha135", ALPHA135), ("alpha152", ALPHA152)]:
    print(f"\n=== {name} ===")
    print(f"Expr: {expr[:80]}...")

    lab = al.AlfarsLab.from_env()
    lab.with_filter("symbols not like '%BJ'")
    lab.with_years(2010, 2010)  # single year
    lab.with_backtest_config(10, "equal", 1, 1, 0.0003)
    lab.register(name, expr)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv") as tf:
        panel = lab.calc(tf.name)

    print(f"  slices: {len(panel.factor_names)}")

    prices = al.AlfarsLab.from_env() \
        .with_filter("symbols not like '%BJ'") \
        .with_years(2010, 2010)._prices  # hack: use evaluate instead
    # Just use evaluate()
    matrices, prices = lab.evaluate()
    mat = matrices[name]
    n = mat.shape[0] * mat.shape[1]
    nan_cnt = int(np.isnan(mat).sum())
    finite_cnt = int(np.isfinite(mat).sum())
    print(f"  matrix: {mat.shape}, NaN={nan_cnt}/{n} ({100*nan_cnt/n:.1f}%), finite={finite_cnt}")

    # Check first few symbols' time series
    for sym in range(min(5, mat.shape[1])):
        col = mat[:, sym]
        finite = np.isfinite(col).sum()
        if finite > 0:
            vals = col[np.isfinite(col)]
            print(f"    sym[{sym}]: finite={finite}, range=[{vals.min():.4f}, {vals.max():.4f}]")
        else:
            print(f"    sym[{sym}]: ALL NaN")
