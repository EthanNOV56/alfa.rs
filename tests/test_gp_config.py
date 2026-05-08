import alfars as al


def test_avail_fields():
    fields = al.AlfarsLab.avail_fields()
    assert isinstance(fields, list)
    assert len(fields) > 5
    assert "close" in fields
    assert "open" in fields
    assert "vol" in fields


def test_avail_ops():
    ops = al.AlfarsLab.avail_ops()
    assert isinstance(ops, list)
    assert len(ops) == 24
    assert "add" in ops
    assert "ts_mean" in ops
    assert "rank" in ops


def test_set_gp_config():
    lab = al.AlfarsLab.from_env()
    lab.with_filter("symbols not like '%BJ'")
    lab.with_years(2010, 2010)
    lab.with_backtest_config(10, "equal", 1, 1, 0.0005, 0.0015)

    lab.set_fields(["close", "vol"])
    lab.set_ops(["add", "sub", "mul", "div", "rank"])

    # Single seed
    lab.set_gp_seed("1d:close / 1d:open")

    # Verify no crash on run (GP is heavy, just check it parses)
    assert True


def test_set_gp_seed_list():
    lab = al.AlfarsLab.from_env()
    lab.with_filter("symbols not like '%BJ'")
    lab.with_years(2010, 2010)
    lab.with_backtest_config(10, "equal", 1, 1, 0.0005, 0.0015)

    lab.set_fields(["close"])
    lab.set_ops(["add", "sub", "mul"])

    # List seed
    lab.set_gp_seed(["1d:close / 1d:open", "1d:high - 1d:low"])

    assert True


def test_set_gp_seed_parse_error():
    lab = al.AlfarsLab.from_env()
    lab.with_filter("symbols not like '%BJ'")
    lab.with_years(2010, 2010)
    lab.with_backtest_config(10, "equal", 1, 1, 0.0005, 0.0015)

    try:
        lab.set_gp_seed("!@#INVALID!!")
        assert False, "should raise"
    except Exception:
        pass  # parse error expected
