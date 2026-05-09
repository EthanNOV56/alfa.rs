#!/usr/bin/env python3
"""
Alpha101 Factor Example.

Each alpha computed independently through winsor→zscore→cap_neu→qcut,
then equal-weight combined via lab.run_multi().

Skipped: alpha48,56,58,59,63,67,69,70,76,79,80,82,87,89,90,91,93,97,100
"""

import alfars as al

ret_expr = "close / ts_delay(close, 1) - 1"


def build_alphas() -> dict[str, str]:
    a: dict[str, str] = {}
    a["alpha1"] = f"(cs_rank(ts_argmax(power(quesval(0, {ret_expr}, close, ts_std({ret_expr}, 20)), 2.0), 5)) - 0.5)"
    a["alpha2"] = f"(-1) * ts_correlation(cs_rank(ts_delta(log(vol), 2)), cs_rank((close - open) / open), 6)"
    a["alpha3"] = "ts_correlation(cs_rank(open), cs_rank(vol), 10) * -1"
    a["alpha4"] = "-1 * ts_rank(cs_rank(low), 9)"
    a["alpha5"] = "cs_rank((open - (ts_sum(vwap, 10) / 10))) * (-1 * abs(cs_rank((close - vwap))))"
    a["alpha6"] = "(-1) * ts_correlation(open, vol, 10)"
    a["alpha7"] = f"quesval2(ts_mean(vol, 20), vol, (-1 * ts_rank(abs(close - ts_delay(close, 7)), 60)) * sign(ts_delta(close, 7)), -1)"
    a["alpha8"] = f"-1 * cs_rank(((ts_sum(open, 5) * ts_sum({ret_expr}, 5)) - ts_delay((ts_sum(open, 5) * ts_sum({ret_expr}, 5)), 10)))"
    a["alpha9"] = "quesval(0, ts_min(ts_delta(close, 1), 5), ts_delta(close, 1), quesval(0, ts_max(ts_delta(close, 1), 5), (-1 * ts_delta(close, 1)), ts_delta(close, 1)))"
    a["alpha10"] = "cs_rank(quesval(0, ts_min(ts_delta(close, 1), 4), ts_delta(close, 1), quesval(0, ts_max(ts_delta(close, 1), 4), (-1 * ts_delta(close, 1)), ts_delta(close, 1))))"
    a["alpha11"] = "(cs_rank(ts_max(vwap - close, 3)) + cs_rank(ts_min(vwap - close, 3))) * cs_rank(ts_delta(vol, 3))"
    a["alpha12"] = "sign(ts_delta(vol, 1)) * (-1 * ts_delta(close, 1))"
    a["alpha13"] = "-1 * cs_rank(ts_covariance(cs_rank(close), cs_rank(vol), 5))"
    a["alpha14"] = f"(-1 * cs_rank(({ret_expr}) - ts_delay({ret_expr}, 3))) * ts_correlation(open, vol, 10)"
    a["alpha15"] = "-1 * ts_sum(cs_rank(ts_correlation(cs_rank(high), cs_rank(vol), 3)), 3)"
    a["alpha16"] = "-1 * cs_rank(ts_covariance(cs_rank(high), cs_rank(vol), 5))"
    a["alpha17"] = "(-1 * cs_rank(ts_rank(close, 10))) * cs_rank(close - 2 * ts_delay(close, 1) + ts_delay(close, 2)) * cs_rank(ts_rank(vol / ts_mean(vol, 20), 5))"
    a["alpha18"] = "-1 * cs_rank((ts_std(abs(close - open), 5) + (close - open)) + ts_correlation(close, open, 10))"
    a["alpha19"] = f"(-1 * sign(ts_delta(close, 7) + (close - ts_delay(close, 7)))) * (cs_rank(ts_sum({ret_expr}, 250) + 1) + 1)"
    a["alpha20"] = "(-1 * cs_rank(open - ts_delay(high, 1))) * cs_rank(open - ts_delay(close, 1)) * cs_rank(open - ts_delay(low, 1))"
    a["alpha21"] = f"quesval2((ts_mean(close, 8) + ts_std(close, 8)), ts_mean(close, 2), -1, quesval2(ts_mean(close, 2), (ts_mean(close, 8) - ts_std(close, 8)), 1, quesval(1, (vol / ts_mean(vol, 20)), 1, -1)))"
    a["alpha22"] = "-1 * ts_delta(ts_correlation(high, vol, 5), 5) * cs_rank(ts_std(close, 20))"
    a["alpha23"] = "quesval2(ts_mean(high, 20), high, -1 * ts_delta(high, 2), 0)"
    a["alpha24"] = "quesval(0.05, ts_delta(ts_sum(close, 100) / 100, 100) / ts_delay(close, 100), (-1 * ts_delta(close, 3)), (-1 * (close - ts_min(close, 100))))"
    a["alpha25"] = f"cs_rank((-1 * {ret_expr}) * ts_mean(vol, 20) * vwap * (high - close))"
    a["alpha26"] = "-1 * ts_max(ts_correlation(ts_rank(vol, 5), ts_rank(high, 5), 5), 3)"
    a["alpha27"] = "quesval(0.5, cs_rank(ts_mean(ts_correlation(cs_rank(vol), cs_rank(vwap), 6), 2)), -1, 1)"
    a["alpha28"] = "cs_scale(ts_correlation(ts_mean(vol, 20), low, 5) + (high + low) / 2 - close)"
    a["alpha29"] = f"ts_min(ts_product(cs_rank(cs_rank(cs_scale(log(ts_sum(ts_min(cs_rank(cs_rank((-1 * cs_rank(ts_delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(ts_delay((-1 * {ret_expr}), 6), 5)"
    a["alpha30"] = "((cs_rank(sign(close - ts_delay(close, 1)) + sign(ts_delay(close, 1) - ts_delay(close, 2)) + sign(ts_delay(close, 2) - ts_delay(close, 3))) * -1 + 1) * ts_sum(vol, 5)) / ts_sum(vol, 20)"
    a["alpha31"] = "(cs_rank(cs_rank(cs_rank(ts_decay_linear((-1) * cs_rank(cs_rank(ts_delta(close, 10))), 10)))) + cs_rank((-1) * ts_delta(close, 3))) + sign(cs_scale(ts_correlation(ts_mean(vol, 20), low, 12)))"
    a["alpha32"] = "cs_scale((ts_sum(close, 7) / 7 - close)) + (20 * cs_scale(ts_correlation(vwap, ts_delay(close, 5), 230)))"
    a["alpha33"] = "cs_rank((-1) * (open / close * -1 + 1))"
    a["alpha34"] = f"cs_rank((cs_rank(ts_std({ret_expr}, 2) / ts_std({ret_expr}, 5)) * -1 + 1) + (cs_rank(ts_delta(close, 1)) * -1 + 1))"
    a["alpha35"] = f"(ts_rank(vol, 32) * (ts_rank((close + high - low), 16) * -1 + 1)) * (ts_rank({ret_expr}, 32) * -1 + 1)"
    a["alpha36"] = f"((((2.21 * cs_rank(ts_correlation((close - open), ts_delay(vol, 1), 15))) + (0.7 * cs_rank((open - close)))) + (0.73 * cs_rank(ts_rank(ts_delay((-1) * {ret_expr}, 6), 5)))) + cs_rank(abs(ts_correlation(vwap, ts_mean(vol, 20), 6)))) + (0.6 * cs_rank(((ts_sum(close, 200) / 200 - open) * (close - open))))"
    a["alpha37"] = "cs_rank(ts_correlation(ts_delay((open - close), 1), close, 200)) + cs_rank((open - close))"
    a["alpha38"] = "((-1) * cs_rank(ts_rank(close, 10))) * cs_rank((close / open))"
    a["alpha39"] = f"((-1) * cs_rank((ts_delta(close, 7) * (cs_rank(ts_decay_linear((vol / ts_mean(vol, 20)), 9)) * -1 + 1)))) * (cs_rank(ts_sum({ret_expr}, 250)) + 1)"
    a["alpha40"] = "((-1) * cs_rank(ts_std(high, 10))) * ts_correlation(high, vol, 10)"
    a["alpha41"] = "power((high * low), 0.5) - vwap"
    a["alpha42"] = "cs_rank((vwap - close)) / cs_rank((vwap + close))"
    a["alpha43"] = "ts_rank((vol / ts_mean(vol, 20)), 20) * ts_rank((-1) * ts_delta(close, 7), 8)"
    a["alpha44"] = "(-1) * ts_correlation(high, cs_rank(vol), 5)"
    a["alpha45"] = "(-1) * cs_rank(ts_sum(ts_delay(close, 5), 20) / 20) * ts_correlation(close, vol, 2) * cs_rank(ts_correlation(ts_sum(close, 5), ts_sum(close, 20), 2))"
    a["alpha46"] = f"quesval(0.25, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), -1, quesval(0, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), (-1) * (close - ts_delay(close, 1)), 1))"
    a["alpha47"] = "((cs_rank(power(close, -1)) * vol / ts_mean(vol, 20)) * (high * cs_rank(high - close)) / (ts_sum(high, 5) / 5)) - cs_rank(vwap - ts_delay(vwap, 5))"
    a["alpha49"] = f"quesval(-0.1, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), (-1) * (close - ts_delay(close, 1)), 1)"
    a["alpha50"] = "(-1) * ts_max(cs_rank(ts_correlation(cs_rank(vol), cs_rank(vwap), 5)), 5)"
    a["alpha51"] = f"quesval(-0.05, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), (-1) * (close - ts_delay(close, 1)), 1)"
    a["alpha52"] = f"(((-1) * ts_min(low, 5)) + ts_delay(ts_min(low, 5), 5)) * cs_rank((ts_sum({ret_expr}, 240) - ts_sum({ret_expr}, 20)) / 220) * ts_rank(vol, 5)"
    a["alpha53"] = "(-1) * ts_delta(((close - low) - (high - close)) / (close - low), 9)"
    a["alpha54"] = "((-1) * ((low - close) * power(open, 5))) / ((low - high) * power(close, 5))"
    a["alpha55"] = "(-1) * ts_correlation(cs_rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), cs_rank(vol), 6)"
    a["alpha57"] = "-1 * ((close - vwap) / ts_decay_linear(cs_rank(ts_argmax(close, 30)), 2))"
    a["alpha60"] = "- 1 * ((2 * cs_scale(cs_rank((((close - low) - (high - close)) / (high - low)) * vol))) - cs_scale(cs_rank(ts_argmax(close, 10))))"
    a["alpha61"] = "quesval2(cs_rank(vwap - ts_min(vwap, 16)), cs_rank(ts_correlation(vwap, ts_mean(vol, 180), 18)), 1, 0)"
    a["alpha62"] = "lt(cs_rank(ts_correlation(vwap, ts_sum(ts_mean(vol, 20), 22), 10)), cs_rank(lt(cs_rank(open) + cs_rank(open), cs_rank((high + low) / 2) + cs_rank(high)))) * -1"
    a["alpha64"] = "lt(cs_rank(ts_correlation(ts_sum(((open * 0.178404) + (low * (1 - 0.178404))), 13), ts_sum(ts_mean(vol, 120), 13), 17)), cs_rank(ts_delta((((high + low) / 2 * 0.178404) + (vwap * (1 - 0.178404))), 4))) * -1"
    a["alpha65"] = "lt(cs_rank(ts_correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), ts_sum(ts_mean(vol, 60), 9), 6)), cs_rank(open - ts_min(open, 14))) * -1"
    a["alpha66"] = "(cs_rank(ts_decay_linear(ts_delta(vwap, 4), 7)) + ts_rank(ts_decay_linear((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2)), 11), 7)) * -1"
    a["alpha68"] = "lt(ts_rank(ts_correlation(cs_rank(high), cs_rank(ts_mean(vol, 15)), 9), 14), cs_rank(ts_delta((close * 0.518371 + low * (1 - 0.518371)), 1))) * -1"
    a["alpha71"] = "gt(ts_rank(ts_decay_linear(ts_correlation(ts_rank(close, 3), ts_rank(ts_mean(vol, 180), 12), 18), 4), 16), ts_rank(ts_decay_linear(power(cs_rank((low + open) - (vwap + vwap)), 2), 16), 4))"
    a["alpha72"] = "cs_rank(ts_decay_linear(ts_correlation((high + low) / 2, ts_mean(vol, 40), 9), 10)) / cs_rank(ts_decay_linear(ts_correlation(ts_rank(vwap, 4), ts_rank(vol, 19), 7), 3))"
    a["alpha73"] = "gt(cs_rank(ts_decay_linear(ts_delta(vwap, 5), 3)), ts_rank(ts_decay_linear((ts_delta(open * 0.147155 + low * 0.852845, 2) / (open * 0.147155 + low * 0.852845)) * -1, 3), 17)) * -1"
    a["alpha74"] = "quesval2(cs_rank(ts_correlation(close, ts_sum(ts_mean(vol, 30), 37), 15)), cs_rank(ts_correlation(cs_rank(high * 0.0261661 + vwap * 0.9738339), cs_rank(vol), 11)), 1, 0) * -1"
    a["alpha75"] = "quesval2(cs_rank(ts_correlation(vwap, vol, 4)), cs_rank(ts_correlation(cs_rank(low), cs_rank(ts_mean(vol, 50)), 12)), 1, 0)"
    a["alpha77"] = "lt(cs_rank(ts_decay_linear((((high + low) / 2 + high) - (vwap + high)), 20)), cs_rank(ts_decay_linear(ts_correlation((high + low) / 2, ts_mean(vol, 40), 3), 6)))"
    a["alpha78"] = "power(cs_rank(ts_correlation(ts_sum((low * 0.352233) + (vwap * (1 - 0.352233)), 20), ts_sum(ts_mean(vol, 40), 20), 7)), cs_rank(ts_correlation(cs_rank(vwap), cs_rank(vol), 6)))"
    a["alpha81"] = "quesval2(cs_rank(log(ts_product(cs_rank(power(cs_rank(ts_correlation(vwap, ts_sum(ts_mean(vol, 10), 50), 8)), 4)), 15))), cs_rank(ts_correlation(cs_rank(vwap), cs_rank(vol), 5)), 1, 0) * -1"
    a["alpha83"] = "(cs_rank(ts_delay((high - low) / (ts_sum(close, 5) / 5), 2)) * cs_rank(cs_rank(vol))) / (((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close))"
    a["alpha84"] = "power(ts_rank(vwap - ts_max(vwap, 15), 21), ts_delta(close, 5))"
    a["alpha85"] = "power(cs_rank(ts_correlation(high * 0.876703 + close * 0.123297, ts_mean(vol, 30), 10)), cs_rank(ts_correlation(ts_rank((high + low) / 2, 4), ts_rank(vol, 10), 7)))"
    a["alpha86"] = "quesval2(ts_rank(ts_correlation(close, ts_sum(ts_mean(vol, 20), 15), 6), 20), cs_rank((open + close) - (vwap + open)), 1, 0) * -1"
    a["alpha88"] = "lt(cs_rank(ts_decay_linear((cs_rank(open) + cs_rank(low)) - (cs_rank(high) + cs_rank(close)), 8)), ts_rank(ts_decay_linear(ts_correlation(ts_rank(close, 8), ts_rank(ts_mean(vol, 60), 21), 8), 7), 3))"
    a["alpha92"] = "lt(ts_rank(ts_decay_linear(quesval2(((high + low) / 2 + close), (low + open), 1, 0), 15), 19), ts_rank(ts_decay_linear(ts_correlation(cs_rank(low), cs_rank(ts_mean(vol, 30)), 8), 7), 7))"
    a["alpha94"] = "power(cs_rank(vwap - ts_min(vwap, 12)), ts_rank(ts_correlation(ts_rank(vwap, 20), ts_rank(ts_mean(vol, 60), 4), 18), 3)) * -1"
    a["alpha95"] = "quesval2(cs_rank(open - ts_min(open, 12)), ts_rank(power(cs_rank(ts_correlation(ts_sum((high + low) / 2, 19), ts_sum(ts_mean(vol, 40), 19), 13)), 5), 12), 1, 0)"
    a["alpha96"] = "gt(ts_rank(ts_decay_linear(ts_correlation(cs_rank(vwap), cs_rank(vol), 4), 4), 8), ts_rank(ts_decay_linear(ts_argmax(ts_correlation(ts_rank(close, 7), ts_rank(ts_mean(vol, 60), 4), 4), 13), 14), 13)) * -1"
    a["alpha98"] = "cs_rank(ts_decay_linear(ts_correlation(vwap, ts_sum(ts_mean(vol, 5), 26), 5), 7)) - cs_rank(ts_decay_linear(ts_rank(ts_argmin(ts_correlation(cs_rank(open), cs_rank(ts_mean(vol, 15)), 21), 9), 7), 8))"
    a["alpha99"] = "quesval2(cs_rank(ts_correlation(ts_sum((high + low) / 2, 20), ts_sum(ts_mean(vol, 60), 20), 9)), cs_rank(ts_correlation(low, vol, 6)), 1, 0) * -1"
    a["alpha101"] = "((close - open) / ((high - low) + 0.001))"
    return a


SKIPPED = {
    "alpha48", "alpha56", "alpha58", "alpha59", "alpha63", "alpha67",
    "alpha69", "alpha70", "alpha76", "alpha79", "alpha80", "alpha82",
    "alpha87", "alpha89", "alpha90", "alpha91", "alpha93", "alpha97", "alpha100",
}


def main():
    alphas = build_alphas()
    print(f"Alphas defined: {len(alphas)}")

    lab = al.AlfarsLab.from_env()
    lab.set_pool("symbols not like '%BJ'")
    lab.set_duration(2024, 2024)

    for name, expr in alphas.items():
        if name in SKIPPED:
            continue
        try:
            lab.register(name, expr)
        except Exception as e:
            print(f"  REGISTER FAIL {name}: {e}")

    names = sorted(alphas.keys() - SKIPPED)
    print(f"Registered {len(names)}, computing...")

    # Compute all factors through full pipeline (winsor→zscore→cap_neu→qcut)
    matrices, prices = lab.evaluate()
    valid_mats = [matrices[n] for n in names if n in matrices]
    print(f"Valid factor matrices: {len(valid_mats)}")

    # Equal-weight multi-factor backtest
    result = lab.run_multi(valid_mats, prices)

    print(f"\n── Backtest Results ({len(valid_mats)} factors) ──")
    print(f"  IC mean:   {result.ic_mean:.6f}")
    print(f"  IC IR:     {result.ic_ir:.4f}")
    print(f"  Total ret: {result.total_return:.4f}")
    print(f"  Sharpe:    {result.sharpe_ratio:.4f}")
    print(f"  Max DD:    {result.max_drawdown:.4f}")
    print(f"  Turnover:  {result.turnover:.4f}")


if __name__ == "__main__":
    main()
