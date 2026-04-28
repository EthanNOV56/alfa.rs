#!/usr/bin/env python3
"""
Alpha191 Factor Example.

Computes Alpha191 factors from daily OHLCV data using the full Rust pipeline:
ClickHouseSource → DataLayer → FactorRegistry.compute_factor_matrices_1d →
FactorCombiner → BacktestEngine.run_with_prices.

Operators translated from numpy/pandas to alfars ts_/cs_ canonical names:
  Rank→cs_rank  Delay→ts_delay  Delta→ts_delta  Corr→ts_correlation
  Cov→ts_covariance  Sum→ts_sum  Mean→ts_mean  Std→ts_std
  Tsrank→ts_rank  Tsmax→ts_max  Tsmin→ts_min  Product→ts_product
  Sign→sign  Max→max  Min→min  Abs→abs  Log→log  Power→power
  Sma→ts_sma  Decaylinear→ts_decay_linear  Highday→ts_highday
  Lowday→ts_lowday  Wma→ts_wma  Count→ts_sum

Alphas skipped:
  - Regbeta: alpha021,030,116,147
  - Benchmark data: alpha075,149,181,182
  - SELF recursion: alpha143
  - Rowmax/Rowmin: alpha165,183
  - Complex/broken: alpha055,137,166,190
  - amount-only (needs data): alpha070,095,132
"""

import alfars as al
from alfars._core import (
    ClickHouseSource,
    DataLayer,
    FactorCombiner,
    FactorRegistry,
    PyBacktestEngine,
)

ret_expr = "close / ts_delay(close, 1) - 1"


def build_alphas() -> dict[str, str]:
    a: dict[str, str] = {}

    a["alpha001"] = "-1 * ts_correlation(cs_rank(ts_delta(log(volume), 1)), cs_rank((close - open) / open), 6)"
    a["alpha002"] = "-1 * ts_delta(((close - low) - (high - close)) / (high - low), 1)"
    a["alpha003"] = "ts_sum(quesval2(close, ts_delay(close, 1), 0, quesval(0, close - ts_delay(close, 1), close - min(low, ts_delay(close, 1)), close - max(high, ts_delay(close, 1)))), 6)"
    a["alpha004"] = (
        "quesval2(ts_sum(close, 8) / 8 + ts_std(close, 8), ts_sum(close, 2) / 2, "
        "-1, quesval2(ts_sum(close, 2) / 2, ts_sum(close, 8) / 8 + ts_std(close, 8), "
        "quesval2(ts_sum(close, 8) / 8 - ts_std(close, 8), ts_sum(close, 2) / 2, "
        "quesval(0.999, volume / ts_mean(volume, 20), 1, -1), 1), 1))"
    )
    a["alpha005"] = "-1 * ts_max(ts_correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)"
    a["alpha006"] = "-1 * cs_rank(sign(ts_delta((open * 0.85 + high * 0.15), 4)))"
    a["alpha007"] = "(cs_rank(ts_max(vwap - close, 3)) + cs_rank(ts_min(vwap - close, 3))) * cs_rank(ts_delta(volume, 3))"
    a["alpha008"] = "cs_rank(ts_delta(((high + low) / 2 * 0.2 + vwap * 0.8), 4) * -1)"
    a["alpha009"] = "ts_sma(((high + low) / 2 - (ts_delay(high, 1) + ts_delay(low, 1)) / 2) * (high - low) / volume, 7, 2)"
    a["alpha010"] = f"cs_rank(ts_max(power(quesval(0, {ret_expr}, ts_std({ret_expr}, 20), close), 2), 5))"
    a["alpha011"] = "ts_sum(((close - low) - (high - close)) / (high - low) * volume, 6)"
    a["alpha012"] = "cs_rank(open - ts_sum(vwap, 10) / 10) * -1 * cs_rank(abs(close - vwap))"
    a["alpha013"] = "power(high * low, 0.5) - vwap"
    a["alpha014"] = "close - ts_delay(close, 5)"
    a["alpha015"] = "open / ts_delay(close, 1) - 1"
    a["alpha016"] = "-1 * ts_max(cs_rank(ts_correlation(cs_rank(volume), cs_rank(vwap), 5)), 5)"
    a["alpha017"] = "power(cs_rank(vwap - ts_max(vwap, 15)), ts_delta(close, 5))"
    a["alpha018"] = "close / ts_delay(close, 5)"
    a["alpha019"] = (
        "quesval(0, close - ts_delay(close, 5), "
        "quesval(0, ts_delay(close, 5) - close, "
        "(close - ts_delay(close, 5)) / close, (close - ts_delay(close, 5)) / ts_delay(close, 5)), 0)"
    )
    a["alpha020"] = "(close - ts_delay(close, 6)) / ts_delay(close, 6) * 100"
    # alpha021: Regbeta — skipped
    a["alpha022"] = (
        "ts_sma(((close - ts_mean(close, 6)) / ts_mean(close, 6) "
        "- ts_delay((close - ts_mean(close, 6)) / ts_mean(close, 6), 3)), 12, 1)"
    )
    a["alpha023"] = (
        "ts_sma(quesval(0, close - ts_delay(close, 1), ts_std(close, 20), 0), 20, 1) "
        "/ (ts_sma(quesval(0, close - ts_delay(close, 1), ts_std(close, 20), 0), 20, 1) "
        "+ ts_sma(quesval(0, ts_delay(close, 1) - close, ts_std(close, 20), 0), 20, 1)) * 100"
    )
    a["alpha024"] = "ts_sma(close - ts_delay(close, 5), 5, 1)"
    a["alpha025"] = (
        f"(-1 * cs_rank((ts_delta(close, 7) * "
        f"(1 - cs_rank(ts_decay_linear(volume / ts_mean(volume, 20), 9)))))) "
        f"* (1 + cs_rank(ts_sum({ret_expr}, 250)))"
    )
    a["alpha026"] = "((ts_sum(close, 7) / 7 - close)) + ts_correlation(vwap, ts_delay(close, 5), 230)"
    a["alpha027"] = (
        "ts_wma(((close - ts_delay(close, 3)) / ts_delay(close, 3) * 100 "
        "+ (close - ts_delay(close, 6)) / ts_delay(close, 6) * 100), 12)"
    )
    a["alpha028"] = (
        "3 * ts_sma((close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9)) * 100, 3, 1) "
        "- 2 * ts_sma(ts_sma((close - ts_min(low, 9)) / (ts_max(high, 9) - ts_max(low, 9)) * 100, 3, 1), 3, 1)"
    )
    a["alpha029"] = "(close - ts_delay(close, 6)) / ts_delay(close, 6) * volume"
    # alpha030: Regbeta — skipped
    a["alpha031"] = "(close - ts_mean(close, 12)) / ts_mean(close, 12) * 100"
    a["alpha032"] = "-1 * ts_sum(cs_rank(ts_correlation(cs_rank(high), cs_rank(volume), 3)), 3)"
    a["alpha033"] = (
        f"(((-1 * ts_min(low, 5)) + ts_delay(ts_min(low, 5), 5)) "
        f"* cs_rank((ts_sum({ret_expr}, 240) - ts_sum({ret_expr}, 20)) / 220)) "
        f"* ts_rank(volume, 5)"
    )
    a["alpha034"] = "ts_mean(close, 12) / close"
    a["alpha035"] = (
        "min(cs_rank(ts_decay_linear(ts_delta(open, 1), 15)), "
        "cs_rank(ts_decay_linear(ts_correlation(volume, open, 17), 7))) * -1"
    )
    a["alpha036"] = "cs_rank(ts_sum(ts_correlation(cs_rank(volume), cs_rank(vwap), 6), 2))"
    a["alpha037"] = (
        "-1 * cs_rank((ts_sum(open, 5) * ts_sum(close / ts_delay(close, 1) - 1, 5) "
        "- ts_delay(ts_sum(open, 5) * ts_sum(close / ts_delay(close, 1) - 1, 5), 10)))"
    )
    a["alpha038"] = "quesval(0, ts_sum(high, 20) / 20 - high, -1 * ts_delta(high, 2), 0)"
    a["alpha039"] = (
        "(cs_rank(ts_decay_linear(ts_delta(close, 2), 8)) "
        "- cs_rank(ts_decay_linear(ts_correlation(vwap * 0.3 + open * 0.7, "
        "ts_sum(ts_mean(volume, 180), 37), 14), 12))) * -1"
    )
    a["alpha040"] = (
        "ts_sum(quesval(0, close - ts_delay(close, 1), volume, 0), 26) "
        "/ ts_sum(quesval(0, ts_delay(close, 1) - close, volume, 0), 26) * 100"
    )
    a["alpha041"] = "cs_rank(ts_max(ts_delta(vwap, 3), 5)) * -1"
    a["alpha042"] = "-1 * cs_rank(ts_std(high, 10)) * ts_correlation(high, volume, 10)"
    a["alpha043"] = (
        "ts_sum(quesval2(close, ts_delay(close, 1), "
        "quesval2(ts_delay(close, 1), close, -volume, volume), 0), 6)"
    )
    a["alpha044"] = (
        "ts_rank(ts_decay_linear(ts_correlation(low, ts_mean(volume, 10), 7), 6), 4) "
        "+ ts_rank(ts_decay_linear(ts_delta(vwap, 3), 10), 15)"
    )
    a["alpha045"] = (
        "cs_rank(ts_delta(close * 0.6 + open * 0.4, 1)) "
        "* cs_rank(ts_correlation(vwap, ts_mean(volume, 150), 15))"
    )
    a["alpha046"] = "(ts_mean(close, 3) + ts_mean(close, 6) + ts_mean(close, 12) + ts_mean(close, 24)) / (4 * close)"
    a["alpha047"] = "ts_sma((ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6)) * 100, 9, 1)"
    a["alpha048"] = (
        "(-1 * (cs_rank((sign(close - ts_delay(close, 1)) "
        "+ sign(ts_delay(close, 1) - ts_delay(close, 2)) "
        "+ sign(ts_delay(close, 2) - ts_delay(close, 3)))) * ts_sum(volume, 5))) "
        "/ ts_sum(volume, 20)"
    )
    a["alpha049"] = (
        "ts_sum(quesval(0, ts_delay(high, 1) + ts_delay(low, 1) - high - low, "
        "max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12) "
        "/ (ts_sum(quesval(0, ts_delay(high, 1) + ts_delay(low, 1) - high - low, "
        "max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12) "
        "+ ts_sum(quesval(0, high + low - ts_delay(high, 1) - ts_delay(low, 1), "
        "max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12))"
    )
    a["alpha050"] = (
        "(ts_sum(quesval(0, high + low - ts_delay(high, 1) - ts_delay(low, 1), "
        "max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12) "
        "- ts_sum(quesval(0, ts_delay(high, 1) + ts_delay(low, 1) - high - low, "
        "max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12)) "
        "/ (ts_sum(quesval(0, high + low - ts_delay(high, 1) - ts_delay(low, 1), "
        "max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12) "
        "+ ts_sum(quesval(0, ts_delay(high, 1) + ts_delay(low, 1) - high - low, "
        "max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12))"
    )
    a["alpha051"] = (
        "ts_sum(quesval(0, high + low - ts_delay(high, 1) - ts_delay(low, 1), "
        "max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12) "
        "/ (ts_sum(quesval(0, high + low - ts_delay(high, 1) - ts_delay(low, 1), "
        "max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12) "
        "+ ts_sum(quesval(0, ts_delay(high, 1) + ts_delay(low, 1) - high - low, "
        "max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12))"
    )
    a["alpha052"] = (
        "ts_sum(max(high - ts_delay((high + low + close) / 3, 1), 0), 26) "
        "/ ts_sum(max(ts_delay((high + low + close) / 3, 1) - low, 0), 26) * 100"
    )
    a["alpha053"] = "ts_sum(gt(close, ts_delay(close, 1)), 12) / 12 * 100"
    a["alpha054"] = "-1 * cs_rank(ts_std(abs(close - open), 20) + (close - open) + ts_correlation(close, open, 10))"
    # alpha055: broken — skip
    a["alpha056"] = (
        "quesval2(cs_rank(open - ts_min(open, 12)), "
        "cs_rank(power(cs_rank(ts_correlation(ts_sum((high + low) / 2, 19), "
        "ts_sum(ts_mean(volume, 40), 19), 13)), 5)), 1, 0)"
    )
    a["alpha057"] = "ts_sma((close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9)) * 100, 3, 1)"
    a["alpha058"] = "ts_sum(gt(close, ts_delay(close, 1)), 20) / 20 * 100"
    a["alpha059"] = (
        "ts_sum(quesval2(close, ts_delay(close, 1), "
        "quesval2(ts_delay(close, 1), close, "
        "close - max(high, ts_delay(close, 1)), close - min(low, ts_delay(close, 1))), 0), 20)"
    )
    a["alpha060"] = "ts_sum(((close - low) - (high - close)) / (high - low) * volume, 20)"
    a["alpha061"] = (
        "max(cs_rank(ts_decay_linear(ts_delta(vwap, 1), 12)), "
        "cs_rank(ts_decay_linear(ts_rank(ts_correlation(low, ts_mean(volume, 80), 8), 17), 17))) * -1"
    )
    a["alpha062"] = "-1 * ts_correlation(high, cs_rank(volume), 5)"
    a["alpha063"] = (
        "ts_sma(max(close - ts_delay(close, 1), 0), 6, 1) "
        "/ ts_sma(abs(close - ts_delay(close, 1)), 6, 1) * 100"
    )
    a["alpha064"] = (
        "max(cs_rank(ts_decay_linear(ts_correlation(cs_rank(vwap), cs_rank(volume), 4), 4)), "
        "cs_rank(ts_decay_linear(ts_max(ts_correlation(cs_rank(close), cs_rank(ts_mean(volume, 60)), 4), 13), 14))) * -1"
    )
    a["alpha065"] = "ts_mean(close, 6) / close"
    a["alpha066"] = "(close - ts_mean(close, 6)) / ts_mean(close, 6) * 100"
    a["alpha067"] = (
        "ts_sma(max(close - ts_delay(close, 1), 0), 24, 1) "
        "/ ts_sma(abs(close - ts_delay(close, 1)), 24, 1) * 100"
    )
    a["alpha068"] = "ts_sma(((high + low) / 2 - (ts_delay(high, 1) + ts_delay(low, 1)) / 2) * (high - low) / volume, 15, 2)"
    a["alpha069"] = (
        "quesval2(ts_sum(quesval(0, open - ts_delay(open, 1), 0, max(high - open, open - ts_delay(open, 1))), 20), "
        "ts_sum(quesval(0, ts_delay(open, 1) - open, 0, max(open - low, open - ts_delay(open, 1))), 20), "
        "(ts_sum(quesval(0, open - ts_delay(open, 1), 0, max(high - open, open - ts_delay(open, 1))), 20) "
        "- ts_sum(quesval(0, ts_delay(open, 1) - open, 0, max(open - low, open - ts_delay(open, 1))), 20)) "
        "/ ts_sum(quesval(0, open - ts_delay(open, 1), 0, max(high - open, open - ts_delay(open, 1))), 20), "
        "(ts_sum(quesval(0, open - ts_delay(open, 1), 0, max(high - open, open - ts_delay(open, 1))), 20) "
        "- ts_sum(quesval(0, ts_delay(open, 1) - open, 0, max(open - low, open - ts_delay(open, 1))), 20)) "
        "/ ts_sum(quesval(0, ts_delay(open, 1) - open, 0, max(open - low, open - ts_delay(open, 1))), 20))"
    )
    a["alpha070"] = "ts_std(amount, 6)"
    a["alpha071"] = "(close - ts_mean(close, 24)) / ts_mean(close, 24) * 100"
    a["alpha072"] = "ts_sma((ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6)) * 100, 15, 1)"
    a["alpha073"] = (
        "(ts_rank(ts_decay_linear(ts_decay_linear(ts_correlation(close, volume, 10), 16), 4), 5) "
        "- cs_rank(ts_decay_linear(ts_correlation(vwap, ts_mean(volume, 30), 4), 3))) * -1"
    )
    a["alpha074"] = (
        "cs_rank(ts_correlation(ts_sum(low * 0.35 + vwap * 0.65, 20), "
        "ts_sum(ts_mean(volume, 40), 20), 7)) "
        "+ cs_rank(ts_correlation(cs_rank(vwap), cs_rank(volume), 6))"
    )
    # alpha075: benchmark data — skip
    a["alpha076"] = (
        "ts_std(abs(close / ts_delay(close, 1) - 1) / volume, 20) "
        "/ ts_mean(abs(close / ts_delay(close, 1) - 1) / volume, 20)"
    )
    a["alpha077"] = (
        "min(cs_rank(ts_decay_linear(((high + low) / 2 + high - vwap - high), 20)), "
        "cs_rank(ts_decay_linear(ts_correlation((high + low) / 2, ts_mean(volume, 40), 3), 6)))"
    )
    a["alpha078"] = (
        "((high + low + close) / 3 - ts_mean((high + low + close) / 3, 12)) "
        "/ (0.015 * ts_mean(abs(close - ts_mean((high + low + close) / 3, 12)), 12))"
    )
    a["alpha079"] = (
        "ts_sma(max(close - ts_delay(close, 1), 0), 12, 1) "
        "/ ts_sma(abs(close - ts_delay(close, 1)), 12, 1) * 100"
    )
    a["alpha080"] = "(volume - ts_delay(volume, 5)) / ts_delay(volume, 5) * 100"
    a["alpha081"] = "ts_sma(volume, 21, 2)"
    a["alpha082"] = "ts_sma((ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6)) * 100, 20, 1)"
    a["alpha083"] = "-1 * cs_rank(ts_covariance(cs_rank(high), cs_rank(volume), 5))"
    a["alpha084"] = (
        "ts_sum(quesval2(close, ts_delay(close, 1), "
        "quesval2(ts_delay(close, 1), close, -volume, volume), 0), 20)"
    )
    a["alpha085"] = "ts_rank(volume / ts_mean(volume, 20), 20) * ts_rank(-1 * ts_delta(close, 7), 8)"
    a["alpha086"] = (
        "quesval(0.25, (ts_delay(close, 20) - ts_delay(close, 10)) / 10 "
        "- (ts_delay(close, 10) - close) / 10, -1, "
        "quesval(0, (ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10, "
        "1, -1 * (close - ts_delay(close, 1))))"
    )
    a["alpha087"] = (
        "(cs_rank(ts_decay_linear(ts_delta(vwap, 4), 7)) "
        "+ ts_rank(ts_decay_linear(((low - vwap) / (open - (high + low) / 2)), 11), 7)) * -1"
    )
    a["alpha088"] = "(close - ts_delay(close, 20)) / ts_delay(close, 20) * 100"
    a["alpha089"] = "2 * (ts_sma(close, 13, 2) - ts_sma(close, 27, 2) - ts_sma(ts_sma(close, 13, 2) - ts_sma(close, 27, 2), 10, 2))"
    a["alpha090"] = "cs_rank(ts_correlation(cs_rank(vwap), cs_rank(volume), 5)) * -1"
    a["alpha091"] = "cs_rank(close - ts_max(close, 5)) * cs_rank(ts_correlation(ts_mean(volume, 40), low, 5)) * -1"
    a["alpha092"] = (
        "max(cs_rank(ts_decay_linear(ts_delta(close * 0.35 + vwap * 0.65, 2), 3)), "
        "ts_rank(ts_decay_linear(abs(ts_correlation(ts_mean(volume, 180), close, 13)), 5), 15)) * -1"
    )
    a["alpha093"] = (
        "ts_sum(quesval(0, open - ts_delay(open, 1), "
        "0, max(open - low, open - ts_delay(open, 1))), 20)"
    )
    a["alpha094"] = (
        "ts_sum(quesval2(close, ts_delay(close, 1), "
        "quesval2(ts_delay(close, 1), close, -volume, volume), 0), 30)"
    )
    a["alpha095"] = "ts_std(amount, 20)"
    a["alpha096"] = "ts_sma(ts_sma((close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9)) * 100, 3, 1), 3, 1)"
    a["alpha097"] = "ts_std(volume, 10)"
    a["alpha098"] = (
        "quesval(0.05, ts_delta(ts_sum(close, 100) / 100, 100) / ts_delay(close, 100), "
        "-1 * (close - ts_min(close, 100)), -1 * ts_delta(close, 3))"
    )
    a["alpha099"] = "-1 * cs_rank(ts_covariance(cs_rank(close), cs_rank(volume), 5))"
    a["alpha100"] = "ts_std(volume, 20)"
    a["alpha101"] = (
        "quesval2(cs_rank(ts_correlation(close, ts_sum(ts_mean(volume, 30), 37), 15)), "
        "cs_rank(ts_correlation(cs_rank(high * 0.1 + vwap * 0.9), cs_rank(volume), 11)), 1, 0) * -1"
    )
    a["alpha102"] = (
        "ts_sma(max(volume - ts_delay(volume, 1), 0), 6, 1) "
        "/ ts_sma(abs(volume - ts_delay(volume, 1)), 6, 1) * 100"
    )
    a["alpha103"] = "((20 - ts_lowday(low, 20)) / 20) * 100"
    a["alpha104"] = "-1 * ts_delta(ts_correlation(high, volume, 5), 5) * cs_rank(ts_std(close, 20))"
    a["alpha105"] = "-1 * ts_correlation(cs_rank(open), cs_rank(volume), 10)"
    a["alpha106"] = "close - ts_delay(close, 20)"
    a["alpha107"] = (
        "(-1 * cs_rank(open - ts_delay(high, 1))) "
        "* cs_rank(open - ts_delay(close, 1)) * cs_rank(open - ts_delay(low, 1))"
    )
    a["alpha108"] = "power(cs_rank(high - ts_min(high, 2)), cs_rank(ts_correlation(vwap, ts_mean(volume, 120), 6))) * -1"
    a["alpha109"] = "ts_sma(high - low, 10, 2) / ts_sma(ts_sma(high - low, 10, 2), 10, 2)"
    a["alpha110"] = (
        "ts_sum(max(high - ts_delay(close, 1), 0), 20) "
        "/ ts_sum(max(ts_delay(close, 1) - low, 0), 20) * 100"
    )
    a["alpha111"] = (
        "ts_sma(volume * ((close - low) - (high - close)) / (high - low), 11, 2) "
        "- ts_sma(volume * ((close - low) - (high - close)) / (high - low), 4, 2)"
    )
    a["alpha112"] = (
        "(ts_sum(quesval(0, close - ts_delay(close, 1), close - ts_delay(close, 1), 0), 12) "
        "- ts_sum(quesval(0, ts_delay(close, 1) - close, abs(close - ts_delay(close, 1)), 0), 12)) "
        "/ (ts_sum(quesval(0, close - ts_delay(close, 1), close - ts_delay(close, 1), 0), 12) "
        "+ ts_sum(quesval(0, ts_delay(close, 1) - close, abs(close - ts_delay(close, 1)), 0), 12)) * 100"
    )
    a["alpha113"] = (
        "-1 * (cs_rank(ts_sum(ts_delay(close, 5), 20) / 20) "
        "* ts_correlation(close, volume, 2) "
        "* cs_rank(ts_correlation(ts_sum(close, 5), ts_sum(close, 20), 2)))"
    )
    a["alpha114"] = (
        "(cs_rank(ts_delay((high - low) / (ts_sum(close, 5) / 5), 2)) * cs_rank(cs_rank(volume))) "
        "/ (((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close))"
    )
    a["alpha115"] = (
        "power(cs_rank(ts_correlation(high * 0.9 + close * 0.1, ts_mean(volume, 30), 10)), "
        "cs_rank(ts_correlation(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7)))"
    )
    # alpha116: Regbeta — skipped
    a["alpha117"] = (
        f"(ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16))) "
        f"* (1 - ts_rank({ret_expr}, 32))"
    )
    a["alpha118"] = "ts_sum(high - open, 20) / ts_sum(open - low, 20) * 100"
    a["alpha119"] = (
        "cs_rank(ts_decay_linear(ts_correlation(vwap, ts_sum(ts_mean(volume, 5), 26), 5), 7)) "
        "- cs_rank(ts_decay_linear(ts_rank(ts_min(ts_correlation(cs_rank(open), "
        "cs_rank(ts_mean(volume, 15)), 21), 9), 7), 8))"
    )
    a["alpha120"] = "cs_rank(vwap - close) / cs_rank(vwap + close)"
    a["alpha121"] = (
        "power(cs_rank(vwap - ts_min(vwap, 12)), "
        "ts_rank(ts_correlation(ts_rank(vwap, 20), ts_rank(ts_mean(volume, 60), 2), 18), 3)) * -1"
    )
    a["alpha122"] = (
        "(ts_sma(ts_sma(ts_sma(log(close), 13, 2), 13, 2), 13, 2) "
        "- ts_delay(ts_sma(ts_sma(ts_sma(log(close), 13, 2), 13, 2), 13, 2), 1)) "
        "/ ts_delay(ts_sma(ts_sma(ts_sma(log(close), 13, 2), 13, 2), 13, 2), 1)"
    )
    a["alpha123"] = (
        "quesval2(cs_rank(ts_correlation(ts_sum((high + low) / 2, 20), "
        "ts_sum(ts_mean(volume, 60), 20), 9)), "
        "cs_rank(ts_correlation(low, volume, 6)), 0, -1)"
    )
    a["alpha124"] = "(close - vwap) / ts_decay_linear(cs_rank(ts_max(close, 30)), 2)"
    a["alpha125"] = (
        "cs_rank(ts_decay_linear(ts_correlation(vwap, ts_mean(volume, 80), 17), 20)) "
        "/ cs_rank(ts_decay_linear(ts_delta(close * 0.5 + vwap * 0.5, 3), 16))"
    )
    a["alpha126"] = "(close + high + low) / 3"
    a["alpha127"] = "power(ts_mean(power(100 * (close - ts_max(close, 12)) / ts_max(close, 12), 2), 12), 0.5)"
    a["alpha128"] = (
        "100 - (100 / (1 + ts_sum(quesval(0, (high + low + close) / 3 "
        "- ts_delay((high + low + close) / 3, 1), "
        "(high + low + close) / 3 * volume, 0), 14) "
        "/ ts_sum(quesval(0, ts_delay((high + low + close) / 3, 1) "
        "- (high + low + close) / 3, "
        "(high + low + close) / 3 * volume, 0), 14)))"
    )
    a["alpha129"] = (
        "ts_sum(quesval(0, close - ts_delay(close, 1), "
        "0, abs(close - ts_delay(close, 1))), 12)"
    )
    a["alpha130"] = (
        "cs_rank(ts_decay_linear(ts_correlation((high + low) / 2, ts_mean(volume, 40), 9), 10)) "
        "/ cs_rank(ts_decay_linear(ts_correlation(cs_rank(vwap), cs_rank(volume), 7), 3))"
    )
    a["alpha131"] = (
        "power(cs_rank(ts_delta(vwap, 1)), "
        "ts_rank(ts_correlation(close, ts_mean(volume, 50), 18), 18))"
    )
    a["alpha132"] = "ts_mean(amount, 20)"
    a["alpha133"] = (
        "((20 - ts_highday(high, 20)) / 20) * 100 - ((20 - ts_lowday(low, 20)) / 20) * 100"
    )
    a["alpha134"] = "(close - ts_delay(close, 12)) / ts_delay(close, 12) * volume"
    a["alpha135"] = "ts_sma(ts_delay(close / ts_delay(close, 20), 1), 20, 1)"
    a["alpha136"] = f"(-1 * cs_rank(ts_delta({ret_expr}, 3))) * ts_correlation(open, volume, 10)"
    # alpha137: broken — skip
    a["alpha138"] = (
        "(cs_rank(ts_decay_linear(ts_delta(low * 0.7 + vwap * 0.3, 3), 20)) "
        "- ts_rank(ts_decay_linear(ts_rank(ts_correlation(ts_rank(low, 8), "
        "ts_rank(ts_mean(volume, 60), 17), 5), 19), 16), 7)) * -1"
    )
    a["alpha139"] = "-1 * ts_correlation(open, volume, 10)"
    a["alpha140"] = (
        "min(cs_rank(ts_decay_linear((cs_rank(open) + cs_rank(low) - cs_rank(high) - cs_rank(close)), 8)), "
        "ts_rank(ts_decay_linear(ts_correlation(ts_rank(close, 8), ts_rank(ts_mean(volume, 60), 20), 8), 7), 3))"
    )
    a["alpha141"] = "cs_rank(ts_correlation(cs_rank(high), cs_rank(ts_mean(volume, 15)), 9)) * -1"
    a["alpha142"] = (
        "(-1 * cs_rank(ts_rank(close, 10))) "
        "* cs_rank(ts_delta(ts_delta(close, 1), 1)) "
        "* cs_rank(ts_rank(volume / ts_mean(volume, 20), 5))"
    )
    # alpha143: SELF recursion — skipped
    a["alpha144"] = (
        "ts_sum(quesval(0, ts_delay(close, 1) - close, "
        "abs(close / ts_delay(close, 1) - 1) / amount, 0), 20) "
        "/ ts_sum(gt(ts_delay(close, 1), close), 20)"
    )
    a["alpha145"] = "(ts_mean(volume, 9) - ts_mean(volume, 26)) / ts_mean(volume, 12) * 100"
    a["alpha146"] = (
        "ts_mean((close / ts_delay(close, 1) - 1 - ts_sma(close / ts_delay(close, 1) - 1, 61, 2)), 20) "
        "* (close / ts_delay(close, 1) - 1 - ts_sma(close / ts_delay(close, 1) - 1, 61, 2)) "
        "/ ts_sma(power(close / ts_delay(close, 1) - 1 "
        "- ts_sma(close / ts_delay(close, 1) - 1, 61, 2), 2), 61, 2)"
    )
    # alpha147: Regbeta — skipped
    a["alpha148"] = (
        "quesval2(cs_rank(ts_correlation(open, ts_sum(ts_mean(volume, 60), 9), 6)), "
        "cs_rank(open - ts_min(open, 14)), 0, -1)"
    )
    # alpha149: benchmark data — skipped
    a["alpha150"] = "(close + high + low) / 3 * volume"
    a["alpha151"] = "ts_sma(close - ts_delay(close, 20), 20, 1)"
    a["alpha152"] = (
        "ts_sma(ts_mean(ts_delay(ts_sma(ts_delay(close / ts_delay(close, 9), 1), 9, 1), 1), 12) "
        "- ts_mean(ts_delay(ts_sma(ts_delay(close / ts_delay(close, 9), 1), 9, 1), 1), 26), 9, 1)"
    )
    a["alpha153"] = "(ts_mean(close, 3) + ts_mean(close, 6) + ts_mean(close, 12) + ts_mean(close, 24)) / 4"
    a["alpha154"] = (
        "quesval2(cs_rank(vwap - ts_min(vwap, 16)), "
        "cs_rank(ts_correlation(vwap, ts_mean(volume, 180), 18)), 1, 0)"
    )
    a["alpha155"] = (
        "ts_sma(volume, 13, 2) - ts_sma(volume, 27, 2) "
        "- ts_sma(ts_sma(volume, 13, 2) - ts_sma(volume, 27, 2), 10, 2)"
    )
    a["alpha156"] = (
        "max(cs_rank(ts_decay_linear(ts_delta(vwap, 5), 3)), "
        "cs_rank(ts_decay_linear((ts_delta(open * 0.15 + low * 0.85, 2) "
        "/ (open * 0.15 + low * 0.85)) * -1, 3))) * -1"
    )
    a["alpha157"] = (
        f"(ts_min(ts_product(cs_rank(cs_rank(log(ts_sum(ts_min(cs_rank(cs_rank((-1 * cs_rank(ts_delta(close - 1, 5))))), 2), 1)))), 1), 5) "
        f"+ ts_rank(ts_delay(-1 * ({ret_expr}), 6), 5))"
    )
    a["alpha158"] = "((high - ts_sma(close, 15, 2)) - (low - ts_sma(close, 15, 2))) / close"
    a["alpha159"] = (
        "((close - ts_sum(min(low, ts_delay(close, 1)), 6)) / ts_sum(max(high, ts_delay(close, 1)) - min(low, ts_delay(close, 1)), 6) * 12 * 24 "
        "+ (close - ts_sum(min(low, ts_delay(close, 1)), 12)) / ts_sum(max(high, ts_delay(close, 1)) - min(low, ts_delay(close, 1)), 12) * 6 * 24 "
        "+ (close - ts_sum(min(low, ts_delay(close, 1)), 24)) / ts_sum(max(high, ts_delay(close, 1)) - min(low, ts_delay(close, 1)), 24) * 6 * 24) "
        "* 100 / (6 * 12 + 6 * 24 + 12 * 24)"
    )
    a["alpha160"] = "ts_sma(quesval(0, ts_delay(close, 1) - close, ts_std(close, 20), 0), 20, 1)"
    a["alpha161"] = (
        "ts_mean(max(max(high - low, abs(ts_delay(close, 1) - high)), "
        "abs(ts_delay(close, 1) - low)), 12)"
    )
    a["alpha162"] = (
        "(ts_sma(max(close - ts_delay(close, 1), 0), 12, 1) "
        "/ ts_sma(abs(close - ts_delay(close, 1)), 12, 1) * 100 "
        "- ts_min(ts_sma(max(close - ts_delay(close, 1), 0), 12, 1) "
        "/ ts_sma(abs(close - ts_delay(close, 1)), 12, 1) * 100, 12)) "
        "/ (ts_max(ts_sma(max(close - ts_delay(close, 1), 0), 12, 1) "
        "/ ts_sma(abs(close - ts_delay(close, 1)), 12, 1) * 100, 12) "
        "- ts_min(ts_sma(max(close - ts_delay(close, 1), 0), 12, 1) "
        "/ ts_sma(abs(close - ts_delay(close, 1)), 12, 1) * 100, 12))"
    )
    a["alpha163"] = f"cs_rank(((-1 * ({ret_expr})) * ts_mean(volume, 20) * vwap * (high - close)))"
    a["alpha164"] = (
        "ts_sma((quesval2(close, ts_delay(close, 1), "
        "1 / abs(close - ts_delay(close, 1)), 1) "
        "- ts_min(quesval2(close, ts_delay(close, 1), "
        "1 / abs(close - ts_delay(close, 1)), 1), 12)) "
        "/ (high - low) * 100, 13, 2)"
    )
    # alpha165,166: Rowmax/Rowmin, broken — skipped
    a["alpha167"] = (
        "ts_sum(quesval(0, close - ts_delay(close, 1), close - ts_delay(close, 1), 0), 12)"
    )
    a["alpha168"] = "-1 * volume / ts_mean(volume, 20)"
    a["alpha169"] = (
        "ts_sma(ts_mean(ts_delay(ts_sma(close - ts_delay(close, 1), 9, 1), 1), 12) "
        "- ts_mean(ts_delay(ts_sma(close - ts_delay(close, 1), 9, 1), 1), 26), 10, 1)"
    )
    a["alpha170"] = (
        "(cs_rank(power(close, -1)) * volume / ts_mean(volume, 20) "
        "* high * cs_rank(high - close) / (ts_sum(high, 5) / 5)) "
        "- cs_rank(vwap - ts_delay(vwap, 5))"
    )
    a["alpha171"] = "-1 * (low - close) * power(open, 5) / ((close - high) * power(close, 5))"
    a["alpha172"] = (
        "ts_mean(abs(ts_sum(quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), "
        "quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, "
        "0, ts_delay(low, 1) - low), 0), 14) * 100 "
        "/ ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) "
        "- ts_sum(quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, "
        "quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), "
        "0, high - ts_delay(high, 1)), 0), 14) * 100 "
        "/ ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14)) "
        "/ (ts_sum(quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), "
        "quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, "
        "0, ts_delay(low, 1) - low), 0), 14) * 100 "
        "/ ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) "
        "+ ts_sum(quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, "
        "quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), "
        "0, high - ts_delay(high, 1)), 0), 14) * 100 "
        "/ ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14)) * 100, 6)"
    )
    a["alpha173"] = (
        "3 * ts_sma(close, 13, 2) - 2 * ts_sma(ts_sma(close, 13, 2), 13, 2) "
        "+ ts_sma(ts_sma(ts_sma(log(close), 13, 2), 13, 2), 13, 2)"
    )
    a["alpha174"] = "ts_sma(quesval(0, close - ts_delay(close, 1), ts_std(close, 20), 0), 20, 1)"
    a["alpha175"] = (
        "ts_mean(max(max(high - low, abs(ts_delay(close, 1) - high)), "
        "abs(ts_delay(close, 1) - low)), 6)"
    )
    a["alpha176"] = (
        "ts_correlation(cs_rank((close - ts_min(low, 12)) "
        "/ (ts_max(high, 12) - ts_min(low, 12))), cs_rank(volume), 6)"
    )
    a["alpha177"] = "((20 - ts_highday(high, 20)) / 20) * 100"
    a["alpha178"] = "(close - ts_delay(close, 1)) / ts_delay(close, 1) * volume"
    a["alpha179"] = (
        "cs_rank(ts_correlation(vwap, volume, 4)) "
        "* cs_rank(ts_correlation(cs_rank(low), cs_rank(ts_mean(volume, 50)), 12))"
    )
    a["alpha180"] = (
        "quesval2(ts_mean(volume, 20), volume, "
        "-1 * ts_rank(abs(ts_delta(close, 7)), 60) * sign(ts_delta(close, 7)), -1 * volume)"
    )
    # alpha181,182: benchmark data — skipped
    # alpha183: Rowmax/Rowmin — skipped
    a["alpha184"] = (
        "cs_rank(ts_correlation(ts_delay(open - close, 1), close, 200)) "
        "+ cs_rank(open - close)"
    )
    a["alpha185"] = "cs_rank(-1 * power(1 - open / close, 2))"
    a["alpha186"] = (
        "(ts_mean(abs(ts_sum(quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), "
        "quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, "
        "0, ts_delay(low, 1) - low), 0), 14) * 100 "
        "/ ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) "
        "- ts_sum(quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, "
        "quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), "
        "0, high - ts_delay(high, 1)), 0), 14) * 100 "
        "/ ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14)) "
        "/ (ts_sum(quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), "
        "quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, "
        "0, ts_delay(low, 1) - low), 0), 14) * 100 "
        "/ ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) "
        "+ ts_sum(quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, "
        "quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), "
        "0, high - ts_delay(high, 1)), 0), 14) * 100 "
        "/ ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14)) * 100, 6) "
        "+ ts_delay(ts_mean(abs(ts_sum(quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), "
        "quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, "
        "0, ts_delay(low, 1) - low), 0), 14) * 100 "
        "/ ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) "
        "- ts_sum(quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, "
        "quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), "
        "0, high - ts_delay(high, 1)), 0), 14) * 100 "
        "/ ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14)) "
        "/ (ts_sum(quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), "
        "quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, "
        "0, ts_delay(low, 1) - low), 0), 14) * 100 "
        "/ ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) "
        "+ ts_sum(quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, "
        "quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), "
        "0, high - ts_delay(high, 1)), 0), 14) * 100 "
        "/ ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14)) * 100, 6), 6)) / 2"
    )
    a["alpha187"] = (
        "ts_sum(quesval(0, open - ts_delay(open, 1), "
        "0, max(high - open, open - ts_delay(open, 1))), 20)"
    )
    a["alpha188"] = (
        "(high - low - ts_sma(high - low, 11, 2)) / ts_sma(high - low, 11, 2) * 100"
    )
    a["alpha189"] = "ts_mean(abs(close - ts_mean(close, 6)), 6)"
    # alpha190: extremely complex — skip
    a["alpha191"] = "(ts_correlation(ts_mean(volume, 20), low, 5) + (high + low) / 2) - close"

    return a


SKIPPED = {
    "alpha021": "Regbeta",
    "alpha030": "Regbeta",
    "alpha055": "broken formula",
    "alpha075": "benchmark data",
    "alpha116": "Regbeta",
    "alpha137": "broken formula",
    "alpha143": "SELF recursion",
    "alpha147": "Regbeta",
    "alpha149": "benchmark data",
    "alpha165": "Rowmax/Rowmin",
    "alpha166": "broken formula",
    "alpha181": "benchmark data",
    "alpha182": "benchmark data",
    "alpha183": "Rowmax/Rowmin",
    "alpha190": "extremely complex",
}


def main():
    alphas = build_alphas()
    print(f"Alphas defined: {len(alphas)}")
    print(f"Alphas skipped: {len(SKIPPED)}")
    for name, reason in sorted(SKIPPED.items()):
        print(f"  {name}: {reason}")

    ch = ClickHouseSource.from_env()
    dl = DataLayer(ch)
    dl.set_pre_filter("2024-01-01:2025-01-01 symbols not like '%BJ'")

    registry = FactorRegistry("default")
    names = []
    for name, expr in alphas.items():
        try:
            registry.register(name, expr)
            names.append(name)
        except Exception as e:
            print(f"  REGISTER FAIL {name}: {e}")

    print(f"Registered {len(names)} factors, computing...")
    factor_mats, prices = registry.compute_factor_matrices_1d(dl)

    mats = []
    for name in names:
        if name not in factor_mats:
            continue
        mats.append(factor_mats[name])

    print(f"Valid factors: {len(mats)}")

    if len(mats) < 2:
        print("Not enough valid factors")
        return

    print("\nCombining factors (equal-weight)...")
    combined = FactorCombiner.equal_weight(mats)

    print("Running backtest...")
    engine = PyBacktestEngine(10, "equal", 1, 1, 0.001)
    result = engine.run_with_prices(combined, prices)

    print(f"\n── Backtest Results ({len(mats)} factors) ──")
    print(f"  IC mean:   {result.ic_mean:.6f}")
    print(f"  IC IR:     {result.ic_ir:.4f}")
    print(f"  Total ret: {result.total_return:.4f}")
    print(f"  Sharpe:    {result.sharpe_ratio:.4f}")
    print(f"  Max DD:    {result.max_drawdown:.4f}")
    print(f"  Turnover:  {result.turnover:.4f}")


if __name__ == "__main__":
    main()
