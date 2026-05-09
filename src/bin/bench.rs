use alfars::data::clickhouse::ClickHouseSource;
use alfars::lab::AlfarsLab;
use std::time::Instant;
const ALPHAS: &[(&str, &str)] = &[
    (
        "a101_alpha1",
        "(cs_rank(ts_argmax(power(quesval(0, close / ts_delay(close, 1) - 1, close, ts_std(close / ts_delay(close, 1) - 1, 20)), 2.0), 5)) - 0.5)",
    ),
    (
        "a101_alpha2",
        "(-1) * ts_correlation(cs_rank(ts_delta(log(vol), 2)), cs_rank((close - open) / open), 6)",
    ),
    (
        "a101_alpha3",
        "ts_correlation(cs_rank(open), cs_rank(vol), 10) * -1",
    ),
    ("a101_alpha4", "-1 * ts_rank(cs_rank(low), 9)"),
    (
        "a101_alpha5",
        "cs_rank((open - (ts_sum(vwap, 10) / 10))) * (-1 * abs(cs_rank((close - vwap))))",
    ),
    ("a101_alpha6", "(-1) * ts_correlation(open, vol, 10)"),
    (
        "a101_alpha7",
        "quesval2(ts_mean(vol, 20), vol, (-1 * ts_rank(abs(close - ts_delay(close, 7)), 60)) * sign(ts_delta(close, 7)), -1)",
    ),
    (
        "a101_alpha8",
        "-1 * cs_rank(((ts_sum(open, 5) * ts_sum(close / ts_delay(close, 1) - 1, 5)) - ts_delay((ts_sum(open, 5) * ts_sum(close / ts_delay(close, 1) - 1, 5)), 10)))",
    ),
    (
        "a101_alpha9",
        "quesval(0, ts_min(ts_delta(close, 1), 5), ts_delta(close, 1), quesval(0, ts_max(ts_delta(close, 1), 5), (-1 * ts_delta(close, 1)), ts_delta(close, 1)))",
    ),
    (
        "a101_alpha10",
        "cs_rank(quesval(0, ts_min(ts_delta(close, 1), 4), ts_delta(close, 1), quesval(0, ts_max(ts_delta(close, 1), 4), (-1 * ts_delta(close, 1)), ts_delta(close, 1))))",
    ),
    (
        "a101_alpha11",
        "(cs_rank(ts_max(vwap - close, 3)) + cs_rank(ts_min(vwap - close, 3))) * cs_rank(ts_delta(vol, 3))",
    ),
    (
        "a101_alpha12",
        "sign(ts_delta(vol, 1)) * (-1 * ts_delta(close, 1))",
    ),
    (
        "a101_alpha13",
        "-1 * cs_rank(ts_covariance(cs_rank(close), cs_rank(vol), 5))",
    ),
    (
        "a101_alpha14",
        "(-1 * cs_rank((close / ts_delay(close, 1) - 1) - ts_delay(close / ts_delay(close, 1) - 1, 3))) * ts_correlation(open, vol, 10)",
    ),
    (
        "a101_alpha15",
        "-1 * ts_sum(cs_rank(ts_correlation(cs_rank(high), cs_rank(vol), 3)), 3)",
    ),
    (
        "a101_alpha16",
        "-1 * cs_rank(ts_covariance(cs_rank(high), cs_rank(vol), 5))",
    ),
    (
        "a101_alpha17",
        "(-1 * cs_rank(ts_rank(close, 10))) * cs_rank(close - 2 * ts_delay(close, 1) + ts_delay(close, 2)) * cs_rank(ts_rank(vol / ts_mean(vol, 20), 5))",
    ),
    (
        "a101_alpha18",
        "-1 * cs_rank((ts_std(abs(close - open), 5) + (close - open)) + ts_correlation(close, open, 10))",
    ),
    (
        "a101_alpha19",
        "(-1 * sign(ts_delta(close, 7) + (close - ts_delay(close, 7)))) * (cs_rank(ts_sum(close / ts_delay(close, 1) - 1, 250) + 1) + 1)",
    ),
    (
        "a101_alpha20",
        "(-1 * cs_rank(open - ts_delay(high, 1))) * cs_rank(open - ts_delay(close, 1)) * cs_rank(open - ts_delay(low, 1))",
    ),
    (
        "a101_alpha21",
        "quesval2((ts_mean(close, 8) + ts_std(close, 8)), ts_mean(close, 2), -1, quesval2(ts_mean(close, 2), (ts_mean(close, 8) - ts_std(close, 8)), 1, quesval(1, (vol / ts_mean(vol, 20)), 1, -1)))",
    ),
    (
        "a101_alpha22",
        "-1 * ts_delta(ts_correlation(high, vol, 5), 5) * cs_rank(ts_std(close, 20))",
    ),
    (
        "a101_alpha23",
        "quesval2(ts_mean(high, 20), high, -1 * ts_delta(high, 2), 0)",
    ),
    (
        "a101_alpha24",
        "quesval(0.05, ts_delta(ts_sum(close, 100) / 100, 100) / ts_delay(close, 100), (-1 * ts_delta(close, 3)), (-1 * (close - ts_min(close, 100))))",
    ),
    (
        "a101_alpha25",
        "cs_rank((-1 * close / ts_delay(close, 1) - 1) * ts_mean(vol, 20) * vwap * (high - close))",
    ),
    (
        "a101_alpha26",
        "-1 * ts_max(ts_correlation(ts_rank(vol, 5), ts_rank(high, 5), 5), 3)",
    ),
    (
        "a101_alpha27",
        "quesval(0.5, cs_rank(ts_mean(ts_correlation(cs_rank(vol), cs_rank(vwap), 6), 2)), -1, 1)",
    ),
    (
        "a101_alpha28",
        "cs_scale(ts_correlation(ts_mean(vol, 20), low, 5) + (high + low) / 2 - close)",
    ),
    (
        "a101_alpha29",
        "ts_min(ts_product(cs_rank(cs_rank(cs_scale(log(ts_sum(ts_min(cs_rank(cs_rank((-1 * cs_rank(ts_delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(ts_delay((-1 * close / ts_delay(close, 1) - 1), 6), 5)",
    ),
    (
        "a101_alpha30",
        "((cs_rank(sign(close - ts_delay(close, 1)) + sign(ts_delay(close, 1) - ts_delay(close, 2)) + sign(ts_delay(close, 2) - ts_delay(close, 3))) * -1 + 1) * ts_sum(vol, 5)) / ts_sum(vol, 20)",
    ),
    (
        "a101_alpha31",
        "(cs_rank(cs_rank(cs_rank(ts_decay_linear((-1) * cs_rank(cs_rank(ts_delta(close, 10))), 10)))) + cs_rank((-1) * ts_delta(close, 3))) + sign(cs_scale(ts_correlation(ts_mean(vol, 20), low, 12)))",
    ),
    (
        "a101_alpha32",
        "cs_scale((ts_sum(close, 7) / 7 - close)) + (20 * cs_scale(ts_correlation(vwap, ts_delay(close, 5), 230)))",
    ),
    ("a101_alpha33", "cs_rank((-1) * (open / close * -1 + 1))"),
    (
        "a101_alpha34",
        "cs_rank((cs_rank(ts_std(close / ts_delay(close, 1) - 1, 2) / ts_std(close / ts_delay(close, 1) - 1, 5)) * -1 + 1) + (cs_rank(ts_delta(close, 1)) * -1 + 1))",
    ),
    (
        "a101_alpha35",
        "(ts_rank(vol, 32) * (ts_rank((close + high - low), 16) * -1 + 1)) * (ts_rank(close / ts_delay(close, 1) - 1, 32) * -1 + 1)",
    ),
    (
        "a101_alpha36",
        "((((2.21 * cs_rank(ts_correlation((close - open), ts_delay(vol, 1), 15))) + (0.7 * cs_rank((open - close)))) + (0.73 * cs_rank(ts_rank(ts_delay((-1) * close / ts_delay(close, 1) - 1, 6), 5)))) + cs_rank(abs(ts_correlation(vwap, ts_mean(vol, 20), 6)))) + (0.6 * cs_rank(((ts_sum(close, 200) / 200 - open) * (close - open))))",
    ),
    (
        "a101_alpha37",
        "cs_rank(ts_correlation(ts_delay((open - close), 1), close, 200)) + cs_rank((open - close))",
    ),
    (
        "a101_alpha38",
        "((-1) * cs_rank(ts_rank(close, 10))) * cs_rank((close / open))",
    ),
    (
        "a101_alpha39",
        "((-1) * cs_rank((ts_delta(close, 7) * (cs_rank(ts_decay_linear((vol / ts_mean(vol, 20)), 9)) * -1 + 1)))) * (cs_rank(ts_sum(close / ts_delay(close, 1) - 1, 250)) + 1)",
    ),
    (
        "a101_alpha40",
        "((-1) * cs_rank(ts_std(high, 10))) * ts_correlation(high, vol, 10)",
    ),
    ("a101_alpha41", "power((high * low), 0.5) - vwap"),
    (
        "a101_alpha42",
        "cs_rank((vwap - close)) / cs_rank((vwap + close))",
    ),
    (
        "a101_alpha43",
        "ts_rank((vol / ts_mean(vol, 20)), 20) * ts_rank((-1) * ts_delta(close, 7), 8)",
    ),
    (
        "a101_alpha44",
        "(-1) * ts_correlation(high, cs_rank(vol), 5)",
    ),
    (
        "a101_alpha45",
        "(-1) * cs_rank(ts_sum(ts_delay(close, 5), 20) / 20) * ts_correlation(close, vol, 2) * cs_rank(ts_correlation(ts_sum(close, 5), ts_sum(close, 20), 2))",
    ),
    (
        "a101_alpha46",
        "quesval(0.25, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), -1, quesval(0, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), (-1) * (close - ts_delay(close, 1)), 1))",
    ),
    (
        "a101_alpha47",
        "((cs_rank(power(close, -1)) * vol / ts_mean(vol, 20)) * (high * cs_rank(high - close)) / (ts_sum(high, 5) / 5)) - cs_rank(vwap - ts_delay(vwap, 5))",
    ),
    (
        "a101_alpha49",
        "quesval(-0.1, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), (-1) * (close - ts_delay(close, 1)), 1)",
    ),
    (
        "a101_alpha50",
        "(-1) * ts_max(cs_rank(ts_correlation(cs_rank(vol), cs_rank(vwap), 5)), 5)",
    ),
    (
        "a101_alpha51",
        "quesval(-0.05, ((ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10), (-1) * (close - ts_delay(close, 1)), 1)",
    ),
    (
        "a101_alpha52",
        "(((-1) * ts_min(low, 5)) + ts_delay(ts_min(low, 5), 5)) * cs_rank((ts_sum(close / ts_delay(close, 1) - 1, 240) - ts_sum(close / ts_delay(close, 1) - 1, 20)) / 220) * ts_rank(vol, 5)",
    ),
    (
        "a101_alpha53",
        "(-1) * ts_delta(((close - low) - (high - close)) / (close - low), 9)",
    ),
    (
        "a101_alpha54",
        "((-1) * ((low - close) * power(open, 5))) / ((low - high) * power(close, 5))",
    ),
    (
        "a101_alpha55",
        "(-1) * ts_correlation(cs_rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), cs_rank(vol), 6)",
    ),
    (
        "a101_alpha57",
        "-1 * ((close - vwap) / ts_decay_linear(cs_rank(ts_argmax(close, 30)), 2))",
    ),
    (
        "a101_alpha60",
        "- 1 * ((2 * cs_scale(cs_rank((((close - low) - (high - close)) / (high - low)) * vol))) - cs_scale(cs_rank(ts_argmax(close, 10))))",
    ),
    (
        "a101_alpha61",
        "quesval2(cs_rank(vwap - ts_min(vwap, 16)), cs_rank(ts_correlation(vwap, ts_mean(vol, 180), 18)), 1, 0)",
    ),
    (
        "a101_alpha62",
        "lt(cs_rank(ts_correlation(vwap, ts_sum(ts_mean(vol, 20), 22), 10)), cs_rank(lt(cs_rank(open) + cs_rank(open), cs_rank((high + low) / 2) + cs_rank(high)))) * -1",
    ),
    (
        "a101_alpha64",
        "lt(cs_rank(ts_correlation(ts_sum(((open * 0.178404) + (low * (1 - 0.178404))), 13), ts_sum(ts_mean(vol, 120), 13), 17)), cs_rank(ts_delta((((high + low) / 2 * 0.178404) + (vwap * (1 - 0.178404))), 4))) * -1",
    ),
    (
        "a101_alpha65",
        "lt(cs_rank(ts_correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), ts_sum(ts_mean(vol, 60), 9), 6)), cs_rank(open - ts_min(open, 14))) * -1",
    ),
    (
        "a101_alpha66",
        "(cs_rank(ts_decay_linear(ts_delta(vwap, 4), 7)) + ts_rank(ts_decay_linear((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2)), 11), 7)) * -1",
    ),
    (
        "a101_alpha68",
        "lt(ts_rank(ts_correlation(cs_rank(high), cs_rank(ts_mean(vol, 15)), 9), 14), cs_rank(ts_delta((close * 0.518371 + low * (1 - 0.518371)), 1))) * -1",
    ),
    (
        "a101_alpha71",
        "gt(ts_rank(ts_decay_linear(ts_correlation(ts_rank(close, 3), ts_rank(ts_mean(vol, 180), 12), 18), 4), 16), ts_rank(ts_decay_linear(power(cs_rank((low + open) - (vwap + vwap)), 2), 16), 4))",
    ),
    (
        "a101_alpha72",
        "cs_rank(ts_decay_linear(ts_correlation((high + low) / 2, ts_mean(vol, 40), 9), 10)) / cs_rank(ts_decay_linear(ts_correlation(ts_rank(vwap, 4), ts_rank(vol, 19), 7), 3))",
    ),
    (
        "a101_alpha73",
        "gt(cs_rank(ts_decay_linear(ts_delta(vwap, 5), 3)), ts_rank(ts_decay_linear((ts_delta(open * 0.147155 + low * 0.852845, 2) / (open * 0.147155 + low * 0.852845)) * -1, 3), 17)) * -1",
    ),
    (
        "a101_alpha74",
        "quesval2(cs_rank(ts_correlation(close, ts_sum(ts_mean(vol, 30), 37), 15)), cs_rank(ts_correlation(cs_rank(high * 0.0261661 + vwap * 0.9738339), cs_rank(vol), 11)), 1, 0) * -1",
    ),
    (
        "a101_alpha75",
        "quesval2(cs_rank(ts_correlation(vwap, vol, 4)), cs_rank(ts_correlation(cs_rank(low), cs_rank(ts_mean(vol, 50)), 12)), 1, 0)",
    ),
    (
        "a101_alpha77",
        "lt(cs_rank(ts_decay_linear((((high + low) / 2 + high) - (vwap + high)), 20)), cs_rank(ts_decay_linear(ts_correlation((high + low) / 2, ts_mean(vol, 40), 3), 6)))",
    ),
    (
        "a101_alpha78",
        "power(cs_rank(ts_correlation(ts_sum((low * 0.352233) + (vwap * (1 - 0.352233)), 20), ts_sum(ts_mean(vol, 40), 20), 7)), cs_rank(ts_correlation(cs_rank(vwap), cs_rank(vol), 6)))",
    ),
    (
        "a101_alpha81",
        "quesval2(cs_rank(log(ts_product(cs_rank(power(cs_rank(ts_correlation(vwap, ts_sum(ts_mean(vol, 10), 50), 8)), 4)), 15))), cs_rank(ts_correlation(cs_rank(vwap), cs_rank(vol), 5)), 1, 0) * -1",
    ),
    (
        "a101_alpha83",
        "(cs_rank(ts_delay((high - low) / (ts_sum(close, 5) / 5), 2)) * cs_rank(cs_rank(vol))) / (((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close))",
    ),
    (
        "a101_alpha84",
        "power(ts_rank(vwap - ts_max(vwap, 15), 21), ts_delta(close, 5))",
    ),
    (
        "a101_alpha85",
        "power(cs_rank(ts_correlation(high * 0.876703 + close * 0.123297, ts_mean(vol, 30), 10)), cs_rank(ts_correlation(ts_rank((high + low) / 2, 4), ts_rank(vol, 10), 7)))",
    ),
    (
        "a101_alpha86",
        "quesval2(ts_rank(ts_correlation(close, ts_sum(ts_mean(vol, 20), 15), 6), 20), cs_rank((open + close) - (vwap + open)), 1, 0) * -1",
    ),
    (
        "a101_alpha88",
        "lt(cs_rank(ts_decay_linear((cs_rank(open) + cs_rank(low)) - (cs_rank(high) + cs_rank(close)), 8)), ts_rank(ts_decay_linear(ts_correlation(ts_rank(close, 8), ts_rank(ts_mean(vol, 60), 21), 8), 7), 3))",
    ),
    (
        "a101_alpha92",
        "lt(ts_rank(ts_decay_linear(quesval2(((high + low) / 2 + close), (low + open), 1, 0), 15), 19), ts_rank(ts_decay_linear(ts_correlation(cs_rank(low), cs_rank(ts_mean(vol, 30)), 8), 7), 7))",
    ),
    (
        "a101_alpha94",
        "power(cs_rank(vwap - ts_min(vwap, 12)), ts_rank(ts_correlation(ts_rank(vwap, 20), ts_rank(ts_mean(vol, 60), 4), 18), 3)) * -1",
    ),
    (
        "a101_alpha95",
        "quesval2(cs_rank(open - ts_min(open, 12)), ts_rank(power(cs_rank(ts_correlation(ts_sum((high + low) / 2, 19), ts_sum(ts_mean(vol, 40), 19), 13)), 5), 12), 1, 0)",
    ),
    (
        "a101_alpha96",
        "gt(ts_rank(ts_decay_linear(ts_correlation(cs_rank(vwap), cs_rank(vol), 4), 4), 8), ts_rank(ts_decay_linear(ts_argmax(ts_correlation(ts_rank(close, 7), ts_rank(ts_mean(vol, 60), 4), 4), 13), 14), 13)) * -1",
    ),
    (
        "a101_alpha98",
        "cs_rank(ts_decay_linear(ts_correlation(vwap, ts_sum(ts_mean(vol, 5), 26), 5), 7)) - cs_rank(ts_decay_linear(ts_rank(ts_argmin(ts_correlation(cs_rank(open), cs_rank(ts_mean(vol, 15)), 21), 9), 7), 8))",
    ),
    (
        "a101_alpha99",
        "quesval2(cs_rank(ts_correlation(ts_sum((high + low) / 2, 20), ts_sum(ts_mean(vol, 60), 20), 9)), cs_rank(ts_correlation(low, vol, 6)), 1, 0) * -1",
    ),
    ("a101_alpha101", "((close - open) / ((high - low) + 0.001))"),
    ("a158_kmid", "(close - open) / open"),
    ("a158_klen", "(high - low) / open"),
    ("a158_kmid_2", "(close - open) / (high - low + 1e-12)"),
    ("a158_kup", "(high - gt(open, close)) / open"),
    (
        "a158_kup_2",
        "(high - gt(open, close)) / (high - low + 1e-12)",
    ),
    ("a158_klow", "(lt(open, close) - low) / open"),
    (
        "a158_klow_2",
        "(lt(open, close) - low) / (high - low + 1e-12)",
    ),
    ("a158_ksft", "(close * 2 - high - low) / open"),
    (
        "a158_ksft_2",
        "(close * 2 - high - low) / (high - low + 1e-12)",
    ),
    ("a158_open_0", "open / close"),
    ("a158_high_0", "high / close"),
    ("a158_low_0", "low / close"),
    ("a158_vwap_0", "vwap / close"),
    ("a158_roc_5", "ts_delay(close, 5) / close"),
    ("a158_ma_5", "ts_mean(close, 5) / close"),
    ("a158_std_5", "ts_std(close, 5) / close"),
    ("a158_beta_5", "ts_slope(close, 5) / close"),
    ("a158_rsqr_5", "ts_rsquare(close, 5)"),
    ("a158_resi_5", "ts_resi(close, 5) / close"),
    ("a158_max_5", "ts_max(high, 5) / close"),
    ("a158_min_5", "ts_min(low, 5) / close"),
    ("a158_qtlu_5", "ts_quantile(close, 5, 0.8) / close"),
    ("a158_qtld_5", "ts_quantile(close, 5, 0.2) / close"),
    ("a158_rank_5", "ts_rank(close, 5)"),
    (
        "a158_rsv_5",
        "(close - ts_min(low, 5)) / (ts_max(high, 5) - ts_min(low, 5) + 1e-12)",
    ),
    ("a158_imax_5", "ts_argmax(high, 5) / 5"),
    ("a158_imin_5", "ts_argmin(low, 5) / 5"),
    (
        "a158_imxd_5",
        "(ts_argmax(high, 5) - ts_argmin(low, 5)) / 5",
    ),
    ("a158_corr_5", "ts_correlation(close, log(volume + 1), 5)"),
    (
        "a158_cord_5",
        "ts_correlation(close / ts_delay(close, 1), log(volume / ts_delay(volume, 1) + 1), 5)",
    ),
    ("a158_cntp_5", "ts_mean(gt(close, ts_delay(close, 1)), 5)"),
    ("a158_cntn_5", "ts_mean(lt(close, ts_delay(close, 1)), 5)"),
    (
        "a158_cntd_5",
        "ts_mean(gt(close, ts_delay(close, 1)), 5) - ts_mean(lt(close, ts_delay(close, 1)), 5)",
    ),
    (
        "a158_sump_5",
        "ts_sum(gt(close - ts_delay(close, 1), 0), 5) / (ts_sum(abs(close - ts_delay(close, 1)), 5) + 1e-12)",
    ),
    (
        "a158_sumn_5",
        "ts_sum(gt(ts_delay(close, 1) - close, 0), 5) / (ts_sum(abs(close - ts_delay(close, 1)), 5) + 1e-12)",
    ),
    (
        "a158_sumd_5",
        "(ts_sum(gt(close - ts_delay(close, 1), 0), 5) - ts_sum(gt(ts_delay(close, 1) - close, 0), 5)) / (ts_sum(abs(close - ts_delay(close, 1)), 5) + 1e-12)",
    ),
    ("a158_vma_5", "ts_mean(volume, 5) / (volume + 1e-12)"),
    ("a158_vstd_5", "ts_std(volume, 5) / (volume + 1e-12)"),
    (
        "a158_wvma_5",
        "ts_std(abs(close / ts_delay(close, 1) - 1) * volume, 5) / (ts_mean(abs(close / ts_delay(close, 1) - 1) * volume, 5) + 1e-12)",
    ),
    (
        "a158_vsump_5",
        "ts_sum(gt(volume - ts_delay(volume, 1), 0), 5) / (ts_sum(abs(volume - ts_delay(volume, 1)), 5) + 1e-12)",
    ),
    (
        "a158_vsumn_5",
        "ts_sum(gt(ts_delay(volume, 1) - volume, 0), 5) / (ts_sum(abs(volume - ts_delay(volume, 1)), 5) + 1e-12)",
    ),
    (
        "a158_vsumd_5",
        "(ts_sum(gt(volume - ts_delay(volume, 1), 0), 5) - ts_sum(gt(ts_delay(volume, 1) - volume, 0), 5)) / (ts_sum(abs(volume - ts_delay(volume, 1)), 5) + 1e-12)",
    ),
    ("a158_roc_10", "ts_delay(close, 10) / close"),
    ("a158_ma_10", "ts_mean(close, 10) / close"),
    ("a158_std_10", "ts_std(close, 10) / close"),
    ("a158_beta_10", "ts_slope(close, 10) / close"),
    ("a158_rsqr_10", "ts_rsquare(close, 10)"),
    ("a158_resi_10", "ts_resi(close, 10) / close"),
    ("a158_max_10", "ts_max(high, 10) / close"),
    ("a158_min_10", "ts_min(low, 10) / close"),
    ("a158_qtlu_10", "ts_quantile(close, 10, 0.8) / close"),
    ("a158_qtld_10", "ts_quantile(close, 10, 0.2) / close"),
    ("a158_rank_10", "ts_rank(close, 10)"),
    (
        "a158_rsv_10",
        "(close - ts_min(low, 10)) / (ts_max(high, 10) - ts_min(low, 10) + 1e-12)",
    ),
    ("a158_imax_10", "ts_argmax(high, 10) / 10"),
    ("a158_imin_10", "ts_argmin(low, 10) / 10"),
    (
        "a158_imxd_10",
        "(ts_argmax(high, 10) - ts_argmin(low, 10)) / 10",
    ),
    ("a158_corr_10", "ts_correlation(close, log(volume + 1), 10)"),
    (
        "a158_cord_10",
        "ts_correlation(close / ts_delay(close, 1), log(volume / ts_delay(volume, 1) + 1), 10)",
    ),
    ("a158_cntp_10", "ts_mean(gt(close, ts_delay(close, 1)), 10)"),
    ("a158_cntn_10", "ts_mean(lt(close, ts_delay(close, 1)), 10)"),
    (
        "a158_cntd_10",
        "ts_mean(gt(close, ts_delay(close, 1)), 10) - ts_mean(lt(close, ts_delay(close, 1)), 10)",
    ),
    (
        "a158_sump_10",
        "ts_sum(gt(close - ts_delay(close, 1), 0), 10) / (ts_sum(abs(close - ts_delay(close, 1)), 10) + 1e-12)",
    ),
    (
        "a158_sumn_10",
        "ts_sum(gt(ts_delay(close, 1) - close, 0), 10) / (ts_sum(abs(close - ts_delay(close, 1)), 10) + 1e-12)",
    ),
    (
        "a158_sumd_10",
        "(ts_sum(gt(close - ts_delay(close, 1), 0), 10) - ts_sum(gt(ts_delay(close, 1) - close, 0), 10)) / (ts_sum(abs(close - ts_delay(close, 1)), 10) + 1e-12)",
    ),
    ("a158_vma_10", "ts_mean(volume, 10) / (volume + 1e-12)"),
    ("a158_vstd_10", "ts_std(volume, 10) / (volume + 1e-12)"),
    (
        "a158_wvma_10",
        "ts_std(abs(close / ts_delay(close, 1) - 1) * volume, 10) / (ts_mean(abs(close / ts_delay(close, 1) - 1) * volume, 10) + 1e-12)",
    ),
    (
        "a158_vsump_10",
        "ts_sum(gt(volume - ts_delay(volume, 1), 0), 10) / (ts_sum(abs(volume - ts_delay(volume, 1)), 10) + 1e-12)",
    ),
    (
        "a158_vsumn_10",
        "ts_sum(gt(ts_delay(volume, 1) - volume, 0), 10) / (ts_sum(abs(volume - ts_delay(volume, 1)), 10) + 1e-12)",
    ),
    (
        "a158_vsumd_10",
        "(ts_sum(gt(volume - ts_delay(volume, 1), 0), 10) - ts_sum(gt(ts_delay(volume, 1) - volume, 0), 10)) / (ts_sum(abs(volume - ts_delay(volume, 1)), 10) + 1e-12)",
    ),
    ("a158_roc_20", "ts_delay(close, 20) / close"),
    ("a158_ma_20", "ts_mean(close, 20) / close"),
    ("a158_std_20", "ts_std(close, 20) / close"),
    ("a158_beta_20", "ts_slope(close, 20) / close"),
    ("a158_rsqr_20", "ts_rsquare(close, 20)"),
    ("a158_resi_20", "ts_resi(close, 20) / close"),
    ("a158_max_20", "ts_max(high, 20) / close"),
    ("a158_min_20", "ts_min(low, 20) / close"),
    ("a158_qtlu_20", "ts_quantile(close, 20, 0.8) / close"),
    ("a158_qtld_20", "ts_quantile(close, 20, 0.2) / close"),
    ("a158_rank_20", "ts_rank(close, 20)"),
    (
        "a158_rsv_20",
        "(close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20) + 1e-12)",
    ),
    ("a158_imax_20", "ts_argmax(high, 20) / 20"),
    ("a158_imin_20", "ts_argmin(low, 20) / 20"),
    (
        "a158_imxd_20",
        "(ts_argmax(high, 20) - ts_argmin(low, 20)) / 20",
    ),
    ("a158_corr_20", "ts_correlation(close, log(volume + 1), 20)"),
    (
        "a158_cord_20",
        "ts_correlation(close / ts_delay(close, 1), log(volume / ts_delay(volume, 1) + 1), 20)",
    ),
    ("a158_cntp_20", "ts_mean(gt(close, ts_delay(close, 1)), 20)"),
    ("a158_cntn_20", "ts_mean(lt(close, ts_delay(close, 1)), 20)"),
    (
        "a158_cntd_20",
        "ts_mean(gt(close, ts_delay(close, 1)), 20) - ts_mean(lt(close, ts_delay(close, 1)), 20)",
    ),
    (
        "a158_sump_20",
        "ts_sum(gt(close - ts_delay(close, 1), 0), 20) / (ts_sum(abs(close - ts_delay(close, 1)), 20) + 1e-12)",
    ),
    (
        "a158_sumn_20",
        "ts_sum(gt(ts_delay(close, 1) - close, 0), 20) / (ts_sum(abs(close - ts_delay(close, 1)), 20) + 1e-12)",
    ),
    (
        "a158_sumd_20",
        "(ts_sum(gt(close - ts_delay(close, 1), 0), 20) - ts_sum(gt(ts_delay(close, 1) - close, 0), 20)) / (ts_sum(abs(close - ts_delay(close, 1)), 20) + 1e-12)",
    ),
    ("a158_vma_20", "ts_mean(volume, 20) / (volume + 1e-12)"),
    ("a158_vstd_20", "ts_std(volume, 20) / (volume + 1e-12)"),
    (
        "a158_wvma_20",
        "ts_std(abs(close / ts_delay(close, 1) - 1) * volume, 20) / (ts_mean(abs(close / ts_delay(close, 1) - 1) * volume, 20) + 1e-12)",
    ),
    (
        "a158_vsump_20",
        "ts_sum(gt(volume - ts_delay(volume, 1), 0), 20) / (ts_sum(abs(volume - ts_delay(volume, 1)), 20) + 1e-12)",
    ),
    (
        "a158_vsumn_20",
        "ts_sum(gt(ts_delay(volume, 1) - volume, 0), 20) / (ts_sum(abs(volume - ts_delay(volume, 1)), 20) + 1e-12)",
    ),
    (
        "a158_vsumd_20",
        "(ts_sum(gt(volume - ts_delay(volume, 1), 0), 20) - ts_sum(gt(ts_delay(volume, 1) - volume, 0), 20)) / (ts_sum(abs(volume - ts_delay(volume, 1)), 20) + 1e-12)",
    ),
    ("a158_roc_30", "ts_delay(close, 30) / close"),
    ("a158_ma_30", "ts_mean(close, 30) / close"),
    ("a158_std_30", "ts_std(close, 30) / close"),
    ("a158_beta_30", "ts_slope(close, 30) / close"),
    ("a158_rsqr_30", "ts_rsquare(close, 30)"),
    ("a158_resi_30", "ts_resi(close, 30) / close"),
    ("a158_max_30", "ts_max(high, 30) / close"),
    ("a158_min_30", "ts_min(low, 30) / close"),
    ("a158_qtlu_30", "ts_quantile(close, 30, 0.8) / close"),
    ("a158_qtld_30", "ts_quantile(close, 30, 0.2) / close"),
    ("a158_rank_30", "ts_rank(close, 30)"),
    (
        "a158_rsv_30",
        "(close - ts_min(low, 30)) / (ts_max(high, 30) - ts_min(low, 30) + 1e-12)",
    ),
    ("a158_imax_30", "ts_argmax(high, 30) / 30"),
    ("a158_imin_30", "ts_argmin(low, 30) / 30"),
    (
        "a158_imxd_30",
        "(ts_argmax(high, 30) - ts_argmin(low, 30)) / 30",
    ),
    ("a158_corr_30", "ts_correlation(close, log(volume + 1), 30)"),
    (
        "a158_cord_30",
        "ts_correlation(close / ts_delay(close, 1), log(volume / ts_delay(volume, 1) + 1), 30)",
    ),
    ("a158_cntp_30", "ts_mean(gt(close, ts_delay(close, 1)), 30)"),
    ("a158_cntn_30", "ts_mean(lt(close, ts_delay(close, 1)), 30)"),
    (
        "a158_cntd_30",
        "ts_mean(gt(close, ts_delay(close, 1)), 30) - ts_mean(lt(close, ts_delay(close, 1)), 30)",
    ),
    (
        "a158_sump_30",
        "ts_sum(gt(close - ts_delay(close, 1), 0), 30) / (ts_sum(abs(close - ts_delay(close, 1)), 30) + 1e-12)",
    ),
    (
        "a158_sumn_30",
        "ts_sum(gt(ts_delay(close, 1) - close, 0), 30) / (ts_sum(abs(close - ts_delay(close, 1)), 30) + 1e-12)",
    ),
    (
        "a158_sumd_30",
        "(ts_sum(gt(close - ts_delay(close, 1), 0), 30) - ts_sum(gt(ts_delay(close, 1) - close, 0), 30)) / (ts_sum(abs(close - ts_delay(close, 1)), 30) + 1e-12)",
    ),
    ("a158_vma_30", "ts_mean(volume, 30) / (volume + 1e-12)"),
    ("a158_vstd_30", "ts_std(volume, 30) / (volume + 1e-12)"),
    (
        "a158_wvma_30",
        "ts_std(abs(close / ts_delay(close, 1) - 1) * volume, 30) / (ts_mean(abs(close / ts_delay(close, 1) - 1) * volume, 30) + 1e-12)",
    ),
    (
        "a158_vsump_30",
        "ts_sum(gt(volume - ts_delay(volume, 1), 0), 30) / (ts_sum(abs(volume - ts_delay(volume, 1)), 30) + 1e-12)",
    ),
    (
        "a158_vsumn_30",
        "ts_sum(gt(ts_delay(volume, 1) - volume, 0), 30) / (ts_sum(abs(volume - ts_delay(volume, 1)), 30) + 1e-12)",
    ),
    (
        "a158_vsumd_30",
        "(ts_sum(gt(volume - ts_delay(volume, 1), 0), 30) - ts_sum(gt(ts_delay(volume, 1) - volume, 0), 30)) / (ts_sum(abs(volume - ts_delay(volume, 1)), 30) + 1e-12)",
    ),
    ("a158_roc_60", "ts_delay(close, 60) / close"),
    ("a158_ma_60", "ts_mean(close, 60) / close"),
    ("a158_std_60", "ts_std(close, 60) / close"),
    ("a158_beta_60", "ts_slope(close, 60) / close"),
    ("a158_rsqr_60", "ts_rsquare(close, 60)"),
    ("a158_resi_60", "ts_resi(close, 60) / close"),
    ("a158_max_60", "ts_max(high, 60) / close"),
    ("a158_min_60", "ts_min(low, 60) / close"),
    ("a158_qtlu_60", "ts_quantile(close, 60, 0.8) / close"),
    ("a158_qtld_60", "ts_quantile(close, 60, 0.2) / close"),
    ("a158_rank_60", "ts_rank(close, 60)"),
    (
        "a158_rsv_60",
        "(close - ts_min(low, 60)) / (ts_max(high, 60) - ts_min(low, 60) + 1e-12)",
    ),
    ("a158_imax_60", "ts_argmax(high, 60) / 60"),
    ("a158_imin_60", "ts_argmin(low, 60) / 60"),
    (
        "a158_imxd_60",
        "(ts_argmax(high, 60) - ts_argmin(low, 60)) / 60",
    ),
    ("a158_corr_60", "ts_correlation(close, log(volume + 1), 60)"),
    (
        "a158_cord_60",
        "ts_correlation(close / ts_delay(close, 1), log(volume / ts_delay(volume, 1) + 1), 60)",
    ),
    ("a158_cntp_60", "ts_mean(gt(close, ts_delay(close, 1)), 60)"),
    ("a158_cntn_60", "ts_mean(lt(close, ts_delay(close, 1)), 60)"),
    (
        "a158_cntd_60",
        "ts_mean(gt(close, ts_delay(close, 1)), 60) - ts_mean(lt(close, ts_delay(close, 1)), 60)",
    ),
    (
        "a158_sump_60",
        "ts_sum(gt(close - ts_delay(close, 1), 0), 60) / (ts_sum(abs(close - ts_delay(close, 1)), 60) + 1e-12)",
    ),
    (
        "a158_sumn_60",
        "ts_sum(gt(ts_delay(close, 1) - close, 0), 60) / (ts_sum(abs(close - ts_delay(close, 1)), 60) + 1e-12)",
    ),
    (
        "a158_sumd_60",
        "(ts_sum(gt(close - ts_delay(close, 1), 0), 60) - ts_sum(gt(ts_delay(close, 1) - close, 0), 60)) / (ts_sum(abs(close - ts_delay(close, 1)), 60) + 1e-12)",
    ),
    ("a158_vma_60", "ts_mean(volume, 60) / (volume + 1e-12)"),
    ("a158_vstd_60", "ts_std(volume, 60) / (volume + 1e-12)"),
    (
        "a158_wvma_60",
        "ts_std(abs(close / ts_delay(close, 1) - 1) * volume, 60) / (ts_mean(abs(close / ts_delay(close, 1) - 1) * volume, 60) + 1e-12)",
    ),
    (
        "a158_vsump_60",
        "ts_sum(gt(volume - ts_delay(volume, 1), 0), 60) / (ts_sum(abs(volume - ts_delay(volume, 1)), 60) + 1e-12)",
    ),
    (
        "a158_vsumn_60",
        "ts_sum(gt(ts_delay(volume, 1) - volume, 0), 60) / (ts_sum(abs(volume - ts_delay(volume, 1)), 60) + 1e-12)",
    ),
    (
        "a158_vsumd_60",
        "(ts_sum(gt(volume - ts_delay(volume, 1), 0), 60) - ts_sum(gt(ts_delay(volume, 1) - volume, 0), 60)) / (ts_sum(abs(volume - ts_delay(volume, 1)), 60) + 1e-12)",
    ),
    (
        "a191_alpha001",
        "-1 * ts_correlation(cs_rank(ts_delta(log(volume), 1)), cs_rank((close - open) / open), 6)",
    ),
    (
        "a191_alpha002",
        "-1 * ts_delta(((close - low) - (high - close)) / (high - low), 1)",
    ),
    (
        "a191_alpha003",
        "ts_sum(quesval2(close, ts_delay(close, 1), 0, quesval(0, close - ts_delay(close, 1), close - min(low, ts_delay(close, 1)), close - max(high, ts_delay(close, 1)))), 6)",
    ),
    (
        "a191_alpha004",
        "quesval2(ts_sum(close, 8) / 8 + ts_std(close, 8), ts_sum(close, 2) / 2, -1, quesval2(ts_sum(close, 2) / 2, ts_sum(close, 8) / 8 + ts_std(close, 8), quesval2(ts_sum(close, 8) / 8 - ts_std(close, 8), ts_sum(close, 2) / 2, quesval(0.999, volume / ts_mean(volume, 20), 1, -1), 1), 1))",
    ),
    (
        "a191_alpha005",
        "-1 * ts_max(ts_correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)",
    ),
    (
        "a191_alpha006",
        "-1 * cs_rank(sign(ts_delta((open * 0.85 + high * 0.15), 4)))",
    ),
    (
        "a191_alpha007",
        "(cs_rank(ts_max(vwap - close, 3)) + cs_rank(ts_min(vwap - close, 3))) * cs_rank(ts_delta(volume, 3))",
    ),
    (
        "a191_alpha008",
        "cs_rank(ts_delta(((high + low) / 2 * 0.2 + vwap * 0.8), 4) * -1)",
    ),
    (
        "a191_alpha009",
        "ts_sma(((high + low) / 2 - (ts_delay(high, 1) + ts_delay(low, 1)) / 2) * (high - low) / volume, 7, 2)",
    ),
    (
        "a191_alpha010",
        "cs_rank(ts_max(power(quesval(0, close / ts_delay(close, 1) - 1, ts_std(close / ts_delay(close, 1) - 1, 20), close), 2), 5))",
    ),
    (
        "a191_alpha011",
        "ts_sum(((close - low) - (high - close)) / (high - low) * volume, 6)",
    ),
    (
        "a191_alpha012",
        "cs_rank(open - ts_sum(vwap, 10) / 10) * -1 * cs_rank(abs(close - vwap))",
    ),
    ("a191_alpha013", "power(high * low, 0.5) - vwap"),
    ("a191_alpha014", "close - ts_delay(close, 5)"),
    ("a191_alpha015", "open / ts_delay(close, 1) - 1"),
    (
        "a191_alpha016",
        "-1 * ts_max(cs_rank(ts_correlation(cs_rank(volume), cs_rank(vwap), 5)), 5)",
    ),
    (
        "a191_alpha017",
        "power(cs_rank(vwap - ts_max(vwap, 15)), ts_delta(close, 5))",
    ),
    ("a191_alpha018", "close / ts_delay(close, 5)"),
    (
        "a191_alpha019",
        "quesval(0, close - ts_delay(close, 5), quesval(0, ts_delay(close, 5) - close, (close - ts_delay(close, 5)) / close, (close - ts_delay(close, 5)) / ts_delay(close, 5)), 0)",
    ),
    (
        "a191_alpha020",
        "(close - ts_delay(close, 6)) / ts_delay(close, 6) * 100",
    ),
    (
        "a191_alpha022",
        "ts_sma(((close - ts_mean(close, 6)) / ts_mean(close, 6) - ts_delay((close - ts_mean(close, 6)) / ts_mean(close, 6), 3)), 12, 1)",
    ),
    (
        "a191_alpha023",
        "ts_sma(quesval(0, close - ts_delay(close, 1), ts_std(close, 20), 0), 20, 1) / (ts_sma(quesval(0, close - ts_delay(close, 1), ts_std(close, 20), 0), 20, 1) + ts_sma(quesval(0, ts_delay(close, 1) - close, ts_std(close, 20), 0), 20, 1)) * 100",
    ),
    ("a191_alpha024", "ts_sma(close - ts_delay(close, 5), 5, 1)"),
    (
        "a191_alpha025",
        "(-1 * cs_rank((ts_delta(close, 7) * (1 - cs_rank(ts_decay_linear(volume / ts_mean(volume, 20), 9)))))) * (1 + cs_rank(ts_sum(close / ts_delay(close, 1) - 1, 250)))",
    ),
    (
        "a191_alpha026",
        "((ts_sum(close, 7) / 7 - close)) + ts_correlation(vwap, ts_delay(close, 5), 230)",
    ),
    (
        "a191_alpha027",
        "ts_wma(((close - ts_delay(close, 3)) / ts_delay(close, 3) * 100 + (close - ts_delay(close, 6)) / ts_delay(close, 6) * 100), 12)",
    ),
    (
        "a191_alpha028",
        "3 * ts_sma((close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9)) * 100, 3, 1) - 2 * ts_sma(ts_sma((close - ts_min(low, 9)) / (ts_max(high, 9) - ts_max(low, 9)) * 100, 3, 1), 3, 1)",
    ),
    (
        "a191_alpha029",
        "(close - ts_delay(close, 6)) / ts_delay(close, 6) * volume",
    ),
    (
        "a191_alpha031",
        "(close - ts_mean(close, 12)) / ts_mean(close, 12) * 100",
    ),
    (
        "a191_alpha032",
        "-1 * ts_sum(cs_rank(ts_correlation(cs_rank(high), cs_rank(volume), 3)), 3)",
    ),
    (
        "a191_alpha033",
        "(((-1 * ts_min(low, 5)) + ts_delay(ts_min(low, 5), 5)) * cs_rank((ts_sum(close / ts_delay(close, 1) - 1, 240) - ts_sum(close / ts_delay(close, 1) - 1, 20)) / 220)) * ts_rank(volume, 5)",
    ),
    ("a191_alpha034", "ts_mean(close, 12) / close"),
    (
        "a191_alpha035",
        "min(cs_rank(ts_decay_linear(ts_delta(open, 1), 15)), cs_rank(ts_decay_linear(ts_correlation(volume, open, 17), 7))) * -1",
    ),
    (
        "a191_alpha036",
        "cs_rank(ts_sum(ts_correlation(cs_rank(volume), cs_rank(vwap), 6), 2))",
    ),
    (
        "a191_alpha037",
        "-1 * cs_rank((ts_sum(open, 5) * ts_sum(close / ts_delay(close, 1) - 1, 5) - ts_delay(ts_sum(open, 5) * ts_sum(close / ts_delay(close, 1) - 1, 5), 10)))",
    ),
    (
        "a191_alpha038",
        "quesval(0, ts_sum(high, 20) / 20 - high, -1 * ts_delta(high, 2), 0)",
    ),
    (
        "a191_alpha039",
        "(cs_rank(ts_decay_linear(ts_delta(close, 2), 8)) - cs_rank(ts_decay_linear(ts_correlation(vwap * 0.3 + open * 0.7, ts_sum(ts_mean(volume, 180), 37), 14), 12))) * -1",
    ),
    (
        "a191_alpha040",
        "ts_sum(quesval(0, close - ts_delay(close, 1), volume, 0), 26) / ts_sum(quesval(0, ts_delay(close, 1) - close, volume, 0), 26) * 100",
    ),
    (
        "a191_alpha041",
        "cs_rank(ts_max(ts_delta(vwap, 3), 5)) * -1",
    ),
    (
        "a191_alpha042",
        "-1 * cs_rank(ts_std(high, 10)) * ts_correlation(high, volume, 10)",
    ),
    (
        "a191_alpha043",
        "ts_sum(quesval2(close, ts_delay(close, 1), quesval2(ts_delay(close, 1), close, -volume, volume), 0), 6)",
    ),
    (
        "a191_alpha044",
        "ts_rank(ts_decay_linear(ts_correlation(low, ts_mean(volume, 10), 7), 6), 4) + ts_rank(ts_decay_linear(ts_delta(vwap, 3), 10), 15)",
    ),
    (
        "a191_alpha045",
        "cs_rank(ts_delta(close * 0.6 + open * 0.4, 1)) * cs_rank(ts_correlation(vwap, ts_mean(volume, 150), 15))",
    ),
    (
        "a191_alpha046",
        "(ts_mean(close, 3) + ts_mean(close, 6) + ts_mean(close, 12) + ts_mean(close, 24)) / (4 * close)",
    ),
    (
        "a191_alpha047",
        "ts_sma((ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6)) * 100, 9, 1)",
    ),
    (
        "a191_alpha048",
        "(-1 * (cs_rank((sign(close - ts_delay(close, 1)) + sign(ts_delay(close, 1) - ts_delay(close, 2)) + sign(ts_delay(close, 2) - ts_delay(close, 3)))) * ts_sum(volume, 5))) / ts_sum(volume, 20)",
    ),
    (
        "a191_alpha049",
        "ts_sum(quesval(0, ts_delay(high, 1) + ts_delay(low, 1) - high - low, max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12) / (ts_sum(quesval(0, ts_delay(high, 1) + ts_delay(low, 1) - high - low, max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12) + ts_sum(quesval(0, high + low - ts_delay(high, 1) - ts_delay(low, 1), max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12))",
    ),
    (
        "a191_alpha050",
        "(ts_sum(quesval(0, high + low - ts_delay(high, 1) - ts_delay(low, 1), max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12) - ts_sum(quesval(0, ts_delay(high, 1) + ts_delay(low, 1) - high - low, max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12)) / (ts_sum(quesval(0, high + low - ts_delay(high, 1) - ts_delay(low, 1), max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12) + ts_sum(quesval(0, ts_delay(high, 1) + ts_delay(low, 1) - high - low, max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12))",
    ),
    (
        "a191_alpha051",
        "ts_sum(quesval(0, high + low - ts_delay(high, 1) - ts_delay(low, 1), max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12) / (ts_sum(quesval(0, high + low - ts_delay(high, 1) - ts_delay(low, 1), max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12) + ts_sum(quesval(0, ts_delay(high, 1) + ts_delay(low, 1) - high - low, max(abs(high - ts_delay(high, 1)), abs(low - ts_delay(low, 1))), 0), 12))",
    ),
    (
        "a191_alpha052",
        "ts_sum(max(high - ts_delay((high + low + close) / 3, 1), 0), 26) / ts_sum(max(ts_delay((high + low + close) / 3, 1) - low, 0), 26) * 100",
    ),
    (
        "a191_alpha053",
        "ts_sum(gt(close, ts_delay(close, 1)), 12) / 12 * 100",
    ),
    (
        "a191_alpha054",
        "-1 * cs_rank(ts_std(abs(close - open), 20) + (close - open) + ts_correlation(close, open, 10))",
    ),
    (
        "a191_alpha056",
        "quesval2(cs_rank(open - ts_min(open, 12)), cs_rank(power(cs_rank(ts_correlation(ts_sum((high + low) / 2, 19), ts_sum(ts_mean(volume, 40), 19), 13)), 5)), 1, 0)",
    ),
    (
        "a191_alpha057",
        "ts_sma((close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9)) * 100, 3, 1)",
    ),
    (
        "a191_alpha058",
        "ts_sum(gt(close, ts_delay(close, 1)), 20) / 20 * 100",
    ),
    (
        "a191_alpha059",
        "ts_sum(quesval2(close, ts_delay(close, 1), quesval2(ts_delay(close, 1), close, close - max(high, ts_delay(close, 1)), close - min(low, ts_delay(close, 1))), 0), 20)",
    ),
    (
        "a191_alpha060",
        "ts_sum(((close - low) - (high - close)) / (high - low) * volume, 20)",
    ),
    (
        "a191_alpha061",
        "max(cs_rank(ts_decay_linear(ts_delta(vwap, 1), 12)), cs_rank(ts_decay_linear(ts_rank(ts_correlation(low, ts_mean(volume, 80), 8), 17), 17))) * -1",
    ),
    (
        "a191_alpha062",
        "-1 * ts_correlation(high, cs_rank(volume), 5)",
    ),
    (
        "a191_alpha063",
        "ts_sma(max(close - ts_delay(close, 1), 0), 6, 1) / ts_sma(abs(close - ts_delay(close, 1)), 6, 1) * 100",
    ),
    (
        "a191_alpha064",
        "max(cs_rank(ts_decay_linear(ts_correlation(cs_rank(vwap), cs_rank(volume), 4), 4)), cs_rank(ts_decay_linear(ts_max(ts_correlation(cs_rank(close), cs_rank(ts_mean(volume, 60)), 4), 13), 14))) * -1",
    ),
    ("a191_alpha065", "ts_mean(close, 6) / close"),
    (
        "a191_alpha066",
        "(close - ts_mean(close, 6)) / ts_mean(close, 6) * 100",
    ),
    (
        "a191_alpha067",
        "ts_sma(max(close - ts_delay(close, 1), 0), 24, 1) / ts_sma(abs(close - ts_delay(close, 1)), 24, 1) * 100",
    ),
    (
        "a191_alpha068",
        "ts_sma(((high + low) / 2 - (ts_delay(high, 1) + ts_delay(low, 1)) / 2) * (high - low) / volume, 15, 2)",
    ),
    (
        "a191_alpha069",
        "quesval2(ts_sum(quesval(0, open - ts_delay(open, 1), 0, max(high - open, open - ts_delay(open, 1))), 20), ts_sum(quesval(0, ts_delay(open, 1) - open, 0, max(open - low, open - ts_delay(open, 1))), 20), (ts_sum(quesval(0, open - ts_delay(open, 1), 0, max(high - open, open - ts_delay(open, 1))), 20) - ts_sum(quesval(0, ts_delay(open, 1) - open, 0, max(open - low, open - ts_delay(open, 1))), 20)) / ts_sum(quesval(0, open - ts_delay(open, 1), 0, max(high - open, open - ts_delay(open, 1))), 20), (ts_sum(quesval(0, open - ts_delay(open, 1), 0, max(high - open, open - ts_delay(open, 1))), 20) - ts_sum(quesval(0, ts_delay(open, 1) - open, 0, max(open - low, open - ts_delay(open, 1))), 20)) / ts_sum(quesval(0, ts_delay(open, 1) - open, 0, max(open - low, open - ts_delay(open, 1))), 20))",
    ),
    ("a191_alpha070", "ts_std(amount, 6)"),
    (
        "a191_alpha071",
        "(close - ts_mean(close, 24)) / ts_mean(close, 24) * 100",
    ),
    (
        "a191_alpha072",
        "ts_sma((ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6)) * 100, 15, 1)",
    ),
    (
        "a191_alpha073",
        "(ts_rank(ts_decay_linear(ts_decay_linear(ts_correlation(close, volume, 10), 16), 4), 5) - cs_rank(ts_decay_linear(ts_correlation(vwap, ts_mean(volume, 30), 4), 3))) * -1",
    ),
    (
        "a191_alpha074",
        "cs_rank(ts_correlation(ts_sum(low * 0.35 + vwap * 0.65, 20), ts_sum(ts_mean(volume, 40), 20), 7)) + cs_rank(ts_correlation(cs_rank(vwap), cs_rank(volume), 6))",
    ),
    (
        "a191_alpha076",
        "ts_std(abs(close / ts_delay(close, 1) - 1) / volume, 20) / ts_mean(abs(close / ts_delay(close, 1) - 1) / volume, 20)",
    ),
    (
        "a191_alpha077",
        "min(cs_rank(ts_decay_linear(((high + low) / 2 + high - vwap - high), 20)), cs_rank(ts_decay_linear(ts_correlation((high + low) / 2, ts_mean(volume, 40), 3), 6)))",
    ),
    (
        "a191_alpha078",
        "((high + low + close) / 3 - ts_mean((high + low + close) / 3, 12)) / (0.015 * ts_mean(abs(close - ts_mean((high + low + close) / 3, 12)), 12))",
    ),
    (
        "a191_alpha079",
        "ts_sma(max(close - ts_delay(close, 1), 0), 12, 1) / ts_sma(abs(close - ts_delay(close, 1)), 12, 1) * 100",
    ),
    (
        "a191_alpha080",
        "(volume - ts_delay(volume, 5)) / ts_delay(volume, 5) * 100",
    ),
    ("a191_alpha081", "ts_sma(volume, 21, 2)"),
    (
        "a191_alpha082",
        "ts_sma((ts_max(high, 6) - close) / (ts_max(high, 6) - ts_min(low, 6)) * 100, 20, 1)",
    ),
    (
        "a191_alpha083",
        "-1 * cs_rank(ts_covariance(cs_rank(high), cs_rank(volume), 5))",
    ),
    (
        "a191_alpha084",
        "ts_sum(quesval2(close, ts_delay(close, 1), quesval2(ts_delay(close, 1), close, -volume, volume), 0), 20)",
    ),
    (
        "a191_alpha085",
        "ts_rank(volume / ts_mean(volume, 20), 20) * ts_rank(-1 * ts_delta(close, 7), 8)",
    ),
    (
        "a191_alpha086",
        "quesval(0.25, (ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10, -1, quesval(0, (ts_delay(close, 20) - ts_delay(close, 10)) / 10 - (ts_delay(close, 10) - close) / 10, 1, -1 * (close - ts_delay(close, 1))))",
    ),
    (
        "a191_alpha087",
        "(cs_rank(ts_decay_linear(ts_delta(vwap, 4), 7)) + ts_rank(ts_decay_linear(((low - vwap) / (open - (high + low) / 2)), 11), 7)) * -1",
    ),
    (
        "a191_alpha088",
        "(close - ts_delay(close, 20)) / ts_delay(close, 20) * 100",
    ),
    (
        "a191_alpha089",
        "2 * (ts_sma(close, 13, 2) - ts_sma(close, 27, 2) - ts_sma(ts_sma(close, 13, 2) - ts_sma(close, 27, 2), 10, 2))",
    ),
    (
        "a191_alpha090",
        "cs_rank(ts_correlation(cs_rank(vwap), cs_rank(volume), 5)) * -1",
    ),
    (
        "a191_alpha091",
        "cs_rank(close - ts_max(close, 5)) * cs_rank(ts_correlation(ts_mean(volume, 40), low, 5)) * -1",
    ),
    (
        "a191_alpha092",
        "max(cs_rank(ts_decay_linear(ts_delta(close * 0.35 + vwap * 0.65, 2), 3)), ts_rank(ts_decay_linear(abs(ts_correlation(ts_mean(volume, 180), close, 13)), 5), 15)) * -1",
    ),
    (
        "a191_alpha093",
        "ts_sum(quesval(0, open - ts_delay(open, 1), 0, max(open - low, open - ts_delay(open, 1))), 20)",
    ),
    (
        "a191_alpha094",
        "ts_sum(quesval2(close, ts_delay(close, 1), quesval2(ts_delay(close, 1), close, -volume, volume), 0), 30)",
    ),
    ("a191_alpha095", "ts_std(amount, 20)"),
    (
        "a191_alpha096",
        "ts_sma(ts_sma((close - ts_min(low, 9)) / (ts_max(high, 9) - ts_min(low, 9)) * 100, 3, 1), 3, 1)",
    ),
    ("a191_alpha097", "ts_std(volume, 10)"),
    (
        "a191_alpha098",
        "quesval(0.05, ts_delta(ts_sum(close, 100) / 100, 100) / ts_delay(close, 100), -1 * (close - ts_min(close, 100)), -1 * ts_delta(close, 3))",
    ),
    (
        "a191_alpha099",
        "-1 * cs_rank(ts_covariance(cs_rank(close), cs_rank(volume), 5))",
    ),
    ("a191_alpha100", "ts_std(volume, 20)"),
    (
        "a191_alpha101",
        "quesval2(cs_rank(ts_correlation(close, ts_sum(ts_mean(volume, 30), 37), 15)), cs_rank(ts_correlation(cs_rank(high * 0.1 + vwap * 0.9), cs_rank(volume), 11)), 1, 0) * -1",
    ),
    (
        "a191_alpha102",
        "ts_sma(max(volume - ts_delay(volume, 1), 0), 6, 1) / ts_sma(abs(volume - ts_delay(volume, 1)), 6, 1) * 100",
    ),
    ("a191_alpha103", "((20 - ts_lowday(low, 20)) / 20) * 100"),
    (
        "a191_alpha104",
        "-1 * ts_delta(ts_correlation(high, volume, 5), 5) * cs_rank(ts_std(close, 20))",
    ),
    (
        "a191_alpha105",
        "-1 * ts_correlation(cs_rank(open), cs_rank(volume), 10)",
    ),
    ("a191_alpha106", "close - ts_delay(close, 20)"),
    (
        "a191_alpha107",
        "(-1 * cs_rank(open - ts_delay(high, 1))) * cs_rank(open - ts_delay(close, 1)) * cs_rank(open - ts_delay(low, 1))",
    ),
    (
        "a191_alpha108",
        "power(cs_rank(high - ts_min(high, 2)), cs_rank(ts_correlation(vwap, ts_mean(volume, 120), 6))) * -1",
    ),
    (
        "a191_alpha109",
        "ts_sma(high - low, 10, 2) / ts_sma(ts_sma(high - low, 10, 2), 10, 2)",
    ),
    (
        "a191_alpha110",
        "ts_sum(max(high - ts_delay(close, 1), 0), 20) / ts_sum(max(ts_delay(close, 1) - low, 0), 20) * 100",
    ),
    (
        "a191_alpha111",
        "ts_sma(volume * ((close - low) - (high - close)) / (high - low), 11, 2) - ts_sma(volume * ((close - low) - (high - close)) / (high - low), 4, 2)",
    ),
    (
        "a191_alpha112",
        "(ts_sum(quesval(0, close - ts_delay(close, 1), close - ts_delay(close, 1), 0), 12) - ts_sum(quesval(0, ts_delay(close, 1) - close, abs(close - ts_delay(close, 1)), 0), 12)) / (ts_sum(quesval(0, close - ts_delay(close, 1), close - ts_delay(close, 1), 0), 12) + ts_sum(quesval(0, ts_delay(close, 1) - close, abs(close - ts_delay(close, 1)), 0), 12)) * 100",
    ),
    (
        "a191_alpha113",
        "-1 * (cs_rank(ts_sum(ts_delay(close, 5), 20) / 20) * ts_correlation(close, volume, 2) * cs_rank(ts_correlation(ts_sum(close, 5), ts_sum(close, 20), 2)))",
    ),
    (
        "a191_alpha114",
        "(cs_rank(ts_delay((high - low) / (ts_sum(close, 5) / 5), 2)) * cs_rank(cs_rank(volume))) / (((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close))",
    ),
    (
        "a191_alpha115",
        "power(cs_rank(ts_correlation(high * 0.9 + close * 0.1, ts_mean(volume, 30), 10)), cs_rank(ts_correlation(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7)))",
    ),
    (
        "a191_alpha117",
        "(ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16))) * (1 - ts_rank(close / ts_delay(close, 1) - 1, 32))",
    ),
    (
        "a191_alpha118",
        "ts_sum(high - open, 20) / ts_sum(open - low, 20) * 100",
    ),
    (
        "a191_alpha119",
        "cs_rank(ts_decay_linear(ts_correlation(vwap, ts_sum(ts_mean(volume, 5), 26), 5), 7)) - cs_rank(ts_decay_linear(ts_rank(ts_min(ts_correlation(cs_rank(open), cs_rank(ts_mean(volume, 15)), 21), 9), 7), 8))",
    ),
    (
        "a191_alpha120",
        "cs_rank(vwap - close) / cs_rank(vwap + close)",
    ),
    (
        "a191_alpha121",
        "power(cs_rank(vwap - ts_min(vwap, 12)), ts_rank(ts_correlation(ts_rank(vwap, 20), ts_rank(ts_mean(volume, 60), 2), 18), 3)) * -1",
    ),
    (
        "a191_alpha122",
        "(ts_sma(ts_sma(ts_sma(log(close), 13, 2), 13, 2), 13, 2) - ts_delay(ts_sma(ts_sma(ts_sma(log(close), 13, 2), 13, 2), 13, 2), 1)) / ts_delay(ts_sma(ts_sma(ts_sma(log(close), 13, 2), 13, 2), 13, 2), 1)",
    ),
    (
        "a191_alpha123",
        "quesval2(cs_rank(ts_correlation(ts_sum((high + low) / 2, 20), ts_sum(ts_mean(volume, 60), 20), 9)), cs_rank(ts_correlation(low, volume, 6)), 0, -1)",
    ),
    (
        "a191_alpha124",
        "(close - vwap) / ts_decay_linear(cs_rank(ts_max(close, 30)), 2)",
    ),
    (
        "a191_alpha125",
        "cs_rank(ts_decay_linear(ts_correlation(vwap, ts_mean(volume, 80), 17), 20)) / cs_rank(ts_decay_linear(ts_delta(close * 0.5 + vwap * 0.5, 3), 16))",
    ),
    ("a191_alpha126", "(close + high + low) / 3"),
    (
        "a191_alpha127",
        "power(ts_mean(power(100 * (close - ts_max(close, 12)) / ts_max(close, 12), 2), 12), 0.5)",
    ),
    (
        "a191_alpha128",
        "100 - (100 / (1 + ts_sum(quesval(0, (high + low + close) / 3 - ts_delay((high + low + close) / 3, 1), (high + low + close) / 3 * volume, 0), 14) / ts_sum(quesval(0, ts_delay((high + low + close) / 3, 1) - (high + low + close) / 3, (high + low + close) / 3 * volume, 0), 14)))",
    ),
    (
        "a191_alpha129",
        "ts_sum(quesval(0, close - ts_delay(close, 1), 0, abs(close - ts_delay(close, 1))), 12)",
    ),
    (
        "a191_alpha130",
        "cs_rank(ts_decay_linear(ts_correlation((high + low) / 2, ts_mean(volume, 40), 9), 10)) / cs_rank(ts_decay_linear(ts_correlation(cs_rank(vwap), cs_rank(volume), 7), 3))",
    ),
    (
        "a191_alpha131",
        "power(cs_rank(ts_delta(vwap, 1)), ts_rank(ts_correlation(close, ts_mean(volume, 50), 18), 18))",
    ),
    ("a191_alpha132", "ts_mean(amount, 20)"),
    (
        "a191_alpha133",
        "((20 - ts_highday(high, 20)) / 20) * 100 - ((20 - ts_lowday(low, 20)) / 20) * 100",
    ),
    (
        "a191_alpha134",
        "(close - ts_delay(close, 12)) / ts_delay(close, 12) * volume",
    ),
    (
        "a191_alpha135",
        "ts_sma(ts_delay(close / ts_delay(close, 20), 1), 20, 1)",
    ),
    (
        "a191_alpha136",
        "(-1 * cs_rank(ts_delta(close / ts_delay(close, 1) - 1, 3))) * ts_correlation(open, volume, 10)",
    ),
    (
        "a191_alpha138",
        "(cs_rank(ts_decay_linear(ts_delta(low * 0.7 + vwap * 0.3, 3), 20)) - ts_rank(ts_decay_linear(ts_rank(ts_correlation(ts_rank(low, 8), ts_rank(ts_mean(volume, 60), 17), 5), 19), 16), 7)) * -1",
    ),
    ("a191_alpha139", "-1 * ts_correlation(open, volume, 10)"),
    (
        "a191_alpha140",
        "min(cs_rank(ts_decay_linear((cs_rank(open) + cs_rank(low) - cs_rank(high) - cs_rank(close)), 8)), ts_rank(ts_decay_linear(ts_correlation(ts_rank(close, 8), ts_rank(ts_mean(volume, 60), 20), 8), 7), 3))",
    ),
    (
        "a191_alpha141",
        "cs_rank(ts_correlation(cs_rank(high), cs_rank(ts_mean(volume, 15)), 9)) * -1",
    ),
    (
        "a191_alpha142",
        "(-1 * cs_rank(ts_rank(close, 10))) * cs_rank(ts_delta(ts_delta(close, 1), 1)) * cs_rank(ts_rank(volume / ts_mean(volume, 20), 5))",
    ),
    (
        "a191_alpha144",
        "ts_sum(quesval(0, ts_delay(close, 1) - close, abs(close / ts_delay(close, 1) - 1) / amount, 0), 20) / ts_sum(gt(ts_delay(close, 1), close), 20)",
    ),
    (
        "a191_alpha145",
        "(ts_mean(volume, 9) - ts_mean(volume, 26)) / ts_mean(volume, 12) * 100",
    ),
    (
        "a191_alpha146",
        "ts_mean((close / ts_delay(close, 1) - 1 - ts_sma(close / ts_delay(close, 1) - 1, 61, 2)), 20) * (close / ts_delay(close, 1) - 1 - ts_sma(close / ts_delay(close, 1) - 1, 61, 2)) / ts_sma(power(close / ts_delay(close, 1) - 1 - ts_sma(close / ts_delay(close, 1) - 1, 61, 2), 2), 61, 2)",
    ),
    (
        "a191_alpha148",
        "quesval2(cs_rank(ts_correlation(open, ts_sum(ts_mean(volume, 60), 9), 6)), cs_rank(open - ts_min(open, 14)), 0, -1)",
    ),
    ("a191_alpha150", "(close + high + low) / 3 * volume"),
    (
        "a191_alpha151",
        "ts_sma(close - ts_delay(close, 20), 20, 1)",
    ),
    (
        "a191_alpha152",
        "ts_sma(ts_mean(ts_delay(ts_sma(ts_delay(close / ts_delay(close, 9), 1), 9, 1), 1), 12) - ts_mean(ts_delay(ts_sma(ts_delay(close / ts_delay(close, 9), 1), 9, 1), 1), 26), 9, 1)",
    ),
    (
        "a191_alpha153",
        "(ts_mean(close, 3) + ts_mean(close, 6) + ts_mean(close, 12) + ts_mean(close, 24)) / 4",
    ),
    (
        "a191_alpha154",
        "quesval2(cs_rank(vwap - ts_min(vwap, 16)), cs_rank(ts_correlation(vwap, ts_mean(volume, 180), 18)), 1, 0)",
    ),
    (
        "a191_alpha155",
        "ts_sma(volume, 13, 2) - ts_sma(volume, 27, 2) - ts_sma(ts_sma(volume, 13, 2) - ts_sma(volume, 27, 2), 10, 2)",
    ),
    (
        "a191_alpha156",
        "max(cs_rank(ts_decay_linear(ts_delta(vwap, 5), 3)), cs_rank(ts_decay_linear((ts_delta(open * 0.15 + low * 0.85, 2) / (open * 0.15 + low * 0.85)) * -1, 3))) * -1",
    ),
    (
        "a191_alpha157",
        "(ts_min(ts_product(cs_rank(cs_rank(log(ts_sum(ts_min(cs_rank(cs_rank((-1 * cs_rank(ts_delta(close - 1, 5))))), 2), 1)))), 1), 5) + ts_rank(ts_delay(-1 * (close / ts_delay(close, 1) - 1), 6), 5))",
    ),
    (
        "a191_alpha158",
        "((high - ts_sma(close, 15, 2)) - (low - ts_sma(close, 15, 2))) / close",
    ),
    (
        "a191_alpha159",
        "((close - ts_sum(min(low, ts_delay(close, 1)), 6)) / ts_sum(max(high, ts_delay(close, 1)) - min(low, ts_delay(close, 1)), 6) * 12 * 24 + (close - ts_sum(min(low, ts_delay(close, 1)), 12)) / ts_sum(max(high, ts_delay(close, 1)) - min(low, ts_delay(close, 1)), 12) * 6 * 24 + (close - ts_sum(min(low, ts_delay(close, 1)), 24)) / ts_sum(max(high, ts_delay(close, 1)) - min(low, ts_delay(close, 1)), 24) * 6 * 24) * 100 / (6 * 12 + 6 * 24 + 12 * 24)",
    ),
    (
        "a191_alpha160",
        "ts_sma(quesval(0, ts_delay(close, 1) - close, ts_std(close, 20), 0), 20, 1)",
    ),
    (
        "a191_alpha161",
        "ts_mean(max(max(high - low, abs(ts_delay(close, 1) - high)), abs(ts_delay(close, 1) - low)), 12)",
    ),
    (
        "a191_alpha162",
        "(ts_sma(max(close - ts_delay(close, 1), 0), 12, 1) / ts_sma(abs(close - ts_delay(close, 1)), 12, 1) * 100 - ts_min(ts_sma(max(close - ts_delay(close, 1), 0), 12, 1) / ts_sma(abs(close - ts_delay(close, 1)), 12, 1) * 100, 12)) / (ts_max(ts_sma(max(close - ts_delay(close, 1), 0), 12, 1) / ts_sma(abs(close - ts_delay(close, 1)), 12, 1) * 100, 12) - ts_min(ts_sma(max(close - ts_delay(close, 1), 0), 12, 1) / ts_sma(abs(close - ts_delay(close, 1)), 12, 1) * 100, 12))",
    ),
    (
        "a191_alpha163",
        "cs_rank(((-1 * (close / ts_delay(close, 1) - 1)) * ts_mean(volume, 20) * vwap * (high - close)))",
    ),
    (
        "a191_alpha164",
        "ts_sma((quesval2(close, ts_delay(close, 1), 1 / abs(close - ts_delay(close, 1)), 1) - ts_min(quesval2(close, ts_delay(close, 1), 1 / abs(close - ts_delay(close, 1)), 1), 12)) / (high - low) * 100, 13, 2)",
    ),
    (
        "a191_alpha167",
        "ts_sum(quesval(0, close - ts_delay(close, 1), close - ts_delay(close, 1), 0), 12)",
    ),
    ("a191_alpha168", "-1 * volume / ts_mean(volume, 20)"),
    (
        "a191_alpha169",
        "ts_sma(ts_mean(ts_delay(ts_sma(close - ts_delay(close, 1), 9, 1), 1), 12) - ts_mean(ts_delay(ts_sma(close - ts_delay(close, 1), 9, 1), 1), 26), 10, 1)",
    ),
    (
        "a191_alpha170",
        "(cs_rank(power(close, -1)) * volume / ts_mean(volume, 20) * high * cs_rank(high - close) / (ts_sum(high, 5) / 5)) - cs_rank(vwap - ts_delay(vwap, 5))",
    ),
    (
        "a191_alpha171",
        "-1 * (low - close) * power(open, 5) / ((close - high) * power(close, 5))",
    ),
    (
        "a191_alpha172",
        "ts_mean(abs(ts_sum(quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, 0, ts_delay(low, 1) - low), 0), 14) * 100 / ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) - ts_sum(quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), 0, high - ts_delay(high, 1)), 0), 14) * 100 / ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14)) / (ts_sum(quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, 0, ts_delay(low, 1) - low), 0), 14) * 100 / ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) + ts_sum(quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), 0, high - ts_delay(high, 1)), 0), 14) * 100 / ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14)) * 100, 6)",
    ),
    (
        "a191_alpha173",
        "3 * ts_sma(close, 13, 2) - 2 * ts_sma(ts_sma(close, 13, 2), 13, 2) + ts_sma(ts_sma(ts_sma(log(close), 13, 2), 13, 2), 13, 2)",
    ),
    (
        "a191_alpha174",
        "ts_sma(quesval(0, close - ts_delay(close, 1), ts_std(close, 20), 0), 20, 1)",
    ),
    (
        "a191_alpha175",
        "ts_mean(max(max(high - low, abs(ts_delay(close, 1) - high)), abs(ts_delay(close, 1) - low)), 6)",
    ),
    (
        "a191_alpha176",
        "ts_correlation(cs_rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), cs_rank(volume), 6)",
    ),
    ("a191_alpha177", "((20 - ts_highday(high, 20)) / 20) * 100"),
    (
        "a191_alpha178",
        "(close - ts_delay(close, 1)) / ts_delay(close, 1) * volume",
    ),
    (
        "a191_alpha179",
        "cs_rank(ts_correlation(vwap, volume, 4)) * cs_rank(ts_correlation(cs_rank(low), cs_rank(ts_mean(volume, 50)), 12))",
    ),
    (
        "a191_alpha180",
        "quesval2(ts_mean(volume, 20), volume, -1 * ts_rank(abs(ts_delta(close, 7)), 60) * sign(ts_delta(close, 7)), -1 * volume)",
    ),
    (
        "a191_alpha184",
        "cs_rank(ts_correlation(ts_delay(open - close, 1), close, 200)) + cs_rank(open - close)",
    ),
    ("a191_alpha185", "cs_rank(-1 * power(1 - open / close, 2))"),
    (
        "a191_alpha186",
        "(ts_mean(abs(ts_sum(quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, 0, ts_delay(low, 1) - low), 0), 14) * 100 / ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) - ts_sum(quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), 0, high - ts_delay(high, 1)), 0), 14) * 100 / ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14)) / (ts_sum(quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, 0, ts_delay(low, 1) - low), 0), 14) * 100 / ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) + ts_sum(quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), 0, high - ts_delay(high, 1)), 0), 14) * 100 / ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14)) * 100, 6) + ts_delay(ts_mean(abs(ts_sum(quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, 0, ts_delay(low, 1) - low), 0), 14) * 100 / ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) - ts_sum(quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), 0, high - ts_delay(high, 1)), 0), 14) * 100 / ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14)) / (ts_sum(quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, 0, ts_delay(low, 1) - low), 0), 14) * 100 / ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14) + ts_sum(quesval2(high - ts_delay(high, 1), ts_delay(low, 1) - low, quesval2(ts_delay(low, 1) - low, high - ts_delay(high, 1), 0, high - ts_delay(high, 1)), 0), 14) * 100 / ts_sum(max(max(high - low, abs(high - ts_delay(close, 1))), abs(low - ts_delay(close, 1))), 14)) * 100, 6), 6)) / 2",
    ),
    (
        "a191_alpha187",
        "ts_sum(quesval(0, open - ts_delay(open, 1), 0, max(high - open, open - ts_delay(open, 1))), 20)",
    ),
    (
        "a191_alpha188",
        "(high - low - ts_sma(high - low, 11, 2)) / ts_sma(high - low, 11, 2) * 100",
    ),
    (
        "a191_alpha189",
        "ts_mean(abs(close - ts_mean(close, 6)), 6)",
    ),
    (
        "a191_alpha191",
        "(ts_correlation(ts_mean(volume, 20), low, 5) + (high + low) / 2) - close",
    ),
];
fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    let nt: usize = std::env::var("RAYON_NUM_THREADS")
        .unwrap_or_default()
        .parse()
        .unwrap_or(0);
    if nt > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build_global()
            .map_err(|e| format!("rayon:{}", e))?;
    }
    eprintln!("rayon:{}", rayon::current_num_threads());
    let mut lab = AlfarsLab::new(ClickHouseSource::from_env());
    lab.set_pool("symbols not like '%BJ'");
    lab.set_duration(2024, 2024);
    for (n, e) in ALPHAS {
        lab.register(n, e)?;
    }
    eprintln!("reg:{}", ALPHAS.len());
    let t0 = Instant::now();
    let p = lab.calc(Some(".tests/b416.csv"))?;
    let t = t0.elapsed();
    eprintln!(
        "done:{:.1}s rec:{}",
        t.as_secs_f64(),
        p.slices.iter().map(|s| s.groups.len()).sum::<usize>()
    );
    Ok(())
}
