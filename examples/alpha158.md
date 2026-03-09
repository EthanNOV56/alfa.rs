        "kmid", "(close - open) / open";
        "klen", "(high - low) / open";
        "kmid_2", "(close - open) / (high - low + 1e-12)";
        "kup", "(high - ts_greater(open, close)) / open";
        "kup_2", "(high - ts_greater(open, close)) / (high - low + 1e-12)";
        "klow", "(ts_less(open, close) - low) / open";
        "klow_2", "((ts_less(open, close) - low) / (high - low + 1e-12))";
        "ksft", "(close * 2 - high - low) / open";
        "ksft_2", "(close * 2 - high - low) / (high - low + 1e-12)";

        # Price change features
        for field in ["open", "high", "low", "vwap"]:
            f"{field}_0", f"{field} / close";

        # Time series features
        windows: list[int] = [5, 10, 20, 30, 60]

        for w in windows:
            f"roc_{w}", f"ts_delay(close, {w}) / close";

        for w in windows:
            f"ma_{w}", f"ts_mean(close, {w}) / close";

        for w in windows:
            f"std_{w}", f"ts_std(close, {w}) / close";

        for w in windows:
            f"beta_{w}", f"ts_slope(close, {w}) / close";

        for w in windows:
            f"rsqr_{w}", f"ts_rsquare(close, {w})";

        for w in windows:
            f"resi_{w}", f"ts_resi(close, {w}) / close";

        for w in windows:
            f"max_{w}", f"ts_max(high, {w}) / close";

        for w in windows:
            f"min_{w}", f"ts_min(low, {w}) / close";

        for w in windows:
            f"qtlu_{w}", f"ts_quantile(close, {w}, 0.8) / close";

        for w in windows:
            f"qtld_{w}", f"ts_quantile(close, {w}, 0.2) / close";

        for w in windows:
            f"rank_{w}", f"ts_rank(close, {w})";

        for w in windows:
            f"rsv_{w}", f"(close - ts_min(low, {w})) / (ts_max(high, {w}) - ts_min(low, {w}) + 1e-12)";

        for w in windows:
            f"imax_{w}", f"ts_argmax(high, {w}) / {w}";

        for w in windows:
            f"imin_{w}", f"ts_argmin(low, {w}) / {w}";

        for w in windows:
            f"imxd_{w}", f"(ts_argmax(high, {w}) - ts_argmin(low, {w})) / {w}";

        for w in windows:
            f"corr_{w}", f"ts_corr(close, ts_log(volume + 1), {w})";

        for w in windows:
            f"cord_{w}", f"ts_corr(close / ts_delay(close, 1), ts_log(volume / ts_delay(volume, 1) + 1), {w})";

        for w in windows:
            f"cntp_{w}", f"ts_mean(close > ts_delay(close, 1), {w})";

        for w in windows:
            f"cntn_{w}", f"ts_mean(close < ts_delay(close, 1), {w})";

        for w in windows:
            f"cntd_{w}", f"ts_mean(close > ts_delay(close, 1), {w}) - ts_mean(close < ts_delay(close, 1), {w})";

        for w in windows:
            f"sump_{w}", f"ts_sum(ts_greater(close - ts_delay(close, 1), 0), {w}) / (ts_sum(ts_abs(close - ts_delay(close, 1)), {w}) + 1e-12)";

        for w in windows:
            f"sumn_{w}", f"ts_sum(ts_greater(ts_delay(close, 1) - close, 0), {w}) / (ts_sum(ts_abs(close - ts_delay(close, 1)), {w}) + 1e-12)";

        for w in windows:
            f"sumd_{w}", f"(ts_sum(ts_greater(close - ts_delay(close, 1), 0), {w}) - ts_sum(ts_greater(ts_delay(close, 1) - close, 0), {w})) / (ts_sum(ts_abs(close - ts_delay(close, 1)), {w}) + 1e-12)";

        for w in windows:
            f"vma_{w}", f"ts_mean(volume, {w}) / (volume + 1e-12)";

        for w in windows:
            f"vstd_{w}", f"ts_std(volume, {w}) / (volume + 1e-12)";

        for w in windows:
            f"wvma_{w}", f"ts_std(ts_abs(close / ts_delay(close, 1) - 1) * volume, {w}) / (ts_mean(ts_abs(close / ts_delay(close, 1) - 1) * volume, {w}) + 1e-12)";

        for w in windows:
            f"vsump_{w}", f"ts_sum(ts_greater(volume - ts_delay(volume, 1), 0), {w}) / (ts_sum(ts_abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)";

        for w in windows:
            f"vsumn_{w}", f"ts_sum(ts_greater(ts_delay(volume, 1) - volume, 0), {w}) / (ts_sum(ts_abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)";

        for w in windows:
            f"vsumd_{w}", f"(ts_sum(ts_greater(volume - ts_delay(volume, 1), 0), {w}) - ts_sum(ts_greater(ts_delay(volume, 1) - volume, 0), {w})) / (ts_sum(ts_abs(volume - ts_delay(volume, 1)), {w}) + 1e-12)";

        # Set label
        self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1";
