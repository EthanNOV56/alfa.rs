//! Portfolio simulation and weight computation functions.

use ndarray::{Array1, Array2};

use crate::WeightMethod;

pub(crate) fn compute_quantile_groups(
    factor: &Array2<f64>,
    quantiles: usize,
) -> Result<Array2<usize>, String> {
    let (n_days, n_assets) = factor.dim();
    let mut groups = Array2::<usize>::zeros((n_days, n_assets));

    for day in 0..n_days {
        let factor_row = factor.row(day);
        let mut valid_data: Vec<(usize, f64)> = factor_row
            .iter()
            .enumerate()
            .filter(|&(_, &v)| !v.is_nan())
            .map(|(i, &v)| (i, v))
            .collect();

        if valid_data.len() < quantiles {
            continue;
        }

        valid_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let n_valid = valid_data.len() as f64;
        let bins = quantiles as f64;

        let mut i = 0usize;
        while i < valid_data.len() {
            let mut j = i + 1;
            while j < valid_data.len() && valid_data[j].1 == valid_data[i].1 {
                j += 1;
            }
            let avg_rank = (i + 1 + j) as f64 / 2.0;
            let q = ((avg_rank - 1.0) * bins / n_valid).floor() as usize;
            let q = q.min(quantiles - 1) + 1;
            for k in i..j {
                let (asset_idx, _) = valid_data[k];
                groups[[day, asset_idx]] = q;
            }
            i = j;
        }
    }

    Ok(groups)
}

/// Compute group weights based on factor values
pub(crate) fn compute_group_weights(
    factor: &Array2<f64>,
    group_labels: &Array2<usize>,
    quantiles: usize,
    weight_method: WeightMethod,
) -> Array2<f64> {
    let (n_days, n_assets) = factor.dim();
    let mut group_weights = Array2::<f64>::zeros((n_days, n_assets));

    for day in 0..n_days {
        let labels_today = group_labels.row(day);
        let factor_today = factor.row(day);

        let mut group_assets: Vec<Vec<usize>> = vec![Vec::new(); quantiles];
        for asset in 0..n_assets {
            let label = labels_today[asset];
            if label > 0 && label <= quantiles {
                group_assets[label - 1].push(asset);
            }
        }

        for group in 0..quantiles {
            let assets = &group_assets[group];
            if assets.is_empty() {
                continue;
            }

            let group_weight = 1.0 / quantiles as f64;

            let weights: Vec<f64> = match weight_method {
                WeightMethod::Equal => {
                    let w = group_weight / assets.len() as f64;
                    vec![w; assets.len()]
                }
                WeightMethod::Weighted => {
                    let total_factor: f64 = assets
                        .iter()
                        .map(|&idx| factor_today[idx])
                        .filter(|&v| !v.is_nan())
                        .sum();
                    if total_factor == 0.0 {
                        vec![0.0; assets.len()]
                    } else {
                        assets
                            .iter()
                            .map(|&idx| {
                                let f = factor_today[idx];
                                if f.is_nan() {
                                    0.0
                                } else {
                                    group_weight * f / total_factor
                                }
                            })
                            .collect()
                    }
                }
            };

            for (&asset, &weight) in assets.iter().zip(weights.iter()) {
                group_weights[[day, asset]] = weight;
            }
        }
    }

    group_weights
}

/// Simulate per-group NAV using pre-computed group labels.
pub(crate) fn simulate_groups(
    group_labels: &Array2<usize>,
    group_weights: &Array2<f64>,
    quantiles: usize,
    n_days: usize,
    n_assets: usize,
    tradable: &Array2<f64>,
    open: &Array2<f64>,
    close: &Array2<f64>,
    vwap: &Array2<f64>,
    fee_rate: f64,
) -> Result<Array2<f64>, String> {
    let mut group_returns = Array2::<f64>::zeros((n_days - 1, quantiles));

    for group in 0..quantiles {
        let mut nv = 1.0f64;
        let mut prev_shares: Vec<f64> = vec![0.0f64; n_assets];

        for day in 1..n_days {
            let mut pool_count = 0usize;
            let mut in_pool = vec![false; n_assets];
            for a in 0..n_assets {
                if group_labels[[day - 1, a]] == group + 1 && tradable[[day, a]] > 0.5 {
                    in_pool[a] = true;
                    pool_count += 1;
                }
            }

            let mut asset = 0.0f64;
            for a in 0..n_assets {
                if tradable[[day, a]] > 0.5 {
                    let s = prev_shares[a];
                    if s.is_finite() && s != 0.0 {
                        asset += s * open[[day, a]];
                    }
                }
            }
            if asset <= 0.0 {
                asset = nv;
            }

            if pool_count == 0 {
                let mut asset_close = 0.0f64;
                for a in 0..n_assets {
                    let cl = close[[day, a]];
                    if cl.is_finite() && prev_shares[a].is_finite() {
                        asset_close += prev_shares[a] * cl;
                    }
                }
                let new_nv = asset_close.max(0.0);
                group_returns[[day - 1, group]] = new_nv / nv - 1.0;
                nv = new_nv;
                continue;
            }

            let mut total_weight = 0.0f64;
            for a in 0..n_assets {
                if in_pool[a] {
                    let w = group_weights[[day - 1, a]];
                    if w.is_finite() && w > 0.0 {
                        total_weight += w;
                    }
                }
            }
            if total_weight == 0.0 {
                total_weight = pool_count as f64;
            }

            let mut new_shares: Vec<f64> = vec![0.0f64; n_assets];
            let mut asset_close = 0.0f64;
            let mut fee_dollars = 0.0f64;

            for a in 0..n_assets {
                if in_pool[a] {
                    let op = open[[day, a]];
                    let w = group_weights[[day - 1, a]];
                    let alloc = if w.is_finite() && w > 0.0 {
                        asset * (w / total_weight)
                    } else {
                        asset / pool_count as f64
                    };
                    if op.is_finite() && op > 0.0 {
                        new_shares[a] = alloc / op;
                        let cl = close[[day, a]];
                        if cl.is_finite() {
                            asset_close += new_shares[a] * cl;
                        }
                    }
                } else if tradable[[day, a]] <= 0.5 {
                    new_shares[a] = prev_shares[a];
                    let cl = close[[day, a]];
                    if cl.is_finite() {
                        asset_close += new_shares[a] * cl;
                    }
                }

                let delta = new_shares[a] - prev_shares[a];
                let vp = vwap[[day, a]];
                if delta.abs() > 1e-15 && vp.is_finite() && vp > 0.0 {
                    fee_dollars += delta.abs() * fee_rate * vp;
                }
            }

            let new_nv = (asset_close - fee_dollars).max(0.0);
            group_returns[[day - 1, group]] = new_nv / nv - 1.0;

            nv = new_nv;
            prev_shares = new_shares;
        }
    }

    Ok(group_returns)
}

/// Compute holding return using adjusted prices
pub fn compute_holding_return(
    weights: &Array2<f64>,
    close: &Array2<f64>,
    adj_factor: &Array2<f64>,
) -> Array2<f64> {
    let (n_days, n_symbols) = weights.dim();
    let (_, n_symbols_check) = close.dim();
    assert_eq!(
        n_symbols, n_symbols_check,
        "Weights and close must have same number of symbols"
    );

    let last_adj_factor = adj_factor.row(n_days - 1);
    let mut adj_close = Array2::<f64>::zeros((n_days, n_symbols));
    for day in 0..n_days {
        let adj_factors = adj_factor.row(day);
        for symbol in 0..n_symbols {
            let adj = adj_factors[symbol];
            let last_adj = last_adj_factor[symbol];
            if !adj.is_nan() && !last_adj.is_nan() && last_adj != 0.0 {
                adj_close[[day, symbol]] = close[[day, symbol]] * adj / last_adj;
            } else {
                adj_close[[day, symbol]] = close[[day, symbol]];
            }
        }
    }

    let weights_lag = weights.slice(ndarray::s![0..n_days - 1, ..]);
    let adj_close_lag = adj_close.slice(ndarray::s![0..n_days - 1, ..]);
    let adj_close_current = adj_close.slice(ndarray::s![1.., ..]);
    let price_returns = (&adj_close_current / &adj_close_lag) - 1.0;

    let weighted_returns = &weights_lag * &price_returns;
    let day_returns = weighted_returns.sum_axis(ndarray::Axis(1));

    let mut returns = ndarray::Array2::<f64>::zeros((n_days, 1));
    for day in 1..n_days {
        returns[[day, 0]] = day_returns[day - 1];
    }

    returns
}

/// Compute trading return using adjusted prices with limit-up/down handling
pub fn compute_trading_return(
    weights: &Array2<f64>,
    close: &Array2<f64>,
    open: &Array2<f64>,
    vwap: &Array2<f64>,
    adj_factor: &Array2<f64>,
    fee: f64,
    slippage: f64,
    tradable: &Array2<f64>,
) -> Array2<f64> {
    let (n_days, n_symbols) = weights.dim();
    let total_cost = fee + slippage;

    let last_adj_factor = adj_factor.row(n_days - 1);
    let mut adj_close = Array2::<f64>::zeros((n_days, n_symbols));
    let mut adj_open = Array2::<f64>::zeros((n_days, n_symbols));
    let mut adj_vwap = Array2::<f64>::zeros((n_days, n_symbols));

    for day in 0..n_days {
        let adj_factors = adj_factor.row(day);
        for symbol in 0..n_symbols {
            let adj = adj_factors[symbol];
            let last_adj = last_adj_factor[symbol];
            if !adj.is_nan() && !last_adj.is_nan() && last_adj != 0.0 {
                adj_close[[day, symbol]] = close[[day, symbol]] * adj / last_adj;
                adj_open[[day, symbol]] = open[[day, symbol]] * adj / last_adj;
                adj_vwap[[day, symbol]] = vwap[[day, symbol]] * adj / last_adj;
            } else {
                adj_close[[day, symbol]] = close[[day, symbol]];
                adj_open[[day, symbol]] = open[[day, symbol]];
                adj_vwap[[day, symbol]] = vwap[[day, symbol]];
            }
        }
    }

    let mut returns = ndarray::Array2::<f64>::zeros((n_days, 1));

    let tradable_0 = tradable.row(0);
    let adj_open_0 = adj_open.row(0);
    let adj_close_0 = adj_close.row(0);
    let adj_vwap_0 = adj_vwap.row(0);
    let weights_0 = weights.row(0);

    let mut asset_0 = 0.0;
    let mut fee_0 = 0.0;
    for symbol in 0..n_symbols {
        let w = weights_0[symbol];
        if w != 0.0 && tradable_0[symbol] > 0.5 {
            let shares = w / adj_open_0[symbol];
            asset_0 += shares * adj_close_0[symbol];
            fee_0 += w.abs() * total_cost;
        }
    }
    let nav_0 = if asset_0 > 0.0 { asset_0 - fee_0 } else { 1.0 };
    returns[[0, 0]] = 0.0;

    if n_days > 1 {
        let weights_lag = weights.slice(ndarray::s![0..n_days - 1, ..]);
        let weights_current = weights.slice(ndarray::s![1.., ..]);
        let adj_open_current = adj_open.slice(ndarray::s![1.., ..]);
        let adj_close_current = adj_close.slice(ndarray::s![1.., ..]);
        let adj_vwap_current = adj_vwap.slice(ndarray::s![1.., ..]);
        let tradable_current = tradable.slice(ndarray::s![1.., ..]);

        for day_idx in 0..(n_days - 1) {
            let day = day_idx + 1;
            let prev_day = day - 1;

            let prev_weights = weights_lag.row(day_idx);
            let prev_tradable = tradable.slice(ndarray::s![prev_day, ..]);
            let curr_weights = weights_current.row(day_idx);
            let curr_tradable = tradable_current.row(day_idx);
            let curr_open = adj_open_current.row(day_idx);
            let curr_close = adj_close_current.row(day_idx);
            let curr_vwap = adj_vwap_current.row(day_idx);

            let mut curr_asset = 0.0;
            let mut prev_asset = 0.0;
            let mut fee = 0.0;

            for symbol in 0..n_symbols {
                let pt = prev_tradable[symbol];
                let ct = curr_tradable[symbol];
                let pw = prev_weights[symbol];
                let cw = curr_weights[symbol];
                let co = curr_open[symbol];
                let ccl = curr_close[symbol];
                let cv = curr_vwap[symbol];

                let effective_weight = if ct <= 0.5 { pw } else { cw };

                if pt > 0.5 && pw != 0.0 {
                    let prev_shares = pw / adj_close.row(prev_day)[symbol];
                    prev_asset += prev_shares * ccl;
                }

                if ct > 0.5 && effective_weight != 0.0 {
                    let curr_shares = effective_weight / co;
                    curr_asset += curr_shares * ccl;
                }

                let weight_change = effective_weight - pw;
                if weight_change.abs() > 1e-10 {
                    let fee_rate = fee + slippage;
                    fee += weight_change.abs() * fee_rate;
                }
            }

            let nav = if prev_asset > 0.0 {
                curr_asset - fee
            } else {
                curr_asset
            };
            let day_return = if prev_asset > 0.0 {
                nav / prev_asset - 1.0
            } else {
                0.0
            };
            returns[[day, 0]] = day_return;
        }
    }

    returns
}

/// Compute total portfolio return combining holding and trading returns
pub fn compute_portfolio_return(
    weights: &Array2<f64>,
    close: &Array2<f64>,
    open: &Array2<f64>,
    vwap: &Array2<f64>,
    adj_factor: &Array2<f64>,
    fee: f64,
    slippage: f64,
    tradable: &Array2<f64>,
) -> Array2<f64> {
    let holding_return = compute_holding_return(weights, close, adj_factor);
    let trading_return = compute_trading_return(
        weights, close, open, vwap, adj_factor, fee, slippage, tradable,
    );
    holding_return + trading_return
}
