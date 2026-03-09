"""
Pure Python fallback implementation of backtest functions.
Used when Rust extension is not available.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PyBacktestEngine:
    """Mock engine for fallback mode."""
    pass


@dataclass
class PyBacktestResult:
    """Mock result for fallback mode."""
    group_returns: np.ndarray
    group_cum_returns: np.ndarray
    long_short_returns: np.ndarray
    long_short_cum_return: float
    ic_series: np.ndarray
    ic_mean: float
    ic_ir: float


def quantile_backtest(
    factor: np.ndarray,
    returns: np.ndarray,
    quantiles: int = 10,
    weight_method: str = "equal",
    long_top_n: int = 1,
    short_top_n: int = 1,
    commission_rate: float = 0.0,
    weights: Optional[np.ndarray] = None,
) -> PyBacktestResult:
    """
    Pure Python implementation of quantile backtest.
    """
    n_days, n_assets = factor.shape
    
    # Initialize results
    group_returns = np.zeros((n_days - 1, quantiles))
    group_cum_returns = np.zeros((n_days - 1, quantiles))
    ic_series = np.zeros(n_days - 1)
    
    # Process each day
    for day in range(n_days - 1):
        # Get factor values for current day
        factor_today = factor[day]
        returns_tomorrow = returns[day + 1]
        
        # Remove NaN values
        valid_mask = ~np.isnan(factor_today) & ~np.isnan(returns_tomorrow)
        if valid_mask.sum() < quantiles:
            continue
        
        factor_valid = factor_today[valid_mask]
        returns_valid = returns_tomorrow[valid_mask]
        indices_valid = np.where(valid_mask)[0]
        
        # Sort by factor value
        sort_idx = np.argsort(factor_valid)
        factor_sorted = factor_valid[sort_idx]
        returns_sorted = returns_valid[sort_idx]
        indices_sorted = indices_valid[sort_idx]
        
        # Assign quantile groups
        n_valid = len(factor_sorted)
        group_size = n_valid // quantiles
        groups = np.zeros(n_assets, dtype=int)
        
        for i, idx in enumerate(indices_sorted):
            group = min(i // group_size, quantiles - 1)
            groups[idx] = group + 1  # 1-indexed
        
        # Compute group returns (equal weight)
        for g in range(1, quantiles + 1):
            group_mask = groups == g
            if group_mask.any():
                group_return = returns_tomorrow[group_mask].mean()
                group_returns[day, g - 1] = group_return
        
        # Compute IC
        ic = np.corrcoef(factor_today[valid_mask], returns_tomorrow[valid_mask])[0, 1]
        ic_series[day] = ic
    
    # Compute cumulative returns
    for g in range(quantiles):
        cum = 1.0
        for day in range(n_days - 1):
            cum *= 1.0 + group_returns[day, g]
            group_cum_returns[day, g] = cum - 1.0
    
    # Compute long-short portfolio
    long_groups = list(range(quantiles - long_top_n, quantiles))
    short_groups = list(range(short_top_n))
    
    long_short_returns = np.zeros(n_days - 1)
    for day in range(n_days - 1):
        long_return = group_returns[day, long_groups].mean() if long_groups else 0.0
        short_return = group_returns[day, short_groups].mean() if short_groups else 0.0
        long_short_returns[day] = long_return - short_return - commission_rate
    
    # Compute cumulative long-short return
    long_short_cum_return = (1.0 + long_short_returns).prod() - 1.0
    
    # Compute IC statistics
    valid_ic = ic_series[~np.isnan(ic_series)]
    ic_mean = valid_ic.mean() if len(valid_ic) > 0 else 0.0
    ic_std = valid_ic.std() if len(valid_ic) > 1 else 0.0
    ic_ir = ic_mean / ic_std if ic_std != 0 else 0.0
    
    return PyBacktestResult(
        group_returns=group_returns,
        group_cum_returns=group_cum_returns,
        long_short_returns=long_short_returns,
        long_short_cum_return=long_short_cum_return,
        ic_series=ic_series,
        ic_mean=ic_mean,
        ic_ir=ic_ir,
    )


def compute_ic(
    factor: np.ndarray,
    returns: np.ndarray,
) -> Tuple[float, float]:
    """
    Pure Python implementation of IC computation.
    """
    n_days, n_assets = factor.shape
    ic_vals = []
    
    for day in range(n_days - 1):
        factor_today = factor[day]
        returns_tomorrow = returns[day + 1]
        
        valid_mask = ~np.isnan(factor_today) & ~np.isnan(returns_tomorrow)
        if valid_mask.sum() < 2:
            continue
        
        ic = np.corrcoef(factor_today[valid_mask], returns_tomorrow[valid_mask])[0, 1]
        if not np.isnan(ic):
            ic_vals.append(ic)
    
    if not ic_vals:
        return 0.0, 0.0
    
    ic_mean = np.mean(ic_vals)
    ic_std = np.std(ic_vals)
    ic_ir = ic_mean / ic_std if ic_std != 0 else 0.0
    
    return ic_mean, ic_ir