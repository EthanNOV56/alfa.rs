#!/usr/bin/env python3
"""Test the weighted close ratio factor."""

import numpy as np
import alfars as al

# Create sample data: 10 days, 3 assets
np.random.seed(42)
n_days = 10
n_assets = 3

close = np.random.rand(n_days, n_assets) * 100 + 50  # 50-150
volume = np.random.rand(n_days, n_assets) * 1000000 + 100000  # 1M-2M

print("=== Test Data ===")
print(f"Close shape: {close.shape}")
print(f"Volume shape: {volume.shape}")
print()

# Test 1: ts_sum with window=0 (expanding)
print("=== Test 1: ts_sum with expanding window (window=0) ===")
data = {"close": close, "volume": volume}
result = al.evaluate_expression(
    al.ts_sum(al.Expr.col("close"), 0),
    data,
    n_days,
    n_assets
)
print("ts_sum(close, 0) - cumulative sum from start:")
print(result)
print()

# Test 2: ts_count with window=0
print("=== Test 2: ts_count with expanding window (window=0) ===")
result = al.evaluate_expression(
    al.ts_count(al.Expr.col("close"), 0),
    data,
    n_days,
    n_assets
)
print("ts_count(close, 0) - cumulative count from start:")
print(result)
print()

# Test 3: Weighted close ratio factor
# Formula: (sum(volume * close, 0) / sum(volume, 0)) / (sum(close, 0) / count(close, 0))
# = (sum(vol * close) / sum(vol)) / (sum(close) / T)
print("=== Test 3: Weighted Close Ratio Factor ===")

# Build expression: (sum(vol * close, 0) / sum(vol, 0)) / (sum(close, 0) / count(close, 0))
vol_close = al.Expr.col("volume") * al.Expr.col("close")
sum_vol_close = al.ts_sum(vol_close, 0)
sum_vol = al.ts_sum(al.Expr.col("volume"), 0)
sum_close = al.ts_sum(al.Expr.col("close"), 0)
count_close = al.ts_count(al.Expr.col("close"), 0)

# vwap = sum(vol * close) / sum(vol)
vwap = sum_vol_close / sum_vol

# arithmetic_mean = sum(close) / count(close)
arithmetic_mean = sum_close / count_close

# ratio = vwap / arithmetic_mean
weighted_close_ratio = vwap / arithmetic_mean

result = al.evaluate_expression(weighted_close_ratio, data, n_days, n_assets)
print("Weighted Close Ratio Factor:")
print(result)
print()

# Test 4: Verify with manual calculation for asset 0
print("=== Test 4: Manual Verification for Asset 0 ===")
close_0 = close[:, 0]
vol_0 = volume[:, 0]
vol_close_0 = vol_0 * close_0

cumsum_vol_close = np.cumsum(vol_close_0)
cumsum_vol = np.cumsum(vol_0)
cumsum_close = np.cumsum(close_0)
count = np.arange(1, n_days + 1)

vwap_manual = cumsum_vol_close / cumsum_vol
arithmetic_manual = cumsum_close / count
ratio_manual = vwap_manual / arithmetic_manual

print(f"Manual calculation for asset 0:")
print(f"  vwap: {vwap_manual}")
print(f"  arithmetic mean: {arithmetic_manual}")
print(f"  ratio: {ratio_manual}")
print()

print("Expected (from expression):")
print(f"  {result[:, 0]}")
print()

print("Match:", np.allclose(result[:, 0], ratio_manual, equal_nan=True))
