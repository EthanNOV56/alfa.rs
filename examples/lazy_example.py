#!/usr/bin/env python3
"""
Example of using the lazy evaluation system in exprs.

This demonstrates how to build factor expressions lazily with automatic
optimization and efficient execution.
"""

import numpy as np

# Import exprs
try:
    from exprs._core import LazyFrame, Expr, rolling_window, expanding_window
    print("Successfully imported exprs lazy modules")
except ImportError:
    print("Warning: Could not import from exprs._core. Building extension first...")
    # Try to build the extension
    import subprocess
    import sys
    import os
    
    # Change to the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_dir)
    
    # Build the extension
    print("Building extension with maturin...")
    result = subprocess.run([sys.executable, "-m", "maturin", "develop", "--release"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        sys.exit(1)
    
    print("Build successful, trying to import again...")
    from exprs._core import LazyFrame, Expr, rolling_window, expanding_window

# Create sample market data
np.random.seed(42)
n_days = 100
n_assets = 50

# Generate price data with some structure
prices = 100 + np.cumsum(np.random.randn(n_days, n_assets) * 0.02, axis=0)
volumes = 1000000 + np.random.randn(n_days, n_assets) * 200000

# Compute returns
returns = np.zeros_like(prices)
returns[1:] = prices[1:] / prices[:-1] - 1.0
returns[0] = np.nan

# Create data dictionary for LazyFrame
data = {
    "close": prices,
    "volume": volumes,
    "returns": returns
}

print(f"Created sample data: {n_days} days × {n_assets} assets")

# ============================================================================
# Example 1: Basic lazy factor construction
# ============================================================================

print("\n" + "="*80)
print("Example 1: Basic lazy factor construction")
print("="*80)

# Create a LazyFrame from the data
lazy_df = LazyFrame.scan(data)

# Build a simple factor: price momentum (20-day return)
momentum_expr = Expr.col("close").pct_change(20)  # 20-day return

# Add the factor column
lazy_with_factor = lazy_df.with_columns([("momentum_20d", momentum_expr)])

# Explain the plan (unoptimized)
print("\nLogical plan (unoptimized):")
print(lazy_with_factor.explain(optimized=False))

# Explain the plan (optimized)
print("\nLogical plan (optimized):")
print(lazy_with_factor.explain(optimized=True))

# Collect the results
print("\nExecuting lazy computation...")
results = lazy_with_factor.collect()

print(f"Computed {len(results)} columns:")
for col_name in results.keys():
    array = results[col_name]
    print(f"  - {col_name}: shape={array.shape}, dtype={array.dtype}")

# Extract the factor for analysis
momentum_factor = results["momentum_20d"]
print(f"\nMomentum factor stats:")
print(f"  Mean: {np.nanmean(momentum_factor):.6f}")
print(f"  Std: {np.nanstd(momentum_factor):.6f}")
print(f"  Min: {np.nanmin(momentum_factor):.6f}")
print(f"  Max: {np.nanmax(momentum_factor):.6f}")

# ============================================================================
# Example 2: Complex factor with multiple operations
# ============================================================================

print("\n" + "="*80)
print("Example 2: Complex factor with multiple operations")
print("="*80)

# Start fresh
lazy_df2 = LazyFrame.scan(data)

# Build a more complex factor:
# 1. Volume-adjusted price: close * log(volume)
# 2. 10-day moving average of volume-adjusted price
# 3. Z-score of the moving average (relative to 20-day mean and std)

# Volume-adjusted price
volume_adj_expr = Expr.col("close") * Expr.col("volume").log()

# 10-day moving average
ma_10_expr = volume_adj_expr.rolling_mean(10)

# Z-score: (value - 20-day mean) / 20-day std
ma_20_expr = volume_adj_expr.rolling_mean(20)
std_20_expr = volume_adj_expr.rolling_std(20)
zscore_expr = (ma_10_expr - ma_20_expr) / std_20_expr

# Add all intermediate columns and final factor
lazy_complex = lazy_df2.with_columns([
    ("volume_adj_price", volume_adj_expr),
    ("ma_10d", ma_10_expr),
    ("ma_20d", ma_20_expr),
    ("std_20d", std_20_expr),
    ("volume_zscore", zscore_expr)
])

print("\nExecuting complex factor computation...")
complex_results = lazy_complex.collect()

print(f"\nComputed complex factor stats:")
for col_name in ["volume_adj_price", "ma_10d", "ma_20d", "std_20d", "volume_zscore"]:
    if col_name in complex_results:
        array = complex_results[col_name]
        valid_data = array[~np.isnan(array)]
        if len(valid_data) > 0:
            print(f"  {col_name}:")
            print(f"    Shape: {array.shape}, Valid: {len(valid_data)}/{array.size}")
            print(f"    Mean: {np.mean(valid_data):.6f}, Std: {np.std(valid_data):.6f}")

# ============================================================================
# Example 3: Stateful operations (cumulative, EMA)
# ============================================================================

print("\n" + "="*80)
print("Example 3: Stateful operations")
print("="*80)

# Note: Stateful operations need to be implemented in the execution engine
# For now, we'll demonstrate the API

lazy_df3 = LazyFrame.scan(data)

# Create expressions for stateful operations
# (These would need proper implementation in the execution engine)
cumsum_expr = Expr.col("returns").cumsum()
# ema_expr = Expr.col("close").ema(20)  # 20-day EMA

lazy_stateful = lazy_df3.with_columns([
    ("cumulative_returns", cumsum_expr),
    # ("ema_20d", ema_expr)  # Would need EMA implementation
])

print("\nLogical plan for stateful operations:")
print(lazy_stateful.explain(optimized=True))

# Try to execute (may fail if stateful ops not implemented)
try:
    stateful_results = lazy_stateful.collect()
    print("\nStateful computation successful!")
    if "cumulative_returns" in stateful_results:
        cr = stateful_results["cumulative_returns"]
        print(f"Cumulative returns shape: {cr.shape}")
except Exception as e:
    print(f"\nStateful execution not fully implemented yet: {e}")

# ============================================================================
# Example 4: Window operations with specification
# ============================================================================

print("\n" + "="*80)
print("Example 4: Window operations with specification")
print("="*80)

# Create window specifications
rolling_20d = rolling_window(20, min_periods=10)
expanding = expanding_window(min_periods=5)

print(f"Rolling window spec: {rolling_20d}")
print(f"Expanding window spec: {expanding}")

# ============================================================================
# Example 5: Realistic factor - Reference Price (RP) and CGO
# ============================================================================

print("\n" + "="*80)
print("Example 5: Realistic factor - Reference Price and CGO")
print("="*80)

"""
Reference Price (RP) factor formula:
RPₜ = (1/k) Σₙ₌₀ᵀ⁻¹ [ Vₜ₋ₙ Πₛ₌₁ⁿ⁻¹ (1 - Vₜ₋ₙ₊ₛ) ] Pₜ₋ₙ

Capital Gains Overhang (CGO):
CGOᵢₜ = (Closeᵢₜ - RPₜ) / Closeᵢₜ

Where V is normalized volume.
"""

# This would be a complex implementation requiring:
# 1. Volume normalization
# 2. Cumulative product of (1 - V) over window
# 3. Weighted sum of prices
# 4. Final CGO calculation

# For now, we'll demonstrate a simplified version
lazy_df5 = LazyFrame.scan(data)

# Simplified: Normalized volume (relative to 20-day mean)
norm_volume_expr = Expr.col("volume") / Expr.col("volume").rolling_mean(20)

# Simplified reference price: 10-day volume-weighted average
weighted_price_expr = Expr.col("close") * norm_volume_expr
ref_price_expr = weighted_price_expr.rolling_sum(10) / norm_volume_expr.rolling_sum(10)

# CGO calculation
cgo_expr = (Expr.col("close") - ref_price_expr) / Expr.col("close")

lazy_cgo = lazy_df5.with_columns([
    ("norm_volume", norm_volume_expr),
    ("ref_price", ref_price_expr),
    ("CGO", cgo_expr)
])

print("\nExecuting CGO factor computation...")
try:
    cgo_results = lazy_cgo.collect()
    print(f"Successfully computed CGO factor")
    
    if "CGO" in cgo_results:
        cgo_factor = cgo_results["CGO"]
        valid_cgo = cgo_factor[~np.isnan(cgo_factor)]
        if len(valid_cgo) > 0:
            print(f"CGO factor statistics:")
            print(f"  Mean: {np.mean(valid_cgo):.6f}")
            print(f"  Std: {np.std(valid_cgo):.6f}")
            print(f"  Min: {np.min(valid_cgo):.6f}")
            print(f"  Max: {np.max(valid_cgo):.6f}")
            
            # Check if CGO has predictive power for returns
            # (simplified correlation calculation)
            returns_next_day = np.roll(data["returns"], -1, axis=0)
            returns_next_day[-1, :] = np.nan
            
            # Flatten and correlate
            cgo_flat = cgo_factor[:-1].flatten()
            returns_flat = returns_next_day[:-1].flatten()
            
            mask = ~np.isnan(cgo_flat) & ~np.isnan(returns_flat)
            if np.sum(mask) > 10:
                corr = np.corrcoef(cgo_flat[mask], returns_flat[mask])[0, 1]
                print(f"  Correlation with next-day returns: {corr:.6f}")
except Exception as e:
    print(f"CGO computation failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("Summary")
print("="*80)

print("\nLazy evaluation system provides:")
print("1. Declarative factor construction")
print("2. Automatic optimization of computation graphs")
print("3. Parallel execution across assets")
print("4. Caching of intermediate results")
print("5. Integration with existing expression system")

print("\nKey benefits for factor research:")
print("- Natural expression of complex formulas")
print("- Reuse of intermediate computations")
print("- Easy experimentation with different parameters")
print("- Good performance through optimization")

print("\nNote: Some features (stateful ops, advanced windows) need implementation")
print("in the execution engine, but the API is ready for use.")