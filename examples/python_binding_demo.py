#!/usr/bin/env python3
"""
Python binding demonstration for exprs

This example shows how to use the Python bindings for:
1. Expression building and evaluation
2. DataFrame operations
3. Backtesting
"""

import sys
import numpy as np

# Try to import the exprs module
try:
    import alfars as aexpr

    print("✅ Successfully imported exprs")
except ImportError as e:
    print(f"❌ Failed to import alfars: {e}")
    print("\nTry building the module first:")
    print("  maturin develop  # for development install")
    print("  or")
    print("  pip install .    # for regular install")
    sys.exit(1)


def demo_expressions():
    """Demonstrate expression building and evaluation"""
    print("\n" + "=" * 60)
    print("1. EXPRESSION BUILDING AND EVALUATION")
    print("=" * 60)

    # Create basic expressions
    print("\n📝 Creating expressions:")
    close = aexpr.Expr.col("close")
    volume = aexpr.Expr.col("volume")
    print(f"  close column: {close}")
    print(f"  volume column: {volume}")

    # Build complex expression: (close * 2) + (volume / 1000)
    expr = close.mul(aexpr.Expr.lit_float(2.0)).add(
        volume.div(aexpr.Expr.lit_float(1000.0))
    )
    print(f"  Complex expression: {expr}")

    # Create a simple DataFrame
    print("\n📊 Creating DataFrame:")
    df = aexpr.DataFrame.from_dict(
        {
            "close": [100.0, 101.0, 102.5, 103.0, 104.0],
            "volume": [1000.0, 1200.0, 800.0, 1500.0, 900.0],
        }
    )
    print(f"  DataFrame shape: {df.n_rows()} rows, {df.n_cols()} columns")
    print(f"  Columns: {df.column_names()}")

    # Evaluate expression
    print("\n⚡ Evaluating expression on DataFrame:")
    result = df.evaluate(expr)
    print(f"  Result type: {type(result).__name__}")
    print(f"  Result length: {result.len()}")
    print(f"  Result values: {result.to_list()}")
    print(f"  Result mean: {result.mean():.2f}")
    print(f"  Result std: {result.std(1.0):.2f}")

    # Test Series operations
    print("\n📈 Testing Series operations:")
    series = aexpr.Series.new([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"  Original series: {series.to_list()}")
    print(f"  Lag(1): {series.lag(1).to_list()}")
    print(f"  Diff(1): {series.diff(1).to_list()}")
    print(f"  Moving Average(3): {series.moving_average(3).to_list()}")
    print(f"  EMA(3): {series.ema(3).to_list()}")

    return df, expr


def demo_factor_functions():
    """Demonstrate pre-built factor functions"""
    print("\n" + "=" * 60)
    print("2. PRE-BUILT FACTOR FUNCTIONS")
    print("=" * 60)

    # WCR factor
    wcr = aexpr.create_wcr_factor("close", "volume", 10)
    print(f"\n📊 WCR factor: {wcr}")
    print("  Formula: (close * volume) / moving_average(close, 10)")

    # Momentum factor
    momentum = aexpr.create_momentum_factor("close", 5)
    print(f"\n📈 Momentum factor: {momentum}")
    print("  Formula: (close - lag(close, 5)) / lag(close, 5)")

    # Volatility factor
    volatility = aexpr.create_volatility_factor("close", 20)
    print(f"\n📉 Volatility factor: {volatility}")
    print("  Formula: rolling_std(close, 20)")

    return wcr, momentum, volatility


def demo_backtesting():
    """Demonstrate backtesting functionality"""
    print("\n" + "=" * 60)
    print("3. BACKTESTING DEMONSTRATION")
    print("=" * 60)

    # Create backtest engine
    print("\n⚙️ Creating backtest engine:")
    engine = aexpr.BacktestEngine("equal", 0.0005, True)
    print(f"  Engine created: {engine}")
    print("  Weight method: equal weighting")
    print("  Commission rate: 0.05%")
    print("  IC calculation: enabled")

    # Simulate some data
    print("\n📊 Simulating backtest data:")
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    dates = [f"2023-01-{i + 1:02d}" for i in range(10)]

    # Generate random factors and returns
    np.random.seed(42)
    n_dates = len(dates)
    n_symbols = len(symbols)

    factors = []
    returns = []

    for i in range(n_dates):
        # Random factors (mean 0, std 1)
        day_factors = list(np.random.randn(n_symbols))
        factors.append(day_factors)

        # Returns correlated with factors plus noise
        day_returns = [f * 0.1 + np.random.randn() * 0.02 for f in day_factors]
        returns.append(day_returns)

    print(f"  Symbols: {symbols}")
    print(f"  Dates: {dates[:3]}... ({len(dates)} total)")
    print(f"  Factors shape: {len(factors)} x {len(factors[0])}")
    print(f"  Returns shape: {len(returns)} x {len(returns[0])}")

    # Run backtest with 3 groups
    print("\n🏃 Running backtest (qcut with 3 groups)...")
    try:
        result = engine.qcut_backtest(symbols, dates, factors, returns, 3)
        print("  ✅ Backtest completed successfully!")

        # Display results
        print("\n📊 Backtest Results:")
        print(f"  Sharpe Ratio: {result.sharpe_ratio():.3f}")
        print(f"  Max Drawdown: {result.max_drawdown():.2%}")
        print(f"  Annualized Return: {result.annualized_return():.2%}")
        print(f"  Annualized Volatility: {result.annualized_volatility():.2%}")
        print(f"  Win Rate: {result.win_rate():.2%}")
        print(f"  Long-Short Cumulative Return: {result.long_short_cum_return():.2%}")

        if result.ic() is not None:
            print(f"  Information Coefficient (IC): {result.ic():.3f}")

        # Show group returns
        group_returns = result.group_returns()
        print(
            f"\n📈 Group Returns (shape: {len(group_returns)} x {len(group_returns[0])}):"
        )
        for i in range(min(3, len(group_returns))):
            print(f"  Group {i + 1} returns (first 5 days): {group_returns[i][:5]}")

        return result
    except Exception as e:
        print(f"  ❌ Backtest failed: {e}")
        print("  (This might be due to insufficient data or other issues)")
        return None


def demo_clickhouse():
    """Demonstrate ClickHouse integration (if available)"""
    print("\n" + "=" * 60)
    print("4. CLICKHOUSE INTEGRATION")
    print("=" * 60)

    print("\n💾 Note: ClickHouse integration requires a running ClickHouse server.")
    print("   You can skip this section if ClickHouse is not available.")

    response = input("\nDo you want to test ClickHouse integration? (y/N): ")
    if response.lower() != "y":
        print("  Skipping ClickHouse demo...")
        return None

    try:
        # Try to connect to local ClickHouse
        print("\n🔗 Connecting to ClickHouse (localhost:9000)...")
        ch = aexpr.ClickHouseProvider(
            "localhost", 9000, "readonly", "readonly", "default"
        )
        print("  ✅ Connected successfully!")

        # Simple query
        print("\n📋 Executing test query...")
        results = ch.query("SELECT 1 as test_value, 'hello' as greeting")

        if results:
            print(f"  ✅ Query returned {len(results)} rows")
            print(f"  First row: {results[0]}")
        else:
            print("  ⚠️ Query returned no results")

        return ch
    except Exception as e:
        print(f"  ❌ ClickHouse connection failed: {e}")
        print("\n  To test ClickHouse, make sure:")
        print("  1. ClickHouse is running on localhost:9000")
        print("  2. User 'readonly' with password 'readonly' exists")
        print("  3. Database 'default' is accessible")
        return None


def performance_comparison():
    """Compare performance with pure Python"""
    print("\n" + "=" * 60)
    print("5. PERFORMANCE COMPARISON")
    print("=" * 60)

    print("\n⏱️ Generating large dataset for performance test...")
    n_rows = 10000
    n_cols = 10

    # Generate random data
    np.random.seed(42)
    data = {}
    for i in range(n_cols):
        data[f"col_{i}"] = np.random.randn(n_rows).tolist()

    print(f"  Dataset: {n_rows:,} rows x {n_cols} columns")

    # Create DataFrame
    import time

    start = time.time()
    df = aexpr.DataFrame.from_dict(data)
    rust_load_time = time.time() - start
    print(f"\n🦀 Rust DataFrame creation: {rust_load_time * 1000:.1f}ms")

    # Complex expression evaluation
    print("\n🧮 Evaluating complex expression...")
    expr = aexpr.Expr.col("col_0")
    for i in range(1, 5):
        expr = expr.add(aexpr.Expr.col(f"col_{i}"))
    expr = expr.div(aexpr.Expr.lit_float(5.0))

    start = time.time()
    result = df.evaluate(expr)
    rust_eval_time = time.time() - start
    print(f"  Rust evaluation time: {rust_eval_time * 1000:.1f}ms")

    # Compare with pure Python (simple implementation)
    print("\n🐍 Pure Python equivalent (for comparison)...")
    start = time.time()
    python_result = []
    for i in range(n_rows):
        val = 0.0
        for j in range(5):
            val += data[f"col_{j}"][i]
        val /= 5.0
        python_result.append(val)
    python_time = time.time() - start

    print(f"  Python evaluation time: {python_time * 1000:.1f}ms")
    print(f"  Speedup: {python_time / rust_eval_time:.1f}x")

    # Verify results match
    rust_values = result.to_list()
    max_diff = max(abs(r - p) for r, p in zip(rust_values, python_result))
    print(f"  Max difference: {max_diff:.6f}")
    if max_diff < 1e-10:
        print("  ✅ Results match perfectly!")
    else:
        print("  ⚠️ Small numerical differences detected")


def main():
    """Run all demonstrations"""
    print("🌟 ALPHA-EXPR PYTHON BINDING DEMONSTRATION")
    print("=" * 60)

    # Check version
    print(f"Version: {aexpr.__version__}")

    # Run demos
    demo_expressions()
    demo_factor_functions()
    demo_backtesting()
    demo_clickhouse()
    performance_comparison()

    print("\n" + "=" * 60)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("\nSummary of capabilities demonstrated:")
    print("✅ Expression building and evaluation")
    print("✅ DataFrame operations with vectorized computation")
    print("✅ Time series functions (lag, diff, moving averages)")
    print("✅ Backtesting with multiple weighting methods")
    print("✅ Performance metrics calculation")
    print("✅ ClickHouse database integration")
    print("✅ Significant performance advantages over pure Python")

    print("\nNext steps:")
    print("1. Build more complex alpha factors using the expression API")
    print("2. Connect to your ClickHouse database for real data")
    print("3. Run backtests on historical factor data")
    print("4. Integrate with your existing Python quantitative workflow")


if __name__ == "__main__":
    main()
