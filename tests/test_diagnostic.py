#!/usr/bin/env python3
"""
Diagnostic script to check what's exported by the _core module.
"""

import sys
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

try:
    # Try to import _core directly
    import alpha_expr._core as core

    print("Successfully imported _core module")
    print(f"Module location: {core.__file__}")

    # List all public attributes
    public_attrs = [attr for attr in dir(core) if not attr.startswith("_")]
    print(f"\nPublic attributes ({len(public_attrs)}):")
    for attr in sorted(public_attrs):
        obj = getattr(core, attr)
        print(f"  {attr}: {type(obj).__name__}")

    # Check for specific classes we expect
    expected = [
        "PyBacktestEngine",
        "PyBacktestResult",
        "Expr",
        "Series",
        "DataFrame",
        "LazyFrame",
        "quantile_backtest",
        "compute_ic",
        "evaluate_expression",
        "lag",
        "diff",
        "rolling_mean",
        "cumsum",
        "cumprod",
        "rolling_window",
        "expanding_window",
    ]

    print(f"\nChecking for expected exports:")
    for exp in expected:
        if hasattr(core, exp):
            print(f"  ✓ {exp}")
        else:
            print(f"  ✗ {exp} (missing)")

except Exception as e:
    print(f"Error importing _core: {e}")
    import traceback

    traceback.print_exc()
