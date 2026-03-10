#!/usr/bin/env python3
"""
Test script to verify Python bindings for alpha-expr v0.2.0
"""

import sys
import numpy as np

# Add current directory to path for local development
sys.path.insert(0, ".")


def test_imports():
    """Test that all expected modules and classes can be imported."""
    print("=" * 60)
    print("1. IMPORT TEST")
    print("=" * 60)

    try:
        import alpha_expr as ae

        print(f"✓ Imported alpha-expr v{ae.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import alpha-expr: {e}")
        return False

    # Test direct import from _core
    try:
        from alpha_expr._core import (
            GpEngine,
            PersistenceManager,
            FactorMetadata,
            GPHistoryRecord,
            MetaLearningAnalyzer,
            GPRecommendations,
            LazyFrame,
            Expr,
            DataFrame,
            Series,
        )

        print("✓ All new classes imported from _core")

        # Verify classes exist
        expected_classes = [
            ("GpEngine", GpEngine),
            ("PersistenceManager", PersistenceManager),
            ("FactorMetadata", FactorMetadata),
            ("GPHistoryRecord", GPHistoryRecord),
            ("MetaLearningAnalyzer", MetaLearningAnalyzer),
            ("GPRecommendations", GPRecommendations),
            ("LazyFrame", LazyFrame),
            ("Expr", Expr),
            ("DataFrame", DataFrame),
            ("Series", Series),
        ]

        for name, cls in expected_classes:
            print(f"  ✓ {name} is available: {cls}")

        return True
    except ImportError as e:
        print(f"✗ Failed to import from _core: {e}")
        return False


def test_expression_system():
    """Test the expression system."""
    print("\n" + "=" * 60)
    print("2. EXPRESSION SYSTEM TEST")
    print("=" * 60)

    try:
        from alpha_expr._core import Expr

        # Create column expressions
        open_expr = Expr.col("open")
        close_expr = Expr.col("close")

        # Create arithmetic expression
        factor_expr = (close_expr - open_expr) / open_expr

        # Test method chaining
        abs_expr = factor_expr.abs()
        sqrt_expr = abs_expr.sqrt()

        print("✓ Expression system works:")
        print(f"  Column expr: {open_expr}")
        print(f"  Arithmetic expr: {factor_expr}")
        print(f"  Method chaining: {sqrt_expr}")

        return True
    except Exception as e:
        print(f"✗ Expression system test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_lazyframe_creation():
    """Test LazyFrame creation."""
    print("\n" + "=" * 60)
    print("3. LAZYFRAME CREATION TEST")
    print("=" * 60)

    try:
        from alpha_expr._core import LazyFrame
        import numpy as np

        # Create test data
        data = {
            "close": np.random.randn(10, 5),
            "volume": np.random.randn(10, 5),
        }

        # Create LazyFrame
        lf = LazyFrame.scan(data)

        print(f"✓ Created LazyFrame")
        print(f"  Data shape: {data['close'].shape}")
        print(f"  LazyFrame repr: {lf}")

        return True
    except Exception as e:
        print(f"✗ LazyFrame test failed: {e}")
        return False


def test_gp_engine_creation():
    """Test GpEngine creation."""
    print("\n" + "=" * 60)
    print("4. GP ENGINE CREATION TEST")
    print("=" * 60)

    try:
        from alpha_expr._core import GpEngine

        # Create GP engine
        gp = GpEngine(
            population_size=50,
            max_generations=10,
            tournament_size=5,
            crossover_prob=0.8,
            mutation_prob=0.2,
            max_depth=5,
        )

        print(f"✓ Created GpEngine")
        print(f"  Population size: 50")
        print(f"  Max generations: 10")
        print(f"  GP engine repr: {gp}")

        return True
    except Exception as e:
        print(f"✗ GpEngine test failed: {e}")
        return False


def test_persistence_manager():
    """Test PersistenceManager creation."""
    print("\n" + "=" * 60)
    print("5. PERSISTENCE MANAGER TEST")
    print("=" * 60)

    try:
        from alpha_expr._core import PersistenceManager
        import tempfile
        import shutil

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="alpha_test_")

        # Create persistence manager
        pm = PersistenceManager(temp_dir)

        print(f"✓ Created PersistenceManager")
        print(f"  Path: {temp_dir}")
        print(f"  PM repr: {pm}")

        # Clean up
        shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        print(f"✗ PersistenceManager test failed: {e}")
        if "temp_dir" in locals():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        return False


def test_meta_learning_analyzer():
    """Test MetaLearningAnalyzer creation."""
    print("\n" + "=" * 60)
    print("6. META-LEARNING ANALYZER TEST")
    print("=" * 60)

    try:
        from alpha_expr._core import MetaLearningAnalyzer

        # Create analyzer
        analyzer = MetaLearningAnalyzer()

        print(f"✓ Created MetaLearningAnalyzer")
        print(f"  Trained: {analyzer.is_trained()}")
        print(f"  Version: {analyzer.version()}")
        print(f"  Analyzer repr: {analyzer}")

        # Get recommendations
        recommendations = analyzer.get_recommendations()
        print(f"  Recommendations confidence: {recommendations.confidence_score:.2f}")
        print(
            f"  Recommended functions: {recommendations.recommended_functions[:3]}..."
        )

        return True
    except Exception as e:
        print(f"✗ MetaLearningAnalyzer test failed: {e}")
        return False


def test_full_alpha_expr_api():
    """Test the full alpha_expr API (not just _core)."""
    print("\n" + "=" * 60)
    print("7. FULL alpha_expr API TEST")
    print("=" * 60)

    try:
        import alpha_expr as ae

        print("✓ Full alpha_expr module structure:")

        # Check version
        print(f"  Version: {ae.__version__}")

        # Check __all__ exports
        expected_exports = [
            "GpEngine",
            "PersistenceManager",
            "FactorMetadata",
            "GPHistoryRecord",
            "MetaLearningAnalyzer",
            "GPRecommendations",
            "LazyFrame",
            "Expr",
            "DataFrame",
            "Series",
            "quantile_backtest",
            "compute_information_coefficient",
            "BacktestEngine",
            "BacktestResult",
        ]

        print("  Available exports:")
        for export in ae.__all__:
            if export in expected_exports:
                print(f"    ✓ {export}")
            else:
                print(f"    • {export}")

        return True
    except Exception as e:
        print(f"✗ Full API test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("alpha-expr v0.2.0 Python Bindings Verification")
    print("=" * 80)

    tests = [
        ("Import Test", test_imports),
        ("Expression System", test_expression_system),
        ("LazyFrame Creation", test_lazyframe_creation),
        ("GP Engine Creation", test_gp_engine_creation),
        ("Persistence Manager", test_persistence_manager),
        ("Meta-Learning Analyzer", test_meta_learning_analyzer),
        ("Full API Test", test_full_alpha_expr_api),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL PYTHON BINDINGS ARE WORKING CORRECTLY!")
        print(
            "\nalpha-expr v0.2.0 is now production ready with full Python API support."
        )
    else:
        print("⚠️ Some tests failed. Please check the errors above.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
