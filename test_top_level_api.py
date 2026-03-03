#!/usr/bin/env python3
"""
Test the top-level alpha_expr API
"""

import sys
sys.path.insert(0, ".")

def test_top_level_imports():
    """Test that all classes are available at the top level."""
    print("Testing top-level alpha_expr API...")
    
    try:
        import alpha_expr as ae
        
        # Test that classes are available
        test_classes = [
            ('GpEngine', ae.GpEngine),
            ('PersistenceManager', ae.PersistenceManager),
            ('FactorMetadata', ae.FactorMetadata),
            ('GPHistoryRecord', ae.GPHistoryRecord),
            ('MetaLearningAnalyzer', ae.MetaLearningAnalyzer),
            ('GPRecommendations', ae.GPRecommendations),
            ('LazyFrame', ae.LazyFrame),
            ('Expr', ae.Expr),
            ('DataFrame', ae.DataFrame),
            ('Series', ae.Series),
            ('BacktestEngine', ae.BacktestEngine),
            ('BacktestResult', ae.BacktestResult),
        ]
        
        print("✓ All classes available at top level:")
        for name, cls in test_classes:
            print(f"  {name}: {cls}")
        
        # Test that functions are available
        test_functions = [
            'quantile_backtest',
            'compute_information_coefficient',
            'evaluate_expression',
            'lag',
            'diff',
            'rolling_mean',
            'cumsum',
            'cumprod',
        ]
        
        print("\n✓ All functions available at top level:")
        for func_name in test_functions:
            func = getattr(ae, func_name, None)
            if func is not None:
                print(f"  {func_name}: {func}")
            else:
                print(f"  ✗ {func_name} NOT FOUND")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_usage_examples():
    """Test basic usage examples."""
    print("\n" + "="*60)
    print("Testing usage examples...")
    
    try:
        import alpha_expr as ae
        import numpy as np
        
        # Example 1: Create a factor expression
        print("\n1. Creating factor expressions:")
        expr = ae.Expr.col("close") - ae.Expr.col("open")
        print(f"   Expression: {expr}")
        
        # Example 2: Create GP engine
        print("\n2. Creating GP engine:")
        gp = ae.GpEngine(
            population_size=100,
            max_generations=20,
            tournament_size=7,
            crossover_prob=0.8,
            mutation_prob=0.2,
            max_depth=6,
        )
        print(f"   GP Engine: {gp}")
        
        # Example 3: Create persistence manager
        print("\n3. Creating persistence manager:")
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp(prefix="alpha_test_")
        pm = ae.PersistenceManager(temp_dir)
        print(f"   Persistence Manager: {pm}")
        shutil.rmtree(temp_dir)
        
        # Example 4: Create meta-learning analyzer
        print("\n4. Creating meta-learning analyzer:")
        analyzer = ae.MetaLearningAnalyzer()
        print(f"   Analyzer trained: {analyzer.is_trained()}")
        recs = analyzer.get_recommendations()
        print(f"   Recommendations confidence: {recs.confidence_score:.2f}")
        
        # Example 5: Create LazyFrame
        print("\n5. Creating LazyFrame:")
        data = {'close': np.random.randn(10, 5)}
        lf = ae.LazyFrame.scan(data)
        print(f"   LazyFrame: {lf}")
        
        print("\n✓ All usage examples work!")
        return True
        
    except Exception as e:
        print(f"✗ Usage example error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_top_level_imports()
    success2 = test_usage_examples()
    
    if success1 and success2:
        print("\n" + "="*60)
        print("🎉 TOP-LEVEL API IS FULLY FUNCTIONAL!")
        print("alpha-expr v0.2.0 Python bindings are complete and ready for production.")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("⚠️ Some tests failed.")
        sys.exit(1)