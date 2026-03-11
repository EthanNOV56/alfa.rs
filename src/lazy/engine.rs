//! Lazy evaluation engine
//!
//! Re-exports all lazy evaluation components and provides helper functions.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::Array2;

use crate::expr::Expr;

// Re-export types from submodules
pub use super::frame::{LazyFrame, LazyFrameBuilder};
pub use super::optimizer::LazyOptimizer;
pub use super::executor::LazyExecutor;
pub use super::plan::{
    DataFormat, DataSource, JoinType, LogicalPlan, OptimizationLevel, StatefulExpr, WindowKind,
    WindowSpec,
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a rolling window specification
pub fn rolling_window(size: usize, min_periods: Option<usize>) -> WindowSpec {
    WindowSpec {
        kind: WindowKind::Rolling,
        size: Some(size),
        min_periods: min_periods.unwrap_or(1),
    }
}

/// Create an expanding window specification
pub fn expanding_window(min_periods: Option<usize>) -> WindowSpec {
    WindowSpec {
        kind: WindowKind::Expanding,
        size: None,
        min_periods: min_periods.unwrap_or(1),
    }
}

/// Create a cumulative sum expression
pub fn cumsum(expr: Expr) -> StatefulExpr {
    StatefulExpr::CumSum(expr)
}

/// Create a cumulative product expression
pub fn cumprod(expr: Expr) -> StatefulExpr {
    StatefulExpr::CumProd(expr)
}

/// Create an exponential moving average expression
pub fn ema(expr: Expr, span: usize) -> StatefulExpr {
    let alpha = 2.0 / (span as f64 + 1.0);
    StatefulExpr::Ema(expr, alpha)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;

    /// Create test data with close and volume columns (3 days × 4 assets)
    fn create_test_data() -> HashMap<String, Array2<f64>> {
        let mut data = HashMap::new();
        data.insert(
            "close".to_string(),
            Array2::from_shape_vec(
                (3, 4),
                vec![
                    1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0,
                ],
            )
            .unwrap(),
        );
        data.insert(
            "volume".to_string(),
            Array2::from_shape_vec(
                (3, 4),
                vec![
                    100.0, 200.0, 300.0, 400.0, 400.0, 300.0, 200.0, 100.0, 200.0, 300.0, 400.0, 100.0,
                ],
            )
            .unwrap(),
        );
        data
    }

    // ========================================================================
    // LogicalPlan and Data Structure Tests
    // ========================================================================

    #[test]
    fn test_logical_plan_scan() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let plan = LogicalPlan::Scan {
            source,
            projection: Some(vec!["close".to_string()]),
            selection: None,
        };

        match &plan {
            LogicalPlan::Scan { source, projection, selection } => {
                assert!(matches!(source, DataSource::NumpyArrays(_)));
                assert!(projection.is_some());
                assert!(selection.is_none());
            }
            _ => panic!("Expected Scan plan"),
        }
    }

    #[test]
    fn test_logical_plan_projection() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);
        let input = Arc::new(LogicalPlan::Scan {
            source,
            projection: None,
            selection: None,
        });

        let exprs = vec![
            ("return".to_string(), Expr::col("close")),
        ];

        let plan = LogicalPlan::Projection {
            input: input.clone(),
            exprs: exprs.clone(),
        };

        match &plan {
            LogicalPlan::Projection { input: p_input, exprs: p_exprs } => {
                assert!(Arc::ptr_eq(&input, p_input));
                assert_eq!(p_exprs.len(), 1);
            }
            _ => panic!("Expected Projection plan"),
        }
    }

    #[test]
    fn test_logical_plan_filter() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);
        let input = Arc::new(LogicalPlan::Scan {
            source,
            projection: None,
            selection: None,
        });

        let predicate = Expr::col("close").gt(Expr::lit(2.0));

        let plan = LogicalPlan::Filter {
            input: input.clone(),
            predicate: predicate.clone(),
        };

        match &plan {
            LogicalPlan::Filter { input: p_input, predicate: p_pred } => {
                assert!(Arc::ptr_eq(&input, p_input));
                assert!(matches!(p_pred, Expr::BinaryExpr { .. }));
            }
            _ => panic!("Expected Filter plan"),
        }
    }

    #[test]
    fn test_logical_plan_window() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);
        let input = Arc::new(LogicalPlan::Scan {
            source,
            projection: None,
            selection: None,
        });

        let expr = Expr::col("close");
        let window_spec = WindowSpec {
            kind: WindowKind::Rolling,
            size: Some(3),
            min_periods: 1,
        };

        let plan = LogicalPlan::Window {
            input: input.clone(),
            expr: expr.clone(),
            window_spec: window_spec.clone(),
            output_name: "ma_close".to_string(),
        };

        match &plan {
            LogicalPlan::Window {
                input: w_input,
                expr: w_expr,
                window_spec: w_spec,
                output_name,
            } => {
                assert!(Arc::ptr_eq(&input, w_input));
                assert!(matches!(w_expr, Expr::Column(_)));
                assert!(matches!(w_spec.kind, WindowKind::Rolling));
                assert_eq!(output_name, "ma_close");
            }
            _ => panic!("Expected Window plan"),
        }
    }

    #[test]
    fn test_logical_plan_stateful() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);
        let input = Arc::new(LogicalPlan::Scan {
            source,
            projection: None,
            selection: None,
        });

        let expr = StatefulExpr::CumSum(Expr::col("close"));

        let plan = LogicalPlan::Stateful {
            input: input.clone(),
            expr: expr.clone(),
            output_name: "cumsum_close".to_string(),
        };

        match &plan {
            LogicalPlan::Stateful {
                input: s_input,
                expr: s_expr,
                output_name,
            } => {
                assert!(Arc::ptr_eq(&input, s_input));
                assert!(matches!(s_expr, StatefulExpr::CumSum(_)));
                assert_eq!(output_name, "cumsum_close");
            }
            _ => panic!("Expected Stateful plan"),
        }
    }

    #[test]
    fn test_logical_plan_cache() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);
        let input = Arc::new(LogicalPlan::Scan {
            source,
            projection: None,
            selection: None,
        });

        let plan = LogicalPlan::Cache {
            input: input.clone(),
            key: "test_cache".to_string(),
        };

        match &plan {
            LogicalPlan::Cache { input: c_input, key } => {
                assert!(Arc::ptr_eq(&input, c_input));
                assert_eq!(key, "test_cache");
            }
            _ => panic!("Expected Cache plan"),
        }
    }

    #[test]
    fn test_logical_plan_join() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let left = Arc::new(LogicalPlan::Scan {
            source: source.clone(),
            projection: None,
            selection: None,
        });
        let right = Arc::new(LogicalPlan::Scan {
            source,
            projection: None,
            selection: None,
        });

        let plan = LogicalPlan::Join {
            left: left.clone(),
            right: right.clone(),
            on: vec!["symbol".to_string()],
            how: JoinType::Inner,
        };

        match &plan {
            LogicalPlan::Join {
                left: j_left,
                right: j_right,
                on,
                how,
            } => {
                assert!(Arc::ptr_eq(&left, j_left));
                assert!(Arc::ptr_eq(&right, j_right));
                assert_eq!(on.len(), 1);
                assert!(matches!(how, JoinType::Inner));
            }
            _ => panic!("Expected Join plan"),
        }
    }

    #[test]
    fn test_data_source_numpy_arrays() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data.clone());

        match &source {
            DataSource::NumpyArrays(arrays) => {
                assert_eq!(arrays.len(), 2);
                assert!(arrays.contains_key("close"));
                assert!(arrays.contains_key("volume"));
            }
            _ => panic!("Expected NumpyArrays source"),
        }
    }

    #[test]
    fn test_data_source_external() {
        let source = DataSource::External {
            path: "data/test.csv".to_string(),
            format: DataFormat::Csv,
        };

        match &source {
            DataSource::External { path, format } => {
                assert_eq!(path, "data/test.csv");
                assert!(matches!(format, DataFormat::Csv));
            }
            _ => panic!("Expected External source"),
        }
    }

    #[test]
    fn test_window_spec_rolling() {
        let spec = WindowSpec {
            kind: WindowKind::Rolling,
            size: Some(5),
            min_periods: 2,
        };

        assert!(matches!(spec.kind, WindowKind::Rolling));
        assert_eq!(spec.size, Some(5));
        assert_eq!(spec.min_periods, 2);
    }

    #[test]
    fn test_window_spec_expanding() {
        let spec = WindowSpec {
            kind: WindowKind::Expanding,
            size: None,
            min_periods: 1,
        };

        assert!(matches!(spec.kind, WindowKind::Expanding));
        assert!(spec.size.is_none());
        assert_eq!(spec.min_periods, 1);
    }

    #[test]
    fn test_stateful_expr_variants() {
        let expr = Expr::col("close");

        let cumsum = StatefulExpr::CumSum(expr.clone());
        assert!(matches!(cumsum, StatefulExpr::CumSum(_)));

        let cumprod = StatefulExpr::CumProd(expr.clone());
        assert!(matches!(cumprod, StatefulExpr::CumProd(_)));

        let cummax = StatefulExpr::CumMax(expr.clone());
        assert!(matches!(cummax, StatefulExpr::CumMax(_)));

        let cummin = StatefulExpr::CumMin(expr.clone());
        assert!(matches!(cummin, StatefulExpr::CumMin(_)));

        let ema = StatefulExpr::Ema(expr.clone(), 0.5);
        assert!(matches!(ema, StatefulExpr::Ema(_, _)));
    }

    #[test]
    fn test_join_type_variants() {
        let inner = JoinType::Inner;
        let left = JoinType::Left;
        let right = JoinType::Right;
        let outer = JoinType::Outer;

        assert!(matches!(inner, JoinType::Inner));
        assert!(matches!(left, JoinType::Left));
        assert!(matches!(right, JoinType::Right));
        assert!(matches!(outer, JoinType::Outer));
    }

    // ========================================================================
    // LazyFrame API Tests
    // ========================================================================

    #[test]
    fn test_lazy_frame_scan() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let lf = LazyFrame::scan(source);

        match &*lf.logical_plan {
            LogicalPlan::Scan { source, projection, selection } => {
                assert!(matches!(source, DataSource::NumpyArrays(_)));
                assert!(projection.is_none());
                assert!(selection.is_none());
            }
            _ => panic!("Expected Scan plan"),
        }
    }

    #[test]
    fn test_lazy_frame_with_columns() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let lf = LazyFrame::scan(source)
            .with_columns(vec![
                ("return", Expr::col("close")),
            ]);

        match &*lf.logical_plan {
            LogicalPlan::Projection { exprs, .. } => {
                assert_eq!(exprs.len(), 1);
                assert_eq!(exprs[0].0, "return");
            }
            _ => panic!("Expected Projection plan"),
        }
    }

    #[test]
    fn test_lazy_frame_filter() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let predicate = Expr::col("close").gt(Expr::lit(2.0));

        let lf = LazyFrame::scan(source).filter(predicate);

        match &*lf.logical_plan {
            LogicalPlan::Filter { predicate, .. } => {
                assert!(matches!(predicate, Expr::BinaryExpr { .. }));
            }
            _ => panic!("Expected Filter plan"),
        }
    }

    #[test]
    fn test_lazy_frame_with_window() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let window_spec = WindowSpec {
            kind: WindowKind::Rolling,
            size: Some(3),
            min_periods: 1,
        };

        let lf = LazyFrame::scan(source)
            .with_window(Expr::col("close"), window_spec, "ma_close");

        match &*lf.logical_plan {
            LogicalPlan::Window { output_name, .. } => {
                assert_eq!(output_name, "ma_close");
            }
            _ => panic!("Expected Window plan"),
        }
    }

    #[test]
    fn test_lazy_frame_with_stateful() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let expr = StatefulExpr::CumSum(Expr::col("close"));

        let lf = LazyFrame::scan(source)
            .with_stateful(expr, "cumsum_close");

        match &*lf.logical_plan {
            LogicalPlan::Stateful { output_name, .. } => {
                assert_eq!(output_name, "cumsum_close");
            }
            _ => panic!("Expected Stateful plan"),
        }
    }

    #[test]
    fn test_lazy_frame_cache() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let lf = LazyFrame::scan(source).cache(Some("my_cache"));

        match &*lf.logical_plan {
            LogicalPlan::Cache { key, .. } => {
                assert_eq!(key, "my_cache");
            }
            _ => panic!("Expected Cache plan"),
        }
    }

    #[test]
    fn test_lazy_frame_join() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let lf1 = LazyFrame::scan(source.clone());
        let lf2 = LazyFrame::scan(source);

        let lf = lf1.join(lf2, vec!["symbol".to_string()], JoinType::Inner);

        match &*lf.logical_plan {
            LogicalPlan::Join { on, how, .. } => {
                assert_eq!(on.len(), 1);
                assert!(matches!(how, JoinType::Inner));
            }
            _ => panic!("Expected Join plan"),
        }
    }

    #[test]
    fn test_lazy_frame_chaining() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let lf = LazyFrame::scan(source)
            .filter(Expr::col("close"))
            .with_columns(vec![
                ("ma5", Expr::col("close")),
            ])
            .with_stateful(StatefulExpr::CumSum(Expr::col("ma5")), "cumsum_ma5")
            .cache(Some("chained_cache"));

        // Check the top-level plan is Cache
        match &*lf.logical_plan {
            LogicalPlan::Cache { .. } => {}
            _ => panic!("Expected Cache plan at top"),
        }

        // Verify optimization level is preserved
        let lf2 = lf.optimization_level(OptimizationLevel::Aggressive);
        assert!(matches!(lf2.optimization_level, OptimizationLevel::Aggressive));
    }

    // ========================================================================
    // Optimizer Tests
    // ========================================================================

    #[test]
    fn test_optimizer_none_level() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);
        let plan = LogicalPlan::Scan {
            source,
            projection: None,
            selection: None,
        };

        let optimizer = LazyOptimizer::new(OptimizationLevel::None);
        let optimized = optimizer.optimize(plan);

        assert!(matches!(optimized, LogicalPlan::Scan { .. }));
    }

    #[test]
    fn test_optimizer_basic_level() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);
        let plan = LogicalPlan::Scan {
            source,
            projection: None,
            selection: None,
        };

        let optimizer = LazyOptimizer::new(OptimizationLevel::Basic);
        let optimized = optimizer.optimize(plan);

        assert!(matches!(optimized, LogicalPlan::Scan { .. }));
    }

    #[test]
    fn test_optimizer_default_level() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);
        let plan = LogicalPlan::Scan {
            source,
            projection: None,
            selection: None,
        };

        let optimizer = LazyOptimizer::new(OptimizationLevel::Default);
        let optimized = optimizer.optimize(plan);

        assert!(matches!(optimized, LogicalPlan::Scan { .. }));
    }

    #[test]
    fn test_optimizer_aggressive_level() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);
        let plan = LogicalPlan::Scan {
            source,
            projection: None,
            selection: None,
        };

        let optimizer = LazyOptimizer::new(OptimizationLevel::Aggressive);
        let optimized = optimizer.optimize(plan);

        assert!(matches!(optimized, LogicalPlan::Scan { .. }));
    }

    #[test]
    fn test_predicate_pushdown() {
        // Create a plan with filter above projection
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let predicate = Expr::col("close").gt(Expr::lit(2.0));

        // Build: Filter -> Projection -> Scan
        let plan = LogicalPlan::Filter {
            input: Arc::new(LogicalPlan::Projection {
                input: Arc::new(LogicalPlan::Scan {
                    source,
                    projection: None,
                    selection: None,
                }),
                exprs: vec![("close".to_string(), Expr::col("close"))],
            }),
            predicate,
        };

        let optimizer = LazyOptimizer::new(OptimizationLevel::Default);
        let optimized = optimizer.optimize(plan);

        // The optimizer pushes the filter below projection
        // Original: Filter -> Projection -> Scan
        // After optimization: Projection -> Filter -> Scan
        match &optimized {
            LogicalPlan::Projection { input, .. } => {
                match &**input {
                    LogicalPlan::Filter { .. } => {}
                    other => panic!("Expected filter inside projection, got {:?}", other),
                }
            }
            other => panic!("Expected projection at top, got {:?}", other),
        }
    }

    #[test]
    fn test_extract_column_references() {
        let optimizer = LazyOptimizer::new(OptimizationLevel::Default);

        // Test simple column
        let expr = Expr::col("close");
        let cols = optimizer.extract_column_references(&expr);
        assert!(cols.contains("close"));
        assert_eq!(cols.len(), 1);

        // Test binary expression
        let expr = Expr::col("close") + Expr::col("volume");
        let cols = optimizer.extract_column_references(&expr);
        assert!(cols.contains("close"));
        assert!(cols.contains("volume"));
        assert_eq!(cols.len(), 2);

        // Test literal (no columns)
        let expr = Expr::lit(1.0);
        let cols = optimizer.extract_column_references(&expr);
        assert!(cols.is_empty());
    }

    // ========================================================================
    // Executor Tests
    // ========================================================================

    #[test]
    fn test_executor_scan() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let plan = LogicalPlan::Scan {
            source,
            projection: None,
            selection: None,
        };

        let mut executor = LazyExecutor::new();
        let result = executor.execute(&plan).unwrap();

        assert!(result.contains_key("close"));
        assert!(result.contains_key("volume"));
        assert_eq!(result["close"].dim(), (3, 4));
    }

    #[test]
    fn test_executor_scan_with_projection() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let plan = LogicalPlan::Scan {
            source,
            projection: Some(vec!["close".to_string()]),
            selection: None,
        };

        let mut executor = LazyExecutor::new();
        let result = executor.execute(&plan).unwrap();

        assert!(result.contains_key("close"));
        assert!(!result.contains_key("volume"));
    }

    #[test]
    fn test_executor_projection() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let plan = LogicalPlan::Projection {
            input: Arc::new(LogicalPlan::Scan {
                source,
                projection: None,
                selection: None,
            }),
            exprs: vec![
                ("double_close".to_string(), Expr::col("close") * Expr::lit(2.0)),
            ],
        };

        let mut executor = LazyExecutor::new();
        let result = executor.execute(&plan).unwrap();

        assert!(result.contains_key("close"));
        assert!(result.contains_key("double_close"));
    }

    #[test]
    fn test_executor_filter() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let predicate = Expr::col("close").gt(Expr::lit(2.0));

        let plan = LogicalPlan::Filter {
            input: Arc::new(LogicalPlan::Scan {
                source,
                projection: None,
                selection: None,
            }),
            predicate,
        };

        let mut executor = LazyExecutor::new();
        let result = executor.execute(&plan).unwrap();

        assert!(result.contains_key("close"));
    }

    #[test]
    fn test_executor_window() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let plan = LogicalPlan::Window {
            input: Arc::new(LogicalPlan::Scan {
                source,
                projection: None,
                selection: None,
            }),
            expr: Expr::col("close"),
            window_spec: WindowSpec {
                kind: WindowKind::Rolling,
                size: Some(2),
                min_periods: 1,
            },
            output_name: "ma_close".to_string(),
        };

        let mut executor = LazyExecutor::new();
        let result = executor.execute(&plan).unwrap();

        assert!(result.contains_key("close"));
        assert!(result.contains_key("ma_close"));
    }

    #[test]
    fn test_executor_stateful_cumsum() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let plan = LogicalPlan::Stateful {
            input: Arc::new(LogicalPlan::Scan {
                source,
                projection: None,
                selection: None,
            }),
            expr: StatefulExpr::CumSum(Expr::col("close")),
            output_name: "cumsum_close".to_string(),
        };

        let mut executor = LazyExecutor::new();
        let result = executor.execute(&plan).unwrap();

        assert!(result.contains_key("close"));
        assert!(result.contains_key("cumsum_close"));
    }

    #[test]
    fn test_executor_cache() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let plan = LogicalPlan::Cache {
            input: Arc::new(LogicalPlan::Scan {
                source,
                projection: None,
                selection: None,
            }),
            key: "test_cache".to_string(),
        };

        let mut executor = LazyExecutor::new();

        // First execution should populate cache
        let result1 = executor.execute(&plan).unwrap();
        assert!(result1.contains_key("close"));

        // Second execution should use cache
        let result2 = executor.execute(&plan).unwrap();
        assert!(result2.contains_key("close"));
    }

    #[test]
    fn test_executor_join_inner() {
        let mut data1 = HashMap::new();
        data1.insert(
            "close".to_string(),
            Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );

        let mut data2 = HashMap::new();
        data2.insert(
            "volume".to_string(),
            Array2::from_shape_vec((3, 2), vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0]).unwrap(),
        );

        let plan = LogicalPlan::Join {
            left: Arc::new(LogicalPlan::Scan {
                source: DataSource::NumpyArrays(data1),
                projection: None,
                selection: None,
            }),
            right: Arc::new(LogicalPlan::Scan {
                source: DataSource::NumpyArrays(data2),
                projection: None,
                selection: None,
            }),
            on: vec!["symbol".to_string()],
            how: JoinType::Inner,
        };

        let mut executor = LazyExecutor::new();
        let result = executor.execute(&plan).unwrap();

        assert!(result.contains_key("close"));
        assert!(result.contains_key("volume"));
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_empty_data() {
        let data = HashMap::new();
        let source = DataSource::NumpyArrays(data);

        let plan = LogicalPlan::Scan {
            source,
            projection: None,
            selection: None,
        };

        let mut executor = LazyExecutor::new();
        let result = executor.execute(&plan).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn test_lazy_frame_explain() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let lf = LazyFrame::scan(source);

        let unoptimized = lf.explain(false);
        assert!(unoptimized.contains("Scan"));

        let optimized = lf.explain(true);
        assert!(optimized.contains("Scan"));
    }

    #[test]
    fn test_lazy_frame_optimization_level() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let lf = LazyFrame::scan(source)
            .optimization_level(OptimizationLevel::None);

        assert!(matches!(lf.optimization_level, OptimizationLevel::None));

        let lf2 = lf.optimization_level(OptimizationLevel::Aggressive);
        assert!(matches!(lf2.optimization_level, OptimizationLevel::Aggressive));
    }

    #[test]
    fn test_lazy_frame_collect() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let lf = LazyFrame::scan(source);
        let result = lf.collect().unwrap();

        assert!(result.contains_key("close"));
        assert!(result.contains_key("volume"));
    }

    #[test]
    fn test_lazy_frame_collect_factor() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let lf = LazyFrame::scan(source);
        let result = lf.collect_factor("close").unwrap();

        assert_eq!(result.dim(), (3, 4));
    }

    #[test]
    fn test_lazy_frame_collect_factor_not_found() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let lf = LazyFrame::scan(source);
        let result = lf.collect_factor("nonexistent");

        assert!(result.is_err());
    }

    #[test]
    fn test_rolling_window_helper() {
        let spec = rolling_window(5, Some(2));

        assert!(matches!(spec.kind, WindowKind::Rolling));
        assert_eq!(spec.size, Some(5));
        assert_eq!(spec.min_periods, 2);
    }

    #[test]
    fn test_expanding_window_helper() {
        let spec = expanding_window(Some(3));

        assert!(matches!(spec.kind, WindowKind::Expanding));
        assert!(spec.size.is_none());
        assert_eq!(spec.min_periods, 3);
    }

    #[test]
    fn test_cumsum_helper() {
        let expr = cumsum(Expr::col("close"));
        assert!(matches!(expr, StatefulExpr::CumSum(_)));
    }

    #[test]
    fn test_cumprod_helper() {
        let expr = cumprod(Expr::col("close"));
        assert!(matches!(expr, StatefulExpr::CumProd(_)));
    }

    #[test]
    fn test_ema_helper() {
        let expr = ema(Expr::col("close"), 10);
        match &expr {
            StatefulExpr::Ema(_, alpha) => {
                // alpha = 2 / (span + 1) = 2 / 11
                let expected = 2.0 / 11.0;
                assert!((alpha - expected).abs() < 1e-10);
            }
            _ => panic!("Expected Ema"),
        }
    }

    // ========================================================================
    // LazyFrameBuilder Tests
    // ========================================================================

    #[test]
    fn test_lazy_frame_builder_scan() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let builder = LazyFrameBuilder::scan(source);
        let lf = builder.build().unwrap();

        assert!(matches!(&*lf.logical_plan, LogicalPlan::Scan { .. }));
    }

    #[test]
    fn test_lazy_frame_builder_chaining() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let result = LazyFrameBuilder::scan(source)
            .with_columns(vec![("ma".to_string(), Expr::col("close"))])
            .filter(Expr::col("close"))
            .cache(Some("builder_cache"))
            .optimization_level(OptimizationLevel::Default)
            .collect()
            .unwrap();

        assert!(result.contains_key("close"));
        assert!(result.contains_key("ma"));
    }

    #[test]
    fn test_lazy_frame_builder_collect_factor() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let result = LazyFrameBuilder::scan(source)
            .collect_factor("close")
            .unwrap();

        assert_eq!(result.dim(), (3, 4));
    }

    #[test]
    fn test_lazy_frame_builder_no_frame() {
        let builder = LazyFrameBuilder { lazy_frame: None };
        let result = builder.collect();

        assert!(result.is_err());
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_pipeline() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        // Full pipeline: scan -> filter -> projection -> window -> stateful -> cache
        let result = LazyFrameBuilder::scan(source)
            .filter(Expr::col("close").gt(Expr::lit(1.0)))
            .with_columns(vec![
                ("return".to_string(), Expr::col("close") * Expr::lit(2.0)),
            ])
            .with_window(
                Expr::col("close"),
                WindowSpec {
                    kind: WindowKind::Rolling,
                    size: Some(2),
                    min_periods: 1,
                },
                "ma_close",
            )
            .with_stateful(
                StatefulExpr::CumSum(Expr::col("return")),
                "cumsum_return",
            )
            .cache(Some("full_pipeline"))
            .optimization_level(OptimizationLevel::Default)
            .collect()
            .unwrap();

        assert!(result.contains_key("close"));
        assert!(result.contains_key("volume"));
        assert!(result.contains_key("return"));
        assert!(result.contains_key("ma_close"));
        assert!(result.contains_key("cumsum_return"));
    }

    #[test]
    fn test_multiple_projections() {
        let data = create_test_data();
        let source = DataSource::NumpyArrays(data);

        let result = LazyFrameBuilder::scan(source)
            .with_columns(vec![
                ("double".to_string(), Expr::col("close") * Expr::lit(2.0)),
                ("triple".to_string(), Expr::col("close") * Expr::lit(3.0)),
            ])
            .collect()
            .unwrap();

        assert!(result.contains_key("double"));
        assert!(result.contains_key("triple"));
    }
}
