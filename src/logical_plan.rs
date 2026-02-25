//! Logical plan representing relational algebra operations
//!
//! This module provides a logical plan representation that can be optimized
//! and transformed before being converted to a physical execution plan.

use std::sync::Arc;
use crate::expr::Expr;

/// A logical plan node representing a relational algebra operation
#[derive(Clone)]
pub enum LogicalPlan {
    /// Scan a data source (e.g., table, CSV file)
    Scan {
        source_name: String,
        projection: Option<Vec<String>>,
        filters: Vec<Expr>,
    },
    /// Project specific columns (possibly with computed expressions)
    Projection {
        input: Arc<LogicalPlan>,
        expr: Vec<Expr>,
        schema: Vec<(String, crate::expr::DataType)>,
    },
    /// Filter rows based on a predicate
    Filter {
        input: Arc<LogicalPlan>,
        predicate: Expr,
    },
    /// Aggregate rows using grouping and aggregation expressions
    Aggregate {
        input: Arc<LogicalPlan>,
        group_expr: Vec<Expr>,
        agg_expr: Vec<Expr>,
    },
    /// Join two logical plans
    Join {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        join_type: JoinType,
        condition: Option<Expr>,
    },
    /// Sort rows by expressions
    Sort {
        input: Arc<LogicalPlan>,
        sort_expr: Vec<Expr>,
        descending: Vec<bool>,
    },
    /// Limit the number of rows
    Limit {
        input: Arc<LogicalPlan>,
        limit: usize,
        offset: usize,
    },
    /// Set operation (union, intersect, except)
    SetOperation {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        op: SetOp,
        all: bool,
    },
    /// Expression evaluation (for standalone expression trees)
    Expression(Expr),
}

/// Join types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

/// Set operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOp {
    Union,
    Intersect,
    Except,
}

impl LogicalPlan {
    /// Create a scan logical plan
    pub fn scan(source_name: impl Into<String>) -> Self {
        LogicalPlan::Scan {
            source_name: source_name.into(),
            projection: None,
            filters: vec![],
        }
    }

    /// Create a projection logical plan
    pub fn projection(input: LogicalPlan, expr: Vec<Expr>, schema: Vec<(String, crate::expr::DataType)>) -> Self {
        LogicalPlan::Projection {
            input: Arc::new(input),
            expr,
            schema,
        }
    }

    /// Create a filter logical plan
    pub fn filter(input: LogicalPlan, predicate: Expr) -> Self {
        LogicalPlan::Filter {
            input: Arc::new(input),
            predicate,
        }
    }

    /// Create an aggregate logical plan
    pub fn aggregate(input: LogicalPlan, group_expr: Vec<Expr>, agg_expr: Vec<Expr>) -> Self {
        LogicalPlan::Aggregate {
            input: Arc::new(input),
            group_expr,
            agg_expr,
        }
    }

    /// Create a join logical plan
    pub fn join(left: LogicalPlan, right: LogicalPlan, join_type: JoinType, condition: Option<Expr>) -> Self {
        LogicalPlan::Join {
            left: Arc::new(left),
            right: Arc::new(right),
            join_type,
            condition,
        }
    }

    /// Create a sort logical plan
    pub fn sort(input: LogicalPlan, sort_expr: Vec<Expr>, descending: Vec<bool>) -> Self {
        LogicalPlan::Sort {
            input: Arc::new(input),
            sort_expr,
            descending,
        }
    }

    /// Create a limit logical plan
    pub fn limit(input: LogicalPlan, limit: usize, offset: usize) -> Self {
        LogicalPlan::Limit {
            input: Arc::new(input),
            limit,
            offset,
        }
    }

    /// Create a set operation logical plan
    pub fn set_operation(left: LogicalPlan, right: LogicalPlan, op: SetOp, all: bool) -> Self {
        LogicalPlan::SetOperation {
            left: Arc::new(left),
            right: Arc::new(right),
            op,
            all,
        }
    }

    /// Create an expression logical plan
    pub fn expression(expr: Expr) -> Self {
        LogicalPlan::Expression(expr)
    }

    /// Get the children of this logical plan
    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::Scan { .. } => vec![],
            LogicalPlan::Projection { input, .. } => vec![input],
            LogicalPlan::Filter { input, .. } => vec![input],
            LogicalPlan::Aggregate { input, .. } => vec![input],
            LogicalPlan::Join { left, right, .. } => vec![left, right],
            LogicalPlan::Sort { input, .. } => vec![input],
            LogicalPlan::Limit { input, .. } => vec![input],
            LogicalPlan::SetOperation { left, right, .. } => vec![left, right],
            LogicalPlan::Expression(_) => vec![],
        }
    }

    /// Transform the logical plan using a visitor
    pub fn transform(self, f: &mut dyn FnMut(LogicalPlan) -> LogicalPlan) -> LogicalPlan {
        let transformed_children = self.children()
            .iter()
            .map(|child| (*child).clone().transform(f))
            .collect::<Vec<_>>();
        
        let mut plan = match self {
            LogicalPlan::Scan { source_name, projection, filters } => {
                LogicalPlan::Scan { source_name, projection, filters }
            }
            LogicalPlan::Projection { input: _, expr, schema } => {
                let input = transformed_children[0].clone();
                LogicalPlan::Projection {
                    input: Arc::new(input),
                    expr,
                    schema,
                }
            }
            LogicalPlan::Filter { input: _, predicate } => {
                let input = transformed_children[0].clone();
                LogicalPlan::Filter {
                    input: Arc::new(input),
                    predicate,
                }
            }
            LogicalPlan::Aggregate { input: _, group_expr, agg_expr } => {
                let input = transformed_children[0].clone();
                LogicalPlan::Aggregate {
                    input: Arc::new(input),
                    group_expr,
                    agg_expr,
                }
            }
            LogicalPlan::Join { left: _, right: _, join_type, condition } => {
                let left = transformed_children[0].clone();
                let right = transformed_children[1].clone();
                LogicalPlan::Join {
                    left: Arc::new(left),
                    right: Arc::new(right),
                    join_type,
                    condition,
                }
            }
            LogicalPlan::Sort { input: _, sort_expr, descending } => {
                let input = transformed_children[0].clone();
                LogicalPlan::Sort {
                    input: Arc::new(input),
                    sort_expr,
                    descending,
                }
            }
            LogicalPlan::Limit { input: _, limit, offset } => {
                let input = transformed_children[0].clone();
                LogicalPlan::Limit {
                    input: Arc::new(input),
                    limit,
                    offset,
                }
            }
            LogicalPlan::SetOperation { left: _, right: _, op, all } => {
                let left = transformed_children[0].clone();
                let right = transformed_children[1].clone();
                LogicalPlan::SetOperation {
                    left: Arc::new(left),
                    right: Arc::new(right),
                    op,
                    all,
                }
            }
            LogicalPlan::Expression(expr) => LogicalPlan::Expression(expr),
        };
        
        plan = f(plan);
        plan
    }
}

impl std::fmt::Debug for LogicalPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogicalPlan::Scan { source_name, projection, filters } => {
                write!(f, "Scan({}", source_name)?;
                if let Some(proj) = projection {
                    write!(f, ", projection={:?}", proj)?;
                }
                if !filters.is_empty() {
                    write!(f, ", filters={:?}", filters)?;
                }
                write!(f, ")")
            }
            LogicalPlan::Projection { expr, .. } => {
                write!(f, "Projection({:?})", expr)
            }
            LogicalPlan::Filter { predicate, .. } => {
                write!(f, "Filter({:?})", predicate)
            }
            LogicalPlan::Aggregate { group_expr, agg_expr, .. } => {
                write!(f, "Aggregate(group={:?}, agg={:?})", group_expr, agg_expr)
            }
            LogicalPlan::Join { join_type, condition, .. } => {
                write!(f, "Join({:?}, condition={:?})", join_type, condition)
            }
            LogicalPlan::Sort { sort_expr, descending, .. } => {
                write!(f, "Sort({:?}, descending={:?})", sort_expr, descending)
            }
            LogicalPlan::Limit { limit, offset, .. } => {
                write!(f, "Limit(limit={}, offset={})", limit, offset)
            }
            LogicalPlan::SetOperation { op, all, .. } => {
                write!(f, "{:?}(all={})", op, all)
            }
            LogicalPlan::Expression(expr) => {
                write!(f, "Expression({:?})", expr)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{Expr, DataType};

    #[test]
    fn test_scan_creation() {
        let scan = LogicalPlan::scan("table1");
        assert!(matches!(scan, LogicalPlan::Scan { source_name, .. } if source_name == "table1"));
    }

    #[test]
    fn test_filter_creation() {
        let scan = LogicalPlan::scan("table1");
        let filter = LogicalPlan::filter(scan, Expr::col("x").gt(Expr::lit_int(0)));
        assert!(matches!(filter, LogicalPlan::Filter { .. }));
    }

    #[test]
    fn test_projection_creation() {
        let scan = LogicalPlan::scan("table1");
        let projection = LogicalPlan::projection(
            scan,
            vec![Expr::col("x"), Expr::col("y")],
            vec![
                ("x".to_string(), DataType::Integer),
                ("y".to_string(), DataType::Integer),
            ],
        );
        assert!(matches!(projection, LogicalPlan::Projection { .. }));
    }

    #[test]
    fn test_aggregate_creation() {
        let scan = LogicalPlan::scan("table1");
        let aggregate = LogicalPlan::aggregate(
            scan,
            vec![Expr::col("group")],
            vec![Expr::col("value").sum()],
        );
        assert!(matches!(aggregate, LogicalPlan::Aggregate { .. }));
    }

    #[test]
    fn test_children_method() {
        let scan = LogicalPlan::scan("table1");
        assert_eq!(scan.children().len(), 0);
        
        let filter = LogicalPlan::filter(scan, Expr::lit_bool(true));
        assert_eq!(filter.children().len(), 1);
        
        let join = LogicalPlan::join(
            LogicalPlan::scan("left"),
            LogicalPlan::scan("right"),
            JoinType::Inner,
            Some(Expr::lit_bool(true)),
        );
        assert_eq!(join.children().len(), 2);
    }

    #[test]
    fn test_transform_method() {
        let scan = LogicalPlan::scan("table1");
        let filter = LogicalPlan::filter(scan, Expr::col("x").gt(Expr::lit_int(0)));
        
        // Transform that does nothing
        let mut transform_fn = |plan: LogicalPlan| plan;
        let transformed = filter.transform(&mut transform_fn);
        assert!(matches!(transformed, LogicalPlan::Filter { .. }));
    }
}