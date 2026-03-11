//! Optimizer for lazy evaluation plans
//!
//! Provides optimization passes for logical plans including predicate pushdown,
//! projection pruning, and common subexpression elimination.

use std::sync::Arc;

use crate::expr::{Expr, Expr as E};

use super::plan::*;

// ============================================================================
// Lazy Optimizer
// ============================================================================

/// Optimizer for logical plans
pub struct LazyOptimizer {
    level: OptimizationLevel,
}

impl LazyOptimizer {
    pub fn new(level: OptimizationLevel) -> Self {
        Self { level }
    }

    pub fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        match self.level {
            OptimizationLevel::None => plan,
            OptimizationLevel::Basic => self.optimize_basic(plan),
            OptimizationLevel::Default => self.optimize_default(plan),
            OptimizationLevel::Aggressive => self.optimize_aggressive(plan),
        }
    }

    fn optimize_basic(&self, plan: LogicalPlan) -> LogicalPlan {
        // Basic optimizations: constant folding and simple CSE
        let plan = self.constant_folding(plan);
        let plan = self.simple_cse(plan);
        plan
    }

    fn optimize_default(&self, plan: LogicalPlan) -> LogicalPlan {
        // Default optimizations: basic + predicate pushdown + projection pruning
        let plan = self.optimize_basic(plan);
        let plan = self.predicate_pushdown(plan);
        let plan = self.projection_pruning(plan);
        plan
    }

    fn optimize_aggressive(&self, plan: LogicalPlan) -> LogicalPlan {
        // Aggressive optimizations: default + join reordering + advanced CSE
        let plan = self.optimize_default(plan);
        let plan = self.join_reordering(plan);
        let plan = self.advanced_cse(plan);
        plan
    }

    /// Constant folding: evaluate constant expressions at optimization time
    fn constant_folding(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement constant folding
        // For now, just pass through
        plan
    }

    /// Simple common subexpression elimination
    fn simple_cse(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement simple CSE
        // For now, just pass through
        plan
    }

    /// Predicate pushdown: move filters closer to data sources
    fn predicate_pushdown(&self, plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Filter { input, predicate } => {
                let optimized_input = self.predicate_pushdown(input.as_ref().clone());

                // Try to push predicate through projection
                if let LogicalPlan::Projection {
                    input: proj_input,
                    exprs,
                } = &optimized_input
                {
                    // Check if predicate only uses columns that are preserved in projection
                    let used_columns = self.extract_column_references(&predicate);
                    let projected_columns: std::collections::HashSet<_> =
                        exprs.iter().map(|(name, _)| name.clone()).collect();

                    // Also include columns from input that are passed through unchanged
                    // For now, we assume all columns used must be in projection
                    if used_columns.is_subset(&projected_columns) {
                        // Push filter below projection
                        return LogicalPlan::Projection {
                            input: Arc::new(LogicalPlan::Filter {
                                input: proj_input.clone(),
                                predicate,
                            }),
                            exprs: exprs.clone(),
                        };
                    }
                }

                LogicalPlan::Filter {
                    input: Arc::new(optimized_input),
                    predicate,
                }
            }
            LogicalPlan::Projection { input, exprs } => LogicalPlan::Projection {
                input: Arc::new(self.predicate_pushdown(input.as_ref().clone())),
                exprs,
            },
            LogicalPlan::Window {
                input,
                expr,
                window_spec,
                output_name,
            } => LogicalPlan::Window {
                input: Arc::new(self.predicate_pushdown(input.as_ref().clone())),
                expr,
                window_spec,
                output_name,
            },
            LogicalPlan::Stateful {
                input,
                expr,
                output_name,
            } => LogicalPlan::Stateful {
                input: Arc::new(self.predicate_pushdown(input.as_ref().clone())),
                expr,
                output_name,
            },
            LogicalPlan::Cache { input, key } => LogicalPlan::Cache {
                input: Arc::new(self.predicate_pushdown(input.as_ref().clone())),
                key,
            },
            LogicalPlan::Join {
                left,
                right,
                on,
                how,
            } => LogicalPlan::Join {
                left: Arc::new(self.predicate_pushdown(left.as_ref().clone())),
                right: Arc::new(self.predicate_pushdown(right.as_ref().clone())),
                on,
                how,
            },
            LogicalPlan::Scan { .. } => plan,
        }
    }

    /// Extract column references from an expression
    pub fn extract_column_references(&self, expr: &Expr) -> std::collections::HashSet<String> {
        let mut columns = std::collections::HashSet::new();

        match expr {
            E::Column(name) => {
                columns.insert(name.clone());
            }
            E::BinaryExpr { left, right, .. } => {
                columns.extend(self.extract_column_references(left));
                columns.extend(self.extract_column_references(right));
            }
            E::UnaryExpr { expr, .. } => {
                columns.extend(self.extract_column_references(expr));
            }
            E::FunctionCall { args, .. } => {
                for arg in args {
                    columns.extend(self.extract_column_references(arg));
                }
            }
            E::Aggregate { expr, .. } => {
                columns.extend(self.extract_column_references(expr));
            }
            E::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                columns.extend(self.extract_column_references(condition));
                columns.extend(self.extract_column_references(then_expr));
                columns.extend(self.extract_column_references(else_expr));
            }
            E::Cast { expr, .. } => {
                columns.extend(self.extract_column_references(expr));
            }
            E::Literal(_) => {}
        }

        columns
    }

    /// Projection pruning: remove unused columns from projections
    fn projection_pruning(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement projection pruning
        // For now, just pass through
        plan
    }

    /// Join reordering
    fn join_reordering(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement join reordering
        // For now, just pass through
        plan
    }

    /// Advanced common subexpression elimination
    fn advanced_cse(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement advanced CSE
        // For now, just pass through
        plan
    }
}
