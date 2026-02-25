//! Optimization framework for logical plans and expressions
//!
//! This module provides rule-based optimization for transforming logical plans
//! and expressions to more efficient forms.

use crate::logical_plan::LogicalPlan;
use crate::expr::{Expr, Literal, BinaryOp, UnaryOp};
use std::sync::Arc;

/// An optimization rule that can transform a logical plan
pub trait OptimizerRule {
    /// Apply the rule to a logical plan, returning a new optimized plan
    fn optimize(&self, plan: LogicalPlan) -> LogicalPlan;
    
    /// Get the name of this optimization rule
    fn name(&self) -> &str;
}

/// A collection of optimization rules applied in sequence
pub struct Optimizer {
    rules: Vec<Box<dyn OptimizerRule>>,
}

impl Optimizer {
    /// Create a new optimizer with no rules
    pub fn new() -> Self {
        Self { rules: vec![] }
    }
    
    /// Add an optimization rule
    pub fn add_rule<R: OptimizerRule + 'static>(mut self, rule: R) -> Self {
        self.rules.push(Box::new(rule));
        self
    }
    
    /// Optimize a logical plan by applying all rules in sequence
    pub fn optimize(&self, mut plan: LogicalPlan) -> LogicalPlan {
        for rule in &self.rules {
            plan = rule.optimize(plan);
        }
        plan
    }
    
    /// Create a default optimizer with common optimization rules
    pub fn default() -> Self {
        Self::new()
            .add_rule(ConstantFolding::new())
            .add_rule(PredicatePushdown::new())
            .add_rule(ProjectionPushdown::new())
            .add_rule(EliminateCommonSubexpressions::new())
    }
}

// ----------------------------------------------------------------------------
// Common optimization rules
// ----------------------------------------------------------------------------

/// Constant folding: evaluate constant expressions at compile time
pub struct ConstantFolding {
    name: String,
}

impl ConstantFolding {
    pub fn new() -> Self {
        Self {
            name: "constant_folding".to_string(),
        }
    }
    
    fn fold_expr(&self, expr: Expr) -> Expr {
        match expr {
            // Literals stay as they are
            Expr::Literal(_) | Expr::Column(_) => expr,
            
            // Binary expressions: fold left and right, then try to evaluate if both are literals
            Expr::BinaryExpr { left, op, right } => {
                let left_folded = self.fold_expr(left.as_ref().clone());
                let right_folded = self.fold_expr(right.as_ref().clone());
                
                // If both sides are literals, try to evaluate
                if let (Expr::Literal(left_lit), Expr::Literal(right_lit)) = (&left_folded, &right_folded) {
                    // For now, only handle integer arithmetic
                    match (left_lit, op, right_lit) {
                        (Literal::Integer(l), BinaryOp::Add, Literal::Integer(r)) => {
                            return Expr::Literal(Literal::Integer(l + r));
                        }
                        (Literal::Integer(l), BinaryOp::Subtract, Literal::Integer(r)) => {
                            return Expr::Literal(Literal::Integer(l - r));
                        }
                        (Literal::Integer(l), BinaryOp::Multiply, Literal::Integer(r)) => {
                            return Expr::Literal(Literal::Integer(l * r));
                        }
                        (Literal::Integer(l), BinaryOp::Divide, Literal::Integer(r)) if *r != 0 => {
                            return Expr::Literal(Literal::Integer(l / r));
                        }
                        _ => {} // Fall through to return the folded binary expression
                    }
                }
                
                Expr::BinaryExpr {
                    left: Arc::new(left_folded),
                    op,
                    right: Arc::new(right_folded),
                }
            }
            
            // Unary expressions: fold the inner expression, then try to evaluate
            Expr::UnaryExpr { op, expr } => {
                let folded_expr = self.fold_expr(expr.as_ref().clone());
                if let Expr::Literal(lit) = &folded_expr {
                    match (lit, op) {
                        (Literal::Integer(v), UnaryOp::Negate) => {
                            return Expr::Literal(Literal::Integer(-v));
                        }
                        (Literal::Float(v), UnaryOp::Negate) => {
                            return Expr::Literal(Literal::Float(-v));
                        }
                        (Literal::Boolean(v), UnaryOp::Not) => {
                            return Expr::Literal(Literal::Boolean(!v));
                        }
                        _ => {} // Fall through
                    }
                }
                
                Expr::UnaryExpr {
                    op,
                    expr: Arc::new(folded_expr),
                }
            }
            
            // For other expression types, recursively fold subexpressions
            Expr::FunctionCall { name, args } => {
                let folded_args = args.into_iter().map(|arg| self.fold_expr(arg)).collect();
                Expr::FunctionCall { name, args: folded_args }
            }
            Expr::Aggregate { op, expr, distinct } => {
                let folded_expr = self.fold_expr(expr.as_ref().clone());
                Expr::Aggregate { op, expr: Arc::new(folded_expr), distinct }
            }
            Expr::Conditional { condition, then_expr, else_expr } => {
                let folded_cond = self.fold_expr(condition.as_ref().clone());
                let folded_then = self.fold_expr(then_expr.as_ref().clone());
                let folded_else = self.fold_expr(else_expr.as_ref().clone());
                Expr::Conditional {
                    condition: Arc::new(folded_cond),
                    then_expr: Arc::new(folded_then),
                    else_expr: Arc::new(folded_else),
                }
            }
            Expr::Cast { expr, data_type } => {
                let folded_expr = self.fold_expr(expr.as_ref().clone());
                Expr::Cast { expr: Arc::new(folded_expr), data_type }
            }
        }
    }
}

impl OptimizerRule for ConstantFolding {
    fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        let mut transform_fn = |plan: LogicalPlan| {
            match plan {
                LogicalPlan::Filter { input, predicate } => {
                    let folded_predicate = self.fold_expr(predicate);
                    LogicalPlan::Filter { input, predicate: folded_predicate }
                }
                LogicalPlan::Projection { input, expr, schema } => {
                    let folded_expr = expr.into_iter().map(|e| self.fold_expr(e)).collect();
                    LogicalPlan::Projection { input, expr: folded_expr, schema }
                }
                LogicalPlan::Aggregate { input, group_expr, agg_expr } => {
                    let folded_group = group_expr.into_iter().map(|e| self.fold_expr(e)).collect();
                    let folded_agg = agg_expr.into_iter().map(|e| self.fold_expr(e)).collect();
                    LogicalPlan::Aggregate { input, group_expr: folded_group, agg_expr: folded_agg }
                }
                LogicalPlan::Expression(expr) => {
                    LogicalPlan::Expression(self.fold_expr(expr))
                }
                _ => plan,
            }
        };
        plan.transform(&mut transform_fn)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Predicate pushdown: push filters closer to data sources
pub struct PredicatePushdown {
    name: String,
}

impl PredicatePushdown {
    pub fn new() -> Self {
        Self {
            name: "predicate_pushdown".to_string(),
        }
    }
}

impl OptimizerRule for PredicatePushdown {
    fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement predicate pushdown
        // This would push filter conditions through projections and joins
        // when possible to reduce the amount of data processed
        plan
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Projection pushdown: eliminate unused columns early
pub struct ProjectionPushdown {
    name: String,
}

impl ProjectionPushdown {
    pub fn new() -> Self {
        Self {
            name: "projection_pushdown".to_string(),
        }
    }
}

impl OptimizerRule for ProjectionPushdown {
    fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement projection pushdown
        // This would track which columns are actually needed and
        // push projections down to eliminate unused columns early
        plan
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Eliminate common subexpressions: reuse computed expressions
pub struct EliminateCommonSubexpressions {
    name: String,
}

impl EliminateCommonSubexpressions {
    pub fn new() -> Self {
        Self {
            name: "eliminate_common_subexpressions".to_string(),
        }
    }
}

impl OptimizerRule for EliminateCommonSubexpressions {
    fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        // TODO: Implement common subexpression elimination
        // This would identify identical expression trees and compute them once
        plan
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

// ----------------------------------------------------------------------------
// Expression-specific optimizations
// ----------------------------------------------------------------------------

/// Optimize expressions directly (without a full logical plan)
pub fn optimize_expr(expr: Expr) -> Expr {
    // TODO: Implement expression-level optimizations
    expr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;
    
    #[test]
    fn test_optimizer_creation() {
        let optimizer = Optimizer::default();
        let plan = LogicalPlan::expression(Expr::lit_int(5));
        let optimized = optimizer.optimize(plan);
        assert!(matches!(optimized, LogicalPlan::Expression(_)));
    }

    #[test]
    fn test_constant_folding() {
        let optimizer = Optimizer::default();
        
        // Test that 1 + 2 gets folded to 3
        let expr = Expr::lit_int(1).add(Expr::lit_int(2));
        let plan = LogicalPlan::expression(expr);
        let optimized = optimizer.optimize(plan);
        
        // The optimized plan should be a literal 3
        if let LogicalPlan::Expression(Expr::Literal(Literal::Integer(3))) = optimized {
            // Success
        } else {
            panic!("Expected folded literal 3, got {:?}", optimized);
        }
        
        // Test that (5 * 2) - 3 gets folded to 7
        let expr = Expr::lit_int(5).mul(Expr::lit_int(2)).sub(Expr::lit_int(3));
        let plan = LogicalPlan::expression(expr);
        let optimized = optimizer.optimize(plan);
        
        if let LogicalPlan::Expression(Expr::Literal(Literal::Integer(7))) = optimized {
            // Success
        } else {
            panic!("Expected folded literal 7, got {:?}", optimized);
        }
        
        // Test that expressions with columns are not folded
        let expr = Expr::col("x").add(Expr::lit_int(1));
        let plan = LogicalPlan::expression(expr);
        let optimized = optimizer.optimize(plan);
        
        // Should still be a binary expression
        assert!(matches!(optimized, LogicalPlan::Expression(Expr::BinaryExpr { .. })));
    }
}