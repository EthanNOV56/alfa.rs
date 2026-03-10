//! Expression optimizer for vectorized evaluation
//!
//! This module provides optimizations specifically for improving the performance
//! of vectorized expression evaluation, including:
//! - Common subexpression elimination (CSE)
//! - Constant folding for vectorized operations
//! - Algebraic simplifications
//! - Evaluation plan optimization

use crate::expr::{BinaryOp, Expr, Literal, UnaryOp};
use std::sync::Arc;

// ============================================================================
// Expression Optimizer
// ============================================================================

/// Optimizer for vectorized expression evaluation
pub struct ExpressionOptimizer {
    /// Whether to apply common subexpression elimination
    enable_cse: bool,
    /// Whether to apply constant folding
    enable_constant_folding: bool,
    /// Whether to apply algebraic simplifications
    enable_algebraic_simplification: bool,
}

impl ExpressionOptimizer {
    /// Create a new optimizer with all optimizations enabled
    pub fn new() -> Self {
        Self {
            enable_cse: true,
            enable_constant_folding: true,
            enable_algebraic_simplification: true,
        }
    }

    /// Create an optimizer with specific optimizations enabled
    pub fn with_options(cse: bool, constant_folding: bool, algebraic: bool) -> Self {
        Self {
            enable_cse: cse,
            enable_constant_folding: constant_folding,
            enable_algebraic_simplification: algebraic,
        }
    }

    /// Optimize an expression for vectorized evaluation
    pub fn optimize(&self, expr: Expr) -> Expr {
        let mut expr = expr;

        // Apply optimizations in a specific order
        if self.enable_constant_folding {
            expr = self.fold_constants(expr);
        }

        if self.enable_algebraic_simplification {
            expr = self.simplify_algebraic(expr);
        }

        if self.enable_cse {
            expr = self.eliminate_common_subexpressions(expr);
        }

        expr
    }

    // ==================== Constant Folding ====================

    /// Fold constant expressions (evaluate them at optimization time)
    fn fold_constants(&self, expr: Expr) -> Expr {
        match expr {
            // Literals and columns stay as they are
            Expr::Literal(_) | Expr::Column(_) => expr,

            // Binary expressions
            Expr::BinaryExpr { left, op, right } => {
                let left_folded = self.fold_constants(left.as_ref().clone());
                let right_folded = self.fold_constants(right.as_ref().clone());

                // Try to evaluate if both sides are literals
                if let (Expr::Literal(left_lit), Expr::Literal(right_lit)) =
                    (&left_folded, &right_folded)
                {
                    match (left_lit, op, right_lit) {
                        // Integer operations
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

                        // Float operations
                        (Literal::Float(l), BinaryOp::Add, Literal::Float(r)) => {
                            return Expr::Literal(Literal::Float(l + r));
                        }
                        (Literal::Float(l), BinaryOp::Subtract, Literal::Float(r)) => {
                            return Expr::Literal(Literal::Float(l - r));
                        }
                        (Literal::Float(l), BinaryOp::Multiply, Literal::Float(r)) => {
                            return Expr::Literal(Literal::Float(l * r));
                        }
                        (Literal::Float(l), BinaryOp::Divide, Literal::Float(r)) if *r != 0.0 => {
                            return Expr::Literal(Literal::Float(l / r));
                        }

                        // Mixed integer/float operations
                        (Literal::Integer(l), op, Literal::Float(r)) => {
                            let l_f = *l as f64;
                            let result = match op {
                                BinaryOp::Add => l_f + r,
                                BinaryOp::Subtract => l_f - r,
                                BinaryOp::Multiply => l_f * r,
                                BinaryOp::Divide if *r != 0.0 => l_f / r,
                                _ => {
                                    return Expr::BinaryExpr {
                                        left: Arc::new(left_folded),
                                        op,
                                        right: Arc::new(right_folded),
                                    }
                                }
                            };
                            return Expr::Literal(Literal::Float(result));
                        }
                        (Literal::Float(l), op, Literal::Integer(r)) => {
                            let r_f = *r as f64;
                            let result = match op {
                                BinaryOp::Add => l + r_f,
                                BinaryOp::Subtract => l - r_f,
                                BinaryOp::Multiply => l * r_f,
                                BinaryOp::Divide if r_f != 0.0 => l / r_f,
                                _ => {
                                    return Expr::BinaryExpr {
                                        left: Arc::new(left_folded),
                                        op,
                                        right: Arc::new(right_folded),
                                    }
                                }
                            };
                            return Expr::Literal(Literal::Float(result));
                        }

                        _ => {} // Fall through
                    }
                }

                // Special case: x + 0 = x, 0 + x = x
                if self.enable_algebraic_simplification {
                    match (&left_folded, op, &right_folded) {
                        (Expr::Literal(Literal::Integer(0)), BinaryOp::Add, _) => {
                            return right_folded
                        }
                        (_, BinaryOp::Add, Expr::Literal(Literal::Integer(0))) => {
                            return left_folded
                        }
                        (Expr::Literal(Literal::Float(f)), BinaryOp::Add, _) if *f == 0.0 => {
                            return right_folded
                        }
                        (_, BinaryOp::Add, Expr::Literal(Literal::Float(f))) if *f == 0.0 => {
                            return left_folded
                        }

                        // x - 0 = x
                        (_, BinaryOp::Subtract, Expr::Literal(Literal::Integer(0))) => {
                            return left_folded
                        }
                        (_, BinaryOp::Subtract, Expr::Literal(Literal::Float(f))) if *f == 0.0 => {
                            return left_folded
                        }

                        // x * 1 = x, 1 * x = x
                        (Expr::Literal(Literal::Integer(1)), BinaryOp::Multiply, _) => {
                            return right_folded
                        }
                        (_, BinaryOp::Multiply, Expr::Literal(Literal::Integer(1))) => {
                            return left_folded
                        }
                        (Expr::Literal(Literal::Float(f)), BinaryOp::Multiply, _) if *f == 1.0 => {
                            return right_folded
                        }
                        (_, BinaryOp::Multiply, Expr::Literal(Literal::Float(f))) if *f == 1.0 => {
                            return left_folded
                        }

                        // x * 0 = 0, 0 * x = 0
                        (Expr::Literal(Literal::Integer(0)), BinaryOp::Multiply, _) => {
                            return Expr::Literal(Literal::Integer(0))
                        }
                        (_, BinaryOp::Multiply, Expr::Literal(Literal::Integer(0))) => {
                            return Expr::Literal(Literal::Integer(0))
                        }
                        (Expr::Literal(Literal::Float(f)), BinaryOp::Multiply, _) if *f == 0.0 => {
                            return Expr::Literal(Literal::Float(0.0))
                        }
                        (_, BinaryOp::Multiply, Expr::Literal(Literal::Float(f))) if *f == 0.0 => {
                            return Expr::Literal(Literal::Float(0.0))
                        }

                        // x / 1 = x
                        (_, BinaryOp::Divide, Expr::Literal(Literal::Integer(1))) => {
                            return left_folded
                        }
                        (_, BinaryOp::Divide, Expr::Literal(Literal::Float(f))) if *f == 1.0 => {
                            return left_folded
                        }

                        _ => {}
                    }
                }

                Expr::BinaryExpr {
                    left: Arc::new(left_folded),
                    op,
                    right: Arc::new(right_folded),
                }
            }

            // Unary expressions
            Expr::UnaryExpr { op, expr } => {
                let folded_expr = self.fold_constants(expr.as_ref().clone());

                // Try to evaluate if the inner expression is a literal
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
                        (Literal::Integer(v), UnaryOp::Abs) => {
                            return Expr::Literal(Literal::Integer(v.abs()));
                        }
                        (Literal::Float(v), UnaryOp::Abs) => {
                            return Expr::Literal(Literal::Float(v.abs()));
                        }
                        (Literal::Float(v), UnaryOp::Sqrt) if *v >= 0.0 => {
                            return Expr::Literal(Literal::Float(v.sqrt()));
                        }
                        (Literal::Float(v), UnaryOp::Log) if *v > 0.0 => {
                            return Expr::Literal(Literal::Float(v.ln()));
                        }
                        (Literal::Float(v), UnaryOp::Exp) => {
                            return Expr::Literal(Literal::Float(v.exp()));
                        }
                        _ => {} // Fall through
                    }
                }

                // Special case: -(-x) = x
                if self.enable_algebraic_simplification {
                    if let Expr::UnaryExpr {
                        op: UnaryOp::Negate,
                        expr: inner,
                    } = &folded_expr
                    {
                        if op == UnaryOp::Negate {
                            return inner.as_ref().clone();
                        }
                    }
                }

                Expr::UnaryExpr {
                    op,
                    expr: Arc::new(folded_expr),
                }
            }

            // Function calls
            Expr::FunctionCall { name, args } => {
                let folded_args = args
                    .into_iter()
                    .map(|arg| self.fold_constants(arg))
                    .collect();
                Expr::FunctionCall {
                    name,
                    args: folded_args,
                }
            }

            // Other expression types (preserve structure)
            Expr::Aggregate { op, expr, distinct } => {
                let folded_expr = self.fold_constants(expr.as_ref().clone());
                Expr::Aggregate {
                    op,
                    expr: Arc::new(folded_expr),
                    distinct,
                }
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let folded_cond = self.fold_constants(condition.as_ref().clone());
                let folded_then = self.fold_constants(then_expr.as_ref().clone());
                let folded_else = self.fold_constants(else_expr.as_ref().clone());
                Expr::Conditional {
                    condition: Arc::new(folded_cond),
                    then_expr: Arc::new(folded_then),
                    else_expr: Arc::new(folded_else),
                }
            }
            Expr::Cast { expr, data_type } => {
                let folded_expr = self.fold_constants(expr.as_ref().clone());
                Expr::Cast {
                    expr: Arc::new(folded_expr),
                    data_type,
                }
            }
        }
    }

    // ==================== Algebraic Simplification ====================

    /// Apply algebraic simplifications
    fn simplify_algebraic(&self, expr: Expr) -> Expr {
        match expr {
            // Binary expressions with algebraic simplifications
            Expr::BinaryExpr { left, op, right } => {
                let left_simplified = self.simplify_algebraic(left.as_ref().clone());
                let right_simplified = self.simplify_algebraic(right.as_ref().clone());

                // Apply algebraic rules
                match (&left_simplified, op, &right_simplified) {
                    // x + x = 2 * x
                    (left_expr, BinaryOp::Add, right_expr) if left_expr == right_expr => {
                        return Expr::BinaryExpr {
                            left: Arc::new(left_simplified),
                            op: BinaryOp::Multiply,
                            right: Arc::new(Expr::Literal(Literal::Integer(2))),
                        };
                    }

                    // x - x = 0
                    (left_expr, BinaryOp::Subtract, right_expr) if left_expr == right_expr => {
                        return Expr::Literal(Literal::Integer(0));
                    }

                    // (a + b) + c = a + (b + c)  (associativity - we don't reorder, but could in future)
                    // For now, just return the simplified expression
                    _ => {}
                }

                Expr::BinaryExpr {
                    left: Arc::new(left_simplified),
                    op,
                    right: Arc::new(right_simplified),
                }
            }

            // Recursively simplify other expression types
            Expr::UnaryExpr { op, expr } => {
                let simplified_expr = self.simplify_algebraic(expr.as_ref().clone());
                Expr::UnaryExpr {
                    op,
                    expr: Arc::new(simplified_expr),
                }
            }
            Expr::FunctionCall { name, args } => {
                let simplified_args = args
                    .into_iter()
                    .map(|arg| self.simplify_algebraic(arg))
                    .collect();
                Expr::FunctionCall {
                    name,
                    args: simplified_args,
                }
            }
            Expr::Aggregate { op, expr, distinct } => {
                let simplified_expr = self.simplify_algebraic(expr.as_ref().clone());
                Expr::Aggregate {
                    op,
                    expr: Arc::new(simplified_expr),
                    distinct,
                }
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let simplified_cond = self.simplify_algebraic(condition.as_ref().clone());
                let simplified_then = self.simplify_algebraic(then_expr.as_ref().clone());
                let simplified_else = self.simplify_algebraic(else_expr.as_ref().clone());
                Expr::Conditional {
                    condition: Arc::new(simplified_cond),
                    then_expr: Arc::new(simplified_then),
                    else_expr: Arc::new(simplified_else),
                }
            }
            Expr::Cast { expr, data_type } => {
                let simplified_expr = self.simplify_algebraic(expr.as_ref().clone());
                Expr::Cast {
                    expr: Arc::new(simplified_expr),
                    data_type,
                }
            }

            // Literals and columns don't need algebraic simplification
            _ => expr,
        }
    }

    // ==================== Common Subexpression Elimination ====================

    /// Eliminate common subexpressions
    fn eliminate_common_subexpressions(&self, expr: Expr) -> Expr {
        // TODO: Implement proper CSE
        // For now, return the expression unchanged
        expr
    }
}

// ============================================================================
// Evaluation Plan Optimizer
// ============================================================================

/// Optimized evaluation plan for vectorized expressions
pub struct EvaluationPlan {
    /// Root expression to evaluate
    pub root: Expr,
    /// Pre-computed subexpressions (common subexpressions)
    pub precomputed: Vec<(String, Expr)>,
}

impl EvaluationPlan {
    /// Create an evaluation plan from an expression
    pub fn from_expr(expr: Expr, optimizer: &ExpressionOptimizer) -> Self {
        let optimized = optimizer.optimize(expr);

        // TODO: Extract common subexpressions and create let-bindings
        // For now, just wrap the optimized expression

        EvaluationPlan {
            root: optimized,
            precomputed: Vec::new(),
        }
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Optimize an expression using default optimizations
pub fn optimize_expression(expr: Expr) -> Expr {
    let optimizer = ExpressionOptimizer::new();
    optimizer.optimize(expr)
}

/// Create an optimized evaluation plan
pub fn create_evaluation_plan(expr: Expr) -> EvaluationPlan {
    let optimizer = ExpressionOptimizer::new();
    EvaluationPlan::from_expr(expr, &optimizer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;

    #[test]
    fn test_constant_folding() {
        let optimizer = ExpressionOptimizer::new();

        // Test integer arithmetic
        let expr = Expr::lit_int(5).add(Expr::lit_int(3));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Integer(8))));

        // Test float arithmetic
        let expr = Expr::lit_float(2.5).mul(Expr::lit_float(4.0));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Float(f)) if f == 10.0));

        // Test mixed integer/float
        let expr = Expr::lit_int(5).add(Expr::lit_float(2.5));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Float(f)) if f == 7.5));
    }

    #[test]
    fn test_algebraic_simplification() {
        let optimizer = ExpressionOptimizer::new();

        // Test x + 0 = x
        let expr = Expr::col("price").add(Expr::lit_int(0));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Column(name) if name == "price"));

        // Test 0 + x = x
        let expr = Expr::lit_int(0).add(Expr::col("volume"));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Column(name) if name == "volume"));

        // Test x * 1 = x
        let expr = Expr::col("close").mul(Expr::lit_float(1.0));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Column(name) if name == "close"));

        // Test x * 0 = 0
        let expr = Expr::col("value").mul(Expr::lit_int(0));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Integer(0))));
    }

    #[test]
    fn test_double_negation() {
        let optimizer = ExpressionOptimizer::new();

        // Test -(-x) = x
        let expr = Expr::col("price").neg().neg();
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Column(name) if name == "price"));
    }

    #[test]
    fn test_complex_optimization() {
        let optimizer = ExpressionOptimizer::new();

        // Test (x + 0) * 1 + (5 * 0) = x
        let expr = Expr::col("price")
            .add(Expr::lit_int(0))
            .mul(Expr::lit_int(1))
            .add(Expr::lit_int(5).mul(Expr::lit_int(0)));

        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Column(name) if name == "price"));
    }

    #[test]
    fn test_preserve_non_trivial_expressions() {
        let optimizer = ExpressionOptimizer::new();

        // Test that non-trivial expressions are preserved
        let expr = Expr::col("close").add(Expr::col("open"));
        let optimized = optimizer.optimize(expr.clone());

        // Should still be a binary expression
        assert!(matches!(optimized, Expr::BinaryExpr { .. }));

        // Test that x + x is simplified to 2 * x
        let expr = Expr::col("value").add(Expr::col("value"));
        let optimized = optimizer.optimize(expr);

        match optimized {
            Expr::BinaryExpr {
                left,
                op: BinaryOp::Multiply,
                right,
            } => {
                assert!(matches!(left.as_ref(), Expr::Column(name) if name == "value"));
                assert!(matches!(right.as_ref(), Expr::Literal(Literal::Integer(2))));
            }
            _ => panic!("Expected 2 * value, got {:?}", optimized),
        }
    }
}
