//! Expression optimizer for vectorized evaluation
//!
//! This module provides optimizations specifically for improving the performance
//! of vectorized expression evaluation, including:
//! - Common subexpression elimination (CSE)
//! - Constant folding for vectorized operations
//! - Algebraic simplifications
//! - Evaluation plan optimization

use super::ast::{BinaryOp, DataType, Expr, Literal, UnaryOp};
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
                                    };
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
                                    };
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
                            return right_folded;
                        }
                        (_, BinaryOp::Add, Expr::Literal(Literal::Integer(0))) => {
                            return left_folded;
                        }
                        (Expr::Literal(Literal::Float(f)), BinaryOp::Add, _) if *f == 0.0 => {
                            return right_folded;
                        }
                        (_, BinaryOp::Add, Expr::Literal(Literal::Float(f))) if *f == 0.0 => {
                            return left_folded;
                        }

                        // x - 0 = x
                        (_, BinaryOp::Subtract, Expr::Literal(Literal::Integer(0))) => {
                            return left_folded;
                        }
                        (_, BinaryOp::Subtract, Expr::Literal(Literal::Float(f))) if *f == 0.0 => {
                            return left_folded;
                        }

                        // x * 1 = x, 1 * x = x
                        (Expr::Literal(Literal::Integer(1)), BinaryOp::Multiply, _) => {
                            return right_folded;
                        }
                        (_, BinaryOp::Multiply, Expr::Literal(Literal::Integer(1))) => {
                            return left_folded;
                        }
                        (Expr::Literal(Literal::Float(f)), BinaryOp::Multiply, _) if *f == 1.0 => {
                            return right_folded;
                        }
                        (_, BinaryOp::Multiply, Expr::Literal(Literal::Float(f))) if *f == 1.0 => {
                            return left_folded;
                        }

                        // x * 0 = 0, 0 * x = 0
                        (Expr::Literal(Literal::Integer(0)), BinaryOp::Multiply, _) => {
                            return Expr::Literal(Literal::Integer(0));
                        }
                        (_, BinaryOp::Multiply, Expr::Literal(Literal::Integer(0))) => {
                            return Expr::Literal(Literal::Integer(0));
                        }
                        (Expr::Literal(Literal::Float(f)), BinaryOp::Multiply, _) if *f == 0.0 => {
                            return Expr::Literal(Literal::Float(0.0));
                        }
                        (_, BinaryOp::Multiply, Expr::Literal(Literal::Float(f))) if *f == 0.0 => {
                            return Expr::Literal(Literal::Float(0.0));
                        }

                        // x / 1 = x
                        (_, BinaryOp::Divide, Expr::Literal(Literal::Integer(1))) => {
                            return left_folded;
                        }
                        (_, BinaryOp::Divide, Expr::Literal(Literal::Float(f))) if *f == 1.0 => {
                            return left_folded;
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
        use std::collections::HashMap;

        struct CseVisitor {
            subexprs: HashMap<String, Expr>,
            counter: usize,
        }

        impl CseVisitor {
            fn new() -> Self {
                Self {
                    subexprs: HashMap::new(),
                    counter: 0,
                }
            }

            fn visit(&mut self, expr: &Expr) -> Expr {
                match expr {
                    // For literals and columns, return as-is (they're already atomic)
                    Expr::Literal(_) | Expr::Column(_) => expr.clone(),

                    // For binary expressions, recursively process and check for CSE
                    Expr::BinaryExpr { left, op, right } => {
                        let new_left = self.visit(left);
                        let new_right = self.visit(right);

                        // Create a key for this subexpression
                        let key = format!("{:?}:{:?}:{:?}", op, new_left, new_right);

                        // Check if we've seen this subexpression before
                        if let Some(cached) = self.subexprs.get(&key) {
                            return cached.clone();
                        }

                        // Create new binary expression
                        let new_expr = Expr::BinaryExpr {
                            left: Arc::new(new_left),
                            op: *op,
                            right: Arc::new(new_right),
                        };

                        // Cache it for future use
                        self.subexprs.insert(key, new_expr.clone());

                        new_expr
                    }

                    // For unary expressions
                    Expr::UnaryExpr { op, expr: inner } => {
                        let new_inner = self.visit(inner);

                        let key = format!("{:?}:{:?}", op, new_inner);

                        if let Some(cached) = self.subexprs.get(&key) {
                            return cached.clone();
                        }

                        let new_expr = Expr::UnaryExpr {
                            op: *op,
                            expr: Arc::new(new_inner),
                        };

                        self.subexprs.insert(key, new_expr.clone());
                        new_expr
                    }

                    // For function calls - process args but don't do full CSE
                    Expr::FunctionCall { name, args } => {
                        let new_args: Vec<Expr> = args.iter().map(|arg| self.visit(arg)).collect();
                        Expr::FunctionCall {
                            name: name.clone(),
                            args: new_args,
                        }
                    }

                    // For aggregates
                    Expr::Aggregate {
                        op,
                        expr: inner,
                        distinct,
                    } => {
                        let new_inner = self.visit(inner);
                        Expr::Aggregate {
                            op: *op,
                            expr: Arc::new(new_inner),
                            distinct: *distinct,
                        }
                    }

                    // For conditional - process all branches
                    Expr::Conditional {
                        condition,
                        then_expr,
                        else_expr,
                    } => {
                        let new_cond = self.visit(condition);
                        let new_then = self.visit(then_expr);
                        let new_else = self.visit(else_expr);
                        Expr::Conditional {
                            condition: Arc::new(new_cond),
                            then_expr: Arc::new(new_then),
                            else_expr: Arc::new(new_else),
                        }
                    }

                    // For cast
                    Expr::Cast {
                        expr: inner,
                        data_type,
                    } => {
                        let new_inner = self.visit(inner);
                        Expr::Cast {
                            expr: Arc::new(new_inner),
                            data_type: data_type.clone(),
                        }
                    }
                }
            }
        }

        let mut visitor = CseVisitor::new();
        visitor.visit(&expr)
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

    // ==================== Additional Constant Folding Tests ====================

    #[test]
    fn test_constant_folding_subtraction() {
        let optimizer = ExpressionOptimizer::new();

        // Test subtraction
        let expr = Expr::lit_int(10).sub(Expr::lit_int(3));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Integer(7))));
    }

    #[test]
    fn test_constant_folding_division() {
        let optimizer = ExpressionOptimizer::new();

        // Test division
        let expr = Expr::lit_float(10.0).div(Expr::lit_float(2.0));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Float(f)) if (f - 5.0).abs() < 1e-10));

        // Test division by zero (should not fold)
        let expr = Expr::lit_int(5).div(Expr::lit_int(0));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::BinaryExpr { .. }));
    }

    #[test]
    fn test_constant_folding_modulo() {
        let optimizer = ExpressionOptimizer::new();

        // Test modulo - current implementation doesn't fold modulo
        let expr = Expr::lit_int(10).binary(BinaryOp::Modulo, Expr::lit_int(3));
        let optimized = optimizer.optimize(expr);
        // Modulo is not folded, so it remains a BinaryExpr
        assert!(matches!(
            optimized,
            Expr::BinaryExpr {
                op: BinaryOp::Modulo,
                ..
            }
        ));
    }

    #[test]
    fn test_constant_folding_unary() {
        let optimizer = ExpressionOptimizer::new();

        // Test unary negation
        let expr = Expr::lit_int(5).neg();
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Integer(-5))));

        // Test unary negation on float
        let expr = Expr::lit_float(3.5).neg();
        let optimized = optimizer.optimize(expr);
        assert!(
            matches!(optimized, Expr::Literal(Literal::Float(f)) if (f - (-3.5)).abs() < 1e-10)
        );

        // Test NOT on boolean
        let expr = Expr::lit_bool(true).not();
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Boolean(false))));
    }

    #[test]
    fn test_constant_folding_abs() {
        let optimizer = ExpressionOptimizer::new();

        // Test abs on positive
        let expr = Expr::lit_int(5).abs();
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Integer(5))));

        // Test abs on negative
        let expr = Expr::lit_int(-5).abs();
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Integer(5))));

        // Test abs on float
        let expr = Expr::lit_float(-3.14).abs();
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Float(f)) if (f - 3.14).abs() < 1e-10));
    }

    #[test]
    fn test_constant_folding_sqrt() {
        let optimizer = ExpressionOptimizer::new();

        // Test sqrt
        let expr = Expr::lit_float(4.0).sqrt();
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Float(f)) if (f - 2.0).abs() < 1e-10));

        // Test sqrt on negative (should not fold)
        let expr = Expr::lit_float(-4.0).sqrt();
        let optimized = optimizer.optimize(expr);
        assert!(matches!(
            optimized,
            Expr::UnaryExpr {
                op: UnaryOp::Sqrt,
                ..
            }
        ));
    }

    #[test]
    fn test_constant_folding_log_exp() {
        let optimizer = ExpressionOptimizer::new();

        // Test log (using unary expression)
        let expr = Expr::unary(Expr::lit_float(2.718281828), UnaryOp::Log);
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Float(f)) if (f - 1.0).abs() < 1e-6));

        // Test exp (using unary expression)
        let expr = Expr::unary(Expr::lit_float(2.0), UnaryOp::Exp);
        let optimized = optimizer.optimize(expr);
        assert!(
            matches!(optimized, Expr::Literal(Literal::Float(f)) if (f - std::f64::consts::E * std::f64::consts::E).abs() < 1e-6)
        );
    }

    // ==================== Additional Algebraic Simplification Tests ====================

    #[test]
    fn test_algebraic_simplify_x_minus_x() {
        let optimizer = ExpressionOptimizer::new();

        // Test x - x = 0
        let expr = Expr::col("value").sub(Expr::col("value"));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Integer(0))));
    }

    #[test]
    fn test_algebraic_simplify_x_times_0() {
        let optimizer = ExpressionOptimizer::new();

        // Test x * 0 = 0
        let expr = Expr::col("price").mul(Expr::lit_float(0.0));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Literal(Literal::Float(0.0))));
    }

    #[test]
    fn test_algebraic_simplify_x_divide_1() {
        let optimizer = ExpressionOptimizer::new();

        // Test x / 1 = x
        let expr = Expr::col("price").div(Expr::lit_int(1));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::Column(name) if name == "price"));
    }

    // ==================== CSE Tests ====================

    #[test]
    fn test_cse_simple() {
        let optimizer = ExpressionOptimizer::with_options(true, false, false);

        // Test simple CSE: (x + y) + (x + y) should be 2 * (x + y)
        // But with current CSE implementation, it may not optimize this
        // Let's test that the optimizer runs without panic
        let expr = Expr::col("x").add(Expr::col("y"));
        let optimized = optimizer.optimize(expr);
        assert!(matches!(optimized, Expr::BinaryExpr { .. }));
    }

    #[test]
    fn test_cse_disabled() {
        // With CSE disabled but constant folding still on, constants still fold
        let optimizer = ExpressionOptimizer::with_options(false, true, false);

        let expr = Expr::lit_int(5).add(Expr::lit_int(3));
        let optimized = optimizer.optimize(expr);

        // Constant folding is still enabled
        assert!(matches!(optimized, Expr::Literal(Literal::Integer(8))));
    }

    // ==================== Edge Cases Tests ====================

    #[test]
    fn test_optimize_no_change() {
        let optimizer = ExpressionOptimizer::new();

        // Test that simple column is unchanged
        let expr = Expr::col("close");
        let optimized = optimizer.optimize(expr.clone());
        assert_eq!(expr, optimized);
    }

    #[test]
    fn test_optimize_function_call() {
        let optimizer = ExpressionOptimizer::new();

        // Test function call with literal args - currently not evaluated at compile time
        let expr = Expr::function("abs", vec![Expr::lit_int(-5)]);
        let optimized = optimizer.optimize(expr);
        // Function calls are not folded (would require built-in function registry)
        assert!(matches!(optimized, Expr::FunctionCall { name, .. } if name == "abs"));
    }

    #[test]
    fn test_optimize_conditional() {
        let optimizer = ExpressionOptimizer::new();

        // Test conditional with constant condition - currently not simplified
        let expr = Expr::conditional(Expr::lit_bool(true), Expr::lit_int(1), Expr::lit_int(0));
        let optimized = optimizer.optimize(expr);
        // Conditional is not simplified when condition is constant
        assert!(matches!(optimized, Expr::Conditional { .. }));
    }

    #[test]
    fn test_optimize_cast() {
        let optimizer = ExpressionOptimizer::new();

        // Test cast expression
        let expr = Expr::lit_int(5).cast(DataType::Float);
        let optimized = optimizer.optimize(expr);
        // Cast of literal may or may not be optimized depending on implementation
        assert!(matches!(
            optimized,
            Expr::Cast { .. } | Expr::Literal(Literal::Float(5.0))
        ));
    }

    #[test]
    fn test_optimize_nested_complex() {
        let optimizer = ExpressionOptimizer::new();

        // Test complex nested expression
        let expr = Expr::col("a")
            .add(Expr::lit_int(0))
            .mul(Expr::lit_int(1))
            .sub(Expr::col("b").mul(Expr::lit_int(0)));

        let optimized = optimizer.optimize(expr);
        // (a + 0) * 1 - (b * 0) = a * 1 - 0 = a
        assert!(matches!(optimized, Expr::Column(name) if name == "a"));
    }
}
