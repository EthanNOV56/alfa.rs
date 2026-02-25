use alpha_expr::expr::{Expr, Literal};
use alpha_expr::logical_plan::LogicalPlan;
use alpha_expr::optimizer::{Optimizer, ConstantFolding};

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
        panic!("Optimized plan is not a literal 3: {:?}", optimized);
    }
}