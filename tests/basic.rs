use alpha_expr::expr::Expr;

#[test]
fn smoke() {
    let e = Expr::col("close");
    assert!(matches!(e, Expr::Column(_)));
}