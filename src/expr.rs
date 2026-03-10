//! Expression system inspired by Polars' expression API
//!
//! This module provides a strongly-typed expression tree that can represent
//! complex computations and be optimized/evaluated efficiently.

use std::fmt;
use std::sync::Arc;

/// Data type supported by the expression system
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataType {
    Boolean,
    Integer,
    Float,
    String,
    // TODO: Add more types as needed
}

/// Dimension types for financial expressions (for GP factor mining)
/// Used to prevent generating invalid expressions like unnormalized price/volume
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dimension {
    /// Price dimension (close, open, high, low)
    Price,
    /// Return dimension (daily returns, log returns)
    Return,
    /// Volume dimension
    Volume,
    /// Amount dimension (trading amount)
    Amount,
    /// Ratio dimension (normalized/dimensionless)
    Ratio,
    /// No specific dimension
    Dimensionless,
    /// Unknown dimension (needs inference)
    Unknown,
}

impl Dimension {
    /// Check if two dimensions are compatible for binary operations
    pub fn is_compatible_with(&self, other: &Dimension) -> bool {
        match (self, other) {
            // Same dimensions are always compatible
            (d1, d2) if d1 == d2 => true,
            // Return / Return = Ratio (e.g., return / return)
            (Dimension::Return, Dimension::Return) => true,
            // Volume / Volume = Ratio
            (Dimension::Volume, Dimension::Volume) => true,
            // Price / Price = Ratio
            (Dimension::Price, Dimension::Price) => true,
            // Return / Volume = Ratio
            (Dimension::Return, Dimension::Volume) => true,
            (Dimension::Volume, Dimension::Return) => true,
            // Any / Ratio = Any (multiplying by ratio preserves dimension)
            (_, Dimension::Ratio) => true,
            (Dimension::Ratio, _) => true,
            // Dimensionless can combine with anything
            (Dimension::Dimensionless, _) => true,
            (_, Dimension::Dimensionless) => true,
            // Unknown is compatible with anything (needs inference)
            (Dimension::Unknown, _) => true,
            (_, Dimension::Unknown) => true,
            // Otherwise incompatible
            _ => false,
        }
    }

    /// Get resulting dimension from binary operation
    pub fn binary_result(&self, op: BinaryOp, other: &Dimension) -> Dimension {
        match op {
            BinaryOp::Add | BinaryOp::Subtract => {
                // Addition/subtraction: dimensions must match
                if self.is_compatible_with(other) {
                    if *self == Dimension::Dimensionless || *other == Dimension::Dimensionless {
                        return Dimension::Dimensionless;
                    }
                    if *self == *other {
                        return self.clone();
                    }
                    // Mixed: return dimensionless (e.g., price - price)
                    Dimension::Dimensionless
                } else {
                    Dimension::Unknown
                }
            }
            BinaryOp::Multiply | BinaryOp::Divide => {
                // Multiplication/division: dimensions combine
                match (self, other) {
                    (Dimension::Return, Dimension::Return) => Dimension::Ratio,
                    (Dimension::Volume, Dimension::Volume) => Dimension::Ratio,
                    (Dimension::Price, Dimension::Price) => Dimension::Ratio,
                    (Dimension::Return, Dimension::Volume) => Dimension::Ratio,
                    (Dimension::Volume, Dimension::Return) => Dimension::Ratio,
                    (d, Dimension::Ratio) | (Dimension::Ratio, d) => d.clone(),
                    (Dimension::Dimensionless, d) | (d, Dimension::Dimensionless) => d.clone(),
                    (Dimension::Unknown, _) | (_, Dimension::Unknown) => Dimension::Unknown,
                    _ => Dimension::Ratio, // Most multiplications result in ratio
                }
            }
            BinaryOp::Modulo => Dimension::Dimensionless, // Modulo always dimensionless
            BinaryOp::Equal
            | BinaryOp::NotEqual
            | BinaryOp::GreaterThan
            | BinaryOp::GreaterThanOrEqual
            | BinaryOp::LessThan
            | BinaryOp::LessThanOrEqual
            | BinaryOp::And
            | BinaryOp::Or => Dimension::Dimensionless, // Comparison/logical result
        }
    }

    /// Check if this dimension is valid for a factor output
    pub fn is_valid_factor_dimension(&self) -> bool {
        matches!(
            self,
            Dimension::Ratio | Dimension::Dimensionless | Dimension::Return
        )
    }
}

/// Wrapper for expressions with dimension information
#[derive(Clone)]
pub struct TypedExpr {
    pub expr: Expr,
    pub dimension: Dimension,
}

impl TypedExpr {
    /// Create a new typed expression
    pub fn new(expr: Expr, dimension: Dimension) -> Self {
        Self { expr, dimension }
    }

    /// Infer dimension from expression
    pub fn infer_dimension(expr: &Expr) -> Dimension {
        match expr {
            Expr::Literal(lit) => match lit {
                Literal::Float(_) | Literal::Integer(_) => Dimension::Dimensionless,
                Literal::String(_) => Dimension::Unknown,
                Literal::Boolean(_) => Dimension::Dimensionless,
                Literal::Null => Dimension::Unknown,
            },
            Expr::Column(name) => {
                // Infer from column name
                let lower = name.to_lowercase();
                if lower.contains("return") || lower.contains("ret") {
                    Dimension::Return
                } else if lower.contains("volume") || lower.contains("vol") {
                    Dimension::Volume
                } else if lower.contains("amount") {
                    Dimension::Amount
                } else if lower.contains("price")
                    || lower.contains("close")
                    || lower.contains("open")
                    || lower.contains("high")
                    || lower.contains("low")
                {
                    Dimension::Price
                } else {
                    Dimension::Unknown
                }
            }
            Expr::BinaryExpr { left, op, right } => {
                let dim_left = Self::infer_dimension(left);
                let dim_right = Self::infer_dimension(right);
                dim_left.binary_result(*op, &dim_right)
            }
            Expr::UnaryExpr { expr, .. } => Self::infer_dimension(expr),
            Expr::FunctionCall { name, .. } => {
                // Most functions output ratio/dimensionless (rank, scale, etc.)
                let lower = name.to_lowercase();
                if lower.contains("rank") || lower.contains("scale") {
                    Dimension::Ratio
                } else if lower.contains("delay") || lower.contains("diff") {
                    // These preserve dimension
                    Dimension::Unknown
                } else {
                    Dimension::Dimensionless
                }
            }
            Expr::Aggregate { .. } => Dimension::Dimensionless,
            Expr::Conditional { .. } => Dimension::Unknown,
            Expr::Cast { .. } => Dimension::Unknown,
        }
    }
}

/// An expression node in the computation graph
#[derive(Clone, PartialEq)]
pub enum Expr {
    /// A literal value
    Literal(Literal),
    /// Reference to a column by name
    Column(String),
    /// Binary operation between two expressions
    BinaryExpr {
        left: Arc<Expr>,
        op: BinaryOp,
        right: Arc<Expr>,
    },
    /// Unary operation on an expression
    UnaryExpr { op: UnaryOp, expr: Arc<Expr> },
    /// Function call with arguments
    FunctionCall { name: String, args: Vec<Expr> },
    /// Aggregate expression (e.g., sum, mean)
    Aggregate {
        op: AggregateOp,
        expr: Arc<Expr>,
        distinct: bool,
    },
    /// Conditional expression (if-then-else)
    Conditional {
        condition: Arc<Expr>,
        then_expr: Arc<Expr>,
        else_expr: Arc<Expr>,
    },
    /// Cast expression to a different type
    Cast {
        expr: Arc<Expr>,
        data_type: DataType,
    },
}

/// Literal value
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Null,
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    And,
    Or,
    // TODO: Add more operators
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Negate,
    Not,
    Abs,
    Sqrt,
    Log,
    Exp,
    // TODO: Add more operators
}

/// Aggregate operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateOp {
    Sum,
    Mean,
    Min,
    Max,
    Count,
    StdDev,
    Variance,
    // TODO: Add more aggregates
}

impl Expr {
    /// Create a literal boolean expression
    pub fn lit_bool(value: bool) -> Self {
        Expr::Literal(Literal::Boolean(value))
    }

    /// Create a literal integer expression
    pub fn lit_int(value: i64) -> Self {
        Expr::Literal(Literal::Integer(value))
    }

    /// Create a literal float expression
    pub fn lit_float(value: f64) -> Self {
        Expr::Literal(Literal::Float(value))
    }

    /// Create a literal string expression
    pub fn lit_string(value: impl Into<String>) -> Self {
        Expr::Literal(Literal::String(value.into()))
    }

    /// Create a column reference expression
    pub fn col(name: impl Into<String>) -> Self {
        Expr::Column(name.into())
    }

    /// Create a binary expression
    pub fn binary(self, op: BinaryOp, right: Expr) -> Self {
        Expr::BinaryExpr {
            left: Arc::new(self),
            op,
            right: Arc::new(right),
        }
    }

    /// Create an addition expression
    pub fn add(self, right: Expr) -> Self {
        self.binary(BinaryOp::Add, right)
    }

    /// Create a subtraction expression
    pub fn sub(self, right: Expr) -> Self {
        self.binary(BinaryOp::Subtract, right)
    }

    /// Create a multiplication expression
    pub fn mul(self, right: Expr) -> Self {
        self.binary(BinaryOp::Multiply, right)
    }

    /// Create a division expression
    pub fn div(self, right: Expr) -> Self {
        self.binary(BinaryOp::Divide, right)
    }

    /// Create an equality comparison expression
    pub fn eq(self, right: Expr) -> Self {
        self.binary(BinaryOp::Equal, right)
    }

    /// Create a greater-than comparison expression
    pub fn gt(self, right: Expr) -> Self {
        self.binary(BinaryOp::GreaterThan, right)
    }

    /// Create a logical AND expression
    pub fn and(self, right: Expr) -> Self {
        self.binary(BinaryOp::And, right)
    }

    /// Create a logical OR expression
    pub fn or(self, right: Expr) -> Self {
        self.binary(BinaryOp::Or, right)
    }

    /// Create a unary expression
    pub fn unary(self, op: UnaryOp) -> Self {
        Expr::UnaryExpr {
            op,
            expr: Arc::new(self),
        }
    }

    /// Create a negation expression
    pub fn neg(self) -> Self {
        self.unary(UnaryOp::Negate)
    }

    /// Create a NOT expression
    pub fn not(self) -> Self {
        self.unary(UnaryOp::Not)
    }

    /// Create an absolute value expression
    pub fn abs(self) -> Self {
        self.unary(UnaryOp::Abs)
    }

    /// Create a square root expression
    pub fn sqrt(self) -> Self {
        self.unary(UnaryOp::Sqrt)
    }

    /// Create a function call expression
    pub fn function(name: impl Into<String>, args: Vec<Expr>) -> Self {
        Expr::FunctionCall {
            name: name.into(),
            args,
        }
    }

    /// Create an aggregate expression
    pub fn aggregate(self, op: AggregateOp, distinct: bool) -> Self {
        Expr::Aggregate {
            op,
            expr: Arc::new(self),
            distinct,
        }
    }

    /// Create a sum aggregate expression
    pub fn sum(self) -> Self {
        self.aggregate(AggregateOp::Sum, false)
    }

    /// Create a mean aggregate expression
    pub fn mean(self) -> Self {
        self.aggregate(AggregateOp::Mean, false)
    }

    /// Create a conditional expression
    pub fn conditional(condition: Expr, then_expr: Expr, else_expr: Expr) -> Self {
        Expr::Conditional {
            condition: Arc::new(condition),
            then_expr: Arc::new(then_expr),
            else_expr: Arc::new(else_expr),
        }
    }

    /// Create a cast expression
    pub fn cast(self, data_type: DataType) -> Self {
        Expr::Cast {
            expr: Arc::new(self),
            data_type,
        }
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Literal(lit) => write!(f, "{:?}", lit),
            Expr::Column(name) => write!(f, "#{}", name),
            Expr::BinaryExpr { left, op, right } => {
                write!(f, "({:?} {:?} {:?})", left, op, right)
            }
            Expr::UnaryExpr { op, expr } => write!(f, "({:?} {:?})", op, expr),
            Expr::FunctionCall { name, args } => {
                write!(f, "{}({:?})", name, args)
            }
            Expr::Aggregate { op, expr, distinct } => {
                if *distinct {
                    write!(f, "{:?}_distinct({:?})", op, expr)
                } else {
                    write!(f, "{:?}({:?})", op, expr)
                }
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => write!(
                f,
                "if {:?} then {:?} else {:?}",
                condition, then_expr, else_expr
            ),
            Expr::Cast { expr, data_type } => write!(f, "cast({:?} as {:?})", expr, data_type),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// Operator overloading for easier expression building
impl std::ops::Add for Expr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.binary(BinaryOp::Add, rhs)
    }
}

impl std::ops::Sub for Expr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.binary(BinaryOp::Subtract, rhs)
    }
}

impl std::ops::Mul for Expr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.binary(BinaryOp::Multiply, rhs)
    }
}

impl std::ops::Div for Expr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.binary(BinaryOp::Divide, rhs)
    }
}

impl Expr {
    /// Add an alias to this expression
    pub fn alias(self, _name: impl Into<String>) -> Self {
        // For now, just return the expression itself
        // In a real implementation, this would create an Alias expression variant
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_expression() {
        let expr = Expr::col("x")
            .add(Expr::lit_int(5))
            .mul(Expr::col("y"))
            .gt(Expr::lit_float(10.0));

        println!("Expression: {}", expr);
        assert!(matches!(expr, Expr::BinaryExpr { .. }));
    }

    #[test]
    fn test_operator_overloading() {
        let expr = Expr::col("a") + Expr::col("b") * Expr::lit_int(2);
        println!("Expression: {}", expr);
    }

    #[test]
    fn test_literal_creation() {
        let bool_lit = Expr::lit_bool(true);
        let int_lit = Expr::lit_int(42);
        let float_lit = Expr::lit_float(3.14);
        let string_lit = Expr::lit_string("hello");

        assert!(matches!(bool_lit, Expr::Literal(Literal::Boolean(true))));
        assert!(matches!(int_lit, Expr::Literal(Literal::Integer(42))));
        assert!(matches!(float_lit, Expr::Literal(Literal::Float(x)) if (x - 3.14).abs() < 1e-10));
        assert!(matches!(string_lit, Expr::Literal(Literal::String(ref s)) if s == "hello"));
    }

    #[test]
    fn test_column_reference() {
        let col_expr = Expr::col("price");
        assert!(matches!(col_expr, Expr::Column(ref name) if name == "price"));
    }

    #[test]
    fn test_unary_operators() {
        let expr = Expr::lit_int(5).neg();
        assert!(matches!(
            expr,
            Expr::UnaryExpr {
                op: UnaryOp::Negate,
                ..
            }
        ));

        let expr2 = Expr::lit_bool(false).not();
        assert!(matches!(
            expr2,
            Expr::UnaryExpr {
                op: UnaryOp::Not,
                ..
            }
        ));

        let expr3 = Expr::lit_float(4.0).sqrt();
        assert!(matches!(
            expr3,
            Expr::UnaryExpr {
                op: UnaryOp::Sqrt,
                ..
            }
        ));
    }

    #[test]
    fn test_function_call() {
        let expr = Expr::function("pow", vec![Expr::lit_float(2.0), Expr::lit_float(3.0)]);
        assert!(matches!(expr, Expr::FunctionCall { name, .. } if name == "pow"));
    }

    #[test]
    fn test_aggregate_expression() {
        let expr = Expr::col("value").sum();
        assert!(matches!(
            expr,
            Expr::Aggregate {
                op: AggregateOp::Sum,
                ..
            }
        ));

        let expr2 = Expr::col("value").mean();
        assert!(matches!(
            expr2,
            Expr::Aggregate {
                op: AggregateOp::Mean,
                ..
            }
        ));
    }

    #[test]
    fn test_conditional_expression() {
        let expr = Expr::conditional(Expr::lit_bool(true), Expr::lit_int(1), Expr::lit_int(0));
        assert!(matches!(expr, Expr::Conditional { .. }));
    }

    #[test]
    fn test_cast_expression() {
        let expr = Expr::lit_int(5).cast(DataType::Float);
        assert!(matches!(
            expr,
            Expr::Cast {
                data_type: DataType::Float,
                ..
            }
        ));
    }

    #[test]
    fn test_debug_format() {
        let expr = Expr::col("x") + Expr::lit_int(5);
        let debug_str = format!("{:?}", expr);
        // Just ensure it doesn't panic and produces some output
        assert!(!debug_str.is_empty());
    }
}
