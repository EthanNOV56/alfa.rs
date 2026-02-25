//! Expression system inspired by Polars' expression API
//!
//! This module provides a strongly-typed expression tree that can represent
//! complex computations and be optimized/evaluated efficiently.

use std::fmt;
use std::sync::Arc;

/// Data type supported by the expression system
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Boolean,
    Integer,
    Float,
    String,
    // TODO: Add more types as needed
}

/// An expression node in the computation graph
#[derive(Clone)]
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
    UnaryExpr {
        op: UnaryOp,
        expr: Arc<Expr>,
    },
    /// Function call with arguments
    FunctionCall {
        name: String,
        args: Vec<Expr>,
    },
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            } => write!(f, "if {:?} then {:?} else {:?}", condition, then_expr, else_expr),
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
}