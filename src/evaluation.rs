//! Expression evaluation engine
//!
//! This module provides functionality to evaluate expressions against data.

use std::collections::HashMap;
use crate::expr::{Expr, Literal, BinaryOp, UnaryOp, AggregateOp, DataType};

/// Context for evaluating expressions
pub struct EvaluationContext {
    /// Column name to value mapping for the current row
    row_data: HashMap<String, Literal>,
    /// Aggregated data for window functions (optional)
    window_data: Option<HashMap<String, Vec<Literal>>>,
}

impl EvaluationContext {
    /// Create a new evaluation context with empty row data
    pub fn new() -> Self {
        Self {
            row_data: HashMap::new(),
            window_data: None,
        }
    }
    
    /// Create a new evaluation context with the given row data
    pub fn with_row_data(data: HashMap<String, Literal>) -> Self {
        Self {
            row_data: data,
            window_data: None,
        }
    }
    
    /// Set a column value in the current row
    pub fn set_column(&mut self, name: String, value: Literal) {
        self.row_data.insert(name, value);
    }
    
    /// Get a column value from the current row
    pub fn get_column(&self, name: &str) -> Option<&Literal> {
        self.row_data.get(name)
    }
    
    /// Evaluate an expression in this context
    pub fn evaluate(&self, expr: &Expr) -> Result<Literal, EvaluationError> {
        match expr {
            Expr::Literal(lit) => Ok(lit.clone()),
            Expr::Column(name) => {
                self.get_column(name)
                    .cloned()
                    .ok_or_else(|| EvaluationError::ColumnNotFound(name.clone()))
            }
            Expr::BinaryExpr { left, op, right } => {
                let left_val = self.evaluate(left)?;
                let right_val = self.evaluate(right)?;
                eval_binary_op(&left_val, *op, &right_val)
            }
            Expr::UnaryExpr { op, expr } => {
                let val = self.evaluate(expr)?;
                eval_unary_op(&val, *op)
            }
            Expr::FunctionCall { name, args } => {
                let arg_values = args.iter()
                    .map(|arg| self.evaluate(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                eval_function(name, &arg_values)
            }
            Expr::Aggregate { .. } => {
                // Aggregate expressions require window data
                Err(EvaluationError::UnsupportedOperation(
                    "Aggregate expressions require window context".to_string()
                ))
            }
            Expr::Conditional { condition, then_expr, else_expr } => {
                let cond_val = self.evaluate(condition)?;
                if let Literal::Boolean(true) = cond_val {
                    self.evaluate(then_expr)
                } else {
                    self.evaluate(else_expr)
                }
            }
            Expr::Cast { expr, data_type } => {
                let val = self.evaluate(expr)?;
                cast_value(&val, data_type)
            }
        }
    }
}

/// Evaluate a binary operation
fn eval_binary_op(left: &Literal, op: BinaryOp, right: &Literal) -> Result<Literal, EvaluationError> {
    match (left, op, right) {
        // Arithmetic operations
        (Literal::Integer(l), BinaryOp::Add, Literal::Integer(r)) => Ok(Literal::Integer(l + r)),
        (Literal::Float(l), BinaryOp::Add, Literal::Float(r)) => Ok(Literal::Float(l + r)),
        (Literal::Integer(l), BinaryOp::Subtract, Literal::Integer(r)) => Ok(Literal::Integer(l - r)),
        (Literal::Float(l), BinaryOp::Subtract, Literal::Float(r)) => Ok(Literal::Float(l - r)),
        (Literal::Integer(l), BinaryOp::Multiply, Literal::Integer(r)) => Ok(Literal::Integer(l * r)),
        (Literal::Float(l), BinaryOp::Multiply, Literal::Float(r)) => Ok(Literal::Float(l * r)),
        (Literal::Integer(l), BinaryOp::Divide, Literal::Integer(r)) => {
            if *r == 0 {
                Err(EvaluationError::DivisionByZero)
            } else {
                Ok(Literal::Integer(l / r))
            }
        }
        (Literal::Float(l), BinaryOp::Divide, Literal::Float(r)) => {
            if *r == 0.0 {
                Err(EvaluationError::DivisionByZero)
            } else {
                Ok(Literal::Float(l / r))
            }
        }
        (Literal::Integer(l), BinaryOp::Modulo, Literal::Integer(r)) => {
            if *r == 0 {
                Err(EvaluationError::DivisionByZero)
            } else {
                Ok(Literal::Integer(l % r))
            }
        }
        
        // Comparison operations
        (Literal::Integer(l), BinaryOp::Equal, Literal::Integer(r)) => Ok(Literal::Boolean(l == r)),
        (Literal::Float(l), BinaryOp::Equal, Literal::Float(r)) => Ok(Literal::Boolean(l == r)),
        (Literal::Boolean(l), BinaryOp::Equal, Literal::Boolean(r)) => Ok(Literal::Boolean(l == r)),
        (Literal::String(l), BinaryOp::Equal, Literal::String(r)) => Ok(Literal::Boolean(l == r)),
        
        (Literal::Integer(l), BinaryOp::NotEqual, Literal::Integer(r)) => Ok(Literal::Boolean(l != r)),
        (Literal::Float(l), BinaryOp::NotEqual, Literal::Float(r)) => Ok(Literal::Boolean(l != r)),
        (Literal::Boolean(l), BinaryOp::NotEqual, Literal::Boolean(r)) => Ok(Literal::Boolean(l != r)),
        (Literal::String(l), BinaryOp::NotEqual, Literal::String(r)) => Ok(Literal::Boolean(l != r)),
        
        (Literal::Integer(l), BinaryOp::GreaterThan, Literal::Integer(r)) => Ok(Literal::Boolean(l > r)),
        (Literal::Float(l), BinaryOp::GreaterThan, Literal::Float(r)) => Ok(Literal::Boolean(l > r)),
        
        (Literal::Integer(l), BinaryOp::GreaterThanOrEqual, Literal::Integer(r)) => Ok(Literal::Boolean(l >= r)),
        (Literal::Float(l), BinaryOp::GreaterThanOrEqual, Literal::Float(r)) => Ok(Literal::Boolean(l >= r)),
        
        (Literal::Integer(l), BinaryOp::LessThan, Literal::Integer(r)) => Ok(Literal::Boolean(l < r)),
        (Literal::Float(l), BinaryOp::LessThan, Literal::Float(r)) => Ok(Literal::Boolean(l < r)),
        
        (Literal::Integer(l), BinaryOp::LessThanOrEqual, Literal::Integer(r)) => Ok(Literal::Boolean(l <= r)),
        (Literal::Float(l), BinaryOp::LessThanOrEqual, Literal::Float(r)) => Ok(Literal::Boolean(l <= r)),
        
        // Logical operations
        (Literal::Boolean(l), BinaryOp::And, Literal::Boolean(r)) => Ok(Literal::Boolean(*l && *r)),
        (Literal::Boolean(l), BinaryOp::Or, Literal::Boolean(r)) => Ok(Literal::Boolean(*l || *r)),
        
        _ => Err(EvaluationError::TypeMismatch {
            op: format!("{:?}", op),
            left_type: type_of_literal(left),
            right_type: type_of_literal(right),
        }),
    }
}

/// Evaluate a unary operation
fn eval_unary_op(value: &Literal, op: UnaryOp) -> Result<Literal, EvaluationError> {
    match (value, op) {
        (Literal::Integer(v), UnaryOp::Negate) => Ok(Literal::Integer(-v)),
        (Literal::Float(v), UnaryOp::Negate) => Ok(Literal::Float(-v)),
        (Literal::Boolean(v), UnaryOp::Not) => Ok(Literal::Boolean(!v)),
        (Literal::Integer(v), UnaryOp::Abs) => Ok(Literal::Integer(v.abs())),
        (Literal::Float(v), UnaryOp::Abs) => Ok(Literal::Float(v.abs())),
        (Literal::Float(v), UnaryOp::Sqrt) => {
            if *v < 0.0 {
                Err(EvaluationError::InvalidOperation("sqrt of negative number".to_string()))
            } else {
                Ok(Literal::Float(v.sqrt()))
            }
        }
        (Literal::Float(v), UnaryOp::Log) => {
            if *v <= 0.0 {
                Err(EvaluationError::InvalidOperation("log of non-positive number".to_string()))
            } else {
                Ok(Literal::Float(v.ln()))
            }
        }
        (Literal::Float(v), UnaryOp::Exp) => Ok(Literal::Float(v.exp())),
        _ => Err(EvaluationError::TypeMismatch {
            op: format!("{:?}", op),
            left_type: type_of_literal(value),
            right_type: "".to_string(),
        }),
    }
}

/// Evaluate a function call
fn eval_function(name: &str, args: &[Literal]) -> Result<Literal, EvaluationError> {
    match (name, args) {
        ("pow", [Literal::Float(base), Literal::Float(exp)]) => {
            Ok(Literal::Float(base.powf(*exp)))
        }
        ("sin", [Literal::Float(v)]) => Ok(Literal::Float(v.sin())),
        ("cos", [Literal::Float(v)]) => Ok(Literal::Float(v.cos())),
        ("tan", [Literal::Float(v)]) => Ok(Literal::Float(v.tan())),
        ("abs", [Literal::Integer(v)]) => Ok(Literal::Integer(v.abs())),
        ("abs", [Literal::Float(v)]) => Ok(Literal::Float(v.abs())),
        ("round", [Literal::Float(v)]) => Ok(Literal::Float(v.round())),
        ("floor", [Literal::Float(v)]) => Ok(Literal::Float(v.floor())),
        ("ceil", [Literal::Float(v)]) => Ok(Literal::Float(v.ceil())),
        _ => Err(EvaluationError::UnknownFunction(name.to_string())),
    }
}

/// Cast a value to a different type
fn cast_value(value: &Literal, target_type: &DataType) -> Result<Literal, EvaluationError> {
    match (value, target_type) {
        (Literal::Integer(v), DataType::Float) => Ok(Literal::Float(*v as f64)),
        (Literal::Float(v), DataType::Integer) => Ok(Literal::Integer(*v as i64)),
        (Literal::Integer(v), DataType::String) => Ok(Literal::String(v.to_string())),
        (Literal::Float(v), DataType::String) => Ok(Literal::String(v.to_string())),
        (Literal::Boolean(v), DataType::String) => Ok(Literal::String(v.to_string())),
        _ => Err(EvaluationError::InvalidCast {
            from_type: type_of_literal(value),
            to_type: format!("{:?}", target_type),
        }),
    }
}

/// Get the type of a literal as a string
fn type_of_literal(lit: &Literal) -> String {
    match lit {
        Literal::Boolean(_) => "boolean",
        Literal::Integer(_) => "integer",
        Literal::Float(_) => "float",
        Literal::String(_) => "string",
        Literal::Null => "null",
    }.to_string()
}

/// Errors that can occur during evaluation
#[derive(Debug, thiserror::Error)]
pub enum EvaluationError {
    #[error("Column not found: {0}")]
    ColumnNotFound(String),
    
    #[error("Division by zero")]
    DivisionByZero,
    
    #[error("Type mismatch for operation {op}: left is {left_type}, right is {right_type}")]
    TypeMismatch {
        op: String,
        left_type: String,
        right_type: String,
    },
    
    #[error("Invalid cast from {from_type} to {to_type}")]
    InvalidCast {
        from_type: String,
        to_type: String,
    },
    
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    #[error("Unknown function: {0}")]
    UnknownFunction(String),
    
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;
    
    #[test]
    fn test_evaluate_literal() {
        let ctx = EvaluationContext::new();
        let expr = Expr::lit_int(42);
        assert_eq!(ctx.evaluate(&expr).unwrap(), Literal::Integer(42));
    }
    
    #[test]
    fn test_evaluate_column() {
        let mut ctx = EvaluationContext::new();
        ctx.set_column("x".to_string(), Literal::Integer(10));
        let expr = Expr::col("x");
        assert_eq!(ctx.evaluate(&expr).unwrap(), Literal::Integer(10));
    }
    
    #[test]
    fn test_evaluate_binary_expression() {
        let mut ctx = EvaluationContext::new();
        ctx.set_column("a".to_string(), Literal::Integer(5));
        ctx.set_column("b".to_string(), Literal::Integer(3));
        
        let expr = Expr::col("a").add(Expr::col("b"));
        assert_eq!(ctx.evaluate(&expr).unwrap(), Literal::Integer(8));
        
        let expr = Expr::col("a").mul(Expr::col("b"));
        assert_eq!(ctx.evaluate(&expr).unwrap(), Literal::Integer(15));
    }
    
    #[test]
    fn test_evaluate_unary_expression() {
        let mut ctx = EvaluationContext::new();
        ctx.set_column("x".to_string(), Literal::Integer(-5));
        
        let expr = Expr::col("x").neg();
        assert_eq!(ctx.evaluate(&expr).unwrap(), Literal::Integer(5));
        
        let expr = Expr::lit_bool(false).not();
        assert_eq!(ctx.evaluate(&expr).unwrap(), Literal::Boolean(true));
    }

    #[test]
    fn test_division_by_zero() {
        let ctx = EvaluationContext::new();
        let expr = Expr::lit_int(5).div(Expr::lit_int(0));
        match ctx.evaluate(&expr) {
            Err(EvaluationError::DivisionByZero) => (),
            other => panic!("Expected DivisionByZero error, got {:?}", other),
        }
        
        let expr = Expr::lit_float(5.0).div(Expr::lit_float(0.0));
        match ctx.evaluate(&expr) {
            Err(EvaluationError::DivisionByZero) => (),
            other => panic!("Expected DivisionByZero error, got {:?}", other),
        }
    }

    #[test]
    fn test_type_mismatch() {
        let ctx = EvaluationContext::new();
        // Adding integer and boolean should fail
        let expr = Expr::lit_int(5).add(Expr::lit_bool(true));
        match ctx.evaluate(&expr) {
            Err(EvaluationError::TypeMismatch { .. }) => (),
            other => panic!("Expected TypeMismatch error, got {:?}", other),
        }
    }

    #[test]
    fn test_unknown_column() {
        let ctx = EvaluationContext::new();
        let expr = Expr::col("nonexistent");
        match ctx.evaluate(&expr) {
            Err(EvaluationError::ColumnNotFound(name)) if name == "nonexistent" => (),
            other => panic!("Expected ColumnNotFound error, got {:?}", other),
        }
    }

    #[test]
    fn test_function_evaluation() {
        let ctx = EvaluationContext::new();
        // Test pow function
        let expr = Expr::function("pow", vec![Expr::lit_float(2.0), Expr::lit_float(3.0)]);
        match ctx.evaluate(&expr) {
            Ok(Literal::Float(x)) if (x - 8.0).abs() < 1e-10 => (),
            other => panic!("Expected pow(2.0, 3.0) = 8.0, got {:?}", other),
        }
        
        // Test sin function
        let expr = Expr::function("sin", vec![Expr::lit_float(0.0)]);
        match ctx.evaluate(&expr) {
            Ok(Literal::Float(x)) if x.abs() < 1e-10 => (),
            other => panic!("Expected sin(0.0) = 0.0, got {:?}", other),
        }
        
        // Test unknown function
        let expr = Expr::function("unknown_func", vec![Expr::lit_int(1)]);
        match ctx.evaluate(&expr) {
            Err(EvaluationError::UnknownFunction(name)) if name == "unknown_func" => (),
            other => panic!("Expected UnknownFunction error, got {:?}", other),
        }
    }

    #[test]
    fn test_conditional_evaluation() {
        let mut ctx = EvaluationContext::new();
        ctx.set_column("cond".to_string(), Literal::Boolean(true));
        ctx.set_column("a".to_string(), Literal::Integer(10));
        ctx.set_column("b".to_string(), Literal::Integer(20));
        
        let expr = Expr::conditional(
            Expr::col("cond"),
            Expr::col("a"),
            Expr::col("b"),
        );
        assert_eq!(ctx.evaluate(&expr).unwrap(), Literal::Integer(10));
        
        // Test false condition
        ctx.set_column("cond".to_string(), Literal::Boolean(false));
        assert_eq!(ctx.evaluate(&expr).unwrap(), Literal::Integer(20));
    }

    #[test]
    fn test_cast_evaluation() {
        let ctx = EvaluationContext::new();
        let expr = Expr::lit_int(42).cast(DataType::Float);
        match ctx.evaluate(&expr) {
            Ok(Literal::Float(x)) if (x - 42.0).abs() < 1e-10 => (),
            other => panic!("Expected cast to float 42.0, got {:?}", other),
        }
        
        // Invalid cast should fail
        let expr = Expr::lit_string("hello").cast(DataType::Integer);
        match ctx.evaluate(&expr) {
            Err(EvaluationError::InvalidCast { .. }) => (),
            other => panic!("Expected InvalidCast error, got {:?}", other),
        }
    }
}