//! Dimension inference for expressions
//!
//! This module provides functionality to infer whether expressions represent
//! scalar values or time series, ensuring dimensional consistency in computations.

use crate::expr::{Expr, BinaryOp, UnaryOp, AggregateOp};

/// Dimension kind (scalar vs time series)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DimKind {
    Scalar,
    TimeSeries,
}

/// Dimension information for an expression
#[derive(Clone, Debug)]
pub struct Dimension {
    pub kind: DimKind,
    pub name: Option<String>,
}

impl Dimension {
    /// Create a new scalar dimension
    pub fn new_scalar() -> Self {
        Dimension {
            kind: DimKind::Scalar,
            name: None,
        }
    }
    
    /// Create a new time series dimension
    pub fn new_timeseries() -> Self {
        Dimension {
            kind: DimKind::TimeSeries,
            name: None,
        }
    }
}

/// Context mapping column names to their dimensions
pub type DimensionContext = Vec<(String, Dimension)>;

/// Dimension inference error
#[derive(Debug)]
pub struct DimensionError(pub String);

impl std::fmt::Display for DimensionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dimension error: {}", self.0)
    }
}

impl std::error::Error for DimensionError {}

/// Infer the dimension of an expression given a context
pub fn infer_dimension(
    expr: &Expr,
    ctx: &DimensionContext,
) -> Result<Dimension, DimensionError> {
    match expr {
        Expr::Literal(_) => Ok(Dimension::new_scalar()),
        Expr::Column(name) => {
            for (col_name, dim) in ctx.iter() {
                if col_name == name {
                    return Ok(dim.clone());
                }
            }
            // Default to TimeSeries if column not found in context
            // This assumes unknown columns are likely time series in financial contexts
            Ok(Dimension::new_timeseries())
        }
        Expr::BinaryExpr { left, op, right } => {
            let left_dim = infer_dimension(left, ctx)?;
            let right_dim = infer_dimension(right, ctx)?;
            
            match op {
                BinaryOp::Add | BinaryOp::Subtract => {
                    // Addition/subtraction requires same dimensions
                    if left_dim.kind == right_dim.kind {
                        Ok(left_dim)
                    } else {
                        Err(DimensionError(format!(
                            "Add/Subtract dimension mismatch: {:?} vs {:?}",
                            left_dim.kind, right_dim.kind
                        )))
                    }
                }
                BinaryOp::Multiply | BinaryOp::Divide => {
                    // Multiplication/division allow mixed dimensions
                    // TimeSeries * Scalar -> TimeSeries
                    // Scalar * TimeSeries -> TimeSeries
                    // Scalar * Scalar -> Scalar
                    match (&left_dim.kind, &right_dim.kind) {
                        (DimKind::TimeSeries, DimKind::Scalar) => Ok(left_dim),
                        (DimKind::Scalar, DimKind::TimeSeries) => Ok(right_dim),
                        (DimKind::Scalar, DimKind::Scalar) => Ok(Dimension::new_scalar()),
                        _ => Err(DimensionError(format!(
                            "Multiply/Divide unsupported dimension combination: {:?} * {:?}",
                            left_dim.kind, right_dim.kind
                        ))),
                    }
                }
                BinaryOp::Modulo => {
                    // Modulo requires both to be scalars
                    if left_dim.kind == DimKind::Scalar && right_dim.kind == DimKind::Scalar {
                        Ok(Dimension::new_scalar())
                    } else {
                        Err(DimensionError("Modulo requires scalar dimensions".to_string()))
                    }
                }
                // Comparison operators produce scalar booleans
                BinaryOp::Equal | BinaryOp::NotEqual |
                BinaryOp::GreaterThan | BinaryOp::GreaterThanOrEqual |
                BinaryOp::LessThan | BinaryOp::LessThanOrEqual => {
                    // Allow TimeSeries vs Scalar comparisons (e.g., price > 100.0)
                    // Both produce scalar boolean results
                    Ok(Dimension::new_scalar())
                }
                // Logical operators require scalar booleans
                BinaryOp::And | BinaryOp::Or => {
                    if left_dim.kind == DimKind::Scalar && right_dim.kind == DimKind::Scalar {
                        Ok(Dimension::new_scalar())
                    } else {
                        Err(DimensionError("Logical operators require scalar dimensions".to_string()))
                    }
                }
            }
        }
        Expr::UnaryExpr { op: _, expr: inner } => {
            // Unary operations preserve dimension
            infer_dimension(inner, ctx)
        }
        Expr::FunctionCall { name: _, args } => {
            // Conservative rule: if any argument is TimeSeries, result is TimeSeries
            // Otherwise, result is Scalar
            let mut found_timeseries = false;
            for arg in args {
                let dim = infer_dimension(arg, ctx)?;
                if dim.kind == DimKind::TimeSeries {
                    found_timeseries = true;
                    break;
                }
            }
            if found_timeseries {
                Ok(Dimension::new_timeseries())
            } else {
                Ok(Dimension::new_scalar())
            }
        }
        Expr::Aggregate { op: _, expr: inner, distinct: _ } => {
            // Aggregate functions convert TimeSeries to Scalar
            // Check that the input is valid
            let _ = infer_dimension(inner, ctx)?;
            Ok(Dimension::new_scalar())
        }
        Expr::Conditional { condition, then_expr, else_expr } => {
            // Condition must be scalar
            let cond_dim = infer_dimension(condition, ctx)?;
            if cond_dim.kind != DimKind::Scalar {
                return Err(DimensionError("Condition must be scalar".to_string()));
            }
            
            // Branches can have mixed dimensions (TimeSeries vs Scalar)
            // Similar to multiplication: TimeSeries wins over Scalar
            let then_dim = infer_dimension(then_expr, ctx)?;
            let else_dim = infer_dimension(else_expr, ctx)?;
            
            match (&then_dim.kind, &else_dim.kind) {
                (DimKind::TimeSeries, DimKind::Scalar) => Ok(then_dim),
                (DimKind::Scalar, DimKind::TimeSeries) => Ok(else_dim),
                (DimKind::TimeSeries, DimKind::TimeSeries) => Ok(then_dim),
                (DimKind::Scalar, DimKind::Scalar) => Ok(then_dim),
            }
        }
        Expr::Cast { expr: inner, data_type: _ } => {
            // Cast preserves dimension
            infer_dimension(inner, ctx)
        }
    }
}

/// Validate that an expression has scalar dimension
pub fn validate_scalar(expr: &Expr, ctx: &DimensionContext) -> Result<(), DimensionError> {
    let dim = infer_dimension(expr, ctx)?;
    if dim.kind == DimKind::Scalar {
        Ok(())
    } else {
        Err(DimensionError("Expected scalar dimension".to_string()))
    }
}

/// Validate that an expression has time series dimension
pub fn validate_timeseries(expr: &Expr, ctx: &DimensionContext) -> Result<(), DimensionError> {
    let dim = infer_dimension(expr, ctx)?;
    if dim.kind == DimKind::TimeSeries {
        Ok(())
    } else {
        Err(DimensionError("Expected time series dimension".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{Expr, Literal};
    
    #[test]
    fn test_literal_dimension() {
        let ctx = DimensionContext::new();
        let lit = Expr::Literal(Literal::Float(42.0));
        let dim = infer_dimension(&lit, &ctx).unwrap();
        assert_eq!(dim.kind, DimKind::Scalar);
    }
    
    #[test]
    fn test_column_dimension() {
        let mut ctx = DimensionContext::new();
        ctx.push(("price".to_string(), Dimension::new_timeseries()));
        ctx.push(("volume".to_string(), Dimension::new_timeseries()));
        ctx.push(("risk_free_rate".to_string(), Dimension::new_scalar()));
        
        let price = Expr::Column("price".to_string());
        let dim = infer_dimension(&price, &ctx).unwrap();
        assert_eq!(dim.kind, DimKind::TimeSeries);
        
        let rate = Expr::Column("risk_free_rate".to_string());
        let dim = infer_dimension(&rate, &ctx).unwrap();
        assert_eq!(dim.kind, DimKind::Scalar);
        
        // Unknown column defaults to TimeSeries
        let unknown = Expr::Column("unknown".to_string());
        let dim = infer_dimension(&unknown, &ctx).unwrap();
        assert_eq!(dim.kind, DimKind::TimeSeries);
    }
    
    #[test]
    fn test_binary_operations() {
        let mut ctx = DimensionContext::new();
        ctx.push(("price".to_string(), Dimension::new_timeseries()));
        ctx.push(("volume".to_string(), Dimension::new_timeseries()));
        ctx.push(("factor".to_string(), Dimension::new_scalar()));
        
        // TimeSeries + TimeSeries = TimeSeries
        let expr = Expr::Column("price".to_string()).add(Expr::Column("volume".to_string()));
        let dim = infer_dimension(&expr, &ctx).unwrap();
        assert_eq!(dim.kind, DimKind::TimeSeries);
        
        // TimeSeries * Scalar = TimeSeries
        let expr = Expr::Column("price".to_string()).mul(Expr::Column("factor".to_string()));
        let dim = infer_dimension(&expr, &ctx).unwrap();
        assert_eq!(dim.kind, DimKind::TimeSeries);
        
        // Scalar * Scalar = Scalar
        let expr = Expr::Literal(Literal::Float(2.0)).mul(Expr::Column("factor".to_string()));
        let dim = infer_dimension(&expr, &ctx).unwrap();
        assert_eq!(dim.kind, DimKind::Scalar);
        
        // TimeSeries + Scalar should error
        let expr = Expr::Column("price".to_string()).add(Expr::Column("factor".to_string()));
        let result = infer_dimension(&expr, &ctx);
        assert!(result.is_err());
        
        // Comparison produces scalar
        let expr = Expr::Column("price".to_string()).gt(Expr::Literal(Literal::Float(100.0)));
        let dim = infer_dimension(&expr, &ctx).unwrap();
        assert_eq!(dim.kind, DimKind::Scalar);
    }
    
    #[test]
    fn test_conditional_dimension() {
        let mut ctx = DimensionContext::new();
        ctx.push(("price".to_string(), Dimension::new_timeseries()));
        ctx.push(("threshold".to_string(), Dimension::new_scalar()));
        
        // Valid conditional: scalar condition, both branches same dimension
        let expr = Expr::conditional(
            Expr::Column("price".to_string()).gt(Expr::Column("threshold".to_string())),
            Expr::Column("price".to_string()),
            Expr::Literal(Literal::Float(0.0)),
        );
        let dim = infer_dimension(&expr, &ctx).unwrap();
        assert_eq!(dim.kind, DimKind::TimeSeries);
        
        // Mixed dimensions: TimeSeries vs Scalar -> TimeSeries
        let expr = Expr::conditional(
            Expr::Literal(Literal::Boolean(true)),
            Expr::Column("price".to_string()),  // TimeSeries
            Expr::Literal(Literal::Float(0.0)), // Scalar
        );
        let dim = infer_dimension(&expr, &ctx).unwrap();
        assert_eq!(dim.kind, DimKind::TimeSeries);
    }
    
    #[test]
    fn test_function_call_dimension() {
        let mut ctx = DimensionContext::new();
        ctx.push(("price".to_string(), Dimension::new_timeseries()));
        ctx.push(("factor".to_string(), Dimension::new_scalar()));
        
        // Function with TimeSeries argument -> TimeSeries
        let expr = Expr::function("sqrt", vec![Expr::Column("price".to_string())]);
        let dim = infer_dimension(&expr, &ctx).unwrap();
        assert_eq!(dim.kind, DimKind::TimeSeries);
        
        // Function with only Scalar arguments -> Scalar
        let expr = Expr::function("max", vec![
            Expr::Literal(Literal::Float(1.0)),
            Expr::Column("factor".to_string()),
        ]);
        let dim = infer_dimension(&expr, &ctx).unwrap();
        assert_eq!(dim.kind, DimKind::Scalar);
    }
    
    #[test]
    fn test_aggregate_dimension() {
        let mut ctx = DimensionContext::new();
        ctx.push(("price".to_string(), Dimension::new_timeseries()));
        
        // Aggregate of TimeSeries -> Scalar
        let expr = Expr::Column("price".to_string()).aggregate(AggregateOp::Sum, false);
        let dim = infer_dimension(&expr, &ctx).unwrap();
        assert_eq!(dim.kind, DimKind::Scalar);
    }
}