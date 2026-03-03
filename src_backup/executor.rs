//! Expression and plan execution engine
//!
//! This module provides functionality to execute expressions and logical plans
//! against data providers.

use crate::data_provider::{DataProvider, Series};
use crate::expr::{Expr, BinaryOp, UnaryOp, AggregateOp, Literal};
use crate::logical_plan::LogicalPlan;
use std::collections::HashMap;

/// Result of executing a plan or expression
#[derive(Debug, Clone)]
pub struct ExecResult {
    /// Scalar value result (if applicable)
    pub value: Option<f64>,
    /// Series result (if applicable)
    pub series: Option<Series>,
    /// Error message (if execution failed)
    pub error: Option<String>,
}

impl ExecResult {
    /// Create a successful scalar result
    pub fn scalar(value: f64) -> Self {
        ExecResult {
            value: Some(value),
            series: None,
            error: None,
        }
    }
    
    /// Create a successful series result
    pub fn series(series: Series) -> Self {
        ExecResult {
            value: None,
            series: Some(series),
            error: None,
        }
    }
    
    /// Create an error result
    pub fn error(msg: String) -> Self {
        ExecResult {
            value: None,
            series: None,
            error: Some(msg),
        }
    }
    
    /// Check if execution was successful
    pub fn is_ok(&self) -> bool {
        self.error.is_none()
    }
    
    /// Get the scalar value or panic if not a scalar
    pub fn unwrap_scalar(&self) -> f64 {
        self.value.expect("Expected scalar value")
    }
    
    /// Get the series or panic if not a series
    pub fn unwrap_series(&self) -> &Series {
        self.series.as_ref().expect("Expected series")
    }
}

/// Trait for executing logical plans against data providers
pub trait Executor {
    /// Execute a logical plan against a data provider
    fn execute_plan(
        &self,
        plan: &LogicalPlan,
        provider: &dyn DataProvider,
        ctx: Option<&HashMap<String, String>>,
    ) -> ExecResult;
    
    /// Evaluate an expression against a data provider
    fn evaluate_expr(
        &self,
        expr: &Expr,
        provider: &dyn DataProvider,
        ctx: Option<&HashMap<String, String>>,
    ) -> ExecResult;
}

/// Simple expression evaluator that handles basic operations
pub struct EvalExecutor;

impl EvalExecutor {
    /// Evaluate a binary operation
    fn eval_binary(
        &self,
        op: BinaryOp,
        left: f64,
        right: f64,
    ) -> Result<f64, String> {
        match op {
            BinaryOp::Add => Ok(left + right),
            BinaryOp::Subtract => Ok(left - right),
            BinaryOp::Multiply => Ok(left * right),
            BinaryOp::Divide => {
                if right == 0.0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(left / right)
                }
            }
            BinaryOp::Modulo => {
                if right == 0.0 {
                    Err("Modulo by zero".to_string())
                } else {
                    Ok(left % right)
                }
            }
            // Comparison operators return 1.0 for true, 0.0 for false
            BinaryOp::Equal => Ok((left == right) as i64 as f64),
            BinaryOp::NotEqual => Ok((left != right) as i64 as f64),
            BinaryOp::GreaterThan => Ok((left > right) as i64 as f64),
            BinaryOp::GreaterThanOrEqual => Ok((left >= right) as i64 as f64),
            BinaryOp::LessThan => Ok((left < right) as i64 as f64),
            BinaryOp::LessThanOrEqual => Ok((left <= right) as i64 as f64),
            // Logical operators treat 0.0 as false, non-zero as true
            BinaryOp::And => Ok(((left != 0.0) && (right != 0.0)) as i64 as f64),
            BinaryOp::Or => Ok(((left != 0.0) || (right != 0.0)) as i64 as f64),
        }
    }
    
    /// Evaluate a unary operation
    fn eval_unary(&self, op: UnaryOp, value: f64) -> Result<f64, String> {
        match op {
            UnaryOp::Negate => Ok(-value),
            UnaryOp::Not => Ok((value == 0.0) as i64 as f64),
            UnaryOp::Abs => Ok(value.abs()),
            UnaryOp::Sqrt => {
                if value < 0.0 {
                    Err("sqrt of negative number".to_string())
                } else {
                    Ok(value.sqrt())
                }
            }
            UnaryOp::Log => {
                if value <= 0.0 {
                    Err("log of non-positive number".to_string())
                } else {
                    Ok(value.ln())
                }
            }
            UnaryOp::Exp => Ok(value.exp()),
        }
    }
    
    /// Evaluate an aggregate operation on a series
    fn eval_aggregate(
        &self,
        op: AggregateOp,
        series: &Series,
        distinct: bool,
    ) -> Result<f64, String> {
        if series.is_empty() {
            return Err("Cannot aggregate empty series".to_string());
        }
        
        // For now, distinct aggregation is not supported due to f64 hashing issues
        // We'll implement a simple version using vector deduplication with tolerance
        if distinct {
            // Simple deduplication with epsilon tolerance
            let mut unique = Vec::new();
            let epsilon = 1e-10_f64;
            
            for &value in series {
                if !unique.iter().any(|v: &f64| (*v - value).abs() < epsilon) {
                    unique.push(value);
                }
            }
            
            match op {
                AggregateOp::Sum => Ok(unique.iter().sum()),
                AggregateOp::Mean => {
                    let sum: f64 = unique.iter().sum();
                    Ok(sum / unique.len() as f64)
                }
                AggregateOp::Max => {
                    unique.iter()
                        .copied()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or_else(|| "Cannot compute max of empty series".to_string())
                }
                AggregateOp::Min => {
                    unique.iter()
                        .copied()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or_else(|| "Cannot compute min of empty series".to_string())
                }
                AggregateOp::Count => Ok(unique.len() as f64),
                AggregateOp::StdDev => {
                    let mean = unique.iter().sum::<f64>() / unique.len() as f64;
                    let variance = unique.iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f64>() / unique.len() as f64;
                    Ok(variance.sqrt())
                }
                AggregateOp::Variance => {
                    let mean = unique.iter().sum::<f64>() / unique.len() as f64;
                    let variance = unique.iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f64>() / unique.len() as f64;
                    Ok(variance)
                }
            }
        } else {
            match op {
                AggregateOp::Sum => Ok(series.iter().sum()),
                AggregateOp::Mean => {
                    let sum: f64 = series.iter().sum();
                    Ok(sum / series.len() as f64)
                }
                AggregateOp::Max => {
                    series.iter()
                        .copied()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or_else(|| "Cannot compute max of empty series".to_string())
                }
                AggregateOp::Min => {
                    series.iter()
                        .copied()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .ok_or_else(|| "Cannot compute min of empty series".to_string())
                }
                AggregateOp::Count => Ok(series.len() as f64),
                AggregateOp::StdDev => {
                    let mean = series.iter().sum::<f64>() / series.len() as f64;
                    let variance = series.iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f64>() / series.len() as f64;
                    Ok(variance.sqrt())
                }
                AggregateOp::Variance => {
                    let mean = series.iter().sum::<f64>() / series.len() as f64;
                    let variance = series.iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f64>() / series.len() as f64;
                    Ok(variance)
                }
            }
        }
    }
}

impl Executor for EvalExecutor {
    fn evaluate_expr(
        &self,
        expr: &Expr,
        provider: &dyn DataProvider,
        ctx: Option<&HashMap<String, String>>,
    ) -> ExecResult {
        match expr {
            Expr::Literal(lit) => {
                match lit {
                    Literal::Float(f) => ExecResult::scalar(*f),
                    Literal::Integer(i) => ExecResult::scalar(*i as f64),
                    Literal::Boolean(b) => ExecResult::scalar((*b as i64) as f64),
                    _ => ExecResult::error(format!("Unsupported literal type: {:?}", lit)),
                }
            }
            Expr::Column(name) => {
                // First try to get as scalar
                if let Some(scalar) = provider.get_scalar(name) {
                    return ExecResult::scalar(scalar);
                }
                
                // If context is provided, try to get as grouped series
                if let Some(ctx_map) = ctx {
                    if let (Some(table), Some(symbol), Some(trading_date)) = (
                        ctx_map.get("table"),
                        ctx_map.get("symbol"),
                        ctx_map.get("trading_date"),
                    ) {
                        if let Some(series) = provider.get_series_group(table, symbol, trading_date, name) {
                            // For now, we'll return the last value of the series
                            // In a full implementation, we might want to handle series differently
                            if let Some(last) = series.last() {
                                return ExecResult::scalar(*last);
                            }
                        }
                    }
                }
                
                // Try to get as plain series
                if let Some(series) = provider.get_series(name) {
                    if let Some(last) = series.last() {
                        return ExecResult::scalar(*last);
                    }
                }
                
                ExecResult::error(format!("Column not found: {}", name))
            }
            Expr::BinaryExpr { left, op, right } => {
                let left_result = self.evaluate_expr(left, provider, ctx);
                let right_result = self.evaluate_expr(right, provider, ctx);
                
                if !left_result.is_ok() {
                    return left_result;
                }
                if !right_result.is_ok() {
                    return right_result;
                }
                
                let left_val = left_result.unwrap_scalar();
                let right_val = right_result.unwrap_scalar();
                
                match self.eval_binary(*op, left_val, right_val) {
                    Ok(result) => ExecResult::scalar(result),
                    Err(err) => ExecResult::error(err),
                }
            }
            Expr::UnaryExpr { op, expr: inner } => {
                let inner_result = self.evaluate_expr(inner, provider, ctx);
                if !inner_result.is_ok() {
                    return inner_result;
                }
                
                let inner_val = inner_result.unwrap_scalar();
                match self.eval_unary(*op, inner_val) {
                    Ok(result) => ExecResult::scalar(result),
                    Err(err) => ExecResult::error(err),
                }
            }
            Expr::Aggregate { op, expr: inner, distinct } => {
                // For now, we assume the inner expression refers to a column
                // that can be retrieved as a series
                if let Expr::Column(col_name) = inner.as_ref() {
                    // Try to get series from provider
                    let series = if let Some(ctx_map) = ctx {
                        if let (Some(table), Some(symbol), Some(trading_date)) = (
                            ctx_map.get("table"),
                            ctx_map.get("symbol"),
                            ctx_map.get("trading_date"),
                        ) {
                            provider.get_series_group(table, symbol, trading_date, col_name)
                        } else {
                            None
                        }
                    } else {
                        None
                    }.or_else(|| provider.get_series(col_name));
                    
                    if let Some(series) = series {
                        match self.eval_aggregate(*op, &series, *distinct) {
                            Ok(result) => ExecResult::scalar(result),
                            Err(err) => ExecResult::error(err),
                        }
                    } else {
                        ExecResult::error(format!("Cannot get series for column: {}", col_name))
                    }
                } else {
                    // TODO: Support aggregate over arbitrary expressions
                    ExecResult::error("Aggregate over non-column expressions not yet supported".to_string())
                }
            }
            Expr::FunctionCall { name, args } => {
                // For now, implement a few simple functions
                if args.is_empty() {
                    return ExecResult::error(format!("Function {} called with no arguments", name));
                }
                
                // Evaluate all arguments
                let mut arg_values = Vec::new();
                for arg in args {
                    let arg_result = self.evaluate_expr(arg, provider, ctx);
                    if !arg_result.is_ok() {
                        return arg_result;
                    }
                    arg_values.push(arg_result.unwrap_scalar());
                }
                
                match name.as_str() {
                    "sqrt" => {
                        if arg_values.len() != 1 {
                            return ExecResult::error("sqrt requires exactly 1 argument".to_string());
                        }
                        let x = arg_values[0];
                        if x < 0.0 {
                            ExecResult::error("sqrt of negative number".to_string())
                        } else {
                            ExecResult::scalar(x.sqrt())
                        }
                    }
                    "abs" => {
                        if arg_values.len() != 1 {
                            return ExecResult::error("abs requires exactly 1 argument".to_string());
                        }
                        ExecResult::scalar(arg_values[0].abs())
                    }
                    "log" => {
                        if arg_values.len() != 1 {
                            return ExecResult::error("log requires exactly 1 argument".to_string());
                        }
                        let x = arg_values[0];
                        if x <= 0.0 {
                            ExecResult::error("log of non-positive number".to_string())
                        } else {
                            ExecResult::scalar(x.ln())
                        }
                    }
                    "exp" => {
                        if arg_values.len() != 1 {
                            return ExecResult::error("exp requires exactly 1 argument".to_string());
                        }
                        ExecResult::scalar(arg_values[0].exp())
                    }
                    "pow" => {
                        if arg_values.len() != 2 {
                            return ExecResult::error("pow requires exactly 2 arguments".to_string());
                        }
                        ExecResult::scalar(arg_values[0].powf(arg_values[1]))
                    }
                    _ => ExecResult::error(format!("Unknown function: {}", name)),
                }
            }
            Expr::Conditional { condition, then_expr, else_expr } => {
                let cond_result = self.evaluate_expr(condition, provider, ctx);
                if !cond_result.is_ok() {
                    return cond_result;
                }
                
                let cond_val = cond_result.unwrap_scalar();
                // Non-zero means true
                if cond_val != 0.0 {
                    self.evaluate_expr(then_expr, provider, ctx)
                } else {
                    self.evaluate_expr(else_expr, provider, ctx)
                }
            }
            Expr::Cast { expr: inner, data_type: _ } => {
                // For now, just evaluate the inner expression
                // Type casting would be implemented here
                self.evaluate_expr(inner, provider, ctx)
            }
        }
    }
    
    fn execute_plan(
        &self,
        plan: &LogicalPlan,
        provider: &dyn DataProvider,
        ctx: Option<&HashMap<String, String>>,
    ) -> ExecResult {
        match plan {
            LogicalPlan::Expression(expr) => {
                self.evaluate_expr(expr, provider, ctx)
            }
            LogicalPlan::Projection { expr, input: _input, .. } => {
                // For now, just evaluate the first expression in the projection
                if expr.is_empty() {
                    return ExecResult::error("Empty projection".to_string());
                }
                
                // First, we need to execute the input to get the context
                // For simplicity, we'll just evaluate the expression directly
                // assuming all columns are available from the provider
                self.evaluate_expr(&expr[0], provider, ctx)
            }
            LogicalPlan::Filter { predicate, input } => {
                // Execute the input first
                let input_result = self.execute_plan(input, provider, ctx);
                if !input_result.is_ok() {
                    return input_result;
                }
                
                // Evaluate the predicate
                let pred_result = self.evaluate_expr(predicate, provider, ctx);
                if !pred_result.is_ok() {
                    return pred_result;
                }
                
                // If predicate is true (non-zero), return input result
                let pred_val = pred_result.unwrap_scalar();
                if pred_val != 0.0 {
                    input_result
                } else {
                    ExecResult::error("Filter predicate false".to_string())
                }
            }
            LogicalPlan::Scan { source_name: _source_name, projection, filters } => {
                // For scans, we need to set up the execution context
                // For now, we'll just evaluate any filters and return a dummy result
                for filter in filters {
                    let filter_result = self.evaluate_expr(filter, provider, ctx);
                    if !filter_result.is_ok() {
                        return filter_result;
                    }
                    // If any filter is false, return empty result
                    let filter_val = filter_result.unwrap_scalar();
                    if filter_val == 0.0 {
                        return ExecResult::error("Scan filter false".to_string());
                    }
                }
                
                // If there's a projection, try to evaluate the first column
                if let Some(cols) = projection {
                    if !cols.is_empty() {
                        let col_expr = Expr::Column(cols[0].clone());
                        return self.evaluate_expr(&col_expr, provider, ctx);
                    }
                }
                
                // Default: return a dummy scalar
                ExecResult::scalar(1.0)
            }
            _ => ExecResult::error(format!("Unsupported plan node: {:?}", plan)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;
    
    #[test]
    fn test_eval_literal() {
        let provider = crate::data_provider::NullProvider::new();
        let executor = EvalExecutor;
        
        let expr = Expr::Literal(Literal::Float(42.0));
        let result = executor.evaluate_expr(&expr, &provider, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap_scalar(), 42.0);
        
        let expr = Expr::Literal(Literal::Integer(100));
        let result = executor.evaluate_expr(&expr, &provider, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap_scalar(), 100.0);
    }
    
    #[test]
    fn test_eval_binary_operations() {
        let provider = crate::data_provider::NullProvider::new();
        let executor = EvalExecutor;
        
        // 2 + 3 = 5
        let expr = Expr::Literal(Literal::Float(2.0)).add(Expr::Literal(Literal::Float(3.0)));
        let result = executor.evaluate_expr(&expr, &provider, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap_scalar(), 5.0);
        
        // 10 / 2 = 5
        let expr = Expr::Literal(Literal::Float(10.0)).div(Expr::Literal(Literal::Float(2.0)));
        let result = executor.evaluate_expr(&expr, &provider, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap_scalar(), 5.0);
        
        // Division by zero
        let expr = Expr::Literal(Literal::Float(10.0)).div(Expr::Literal(Literal::Float(0.0)));
        let result = executor.evaluate_expr(&expr, &provider, None);
        assert!(!result.is_ok());
    }
    
    #[test]
    fn test_eval_column() {
        let mut scalars = std::collections::HashMap::new();
        scalars.insert("price".to_string(), 150.0);
        scalars.insert("volume".to_string(), 1000.0);
        let provider = crate::data_provider::MockProvider::new(scalars);
        let executor = EvalExecutor;
        
        let expr = Expr::Column("price".to_string());
        let result = executor.evaluate_expr(&expr, &provider, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap_scalar(), 150.0);
        
        let expr = Expr::Column("unknown".to_string());
        let result = executor.evaluate_expr(&expr, &provider, None);
        assert!(!result.is_ok());
    }
    
    #[test]
    fn test_eval_function() {
        let provider = crate::data_provider::NullProvider::new();
        let executor = EvalExecutor;
        
        // sqrt(16) = 4
        let expr = Expr::function("sqrt", vec![Expr::Literal(Literal::Float(16.0))]);
        let result = executor.evaluate_expr(&expr, &provider, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap_scalar(), 4.0);
        
        // sqrt(-1) should error
        let expr = Expr::function("sqrt", vec![Expr::Literal(Literal::Float(-1.0))]);
        let result = executor.evaluate_expr(&expr, &provider, None);
        assert!(!result.is_ok());
    }
    
    #[test]
    fn test_eval_conditional() {
        let provider = crate::data_provider::NullProvider::new();
        let executor = EvalExecutor;
        
        // if true then 1 else 2
        let expr = Expr::conditional(
            Expr::Literal(Literal::Boolean(true)),
            Expr::Literal(Literal::Float(1.0)),
            Expr::Literal(Literal::Float(2.0)),
        );
        let result = executor.evaluate_expr(&expr, &provider, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap_scalar(), 1.0);
        
        // if false then 1 else 2
        let expr = Expr::conditional(
            Expr::Literal(Literal::Boolean(false)),
            Expr::Literal(Literal::Float(1.0)),
            Expr::Literal(Literal::Float(2.0)),
        );
        let result = executor.evaluate_expr(&expr, &provider, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap_scalar(), 2.0);
    }
}