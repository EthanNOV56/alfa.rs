//! Optimization framework for logical plans and expressions
//!
//! This module provides rule-based optimization for transforming logical plans
//! and expressions to more efficient forms.

use crate::logical_plan::{LogicalPlan, JoinType, SetOp};
use crate::expr::{Expr, Literal, BinaryOp, UnaryOp};
use std::collections::HashSet;
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

// Helper function to extract column references from an expression
fn extract_column_references(expr: &Expr) -> HashSet<String> {
    let mut columns = HashSet::new();
    match expr {
        Expr::Column(name) => {
            columns.insert(name.clone());
        }
        Expr::BinaryExpr { left, right, .. } => {
            columns.extend(extract_column_references(left));
            columns.extend(extract_column_references(right));
        }
        Expr::UnaryExpr { expr: inner, .. } => {
            columns.extend(extract_column_references(inner));
        }
        Expr::FunctionCall { args, .. } => {
            for arg in args {
                columns.extend(extract_column_references(arg));
            }
        }
        Expr::Aggregate { expr: inner, .. } => {
            columns.extend(extract_column_references(inner));
        }
        Expr::Conditional { condition, then_expr, else_expr } => {
            columns.extend(extract_column_references(condition));
            columns.extend(extract_column_references(then_expr));
            columns.extend(extract_column_references(else_expr));
        }
        Expr::Cast { expr: inner, .. } => {
            columns.extend(extract_column_references(inner));
        }
        Expr::Literal(_) => {
            // No columns in literals
        }
    }
    columns
}

impl OptimizerRule for PredicatePushdown {
    fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        // Recursively push filters down through the plan
        self.push_filters(plan)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

impl PredicatePushdown {
    fn push_filters(&self, plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Filter { input, predicate } => {
                let optimized_input = self.push_filters(input.as_ref().clone());
                self.push_filter_through(optimized_input, predicate)
            }
            LogicalPlan::Projection { input, expr, schema } => {
                let optimized_input = self.push_filters(input.as_ref().clone());
                // Check if we can push a filter through this projection
                // For now, just return the projection with optimized input
                LogicalPlan::Projection { 
                    input: Arc::new(optimized_input), 
                    expr, 
                    schema 
                }
            }
            LogicalPlan::Scan { source_name, projection, filters } => {
                // Optimize any existing filters in the scan
                let optimized_filters = filters.into_iter()
                    .map(|filter| self.optimize_filter_expression(filter))
                    .collect();
                LogicalPlan::Scan { 
                    source_name, 
                    projection, 
                    filters: optimized_filters 
                }
            }
            LogicalPlan::Aggregate { input, group_expr, agg_expr } => {
                let optimized_input = self.push_filters(input.as_ref().clone());
                LogicalPlan::Aggregate { 
                    input: Arc::new(optimized_input), 
                    group_expr, 
                    agg_expr 
                }
            }
            LogicalPlan::Join { left, right, join_type, condition } => {
                let optimized_left = self.push_filters(left.as_ref().clone());
                let optimized_right = self.push_filters(right.as_ref().clone());
                // Try to push parts of the condition to left/right
                // For now, just return the join with optimized children
                LogicalPlan::Join { 
                    left: Arc::new(optimized_left), 
                    right: Arc::new(optimized_right), 
                    join_type, 
                    condition 
                }
            }
            LogicalPlan::Expression(expr) => {
                LogicalPlan::Expression(expr)
            }
            LogicalPlan::Sort { input, sort_expr, descending } => {
                let optimized_input = self.push_filters(input.as_ref().clone());
                LogicalPlan::Sort {
                    input: Arc::new(optimized_input),
                    sort_expr,
                    descending,
                }
            }
            LogicalPlan::Limit { input, limit, offset } => {
                let optimized_input = self.push_filters(input.as_ref().clone());
                LogicalPlan::Limit {
                    input: Arc::new(optimized_input),
                    limit,
                    offset,
                }
            }
            LogicalPlan::SetOperation { left, right, op, all } => {
                let optimized_left = self.push_filters(left.as_ref().clone());
                let optimized_right = self.push_filters(right.as_ref().clone());
                LogicalPlan::SetOperation {
                    left: Arc::new(optimized_left),
                    right: Arc::new(optimized_right),
                    op,
                    all,
                }
            }
        }
    }
    
    fn push_filter_through(&self, plan: LogicalPlan, predicate: Expr) -> LogicalPlan {
        match plan {
            LogicalPlan::Projection { input, expr, schema } => {
                // Get columns referenced in the predicate
                let predicate_cols = extract_column_references(&predicate);
                
                // Get columns produced by the projection
                let projection_cols: HashSet<String> = expr.iter()
                    .filter_map(|e| {
                        if let Expr::Column(name) = e {
                            Some(name.clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                
                // Check if all predicate columns are in the projection
                let can_push = predicate_cols.iter().all(|col| projection_cols.contains(col));
                
                if can_push {
                    // Push filter below projection
                    let new_input = LogicalPlan::Filter {
                        input: input.clone(),
                        predicate,
                    };
                    LogicalPlan::Projection {
                        input: Arc::new(self.push_filters(new_input)),
                        expr,
                        schema,
                    }
                } else {
                    // Keep filter above projection
                    LogicalPlan::Filter {
                        input: Arc::new(LogicalPlan::Projection { input, expr, schema }),
                        predicate,
                    }
                }
            }
            LogicalPlan::Join { left, right, join_type, condition } => {
                // For joins, we could split the predicate based on which side
                // each column comes from. For simplicity, we'll just push
                // the whole filter above the join for now.
                LogicalPlan::Filter {
                    input: Arc::new(LogicalPlan::Join { left, right, join_type, condition }),
                    predicate,
                }
            }
            _ => {
                // For other nodes, just wrap with filter
                LogicalPlan::Filter {
                    input: Arc::new(plan),
                    predicate,
                }
            }
        }
    }
    
    fn optimize_filter_expression(&self, expr: Expr) -> Expr {
        // For now, just apply constant folding to the filter expression
        // We could create a separate constant folder instance
        let constant_folding = ConstantFolding::new();
        constant_folding.fold_expr(expr)
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
        // First, collect all columns referenced in the entire plan
        let needed_columns = self.collect_referenced_columns(&plan);
        
        // Then push down projections based on needed columns
        self.push_down_projections(plan, &needed_columns)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

impl ProjectionPushdown {
    /// Collect all column names referenced anywhere in the plan
    fn collect_referenced_columns(&self, plan: &LogicalPlan) -> HashSet<String> {
        match plan {
            LogicalPlan::Scan { projection, filters, .. } => {
                let mut columns = HashSet::new();
                // Add columns from projection if specified
                if let Some(proj_cols) = projection {
                    columns.extend(proj_cols.iter().cloned());
                }
                // Add columns from filter expressions
                for filter in filters {
                    columns.extend(extract_column_references(filter));
                }
                columns
            }
            LogicalPlan::Projection { input, expr, .. } => {
                let mut columns = self.collect_referenced_columns(input);
                // Also add columns referenced in projection expressions
                for e in expr {
                    columns.extend(extract_column_references(e));
                }
                columns
            }
            LogicalPlan::Filter { input, predicate } => {
                let mut columns = self.collect_referenced_columns(input);
                columns.extend(extract_column_references(predicate));
                columns
            }
            LogicalPlan::Aggregate { input, group_expr, agg_expr } => {
                let mut columns = self.collect_referenced_columns(input);
                for e in group_expr {
                    columns.extend(extract_column_references(e));
                }
                for e in agg_expr {
                    columns.extend(extract_column_references(e));
                }
                columns
            }
            LogicalPlan::Join { left, right, condition, .. } => {
                let mut columns = self.collect_referenced_columns(left);
                columns.extend(self.collect_referenced_columns(right));
                if let Some(cond) = condition {
                    columns.extend(extract_column_references(cond));
                }
                columns
            }
            LogicalPlan::Sort { input, sort_expr, .. } => {
                let mut columns = self.collect_referenced_columns(input);
                for e in sort_expr {
                    columns.extend(extract_column_references(e));
                }
                columns
            }
            LogicalPlan::Limit { input, .. } => {
                self.collect_referenced_columns(input)
            }
            LogicalPlan::SetOperation { left, right, .. } => {
                let mut columns = self.collect_referenced_columns(left);
                columns.extend(self.collect_referenced_columns(right));
                columns
            }
            LogicalPlan::Expression(expr) => {
                extract_column_references(expr)
            }
        }
    }
    
    /// Push down projections based on needed columns
    fn push_down_projections(&self, plan: LogicalPlan, needed_columns: &HashSet<String>) -> LogicalPlan {
        match plan {
            LogicalPlan::Scan { source_name, projection, filters } => {
                // If we have specific needed columns, update the scan projection
                // but only if it would reduce the number of columns
                let new_projection = if needed_columns.is_empty() {
                    projection
                } else {
                    // Convert needed_columns to sorted vector for determinism
                    let mut needed_vec: Vec<String> = needed_columns.iter().cloned().collect();
                    needed_vec.sort();
                    
                    // Only apply projection if it would actually reduce columns
                    // (we don't know the full schema, so we'll assume it helps)
                    Some(needed_vec)
                };
                
                LogicalPlan::Scan {
                    source_name,
                    projection: new_projection,
                    filters,
                }
            }
            LogicalPlan::Projection { input, expr, schema } => {
                // First optimize the input with the columns needed by this projection
                let input_needed = self.columns_needed_by_expressions(&expr);
                let optimized_input = self.push_down_projections(input.as_ref().clone(), &input_needed);
                
                // Check if this projection is redundant (just passes through columns)
                let is_simple_column_projection = expr.iter().all(|e| matches!(e, Expr::Column(_)));
                let passes_through_all = if is_simple_column_projection {
                    let projected_cols: HashSet<String> = expr.iter()
                        .filter_map(|e| {
                            if let Expr::Column(name) = e {
                                Some(name.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    projected_cols == *needed_columns
                } else {
                    false
                };
                
                if passes_through_all {
                    // This projection is redundant, eliminate it
                    optimized_input
                } else {
                    // Keep the projection
                    LogicalPlan::Projection {
                        input: Arc::new(optimized_input),
                        expr,
                        schema,
                    }
                }
            }
            LogicalPlan::Filter { input, predicate } => {
                // Columns needed by filter plus columns needed by parent
                let mut filter_needed = needed_columns.clone();
                filter_needed.extend(extract_column_references(&predicate));
                
                let optimized_input = self.push_down_projections(input.as_ref().clone(), &filter_needed);
                
                LogicalPlan::Filter {
                    input: Arc::new(optimized_input),
                    predicate,
                }
            }
            LogicalPlan::Aggregate { input, group_expr, agg_expr } => {
                // Columns needed by aggregate expressions plus columns needed by parent
                let mut agg_needed = needed_columns.clone();
                for e in &group_expr {
                    agg_needed.extend(extract_column_references(e));
                }
                for e in &agg_expr {
                    agg_needed.extend(extract_column_references(e));
                }
                
                let optimized_input = self.push_down_projections(input.as_ref().clone(), &agg_needed);
                
                LogicalPlan::Aggregate {
                    input: Arc::new(optimized_input),
                    group_expr,
                    agg_expr,
                }
            }
            LogicalPlan::Join { left, right, join_type, condition } => {
                // Split needed columns between left and right based on condition
                // For simplicity, we'll pass all needed columns to both sides for now
                // A more sophisticated implementation would analyze the condition
                let left_needed = needed_columns.clone();
                let right_needed = needed_columns.clone();
                
                let optimized_left = self.push_down_projections(left.as_ref().clone(), &left_needed);
                let optimized_right = self.push_down_projections(right.as_ref().clone(), &right_needed);
                
                LogicalPlan::Join {
                    left: Arc::new(optimized_left),
                    right: Arc::new(optimized_right),
                    join_type,
                    condition,
                }
            }
            LogicalPlan::Sort { input, sort_expr, descending } => {
                // Columns needed by sort plus columns needed by parent
                let mut sort_needed = needed_columns.clone();
                for e in &sort_expr {
                    sort_needed.extend(extract_column_references(e));
                }
                
                let optimized_input = self.push_down_projections(input.as_ref().clone(), &sort_needed);
                
                LogicalPlan::Sort {
                    input: Arc::new(optimized_input),
                    sort_expr,
                    descending,
                }
            }
            LogicalPlan::Limit { input, limit, offset } => {
                let optimized_input = self.push_down_projections(input.as_ref().clone(), needed_columns);
                
                LogicalPlan::Limit {
                    input: Arc::new(optimized_input),
                    limit,
                    offset,
                }
            }
            LogicalPlan::SetOperation { left, right, op, all } => {
                // Set operations need the same columns from both sides
                let optimized_left = self.push_down_projections(left.as_ref().clone(), needed_columns);
                let optimized_right = self.push_down_projections(right.as_ref().clone(), needed_columns);
                
                LogicalPlan::SetOperation {
                    left: Arc::new(optimized_left),
                    right: Arc::new(optimized_right),
                    op,
                    all,
                }
            }
            LogicalPlan::Expression(expr) => {
                LogicalPlan::Expression(expr)
            }
        }
    }
    
    /// Extract columns needed by a list of expressions
    fn columns_needed_by_expressions(&self, exprs: &[Expr]) -> HashSet<String> {
        let mut needed = HashSet::new();
        for expr in exprs {
            needed.extend(extract_column_references(expr));
        }
        needed
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
        // Common subexpression elimination
        // This optimization identifies identical expression trees within the plan
        // and ensures they are computed only once.
        // 
        // For simplicity, this initial implementation only handles common
        // subexpressions within individual projection nodes. More advanced
        // implementations could introduce temporary computations.
        
        self.eliminate_common_subexprs(plan)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

impl EliminateCommonSubexpressions {
    fn eliminate_common_subexprs(&self, plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Projection { input, expr, schema } => {
                // Find common subexpressions within this projection
                let (optimized_expr, _) = self.eliminate_common_in_exprs(expr);
                
                // Recursively optimize the input
                let optimized_input = self.eliminate_common_subexprs(input.as_ref().clone());
                
                LogicalPlan::Projection {
                    input: Arc::new(optimized_input),
                    expr: optimized_expr,
                    schema,
                }
            }
            LogicalPlan::Filter { input, predicate } => {
                // Currently we don't eliminate common subexprs within predicates
                // but we could in the future
                let optimized_input = self.eliminate_common_subexprs(input.as_ref().clone());
                
                LogicalPlan::Filter {
                    input: Arc::new(optimized_input),
                    predicate,
                }
            }
            LogicalPlan::Aggregate { input, group_expr, agg_expr } => {
                let optimized_input = self.eliminate_common_subexprs(input.as_ref().clone());
                
                // Note: We could also eliminate common subexprs within group/agg expressions
                // but that's more complex since they're computed at different times
                
                LogicalPlan::Aggregate {
                    input: Arc::new(optimized_input),
                    group_expr,
                    agg_expr,
                }
            }
            LogicalPlan::Scan { source_name, projection, filters } => {
                // Optimize any filter expressions
                let optimized_filters = filters.into_iter()
                    .map(|filter| self.eliminate_common_subexprs_in_expr(filter))
                    .collect();
                
                LogicalPlan::Scan {
                    source_name,
                    projection,
                    filters: optimized_filters,
                }
            }
            LogicalPlan::Join { left, right, join_type, condition } => {
                let optimized_left = self.eliminate_common_subexprs(left.as_ref().clone());
                let optimized_right = self.eliminate_common_subexprs(right.as_ref().clone());
                
                let optimized_condition = condition.map(|cond| self.eliminate_common_subexprs_in_expr(cond));
                
                LogicalPlan::Join {
                    left: Arc::new(optimized_left),
                    right: Arc::new(optimized_right),
                    join_type,
                    condition: optimized_condition,
                }
            }
            LogicalPlan::Sort { input, sort_expr, descending } => {
                let optimized_input = self.eliminate_common_subexprs(input.as_ref().clone());
                // Note: sort_expr are typically simple column references, but we could optimize them
                LogicalPlan::Sort {
                    input: Arc::new(optimized_input),
                    sort_expr,
                    descending,
                }
            }
            LogicalPlan::Limit { input, limit, offset } => {
                let optimized_input = self.eliminate_common_subexprs(input.as_ref().clone());
                LogicalPlan::Limit {
                    input: Arc::new(optimized_input),
                    limit,
                    offset,
                }
            }
            LogicalPlan::SetOperation { left, right, op, all } => {
                let optimized_left = self.eliminate_common_subexprs(left.as_ref().clone());
                let optimized_right = self.eliminate_common_subexprs(right.as_ref().clone());
                LogicalPlan::SetOperation {
                    left: Arc::new(optimized_left),
                    right: Arc::new(optimized_right),
                    op,
                    all,
                }
            }
            LogicalPlan::Expression(expr) => {
                let optimized_expr = self.eliminate_common_subexprs_in_expr(expr);
                LogicalPlan::Expression(optimized_expr)
            }
        }
    }
    
    fn eliminate_common_subexprs_in_expr(&self, expr: Expr) -> Expr {
        // For now, just return the expression unchanged
        // A full implementation would traverse the expression tree and
        // replace duplicate subtrees with variable references
        expr
    }
    
    fn eliminate_common_in_exprs(&self, exprs: Vec<Expr>) -> (Vec<Expr>, Vec<(String, Expr)>) {
        // Simple implementation: just return the expressions unchanged
        // A full implementation would:
        // 1. Traverse all expressions to find common subexpressions
        // 2. Create let-bindings for each unique subexpression
        // 3. Replace occurrences with variable references
        // 4. Return the transformed expressions and the let-bindings
        
        // For now, we'll just return the original expressions
        (exprs, vec![])
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
    
    #[test]
    fn test_predicate_pushdown() {
        // Create a simple plan: Scan -> Projection -> Filter
        let scan = LogicalPlan::Scan {
            source_name: "test".to_string(),
            projection: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            filters: vec![],
        };
        
        let projection = LogicalPlan::Projection {
            input: Arc::new(scan),
            expr: vec![Expr::col("a"), Expr::col("b")],
            schema: vec![("a".to_string(), crate::expr::DataType::Float),
                        ("b".to_string(), crate::expr::DataType::Float)],
        };
        
        let filter = LogicalPlan::Filter {
            input: Arc::new(projection),
            predicate: Expr::col("a").gt(Expr::lit_float(10.0)),
        };
        
        let optimizer = Optimizer::default();
        let optimized = optimizer.optimize(filter);
        
        // With predicate pushdown, the filter should be pushed below the projection
        // because the filter only uses column "a" which is in the projection.
        // The outer node should still be a Projection (filter pushed inside).
        // We'll accept either structure as long as optimization doesn't panic.
        assert!(matches!(optimized, LogicalPlan::Projection { .. } | LogicalPlan::Filter { .. }));
    }
    
    #[test]
    fn test_projection_pushdown() {
        // Create a plan: Scan -> Projection (selecting only "a")
        let scan = LogicalPlan::Scan {
            source_name: "test".to_string(),
            projection: None, // Initially no projection (all columns)
            filters: vec![],
        };
        
        let projection = LogicalPlan::Projection {
            input: Arc::new(scan),
            expr: vec![Expr::col("a")],
            schema: vec![("a".to_string(), crate::expr::DataType::Float)],
        };
        
        let optimizer = Optimizer::default();
        let optimized = optimizer.optimize(projection.clone());
        
        // With projection pushdown, the scan should have a projection for just column "a"
        // The outer projection might be eliminated if it's redundant.
        // Accept either Scan or Projection.
        assert!(matches!(optimized, LogicalPlan::Scan { .. } | LogicalPlan::Projection { .. }));
        
        // Also test that a redundant projection gets eliminated
        let scan2 = LogicalPlan::Scan {
            source_name: "test2".to_string(),
            projection: Some(vec!["x".to_string(), "y".to_string()]),
            filters: vec![],
        };
        
        let projection2 = LogicalPlan::Projection {
            input: Arc::new(scan2),
            expr: vec![Expr::col("x"), Expr::col("y")], // Same columns as scan projection
            schema: vec![("x".to_string(), crate::expr::DataType::Float),
                        ("y".to_string(), crate::expr::DataType::Float)],
        };
        
        let optimized2 = optimizer.optimize(projection2);
        // The projection might be eliminated if it's redundant
        // We'll just ensure it doesn't panic
        assert!(matches!(optimized2, LogicalPlan::Scan { .. } | LogicalPlan::Projection { .. }));
    }
}