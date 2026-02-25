//! Alpha computation extensions for quantitative finance
//!
//! This module provides specialized expressions and operations for computing
//! alpha factors and other quantitative finance indicators.

use crate::expr::Expr;
use crate::evaluation::EvaluationContext;

/// Alpha factor expression builder
pub struct AlphaBuilder {
    base_expr: Expr,
}

impl AlphaBuilder {
    /// Create a new alpha builder from a base expression
    pub fn new(base_expr: Expr) -> Self {
        Self { base_expr }
    }
    
    /// Create a momentum alpha factor (price change over N periods)
    pub fn momentum(self, periods: i64) -> Expr {
        let base = self.base_expr;
        // TODO: Implement proper time series lag operation
        // For now, create a placeholder expression
        Expr::function(
            "momentum",
            vec![base, Expr::lit_int(periods)]
        )
    }
    
    /// Create a moving average alpha factor
    pub fn moving_average(self, window: i64) -> Expr {
        let base = self.base_expr;
        Expr::function(
            "moving_average",
            vec![base, Expr::lit_int(window)]
        )
    }
    
    /// Create a volatility alpha factor (standard deviation over N periods)
    pub fn volatility(self, periods: i64) -> Expr {
        let base = self.base_expr;
        Expr::function(
            "volatility",
            vec![base, Expr::lit_int(periods)]
        )
    }
    
    /// Create a Sharpe ratio alpha factor
    pub fn sharpe_ratio(self, returns_expr: Expr, risk_free_expr: Expr, periods: i64) -> Expr {
        Expr::function(
            "sharpe_ratio",
            vec![returns_expr, risk_free_expr, Expr::lit_int(periods)]
        )
    }
    
    /// Create a correlation alpha factor between two series
    pub fn correlation(self, other: Expr, periods: i64) -> Expr {
        let base = self.base_expr;
        Expr::function(
            "correlation",
            vec![base, other, Expr::lit_int(periods)]
        )
    }
    
    /// Create a beta alpha factor (relative to market)
    pub fn beta(self, market_expr: Expr, periods: i64) -> Expr {
        let base = self.base_expr;
        Expr::function(
            "beta",
            vec![base, market_expr, Expr::lit_int(periods)]
        )
    }
    
    /// Create an R-squared alpha factor
    pub fn r_squared(self, market_expr: Expr, periods: i64) -> Expr {
        let base = self.base_expr;
        Expr::function(
            "r_squared",
            vec![base, market_expr, Expr::lit_int(periods)]
        )
    }
    
    /// Create a maximum drawdown alpha factor
    pub fn max_drawdown(self, periods: i64) -> Expr {
        let base = self.base_expr;
        Expr::function(
            "max_drawdown",
            vec![base, Expr::lit_int(periods)]
        )
    }
    
    /// Create a value at risk (VaR) alpha factor
    pub fn var(self, confidence_level: f64, periods: i64) -> Expr {
        let base = self.base_expr;
        Expr::function(
            "var",
            vec![base, Expr::lit_float(confidence_level), Expr::lit_int(periods)]
        )
    }
}

/// Extension trait for adding alpha methods to Expr
pub trait AlphaExprExt {
    /// Convert an expression to an alpha builder
    fn alpha(self) -> AlphaBuilder;
}

impl AlphaExprExt for Expr {
    fn alpha(self) -> AlphaBuilder {
        AlphaBuilder::new(self)
    }
}

/// Time series operations for alpha computation
pub mod timeseries {
    use super::*;
    
    /// Lag operation: shift series back by N periods
    pub fn lag(expr: Expr, periods: i64) -> Expr {
        Expr::function("lag", vec![expr, Expr::lit_int(periods)])
    }
    
    /// Difference operation: compute change over N periods
    pub fn diff(expr: Expr, periods: i64) -> Expr {
        let lagged = lag(expr.clone(), periods);
        expr - lagged
    }
    
    /// Percentage change operation
    pub fn pct_change(expr: Expr, periods: i64) -> Expr {
        let diff_expr = diff(expr.clone(), periods);
        let lagged = lag(expr, periods);
        diff_expr / lagged * Expr::lit_float(100.0)
    }
    
    /// Rolling window operation
    pub fn rolling(expr: Expr, window: i64, func: &str) -> Expr {
        Expr::function(
            &format!("rolling_{}", func),
            vec![expr, Expr::lit_int(window)]
        )
    }
    
    /// Exponential moving average
    pub fn ema(expr: Expr, span: i64) -> Expr {
        Expr::function("ema", vec![expr, Expr::lit_int(span)])
    }
}

/// Predefined alpha factors from academic literature
pub mod factors {
    use super::*;
    
    /// Momentum factor (Jegadeesh and Titman, 1993)
    pub fn momentum(price_expr: Expr, formation_period: i64, holding_period: i64) -> Expr {
        let ret = timeseries::pct_change(price_expr, formation_period);
        timeseries::rolling(ret, holding_period, "mean")
    }
    
    /// Value factor (Fama and French, 1992)
    pub fn value(book_value_expr: Expr, market_value_expr: Expr) -> Expr {
        book_value_expr / market_value_expr
    }
    
    /// Size factor (Fama and French, 1992)
    pub fn size(market_cap_expr: Expr) -> Expr {
        Expr::function("log", vec![market_cap_expr])
    }
    
    /// Quality factor (Asness et al., 2013)
    pub fn quality(profitability_expr: Expr, growth_expr: Expr, safety_expr: Expr) -> Expr {
        let z_profit = z_score(profitability_expr);
        let z_growth = z_score(growth_expr);
        let z_safety = z_score(safety_expr);
        (z_profit + z_growth + z_safety) / Expr::lit_float(3.0)
    }
    
    /// Low volatility factor (Ang et al., 2006)
    pub fn low_volatility(returns_expr: Expr, window: i64) -> Expr {
        let volatility = returns_expr.clone().alpha().volatility(window);
        Expr::lit_float(1.0) / volatility
    }
    
    /// Z-score normalization
    pub fn z_score(expr: Expr) -> Expr {
        let mean = expr.clone().alpha().moving_average(20);
        let std = expr.clone().alpha().volatility(20);
        (expr - mean) / std
    }
}

/// Alpha factor evaluation context with time series support
pub struct AlphaEvaluationContext {
    base_context: EvaluationContext,
    // TODO: Add time series data storage
}

impl AlphaEvaluationContext {
    /// Create a new alpha evaluation context
    pub fn new() -> Self {
        Self {
            base_context: EvaluationContext::new(),
        }
    }
    
    /// Evaluate an alpha expression
    pub fn evaluate_alpha(&self, expr: &Expr) -> Result<f64, crate::evaluation::EvaluationError> {
        // For now, delegate to base evaluation
        let result = self.base_context.evaluate(expr)?;
        match result {
            crate::expr::Literal::Float(f) => Ok(f),
            crate::expr::Literal::Integer(i) => Ok(i as f64),
            _ => Err(crate::evaluation::EvaluationError::InvalidOperation(
                "Alpha evaluation requires numeric result".to_string()
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;
    
    #[test]
    fn test_alpha_builder() {
        let price = Expr::col("price");
        let alpha = price.clone().alpha();
        let momentum = alpha.momentum(20);
        
        assert!(matches!(momentum, Expr::FunctionCall { .. }));
    }
    
    #[test]
    fn test_timeseries_operations() {
        let price = Expr::col("price");
        let lagged = timeseries::lag(price.clone(), 1);
        let diff = timeseries::diff(price.clone(), 1);
        
        assert!(matches!(lagged, Expr::FunctionCall { .. }));
        assert!(matches!(diff, Expr::BinaryExpr { .. }));
    }
    
    #[test]
    fn test_predefined_factors() {
        let price = Expr::col("price");
        let momentum = factors::momentum(price, 20, 10);
        
        assert!(matches!(momentum, Expr::FunctionCall { .. }));
    }
}