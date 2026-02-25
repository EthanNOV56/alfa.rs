//! Alpha computation extensions for quantitative finance
//!
//! This module provides specialized expressions and operations for computing
//! alpha factors and other quantitative finance indicators.

use crate::expr::Expr;
use crate::evaluation::EvaluationContext;
use crate::timeseries::TimeSeries;

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
    pub fn momentum(&self, periods: i64) -> Expr {
        let base = self.base_expr.clone();
        // TODO: Implement proper time series lag operation
        // For now, create a placeholder expression
        Expr::function(
            "momentum",
            vec![base, Expr::lit_int(periods)]
        )
    }
    
    /// Create a moving average alpha factor
    pub fn moving_average(&self, window: i64) -> Expr {
        let base = self.base_expr.clone();
        Expr::function(
            "moving_average",
            vec![base, Expr::lit_int(window)]
        )
    }
    
    /// Create a volatility alpha factor (standard deviation over N periods)
    pub fn volatility(&self, periods: i64) -> Expr {
        let base = self.base_expr.clone();
        Expr::function(
            "volatility",
            vec![base, Expr::lit_int(periods)]
        )
    }
    
    /// Create a Sharpe ratio alpha factor
    pub fn sharpe_ratio(&self, returns_expr: Expr, risk_free_expr: Expr, periods: i64) -> Expr {
        Expr::function(
            "sharpe_ratio",
            vec![returns_expr, risk_free_expr, Expr::lit_int(periods)]
        )
    }
    
    /// Create a correlation alpha factor between two series
    pub fn correlation(&self, other: Expr, periods: i64) -> Expr {
        let base = self.base_expr.clone();
        Expr::function(
            "correlation",
            vec![base, other, Expr::lit_int(periods)]
        )
    }
    
    /// Create a beta alpha factor (relative to market)
    pub fn beta(&self, market_expr: Expr, periods: i64) -> Expr {
        let base = self.base_expr.clone();
        Expr::function(
            "beta",
            vec![base, market_expr, Expr::lit_int(periods)]
        )
    }
    
    /// Create an R-squared alpha factor
    pub fn r_squared(&self, market_expr: Expr, periods: i64) -> Expr {
        let base = self.base_expr.clone();
        Expr::function(
            "r_squared",
            vec![base, market_expr, Expr::lit_int(periods)]
        )
    }
    
    /// Create a maximum drawdown alpha factor
    pub fn max_drawdown(&self, periods: i64) -> Expr {
        let base = self.base_expr.clone();
        Expr::function(
            "max_drawdown",
            vec![base, Expr::lit_int(periods)]
        )
    }
    
    /// Create a value at risk (VaR) alpha factor
    pub fn var(&self, confidence_level: f64, periods: i64) -> Expr {
        let base = self.base_expr.clone();
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

/// Actual computation of alpha factors using time series data
pub mod compute {
    use super::TimeSeries;
    use ndarray;
    
    /// Compute momentum factor: price change over N periods
    pub fn momentum(series: &TimeSeries, periods: usize) -> TimeSeries {
        series.pct_change(periods)
    }
    
    /// Compute simple moving average
    pub fn moving_average(series: &TimeSeries, window: usize) -> TimeSeries {
        series.moving_average(window)
    }
    
    /// Compute volatility (standard deviation) over window
    pub fn volatility(series: &TimeSeries, window: usize) -> TimeSeries {
        series.rolling_std(window)
    }
    
    /// Compute Sharpe ratio (annualized)
    pub fn sharpe_ratio(series: &TimeSeries, risk_free_rate: f64, window: usize) -> TimeSeries {
        series.sharpe_ratio(risk_free_rate, window)
    }
    
    /// Compute correlation between two series
    pub fn correlation(series1: &TimeSeries, series2: &TimeSeries, window: usize) -> TimeSeries {
        series1.correlation(series2, window)
    }
    
    /// Compute beta (relative to market)
    pub fn beta(series: &TimeSeries, market: &TimeSeries, window: usize) -> TimeSeries {
        series.beta(market, window)
    }
    
    /// Compute R-squared (coefficient of determination)
    pub fn r_squared(series: &TimeSeries, market: &TimeSeries, window: usize) -> TimeSeries {
        let correlation_series = series.correlation(market, window);
        // R² = correlation²
        let r_sq_data = correlation_series.data().mapv(|x| x * x);
        TimeSeries::from_array(r_sq_data)
    }
    
    /// Compute maximum drawdown
    pub fn max_drawdown(series: &TimeSeries, window: usize) -> TimeSeries {
        series.max_drawdown(window)
    }
    
    /// Compute value at risk (VaR) using historical simulation
    pub fn var(series: &TimeSeries, confidence_level: f64, window: usize) -> TimeSeries {
        let returns = series.pct_change(1);
        let n = returns.len();
        let mut result = ndarray::Array1::zeros(n);
        
        for i in 0..n {
            let start = if i + 1 >= window { i + 1 - window } else { 0 };
            let slice = returns.data().slice(ndarray::s![start..=i]);
            
            if slice.len() == 0 {
                result[i] = f64::NAN;
                continue;
            }
            
            // Sort returns and find VaR
            let mut sorted: Vec<f64> = slice.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let index = ((1.0 - confidence_level) * sorted.len() as f64).floor() as usize;
            result[i] = sorted.get(index).copied().unwrap_or(f64::NAN);
        }
        
        TimeSeries::from_array(result)
    }
    
    /// Compute low volatility factor (inverse volatility)
    pub fn low_volatility(series: &TimeSeries, window: usize) -> TimeSeries {
        let vol = series.rolling_std(window);
        let inv_vol_data = vol.data().mapv(|x| 1.0 / x);
        TimeSeries::from_array(inv_vol_data)
    }
    
    /// Compute Z-score normalization
    pub fn z_score(series: &TimeSeries, window: usize) -> TimeSeries {
        let mean = series.moving_average(window);
        let std = series.rolling_std(window);
        let z_data = (series.data() - mean.data()) / std.data();
        TimeSeries::from_array(z_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;
    
    #[test]
    fn test_alpha_builder() {
        let price = Expr::col("price");
        
        // Each method consumes the builder, so we need to recreate it
        let momentum = price.clone().alpha().momentum(20);
        assert!(matches!(momentum, Expr::FunctionCall { .. }));
        
        let volatility = price.clone().alpha().volatility(30);
        assert!(matches!(volatility, Expr::FunctionCall { .. }));
        
        let moving_avg = price.clone().alpha().moving_average(50);
        assert!(matches!(moving_avg, Expr::FunctionCall { .. }));
    }
    
    #[test]
    fn test_timeseries_operations() {
        let price = Expr::col("price");
        let lagged = timeseries::lag(price.clone(), 1);
        let diff = timeseries::diff(price.clone(), 1);
        let pct_change = timeseries::pct_change(price.clone(), 1);
        let rolling = timeseries::rolling(price.clone(), 10, "mean");
        let ema = timeseries::ema(price.clone(), 20);
        
        assert!(matches!(lagged, Expr::FunctionCall { .. }));
        assert!(matches!(diff, Expr::BinaryExpr { .. }));
        assert!(matches!(pct_change, Expr::BinaryExpr { .. }));
        assert!(matches!(rolling, Expr::FunctionCall { .. }));
        assert!(matches!(ema, Expr::FunctionCall { .. }));
    }
    
    #[test]
    fn test_predefined_factors() {
        let price = Expr::col("price");
        let momentum = factors::momentum(price.clone(), 20, 10);
        assert!(matches!(momentum, Expr::FunctionCall { .. }));
        
        let book_value = Expr::col("book_value");
        let market_value = Expr::col("market_value");
        let value = factors::value(book_value, market_value);
        assert!(matches!(value, Expr::BinaryExpr { .. }));
        
        let market_cap = Expr::col("market_cap");
        let size = factors::size(market_cap);
        assert!(matches!(size, Expr::FunctionCall { .. }));
        
        let returns = Expr::col("returns");
        let low_vol = factors::low_volatility(returns.clone(), 30);
        assert!(matches!(low_vol, Expr::BinaryExpr { .. }));
        
        let profitability = Expr::col("profitability");
        let growth = Expr::col("growth");
        let safety = Expr::col("safety");
        let quality = factors::quality(profitability, growth, safety);
        assert!(matches!(quality, Expr::BinaryExpr { .. }));
    }
    
    #[test]
    fn test_z_score() {
        let expr = Expr::col("x");
        let z = factors::z_score(expr);
        // z = (x - mean) / std
        assert!(matches!(z, Expr::BinaryExpr { .. }));
    }
    
    #[test]
    fn test_alpha_evaluation_context() {
        let ctx = AlphaEvaluationContext::new();
        let expr = Expr::lit_float(42.0);
        match ctx.evaluate_alpha(&expr) {
            Ok(val) => assert!((val - 42.0).abs() < 1e-10),
            Err(e) => panic!("Expected evaluation to succeed, got {:?}", e),
        }
    }
}