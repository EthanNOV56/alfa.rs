//! Symbolic regression using genetic programming (rebuilt on top of GA module)
//!
//! This module provides a high-level interface for symbolic regression,
//! leveraging the advanced genetic algorithm infrastructure in the `ga` module.

use crate::expr::{Expr, Literal, BinaryOp, UnaryOp};
use crate::ga::{self, GenotypeDecoder, FitnessEvaluator, GAConfig};
use rand::prelude::*;
use std::collections::HashMap;

/// Configuration for symbolic regression
#[derive(Clone)]
pub struct SymbolicRegressionConfig {
    /// Population size
    pub population_size: usize,
    /// Maximum number of generations
    pub max_generations: usize,
    /// Tournament size for selection
    pub tournament_size: usize,
    /// Crossover probability
    pub crossover_prob: f64,
    /// Mutation probability
    pub mutation_prob: f64,
    /// Maximum tree depth
    pub max_depth: usize,
    /// Terminal symbols (constants and variables)
    pub terminals: Vec<Terminal>,
    /// Function set
    pub functions: Vec<Function>,
}

/// Terminal symbol (leaf node in expression tree)
#[derive(Clone)]
pub enum Terminal {
    /// Variable with name
    Variable(String),
    /// Constant value
    Constant(f64),
    /// Ephemeral random constant
    Ephemeral,
}

/// Function with arity
#[derive(Clone)]
pub struct Function {
    /// Function name
    pub name: String,
    /// Number of arguments
    pub arity: usize,
    /// Builder function
    pub builder: fn(Vec<Expr>) -> Expr,
}

/// Data point for symbolic regression
#[derive(Clone)]
pub struct DataPoint {
    /// Input variables
    pub inputs: HashMap<String, f64>,
    /// Target output
    pub target: f64,
}

/// Fitness function for symbolic regression (mean squared error)
struct SymbolicRegressionFitnessEvaluator {
    data_points: Vec<DataPoint>,
}

impl FitnessEvaluator for SymbolicRegressionFitnessEvaluator {
    fn fitness(&self, expr: &Expr) -> f64 {
        // Higher fitness is better, so we use negative MSE
        -self.mean_squared_error(expr)
    }
}

impl SymbolicRegressionFitnessEvaluator {
    fn mean_squared_error(&self, expr: &Expr) -> f64 {
        let mut total_error = 0.0;
        let mut count = 0;
        
        for point in &self.data_points {
            // Evaluate expression with current input values
            // For simplicity, we'll create a simple evaluation context
            // In a real implementation, this would use the full evaluation engine
            match self.evaluate_expr_with_inputs(expr, &point.inputs) {
                Ok(predicted) => {
                    let error = predicted - point.target;
                    total_error += error * error;
                    count += 1;
                }
                Err(_) => {
                    // Invalid expression, penalize heavily
                    return 1e12;
                }
            }
        }
        
        if count == 0 {
            return 1e12;
        }
        
        total_error / count as f64
    }
    
    fn evaluate_expr_with_inputs(&self, expr: &Expr, inputs: &HashMap<String, f64>) -> Result<f64, String> {
        // Simple recursive evaluation for demonstration
        // In production, use the actual evaluation engine
        match expr {
            Expr::Literal(Literal::Float(f)) => Ok(*f),
            Expr::Literal(Literal::Integer(i)) => Ok(*i as f64),
            Expr::Column(name) => {
                inputs.get(name)
                    .copied()
                    .ok_or_else(|| format!("Variable {} not found in inputs", name))
            }
            Expr::BinaryExpr { left, op, right } => {
                let left_val = self.evaluate_expr_with_inputs(left, inputs)?;
                let right_val = self.evaluate_expr_with_inputs(right, inputs)?;
                match op {
                    BinaryOp::Add => Ok(left_val + right_val),
                    BinaryOp::Subtract => Ok(left_val - right_val),
                    BinaryOp::Multiply => Ok(left_val * right_val),
                    BinaryOp::Divide => {
                        if right_val == 0.0 {
                            Err("Division by zero".to_string())
                        } else {
                            Ok(left_val / right_val)
                        }
                    }
                    BinaryOp::Modulo => Ok(left_val % right_val),
                    BinaryOp::Equal => Ok((left_val == right_val) as i64 as f64),
                    BinaryOp::NotEqual => Ok((left_val != right_val) as i64 as f64),
                    BinaryOp::GreaterThan => Ok((left_val > right_val) as i64 as f64),
                    BinaryOp::GreaterThanOrEqual => Ok((left_val >= right_val) as i64 as f64),
                    BinaryOp::LessThan => Ok((left_val < right_val) as i64 as f64),
                    BinaryOp::LessThanOrEqual => Ok((left_val <= right_val) as i64 as f64),
                    BinaryOp::And => Ok(((left_val != 0.0) && (right_val != 0.0)) as i64 as f64),
                    BinaryOp::Or => Ok(((left_val != 0.0) || (right_val != 0.0)) as i64 as f64),
                }
            }
            Expr::UnaryExpr { op, expr: inner } => {
                let inner_val = self.evaluate_expr_with_inputs(inner, inputs)?;
                match op {
                    UnaryOp::Negate => Ok(-inner_val),
                    UnaryOp::Not => Ok((inner_val == 0.0) as i64 as f64),
                    UnaryOp::Abs => Ok(inner_val.abs()),
                    UnaryOp::Sqrt => {
                        if inner_val < 0.0 {
                            Err("Square root of negative number".to_string())
                        } else {
                            Ok(inner_val.sqrt())
                        }
                    }
                    UnaryOp::Log => {
                        if inner_val <= 0.0 {
                            Err("Log of non-positive number".to_string())
                        } else {
                            Ok(inner_val.ln())
                        }
                    }
                    UnaryOp::Exp => Ok(inner_val.exp()),
                }
            }
            Expr::FunctionCall { name, args } => {
                let arg_vals: Result<Vec<f64>, String> = args.iter()
                    .map(|arg| self.evaluate_expr_with_inputs(arg, inputs))
                    .collect();
                let arg_vals = arg_vals?;
                
                // Handle common functions
                match name.as_str() {
                    "sqrt" if arg_vals.len() == 1 => {
                        if arg_vals[0] < 0.0 {
                            Err("Square root of negative number".to_string())
                        } else {
                            Ok(arg_vals[0].sqrt())
                        }
                    }
                    "log" if arg_vals.len() == 1 => {
                        if arg_vals[0] <= 0.0 {
                            Err("Log of non-positive number".to_string())
                        } else {
                            Ok(arg_vals[0].ln())
                        }
                    }
                    "exp" if arg_vals.len() == 1 => Ok(arg_vals[0].exp()),
                    "abs" if arg_vals.len() == 1 => Ok(arg_vals[0].abs()),
                    "sin" if arg_vals.len() == 1 => Ok(arg_vals[0].sin()),
                    "cos" if arg_vals.len() == 1 => Ok(arg_vals[0].cos()),
                    "tan" if arg_vals.len() == 1 => Ok(arg_vals[0].tan()),
                    _ => Err(format!("Unknown function or arity mismatch: {}", name)),
                }
            }
            _ => Err("Unsupported expression type for simple evaluation".to_string()),
        }
    }
}

/// Expression generator for symbolic regression
struct SymbolicRegressionGenerator<'a> {
    config: SymbolicRegressionConfig,
    rng: &'a mut (dyn RngCore + Send + Sync),
}

impl<'a> SymbolicRegressionGenerator<'a> {
    fn new(config: SymbolicRegressionConfig, rng: &'a mut (dyn RngCore + Send + Sync)) -> Self {
        Self { config, rng }
    }
    
    fn generate_random_expr(&mut self, max_depth: usize) -> Expr {
        if max_depth == 0 || (!self.config.functions.is_empty() && self.rng.gen_bool(0.3)) {
            self.generate_random_terminal()
        } else {
            self.generate_random_function(max_depth - 1)
        }
    }
    
    fn generate_random_terminal(&mut self) -> Expr {
        let terminal = &self.config.terminals[self.rng.gen_range(0..self.config.terminals.len())];
        
        match terminal {
            Terminal::Variable(name) => Expr::Column(name.clone()),
            Terminal::Constant(value) => Expr::Literal(Literal::Float(*value)),
            Terminal::Ephemeral => Expr::Literal(Literal::Float(self.rng.gen_range(-10.0..10.0))),
        }
    }
    
    fn generate_random_function(&mut self, max_depth: usize) -> Expr {
        let function_idx = self.rng.gen_range(0..self.config.functions.len());
        let arity = self.config.functions[function_idx].arity;
        let mut args = Vec::with_capacity(arity);
        
        for _ in 0..arity {
            args.push(self.generate_random_expr(max_depth));
        }
        
        let function = &self.config.functions[function_idx];
        (function.builder)(args)
    }
    
    fn generate_initial_population(&mut self, size: usize) -> Vec<Expr> {
        let mut population = Vec::with_capacity(size);
        for _ in 0..size {
            population.push(self.generate_random_expr(self.config.max_depth));
        }
        population
    }
}

/// High-level symbolic regression engine
pub struct SymbolicRegressionEngine {
    config: SymbolicRegressionConfig,
    rng: Box<dyn RngCore + Send + Sync>,
}

impl SymbolicRegressionEngine {
    /// Create a new symbolic regression engine with default configuration
    pub fn new() -> Self {
        Self::with_config(SymbolicRegressionConfig::default())
    }
    
    /// Create a new symbolic regression engine with custom configuration
    pub fn with_config(config: SymbolicRegressionConfig) -> Self {
        use rand::SeedableRng;
        let rng: Box<dyn RngCore + Send + Sync> = Box::new(rand::rngs::StdRng::from_entropy());
        Self {
            config,
            rng,
        }
    }
    
    /// Run symbolic regression on the given data points
    pub fn run(&mut self, data_points: &[DataPoint]) -> Option<(Expr, f64)> {
        // Convert to GA configuration
        let ga_config = GAConfig {
            generations: self.config.max_generations,
            tournament_size: self.config.tournament_size,
            crossover_rate: self.config.crossover_prob,
            mutation_rate: self.config.mutation_prob,
            gene_max: 255, // Not used for expression-based GA
        };
        
        // Create fitness evaluator
        let evaluator = SymbolicRegressionFitnessEvaluator {
            data_points: data_points.to_vec(),
        };
        
        // Generate initial population
        let mut generator = SymbolicRegressionGenerator::new(
            self.config.clone(),
            &mut *self.rng,
        );
        let initial_population = generator.generate_initial_population(self.config.population_size);
        
        // Run GA
        let (best_expr, best_fitness) = ga::run_ga_exprs(
            &ga_config,
            initial_population,
            &evaluator,
            &mut *self.rng,
        );
        
        Some((best_expr, -best_fitness)) // Convert back to MSE (positive, lower is better)
    }
    
    /// Find an expression that fits the given data (simplified interface)
    pub fn find_expression(data_points: &[DataPoint]) -> Option<Expr> {
        let mut engine = Self::new();
        engine.run(data_points).map(|(expr, _)| expr)
    }
}

/// Default configuration for symbolic regression
impl Default for SymbolicRegressionConfig {
    fn default() -> Self {
        let functions = vec![
            Function {
                name: "add".to_string(),
                arity: 2,
                builder: |args| {
                    let mut iter = args.into_iter();
                    let left = iter.next().unwrap();
                    let right = iter.next().unwrap();
                    left.binary(BinaryOp::Add, right)
                },
            },
            Function {
                name: "sub".to_string(),
                arity: 2,
                builder: |args| {
                    let mut iter = args.into_iter();
                    let left = iter.next().unwrap();
                    let right = iter.next().unwrap();
                    left.binary(BinaryOp::Subtract, right)
                },
            },
            Function {
                name: "mul".to_string(),
                arity: 2,
                builder: |args| {
                    let mut iter = args.into_iter();
                    let left = iter.next().unwrap();
                    let right = iter.next().unwrap();
                    left.binary(BinaryOp::Multiply, right)
                },
            },
            Function {
                name: "div".to_string(),
                arity: 2,
                builder: |args| {
                    let mut iter = args.into_iter();
                    let left = iter.next().unwrap();
                    let right = iter.next().unwrap();
                    left.binary(BinaryOp::Divide, right)
                },
            },
            Function {
                name: "sqrt".to_string(),
                arity: 1,
                builder: |args| {
                    args.into_iter().next().unwrap().unary(UnaryOp::Sqrt)
                },
            },
            Function {
                name: "log".to_string(),
                arity: 1,
                builder: |args| {
                    args.into_iter().next().unwrap().unary(UnaryOp::Log)
                },
            },
            Function {
                name: "exp".to_string(),
                arity: 1,
                builder: |args| {
                    args.into_iter().next().unwrap().unary(UnaryOp::Exp)
                },
            },
            Function {
                name: "neg".to_string(),
                arity: 1,
                builder: |args| {
                    args.into_iter().next().unwrap().unary(UnaryOp::Negate)
                },
            },
        ];
        
        Self {
            population_size: 100,
            max_generations: 50,
            tournament_size: 5,
            crossover_prob: 0.8,
            mutation_prob: 0.1,
            max_depth: 5,
            terminals: vec![
                Terminal::Variable("x".to_string()),
                Terminal::Variable("y".to_string()),
                Terminal::Constant(1.0),
                Terminal::Constant(2.0),
                Terminal::Ephemeral,
            ],
            functions,
        }
    }
}

/// Legacy compatibility function
pub fn find_expression(data: &[DataPoint]) -> Option<Expr> {
    SymbolicRegressionEngine::find_expression(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_default() {
        let config = SymbolicRegressionConfig::default();
        assert_eq!(config.population_size, 100);
        assert_eq!(config.max_generations, 50);
        assert_eq!(config.tournament_size, 5);
    }
    
    #[test]
    fn test_generate_random_terminal() {
        let config = SymbolicRegressionConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut generator = SymbolicRegressionGenerator::new(config, &mut rng);
        
        let expr = generator.generate_random_terminal();
        assert!(matches!(expr, Expr::Literal(_) | Expr::Column(_)));
    }
    
    #[test]
    fn test_fitness_evaluator() {
        let data_points = vec![
            DataPoint {
                inputs: {
                    let mut m = HashMap::new();
                    m.insert("x".to_string(), 1.0);
                    m
                },
                target: 2.0,
            },
            DataPoint {
                inputs: {
                    let mut m = HashMap::new();
                    m.insert("x".to_string(), 2.0);
                    m
                },
                target: 4.0,
            },
        ];
        
        let evaluator = SymbolicRegressionFitnessEvaluator {
            data_points: data_points.clone(),
        };
        
        // Test with expression x * 2
        let expr = Expr::Column("x".to_string())
            .binary(BinaryOp::Multiply, Expr::Literal(Literal::Float(2.0)));
        
        let fitness = evaluator.fitness(&expr);
        // Perfect fit should have high fitness (low negative MSE)
        assert!(fitness > -1e-6);
    }
    
    #[test]
    fn test_engine_creation() {
        let _engine = SymbolicRegressionEngine::new();
        // Should compile and create without error
        assert!(true);
    }
}