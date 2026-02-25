//! Symbolic regression using genetic programming
//!
//! This module provides functionality to discover mathematical expressions
//! that fit given data using evolutionary algorithms.

use rand::Rng;
use std::collections::HashMap;
use crate::expr::{Expr, Literal};

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
    /// Fitness function
    pub fitness_fn: FitnessFunction,
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

/// Fitness function type
#[derive(Clone)]
pub enum FitnessFunction {
    /// Mean squared error
    MSE,
    /// Mean absolute error
    MAE,
    /// R-squared
    RSquared,
    /// Custom fitness function
    Custom(fn(&Expr, &[DataPoint]) -> f64),
}

/// Data point for symbolic regression
#[derive(Clone)]
pub struct DataPoint {
    /// Input variables
    pub inputs: HashMap<String, f64>,
    /// Target output
    pub target: f64,
}

/// An individual in the population
#[derive(Clone)]
pub struct Individual {
    /// Expression tree
    pub expr: Expr,
    /// Fitness score (lower is better)
    pub fitness: f64,
    /// Depth of the expression tree
    pub depth: usize,
}

/// Symbolic regression engine
pub struct SymbolicRegression {
    config: SymbolicRegressionConfig,
    population: Vec<Individual>,
    rng: rand::rngs::ThreadRng,
    best_individual: Option<Individual>,
    generation: usize,
}

impl SymbolicRegression {
    /// Create a new symbolic regression engine
    pub fn new(config: SymbolicRegressionConfig) -> Self {
        Self {
            config,
            population: vec![],
            rng: rand::thread_rng(),
            best_individual: None,
            generation: 0,
        }
    }
    
    /// Initialize the population with random expressions
    pub fn initialize_population(&mut self) {
        self.population.clear();
        
        for _ in 0..self.config.population_size {
            let expr = self.generate_random_expr(self.config.max_depth, true);
            let fitness = self.evaluate_fitness(&expr);
            let depth = self.calculate_depth(&expr);
            
            self.population.push(Individual {
                expr,
                fitness,
                depth,
            });
        }
        
        self.update_best_individual();
    }
    
    /// Run the evolutionary algorithm
    pub fn run(&mut self, _data: &[DataPoint]) -> Option<&Individual> {
        self.initialize_population();
        
        for generation in 0..self.config.max_generations {
            self.generation = generation;
            
            // Create new population
            let mut new_population = Vec::with_capacity(self.config.population_size);
            
            // Elitism: keep the best individual
            if let Some(best) = self.best_individual.clone() {
                new_population.push(best);
            }
            
            // Fill the rest of the population
            while new_population.len() < self.config.population_size {
                let parent1 = self.tournament_selection();
                let parent2 = self.tournament_selection();
                
                let (child1, child2) = if self.rng.gen_range(0.0..1.0) < self.config.crossover_prob {
                    self.crossover(&parent1.expr, &parent2.expr)
                } else {
                    (parent1.expr.clone(), parent2.expr.clone())
                };
                
                let child1 = if self.rng.gen_range(0.0..1.0) < self.config.mutation_prob {
                    self.mutate(&child1)
                } else {
                    child1.clone()
                };
                
                let child2 = if self.rng.gen_range(0.0..1.0) < self.config.mutation_prob {
                    self.mutate(&child2)
                } else {
                    child2.clone()
                };
                
                let fitness1 = self.evaluate_fitness(&child1);
                let fitness2 = self.evaluate_fitness(&child2);
                let depth1 = self.calculate_depth(&child1);
                let depth2 = self.calculate_depth(&child2);
                
                new_population.push(Individual {
                    expr: child1,
                    fitness: fitness1,
                    depth: depth1,
                });
                
                if new_population.len() < self.config.population_size {
                    new_population.push(Individual {
                        expr: child2,
                        fitness: fitness2,
                        depth: depth2,
                    });
                }
            }
            
            self.population = new_population;
            self.update_best_individual();
            
            // Log progress
            if generation % 10 == 0 {
                println!("Generation {}: best fitness = {:.6}", 
                    generation, 
                    self.best_individual.as_ref().unwrap().fitness
                );
            }
        }
        
        self.best_individual.as_ref()
    }
    
    /// Generate a random expression
    fn generate_random_expr(&mut self, max_depth: usize, allow_functions: bool) -> Expr {
        if max_depth == 0 || (!allow_functions && self.rng.gen_range(0.0..1.0) < 0.7) {
            // Generate terminal
            self.generate_random_terminal()
        } else {
            // Generate function
            let function_idx = self.rng.gen_range(0..self.config.functions.len());
            let function = self.config.functions[function_idx].clone();
            let mut args = Vec::with_capacity(function.arity);
            
            for _ in 0..function.arity {
                args.push(self.generate_random_expr(max_depth - 1, true));
            }
            
            (function.builder)(args)
        }
    }
    
    /// Generate a random terminal
    fn generate_random_terminal(&mut self) -> Expr {
        let terminal = &self.config.terminals[self.rng.gen_range(0..self.config.terminals.len())];
        
        match terminal {
            Terminal::Variable(name) => Expr::col(name.clone()),
            Terminal::Constant(value) => Expr::lit_float(*value),
            Terminal::Ephemeral => Expr::lit_float(self.rng.gen_range(-10.0..10.0)),
        }
    }
    
    /// Evaluate fitness of an expression
    fn evaluate_fitness(&self, _expr: &Expr) -> f64 {
        // TODO: Use actual data points
        // For now, return a placeholder
        0.0
    }
    
    /// Calculate depth of an expression
    fn calculate_depth(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Literal(_) | Expr::Column(_) => 1,
            Expr::BinaryExpr { left, right, .. } => {
                1 + self.calculate_depth(left).max(self.calculate_depth(right))
            }
            Expr::UnaryExpr { expr, .. } => 1 + self.calculate_depth(expr),
            Expr::FunctionCall { args, .. } => {
                1 + args.iter().map(|arg| self.calculate_depth(arg)).max().unwrap_or(0)
            }
            Expr::Aggregate { expr, .. } => 1 + self.calculate_depth(expr),
            Expr::Conditional { condition, then_expr, else_expr } => {
                1 + self.calculate_depth(condition)
                    .max(self.calculate_depth(then_expr))
                    .max(self.calculate_depth(else_expr))
            }
            Expr::Cast { expr, .. } => 1 + self.calculate_depth(expr),
        }
    }
    
    /// Tournament selection
    fn tournament_selection(&mut self) -> Individual {
        let mut best_idx = self.rng.gen_range(0..self.population.len());
        
        for _ in 1..self.config.tournament_size {
            let candidate_idx = self.rng.gen_range(0..self.population.len());
            if self.population[candidate_idx].fitness < self.population[best_idx].fitness {
                best_idx = candidate_idx;
            }
        }
        
        self.population[best_idx].clone()
    }
    
    /// Crossover two expressions
    fn crossover(&mut self, expr1: &Expr, expr2: &Expr) -> (Expr, Expr) {
        // TODO: Implement subtree crossover
        (expr1.clone(), expr2.clone())
    }
    
    /// Mutate an expression
    fn mutate(&mut self, expr: &Expr) -> Expr {
        // TODO: Implement mutation operators
        expr.clone()
    }
    
    /// Update the best individual found so far
    fn update_best_individual(&mut self) {
        let best = self.population.iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap();
        
        if self.best_individual.as_ref().map_or(true, |current| best.fitness < current.fitness) {
            self.best_individual = Some(best.clone());
        }
    }
}

/// Default configuration for symbolic regression
impl Default for SymbolicRegressionConfig {
    fn default() -> Self {
        use crate::expr::{BinaryOp, UnaryOp};
        
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
            fitness_fn: FitnessFunction::MSE,
        }
    }
}

/// Helper function to run symbolic regression on simple data
pub fn find_expression(data: &[DataPoint]) -> Option<Expr> {
    let config = SymbolicRegressionConfig::default();
    let mut engine = SymbolicRegression::new(config);
    
    engine.run(data).map(|ind| ind.expr.clone())
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
    fn test_random_expr_generation() {
        let config = SymbolicRegressionConfig::default();
        let mut engine = SymbolicRegression::new(config);
        let expr = engine.generate_random_expr(3, true);
        
        // Expression should be valid
        assert!(matches!(expr, Expr::Literal(_) | Expr::Column(_) | Expr::BinaryExpr { .. } | Expr::UnaryExpr { .. } | Expr::FunctionCall { .. }));
    }
}