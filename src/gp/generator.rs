//! Expression generator for creating random expression trees.
//!
//! Supports multiple generation strategies: uniform random, niche-biased,
//! and operator-feedback-guided generation.

use crate::data::frequency;
use crate::expr::{Expr, Literal};
use crate::gp::types::{Function, GPConfig, Niche, OperatorFeedback, Terminal};
use rand::Rng;

/// Expression generator
pub struct ExpressionGenerator<'a> {
    config: &'a GPConfig,
    terminals: Vec<Terminal>,
    functions: Vec<Function>,
}

impl<'a> ExpressionGenerator<'a> {
    /// Create a new expression generator
    pub fn new(config: &'a GPConfig, terminals: Vec<Terminal>, functions: Vec<Function>) -> Self {
        Self {
            config,
            terminals,
            functions,
        }
    }

    /// Generate a random expression (uniform distribution).
    pub fn generate_random_expr<R: Rng + ?Sized>(&self, max_depth: usize, rng: &mut R) -> Expr {
        if max_depth == 0 || (!self.functions.is_empty() && rng.gen_bool(0.3)) {
            self.generate_random_terminal(rng)
        } else {
            self.generate_random_function(max_depth - 1, rng)
        }
    }

    /// Generate expression using operator feedback weights for function selection.
    /// Falls back to uniform random if feedback is None or all weights are zero.
    pub(crate) fn generate_feedback_expr<R: Rng + ?Sized>(
        &self,
        feedback: &OperatorFeedback,
        max_depth: usize,
        rng: &mut R,
    ) -> Expr {
        if max_depth == 0 || (!self.functions.is_empty() && rng.gen_bool(0.3)) {
            self.generate_random_terminal(rng)
        } else {
            self.generate_feedback_function(feedback, max_depth - 1, rng)
        }
    }

    /// Generate a function with operator-feedback-biased selection.
    fn generate_feedback_function<R: Rng + ?Sized>(
        &self,
        feedback: &OperatorFeedback,
        max_depth: usize,
        rng: &mut R,
    ) -> Expr {
        // Compute weights: max(avg_fitness, 0.01) to keep all operators reachable
        let weights: Vec<f64> = self
            .functions
            .iter()
            .map(|f| feedback.avg_fitness(&f.name).max(0.01))
            .collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return self.generate_random_function(max_depth, rng);
        }
        let mut r = rng.gen_range(0.0..total);
        for (i, w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return self.build_function(i, max_depth, rng);
            }
        }
        self.build_function(0, max_depth, rng)
    }

    /// Generate an expression biased toward a niche.
    pub fn generate_niche_expr<R: Rng + ?Sized>(
        &self,
        niche: Niche,
        max_depth: usize,
        rng: &mut R,
    ) -> Expr {
        if max_depth == 0 || (!self.functions.is_empty() && rng.gen_bool(0.3)) {
            self.generate_weighted_terminal(niche, rng)
        } else {
            self.generate_weighted_function(niche, max_depth - 1, rng)
        }
    }

    /// Generate a random terminal (uniform).
    pub fn generate_random_terminal<R: Rng + ?Sized>(&self, rng: &mut R) -> Expr {
        let terminal = &self.terminals[rng.gen_range(0..self.terminals.len())];
        self.build_terminal(terminal, rng)
    }

    /// Generate a terminal with niche-biased weights.
    fn generate_weighted_terminal<R: Rng + ?Sized>(&self, niche: Niche, rng: &mut R) -> Expr {
        if niche == Niche::Mixed {
            return self.generate_random_terminal(rng);
        }
        let weights: Vec<f64> = self
            .terminals
            .iter()
            .map(|t| niche.terminal_weight(t))
            .collect();
        let total: f64 = weights.iter().sum();
        let mut r = rng.gen_range(0.0..total);
        for (i, w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return self.build_terminal(&self.terminals[i], rng);
            }
        }
        self.build_terminal(&self.terminals[0], rng)
    }

    fn build_terminal<R: Rng + ?Sized>(&self, terminal: &Terminal, rng: &mut R) -> Expr {
        match terminal {
            Terminal::Variable(name) => Expr::Column(name.clone()),
            Terminal::Constant(value) => Expr::Literal(Literal::Float(*value)),
            Terminal::Ephemeral => Expr::Literal(Literal::Float(rng.gen_range(-10.0..10.0))),
        }
    }

    /// Generate a random function (uniform).
    pub fn generate_random_function<R: Rng + ?Sized>(&self, max_depth: usize, rng: &mut R) -> Expr {
        let function_idx = rng.gen_range(0..self.functions.len());
        self.build_function(function_idx, max_depth, rng)
    }

    /// Generate a function with niche-biased weights.
    fn generate_weighted_function<R: Rng + ?Sized>(
        &self,
        niche: Niche,
        max_depth: usize,
        rng: &mut R,
    ) -> Expr {
        if niche == Niche::Mixed {
            return self.generate_random_function(max_depth, rng);
        }
        let weights: Vec<f64> = self
            .functions
            .iter()
            .map(|f| niche.function_weight(&f.name))
            .collect();
        let total: f64 = weights.iter().sum();
        let mut r = rng.gen_range(0.0..total);
        for (i, w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return self.build_function(i, max_depth, rng);
            }
        }
        self.build_function(0, max_depth, rng)
    }

    fn build_function<R: Rng + ?Sized>(
        &self,
        function_idx: usize,
        max_depth: usize,
        rng: &mut R,
    ) -> Expr {
        let arity = self.functions[function_idx].arity;
        let mut args = Vec::with_capacity(arity);
        for _ in 0..arity {
            args.push(self.generate_random_expr(max_depth, rng));
        }
        let function = &self.functions[function_idx];
        let mut expr = (function.builder)(args);

        // Frequency-aware: randomly annotate aggregation calls with freq
        if self.config.use_frequencies {
            if let Expr::FunctionCall { freq, args, .. } = &mut expr {
                if !args.is_empty() {
                    if let Some(data_freq) =
                        args.first().and_then(|a| frequency::infer_frequency(a))
                    {
                        if rng.gen_bool(0.4) {
                            let candidates = frequency::valid_agg_freqs(&data_freq);
                            if !candidates.is_empty() {
                                *freq =
                                    Some(candidates[rng.gen_range(0..candidates.len())].clone());
                            }
                        }
                    }
                }
            }
        }

        expr
    }

    /// Generate initial population (uniform).
    pub fn generate_initial_population<R: Rng + ?Sized>(
        &self,
        size: usize,
        rng: &mut R,
    ) -> Vec<Expr> {
        let mut population = Vec::with_capacity(size);
        for _ in 0..size {
            population.push(self.generate_random_expr(self.config.max_depth, rng));
        }
        population
    }

    /// Generate a diverse initial population with equal representation across niches.
    pub fn generate_diverse_population<R: Rng + ?Sized>(
        &self,
        size: usize,
        rng: &mut R,
    ) -> Vec<Expr> {
        let n_niches = Niche::ALL.len();
        let per_niche = size / n_niches;
        let remainder = size % n_niches;
        let mut population = Vec::with_capacity(size);

        for (i, &niche) in Niche::ALL.iter().enumerate() {
            let count = if i < remainder {
                per_niche + 1
            } else {
                per_niche
            };
            for _ in 0..count {
                population.push(self.generate_niche_expr(niche, self.config.max_depth, rng));
            }
        }

        // Shuffle to mix niches
        use rand::seq::SliceRandom;
        population.shuffle(rng);

        population
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_expression_generation() {
        let config = GPConfig::default();
        let mut rng = StdRng::seed_from_u64(42);

        let terminals = vec![
            Terminal::Variable("x".to_string()),
            Terminal::Constant(1.0),
            Terminal::Ephemeral,
        ];

        let functions = vec![Function::add(), Function::mul()];

        let generator = ExpressionGenerator::new(&config, terminals, functions);

        let expr = generator.generate_random_expr(3, &mut rng);
        assert!(matches!(
            expr,
            Expr::BinaryExpr { .. } | Expr::Literal(_) | Expr::Column(_)
        ));
    }
}
