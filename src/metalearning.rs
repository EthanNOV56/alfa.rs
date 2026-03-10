//! Meta-learning for alpha factor discovery
//!
//! This module analyzes historical GP runs to learn patterns and optimize
//! future factor mining through adaptive hyperparameter tuning and
//! intelligent search guidance.

use crate::gp::GPConfig;
use crate::persistence::{FactorMetadata, GPHistoryRecord};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Meta-learning model for factor discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningModel {
    /// Model version
    pub version: u32,
    /// Trained timestamp
    pub trained_at: u64,
    /// Function usage statistics
    pub function_stats: HashMap<String, FunctionStats>,
    /// Terminal usage statistics
    pub terminal_stats: HashMap<String, TerminalStats>,
    /// Expression pattern statistics
    pub pattern_stats: Vec<PatternStat>,
    /// Hyperparameter recommendations
    pub hyperparameter_recs: HyperparameterRecommendations,
    /// Performance correlation analysis
    pub performance_correlations: PerformanceCorrelations,
    /// Learning history
    pub learning_history: Vec<LearningRecord>,
}

/// Statistics for function usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionStats {
    /// Function name
    pub name: String,
    /// Total usage count
    pub usage_count: u32,
    /// Usage in high-performing factors (IC > threshold)
    pub high_perf_usage: u32,
    /// Average complexity when used
    pub avg_complexity: f64,
    /// Average IC when used
    pub avg_ic: f64,
    /// Common argument patterns
    pub argument_patterns: Vec<ArgumentPattern>,
}

/// Statistics for terminal usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalStats {
    /// Terminal name (or type)
    pub name: String,
    /// Total usage count
    pub usage_count: u32,
    /// Usage in high-performing factors
    pub high_perf_usage: u32,
    /// Average IC when used
    pub avg_ic: f64,
    /// Common co-occurring terminals
    pub co_occurrences: HashMap<String, f64>,
}

/// Argument pattern for functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentPattern {
    /// Pattern identifier
    pub pattern: String,
    /// Occurrence count
    pub count: u32,
    /// Average IC for this pattern
    pub avg_ic: f64,
}

/// Expression pattern statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStat {
    /// Pattern type (e.g., "ratio", "difference", "normalized")
    pub pattern_type: String,
    /// Example expression
    pub example: String,
    /// Occurrence count
    pub occurrence: u32,
    /// Success rate (IC > threshold)
    pub success_rate: f64,
    /// Average complexity
    pub avg_complexity: f64,
    /// Average IC
    pub avg_ic: f64,
}

/// Hyperparameter recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterRecommendations {
    /// Recommended population size
    pub population_size: (usize, usize), // (min, max)
    /// Recommended max generations
    pub max_generations: (usize, usize),
    /// Recommended tournament size
    pub tournament_size: (usize, usize),
    /// Recommended crossover probability
    pub crossover_prob: (f64, f64),
    /// Recommended mutation probability
    pub mutation_prob: (f64, f64),
    /// Recommended max depth
    pub max_depth: (usize, usize),
    /// Confidence scores for recommendations
    pub confidence: HashMap<String, f64>,
}

/// Performance correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCorrelations {
    /// Correlation between complexity and IC
    pub complexity_ic_corr: f64,
    /// Correlation between depth and IC
    pub depth_ic_corr: f64,
    /// Correlation between function count and IC
    pub function_count_ic_corr: f64,
    /// Optimal complexity range
    pub optimal_complexity_range: (f64, f64),
    /// Optimal depth range
    pub optimal_depth_range: (f64, f64),
}

/// Learning record for model updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRecord {
    /// Timestamp
    pub timestamp: u64,
    /// Number of factors analyzed
    pub factors_analyzed: u32,
    /// Number of GP runs analyzed
    pub runs_analyzed: u32,
    /// Model improvement metrics
    pub improvement: ModelImprovement,
}

/// Model improvement metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelImprovement {
    /// Prediction accuracy improvement
    pub accuracy_improvement: f64,
    /// Recommendation effectiveness improvement
    pub effectiveness_improvement: f64,
    /// Coverage improvement
    pub coverage_improvement: f64,
}

/// Meta-learning analyzer
pub struct MetaLearningAnalyzer {
    /// Base model
    model: MetaLearningModel,
    /// High performance threshold (IC)
    high_perf_threshold: f64,
    /// Minimum data points for analysis
    min_data_points: u32,
}

impl MetaLearningAnalyzer {
    /// Create a new meta-learning analyzer
    pub fn new() -> Self {
        Self {
            model: MetaLearningModel {
                version: 1,
                trained_at: 0,
                function_stats: HashMap::new(),
                terminal_stats: HashMap::new(),
                pattern_stats: Vec::new(),
                hyperparameter_recs: HyperparameterRecommendations {
                    population_size: (50, 200),
                    max_generations: (30, 100),
                    tournament_size: (3, 7),
                    crossover_prob: (0.7, 0.9),
                    mutation_prob: (0.05, 0.2),
                    max_depth: (3, 7),
                    confidence: HashMap::new(),
                },
                performance_correlations: PerformanceCorrelations {
                    complexity_ic_corr: 0.0,
                    depth_ic_corr: 0.0,
                    function_count_ic_corr: 0.0,
                    optimal_complexity_range: (3.0, 10.0),
                    optimal_depth_range: (2.0, 5.0),
                },
                learning_history: Vec::new(),
            },
            high_perf_threshold: 0.1, // IC > 0.1 is high performing
            min_data_points: 50,
        }
    }

    /// Train the model on historical data
    pub fn train(
        &mut self,
        factors: &[FactorMetadata],
        gp_runs: &[GPHistoryRecord],
    ) -> Result<(), String> {
        if factors.len() < self.min_data_points as usize {
            return Err(format!(
                "Insufficient data: need at least {} factors, got {}",
                self.min_data_points,
                factors.len()
            ));
        }

        // Analyze function usage
        self.analyze_function_usage(factors)?;

        // Analyze terminal usage
        self.analyze_terminal_usage(factors)?;

        // Analyze expression patterns
        self.analyze_expression_patterns(factors)?;

        // Analyze performance correlations
        self.analyze_performance_correlations(factors)?;

        // Analyze hyperparameter effectiveness
        self.analyze_hyperparameters(gp_runs)?;

        // Update model metadata
        self.model.version += 1;
        self.model.trained_at = crate::persistence::PersistenceManager::current_timestamp();

        // Record learning
        self.record_learning(factors.len() as u32, gp_runs.len() as u32);

        Ok(())
    }

    /// Analyze function usage patterns
    fn analyze_function_usage(&mut self, factors: &[FactorMetadata]) -> Result<(), String> {
        let mut function_stats = HashMap::new();

        for factor in factors {
            let expr = &factor.expression;
            let is_high_perf = factor.metrics.ic_mean.abs() > self.high_perf_threshold;

            // Extract functions from expression (simplified parsing)
            let functions = self.extract_functions_from_expr(expr);

            for func_name in functions {
                let stats = function_stats
                    .entry(func_name.clone())
                    .or_insert(FunctionStats {
                        name: func_name.clone(),
                        usage_count: 0,
                        high_perf_usage: 0,
                        avg_complexity: 0.0,
                        avg_ic: 0.0,
                        argument_patterns: Vec::new(),
                    });

                stats.usage_count += 1;
                if is_high_perf {
                    stats.high_perf_usage += 1;
                }

                // Update running averages
                let n = stats.usage_count as f64;
                stats.avg_ic = (stats.avg_ic * (n - 1.0) + factor.metrics.ic_mean) / n;
                stats.avg_complexity =
                    (stats.avg_complexity * (n - 1.0) + factor.metrics.complexity_penalty) / n;
            }
        }

        self.model.function_stats = function_stats;
        Ok(())
    }

    /// Extract functions from expression string (simplified)
    fn extract_functions_from_expr(&self, expr: &str) -> Vec<String> {
        // This is a simplified implementation
        // In production, would use proper expression parsing

        let mut functions = Vec::new();

        // Common function patterns
        let function_patterns = [
            ("sqrt", "sqrt"),
            ("log", "log"),
            ("abs", "abs"),
            ("exp", "exp"),
            ("sin", "sin"),
            ("cos", "cos"),
            ("tan", "tan"),
            ("atan", "atan"),
            ("pow", "pow"),
            ("max", "max"),
            ("min", "min"),
            ("mean", "mean"),
            ("std", "std"),
            ("sum", "sum"),
        ];

        let expr_lower = expr.to_lowercase();
        for (name, pattern) in function_patterns.iter() {
            if expr_lower.contains(pattern) {
                functions.push(name.to_string());
            }
        }

        functions
    }

    /// Analyze terminal usage patterns
    fn analyze_terminal_usage(&mut self, factors: &[FactorMetadata]) -> Result<(), String> {
        let mut terminal_stats = HashMap::new();

        // Common financial terminals
        let common_terminals = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "returns",
            "volatility",
            "momentum",
            "rsi",
            "macd",
            "bollinger",
            "atr",
            "adx",
        ];

        for &terminal in &common_terminals {
            let mut usage_count = 0;
            let mut high_perf_usage = 0;
            let mut total_ic = 0.0;

            for factor in factors {
                let expr_lower = factor.expression.to_lowercase();
                if expr_lower.contains(terminal) {
                    usage_count += 1;
                    if factor.metrics.ic_mean.abs() > self.high_perf_threshold {
                        high_perf_usage += 1;
                    }
                    total_ic += factor.metrics.ic_mean;
                }
            }

            if usage_count > 0 {
                let stats = TerminalStats {
                    name: terminal.to_string(),
                    usage_count,
                    high_perf_usage,
                    avg_ic: total_ic / usage_count as f64,
                    co_occurrences: HashMap::new(),
                };
                terminal_stats.insert(terminal.to_string(), stats);
            }
        }

        // Analyze co-occurrences
        let terminal_names: Vec<String> = terminal_stats.keys().cloned().collect();

        for terminal1 in &terminal_names {
            if let Some(stats1) = terminal_stats.get_mut(terminal1) {
                for terminal2 in &terminal_names {
                    if terminal1 != terminal2 {
                        let mut co_occur_count = 0;
                        let mut total_pairs = 0;

                        for factor in factors {
                            let expr_lower = factor.expression.to_lowercase();
                            if expr_lower.contains(terminal1) && expr_lower.contains(terminal2) {
                                co_occur_count += 1;
                            }
                            if expr_lower.contains(terminal1) || expr_lower.contains(terminal2) {
                                total_pairs += 1;
                            }
                        }

                        if total_pairs > 0 {
                            let co_occurrence_rate = co_occur_count as f64 / total_pairs as f64;
                            stats1
                                .co_occurrences
                                .insert(terminal2.clone(), co_occurrence_rate);
                        }
                    }
                }
            }
        }

        self.model.terminal_stats = terminal_stats;
        Ok(())
    }

    /// Analyze expression patterns
    fn analyze_expression_patterns(&mut self, factors: &[FactorMetadata]) -> Result<(), String> {
        let mut patterns = HashMap::new();

        for factor in factors {
            let pattern_type = self.classify_expression_pattern(&factor.expression);
            let is_success = factor.metrics.ic_mean.abs() > self.high_perf_threshold;

            let entry = patterns.entry(pattern_type.clone()).or_insert((
                0,                         // occurrence
                0,                         // success_count
                0.0,                       // total_complexity
                0.0,                       // total_ic
                factor.expression.clone(), // example
            ));

            entry.0 += 1;
            if is_success {
                entry.1 += 1;
            }
            entry.2 += factor.metrics.complexity_penalty;
            entry.3 += factor.metrics.ic_mean;
        }

        let mut pattern_stats = Vec::new();
        for (pattern_type, (occurrence, success_count, total_complexity, total_ic, example)) in
            patterns
        {
            pattern_stats.push(PatternStat {
                pattern_type,
                example,
                occurrence,
                success_rate: success_count as f64 / occurrence as f64,
                avg_complexity: total_complexity / occurrence as f64,
                avg_ic: total_ic / occurrence as f64,
            });
        }

        // Sort by success rate
        pattern_stats.sort_by(|a, b| {
            b.success_rate
                .partial_cmp(&a.success_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.model.pattern_stats = pattern_stats;
        Ok(())
    }

    /// Classify expression pattern
    fn classify_expression_pattern(&self, expr: &str) -> String {
        let expr_lower = expr.to_lowercase();

        if expr_lower.contains("/") && !expr_lower.contains("1/") {
            return "ratio".to_string();
        }

        if expr_lower.contains("-") {
            return "difference".to_string();
        }

        if expr_lower.contains("*") {
            return "product".to_string();
        }

        if expr_lower.contains("log") || expr_lower.contains("exp") {
            return "log_transform".to_string();
        }

        if expr_lower.contains("sqrt") {
            return "sqrt_transform".to_string();
        }

        if expr_lower.contains("mean") || expr_lower.contains("avg") {
            return "average".to_string();
        }

        if expr_lower.contains("std") || expr_lower.contains("volatility") {
            return "volatility".to_string();
        }

        "simple".to_string()
    }

    /// Analyze performance correlations
    fn analyze_performance_correlations(
        &mut self,
        factors: &[FactorMetadata],
    ) -> Result<(), String> {
        if factors.is_empty() {
            return Ok(());
        }

        let mut complexity_values = Vec::new();
        let mut ic_values = Vec::new();
        let mut depth_values = Vec::new();
        let mut function_counts = Vec::new();

        for factor in factors {
            complexity_values.push(factor.metrics.complexity_penalty);
            ic_values.push(factor.metrics.ic_mean);

            // Estimate depth from expression
            let depth = self.estimate_expression_depth(&factor.expression);
            depth_values.push(depth as f64);

            // Count functions
            let function_count = self.extract_functions_from_expr(&factor.expression).len();
            function_counts.push(function_count as f64);
        }

        // Calculate correlations
        self.model.performance_correlations.complexity_ic_corr =
            self.calculate_correlation(&complexity_values, &ic_values);
        self.model.performance_correlations.depth_ic_corr =
            self.calculate_correlation(&depth_values, &ic_values);
        self.model.performance_correlations.function_count_ic_corr =
            self.calculate_correlation(&function_counts, &ic_values);

        // Find optimal ranges (top 25% by IC)
        let mut factors_with_complexity: Vec<_> = factors
            .iter()
            .map(|f| (f.metrics.complexity_penalty, f.metrics.ic_mean.abs()))
            .collect();
        factors_with_complexity.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_n = (factors_with_complexity.len() as f64 * 0.25).ceil() as usize;
        let top_complexities: Vec<_> = factors_with_complexity
            [..top_n.min(factors_with_complexity.len())]
            .iter()
            .map(|(c, _)| *c)
            .collect();

        if !top_complexities.is_empty() {
            let min_complexity = top_complexities
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let max_complexity = top_complexities
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            self.model.performance_correlations.optimal_complexity_range =
                (min_complexity, max_complexity);
        }

        Ok(())
    }

    /// Estimate expression depth
    fn estimate_expression_depth(&self, expr: &str) -> usize {
        // Simplified depth estimation based on parentheses
        let mut max_depth = 0;
        let mut current_depth = 0;

        for ch in expr.chars() {
            match ch {
                '(' => {
                    current_depth += 1;
                    max_depth = max_depth.max(current_depth);
                }
                ')' => {
                    if current_depth > 0 {
                        current_depth -= 1;
                    }
                }
                _ => {}
            }
        }

        max_depth
    }

    /// Calculate Pearson correlation
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|b| b * b).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Analyze hyperparameter effectiveness from GP runs
    fn analyze_hyperparameters(&mut self, gp_runs: &[GPHistoryRecord]) -> Result<(), String> {
        if gp_runs.is_empty() {
            return Ok(());
        }

        let mut population_size_success = Vec::new();
        let mut max_generations_success = Vec::new();
        let mut tournament_size_success = Vec::new();
        let mut crossover_prob_success = Vec::new();
        let mut mutation_prob_success = Vec::new();
        let mut max_depth_success = Vec::new();

        for run in gp_runs {
            let success = run.best_factor.metrics.ic_mean.abs() > self.high_perf_threshold;

            if success {
                population_size_success.push(run.config.population_size as f64);
                max_generations_success.push(run.config.max_generations as f64);
                tournament_size_success.push(run.config.tournament_size as f64);
                crossover_prob_success.push(run.config.crossover_prob);
                mutation_prob_success.push(run.config.mutation_prob);
                max_depth_success.push(run.config.max_depth as f64);
            }
        }

        // Update recommendations based on successful runs
        if !population_size_success.is_empty() {
            let min_pop = population_size_success
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min) as usize;
            let max_pop = population_size_success
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max) as usize;
            self.model.hyperparameter_recs.population_size = (min_pop, max_pop);
        }

        if !max_generations_success.is_empty() {
            let min_gen = max_generations_success
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min) as usize;
            let max_gen = max_generations_success
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max) as usize;
            self.model.hyperparameter_recs.max_generations = (min_gen, max_gen);
        }

        if !tournament_size_success.is_empty() {
            let min_tourn = tournament_size_success
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min) as usize;
            let max_tourn = tournament_size_success
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max) as usize;
            self.model.hyperparameter_recs.tournament_size = (min_tourn, max_tourn);
        }

        if !crossover_prob_success.is_empty() {
            let min_cross = crossover_prob_success
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let max_cross = crossover_prob_success
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            self.model.hyperparameter_recs.crossover_prob = (min_cross, max_cross);
        }

        if !mutation_prob_success.is_empty() {
            let min_mut = mutation_prob_success
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let max_mut = mutation_prob_success
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            self.model.hyperparameter_recs.mutation_prob = (min_mut, max_mut);
        }

        if !max_depth_success.is_empty() {
            let min_depth = max_depth_success
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min) as usize;
            let max_depth = max_depth_success
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max) as usize;
            self.model.hyperparameter_recs.max_depth = (min_depth, max_depth);
        }

        Ok(())
    }

    /// Record learning progress
    fn record_learning(&mut self, factors_analyzed: u32, runs_analyzed: u32) {
        let record = LearningRecord {
            timestamp: crate::persistence::PersistenceManager::current_timestamp(),
            factors_analyzed,
            runs_analyzed,
            improvement: ModelImprovement {
                accuracy_improvement: 0.0, // Would need baseline comparison
                effectiveness_improvement: 0.0,
                coverage_improvement: 0.0,
            },
        };

        self.model.learning_history.push(record);

        // Keep only last 100 records
        if self.model.learning_history.len() > 100 {
            self.model.learning_history.remove(0);
        }
    }

    /// Get recommendations for next GP run
    pub fn get_recommendations(&self, target_complexity: Option<f64>) -> GPRecommendations {
        let mut functions = Vec::new();
        let mut terminals = Vec::new();

        // Recommend functions based on success rate
        let mut function_candidates: Vec<_> = self.model.function_stats.values().collect();
        function_candidates.sort_by(|a, b| {
            let success_rate_a = a.high_perf_usage as f64 / a.usage_count as f64;
            let success_rate_b = b.high_perf_usage as f64 / b.usage_count as f64;
            success_rate_b
                .partial_cmp(&success_rate_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top 5-8 functions
        let n_functions = function_candidates
            .len()
            .min(8)
            .max(5.min(function_candidates.len()));
        for i in 0..n_functions {
            functions.push(function_candidates[i].name.clone());
        }

        // Recommend terminals based on usage and IC
        let mut terminal_candidates: Vec<_> = self.model.terminal_stats.values().collect();
        terminal_candidates.sort_by(|a, b| {
            b.avg_ic
                .partial_cmp(&a.avg_ic)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top 5-10 terminals
        let n_terminals = terminal_candidates
            .len()
            .min(10)
            .max(5.min(terminal_candidates.len()));
        for i in 0..n_terminals {
            terminals.push(terminal_candidates[i].name.clone());
        }

        // Get hyperparameters
        let hyperparams = self.model.hyperparameter_recs.clone();

        // Determine target complexity
        let target_complexity = target_complexity.unwrap_or_else(|| {
            // Use optimal complexity range midpoint
            let (min, max) = self.model.performance_correlations.optimal_complexity_range;
            (min + max) / 2.0
        });

        GPRecommendations {
            recommended_functions: functions,
            recommended_terminals: terminals,
            hyperparameters: hyperparams,
            target_complexity,
            confidence_score: self.calculate_recommendation_confidence(),
        }
    }

    /// Calculate confidence score for recommendations
    fn calculate_recommendation_confidence(&self) -> f64 {
        let mut confidence = 0.0;
        let mut factors = 0;

        // Base confidence on amount of training data
        let total_factors = self
            .model
            .function_stats
            .values()
            .map(|s| s.usage_count)
            .sum::<u32>();
        let data_confidence = (total_factors as f64 / self.min_data_points as f64).min(1.0);
        confidence += data_confidence * 0.3;
        factors += 1;

        // Confidence from clear patterns in function success rates
        let mut success_rates: Vec<_> = self
            .model
            .function_stats
            .values()
            .map(|s| s.high_perf_usage as f64 / s.usage_count as f64)
            .collect();
        success_rates.sort_by(|a, b| b.partial_cmp(a).unwrap());

        if success_rates.len() >= 3 {
            let top_avg = success_rates[..3].iter().sum::<f64>() / 3.0;
            let bottom_avg = success_rates[success_rates.len() - 3..].iter().sum::<f64>() / 3.0;
            let pattern_confidence = (top_avg - bottom_avg).max(0.0).min(1.0);
            confidence += pattern_confidence * 0.4;
            factors += 1;
        }

        // Confidence from performance correlations
        let correlation_confidence = self
            .model
            .performance_correlations
            .complexity_ic_corr
            .abs()
            .max(self.model.performance_correlations.depth_ic_corr.abs());
        confidence += correlation_confidence * 0.3;
        factors += 1;

        if factors > 0 {
            confidence / factors as f64
        } else {
            0.5 // Default moderate confidence
        }
    }

    /// Get the trained model
    pub fn get_model(&self) -> &MetaLearningModel {
        &self.model
    }

    /// Save model to file
    pub fn save_model(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.model)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load model from file
    pub fn load_model(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let model: MetaLearningModel = serde_json::from_str(&json)?;

        Ok(Self {
            model,
            high_perf_threshold: 0.1,
            min_data_points: 50,
        })
    }

    /// Get high performance threshold
    pub fn get_high_perf_threshold(&self) -> f64 {
        self.high_perf_threshold
    }

    /// Set high performance threshold
    pub fn set_high_perf_threshold(&mut self, threshold: f64) {
        self.high_perf_threshold = threshold;
    }

    /// Get minimum data points required for training
    pub fn get_min_data_points(&self) -> u32 {
        self.min_data_points
    }

    /// Set minimum data points required for training
    pub fn set_min_data_points(&mut self, min_points: u32) {
        self.min_data_points = min_points;
    }

    /// Get model version
    pub fn version(&self) -> u32 {
        self.model.version
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        self.model.trained_at > 0
    }
}

/// Recommendations for GP configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPRecommendations {
    /// Recommended functions to use
    pub recommended_functions: Vec<String>,
    /// Recommended terminals to use
    pub recommended_terminals: Vec<String>,
    /// Recommended hyperparameter ranges
    pub hyperparameters: HyperparameterRecommendations,
    /// Target complexity level
    pub target_complexity: f64,
    /// Confidence score for recommendations (0.0 to 1.0)
    pub confidence_score: f64,
}

impl GPRecommendations {
    /// Create a GPConfig from recommendations (using midpoint of ranges)
    pub fn to_gp_config(&self) -> GPConfig {
        let pop_range = self.hyperparameters.population_size;
        let gen_range = self.hyperparameters.max_generations;
        let tourn_range = self.hyperparameters.tournament_size;
        let crossover_range = self.hyperparameters.crossover_prob;
        let mutation_range = self.hyperparameters.mutation_prob;
        let depth_range = self.hyperparameters.max_depth;

        GPConfig {
            population_size: ((pop_range.0 + pop_range.1) / 2) as usize,
            max_generations: ((gen_range.0 + gen_range.1) / 2) as usize,
            tournament_size: ((tourn_range.0 + tourn_range.1) / 2) as usize,
            crossover_prob: (crossover_range.0 + crossover_range.1) / 2.0,
            mutation_prob: (mutation_range.0 + mutation_range.1) / 2.0,
            max_depth: ((depth_range.0 + depth_range.1) / 2) as usize,
        }
    }

    /// Create a GPConfig with random values within recommended ranges
    pub fn to_random_gp_config(&self, rng: &mut impl rand::Rng) -> GPConfig {
        let pop_range = self.hyperparameters.population_size;
        let gen_range = self.hyperparameters.max_generations;
        let tourn_range = self.hyperparameters.tournament_size;
        let crossover_range = self.hyperparameters.crossover_prob;
        let mutation_range = self.hyperparameters.mutation_prob;
        let depth_range = self.hyperparameters.max_depth;

        GPConfig {
            population_size: rng.gen_range(pop_range.0..=pop_range.1),
            max_generations: rng.gen_range(gen_range.0..=gen_range.1),
            tournament_size: rng.gen_range(tourn_range.0..=tourn_range.1),
            crossover_prob: rng.gen_range(crossover_range.0..=crossover_range.1),
            mutation_prob: rng.gen_range(mutation_range.0..=mutation_range.1),
            max_depth: rng.gen_range(depth_range.0..=depth_range.1),
        }
    }

    /// Check if recommendations are valid
    pub fn is_valid(&self) -> bool {
        !self.recommended_functions.is_empty()
            && !self.recommended_terminals.is_empty()
            && self.confidence_score > 0.1
    }

    /// Get confidence level as string
    pub fn confidence_level(&self) -> &'static str {
        match self.confidence_score {
            x if x >= 0.8 => "high",
            x if x >= 0.6 => "medium",
            x if x >= 0.4 => "low",
            _ => "very_low",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::{create_factor_metadata, PerformanceMetrics};
    use std::collections::HashMap;

    fn create_test_factor(name: &str, ic: f64, complexity: f64) -> FactorMetadata {
        let metrics = PerformanceMetrics {
            ic_mean: ic,
            ic_ir: ic * 2.0,
            turnover: 0.1,
            complexity_penalty: complexity,
            combined_score: ic * 0.8,
            eval_period: ("2023-01-01".to_string(), "2023-12-31".to_string()),
            n_assets: 100,
            n_days: 252,
            custom_metrics: HashMap::new(),
        };

        FactorMetadata {
            id: format!("test_factor_{}", name),
            expression: format!("close_{} * volume", name),
            created_at: 1234567890,
            updated_at: 1234567890,
            metrics,
            tags: vec!["test".to_string()],
            gp_config: None,
            dataset_info: None,
        }
    }

    #[test]
    fn test_meta_learning_analyzer_creation() {
        let analyzer = MetaLearningAnalyzer::new();
        assert_eq!(analyzer.version(), 1);
        assert!(!analyzer.is_trained());
        assert_eq!(analyzer.get_high_perf_threshold(), 0.1);
    }

    #[test]
    fn test_analyze_function_usage() {
        let mut analyzer = MetaLearningAnalyzer::new();

        let mut factors = Vec::new();
        factors.push(create_test_factor("high_ic", 0.15, 3.0));
        factors.push(create_test_factor("low_ic", 0.05, 5.0));

        // Should fail due to insufficient data
        let result = analyzer.train(&factors, &[]);
        assert!(result.is_err());

        // Set lower threshold for testing
        analyzer.min_data_points = 2;
        let result = analyzer.train(&factors, &[]);
        assert!(result.is_ok());

        assert!(analyzer.is_trained());
        assert_eq!(analyzer.version(), 2);
    }

    #[test]
    fn test_get_recommendations() {
        let mut analyzer = MetaLearningAnalyzer::new();
        analyzer.min_data_points = 5;

        // Create enough test factors
        let mut factors = Vec::new();
        for i in 0..10 {
            let ic = if i < 3 { 0.15 } else { 0.05 };
            factors.push(create_test_factor(
                &format!("factor{}", i),
                ic,
                (i + 3) as f64,
            ));
        }

        // Create dummy GP runs
        let gp_runs = Vec::new();

        let result = analyzer.train(&factors, &gp_runs);
        assert!(result.is_ok());

        let recommendations = analyzer.get_recommendations(None);
        assert!(recommendations.is_valid());
        assert!(recommendations.confidence_score > 0.0);
        assert!(!recommendations.recommended_functions.is_empty());

        // Test GP config creation
        let config = recommendations.to_gp_config();
        assert!(config.population_size > 0);
        assert!(config.max_generations > 0);
        assert!(config.tournament_size > 0);
        assert!(config.crossover_prob > 0.0);
        assert!(config.mutation_prob > 0.0);
        assert!(config.max_depth > 0);
    }

    #[test]
    fn test_gp_recommendations_methods() {
        let recommendations = GPRecommendations {
            recommended_functions: vec!["sqrt".to_string(), "log".to_string()],
            recommended_terminals: vec!["close".to_string(), "volume".to_string()],
            hyperparameters: HyperparameterRecommendations {
                population_size: (50, 200),
                max_generations: (30, 100),
                tournament_size: (3, 7),
                crossover_prob: (0.7, 0.9),
                mutation_prob: (0.05, 0.2),
                max_depth: (3, 7),
                confidence: HashMap::new(),
            },
            target_complexity: 5.0,
            confidence_score: 0.75,
        };

        assert!(recommendations.is_valid());
        assert_eq!(recommendations.confidence_level(), "medium");

        let config = recommendations.to_gp_config();
        assert_eq!(config.population_size, 125); // (50 + 200) / 2
        assert_eq!(config.max_generations, 65); // (30 + 100) / 2
        assert_eq!(config.tournament_size, 5); // (3 + 7) / 2
        assert!((config.crossover_prob - 0.8).abs() < 0.01); // (0.7 + 0.9) / 2
        assert!((config.mutation_prob - 0.125).abs() < 0.01); // (0.05 + 0.2) / 2
        assert_eq!(config.max_depth, 5); // (3 + 7) / 2
    }

    #[test]
    fn test_model_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let model_path = temp_dir.path().join("model.json");

        let mut analyzer = MetaLearningAnalyzer::new();
        analyzer.min_data_points = 5;

        // Create test data
        let mut factors = Vec::new();
        for i in 0..10 {
            factors.push(create_test_factor(&format!("factor{}", i), 0.1, 5.0));
        }

        analyzer.train(&factors, &[]).unwrap();

        // Save model
        analyzer.save_model(&model_path).unwrap();
        assert!(model_path.exists());

        // Load model
        let loaded_analyzer = MetaLearningAnalyzer::load_model(&model_path).unwrap();
        assert_eq!(loaded_analyzer.version(), analyzer.version());
        assert!(loaded_analyzer.is_trained());
    }

    #[test]
    fn test_classify_expression_pattern() {
        let analyzer = MetaLearningAnalyzer::new();

        assert_eq!(analyzer.classify_expression_pattern("a / b"), "ratio");
        assert_eq!(analyzer.classify_expression_pattern("a - b"), "difference");
        assert_eq!(analyzer.classify_expression_pattern("a * b"), "product");
        assert_eq!(
            analyzer.classify_expression_pattern("log(a)"),
            "log_transform"
        );
        assert_eq!(
            analyzer.classify_expression_pattern("sqrt(a)"),
            "sqrt_transform"
        );
        assert_eq!(analyzer.classify_expression_pattern("mean(a)"), "average");
        assert_eq!(analyzer.classify_expression_pattern("std(a)"), "volatility");
        assert_eq!(analyzer.classify_expression_pattern("a + b"), "simple");
    }
}
