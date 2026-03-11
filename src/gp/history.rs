//! GP history and statistics types
//!
//! This module contains types for tracking GP evolution history
//! and population statistics.

use crate::persistence::FactorMetadata;
use serde::{Deserialize, Serialize};

/// GP evolution history record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPHistoryRecord {
    /// Run identifier
    pub run_id: String,
    /// Start timestamp
    pub start_time: u64,
    /// End timestamp
    pub end_time: u64,
    /// GP configuration used
    pub config: crate::gp::GPConfig,
    /// Terminal set used
    pub terminals: Vec<String>,
    /// Function set used
    pub functions: Vec<String>,
    /// Best factor discovered
    pub best_factor: FactorMetadata,
    /// Evolution progress (generation, best_fitness)
    pub progress: Vec<(usize, f64)>,
    /// Population statistics
    pub population_stats: Vec<PopulationStats>,
}

/// Population statistics for a generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationStats {
    /// Generation number
    pub generation: usize,
    /// Average fitness
    pub avg_fitness: f64,
    /// Best fitness
    pub best_fitness: f64,
    /// Worst fitness
    pub worst_fitness: f64,
    /// Average complexity
    pub avg_complexity: f64,
    /// Diversity metric
    pub diversity: f64,
}

/// Helper function to create GPHistoryRecord
#[allow(dead_code)]
pub fn create_gp_history_record(
    run_id: String,
    start_time: u64,
    end_time: u64,
    config: crate::gp::GPConfig,
    terminals: Vec<String>,
    functions: Vec<String>,
    best_factor: FactorMetadata,
    progress: Vec<(usize, f64)>,
    population_stats: Vec<PopulationStats>,
) -> GPHistoryRecord {
    GPHistoryRecord {
        run_id,
        start_time,
        end_time,
        config,
        terminals,
        functions,
        best_factor,
        progress,
        population_stats,
    }
}
