//! Persistent storage for alpha-expr factor mining results
//!
//! This module provides functionality to save, load, and manage discovered factors,
//! GP evolution history, and performance metrics across sessions.

use crate::gp::{GPConfig, MultiObjectiveFitness};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Metadata for a discovered factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorMetadata {
    /// Unique identifier for this factor
    pub id: String,
    /// Factor expression in string format
    pub expression: String,
    /// Creation timestamp (Unix epoch in seconds)
    pub created_at: u64,
    /// Last evaluation timestamp
    pub updated_at: u64,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Tags for organization and search
    pub tags: Vec<String>,
    /// GP configuration used to discover this factor
    pub gp_config: Option<GPConfig>,
    /// Source dataset information
    pub dataset_info: Option<DatasetInfo>,
}

/// Performance metrics for a factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Information Coefficient (IC)
    pub ic_mean: f64,
    /// Information Ratio (IR)
    pub ic_ir: f64,
    /// Turnover rate
    pub turnover: f64,
    /// Complexity penalty
    pub complexity_penalty: f64,
    /// Combined fitness score
    pub combined_score: f64,
    /// Evaluation date range
    pub eval_period: (String, String), // (start_date, end_date)
    /// Number of assets used in evaluation
    pub n_assets: usize,
    /// Number of days used in evaluation
    pub n_days: usize,
    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset name or identifier
    pub name: String,
    /// Column names used in factor
    pub columns_used: Vec<String>,
    /// Date range of data
    pub date_range: (String, String),
    /// Number of assets
    pub n_assets: usize,
}

/// Expression cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionCacheEntry {
    /// Expression string (canonical representation)
    pub expression: String,
    /// Factor matrix shape (days, assets)
    pub shape: (usize, usize),
    /// Computed factor matrix (optional, can be large)
    pub factor_matrix: Option<Vec<f64>>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Last accessed timestamp
    pub last_accessed: u64,
    /// Access count
    pub access_count: u32,
}

/// Persistent storage manager
pub struct PersistenceManager {
    /// Base directory for storage
    base_dir: PathBuf,
    /// In-memory factor cache
    factor_cache: HashMap<String, FactorMetadata>,
    /// In-memory GP history cache
    history_cache: HashMap<String, GPHistoryRecord>,
    /// Expression cache manager
    expr_cache_manager: ExpressionCacheManager,
}

impl PersistenceManager {
    /// Create a new persistence manager
    pub fn new(base_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Create directory structure
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(base_dir.join("factors"))?;
        fs::create_dir_all(base_dir.join("history"))?;
        fs::create_dir_all(base_dir.join("cache"))?;
        fs::create_dir_all(base_dir.join("metalearning"))?;

        let expr_cache_manager = ExpressionCacheManager::new(base_dir.join("cache"))?;

        Ok(Self {
            base_dir,
            factor_cache: HashMap::new(),
            history_cache: HashMap::new(),
            expr_cache_manager,
        })
    }

    /// Save a discovered factor
    pub fn save_factor(&mut self, factor: &FactorMetadata) -> std::io::Result<()> {
        let factor_path = self.base_dir.join("factors").join(&factor.id);

        // Serialize to JSON
        let json = serde_json::to_string_pretty(factor)?;
        fs::write(factor_path.with_extension("json"), json)?;

        // Update cache
        self.factor_cache.insert(factor.id.clone(), factor.clone());

        Ok(())
    }

    /// Load a factor by ID
    pub fn load_factor(&mut self, factor_id: &str) -> std::io::Result<Option<FactorMetadata>> {
        // Check cache first
        if let Some(factor) = self.factor_cache.get(factor_id) {
            return Ok(Some(factor.clone()));
        }

        let factor_path = self
            .base_dir
            .join("factors")
            .join(factor_id)
            .with_extension("json");
        if !factor_path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(factor_path)?;
        let factor: FactorMetadata = serde_json::from_str(&json)?;

        // Update cache
        self.factor_cache.insert(factor.id.clone(), factor.clone());

        Ok(Some(factor))
    }

    /// Search factors by criteria
    pub fn search_factors(
        &self,
        min_ic: Option<f64>,
        min_ir: Option<f64>,
        max_complexity: Option<f64>,
        tags: &[String],
    ) -> Vec<FactorMetadata> {
        self.factor_cache
            .values()
            .filter(|factor| {
                // Filter by IC
                if let Some(min) = min_ic {
                    if factor.metrics.ic_mean.abs() < min.abs() {
                        return false;
                    }
                }

                // Filter by IR
                if let Some(min) = min_ir {
                    if factor.metrics.ic_ir.abs() < min.abs() {
                        return false;
                    }
                }

                // Filter by complexity
                if let Some(max) = max_complexity {
                    if factor.metrics.complexity_penalty > max {
                        return false;
                    }
                }

                // Filter by tags
                if !tags.is_empty() {
                    let has_all_tags = tags.iter().all(|tag| factor.tags.contains(tag));
                    if !has_all_tags {
                        return false;
                    }
                }

                true
            })
            .cloned()
            .collect()
    }

    /// Save GP history record
    pub fn save_gp_history(&mut self, record: &GPHistoryRecord) -> std::io::Result<()> {
        let history_path = self.base_dir.join("history").join(&record.run_id);

        // Serialize to JSON
        let json = serde_json::to_string_pretty(record)?;
        fs::write(history_path.with_extension("json"), json)?;

        // Update cache
        self.history_cache
            .insert(record.run_id.clone(), record.clone());

        Ok(())
    }

    /// Load GP history by run ID
    pub fn load_gp_history(&mut self, run_id: &str) -> std::io::Result<Option<GPHistoryRecord>> {
        // Check cache first
        if let Some(record) = self.history_cache.get(run_id) {
            return Ok(Some(record.clone()));
        }

        let history_path = self
            .base_dir
            .join("history")
            .join(run_id)
            .with_extension("json");
        if !history_path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(history_path)?;
        let record: GPHistoryRecord = serde_json::from_str(&json)?;

        // Update cache
        self.history_cache
            .insert(record.run_id.clone(), record.clone());

        Ok(Some(record))
    }

    /// Get expression cache manager
    pub fn expr_cache(&self) -> &ExpressionCacheManager {
        &self.expr_cache_manager
    }

    /// Get expression cache manager (mutable)
    pub fn expr_cache_mut(&mut self) -> &mut ExpressionCacheManager {
        &mut self.expr_cache_manager
    }

    /// Generate a unique factor ID
    pub fn generate_factor_id(&self, expr: &str) -> String {
        use sha2::{Digest, Sha256};

        // Create hash from expression
        let mut hasher = Sha256::new();
        hasher.update(expr.as_bytes());
        let result = hasher.finalize();

        // Convert to hex string
        let hash = format!("{:x}", result);

        // Take first 16 characters for ID
        format!("factor_{}", &hash[..16])
    }

    /// Generate a unique run ID
    pub fn generate_run_id(&self) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        format!("run_{}", timestamp)
    }

    /// Get current timestamp
    pub fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Load all factors from disk (for initialization)
    pub fn load_all_factors(&mut self) -> std::io::Result<usize> {
        let factors_dir = self.base_dir.join("factors");

        if !factors_dir.exists() {
            return Ok(0);
        }

        let mut count = 0;
        for entry in fs::read_dir(factors_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let json = fs::read_to_string(&path)?;
                if let Ok(factor) = serde_json::from_str::<FactorMetadata>(&json) {
                    self.factor_cache.insert(factor.id.clone(), factor);
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Load all GP history from disk (for initialization)
    pub fn load_all_history(&mut self) -> std::io::Result<usize> {
        let history_dir = self.base_dir.join("history");

        if !history_dir.exists() {
            return Ok(0);
        }

        let mut count = 0;
        for entry in fs::read_dir(history_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let json = fs::read_to_string(&path)?;
                if let Ok(record) = serde_json::from_str::<GPHistoryRecord>(&json) {
                    self.history_cache.insert(record.run_id.clone(), record);
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Get all loaded factors
    pub fn get_all_factors(&self) -> Vec<FactorMetadata> {
        self.factor_cache.values().cloned().collect()
    }

    /// Get all loaded GP history records
    pub fn get_all_history(&self) -> Vec<GPHistoryRecord> {
        self.history_cache.values().cloned().collect()
    }

    /// Clear all in-memory data (but not disk)
    pub fn clear_memory(&mut self) {
        self.factor_cache.clear();
        self.history_cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> std::io::Result<CacheStats> {
        self.expr_cache_manager.get_stats()
    }
}

/// Expression cache manager
pub struct ExpressionCacheManager {
    cache_dir: PathBuf,
    max_cache_size: usize,
    current_size: usize,
}

impl ExpressionCacheManager {
    /// Create a new expression cache manager
    pub fn new(cache_dir: PathBuf) -> std::io::Result<Self> {
        fs::create_dir_all(&cache_dir)?;

        Ok(Self {
            cache_dir,
            max_cache_size: 1000, // Default: 1000 expressions
            current_size: 0,
        })
    }

    /// Set maximum cache size
    pub fn set_max_size(&mut self, max_size: usize) {
        self.max_cache_size = max_size;
    }

    /// Save expression evaluation result
    pub fn save_expression(
        &mut self,
        expr: &str,
        shape: (usize, usize),
        factor_matrix: Option<&[f64]>,
        metrics: &PerformanceMetrics,
    ) -> std::io::Result<()> {
        // Generate cache key
        let cache_key = self.generate_cache_key(expr);
        let cache_path = self.cache_dir.join(&cache_key);

        // Create cache entry
        let entry = ExpressionCacheEntry {
            expression: expr.to_string(),
            shape,
            factor_matrix: factor_matrix.map(|fm| fm.to_vec()),
            metrics: metrics.clone(),
            last_accessed: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            access_count: 1,
        };

        // Serialize and save
        let json = serde_json::to_string(&entry)?;
        fs::write(cache_path.with_extension("json"), json)?;

        // Update size tracking
        self.current_size += 1;

        // Check if we need to evict old entries
        if self.current_size > self.max_cache_size {
            self.evict_old_entries()?;
        }

        Ok(())
    }

    /// Load expression evaluation result
    pub fn load_expression(&mut self, expr: &str) -> std::io::Result<Option<ExpressionCacheEntry>> {
        let cache_key = self.generate_cache_key(expr);
        let cache_path = self.cache_dir.join(cache_key).with_extension("json");

        if !cache_path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(&cache_path)?;
        let mut entry: ExpressionCacheEntry = serde_json::from_str(&json)?;

        // Update access metadata
        entry.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        entry.access_count += 1;

        // Save updated metadata
        let updated_json = serde_json::to_string(&entry)?;
        fs::write(cache_path, updated_json)?;

        Ok(Some(entry))
    }

    /// Generate cache key from expression
    fn generate_cache_key(&self, expr: &str) -> String {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(expr.as_bytes());
        let result = hasher.finalize();

        format!("{:x}", result)
    }

    /// Evict old cache entries based on LRU policy
    fn evict_old_entries(&mut self) -> std::io::Result<()> {
        let mut entries = Vec::new();

        // Collect all cache entries
        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(json) = fs::read_to_string(&path) {
                    if let Ok(cache_entry) = serde_json::from_str::<ExpressionCacheEntry>(&json) {
                        entries.push((path, cache_entry));
                    }
                }
            }
        }

        // Sort by last_accessed (oldest first)
        entries.sort_by(|a, b| a.1.last_accessed.cmp(&b.1.last_accessed));

        // Remove oldest entries until we're under limit
        let entries_to_remove = self.current_size.saturating_sub(self.max_cache_size);
        for i in 0..entries_to_remove {
            if i < entries.len() {
                let (path, _) = &entries[i];
                fs::remove_file(path)?;
            }
        }

        self.current_size = self.max_cache_size.min(entries.len() - entries_to_remove);

        Ok(())
    }

    /// Clear entire cache
    pub fn clear_cache(&mut self) -> std::io::Result<()> {
        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                fs::remove_file(path)?;
            }
        }

        self.current_size = 0;

        Ok(())
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> std::io::Result<CacheStats> {
        let mut total_size = 0;
        let mut total_entries = 0;
        let mut avg_access_count = 0.0;

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                total_entries += 1;

                // Get file size
                if let Ok(metadata) = fs::metadata(&path) {
                    total_size += metadata.len();
                }

                // Get access count (if we can read the file)
                if let Ok(json) = fs::read_to_string(&path) {
                    if let Ok(cache_entry) = serde_json::from_str::<ExpressionCacheEntry>(&json) {
                        avg_access_count += cache_entry.access_count as f64;
                    }
                }
            }
        }

        if total_entries > 0 {
            avg_access_count /= total_entries as f64;
        }

        Ok(CacheStats {
            total_entries,
            total_size_bytes: total_size,
            avg_access_count,
            max_size: self.max_cache_size,
        })
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size_bytes: u64,
    pub avg_access_count: f64,
    pub max_size: usize,
}

/// Helper function to create FactorMetadata from expression and fitness
pub fn create_factor_metadata(
    expr: &str,
    fitness: &MultiObjectiveFitness,
    eval_period: (String, String),
    n_assets: usize,
    n_days: usize,
) -> FactorMetadata {
    let timestamp = PersistenceManager::current_timestamp();
    let id = format!("factor_{}", timestamp); // Will be replaced by proper ID

    FactorMetadata {
        id,
        expression: expr.to_string(),
        created_at: timestamp,
        updated_at: timestamp,
        metrics: PerformanceMetrics {
            ic_mean: fitness.ic_score,
            ic_ir: fitness.ir_score,
            turnover: fitness.turnover_penalty,
            complexity_penalty: fitness.complexity_penalty,
            combined_score: fitness.combined_score,
            eval_period,
            n_assets,
            n_days,
            custom_metrics: HashMap::new(),
        },
        tags: vec![],
        gp_config: None,
        dataset_info: None,
    }
}

// Re-export from gp::history
pub use crate::gp::history::{GPHistoryRecord, PopulationStats, create_gp_history_record};

// Re-export al types and functions
pub use crate::al::parser::{AlFactor, AlParser};

impl PersistenceManager {
    /// Load factors from .al files in ~/.alfars/ directory
    pub fn load_from_al(&mut self) -> Result<Vec<AlFactor>, String> {
        let factors = AlParser::load_from_default_dir()?;
        Ok(factors)
    }

    /// Save a factor to .al file in ~/.alfars/ directory
    pub fn save_to_al(&self, factor: &AlFactor, filename: Option<&str>) -> Result<PathBuf, String> {
        AlParser::save_to_default_dir(factor, filename)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_factor_metadata_creation() {
        let fitness = MultiObjectiveFitness {
            ic_score: 0.15,
            ir_score: 2.5,
            turnover_penalty: 0.1,
            complexity_penalty: 0.05,
            combined_score: 0.8,
        };

        let metadata = create_factor_metadata(
            "close / open - 1.0",
            &fitness,
            ("2023-01-01".to_string(), "2023-12-31".to_string()),
            100,
            252,
        );

        assert_eq!(metadata.metrics.ic_mean, 0.15);
        assert_eq!(metadata.metrics.ic_ir, 2.5);
        assert_eq!(metadata.expression, "close / open - 1.0");
    }

    #[test]
    fn test_persistence_manager_basic() {
        let temp_dir = tempdir().unwrap();
        let mut manager = PersistenceManager::new(temp_dir.path()).unwrap();

        let fitness = MultiObjectiveFitness {
            ic_score: 0.1,
            ir_score: 1.5,
            turnover_penalty: 0.05,
            complexity_penalty: 0.03,
            combined_score: 0.7,
        };

        let metadata = create_factor_metadata(
            "high - low",
            &fitness,
            ("2023-01-01".to_string(), "2023-12-31".to_string()),
            50,
            200,
        );

        // Test saving
        manager.save_factor(&metadata).unwrap();

        // Test loading
        let loaded = manager.load_factor(&metadata.id).unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.expression, "high - low");
        assert_eq!(loaded.metrics.ic_mean, 0.1);
    }

    #[test]
    fn test_expression_cache() {
        let temp_dir = tempdir().unwrap();
        let cache_dir = temp_dir.path().join("cache");
        let mut cache_manager = ExpressionCacheManager::new(cache_dir).unwrap();

        let metrics = PerformanceMetrics {
            ic_mean: 0.12,
            ic_ir: 1.8,
            turnover: 0.08,
            complexity_penalty: 0.04,
            combined_score: 0.75,
            eval_period: ("2023-01-01".to_string(), "2023-12-31".to_string()),
            n_assets: 100,
            n_days: 252,
            custom_metrics: HashMap::new(),
        };

        // Test saving
        let expr = "close * volume";
        cache_manager
            .save_expression(expr, (10, 20), None, &metrics)
            .unwrap();

        // Test loading
        let loaded = cache_manager.load_expression(expr).unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.expression, "close * volume");
        assert_eq!(loaded.metrics.ic_mean, 0.12);
        assert_eq!(loaded.access_count, 2); // 1 from save, 1 from load
    }

    #[test]
    fn test_search_factors() {
        let temp_dir = tempdir().unwrap();
        let mut manager = PersistenceManager::new(temp_dir.path()).unwrap();

        // Create test factors
        let mut factor1 = create_factor_metadata(
            "factor1",
            &MultiObjectiveFitness {
                ic_score: 0.15,
                ir_score: 2.0,
                turnover_penalty: 0.1,
                complexity_penalty: 0.05,
                combined_score: 0.8,
            },
            ("2023-01-01".to_string(), "2023-12-31".to_string()),
            100,
            252,
        );
        factor1.tags = vec!["momentum".to_string(), "high_ic".to_string()];
        manager.factor_cache.insert(factor1.id.clone(), factor1);

        let mut factor2 = create_factor_metadata(
            "factor2",
            &MultiObjectiveFitness {
                ic_score: 0.05,
                ir_score: 0.8,
                turnover_penalty: 0.02,
                complexity_penalty: 0.01,
                combined_score: 0.5,
            },
            ("2023-01-01".to_string(), "2023-12-31".to_string()),
            100,
            252,
        );
        factor2.tags = vec!["value".to_string()];
        manager.factor_cache.insert(factor2.id.clone(), factor2);

        // Search by IC
        let high_ic_factors = manager.search_factors(Some(0.1), None, None, &[]);
        assert_eq!(high_ic_factors.len(), 1);
        assert!(high_ic_factors[0].metrics.ic_mean >= 0.1);

        // Search by tag
        let momentum_factors = manager.search_factors(None, None, None, &["momentum".to_string()]);
        assert_eq!(momentum_factors.len(), 1);
        assert!(momentum_factors[0].tags.contains(&"momentum".to_string()));
    }
}
