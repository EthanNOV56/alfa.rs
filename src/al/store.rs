//! Factor storage module using JSON files
//!
//! This module provides factor storage and retrieval using JSON files
//! for metadata and compressed binary files for factor values.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Factor record containing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorRecord {
    pub id: String,
    pub name: String,
    pub expression: String,
    pub category: String,
    pub description: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub version: u32,
    pub ic_mean: f64,
    pub ic_ir: f64,
    pub turnover: f64,
    pub combined_score: f64,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Factor values storage
pub struct FactorValues {
    pub id: String,
    pub version: u32,
    pub data: Vec<f64>,
    pub shape_days: usize,
    pub shape_assets: usize,
    pub created_at: u64,
}

/// Factor store using JSON files
pub struct FactorStore {
    data_dir: PathBuf,
}

/// Error type for factor store operations
#[derive(Debug)]
pub struct FactorStoreError {
    msg: String,
}

impl FactorStoreError {
    fn new(msg: &str) -> Self {
        Self {
            msg: msg.to_string(),
        }
    }
}

impl std::fmt::Display for FactorStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FactorStoreError: {}", self.msg)
    }
}

impl std::error::Error for FactorStoreError {}

type Result<T> = std::result::Result<T, FactorStoreError>;

impl FactorStore {
    /// Create a new FactorStore
    pub fn new(_clickhouse_url: &str, database: &str, data_dir: Option<&Path>) -> Result<Self> {
        let dir = data_dir
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::temp_dir().join(database));

        // Create directories
        fs::create_dir_all(&dir)
            .map_err(|e| FactorStoreError::new(&format!("Failed to create data dir: {}", e)))?;

        fs::create_dir_all(dir.join("factors"))
            .map_err(|e| FactorStoreError::new(&format!("Failed to create factors dir: {}", e)))?;

        fs::create_dir_all(dir.join("values"))
            .map_err(|e| FactorStoreError::new(&format!("Failed to create values dir: {}", e)))?;

        Ok(Self { data_dir: dir })
    }

    /// Initialize - nothing to do for file-based storage
    pub fn init_tables(&mut self) -> Result<()> {
        Ok(())
    }

    /// Register a new factor
    pub fn register_factor(&mut self, record: &FactorRecord) -> Result<String> {
        let id = if record.id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            record.id.clone()
        };

        let mut record = record.clone();
        record.id = id.clone();

        // Save to JSON file
        let path = self.data_dir.join("factors").join(format!("{}.json", id));
        let json = serde_json::to_string_pretty(&record)
            .map_err(|e| FactorStoreError::new(&format!("JSON error: {}", e)))?;
        fs::write(&path, json).map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;

        Ok(id)
    }

    /// Update an existing factor
    pub fn update_factor(&mut self, id: &str, record: &FactorRecord) -> Result<()> {
        let path = self.data_dir.join("factors").join(format!("{}.json", id));
        let json = serde_json::to_string_pretty(&record)
            .map_err(|e| FactorStoreError::new(&format!("JSON error: {}", e)))?;
        fs::write(&path, json).map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;

        Ok(())
    }

    /// Get a factor by ID
    pub fn get_factor(&self, id: &str) -> Result<Option<FactorRecord>> {
        let path = self.data_dir.join("factors").join(format!("{}.json", id));

        if !path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(&path)
            .map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;
        let record: FactorRecord = serde_json::from_str(&json)
            .map_err(|e| FactorStoreError::new(&format!("JSON error: {}", e)))?;

        Ok(Some(record))
    }

    /// Delete a factor by ID
    pub fn delete_factor(&mut self, id: &str) -> Result<bool> {
        let path = self.data_dir.join("factors").join(format!("{}.json", id));

        if path.exists() {
            fs::remove_file(&path)
                .map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;
        }

        // Delete all value files for this factor
        let values_dir = self.data_dir.join("values");
        if values_dir.exists() {
            for entry in fs::read_dir(&values_dir)
                .map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?
            {
                let entry =
                    entry.map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(id) {
                    fs::remove_file(entry.path())
                        .map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;
                }
            }
        }

        Ok(true)
    }

    /// List factors with optional filters
    pub fn list_factors(
        &self,
        category: Option<&str>,
        tags: &[String],
    ) -> Result<Vec<FactorRecord>> {
        let factors_dir = self.data_dir.join("factors");
        let mut result = Vec::new();

        for entry in fs::read_dir(&factors_dir)
            .map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?
        {
            let entry = entry.map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false) {
                let json = fs::read_to_string(&path)
                    .map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;
                let record: FactorRecord = serde_json::from_str(&json)
                    .map_err(|e| FactorStoreError::new(&format!("JSON error: {}", e)))?;

                // Apply filters
                let matches_category = category.map(|c| record.category == c).unwrap_or(true);
                let matches_tags = tags.is_empty() || tags.iter().all(|t| record.tags.contains(t));

                if matches_category && matches_tags {
                    result.push(record);
                }
            }
        }

        Ok(result)
    }

    /// Search factors by performance metrics
    pub fn search_factors(
        &self,
        min_ic: Option<f64>,
        min_ir: Option<f64>,
    ) -> Result<Vec<FactorRecord>> {
        let mut result = self.list_factors(None, &[])?;

        if let Some(ic) = min_ic {
            result.retain(|r| r.ic_mean >= ic);
        }
        if let Some(ir) = min_ir {
            result.retain(|r| r.ic_ir >= ir);
        }

        // Sort by ic_mean descending
        result.sort_by(|a, b| b.ic_mean.partial_cmp(&a.ic_mean).unwrap());

        Ok(result)
    }

    /// Save factor values to disk
    pub fn save_values(
        &mut self,
        id: &str,
        version: u32,
        data: &[f64],
        shape_days: usize,
        shape_assets: usize,
    ) -> Result<()> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Convert f64 to bytes for compression
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Compress data with LZ4
        let compressed =
            lz4::block::compress(&bytes, Some(lz4::block::CompressionMode::DEFAULT), true)
                .map_err(|e| FactorStoreError::new(&format!("LZ4 compress error: {}", e)))?;

        // Save to file
        let data_path = self
            .data_dir
            .join("values")
            .join(format!("{}_{}.lz4", id, version));
        fs::write(&data_path, &compressed)
            .map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;

        // Save metadata
        let metadata_path = self
            .data_dir
            .join("values")
            .join(format!("{}_{}.meta.json", id, version));
        let metadata = serde_json::json!({
            "id": id,
            "version": version,
            "shape_days": shape_days,
            "shape_assets": shape_assets,
            "created_at": now,
            "data_path": data_path.to_string_lossy(),
        });
        let json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| FactorStoreError::new(&format!("JSON error: {}", e)))?;
        fs::write(&metadata_path, json)
            .map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;

        Ok(())
    }

    /// Load factor values from disk
    pub fn load_values(&self, id: &str, version: u32) -> Result<Option<FactorValues>> {
        let metadata_path = self
            .data_dir
            .join("values")
            .join(format!("{}_{}.meta.json", id, version));

        if !metadata_path.exists() {
            return Ok(None);
        }

        #[derive(Deserialize)]
        struct ValueMetadata {
            shape_days: usize,
            shape_assets: usize,
            created_at: u64,
            data_path: String,
        }

        let json = fs::read_to_string(&metadata_path)
            .map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;
        let metadata: ValueMetadata = serde_json::from_str(&json)
            .map_err(|e| FactorStoreError::new(&format!("JSON error: {}", e)))?;

        let compressed = fs::read(&metadata.data_path)
            .map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;
        let size = metadata.shape_days * metadata.shape_assets * 8;
        let bytes = lz4::block::decompress(&compressed, Some(size as i32))
            .map_err(|e| FactorStoreError::new(&format!("LZ4 decompress error: {}", e)))?;

        // Convert bytes back to f64
        let data: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(Some(FactorValues {
            id: id.to_string(),
            version,
            data,
            shape_days: metadata.shape_days,
            shape_assets: metadata.shape_assets,
            created_at: metadata.created_at,
        }))
    }

    /// Export factor to JSON file
    pub fn export_factor(&self, id: &str, path: &Path) -> Result<FactorRecord> {
        let record = self
            .get_factor(id)?
            .ok_or_else(|| FactorStoreError::new(&format!("Factor not found: {}", id)))?;

        let json = serde_json::to_string_pretty(&record)
            .map_err(|e| FactorStoreError::new(&format!("JSON error: {}", e)))?;
        fs::write(path, json).map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;

        Ok(record)
    }

    /// Import factor from JSON file
    pub fn import_factor(&mut self, path: &Path) -> Result<FactorRecord> {
        let json = fs::read_to_string(path)
            .map_err(|e| FactorStoreError::new(&format!("IO error: {}", e)))?;
        let record: FactorRecord = serde_json::from_str(&json)
            .map_err(|e| FactorStoreError::new(&format!("JSON error: {}", e)))?;

        self.register_factor(&record)?;

        Ok(record)
    }
}
