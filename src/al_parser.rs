//! .al file parser for factor expression storage
//!
//! This module provides functionality to parse and save factor expressions
//! in human-readable TOML format (.al files).

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Factor stored in .al file format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlFactor {
    /// Factor name
    pub name: String,
    /// Factor expression (the core formula)
    pub expression: String,
    /// Human-readable description
    #[serde(default)]
    pub description: String,
    /// Dimension: return, price, volume, ratio, dimensionless
    #[serde(default = "default_dimension")]
    pub dimension: String,
    /// Tags for organization
    #[serde(default)]
    pub tags: Vec<String>,
    /// Whether this factor is read-only (from ~/.alfars/)
    #[serde(default)]
    pub readonly: bool,
}

fn default_dimension() -> String {
    "dimensionless".to_string()
}

/// .al file parser for factor expressions
pub struct AlParser;

impl AlParser {
    /// Get the default directory (~/.alfars/)
    pub fn default_dir() -> PathBuf {
        dirs::home_dir()
            .map(|p| p.join(".alfars"))
            .unwrap_or_else(|| PathBuf::from(".alfars"))
    }

    /// Get the user directory (~/.alfars/user/)
    pub fn user_dir() -> PathBuf {
        Self::default_dir().join("user")
    }

    /// Parse a single .al file
    pub fn parse_file(path: &Path) -> Result<AlFactor, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file {}: {}", path.display(), e))?;

        let factor: AlFactor = toml::from_str(&content)
            .map_err(|e| format!("Failed to parse TOML in {}: {}", path.display(), e))?;

        Ok(factor)
    }

    /// Parse all .al files in a directory
    pub fn parse_directory(path: &Path) -> Result<Vec<AlFactor>, String> {
        if !path.exists() {
            return Ok(vec![]);
        }

        let mut factors = Vec::new();

        for entry in fs::read_dir(path)
            .map_err(|e| format!("Failed to read directory {}: {}", path.display(), e))?
        {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("Warning: Failed to read entry: {}", e);
                    continue;
                }
            };

            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("al") {
                match Self::parse_file(&path) {
                    Ok(factor) => factors.push(factor),
                    Err(e) => {
                        eprintln!("Warning: Failed to parse {}: {}", path.display(), e);
                    }
                }
            }
        }

        Ok(factors)
    }

    /// Save a factor to a .al file
    /// Returns the path where the file was saved
    pub fn save_factor(
        factor: &AlFactor,
        dir: &Path,
        filename: Option<&str>,
    ) -> Result<PathBuf, String> {
        // Create directory if it doesn't exist
        fs::create_dir_all(dir)
            .map_err(|e| format!("Failed to create directory {}: {}", dir.display(), e))?;

        // Generate filename from name or use provided one
        let filename = filename.unwrap_or(&factor.name);
        let safe_filename = filename
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>();

        let file_path = dir.join(format!("{}.al", safe_filename));

        // Serialize to TOML
        let toml_content = toml::to_string_pretty(factor)
            .map_err(|e| format!("Failed to serialize factor: {}", e))?;

        fs::write(&file_path, toml_content)
            .map_err(|e| format!("Failed to write file {}: {}", file_path.display(), e))?;

        Ok(file_path)
    }

    /// Load factors from the default ~/.alfars/ directory
    pub fn load_from_default_dir() -> Result<Vec<AlFactor>, String> {
        let default_dir = Self::default_dir();
        Self::parse_directory(&default_dir)
    }

    /// Load all factors from both ~/.alfars/ (readonly) and ~/.alfars/user/ (writable)
    /// Files in the main directory are marked as readonly, files in user/ are writable
    /// Also loads from subdirectories like alpha101/, alpha191/
    pub fn load_all_with_readonly_flag() -> Result<Vec<AlFactor>, String> {
        let mut all_factors = Vec::new();
        let default_dir = Self::default_dir();
        let user_dir = Self::user_dir();

        // Helper function to recursively load .al files from a directory
        fn load_from_dir(dir: &Path, readonly: bool, skip_dirs: &[&str]) -> Vec<AlFactor> {
            let mut factors = Vec::new();
            if !dir.exists() {
                return factors;
            }

            let entries = match fs::read_dir(dir) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("Warning: Failed to read directory {}: {}", dir.display(), e);
                    return factors;
                }
            };

            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    // Skip certain directories
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if !skip_dirs.contains(&name) {
                            factors.extend(load_from_dir(&path, readonly, skip_dirs));
                        }
                    }
                } else if path.extension().and_then(|s| s.to_str()) == Some("al") {
                    if let Ok(mut factor) = AlParser::parse_file(&path) {
                        factor.readonly = readonly;
                        factors.push(factor);
                    }
                }
            }
            factors
        }

        // Load from main directory and all subdirectories (readonly)
        if default_dir.exists() {
            let factors = load_from_dir(&default_dir, true, &["user"]);
            all_factors.extend(factors);
        }

        // Load from user directory (writable)
        if user_dir.exists() {
            for entry in fs::read_dir(&user_dir)
                .map_err(|e| format!("Failed to read directory {}: {}", user_dir.display(), e))?
            {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("al") {
                        if let Ok(mut factor) = Self::parse_file(&path) {
                            factor.readonly = false;
                            all_factors.push(factor);
                        }
                    }
                }
            }
        }

        Ok(all_factors)
    }

    /// Save a factor to the user directory (~/.alfars/user/)
    pub fn save_to_user_dir(factor: &AlFactor, filename: Option<&str>) -> Result<PathBuf, String> {
        let user_dir = Self::user_dir();
        // Always save as writable
        let mut factor = factor.clone();
        factor.readonly = false;
        Self::save_factor(&factor, &user_dir, filename)
    }

    /// Save a factor to the default ~/.alfars/ directory
    pub fn save_to_default_dir(
        factor: &AlFactor,
        filename: Option<&str>,
    ) -> Result<PathBuf, String> {
        let default_dir = Self::default_dir();
        Self::save_factor(factor, &default_dir, filename)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parse_al_file() {
        let toml_content = r#"
name = "Alpha001"
description = "Time series rank of max power returns over 5 days"
dimension = "return"
tags = ["momentum", "alpha101"]

expression = "rank(ts_argmax(power(returns, 2), 5)) - 0.5"
"#;

        let factor: AlFactor = toml::from_str(toml_content).unwrap();
        assert_eq!(factor.name, "Alpha001");
        assert_eq!(
            factor.expression,
            "rank(ts_argmax(power(returns, 2), 5)) - 0.5"
        );
        assert_eq!(factor.dimension, "return");
        assert_eq!(factor.tags, vec!["momentum", "alpha101"]);
    }

    #[test]
    fn test_save_and_load_factor() {
        let temp_dir = tempdir().unwrap();

        let factor = AlFactor {
            name: "TestAlpha".to_string(),
            expression: "rank(close)".to_string(),
            description: "Test factor".to_string(),
            dimension: "price".to_string(),
            tags: vec!["test".to_string()],
            readonly: false,
        };

        // Save
        let saved_path = AlParser::save_factor(&factor, temp_dir.path(), None).unwrap();
        assert!(saved_path.exists());

        // Load
        let loaded = AlParser::parse_file(&saved_path).unwrap();
        assert_eq!(loaded.name, "TestAlpha");
        assert_eq!(loaded.expression, "rank(close)");
    }

    #[test]
    fn test_parse_directory() {
        let temp_dir = tempdir().unwrap();

        // Create test files
        let factor1 = AlFactor {
            name: "Alpha1".to_string(),
            expression: "close".to_string(),
            description: "".to_string(),
            dimension: "price".to_string(),
            tags: vec![],
            readonly: false,
        };
        let factor2 = AlFactor {
            name: "Alpha2".to_string(),
            expression: "volume".to_string(),
            description: "".to_string(),
            dimension: "volume".to_string(),
            tags: vec![],
            readonly: false,
        };

        AlParser::save_factor(&factor1, temp_dir.path(), None).unwrap();
        AlParser::save_factor(&factor2, temp_dir.path(), None).unwrap();

        // Parse directory
        let factors = AlParser::parse_directory(temp_dir.path()).unwrap();
        assert_eq!(factors.len(), 2);
    }

    #[test]
    fn test_default_dimension() {
        let toml_content = r#"
name = "TestAlpha"
expression = "close"
"#;

        let factor: AlFactor = toml::from_str(toml_content).unwrap();
        assert_eq!(factor.dimension, "dimensionless");
    }
}
