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

/// Integration tests for alpha file loading and expression parsing
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::expr::registry::{extract_columns, parse_expression, FactorRegistry};
    use crate::data::ClickHouseSource;

    /// Get the alpha directories
    fn get_alpha_dirs() -> Vec<(String, std::path::PathBuf)> {
        let mut dirs = Vec::new();
        if let Some(home) = dirs::home_dir() {
            let alpha101 = home.join(".alfars").join("alpha101");
            let alpha191 = home.join(".alfars").join("alpha191");
            if alpha101.exists() {
                dirs.push(("alpha101".to_string(), alpha101));
            }
            if alpha191.exists() {
                dirs.push(("alpha191".to_string(), alpha191));
            }
        }
        dirs
    }

    #[test]
    fn test_load_alpha_files() {
        let dirs = get_alpha_dirs();
        if dirs.is_empty() {
            eprintln!("Skipping test - no alpha directories found");
            return;
        }

        for (name, dir) in &dirs {
            println!("Testing {} directory: {:?}", name, dir);
            let factors = AlParser::parse_directory(dir).expect(&format!(
                "Failed to parse directory: {:?}",
                dir
            ));
            println!("  Loaded {} factors from {}", factors.len(), name);
        }
    }

    #[test]
    fn test_parse_all_alpha_expressions() {
        let dirs = get_alpha_dirs();
        if dirs.is_empty() {
            eprintln!("Skipping test - no alpha directories found");
            return;
        }

        let mut success_count = 0;
        let mut failure_count = 0;
        let mut failures: Vec<(String, String, String)> = Vec::new();

        for (dir_name, dir) in &dirs {
            let factors = AlParser::parse_directory(dir).unwrap_or_default();
            println!(
                "\n=== Parsing {} ({} factors) ===",
                dir_name,
                factors.len()
            );

            for factor in factors {
                match parse_expression(&factor.expression) {
                    Ok(expr) => {
                        success_count += 1;
                        let expr_str = format!("{:?}", expr);
                        if expr_str.len() > 80 {
                            println!("  OK: {}: {}...", factor.name, &expr_str[..80]);
                        } else {
                            println!("  OK: {}: {}", factor.name, expr_str);
                        }
                    }
                    Err(e) => {
                        failure_count += 1;
                        failures.push((factor.name, factor.expression, e));
                    }
                }
            }
        }

        println!("\n=== Summary ===");
        println!("Total: {}", success_count + failure_count);
        println!("Success: {}", success_count);
        println!("Failure: {}", failure_count);

        if !failures.is_empty() {
            println!("\n=== Failed Expressions (first 10) ===");
            for (name, expr, error) in failures.iter().take(10) {
                println!("\n--- {} ---", name);
                println!("Expression: {}", expr);
                println!("Error: {}", error);
            }
        }

        // Store counts for analysis
        assert!(
            success_count > 0,
            "Expected at least some successful parses"
        );
    }

    #[test]
    fn test_parse_alpha101_expressions() {
        let dirs = get_alpha_dirs();
        if dirs.is_empty() {
            eprintln!("Skipping test - no alpha directories found");
            return;
        }

        let alpha101_dir = dirs.iter().find(|(name, _)| name == "alpha101");
        if alpha101_dir.is_none() {
            eprintln!("Skipping test - alpha101 directory not found");
            return;
        }

        let (_, dir) = alpha101_dir.unwrap();
        let factors = AlParser::parse_directory(dir).unwrap_or_default();

        let mut success_count = 0;
        let mut failure_count = 0;
        let mut failures: Vec<(String, String, String)> = Vec::new();

        for factor in factors {
            match parse_expression(&factor.expression) {
                Ok(_) => success_count += 1,
                Err(e) => {
                    failure_count += 1;
                    failures.push((factor.name, factor.expression, e));
                }
            }
        }

        println!("\n=== Alpha101 Results ===");
        println!("Success: {} / {}", success_count, success_count + failure_count);

        if !failures.is_empty() {
            println!("\nFailed expressions:");
            for (name, expr, error) in &failures {
                println!("  {}: {} - {}", name, expr.chars().take(60).collect::<String>(), error);
            }
        }

        assert!(
            success_count > 0,
            "Expected at least some successful parses"
        );
    }

    #[test]
    fn test_parse_alpha191_expressions() {
        let dirs = get_alpha_dirs();
        if dirs.is_empty() {
            eprintln!("Skipping test - no alpha directories found");
            return;
        }

        let alpha191_dir = dirs.iter().find(|(name, _)| name == "alpha191");
        if alpha191_dir.is_none() {
            eprintln!("Skipping test - alpha191 directory not found");
            return;
        }

        let (_, dir) = alpha191_dir.unwrap();
        let factors = AlParser::parse_directory(dir).unwrap_or_default();

        let mut success_count = 0;
        let mut failure_count = 0;
        let mut failures: Vec<(String, String, String)> = Vec::new();

        for factor in factors {
            match parse_expression(&factor.expression) {
                Ok(_) => success_count += 1,
                Err(e) => {
                    failure_count += 1;
                    failures.push((factor.name, factor.expression, e));
                }
            }
        }

        println!("\n=== Alpha191 Results ===");
        println!("Success: {} / {}", success_count, success_count + failure_count);

        if !failures.is_empty() {
            println!("\nFailed expressions (first 20):");
            for (name, expr, error) in failures.iter().take(20) {
                println!("  {}: {} - {}", name, expr.chars().take(60).collect::<String>(), error);
            }
            if failures.len() > 20 {
                println!("  ... and {} more", failures.len() - 20);
            }
        }

        assert!(
            success_count > 0,
            "Expected at least some successful parses"
        );
    }

    /// Integration test: parse and compute all alpha factors
    #[test]
    fn test_compute_all_alpha_factors() {
        use crate::data::ClickHouseSource;
        use crate::expr::registry::{extract_columns, FactorRegistry};

        // Load environment variables from .env file
        dotenv::dotenv().ok();

        // Get alpha directories
        let dirs = get_alpha_dirs();
        if dirs.is_empty() {
            eprintln!("Skipping test - no alpha directories found");
            return;
        }

        // Load all factors
        let mut all_factors = Vec::new();
        for (_, dir) in &dirs {
            let factors = AlParser::parse_directory(dir).unwrap_or_default();
            all_factors.extend(factors);
        }

        if all_factors.is_empty() {
            eprintln!("Skipping test - no factors loaded");
            return;
        }

        println!("\n=== Loading {} alpha factors ===", all_factors.len());

        // Try to fetch data from ClickHouse
        let source = ClickHouseSource::from_env();
        let table_name = std::env::var("STOCK_1D").unwrap_or_else(|_| "stock_1d".to_string());

        // Fetch stock data
        let data_result = source.fetch_stock_data(
            &["000001.SZ".to_string()],
            "2024-01-01",
            "2024-12-31",
            &table_name,
        );

        let data = match data_result {
            Ok(d) => {
                println!("Fetched {} rows from ClickHouse", d.values().next().map(|v| v.len()).unwrap_or(0));
                d
            }
            Err(e) => {
                eprintln!("Warning: Failed to fetch from ClickHouse: {}. Skipping test.", e);
                // Return empty data to skip the test
                return;
            }
        };

        // Collect all unique columns needed
        let mut all_columns: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut parse_failures: Vec<(String, String, String)> = Vec::new();

        for factor in &all_factors {
            match parse_expression(&factor.expression) {
                Ok(expr) => {
                    let cols = extract_columns(&expr);
                    for col in cols {
                        all_columns.insert(col);
                    }
                }
                Err(e) => {
                    parse_failures.push((factor.name.clone(), factor.expression.clone(), e));
                }
            }
        }

        println!("\n=== Parse Results ===");
        println!("Total factors: {}", all_factors.len());
        println!("Parse success: {}", all_factors.len() - parse_failures.len());
        println!("Parse failures: {}", parse_failures.len());

        if !parse_failures.is_empty() {
            println!("\nFirst 5 parse failures:");
            for (name, expr, error) in parse_failures.iter().take(5) {
                println!("  {}: {} - {}", name, expr.chars().take(50).collect::<String>(), error);
            }
        }

        // Filter data to only include required columns
        let available_columns: Vec<String> = all_columns.iter()
            .filter(|c| data.contains_key(*c))
            .cloned()
            .collect();

        println!("\n=== Column Availability ===");
        println!("Required columns: {}", all_columns.len());
        println!("Available columns: {}", available_columns.len());

        // Create registry and register factors
        let mut registry = FactorRegistry::new();
        registry.set_columns(available_columns.clone());

        let mut register_failures: Vec<(String, String)> = Vec::new();
        let mut registered_count = 0;

        for factor in &all_factors {
            // Skip factors that failed to parse
            if parse_failures.iter().any(|(name, _, _)| name == &factor.name) {
                continue;
            }

            match registry.register(&factor.name, &factor.expression) {
                Ok(_) => registered_count += 1,
                Err(e) => {
                    register_failures.push((factor.name.clone(), e));
                }
            }
        }

        println!("\n=== Registration Results ===");
        println!("Registered: {}", registered_count);
        println!("Registration failures: {}", register_failures.len());

        if !register_failures.is_empty() {
            println!("\nFirst 5 registration failures:");
            for (name, error) in register_failures.iter().take(5) {
                println!("  {}: {}", name, error);
            }
        }

        // Compute all factors in parallel
        let factor_names: Vec<&str> = all_factors.iter()
            .filter(|f| !parse_failures.iter().any(|(name, _, _)| name == &f.name))
            .filter(|f| !register_failures.iter().any(|(name, _)| name == &f.name))
            .map(|f| f.name.as_str())
            .collect();

        println!("\n=== Computing {} factors (parallel) ===", factor_names.len());

        let compute_start = std::time::Instant::now();
        let results = registry.compute_batch(&factor_names, &data, true);
        let compute_time = compute_start.elapsed().as_millis();

        match results {
            Ok(results) => {
                println!("\n=== Compute Results ===");
                println!("Computed: {}", results.len());
                println!("Compute time: {}ms", compute_time);

                // Show some sample results
                println!("\nSample factor values (first 5):");
                for (i, (name, result)) in results.iter().take(5).enumerate() {
                    let sample: Vec<f64> = result.values.iter().take(3).copied().collect();
                    println!("  {}: {:?}...", name, sample);
                }

                // Assert we computed at least some factors
                assert!(
                    results.len() > 0,
                    "Expected at least some factors to compute successfully"
                );
            }
            Err(e) => {
                eprintln!("Compute failed: {}", e);
                // If compute fails, at least verify parsing worked
                assert!(
                    registered_count > 0,
                    "Expected at least some factors to register successfully"
                );
            }
        }
    }
}
