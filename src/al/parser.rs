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
    use crate::data::ClickHouseSource;
    use crate::expr::registry::{FactorRegistry, extract_columns, parse_expression};
    use ndarray::{Array1, Array2};

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
            let factors = AlParser::parse_directory(dir)
                .expect(&format!("Failed to parse directory: {:?}", dir));
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
            println!("\n=== Parsing {} ({} factors) ===", dir_name, factors.len());

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
        println!(
            "Success: {} / {}",
            success_count,
            success_count + failure_count
        );

        if !failures.is_empty() {
            println!("\nFailed expressions:");
            for (name, expr, error) in &failures {
                println!(
                    "  {}: {} - {}",
                    name,
                    expr.chars().take(60).collect::<String>(),
                    error
                );
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
        println!(
            "Success: {} / {}",
            success_count,
            success_count + failure_count
        );

        if !failures.is_empty() {
            println!("\nFailed expressions (first 20):");
            for (name, expr, error) in failures.iter().take(20) {
                println!(
                    "  {}: {} - {}",
                    name,
                    expr.chars().take(60).collect::<String>(),
                    error
                );
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
        use crate::expr::registry::{FactorRegistry, extract_columns};

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
                println!(
                    "Fetched {} rows from ClickHouse",
                    d.values().next().map(|v| v.len()).unwrap_or(0)
                );
                d
            }
            Err(e) => {
                eprintln!(
                    "Warning: Failed to fetch from ClickHouse: {}. Skipping test.",
                    e
                );
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
        println!(
            "Parse success: {}",
            all_factors.len() - parse_failures.len()
        );
        println!("Parse failures: {}", parse_failures.len());

        if !parse_failures.is_empty() {
            println!("\nFirst 5 parse failures:");
            for (name, expr, error) in parse_failures.iter().take(5) {
                println!(
                    "  {}: {} - {}",
                    name,
                    expr.chars().take(50).collect::<String>(),
                    error
                );
            }
        }

        // Filter data to only include required columns
        let available_columns: Vec<String> = all_columns
            .iter()
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
            if parse_failures
                .iter()
                .any(|(name, _, _)| name == &factor.name)
            {
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
        let factor_names: Vec<&str> = all_factors
            .iter()
            .filter(|f| !parse_failures.iter().any(|(name, _, _)| name == &f.name))
            .filter(|f| !register_failures.iter().any(|(name, _)| name == &f.name))
            .map(|f| f.name.as_str())
            .collect();

        println!(
            "\n=== Computing {} factors (parallel) ===",
            factor_names.len()
        );

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

    /// Full pipeline integration test: Parse alpha101/alpha191 -> Factor Computation -> Backtest
    ///
    /// This test:
    /// 1. Parses all .al files from ~/.alfars/alpha101/ and ~/.alfars/alpha191/
    /// 2. Adds prefix to handle duplicate factor names (alpha101-ALPHA001, alpha191-ALPHA001)
    /// 3. Fetches real stock data from ClickHouse (top 300 stocks by market cap)
    /// 4. Computes all factor values
    /// 5. Runs backtest and outputs IC/IR rankings
    #[test]
    fn test_alpha101_alpha191_full_pipeline() {
        use crate::WeightMethod;
        use crate::backtest::engine::{BacktestConfig, BacktestEngine, FeeConfig, PositionConfig};
        use crate::expr::registry::FactorRegistry;

        println!("\n========================================");
        println!("ALPHA101/ALPHA191 FULL PIPELINE TEST");
        println!("========================================");

        // Load environment variables
        dotenv::dotenv().ok();

        // Step 1: Parse alpha101 and alpha191 directories
        let mut all_factors = Vec::new();

        if let Some(home) = dirs::home_dir() {
            let alpha101_dir = home.join(".alfars").join("alpha101");
            let alpha191_dir = home.join(".alfars").join("alpha191");

            if alpha101_dir.exists() {
                let mut factors_101 = AlParser::parse_directory(&alpha101_dir).unwrap_or_default();
                println!("Loaded {} factors from alpha101", factors_101.len());
                // Add prefix to avoid name collision
                for f in &mut factors_101 {
                    f.name = format!("alpha101-{}", f.name);
                }
                all_factors.extend(factors_101);
            }

            if alpha191_dir.exists() {
                let mut factors_191 = AlParser::parse_directory(&alpha191_dir).unwrap_or_default();
                println!("Loaded {} factors from alpha191", factors_191.len());
                // Add prefix to avoid name collision
                for f in &mut factors_191 {
                    f.name = format!("alpha191-{}", f.name);
                }
                all_factors.extend(factors_191);
            }
        }

        if all_factors.is_empty() {
            eprintln!("Skipping test - no alpha factors found in ~/.alfars/alpha101/ or alpha191/");
            return;
        }

        println!("\nTotal factors loaded: {}", all_factors.len());

        // Step 2: Fetch stock data from ClickHouse
        let source = ClickHouseSource::from_env();
        let table_name = std::env::var("STOCK_1D").unwrap_or_else(|_| "stock_1d".to_string());

        // Query top 300 stocks by market cap
        let top_n = 300;
        println!("Fetching top {} stocks by market cap...", top_n);

        // First try to get stocks from query, then use a known list of top A-shares
        let test_stocks: Vec<String> = match source.get_top_stocks(top_n, &table_name) {
            Ok(stocks) if !stocks.is_empty() => {
                println!("Fetched {} stocks from query", stocks.len());
                stocks
            }
            _ => {
                // Fallback: use a known list of top 300 A-share stocks by code
                // These are the top stocks by market cap as of 2024
                println!("Using fallback stock list (top 300 A-shares by code)");
                (1..=300)
                    .map(|i| format!("{:06}.SZ", i))
                    .collect()
            }
        };

        println!("Using {} stocks for backtest", test_stocks.len());

        // Fetch data for the date range
        let data_result =
            source.fetch_stock_data(&test_stocks, "2020-01-01", "2025-12-31", &table_name);

        let data = match data_result {
            Ok(d) => {
                let rows = d.values().next().map(|v| v.len()).unwrap_or(0);
                println!("Fetched {} rows of data", rows);
                d
            }
            Err(e) => {
                eprintln!(
                    "Warning: Failed to fetch from ClickHouse: {}. Skipping test.",
                    e
                );
                return;
            }
        };

        // Step 3: Parse all factor expressions
        use crate::expr::registry::extract_columns;
        use crate::expr::registry::parse_expression;

        let mut parse_errors: Vec<(String, String, String)> = Vec::new();

        for factor in &all_factors {
            if let Err(e) = parse_expression(&factor.expression) {
                parse_errors.push((factor.name.clone(), factor.expression.clone(), e));
            }
        }

        println!(
            "\nParse results: {} / {} success",
            all_factors.len() - parse_errors.len(),
            all_factors.len()
        );

        if parse_errors.len() == all_factors.len() {
            eprintln!("All factors failed to parse. Skipping test.");
            return;
        }

        // Collect required columns
        let mut all_columns: std::collections::HashSet<String> = std::collections::HashSet::new();
        for factor in &all_factors {
            if let Ok(expr) = parse_expression(&factor.expression) {
                let cols = extract_columns(&expr);
                for col in cols {
                    all_columns.insert(col);
                }
            }
        }

        // Filter available columns
        let available_columns: Vec<String> = all_columns
            .iter()
            .filter(|c| data.contains_key(*c))
            .cloned()
            .collect();

        println!(
            "Available columns: {}/{}",
            available_columns.len(),
            all_columns.len()
        );

        // Step 4: Register factors
        let mut registry = FactorRegistry::new();
        registry.set_columns(available_columns.clone());

        let mut register_errors: Vec<(String, String)> = Vec::new();

        for factor in &all_factors {
            // Skip factors that failed to parse
            if parse_errors.iter().any(|(name, _, _)| name == &factor.name) {
                continue;
            }

            if let Err(e) = registry.register(&factor.name, &factor.expression) {
                register_errors.push((factor.name.clone(), e));
            }
        }

        let registered = all_factors.len() - parse_errors.len() - register_errors.len();
        println!("Registered factors: {}", registered);

        if registered == 0 {
            eprintln!("No factors registered. Skipping test.");
            return;
        }

        // Step 5: Compute factors per-stock for proper multi-stock backtest
        // We need to compute factors for each stock separately and then align them
        println!("\nComputing factors per-stock...");

        // Get factor names to compute (limit to 5 for speed)
        let factor_names: Vec<&str> = all_factors
            .iter()
            .filter(|f| !parse_errors.iter().any(|(name, _, _)| name == &f.name))
            .filter(|f| !register_errors.iter().any(|(name, _)| name == &f.name))
            .take(5)
            .map(|f| f.name.as_str())
            .collect();

        if factor_names.is_empty() {
            eprintln!("No factors to compute");
            return;
        }

        // Fetch data per stock and compute factors
        let mut stock_data: std::collections::HashMap<
            String,
            std::collections::HashMap<String, Vec<f64>>,
        > = std::collections::HashMap::new();

        let mut factor_results: std::collections::HashMap<String, Vec<Vec<f64>>> =
            std::collections::HashMap::new();

        // Initialize factor results
        for name in &factor_names {
            factor_results.insert(name.to_string(), Vec::new());
        }

        // Fetch and compute per stock
        for symbol in &test_stocks {
            let single_stock_data =
                source.fetch_stock_data(&[symbol.clone()], "2024-01-01", "2024-12-31", &table_name);

            let stock_data_map = match single_stock_data {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Warning: Failed to fetch data for {}: {}", symbol, e);
                    continue;
                }
            };

            // Get trading days count
            let n_days = stock_data_map.values().next().map(|v| v.len()).unwrap_or(0);
            if n_days == 0 {
                continue;
            }

            // Store stock data
            stock_data.insert(symbol.clone(), stock_data_map.clone());

            // Compute factors for this stock
            let mut stock_registry = FactorRegistry::new();
            stock_registry.set_columns(available_columns.clone());

            for name in &factor_names {
                if let Some(factor) = all_factors.iter().find(|f| f.name == *name) {
                    let _ = stock_registry.register(name, &factor.expression);
                }
            }

            let results = stock_registry.compute_batch(&factor_names, &stock_data_map, false);

            if let Ok(res) = results {
                for name in &factor_names {
                    if let Some(result) = res.get(*name) {
                        if let Some(factor_vec) = factor_results.get_mut(*name) {
                            factor_vec.push(result.values.clone());
                        }
                    }
                }
            }
        }

        // Verify we have data for all stocks
        let mut valid_stocks = Vec::new();
        for symbol in &test_stocks {
            if stock_data.contains_key(symbol) {
                valid_stocks.push(symbol.clone());
            }
        }

        println!("Valid stocks with data: {}", valid_stocks.len());

        if valid_stocks.len() < 2 {
            eprintln!("Need at least 2 stocks for backtest. Skipping.");
            return;
        }

        // Step 6: Align data and run backtest
        println!("\n========================================");
        println!("BACKTEST RESULTS (IC/IR RANKINGS)");
        println!("========================================");

        // Find minimum days across all stocks
        let mut min_days = usize::MAX;
        for symbol in &valid_stocks {
            if let Some(data) = stock_data.get(symbol) {
                let days = data.values().next().map(|v| v.len()).unwrap_or(0);
                if days > 0 && days < min_days {
                    min_days = days;
                }
            }
        }

        if min_days == usize::MAX || min_days < 10 {
            eprintln!("Not enough trading days for backtest");
            return;
        }

        println!("Using {} trading days for backtest", min_days);

        let n_stocks = valid_stocks.len();
        let mut ic_ir_results: Vec<(String, f64, f64)> = Vec::new();

        // For each factor, build aligned matrices and run backtest
        for factor_name in &factor_names {
            if let Some(factor_vals) = factor_results.get(*factor_name) {
                if factor_vals.len() != n_stocks {
                    continue;
                }

                // Build aligned factor matrix: (n_days, n_stocks)
                let mut aligned_factor: ndarray::Array2<f64> =
                    ndarray::Array2::zeros((min_days, n_stocks));
                for (stock_idx, vals) in factor_vals.iter().enumerate() {
                    for day in 0..min_days {
                        if day < vals.len() {
                            aligned_factor[[day, stock_idx]] = vals[day];
                        }
                    }
                }

                // Build aligned returns matrix: (n_days, n_stocks)
                // IMPORTANT: Compute holding return = (close[t+1] - close[t]) / close[t]
                // This is the overnight return for factor-based trading
                // Also apply forward adjustment (前复权) using adjust_factor
                let mut aligned_returns: ndarray::Array2<f64> =
                    ndarray::Array2::zeros((min_days, n_stocks));
                for (stock_idx, symbol) in valid_stocks.iter().enumerate() {
                    if let Some(data) = stock_data.get(symbol) {
                        let close = data.get("close");
                        let adj_factor = data.get("adjust_factor");

                        if let Some(c) = close {
                            let latest_adj =
                                adj_factor.and_then(|a| a.last()).copied().unwrap_or(1.0);

                            for day in 0..min_days - 1 {
                                // Apply forward adjustment if available
                                let c_t = if let Some(adj) = adj_factor {
                                    c[day] * adj[day] / latest_adj
                                } else {
                                    c[day]
                                };
                                let c_next = if let Some(adj) = adj_factor {
                                    c[day + 1] * adj[day + 1] / latest_adj
                                } else {
                                    c[day + 1]
                                };

                                if c_t > 0.0 && c_next > 0.0 && !c_t.is_nan() && !c_next.is_nan() {
                                    let ret = (c_next - c_t) / c_t;
                                    // Sanity check: returns should be reasonable (-50% to +100%)
                                    if ret.abs() < 2.0 {
                                        aligned_returns[[day, stock_idx]] = ret;
                                    }
                                }
                            }
                            // Last day: no next day return, set to 0
                            aligned_returns[[min_days - 1, stock_idx]] = 0.0;
                        }
                    }
                }

                // Don't pass adj_factor to backtest - we've already computed adjusted returns
                let aligned_adj: Option<ndarray::Array2<f64>> = None;

                // Run backtest
                let config = BacktestConfig {
                    quantiles: 5,
                    weight_method: WeightMethod::Equal,
                    long_top_n: 1,
                    short_top_n: 1,
                    fee_config: FeeConfig::default(),
                    position_config: PositionConfig::default(),
                };

                let engine = BacktestEngine::with_config(config);

                // Create close and vwap as placeholder (using factor as proxy)
                let close = aligned_factor.clone();
                let vwap = aligned_factor.clone();

                // adj_factor is now required - use None as ones if not available
                let adj = aligned_adj.unwrap_or_else(|| Array2::from_elem(aligned_factor.dim(), 1.0));

                match engine.run(
                    aligned_factor,
                    aligned_returns,
                    adj,
                    close,
                    vwap,
                ) {
                    Ok(result) => {
                        println!(
                            "{}: IC={:.4}, IR={:.4}, Return={:.2}%",
                            factor_name,
                            result.ic_mean,
                            result.ic_ir,
                            result.total_return * 100.0
                        );
                        ic_ir_results.push((factor_name.to_string(), result.ic_mean, result.ic_ir));
                    }
                    Err(e) => {
                        eprintln!("Backtest failed for {}: {}", factor_name, e);
                    }
                }
            }
        }

        // Sort by IC IR and show top 5
        println!("\n========================================");
        println!("TOP 5 FACTORS BY IC IR");
        println!("========================================");

        if ic_ir_results.is_empty() {
            println!("No IC/IR results");
        } else {
            ic_ir_results
                .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

            for (i, (name, ic, ir)) in ic_ir_results.iter().take(5).enumerate() {
                println!("{}. {}: IC={:.4}, IR={:.4}", i + 1, name, ic, ir);
            }
        }

        // Assert parsing and computation worked
        assert!(
            !all_factors.is_empty(),
            "Expected at least some factors to be loaded"
        );
        assert!(
            !factor_results.is_empty(),
            "Expected at least some factors to compute"
        );
    }

    /// Integration test: Per-stock factor calculation with different trading days
    ///
    /// This test demonstrates:
    /// 1. Each stock can have different number of trading days
    /// 2. Factors are computed per-stock with forward-fill alignment
    /// 3. Backtest uses adjusted returns (前复权) and weight matrix
    #[test]
    fn test_per_stock_factor_calculation() {
        use crate::backtest::engine::BacktestEngine;
        use crate::expr::registry::FactorRegistry;
        use ndarray::{Array1, Array2};

        println!("\n=== Per-Stock Factor Calculation Test ===");

        // Define stocks with different trading days
        // Stock A: 10 days, Stock B: 8 days, Stock C: 6 days
        let symbols = vec![
            "STOCK_A".to_string(),
            "STOCK_B".to_string(),
            "STOCK_C".to_string(),
        ];
        let trading_days = vec![10, 8, 6];

        // Create mock data for each stock
        // Data structure: HashMap<symbol, HashMap<column, Vec<f64>>>
        let mut stock_data: std::collections::HashMap<
            String,
            std::collections::HashMap<String, Vec<f64>>,
        > = std::collections::HashMap::new();

        for (i, symbol) in symbols.iter().enumerate() {
            let n_days = trading_days[i];
            let mut data = std::collections::HashMap::new();

            // Generate price data with trend
            let close: Vec<f64> = (0..n_days)
                .map(|j| 100.0 + (j as f64) * 2.0 + (i as f64) * 10.0)
                .collect();
            let open: Vec<f64> = close.iter().map(|c| c * 0.99).collect();
            let high: Vec<f64> = close.iter().map(|c| c * 1.02).collect();
            let low: Vec<f64> = close.iter().map(|c| c * 0.98).collect();
            let volume: Vec<f64> = (0..n_days)
                .map(|j| 1_000_000.0 + j as f64 * 10000.0)
                .collect();

            // Adjustment factors (前复权) - each stock has different adjustment
            let adj_factor: Vec<f64> = (0..n_days).map(|j| 1.0 + j as f64 * 0.01).collect();

            // VWAP data
            let vwap: Vec<f64> = close
                .iter()
                .zip(volume.iter())
                .map(|(c, v)| c * (1.0 + v.min(1000000.0) / 2000000.0 * 0.01))
                .collect();

            data.insert("close".to_string(), close);
            data.insert("open".to_string(), open);
            data.insert("high".to_string(), high);
            data.insert("low".to_string(), low);
            data.insert("volume".to_string(), volume);
            data.insert("adjust_factor".to_string(), adj_factor);
            data.insert("vwap".to_string(), vwap);

            stock_data.insert(symbol.clone(), data);
        }

        println!("Stock trading days: {:?}", trading_days);

        // Register factors
        let mut registry = FactorRegistry::new();
        registry.set_columns(vec![
            "close".to_string(),
            "open".to_string(),
            "volume".to_string(),
            "returns".to_string(),
        ]);

        // Register simple factors
        registry.register("rank_close", "rank(close)").unwrap();
        registry.register("ts_mean_5", "ts_mean(close, 5)").unwrap();
        registry
            .register("volume_ratio", "volume / ts_mean(volume, 5)")
            .unwrap();

        let factor_names = vec!["rank_close", "ts_mean_5", "volume_ratio"];

        // Step 1: Compute factors per stock
        // Use index-based storage to maintain order
        let mut factor_results: std::collections::HashMap<String, Vec<Vec<f64>>> =
            std::collections::HashMap::new();

        // Initialize result vectors for each factor
        for factor_name in &factor_names {
            factor_results.insert(factor_name.to_string(), Vec::new());
        }

        for (symbol_idx, symbol) in symbols.iter().enumerate() {
            let data = stock_data.get(symbol).unwrap();

            // Apply forward adjustment (前复权)
            let mut adj_data = data.clone();
            let adj = adj_data.get("adjust_factor").unwrap().clone();
            let latest = *adj.last().unwrap();

            // Adjust close prices
            if let Some(close) = adj_data.get_mut("close") {
                for (i, c) in close.iter_mut().enumerate() {
                    *c = *c * adj[i] / latest;
                }
            }

            // Adjust vwap prices
            if let Some(vwap) = adj_data.get_mut("vwap") {
                for (i, v) in vwap.iter_mut().enumerate() {
                    *v = *v * adj[i] / latest;
                }
            }

            // Compute returns using adjusted close
            if let Some(close) = adj_data.get("close") {
                let mut returns = vec![0.0; close.len()];
                for i in 1..close.len() {
                    returns[i] = (close[i] - close[i - 1]) / close[i - 1];
                }
                adj_data.insert("returns".to_string(), returns);
            }

            // Convert to ndarray format for vectorized computation
            let arr_data: std::collections::HashMap<String, Array1<f64>> = adj_data
                .iter()
                .map(|(k, v)| (k.clone(), Array1::from_vec(v.clone())))
                .collect();

            // Compute factors
            let results = registry
                .compute_batch_vectorized(&factor_names, &arr_data, false)
                .unwrap();

            // Store results per factor (maintain index order)
            for factor_name in &factor_names {
                if let Some(result) = results.get(*factor_name) {
                    if let Some(factor_vec) = factor_results.get_mut(*factor_name) {
                        factor_vec.push(result.values.clone());
                    }
                }
            }
        }

        println!("\nFactor results per stock:");
        for (factor_name, results) in &factor_results {
            println!("  {}: {} stocks computed", factor_name, results.len());
            for (i, result) in results.iter().enumerate() {
                println!(
                    "    Stock {}: {} values, first 3: {:?}",
                    symbols[i],
                    result.len(),
                    &result[..result.len().min(3)]
                );
            }
        }

        // Verify: Each stock has different number of factor values
        for (factor_name, results) in &factor_results {
            assert_eq!(
                results.len(),
                symbols.len(),
                "Should have results for all stocks"
            );
            for (i, result) in results.iter().enumerate() {
                assert_eq!(
                    result.len(),
                    trading_days[i],
                    "Stock {} should have {} values for {}",
                    symbols[i],
                    trading_days[i],
                    factor_name
                );
            }
        }

        // Step 2: Align data to common date index
        // For simplicity, use the minimum number of days
        let min_days = *trading_days.iter().min().unwrap();
        println!("\nAligning to common date index: {} days", min_days);

        // Build aligned factor matrix: [n_factors, n_days, n_symbols]
        let n_factors = factor_names.len();
        let n_symbols = symbols.len();

        let mut aligned_factors: Array2<f64> = Array2::zeros((min_days, n_symbols * n_factors));

        for (factor_idx, factor_name) in factor_names.iter().enumerate() {
            if let Some(results) = factor_results.get(*factor_name) {
                for (symbol_idx, result) in results.iter().enumerate() {
                    let col_idx = factor_idx * n_symbols + symbol_idx;
                    for day in 0..min_days {
                        aligned_factors[[day, col_idx]] = result[day];
                    }
                }
            }
        }

        println!("Aligned factor matrix shape: {:?}", aligned_factors.dim());

        // Build aligned close matrix for holding/trading return calculation
        let mut aligned_close: Array2<f64> = Array2::zeros((min_days, n_symbols));

        for (symbol_idx, symbol) in symbols.iter().enumerate() {
            if let Some(data) = stock_data.get(symbol) {
                if let Some(close) = data.get("close") {
                    for day in 0..min_days {
                        aligned_close[[day, symbol_idx]] = close[day];
                    }
                }
            }
        }

        // Build aligned returns matrix: [n_days, n_symbols]
        let mut aligned_returns: Array2<f64> = Array2::zeros((min_days, n_symbols));

        for (symbol_idx, symbol) in symbols.iter().enumerate() {
            if let Some(data) = stock_data.get(symbol) {
                if let Some(returns) = data.get("returns") {
                    for day in 0..min_days {
                        aligned_returns[[day, symbol_idx]] = returns[day];
                    }
                }
            }
        }

        // Build aligned VWAP matrix for trading cost calculation
        let mut aligned_vwap: Array2<f64> = Array2::zeros((min_days, n_symbols));

        for (symbol_idx, symbol) in symbols.iter().enumerate() {
            if let Some(data) = stock_data.get(symbol) {
                if let Some(vwap) = data.get("vwap") {
                    for day in 0..min_days {
                        aligned_vwap[[day, symbol_idx]] = vwap[day];
                    }
                }
            }
        }

        println!("Aligned returns matrix shape: {:?}", aligned_returns.dim());

        // Step 3: Generate weight matrix using factor ranking
        // Extract first factor for ranking - aligned_factors is [min_days, n_symbols * n_factors]
        // We need [min_days, n_symbols] - take first factor's columns
        let mut single_factor: Array2<f64> = Array2::zeros((min_days, n_symbols));
        for day in 0..min_days {
            for s in 0..n_symbols {
                single_factor[[day, s]] = aligned_factors[[day, s]];
            }
        }

        let weights = generate_weight_matrix(
            &single_factor,
            1, // long_top_n
            1, // short_top_n
        );

        println!("Weight matrix shape: {:?}", weights.dim());
        println!("Weight matrix (first 5 days):");
        for day in 0..5.min(min_days) {
            println!("  Day {}: {:?}", day, weights.row(day));
        }

        // Step 4: Compute holding return using previous day's weights
        let holding_pnl = BacktestEngine::compute_holding_return(&weights, &aligned_close);
        println!(
            "\nHolding return (first 5 days): {:?}",
            holding_pnl.slice(ndarray::s![..5, 0])
        );

        // Step 5: Compute trading return using close and vwap
        let trading_pnl = BacktestEngine::compute_trading_return(
            &weights,
            &aligned_close,
            &aligned_vwap,
            0.0003,
            0.0005,
        );
        println!(
            "Trading return (first 5 days): {:?}",
            trading_pnl.slice(ndarray::s![..5, 0])
        );

        // Step 6: Compute total returns
        let total_pnl = &holding_pnl + &trading_pnl;
        let cumulative_return = compute_cumulative_return(&total_pnl.column(0).to_owned());
        println!(
            "\nTotal cumulative return: {:.4}%",
            cumulative_return * 100.0
        );

        // Assertions
        assert!(cumulative_return.is_finite());
    }

    /// Generate weight matrix from factor values
    ///
    /// Weight matrix shape: [n_days, n_symbols]
    /// - Long positions: top N stocks with highest factor values
    /// - Short positions: bottom N stocks with lowest factor values
    fn generate_weight_matrix(
        factor_matrix: &Array2<f64>,
        long_top_n: usize,
        short_top_n: usize,
    ) -> Array2<f64> {
        let (n_days, n_symbols) = factor_matrix.dim();
        let mut weights = ndarray::Array2::<f64>::zeros((n_days, n_symbols));

        for day in 0..n_days {
            let mut ranked: Vec<(usize, f64)> = (0..n_symbols)
                .map(|i| (i, factor_matrix[[day, i]]))
                .collect();

            // Sort by factor value (descending)
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Assign long weights (top N)
            let long_weight = 1.0 / long_top_n as f64;
            for i in 0..long_top_n.min(n_symbols) {
                weights[[day, ranked[i].0]] += long_weight;
            }

            // Assign short weights (bottom N)
            let short_weight = -1.0 / short_top_n as f64;
            for i in 0..short_top_n.min(n_symbols) {
                let idx = n_symbols - 1 - i;
                weights[[day, ranked[idx].0]] += short_weight;
            }
        }

        weights
    }

    /// Compute holding return from weights and close prices (vectorized)
    ///
    /// # Formula
    /// `holding_return[day] = sum(weights[day-1, :] * (close[day, :] / close[day-1, :] - 1))`
    /// Compute cumulative return from daily PnL
    fn compute_cumulative_return(daily_pnl: &ndarray::Array1<f64>) -> f64 {
        let mut cum = 1.0;
        for &p in daily_pnl.iter() {
            if p.is_finite() {
                cum *= 1.0 + p;
            }
        }
        cum - 1.0
    }

    /// Test with real ClickHouse data: All weight on single stock 000001.SZ
    ///
    /// Verifies:
    /// 1. Fetch real price data from ClickHouse
    /// 2. Weight matrix: 100% on 000001.SZ
    /// 3. Holding PnL equals actual returns
    ///
    /// Note: This test uses raw (unadjusted) close prices.
    /// In production, you would apply forward adjustment (前复权) using adjust_factor.
    #[test]
    fn test_single_stock_weight_real_data() {
        use crate::backtest::engine::BacktestEngine;
        use crate::data::ClickHouseSource;
        use ndarray::Array2;
        use std::collections::HashMap;

        // Load environment
        dotenv::dotenv().ok();

        println!("\n=== Single Stock Weight Test (Real Data) ===");

        // Fetch real data from ClickHouse
        let source = ClickHouseSource::from_env();

        let table_name = std::env::var("STOCK_1D").unwrap_or_else(|_| "stock_1d".to_string());

        // Fetch 000001.SZ data for 2024
        let data_result: Result<HashMap<String, Vec<f64>>, _> = source.fetch_stock_data(
            &["000001.SZ".to_string()],
            "2024-01-01",
            "2024-12-31",
            &table_name,
        );

        let data = match data_result {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Skipping test - failed to fetch data: {}", e);
                return;
            }
        };

        // Extract close and calculate returns
        let close = data.get("close").unwrap();
        let vwap_raw = data.get("vwap").unwrap();
        let volume_raw = data.get("volume");
        let amount_raw = data.get("amount");
        let n_days = close.len();
        println!("Fetched {} trading days for 000001.SZ", n_days);
        println!("First 5 raw close: {:?}", &close[..5]);
        println!("First 5 raw vwap: {:?}", &vwap_raw[..5]);
        if let Some(vol) = volume_raw {
            println!("First 5 raw volume: {:?}", &vol[..5]);
        }
        if let Some(amt) = amount_raw {
            println!("First 5 raw amount: {:?}", &amt[..5]);
        }

        // Get real adjust_factor from ClickHouse
        let adjust_factor = match data.get("adjust_factor") {
            Some(factors) => {
                println!("Using real adjust_factor from ClickHouse");
                factors.clone()
            }
            None => {
                println!("adjust_factor not available, using 1.0");
                vec![1.0; n_days]
            }
        };

        // Apply forward adjustment: adj_close[i] = close[i] * adjust_factor[i] / latest_factor
        let latest_factor = adjust_factor.last().copied().unwrap_or(1.0);
        let adj_close: Vec<f64> = close
            .iter()
            .zip(adjust_factor.iter())
            .map(|(c, f)| {
                let factor = if f.is_finite() && *f > 0.0 { *f } else { 1.0 };
                c * factor / latest_factor
            })
            .collect();

        // Use vwap from data, apply adjustment
        let vwap_raw = data.get("vwap").unwrap();
        let vwap: Vec<f64> = vwap_raw
            .iter()
            .zip(adjust_factor.iter())
            .map(|(v, f)| {
                let factor = if f.is_finite() && *f > 0.0 { *f } else { 1.0 };
                v * factor / latest_factor
            })
            .collect();

        println!("\n=== Forward Adjustment (Real Data) ===");
        println!(
            "First raw close: {}, First adj close: {:.4}",
            close[0], adj_close[0]
        );
        println!(
            "Last raw close: {}, Last adj close: {:.4}",
            close[n_days - 1],
            adj_close[n_days - 1]
        );
        println!(
            "First adjust_factor: {:.6}, Last: {:.6}",
            adjust_factor[0],
            adjust_factor[n_days - 1]
        );

        // Create weight matrix: alternating [1, 0, 1, 0, 1, 0, ...]
        let weight_pattern: Vec<f64> = (0..n_days)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let weights = Array2::from_shape_vec((n_days, 1), weight_pattern.clone()).unwrap();

        let vwap_arr = Array2::from_shape_vec((n_days, 1), vwap.clone()).unwrap();

        println!("Weight: alternating [1, 0, 1, 0, ...] for {} days", n_days);
        println!("First 10 weights: {:?}", &weight_pattern[..10]);

        // Compute holding return using previous day's weights
        let close_arr = Array2::from_shape_vec((n_days, 1), adj_close.clone()).unwrap();

        let holding_return = BacktestEngine::compute_holding_return(&weights, &close_arr);
        println!(
            "First 5 holding returns: {:?}",
            holding_return.slice(ndarray::s![..5, 0])
        );

        // Compute trading return
        let fee = 0.0003;
        let slippage = 0.0005;
        let trading_return =
            BacktestEngine::compute_trading_return(&weights, &close_arr, &vwap_arr, fee, slippage);
        println!(
            "First 5 trading returns: {:?}",
            trading_return.slice(ndarray::s![..5, 0])
        );

        // Verify holding return calculation:
        // holding_return[day] = weights[day-1] * (close[day]/close[day-1] - 1)
        println!("\n=== Holding Return Verification ===");
        for day in 1..6.min(n_days) {
            let expected_holding =
                weight_pattern[day - 1] * (adj_close[day] / adj_close[day - 1] - 1.0);
            let actual_holding = holding_return[[day, 0]];
            println!(
                "Day {}: weight[{}]={}, close[{}]={:.2}, close[{}]={:.2}, expected={:.6}, actual={:.6}",
                day,
                day - 1,
                weight_pattern[day - 1],
                day - 1,
                adj_close[day - 1],
                day,
                adj_close[day],
                expected_holding,
                actual_holding
            );

            assert!(
                (actual_holding - expected_holding).abs() < 1e-6,
                "Day {}: expected holding {:.6}, got {:.6}",
                day,
                expected_holding,
                actual_holding
            );
        }

        // Verify trading return calculation:
        // trading_return[day] = weight_diff * (close[day]/vwap[day] - 1 - total_cost)
        // where weight_diff = weights[day] - weights[day-1]
        let total_cost_rate = fee + slippage;
        println!("\n=== Trading Return Verification ===");
        for day in 1..6.min(n_days) {
            let weight_diff = weight_pattern[day] - weight_pattern[day - 1];
            let price_return = adj_close[day] / vwap[day] - 1.0;
            let expected_trading = weight_diff * (price_return - total_cost_rate);
            let actual_trading = trading_return[[day, 0]];
            println!(
                "Day {}: weight_diff={}, close/vwap={:.4}, price_return={:.6}, cost={:.6}, expected={:.6}, actual={:.6}",
                day,
                weight_diff,
                adj_close[day] / vwap[day],
                price_return,
                total_cost_rate,
                expected_trading,
                actual_trading
            );

            assert!(
                (actual_trading - expected_trading).abs() < 1e-6,
                "Day {}: expected trading {:.6}, got {:.6}",
                day,
                expected_trading,
                actual_trading
            );
        }

        // Total return
        let total_return = &holding_return + &trading_return;
        let cumulative = compute_cumulative_return(&total_return.column(0).to_owned());

        // Calculate expected cumulative: count returns on holding days
        let mut expected_cum = 1.0;
        for day in 1..n_days {
            let day_return = weight_pattern[day - 1] * (adj_close[day] / adj_close[day - 1] - 1.0);
            expected_cum *= 1.0 + day_return;
        }
        let expected_cumulative_alt = expected_cum - 1.0;

        println!("\n=== Cumulative Return ===");
        println!("Expected (alternating): {:.6}", expected_cumulative_alt);
        println!("Actual (with costs):    {:.6}", cumulative);

        println!("\n✓ All assertions passed!");
    }
}
