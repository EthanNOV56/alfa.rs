//! DataLayer - Intelligent data fetching
//!
//! Provides automatic data loading with prefixed column names.

use crate::data::clickhouse::ClickHouseSource;
use crate::data::source::{DataError, QueryFilter};
use crate::expr::ast::Frequency;
use crate::expr::registry::config::FactorSlice;
use arrow::array::{
    Array, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    PrimitiveArray, StringArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType, Date32Type};
use arrow::record_batch::RecordBatch;
use arrow::ipc::reader::StreamReader;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// PreFilter - parsed filter conditions from pre_filter string
#[derive(Debug, Clone)]
pub struct PreFilter {
    pub start_date: Option<String>,
    pub end_date: Option<String>,
    pub symbols: Option<Vec<String>>,
    pub symbols_like: Vec<String>,         // e.g., "%SH"
    pub exclude_symbols_like: Vec<String>, // e.g., "%BJ"
    pub conditions: Vec<String>,
}

const TABLE_1D: &str = "stock_1d";
const TABLE_5M: &str = "stock_5m";

impl Default for PreFilter {
    fn default() -> Self {
        Self {
            start_date: None,
            end_date: None,
            symbols: None,
            symbols_like: Vec::new(),
            exclude_symbols_like: Vec::new(),
            conditions: Vec::new(),
        }
    }
}

impl PreFilter {
    /// Parse a pre_filter string into structured conditions
    /// Format examples:
    /// - "2020-01-01:2025-01-01" - date range
    /// - "2020-01-01:2025-01-01 symbols like '%SH'" - date range + symbol filter
    /// - "2020-01-01:2025-01-01 symbols not like '%BJ'" - exclude BJ stocks
    pub fn parse(filter_str: &str) -> Self {
        let mut result = PreFilter::default();
        let filter_str = filter_str.trim();

        // Split by whitespace for initial tokens
        let tokens: Vec<&str> = filter_str.split_whitespace().collect();

        let mut i = 0;
        while i < tokens.len() {
            let token = tokens[i];

            // Check for date range "start:end"
            if token.contains(':') && !token.contains("like") && !token.contains("not") {
                let parts: Vec<&str> = token.split(':').collect();
                if parts.len() == 2 {
                    result.start_date = Some(parts[0].to_string());
                    result.end_date = Some(parts[1].to_string());
                }
            }
            // Check for "symbols" keyword
            else if token == "symbols" {
                // Look ahead for "like" or "not like"
                if i + 2 < tokens.len() {
                    let next = tokens[i + 1];
                    let third = tokens[i + 2];
                    if next == "like" {
                        result.symbols_like.push(third.trim_matches('\'').to_string());
                        i += 3;
                        continue;
                    } else if next == "not" && i + 3 < tokens.len() && tokens[i + 2] == "like" {
                        result.exclude_symbols_like.push(tokens[i + 3].trim_matches('\'').to_string());
                        i += 4;
                        continue;
                    }
                }
            }
            // Check for "not" keyword (standalone not like)
            else if token == "not" && i + 2 < tokens.len() && tokens[i + 1] == "like" {
                result.exclude_symbols_like.push(tokens[i + 2].trim_matches('\'').to_string());
                i += 3;
                continue;
            }
            // Check for "like" with pattern
            else if token == "like" {
                if i + 1 < tokens.len() {
                    result.exclude_symbols_like.push(tokens[i + 1].trim_matches('\'').to_string());
                    i += 2;
                    continue;
                }
            }
            // Other conditions (e.g., "close > 100", "volume > 1000000")
            else {
                // Check if it looks like a SQL condition
                if token.contains('>') || token.contains('<') || token.contains('=') ||
                   token.contains("AND") || token.contains("OR") {
                    result.conditions.push(token.to_string());
                }
            }

            i += 1;
        }

        result
    }

    /// Build SQL WHERE clause from parsed conditions
    pub fn to_sql_conditions(&self) -> String {
        let mut conditions = Vec::new();

        if let Some(ref symbols) = self.symbols {
            if !symbols.is_empty() {
                let sym_list = symbols.iter()
                    .map(|s| format!("'{}'", s))
                    .collect::<Vec<_>>()
                    .join(", ");
                conditions.push(format!("symbol IN ({})", sym_list));
            }
        }

        for pattern in &self.symbols_like {
            conditions.push(format!("symbol LIKE '{}'", pattern));
        }

        for pattern in &self.exclude_symbols_like {
            conditions.push(format!("symbol NOT LIKE '{}'", pattern));
        }

        for cond in &self.conditions {
            conditions.push(cond.clone());
        }

        if conditions.is_empty() {
            String::new()
        } else {
            conditions.join(" AND ")
        }
    }
}

/// Parse a date string like "2024-01-15" to YYYYMMDD f64
fn parse_date_to_ymd(s: &str) -> f64 {
    let parts: Vec<i32> = s.split('-').filter_map(|p| p.parse().ok()).collect();
    if parts.len() != 3 {
        return f64::NAN;
    }
    let (year, month, day) = (parts[0], parts[1], parts[2]);
    (year as f64 * 10000.0) + (month as f64 * 100.0) + day as f64
}

/// Convert Date32 epoch days to YYYYMMDD f64
/// ClickHouse Arrow format sends Date columns as Date32 (days since 1970-01-01).
fn date32_to_ymd_f64(epoch_days: i32) -> f64 {
    if epoch_days < 0 {
        return f64::NAN;
    }
    let z = epoch_days as i64 + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y * 10000 + m as i64 * 100 + d as i64) as f64
}

/// Convert any Arrow numeric array to f64 values and push into vec
fn numeric_to_f64(
    array: &dyn Array,
    col_name: &str,
    vec: &mut Vec<f64>,
) -> Result<(), DataError> {
    macro_rules! push_as_f64 {
        ($arr:expr, $vec:expr) => {
            for v in $arr.values() {
                $vec.push(*v as f64);
            }
        };
    }

    if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
        vec.extend_from_slice(arr.values());
    } else if let Some(arr) = array.as_any().downcast_ref::<Float32Array>() {
        push_as_f64!(arr, vec);
    } else if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
        push_as_f64!(arr, vec);
    } else if let Some(arr) = array.as_any().downcast_ref::<UInt32Array>() {
        push_as_f64!(arr, vec);
    } else if let Some(arr) = array.as_any().downcast_ref::<UInt16Array>() {
        push_as_f64!(arr, vec);
    } else if let Some(arr) = array.as_any().downcast_ref::<UInt8Array>() {
        push_as_f64!(arr, vec);
    } else if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
        push_as_f64!(arr, vec);
    } else if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
        push_as_f64!(arr, vec);
    } else if let Some(arr) = array.as_any().downcast_ref::<Int16Array>() {
        push_as_f64!(arr, vec);
    } else if let Some(arr) = array.as_any().downcast_ref::<Int8Array>() {
        push_as_f64!(arr, vec);
    } else {
        return Err(DataError::Query(format!(
            "Column '{}' unsupported numeric type: {:?}",
            col_name,
            array.data_type()
        )));
    }
    Ok(())
}

/// Price data matrices for backtest (n_dates × n_symbols)
#[derive(Debug, Clone)]
pub struct PriceMatrix {
    pub dates: Vec<i64>,
    pub symbols: Vec<String>,
    pub close: Array2<f64>,
    pub open: Array2<f64>,
    pub high: Array2<f64>,
    pub low: Array2<f64>,
    pub vwap: Array2<f64>,
    pub returns: Array2<f64>,
    pub tradable: Array2<f64>,
}

impl PriceMatrix {
    /// Build a qcut matrix (Option<i32>) aligned with this PriceMatrix from FactorSlices.
    ///
    /// Returns `(n_dates, n_symbols)` where each cell is the qcut group (0..9) or `-1`
    /// (encoded as i32, -1 = None) for symbols with no data.
    pub fn build_qcut_matrix(
        &self,
        slices: &[FactorSlice],
    ) -> Array2<i32> {
        let n_dates = self.dates.len();
        let n_syms = self.symbols.len();

        let mut sym_to_idx: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for (i, s) in self.symbols.iter().enumerate() {
            sym_to_idx.insert(s.as_str(), i);
        }

        let mut date_to_idx: std::collections::HashMap<i64, usize> =
            std::collections::HashMap::new();
        for (i, &d) in self.dates.iter().enumerate() {
            date_to_idx.insert(d, i);
        }

        let mut mat = Array2::<i32>::from_elem((n_dates, n_syms), -1i32);

        for s in slices {
            for i in 0..s.groups.len() {
                let (date, sym_idx) = s.groups[i];
                if (sym_idx as usize) < s.symbols.len() {
                    let sym_str = &s.symbols[sym_idx as usize];
                    if let (Some(&di), Some(&si)) =
                        (date_to_idx.get(&date), sym_to_idx.get(sym_str.as_str()))
                    {
                        mat[[di, si]] = s.qcut[i].unwrap_or(-1i32);
                    }
                }
            }
        }

        mat
    }

    /// Build a factor matrix aligned with this price matrix from FactorSlices.
    ///
    /// Each `FactorSlice` carries its own symbol list. The returned `Array2<f64>`
    /// has shape `[n_dates, n_symbols]` with `f64::NAN` where no factor value is available.
    pub fn build_factor_matrix(
        &self,
        slices: &[FactorSlice],
    ) -> Array2<f64> {
        let n_dates = self.dates.len();
        let n_syms = self.symbols.len();

        let mut sym_to_idx: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for (i, s) in self.symbols.iter().enumerate() {
            sym_to_idx.insert(s.as_str(), i);
        }

        let mut date_to_idx: std::collections::HashMap<i64, usize> =
            std::collections::HashMap::new();
        for (i, &d) in self.dates.iter().enumerate() {
            date_to_idx.insert(d, i);
        }

        let mut mat = Array2::<f64>::from_elem((n_dates, n_syms), f64::NAN);

        for s in slices {
            for i in 0..s.groups.len() {
                let (date, sym_idx) = s.groups[i];
                if (sym_idx as usize) < s.symbols.len() {
                    let sym_str = &s.symbols[sym_idx as usize];
                    if let (Some(&di), Some(&si)) =
                        (date_to_idx.get(&date), sym_to_idx.get(sym_str.as_str()))
                    {
                        mat[[di, si]] = s.cap_neued[i];
                    }
                }
            }
        }

        mat
    }
}

/// Data layer for intelligent data fetching
pub struct DataLayer {
    source: ClickHouseSource,
    filter: QueryFilter,
    pre_filter: String,
    /// Cache for storing fetched data to avoid repeated queries
    cache: HashMap<String, HashMap<String, Array1<f64>>>,
    /// Cached unique symbols from last 5m query (sorted)
    symbols_5m: Vec<String>,
}

impl DataLayer {
    /// Get a reference to the underlying ClickHouse source.
    pub fn source(&self) -> &ClickHouseSource {
        &self.source
    }

    /// Create a new DataLayer from ClickHouseSource
    pub fn new(source: ClickHouseSource) -> Self {
        Self {
            source,
            filter: QueryFilter::default(),
            pre_filter: String::new(),
            cache: HashMap::new(),
            symbols_5m: Vec::new(),
        }
    }

    /// Set the filter for all queries
    pub fn set_filter(&mut self, filter: QueryFilter) {
        self.filter = filter;
    }

    /// Set the pre_filter string
    pub fn set_pre_filter(&mut self, filter: &str) {
        self.pre_filter = filter.to_string();
    }

    /// Get the current pre_filter string.
    pub fn pre_filter(&self) -> &str {
        &self.pre_filter
    }

    /// Clear the internal cache to free memory
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.symbols_5m.clear();
    }

    /// Get the unique symbols from the last 5m query (sorted)
    pub fn get_symbols_5m(&self) -> &[String] {
        &self.symbols_5m
    }

    /// Query data for specified fields
    /// Automatically determines which tables to query based on field prefixes
    /// Fields format: "5m:close", "1d:volume", etc.
    pub fn query(&mut self, fields: Vec<String>) -> Result<HashMap<String, Array1<f64>>, crate::data::source::DataError> {
        let pre = PreFilter::parse(&self.pre_filter);

        // Group fields by frequency prefix
        let mut freq_1d_cols: Vec<String> = Vec::new();
        let mut freq_5m_cols: Vec<String> = Vec::new();

        for field in &fields {
            if let Some(colon_pos) = field.find(':') {
                let prefix = &field[..colon_pos];
                let col_name = &field[colon_pos + 1..];
                match prefix {
                    "1d" => freq_1d_cols.push(col_name.to_string()),
                    "5m" => freq_5m_cols.push(col_name.to_string()),
                    _ => {}
                }
            }
        }

        let mut result = HashMap::new();

        // Query 1d data if needed (cached per-column-set)
        if !freq_1d_cols.is_empty() {
            let key = format!("1d:{:?}", freq_1d_cols);
            if let Some(cached) = self.cache.get(&key) {
                result.extend(cached.clone());
            } else {
                let data = self.query_1d_with_columns(&freq_1d_cols, &pre)?;
                result.extend(data.clone());
                self.cache.insert(key, data);
            }
        }

        // Query 5m data — cache ALL columns under "5m:full" on first query,
        // serve subsequent queries (with same or subset of columns) from cache
        if !freq_5m_cols.is_empty() {
            if let Some(full_data) = self.cache.get("5m:full") {
                // Serve from cache; fetch any missing columns
                let mut missing: Vec<String> = Vec::new();
                for col in &freq_5m_cols {
                    let key = format!("5m:{}", col);
                    if let Some(arr) = full_data.get(&key) {
                        result.insert(key, arr.clone());
                    } else {
                        missing.push(col.clone());
                    }
                }
                if !missing.is_empty() {
                    let data = self.query_5m_with_columns(&missing, &pre)?;
                    if let Some(full) = self.cache.get_mut("5m:full") {
                        full.extend(data.clone());
                    }
                    result.extend(data);
                }
            } else {
                let data = self.query_5m_with_columns(&freq_5m_cols, &pre)?;
                self.cache.insert("5m:full".to_string(), data.clone());
                result.extend(data);
            }
        }

        Ok(result)
    }

    /// Build SQL and execute Arrow query for a table
    ///
    /// Handles Arrow IPC parsing and converts all column types to Array1<f64>:
    /// - Float64: direct copy
    /// - Date32: epoch days → YYYYMMDD f64
    /// - Utf8 (symbol): string → numeric index via symbol encoding
    fn query_arrow_impl(
        &mut self,
        columns: &[String],
        table_name: &str,
        pre: &PreFilter,
        prefix: &str,
    ) -> Result<HashMap<String, Array1<f64>>, DataError> {
        // Build SQL (matches query_with_filter logic)
        let columns_str = columns.join(", ");
        let mut where_clauses = vec!["1=1".to_string()];

        if let Some(ref start) = pre.start_date {
            where_clauses.push(format!("trading_date >= '{}'", start));
        }
        if let Some(ref end) = pre.end_date {
            where_clauses.push(format!("trading_date <= '{}'", end));
        }
        for pattern in &pre.symbols_like {
            where_clauses.push(format!("symbol LIKE '{}'", pattern));
        }
        for pattern in &pre.exclude_symbols_like {
            where_clauses.push(format!("symbol NOT LIKE '{}'", pattern));
        }
        for cond in &pre.conditions {
            where_clauses.push(cond.clone());
        }

        let sql = format!(
            "SELECT {} FROM {} WHERE {} ORDER BY symbol, trading_date",
            columns_str,
            table_name,
            where_clauses.join(" AND ")
        );

        let bytes = self.source.query_raw_arrow(&sql)?;
        if bytes.is_empty() {
            return Ok(HashMap::new());
        }

        let cursor = std::io::Cursor::new(bytes);
        let reader = StreamReader::try_new(cursor, None)
            .map_err(|e| DataError::Query(format!("Arrow IPC parse: {}", e)))?;
        let schema = reader.schema();

        // Classify each column by its Arrow type
        enum ColKind {
            Float,
            Date,
            Symbol,
        }
        let mut col_specs: Vec<(usize, String, ColKind)> = Vec::new();
        for (i, field) in schema.fields().iter().enumerate() {
            let kind: ColKind = match field.data_type() {
                DataType::Float64 => ColKind::Float,
                DataType::Date32 => ColKind::Date,
                DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View => {
                    if field.name() == "symbol" {
                        ColKind::Symbol
                    } else {
                        ColKind::Float
                    }
                }
                DataType::Int32 | DataType::Int64 | DataType::UInt32 | DataType::UInt64 => {
                    ColKind::Float
                }
                _ => {
                    return Err(DataError::Query(format!(
                        "Unsupported Arrow type for column '{}': {:?}",
                        field.name(),
                        field.data_type()
                    )));
                }
            };
            col_specs.push((i, field.name().clone(), kind));
        }

        // Collect all batches first (needed for two-pass symbol encoding)
        let mut batches: Vec<RecordBatch> = Vec::new();
        for batch_result in reader {
            batches.push(
                batch_result
                    .map_err(|e| DataError::Query(format!("Arrow batch read: {}", e)))?,
            );
        }

        // Build symbol encoding (pass 1: scan only, allocate Strings only for unique symbols)
        let t_encode = std::time::Instant::now();
        let symbol_to_idx: HashMap<String, f64>;
        if !self.symbols_5m.is_empty() {
            symbol_to_idx = self
                .symbols_5m
                .iter()
                .enumerate()
                .map(|(i, s)| (s.clone(), i as f64))
                .collect();
        } else {
            let mut unique_set: std::collections::HashSet<String> = std::collections::HashSet::new();
            for batch in &batches {
                let arrays = batch.columns();
                for &(col_idx, _, ref kind) in &col_specs {
                    if matches!(kind, ColKind::Symbol) {
                        if let Some(arr) = arrays[col_idx].as_any().downcast_ref::<StringArray>() {
                            for i in 0..batch.num_rows() {
                                let s = arr.value(i);
                                if !unique_set.contains(s) {
                                    unique_set.insert(s.to_string());
                                }
                            }
                        }
                    }
                }
            }
            let mut unique: Vec<String> = unique_set.into_iter().collect();
            unique.sort();
            symbol_to_idx = unique
                .iter()
                .enumerate()
                .map(|(i, s)| (s.clone(), i as f64))
                .collect();
            self.symbols_5m = unique;
        }

        // Pass 2: process all columns, encode symbols via HashMap lookup (no allocation)
        let t_parse = std::time::Instant::now();
        let mut col_vecs: HashMap<String, Vec<f64>> = HashMap::new();
        for (_, name, _) in &col_specs {
            col_vecs.insert(name.clone(), Vec::new());
        }

        for batch in &batches {
            let arrays = batch.columns();
            let n_rows = batch.num_rows();

            for &(col_idx, ref name, ref kind) in &col_specs {
                let array = &arrays[col_idx];
                let vec = col_vecs.get_mut(name).unwrap();

                match kind {
                    ColKind::Float => {
                        numeric_to_f64(array, name, vec)?;
                    }
                    ColKind::Date => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<PrimitiveArray<Date32Type>>()
                            .ok_or_else(|| {
                                DataError::Query(format!(
                                    "Column '{}' expected Date32", name
                                ))
                            })?;
                        for i in 0..n_rows {
                            vec.push(date32_to_ymd_f64(arr.value(i)));
                        }
                    }
                    ColKind::Symbol => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| {
                                DataError::Query(format!(
                                    "Column '{}' expected StringArray", name
                                ))
                            })?;
                        for i in 0..n_rows {
                            vec.push(
                                symbol_to_idx.get(arr.value(i)).copied().unwrap_or(f64::NAN),
                            );
                        }
                    }
                }
            }
        }
        let t_parse_ms = t_parse.elapsed().as_millis();

        // Build final result (Vec → Array1 without clone via from_vec)
        let mut result = HashMap::new();
        for (_, name, kind) in &col_specs {
            let key = format!("{}:{}", prefix, name);
            if let Some(vec) = col_vecs.remove(name) {
                result.insert(key, Array1::from_vec(vec));
            }
        }
        let n_rows = result.values().next().map(|v| v.len()).unwrap_or(0);
        eprintln!("    arrow parse: {}ms  encode: {}ms  rows={}", t_parse_ms, t_encode.elapsed().as_millis(), n_rows);

        Ok(result)
    }

    /// Query 1d data with specific columns
    fn query_1d_with_columns(&mut self, columns: &[String], pre: &PreFilter) -> Result<HashMap<String, Array1<f64>>, DataError> {
        let mut cols = vec!["trading_date".to_string(), "symbol".to_string()];
        for col in columns {
            if !cols.contains(col) {
                cols.push(col.clone());
            }
        }
        self.query_arrow_impl(&cols, TABLE_1D, pre, "1d")
    }

    /// Query 5m data with specific columns
    fn query_5m_with_columns(&mut self, columns: &[String], pre: &PreFilter) -> Result<HashMap<String, Array1<f64>>, DataError> {
        let mut cols = vec!["trading_date".to_string(), "symbol".to_string()];
        for col in columns {
            if !cols.contains(col) {
                cols.push(col.clone());
            }
        }
        self.query_arrow_impl(&cols, TABLE_5M, pre, "5m")
    }

    /// Build a (date, symbol) → free_float_cap map for cap_neu neutralization.
    ///
    /// Queries 1d close and free_float_shares, computes free_float_cap = close × shares.
    pub fn build_free_float_cap_map(
        &mut self,
    ) -> Result<HashMap<(i64, usize), f64>, crate::data::source::DataError> {
        let data = self.query(vec![
            "1d:close".to_string(),
            "1d:free_float_shares".to_string(),
            "1d:trading_date".to_string(),
            "1d:symbol".to_string(),
        ])?;

        let dates = data.get("1d:trading_date").ok_or(DataError::NotFound(
            "1d:trading_date column missing".into(),
        ))?;
        let symbols = data.get("1d:symbol").ok_or(DataError::NotFound(
            "1d:symbol column missing".into(),
        ))?;
        let close = data.get("1d:close").ok_or(DataError::NotFound(
            "1d:close column missing".into(),
        ))?;
        let float_shares = data.get("1d:free_float_shares").ok_or(DataError::NotFound(
            "1d:free_float_shares column missing".into(),
        ))?;

        let sym_list = self.get_symbols_5m();
        let n_syms = sym_list.len();
        let mut map = HashMap::new();
        for i in 0..dates.len() {
            let d = dates[i] as i64;
            if symbols[i].is_nan() {
                continue;
            }
            let s = symbols[i] as usize;
            if s >= n_syms {
                continue;
            }
            let cap = close[i] * float_shares[i];
            if d > 19000101 && cap.is_finite() && cap > 0.0 {
                map.insert((d, s), cap);
            }
        }
        Ok(map)
    }

    /// Query 1d price data for backtest and return as matrices (n_dates × n_symbols).
    ///
    /// Follows `alpha.py::read_prices`: vwap = amount × amount_unit / (volume × volume_unit),
    /// forward-adjusted prices via adjust_factor, backfilled close.
    pub fn query_price_matrix(
        &mut self,
    ) -> Result<PriceMatrix, crate::data::source::DataError> {
        let data = self.query(vec![
            "1d:trading_date".to_string(),
            "1d:symbol".to_string(),
            "1d:open".to_string(),
            "1d:high".to_string(),
            "1d:low".to_string(),
            "1d:close".to_string(),
            "1d:volume".to_string(),
            "1d:amount".to_string(),
            "1d:adjust_factor".to_string(),
        ])?;

        let dates_arr = data.get("1d:trading_date").ok_or(DataError::NotFound(
            "1d:trading_date missing".into(),
        ))?;
        let syms_arr = data.get("1d:symbol").ok_or(DataError::NotFound(
            "1d:symbol missing".into(),
        ))?;
        let open = data.get("1d:open").ok_or(DataError::NotFound("1d:open missing".into()))?;
        let high = data.get("1d:high").ok_or(DataError::NotFound("1d:high missing".into()))?;
        let low = data.get("1d:low").ok_or(DataError::NotFound("1d:low missing".into()))?;
        let close = data.get("1d:close").ok_or(DataError::NotFound("1d:close missing".into()))?;
        let volume = data.get("1d:volume").ok_or(DataError::NotFound("1d:volume missing".into()))?;
        let amount = data.get("1d:amount").ok_or(DataError::NotFound("1d:amount missing".into()))?;
        let adj = data.get("1d:adjust_factor").ok_or(DataError::NotFound("1d:adjust_factor missing".into()))?;

        let sym_list = self.get_symbols_5m();
        let n_syms = sym_list.len();

        // Collect unique sorted dates
        let mut date_set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        for i in 0..dates_arr.len() {
            if syms_arr[i].is_nan() { continue; }
            let d = dates_arr[i] as i64;
            let s = syms_arr[i] as usize;
            if d > 19000101 && s < n_syms { date_set.insert(d); }
        }
        let dates: Vec<i64> = date_set.into_iter().collect();
        let n_dates = dates.len();

        // Raw matrices (NaN-filled)
        let mut close_mat = Array2::<f64>::from_elem((n_dates, n_syms), f64::NAN);
        let mut open_mat = Array2::<f64>::from_elem((n_dates, n_syms), f64::NAN);
        let mut high_mat = Array2::<f64>::from_elem((n_dates, n_syms), f64::NAN);
        let mut low_mat = Array2::<f64>::from_elem((n_dates, n_syms), f64::NAN);
        let mut vwap_mat = Array2::<f64>::from_elem((n_dates, n_syms), f64::NAN);
        let mut adj_mat = Array2::<f64>::from_elem((n_dates, n_syms), f64::NAN);
        // Track which (date, symbol) have data, matching Python's group-by semantics
        let mut has_data = Array2::<bool>::from_elem((n_dates, n_syms), false);

        let date_to_idx: std::collections::HashMap<i64, usize> = dates
            .iter().enumerate().map(|(i, &d)| (d, i)).collect();

        for i in 0..dates_arr.len() {
            if syms_arr[i].is_nan() { continue; }
            let d = dates_arr[i] as i64;
            let s = syms_arr[i] as usize;
            if s >= n_syms { continue; }
            if let Some(&di) = date_to_idx.get(&d) {
                close_mat[[di, s]] = close[i];
                open_mat[[di, s]] = open[i];
                high_mat[[di, s]] = high[i];
                low_mat[[di, s]] = low[i];
                adj_mat[[di, s]] = adj[i];
                has_data[[di, s]] = true;
                // vwap = amount * amount_unit / (volume * volume_unit)
                let factor = self.source.amount_unit() / self.source.volume_unit();
                if volume[i].is_finite() && volume[i] > 0.0 {
                    vwap_mat[[di, s]] = amount[i] * factor / volume[i];
                }
            }
        }

        // Forward-adjust prices: divide adj_factor by last_adj per symbol.
        // Matches Python's pl.col("adjust_factor") / pl.col("adjust_factor").last().over("symbol").
        // The "last" is the last date the symbol appears in the data (has_data==true),
        // even if adj_factor is NaN on that date. IEEE 754 propagates NaN correctly.
        for s in 0..n_syms {
            let last_adj = (0..n_dates).rev()
                .find(|&d| has_data[[d, s]])
                .map(|d| adj_mat[[d, s]])
                .unwrap_or(1.0);
            for d in 0..n_dates {
                let f = adj_mat[[d, s]] / last_adj;
                close_mat[[d, s]] *= f;
                open_mat[[d, s]] *= f;
                high_mat[[d, s]] *= f;
                low_mat[[d, s]] *= f;
                vwap_mat[[d, s]] *= f;
            }
        }

        // Backfill close only, matching alpha.py:
        //   close.fill_nan(None).fill_null(strategy="backward").over("symbol")
        // Only NaN is converted to null and backfilled; 0.0 stays as-is.
        // open/high/low/vwap are NOT backfilled in Python; NaN → fill_nan(0) in backtest.
        for s in 0..n_syms {
            let mut last_close = f64::NAN;
            for d in (0..n_dates).rev() {
                if close_mat[[d, s]].is_finite() {
                    last_close = close_mat[[d, s]];
                } else if last_close.is_finite() {
                    close_mat[[d, s]] = last_close;
                }
            }
        }

        // Compute returns: close[t] / close[t-1] - 1
        let mut returns = Array2::<f64>::zeros((n_dates, n_syms));
        for s in 0..n_syms {
            for d in 1..n_dates {
                let prev = close_mat[[d - 1, s]];
                let curr = close_mat[[d, s]];
                if prev.is_finite() && curr.is_finite() && prev != 0.0 {
                    returns[[d, s]] = curr / prev - 1.0;
                }
            }
        }

        // tradable = high > low
        let mut tradable = Array2::<f64>::zeros((n_dates, n_syms));
        for d in 0..n_dates {
            for s in 0..n_syms {
                if high_mat[[d, s]].is_finite() && low_mat[[d, s]].is_finite()
                    && high_mat[[d, s]] > low_mat[[d, s]]
                {
                    tradable[[d, s]] = 1.0;
                }
            }
        }

        Ok(PriceMatrix {
            dates,
            symbols: sym_list.to_vec(),
            close: close_mat,
            open: open_mat,
            high: high_mat,
            low: low_mat,
            vwap: vwap_mat,
            returns,
            tradable,
        })
    }

    /// Fetch data by frequency
    /// Returns columns with frequency prefix (e.g., "1d:close", "5m:vol")
    pub fn fetch(&mut self, freq: Frequency) -> Result<HashMap<String, Array1<f64>>, crate::data::source::DataError> {
        match freq {
            Frequency::Daily => {
                let pre = PreFilter::parse(&self.pre_filter);
                self.query_1d_with_columns(&vec![], &pre)
            }
            Frequency::Minute5 => {
                let pre = PreFilter::parse(&self.pre_filter);
                self.query_5m_with_columns(&vec![], &pre)
            }
            _ => Err(crate::data::source::DataError::NotFound(
                format!("Frequency {:?} not supported yet", freq)
            )),
        }
    }
}
