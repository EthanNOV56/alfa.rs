//! DataPool — centralised data lifecycle manager for the pipeline.
//!
//! Owns a shared PriceMatrix (Arc, queried once) and optionally caches
//! year-level DataLayer instances to avoid redundant ClickHouse queries.
//! Provides configurable memory/time tradeoff via [`CachePolicy`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex, atomic::AtomicUsize, atomic::Ordering};

use crate::data::clickhouse::ClickHouseSource;
use crate::data::layer::{DataLayer, PriceMatrix};

// ---------------------------------------------------------------------------
// CachePolicy — controls whether year-level 5m DataLayers are retained
// ---------------------------------------------------------------------------

/// Cache policy for year-level DataLayers.
///
/// Each DataLayer holds ~2 GB of raw 5m Arrow data during query processing.
/// Caching avoids re-querying ClickHouse at the cost of memory.
///
/// Estimated per-year memory: ~2 GB (5m data for ~5 000 Chinese A-shares).
#[derive(Debug, Clone)]
pub enum CachePolicy {
    /// Drop all intermediate data immediately after use.
    /// Lowest peak memory (~4 GB total). Highest ClickHouse load (re-queries
    /// every batch).
    ///
    /// Recommended for: machines with < 16 GB RAM, single-factor workflows.
    DropAll,

    /// Keep only the most recently used year's DataLayer.
    /// ~2 GB extra memory overhead. Effective when processing years in order.
    ///
    /// Recommended for: 16 GB machines, medium factor count (50–100).
    KeepMostRecent,

    /// Keep all year DataLayers in memory.
    /// Memory ≈ ~2 GB × number of distinct years. Zero re-queries.
    ///
    /// Recommended for: 32 GB+ machines, 100+ factor pipelines, GP mining.
    KeepAll,

    /// Keep up to N most recently used year DataLayers.
    /// Memory ≈ ~2 GB × N. Evicts least-recently-used when over capacity.
    ///
    /// Recommended for: precise memory tuning on any hardware.
    KeepN(usize),
}

impl Default for CachePolicy {
    fn default() -> Self {
        CachePolicy::DropAll
    }
}

// ---------------------------------------------------------------------------
// DataPoolConfig
// ---------------------------------------------------------------------------

/// Complete configuration surface for the DataPool.
///
/// All defaults are conservative. Tune for your hardware:
///
/// | Scenario          | cache_policy       | batch_size | parallel_years |
/// |-------------------|--------------------|------------|----------------|
/// | 16 GB machine     | DropAll            | 5          | 3              |
/// | 32 GB machine     | KeepMostRecent     | 20         | 5              |
/// | 64 GB+ server     | KeepAll            | 50         | 8              |
/// | GP mining         | KeepAll or KeepN   | —          | 1*             |
///
/// *GP uses sequential year processing internally; parallel_years mainly
///  affects `calc()`.
#[derive(Debug, Clone)]
pub struct DataPoolConfig {
    /// Cache policy for year-level DataLayers (5m data).
    /// Default: DropAll (safest, matches legacy behaviour).
    pub cache_policy: CachePolicy,

    /// Number of years to compute in parallel during multi-year `calc()`.
    ///
    /// Benchmarked on 16-core 32 GB, single CH node:
    ///   threads=3 → 42.8 s, peak 9.5 GB
    ///   threads=5 → 31.3 s, peak 17.0 GB  ← sweet spot
    ///   threads=6 → 37.5 s, peak 18.6 GB  (CH contention)
    ///
    /// Default: 5.
    pub calc_parallel_years: usize,

    /// Soft memory budget in bytes for the pool (0 = unlimited).
    /// When exceeded, cached DataLayers are evicted LRU-first before new
    /// queries. Does NOT evict the shared PriceMatrix.
    ///
    /// Set to e.g. `12_000_000_000` (12 GB) for a 16 GB machine.
    /// Default: 0 (unlimited — caller is responsible for sizing).
    pub memory_budget_bytes: usize,

    /// Number of factors to process per batch in `evaluate_and_backtest_each()`.
    /// Smaller = lower peak memory (factor matrices released sooner).
    /// Larger = fewer DataLayer borrow/return cycles.
    ///
    /// Default: 5 (conservative).
    pub backtest_batch_size: usize,
}

impl Default for DataPoolConfig {
    fn default() -> Self {
        Self {
            cache_policy: CachePolicy::DropAll,
            calc_parallel_years: 5,
            memory_budget_bytes: 0,
            backtest_batch_size: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// DataPool
// ---------------------------------------------------------------------------

/// Centralised data lifecycle manager.
///
/// Responsibilities:
/// 1. Queries and shares a `PriceMatrix` (via Arc) — one query for all consumers.
/// 2. Optionally caches year-level `DataLayer` instances per [`CachePolicy`].
/// 3. Tracks approximate memory and evicts cached DataLayers when over budget.
///
/// Thread-safe: all interior mutability through `Mutex` and `AtomicUsize`.
///
/// # Example
///
/// ```ignore
/// let pool = DataPool::new(source, "symbols not like '%BJ'", DataPoolConfig::default());
/// let prices = pool.get_prices()?;  // Arc<PriceMatrix>
/// let mut dl = pool.borrow_year(2020, "2020-01-01", "2021-01-01")?;
/// // ... use dl ...
/// pool.return_year(2020, dl);
/// ```
pub struct DataPool {
    source: ClickHouseSource,
    pre_filter: String,

    /// Shared price matrix — lazy-init via get_prices(), never evicted.
    prices: Mutex<Option<Arc<PriceMatrix>>>,

    /// Cached DataLayers keyed by year.
    year_caches: Mutex<HashMap<i32, DataLayer>>,

    /// LRU tracking: most recently used year at the back.
    year_lru: Mutex<Vec<i32>>,

    config: DataPoolConfig,

    /// Approximate memory used by cached DataLayers (bytes).
    /// Conservative estimate: 2 GB per year-level DataLayer.
    approx_memory: AtomicUsize,
}

impl DataPool {
    /// Year-level DataLayer memory estimate (conservative, based on benchmarks).
    const YEAR_DL_ESTIMATE_BYTES: usize = 2_000_000_000; // 2 GB

    /// Create a new DataPool from a ClickHouse source and configuration.
    pub fn new(source: ClickHouseSource, pre_filter: String, config: DataPoolConfig) -> Self {
        Self {
            source,
            pre_filter,
            prices: Mutex::new(None),
            year_caches: Mutex::new(HashMap::new()),
            year_lru: Mutex::new(Vec::new()),
            config,
            approx_memory: AtomicUsize::new(0),
        }
    }

    /// Get or initialise the shared PriceMatrix.
    ///
    /// Queries ClickHouse once; subsequent calls return `Arc::clone` of the
    /// cached result (no additional I/O).
    pub fn get_prices(&self) -> Result<Arc<PriceMatrix>, crate::data::source::DataError> {
        let mut guard = self.prices.lock().unwrap();
        if let Some(ref prices) = *guard {
            return Ok(Arc::clone(prices));
        }

        let mut dl = DataLayer::new(self.source.clone());
        dl.set_pre_filter(&self.pre_filter);
        let prices = dl.query_price_matrix()?;

        let arc = Arc::new(prices);
        *guard = Some(Arc::clone(&arc));
        Ok(arc)
    }

    /// Borrow a DataLayer for a specific year.
    ///
    /// Returns a cached instance if available per [`CachePolicy`], otherwise
    /// creates a new DataLayer with the given date range appended to the pool's
    /// base pre_filter.
    ///
    /// The caller **MUST** call [`return_year`] after use so the pool can
    /// cache or drop the DataLayer according to policy.
    pub fn borrow_year(&self, year: i32, start_date: &str, end_date: &str) -> DataLayer {
        // Check cache
        {
            let mut caches = self.year_caches.lock().unwrap();
            if let Some(dl) = caches.remove(&year) {
                // Touch LRU
                let mut lru = self.year_lru.lock().unwrap();
                lru.retain(|&y| y != year);
                lru.push(year);
                return dl;
            }
        }

        // Cache miss — create fresh DataLayer
        let mut dl = DataLayer::new(self.source.clone());
        let filter = if self.pre_filter.is_empty() {
            format!("{}:{}", start_date, end_date)
        } else {
            format!("{}:{} {}", start_date, end_date, self.pre_filter)
        };
        dl.set_pre_filter(&filter);
        dl
    }

    /// Return a DataLayer after use.
    ///
    /// The pool will cache or drop it based on [`CachePolicy`] and memory budget.
    /// Call this promptly to avoid stale data.
    pub fn return_year(&self, year: i32, dl: DataLayer) {
        match self.config.cache_policy {
            CachePolicy::DropAll => {
                // Let dl drop naturally
                drop(dl);
            }
            CachePolicy::KeepMostRecent => {
                // Evict all existing, then insert
                let mut caches = self.year_caches.lock().unwrap();
                let mut lru = self.year_lru.lock().unwrap();
                for &old_year in lru.iter() {
                    caches.remove(&old_year);
                }
                lru.clear();

                caches.insert(year, dl);
                lru.push(year);
                self.approx_memory
                    .store(Self::YEAR_DL_ESTIMATE_BYTES, Ordering::Relaxed);
            }
            CachePolicy::KeepAll => {
                self.enforce_budget();
                let mut caches = self.year_caches.lock().unwrap();
                let mut lru = self.year_lru.lock().unwrap();
                lru.retain(|&y| y != year);
                lru.push(year);
                caches.insert(year, dl);
                self.approx_memory.store(
                    caches.len() * Self::YEAR_DL_ESTIMATE_BYTES,
                    Ordering::Relaxed,
                );
            }
            CachePolicy::KeepN(n) => {
                self.enforce_budget();
                let mut caches = self.year_caches.lock().unwrap();
                let mut lru = self.year_lru.lock().unwrap();

                // Evict LRU if at capacity
                while lru.len() >= n {
                    if let Some(old) = lru.first().copied() {
                        lru.remove(0);
                        caches.remove(&old);
                    } else {
                        break;
                    }
                }

                lru.retain(|&y| y != year);
                lru.push(year);
                caches.insert(year, dl);
                self.approx_memory.store(
                    caches.len() * Self::YEAR_DL_ESTIMATE_BYTES,
                    Ordering::Relaxed,
                );
            }
        }
    }

    /// Evict LRU entries until memory is below budget (if budget > 0).
    fn enforce_budget(&self) {
        let budget = self.config.memory_budget_bytes;
        if budget == 0 {
            return;
        }

        let mut caches = self.year_caches.lock().unwrap();
        let mut lru = self.year_lru.lock().unwrap();

        while !lru.is_empty() && caches.len() * Self::YEAR_DL_ESTIMATE_BYTES > budget {
            if let Some(old) = lru.first().copied() {
                lru.remove(0);
                caches.remove(&old);
            } else {
                break;
            }
        }

        self.approx_memory.store(
            caches.len() * Self::YEAR_DL_ESTIMATE_BYTES,
            Ordering::Relaxed,
        );
    }

    /// Get the current config (read-only).
    pub fn config(&self) -> &DataPoolConfig {
        &self.config
    }

    /// Get the base pre_filter string.
    pub fn pre_filter(&self) -> &str {
        &self.pre_filter
    }

    /// Set the base pre_filter (clears cached prices since they may be stale).
    pub fn set_pre_filter(&mut self, filter: &str) {
        self.pre_filter = filter.to_string();
        // Invalidate cached prices
        *self.prices.lock().unwrap() = None;
    }

    /// Get approximate memory usage of cached DataLayers in bytes.
    pub fn memory_usage(&self) -> usize {
        self.approx_memory.load(Ordering::Relaxed)
    }

    /// Get number of cached year DataLayers.
    pub fn cached_years(&self) -> usize {
        self.year_caches.lock().unwrap().len()
    }

    /// Force-evict all cached DataLayers (keeps the PriceMatrix).
    pub fn clear_caches(&self) {
        let mut caches = self.year_caches.lock().unwrap();
        let mut lru = self.year_lru.lock().unwrap();
        caches.clear();
        lru.clear();
        self.approx_memory.store(0, Ordering::Relaxed);
    }
}
