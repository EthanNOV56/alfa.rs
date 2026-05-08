//! Frequency-aware field definitions and helpers.
//!
//! Central registry mapping canonical field names to their available frequencies
//! and per-table DB expressions. Used by DataLayer for SQL generation and by
//! the GP engine for frequency-aware expression generation and mutation.

use crate::expr::ast::{Expr, Frequency};

// ── Field definitions ────────────────────────────────────────────────────

/// A canonical field with its per-frequency availability and DB expressions.
struct FieldDef {
    canonical: &'static str,
    freq_1d: bool,
    freq_5m: bool,
    freq_1m: bool,
    /// SQL expression on stock_1d (None = use canonical name as column)
    expr_1d: Option<&'static str>,
    /// SQL expression on stock_5m/stock_1m (None = use canonical name as column)
    expr_5m: Option<&'static str>,
}

/// All known fields across frequencies, matching local DB schema.
/// vol: 1d→DB `volume` (手), 5m/1m→DB `vol` (手)
/// amount: 1d→千元, 5m/1m→元. VWAP handles unit conversion.
static FIELDS: &[FieldDef] = &[
    // ── OHLCV (all freqs) ──
    FieldDef {
        canonical: "close",
        freq_1d: true,
        freq_5m: true,
        freq_1m: true,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "open",
        freq_1d: true,
        freq_5m: true,
        freq_1m: true,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "high",
        freq_1d: true,
        freq_5m: true,
        freq_1m: true,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "low",
        freq_1d: true,
        freq_5m: true,
        freq_1m: true,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "vol",
        freq_1d: true,
        freq_5m: true,
        freq_1m: true,
        expr_1d: Some("volume"),
        expr_5m: Some("vol"),
    },
    FieldDef {
        canonical: "amount",
        freq_1d: true,
        freq_5m: true,
        freq_1m: true,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "vwap",
        freq_1d: true,
        freq_5m: true,
        freq_1m: true,
        expr_1d: Some("amount * 1000 / (volume * 100)"),
        expr_5m: Some("amount / vol"),
    },
    // ── adj / limits (1d only) ──
    FieldDef {
        canonical: "prev_close",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "change",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "change_pct",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "adjust_factor",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "up_limit",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "down_limit",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    // ── Turnover (1d only) ──
    FieldDef {
        canonical: "turnover_rate",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "free_turnover_rate",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "volume_ratio",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    // ── Valuation (1d only) ──
    FieldDef {
        canonical: "pe",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "pe_ttm",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "pb",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "ps",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "ps_ttm",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    // ── Dividend (1d only) ──
    FieldDef {
        canonical: "dividend_yield",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "dividend_yield_ttm",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    // ── Shares / market cap (1d only) ──
    FieldDef {
        canonical: "total_shares",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "float_shares",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "free_float_shares",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "market_cap",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "float_market_cap",
        freq_1d: true,
        freq_5m: false,
        freq_1m: false,
        expr_1d: None,
        expr_5m: None,
    },
    // ── 5m/1m only ──
    FieldDef {
        canonical: "pre_close",
        freq_1d: false,
        freq_5m: true,
        freq_1m: true,
        expr_1d: None,
        expr_5m: None,
    },
    FieldDef {
        canonical: "pct_chg",
        freq_1d: false,
        freq_5m: true,
        freq_1m: true,
        expr_1d: None,
        expr_5m: None,
    },
];

// ── Query API ─────────────────────────────────────────────────────────────

/// All supported frequencies, finest to coarsest.
pub fn all_frequencies() -> &'static [Frequency] {
    &[Frequency::Minute1, Frequency::Minute5, Frequency::Daily]
}

/// Canonical field names available at a given frequency.
pub fn fields_for_frequency(freq: &Frequency) -> Vec<&'static str> {
    FIELDS
        .iter()
        .filter(|f| match freq {
            Frequency::Minute1 => f.freq_1m,
            Frequency::Minute5 => f.freq_5m,
            Frequency::Daily => f.freq_1d,
            _ => false,
        })
        .map(|f| f.canonical)
        .collect()
}

/// Check whether a canonical field name exists at the given frequency.
pub fn field_at_frequency(name: &str, freq: &Frequency) -> bool {
    FIELDS.iter().any(|f| {
        f.canonical == name
            && match freq {
                Frequency::Minute1 => f.freq_1m,
                Frequency::Minute5 => f.freq_5m,
                Frequency::Daily => f.freq_1d,
                _ => false,
            }
    })
}

/// Resolve a canonical field name to its DB SQL expression for a given table.
pub fn db_expression_for(canonical: &str, table: &str) -> Option<String> {
    let fd = FIELDS.iter().find(|f| f.canonical == canonical)?;
    let is_5m = table == "stock_5m" || table == "stock_1m";
    if is_5m {
        fd.expr_5m.map(|e| e.to_string())
    } else {
        fd.expr_1d.map(|e| e.to_string())
    }
}

// ── Frequency helpers ─────────────────────────────────────────────────────

/// Check if `agg_freq` can aggregate data at `data_freq` (must be coarser).
pub fn can_aggregate(agg_freq: &Frequency, data_freq: &Frequency) -> bool {
    agg_freq.period_days() > data_freq.period_days()
}

/// Infer the frequency of an expression by walking the tree.
/// Returns None for constant-only expressions.
pub fn infer_frequency(expr: &Expr) -> Option<Frequency> {
    match expr {
        Expr::Column(name) => name.split(':').next().and_then(Frequency::parse),
        Expr::FunctionCall { freq, args, .. } => {
            if let Some(f) = freq {
                Some(f.clone())
            } else if let Some(arg) = args.first() {
                infer_frequency(arg)
            } else {
                None
            }
        }
        Expr::UnaryExpr { expr, .. } => infer_frequency(expr),
        Expr::BinaryExpr { left, right, .. } => {
            match (infer_frequency(left), infer_frequency(right)) {
                (Some(a), Some(b)) => {
                    if a.period_days() < b.period_days() {
                        Some(a)
                    } else {
                        Some(b)
                    }
                }
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            }
        }
        Expr::Conditional { condition, .. } => infer_frequency(condition),
        Expr::Aggregate { expr, .. } => infer_frequency(expr),
        Expr::Cast { expr, .. } => infer_frequency(expr),
        Expr::Literal(_) => None,
    }
}

/// Valid aggregation frequencies for data at `data_freq` (coarser only).
pub fn valid_agg_freqs(data_freq: &Frequency) -> Vec<Frequency> {
    all_frequencies()
        .iter()
        .filter(|f| can_aggregate(f, data_freq))
        .cloned()
        .collect()
}

/// All field names available at daily frequency (GP terminals).
pub fn avail_fields_1d() -> Vec<String> {
    FIELDS
        .iter()
        .filter(|f| f.freq_1d)
        .map(|f| f.canonical.to_string())
        .collect()
}
