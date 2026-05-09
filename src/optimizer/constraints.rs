//! Constraint system for portfolio optimization.
//!
//! Research basis:
//! - Report #7: turnover constraints vs penalties
//! - Report #12: risk exposure control
//! - Report #14: Barra-style factor neutrality constraints
//! - Report #16: multi-objective constraints

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Optimization constraint set.
///
/// All constraints are optional — only the ones explicitly set are enforced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConstraints {
    // ── Weight bounds ──
    /// Long-only (w ≥ 0)
    #[serde(default)]
    pub long_only: bool,

    /// Per-asset maximum weight
    #[serde(default)]
    pub max_position: Option<f64>,

    /// Per-asset minimum weight (for shorting)
    #[serde(default)]
    pub min_position: Option<f64>,

    // ── Position concentration ──
    /// Maximum number of assets with non-zero weight (L0 norm)
    #[serde(default)]
    pub max_assets: Option<usize>,

    // ── Market exposure ──
    /// Market neutral (Σw = 0)
    #[serde(default)]
    pub market_neutral: bool,

    /// Full investment (Σw = 1). Ignored when `market_neutral = true`.
    #[serde(default = "default_true")]
    pub full_investment: bool,

    /// Total leverage limit (Σ|w| ≤ L)
    #[serde(default)]
    pub leverage_limit: Option<f64>,

    // ── Sector / group constraints ──
    /// Group-level weight constraints (e.g., sector cap at 30%)
    #[serde(default)]
    pub group_constraints: Vec<GroupConstraint>,

    // ── Factor exposure constraints ──
    /// Factor exposure constraints (Barra-style neutrality)
    #[serde(default)]
    pub factor_exposure_constraints: Vec<FactorExposureConstraint>,

    // ── Turnover ──
    /// One-day turnover limit: ‖w - w_prev‖₁ ≤ τ
    #[serde(default)]
    pub turnover_limit: Option<f64>,
}

fn default_true() -> bool {
    true
}

impl Default for OptimizerConstraints {
    fn default() -> Self {
        Self {
            long_only: true,
            max_position: None,
            min_position: None,
            max_assets: None,
            market_neutral: false,
            full_investment: true,
            leverage_limit: None,
            group_constraints: vec![],
            factor_exposure_constraints: vec![],
            turnover_limit: None,
        }
    }
}

/// Group-level weight constraint (sector, industry, or custom grouping).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupConstraint {
    /// Group name (e.g., "Financials")
    pub name: String,
    /// Asset indices belonging to this group
    pub members: Vec<usize>,
    /// Minimum aggregate weight for the group
    #[serde(default)]
    pub min_weight: Option<f64>,
    /// Maximum aggregate weight for the group
    #[serde(default)]
    pub max_weight: Option<f64>,
}

/// Factor exposure constraint (Barra-style).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorExposureConstraint {
    /// Factor name
    pub name: String,
    /// Exposures per asset `(n_assets,)`
    pub exposures: Vec<f64>,
    /// Target exposure (0 = fully neutral)
    #[serde(default)]
    pub target: f64,
    /// Allowed deviation from target
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
}

fn default_tolerance() -> f64 {
    0.05
}

// ── Feasibility checking ─────────────────────────────────────────────────────

/// Feasibility check errors.
#[derive(Debug, Clone, PartialEq)]
pub enum FeasibilityError {
    /// `long_only=true` + `market_neutral=true` → no solution (w = 0 is the
    /// only feasible point).
    LongOnlyMarketNeutralConflict,
    /// `max_assets` too small for `market_neutral`.
    MaxAssetsTooSmall,
    /// Group constraint sum exceeds 1.0.
    GroupConstraintOverflow,
    /// Group constraint identically infeasible.
    GroupConstraintInfeasible(String),
}

impl std::fmt::Display for FeasibilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LongOnlyMarketNeutralConflict => {
                write!(f, "long_only=true + market_neutral=true: only solution is w=0")
            }
            Self::MaxAssetsTooSmall => {
                write!(f, "max_assets < 2 combined with market_neutral=true is infeasible")
            }
            Self::GroupConstraintOverflow => {
                write!(f, "sum of group min_weights exceeds 1.0")
            }
            Self::GroupConstraintInfeasible(msg) => {
                write!(f, "group constraint infeasible: {msg}")
            }
        }
    }
}

/// Check whether a constraint set is feasible.
///
/// Returns `Ok(())` if the constraints can theoretically be satisfied,
/// or a `FeasibilityError` describing the first detected conflict.
///
/// # Constraint relaxation priority
///
/// When constraints are infeasible, relax in this order:
///
/// ```text
/// Priority 1 (never relax): long_only, market_neutral
/// Priority 2 (relaxable):    max_position, min_position
/// Priority 3 (very relaxable): turnover_limit, group_constraints
/// Priority 4 (most relaxable): factor_exposure_constraints
/// ```
pub fn check_feasibility(constraints: &OptimizerConstraints) -> Result<(), FeasibilityError> {
    // 1. long_only + market_neutral conflict
    if constraints.long_only && constraints.market_neutral {
        return Err(FeasibilityError::LongOnlyMarketNeutralConflict);
    }

    // 2. max_assets too small for market_neutral
    if constraints.market_neutral {
        if let Some(max) = constraints.max_assets {
            if max < 2 {
                return Err(FeasibilityError::MaxAssetsTooSmall);
            }
        }
    }

    // 3. Group constraint overflow
    let min_sum: f64 = constraints
        .group_constraints
        .iter()
        .filter_map(|g| g.min_weight)
        .sum();
    if min_sum > 1.0 {
        return Err(FeasibilityError::GroupConstraintOverflow);
    }

    if !constraints.group_constraints.is_empty() {
        let max_sum: f64 = constraints
            .group_constraints
            .iter()
            .filter_map(|g| g.max_weight)
            .sum();
        if constraints.full_investment && !constraints.market_neutral && max_sum < 1.0 {
            return Err(FeasibilityError::GroupConstraintInfeasible(
                "sum of group max_weights < 1.0 but full_investment=true".into(),
            ));
        }
    }

    Ok(())
}

// ── Constraint application helpers ──────────────────────────────────────────

/// Clamp weights to [lower, upper] bounds.
pub(crate) fn apply_bounds(
    weights: &mut Array1<f64>,
    constraints: &OptimizerConstraints,
) {
    let lb = constraints.min_position.unwrap_or(if constraints.long_only { 0.0 } else { f64::NEG_INFINITY });
    let ub = constraints.max_position.unwrap_or(f64::INFINITY);
    for w in weights.iter_mut() {
        *w = w.clamp(lb, ub);
    }
}

/// Normalize weights to sum to `target`.
pub(crate) fn normalize_sum(weights: &mut Array1<f64>, target: f64) {
    let s: f64 = weights.iter().sum();
    if s.abs() > 1e-12 {
        let scale = target / s;
        for w in weights.iter_mut() {
            *w *= scale;
        }
    } else {
        // Degenerate case: all weights near zero → equal weight
        let n = weights.len() as f64;
        if n > 0.0 && target != 0.0 {
            for w in weights.iter_mut() {
                *w = target / n;
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// [SYNTHETIC] feasibility: long_only ∩ market_neutral → conflict.
    #[test]
    fn test_feasibility_long_only_market_neutral_conflict() {
        let c = OptimizerConstraints {
            long_only: true,
            market_neutral: true,
            ..Default::default()
        };
        assert_eq!(
            check_feasibility(&c).unwrap_err(),
            FeasibilityError::LongOnlyMarketNeutralConflict
        );
    }

    /// [SYNTHETIC] feasibility: group max_weights sum < 1.0 with
    /// full_investment → infeasible.
    #[test]
    fn test_feasibility_group_underflow() {
        let c = OptimizerConstraints {
            full_investment: true,
            market_neutral: false,
            group_constraints: vec![GroupConstraint {
                name: "A".into(),
                members: vec![0, 1],
                min_weight: None,
                max_weight: Some(0.3),
            }],
            ..Default::default()
        };
        assert!(check_feasibility(&c).is_err());
    }

    /// [SYNTHETIC] feasibility: OK constraints pass.
    #[test]
    fn test_feasibility_ok() {
        let c = OptimizerConstraints {
            long_only: true,
            max_position: Some(0.1),
            max_assets: Some(100),
            ..Default::default()
        };
        assert!(check_feasibility(&c).is_ok());
    }

    /// [SYNTHETIC] feasibility: group min_weights sum > 1.0 → overflow.
    #[test]
    fn test_feasibility_group_overflow() {
        let c = OptimizerConstraints {
            group_constraints: vec![
                GroupConstraint {
                    name: "A".into(),
                    members: vec![0],
                    min_weight: Some(0.6),
                    max_weight: None,
                },
                GroupConstraint {
                    name: "B".into(),
                    members: vec![1],
                    min_weight: Some(0.6),
                    max_weight: None,
                },
            ],
            ..Default::default()
        };
        assert_eq!(
            check_feasibility(&c).unwrap_err(),
            FeasibilityError::GroupConstraintOverflow
        );
    }

    /// [SYNTHETIC] apply_bounds: long_only clamps negatives to 0.
    #[test]
    fn test_apply_bounds_long_only() {
        let c = OptimizerConstraints {
            long_only: true,
            ..Default::default()
        };
        let mut w = Array1::from_vec(vec![-0.1, 0.2, 0.5, -0.3]);
        apply_bounds(&mut w, &c);
        assert!(w.iter().all(|&x| x >= 0.0));
    }

    /// [SYNTHETIC] normalize_sum: scales to target 1.0.
    #[test]
    fn test_normalize_sum() {
        let mut w = Array1::from_vec(vec![0.2, 0.3, 0.1]);
        normalize_sum(&mut w, 1.0);
        assert!((w.sum() - 1.0).abs() < 1e-12);
    }
}
