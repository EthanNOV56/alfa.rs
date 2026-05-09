//! Performance attribution — Brinson model + factor-based attribution.
//!
//! Decomposes active return (portfolio − benchmark) into:
//! - Allocation effect: over/under-weighting sectors
//! - Selection effect: stock-picking within sectors
//! - Interaction effect: cross-term
//!
//! And factor-based: portfolio return = factor contribution + specific return.

use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Per-sector attribution detail.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SectorDetail {
    pub sector_name: String,
    pub portfolio_weight: f64,
    pub benchmark_weight: f64,
    pub portfolio_return: f64,
    pub benchmark_return: f64,
    pub allocation_effect: f64,
    pub selection_effect: f64,
}

/// Brinson performance attribution result.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PerformanceAttribution {
    /// Allocation effect: Σ_s (w_p,s − w_b,s) × (r_b,s − r_b)
    pub allocation_effect: f64,
    /// Selection effect: Σ_s w_b,s × (r_p,s − r_b,s)
    pub selection_effect: f64,
    /// Interaction effect: Σ_s (w_p,s − w_b,s) × (r_p,s − r_b,s)
    pub interaction_effect: f64,
    /// Total active return = allocation + selection + interaction
    pub total_active_return: f64,
    /// Per-sector breakdown.
    pub sector_details: Vec<SectorDetail>,
}

/// Brinson attribution calculator.
///
/// # Parameters
/// - `sector_assignments`: list of (sector_name, Vec<asset_indices>)
///
/// # Example
/// ```
/// // 4 assets: A,B in "tech", C,D in "finance"
/// let attributor = BrinsonAttributor::new(vec![
///     ("tech".into(), vec![0, 1]),
///     ("finance".into(), vec![2, 3]),
/// ]);
/// ```
pub struct BrinsonAttributor {
    sector_names: Vec<String>,
    /// For each sector, the indices of assets in that sector.
    sector_members: Vec<Vec<usize>>,
}

impl BrinsonAttributor {
    pub fn new(sector_assignments: Vec<(String, Vec<usize>)>) -> Self {
        let (names, members): (Vec<_>, Vec<_>) = sector_assignments.into_iter().unzip();
        Self {
            sector_names: names,
            sector_members: members,
        }
    }

    /// Compute Brinson attribution for a single period.
    ///
    /// # Parameters
    /// - `portfolio_weights`: weight per asset (n_assets,), sum to 1.0
    /// - `benchmark_weights`: benchmark weight per asset (n_assets,), sum to 1.0
    /// - `asset_returns`: return per asset over the period (n_assets,)
    ///
    /// # Returns
    /// `PerformanceAttribution` with all three effects and sector detail.
    pub fn attribute(
        &self,
        portfolio_weights: &Array1<f64>,
        benchmark_weights: &Array1<f64>,
        asset_returns: &Array1<f64>,
    ) -> PerformanceAttribution {
        let n_assets = portfolio_weights.len();

        // Aggregate sector-level weights and returns
        let mut sector_details = Vec::with_capacity(self.sector_names.len());

        // Benchmark total return: r_b = Σ_i w_b,i × r_i
        let mut benchmark_total_return = 0.0;
        for i in 0..n_assets {
            benchmark_total_return += benchmark_weights[i] * asset_returns[i];
        }

        let mut total_allocation = 0.0;
        let mut total_selection = 0.0;
        let mut total_interaction = 0.0;

        for s in 0..self.sector_names.len() {
            let members = &self.sector_members[s];

            // Sector weights
            let mut wp_s = 0.0;
            let mut wb_s = 0.0;
            for &a in members {
                wp_s += portfolio_weights[a];
                wb_s += benchmark_weights[a];
            }

            // Sector returns (weighted within sector)
            let mut rp_s = 0.0;
            let mut rb_s = 0.0;
            for &a in members {
                if wp_s > 0.0 {
                    rp_s += (portfolio_weights[a] / wp_s) * asset_returns[a];
                }
                if wb_s > 0.0 {
                    rb_s += (benchmark_weights[a] / wb_s) * asset_returns[a];
                }
            }

            // Brinson effects per sector
            let alloc = (wp_s - wb_s) * (rb_s - benchmark_total_return);
            let select = wb_s * (rp_s - rb_s);
            let interact = (wp_s - wb_s) * (rp_s - rb_s);

            total_allocation += alloc;
            total_selection += select;
            total_interaction += interact;

            sector_details.push(SectorDetail {
                sector_name: self.sector_names[s].clone(),
                portfolio_weight: wp_s,
                benchmark_weight: wb_s,
                portfolio_return: rp_s,
                benchmark_return: rb_s,
                allocation_effect: alloc,
                selection_effect: select,
            });
        }

        PerformanceAttribution {
            allocation_effect: total_allocation,
            selection_effect: total_selection,
            interaction_effect: total_interaction,
            total_active_return: total_allocation + total_selection + total_interaction,
            sector_details,
        }
    }
}

// --------------------------------------------------------------------------
// Factor-based performance attribution
// --------------------------------------------------------------------------

/// Decompose portfolio active return into factor contributions and specific return.
///
/// Given estimated factor returns `f` (n_days × K), asset-level specific returns
/// `ε` (n_days × n_assets), and portfolio weights `w` (n_assets,):
///
/// ```text
///   portfolio_return = Σ_k exposure_k × factor_return_k + specific_return
///   exposure_k = Σ_a w[a] × B[a,k]
/// ```
///
/// Returns a map from factor name → contribution, plus the specific return.
pub fn factor_performance_attribution(
    weights: &Array1<f64>,
    factor_returns: &Array2<f64>,
    specific_returns: &Array1<f64>,
    exposures: &Array2<f64>,
    factor_names: &[String],
) -> HashMap<String, f64> {
    let (_n_days, k) = factor_returns.dim();
    let n_assets = weights.len();
    let mut contributions = HashMap::new();

    for k_idx in 0..k {
        // Average exposure: Σ_a w[a] × B[a,k]
        let mut avg_exposure = 0.0;
        for a in 0..n_assets {
            avg_exposure += weights[a] * exposures[[a, k_idx]];
        }
        // Factor return over period (mean)
        let mut factor_ret = 0.0;
        let mut count = 0;
        for d in 0..factor_returns.dim().0 {
            let fr = factor_returns[[d, k_idx]];
            if fr.is_finite() {
                factor_ret += fr;
                count += 1;
            }
        }
        if count > 0 {
            factor_ret /= count as f64;
        }
        let name = factor_names
            .get(k_idx)
            .cloned()
            .unwrap_or_else(|| format!("factor_{}", k_idx));
        contributions.insert(name, avg_exposure * factor_ret);
    }

    // Specific contribution
    let mut spec_contrib = 0.0;
    for a in 0..n_assets {
        if specific_returns[a].is_finite() {
            spec_contrib += weights[a] * specific_returns[a];
        }
    }
    contributions.insert("specific".into(), spec_contrib);

    contributions
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// All tests use **synthetic data** — no external data sources.

    #[test]
    fn brinson_overweight_winning_sector() {
        // Synthetic: 4 assets, 2 sectors.
        // Portfolio overweights "tech" which outperforms → positive allocation effect.
        let attributor = BrinsonAttributor::new(vec![
            ("tech".into(), vec![0, 1]),
            ("finance".into(), vec![2, 3]),
        ]);

        // Benchmark: equal-weight across assets → 25% each, 50% per sector
        let bw = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        // Portfolio: overweight tech (40% each = 80%), underweight finance (20% each = 40%)
        let pw = Array1::from_vec(vec![0.40, 0.40, 0.10, 0.10]);
        // Returns: tech does well (+10%), finance does poorly (+2%)
        let ret = Array1::from_vec(vec![0.10, 0.10, 0.02, 0.02]);

        let attr = attributor.attribute(&pw, &bw, &ret);

        // Active return = portfolio_return - benchmark_return
        // portfolio: 0.4*0.1+0.4*0.1+0.1*0.02+0.1*0.02 = 0.084
        // benchmark: 0.25*0.1+0.25*0.1+0.25*0.02+0.25*0.02 = 0.06
        // active = 0.024
        let active = attr.total_active_return;
        let expected_active = 0.084 - 0.06;
        assert!(
            (active - expected_active).abs() < 1e-12,
            "Active return should be {:.6}, got {:.6}",
            expected_active,
            active
        );

        // Allocation effect should be positive (overweighted winning sector)
        assert!(
            attr.allocation_effect > 0.0,
            "Allocation effect should be positive when overweighting winning sector, got {:.6}",
            attr.allocation_effect
        );
    }

    #[test]
    fn brinson_equal_weights_zero_attribution() {
        // Synthetic: when portfolio = benchmark, all effects should be zero.
        let attributor = BrinsonAttributor::new(vec![("a".into(), vec![0]), ("b".into(), vec![1])]);

        let w = Array1::from_vec(vec![0.5, 0.5]);
        let ret = Array1::from_vec(vec![0.05, 0.03]);

        let attr = attributor.attribute(&w, &w, &ret);
        assert!((attr.allocation_effect).abs() < 1e-12);
        assert!((attr.selection_effect).abs() < 1e-12);
        assert!((attr.interaction_effect).abs() < 1e-12);
        assert!((attr.total_active_return).abs() < 1e-12);
    }

    #[test]
    fn brinson_three_effects_sum_to_active() {
        // Synthetic: random weights and returns, verify Brinson identity.
        let attributor = BrinsonAttributor::new(vec![
            ("s0".into(), vec![0, 1]),
            ("s1".into(), vec![2, 3, 4]),
        ]);

        let pw = Array1::from_vec(vec![0.3, 0.1, 0.2, 0.3, 0.1]);
        let bw = Array1::from_vec(vec![0.2, 0.2, 0.2, 0.2, 0.2]);
        let ret = Array1::from_vec(vec![0.08, 0.04, 0.02, 0.06, 0.10]);

        let attr = attributor.attribute(&pw, &bw, &ret);

        let port_ret = pw.dot(&ret);
        let bench_ret = bw.dot(&ret);
        let active = port_ret - bench_ret;

        let sum_effects = attr.allocation_effect + attr.selection_effect + attr.interaction_effect;
        assert!(
            (sum_effects - active).abs() < 1e-12,
            "Sum of effects ({:.12}) should equal active return ({:.12})",
            sum_effects,
            active
        );
    }

    // ---------- Factor attribution ----------

    #[test]
    fn factor_attribution_reconstructs_portfolio_return() {
        // Synthetic: 2 assets, 2 factors. Verify factor contributions + specific = portfolio return.
        let weights = Array1::from_vec(vec![0.6, 0.4]);
        let exposures = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        // Factor returns: [0.05, 0.02]
        let factor_returns = Array2::from_shape_vec((1, 2), vec![0.05, 0.02]).unwrap();
        // Specific returns: [0.01, -0.01]
        let specific_returns = Array1::from_vec(vec![0.01, -0.01]);
        let factor_names = vec!["f0".into(), "f1".into()];

        let contribs = factor_performance_attribution(
            &weights,
            &factor_returns,
            &specific_returns,
            &exposures,
            &factor_names,
        );

        // f0 contribution: exposure = 0.6*1.0 + 0.4*0.0 = 0.6; factor_ret = 0.05; contrib = 0.03
        assert!((contribs["f0"] - 0.03).abs() < 1e-10);
        // f1: exposure = 0.6*0 + 0.4*1 = 0.4; factor_ret = 0.02; contrib = 0.008
        assert!((contribs["f1"] - 0.008).abs() < 1e-10);
        // specific: 0.6*0.01 + 0.4*(-0.01) = 0.006 - 0.004 = 0.002
        assert!((contribs["specific"] - 0.002).abs() < 1e-10);

        // Portfolio return: 0.6*(1.0*0.05+0.01) + 0.4*(1.0*0.02-0.01) = 0.6*0.06 + 0.4*0.01 = 0.036+0.004 = 0.04
        let sum_contrib: f64 = contribs.values().sum();
        assert!((sum_contrib - 0.04).abs() < 1e-10);
    }
}
