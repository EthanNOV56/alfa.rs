//! Stress testing — scenario analysis for portfolio risk.
//!
//! Applies factor shocks, volatility multipliers, and correlation shocks
//! to a Barra-style risk decomposition to estimate P&L under stress.

use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// A stress scenario definition.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct StressScenario {
    pub name: String,
    pub description: String,
    /// Per-factor shocks in standard deviation multiples.
    /// e.g. `{"Beta": -3.0}` means the Beta factor drops 3σ.
    pub factor_shocks: HashMap<String, f64>,
    /// Multiply all factor volatilities by this factor (>1 = more volatile).
    pub volatility_multiplier: Option<f64>,
    /// Multiply all pairwise correlations by this factor (1.0 = unchanged).
    pub correlation_multiplier: Option<f64>,
}

/// Result for a single stress scenario.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct StressTestResult {
    pub scenario_name: String,
    pub pre_shock_nav: f64,
    pub post_shock_nav: f64,
    pub loss_amount: f64,
    pub loss_pct: f64,
    /// Per-factor contribution to the loss.
    pub factor_losses: HashMap<String, f64>,
}

impl StressScenario {
    /// 2008 Global Financial Crisis: equity crash, vol spike, correlation → 1.
    pub fn gfc_2008() -> Self {
        Self {
            name: "2008 GFC".into(),
            description: "Global Financial Crisis: equity -3σ, vol ×3, correlation → 1".into(),
            factor_shocks: HashMap::from([
                ("Beta".into(), -3.0),
                ("Momentum".into(), -2.0),
            ]),
            volatility_multiplier: Some(3.0),
            correlation_multiplier: Some(2.0),
        }
    }

    /// 2015 China A-share crash: liquidity crunch.
    pub fn china_2015() -> Self {
        Self {
            name: "2015 China Crash".into(),
            description: "China A-share crash: Beta -2σ, Momentum -1σ, Liquidity -3σ".into(),
            factor_shocks: HashMap::from([
                ("Beta".into(), -2.0),
                ("Momentum".into(), -1.0),
                ("Liquidity".into(), -3.0),
            ]),
            volatility_multiplier: Some(2.0),
            correlation_multiplier: None,
        }
    }

    /// 2020 COVID market shock.
    pub fn covid_2020() -> Self {
        Self {
            name: "COVID-19".into(),
            description: "COVID-19 shock: Beta -2.5σ, vol ×2.5".into(),
            factor_shocks: HashMap::from([("Beta".into(), -2.5)]),
            volatility_multiplier: Some(2.5),
            correlation_multiplier: None,
        }
    }
}

/// Run a single stress scenario against a portfolio risk decomposition.
///
/// # Parameters
/// - `weights`: portfolio weights (n_assets,)
/// - `exposures`: factor exposure matrix (n_assets × K)
/// - `factor_cov`: factor covariance matrix (K × K)
/// - `specific_vars`: per-asset specific variances (n_assets,)
/// - `factor_names`: K factor names (for labelling)
/// - `nav`: current portfolio NAV
pub fn run_stress_test(
    scenario: &StressScenario,
    weights: &Array1<f64>,
    exposures: &Array2<f64>,
    factor_cov: &Array2<f64>,
    specific_vars: &Array1<f64>,
    factor_names: &[String],
    nav: f64,
) -> StressTestResult {
    let (n_assets, k) = exposures.dim();

    // 1. Build stressed factor covariance
    let mut stressed_cov = factor_cov.clone();

    // Apply volatility multiplier
    if let Some(vm) = scenario.volatility_multiplier {
        for i in 0..k {
            for j in 0..k {
                stressed_cov[[i, j]] *= vm;
            }
        }
    }

    // Apply correlation multiplier: C_ij → c_mult × C_ij for i≠j
    if let Some(cm) = scenario.correlation_multiplier {
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    stressed_cov[[i, j]] *= cm;
                }
            }
        }
    }

    // 2. Weighted exposure: e = Bᵀ w
    let mut e = Array1::zeros(k);
    for k_idx in 0..k {
        let mut sum = 0.0;
        for a in 0..n_assets {
            sum += weights[a] * exposures[[a, k_idx]];
        }
        e[k_idx] = sum;
    }

    // 3. Systematic variance under stress
    let mut sys_var_stressed = 0.0;
    for p in 0..k {
        let mut row_sum = 0.0;
        for q in 0..k {
            row_sum += stressed_cov[[p, q]] * e[q];
        }
        sys_var_stressed += e[p] * row_sum;
    }

    // 4. Specific variance (scaled by vol multiplier too)
    let vm = scenario.volatility_multiplier.unwrap_or(1.0);
    let mut spec_var = 0.0;
    for a in 0..n_assets {
        let sv = specific_vars[a];
        if sv.is_finite() && sv > 0.0 {
            spec_var += weights[a] * weights[a] * sv * vm;
        }
    }

    let stressed_var = sys_var_stressed + spec_var;
    let stressed_vol = stressed_var.sqrt();

    // 5. Factor-specific shock losses
    let mut factor_losses = HashMap::new();
    for k_idx in 0..k {
        let name = factor_names
            .get(k_idx)
            .cloned()
            .unwrap_or_else(|| format!("factor_{}", k_idx));
        let shock = scenario.factor_shocks.get(&name).copied().unwrap_or(0.0);
        if shock != 0.0 {
            let factor_vol = factor_cov[[k_idx, k_idx]].sqrt();
            let loss = e[k_idx] * shock * factor_vol * nav;
            factor_losses.insert(name, loss);
        }
    }

    // 6. Total loss = factor shocks + 1σ stressed vol move
    let total_factor_loss: f64 = factor_losses.values().sum();
    let vol_loss = stressed_vol * nav;
    let total_loss = total_factor_loss + vol_loss;

    StressTestResult {
        scenario_name: scenario.name.clone(),
        pre_shock_nav: nav,
        post_shock_nav: (nav + total_loss).max(0.0),
        loss_amount: total_loss,
        loss_pct: if nav > 0.0 { total_loss / nav } else { 0.0 },
        factor_losses,
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// All tests use **synthetic data** — no external data sources.

    #[test]
    fn no_shock_no_loss() {
        // Synthetic: identity cov, zero exposures → zero loss
        let k = 2;
        let n_assets = 3;
        let weights = Array1::from_vec(vec![0.4, 0.4, 0.2]);
        let exposures = Array2::zeros((n_assets, k));
        let factor_cov = Array2::eye(k);
        let specific_vars = Array1::zeros(n_assets);
        let factor_names = vec!["f0".into(), "f1".into()];

        let scenario = StressScenario {
            name: "no_shock".into(),
            description: "zero shock".into(),
            factor_shocks: HashMap::new(),
            volatility_multiplier: None,
            correlation_multiplier: None,
        };

        let result = run_stress_test(
            &scenario,
            &weights,
            &exposures,
            &factor_cov,
            &specific_vars,
            &factor_names,
            1_000_000.0,
        );
        assert!((result.loss_pct - 0.0).abs() < 1e-10);
    }

    #[test]
    fn vol_doubling_increases_risk() {
        // Synthetic: unit exposures, unit cov → vol multiplier increases loss
        let k = 2;
        let n_assets = 2;
        let weights = Array1::from_vec(vec![0.5, 0.5]);
        let exposures = Array2::from_shape_vec((n_assets, k), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let factor_cov = Array2::eye(k);
        let specific_vars = Array1::zeros(n_assets);
        let factor_names = vec!["f0".into(), "f1".into()];

        // Baseline (no multiplier)
        let base = run_stress_test(
            &StressScenario {
                name: "base".into(),
                description: "".into(),
                factor_shocks: HashMap::new(),
                volatility_multiplier: None,
                correlation_multiplier: None,
            },
            &weights, &exposures, &factor_cov, &specific_vars, &factor_names, 1_000_000.0,
        );

        // Vol ×2
        let stressed = run_stress_test(
            &StressScenario {
                name: "vol2x".into(),
                description: "".into(),
                factor_shocks: HashMap::new(),
                volatility_multiplier: Some(2.0),
                correlation_multiplier: None,
            },
            &weights, &exposures, &factor_cov, &specific_vars, &factor_names, 1_000_000.0,
        );

        assert!(stressed.loss_pct.abs() > base.loss_pct.abs());
    }

    #[test]
    fn builtin_scenarios_dont_panic() {
        // Synthetic: minimal valid inputs, verify built-in scenarios compute without panic
        let k = 2;
        let n_assets = 3;
        let weights = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let exposures = Array2::from_shape_vec((n_assets, k), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let factor_cov = Array2::eye(k);
        let specific_vars = Array1::from_vec(vec![0.01, 0.01, 0.01]);
        let factor_names = vec!["Beta".into(), "Momentum".into()];

        for scenario in [StressScenario::gfc_2008(), StressScenario::china_2015(), StressScenario::covid_2020()] {
            let result = run_stress_test(
                &scenario,
                &weights, &exposures, &factor_cov, &specific_vars, &factor_names, 1_000_000.0,
            );
            assert!(result.loss_pct.is_finite());
            assert!(!result.scenario_name.is_empty());
        }
    }
}
