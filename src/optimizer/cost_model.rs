//! Transaction cost model.
//!
//! Research basis: reports #7, #8, #9, #10.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Trait for transaction cost computation.
pub trait CostModel: Send + Sync {
    /// Compute per-asset transaction cost vector.
    ///
    /// # Parameters
    /// - `target_weights`: target portfolio weights
    /// - `prev_weights`: current portfolio weights
    /// - `prices`: per-asset prices
    /// - `volumes`: per-asset volumes (for impact estimation)
    fn compute_cost(
        &self,
        target_weights: &Array1<f64>,
        prev_weights: &Array1<f64>,
        prices: &Array1<f64>,
        volumes: &Array1<f64>,
    ) -> Array1<f64>;

    /// Gradient of cost w.r.t. target weights.
    fn compute_gradient(
        &self,
        target_weights: &Array1<f64>,
        prev_weights: &Array1<f64>,
        prices: &Array1<f64>,
        volumes: &Array1<f64>,
    ) -> Array1<f64>;
}

// ── NoCost ───────────────────────────────────────────────────────────────────

pub struct NoCost;

impl CostModel for NoCost {
    fn compute_cost(
        &self,
        _target_weights: &Array1<f64>,
        _prev_weights: &Array1<f64>,
        _prices: &Array1<f64>,
        _volumes: &Array1<f64>,
    ) -> Array1<f64> {
        Array1::zeros(_target_weights.len())
    }

    fn compute_gradient(
        &self,
        _target_weights: &Array1<f64>,
        _prev_weights: &Array1<f64>,
        _prices: &Array1<f64>,
        _volumes: &Array1<f64>,
    ) -> Array1<f64> {
        Array1::zeros(_target_weights.len())
    }
}

// ── LinearCost ───────────────────────────────────────────────────────────────

pub struct LinearCost {
    pub buy_rate: f64,
    pub sell_rate: f64,
}

impl CostModel for LinearCost {
    fn compute_cost(
        &self,
        target_weights: &Array1<f64>,
        prev_weights: &Array1<f64>,
        _prices: &Array1<f64>,
        _volumes: &Array1<f64>,
    ) -> Array1<f64> {
        let dw = target_weights - prev_weights;
        dw.mapv(|x| if x > 0.0 { x * self.buy_rate } else { -x * self.sell_rate })
    }

    fn compute_gradient(
        &self,
        target_weights: &Array1<f64>,
        prev_weights: &Array1<f64>,
        _prices: &Array1<f64>,
        _volumes: &Array1<f64>,
    ) -> Array1<f64> {
        let dw = target_weights - prev_weights;
        dw.mapv(|x| {
            if x > 0.0 {
                self.buy_rate
            } else if x < 0.0 {
                -self.sell_rate
            } else {
                0.0
            }
        })
    }
}

// ── QuadraticImpact ─────────────────────────────────────────────────────────

pub struct QuadraticImpact {
    /// Impact coefficient η
    pub eta: f64,
}

impl CostModel for QuadraticImpact {
    fn compute_cost(
        &self,
        target_weights: &Array1<f64>,
        prev_weights: &Array1<f64>,
        _prices: &Array1<f64>,
        volumes: &Array1<f64>,
    ) -> Array1<f64> {
        let dw = (target_weights - prev_weights).mapv(|x| x.abs());
        dw.iter()
            .zip(volumes.iter())
            .map(|(&d, &v)| {
                if v > 0.0 {
                    self.eta * (d / v).sqrt() * d
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn compute_gradient(
        &self,
        target_weights: &Array1<f64>,
        prev_weights: &Array1<f64>,
        _prices: &Array1<f64>,
        volumes: &Array1<f64>,
    ) -> Array1<f64> {
        let dw = target_weights - prev_weights;
        dw.iter()
            .zip(volumes.iter())
            .map(|(&d, &v)| {
                if v > 0.0 && d.abs() > 1e-12 {
                    let sign = if d > 0.0 { 1.0 } else { -1.0 };
                    1.5 * self.eta * (d.abs() / v).sqrt() * sign
                } else {
                    0.0
                }
            })
            .collect()
    }
}

// ── CostModelConfig ──────────────────────────────────────────────────────────

/// Cost model configuration (serde-serializable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CostModelConfig {
    None,
    Linear {
        buy_commission: f64,
        sell_commission: f64,
    },
    Quadratic {
        eta: f64,
    },
    Composite {
        buy_commission: f64,
        sell_commission: f64,
        eta: f64,
    },
}

impl CostModelConfig {
    /// Build a `CostModel` trait object from configuration.
    pub fn build(&self) -> Result<Box<dyn CostModel>, String> {
        match self {
            Self::None => Ok(Box::new(NoCost)),
            Self::Linear {
                buy_commission,
                sell_commission,
            } => Ok(Box::new(LinearCost {
                buy_rate: *buy_commission,
                sell_rate: *sell_commission,
            })),
            Self::Quadratic { eta } => Ok(Box::new(QuadraticImpact { eta: *eta })),
            Self::Composite {
                buy_commission,
                sell_commission,
                eta,
            } => Ok(Box::new(CompositeCost {
                buy_rate: *buy_commission,
                sell_rate: *sell_commission,
                eta: *eta,
            })),
        }
    }
}

// ── CompositeCost ────────────────────────────────────────────────────────────

pub struct CompositeCost {
    pub buy_rate: f64,
    pub sell_rate: f64,
    pub eta: f64,
}

impl CostModel for CompositeCost {
    fn compute_cost(
        &self,
        target_weights: &Array1<f64>,
        prev_weights: &Array1<f64>,
        prices: &Array1<f64>,
        volumes: &Array1<f64>,
    ) -> Array1<f64> {
        let linear_cost = LinearCost {
            buy_rate: self.buy_rate,
            sell_rate: self.sell_rate,
        }
        .compute_cost(target_weights, prev_weights, prices, volumes);
        let impact = QuadraticImpact { eta: self.eta }
            .compute_cost(target_weights, prev_weights, prices, volumes);
        linear_cost + impact
    }

    fn compute_gradient(
        &self,
        target_weights: &Array1<f64>,
        prev_weights: &Array1<f64>,
        prices: &Array1<f64>,
        volumes: &Array1<f64>,
    ) -> Array1<f64> {
        let linear_grad = LinearCost {
            buy_rate: self.buy_rate,
            sell_rate: self.sell_rate,
        }
        .compute_gradient(target_weights, prev_weights, prices, volumes);
        let impact_grad = QuadraticImpact { eta: self.eta }
            .compute_gradient(target_weights, prev_weights, prices, volumes);
        linear_grad + impact_grad
    }
}
