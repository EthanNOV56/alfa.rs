//! Backtest configuration types.

use crate::WeightMethod;

/// Slippage configuration with separate buy/sell rates.
#[derive(Debug, Clone)]
pub struct SlippageConfig {
    /// Threshold for large volume trades
    pub large_volume_threshold: f64,
    /// Buy-side slippage rate
    pub buy_slippage: f64,
    /// Sell-side slippage rate
    pub sell_slippage: f64,
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            large_volume_threshold: 1_000_000.0,
            buy_slippage: 0.001,
            sell_slippage: 0.001,
        }
    }
}

/// Fee configuration with separate buy/sell commission.
#[derive(Debug, Clone)]
pub struct FeeConfig {
    /// Buy-side commission rate
    pub buy_commission: f64,
    /// Sell-side commission rate
    pub sell_commission: f64,
    /// Slippage configuration
    pub slippage: SlippageConfig,
    /// Minimum commission per trade
    pub min_commission: f64,
}

impl Default for FeeConfig {
    fn default() -> Self {
        Self {
            buy_commission: 0.0005,
            sell_commission: 0.0015,
            slippage: SlippageConfig::default(),
            min_commission: 5.0,
        }
    }
}

/// Position configuration for long/short portfolios
#[derive(Debug, Clone)]
pub struct PositionConfig {
    /// Long position ratio (e.g., 1.0 for 100%)
    pub long_ratio: f64,
    /// Short position ratio (e.g., 0.5 for 50% - spot shorting)
    pub short_ratio: f64,
    /// Whether to use market neutral strategy (long - short)
    pub market_neutral: bool,
}

impl Default for PositionConfig {
    fn default() -> Self {
        Self {
            long_ratio: 1.0,
            short_ratio: 1.0,
            market_neutral: true,
        }
    }
}

/// Limit up/down handling configuration
#[derive(Debug, Clone)]
pub struct LimitUpDownConfig {
    /// Whether to enable limit up/down handling
    pub enabled: bool,
}

impl Default for LimitUpDownConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

/// Execution configuration: commissions and slippage.
///
/// Separated from BacktestConfig so execution costs can be
/// configured independently of alpha/strategy parameters.
#[derive(Debug, Clone)]
pub struct ExecConfig {
    /// Buy-side commission rate
    pub buy_commission: f64,
    /// Sell-side commission rate
    pub sell_commission: f64,
    /// Buy-side slippage rate
    pub buy_slippage: f64,
    /// Sell-side slippage rate
    pub sell_slippage: f64,
}

impl Default for ExecConfig {
    fn default() -> Self {
        Self {
            buy_commission: 0.0005,
            sell_commission: 0.0015,
            buy_slippage: 0.001,
            sell_slippage: 0.001,
        }
    }
}

impl ExecConfig {
    /// Apply execution config to a FeeConfig.
    pub fn apply_to_fee_config(&self, fee: &mut FeeConfig) {
        fee.buy_commission = self.buy_commission;
        fee.sell_commission = self.sell_commission;
        fee.slippage.buy_slippage = self.buy_slippage;
        fee.slippage.sell_slippage = self.sell_slippage;
    }
}

/// Backtest configuration - follows dependency inversion principle
/// Only contains configuration parameters, no data
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Number of quantile groups for factor ranking
    pub quantiles: usize,
    /// Weight method for portfolio construction
    pub weight_method: WeightMethod,
    /// Number of top stocks to go long
    pub long_top_n: usize,
    /// Number of top stocks to go short
    pub short_top_n: usize,
    /// Rebalance frequency in trading days (1 = daily, 5 = weekly, 20 = monthly)
    pub rebalance_freq: usize,
    /// Fee configuration (commission, slippage)
    pub fee_config: FeeConfig,
    /// Position configuration (long/short ratios)
    pub position_config: PositionConfig,
    /// Limit up/down handling configuration
    pub limit_up_down_config: LimitUpDownConfig,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            quantiles: 10,
            weight_method: WeightMethod::Equal,
            long_top_n: 1,
            short_top_n: 1,
            rebalance_freq: 1,
            fee_config: FeeConfig::default(),
            position_config: PositionConfig::default(),
            limit_up_down_config: LimitUpDownConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_config_default() {
        let config = SlippageConfig::default();
        assert_eq!(config.large_volume_threshold, 1_000_000.0);
        assert_eq!(config.buy_slippage, 0.001);
        assert_eq!(config.sell_slippage, 0.001);
    }

    #[test]
    fn test_fee_config_default() {
        let config = FeeConfig::default();
        assert_eq!(config.buy_commission, 0.0005);
        assert_eq!(config.sell_commission, 0.0015);
        assert_eq!(config.min_commission, 5.0);
    }

    #[test]
    fn test_position_config_default() {
        let config = PositionConfig::default();
        assert_eq!(config.long_ratio, 1.0);
        assert_eq!(config.short_ratio, 1.0);
        assert!(config.market_neutral);
    }

    #[test]
    fn test_slippage_config_custom() {
        let config = SlippageConfig {
            large_volume_threshold: 5_000_000.0,
            buy_slippage: 0.002,
            sell_slippage: 0.003,
        };
        assert_eq!(config.large_volume_threshold, 5_000_000.0);
        assert_eq!(config.buy_slippage, 0.002);
        assert_eq!(config.sell_slippage, 0.003);
    }

    #[test]
    fn test_fee_config_custom() {
        let slippage = SlippageConfig::default();
        let config = FeeConfig {
            buy_commission: 0.001,
            sell_commission: 0.002,
            slippage,
            min_commission: 10.0,
        };
        assert_eq!(config.buy_commission, 0.001);
        assert_eq!(config.sell_commission, 0.002);
        assert_eq!(config.min_commission, 10.0);
    }

    #[test]
    fn test_position_config_long_only() {
        let config = PositionConfig {
            long_ratio: 1.0,
            short_ratio: 0.0,
            market_neutral: false,
        };
        assert_eq!(config.long_ratio, 1.0);
        assert_eq!(config.short_ratio, 0.0);
        assert!(!config.market_neutral);
    }

    #[test]
    fn test_position_config_long_short() {
        let config = PositionConfig {
            long_ratio: 1.0,
            short_ratio: 0.5,
            market_neutral: true,
        };
        assert_eq!(config.long_ratio, 1.0);
        assert_eq!(config.short_ratio, 0.5);
        assert!(config.market_neutral);
    }
}
