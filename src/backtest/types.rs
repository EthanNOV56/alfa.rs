//! Backtest result types and serialization.

use ndarray::{Array1, Array2};

/// Return type for automatic return computation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReturnType {
    /// Holding return: (close[t+1] - close[t]) / close[t]
    Holding,
    /// Intraday trading return: (close[t] - open[t]) / open[t]
    Trading,
}

/// Enhanced backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Trading dates (YYYYMMDD), length n_days
    pub dates: Vec<i64>,
    /// Group returns (quantile-based)
    pub group_returns: Array2<f64>,
    /// Group cumulative returns
    pub group_cum_returns: Array2<f64>,
    /// Long-short daily returns
    pub long_short_returns: Array1<f64>,
    /// Long-short cumulative return (final scalar)
    pub long_short_cum_return: f64,
    /// Long-short cumulative NAV curve [n_days-1], starts at 1.0
    pub long_short_cum_returns: Array1<f64>,
    /// Long leg cumulative NAV curve [n_days-1]
    pub long_cum_returns: Array1<f64>,
    /// Short leg cumulative NAV curve [n_days-1]
    pub short_cum_returns: Array1<f64>,
    /// Passive benchmark daily returns (equal-weight all tradable stocks)
    pub passive_returns: Array1<f64>,
    /// Passive benchmark cumulative NAV curve [n_days-1]
    pub passive_cum_returns: Array1<f64>,
    /// IC series
    pub ic_series: Array1<f64>,
    /// IC mean
    pub ic_mean: f64,
    /// IC IR (Information Ratio)
    pub ic_ir: f64,
    /// Long group IC mean (top quantile groups only)
    pub long_ic_mean: f64,
    /// Long group IC IR
    pub long_ic_ir: f64,
    /// Short group IC mean (bottom quantile groups only)
    pub short_ic_mean: f64,
    /// Short group IC IR
    pub short_ic_ir: f64,
    /// Long+short combined IC mean
    pub long_short_ic_mean: f64,
    /// Long+short combined IC IR
    pub long_short_ic_ir: f64,
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Turnover rate (group-label-based)
    pub turnover: f64,
    /// Weight-based turnover rate (sum of absolute weight changes)
    pub weight_turnover: f64,
    /// Win rate: fraction of days with positive long-short return
    pub win_rate: f64,
    /// Calmar ratio: annualized_return / max_drawdown
    pub calmar_ratio: f64,
    /// Long-only returns
    pub long_returns: Array1<f64>,
    /// Short-only returns
    pub short_returns: Array1<f64>,
}

impl Default for BacktestResult {
    fn default() -> Self {
        Self {
            dates: vec![],
            group_returns: Array2::zeros((0, 0)),
            group_cum_returns: Array2::zeros((0, 0)),
            long_short_returns: Array1::zeros(0),
            long_short_cum_return: 0.0,
            long_short_cum_returns: Array1::zeros(0),
            long_cum_returns: Array1::zeros(0),
            short_cum_returns: Array1::zeros(0),
            passive_returns: Array1::zeros(0),
            passive_cum_returns: Array1::zeros(0),
            ic_series: Array1::zeros(0),
            ic_mean: 0.0,
            ic_ir: 0.0,
            long_ic_mean: 0.0,
            long_ic_ir: 0.0,
            short_ic_mean: 0.0,
            short_ic_ir: 0.0,
            long_short_ic_mean: 0.0,
            long_short_ic_ir: 0.0,
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            turnover: 0.0,
            weight_turnover: 0.0,
            win_rate: 0.0,
            calmar_ratio: 0.0,
            long_returns: Array1::zeros(0),
            short_returns: Array1::zeros(0),
        }
    }
}

impl BacktestResult {
    /// Write NAV curves to CSV file (date,nv,group).
    pub fn to_csv<P: AsRef<std::path::Path>>(&self, path: P) -> csv::Result<()> {
        let mut wtr = csv::Writer::from_path(&path)?;
        self.write_nav_csv(&mut wtr)
    }

    /// Write group NAV curves to CSV writer.
    pub fn write_nav_csv<W: std::io::Write>(&self, wtr: &mut csv::Writer<W>) -> csv::Result<()> {
        let dates = &self.dates;
        wtr.write_record(&["date", "nv", "group"])?;
        let fmt_date = |d: i64| -> String {
            let yr = d / 10000;
            let mo = (d % 10000) / 100;
            let dy = d % 100;
            format!("{:04}-{:02}-{:02}", yr, mo, dy)
        };
        for g in 0..self.group_returns.ncols() {
            wtr.write_record(&[&fmt_date(dates[0]), "1.0", &g.to_string()])?;
            for t in 0..self.group_returns.nrows() {
                let nv = 1.0 + self.group_cum_returns[[t, g]];
                let date_idx = t + 1;
                if date_idx < dates.len() {
                    wtr.write_record(&[
                        &fmt_date(dates[date_idx]),
                        &nv.to_string(),
                        &g.to_string(),
                    ])?;
                }
            }
        }
        // Write long-short NAV curve
        wtr.write_record(&[&fmt_date(dates[0]), "1.0", "long_short"])?;
        for t in 0..self.long_short_cum_returns.len() {
            let date_idx = t + 1;
            if date_idx < dates.len() {
                wtr.write_record(&[
                    &fmt_date(dates[date_idx]),
                    &self.long_short_cum_returns[t].to_string(),
                    "long_short",
                ])?;
            }
        }
        Ok(())
    }
}
