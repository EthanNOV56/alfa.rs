/**
 * Type definitions for the Alfa.rs visualization
 */

export interface BacktestRequest {
  factor: number[][];
  returns: number[][];
  dates?: string[];
  quantiles?: number;
  weight_method?: string;
  long_top_n?: number;
  short_top_n?: number;
  commission_rate?: number;
}

export interface NavData {
  dates: string[];
  quantiles: number[][];
  long_short: number[];
  benchmark: number[];
  ic_series: number[];
  metrics: Metrics;
}

export interface Metrics {
  long_short_cum_return: number;
  total_return: number;
  annualized_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  turnover: number;
  ic_mean: number;
  ic_ir: number;
}

export interface ChartTooltip {
  date: string;
  values: Record<string, number>;
  changes: Record<string, number>;
}
