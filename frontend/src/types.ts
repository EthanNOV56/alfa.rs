/**
 * Type definitions for the Alfa.rs visualization
 */

export interface BacktestRequest {
  factor?: number[][];
  returns?: number[][];
  dates?: string[];
  quantiles?: number;
  weight_method?: string;
  long_top_n?: number;
  short_top_n?: number;
  commission_rate?: number;
  /** Data source config for on-demand loading from database */
  dataSource?: DataSourceConfig;
}

export interface NavData {
  dates: string[];
  quantiles: number[][];
  long_short?: number[];  // May be undefined
  longShort?: number[];   // FastAPI camelCase
  benchmark: number[];
  ic_series?: number[];   // May be undefined
  icSeries?: number[];    // FastAPI camelCase
  metrics: Metrics;
}

export interface Metrics {
  long_short_cum_return?: number;
  longShortCumReturn?: number;
  total_return?: number;
  totalReturn?: number;
  annualized_return?: number;
  annualizedReturn?: number;
  sharpe_ratio?: number;
  sharpeRatio?: number;
  max_drawdown?: number;
  maxDrawdown?: number;
  turnover?: number;
  ic_mean?: number;
  icMean?: number;
  ic_ir?: number;
  icIr?: number;
}

export interface ChartTooltip {
  date: string;
  values: Record<string, number>;
  changes: Record<string, number>;
}

// Factor library types
export interface Factor {
  id: string;
  name: string;
  expression: string;
  description: string;
}

export interface FactorListResponse {
  factors: Factor[];
}

export interface FactorComputeRequest {
  factorId: string;
  nDays?: number;
  nAssets?: number;
  /** Data source config for on-demand loading from database */
  dataSource?: DataSourceConfig;
}

export interface FactorComputeResponse {
  factorId: string;
  factor: number[][];
  returns: number[][];
  dates: string[];
}

// GP Mining types
export interface GpMineRequest {
  population_size: number;
  max_generations: number;
  terminal_set: string[];
  function_set: string[];
  n_days?: number;
  n_assets?: number;
  target_ic?: number;
  /// Seed expression for mutation-based GP mining
  seed_expression?: string;
  /** Data source config for on-demand loading from database */
  dataSource?: DataSourceConfig;
}

export interface GpFactor {
  id: string;
  name: string;
  expression: string;
  ic_mean: number;
  ic_ir: number;
  fitness: number;
}

export interface GpMineResponse {
  factors: GpFactor[];
  best_factor: GpFactor;
  bestFactor?: GpFactor;
  generations: number;
  elapsed_time: number;
}

// Alpha Library types
export interface Alpha {
  name: string;
  expression: string;
  description: string;
  dimension: string;
  tags: string[];
  readonly: boolean;
}

export interface AlphaListResponse {
  alphas: Alpha[];
}

export interface SaveAlphaRequest {
  name: string;
  expression: string;
  description?: string;
  dimension?: string;
  tags?: string[];
}

export interface SaveAlphaResponse {
  success: boolean;
  path: string;
  message: string;
}

// Database configuration types
export interface DbConfig {
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
  connected: boolean;
}

export interface SetDbConfigRequest {
  host: string;
  port?: number;
  database?: string;
  username?: string;
  password?: string;
}

export interface SymbolInfo {
  symbol: string;
  name: string;
}

export interface DateRange {
  minDate: string;
  maxDate: string;
}

export interface LoadDataRequest {
  symbols: string[];
  startDate: string;
  endDate: string;
  /** Table name: stock_1d, stock_1m, stock_5m (default: stock_1d) */
  table?: string;
  /** Filter conditions for stock pool (AND logic) */
  filters?: FilterCondition[];
}

export interface FilterCondition {
  column: string;
  operator: string; // ">", "<", ">=", "<=", "=", "!=", "LIKE", "NOT LIKE"
  value: string;
}

export interface GetTablesRequest {
  database?: string;
}

export interface LoadDataResponse {
  dates: string[];
  symbols: string[];
  close: number[][];
  open: number[][];
  high: number[][];
  low: number[][];
  volume: number[][];
  returns: number[][];
}

// Column mapping types
export interface ColumnMapping {
  close: string;
  open: string;
  high: string;
  low: string;
  volume: string;
  symbol?: string;
  tradingDate?: string;
  pe?: string;
  roe?: string;
  marketCap?: string;
}

export interface SetColumnMappingRequest {
  close: string;
  open: string;
  high: string;
  low: string;
  volume: string;
  symbol?: string;
  tradingDate?: string;
  pe?: string;
  roe?: string;
  marketCap?: string;
}

// Table mapping types
export interface TableMapping {
  stock_1day: string;
  stock_5min?: string;
  stock_1min?: string;
}

export interface SetTableMappingRequest {
  stock_1day: string;
  stock_5min?: string;
  stock_1min?: string;
}

export interface TableValidationResult {
  stock_1day_exists: boolean;
  stock_5min_exists: boolean;
  stock_1min_exists: boolean;
  available_tables: string[];
  current_mapping: TableMapping;
}

export interface ColumnInfo {
  name: string;
  columnType: string;
}

// Data source configuration for on-demand data loading
// Note: table, startDate, endDate, filters are set at execution time (backtest/GP mining)
// Only columnMapping needs to be stored in the Data Source page
export interface DataSourceConfig {
  /** Table name: stock_1d, stock_1m, stock_5m */
  table: string;
  /** Start date for data range */
  startDate: string;
  /** End date for data range */
  endDate: string;
  /** Filter conditions for stock pool (AND logic) */
  filters?: FilterCondition[];
  /** Column mapping (uses saved mapping from Data Source page) */
  columnMapping?: ColumnMapping;
}
