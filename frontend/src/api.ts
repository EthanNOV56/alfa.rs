/**
 * API client for Alfa.rs backtest server
 */

import type {
  BacktestRequest,
  NavData,
  FactorListResponse,
  FactorComputeRequest,
  FactorComputeResponse,
  GpMineRequest,
  GpMineResponse,
  AlphaListResponse,
  SaveAlphaRequest,
  SaveAlphaResponse,
  DbConfig,
  SetDbConfigRequest,
  SymbolInfo,
  DateRange,
  LoadDataRequest,
  LoadDataResponse,
  GetTablesRequest,
  ColumnMapping,
  SetColumnMappingRequest,
  ColumnInfo,
  TableMapping,
  SetTableMappingRequest,
} from './types';

const API_BASE = '/api';

/**
 * Helper to extract error message from response
 */
async function handleApiError(response: Response): Promise<string> {
  // Try to parse as JSON first, fallback to text
  try {
    const error = await response.json();
    return error.detail || error.message || JSON.stringify(error);
  } catch {
    // Not JSON, try to get text
    return await response.text().catch(() => 'Unknown error');
  }
}

/**
 * Run a backtest and get NAV data for visualization
 */
export async function runBacktest(request: BacktestRequest): Promise<NavData> {
  console.log('[runBacktest] Starting request, factor shape:', request.factor?.length, 'x', request.factor?.[0]?.length);
  const requestBody = JSON.stringify(request);
  console.log('[runBacktest] Request body size:', requestBody.length, 'bytes');

  const response = await fetch(`${API_BASE}/backtest`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: requestBody,
  });
  console.log('[runBacktest] Response status:', response.status);

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('API Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Health check
 */
export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

// ============================================================================
// Factor Library API
// ============================================================================

/**
 * Get list of predefined factors
 */
export async function listFactors(): Promise<FactorListResponse> {
  const response = await fetch(`${API_BASE}/factors`);

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('listFactors Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Compute factor values for a predefined factor
 */
export async function computeFactor(
  request: FactorComputeRequest
): Promise<FactorComputeResponse> {
  const response = await fetch(`${API_BASE}/factors/compute`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('computeFactor Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

// ============================================================================
// GP Mining API
// ============================================================================

/**
 * Run GP factor mining
 */
export async function mineFactors(
  request: GpMineRequest
): Promise<GpMineResponse> {
  const response = await fetch(`${API_BASE}/gp/mine`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('mineFactors Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

// ============================================================================
// Alpha Library API
// ============================================================================

/**
 * Get list of alpha factors from ~/.alfars/ (readonly) and ~/.alfars/user/ (writable)
 */
export async function listAlphas(): Promise<AlphaListResponse> {
  const response = await fetch(`${API_BASE}/alphas`);

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('listAlphas Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Save a new alpha factor to ~/.alfars/user/
 */
export async function saveAlpha(request: SaveAlphaRequest): Promise<SaveAlphaResponse> {
  const response = await fetch(`${API_BASE}/alphas`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('saveAlpha Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

// ============================================================================
// Database Configuration API
// ============================================================================

/**
 * Get database configuration
 */
export async function getDbConfig(): Promise<DbConfig> {
  const response = await fetch(`${API_BASE}/config`);

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('getDbConfig Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Set database configuration
 */
export async function setDbConfig(request: SetDbConfigRequest): Promise<DbConfig> {
  const response = await fetch(`${API_BASE}/config`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('setDbConfig Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Get available symbols from database
 */
export async function getSymbols(params: { table: string }): Promise<SymbolInfo[]> {
  const response = await fetch(`${API_BASE}/data/symbols?table=${encodeURIComponent(params.table)}`);

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('getSymbols Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Get date range for available data
 */
export async function getDateRange(params: { table: string }): Promise<DateRange> {
  const response = await fetch(`${API_BASE}/data/range?table=${encodeURIComponent(params.table)}`);

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('getDateRange Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Load data from database
 */
export async function loadData(request: LoadDataRequest): Promise<LoadDataResponse> {
  const response = await fetch(`${API_BASE}/data/load`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('loadData Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Get available tables in the database
 */
export async function getTables(request: GetTablesRequest): Promise<string[]> {
  const response = await fetch(`${API_BASE}/data/tables`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('getTables Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

// ============================================================================
// Column Mapping API
// ============================================================================

/**
 * Get column mapping configuration for a specific table
 */
export async function getColumnMapping(table: string = 'stock_1d'): Promise<ColumnMapping> {
  const response = await fetch(`${API_BASE}/config/column-mapping?table=${encodeURIComponent(table)}`);

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('getColumnMapping Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Set column mapping configuration for a specific table
 */
export async function setColumnMapping(request: SetColumnMappingRequest): Promise<ColumnMapping> {
  const response = await fetch(`${API_BASE}/config/column-mapping`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('setColumnMapping Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

// ============================================================================
// Column Info API
// ============================================================================

/**
 * Get column information for a table
 */
export async function getColumns(params: { table: string }): Promise<ColumnInfo[]> {
  const response = await fetch(`${API_BASE}/data/columns?table=${encodeURIComponent(params.table)}`);

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('getColumns Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Get available filter options (columns that can be used for filtering)
 */
export async function getFilterOptions(params: { table: string }): Promise<ColumnInfo[]> {
  const response = await fetch(`${API_BASE}/data/filter-options?table=${encodeURIComponent(params.table)}`);

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('getFilterOptions Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

// ============================================================================
// Table Mapping API
// ============================================================================

/**
 * Get table mapping configuration
 */
export async function getTableMapping(): Promise<TableMapping> {
  const response = await fetch(`${API_BASE}/config/table-mapping`);

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('getTableMapping Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Set table mapping configuration
 */
export async function setTableMapping(request: SetTableMappingRequest): Promise<TableMapping> {
  const response = await fetch(`${API_BASE}/config/table-mapping`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('setTableMapping Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Table validation result
 */
export interface TableValidationResult {
  stock_1day_exists: boolean;
  stock_5min_exists: boolean;
  stock_1min_exists: boolean;
  available_tables: string[];
  current_mapping: TableMapping;
}

/**
 * Validate table existence and get available tables
 */
export async function validateTables(): Promise<TableValidationResult> {
  const response = await fetch(`${API_BASE}/data/validate-tables`);

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('validateTables Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Column validation result
 */
export interface ColumnValidationResult {
  valid: boolean;
  available_columns: string[];
  current_mapping: {
    close: string;
    open: string;
    high: string;
    low: string;
    volume: string;
    symbol: string;
    tradingDate: string;
  };
  missing_columns: string[];
}

/**
 * Validate column existence and get available columns for table mapping
 */
export async function validateColumns(table?: string): Promise<ColumnValidationResult> {
  const url = table
    ? `${API_BASE}/data/validate-columns?table=${encodeURIComponent(table)}`
    : `${API_BASE}/data/validate-columns`;
  const response = await fetch(url);

  if (!response.ok) {
    const errorMessage = await handleApiError(response);
    console.error('validateColumns Error:', response.status, errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
}
