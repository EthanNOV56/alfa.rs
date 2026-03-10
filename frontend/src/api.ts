/**
 * API client for Alfa.rs backtest server
 */

import type { BacktestRequest, NavData } from './types';

const API_BASE = '/api';

/**
 * Run a backtest and get NAV data for visualization
 */
export async function runBacktest(request: BacktestRequest): Promise<NavData> {
  const response = await fetch(`${API_BASE}/backtest`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
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

/**
 * Generate demo data for testing
 */
export function generateDemoData(): BacktestRequest {
  const nDays = 100;
  const nAssets = 50;

  // Generate random factor data with some signal
  const factor: number[][] = [];
  const returns: number[][] = [];
  const dates: string[] = [];

  // Create dates
  const startDate = new Date('2024-01-01');
  for (let i = 0; i < nDays; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    dates.push(date.toISOString().split('T')[0]);
  }

  // Generate factor and returns with some correlation
  for (let d = 0; d < nDays; d++) {
    const dayFactor: number[] = [];
    const dayReturns: number[] = [];

    for (let a = 0; a < nAssets; a++) {
      // Random factor value
      const f = Math.random();
      dayFactor.push(f);

      // Returns with some correlation to factor
      // Higher factor -> higher returns (positive alpha)
      const signal = (f - 0.5) * 0.02; // Signal component
      const noise = (Math.random() - 0.5) * 0.02; // Noise component
      dayReturns.push(signal + noise);
    }

    factor.push(dayFactor);
    returns.push(dayReturns);
  }

  return {
    factor,
    returns,
    dates,
    quantiles: 5,
    weight_method: 'equal',
    long_top_n: 1,
    short_top_n: 1,
    commission_rate: 0.001,
  };
}
