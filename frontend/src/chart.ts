/**
 * Chart rendering module using Chart.js
 */

import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import type { NavData, Metrics } from './types';

// Register Chart.js components
Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Legend,
  Filler
);

const QUANTILE_COLORS = [
  '#ef4444', // Q1 (red - worst)
  '#f97316',
  '#eab308',
  '#84cc16',
  '#22c55e',
  '#14b8a6',
  '#0ea5e9',
  '#6366f1',
  '#8b5cf6',
  '#d946ef', // Q10 (purple - best)
];

const LONG_SHORT_COLOR = '#10b981';
const BENCHMARK_COLOR = '#6b7280';

// Track multiple chart instances
const chartInstances: Chart[] = [];

/**
 * Format percentage
 */
function formatPercent(value: number | undefined): string {
  if (value === undefined) return '--';
  return `${(value * 100).toFixed(2)}%`;
}

/**
 * Format number
 */
function formatNumber(value: number | undefined): string {
  if (value === undefined) return '--';
  return value.toFixed(4);
}

/**
 * Get long_short data (handles both snake_case and camelCase)
 */
function getLongShort(data: NavData): number[] {
  return data.long_short || data.longShort || [];
}

/**
 * Create or update the NAV chart
 */
export function renderChart(navData: NavData, canvas: HTMLCanvasElement): void {
  try {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('Failed to get canvas context');
      return;
    }

    const { dates, quantiles, benchmark } = navData;
    const long_short = getLongShort(navData);

    // Validate data
    if (!dates || dates.length === 0) {
      console.error('Invalid navData: missing dates');
      return;
    }

    // Prepare datasets
    const datasets: any[] = [];

    // Quantile curves
    quantiles.forEach((values, index) => {
    const color = QUANTILE_COLORS[index % QUANTILE_COLORS.length];
    datasets.push({
      label: `Q${index + 1}`,
      data: values,
      borderColor: color,
      backgroundColor: color + '20',
      borderWidth: 1.5,
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.1,
      fill: false,
    });
  });

  // Long-short curve
  if (long_short.length > 0) {
    datasets.push({
      label: 'Long-Short',
      data: long_short,
      borderColor: LONG_SHORT_COLOR,
      backgroundColor: LONG_SHORT_COLOR + '20',
      borderWidth: 2.5,
      pointRadius: 0,
      pointHoverRadius: 5,
      tension: 0.1,
      fill: false,
    });
  }

  // Benchmark curve
  if (benchmark.length > 0) {
    datasets.push({
      label: 'Benchmark',
      data: benchmark,
      borderColor: BENCHMARK_COLOR,
      backgroundColor: BENCHMARK_COLOR + '10',
      borderWidth: 2,
      borderDash: [5, 5],
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.1,
      fill: false,
    });
  }

  // Create chart and track it
  const newChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            usePointStyle: true,
            padding: 20,
          },
        },
        tooltip: {
          enabled: true,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          titleFont: { size: 12 },
          bodyFont: { size: 11 },
          padding: 12,
          callbacks: {
            title: (items) => {
              if (items.length > 0) {
                return `Date: ${items[0].label}`;
              }
              return '';
            },
            label: (context) => {
              const value = context.parsed.y;
              if (value === null) return '';
              const prevValue = context.dataset.data[context.dataIndex - 1];
              const change = typeof prevValue === 'number' && context.dataIndex > 0
                ? value - prevValue
                : value - 1;
              const changeStr = formatPercent(change);
              const valueStr = formatNumber(value);
              return `${context.dataset.label}: ${valueStr} (${changeStr})`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: {
            display: false,
          },
          ticks: {
            maxTicksLimit: 10,
            maxRotation: 45,
          },
        },
        y: {
          grid: {
            color: 'rgba(0, 0, 0, 0.05)',
          },
          ticks: {
            callback: (value) => formatNumber(value as number),
          },
        },
      },
    },
  });

  // Track this chart instance
  chartInstances.push(newChart);
  } catch (error) {
    console.error('Error rendering chart:', error);
  }
}

/**
 * Destroy all charts
 */
export function destroyChart(): void {
  chartInstances.forEach(chart => chart.destroy());
  chartInstances.length = 0;
}

/**
 * Get metric value (handles both snake_case and camelCase)
 */
function getMetricValue(metrics: Metrics, snakeKey: keyof Metrics, camelKey: keyof Metrics): number | undefined {
  return metrics[snakeKey] as number | undefined ?? metrics[camelKey] as number | undefined;
}

/**
 * Update metrics display
 */
export function updateMetrics(metrics: Metrics): void {
  const elements: Record<string, string> = {
    lsReturn: formatPercent(getMetricValue(metrics, 'long_short_cum_return', 'longShortCumReturn')),
    annReturn: formatPercent(getMetricValue(metrics, 'annualized_return', 'annualizedReturn')),
    sharpe: formatNumber(getMetricValue(metrics, 'sharpe_ratio', 'sharpeRatio')),
    maxDd: formatPercent(getMetricValue(metrics, 'max_drawdown', 'maxDrawdown')),
    icMean: formatNumber(getMetricValue(metrics, 'ic_mean', 'icMean')),
    icIr: formatNumber(getMetricValue(metrics, 'ic_ir', 'icIr')),
  };

  for (const [id, value] of Object.entries(elements)) {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = value;
    }
  }
}
