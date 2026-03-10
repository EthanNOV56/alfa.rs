/**
 * Main entry point for Alfa.rs visualization frontend
 */

import { runBacktest, generateDemoData, healthCheck } from './api';
import { renderChart, destroyChart, updateMetrics } from './chart';
import type { NavData } from './types';

/**
 * Initialize the application
 */
async function init(): Promise<void> {
  console.log('Alfa.rs Factor Visualization');

  // Check server health
  const isHealthy = await healthCheck();
  if (!isHealthy) {
    console.warn('Backend server not available. Please start the server first.');
  }

  // Get DOM elements
  const canvas = document.getElementById('navChart') as HTMLCanvasElement;
  const runDemoBtn = document.getElementById('runDemo') as HTMLButtonElement;
  const clearBtn = document.getElementById('clearChart') as HTMLButtonElement;

  if (!canvas) {
    console.error('Canvas element not found');
    return;
  }

  // Set up event listeners
  runDemoBtn?.addEventListener('click', async () => {
    try {
      runDemoBtn.disabled = true;
      runDemoBtn.textContent = 'Running...';

      // Generate demo data
      const request = generateDemoData();
      console.log('Running backtest with', request.factor.length, 'days,', request.factor[0].length, 'assets');

      // Run backtest
      const navData: NavData = await runBacktest(request);

      // Render chart
      renderChart(navData, canvas);

      // Update metrics
      updateMetrics(navData.metrics);

      console.log('Backtest complete. Metrics:', navData.metrics);
    } catch (error) {
      console.error('Backtest failed:', error);
      alert(`Backtest failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      runDemoBtn.disabled = false;
      runDemoBtn.textContent = 'Run Demo Backtest';
    }
  });

  clearBtn?.addEventListener('click', () => {
    destroyChart();
    // Reset metrics
    const metricIds = ['lsReturn', 'annReturn', 'sharpe', 'maxDd', 'icMean', 'icIr'];
    metricIds.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = '--';
    });
  });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
