/**
 * Main entry point for Alfa.rs Factor Visualization frontend
 */

import {
  runBacktest,
  healthCheck,
  computeFactor,
  mineFactors,
  listAlphas,
  saveAlpha,
  getDbConfig,
  setDbConfig,
  getDateRange,
  getTables,
  getColumnMapping,
  setColumnMapping,
  getColumns,
  getTableMapping,
  setTableMapping,
  validateTables,
  validateColumns,
} from './api';
import { renderChart, destroyChart, updateMetrics } from './chart';
import type { NavData, GpMineRequest, GpFactor, Alpha, FactorComputeResponse, FilterCondition, ColumnInfo, DataSourceConfig, TableMapping, Metrics } from './types';

// Chart instances
// Note: Chart instance is stored in chart.ts module, managed by renderChart/destroyChart

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

  // Set up navigation
  setupNavigation();

  // Initialize data source page
  initDataSource();

  // Initialize backtest
  initBacktest();

  // Initialize GP mining
  initGpMining();

  // Initialize Alpha library
  initAlphaLibrary();
}

/**
 * Set up navigation tabs
 */
function setupNavigation(): void {
  const tabs = document.querySelectorAll('.nav-tab');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const pageId = tab.getAttribute('data-page');
      if (!pageId) return;

      navigateTo(pageId);
    });
  });
}

/**
 * Navigate to a specific page
 */
function navigateTo(pageId: string): void {
  const tabs = document.querySelectorAll('.nav-tab');
  const pages = document.querySelectorAll('.page');

  // Update tab active state
  tabs.forEach(t => t.classList.remove('active'));
  const targetTab = document.querySelector(`.nav-tab[data-page="${pageId}"]`);
  if (targetTab) {
    targetTab.classList.add('active');
  }

  // Show selected page
  pages.forEach(p => p.classList.add('hidden'));
  document.getElementById(`page-${pageId}`)?.classList.remove('hidden');
}

// Store loaded data for backtest

// Store filter conditions for backtest page
let backtestFilterConditions: FilterCondition[] = [];

// Store filter conditions for GP mining page
let miningFilterConditions: FilterCondition[] = [];

// Store available columns for backtest
let backtestAvailableColumns: ColumnInfo[] = [];

// Store available columns for GP mining
let miningAvailableColumns: ColumnInfo[] = [];

// Store pending alpha for backtest (from Alpha Library page)
interface PendingAlphaBacktest {
  expression: string;
  name: string;
}
let pendingAlphaBacktest: PendingAlphaBacktest | null = null;

// Store pending alpha for GP Mining (from Alpha Library page)
interface PendingAlphaMining {
  expression: string;
  name: string;
}
let pendingAlphaForMining: PendingAlphaMining | null = null;

// Store selected alpha for GP Mining (seed alpha)
let selectedAlphaForMining: Alpha | null = null;

/**
 * Initialize data source functionality
 */
function initDataSource(): void {
  const connectBtn = document.getElementById('connectDb') as HTMLButtonElement;

  if (!connectBtn) return;

  // Load current config on init
  loadDbConfig();

  connectBtn.addEventListener('click', async () => {
    const host = (document.getElementById('dbHost') as HTMLInputElement).value;
    const port = parseInt((document.getElementById('dbPort') as HTMLInputElement).value);
    const username = (document.getElementById('dbUsername') as HTMLInputElement).value;
    const password = (document.getElementById('dbPassword') as HTMLInputElement).value;

    const connectMessage = document.getElementById('dbConnectMessage');
    connectBtn.disabled = true;
    connectBtn.textContent = 'Connecting...';

    try {
      const config = await setDbConfig({
        host,
        port,
        database: 'default', // Use default database
        username,
        password,
      });

      updateDbStatus(config.connected);

      if (connectMessage) {
        connectMessage.textContent = config.connected ? 'Connected successfully!' : 'Connection failed';
        connectMessage.className = config.connected ? 'message success' : 'message error';
      }

      if (config.connected) {
        // Show column mapping section
        document.getElementById('columnMappingSection')?.classList.remove('hidden');
        // Show table mapping section
        document.getElementById('tableMappingSection')?.classList.remove('hidden');
        // Validate tables and show available tables
        checkTableValidation();
      }
    } catch (error) {
      console.error('Failed to connect:', error);
      updateDbStatus(false);
      if (connectMessage) {
        connectMessage.textContent = error instanceof Error ? error.message : 'Connection failed';
        connectMessage.className = 'message error';
      }
    } finally {
      connectBtn.disabled = false;
      connectBtn.textContent = 'Connect';
    }
  });

  // Validate and save column mapping button
  const saveMappingBtn = document.getElementById('validateColumnsBtn') as HTMLButtonElement;
  saveMappingBtn?.addEventListener('click', async () => {
    const mappingMessage = document.getElementById('mappingMessage');
    const tableSelect = document.getElementById('columnMappingTableSelect') as HTMLSelectElement;
    const selectedTable = tableSelect?.value || 'stock_1d';

    try {
      const mapping = {
        table: selectedTable,
        mapping: {
          close: (document.getElementById('mapClose') as HTMLInputElement).value || 'close',
          open: (document.getElementById('mapOpen') as HTMLInputElement).value || 'open',
          high: (document.getElementById('mapHigh') as HTMLInputElement).value || 'high',
          low: (document.getElementById('mapLow') as HTMLInputElement).value || 'low',
          volume: (document.getElementById('mapVolume') as HTMLInputElement).value || 'volume',
          symbol: (document.getElementById('mapSymbol') as HTMLInputElement).value || 'symbol',
          tradingDate: (document.getElementById('mapTradingDate') as HTMLInputElement).value || 'trading_date',
        },
      };

      // Validate columns first
      const validationResult = await validateColumns(selectedTable);

      // Update validation status display
      updateColumnValidationStatus(validationResult);

      if (!validationResult.valid && validationResult.missing_columns) {
        if (mappingMessage) {
          mappingMessage.textContent = `Warning: Some columns not found: ${validationResult.missing_columns.join(', ')}`;
          mappingMessage.className = 'message error';
        }
        // Continue to save anyway
      }

      await setColumnMapping(mapping);

      if (mappingMessage && validationResult.valid) {
        mappingMessage.textContent = 'Column mapping validated and saved successfully!';
        mappingMessage.className = 'message success';
      }
    } catch (error) {
      console.error('Failed to validate/save column mapping:', error);
      if (mappingMessage) {
        mappingMessage.textContent = error instanceof Error ? error.message : 'Failed to validate mapping';
        mappingMessage.className = 'message error';
      }
    }
  });

  // Set up column dropdown change handlers
  setupColumnDropdowns();

  // Set up table selector change handler
  const columnMappingTableSelect = document.getElementById('columnMappingTableSelect') as HTMLSelectElement;
  columnMappingTableSelect?.addEventListener('change', () => {
    loadColumnMapping(columnMappingTableSelect.value);
  });

  // Load saved column mapping on init
  loadColumnMapping();

  // Save table mapping button
  const saveTableMappingBtn = document.getElementById('saveTableMappingBtn') as HTMLButtonElement;
  saveTableMappingBtn?.addEventListener('click', async () => {
    const tableMappingMessage = document.getElementById('tableMappingMessage');

    try {
      // Use dropdown value if selected, otherwise use input value
      const getTableValue = (selectId: string, inputId: string): string => {
        const select = document.getElementById(selectId) as HTMLSelectElement;
        const input = document.getElementById(inputId) as HTMLInputElement;
        return (select.value || input.value).trim();
      };

      const mapping: TableMapping = {
        stock_1day: getTableValue('table1daySelect', 'table1day') || 'stock_1day',
        stock_5min: getTableValue('table5minSelect', 'table5min') || undefined,
        stock_1min: getTableValue('table1minSelect', 'table1min') || undefined,
      };

      await setTableMapping(mapping);

      if (tableMappingMessage) {
        tableMappingMessage.textContent = 'Table mapping saved successfully!';
        tableMappingMessage.className = 'message success';
      }

      // Re-validate after saving
      checkTableValidation();
    } catch (error) {
      console.error('Failed to save table mapping:', error);
      if (tableMappingMessage) {
        tableMappingMessage.textContent = error instanceof Error ? error.message : 'Failed to save mapping';
        tableMappingMessage.className = 'message error';
      }
    }
  });

  // Refresh tables button
  const refreshTablesBtn = document.getElementById('refreshTablesBtn') as HTMLButtonElement;
  refreshTablesBtn?.addEventListener('click', () => {
    checkTableValidation();
  });

  // Available tables dropdown - fill input when selected
  // Table mapping dropdown handlers - sync dropdown with input field
  const setupTableDropdown = (selectId: string, inputId: string) => {
    const select = document.getElementById(selectId) as HTMLSelectElement;
    const input = document.getElementById(inputId) as HTMLInputElement;
    select?.addEventListener('change', () => {
      if (select.value) {
        input.value = select.value;
      }
    });
  };

  setupTableDropdown('table1daySelect', 'table1day');
  setupTableDropdown('table5minSelect', 'table5min');
  setupTableDropdown('table1minSelect', 'table1min');
}

async function loadTableMapping(): Promise<void> {
  try {
    const mapping = await getTableMapping();
    (document.getElementById('table1day') as HTMLInputElement).value = mapping.stock_1day || 'stock_1day';
    (document.getElementById('table5min') as HTMLInputElement).value = mapping.stock_5min || '';
    (document.getElementById('table1min') as HTMLInputElement).value = mapping.stock_1min || '';
    // Also validate tables
    checkTableValidation();
  } catch (error) {
    console.error('Failed to load table mapping:', error);
  }
}

/**
 * Check table validation and populate available tables dropdown
 */
async function checkTableValidation(): Promise<void> {
  const statusEl = document.getElementById('tableValidationStatus');
  const select1day = document.getElementById('table1daySelect') as HTMLSelectElement;
  const select5min = document.getElementById('table5minSelect') as HTMLSelectElement;
  const select1min = document.getElementById('table1minSelect') as HTMLSelectElement;

  if (!statusEl || !select1day) return;

  try {
    const result = await validateTables();

    // Populate all dropdowns with available tables
    const tables = result.available_tables || [];
    const populateDropdown = (select: HTMLSelectElement, currentValue: string) => {
      select.innerHTML = '<option value="">-- Select table --</option>';
      for (const table of tables) {
        const option = document.createElement('option');
        option.value = table;
        option.textContent = table;
        if (table === currentValue) {
          option.selected = true;
        }
        select.appendChild(option);
      }
    };

    const currentMapping = result.current_mapping || {};
    populateDropdown(select1day, currentMapping.stock_1day || 'stock_1day');
    populateDropdown(select5min, currentMapping.stock_5min || '');
    populateDropdown(select1min, currentMapping.stock_1min || '');

    // Set input values from current mapping
    (document.getElementById('table1day') as HTMLInputElement).value = currentMapping.stock_1day || 'stock_1day';
    (document.getElementById('table5min') as HTMLInputElement).value = currentMapping.stock_5min || '';
    (document.getElementById('table1min') as HTMLInputElement).value = currentMapping.stock_1min || '';

    // Update status message
    const exists1day = result.stock_1day_exists;
    const exists5min = result.stock_5min_exists;
    const exists1min = result.stock_1min_exists;

    if (!exists1day) {
      statusEl.className = 'table-validation-status error';
      statusEl.textContent = `Table "${currentMapping.stock_1day}" not found. Please select the correct 1 Day table from the dropdown.`;
    } else if (!exists5min || !exists1min) {
      statusEl.className = 'table-validation-status warning';
      const missing = [];
      if (!exists5min) missing.push('5min');
      if (!exists1min) missing.push('1min');
      statusEl.textContent = `1 Day table found. Optional tables (${missing.join(', ')}) not found.`;
    } else {
      statusEl.className = 'table-validation-status success';
      statusEl.textContent = 'All configured tables found in the database.';
    }
  } catch (error) {
    console.error('Failed to validate tables:', error);
    statusEl.className = 'table-validation-status error';
    statusEl.textContent = 'Failed to validate tables: ' + (error instanceof Error ? error.message : 'Unknown error');
  }
}

async function loadDbConfig(): Promise<void> {
  try {
    const config = await getDbConfig();
    (document.getElementById('dbHost') as HTMLInputElement).value = config.host;
    (document.getElementById('dbPort') as HTMLInputElement).value = config.port.toString();
    (document.getElementById('dbUsername') as HTMLInputElement).value = config.username;
    (document.getElementById('dbPassword') as HTMLInputElement).value = config.password;
    updateDbStatus(config.connected);

    if (config.connected) {
      document.getElementById('columnMappingSection')?.classList.remove('hidden');
      document.getElementById('tableMappingSection')?.classList.remove('hidden');
      loadTableMapping();
    }
  } catch (error) {
    console.error('Failed to load DB config:', error);
  }
}

async function loadColumnMapping(table?: string): Promise<void> {
  const tableSelect = document.getElementById('columnMappingTableSelect') as HTMLSelectElement;
  const selectedTable = table || tableSelect?.value || 'stock_1d';

  try {
    const mapping = await getColumnMapping(selectedTable);
    (document.getElementById('mapClose') as HTMLInputElement).value = mapping.close || 'close';
    (document.getElementById('mapOpen') as HTMLInputElement).value = mapping.open || 'open';
    (document.getElementById('mapHigh') as HTMLInputElement).value = mapping.high || 'high';
    (document.getElementById('mapLow') as HTMLInputElement).value = mapping.low || 'low';
    (document.getElementById('mapVolume') as HTMLInputElement).value = mapping.volume || 'volume';
    (document.getElementById('mapSymbol') as HTMLInputElement).value = mapping.symbol || 'symbol';
    (document.getElementById('mapTradingDate') as HTMLInputElement).value = mapping.tradingDate || 'trading_date';

    // Populate dropdowns with available columns
    await populateColumnDropdowns(selectedTable);
  } catch (error) {
    console.error('Failed to load column mapping:', error);
  }
}

/**
 * Populate column dropdowns with available columns from database
 */
async function populateColumnDropdowns(table?: string): Promise<void> {
  try {
    const result = await validateColumns(table);
    const columns = result.available_columns;

    // Populate all dropdowns
    const dropdownIds = [
      'mapCloseSelect', 'mapOpenSelect', 'mapHighSelect', 'mapLowSelect',
      'mapVolumeSelect', 'mapSymbolSelect', 'mapTradingDateSelect'
    ];

    dropdownIds.forEach(dropdownId => {
      const select = document.getElementById(dropdownId) as HTMLSelectElement;
      if (select) {
        // Clear existing options except first
        while (select.options.length > 1) {
          select.remove(1);
        }

        // Add available columns
        columns.forEach(col => {
          const option = document.createElement('option');
          option.value = col;
          option.textContent = col;
          select.appendChild(option);
        });
      }
    });

    // Update validation status
    updateColumnValidationStatus(result);

    // Auto-select matching columns
    autoSelectMatchingColumns(columns);
  } catch (error) {
    console.error('Failed to populate column dropdowns:', error);
  }
}

/**
 * Auto-select columns that match the current mapping
 */
function autoSelectMatchingColumns(columns: string[]): void {
  const inputIds = ['mapClose', 'mapOpen', 'mapHigh', 'mapLow', 'mapVolume', 'mapSymbol', 'mapTradingDate'];

  inputIds.forEach(inputId => {
    const input = document.getElementById(inputId) as HTMLInputElement;
    if (input && input.value && columns.includes(input.value)) {
      // Find matching option in dropdown
      const dropdownId = inputId + 'Select';
      const dropdown = document.getElementById(dropdownId) as HTMLSelectElement;
      if (dropdown) {
        dropdown.value = input.value;
      }
    }
  });
}

/**
 * Update column validation status display
 */
function updateColumnValidationStatus(result: { valid?: boolean; missing_columns?: string[] }): void {
  const statusEl = document.getElementById('columnValidationStatus');
  if (!statusEl) return;

  if (result.valid) {
    statusEl.innerHTML = '<span class="status-valid">All columns found in database</span>';
  } else if (result.missing_columns && result.missing_columns.length > 0) {
    statusEl.innerHTML = `<span class="status-invalid">Missing columns: ${result.missing_columns.join(', ')}</span>`;
  } else {
    statusEl.innerHTML = '<span class="status-invalid">Validation failed</span>';
  }
}

/**
 * Set up column dropdown change handlers
 */
function setupColumnDropdowns(): void {
  const mappingFields = ['Close', 'Open', 'High', 'Low', 'Volume', 'Symbol', 'TradingDate'];

  mappingFields.forEach(field => {
    const selectId = `map${field}Select`;
    const inputId = `map${field}`;

    const select = document.getElementById(selectId);
    const input = document.getElementById(inputId) as HTMLInputElement;

    if (select && input) {
      select.addEventListener('change', () => {
        const selectEl = select as HTMLSelectElement;
        if (selectEl.value) {
          input.value = selectEl.value;
        }
      });
    }
  });
}

function updateDbStatus(connected: boolean): void {
  const statusEl = document.getElementById('dbStatus');
  if (statusEl) {
    const indicator = statusEl.querySelector('.status-indicator');
    const text = statusEl.querySelector('span:last-child');
    if (indicator) {
      indicator.className = connected ? 'status-indicator connected' : 'status-indicator disconnected';
    }
    if (text) {
      text.textContent = connected ? 'Connected' : 'Not connected';
    }
  }
}

/**
 * Initialize demo backtest functionality
 */
function initBacktest(): void {
  console.log('[initBacktest] Starting initialization...');
  const canvas = document.getElementById('navChart') as HTMLCanvasElement;
  const runRealBacktestBtn = document.getElementById('runRealBacktest') as HTMLButtonElement;
  const clearBtn = document.getElementById('clearChart') as HTMLButtonElement;

  console.log('[initBacktest] canvas:', !!canvas, 'runRealBacktestBtn:', !!runRealBacktestBtn);

  if (!canvas) {
    console.log('[initBacktest] canvas not found, skipping');
    return;
  }

  // Initialize backtest data source controls
  initBacktestDataControls();

  runRealBacktestBtn?.addEventListener('click', async () => {
    console.log('[initBacktest] Button clicked!');
    const tableSelect = document.getElementById('backtestTableSelect') as HTMLSelectElement;
    const startDate = (document.getElementById('backtestStartDate') as HTMLInputElement).value;
    const endDate = (document.getElementById('backtestEndDate') as HTMLInputElement).value;
    console.log('[initBacktest] table:', tableSelect.value, 'startDate:', startDate, 'endDate:', endDate);

    if (!tableSelect.value) {
      alert('Please select a table first');
      return;
    }

    if (!startDate || !endDate) {
      alert('Please select date range');
      return;
    }

    runRealBacktestBtn.disabled = true;

    // Filter out empty filters
    const activeFilters = backtestFilterConditions.filter(f => f.column && f.value);

    const dataSource: DataSourceConfig = {
      table: tableSelect.value,
      startDate,
      endDate,
      filters: activeFilters,
    };

    try {
      // Check if alphas are selected for batch backtest
      if (selectedAlphasForBacktest.length > 0) {
        // Batch mode: run backtest for each selected alpha in parallel
        runRealBacktestBtn.textContent = `Running (0/${selectedAlphasForBacktest.length})...`;

        // Create container for multiple charts
        const chartContainer = document.querySelector('.chart-container');
        if (chartContainer) {
          // Clear previous charts - also destroy Chart.js instances
          destroyChart();
          chartContainer.innerHTML = '';
        }

        // Track completion order
        let completedCount = 0;

        // Create parallel promise for each alpha
        const alphaPromises = selectedAlphasForBacktest.map(async (alpha, index) => {
          const startTime = Date.now();

          try {
            // Compute factor - create a fresh copy of dataSource for each request
            const computeReq: any = {
              factorId: alpha.expression,
              dataSource: { ...dataSource },
            };

            console.log(`Computing factor ${index + 1}/${selectedAlphasForBacktest.length}: ${alpha.name}`);
            let computeRes: FactorComputeResponse;
            try {
              computeRes = await computeFactor(computeReq);
            } catch (e) {
              console.error(`computeFactor failed for ${alpha.name}:`, e);
              throw e;
            }
            console.log(`Factor computed successfully for ${alpha.name}:`, computeRes);

            // Use cacheId if available, otherwise send full data
            const request: any = {
              quantiles: 5,
              weight_method: 'equal',
              long_top_n: 1,
              short_top_n: 1,
              commission_rate: 0.001,
            };

            // Note: backend returns cache_id (snake_case), not cacheId
            if (computeRes.cache_id) {
              // Use cache ID - much smaller request
              request.cacheId = computeRes.cache_id;
              console.log(`Running backtest for ${alpha.name} using cache: ${request.cacheId}`);
            } else {
              // Fallback to sending full data (for backward compatibility)
              if (!computeRes.factor || !computeRes.returns || !computeRes.dates) {
                throw new Error(`Invalid computeRes for ${alpha.name}: missing factor/returns/dates`);
              }
              request.factor = computeRes.factor;
              request.returns = computeRes.returns;
              request.dates = computeRes.dates;
              console.log(`Running backtest for ${alpha.name} with factor shape: ${request.factor?.length}x${request.factor?.[0]?.length}`);
            }
            let navData: NavData;
            try {
              navData = await runBacktest(request);
            } catch (e) {
              console.error(`runBacktest failed for ${alpha.name}:`, e);
              throw e;
            }
            console.log(`Backtest completed for ${alpha.name}:`, navData.metrics);

            const elapsed = Date.now() - startTime;
            console.log(`Alpha "${alpha.name}" completed in ${elapsed}ms`);

            return {
              name: alpha.name,
              metrics: navData.metrics,
              navData,
              completionOrder: completedCount++,
              elapsed,
            };
          } catch (innerError) {
            console.error(`Failed to process alpha "${alpha.name}" (${index + 1}/${selectedAlphasForBacktest.length}):`, innerError);
            const innerErrorMsg = innerError instanceof Error ? innerError.message : 'Unknown error';
            // Check for network errors
            if (innerErrorMsg.includes('NetworkError') || innerErrorMsg.includes('Failed to fetch')) {
              throw new Error(`Network error: Is the server running? (http://localhost:8000)`);
            }
            throw new Error(`Failed to process alpha "${alpha.name}": ${innerErrorMsg}`);
          }
        });

        // Execute sequentially instead of parallel for debugging
        console.log(`Starting batch backtest for ${alphaPromises.length} alphas (sequential mode)...`);
        const settledResults: Array<{ status: 'fulfilled' | 'rejected'; value?: any; reason?: any }> = [];
        for (const promise of alphaPromises) {
          try {
            const result = await promise;
            settledResults.push({ status: 'fulfilled', value: result });
          } catch (err) {
            settledResults.push({ status: 'rejected', reason: err });
          }
        }

        // Process results - separate successes and failures
        const allResults: Array<{
          name: string;
          metrics: Metrics;
          navData: NavData;
          completionOrder: number;
          elapsed: number;
        }> = [];
        const failures: Array<{ name: string; reason: string }> = [];

        settledResults.forEach((result, index) => {
          if (result.status === 'fulfilled') {
            console.log(`Alpha ${index + 1} completed successfully:`, result.value.name);
            allResults.push(result.value);
          } else {
            console.error(`Alpha ${index + 1} failed:`, result.reason);
            failures.push({
              name: selectedAlphasForBacktest[index]?.name || `Alpha ${index + 1}`,
              reason: result.reason instanceof Error ? result.reason.message : String(result.reason)
            });
          }
        });

        console.log(`Batch complete: ${allResults.length} succeeded, ${failures.length} failed`);

        if (failures.length > 0) {
          console.warn('Failed alphas:', failures);
        }

        // If no results at all, show error
        if (allResults.length === 0) {
          throw new Error(`All ${failures.length} alphas failed. Check console for details.`);
        }

        // Sort by completion order (fastest first)
        allResults.sort((a, b) => a.completionOrder - b.completionOrder);

        // Render all charts in order
        for (const result of allResults) {
          // Create chart section for each alpha
          const chartSection = document.createElement('div');
          chartSection.className = 'alpha-chart-section';
          chartSection.style.marginBottom = '24px';
          chartSection.style.padding = '16px';
          chartSection.style.background = '#f9fafb';
          chartSection.style.borderRadius = '8px';
          chartSection.style.height = '350px';

          // Add title with metrics
          const icValue = result.metrics.ic_mean ?? result.metrics.icMean ?? 0;
          const title = document.createElement('h3');
          title.textContent = `${result.name} (IC: ${icValue.toFixed(4)}, ${result.elapsed}ms)`;
          title.style.margin = '0 0 12px 0';
          title.style.color = '#374151';
          chartSection.appendChild(title);

          // Create canvas for this chart
          const canvas = document.createElement('canvas');
          chartSection.appendChild(canvas);

          if (chartContainer) {
            chartContainer.appendChild(chartSection);
          }

          // Render chart with error handling
          try {
            renderChart(result.navData, canvas);
            console.log(`Chart rendered successfully for ${result.name}`);
          } catch (chartError) {
            console.error(`Failed to render chart for ${result.name}:`, chartError);
          }
        }

        // Update metrics display with first result
        if (allResults.length > 0) {
          updateMetrics(allResults[0].metrics);
        }

        // Show summary in console
        console.log('Batch backtest results:', allResults.map(r => ({
          name: r.name,
          ic: r.metrics.ic_mean ?? r.metrics.icMean,
          elapsed: r.elapsed,
        })));

      } else {
        // Single mode: run backtest with database data directly
        runRealBacktestBtn.textContent = 'Running...';

        const request = {
          dataSource,
          quantiles: 5,
          weight_method: 'equal',
          long_top_n: 1,
          short_top_n: 1,
          commission_rate: 0.001,
        };

        const navData: NavData = await runBacktest(request);

        destroyChart();
        renderChart(navData, canvas);
        updateMetrics(navData.metrics);

        console.log('Backtest complete:', navData.metrics);
      }
    } catch (error) {
      console.error('Backtest failed:', error);
      // Provide more detailed error information
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';
      alert(`Backtest failed: ${errorMsg}`);
    } finally {
      runRealBacktestBtn.disabled = false;
      runRealBacktestBtn.textContent = 'Run Backtest';
    }
  });

  clearBtn?.addEventListener('click', () => {
    destroyChart();
    // Chart cleanup handled by chart.ts module
    resetMetrics();
  });

  // Check for pending alpha from Alpha Library
  checkAndLoadPendingAlpha();

  // Initialize alpha selection
  initAlphaSelection();
}

// Store selected alphas for batch backtest
let selectedAlphasForBacktest: Alpha[] = [];

/**
 * Initialize alpha selection functionality
 */
async function initAlphaSelection(): Promise<void> {
  const categoryFilter = document.getElementById('alphaCategoryFilter') as HTMLSelectElement;
  const clearBtn = document.getElementById('clearSelectedAlphasBtn') as HTMLButtonElement;

  // Load alphas initially
  await loadAlphasForSelection();

  // Category filter handler
  categoryFilter?.addEventListener('change', async () => {
    await loadAlphasForSelection();
  });

  // Clear selection handler
  clearBtn?.addEventListener('click', () => {
    selectedAlphasForBacktest = [];
    updateSelectedAlphasDisplay();
  });
}

/**
 * Load alphas for selection in backtest page
 */
async function loadAlphasForSelection(): Promise<void> {
  const container = document.getElementById('alphaListContainer');
  const categoryFilter = document.getElementById('alphaCategoryFilter') as HTMLSelectElement;

  if (!container) return;

  try {
    const response = await listAlphas();
    const category = categoryFilter?.value || '';

    // Filter by category if selected
    let filteredAlphas = response.alphas;
    if (category) {
      filteredAlphas = response.alphas.filter(a =>
        a.tags?.some(t => t.toLowerCase() === category.toLowerCase()) ||
        a.description.toLowerCase().includes(category.toLowerCase())
      );
    }

    if (filteredAlphas.length === 0) {
      container.innerHTML = '<div class="alpha-item">No alphas found</div>';
      return;
    }

    container.innerHTML = filteredAlphas.map(alpha => {
      const isSelected = selectedAlphasForBacktest.some(a => a.name === alpha.name);
      return `
        <div class="alpha-item ${isSelected ? 'selected' : ''}" data-name="${alpha.name}" data-expression="${encodeURIComponent(alpha.expression)}">
          <input type="checkbox" ${isSelected ? 'checked' : ''} />
          <span class="alpha-name">${alpha.name}</span>
          <span class="alpha-category">${alpha.tags?.[0] || 'N/A'}</span>
        </div>
      `;
    }).join('');

    // Add click handlers
    container.querySelectorAll('.alpha-item').forEach(item => {
      item.addEventListener('click', (e) => {
        if ((e.target as HTMLElement).tagName === 'INPUT') return;

        const checkbox = item.querySelector('input[type="checkbox"]') as HTMLInputElement;
        checkbox.checked = !checkbox.checked;

        const name = item.getAttribute('data-name') || '';
        const expression = decodeURIComponent(item.getAttribute('data-expression') || '');

        if (checkbox.checked) {
          selectedAlphasForBacktest.push({
            name,
            expression,
            description: '',
            dimension: '',
            tags: [],
            readonly: false
          });
          item.classList.add('selected');
        } else {
          selectedAlphasForBacktest = selectedAlphasForBacktest.filter(a => a.name !== name);
          item.classList.remove('selected');
        }

        updateSelectedAlphasDisplay();
      });

      // Checkbox change handler
      const checkbox = item.querySelector('input[type="checkbox"]') as HTMLInputElement;
      checkbox?.addEventListener('change', () => {
        const name = item.getAttribute('data-name') || '';
        const expression = decodeURIComponent(item.getAttribute('data-expression') || '');

        if (checkbox.checked) {
          selectedAlphasForBacktest.push({
            name,
            expression,
            description: '',
            dimension: '',
            tags: [],
            readonly: false
          });
          item.classList.add('selected');
        } else {
          selectedAlphasForBacktest = selectedAlphasForBacktest.filter(a => a.name !== name);
          item.classList.remove('selected');
        }

        updateSelectedAlphasDisplay();
      });
    });

  } catch (error) {
    console.error('Failed to load alphas:', error);
    container.innerHTML = '<div class="alpha-item">Failed to load alphas</div>';
  }
}

/**
 * Update selected alphas display
 */
function updateSelectedAlphasDisplay(): void {
  const infoEl = document.getElementById('selectedAlphasInfo');
  const countEl = infoEl?.querySelector('.count');

  if (countEl) {
    countEl.textContent = `Selected: ${selectedAlphasForBacktest.length} alpha(s)`;
  }

  if (infoEl) {
    if (selectedAlphasForBacktest.length > 0) {
      infoEl.classList.remove('hidden');
    } else {
      infoEl.classList.add('hidden');
    }
  }
}

/**
 * Check and load pending alpha from Alpha Library navigation
 */
function checkAndLoadPendingAlpha(): void {
  if (pendingAlphaBacktest) {
    // Add the pending alpha to selection
    selectedAlphasForBacktest = [{
      name: pendingAlphaBacktest.name,
      expression: pendingAlphaBacktest.expression,
      description: '',
      dimension: '',
      tags: [],
      readonly: false
    }];

    updateSelectedAlphasDisplay();

    // Reload alpha list to show selection
    loadAlphasForSelection();

    // Clear the pending alpha
    pendingAlphaBacktest = null;
  }
}

/**
 * Check and load pending alpha for GP Mining from Alpha Library navigation
 */
function checkAndLoadPendingMiningAlpha(): void {
  if (pendingAlphaForMining) {
    // Set the pending alpha as selected for mining
    selectedAlphaForMining = {
      name: pendingAlphaForMining.name,
      expression: pendingAlphaForMining.expression,
      description: '',
      dimension: '',
      tags: [],
      readonly: false
    };

    updateSelectedMiningAlphaDisplay();

    // Reload alpha list to show selection
    loadMiningAlphasForSelection();

    // Clear the pending alpha
    pendingAlphaForMining = null;
  }
}

/**
 * Update the display for selected mining alpha
 */
function updateSelectedMiningAlphaDisplay(): void {
  const infoEl = document.getElementById('selectedMiningAlphaInfo');
  const countEl = infoEl?.querySelector('.count');

  if (countEl) {
    countEl.textContent = selectedAlphaForMining
      ? `Selected: ${selectedAlphaForMining.name}`
      : 'No alpha selected';
  }

  if (infoEl) {
    if (selectedAlphaForMining) {
      infoEl.classList.remove('hidden');
    } else {
      infoEl.classList.add('hidden');
    }
  }
}

/**
 * Initialize alpha selection for GP Mining page
 */
async function initMiningAlphaSelection(): Promise<void> {
  const categoryFilter = document.getElementById('miningAlphaCategoryFilter') as HTMLSelectElement;
  const clearBtn = document.getElementById('clearSelectedMiningAlphaBtn') as HTMLButtonElement;

  // Load alphas initially
  await loadMiningAlphasForSelection();

  // Category filter handler
  categoryFilter?.addEventListener('change', async () => {
    await loadMiningAlphasForSelection();
  });

  // Clear selection handler
  clearBtn?.addEventListener('click', () => {
    selectedAlphaForMining = null;
    updateSelectedMiningAlphaDisplay();
    loadMiningAlphasForSelection();
  });

  // Check for pending alpha from Alpha Library
  checkAndLoadPendingMiningAlpha();
}

/**
 * Load alphas for selection in GP Mining page
 */
async function loadMiningAlphasForSelection(): Promise<void> {
  const container = document.getElementById('miningAlphaListContainer');
  const categoryFilter = document.getElementById('miningAlphaCategoryFilter') as HTMLSelectElement;

  if (!container) return;

  container.innerHTML = '<div class="loading">Loading alphas...</div>';

  try {
    const category = categoryFilter?.value || '';
    const response = await listAlphas();
    const alphas = response.alphas;

    // Filter by category if selected
    const filteredAlphas = category
      ? alphas.filter(a => a.tags?.some(t => t.toLowerCase() === category.toLowerCase()))
      : alphas;

    if (filteredAlphas.length === 0) {
      container.innerHTML = '<div class="empty">No alphas found</div>';
      return;
    }

    container.innerHTML = filteredAlphas
      .map(alpha => {
        const isSelected = selectedAlphaForMining?.expression === alpha.expression;
        return `
          <div class="alpha-item ${isSelected ? 'selected' : ''}" data-expression="${alpha.expression}">
            <div class="alpha-name">${alpha.name}</div>
            <div class="alpha-expr">${alpha.expression.substring(0, 50)}${alpha.expression.length > 50 ? '...' : ''}</div>
            ${alpha.tags?.length ? `<div class="alpha-tags">${alpha.tags.join(', ')}</div>` : ''}
          </div>
        `;
      })
      .join('');

    // Add click handlers
    container.querySelectorAll('.alpha-item').forEach(item => {
      item.addEventListener('click', () => {
        const expression = item.getAttribute('data-expression');
        const alpha = alphas.find(a => a.expression === expression);
        if (alpha) {
          selectedAlphaForMining = alpha;
          updateSelectedMiningAlphaDisplay();
          loadMiningAlphasForSelection();
        }
      });
    });
  } catch (error) {
    console.error('Failed to load alphas for mining:', error);
    container.innerHTML = '<div class="error">Failed to load alphas</div>';
  }
}

function initBacktestDataControls(): void {
  const refreshTablesBtn = document.getElementById('refreshBacktestTablesBtn') as HTMLButtonElement;
  const addFilterBtn = document.getElementById('addBacktestFilterBtn') as HTMLButtonElement;

  // Check if database is connected before loading tables
  checkDbConnectionAndLoadTables();

  refreshTablesBtn?.addEventListener('click', () => {
    loadBacktestTables();
  });

  // Table selection - load date range and columns
  const tableSelect = document.getElementById('backtestTableSelect') as HTMLSelectElement;
  tableSelect?.addEventListener('change', async () => {
    const table = tableSelect.value;
    if (table) {
      // Load date range
      try {
        const range = await getDateRange({ table });
        (document.getElementById('backtestStartDate') as HTMLInputElement).value = range.minDate;
        (document.getElementById('backtestEndDate') as HTMLInputElement).value = range.maxDate;
      } catch (error) {
        console.error('Failed to load date range:', error);
      }

      // Load columns for filter
      try {
        const columns = await getColumns({ table });
        backtestAvailableColumns = columns;
        renderBacktestFilters();
      } catch (error) {
        console.error('Failed to load columns:', error);
      }
    }
  });

  addFilterBtn?.addEventListener('click', () => {
    backtestFilterConditions.push({ column: '', operator: '=', value: '' });
    renderBacktestFilters();
  });
}

async function checkDbConnectionAndLoadTables(): Promise<void> {
  try {
    const config = await getDbConfig();
    if (config.connected) {
      await loadBacktestTables();
      await loadMiningTables();
    }
  } catch (error) {
    console.error('Failed to check DB connection:', error);
  }
}

async function loadBacktestTables(): Promise<void> {
  const tableSelect = document.getElementById('backtestTableSelect') as HTMLSelectElement;
  const refreshBtn = document.getElementById('refreshBacktestTablesBtn') as HTMLButtonElement;

  if (!tableSelect) return;

  // Check if database is connected first
  try {
    const config = await getDbConfig();
    if (!config.connected) {
      console.log('Database not connected, skipping table load');
      return;
    }
  } catch (error) {
    console.log('Cannot connect to server, skipping table load');
    return;
  }

  refreshBtn.disabled = true;
  refreshBtn.textContent = 'Loading...';

  try {
    const tables = await getTables({ database: 'default' });
    tableSelect.innerHTML = '<option value="">-- Select Table --</option>';
    tables.forEach(table => {
      const option = document.createElement('option');
      option.value = table;
      option.textContent = table;
      tableSelect.appendChild(option);
    });
  } catch (error) {
    console.error('Failed to load tables:', error);
    // Don't show alert, just log the error
  } finally {
    refreshBtn.disabled = false;
    refreshBtn.textContent = 'Refresh';
  }
}

function renderBacktestFilters(): void {
  const container = document.getElementById('backtestFilters');
  if (!container) return;

  container.innerHTML = '';

  backtestFilterConditions.forEach((filter, index) => {
    const filterDiv = document.createElement('div');
    filterDiv.className = 'filter-condition';
    filterDiv.innerHTML = `
      <select class="filter-column" data-index="${index}">
        <option value="">-- Select Column --</option>
        ${backtestAvailableColumns.map(col => `
          <option value="${col.name}" ${col.name === filter.column ? 'selected' : ''}>${col.name}</option>
        `).join('')}
      </select>
      <select class="filter-operator" data-index="${index}">
        <option value="=" ${filter.operator === '=' ? 'selected' : ''}>=</option>
        <option value="!=" ${filter.operator === '!=' ? 'selected' : ''}>!=</option>
        <option value=">" ${filter.operator === '>' ? 'selected' : ''}>&gt;</option>
        <option value=">=" ${filter.operator === '>=' ? 'selected' : ''}>&gt;=</option>
        <option value="<" ${filter.operator === '<' ? 'selected' : ''}>&lt;</option>
        <option value="<=" ${filter.operator === '<=' ? 'selected' : ''}>&lt;=</option>
      </select>
      <input type="text" class="filter-value" data-index="${index}" value="${filter.value}" placeholder="Value" />
      <button class="remove-filter" data-index="${index}">X</button>
    `;
    container.appendChild(filterDiv);
  });

  // Add back the add button
  const newAddBtn = document.createElement('button');
  newAddBtn.id = 'addBacktestFilterBtn';
  newAddBtn.textContent = '+ Add Filter';
  newAddBtn.addEventListener('click', () => {
    backtestFilterConditions.push({ column: '', operator: '=', value: '' });
    renderBacktestFilters();
  });
  container.appendChild(newAddBtn);

  // Add event listeners
  container.querySelectorAll('.filter-column').forEach(el => {
    el.addEventListener('change', (e) => {
      const index = parseInt((e.target as HTMLElement).getAttribute('data-index') || '0');
      backtestFilterConditions[index].column = (e.target as HTMLSelectElement).value;
    });
  });

  container.querySelectorAll('.filter-operator').forEach(el => {
    el.addEventListener('change', (e) => {
      const index = parseInt((e.target as HTMLElement).getAttribute('data-index') || '0');
      backtestFilterConditions[index].operator = (e.target as HTMLSelectElement).value;
    });
  });

  container.querySelectorAll('.filter-value').forEach(el => {
    el.addEventListener('input', (e) => {
      const index = parseInt((e.target as HTMLElement).getAttribute('data-index') || '0');
      backtestFilterConditions[index].value = (e.target as HTMLInputElement).value;
    });
  });

  container.querySelectorAll('.remove-filter').forEach(el => {
    el.addEventListener('click', (e) => {
      const index = parseInt((e.target as HTMLElement).getAttribute('data-index') || '0');
      backtestFilterConditions.splice(index, 1);
      renderBacktestFilters();
    });
  });
}

function initMiningDataControls(): void {
  const refreshTablesBtn = document.getElementById('refreshMiningTablesBtn') as HTMLButtonElement;
  const addFilterBtn = document.getElementById('addMiningFilterBtn') as HTMLButtonElement;

  // Load tables on init
  loadMiningTables();

  refreshTablesBtn?.addEventListener('click', () => {
    loadMiningTables();
  });

  // Table selection - load date range and columns
  const tableSelect = document.getElementById('miningTableSelect') as HTMLSelectElement;
  tableSelect?.addEventListener('change', async () => {
    const table = tableSelect.value;
    if (table) {
      // Load date range
      try {
        const range = await getDateRange({ table });
        (document.getElementById('miningStartDate') as HTMLInputElement).value = range.minDate;
        (document.getElementById('miningEndDate') as HTMLInputElement).value = range.maxDate;
      } catch (error) {
        console.error('Failed to load date range:', error);
      }

      // Load columns for filter
      try {
        const columns = await getColumns({ table });
        miningAvailableColumns = columns;
        renderMiningFilters();
      } catch (error) {
        console.error('Failed to load columns:', error);
      }
    }
  });

  addFilterBtn?.addEventListener('click', () => {
    miningFilterConditions.push({ column: '', operator: '=', value: '' });
    renderMiningFilters();
  });
}

async function loadMiningTables(): Promise<void> {
  const tableSelect = document.getElementById('miningTableSelect') as HTMLSelectElement;
  const refreshBtn = document.getElementById('refreshMiningTablesBtn') as HTMLButtonElement;

  if (!tableSelect) return;

  // Check if database is connected first
  try {
    const config = await getDbConfig();
    if (!config.connected) {
      console.log('Database not connected, skipping table load');
      return;
    }
  } catch (error) {
    console.log('Cannot connect to server, skipping table load');
    return;
  }

  refreshBtn.disabled = true;
  refreshBtn.textContent = 'Loading...';

  try {
    const tables = await getTables({ database: 'default' });
    tableSelect.innerHTML = '<option value="">-- Select Table --</option>';
    tables.forEach(table => {
      const option = document.createElement('option');
      option.value = table;
      option.textContent = table;
      tableSelect.appendChild(option);
    });
  } catch (error) {
    console.error('Failed to load tables:', error);
    // Don't show alert, just log the error
  } finally {
    refreshBtn.disabled = false;
    refreshBtn.textContent = 'Refresh';
  }
}

function renderMiningFilters(): void {
  const container = document.getElementById('miningFilters');
  if (!container) return;

  container.innerHTML = '';

  miningFilterConditions.forEach((filter, index) => {
    const filterDiv = document.createElement('div');
    filterDiv.className = 'filter-condition';
    filterDiv.innerHTML = `
      <select class="filter-column" data-index="${index}">
        <option value="">-- Select Column --</option>
        ${miningAvailableColumns.map(col => `
          <option value="${col.name}" ${col.name === filter.column ? 'selected' : ''}>${col.name}</option>
        `).join('')}
      </select>
      <select class="filter-operator" data-index="${index}">
        <option value="=" ${filter.operator === '=' ? 'selected' : ''}>=</option>
        <option value="!=" ${filter.operator === '!=' ? 'selected' : ''}>!=</option>
        <option value=">" ${filter.operator === '>' ? 'selected' : ''}>&gt;</option>
        <option value=">=" ${filter.operator === '>=' ? 'selected' : ''}>&gt;=</option>
        <option value="<" ${filter.operator === '<' ? 'selected' : ''}>&lt;</option>
        <option value="<=" ${filter.operator === '<=' ? 'selected' : ''}>&lt;=</option>
      </select>
      <input type="text" class="filter-value" data-index="${index}" value="${filter.value}" placeholder="Value" />
      <button class="remove-filter" data-index="${index}">X</button>
    `;
    container.appendChild(filterDiv);
  });

  // Add back the add button
  const newAddBtn = document.createElement('button');
  newAddBtn.id = 'addMiningFilterBtn';
  newAddBtn.textContent = '+ Add Filter';
  newAddBtn.addEventListener('click', () => {
    miningFilterConditions.push({ column: '', operator: '=', value: '' });
    renderMiningFilters();
  });
  container.appendChild(newAddBtn);

  // Add event listeners
  container.querySelectorAll('.filter-column').forEach(el => {
    el.addEventListener('change', (e) => {
      const index = parseInt((e.target as HTMLElement).getAttribute('data-index') || '0');
      miningFilterConditions[index].column = (e.target as HTMLSelectElement).value;
    });
  });

  container.querySelectorAll('.filter-operator').forEach(el => {
    el.addEventListener('change', (e) => {
      const index = parseInt((e.target as HTMLElement).getAttribute('data-index') || '0');
      miningFilterConditions[index].operator = (e.target as HTMLSelectElement).value;
    });
  });

  container.querySelectorAll('.filter-value').forEach(el => {
    el.addEventListener('input', (e) => {
      const index = parseInt((e.target as HTMLElement).getAttribute('data-index') || '0');
      miningFilterConditions[index].value = (e.target as HTMLInputElement).value;
    });
  });

  container.querySelectorAll('.remove-filter').forEach(el => {
    el.addEventListener('click', (e) => {
      const index = parseInt((e.target as HTMLElement).getAttribute('data-index') || '0');
      miningFilterConditions.splice(index, 1);
      renderMiningFilters();
    });
  });
}


/**
 * Initialize GP mining functionality
 */
function initGpMining(): void {
  console.log('[initGpMining] Starting initialization...');
  const runMiningBtn = document.getElementById('runMining') as HTMLButtonElement;
  const miningStatus = document.getElementById('miningStatus');
  const miningResults = document.getElementById('miningResults');

  console.log('[initGpMining] runMiningBtn:', !!runMiningBtn, 'miningStatus:', !!miningStatus, 'miningResults:', !!miningResults);

  if (!runMiningBtn || !miningStatus || !miningResults) {
    console.log('[initGpMining] Missing elements, skipping');
    return;
  }

  // Initialize GP mining data source controls
  initMiningDataControls();

  // Initialize alpha selection for GP Mining
  initMiningAlphaSelection();

  runMiningBtn.addEventListener('click', async () => {
    // Get configuration
    const popSize = parseInt((document.getElementById('popSize') as HTMLInputElement).value);
    const maxGens = parseInt((document.getElementById('maxGens') as HTMLInputElement).value);
    const targetIc = parseFloat((document.getElementById('targetIc') as HTMLInputElement).value);

    // Get terminal set - select all checked checkboxes in the terminal set section
    const terminalSet: string[] = [];
    // Find the terminal set checkboxes by looking at the parent checkbox-group
    document.querySelectorAll('#page-mining .checkbox-group').forEach(group => {
      const label = group.previousElementSibling;
      if (label && label.textContent?.includes('Terminal Set')) {
        group.querySelectorAll('input[type="checkbox"]').forEach(cb => {
          if ((cb as HTMLInputElement).checked) {
            terminalSet.push((cb as HTMLInputElement).value);
          }
        });
      }
    });
    // Fallback: just get all checked checkboxes in the page-mining section
    if (terminalSet.length === 0) {
      document.querySelectorAll('#page-mining input[type="checkbox"]').forEach(cb => {
        if ((cb as HTMLInputElement).checked) {
          terminalSet.push((cb as HTMLInputElement).value);
        }
      });
    }

    // Get function set - select all checked checkboxes in the function set section
    const functionSet: string[] = [];
    // We'll get all checked checkboxes and filter for known function names
    const knownFunctions = ['add', 'sub', 'mul', 'div', 'sqrt', 'abs', 'neg', 'rank', 'ts_mean', 'ts_std', 'ts_max', 'ts_min', 'delay', 'log', 'sign', 'ts_rank', 'decay_linear', 'correlation'];
    document.querySelectorAll('#page-mining input[type="checkbox"]').forEach(cb => {
      const checkbox = cb as HTMLInputElement;
      if (checkbox.checked && knownFunctions.includes(checkbox.value)) {
        functionSet.push(checkbox.value);
      }
    });

    const request: GpMineRequest = {
      population_size: popSize,
      max_generations: maxGens,
      terminal_set: terminalSet,
      function_set: functionSet,
      target_ic: targetIc,
      n_days: 50,
      n_assets: 30,
    };

    // Add seed expression if an alpha is selected for mutation-based mining
    if (selectedAlphaForMining) {
      request.seed_expression = selectedAlphaForMining.expression;
    }

    // Get data source config from controls
    const miningTableSelect = document.getElementById('miningTableSelect') as HTMLSelectElement;
    const miningStartDate = (document.getElementById('miningStartDate') as HTMLInputElement).value;
    const miningEndDate = (document.getElementById('miningEndDate') as HTMLInputElement).value;

    if (!miningTableSelect.value) {
      alert('Please select a table first');
      return;
    }

    if (!miningStartDate || !miningEndDate) {
      alert('Please select date range');
      return;
    }

    request.dataSource = {
      table: miningTableSelect.value,
      startDate: miningStartDate,
      endDate: miningEndDate,
      filters: miningFilterConditions.filter(f => f.column && f.value),
    };

    try {
      runMiningBtn.disabled = true;
      runMiningBtn.textContent = 'Mining...';
      miningStatus.textContent = 'Running GP mining...';
      miningResults.innerHTML = '';

      const response = await mineFactors(request);

      // Show results
      const bestFactor = response.best_factor || response.bestFactor;
      miningStatus.textContent = `Completed ${response.generations} generations in ${response.elapsed_time.toFixed(2)}s`;

      miningResults.innerHTML = `
        <div class="best-factor">
          <h4>Best Factor</h4>
          <div class="factor-name">${bestFactor?.name || 'N/A'}</div>
          <code>${bestFactor?.expression || 'N/A'}</code>
          <div class="factor-stats">
            <span>IC: ${bestFactor?.ic_mean?.toFixed(4) || '--'}</span>
            <span>IR: ${bestFactor?.ic_ir?.toFixed(4) || '--'}</span>
            <span>Fitness: ${bestFactor?.fitness?.toFixed(4) || '--'}</span>
          </div>
          ${bestFactor ? `<button class="save-gp-factor" data-gp-id="${bestFactor.id}">Save to Library</button>` : ''}
        </div>
        <div class="all-factors">
          <h4>All Candidates (${response.factors.length})</h4>
          ${response.factors
            .map(
              f => `
            <div class="gp-factor-item">
              <div class="gp-factor-expr">${f.expression}</div>
              <div class="gp-factor-stats">
                <span>IC: ${f.ic_mean?.toFixed(4) || '--'}</span>
                <span>IR: ${f.ic_ir?.toFixed(4) || '--'}</span>
              </div>
              <button class="save-gp-factor" data-gp-id="${f.id}">Save</button>
            </div>
          `
            )
            .join('')}
        </div>
      `;

      // Add save button handlers
      miningResults.querySelectorAll('.save-gp-factor').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const gpId = btn.getAttribute('data-gp-id');
          const factor = response.factors.find(f => f.id === gpId) || bestFactor;
          if (factor && (window as any).showSaveAlphaForm) {
            (window as any).showSaveAlphaForm(factor);
          }
        });
      });

      console.log('GP mining complete:', response);
    } catch (error) {
      console.error('GP mining failed:', error);
      miningStatus.textContent = `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
    } finally {
      runMiningBtn.disabled = false;
      runMiningBtn.textContent = 'Start Mining';
    }
  });
}

/**
 * Reset metrics display
 */
function resetMetrics(): void {
  const metricIds = ['lsReturn', 'annReturn', 'sharpe', 'maxDd', 'icMean', 'icIr'];
  metricIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = '--';
  });
}

// Current selected alpha
let selectedAlpha: Alpha | null = null;

// GP result to save
let pendingGpResult: GpFactor | null = null;

/**
 * Initialize Alpha Library functionality
 */
async function initAlphaLibrary(): Promise<void> {
  const alphaList = document.getElementById('alphaList');
  const alphaDetail = document.getElementById('alphaDetail');
  const saveAlphaForm = document.getElementById('saveAlphaForm');
  const saveAlphaBtn = document.getElementById('saveAlphaBtn') as HTMLButtonElement | null;
  const saveAlphaMessage = document.getElementById('saveAlphaMessage');

  if (!alphaList) return;

  // Create a non-null reference for use in nested functions
  const alphaListEl = alphaList;

  // Load alphas on init
  loadAlphas();

  async function loadAlphas(): Promise<void> {
    try {
      const response = await listAlphas();
      renderAlphaList(response.alphas);
    } catch (error) {
      console.error('Failed to load alphas:', error);
      alphaListEl.innerHTML = `<div class="error">Failed to load alphas: ${error instanceof Error ? error.message : 'Unknown error'}</div>`;
    }
  }

  function renderAlphaList(alphas: Alpha[]): void {
    if (alphas.length === 0) {
      alphaListEl.innerHTML = '<div class="empty">No .al files found in ~/.alfars/ or ~/.alfars/user/</div>';
      return;
    }

    alphaListEl.innerHTML = alphas
      .map(
        a => `
        <div class="alpha-item ${a.readonly ? 'readonly' : 'writable'}" data-alpha-name="${a.name}">
          <div class="alpha-name">
            ${a.name}
            ${a.readonly ? '<span class="readonly-badge">Read-only</span>' : ''}
          </div>
          <div class="alpha-expr">${a.expression}</div>
        </div>
      `
      )
      .join('');

    // Add click handlers
    alphaListEl.querySelectorAll('.alpha-item').forEach(item => {
      item.addEventListener('click', () => {
        const alphaName = item.getAttribute('data-alpha-name');
        const alpha = alphas.find(a => a.name === alphaName);
        if (alpha) {
          showAlphaDetail(alpha);
        }
      });
    });
  }

  function showAlphaDetail(alpha: Alpha): void {
    selectedAlpha = alpha;
    pendingGpResult = null;

    const title = document.getElementById('alphaDetailTitle');
    const exprEl = document.getElementById('alphaExpression');
    const descEl = document.getElementById('alphaDescription');
    const dimEl = document.getElementById('alphaDimension');
    const tagsEl = document.getElementById('alphaTags');
    const readonlyEl = document.getElementById('alphaReadonly');

    if (title) title.textContent = alpha.name;
    if (exprEl) exprEl.textContent = alpha.expression;
    if (descEl) descEl.textContent = alpha.description || 'No description';
    if (dimEl) dimEl.textContent = alpha.dimension;
    if (tagsEl) tagsEl.textContent = alpha.tags.join(', ') || 'No tags';
    if (readonlyEl) {
      readonlyEl.textContent = alpha.readonly ? 'Read-only' : 'Editable';
      readonlyEl.className = alpha.readonly ? 'readonly-badge readonly' : 'readonly-badge writable';
    }

    alphaDetail?.classList.remove('hidden');
    saveAlphaForm?.classList.add('hidden');

    // Clear previous chart and metrics
    const alphaChart = document.getElementById('alphaChart') as HTMLCanvasElement;
    if (alphaChart) {
      const ctx = alphaChart.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, alphaChart.width, alphaChart.height);
    }
    const alphaIcMeanEl = document.getElementById('alphaIcMean');
    const alphaIcIrEl = document.getElementById('alphaIcIr');
    if (alphaIcMeanEl) alphaIcMeanEl.textContent = '--';
    if (alphaIcIrEl) alphaIcIrEl.textContent = '--';
  }

  // Run Backtest button handler
  const runAlphaBacktestBtn = document.getElementById('runAlphaBacktest') as HTMLButtonElement;
  runAlphaBacktestBtn?.addEventListener('click', async () => {
    if (!selectedAlpha) return;

    // Store the alpha and navigate to Backtest page
    pendingAlphaBacktest = {
      expression: selectedAlpha.expression,
      name: selectedAlpha.name,
    };

    // Navigate to Backtest page
    navigateTo('backtest');

    // Clear the pending alpha after navigation
    pendingAlphaBacktest = null;
  });

  // GP Mining button handler - navigate to GP Mining page
  const gpMineAlphaBtn = document.getElementById('gpMineAlpha') as HTMLButtonElement;
  gpMineAlphaBtn?.addEventListener('click', async () => {
    if (!selectedAlpha) return;

    // Store the alpha and navigate to GP Mining page
    pendingAlphaForMining = {
      expression: selectedAlpha.expression,
      name: selectedAlpha.name,
    };

    // Navigate to GP Mining page
    navigateTo('mining');

    // Clear the pending alpha after navigation
    pendingAlphaForMining = null;
  });
  saveAlphaBtn?.addEventListener('click', async () => {
    const nameInput = document.getElementById('saveAlphaName') as HTMLInputElement;
    const descInput = document.getElementById('saveAlphaDescription') as HTMLInputElement;
    const dimSelect = document.getElementById('saveAlphaDimension') as HTMLSelectElement;
    const tagsInput = document.getElementById('saveAlphaTags') as HTMLInputElement;

    const name = nameInput?.value.trim();
    const expression = pendingGpResult?.expression;
    const description = descInput?.value.trim();
    const dimension = dimSelect?.value;
    const tags = tagsInput?.value
      .split(',')
      .map(t => t.trim())
      .filter(t => t.length > 0);

    if (!name) {
      if (saveAlphaMessage) {
        saveAlphaMessage.textContent = 'Name is required';
        saveAlphaMessage.className = 'save-message error';
      }
      return;
    }

    if (!expression) {
      if (saveAlphaMessage) {
        saveAlphaMessage.textContent = 'No expression to save';
        saveAlphaMessage.className = 'save-message error';
      }
      return;
    }

    try {
      saveAlphaBtn.disabled = true;
      saveAlphaBtn.textContent = 'Saving...';

      const response = await saveAlpha({
        name,
        expression,
        description,
        dimension,
        tags,
      });

      if (saveAlphaMessage) {
        saveAlphaMessage.textContent = response.message;
        saveAlphaMessage.className = 'save-message success';
      }

      // Reload alphas
      await loadAlphas();

      // Clear form
      if (nameInput) nameInput.value = '';
      if (descInput) descInput.value = '';
      if (tagsInput) tagsInput.value = '';
      pendingGpResult = null;
      saveAlphaForm?.classList.add('hidden');
    } catch (error) {
      console.error('Save alpha failed:', error);
      if (saveAlphaMessage) {
        saveAlphaMessage.textContent = `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
        saveAlphaMessage.className = 'save-message error';
      }
    } finally {
      saveAlphaBtn.disabled = false;
      saveAlphaBtn.textContent = 'Save to ~/.alfars/user/';
    }
  });

  // Expose function to show save form from GP mining
  (window as any).showSaveAlphaForm = (gpFactor: GpFactor) => {
    pendingGpResult = gpFactor;

    const exprEl = document.getElementById('saveAlphaExpression');
    const nameInput = document.getElementById('saveAlphaName') as HTMLInputElement;

    if (exprEl) exprEl.textContent = gpFactor.expression;
    if (nameInput) nameInput.value = gpFactor.name.toLowerCase().replace(/\s+/g, '_');

    alphaDetail?.classList.add('hidden');
    saveAlphaForm?.classList.remove('hidden');

    // Switch to alphas tab
    const tabs = document.querySelectorAll('.nav-tab');
    const pages = document.querySelectorAll('.page');
    tabs.forEach(t => t.classList.remove('active'));
    pages.forEach(p => p.classList.add('hidden'));
    document.querySelector('.nav-tab[data-page="alphas"]')?.classList.add('active');
    document.getElementById('page-alphas')?.classList.remove('hidden');

    if (saveAlphaMessage) {
      saveAlphaMessage.textContent = '';
      saveAlphaMessage.className = 'save-message';
    }
  };
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
