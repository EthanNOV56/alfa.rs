# alfars: From epsilon to alpha.

<p align="center">
  <img src="assets/logo.png" alt="alfars" width="200"/>
</p>

**Informed by data;**<br>
**Enlightened by research;**<br>
**Engineered with AI;**<br>
**Executed with algorithms.**


High-performance quant workflow with Rust core and Python bindings.

[![Rust](https://github.com/EthanNOV56/alfa.rs/actions/workflows/rust.yml/badge.svg)](https://github.com/EthanNOV56/alfa.rs/actions)
[![Python](https://img.shields.io/pypi/v/alfars.svg)](https://pypi.org/project/alfars/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

### Core Backtesting
- **High Performance**: Rust core with zero-clone streaming pipeline and per-factor parallelism
- **Flexible API**: NumPy arrays and Pandas Series support
- **Complete Metrics**: qcut(N) grouping, long-short portfolios, IC/group IC, weight turnover, win rate, Calmar ratio
- **Rebalance Control**: Configurable rebalance frequency and passive benchmark
- **Alphalens Compatibility**: Similar API design for easy migration
- **Extensible**: Modular design with custom weights, grouping, and commission models

### Intelligent Factor Mining
- **Expression System**: AST-based expression builder for custom factor computation
- **Lazy Evaluation**: Polars-style delayed computation with query optimization
- **Genetic Programming**: Auto-discover high-performance factor expressions with frequency support (1d/5m/1m), diverse init, and smart mutation
- **Dimension System**: Type-safe factor expressions to prevent invalid calculations
- **Redundancy Detection**: AST dedup and diversity selection to avoid duplicate factors
- **Persistence & Meta-Learning**: Factor library management with FactorPool lifecycle, search, caching, and versioning

### Streaming Pipeline (v0.5.0)
- **DataPool**: High-throughput data access with date-range filtering and mixed-frequency batch support
- **Fault Tolerance**: Result-based error handling for production reliability
- **Zero-Clone Backtest**: Broadcast-safe scalar handling with bounded cross-year parallelism
- **Benchmark Parity**: Validated DAG vs sequential evaluation parity on 82-factor alpha101

## Installation

### Requirements

- **Rust**: 1.70+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- **Python**: 3.8+
- **uv** (recommended): `pip install uv`

### Install from Source

```bash
# Clone repository
git clone https://github.com/EthanNOV56/alfa.rs.git
cd alfa.rs

# Option 1: Full installation with Python bindings
uv pip install -e .
maturin develop --release

# Option 2: Rust-only server (no Python extension needed)
cargo build --release --bin alfars-server
```

### Using pip (future releases)

```bash
pip install alfars
```

## Quick Start

### Basic Usage

```python
import numpy as np
import alfars as al

# Generate sample data
n_days, n_assets = 100, 200
factor = np.random.randn(n_days, n_assets)
returns = np.random.randn(n_days, n_assets) * 0.01 + factor * 0.005

# Run quantile backtest
result = al.quantile_backtest(
    factor=factor,
    returns=returns,
    quantiles=10,
    weight_method="equal",
    long_top_n=1,
    short_top_n=1,
    commission_rate=0.0003,
)

print(f"Long-Short Return: {result.long_short_cum_return:.4%}")
print(f"IC Mean: {result.ic_mean:.4f}")
print(f"IC IR: {result.ic_ir:.4f}")
```

### Start Interactive Lab

```bash
# Option 1: Python FastAPI server (requires maturin develop first)
uv run python -m alfars.lab

# Option 2: Rust HTTP server (recommended - no Python dependency)
cargo run --release --bin alfars-server   # Start Rust backend (port 8000)
cd frontend && npm run dev                 # Start frontend (port 5173)
```

Then open http://localhost:5173 in your browser.

### Genetic Programming Factor Mining

```python
from alfars import GpEngine

# Create GP engine
gp = GpEngine(
    population_size=100,
    max_generations=50,
    max_depth=6,
)

# Set available columns
gp.set_columns(['open', 'high', 'low', 'close', 'volume'])

# Prepare data
data = {
    'close': close_prices,    # shape: (n_days, n_assets)
    'volume': volumes,
}
returns = next_day_returns

# Mine factors
factors = gp.mine_factors(data, returns, num_factors=10)

for expr_str, fitness in factors[:3]:
    print(f"Factor: {expr_str[:60]}... (fitness: {fitness:.4f})")
```

### Expression System

```python
from alfars import Expr, LazyFrame

# Build factor expressions
expr = (Expr.col("close") - Expr.col("open")) / Expr.col("open")
sqrt_expr = expr.abs().sqrt()

# Lazy evaluation
lf = LazyFrame.scan(data)
lf_with_factor = lf.with_columns([("my_factor", expr)])
result = lf_with_factor.collect()
```

### Persistence & Meta-Learning

```python
from alfars import PersistenceManager, MetaLearningAnalyzer

# Factor library
db = PersistenceManager("./factor_library")
db.save_factor(factor_metadata)
factors = db.search_factors(min_ic=0.1)

# Meta-learning recommendations
analyzer = MetaLearningAnalyzer()
analyzer.train(factors, history)
recommendations = analyzer.get_recommendations(target_complexity=4.5)
print(f"Recommended: {recommendations.recommended_functions}")
```

## Performance Benchmarks

| Data Size | Rust | Python | Speedup |
|-----------|------|--------|---------|
| 100×200 | 5.2ms | 42.1ms | 8.1× |
| 500×500 | 68.3ms | 1.2s | 17.6× |
| 1000×1000 | 312ms | 8.7s | 27.9× |

*Test environment: AMD Ryzen 7 5800X, 32GB RAM*

## Project Structure

```
alfars/
├── Cargo.toml              # Rust project config
├── pyproject.toml          # Python project config
├── src/
│   ├── lib.rs             # Core + Python bindings
│   ├── al/                # Alpha file parser + factor store
│   ├── backtest/          # Backtest engine
│   ├── data/              # Data source abstraction
│   ├── expr/              # Expression system (AST, optimizer, registry)
│   ├── gp/                # Genetic programming + history + metalearning
│   ├── lazy/              # Lazy evaluation engine
│   ├── types/             # Series and DataFrame types
│   ├── persistence.rs     # Persistence manager
│   └── bin/server.rs      # Rust HTTP server
├── alfars/                # Python package
│   ├── __init__.py
│   ├── lab.py             # Interactive lab launcher
│   └── server.py          # FastAPI server
├── frontend/              # Interactive UI (Vite + TypeScript)
├── assets/                # Static assets (logo, etc.)
└── tests/                 # Test suite
```

## Development

```bash
# Format code before committing
cargo fmt
ruff format

# Run tests
pytest tests/

# Build release
maturin build --release
```

## Version History

### v0.5.0 (Current)
- **Streaming Pipeline**: DataPool with date-range filtering, mixed-frequency batch support
- **Fault Tolerance**: Result-based error handling for production pipeline reliability
- **Zero-Clone Backtest**: Broadcast-safe scalar handling, bounded cross-year parallelism
- **Backtest Metrics**: rebalance_freq, passive benchmark, group IC, weight turnover, win rate, Calmar ratio
- **GP Frequency Support**: Multi-frequency factor mining (1d/5m/1m) and AST redundancy detection
- **Performance**: SIMD operators, Arc sharing, O(n²) routing fix, per-window allocation elimination
- **Lab**: Server binary bundled in wheel, auto-detection for dev/prod
- **Refactor**: Backtest engine (2610→6 modules), GP engine (2945→6 modules), clean Python class names

### v0.4.0
- Interactive Lab with one-command `alfars lab` for visual factor research
- GP parallelization, dimension system, and ClickHouse integration
- Rust HTTP server as standalone backend

### v0.2.0
- Expression system with AST-based builder
- Lazy evaluation engine (Polars-style)
- Genetic programming factor mining
- Persistence and factor library management
- Meta-learning recommendations

### v0.1.0
- High-performance quantile backtesting
- Alphalens-compatible API
- NumPy/Pandas dual interface

## License

MIT License

## Acknowledgments

- [alphalens](https://github.com/quantopian/alphalens) - API design reference
- [PyO3](https://github.com/PyO3/pyo3) - Rust-Python bindings
- [ndarray](https://github.com/rust-ndarray/ndarray) - Array computing
- [rayon](https://github.com/rayon-rs/rayon) - Data parallelism
- [Polars](https://github.com/pola-rs/polars) - Lazy evaluation design
