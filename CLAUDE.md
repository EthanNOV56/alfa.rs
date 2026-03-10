# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

alfars (alfa.rs) is a high-performance factor expression and backtesting library written in Rust with Python bindings via PyO3. It provides tools for quantitative factor research including expression evaluation, lazy computation, genetic programming-based factor mining, and meta-learning recommendations.

**Python Package Name**: `alfars` (import as `import alfars as al`)

## Common Commands

```bash
# Build and install the Python extension in development mode
maturin develop

# Build with optimizations (recommended for performance testing)
maturin develop --release

# Run tests
pytest tests/

# Run a specific test
pytest tests/test_basic.py::test_quantile_backtest_basic

# Build only the Rust library (no Python extension)
cargo build --release

# Build wheel package for distribution
maturin build --release

# Interactive Visualization - Two Options:

# Option 1: Rust HTTP Server (recommended - no Python dependency)
cargo run --release --bin alfars-server   # Start Rust backend (port 8000)
cd frontend && npm run dev                 # Start frontend (port 5173)

# Option 2: Python FastAPI (requires maturin develop first)
uv run python -m alfars.server            # Start Python backend (port 8000)
cd frontend && npm run dev                # Start frontend (port 5173)
```

## Environment Management

This project uses `uv` for Python dependency management:
```bash
# Run Python with the project environment
uv run python your_script.py
```

## Architecture

### Core Modules (`src/`)

| Module | Purpose |
|--------|---------|
| `expr.rs` | AST-based expression system with Literal, Column, BinaryExpr, UnaryExpr, FunctionCall variants |
| `expr_optimizer.rs` | Expression optimization (constant folding, algebraic simplification) |
| `lazy.rs` | Polars-style lazy evaluation engine with query optimization |
| `gp.rs` | Genetic programming factor mining engine |
| `persistence.rs` | Factor library management with LZ4 compression and UUID identification |
| `metalearning.rs` | Meta-learning analyzer for intelligent GP recommendations |
| `polars_style.rs` | DataFrame/Series compatibility layer |

### Key Types

```rust
// Expression AST
pub enum Expr {
    Literal(Literal),
    Column(String),
    BinaryExpr { left, op, right },
    UnaryExpr { op, expr },
    FunctionCall { name, args },
}

// Dimension system for type safety in factor expressions
pub enum Dimension {
    Price, Return, Volume, Amount, Ratio, Dimensionless,
}
```

### Python Package

The package is named `exprs` (configured in pyproject.toml). The Rust extension is built as `exprs._core` via maturin.

## Design Principles

### 1. Ergonomic Python API
Expose user-friendly Python APIs for factor registration:
- Simple factor registration interface
- Intuitive expression building
- Clear error messages

### 2. AST-based Expression System
Parse expressions into AST (Abstract Syntax Tree):
- Parse factor expressions into tree structures
- Support expression optimization (constant folding, algebraic simplification)
- Support expression serialization/deserialization

### 3. Lazy Evaluation & Optimization
Optimization through Lazy mode:
- Lazy evaluation: compute on demand, avoid unnecessary intermediate results
- Expression optimization: constant folding, common subexpression elimination
- Vectorized execution: leverage SIMD and parallel computing

### 4. GP-based Factor Mining
Genetic programming-based factor mining:
- Auto-generate candidate factor expressions
- Fitness function based on IC/IR metrics
- Support custom operator sets

### 5. Dimension System
Introduce dimension system in GP to prevent generating unreasonable expressions:
- **Price/Return dimensions**: `price`, `return`, `volume`, `amount`
- **Normalization check**: detect non-normalized price/volume expressions
- **Dimension compatibility check**: `price + return` illegal, `return / return` legal
- **Auto-fix**: provide correction suggestions for illegal expressions

## Development Notes

- PyO3 version: 0.28 (upgraded from 0.20)
- When upgrading PyO3, follow the sequence: 0.20 → 0.21 → 0.23 → 0.24 → 0.25 → 0.26 → 0.27 → 0.28
- Common API changes to watch for:
  - `Python::with_gil` → `pyo3::Python::try_attach` + `.ok_or_else()`
  - `dict.into_pyobject(py)` → `dict.into()`
  - `into_pyarray_bound` → `into_pyarray`

### Code Development Standards

1. **Check for similar implementations first**: Before implementing new features, search for similar implementations in the project:
   ```bash
   grep -r "pattern" src/
   glob "src/**/*.rs"
   ```
   If similar code exists, extend it rather than duplicating.

2. **Clean up unused files regularly**:
   - Remove unused test files in `examples/` directory
   - Keep the project directory clean
   - Delete unused code and comments

3. **Use proper Git practices**:
   - Use clear English commit messages
   - Recommended format:
     - `feat: add new feature`
     - `fix: fix bug in xxx`
     - `refactor: simplify xxx`
     - `chore: update dependencies`
   - Use `git add <specific_file>` instead of `git add -A` to avoid committing unintended files

4. **Format code before committing**:
   - Run `cargo fmt` to format Rust code
   - Run `ruff fmt` to format Python code
   - This ensures consistent code style across the project

## Versioning and Release

This project follows [Semantic Versioning](https://semver.org/) (SemVer):

- **MAJOR** (X.0.0): Incompatible API changes
- **MINOR** (x.Y.0): New backwards-compatible functionality
- **PATCH** (x.y.Z): Backwards-compatible bug fixes

### Version Update Rules

1. **MAJOR bump** (e.g., 1.0.0 → 2.0.0): When making breaking changes to the public API
2. **MINOR bump** (e.g., 1.0.0 → 1.1.0): When adding new features without breaking existing API
3. **PATCH bump** (e.g., 1.0.0 → 1.0.1): When fixing bugs without changing API

4. **Automatic Publishing**: After bumping version in both `Cargo.toml` and `pyproject.toml`, automatically run:

   ```bash
   # Update version in Cargo.toml and pyproject.toml first
   # Then publish to both crates.io and PyPI
   cargo publish
   maturin build --release
   maturin publish
   ```

### Release to crates.io

Publish to crates.io promptly after version changes:

```bash
# Update version in Cargo.toml first
# Then publish
cargo publish

# Or use cargo-release for automated releases
cargo install cargo-release
cargo release patch  # or minor, major
```

### Python Package Version

The Python package version (in `pyproject.toml`) should match the Rust crate version. Keep them in sync when releasing.

### Python API Examples

#### Factor Registration
```python
from exprs import AlphaRegistry

registry = AlphaRegistry()

# Register factor expression
alpha_id = registry.register(
    name="alpha_001",
    expression="rank(ts_mean(close, 20))",
    category="momentum"
)

# Register with dimension check
alpha_id = registry.register_with_dimension(
    name="alpha_002",
    expression="return / volume",  # valid: ratio / volume = ratio
    dimension_check=True
)
```

#### Expression Parsing
```python
from exprs import parse_expression

# Parse to AST
ast = parse_expression("rank(ts_mean(close, 20))")

# Optimize expression
optimized = optimize_expression(ast)

# Print expression tree
print(ast)
```

#### Lazy Computation
```python
from exprs import LazyFrame

lf = LazyFrame.from_dataframe(df)
result = lf.select(
    rank(ts_mean(close, 20)) - rank(ts_mean(open, 20))
).filter(close > 100).execute()
```

#### GP Factor Mining
```python
from exprs import GpEngine

engine = GpEngine(
    population_size=100,
    max_generations=50,
    terminal_set=["close", "open", "high", "low", "volume"],
    function_set=["rank", "ts_mean", "ts_std", "delay"],
    dimension_aware=True,  # Enable dimension check
)

best_alpha = engine.evolve(returns, target_ic=0.05)
```

## Data Source: ClickHouse

This project uses ClickHouse for storing historical market data. Connect with:

```python
from clickhouse_connect import get_client

client = get_client(host='localhost', port=8123, username='readonly_user")
```

### Available Tables

| Table | Rows | Columns | Description |
|-------|------|---------|-------------|
| `default.stock_1d` | 18.6M | 30 | Daily OHLCV + fundamentals |
| `default.stock_1m` | 3.1B | 11 | 1-minute bars |
| `default.stock_5m` | 655M | 11 | 5-minute bars |

### Key Columns

**stock_1d:**
- `symbol`, `name`, `trading_date`
- `open`, `high`, `low`, `close`, `volume`, `amount`
- `pe`, `pb`, `market_cap`, `total_shares`, `float_shares`

**stock_1m/stock_5m:**
- `trading_time`, `symbol`
- `close`, `open`, `high`, `low`, `vol`, `amount`
- Materialized columns: `trading_date`, `time_slot`

### Common Queries

```python
# Read daily data for a symbol
df = client.query("""
    SELECT * FROM default.stock_1d
    WHERE symbol = '000001.SZ'
    ORDER BY trading_date DESC
    LIMIT 100
""").result_df()

# Read minute data (limit date range for performance)
df = client.query("""
    SELECT * FROM default.stock_1m
    WHERE symbol = '000001.SZ'
      AND trading_time >= '2025-01-01'
      AND trading_time < '2025-01-02'
    ORDER BY trading_time
""").result_df()

# Aggregate minute to daily
df = client.query("""
    SELECT
        symbol,
        toDate(trading_time) as trading_date,
        any(open) as open,
        max(high) as high,
        min(low) as low,
        anyLast(close) as close,
        sum(vol) as volume
    FROM default.stock_1m
    WHERE symbol = '000001.SZ' AND trading_time >= '2025-01-01'
    GROUP BY symbol, toDate(trading_time)
    ORDER BY trading_date
""").result_df()
```

### Performance Tips

1. Use materialized columns for filtering: `trading_date`, `time_slot`
2. Limit columns: `SELECT symbol, close, volume FROM ...`
3. Limit date range for minute data
4. Use `LIMIT batch_size OFFSET offset` for pagination
