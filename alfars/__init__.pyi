"""
Type stubs for alfars - High-performance factor expression and backtesting library.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

# Note: This is a stub file to provide type hints for the alfars package.
# The actual implementation is in Rust.

# =============================================================================
# Type Aliases
# =============================================================================

ArrayLike = npt.NDArray[np.floating]
DataDict = Dict[str, npt.NDArray]

# =============================================================================
# Core Functions
# =============================================================================

def quantile_backtest(
    factor: npt.NDArray,
    returns: npt.NDArray,
    quantiles: int = 10,
    weight_method: Literal["equal", "weighted"] = "equal",
    long_top_n: int = 1,
    short_top_n: int = 1,
    commission_rate: float = 0.0,
    weights: Optional[npt.NDArray] = None,
) -> BacktestResult: ...
def compute_information_coefficient(
    factor: npt.NDArray,
    returns: npt.NDArray,
) -> Tuple[float, float]: ...
def compute_ic(
    factor: npt.NDArray,
    returns: npt.NDArray,
) -> Tuple[float, float]: ...
def evaluate_expression(
    expr: Expr,
    data: Dict[str, npt.NDArray],
    n_days: int,
    n_assets: int,
) -> npt.NDArray: ...
def parse_expression(expression: str) -> Expr: ...
def optimize_expression(expr: Expr) -> Expr: ...
def create_factor_tear_sheet(
    factor: pd.Series,
    forward_returns: pd.Series,
    quantiles: int = 10,
    periods: Tuple[int, ...] = (1, 5, 10),
    group_neutral: bool = False,
    by_group: bool = False,
    **kwargs: Any,
) -> None: ...
def factor_returns(
    factor: pd.Series,
    forward_returns: pd.Series,
    quantiles: int = 10,
    bins: Optional[List[float]] = None,
    periods: Tuple[int, ...] = (1,),
    weights: Optional[pd.Series] = None,
    groupby: Optional[pd.Series] = None,
    zero_aware: bool = False,
) -> Dict[str, Union[pd.DataFrame, pd.Series]]: ...

# =============================================================================
# Expression Functions
# =============================================================================

def lag(expr: Expr, periods: int) -> Expr: ...
def diff(expr: Expr, periods: int) -> Expr: ...
def rolling_mean(expr: Expr, window: int) -> Expr: ...
def rank(expr: Expr) -> Expr: ...
def ts_rank(expr: Expr, window: int) -> Expr: ...
def ts_corr(expr1: Expr, expr2: Expr, window: int) -> Expr: ...
def ts_cov(expr1: Expr, expr2: Expr, window: int) -> Expr: ...
def ts_sum(expr: Expr, window: int) -> Expr: ...
def ts_count(expr: Expr, window: int) -> Expr: ...
def ts_max(expr: Expr, window: int) -> Expr: ...
def ts_min(expr: Expr, window: int) -> Expr: ...
def ts_argmax(expr: Expr, window: int) -> Expr: ...
def ts_argmin(expr: Expr, window: int) -> Expr: ...
def scale(expr: Expr, window: int) -> Expr: ...
def sign(expr: Expr) -> Expr: ...
def power(expr: Expr, exponent: float) -> Expr: ...
def decay_linear(expr: Expr, periods: int) -> Expr: ...
def cumsum(expr: Expr) -> Expr: ...
def cumprod(expr: Expr) -> Expr: ...

# =============================================================================
# Expr Class
# =============================================================================

class Expr:
    """Factor expression builder with chainable operations."""

    @staticmethod
    def col(name: str) -> Expr:
        """Create a column reference expression."""
        ...

    @staticmethod
    def lit_float(value: float) -> Expr:
        """Create a float literal expression."""
        ...

    @staticmethod
    def lit_int(value: int) -> Expr:
        """Create an integer literal expression."""
        ...

    @staticmethod
    def lit_bool(value: bool) -> Expr:
        """Create a boolean literal expression."""
        ...

    # Arithmetic operations
    def add(self, other: Union[Expr, float, int]) -> Expr: ...
    def sub(self, other: Union[Expr, float, int]) -> Expr: ...
    def mul(self, other: Union[Expr, float, int]) -> Expr: ...
    def div(self, other: Union[Expr, float, int]) -> Expr: ...

    # Unary operations
    def neg(self) -> Expr: ...
    def abs(self) -> Expr: ...

    # Mathematical functions
    def sqrt(self) -> Expr: ...
    def exp(self) -> Expr: ...
    def log(self) -> Expr: ...

    # Comparison operations
    def gt(self, other: Union[Expr, float, int]) -> Expr: ...
    def lt(self, other: Union[Expr, float, int]) -> Expr: ...
    def eq(self, other: Union[Expr, float, int]) -> Expr: ...
    def ne(self, other: Union[Expr, float, int]) -> Expr: ...

# =============================================================================
# Backtest Classes
# =============================================================================

class BacktestEngine:
    """Backtest engine with configurable parameters."""

    def __init__(
        self,
        factor: npt.NDArray,
        returns: npt.NDArray,
        quantiles: int = 10,
        weight_method: Literal["equal", "weighted"] = "equal",
        long_top_n: int = 1,
        short_top_n: int = 1,
        commission_rate: float = 0.0,
        weights: Optional[npt.NDArray] = None,
    ) -> None: ...
    def run(self) -> BacktestResult:
        """Execute the backtest and return results."""
        ...

class BacktestResult:
    """Container for backtest results."""

    group_returns: npt.NDArray
    group_cum_returns: npt.NDArray
    long_short_returns: npt.NDArray
    long_short_cum_return: float
    ic_series: npt.NDArray
    ic_mean: float
    ic_ir: float

    def summary(self) -> str:
        """Return a summary string of the backtest results."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary."""
        ...

    @staticmethod
    def from_rust_result(result: Any) -> BacktestResult:
        """Create BacktestResult from Rust result."""
        ...

# Genetic Programming Classes
# =============================================================================

class GpEngine:
    """Genetic programming factor mining engine."""

    def __init__(
        self,
        population_size: int = 100,
        max_generations: int = 50,
        tournament_size: int = 7,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
        max_depth: int = 6,
        allow_ephemeral: bool = True,
    ) -> None: ...
    def set_columns(
        self, columns: List[str], allow_ephemeral: Optional[bool] = None
    ) -> None:
        """Set available column names for GP."""
        ...

    def mine_factors(
        self,
        data: Dict[str, npt.NDArray],
        returns: npt.NDArray,
        num_factors: int = 10,
        weight_ic: Optional[float] = None,
        weight_ir: Optional[float] = None,
        weight_turnover: Optional[float] = None,
        weight_complexity: Optional[float] = None,
    ) -> List[Tuple[str, float, float, float, float, int]]:
        """
        Mine for factors using genetic programming with multi-objective optimization.

        Returns list of (expression, fitness, ic, ir, turnover, complexity)
        """
        ...

    def test_run(self) -> Dict[str, Any]:
        """Run a test execution of the GP engine."""
        ...

# =============================================================================
# Persistence Classes
# =============================================================================

class PersistenceManager:
    """Factor library persistence manager."""

    def __init__(self, path: str) -> None: ...
    def save_factor(self, metadata: FactorMetadata) -> None:
        """Save a factor to the library."""
        ...

    def load_factor(self, factor_id: str) -> Optional[FactorMetadata]:
        """Load a factor by ID."""
        ...

    def search_factors(
        self,
        min_ic: Optional[float] = None,
        max_complexity: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> List[FactorMetadata]: ...
    def get_all_factors(self) -> List[FactorMetadata]: ...
    def get_all_history(self) -> List[GPHistoryRecord]: ...
    def load_all_factors(self) -> int: ...
    def load_all_history(self) -> int: ...
    def clear_memory(self) -> None: ...
    def cache_stats(self) -> Dict[str, Any]: ...

class FactorMetadata:
    """Metadata container for factors."""

    def __init__(
        self,
        id: str,
        expression: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
    ) -> None: ...

    id: str
    expression: str
    metrics: Dict[str, float]
    tags: List[str]

class GPHistoryRecord:
    """Container for GP run history."""

    run_id: str
    best_factor: FactorMetadata
    config: Dict[str, Any]

# =============================================================================
# Meta Learning Classes
# =============================================================================

class MetaLearningAnalyzer:
    """Meta-learning analyzer for GP recommendations."""

    def __init__(self) -> None: ...
    def set_high_perf_threshold(self, threshold: float) -> None: ...
    def set_min_data_points(self, min_points: int) -> None: ...
    def get_high_perf_threshold(self) -> float: ...
    def get_min_data_points(self) -> int: ...
    def train(
        self,
        factors: List[FactorMetadata],
        history: List[GPHistoryRecord],
    ) -> None: ...
    def is_trained(self) -> bool: ...
    def version(self) -> str: ...
    def confidence_score(self) -> float: ...
    def get_recommendations(
        self,
        target_complexity: Optional[float] = None,
    ) -> GPRecommendations: ...
    def save_model(self, path: str) -> None: ...
    @staticmethod
    def load_model(path: str) -> MetaLearningAnalyzer: ...

class GPRecommendations:
    """GP configuration recommendations."""

    recommended_functions: List[str]
    recommended_terminals: List[str]
    target_complexity: float
    confidence_score: float

    def is_valid(self) -> bool: ...
    def to_gp_config(self) -> Dict[str, Any]: ...
    def confidence_level(self) -> Literal["high", "medium", "low"]: ...

# =============================================================================
# Registry Classes
# =============================================================================

class AlphaRegistry:
    """Registry for factor management."""

    def __init__(self, storage_dir: Optional[str] = None) -> None: ...
    def register(
        self,
        name: str,
        expression: str,
        category: str = "custom",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Register a new factor."""
        ...

    def register_with_validation(
        self,
        name: str,
        expression: str,
        category: str = "custom",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        validate_dimension: bool = True,
    ) -> str:
        """Register a factor with dimension validation."""
        ...

    def get(self, factor_id: str) -> Optional[Dict[str, Any]]: ...
    def delete(self, factor_id: str) -> bool: ...
    def list(self, category: Optional[str] = None) -> List[Dict[str, Any]]: ...

class FactorRegistry:
    """Rust-backed factor registry with expression parsing and compute."""

    def __init__(
        self, mode: Literal["default", "conservative", "high_performance"] = "default"
    ) -> None: ...
    def set_columns(self, columns: List[str]) -> None:
        """Set available column names (e.g., ['close', 'volume', 'open'])."""
        ...

    def columns(self) -> List[str]:
        """Get available column names."""
        ...

    def register(self, name: str, expression: str) -> str:
        """
        Register a factor expression.

        The backend automatically parses the expression and generates a compute plan.
        """
        ...

    def compute(
        self, name: str, data: Dict[str, Union[npt.NDArray, List[float]]]
    ) -> FactorResult: ...
    def compute_batch(
        self,
        names: List[str],
        data: Dict[str, Union[npt.NDArray, List[float]]],
        parallel: bool = False,
    ) -> Dict[str, FactorResult]: ...
    def compute_cs_pipeline(
        self, data_layer: DataLayer
    ) -> Dict[str, FactorSlice]:
        """
        Compute all registered factors via the cross-sectional pipeline.

        Queries 5m data via DataLayer, computes factors, applies
        winsor→zscore→cap_neu→qcut per date. Returns FactorSlice per factor.
        """
        ...
    def compute_factor_matrices_1d(
        self, data_layer: DataLayer
    ) -> Tuple[Dict[str, npt.NDArray[np.float64]], PriceMatrix]:
        """
        Compute factors on 1d data and return 2D matrices aligned with PriceMatrix.

        Returns (factor_matrices_dict, price_matrix).
        """
        ...

    def list(self) -> List[str]: ...
    def get(self, name: str) -> Optional[FactorInfo]: ...
    def unregister(self, name: str) -> bool: ...
    def clear(self) -> None: ...
    def get_config(self) -> Dict[str, Any]: ...

class FactorInfo:
    """
    Factor information container.

    This class is created by FactorRegistry.get() - do not instantiate directly.
    """

    name: str
    expression: str
    description: str
    category: str

    def __repr__(self) -> str: ...

class FactorResult:
    """
    Factor computation result.

    This class is created by FactorRegistry.compute() - do not instantiate directly.
    """

    name: str
    """Factor name"""

    values: npt.NDArray[np.float64]
    """Computed factor values"""

    n_rows: int
    """Number of rows (elements) in the result"""

    n_cols: int
    """Number of columns (not used, kept for compatibility)"""

    compute_time_ms: float
    """Computation time in milliseconds"""

    def __repr__(self) -> str: ...

# =============================================================================
# Dimension System
# =============================================================================

class Dimension:
    """Dimension system for type safety in factor expressions."""

    PRICE: str
    RETURN: str
    VOLUME: str
    AMOUNT: str
    RATIO: str
    DIMENSIONLESS: str

    @staticmethod
    def infer(expr: Expr) -> Dimension: ...
    @staticmethod
    def is_valid_factor(expr: Expr) -> bool: ...

# =============================================================================
# Data Pipeline Classes
# =============================================================================

class ClickHouseSource:
    """ClickHouse data source configured from environment variables."""

    @staticmethod
    def from_env() -> ClickHouseSource:
        """Create from CLICKHOUSE_* or CH_* environment variables."""
        ...
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        database: str = "default",
        username: str = "default",
        password: Optional[str] = None,
    ) -> None: ...
    @property
    def host(self) -> str: ...
    @property
    def port(self) -> int: ...
    @property
    def database(self) -> str: ...
    @property
    def username(self) -> str: ...

class DataLayer:
    """Intelligent data fetching layer backed by ClickHouse."""

    def __init__(self, source: ClickHouseSource) -> None: ...
    def set_pre_filter(self, filter: str) -> None:
        """Set date/symbol filter (e.g., '2024-01-01:2025-01-01 symbols not like \'%BJ\'')."""
        ...
    def query_price_matrix(self) -> PriceMatrix:
        """Query 1d OHLCV data as aligned 2D matrices."""
        ...
    def clear_cache(self) -> None: ...
    @property
    def symbols_5m(self) -> List[str]:
        """Cached symbol list from last query."""
        ...

class PriceMatrix:
    """2D price data matrices (n_dates × n_symbols), forward-adjusted."""

    @property
    def dates(self) -> List[int]: ...
    @property
    def symbols(self) -> List[str]: ...
    @property
    def n_dates(self) -> int: ...
    @property
    def n_symbols(self) -> int: ...
    @property
    def close(self) -> npt.NDArray[np.float64]: ...
    @property
    def open(self) -> npt.NDArray[np.float64]: ...
    @property
    def high(self) -> npt.NDArray[np.float64]: ...
    @property
    def low(self) -> npt.NDArray[np.float64]: ...
    @property
    def vwap(self) -> npt.NDArray[np.float64]: ...
    @property
    def returns(self) -> npt.NDArray[np.float64]: ...
    @property
    def tradable(self) -> npt.NDArray[np.float64]: ...

    def build_factor_matrix(
        self, slices: List[FactorSlice]
    ) -> npt.NDArray[np.float64]:
        """
        Build a (n_dates × n_symbols) factor matrix from FactorSlices.

        Each tuple is (FactorSlice, symbol_list).
        """
        ...

class FactorSlice:
    """Cross-sectional result for one factor × one year."""

    @property
    def factor_name(self) -> str:
        """Which factor this slice belongs to."""
        ...
    @property
    def groups(self) -> List[Tuple[int, int]]:
        """(date_int, symbol_idx) keys for each value."""
        ...
    @property
    def symbols(self) -> List[str]:
        """Symbol strings at each index position."""
        ...
    @property
    def cap_neued(self) -> List[float]:
        """Market-cap neutralized factor values."""
        ...
    @property
    def qcut(self) -> List[Optional[int]]:
        """Quantile group assignments (0..9 or None)."""
        ...

# =============================================================================
# Factor Combination + Position Building
# =============================================================================

class FactorCombiner:
    """Multi-factor combination methods."""

    @staticmethod
    def equal_weight(
        factors: List[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """Equal-weight average, NaN-aware per cell."""
        ...
    @staticmethod
    def rank_average(
        factors: List[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """Average of cross-sectional normalized ranks."""
        ...
    @staticmethod
    def signal_weighted(
        factors: List[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """Weight each factor by z-scored signal magnitude, then sum."""
        ...

class PositionBuilder:
    """Factor value → position/weight matrix conversion."""

    @staticmethod
    def from_factor(
        factor: npt.NDArray[np.float64],
        quantiles: int,
        long_top_n: int,
        short_top_n: int,
    ) -> npt.NDArray[np.float64]:
        """
        Convert factor matrix to long-short position matrix.
        Positions: 1 (long), -1 (short), 0 (neutral).
        """
        ...

# =============================================================================
# Rust-Backed Backtest Engine
# =============================================================================

class PyBacktestEngine:
    """Rust backtest engine with configurable parameters."""

    def __init__(
        self,
        quantiles: int = 10,
        weight_method: Literal["equal", "weighted"] = "equal",
        long_top_n: int = 1,
        short_top_n: int = 1,
        commission_rate: float = 0.0,
    ) -> None: ...
    def run(
        self,
        factor: npt.NDArray[np.float64],
        returns: npt.NDArray[np.float64],
        adj_factor: npt.NDArray[np.float64],
        close: npt.NDArray[np.float64],
        open: npt.NDArray[np.float64],
        vwap: npt.NDArray[np.float64],
        tradable: npt.NDArray[np.float64],
    ) -> PyBacktestResult: ...
    def run_multi(
        self,
        factors: List[npt.NDArray[np.float64]],
        returns: npt.NDArray[np.float64],
        adj_factor: npt.NDArray[np.float64],
        close: npt.NDArray[np.float64],
        open: npt.NDArray[np.float64],
        vwap: npt.NDArray[np.float64],
        tradable: npt.NDArray[np.float64],
    ) -> PyBacktestResult:
        """Multi-factor equal-weight backtest."""
        ...
    def run_with_prices(
        self,
        factor: npt.NDArray[np.float64],
        prices: PriceMatrix,
    ) -> PyBacktestResult:
        """Single-factor backtest with PriceMatrix (auto-aligned)."""
        ...
    def run_multi_with_prices(
        self,
        factors: List[npt.NDArray[np.float64]],
        prices: PriceMatrix,
    ) -> PyBacktestResult:
        """Multi-factor equal-weight backtest with PriceMatrix."""
        ...

class PyBacktestResult:
    """Backtest result from Rust engine."""

    dates: List[int]
    group_returns: npt.NDArray[np.float64]
    group_cum_returns: npt.NDArray[np.float64]
    long_short_returns: npt.NDArray[np.float64]
    long_short_cum_return: float
    long_short_cum_returns: npt.NDArray[np.float64]
    long_cum_returns: npt.NDArray[np.float64]
    short_cum_returns: npt.NDArray[np.float64]
    ic_series: npt.NDArray[np.float64]
    ic_mean: float
    ic_ir: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float
    long_returns: npt.NDArray[np.float64]
    short_returns: npt.NDArray[np.float64]

class PyFeeConfig:
    """Fee configuration for backtest."""

    commission_rate: float

class PyPositionConfig:
    """Position sizing configuration."""

    long_ratio: float
    short_ratio: float
    market_neutral: bool

class PySlippageConfig:
    """Volume-based slippage configuration."""

    large_volume_threshold: float
    large_slippage_rate: float

# Python-side aliases for backtest config
FeeConfig = PyFeeConfig
PositionConfig = PyPositionConfig
SlippageConfig = PySlippageConfig

# =============================================================================
# Module-level attributes
# =============================================================================

HAS_RUST_EXT: bool
np: np.ndarray.__class__
pd: pd.DataFrame.__class__
__version__: str
