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

def rolling_window(
    size: int,
    min_periods: Optional[int] = None,
) -> Any: ...

def expanding_window(
    min_periods: Optional[int] = None,
) -> Any: ...

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

# =============================================================================
# DataFrame and Series Classes
# =============================================================================

class Series:
    """Series data structure for expression evaluation."""

    def __init__(self, data: npt.NDArray) -> None: ...

    def len(self) -> int: ...
    def is_empty(self) -> bool: ...
    def to_list(self) -> List[float]: ...


class DataFrame:
    """DataFrame structure for expression evaluation."""

    def __init__(self, columns: Dict[str, npt.NDArray]) -> None: ...

    def n_rows(self) -> int: ...
    def n_cols(self) -> int: ...
    def column_names(self) -> List[str]: ...
    def evaluate(self, expr: Expr) -> npt.NDArray: ...


class LazyFrame:
    """Lazy evaluation frame for query optimization."""

    @staticmethod
    def scan(data: Dict[str, npt.NDArray]) -> LazyFrame:
        """Create a LazyFrame from a dictionary of data."""
        ...

    def with_columns(self, columns: List[Tuple[str, Expr]]) -> LazyFrame:
        """Add new columns to the LazyFrame."""
        ...

    def join(
        self,
        other: LazyFrame,
        on: List[str],
        how: Literal["inner", "left", "right", "outer"] = "inner",
    ) -> LazyFrame:
        """Join with another LazyFrame."""
        ...

    def collect(self) -> DataFrame:
        """Execute the lazy computation and return DataFrame."""
        ...

    def explain(self, optimized: bool = False) -> str:
        """Return the query plan."""
        ...

# =============================================================================
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
    ) -> None: ...

    def set_columns(self, columns: List[str]) -> None:
        """Set available column names for GP."""
        ...

    def mine_factors(
        self,
        data: Dict[str, npt.NDArray],
        returns: npt.NDArray,
        num_factors: int = 10,
    ) -> List[Tuple[str, float]]:
        """Mine for factors using genetic programming."""
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

    def __init__(self) -> None: ...

    def register(
        self,
        name: str,
        expression: str,
        category: Optional[str] = None,
    ) -> str:
        """Register a new factor."""
        ...

    def register_with_validation(
        self,
        name: str,
        expression: str,
        dimension_check: bool = True,
    ) -> str:
        """Register a factor with dimension validation."""
        ...

    def get(self, name: str) -> Optional[FactorMetadata]: ...
    def delete(self, name: str) -> bool: ...
    def list(self) -> List[str]: ...

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
# Module-level attributes
# =============================================================================

HAS_RUST_EXT: bool
np: np.ndarray.__class__
pd: pd.DataFrame.__class__
__version__: str
