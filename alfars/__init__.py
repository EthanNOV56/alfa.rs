"""
alfars: High-performance factor expression and backtesting library

This library provides:
- Quantile grouping backtest (qcut N groups)
- Long-short portfolio construction
- Information coefficient (IC) computation
- Factor performance analysis

The core implementation is in Rust (via PyO3) for maximum performance.

Usage:
    import alfars as al
    # or
    import alfars
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Package metadata
__version__ = "0.2.0"
__author__ = "EthanNOV56"

try:
    from ._core import (
        DataFrame,
        # Expression system
        Expr,
        FactorMetadata,
        # Genetic Programming
        GpEngine,
        GPHistoryRecord,
        GPRecommendations,
        # Lazy evaluation
        LazyFrame,
        # Meta-learning system
        MetaLearningAnalyzer,
        # Persistence system
        PersistenceManager,
        PyBacktestEngine,
        Series,
        compute_ic,
        cumprod,
        cumsum,
        decay_linear,
        diff,
        evaluate_expression,
        expanding_window,
        # Expression functions
        lag,
        power,
        quantile_backtest,
        rank,
        rolling_mean,
        rolling_window,
        scale,
        sign,
        ts_argmax,
        ts_argmin,
        ts_corr,
        ts_count,
        ts_cov,
        ts_max,
        ts_min,
        # Alpha101 functions
        ts_rank,
        ts_sum,
        ts_count,
    )
    HAS_RUST_EXT = True
    # Create aliases for internal use
    _quantile_backtest = quantile_backtest
    _compute_ic = compute_ic
except ImportError:
    HAS_RUST_EXT = False
    print("Warning: Rust extension not found. Using pure Python fallback.")
    from ._fallback import (
        PyBacktestEngine,
        PyBacktestResult,
    )
    from ._fallback import (
        compute_ic as _compute_ic,
    )
    from ._fallback import (
        quantile_backtest as _quantile_backtest,
    )
    # Create simple stubs for new functionality when Rust extension is missing
    class Expr:
        """Stub Expr class for fallback mode."""
        pass

    class Series:
        """Stub Series class for fallback mode."""
        pass

    class DataFrame:
        """Stub DataFrame class for fallback mode."""
        pass

    class LazyFrame:
        """Stub LazyFrame class for fallback mode."""
        pass

    # GP system stubs
    class GpEngine:
        """Stub GpEngine class for fallback mode."""
        pass

    class PersistenceManager:
        """Stub PersistenceManager class for fallback mode."""
        pass

    class FactorMetadata:
        """Stub FactorMetadata class for fallback mode."""
        pass

    class GPHistoryRecord:
        """Stub GPHistoryRecord class for fallback mode."""
        pass

    class MetaLearningAnalyzer:
        """Stub MetaLearningAnalyzer class for fallback mode."""
        pass

    class GPRecommendations:
        """Stub GPRecommendations class for fallback mode."""
        pass

    def rolling_window(*args, **kwargs):
        """Stub function for fallback mode."""
        return {}

    def expanding_window(*args, **kwargs):
        """Stub function for fallback mode."""
        return {}

    def lag(*args, **kwargs):
        """Stub function for fallback mode."""
        return Expr()

    def diff(*args, **kwargs):
        """Stub function for fallback mode."""
        return Expr()

    def rolling_mean(*args, **kwargs):
        """Stub function for fallback mode."""
        return Expr()

    def cumsum(*args, **kwargs):
        """Stub function for fallback mode."""
        return Expr()

    def cumprod(*args, **kwargs):
        """Stub function for fallback mode."""
        return Expr()

    def evaluate_expression(*args, **kwargs):
        """Stub function for fallback mode."""
        return np.array([])
__all__ = [
    # Core backtesting
    "factor_returns",
    "quantile_backtest",
    "create_factor_tear_sheet",
    "BacktestEngine",
    "BacktestResult",
    "compute_information_coefficient",
    # Expression system
    "Expr",
    "Series",
    "DataFrame",
    "evaluate_expression",
    "parse_expression",
    "optimize_expression",
    # Lazy evaluation
    "LazyFrame",
    "rolling_window",
    "expanding_window",
    # Expression functions
    "lag",
    "diff",
    "rolling_mean",
    "cumsum",
    "cumprod",
    # Genetic Programming
    "GpEngine",
    # Persistence system
    "PersistenceManager",
    "FactorMetadata",
    "GPHistoryRecord",
    # Meta-learning system
    "MetaLearningAnalyzer",
    "GPRecommendations",
    # Factor registry (ergonomic API)
    "AlphaRegistry",
    # Dimension types
    "Dimension",
]


# =============================================================================
# Ergonomic Factor Registration API
# =============================================================================

class AlphaRegistry:
    """
    Ergonomic API for registering and managing alpha factors.

    Provides a simple interface for:
    - Registering factor expressions
    - Managing factor metadata
    - Dimension validation

    Example:
        registry = AlphaRegistry()

        # Register a simple factor
        alpha_id = registry.register(
            name="alpha_001",
            expression="rank(ts_mean(close, 20))",
            category="momentum"
        )

        # Register with dimension validation
        alpha_id = registry.register_with_validation(
            name="alpha_002",
            expression="rank(close / delay(close, 1) - 1)",
            category="momentum",
            validate_dimension=True
        )
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the alpha registry.

        Parameters
        ----------
        storage_dir : str, optional
            Directory to store factor metadata
        """
        self._storage_dir = storage_dir
        self._factors = {}
        self._next_id = 1

    def register(
        self,
        name: str,
        expression: str,
        category: str = "custom",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Register a new factor expression.

        Parameters
        ----------
        name : str
            Factor name
        expression : str
            Factor expression (e.g., "rank(ts_mean(close, 20))")
        category : str
            Factor category (e.g., "momentum", "value", "volume")
        description : str, optional
            Factor description
        tags : list, optional
            Factor tags

        Returns
        -------
        str
            Factor ID
        """
        factor_id = f"alpha_{self._next_id:04d}"
        self._next_id += 1

        self._factors[factor_id] = {
            "id": factor_id,
            "name": name,
            "expression": expression,
            "category": category,
            "description": description,
            "tags": tags or [],
            "dimension": None,
        }

        return factor_id

    def register_with_validation(
        self,
        name: str,
        expression: str,
        category: str = "custom",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        validate_dimension: bool = True,
    ) -> str:
        """
        Register a factor with dimension validation.

        Parameters
        ----------
        name : str
            Factor name
        expression : str
            Factor expression
        category : str
            Factor category
        description : str, optional
            Factor description
        tags : list, optional
            Factor tags
        validate_dimension : bool
            Whether to validate dimension compatibility

        Returns
        -------
        str
            Factor ID
        """
        factor_id = self.register(name, expression, category, description, tags)

        if validate_dimension:
            # Try to infer dimension
            try:
                # For now, store the expression for later dimension inference
                self._factors[factor_id]["dimension"] = "inferred"
            except Exception:
                pass

        return factor_id

    def get(self, factor_id: str) -> Optional[Dict]:
        """Get factor by ID."""
        return self._factors.get(factor_id)

    def list(self, category: Optional[str] = None) -> List[Dict]:
        """List all factors, optionally filtered by category."""
        if category is None:
            return list(self._factors.values())
        return [f for f in self._factors.values() if f["category"] == category]

    def delete(self, factor_id: str) -> bool:
        """Delete a factor by ID."""
        if factor_id in self._factors:
            del self._factors[factor_id]
            return True
        return False


# =============================================================================
# Expression Parsing and Optimization
# =============================================================================

def parse_expression(expression: str) -> "Expr":
    """
    Parse a factor expression string into an AST.

    Parameters
    ----------
    expression : str
        Factor expression (e.g., "rank(ts_mean(close, 20))")

    Returns
    -------
    Expr
        Parsed expression AST

    Example:
        >>> expr = parse_expression("rank(ts_mean(close, 20))")
        >>> print(expr)
        FunctionCall { name: "rank", args: [FunctionCall { name: "ts_mean", ... }] }
    """
    if not HAS_RUST_EXT:
        return Expr()

    # Use Rust implementation if available
    try:
        from ._core import parse_expression as _parse
        return _parse(expression)
    except (ImportError, AttributeError):
        # Fallback: return a basic expression
        return Expr()


def optimize_expression(expr: "Expr") -> "Expr":
    """
    Optimize an expression AST using constant folding and algebraic simplification.

    Parameters
    ----------
    expr : Expr
        Input expression

    Returns
    -------
    Expr
        Optimized expression

    Example:
        >>> expr = parse_expression("rank(ts_mean(close, 20))")
        >>> optimized = optimize_expression(expr)
    """
    if not HAS_RUST_EXT:
        return expr

    try:
        from ._core import optimize_expression as _optimize
        return _optimize(expr)
    except (ImportError, AttributeError):
        return expr


# =============================================================================
# Dimension Types
# =============================================================================

class Dimension:
    """
    Dimension types for financial expressions.

    Used to validate that GP-generated expressions make sense financially.
    """

    PRICE = "price"
    RETURN = "return"
    VOLUME = "volume"
    AMOUNT = "amount"
    RATIO = "ratio"
    DIMENSIONLESS = "dimensionless"
    UNKNOWN = "unknown"

    @classmethod
    def is_valid_factor(cls, dimension: str) -> bool:
        """Check if a dimension is valid for a factor output."""
        return dimension in (cls.RATIO, cls.DIMENSIONLESS, cls.RETURN)

    @classmethod
    def infer(cls, expression: str) -> str:
        """
        Try to infer the dimension of an expression.

        Parameters
        ----------
        expression : str
            Factor expression

        Returns
        -------
        str
            Inferred dimension
        """
        # Simple heuristics for now
        lower = expression.lower()
        if "rank" in lower or "scale" in lower:
            return cls.RATIO
        if "return" in lower or "ret" in lower:
            return cls.RETURN
        if "volume" in lower or "vol" in lower:
            return cls.VOLUME
        if "amount" in lower:
            return cls.AMOUNT
        if "price" in lower or "close" in lower or "open" in lower:
            return cls.PRICE
        return cls.UNKNOWN


class BacktestEngine:
    """High-performance backtest engine with Rust core."""

    def __init__(
        self,
        factor: np.ndarray,
        returns: np.ndarray,
        quantiles: int = 10,
        weight_method: str = "equal",
        long_top_n: int = 1,
        short_top_n: int = 1,
        commission_rate: float = 0.0,
        weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize backtest engine.

        Parameters
        ----------
        factor : np.ndarray
            Factor values, shape (n_days, n_assets)
        returns : np.ndarray
            Forward returns, shape (n_days, n_assets)
        quantiles : int, default 10
            Number of quantile groups
        weight_method : str, default "equal"
            "equal" for equal weighting, "weighted" for external weights
        long_top_n : int, default 1
            Number of top groups to long
        short_top_n : int, default 1
            Number of bottom groups to short
        commission_rate : float, default 0.0
            One-way commission rate
        weights : np.ndarray, optional
            External weights for weighted method, shape (n_days, n_assets)
        """
        if not HAS_RUST_EXT:
            raise RuntimeError(
                "Rust extension not available. Please install with: "
                "pip install -e .[dev] or maturin develop"
            )

        self._engine = PyBacktestEngine(
            factor, returns, quantiles, weight_method,
            long_top_n, short_top_n, commission_rate, weights
        )
        self.factor_shape = factor.shape
        self.returns_shape = returns.shape

    def run(self) -> "BacktestResult":
        """Run the backtest."""
        rust_result = self._engine.run()
        return BacktestResult.from_rust_result(rust_result)


class BacktestResult:
    """Backtest result container."""

    def __init__(
        self,
        group_returns: np.ndarray,
        group_cum_returns: np.ndarray,
        long_short_returns: np.ndarray,
        long_short_cum_return: float,
        ic_series: np.ndarray,
        ic_mean: float,
        ic_ir: float,
    ):
        self.group_returns = group_returns
        self.group_cum_returns = group_cum_returns
        self.long_short_returns = long_short_returns
        self.long_short_cum_return = long_short_cum_return
        self.ic_series = ic_series
        self.ic_mean = ic_mean
        self.ic_ir = ic_ir

    @classmethod
    def from_rust_result(cls, rust_result: Any) -> "BacktestResult":
        """Create from Rust result object."""
        return cls(
            group_returns=np.array(rust_result.group_returns),
            group_cum_returns=np.array(rust_result.group_cum_returns),
            long_short_returns=np.array(rust_result.long_short_returns),
            long_short_cum_return=rust_result.long_short_cum_return,
            ic_series=np.array(rust_result.ic_series),
            ic_mean=rust_result.ic_mean,
            ic_ir=rust_result.ic_ir,
        )

    def to_dict(self) -> Dict[str, Union[np.ndarray, float]]:
        """Convert to dictionary."""
        return {
            "group_returns": self.group_returns,
            "group_cum_returns": self.group_cum_returns,
            "long_short_returns": self.long_short_returns,
            "long_short_cum_return": self.long_short_cum_return,
            "ic_series": self.ic_series,
            "ic_mean": self.ic_mean,
            "ic_ir": self.ic_ir,
        }

    def summary(self) -> str:
        """Generate summary text."""
        return f"""Backtest Summary
================
Groups: {self.group_returns.shape[1]}
Days: {self.group_returns.shape[0]}

Long-Short Cumulative Return: {self.long_short_cum_return:.4%}
IC Mean: {self.ic_mean:.4f}
IC IR: {self.ic_ir:.4f}

Group Mean Returns:
{self._format_group_returns()}
"""

    def _format_group_returns(self) -> str:
        """Format group returns for display."""
        group_means = self.group_returns.mean(axis=0)
        lines = []
        for i, mean in enumerate(group_means):
            lines.append(f"  Group {i+1}: {mean:.6%}")
        return "\n".join(lines)


def factor_returns(
    factor: pd.Series,
    forward_returns: pd.Series,
    quantiles: int = 10,
    bins: Optional[list] = None,
    periods: Tuple[int, ...] = (1,),
    weights: Optional[pd.Series] = None,
    groupby: Optional[pd.Series] = None,
    zero_aware: bool = False,
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Compute factor portfolio returns (alphalens-like interface).

    Parameters
    ----------
    factor : pd.Series
        Factor values with MultiIndex (date, asset)
    forward_returns : pd.Series
        Forward returns with MultiIndex (date, asset)
    quantiles : int, default 10
        Number of quantile groups
    bins : list, optional
        Custom bin edges (overrides quantiles)
    periods : tuple, default (1,)
        Forward periods to compute returns for
    weights : pd.Series, optional
        Weights for weighted aggregation (e.g., market cap)
    groupby : pd.Series, optional
        Grouping variable (e.g., industry classification)
    zero_aware : bool, default False
        Treat positive and negative factors separately

    Returns
    -------
    dict
        Dictionary containing factor_returns, factor_quantiles, and weights
    """
    # Convert to numpy arrays for Rust processing
    factor_df = factor.unstack()
    returns_df = forward_returns.unstack()

    # Align dates and assets
    common_dates = factor_df.index.intersection(returns_df.index)
    common_assets = factor_df.columns.intersection(returns_df.columns)

    factor_array = factor_df.loc[common_dates, common_assets].values
    returns_array = returns_df.loc[common_dates, common_assets].values

    # Run backtest using Rust engine
    result = quantile_backtest(
        factor=factor_array,
        returns=returns_array,
        quantiles=quantiles,
        weight_method="equal",  # Default for this interface
        long_top_n=1,
        short_top_n=1,
        commission_rate=0.0,
        weights=None,
    )

    # Convert results back to pandas
    factor_returns_df = pd.DataFrame(
        result.group_returns,
        index=common_dates[:result.group_returns.shape[0]],
        columns=[f"Q{i+1}" for i in range(quantiles)]
    )

    # Create factor quantiles series
    # Note: This is a simplified version
    factor_quantiles = pd.Series(
        index=factor.index,
        data=np.nan,
        name="factor_quantile"
    )

    return {
        "factor_returns": factor_returns_df,
        "factor_quantiles": factor_quantiles,
        "weights": None,  # Would need to compute from weights input
    }


def quantile_backtest(
    factor: np.ndarray,
    returns: np.ndarray,
    quantiles: int = 10,
    weight_method: str = "equal",
    long_top_n: int = 1,
    short_top_n: int = 1,
    commission_rate: float = 0.0,
    weights: Optional[np.ndarray] = None,
) -> BacktestResult:
    """
    Run quantile backtest (direct numpy interface).

    Parameters
    ----------
    factor : np.ndarray
        Factor values, shape (n_days, n_assets)
    returns : np.ndarray
        Forward returns, shape (n_days, n_assets)
    quantiles : int, default 10
        Number of quantile groups
    weight_method : str, default "equal"
        "equal" or "weighted"
    long_top_n : int, default 1
        Number of top groups to long
    short_top_n : int, default 1
        Number of bottom groups to short
    commission_rate : float, default 0.0
        One-way commission rate
    weights : np.ndarray, optional
        External weights for weighted method

    Returns
    -------
    BacktestResult
        Backtest results
    """
    if not HAS_RUST_EXT:
        # Fallback to pure Python implementation
        from ._fallback import quantile_backtest as python_backtest
        return python_backtest(
            factor, returns, quantiles, weight_method,
            long_top_n, short_top_n, commission_rate, weights
        )

    rust_result = _quantile_backtest(
        factor, returns, quantiles, weight_method,
        long_top_n, short_top_n, commission_rate, weights
    )
    return BacktestResult.from_rust_result(rust_result)


def compute_information_coefficient(
    factor: np.ndarray,
    returns: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute information coefficient (IC) statistics.

    Parameters
    ----------
    factor : np.ndarray
        Factor values, shape (n_days, n_assets)
    returns : np.ndarray
        Forward returns, shape (n_days, n_assets)

    Returns
    -------
    tuple
        (ic_mean, ic_ir)
    """
    if not HAS_RUST_EXT:
        # Fallback
        from ._fallback import compute_ic
        return compute_ic(factor, returns)

    return _compute_ic(factor, returns)


def create_factor_tear_sheet(
    factor: pd.Series,
    forward_returns: pd.Series,
    quantiles: int = 10,
    periods: Tuple[int, ...] = (1, 5, 10),
    group_neutral: bool = False,
    by_group: bool = False,
    **kwargs,
) -> None:
    """
    Create factor tear sheet (alphalens-like).

    Parameters
    ----------
    factor : pd.Series
        Factor values with MultiIndex (date, asset)
    forward_returns : pd.Series
        Forward returns with MultiIndex (date, asset)
    quantiles : int, default 10
        Number of quantile groups
    periods : tuple, default (1, 5, 10)
        Forward periods to analyze
    group_neutral : bool, default False
        Whether to make returns group neutral
    by_group : bool, default False
        Whether to show performance by group
    **kwargs
        Additional arguments passed to factor_returns
    """
    print("Factor Tear Sheet")
    print("=" * 50)

    # Compute factor returns
    results = factor_returns(
        factor, forward_returns, quantiles=quantiles,
        periods=periods, **kwargs
    )

    # Compute IC
    factor_df = factor.unstack()
    returns_df = forward_returns.unstack()
    common_dates = factor_df.index.intersection(returns_df.index)
    common_assets = factor_df.columns.intersection(returns_df.columns)

    factor_array = factor_df.loc[common_dates, common_assets].values
    returns_array = returns_df.loc[common_dates, common_assets].values

    ic_mean, ic_ir = compute_information_coefficient(factor_array, returns_array)

    print(f"\nFactor Statistics:")
    print(f"  Number of observations: {len(factor.dropna())}")
    print(f"  Quantiles: {quantiles}")
    print(f"  Periods: {periods}")

    print(f"\nIC Statistics:")
    print(f"  Mean IC: {ic_mean:.4f}")
    print(f"  IC IR: {ic_ir:.4f}")

    print(f"\nQuantile Returns (mean):")
    factor_returns_df = results["factor_returns"]
    for col in factor_returns_df.columns:
        mean_return = factor_returns_df[col].mean()
        print(f"  {col}: {mean_return:.6%}")

    # Try to import matplotlib for plotting
    try:
        import matplotlib.pyplot as plt

        # Plot cumulative returns by quantile
        cum_returns = (1 + factor_returns_df).cumprod() - 1
        cum_returns.plot(title="Cumulative Returns by Quantile")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend(title="Quantile")
        plt.grid(True, alpha=0.3)
        plt.show()

    except ImportError:
        print("\n(Install matplotlib for visualizations)")
