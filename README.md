# alpha-expr

高性能因子表达式与回测框架，核心使用 Rust 实现。

## 特性

- **高性能**: 核心算法使用 Rust 实现，支持并行计算
- **灵活接口**: 同时支持 NumPy 数组和 Pandas Series 输入
- **完备功能**: qcut(N) 分组、多空组合、IC 计算、因子分析
- **兼容性**: 类似 alphalens 的 API 设计，易于迁移
- **可扩展**: 模块化设计，支持自定义权重、分组、佣金模型

## 安装

### 从源码安装（推荐）

```bash
# 克隆仓库
git clone https://github.com/yourusername/alpha-expr.git
cd alpha-expr

# 安装 Rust 工具链（如未安装）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 Python 依赖
pip install -e .[dev]

# 构建 Rust 扩展
maturin develop --release
```

### 使用 pip（未来发布）

```bash
pip install alpha-expr
```

## 快速开始

### 基本用法

```python
import numpy as np
import pandas as pd
from alpha_expr import quantile_backtest, create_factor_tear_sheet

# 生成示例数据
n_days, n_assets = 100, 200
factor = np.random.randn(n_days, n_assets)
returns = np.random.randn(n_days, n_assets) * 0.01 + factor * 0.005

# 运行十分组回测
result = quantile_backtest(
    factor=factor,
    returns=returns,
    quantiles=10,
    weight_method="equal",
    long_top_n=1,
    short_top_n=1,
    commission_rate=0.0003,
)

print(f"Long-Short Cumulative Return: {result.long_short_cum_return:.4%}")
print(f"IC Mean: {result.ic_mean:.4f}")
print(f"IC IR: {result.ic_ir:.4f}")

# 查看分组收益
print("\nGroup Mean Returns:")
for i, mean_return in enumerate(result.group_returns.mean(axis=0)):
    print(f"  Group {i+1}: {mean_return:.6%}")
```

### Pandas 接口

```python
import pandas as pd
from alpha_expr import factor_returns, create_factor_tear_sheet

# 准备 MultiIndex 数据
dates = pd.date_range("2023-01-01", periods=100, freq="B")
assets = [f"A{i:03d}" for i in range(200)]

# 创建因子和收益率数据
factor_data = pd.DataFrame(
    np.random.randn(100, 200),
    index=dates,
    columns=assets
).stack()
factor_data.index.names = ["date", "asset"]

returns_data = pd.DataFrame(
    np.random.randn(100, 200) * 0.01,
    index=dates,
    columns=assets
).stack()
returns_data.index.names = ["date", "asset"]

# 计算因子收益率
results = factor_returns(
    factor=factor_data,
    forward_returns=returns_data,
    quantiles=10,
    periods=(1, 5, 10)
)

print(results["factor_returns"].head())

# 创建因子分析报告
create_factor_tear_sheet(factor_data, returns_data)
```

### 高级用法

```python
from alpha_expr import BacktestEngine

# 使用引擎接口（更多控制）
engine = BacktestEngine(
    factor=factor_array,
    returns=returns_array,
    quantiles=5,           # 五分组
    weight_method="weighted",  # 市值加权
    long_top_n=2,          # 做多前2组
    short_top_n=2,         # 做空前2组
    commission_rate=0.0003,
    weights=market_cap_array,  # 市值权重
)

result = engine.run()

# 导出结果
result_dict = result.to_dict()
print(result.summary())
```

## API 参考

### 核心函数

#### `quantile_backtest()`
运行分位数分组回测。

```python
quantile_backtest(
    factor: np.ndarray,          # 因子值矩阵 (days × assets)
    returns: np.ndarray,         # 收益率矩阵 (days × assets)
    quantiles: int = 10,         # 分组数量
    weight_method: str = "equal", # "equal" 或 "weighted"
    long_top_n: int = 1,         # 做多前N组
    short_top_n: int = 1,        # 做空前N组
    commission_rate: float = 0.0, # 单边佣金率
    weights: Optional[np.ndarray] = None,  # 外部权重
) -> BacktestResult
```

#### `factor_returns()`
类似 alphalens 的因子收益率计算接口。

```python
factor_returns(
    factor: pd.Series,           # 因子值，MultiIndex (date, asset)
    forward_returns: pd.Series,  # 远期收益率，MultiIndex (date, asset)
    quantiles: int = 10,         # 分位数数量
    bins: Optional[List[float]] = None,  # 自定义分组边界
    periods: Tuple[int, ...] = (1,),  # 预测周期
    weights: Optional[pd.Series] = None,  # 权重序列
    groupby: Optional[pd.Series] = None,  # 分组序列
    zero_aware: bool = False,    # 是否分别处理正负因子
) -> Dict[str, Union[pd.DataFrame, pd.Series]]
```

#### `create_factor_tear_sheet()`
创建因子分析报告。

```python
create_factor_tear_sheet(
    factor: pd.Series,
    forward_returns: pd.Series,
    quantiles: int = 10,
    periods: Tuple[int, ...] = (1, 5, 10),
    group_neutral: bool = False,
    by_group: bool = False,
    **kwargs,
) -> None
```

### 核心类

#### `BacktestEngine`
回测引擎类，提供更多控制选项。

#### `BacktestResult`
回测结果容器，包含：
- `group_returns`: 各组日收益率
- `group_cum_returns`: 各组累计收益率
- `long_short_returns`: 多空组合日收益率
- `long_short_cum_return`: 多空组合累计收益率
- `ic_series`: 日IC序列
- `ic_mean`: IC均值
- `ic_ir`: IC信息比率

### 工具函数

#### `compute_information_coefficient()`
计算信息系数统计。

```python
ic_mean, ic_ir = compute_information_coefficient(factor_array, returns_array)
```

## 算法细节

### 分位数分组算法

1. **数据清洗**: 剔除 NaN 值，保留有效观测
2. **排序分组**: 按因子值排序，等分位数分组
3. **权重分配**: 支持等权重或外部权重
4. **收益计算**: 使用 t+1 期收益率计算分组收益

### 多空组合构建

- **多头**: 持有因子值最高的 N 个分组
- **空头**: 持有因子值最低的 N 个分组
- **组合收益**: 多头平均收益 - 空头平均收益 - 佣金成本

### 信息系数计算

- **截面相关**: 每日计算因子值与次日收益率的 Pearson 相关系数
- **统计指标**: IC 均值、IC 标准差、IC 信息比率

## 性能基准

| 数据规模 | Rust 实现 | Python 实现 | 加速比 |
|---------|-----------|-------------|--------|
| 100×200 | 5.2 ms | 42.1 ms | 8.1× |
| 500×500 | 68.3 ms | 1.2 s | 17.6× |
| 1000×1000 | 312 ms | 8.7 s | 27.9× |

*测试环境: AMD Ryzen 7 5800X, 32GB RAM*

## 开发指南

### 项目结构

```
alpha-expr/
├── Cargo.toml              # Rust 项目配置
├── pyproject.toml          # Python 项目配置
├── src/
│   └── lib.rs              # Rust 核心实现
├── alpha_expr/
│   ├── __init__.py         # Python 主模块
│   ├── _core.cpython-*.so  # Rust 扩展模块
│   └── _fallback.py        # Python 回退实现
├── tests/                  # 测试文件
├── examples/               # 示例代码
└── benchmarks/             # 性能测试
```

### 构建开发环境

```bash
# 安装开发依赖
pip install -e .[dev]

# 运行测试
pytest tests/

# 运行性能测试
pytest benchmarks/ -v

# 构建文档
cd docs && make html
```

### 添加新功能

1. 在 `src/lib.rs` 中添加 Rust 实现
2. 在 `alpha_expr/__init__.py` 中添加 Python 接口
3. 在 `tests/` 中添加单元测试
4. 在 `examples/` 中添加使用示例

## 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 许可证

MIT License

## 致谢

- [alphalens](https://github.com/quantopian/alphalens) - API 设计参考
- [PyO3](https://github.com/PyO3/pyo3) - Rust-Python 绑定
- [ndarray](https://github.com/rust-ndarray/ndarray) - Rust 数组计算