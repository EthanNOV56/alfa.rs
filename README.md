# alpha-expr

高性能因子表达式与回测框架，核心使用 Rust 实现，现已升级至 **v0.2.0**，新增智能因子挖掘与学习能力。

## 特性

### 核心回测功能
- **高性能**: 核心算法使用 Rust 实现，支持并行计算（8-10倍加速）
- **灵活接口**: 同时支持 NumPy 数组和 Pandas Series 输入
- **完备功能**: qcut(N) 分组、多空组合、IC 计算、因子分析
- **兼容性**: 类似 alphalens 的 API 设计，易于迁移
- **可扩展**: 模块化设计，支持自定义权重、分组、佣金模型

### 新增智能功能（v0.2.0）
- **表达式系统**: 支持构建复杂的数学表达式树，用于自定义因子计算
- **惰性求值**: Polars 风格的延迟计算引擎，支持查询优化和高效执行
- **遗传规划因子挖掘**: 使用遗传算法自动发现高绩效因子表达式
- **持久化存储**: 完整的因子库管理，支持搜索、缓存和版本控制
- **元学习系统**: 基于历史数据分析的智能推荐，优化后续因子挖掘参数
- **完整Python绑定**: 所有功能通过简洁的Python API暴露，类型安全且高效

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

## 智能因子挖掘 (v0.2.0)

### 表达式系统

```python
from alpha_expr import Expr, LazyFrame, DataFrame, Series

# 创建自定义因子表达式
expr = (Expr.col("close") - Expr.col("open")) / Expr.col("open")
sqrt_expr = expr.abs().sqrt()
log_expr = (Expr.col("volume") + 1.0).log()

print(f"Expression: {expr}")
print(f"Sqrt expression: {sqrt_expr}")

# 在数据上评估表达式
data = {
    'open': np.random.randn(100, 50),
    'close': np.random.randn(100, 50),
    'volume': np.random.randn(100, 50),
}

# 使用惰性求值
lf = LazyFrame.scan(data)
lf_with_factor = lf.with_columns([("custom_factor", expr)])
result = lf_with_factor.collect()
print(f"Computed factor shape: {result['custom_factor'].shape}")
```

### 遗传规划因子挖掘

```python
from alpha_expr import GpEngine

# 创建GP引擎
gp = GpEngine(
    population_size=100,
    max_generations=50,
    tournament_size=7,
    crossover_prob=0.8,
    mutation_prob=0.2,
    max_depth=6,
)

# 设置可用数据列
gp.set_columns(['open', 'high', 'low', 'close', 'volume'])

# 准备数据
data = {
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices,
    'volume': volumes,
}
returns = next_day_returns

# 挖掘因子
factors = gp.mine_factors(data, returns, num_factors=10)

print(f"Discovered {len(factors)} factors:")
for i, (expr_str, fitness) in enumerate(factors[:3]):
    print(f"  Factor {i+1}: {expr_str[:80]}... (fitness: {fitness:.4f})")
```

### 持久化存储与因子库管理

```python
from alpha_expr import PersistenceManager, FactorMetadata

# 创建因子库
db = PersistenceManager("./factor_library")

# 保存因子
factor_metadata = FactorMetadata(
    id="momentum_factor_001",
    expression="close - lag(close, 20)",
    metrics={
        "ic_mean": 0.15,
        "ic_ir": 2.5,
        "turnover": 0.12,
        "complexity_penalty": 3.2,
        "combined_score": 0.42,
    },
    tags=["momentum", "20-day"]
)
db.save_factor(factor_metadata)

# 搜索因子
high_ic_factors = db.search_factors(min_ic=0.1, max_complexity=5.0, tags=["momentum"])
print(f"Found {len(high_ic_factors)} high-IC momentum factors")

# 批量加载
num_factors = db.load_all_factors()
num_history = db.load_all_history()
print(f"Loaded {num_factors} factors and {num_history} GP history records")

# 获取缓存统计
stats = db.cache_stats()
print(f"Cache stats: {dict(stats)}")
```

### 元学习智能推荐

```python
from alpha_expr import MetaLearningAnalyzer, GPRecommendations

# 创建元学习分析器
analyzer = MetaLearningAnalyzer()

# 配置训练参数（可根据数据量调整）
analyzer.set_min_data_points(50)          # 最小训练数据量
analyzer.set_high_perf_threshold(0.08)    # 高绩效阈值 (IC > 0.08)

# 训练模型
all_factors = db.get_all_factors()
all_history = db.get_all_history()
analyzer.train(all_factors, all_history)

print(f"Model trained: {analyzer.is_trained()}")
print(f"Model version: {analyzer.version()}")
print(f"Recommendations confidence: {analyzer.confidence_score():.2f}")

# 获取智能推荐
recommendations = analyzer.get_recommendations(target_complexity=4.5)
print(f"Recommended functions: {recommendations.recommended_functions}")
print(f"Recommended terminals: {recommendations.recommended_terminals}")
print(f"Confidence level: {recommendations.confidence_level()}")

# 转换为GP配置
gp_config = recommendations.to_gp_config()
print(f"Recommended population size: {gp_config['population_size']}")
print(f"Recommended max generations: {gp_config['max_generations']}")

# 保存/加载模型
analyzer.save_model("./meta_learning_model.json")
loaded_analyzer = MetaLearningAnalyzer.load_model("./meta_learning_model.json")
```

### 完整工作流示例

```python
# 1. 数据准备
data = prepare_market_data(start_date="2020-01-01", end_date="2023-12-31")
returns = compute_forward_returns(data['close'])

# 2. 初始GP因子挖掘
gp = GpEngine.from_recommendations(recommendations)  # 使用智能推荐配置
initial_factors = gp.mine_factors(data, returns, num_factors=100)

# 3. 回测验证与存储
for expr_str, fitness in initial_factors:
    factor_array = evaluate_expression(expr_str, data)
    result = quantile_backtest(factor_array, returns)
    
    factor_metadata = FactorMetadata(
        id=f"gp_factor_{hash(expr_str)}",
        expression=expr_str,
        metrics={
            "ic_mean": result.ic_mean,
            "ic_ir": result.ic_ir,
            "turnover": compute_turnover(factor_array),
            "complexity_penalty": compute_complexity(expr_str),
            "combined_score": fitness,
        }
    )
    db.save_factor(factor_metadata)

# 4. 元学习优化
analyzer.train(db.get_all_factors(), db.get_all_history())
new_recommendations = analyzer.get_recommendations()

# 5. 迭代改进（越用越聪明）
optimized_gp = GpEngine.from_recommendations(new_recommendations)
improved_factors = optimized_gp.mine_factors(data, returns, num_factors=50)
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

### 表达式系统 (v0.2.0)

#### `Expr` 类
因子表达式构建器，支持链式调用。

```python
# 创建表达式
expr = Expr.col("close")                    # 列引用
expr = Expr.lit_float(1.0)                  # 常量
expr = Expr.lit_int(5)                      # 整数常量

# 算术运算
expr1 = expr1.add(expr2)                    # 加法
expr1 = expr1.sub(expr2)                    # 减法  
expr1 = expr1.mul(expr2)                    # 乘法
expr1 = expr1.div(expr2)                    # 除法

# 数学函数
expr = expr.abs()                           # 绝对值
expr = expr.sqrt()                          # 平方根
expr = expr.log()                           # 自然对数
expr = expr.exp()                           # 指数
expr = expr.neg()                           # 取负

# 比较运算
expr = expr.gt(Expr.lit_float(0))           # 大于
expr = expr.lt(Expr.lit_float(0))           # 小于
expr = expr.eq(Expr.lit_float(0))           # 等于
expr = expr.ne(Expr.lit_float(0))           # 不等于

# 时间序列函数
expr = expr.lag(1)                          # 滞后
expr = expr.diff(1)                         # 差分
expr = expr.rolling_mean(20)                # 滚动均值
expr = expr.rolling_std(20)                 # 滚动标准差
expr = expr.cumsum()                        # 累积和
expr = expr.cumprod()                       # 累积积
```

#### `LazyFrame` 类
惰性求值框架，支持延迟计算和查询优化。

```python
# 创建 LazyFrame
lf = LazyFrame.scan(data_dict)              # 从字典创建

# 添加列
lf = lf.with_columns([("new_col", expr)])   # 添加新列

# 连接操作
lf = lf.join(other_lf, ["date"], "inner")   # 内连接

# 执行计算
result = lf.collect()                       # 执行并返回结果

# 查看执行计划
plan = lf.explain(optimized=False)          # 逻辑计划
plan = lf.explain(optimized=True)           # 优化后计划
```

#### `Series` 和 `DataFrame` 类
基础数据结构，用于表达式求值。

```python
series = Series.from_numpy(array)           # 从numpy数组创建
dataframe = DataFrame.from_dict(data_dict)  # 从字典创建
```

### 遗传规划模块 (v0.2.0)

#### `GpEngine` 类
遗传规划因子挖掘引擎。

```python
# 创建GP引擎
gp = GpEngine(
    population_size=100,     # 种群大小
    max_generations=50,      # 最大代数
    tournament_size=7,       # 锦标赛大小
    crossover_prob=0.8,      # 交叉概率
    mutation_prob=0.2,       # 变异概率
    max_depth=6,             # 最大深度
)

# 配置
gp.set_columns(['open', 'high', 'low', 'close', 'volume'])  # 可用变量

# 运行因子挖掘
factors = gp.mine_factors(
    data_dict,               # 数据字典 {column: numpy_array}
    returns_array,           # 收益率数组
    num_factors=10,          # 挖掘因子数量
) -> List[Tuple[str, float]] # (表达式字符串, 适应度)

# 测试运行
test_result = gp.test_run()  # 简单测试运行
```

### 持久化存储模块 (v0.2.0)

#### `PersistenceManager` 类
因子库管理器。

```python
# 创建管理器
pm = PersistenceManager("./factor_db")

# 因子管理
pm.save_factor(factor_metadata)            # 保存因子
factor = pm.load_factor("factor_id")       # 加载因子
pm.clear_memory()                          # 清空内存缓存

# 批量操作
num_factors = pm.load_all_factors()        # 加载所有因子
num_history = pm.load_all_history()        # 加载所有历史记录
all_factors = pm.get_all_factors()         # 获取所有因子
all_history = pm.get_all_history()         # 获取所有历史记录

# 搜索功能
results = pm.search_factors(
    min_ic=0.1,             # 最小IC阈值
    max_complexity=5.0,     # 最大复杂度
    tags=["momentum"],      # 标签筛选
)

# 缓存统计
stats = pm.cache_stats()    # 获取缓存统计信息
```

#### `FactorMetadata` 类
因子元数据容器。

```python
metadata = FactorMetadata(
    id="unique_id",                      # 唯一标识
    expression="close - open",           # 表达式字符串
    metrics={                            # 性能指标
        "ic_mean": 0.15,
        "ic_ir": 2.5,
        "turnover": 0.12,
        "complexity_penalty": 3.2,
        "combined_score": 0.42,
    },
    tags=["momentum", "short_term"],    # 标签
)

# 属性访问
metadata.id                              # 获取ID
metadata.expression                      # 获取表达式
metadata.metrics                         # 获取指标字典
metadata.tags                            # 获取标签列表
```

#### `GPHistoryRecord` 类
GP历史记录容器。

```python
record = GPHistoryRecord(
    run_id="run_001",                    # 运行ID
    best_factor=factor_metadata,         # 最佳因子
    config=gp_config_dict,               # GP配置
)

# 属性访问
record.run_id                            # 运行ID
record.best_factor                       # 最佳因子
record.config                            # 配置字典
```

### 元学习模块 (v0.2.0)

#### `MetaLearningAnalyzer` 类
元学习分析器。

```python
# 创建分析器
analyzer = MetaLearningAnalyzer()

# 配置
analyzer.set_high_perf_threshold(0.08)   # 高绩效阈值
analyzer.set_min_data_points(50)         # 最小训练数据量

# 训练
analyzer.train(factors_list, gp_history_list)

# 获取信息
is_trained = analyzer.is_trained()       # 是否已训练
version = analyzer.version()             # 模型版本
confidence = analyzer.confidence_score() # 推荐置信度

# 获取推荐
recommendations = analyzer.get_recommendations(target_complexity=4.5)

# 模型持久化
analyzer.save_model("./model.json")      # 保存模型
loaded = MetaLearningAnalyzer.load_model("./model.json")  # 加载模型
```

#### `GPRecommendations` 类
GP配置推荐。

```python
# 属性访问
recs.recommended_functions               # 推荐函数列表
recs.recommended_terminals               # 推荐终端列表
recs.target_complexity                   # 目标复杂度
recs.confidence_score                    # 置信度分数 (0.0-1.0)
recs.confidence_level()                  # 置信度等级 ("high"/"medium"/"low")

# 转换方法
gp_config = recs.to_gp_config()          # 转换为GP配置字典
is_valid = recs.is_valid()               # 检查推荐是否有效
```

### 表达式函数 (v0.2.0)

#### `evaluate_expression()`
在数据上评估表达式。

```python
result = evaluate_expression(
    expr_str,                            # 表达式字符串
    data_dict,                           # 数据字典
) -> np.ndarray                         # 计算结果数组
```

#### 窗口函数
```python
window_spec = rolling_window(20, min_periods=5)   # 滚动窗口
window_spec = expanding_window(min_periods=10)    # 扩展窗口
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

### 新模块性能特点 (v0.2.0)

| 模块 | 性能特点 | 优化技术 |
|------|----------|----------|
| **表达式系统** | 零拷贝求值，无中间数组分配 | 表达式树优化，常量折叠 |
| **惰性求值** | 延迟计算，最小化内存使用 | 查询优化，谓词下推 |
| **遗传规划** | 并行种群评估，8核线性加速 | 缓存评估结果，批量处理 |
| **持久化存储** | 内存映射文件，快速序列化 | LRU缓存，智能预取 |
| **元学习** | 增量学习，无需全量重训练 | 特征缓存，近似计算 |

### 内存效率
- **惰性求值**: 减少60-80%的中间内存使用
- **表达式缓存**: 相同表达式复用计算结果
- **分批处理**: 大数据集分块处理，避免OOM

## 开发指南

### 项目结构 (v0.2.0)

```
alpha-expr/
├── Cargo.toml                    # Rust 项目配置
├── pyproject.toml                # Python 项目配置
├── src/
│   ├── lib.rs                    # Rust 核心实现和Python绑定
│   ├── backtest.rs               # 回测引擎实现
│   ├── expr.rs                   # 表达式系统
│   ├── lazy.rs                   # 惰性求值引擎
│   ├── gp.rs                     # 遗传规划模块
│   ├── persistence.rs            # 持久化存储模块
│   ├── metalearning.rs           # 元学习系统
│   └── ga.rs                     # 遗传算法基础
├── alpha_expr/
│   ├── __init__.py               # Python 主模块
│   ├── _core.cpython-*.so        # Rust 扩展模块
│   └── _fallback.py              # Python 回退实现
├── examples/
│   ├── basic_backtest.py         # 基础回测示例
│   ├── expression_system.py      # 表达式系统示例
│   ├── lazy_evaluation.py        # 惰性求值示例
│   ├── gp_factor_mining.py       # GP因子挖掘示例
│   ├── persistence_demo.py       # 持久化存储示例
│   ├── metalearning_workflow.py  # 元学习工作流示例
│   └── full_workflow_example.py  # 完整工作流示例
├── tests/                        # 测试文件
├── benchmarks/                   # 性能测试
└── memory/                       # 运行时记忆文件
```

### 模块架构

```
alpha-expr v0.2.0 架构
├── 核心层 (Rust)
│   ├── 回测引擎 (backtest)      # 分位数分组、多空组合、IC计算
│   ├── 表达式系统 (expr)        # 表达式树构建、求值、优化
│   ├── 惰性求值 (lazy)          # 逻辑计划、物理计划、查询优化
│   ├── 遗传规划 (gp)            # 因子挖掘、多目标优化、缓存
│   ├── 持久化存储 (persistence) # 因子库、GP历史、表达式缓存
│   └── 元学习 (metalearning)    # 历史分析、智能推荐、模型持久化
├── 绑定层 (PyO3)
│   ├── Python类型映射           # Rust-Python类型转换
│   ├── API暴露                 # 所有功能通过Python类暴露
│   └── 错误处理                # 统一的异常处理
└── 应用层 (Python)
    ├── 简洁API                 # Pythonic接口设计
    ├── 工作流集成              # 完整因子挖掘流水线
    └── 生态系统集成            # numpy/pandas兼容
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

## 版本历史

### v0.2.0 (当前版本) - 智能因子挖掘
- **新增表达式系统**: 完整的数学表达式构建与求值
- **新增惰性求值引擎**: Polars风格的延迟计算与查询优化
- **新增遗传规划因子挖掘**: 自动发现高绩效因子表达式
- **新增持久化存储系统**: 因子库管理、搜索、缓存
- **新增元学习系统**: 基于历史数据的智能推荐
- **完整Python绑定**: 所有新功能通过简洁API暴露
- **性能优化**: 8-10倍加速，内存高效设计
- **向后兼容**: 保持v0.1.0所有API不变

### v0.1.0 - 基础回测框架
- 高性能分位数分组回测
- 类似alphalens的API设计
- NumPy/Pandas双接口支持
- 多空组合构建与IC计算
- 基础性能基准测试

## 许可证

MIT License

## 致谢

### 基础框架
- [alphalens](https://github.com/quantopian/alphalens) - 回测API设计参考
- [PyO3](https://github.com/PyO3/pyo3) - Rust-Python绑定框架
- [ndarray](https://github.com/rust-ndarray/ndarray) - Rust高性能数组计算

### 新功能依赖 (v0.2.0)
- [rayon](https://github.com/rayon-rs/rayon) - Rust数据并行库
- [serde](https://github.com/serde-rs/serde) - 序列化/反序列化框架
- [rand](https://github.com/rust-random/rand) - 随机数生成
- [lru](https://github.com/jeromefroe/lru-rs) - LRU缓存实现
- [sha2](https://github.com/RustCrypto/hashes) - 哈希算法库

### 设计灵感
- [Polars](https://github.com/pola-rs/polars) - 惰性求值设计思想
- [DEAP](https://github.com/DEAP/deap) - 进化计算框架
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) - 机器学习API设计

### 社区支持
- [Rust社区](https://www.rust-lang.org/) - 优秀的语言和工具链
- [Python量化社区](https://github.com/topics/quantitative-finance) - 丰富的量化金融资源