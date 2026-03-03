"""
alpha-expr 基本使用示例
展示如何使用量化回测功能
"""

import numpy as np
import pandas as pd
from alpha_expr import (
    quantile_backtest,
    create_factor_tear_sheet,
    BacktestEngine,
    compute_information_coefficient,
)

print("=" * 60)
print("alpha-expr 示例：量化因子回测")
print("=" * 60)

# 设置随机种子
np.random.seed(42)

# 生成模拟数据
print("\n1. 生成模拟数据...")
n_days = 100
n_assets = 200

# 因子数据（具有自相关性的随机游走）
factor = np.zeros((n_days, n_assets))
factor[0] = np.random.randn(n_assets)
for t in range(1, n_days):
    factor[t] = 0.7 * factor[t-1] + 0.3 * np.random.randn(n_assets)

# 收益率数据（包含因子信号和噪声）
signal_strength = 0.05
returns = np.zeros((n_days, n_assets))
for t in range(n_days):
    returns[t] = signal_strength * factor[t] + np.random.randn(n_assets) * 0.01

# 添加一些缺失值（真实数据常见）
missing_mask = np.random.rand(n_days, n_assets) < 0.1
factor[missing_mask] = np.nan
returns[missing_mask] = np.nan

print(f"  数据形状: {factor.shape} (天数 × 资产数)")
print(f"  有效观测: {np.sum(~np.isnan(factor)):,}")
print(f"  缺失比例: {np.mean(np.isnan(factor)):.1%}")

# 方法1：直接使用 quantile_backtest 函数
print("\n2. 运行十分组回测...")
result = quantile_backtest(
    factor=factor,
    returns=returns,
    quantiles=10,
    weight_method="equal",
    long_top_n=1,
    short_top_n=1,
    commission_rate=0.0003,
)

print("\n回测结果:")
print(f"  多空组合累计收益: {result.long_short_cum_return:+.4%}")
print(f"  IC 均值: {result.ic_mean:+.4f}")
print(f"  IC 信息比率: {result.ic_ir:+.4f}")

# 展示分组收益
print("\n分组平均收益 (1=最低, 10=最高):")
group_means = result.group_returns.mean(axis=0)
for i, mean in enumerate(group_means):
    print(f"  第{i+1}组: {mean:+.6%}")

# 方法2：使用 BacktestEngine（更多控制）
print("\n3. 使用 BacktestEngine 进行高级回测...")
engine = BacktestEngine(
    factor=factor,
    returns=returns,
    quantiles=5,            # 五分组
    weight_method="equal",
    long_top_n=2,           # 做多前2组
    short_top_n=2,          # 做空前2组
    commission_rate=0.0005,
)

engine_result = engine.run()
print(engine_result.summary())

# 方法3：计算信息系数
print("\n4. 计算信息系数...")
ic_mean, ic_ir = compute_information_coefficient(factor, returns)
print(f"  IC 均值: {ic_mean:+.4f}")
print(f"  IC 信息比率: {ic_ir:+.4f}")

# 方法4：使用 Pandas 接口（类似 alphalens）
print("\n5. 使用 Pandas 接口...")
# 创建 MultiIndex 数据
dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
assets = [f"STK{i:04d}" for i in range(n_assets)]

factor_df = pd.DataFrame(factor, index=dates, columns=assets)
returns_df = pd.DataFrame(returns, index=dates, columns=assets)

factor_series = factor_df.stack()
factor_series.index.names = ["date", "asset"]

returns_series = returns_df.stack()
returns_series.index.names = ["date", "asset"]

print(f"  因子数据形状: {factor_series.shape}")
print(f"  收益率数据形状: {returns_series.shape}")

# 创建因子分析报告
print("\n6. 生成因子分析报告...")
try:
    create_factor_tear_sheet(
        factor_series,
        returns_series,
        quantiles=10,
        periods=(1, 5),
        group_neutral=False,
    )
except Exception as e:
    print(f"  注意: {e}")
    print("  如需完整功能，请安装 matplotlib")

# 性能对比
print("\n7. 性能对比（简单基准）...")
import time

# 小规模数据
small_factor = np.random.randn(50, 100)
small_returns = np.random.randn(50, 100) * 0.01

start = time.time()
for _ in range(10):
    _ = quantile_backtest(small_factor, small_returns, quantiles=5)
rust_time = time.time() - start

print(f"  Rust 实现 (10次平均): {rust_time/10:.4f} 秒")

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)