"""
alpha-expr 基本功能测试
"""

import numpy as np
import pytest
from alpha_expr import quantile_backtest, compute_information_coefficient


def test_quantile_backtest_basic():
    """测试基本回测功能"""
    # 创建简单测试数据
    n_days, n_assets = 20, 50
    factor = np.random.randn(n_days, n_assets)
    returns = np.random.randn(n_days, n_assets) * 0.01
    
    # 运行回测
    result = quantile_backtest(
        factor=factor,
        returns=returns,
        quantiles=5,
        weight_method="equal",
        long_top_n=1,
        short_top_n=1,
    )
    
    # 检查结果形状
    assert result.group_returns.shape == (n_days - 1, 5)
    assert result.group_cum_returns.shape == (n_days - 1, 5)
    assert result.long_short_returns.shape == (n_days - 1,)
    assert result.ic_series.shape == (n_days - 1,)
    
    # 检查数据类型
    assert np.isscalar(result.long_short_cum_return)
    assert np.isscalar(result.ic_mean)
    assert np.isscalar(result.ic_ir)


def test_quantile_backtest_with_nan():
    """测试包含缺失值的回测"""
    n_days, n_assets = 30, 40
    factor = np.random.randn(n_days, n_assets)
    returns = np.random.randn(n_days, n_assets) * 0.01
    
    # 添加缺失值
    factor[0, 0] = np.nan
    factor[5, 10:15] = np.nan
    returns[10, 20] = np.nan
    
    result = quantile_backtest(
        factor=factor,
        returns=returns,
        quantiles=4,
    )
    
    # 即使有缺失值，也应该能运行
    assert not np.all(np.isnan(result.group_returns))


def test_quantile_backtest_parameter_validation():
    """测试参数验证"""
    n_days, n_assets = 10, 20
    factor = np.random.randn(n_days, n_assets)
    returns = np.random.randn(n_days, n_assets)
    
    # 无效的分组数
    with pytest.raises(Exception):
        quantile_backtest(factor, returns, quantiles=1)  # 至少需要2组
    
    # 无效的权重方法
    with pytest.raises(Exception):
        quantile_backtest(factor, returns, weight_method="invalid")
    
    # 无效的多空头寸
    with pytest.raises(Exception):
        quantile_backtest(factor, returns, long_top_n=0)
    
    # 无效的佣金率
    with pytest.raises(Exception):
        quantile_backtest(factor, returns, commission_rate=-0.1)


def test_compute_ic():
    """测试信息系数计算"""
    n_days, n_assets = 25, 30
    factor = np.random.randn(n_days, n_assets)
    returns = np.random.randn(n_days, n_assets) * 0.01
    
    ic_mean, ic_ir = compute_information_coefficient(factor, returns)
    
    # IC应该在[-1, 1]范围内
    assert -1 <= ic_mean <= 1
    # IR可以是任何值，但通常不会太大
    assert abs(ic_ir) < 10  # 合理的边界


def test_compute_ic_perfect_correlation():
    """测试完全相关的情况"""
    n_days, n_assets = 10, 15
    factor = np.random.randn(n_days, n_assets)
    
    # 创建完全相关的收益率
    returns = factor * 0.01  # 完美线性关系
    
    ic_mean, ic_ir = compute_information_coefficient(factor, returns)
    
    # 应该接近1（由于浮点误差可能不是精确的1）
    assert abs(ic_mean - 1.0) < 0.01


def test_compute_ic_no_correlation():
    """测试无相关性的情况"""
    n_days, n_assets = 20, 25
    factor = np.random.randn(n_days, n_assets)
    returns = np.random.randn(n_days, n_assets) * 0.01
    
    ic_mean, ic_ir = compute_information_coefficient(factor, returns)
    
    # 随机数据的IC应该接近0
    assert abs(ic_mean) < 0.5


def test_backtest_monotonic_factor():
    """测试单调因子（应该有单调的分组收益）"""
    n_days, n_assets = 15, 100
    
    # 创建单调因子（资产编号本身就是因子值）
    factor = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        factor[:, i] = i  # 资产i的因子值为i
    
    # 创建与因子正相关的收益率
    returns = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        returns[:, i] = i * 0.0001 + np.random.randn(n_days) * 0.001
    
    result = quantile_backtest(factor, returns, quantiles=10)
    
    # 分组平均收益应该大致单调递增
    group_means = result.group_returns.mean(axis=0)
    
    # 检查单调性（允许小的波动）
    for i in range(1, len(group_means)):
        # 后面的组应该比前面的组收益高（或至少不低很多）
        assert group_means[i] >= group_means[i-1] - 0.0001


def test_commission_impact():
    """测试佣金对收益的影响"""
    n_days, n_assets = 20, 50
    factor = np.random.randn(n_days, n_assets)
    returns = np.random.randn(n_days, n_assets) * 0.01
    
    # 无佣金回测
    result_no_commission = quantile_backtest(
        factor, returns, commission_rate=0.0
    )
    
    # 有佣金回测
    result_with_commission = quantile_backtest(
        factor, returns, commission_rate=0.001
    )
    
    # 有佣金的累计收益应该更低（或相等）
    assert (result_with_commission.long_short_cum_return <= 
            result_no_commission.long_short_cum_return + 1e-10)


def test_different_quantiles():
    """测试不同分组数量的影响"""
    n_days, n_assets = 25, 60
    factor = np.random.randn(n_days, n_assets)
    returns = np.random.randn(n_days, n_assets) * 0.01
    
    results = {}
    for quantiles in [2, 5, 10]:
        result = quantile_backtest(factor, returns, quantiles=quantiles)
        results[quantiles] = result
        
        # 检查结果形状
        assert result.group_returns.shape[1] == quantiles
    
    # 不同分组数的IC应该相似（因为使用相同数据）
    ic_values = [results[q].ic_mean for q in [2, 5, 10]]
    ic_std = np.std(ic_values)
    assert ic_std < 0.2  # 不应该差异太大


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])