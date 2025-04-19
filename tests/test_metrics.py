import pandas as pd
import numpy as np
from geometric_portfolio import metrics
import pytest

def test_arithmetic_mean():
    returns = pd.Series([0.01, 0.02, 0.03, -0.01])
    periods_per_year = 252
    result = metrics.arithmetic_mean(returns, periods_per_year=periods_per_year)
    expected = (1 + returns.mean()) ** periods_per_year - 1
    assert np.isclose(result, expected)

def test_geometric_mean():
    returns = pd.Series([0.01, 0.02, 0.03, -0.01])
    periods_per_year = 252
    result = metrics.geometric_mean(returns, periods_per_year=periods_per_year)
    expected = np.prod(1 + returns) ** (periods_per_year / len(returns)) - 1
    assert np.isclose(result, expected)

def test_volatility():
    returns = pd.Series([0.01, 0.02, 0.03, -0.01])
    periods_per_year = 252
    result = metrics.volatility(returns, periods_per_year=periods_per_year)
    expected = returns.std(ddof=1) * np.sqrt(periods_per_year)
    assert np.isclose(result, expected)

def test_max_drawdown():
    returns = pd.Series([0.01, -0.02, 0.01, -0.05, 0.02])
    result = metrics.max_drawdown(returns)
    # Compute expected drawdown manually
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    expected = drawdown.min()
    assert np.isclose(result, expected)

def test_sharpe_ratio_zero_vol():
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    assert np.isnan(metrics.sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252))

def test_sharpe_ratio_normal():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    excess = returns - 0.0/252
    ann_excess = excess.mean() * 252
    ann_vol = excess.std(ddof=1) * np.sqrt(252)
    expected = ann_excess / ann_vol
    assert np.isclose(metrics.sharpe_ratio(returns), expected)

def test_best_and_worst_day():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    assert metrics.best_day(returns) == 0.03
    assert metrics.worst_day(returns) == -0.02

def test_best_and_worst_year():
    dates = pd.to_datetime(['2020-01-01','2020-01-02','2021-01-01','2021-01-02'])
    rets = pd.Series([0.01,0.02,-0.01,-0.02], index=dates)
    # 2020: (1.01*1.02)-1 = 0.0302; 2021: (0.99*0.98)-1 = -0.0298
    assert pytest.approx(metrics.best_year(rets), rel=1e-5) == 0.0302
    assert pytest.approx(metrics.worst_year(rets), rel=1e-5) == -0.0298

def test_wealth_and_summary():
    dates = pd.date_range('2020-01-01', periods=3, freq='D')
    returns = pd.Series([0.1, 0.2, -0.1], index=dates)
    # test wealth
    wealth = metrics.wealth(returns, initial_wealth=1.0)
    expected_wealth = pd.Series([1.1, 1.1*1.2, 1.1*1.2*0.9], index=dates)
    assert np.allclose(wealth.values, expected_wealth.values)
    # test summary matches individual metrics
    summary = metrics.summary(returns, risk_free_rate=0.0, periods_per_year=252)
    expected = {
        'Arithmetic Mean': metrics.arithmetic_mean(returns, 252),
        'Geometric Mean': metrics.geometric_mean(returns, 252),
        'Volatility': metrics.volatility(returns, 252),
        'Sharpe Ratio': metrics.sharpe_ratio(returns, 0.0, 252),
        'Max Drawdown': metrics.max_drawdown(returns),
        'Best Day': metrics.best_day(returns),
        'Worst Day': metrics.worst_day(returns),
        'Best Year': metrics.best_year(returns),
        'Worst Year': metrics.worst_year(returns),
    }
    for k, v in expected.items():
        assert np.isclose(summary[k], v)
