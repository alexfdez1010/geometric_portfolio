import pandas as pd
import numpy as np
from geometric_portfolio import metrics

def test_arithmetic_mean():
    returns = pd.Series([0.01, 0.02, 0.03, -0.01])
    result = metrics.arithmetic_mean(returns)
    assert np.isclose(result, 0.0125)

def test_geometric_mean():
    returns = pd.Series([0.01, 0.02, 0.03, -0.01])
    result = metrics.geometric_mean(returns)
    expected = np.exp(np.log1p(returns).mean()) - 1
    assert np.isclose(result, expected)

def test_volatility():
    returns = pd.Series([0.01, 0.02, 0.03, -0.01])
    result = metrics.volatility(returns)
    expected = returns.std(ddof=1)
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
