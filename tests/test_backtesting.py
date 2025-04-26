import pandas as pd
import numpy as np
from geometric_portfolio.backtesting import backtesting


def test_backtesting_basic():
    # Select three well-known tickers
    tickers = ["AAPL", "MSFT", "GOOG"]
    # initial weights that sum to 1
    weights = {"AAPL": 0.3, "MSFT": 0.4, "GOOG": 0.3}
    start = "2020-01-01"
    end = "2025-01-01"
    # set acceptable_diff large to avoid rebalancing in this short period
    acceptable_diff = 0.05

    # Run backtest
    returns, weight_hist = backtesting(
        10000,
        tickers,
        weights,
        start,
        end,
        acceptable_diff,
        fixed_cost=0.01,
        variable_cost=0.001,
    )

    # Assert types
    assert isinstance(returns, pd.Series)
    assert isinstance(weight_hist, pd.DataFrame)

    # Lengths match
    assert len(returns) == weight_hist.shape[0]

    # Weights sum to 1 each day
    sums = weight_hist.sum(axis=1)
    assert np.allclose(sums.values, 1.0, atol=1e-6)

    # Returns should not be all zeros
    assert not np.allclose(returns.values, 0.0)

    # There should be at least one return (few trading days)
    assert len(returns) > 0
