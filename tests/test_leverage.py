import numpy as np
import pandas as pd
import pytest

from geometric_portfolio.leverage import leverage_optimizer
from geometric_portfolio.metrics import geometric_mean

# TODO: Add test cases for leverage_optimizer


@pytest.fixture
def sample_returns():
    """Generate a simple sample daily return series."""
    # Consistent positive return for simplicity
    return pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])


def test_leverage_optimizer_zero_rf_l1(sample_returns):
    """Test case: Risk-free rate = 0, Leverage = 1."""
    leverage = 1.0
    result_df = leverage_optimizer(
        sample_returns,
        maximum_leverage=leverage,
        risk_free_rate=0.0,
    )
    # Since min_leverage == max_leverage, all rows have the same leverage.
    # Check results. Leverage=0 is filtered out as GM=0.
    assert len(result_df) == 999  # Linspace(0,1,1000) but leverage=0 filtered out
    # Find row corresponding to target leverage = 1.0
    result_row = result_df[np.isclose(result_df["Leverage"], leverage)].iloc[0]
    expected_gm = geometric_mean(sample_returns)
    assert np.isclose(result_row["Geometric Mean"], expected_gm)
    assert np.isclose(result_row["Leverage"], leverage)


def test_leverage_optimizer_positive_rf_l1(sample_returns):
    """Test case: Positive risk-free rate, Leverage = 1."""
    rf_rate = 0.02  # 2% annual
    leverage = 1.0
    result_df = leverage_optimizer(
        sample_returns,
        maximum_leverage=leverage,
        risk_free_rate=rf_rate,
    )
    # Since min_leverage == max_leverage, all rows have the same leverage.
    # Check results for target leverage = 1.0
    assert len(result_df) == 1000
    result_row = result_df[np.isclose(result_df["Leverage"], leverage)].iloc[0]
    # For L=1, leveraged returns = asset returns
    expected_gm = geometric_mean(sample_returns)
    # Leverage = 1 means the return should still be the asset's return
    assert np.isclose(result_row["Geometric Mean"], expected_gm)
    assert np.isclose(result_row["Leverage"], leverage)


def test_leverage_optimizer_positive_rf_l0(sample_returns):
    """Test case: Positive risk-free rate, Leverage = 0."""
    rf_rate = 0.03  # 3% annual
    leverage = 0.0
    result_df = leverage_optimizer(
        sample_returns,
        maximum_leverage=leverage,
        risk_free_rate=rf_rate,
    )
    # Since min_leverage == max_leverage, all rows have the same leverage.
    # Check results for target leverage = 0.0
    assert len(result_df) == 1000
    result_row = result_df[np.isclose(result_df["Leverage"], leverage)].iloc[0]

    # Leverage = 0 means the return should be the risk-free rate
    daily_rf = (1 + rf_rate) ** (1 / 252) - 1
    expected_returns_rf = pd.Series([daily_rf] * len(sample_returns))
    expected_gm = geometric_mean(expected_returns_rf)
    assert np.isclose(result_row["Geometric Mean"], expected_gm)
    assert np.isclose(result_row["Leverage"], leverage)


def test_leverage_optimizer_positive_rf_fractional_l(sample_returns):
    """Test case: Positive risk-free rate, 0 < Leverage < 1."""
    rf_rate = 0.01  # 1% annual
    leverage = 0.5
    result_df = leverage_optimizer(
        sample_returns,
        maximum_leverage=leverage,
        risk_free_rate=rf_rate,
    )
    # Since min_leverage == max_leverage, all rows have the same leverage.
    # Check results for target leverage = 0.5
    assert len(result_df) == 1000
    result_row = result_df[np.isclose(result_df["Leverage"], leverage)].iloc[0]

    # Expected return: R_f + L * (R_asset - R_f)
    daily_rf = (1 + rf_rate) ** (1 / 252) - 1
    expected_leveraged_returns = daily_rf + leverage * (sample_returns - daily_rf)
    expected_gm = geometric_mean(expected_leveraged_returns)

    assert np.isclose(result_row["Geometric Mean"], expected_gm)
    assert np.isclose(result_row["Leverage"], leverage)
