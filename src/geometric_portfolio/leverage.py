import pandas as pd
import numpy as np
from geometric_portfolio.metrics import geometric_mean, volatility


def leverage_optimizer(
    returns: pd.Series,
    maximum_leverage: float = 1.0,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Optimize leverage based on geometric mean maximization.

    The leverage is optimized in the range [0, maximum_leverage].

    Args:
        returns: Series of daily returns.
        maximum_leverage: Maximum leverage to consider (negative values are allowed)
        risk_free_rate: Annualized risk-free rate (default is 0.0).

    Returns:
        pd.DataFrame: DataFrame containing the optimal leverage and corresponding metrics.
    """
    leverages, geometric_means, volatilities, calmar_ratios = [], [], [], []

    # De-annualize risk-free rate assuming 252 trading days
    daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1

    for leverage in np.linspace(0.0, maximum_leverage, 1000):
        # Apply the formula: R_leveraged = R_f + L * (R_asset - R_f)
        leveraged_returns = daily_risk_free_rate + leverage * (
            returns - daily_risk_free_rate
        )
        # Avoid negative returns exceeding -100%
        leveraged_returns = leveraged_returns.clip(lower=-1.0)
        geometric_mean_leveraged = geometric_mean(leveraged_returns)

        if geometric_mean_leveraged <= 0:
            continue

        volatility_leveraged = volatility(leveraged_returns)

        leverages.append(leverage)
        geometric_means.append(geometric_mean_leveraged)
        volatilities.append(volatility_leveraged)
        # Handle division by zero for Calmar Ratio
        calmar_ratio = (
            geometric_mean_leveraged / volatility_leveraged
            if volatility_leveraged > 0
            else np.inf
            if geometric_mean_leveraged > 0
            else 0  # Or np.nan, np.inf seems reasonable
        )
        calmar_ratios.append(calmar_ratio)

    return pd.DataFrame(
        {
            "Leverage": leverages,
            "Geometric Mean": geometric_means,
            "Volatility": volatilities,
            "Calmar Ratio": calmar_ratios,
        }
    )
