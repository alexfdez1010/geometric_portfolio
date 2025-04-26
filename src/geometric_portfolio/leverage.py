import pandas as pd
import numpy as np
from geometric_portfolio.metrics import geometric_mean, volatility


def leverage_optimizer(
    returns: pd.Series, minimum_leverage: float = 0.0, maximum_leverage: float = 1.0
) -> pd.DataFrame:
    """
    Find the optimal leverage for a given series of returns.

    Args:
        returns: Series of daily returns.
        minimum_leverage: Minimum leverage to consider (negative values are allowed)
        maximum_leverage: Maximum leverage to consider (negative values are allowed)

    Returns:
        pd.DataFrame: DataFrame containing the optimal leverage and corresponding metrics.
    """
    leverages, geometric_means, volatilities, calmar_ratios = [], [], [], []

    for leverage in np.linspace(minimum_leverage, maximum_leverage, 1000):
        leveraged_returns = leverage * returns
        # Avoid negative returns exceeding -100%
        leveraged_returns = leveraged_returns.clip(lower=-1.0)
        geometric_mean_leveraged = geometric_mean(leveraged_returns)

        if geometric_mean_leveraged <= 0:
            continue

        volatility_leveraged = volatility(leveraged_returns)

        leverages.append(leverage)
        geometric_means.append(geometric_mean_leveraged)
        volatilities.append(volatility_leveraged)
        calmar_ratios.append(geometric_mean_leveraged / volatility_leveraged)

    return pd.DataFrame(
        {
            "Leverage": leverages,
            "Geometric Mean": geometric_means,
            "Volatility": volatilities,
            "Calmar Ratio": calmar_ratios,
        }
    )
