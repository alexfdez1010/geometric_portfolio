import numpy as np
import pandas as pd
from typing import cast


def arithmetic_mean(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized arithmetic mean from daily returns.

    Args:
        returns: Series of daily returns.
        periods_per_year: Trading periods per year (default is 252).

    Returns:
        float: Annualized arithmetic mean of the returns.
    """
    mean = returns.mean()
    return (1 + mean) ** periods_per_year - 1


def geometric_mean(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized geometric mean (compound average) of daily returns.

    Args:
        returns: Series of daily returns.
        periods_per_year: Trading periods per year (default is 252).

    Returns:
        float: Geometric mean of the returns, annualized.
    """
    return np.prod(1 + returns) ** (periods_per_year / len(returns)) - 1


def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized volatility from daily returns.

    Args:
        returns: Series of daily returns.
        periods_per_year: Trading periods per year (default is 252).

    Returns:
        float: Annualized volatility of the returns.
    """
    return cast(float, returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Calculate the annualized Sharpe ratio of returns.

    Args:
        returns: Series of daily returns.
        risk_free_rate: Annual risk-free rate (default is 0.0).
        periods_per_year: Trading periods per year (default is 252).

    Returns:
        float: Annualized Sharpe ratio.
    """
    vol = volatility(returns, periods_per_year)
    if np.isclose(vol, 0):
        return np.nan
    return (arithmetic_mean(returns, periods_per_year) - risk_free_rate) / vol


def calmar_ratio(returns: pd.Series, risk_preference: float = 1.0) -> float:
    """
    Calculate the Calmar ratio of returns (geometric mean / max drawdown ^ risk_preference). This
    ratio wants to maximize the geometric mean while minimizing the max drawdown. Depeding of the
    risk preference, the ratio will be more or less sensitive to the max drawdown. Higher
    risk preference will make the ratio more sensitive to the max drawdown. If risk preference is 0,
    the ratio will be equal to the geometric mean.

    Args:
        returns: Series of daily returns.
        risk_preference: Risk preference parameter (higher implies less risk, if 0 equals to maximize the geometric mean).

    Returns:
        float: Annualized Calmar ratio.
    """
    max_drawdown_value = max_drawdown(returns)
    if np.isclose(max_drawdown_value, 0):
        return np.nan
    return geometric_mean(returns) / (max_drawdown_value**risk_preference)


def max_drawdown(returns: pd.Series) -> float:
    """
    Identify the maximum drawdown of returns.

    Args:
        returns: Series of daily returns.

    Returns:
        float: Maximum drawdown (negative value).
    """
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    return -drawdown.min()


def best_day(returns: pd.Series) -> float:
    """
    Identify the best performing day.

    Args:
        returns: Series of daily returns.

    Returns:
        float: Return for the best day.
    """
    date = returns.idxmax()
    return float(returns.loc[date])


def worst_day(returns: pd.Series) -> float:
    """
    Identify the worst performing day.

    Args:
        returns: Series of daily returns.

    Returns:
        float: Return for the worst day.
    """
    date = returns.idxmin()
    return float(returns.loc[date])


def best_year(returns: pd.Series) -> float:
    """
    Identify the best performing calendar year.

    Args:
        returns: Series of daily returns (DatetimeIndex).

    Returns:
        float: Total return for the best year.
    """
    yearly = returns.groupby(returns.index.year).apply(lambda x: (1 + x).prod() - 1)
    return float(yearly.max())


def worst_year(returns: pd.Series) -> float:
    """
    Identify the worst performing calendar year.

    Args:
        returns: Series of daily returns (DatetimeIndex).

    Returns:
        float: Total return for the worst year.
    """
    yearly = returns.groupby(returns.index.year).apply(lambda x: (1 + x).prod() - 1)
    return float(yearly.min())


def summary(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> pd.Series:
    """
    Compute and display common portfolio metrics.

    Args:
        returns: Series of daily returns.
        risk_free_rate: Annual risk-free rate (default is 0.0).
        periods_per_year: Trading periods per year (default is 252).

    Returns:
        pd.Series: Series of computed metrics.
    """
    metrics = {
        "Geometric Mean": geometric_mean(returns),
        "Max Drawdown": max_drawdown(returns),
        "Calmar Ratio": calmar_ratio(returns),
        "Volatility": volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "Best Day": best_day(returns),
        "Worst Day": worst_day(returns),
        "Best Year": best_year(returns),
        "Worst Year": worst_year(returns),
        "Arithmetic Mean": arithmetic_mean(returns),
    }
    summary_series = pd.Series(metrics)
    return summary_series


def wealth(returns: pd.Series, initial_wealth: float = 1.0) -> pd.Series:
    """
    Compute the wealth index from returns.

    Args:
        returns: Series of daily returns.
        initial_wealth: Initial wealth value (default is 1.0).

    Returns:
        pd.Series: Series of computed wealth index.
    """
    return (1 + returns).cumprod() * initial_wealth
