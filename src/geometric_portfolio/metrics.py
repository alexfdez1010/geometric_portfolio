import numpy as np
import pandas as pd
from typing import cast


def arithmetic_mean(returns: pd.Series) -> float:
    """
    Calculate the arithmetic mean of daily returns.

    Args:
        returns: Series of daily returns.

    Returns:
        float: Arithmetic mean of the returns.
    """
    return cast(float, returns.mean())


def geometric_mean(returns: pd.Series) -> float:
    """
    Calculate the geometric mean (compound average) of daily returns.

    Args:
        returns: Series of daily returns.

    Returns:
        float: Geometric mean of the returns.
    """
    return np.exp(np.log1p(returns).mean()) - 1


def volatility(returns: pd.Series) -> float:
    """
    Calculate the daily volatility (standard deviation) of returns.

    Args:
        returns: Series of daily returns.

    Returns:
        float: Daily volatility of the returns.
    """
    return cast(float, returns.std(ddof=1))


def annual_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized volatility of returns.

    Args:
        returns: Series of daily returns.
        periods_per_year: Trading periods per year (default is 252).

    Returns:
        float: Annualized volatility of the returns.
    """
    return cast(float, returns.std(ddof=1) * np.sqrt(periods_per_year))


def annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized arithmetic return of returns.

    Args:
        returns: Series of daily returns.
        periods_per_year: Trading periods per year (default is 252).

    Returns:
        float: Annualized arithmetic return of the returns.
    """
    return cast(float, returns.mean() * periods_per_year)


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR) of returns.

    Args:
        returns: Series of daily returns.
        periods_per_year: Trading periods per year (default is 252).

    Returns:
        float: CAGR of the returns.
    """
    daily_geo_mean = geometric_mean(returns)
    return (1 + daily_geo_mean) ** periods_per_year - 1


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized Sharpe ratio of returns.

    Args:
        returns: Series of daily returns.
        risk_free_rate: Annual risk-free rate (default is 0.0).
        periods_per_year: Trading periods per year (default is 252).

    Returns:
        float: Annualized Sharpe ratio.
    """
    excess = returns - risk_free_rate / periods_per_year
    ann_excess = excess.mean() * periods_per_year
    ann_vol = excess.std(ddof=1) * np.sqrt(periods_per_year)
    return ann_excess / ann_vol if ann_vol != 0 else np.nan


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
    return drawdown.min()


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
    yearly = returns.groupby("YE").apply(lambda x: (1 + x).prod() - 1)
    return float(yearly.max())


def worst_year(returns: pd.Series) -> float:
    """
    Identify the worst performing calendar year.

    Args:
        returns: Series of daily returns (DatetimeIndex).

    Returns:
        float: Total return for the worst year.
    """
    yearly = returns.groupby("YE").apply(lambda x: (1 + x).prod() - 1)
    return float(yearly.min())


def summary(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> pd.Series:
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
        'Arithmetic Mean': arithmetic_mean(returns),
        'Geometric Mean': geometric_mean(returns),
        'Volatility': volatility(returns),
        'Annual Volatility': annual_volatility(returns, periods_per_year),
        'Annual Return': annual_return(returns, periods_per_year),
        'CAGR': cagr(returns, periods_per_year),
        'Sharpe Ratio': sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'Max Drawdown': max_drawdown(returns),
        'Best Day': best_day(returns),
        'Worst Day': worst_day(returns),
        'Best Year': best_year(returns),
        'Worst Year': worst_year(returns),
    }
    summary_series = pd.Series(metrics)
    print("Portfolio Metrics Summary:\n", summary_series)
    return summary_series