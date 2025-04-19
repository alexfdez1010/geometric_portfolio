import pandas as pd
import numpy as np
import yfinance as yf


def backtesting(
    initial_amount: float,
    tickers: list[str],
    weights: list[float],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    acceptable_diff: float,
    fixed_cost: float = 0.0,
    variable_cost: float = 0.0
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Backtest a portfolio with rebalancing under realistic conditions.

    Args:
        initial_amount: Initial amount of money to invest.
        tickers: List of asset tickers to download from Yahoo Finance.
        weights: List of target weights for each asset (must sum to 1).
        start_date: Backtest start date.
        end_date: Backtest end date.
        acceptable_diff: Threshold for weight deviation to trigger rebalancing.
        fixed_cost: Fixed transaction cost per trade.
        variable_cost: Variable transaction cost as a fraction of trade value.

    Returns:
        A tuple containing:
        - portfolio_returns: pd.Series of daily returns of the strategy.
        - weight_history: pd.DataFrame of daily weights held for each asset.
    """
    if len(tickers) != len(weights):
        raise ValueError("Length of tickers and weights must match")
    
    weights = np.array(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()

    raw_data = yf.download(tickers, start=start_date, end=end_date)
    raw_data = raw_data.dropna()
    close = raw_data["Close"]

    if isinstance(close, pd.Series):
        close = close.to_frame()

    dates = close.index
    # Initialize portfolio
    portfolio_value = initial_amount
    prices0 = close.iloc[0].values
    shares = (weights * portfolio_value) / prices0

    weight_history = []
    returns = []
    prev_value = portfolio_value

    for date in dates:
        price_close = close.loc[date].values
        # update portfolio value at close
        curr_value = np.dot(shares, price_close)
        # daily return
        daily_ret = curr_value / prev_value - 1
        returns.append(daily_ret)
        # current weights
        curr_weights = shares * price_close / curr_value
        weight_history.append(curr_weights.tolist())
        # check for rebalancing
        if np.any(np.abs(curr_weights - weights) > acceptable_diff):
            target_vals = weights * curr_value
            trade_vals = target_vals - shares * price_close
            # compute transaction costs
            costs = np.where(
                trade_vals != 0,
                fixed_cost + variable_cost * np.abs(trade_vals),
                0.0
            )
            total_cost = costs.sum()
            # execute trades
            shares = shares + trade_vals / price_close
            # recalculate portfolio value after trades
            curr_value = np.dot(shares, price_close)
            # deduct costs
            curr_value -= total_cost
            prev_value = curr_value
        else:
            prev_value = curr_value

    portfolio_returns = pd.Series(returns, index=dates)
    weight_history = pd.DataFrame(weight_history, index=dates, columns=tickers)

    return portfolio_returns, weight_history
