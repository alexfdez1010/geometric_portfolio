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
    # Validate inputs
    if len(tickers) != len(weights):
        raise ValueError("Length of tickers and weights must match")
    
    # Normalize weights if they don't sum to 1
    weights = np.array(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()

    # Download data
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if raw_data.empty:
        raise ValueError(f"No data found for tickers {tickers}")
    
    raw_data = raw_data.dropna()
    close = raw_data["Close"]

    # Handle single ticker case
    if isinstance(close, pd.Series):
        close = close.to_frame()

    dates = close.index
    if len(dates) == 0:
        raise ValueError("No valid trading dates in the specified period")
    
    # Initialize portfolio
    portfolio_value = initial_amount
    prices0 = close.iloc[0].values
    shares = (weights * portfolio_value) / prices0

    weight_history = []
    returns = []
    prev_value = portfolio_value

    for i, date in enumerate(dates):
        price_close = close.loc[date].values
        
        # Update portfolio value at close
        curr_value = np.dot(shares, price_close)
        
        # Calculate daily return (skip first day)
        if i > 0:
            daily_ret = curr_value / prev_value - 1
            returns.append(daily_ret)
        
        # Current weights
        curr_weights = shares * price_close / curr_value
        weight_history.append(curr_weights.tolist())
        
        # Check for rebalancing
        if np.any(np.abs(curr_weights - weights) > acceptable_diff):
            # Calculate target shares
            target_vals = weights * curr_value
            trade_vals = target_vals - shares * price_close
            
            # Compute transaction costs
            costs = np.zeros_like(trade_vals)
            non_zero_trades = trade_vals != 0
            costs[non_zero_trades] = fixed_cost + variable_cost * np.abs(trade_vals[non_zero_trades])
            total_cost = costs.sum()
            
            # Execute trades
            shares = shares + trade_vals / price_close
            
            # Deduct costs from portfolio value
            curr_value -= total_cost
        
        prev_value = curr_value

    # Handle case where there's only one date
    if len(returns) == 0 and len(dates) > 0:
        returns = [0.0] * (len(dates) - 1)
    
    # Create return series starting from second date
    portfolio_returns = pd.Series(returns, index=dates[1:] if len(dates) > 1 else dates)
    weight_history = pd.DataFrame(weight_history, index=dates, columns=tickers)

    return portfolio_returns, weight_history
