import pandas as pd
import yfinance as yf


def backtesting(
    initial_amount: float,
    tickers: list[str],
    weights: dict[str, float],
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
    if not all(ticker in weights for ticker in tickers):
        raise ValueError("All tickers must have corresponding weights")
    
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")

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
    # Initialize target weight dict and per-ticker shares dict
    weight_dict = {t: weights.get(t, 0.0) for t in tickers}
    init_prices = close.iloc[0]
    shares = {t: weight_dict[t] * portfolio_value / init_prices[t] for t in tickers}

    weight_history = []
    returns = []
    prev_value = portfolio_value

    for i, date in enumerate(dates):
        price_close = close.loc[date]  # Series indexed by ticker
        
        # Compute portfolio value by summing share*price
        curr_value = sum(shares[t] * price_close[t] for t in tickers)
        
        if i > 0:
            returns.append(curr_value / prev_value - 1)
        
        # Current weights per ticker
        curr_weights = {t: shares[t] * price_close[t] / curr_value for t in tickers}
        weight_history.append(curr_weights.copy())
        
        # Rebalance if any weight deviates beyond threshold
        if any(abs(curr_weights[t] - weight_dict[t]) > acceptable_diff for t in tickers):
            # Compute trade values and costs
            trade_vals = {t: weight_dict[t] * curr_value - shares[t] * price_close[t] for t in tickers}
            total_cost = sum((fixed_cost + variable_cost * abs(trade_vals[t])) for t in tickers if trade_vals[t] != 0)
            # Execute trades
            for t in tickers:
                if trade_vals[t] != 0:
                    shares[t] += trade_vals[t] / price_close[t]
            curr_value -= total_cost
            # Update target weights to actual post-trade
            weight_dict = {t: shares[t] * price_close[t] / curr_value for t in tickers}
        
        prev_value = curr_value

    # Handle case where there's only one date
    if len(returns) == 0 and len(dates) > 0:
        returns = [0.0] * (len(dates) - 1)
    
    # Create return series starting from second date
    portfolio_returns = pd.Series(returns, index=dates[1:] if len(dates) > 1 else dates)
    weight_df = pd.DataFrame(weight_history, index=dates)

    return portfolio_returns, weight_df
