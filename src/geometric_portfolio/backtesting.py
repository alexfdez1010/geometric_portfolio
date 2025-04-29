import pandas as pd
import yfinance as yf


def backtesting(
    initial_amount: float,
    tickers: list[str],
    target_weights: dict[str, float],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    acceptable_diff: float,
    fixed_cost: float = 0.0,
    variable_cost: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Backtest a portfolio with rebalancing under realistic conditions.

    Args:
        initial_amount: Initial amount of money to invest.
        tickers: List of asset tickers to download from Yahoo Finance.
        target_weights: Dict of target weights for each asset (must sum to 1).
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
    if not all(ticker in target_weights for ticker in tickers):
        raise ValueError("All tickers must have corresponding weights")

    if abs(sum(target_weights.values()) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")

    # Download data
    raw_data = yf.download(
        tickers, start=start_date, end=end_date, progress=False, auto_adjust=True
    )
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
    # Initialize shares dict based on target weights
    init_prices = close.iloc[0]
    shares = {
        t: target_weights.get(t, 0.0) * portfolio_value / init_prices[t]
        for t in tickers
    }

    weight_history = []
    returns = []
    prev_value = (
        initial_amount  # Start with initial investment for first calculation base
    )

    for i, date in enumerate(dates):
        price_close = close.loc[date]  # Series indexed by ticker

        # 1. Calculate value before any rebalancing for this day
        value_before_rebalance = sum(shares[t] * price_close[t] for t in tickers)

        # Store current weights (based on value BEFORE rebalance)
        if value_before_rebalance == 0:  # Avoid division by zero if value drops to zero
            current_weights = {t: 0.0 for t in tickers}
        else:
            current_weights = {
                t: shares[t] * price_close[t] / value_before_rebalance for t in tickers
            }
        weight_history.append(current_weights.copy())

        # --- Rebalancing Logic ---
        total_cost = 0.0
        # Check if rebalancing is needed
        if any(
            abs(current_weights.get(t, 0.0) - target_weights.get(t, 0.0))
            > acceptable_diff
            for t in tickers
        ):
            # Calculate trades based on value_before_rebalance
            trade_vals = {
                t: target_weights.get(t, 0.0) * value_before_rebalance
                - shares[t] * price_close[t]
                for t in tickers
            }
            # Calculate total cost for the rebalancing trades
            total_cost = sum(
                (fixed_cost + variable_cost * abs(trade_vals[t]))
                for t in tickers
                if trade_vals[t] != 0
            )
            # Update shares (handle potential division by zero if price is zero)
            for t in tickers:
                if trade_vals[t] != 0 and price_close[t] != 0:
                    shares[t] += trade_vals[t] / price_close[t]
                elif trade_vals[t] != 0 and price_close[t] == 0:
                    # Handle edge case: trying to trade an asset whose price dropped to zero
                    # This might involve logging a warning or adjusting logic based on desired behavior
                    pass  # For now, just skip the trade if price is zero

            # Value after rebalance (shares updated), before cost deduction
            value_after_rebalance_pre_cost = sum(
                shares[t] * price_close[t] for t in tickers
            )
            value_end_of_day = value_after_rebalance_pre_cost - total_cost
        else:
            # If no rebalance, end-of-day value is the value before rebalance
            value_end_of_day = value_before_rebalance

        # Calculate return based on end-of-day values
        daily_return = 0.0  # Default return for first day or if prev_value is 0
        if i > 0 and prev_value != 0:  # Avoid division by zero for prev_value
            daily_return = value_end_of_day / prev_value - 1
            returns.append(daily_return)
        elif i > 0 and prev_value == 0:
            # If previous value was zero, return can be considered infinite or zero, depending on context
            returns.append(0.0)  # Append 0 if portfolio value was wiped out previously

        # Update prev_value for the next iteration's calculation
        prev_value = value_end_of_day

    # Handle case where there's only one date or returns list is empty
    if len(returns) == 0 and len(dates) > 1:
        returns = [0.0] * (len(dates) - 1)
    elif len(returns) == 0 and len(dates) <= 1:
        # If only one date, or no dates, return empty series/df
        return pd.Series(dtype=float), pd.DataFrame(columns=tickers)

    # Create return series starting from second date
    portfolio_returns = pd.Series(returns, index=dates[1:] if len(dates) > 1 else dates)
    # Weight history includes the initial state, align by removing first entry if returns exist
    weight_df = pd.DataFrame(weight_history, index=dates)
    if not portfolio_returns.empty:
        weight_df = weight_df.iloc[1:]
    return portfolio_returns, weight_df
