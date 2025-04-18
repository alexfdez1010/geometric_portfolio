from typing import cast
import pandas as pd
import yfinance as yf

def get_returns(tickets: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get daily returns for a list of tickets from Yahoo Finance.

    Args:
        tickets: List of ticket symbols.
        start_date: Start date for the data.
        end_date: End date for the data.

    Returns:
        pd.DataFrame: Returns for the given tickets.
    """
    df = yf.download(tickets, start=start_date, end=end_date, auto_adjust=True, multi_level_index=True, interval="1d")

    if df is None or 'Close' not in df:
        raise ValueError("No data found for the given tickets.")
    
    df = cast(pd.DataFrame, df['Close'])
    
    # If DataFrame is empty or has no valid columns, raise ValueError
    if df.empty or all([col not in df.columns for col in tickets]):
        raise ValueError("No data found for the given tickets.")
    
    return df.pct_change().dropna()