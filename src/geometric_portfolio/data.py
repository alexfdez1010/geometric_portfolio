from typing import cast
import pandas as pd
import yfinance as yf

def get_returns(tickets: list[str], start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
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
    
    # Find the first date where all assets have data
    first_valid_date = df.dropna().index[0] if not df.dropna().empty else None
    
    if first_valid_date is not None:
        df = df.loc[first_valid_date:]
    
    return df.pct_change(fill_method=None).dropna()