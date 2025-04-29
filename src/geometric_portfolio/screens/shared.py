import pandas as pd
import streamlit as st
from geometric_portfolio.metrics import summary


def show_summary(asset_returns: dict[str, pd.Series | None]):
    """
    Display summary statistics for each portfolio.

    Args:
        asset_returns: dict[str, pd.Series | None] of returns for each asset and portfolio.
    """
    rows = []
    for title, ret in asset_returns.items():
        if ret is None:
            s = pd.Series(dtype=float).rename(title)
        else:
            s = summary(ret).rename(title)
        rows.append(s)
    df = pd.DataFrame(rows)
    percent_cols = [
        "Arithmetic Mean",
        "Geometric Mean",
        "Volatility",
        "Max Drawdown",
        "Best Day",
        "Worst Day",
        "Best Year",
        "Worst Year",
    ]
    fmt_dict = {col: "{:.2%}" for col in percent_cols if col in df.columns}
    styled = df.style.format(fmt_dict)
    st.subheader("Portfolio Metrics Table")
    st.dataframe(styled)
