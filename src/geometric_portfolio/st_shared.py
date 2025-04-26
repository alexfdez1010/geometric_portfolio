import pandas as pd
import streamlit as st
from geometric_portfolio.leverage import leverage_optimizer


def show_leverage(
    returns: pd.Series,
    title: str | None = None,
    minimum_leverage: float = 0.0,
    maximum_leverage: float = 1.0,
):
    """
    Show the optimal leverage for a given series of returns.

    Args:
        returns: Series of daily returns.
        title: Title for the section.
        minimum_leverage: Minimum leverage to consider (default is 0.0).
        maximum_leverage: Maximum leverage to consider (default is 1.0).
    """
    df = leverage_optimizer(
        returns, minimum_leverage=minimum_leverage, maximum_leverage=maximum_leverage
    )

    best_idx = df["Geometric Mean"].idxmax()
    best_l = df.loc[best_idx, "Leverage"]
    best_gm = df.loc[best_idx, "Geometric Mean"]

    if title:
        st.subheader(title)

    st.write(f"Optimal Leverage: {best_l:.2f} with geometric mean {best_gm:.4%}")

    # Display all metrics in a single chart with different colors
    chart_data = df.set_index("Leverage").copy()
    chart_data["Geometric Mean (%)"] = chart_data["Geometric Mean"] * 100
    chart_data["Volatility (%)"] = chart_data["Volatility"] * 100
    chart_data["Calmar Ratio (%)"] = chart_data["Calmar Ratio"] * 100

    st.line_chart(
        chart_data[["Geometric Mean (%)", "Volatility (%)", "Calmar Ratio (%)"]],
        color=["#ff9933", "#33cc33", "#3366ff"],
    )
