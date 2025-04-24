import streamlit as st
from datetime import date
import numpy as np
import pandas as pd

from geometric_portfolio.data import get_returns
from geometric_portfolio.metrics import volatility, geometric_mean
from geometric_portfolio.tickers import TICKERS, resolve_ticker

def main():
    st.set_page_config(page_title="Leverage Optimizer")
    st.title("Leverage Optimizer")

    asset_name = st.sidebar.selectbox("Select Asset", list(TICKERS.keys()))
    ticker = TICKERS[asset_name]
    custom_input = st.sidebar.text_input("Custom Ticker (symbol or name)")
    if custom_input:
        try:
            custom_symbol = resolve_ticker(custom_input)
            ticker = custom_symbol
        except ValueError as e:
            st.sidebar.error(str(e))

    start = st.date_input("Start date", value=date(2020, 1, 1))
    end = st.date_input("End date", value=date.today())
    run = st.button("Find Optimal Leverage")

    if run:
        returns_df = get_returns(tickers=[ticker], start_date=start.isoformat(), end_date=end.isoformat())
        returns = returns_df[ticker]
        
        leverages, geometric_means, volatilities, alejandro_ratios = [], [], [], []
        
        for leverage in np.linspace(-10, 10, 1000):

            leveraged_returns = leverage * returns
            # Avoid negative returns exceeding -100%
            leveraged_returns = leveraged_returns.clip(lower=-1.0)
            geometric_mean_leveraged = geometric_mean(leveraged_returns)

            if geometric_mean_leveraged < 0:
                continue

            volatility_leveraged = volatility(leveraged_returns)
            
            leverages.append(leverage)
            geometric_means.append(geometric_mean_leveraged)
            volatilities.append(volatility_leveraged)
            alejandro_ratios.append(geometric_mean_leveraged / volatility_leveraged)
        
        df = pd.DataFrame({"Leverage": leverages, "Geometric Mean": geometric_means, "Volatility": volatilities, "Alejandro Ratio": alejandro_ratios})
        best_idx = df["Geometric Mean"].idxmax()
        best_l = df.loc[best_idx, "Leverage"]
        best_gm = df.loc[best_idx, "Geometric Mean"]
        st.write(f"Optimal Leverage: {best_l:.2f} with geometric mean {best_gm:.4%}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.line_chart(100 * df.set_index("Leverage")["Geometric Mean"], y_label="Geometric Mean (%)")
        with col2:
            st.line_chart(100 * df.set_index("Leverage")["Volatility"], y_label="Volatility (%)")
        with col3:
            st.line_chart(df.set_index("Leverage")["Alejandro Ratio"], y_label="Alejandro Ratio")

if __name__ == "__main__":
    main()
