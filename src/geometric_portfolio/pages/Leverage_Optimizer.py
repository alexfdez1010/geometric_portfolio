import streamlit as st
from datetime import date

from geometric_portfolio.data import get_returns
from geometric_portfolio.tickers import TICKERS, resolve_ticker
from geometric_portfolio.st_shred import show_leverage

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
        
        show_leverage(returns, title="Leverage Optimizer")

if __name__ == "__main__":
    main()
