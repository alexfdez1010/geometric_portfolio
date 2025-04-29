import streamlit as st
from datetime import date

from geometric_portfolio.data import get_returns
from geometric_portfolio.tickers import TICKERS, resolve_ticker
from geometric_portfolio.st_shared import show_leverage
from geometric_portfolio.screens.page import Page


class LeverageOptimizerPage(Page):
    def render(self):
        st.title("Leverage Optimizer")

        asset_name = st.selectbox(
            "Select Asset", list(TICKERS.keys()), key="lo_asset_select"
        )
        ticker = TICKERS[asset_name]
        custom_input = st.text_input(
            "Custom Ticker (symbol or name)", key="lo_custom_ticker"
        )
        if custom_input:
            try:
                custom_symbol = resolve_ticker(custom_input)
                ticker = custom_symbol
            except ValueError as e:
                st.error(str(e))

        start = st.date_input("Start date", value=date(2020, 1, 1), key="lo_start_date")
        end = st.date_input("End date", value=date.today(), key="lo_end_date")

        # Add input for Risk-Free Rate
        risk_free_rate_pct = st.number_input(
            "Risk-Free Rate (%)",
            value=0.0,
            format="%.2f",
            key="lo_risk_free_rate",
            help="Enter the annualized risk-free rate as a percentage (e.g., 2.25 for 2.25%).",
        )
        risk_free_rate = risk_free_rate_pct / 100.0  # Convert percentage to decimal

        run = st.button("Find Optimal Leverage", key="lo_run_button")

        if run:
            returns_df = get_returns(
                tickers=[ticker], start_date=start.isoformat(), end_date=end.isoformat()
            )
            returns = returns_df[ticker]

            show_leverage(
                returns,
                title="Leverage Optimizer",
                maximum_leverage=20,
                risk_free_rate=risk_free_rate,  # Pass the risk-free rate
            )
