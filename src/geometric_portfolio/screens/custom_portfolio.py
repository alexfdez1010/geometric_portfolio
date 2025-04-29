import streamlit as st
from datetime import date

from geometric_portfolio.screens.shared import show_summary
from geometric_portfolio.backtesting import backtesting
from geometric_portfolio.metrics import wealth
from geometric_portfolio.plot import (
    plot_wealth_evolution,
    plot_returns_distribution,
    plot_correlation_matrix,
)
from geometric_portfolio.tickers import TICKERS, resolve_ticker
from geometric_portfolio.data import get_returns
from geometric_portfolio.screens.page import Page


class CustomPortfolioPage(Page):
    def render(self):
        # st.set_page_config(page_title="Custom Portfolio") # Removed redundant call
        st.title("Custom Portfolio Backtesting")

        # Sidebar inputs
        st.header("Custom Portfolio Inputs")
        selected_names = st.multiselect(
            "Select Assets", list(TICKERS.keys()), key="custom_assets_select"
        )
        selected = [TICKERS[name] for name in selected_names]

        # Custom ticker input
        custom_input = st.text_input("Custom Ticker (symbol or name)")
        if custom_input:
            try:
                custom_symbol = resolve_ticker(custom_input)
                if custom_symbol not in selected:
                    selected.append(custom_symbol)
            except ValueError as e:
                st.error(str(e))

        start = st.date_input("Start date", value=date(2020, 1, 1))
        end = st.date_input("End date", value=date.today())
        initial_amount = st.number_input(
            "Initial amount", min_value=1000.0, value=10000.0, step=100.0
        )
        acceptable_diff = st.number_input(
            "Acceptable difference", min_value=0.0, max_value=1.0, value=0.05, step=0.01
        )
        fixed_cost = st.number_input("Fixed cost", min_value=0.0, value=1.0, step=0.1)
        variable_cost = st.number_input(
            "Variable cost", min_value=0.0, value=0.0, step=0.01
        )

        # Weight assignment
        weights = {}
        if selected:
            st.subheader("Assign Weights")

            for ticker in selected:
                w = st.number_input(
                    f"Weight for {ticker}",
                    min_value=0.0,
                    max_value=1.0,
                    value=round(1.0 / len(selected), 2),
                    step=0.01,
                    format="%.2f",
                )
                weights[ticker] = w

            total = sum(weights.values())

            if total > 0:
                weights = {ticker: w / total for ticker, w in weights.items()}

        run = st.button("Run Backtest", key="custom_run_button")

        if not run:
            return

        if not selected:
            st.warning("Please select at least one asset.")
            return

        with st.spinner("Running backtest..."):
            asset_returns = get_returns(
                tickers=selected, start_date=start.isoformat(), end_date=end.isoformat()
            )

            strategy_returns, _ = backtesting(
                initial_amount=initial_amount,
                tickers=selected,
                target_weights=weights,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                acceptable_diff=acceptable_diff,
                fixed_cost=fixed_cost,
                variable_cost=variable_cost,
            )

            # Display metrics
            st.subheader("Portfolio Metrics")
            df = {
                "Custom Portfolio": strategy_returns,
                **{asset: asset_returns[asset] for asset in asset_returns.columns},
            }
            show_summary(df)

            wealth_strategy = wealth(strategy_returns)

            st.subheader("Wealth Evolution")
            all_returns = {
                asset: wealth(asset_returns[asset]) for asset in asset_returns.columns
            }
            all_returns["Custom Portfolio"] = wealth_strategy

            fig_w = plot_wealth_evolution(all_returns)
            st.pyplot(fig_w)

            st.subheader("Returns Distribution")
            returns = {asset: asset_returns[asset] for asset in asset_returns.columns}
            returns["Custom Portfolio"] = strategy_returns

            fig_r = plot_returns_distribution(returns)
            st.pyplot(fig_r)

            # Correlation matrix
            st.subheader("Correlation Matrix")
            # Combine returns for correlation
            returns_df = asset_returns.copy()
            returns_df["Custom Portfolio"] = strategy_returns
            fig_corr = plot_correlation_matrix(returns_df)
            st.pyplot(fig_corr)
