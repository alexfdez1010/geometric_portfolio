import streamlit as st
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt

from geometric_portfolio.plot import (
    plot_wealth_evolution,
    plot_returns_distribution,
    plot_correlation_matrix,
)

# Only import used metrics
from geometric_portfolio.metrics import wealth
from geometric_portfolio.data import get_returns
from geometric_portfolio.solver import PortfolioSolver
from geometric_portfolio.backtesting import backtesting
from geometric_portfolio.tickers import TICKERS, CATEGORIES, resolve_ticker
from geometric_portfolio.screens.page import Page
from geometric_portfolio.screens.shared import show_summary


# Define the Page class for Geometric Mean
class GeometricMeanPage(Page):
    def display_weights(self, criteria: list[tuple[str, dict[str, float]]]):
        """
        Display the best portfolio weights for each objective.

        Args:
            criteria: List of tuples (title, weights) where weights is a dict of asset weights.
        """
        st.subheader("Best Portfolio Weights")
        cols = st.columns(len(criteria))
        for col, (title, weights) in zip(cols, criteria):
            with col:
                st.write(title)
                dfw = pd.DataFrame.from_dict(weights, orient="index")
                dfw.columns = ["Weight"]
                dfw = dfw[dfw["Weight"] > 0.01]
                dfw["Weight"] = dfw["Weight"].apply(lambda x: f"{x * 100:.2f}%")
                st.table(dfw)

    def compute_asset_returns(
        self,
        returns: pd.DataFrame,
        criteria: list[tuple[str, dict[str, float]]],
        initial_amount: float,
        acceptable_diff: float,
        fixed_cost: float,
        variable_cost: float,
        start: date,
        end: date,
    ) -> dict[str, pd.Series | None]:
        """
        Compute returns for each asset and portfolio.

        Args:
            returns: pd.DataFrame of asset returns.
            criteria: List of tuples (title, weights) where weights is a dict of asset weights.
            initial_amount: Initial amount of money to invest.
            acceptable_diff: Threshold for weight deviation to trigger rebalancing.
            fixed_cost: Fixed transaction cost per trade.
            variable_cost: Variable transaction cost as a fraction of trade value.
            start: Backtest start date.
            end: Backtest end date.

        Returns:
            dict[str, pd.Series | None] of returns for each asset and portfolio.
        """
        asset_returns = {asset: returns[asset] for asset in returns.columns}
        for title, weights in criteria:
            try:
                strategy_returns, _ = backtesting(
                    initial_amount=initial_amount,
                    tickers=list(weights.keys()),
                    target_weights=weights,
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    acceptable_diff=acceptable_diff,
                    fixed_cost=fixed_cost,
                    variable_cost=variable_cost,
                )
                asset_returns[title] = strategy_returns
            except Exception as e:
                st.error(f"Error backtesting '{title}': {e}")
                asset_returns[title] = None

        return asset_returns

    def plot_results(
        self,
        returns: pd.DataFrame,
        criteria: list[tuple[str, dict[str, float]]],
        solver: PortfolioSolver,
    ):
        """
        Plot wealth evolution and returns distribution for each portfolio.

        Args:
            returns: pd.DataFrame of asset returns.
            criteria: List of tuples (title, weights) where weights is a dict of asset weights.
            solver: PortfolioSolver instance.
        """
        st.subheader("Geometric vs Max Drawdown")
        solver.plot_geometric_max_drawdown()
        st.pyplot(plt.gcf())
        # Compute returns for each portfolio
        returns_dict = {asset: returns[asset] for asset in returns.columns}
        for title, weights in criteria:
            try:
                returns_dict[title] = solver.compute_returns(weights)
            except Exception:
                returns_dict[title] = pd.Series(dtype=float)
        # Wealth evolution
        st.subheader("Wealth Evolution")
        wealth_dict = {asset: wealth(returns[asset]) for asset in returns.columns}
        for title, _ in criteria:
            wealth_dict[title] = wealth(returns_dict[title])
        fig = plot_wealth_evolution(wealth_dict)
        st.pyplot(fig)
        # Returns distribution
        st.subheader("Returns Distribution")
        fig2 = plot_returns_distribution(returns_dict)
        st.pyplot(fig2)
        st.subheader("Correlation Matrix")
        fig3 = plot_correlation_matrix(returns)
        st.pyplot(fig3)

    def render(self):
        # st.set_page_config(page_title="Geometric Mean Portfolio") # Removed redundant call
        st.title("Geometric Mean Portfolio Optimization")

        st.header("Inputs")

        # Grouped asset selection by category - moved from sidebar
        selected_names: list[str] = []
        with st.expander("Equity ETFs", expanded=True):
            sel_equity = st.multiselect(
                "Equity ETFs", CATEGORIES["Equity ETFs"], key="gm_equity_select"
            )
            selected_names.extend(sel_equity)
        with st.expander("Leveraged ETFs"):
            sel_lev = st.multiselect(
                "Leveraged ETFs", CATEGORIES["Leveraged ETFs"], key="gm_lev_select"
            )
            selected_names.extend(sel_lev)
        with st.expander("Commodities"):
            sel_comm = st.multiselect(
                "Commodities ETFs", CATEGORIES["Commodities"], key="gm_comm_select"
            )
            selected_names.extend(sel_comm)
        with st.expander("VIX ETFs"):
            sel_vix = st.multiselect(
                "VIX ETFs", CATEGORIES["VIX ETFs"], key="gm_vix_select"
            )
            selected_names.extend(sel_vix)
        with st.expander("Stocks"):
            sel_stocks = st.multiselect(
                "Stocks", CATEGORIES["Stocks"], key="gm_stocks_select"
            )
            selected_names.extend(sel_stocks)

        selected = [TICKERS[name] for name in selected_names]
        custom_input = st.text_input("Custom Ticker (symbol or name)")
        if custom_input:
            try:
                custom_symbol = resolve_ticker(custom_input)
                if custom_symbol not in selected:
                    selected.append(custom_symbol)
            except ValueError as e:
                st.error(str(e))

        # Other inputs - moved from sidebar
        start = st.date_input("Start date", value=date(2020, 1, 1))
        end = st.date_input("End date", value=date.today())
        initial_amount = st.number_input(
            "Initial amount", min_value=1000, value=10000, step=100
        )
        acceptable_diff = st.number_input(
            "Acceptable difference", min_value=0.0, max_value=1.0, value=0.05, step=0.01
        )
        fixed_cost = st.number_input("Fixed cost", min_value=0.0, value=1.0, step=0.1)
        variable_cost = st.number_input(
            "Variable cost", min_value=0.0, value=0.0, step=0.01, format="%.2f"
        )

        run = st.button("Run Optimization", key="gm_run_button")

        if not run:
            return

        if len(selected) < 2:
            st.warning("Please select at least two assets to run the solver.")
            return

        # Fetch data ONCE
        returns_df = get_returns(
            tickers=selected,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )

        solver = PortfolioSolver(returns_df)
        # Run the optimization to get all weights
        weights_geom, weights_drawdown, weights_calmar = solver.run()

        # Criteria list using weights from solver.run()
        criteria = [
            ("Geometric Mean", weights_geom),
            ("Min Drawdown", weights_drawdown),
            ("Max Calmar", weights_calmar),
        ]

        self.display_weights(criteria)

        asset_returns_dict = self.compute_asset_returns(
            returns=returns_df,
            criteria=criteria,
            initial_amount=initial_amount,
            acceptable_diff=acceptable_diff,
            fixed_cost=fixed_cost,
            variable_cost=variable_cost,
            start=start,
            end=end,
        )

        show_summary(asset_returns_dict)

        # Use the SAME returns_df for plotting
        self.plot_results(returns=returns_df, criteria=criteria, solver=solver)
