import streamlit as st
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt

from geometric_portfolio.plot import plot_wealth_evolution, plot_returns_distribution, plot_correlation_matrix
from geometric_portfolio.metrics import wealth, summary
from geometric_portfolio.data import get_returns
from geometric_portfolio.solver import PortfolioSolver
from geometric_portfolio.backtesting import backtesting
from geometric_portfolio.tickers import TICKERS, CATEGORIES, resolve_ticker
from geometric_portfolio.st_shared import show_leverage

def display_weights(criteria: list[tuple[str, dict[str, float]]]):
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
            dfw = pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"])
            dfw = dfw[dfw["Weight"] > 0.01]
            dfw["Weight"] = dfw["Weight"].apply(lambda x: f"{x*100:.2f}%")
            st.table(dfw)

def compute_asset_returns(
    returns: pd.DataFrame,
    criteria: list[tuple[str, dict[str, float]]],
    initial_amount: float,
    acceptable_diff: float,
    fixed_cost: float,
    variable_cost: float,
    start: date,
    end: date
) -> dict[str, pd.Series]:
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
        dict[str, pd.Series] of returns for each asset and portfolio.
    """
    asset_returns = {asset: returns[asset] for asset in returns.columns}
    for title, weights in criteria:
        try:
            asset_returns[title] = backtesting(
                initial_amount=initial_amount,
                tickers=list(weights.keys()),
                weights=list(weights.values()),
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                acceptable_diff=acceptable_diff,
                fixed_cost=fixed_cost,
                variable_cost=variable_cost
            )[0]
        except Exception:
            asset_returns[title] = None
    return asset_returns

def show_summary(asset_returns: dict[str, pd.Series]):
    """
    Display summary statistics for each portfolio.

    Args:
        asset_returns: dict[str, pd.Series] of returns for each asset and portfolio.
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
        'Arithmetic Mean', 'Geometric Mean', 'Volatility', 'Max Drawdown',
        'Best Day', 'Worst Day', 'Best Year', 'Worst Year'
    ]
    fmt_dict = {col: "{:.2%}" for col in percent_cols if col in df.columns}
    styled = df.style.format(fmt_dict)
    st.subheader("Portfolio Metrics Table")
    st.dataframe(styled)

def plot_results(returns: pd.DataFrame, criteria: list[tuple[str, dict[str, float]]], solver: PortfolioSolver):
    """
    Plot wealth evolution and returns distribution for each portfolio.

    Args:
        returns: pd.DataFrame of asset returns.
        criteria: List of tuples (title, weights) where weights is a dict of asset weights.
        solver: PortfolioSolver instance.
    """
    st.subheader("Geometric vs Volatility")
    solver.plot_geometric_volatility_means()
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

def get_inputs() -> tuple[list[str], date, date, float, float, float, float, float, bool]:
    """
    Get user inputs from the sidebar.

    Returns:
        tuple of selected tickers, start date, end date, initial amount, acceptable difference, fixed cost, variable cost, cash interest rate, run button.
    """
    st.sidebar.header("Inputs")
    # Grouped asset selection by category
    selected_names: list[str] = []
    with st.sidebar.expander("Equity ETFs", expanded=True):
        sel_equity = st.multiselect("Equity ETFs", CATEGORIES["Equity ETFs"], default=["S&P 500 (VOO)", "Nasdaq (QQQ)", "SPDR Dow Jones (DIA)"])
        selected_names.extend(sel_equity)
    with st.sidebar.expander("Leveraged ETFs"):
        sel_lev = st.multiselect("Leveraged ETFs", CATEGORIES["Leveraged ETFs"])
        selected_names.extend(sel_lev)
    with st.sidebar.expander("Crypto"):
        sel_crypto = st.multiselect("Crypto", CATEGORIES["Crypto"])
        selected_names.extend(sel_crypto)
    with st.sidebar.expander("Commodities"):
        sel_comm = st.multiselect("Commodities ETFs", CATEGORIES["Commodities"], default=["SPDR Gold Trust (GLD)"])
        selected_names.extend(sel_comm)
    with st.sidebar.expander("VIX ETFs"):
        sel_vix = st.multiselect("VIX ETFs", CATEGORIES["VIX ETFs"])
        selected_names.extend(sel_vix)
    with st.sidebar.expander("Stocks"):
        sel_stocks = st.multiselect("Stocks", CATEGORIES["Stocks"])
        selected_names.extend(sel_stocks)
    selected = [TICKERS[name] for name in selected_names]
    custom_input = st.sidebar.text_input("Custom Ticker (symbol or name)")
    if custom_input:
        try:
            custom_symbol = resolve_ticker(custom_input)
            if custom_symbol not in selected:
                selected.append(custom_symbol)
        except ValueError as e:
            st.sidebar.error(str(e))
    start = st.sidebar.date_input("Start date", value=date(2020, 1, 1))
    end = st.sidebar.date_input("End date", value=date.today())
    initial_amount = st.sidebar.number_input("Initial amount", min_value=1000, value=10000, step=100)
    acceptable_diff = st.sidebar.number_input("Acceptable difference", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    fixed_cost = st.sidebar.number_input("Fixed cost", min_value=0.0, value=1.0, step=0.1)
    variable_cost = st.sidebar.number_input("Variable cost", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    run = st.sidebar.button("Calculate portfolios", type="primary")
    return selected, start, end, initial_amount, acceptable_diff, fixed_cost, variable_cost, run

def main():
    st.set_page_config(page_title="Geometric Portfolio Explorer")
    st.title("Geometric Portfolio Explorer")
    selected, start, end, initial_amount, acceptable_diff, fixed_cost, variable_cost, run = get_inputs()
    if not run:
        return
    if not selected:
        st.warning("Please select at least one asset.")
        return

    with st.spinner("Fetching data and running simulation..."):
        try:
            returns = get_returns(
                tickers=selected,
                start_date=start.isoformat(),
                end_date=end.isoformat()
            )
            solver = PortfolioSolver(returns)
            best_weights_geometric, best_weights_volatility, best_weights_alejandro = solver.run()
        except Exception as e:
            st.error(f"Error in optimization: {e}")
            return

    criteria = [
        ("Highest Geometric Mean", best_weights_geometric),
        ("Lowest Volatility", best_weights_volatility),
        ("Highest Alejandro Ratio", best_weights_alejandro)
    ]
    display_weights(criteria)

    asset_returns = compute_asset_returns(
        returns,
        criteria,
        initial_amount,
        acceptable_diff,
        fixed_cost,
        variable_cost,
        start,
        end
    )
    show_summary(asset_returns)
    plot_results(returns, criteria, solver)

    st.subheader("Leverage Recommendation")

    keys = [
        "Highest Geometric Mean",
        "Lowest Volatility",
        "Highest Alejandro Ratio"
    ]

    for key in keys:
        returns = asset_returns[key]
        show_leverage(returns, title=key, minimum_leverage=0.0, maximum_leverage=10.0)
    
    

if __name__ == "__main__":
    main()
