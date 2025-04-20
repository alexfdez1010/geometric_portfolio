import streamlit as st
from datetime import date

from geometric_portfolio.Geometric_Mean import show_summary
from geometric_portfolio.backtesting import backtesting
from geometric_portfolio.metrics import wealth
from geometric_portfolio.plot import plot_wealth_evolution, plot_returns_distribution
from geometric_portfolio.tickers import TICKERS
from geometric_portfolio.data import get_returns

def main():
    st.set_page_config(page_title="Custom Portfolio")
    st.title("Custom Portfolio Backtesting")

    # Sidebar inputs
    st.sidebar.header("Custom Portfolio Inputs")
    selected_names = st.sidebar.multiselect("Select Assets", list(TICKERS.keys()), default=list(TICKERS.keys())[:3])
    selected = [TICKERS[name] for name in selected_names]
    # Weight assignment
    weights = []
    if selected:
        st.sidebar.subheader("Assign Weights")
        for name in selected_names:
            w = st.sidebar.number_input(
                f"Weight for {name}", min_value=0.0, max_value=1.0,
                value=round(1.0/len(selected_names), 2), step=0.01, format="%.2f"
            )
            weights.append(w)
        total = sum(weights)
        if total > 0:
            weights = [w/total for w in weights]

    start = st.sidebar.date_input("Start date", value=date(2020, 1, 1))
    end = st.sidebar.date_input("End date", value=date.today())
    initial_amount = st.sidebar.number_input("Initial amount", min_value=1000.0, value=10000.0, step=100.0)
    acceptable_diff = st.sidebar.number_input("Acceptable difference", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    fixed_cost = st.sidebar.number_input("Fixed cost", min_value=0.0, value=1.0, step=0.1)
    variable_cost = st.sidebar.number_input("Variable cost", min_value=0.0, value=0.0, step=0.01)
    run = st.sidebar.button("Run Backtest")

    if not run:
        st.stop()
    if not selected:
        st.warning("Please select at least one asset.")
        st.stop()

    with st.spinner("Running backtest..."):
        asset_returns = get_returns(tickers=selected, start_date=start.isoformat(), end_date=end.isoformat())
        strategy_returns, _ = backtesting(
            initial_amount=initial_amount,
            tickers=selected,
            weights=weights,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            acceptable_diff=acceptable_diff,
            fixed_cost=fixed_cost,
            variable_cost=variable_cost
        )

        wealth_strategy = wealth(strategy_returns)

        st.subheader("Wealth Evolution")
        fig_w = plot_wealth_evolution(
            {
                "Custom Portfolio": wealth_strategy,
                **{asset: wealth(asset_returns[asset]) for asset in asset_returns.columns}
            }
        )
        st.pyplot(fig_w)

        st.subheader("Returns Distribution")
        fig_r = plot_returns_distribution(
            {
                "Custom Portfolio": strategy_returns,
                **{asset: asset_returns[asset] for asset in asset_returns.columns}
            }
        )
        st.pyplot(fig_r)

        # Display metrics
        st.subheader("Portfolio Metrics")
        df = {
            "Custom Portfolio": strategy_returns,
            **{asset: asset_returns[asset] for asset in asset_returns.columns}
        }
        show_summary(df)

if __name__ == "__main__":
    main()
