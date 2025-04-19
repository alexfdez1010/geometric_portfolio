import streamlit as st
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt

from geometric_portfolio.plot import plot_wealth_evolution, plot_returns_distribution
from geometric_portfolio.metrics import wealth
from geometric_portfolio.data import get_returns
from geometric_portfolio.solver import PortfolioSolver
from geometric_portfolio.backtesting import backtesting
from geometric_portfolio.metrics import summary

# Available tickers: map full names to symbols
TICKERS = {
    "S&P 500 (VOO)": "VOO",
    "Nasdaq (QQQ)": "QQQ",
    "SPDR Dow Jones (DIA)": "DIA",
    "SPDR Gold Trust (GLD)": "GLD",
    "iShares Silver Trust (SLV)": "SLV",
    "S&P 500 VIX Short-term Futures Index (VIXL.L)": "VIXL.L",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "United States Oil (USO)": "USO",
    "iShares 20+ Year Treasury (TLT)": "TLT",
    "iShares 7-10 Year Treasury (IEF)": "IEF",
    "iPath S&P 500 VIX (VXX)": "VXX",
    "Apple Inc. (AAPL)": "AAPL",
    "Microsoft Corp. (MSFT)": "MSFT",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "NVIDIA Corp. (NVDA)": "NVDA",
    "Tesla Inc. (TSLA)": "TSLA",
    "Meta Platforms Inc. (META)": "META",
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "Walmart Inc. (WMT)": "WMT",
    "Invesco S&P 500 Equal Weight ETF (RSP)": "RSP",
    "First Trust NASDAQ-100 Equal Weighted Index Fund (QQEW)": "QQEW",
    "Invesco S&P SmallCap 600 Equal Weight ETF (EWSC)": "EWSC",
    "Invesco S&P MidCap 400 Equal Weight ETF (EWMC)": "EWMC",
    "ProShares VIX Short-Term Futures ETF (VIXY)": "VIXY",
    "ProShares VIX Mid-Term Futures ETF (VIXM)": "VIXM",
    "ProShares Ultra VIX Short-Term Futures ETF (UVXY)": "UVXY",
}

def get_inputs():
    st.sidebar.header("Inputs")
    selected_names = st.sidebar.multiselect(
        "Select assets",
        list(TICKERS.keys()),
        default=["S&P 500 (VOO)", "Nasdaq (QQQ)", "SPDR Dow Jones (DIA)", "SPDR Gold Trust (GLD)"]
    )
    selected = [TICKERS[name] for name in selected_names]
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
        returns = get_returns(
            tickets=selected,
            start_date=start.isoformat(),
            end_date=end.isoformat()
        )
        solver = PortfolioSolver(returns)
        best_weights_geometric, best_weights_volatility, best_weights_alejandro = solver.run()

    # Display best weights
    st.subheader("Best Portfolio Weights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Highest Geometric Mean")
        df_weights = pd.DataFrame.from_dict(best_weights_geometric, orient="index", columns=["Weight"])
        df_weights["Weight"] = df_weights["Weight"].apply(lambda x: f"{x*100:.2f}%")
        df_weights = df_weights.drop(["geometric_mean", "volatility", "alejandro_ratio"], errors="ignore")
        st.table(df_weights)
    with col2:
        st.write("Lowest Volatility")
        df_weights = pd.DataFrame.from_dict(best_weights_volatility, orient="index", columns=["Weight"])
        df_weights["Weight"] = df_weights["Weight"].apply(lambda x: f"{x*100:.2f}%")
        df_weights = df_weights.drop(["geometric_mean", "volatility", "alejandro_ratio"], errors="ignore")
        st.table(df_weights)
    with col3:
        st.write("Highest Alejandro Ratio")
        df_weights = pd.DataFrame.from_dict(best_weights_alejandro, orient="index", columns=["Weight"])
        df_weights["Weight"] = df_weights["Weight"].apply(lambda x: f"{x*100:.2f}%")
        df_weights = df_weights.drop(["geometric_mean", "volatility", "alejandro_ratio"], errors="ignore")
        st.table(df_weights)

    # Summary table
    asset_returns = {asset: returns[asset] for asset in returns.columns}
    asset_returns["Geometric Mean"] = backtesting(
        initial_amount=initial_amount,
        tickers=list(best_weights_geometric.keys()), 
        weights=list(best_weights_geometric.values()), 
        start_date=start.isoformat(), 
        end_date=end.isoformat(), 
        acceptable_diff=acceptable_diff, 
        fixed_cost=fixed_cost, 
        variable_cost=variable_cost
    )[0]
    
    asset_returns["Volatility"] = backtesting(
        initial_amount=initial_amount,
        tickers=list(best_weights_volatility.keys()), 
        weights=list(best_weights_volatility.values()), 
        start_date=start.isoformat(), 
        end_date=end.isoformat(), 
        acceptable_diff=acceptable_diff, 
        fixed_cost=fixed_cost, 
        variable_cost=variable_cost
    )[0]
    
    asset_returns["Alejandro Ratio"] = backtesting(
        initial_amount=initial_amount,
        tickers=list(best_weights_alejandro.keys()), 
        weights=list(best_weights_alejandro.values()), 
        start_date=start.isoformat(), 
        end_date=end.isoformat(), 
        acceptable_diff=acceptable_diff, 
        fixed_cost=fixed_cost, 
        variable_cost=variable_cost
    )[0]

    print(asset_returns["Geometric Mean"])
    print(asset_returns["Volatility"])
    print(asset_returns["Alejandro Ratio"])
    
    rows = []
    for name, ret in asset_returns.items():
        s = summary(ret).rename(name)
        rows.append(s)
    df_summary = pd.DataFrame(rows)
    percent_cols = [
        'Arithmetic Mean', 'Geometric Mean', 'Volatility', 'Max Drawdown',
        'Best Day', 'Worst Day', 'Best Year', 'Worst Year'
    ]
    fmt_dict = {col: "{:.2%}" for col in percent_cols if col in df_summary.columns}
    styled = df_summary.style.format(fmt_dict)
    st.subheader("Portfolio Metrics Table")
    st.dataframe(styled)

    # Geometric vs Volatility plot
    st.subheader("Geometric vs Volatility")
    solver.plot_geometric_volatility_means()
    st.pyplot(plt.gcf())

    returns_geometric = solver.compute_returns(best_weights_geometric)
    returns_volatility = solver.compute_returns(best_weights_volatility)
    returns_alejandro = solver.compute_returns(best_weights_alejandro)

    # Wealth evolution plot
    st.subheader("Wealth Evolution")
    wealth_dict = {asset: wealth(returns[asset]) for asset in returns.columns}
    wealth_dict["Geometric Mean"] = wealth(returns_geometric)
    wealth_dict["Lowest Volatility"] = wealth(returns_volatility)
    wealth_dict["Highest Alejandro Ratio"] = wealth(returns_alejandro)
    fig = plot_wealth_evolution(wealth_dict)
    st.pyplot(fig)

    # Returns distribution
    st.subheader("Returns Distribution")
    returns_dict = {
        **{asset: returns[asset] for asset in returns.columns},
        "Geometric Mean": returns_geometric,
        "Lowest Volatility": returns_volatility,
        "Highest Alejandro Ratio": returns_alejandro
    }
    fig = plot_returns_distribution(returns_dict)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
