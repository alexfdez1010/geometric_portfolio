import streamlit as st
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt

from geometric_portfolio.data import get_returns
from geometric_portfolio.solver import PortfolioSolver
from geometric_portfolio.metrics import summary, wealth
from geometric_portfolio.plot import plot_wealth_evolution, plot_returns_distribution

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
    run = st.sidebar.button("Calculate portfolios", type="primary")
    return selected, start, end, run

def main():
    st.set_page_config(page_title="Geometric Portfolio Explorer")
    st.title("Geometric Portfolio Explorer")
    selected, start, end, run = get_inputs()
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
    asset_returns["Best Portfolio (Geometric Mean)"] = solver.compute_returns(best_weights_geometric)
    asset_returns["Best Portfolio (Lowest Volatility)"] = solver.compute_returns(best_weights_volatility)
    asset_returns["Best Portfolio (Highest Alejandro Ratio)"] = solver.compute_returns(best_weights_alejandro)
    
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
    wealth_dict["Best Portfolio (Geometric Mean)"] = wealth(returns_geometric)
    wealth_dict["Best Portfolio (Lowest Volatility)"] = wealth(returns_volatility)
    wealth_dict["Best Portfolio (Highest Alejandro Ratio)"] = wealth(returns_alejandro)
    fig = plot_wealth_evolution(wealth_dict)
    st.pyplot(fig)

    # Returns distribution
    st.subheader("Returns Distribution")
    returns_dict = {
        **{asset: returns[asset] for asset in returns.columns},
        "Best Portfolio (Geometric Mean)": returns_geometric,
        "Best Portfolio (Lowest Volatility)": returns_volatility,
        "Best Portfolio (Highest Alejandro Ratio)": returns_alejandro
    }
    fig = plot_returns_distribution(returns_dict)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
