import streamlit as st
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt

from geometric_portfolio.data import get_returns
from geometric_portfolio.montecarlo import MonteCarlo
from geometric_portfolio.metrics import summary, wealth

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
    num_sim = st.sidebar.number_input(
        "Number of simulations", min_value=1000, max_value=20000, value=10000, step=1000
    )
    run = st.sidebar.button("Run Simulation")
    return selected, start, end, num_sim, run

def main():
    st.set_page_config(page_title="Geometric Portfolio Explorer")
    st.title("Geometric Portfolio Explorer")
    selected, start, end, num_sim, run = get_inputs()
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
        mc = MonteCarlo(returns)
        best_weights = mc.run(num_simulations=int(num_sim))

    # Display best weights
    st.subheader("Best Portfolio Weights")
    df_weights = pd.DataFrame.from_dict(best_weights, orient="index", columns=["Weight"])
    df_weights["Weight"] = df_weights["Weight"].apply(lambda x: f"{x*100:.2f}%")
    df_weights = df_weights.drop(["arithmetic_mean", "geometric_mean"], errors="ignore")
    st.table(df_weights)

    # Summary table
    asset_returns = {asset: returns[asset] for asset in returns.columns}
    asset_returns["Best Portfolio"] = mc.compute_returns(best_weights)
    rows = []
    for name, ret in asset_returns.items():
        s = summary(ret).rename(name)
        rows.append(s)
    df_summary = pd.DataFrame(rows)
    percent_cols = [
        'Arithmetic Mean', 'Geometric Mean', 'Volatility', 'Max Drawdown',
        'Best Day', 'Worst Day', 'Best Year', 'Worst Year', 'Sharpe Ratio'
    ]
    fmt_dict = {col: "{:.2%}" for col in percent_cols if col in df_summary.columns}
    styled = df_summary.style.format(fmt_dict)
    st.subheader("Portfolio Metrics Table")
    st.table(styled)

    # Geometric vs Arithmetic Mean plot
    st.subheader("Geometric vs Arithmetic Mean")
    mc.plot_geometric_arithmetic_means(k=10)
    st.pyplot(plt.gcf())

    # Wealth evolution plot
    st.subheader("Wealth Evolution")
    wealth_dict = {asset: wealth(returns[asset]) for asset in returns.columns}
    wealth_dict["Best Portfolio"] = wealth(mc.compute_returns(best_weights))
    df_wealth = pd.DataFrame(wealth_dict)
    fig, ax = plt.subplots()
    df_wealth.plot(ax=ax)
    ax.set_title("Wealth Evolution")
    ax.set_xlabel("Time")
    ax.set_ylabel("Wealth")
    ax.grid(True)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
