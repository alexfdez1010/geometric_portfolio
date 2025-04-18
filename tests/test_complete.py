import pandas as pd
from geometric_portfolio.data import get_returns
from geometric_portfolio.montecarlo import MonteCarlo
from geometric_portfolio.metrics import summary

def test_integration_full_flow():
    # Select assets
    tickers = ["AAPL", "MSFT", "GC=F", "BTC-USD"]
    # Download daily returns for the last year
    returns = get_returns(tickers, start_date="2015-01-01")
    assert isinstance(returns, pd.DataFrame)
    assert set(tickers).issubset(returns.columns)
    assert not returns.empty

    # Run Monte Carlo simulation
    mc = MonteCarlo(returns)
    best_weights = mc.run(num_simulations=500)  # Use 500 for a quick test
    assert isinstance(best_weights, dict)
    print("Best weights found:", best_weights)

    # Compute returns for best weights
    best_returns = mc.compute_returns({k: best_weights[k] for k in tickers})

    # Show metrics summary
    print("\nPortfolio summary for best weights:")
    summary_series = summary(best_returns)
    print(summary_series)
