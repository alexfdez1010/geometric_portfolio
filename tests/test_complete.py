import pandas as pd
from geometric_portfolio.data import get_returns
from geometric_portfolio.solver import PortfolioSolver
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
    solver = PortfolioSolver(returns)
    best_geometric, best_volatility, best_alejandro = solver.run()  
    assert isinstance(best_geometric, dict)
    assert isinstance(best_volatility, dict)
    assert isinstance(best_alejandro, dict)
    print("Best geometric weights found:", best_geometric)
    print("Best volatility weights found:", best_volatility)
    print("Best alejandro weights found:", best_alejandro)

    # Compute returns for best geometric weights
    best_returns = solver.compute_returns({k: best_geometric[k] for k in tickers})

    # Show metrics summary
    print("\nPortfolio summary for best weights:")
    summary_series = summary(best_returns)
    print(summary_series)
