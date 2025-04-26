# Geometric Portfolio

A Python library and Streamlit web application for constructing and analyzing portfolios using geometric mean optimization. The technique is based in given constant weights to assets and keep that weights constant with rebalancing when the difference between the current weights and the constant weights is higher than a threshold.

This work is inspired in the book **Safe Haven** of Mark Spitznagel, teaching me that geometric mean makes much more sense than using arithmetic mean like is used in Modern Portfolio Theory with the Efficient Frontier and so on.

## Why Geometric Mean is Superior for Portfolio Analysis

The geometric mean provides a more accurate representation of investment returns than the arithmetic mean because it accounts for the compounding effect of returns over time. This distinction is crucial in portfolio management:

Consider a portfolio with a 50% chance of gaining 200% and a 50% chance of losing 100%:

- Arithmetic mean: (200% - 100%)/2 = 50% (suggesting profitability)
- Geometric mean: √[(1+2.0)×(1-1.0)]-1 = 0% (indicating complete loss potential)

The arithmetic mean misleadingly suggests this is a profitable investment, while the geometric mean correctly shows that you could lose everything. This occurs because the arithmetic mean fails to account for the sequential nature of returns and their compounding effect.

Traditional portfolio theory focuses also on volatility as the primary risk measure, but this approach is fundamentally flawed. What truly matters to investors is not the day-to-day fluctuations but the maximum potential loss they might face - the maximum drawdown. A portfolio that experiences severe drawdowns requires exponentially larger gains to recover, which is why maximum drawdown provides a more meaningful risk assessment than standard deviation-based volatility measures.

By optimizing for geometric mean while controlling maximum drawdown, we create portfolios that can sustainably grow wealth over time, rather than portfolios that merely look good on paper but fail to deliver real-world results.

## The Role of Leverage

Leverage is a powerful tool that can be viewed as an extension of the Kelly Criterion—a formula used in betting to determine the optimal stake size. While Kelly traditionally suggests what percentage of your capital to allocate (between 0% and 100%), leverage expands this concept by allowing allocations outside these boundaries. With leverage, you can:

1. Allocate more than 100% of your capital (positive leverage > 1)
2. Allocate less than 100% but greater than 0% (fractional leverage between 0 and 1)
3. Bet against an asset with negative leverage (< 0)

This flexibility enables investors to maximize geometric mean returns across a broader spectrum of allocation possibilities, potentially improving long-term growth rates while managing risk. Like Kelly, the optimal leverage point represents the sweet spot where growth is maximized, but with more strategic options available.

Leverage, while powerful, is not without risks. It amplifies each individual outcome rather than simply multiplying final returns. Due to the compounding nature of geometric returns, sharp declines are severely penalized - losing 50% requires a subsequent 100% gain just to break even, not a 50% gain as arithmetic intuition might suggest. When leveraged, these asymmetries become even more pronounced.

The optimization approach helps identify optimal leverage levels that maximize geometric growth, but investors must remain cautious of margin calls and other practical constraints. The fundamental mathematical reality remains: losses and gains behave asymmetrically under compounding, making proper leverage sizing critical to long-term portfolio survival and growth.

## Repository Structure

```text
geometric_portfolio/
├── src/geometric_portfolio
│   ├── Geometric_Mean.py       # Compute geometric mean of return series
│   ├── backtesting.py          # Backtest portfolio and trading strategies
│   ├── data.py                 # Data loading from CSV and yfinance
│   ├── leverage.py             # Optimal leverage calculation
│   ├── metrics.py              # Performance metrics (Sharpe, Calmar, etc.)
│   ├── plot.py                 # Matplotlib and Streamlit chart helpers
│   ├── solver.py               # Optimization routines
│   ├── st_shared.py            # Shared Streamlit components (e.g., show_leverage)
│   ├── tickers.py              # Ticker utilities and validation
│   └── pages
│       ├── Custom_Portfolio.py # Streamlit app for custom ticker portfolio
│       └── Leverage_Optimizer.py # Streamlit app for leverage optimization
├── tests
│   ├── test_backtesting.py     # Tests for backtesting logic
│   ├── test_metrics.py         # Tests for metric calculations
│   └── test_data.py            # Tests for data loading functionality
├── pyproject.toml              # Project metadata and dependencies
├── Makefile                    # Common tasks (test, lint, run)
└── README.md                   # (this file)
```

## Key Components

### Core Modules

- **Geometric_Mean.py**: Calculates the geometric mean of return series, central to portfolio growth estimation.
- **backtesting.py**: Provides functions to simulate portfolio performance over historical data.
- **data.py**: Handles data ingestion from CSV files or Yahoo Finance via `yfinance`.
- **leverage.py**: Generates a table of geometric mean, volatility, and Calmar ratio across leverage levels.
- **metrics.py**: Defines functions for Sharpe ratio, Calmar ratio, drawdown, and other performance metrics.
- **solver.py**: Wraps SciPy optimizers to maximize according to different goals.
- **plot.py**: Contains utilities to render charts in both Matplotlib and Streamlit.
- **tickers.py**: Validates and manages lists of tickers for portfolio construction.
- **st_shared.py**: Shared Streamlit functions, e.g., `show_leverage` to display optimizer results.

### Streamlit Web Application

The web app has three pages:

1. **Geometric Mean** (`Geometric_Mean.py`)
   - UI to select assets and time period.
   - Displays the best portfolio allocation for geometric mean, volatility, and Calmar ratio.
   - Makes a real simulation of the portfolio performance.
   - Note: Slippage cost is not taken into account.

2. **Custom Portfolio** (`Custom_Portfolio.py`)
   - Similar working to Geometric Mean page, but allows to select custom weights for the assets.

3. **Leverage Optimizer** (`Leverage_Optimizer.py`)
   - Leverage in this case is like an extension of Kelly Criterion that can be less than 0 or higher than 1.
   - This allow to compute the optimal allocation of total wealth to the portfolio to maximize the wealth growth.
   - The computations are in **ideal conditions**, so check use the other two pages to get a better understanding of the real performance.
   - The idea of this page is to be used as a guide for selecting interesting assets and weights to be included in the portfolio.

## Installation

```bash
git clone https://github.com/alexfdez1010/geometric_portfolio.git
cd geometric_portfolio
uv sync
```

## Usage

- **Web App**:

  ```bash
  make app
  ```

- **Library**:

  ```python
  from geometric_portfolio.backtesting import backtest_portfolio
  from geometric_portfolio.leverage import leverage_optimizer
  ```

- **CLI Tasks**:

  ```bash
  uv run pytest        # run tests
  uv run ruff check    # lint
  uv run ruff format   # format code
  ```

## Development

Test the code:

```bash
make test
```

Lint the code:

```bash
make lint
```

Format the code:

```bash
make format
```

Pre-commit (combines test, lint and format):

```bash
make pre-commit
```

## Disclaimer

This work is not a financial advice and should not be used as a basis for making investment decisions. The author is not liable for any losses or damages that may result from the use of this work.

## License

You can use this code for personal use, but not for commercial purposes. Check the [LICENSE](LICENSE) file for more details.
