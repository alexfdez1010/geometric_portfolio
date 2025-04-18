import argparse
from src.geometric_portfolio.data import get_returns
from src.geometric_portfolio.montecarlo import MonteCarlo
from src.geometric_portfolio.metrics import summary, wealth
import sys
import pandas as pd
from colorama import Fore, Style, init
import matplotlib.pyplot as plt

def display_summary_table(asset_returns_dict):
    init(autoreset=True)
    summary_rows = []
    for name, ret in asset_returns_dict.items():
        s = summary(ret)
        s = s.rename(name)
        summary_rows.append(s)
    df = pd.DataFrame(summary_rows)

    # Format as percentage where applicable
    percent_cols = [
        'Arithmetic Mean', 'Geometric Mean', 'Volatility', 'Max Drawdown',
        'Best Day', 'Worst Day', 'Best Year', 'Worst Year'
    ]
    def fmt(val, col):
        if col in percent_cols:
            return f"{val * 100:.2f}%"
        return f"{val:.6g}" if isinstance(val, float) else str(val)

    df_fmt = df.copy()
    for col in percent_cols:
        if col in df_fmt.columns:
            df_fmt[col] = df_fmt[col].apply(lambda x: fmt(x, col))
    if 'Sharpe Ratio' in df_fmt.columns:
        df_fmt['Sharpe Ratio'] = df_fmt['Sharpe Ratio'].apply(lambda x: fmt(x, 'Sharpe Ratio'))

    # Use color for best portfolio row
    if 'Best Portfolio' in df_fmt.index:
        table_str = df_fmt.to_markdown(tablefmt="github").splitlines()
        for i, row in enumerate(table_str):
            if row.strip().startswith("| Best Portfolio "):
                table_str[i] = Fore.GREEN + row + Style.RESET_ALL
        table_output = "\n".join(table_str)
    else:
        table_output = df_fmt.to_markdown(tablefmt="github")
    print("\n" + "="*60)
    print(Fore.CYAN + Style.BRIGHT + "Portfolio Metrics Table" + Style.RESET_ALL)
    print("="*60)
    print(table_output)
    print("="*60 + "\n")



def main():
    parser = argparse.ArgumentParser(description="Get daily returns for a list of assets.")
    parser.add_argument(
        "--tickers", "-t",
        required=True,
        nargs='+',
        help="List of asset tickers (space separated, e.g. --tickers AAPL MSFT BTC-USD)"
    )
    parser.add_argument(
        "--start_date", "-s",
        required=False,
        default=None,
        help="Start date in YYYY-MM-DD format (optional)"
    )
    parser.add_argument(
        "--end_date", "-e",
        required=False,
        default=None,
        help="End date in YYYY-MM-DD format (optional)"
    )
    parser.add_argument(
        "--num_simulations", "-n",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations to run (default: 10000)"
    )
    args = parser.parse_args()

    try:
        returns = get_returns(
            tickets=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date
        )

        # Monte Carlo simulation to find best weights
        mc = MonteCarlo(returns)
        print(f"\nRunning Monte Carlo simulation to find best weights... (n={args.num_simulations})")
        best_weights = mc.run(num_simulations=args.num_simulations)
        print("\nBest weights:")
        for asset, weight in best_weights.items():
            if asset in returns.columns:
                print(f"  {asset}: {weight*100:.2f}%")

        # Compute returns for best weights and for each asset
        best_returns = mc.compute_returns(best_weights)
        asset_returns_dict = {asset: returns[asset] for asset in returns.columns}
        asset_returns_dict['Best Portfolio'] = best_returns

        # Display summary table
        display_summary_table(asset_returns_dict)

        # Show plot
        mc.plot_geometric_arithmetic_means(k=10)

        # Plot wealth evolution for each assets and best portfolio
        wealths = {asset: wealth(returns[asset]) for asset in returns.columns}
        wealths['Best Portfolio'] = wealth(best_returns)
        wealths_df = pd.DataFrame(wealths)
        wealths_df.plot()
        plt.title('Wealth Evolution')
        plt.xlabel('Time')
        plt.ylabel('Wealth')
        plt.grid(True)
        plt.show()

    except ValueError as ve:
        print(f"Error: {ve}", file=sys.stderr)
        sys.exit(1)
    except Exception as ex:
        print(f"Unexpected error: {ex}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

