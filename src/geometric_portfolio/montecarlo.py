from typing import cast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geometric_portfolio.metrics import geometric_mean, volatility, alejandro_ratio

class MonteCarlo:
    """
    Monte Carlo simulation for finding best weights for a portfolio maximizing geometric mean return.
    """
    returns: pd.DataFrame
    best_weights_geometric: dict[str, float] | None
    best_weights_volatility: dict[str, float] | None
    best_weights_alejandro: dict[str, float] | None
    last_results: pd.DataFrame

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.best_weights_geometric = None
        self.best_weights_volatility = None
        self.best_weights_alejandro = None
        self.last_results = pd.DataFrame()
        self.number_of_assets = len(returns.columns)
        
    def run(self, num_simulations: int = 10000) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """
        Run Monte Carlo simulation to find best weights for a portfolio maximizing geometric mean return.

        Args:
            num_simulations: Number of simulations to run.

        Returns:
            tuple[dict[str, float], dict[str, float], dict[str, float]]: Tuple containing the best weights found for the portfolio (geometric mean, volatility and alejandro ratio)
        """

        assets = list(self.returns.columns)
        
        # Run Monte Carlo simulation
        for i in range(1, num_simulations + 1):  
            
            if i % 1000 == 0:
                print(f"Simulation {i} of {num_simulations}")

            weights = self._possible_weights(assets)
            returns = self.compute_returns(weights)

            last_simulation = pd.DataFrame([{
                **{asset: weights[asset] for asset in assets},
                "geometric_mean": geometric_mean(returns),
                "volatility": volatility(returns),
                "alejandro_ratio": alejandro_ratio(returns)
            }])
            
            self.last_results = pd.concat([self.last_results, last_simulation], ignore_index=True)
        
        best_index_geometric = self.last_results["geometric_mean"].idxmax()
        self.best_weights_geometric = cast(dict[str, float], self.last_results.loc[best_index_geometric].to_dict())
        
        best_index_volatility = self.last_results["volatility"].idxmin()
        self.best_weights_volatility = cast(dict[str, float], self.last_results.loc[best_index_volatility].to_dict())

        best_index_alejandro = self.last_results["alejandro_ratio"].idxmax()
        self.best_weights_alejandro = cast(dict[str, float], self.last_results.loc[best_index_alejandro].to_dict())
        
        return self.best_weights_geometric, self.best_weights_volatility, self.best_weights_alejandro
    
    def compute_returns(self, weights: dict[str, float]) -> pd.Series:
        """
        Compute the returns for a given set of weights.

        Args:
            weights: Dictionary containing the weights for each asset.

        Returns:
            pd.Series: Returns for the given set of weights.
        """
        # Convert weights to a Series aligned with returns columns
        weight_series = pd.Series(weights)
        
        # Filter to include only assets in the weights dictionary
        common_assets = set(self.returns.columns).intersection(weights.keys())
        
        # Use vectorized operations for efficiency
        filtered_returns = self.returns[list(common_assets)]
        filtered_weights = weight_series[list(common_assets)]
        
        # Matrix multiplication of returns and weights
        portfolio_returns = filtered_returns.mul(filtered_weights).sum(axis=1)
        
        return portfolio_returns
    
    def _possible_weights(self, assets: list[str]) -> dict[str, float]:
        """
        Generate a possible weight for a portfolio with a given number of assets.

        Args:
            assets: List of assets in the portfolio.

        Returns:
            dict[str, float]: Dictionary containing a possible weight for the portfolio.
        """
        
        # Generate random weights and normalize them to sum to 1
        random_values = np.random.random(len(assets))
        normalized_weights = random_values / np.sum(random_values)
        
        # Create dictionary mapping assets to weights
        weights = dict(zip(assets, normalized_weights))
        
        return weights
    
    def plot_geometric_arithmetic_means(self) -> None:
        """
        Plot the geometric and arithmetic means of the last simulation results.
        It will include the assets results and the best results of
        highest geometric mean, lowest volatility and highest alejandro ratio.
        """
        
        # Compute best geometric mean and lowest volatility portfolios
        best_geometric = self.best_weights_geometric
        best_volatility = self.best_weights_volatility
        best_alejandro = self.best_weights_alejandro

        returns_geometric = self.compute_returns(best_geometric)
        returns_volatility = self.compute_returns(best_volatility)
        returns_alejandro = self.compute_returns(best_alejandro)

        # Plot assets returns
        plt.figure(figsize=(10, 6))
        for asset in self.returns.columns:
            returns = self.compute_returns({asset: 1.0})
            plt.scatter(
                volatility(returns) * 100,
                geometric_mean(returns) * 100,
                marker='o',
                s=100,
                alpha=0.5,
                label=asset
            )
        volatility_geometric, geometric_mean_geometric = volatility(returns_geometric) * 100, geometric_mean(returns_geometric) * 100
        volatility_volatility, geometric_mean_volatility = volatility(returns_volatility) * 100, geometric_mean(returns_volatility) * 100
        volatility_alejandro, geometric_mean_alejandro = volatility(returns_alejandro) * 100, geometric_mean(returns_alejandro) * 100

        # Create scatter plot of volatility vs geometric mean
        plt.scatter(
            volatility_geometric,
            geometric_mean_geometric,
            marker='*',
            s=300,
            color='red',
            label='Best Geometric Mean'
        )
        plt.scatter(
            volatility_volatility,
            geometric_mean_volatility,
            marker='d',
            s=200,
            color='blue',
            label='Lowest Volatility'
        )
        plt.scatter(
            volatility_alejandro,
            geometric_mean_alejandro,
            marker='s',
            s=200,
            color='green',
            label='Highest Alejandro Ratio'
        )

        plt.xlabel('Volatility (%)')
        plt.ylabel('Geometric Mean Return (%)')
        plt.title('Portfolio Optimization: Geometric Mean vs Volatility')
        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()