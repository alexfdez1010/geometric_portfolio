from typing import cast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geometric_portfolio.metrics import geometric_mean, arithmetic_mean

class MonteCarlo:
    """
    Monte Carlo simulation for finding best weights for a portfolio maximizing geometric mean return.
    """
    returns: pd.DataFrame
    best_weights: dict[str, float] | None
    last_results: pd.DataFrame

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.best_weights = None
        self.last_results = pd.DataFrame()
        self.number_of_assets = len(returns.columns)
        
    def run(self, num_simulations: int = 10000) -> dict[str, float]:
        """
        Run Monte Carlo simulation to find best weights for a portfolio maximizing geometric mean return.

        Args:
            num_simulations: Number of simulations to run.

        Returns:
            dict[str, float]: Dictionary containing the best weights found for the portfolio.
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
                "arithmetic_mean": arithmetic_mean(returns),
                "geometric_mean": geometric_mean(returns)
            }])
            
            self.last_results = pd.concat([self.last_results, last_simulation], ignore_index=True)
        
        best_index = self.last_results["geometric_mean"].idxmax()
        self.best_weights = cast(dict[str, float], self.last_results.loc[best_index].to_dict())
        
        return self.best_weights
    
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
    
    def plot_geometric_arithmetic_means(self, k: int = 10) -> None:
        """
        Plot the geometric and arithmetic means of the last simulation results.
        It will include the assets results and the k best results according to 
        the geometric mean.

        Args:
            k: Number of top results to display.
        """
        
        # Sort results by geometric mean
        sorted_results = self.last_results.sort_values('geometric_mean', ascending=False)
        
        # Get top k results
        top_k = sorted_results.head(k)
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        
        # Plot top k results
        plt.scatter(
            top_k['arithmetic_mean'] * 100, 
            top_k['geometric_mean'] * 100, 
            color='green', 
            s=100, 
            label=f'Top {k} Portfolios'
        )
        
        # Plot individual assets
        asset_columns = [col for col in self.returns.columns]
        for asset in asset_columns:
            # Create a portfolio with 100% in this asset
            asset_returns = cast(pd.Series, self.returns[asset])
            arith_mean = arithmetic_mean(asset_returns) * 100
            geo_mean = geometric_mean(asset_returns) * 100
            plt.scatter(arith_mean, geo_mean, s=100, label=asset)
        
        # Highlight the best combination with a star
        best_index = sorted_results['geometric_mean'].idxmax()
        best_portfolio = sorted_results.loc[best_index]
        plt.scatter(
            best_portfolio['arithmetic_mean'] * 100,
            best_portfolio['geometric_mean'] * 100,
            marker='*', 
            s=300, 
            color='red',
            label='Best Portfolio'
        )
        
        # Create annotation text with weights for best portfolio
        weight_text = "Best Portfolio Weights:\n"
        for asset in asset_columns:
            if asset in best_portfolio and best_portfolio[asset] > 0:
                weight_text += f"{asset}: {best_portfolio[asset]*100:.1f}%\n"
        
        # Add annotation for best portfolio
        plt.annotate(
            weight_text, 
            xy=(best_portfolio['arithmetic_mean'] * 100, best_portfolio['geometric_mean'] * 100),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7)
        )
        
        # Add labels and title
        plt.xlabel('Arithmetic Mean Return (%)')
        plt.ylabel('Geometric Mean Return (%)')
        plt.title('Portfolio Optimization Results')
        plt.grid(True)
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
        

        
        
        
        
        
        
        
    