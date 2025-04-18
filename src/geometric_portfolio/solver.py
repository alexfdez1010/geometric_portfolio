import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from geometric_portfolio.metrics import geometric_mean, volatility, alejandro_ratio


class PortfolioSolver:
    """
    Numerical optimization solver for portfolio weights maximizing geometric mean,
    minimizing volatility, and maximizing Alejandro ratio.
    """
    returns: pd.DataFrame
    best_weights_geometric: dict[str, float] | None
    best_weights_volatility: dict[str, float] | None
    best_weights_alejandro: dict[str, float] | None

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.best_weights_geometric = None
        self.best_weights_volatility = None
        self.best_weights_alejandro = None

    def compute_returns(self, weights: dict[str, float]) -> pd.Series:
        weight_series = pd.Series(weights)
        common = set(self.returns.columns).intersection(weights.keys())
        filtered = self.returns[list(common)]
        w_filtered = weight_series[list(common)]
        return filtered.mul(w_filtered).sum(axis=1)

    def _possible_weights(self, assets: list[str]) -> dict[str, float]:
        random_vals = np.random.random(len(assets))
        norm = random_vals / np.sum(random_vals)
        return dict(zip(assets, norm))

    def run(self) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """
        Optimize portfolio weights under long-only constraints for three objectives:
        highest geometric mean, lowest volatility, highest Alejandro ratio.

        Returns:
            Tuple of best weights for each objective:
            - Best weights for highest geometric mean
            - Best weights for lowest volatility
            - Best weights for highest Alejandro ratio
        """
        assets = list(self.returns.columns)
        n = len(assets)
        x0 = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}

        def obj_geom(x):
            ret = self.compute_returns(dict(zip(assets, x)))
            return -geometric_mean(ret)

        def obj_vol(x):
            ret = self.compute_returns(dict(zip(assets, x)))
            return volatility(ret)

        def obj_alejandro(x):
            ret = self.compute_returns(dict(zip(assets, x)))
            return -alejandro_ratio(ret)

        sol_g = minimize(obj_geom, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        sol_v = minimize(obj_vol, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        sol_a = minimize(obj_alejandro, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        self.best_weights_geometric = dict(zip(assets, sol_g.x))
        self.best_weights_volatility = dict(zip(assets, sol_v.x))
        self.best_weights_alejandro = dict(zip(assets, sol_a.x))

        return self.best_weights_geometric, self.best_weights_volatility, self.best_weights_alejandro
    
    def plot_geometric_volatility_means(self) -> None:
        """
        Plot the best geometric mean, lowest volatility and highest Alejandro ratio portfolios
        and the assets returns in a geometric volatility space.
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
