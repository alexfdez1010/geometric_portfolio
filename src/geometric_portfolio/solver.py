import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from geometric_portfolio.metrics import geometric_mean, calmar_ratio, max_drawdown


class PortfolioSolver:
    """
    Numerical optimization solver for portfolio weights maximizing geometric mean,
    minimizing max drawdown, and maximizing Calmar ratio.
    """

    returns: pd.DataFrame
    best_weights_geometric: dict[str, float] | None
    best_weights_max_drawdown: dict[str, float] | None
    best_weights_calmar: dict[str, float] | None

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.best_weights_geometric = None
        self.best_weights_max_drawdown = None
        self.best_weights_calmar = None

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
        highest geometric mean, lowest volatility, highest Calmar ratio.

        Returns:
            Tuple of best weights for each objective:
            - Best weights for highest geometric mean
            - Best weights for lowest volatility
            - Best weights for highest Calmar ratio
        """
        assets = list(self.returns.columns)
        n = len(assets)
        x0 = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

        def obj_geom(x):
            ret = self.compute_returns(dict(zip(assets, x)))
            return -geometric_mean(ret)

        def obj_max_drawdown(x):
            ret = self.compute_returns(dict(zip(assets, x)))
            return max_drawdown(ret)

        def obj_calmar(x):
            ret = self.compute_returns(dict(zip(assets, x)))
            return -calmar_ratio(ret)

        sol_g = minimize(
            obj_geom, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )
        sol_v = minimize(
            obj_max_drawdown, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )
        sol_c = minimize(
            obj_calmar, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        self.best_weights_geometric = dict(zip(assets, sol_g.x))
        self.best_weights_max_drawdown = dict(zip(assets, sol_v.x))
        self.best_weights_calmar = dict(zip(assets, sol_c.x))

        return (
            self.best_weights_geometric,
            self.best_weights_max_drawdown,
            self.best_weights_calmar,
        )

    def plot_geometric_max_drawdown(self) -> None:
        """
        Plot the best geometric mean and lowest max drawdown portfolios
        and the assets returns in a geometric max drawdown space.
        """
        if (
            self.best_weights_geometric is None
            or self.best_weights_max_drawdown is None
            or self.best_weights_calmar is None
        ):
            raise ValueError("Best weights must be computed first, use run() method.")

        # Compute best geometric mean and lowest max drawdown portfolios
        best_geometric = self.best_weights_geometric
        best_max_drawdown = self.best_weights_max_drawdown
        best_calmar = self.best_weights_calmar

        returns_geometric = self.compute_returns(best_geometric)
        returns_max_drawdown = self.compute_returns(best_max_drawdown)
        returns_calmar = self.compute_returns(best_calmar)

        # Plot assets returns
        plt.figure(figsize=(10, 6))
        for asset in self.returns.columns:
            returns = self.compute_returns({asset: 1.0})
            plt.scatter(
                max_drawdown(returns) * 100,
                geometric_mean(returns) * 100,
                marker="o",
                s=100,
                alpha=0.5,
                label=asset,
            )
        max_drawdown_geometric, geometric_mean_geometric = (
            max_drawdown(returns_geometric) * 100,
            geometric_mean(returns_geometric) * 100,
        )
        max_drawdown_max_drawdown, geometric_mean_max_drawdown = (
            max_drawdown(returns_max_drawdown) * 100,
            geometric_mean(returns_max_drawdown) * 100,
        )
        max_drawdown_calmar, geometric_mean_calmar = (
            max_drawdown(returns_calmar) * 100,
            geometric_mean(returns_calmar) * 100,
        )

        # Create scatter plot of max drawdown vs geometric mean
        plt.scatter(
            max_drawdown_geometric,
            geometric_mean_geometric,
            marker="*",
            s=300,
            color="red",
            label="Best Geometric Mean",
        )
        plt.scatter(
            max_drawdown_max_drawdown,
            geometric_mean_max_drawdown,
            marker="d",
            s=200,
            color="blue",
            label="Lowest Max Drawdown",
        )
        plt.scatter(
            max_drawdown_calmar,
            geometric_mean_calmar,
            marker="s",
            s=200,
            color="green",
            label="Highest Calmar Ratio",
        )

        plt.xlabel("Max Drawdown (%)")
        plt.ylabel("Geometric Mean Return (%)")
        plt.title("Portfolio Optimization: Geometric Mean vs Max Drawdown")
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()

    def plot_optimization_landscape(self):
        pass
