import pandas as pd
import numpy as np
from geometric_portfolio.montecarlo import MonteCarlo

def test_possible_weights_sum_to_one():
    returns = pd.DataFrame(np.random.randn(10, 3), columns=["A", "B", "C"])
    mc = MonteCarlo(returns)
    weights = mc._possible_weights(["A", "B", "C"])
    assert np.isclose(sum(weights.values()), 1.0)
    assert set(weights.keys()) == {"A", "B", "C"}

def test_compute_returns_shape():
    returns = pd.DataFrame({
        "A": [0.01, 0.02, 0.03],
        "B": [0.04, 0.05, 0.06],
        "C": [0.07, 0.08, 0.09],
    })
    mc = MonteCarlo(returns)
    weights = {"A": 0.2, "B": 0.3, "C": 0.5}
    portfolio_returns = mc.compute_returns(weights)
    assert isinstance(portfolio_returns, pd.Series)
    assert len(portfolio_returns) == len(returns)

def test_run_returns_best_weights():
    returns = pd.DataFrame({
        "A": [0.01, 0.02, 0.03, 0.04],
        "B": [0.04, 0.03, 0.02, 0.01],
    })
    mc = MonteCarlo(returns)
    best_weights = mc.run(num_simulations=20)
    assert isinstance(best_weights, dict)
    assert set(best_weights.keys()) >= {"A", "B"}
    assert abs(sum([best_weights[k] for k in ["A", "B"]]) - 1.0) < 1e-6
