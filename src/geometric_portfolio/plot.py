import matplotlib.pyplot as plt
import pandas as pd


def plot_wealth_evolution(wealth_dict):
    """
    Plot wealth evolution over time.
    wealth_dict: dict mapping asset names to wealth time series (pd.Series or list).
    Returns the matplotlib figure.
    """
    df = pd.DataFrame(wealth_dict)
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.set_title("Wealth Evolution")
    ax.set_xlabel("Time")
    ax.set_ylabel("Wealth")
    ax.grid(True)
    return fig


def plot_returns_distribution(returns_dict, bins=50, alpha=0.5):
    """
    Plot distribution of returns for multiple assets.
    returns_dict: dict mapping asset names to return series (pd.Series).
    bins: number of bins for histogram.
    alpha: transparency for histogram bars.
    Returns the matplotlib figure.
    """
    fig, ax = plt.subplots()
    for name, series in returns_dict.items():
        series.hist(ax=ax, bins=bins, alpha=alpha, label=name)

    ax.set_title("Returns Distribution")
    ax.set_xlabel("Returns")
    ax.set_ylabel("Frequency")
    ax.legend()
    return fig
