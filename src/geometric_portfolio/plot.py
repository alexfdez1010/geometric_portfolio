import matplotlib.pyplot as plt
import pandas as pd


def plot_wealth_evolution(wealth_dict: dict[str, pd.Series | list[float]]) -> plt.Figure:
    """
    Plot wealth evolution over time.
    
    Args:
        wealth_dict: dict mapping asset names to wealth time series (pd.Series or list).
    
    Returns:
        Matplotlib figure.
    """
    df = pd.DataFrame(wealth_dict)
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.set_title("Wealth Evolution")
    ax.set_xlabel("Time")
    ax.set_ylabel("Wealth")
    ax.grid(True)
    return fig


def plot_returns_distribution(returns_dict: dict[str, pd.Series], bins=50, alpha=0.5) -> plt.Figure:
    """
    Plot distribution of returns for multiple assets.

    Args:
        returns_dict: dict mapping asset names to return series (pd.Series).
        bins: number of bins for histogram.
        alpha: transparency for histogram bars.

    Returns:
        Matplotlib figure.
    """
    num_plots = len(returns_dict)
    cols = min(num_plots, 3)  # Maximum 3 columns
    rows = (num_plots + cols - 1) // cols  # Calculate needed rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten() if num_plots > 1 else [axes]
    
    for i, (name, series) in enumerate(returns_dict.items()):
        # Normalize to percentages by using density=True and multiplying by 100
        series.hist(ax=axes[i], bins=bins, alpha=alpha, label=name, density=True)
        axes[i].set_title(f"{name} Returns")
        axes[i].set_xlabel("Returns")
        axes[i].set_ylabel("Percentage (%)")
        # Convert y-axis to percentage
        axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
        axes[i].legend()
    
    # Hide unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig

# New: plot correlation matrix

def plot_correlation_matrix(returns_df: pd.DataFrame) -> plt.Figure:
    """
    Plot correlation matrix of asset returns.
    """
    corr = returns_df.corr()
    fig, ax = plt.subplots(figsize=(len(corr.columns)*0.5+4, len(corr.index)*0.5+4))
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    return fig
