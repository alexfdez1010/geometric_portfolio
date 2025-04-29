import matplotlib

matplotlib.use("Agg")  # Set backend before importing pyplot or running tests

import os
from streamlit.testing.v1 import AppTest

# Define the path to the main app script relative to this test file
APP_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "../src/geometric_portfolio/app.py"
)

# Use descriptive names (keys from TICKERS dict) matching widget options
TEST_ASSETS = ["S&P 500 (VOO)", "Apple Inc. (AAPL)"]


def test_custom_portfolio_screen():
    """Test navigating to Custom Portfolio screen, selecting assets, and running backtest."""
    at = AppTest.from_file(APP_SCRIPT_PATH)
    at.run(timeout=60)  # Initial run of the main app

    # Navigate to Custom Portfolio screen
    at.selectbox(key="page_selector_key").select("Custom Portfolio")
    at.run(timeout=60)  # Run after navigation

    # Interact with Custom Portfolio widgets
    at.multiselect(key="custom_assets_select").select(TEST_ASSETS[0]).select(
        TEST_ASSETS[1]
    )
    at.button(key="custom_run_button").click()
    at.run(timeout=120)  # Run after interactions

    # Check if any exceptions were raised during execution
    assert len(at.exception) == 0, (
        f"Custom Portfolio AppTest raised exceptions: {at.exception}"
    )


def test_geometric_mean_screen():
    """Test navigating to Geometric Mean screen, selecting assets, and running optimization."""
    at = AppTest.from_file(APP_SCRIPT_PATH)
    at.run(timeout=60)  # Initial run of the main app

    # Navigate to Geometric Mean screen
    at.selectbox(key="page_selector_key").select("Geometric Mean")
    at.run(timeout=60)  # Run after navigation

    # Interact with Geometric Mean widgets
    at.multiselect(key="gm_equity_select").select(TEST_ASSETS[0])
    at.multiselect(key="gm_stocks_select").select(TEST_ASSETS[1])
    at.button(key="gm_run_button").click()
    at.run(timeout=180)  # Run after interactions (optimization might take longer)

    # Check if any exceptions were raised during execution
    assert len(at.exception) == 0, (
        f"Geometric Mean AppTest raised exceptions: {at.exception}"
    )


def test_leverage_optimizer_screen():
    """Test navigating to Leverage Optimizer screen, selecting asset, and running optimization."""
    at = AppTest.from_file(APP_SCRIPT_PATH)
    at.run(timeout=60)  # Initial run of the main app

    # Navigate to Leverage Optimizer screen
    at.selectbox(key="page_selector_key").select("Leverage Optimizer")
    at.run(timeout=60)  # Run after navigation

    # Interact with Leverage Optimizer widgets
    at.selectbox(key="lo_asset_select").select(TEST_ASSETS[0])
    at.button(key="lo_run_button").click()
    at.run(timeout=120)  # Run after interactions

    # Check if any exceptions were raised during execution
    assert len(at.exception) == 0, (
        f"Leverage Optimizer AppTest raised exceptions: {at.exception}"
    )
