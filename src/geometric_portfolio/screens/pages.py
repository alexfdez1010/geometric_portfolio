import streamlit as st
from geometric_portfolio.screens.custom_portfolio import CustomPortfolioPage
from geometric_portfolio.screens.leverage_optimizer import LeverageOptimizerPage
from geometric_portfolio.screens.geometric_mean import GeometricMeanPage
from geometric_portfolio.state import get_page_key, set_page_key
from geometric_portfolio.screens.page import Page


PAGES = {
    "Geometric Mean": GeometricMeanPage,
    "Custom Portfolio": CustomPortfolioPage,
    "Leverage Optimizer": LeverageOptimizerPage,
}


def get_page(key: str) -> Page:
    page = PAGES.get(key)

    if not page:
        raise ValueError(f"Page {key} not found.")

    return page()


def page_selector() -> str:
    """
    Select the page to display.
    """
    pages = list(PAGES.keys())

    page_key = st.selectbox(
        "Page",
        pages,
        index=pages.index(get_page_key()),
        key="page_selector_key",
    )

    if page_key != get_page_key():
        set_page_key(page_key)

    return page_key
