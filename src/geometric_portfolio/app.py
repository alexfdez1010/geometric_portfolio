import streamlit as st
from geometric_portfolio.screens.pages import get_page, page_selector


def main():
    st.set_page_config(
        page_title="Geometric Portfolio Explorer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    page_key = page_selector()
    page = get_page(page_key)
    page.render()


if __name__ == "__main__":
    main()
