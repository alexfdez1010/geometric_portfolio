import streamlit as st


@st.cache_resource
def get_page_key() -> str:
    return st.session_state.get("page_key", "Geometric Mean")


def set_page_key(page_key: str) -> None:
    st.session_state["page_key"] = page_key
