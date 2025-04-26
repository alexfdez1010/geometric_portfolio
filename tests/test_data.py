import pandas as pd
import pytest
from geometric_portfolio import data


def test_get_returns_single_ticket():
    df = data.get_returns(["AAPL"], start_date="2024-01-01", end_date="2024-01-15")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 1
    assert "AAPL" in df.columns
    assert not df.empty


def test_get_returns_multiple_tickets():
    tickets = ["AAPL", "MSFT"]
    df = data.get_returns(tickets, start_date="2024-01-01", end_date="2024-01-15")
    assert isinstance(df, pd.DataFrame)
    assert set(tickets).issubset(df.columns)
    assert df.shape[1] == len(tickets)
    assert not df.empty


def test_get_returns_invalid_ticket():
    with pytest.raises(ValueError):
        data.get_returns(
            ["INVALIDTICKER"], start_date="2024-01-01", end_date="2024-01-15"
        )
