import io

import pandas as pd

from analyzer import export


def test_csv_round_trip_with_formatted_dates():
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL"],
            "Purchase_Date": pd.to_datetime(["2023-01-15"]),
            "Current_Value": [1100.5],
        }
    )
    data = export.to_csv_bytes(df)
    assert isinstance(data, bytes)
    back = pd.read_csv(io.BytesIO(data))
    assert back.loc[0, "Purchase_Date"] == "2023-01-15"
    assert back.loc[0, "Current_Value"] == 1100.5
    assert list(back.columns) == ["Ticker", "Purchase_Date", "Current_Value"]


def test_csv_without_date_column_and_empty_frame():
    assert export.to_csv_bytes(pd.DataFrame({"A": [1]})).startswith(b"A")
    assert export.to_csv_bytes(pd.DataFrame(columns=["A", "B"])) == b"A,B\n"


def test_print_css_hides_sidebar_and_controls():
    assert "stSidebar" in export.PRINT_CSS
    assert "display: none" in export.PRINT_CSS
    assert "<style" not in export.PRINT_CSS
