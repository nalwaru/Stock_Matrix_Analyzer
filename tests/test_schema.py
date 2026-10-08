import numpy as np
import pandas as pd

from analyzer.schema import REQUIRED_COLUMNS, validate


def good_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Ticker": ["aapl ", " MSFT"],
        "Shares": [15.0, 10.0],
        "Purchase_Price": [150.25, 280.5],
        "Purchase_Date": ["2023-01-15", "2023-02-10"],
        "Asset_Class": ["Stocks", " Stocks"],
    })


def test_valid_frame_is_normalized():
    result = validate(good_frame())
    assert result.errors == [] and result.warnings == []
    assert list(result.df.columns) == REQUIRED_COLUMNS
    assert result.df["Ticker"].tolist() == ["AAPL", "MSFT"]
    assert result.df["Asset_Class"].tolist() == ["Stocks", "Stocks"]
    assert str(result.df["Purchase_Date"].dtype).startswith("datetime64")


def test_missing_columns_is_an_error():
    raw = good_frame().drop(columns=["Asset_Class", "Shares"])
    result = validate(raw)
    assert result.df.empty
    assert len(result.errors) == 1
    assert "Asset_Class" in result.errors[0] and "Shares" in result.errors[0]


def test_negative_and_zero_shares_dropped_with_warning():
    raw = good_frame()
    raw.loc[1, "Shares"] = -5
    result = validate(raw)
    assert result.df["Ticker"].tolist() == ["AAPL"]
    assert any("Dropped row 3" in w and "shares" in w for w in result.warnings)


def test_padding_rows_and_empty_columns_are_ignored():
    raw = good_frame()
    padding = pd.DataFrame([[np.nan] * 5] * 3, columns=raw.columns)
    raw = pd.concat([raw, padding], ignore_index=True)
    raw["Unnamed: 5"] = np.nan
    raw["Unnamed: 6"] = np.nan
    result = validate(raw)
    assert result.errors == [] and result.warnings == []
    assert len(result.df) == 2


def test_invalid_date_and_price_rows_dropped():
    raw = good_frame()
    raw.loc[0, "Purchase_Date"] = "not-a-date"
    raw["Purchase_Price"] = raw["Purchase_Price"].astype(object)
    raw.loc[1, "Purchase_Price"] = "abc"
    result = validate(raw)
    assert result.df.empty
    assert result.errors == ["No valid holdings found in the file."]
    assert any("date" in w for w in result.warnings)
    assert any("price" in w for w in result.warnings)


def test_fractional_shares_and_zero_price_are_valid():
    raw = good_frame()
    raw.loc[0, "Shares"] = 0.25
    raw.loc[1, "Purchase_Price"] = 0
    result = validate(raw)
    assert result.warnings == []
    assert result.df["Shares"].tolist() == [0.25, 10.0]
