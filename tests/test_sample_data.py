from pathlib import Path

import pandas as pd

SAMPLE = Path(__file__).parent.parent / "data" / "sample_portfolio.csv"


def test_sample_portfolio_has_required_shape():
    df = pd.read_csv(SAMPLE)
    assert list(df.columns) == [
        "Ticker", "Shares", "Purchase_Price", "Purchase_Date", "Asset_Class",
    ]
    assert len(df) == 12
    assert set(df["Asset_Class"]) == {"Stocks", "ETFs", "Crypto", "REITs"}
    assert (df["Shares"] > 0).all()
