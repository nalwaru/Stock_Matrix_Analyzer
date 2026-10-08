import numpy as np
import pandas as pd
import pytest

from analyzer import scenarios


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": ["AAPL", "BTC-USD", "VOO"],
            "Asset_Class": ["Stocks", "Crypto", "ETFs"],
            "Current_Value": [1000.0, 500.0, np.nan],
        }
    )


def test_zero_shock_leaves_values_unchanged():
    out = scenarios.apply_shocks(frame(), {})
    assert out["Simulated_Value"].iloc[:2].tolist() == [1000.0, 500.0]
    result = scenarios.compare(out)
    assert result == {"original": 1500.0, "simulated": 1500.0, "delta": 0.0, "delta_pct": 0.0}


def test_negative_shock_applies_per_asset_class():
    out = scenarios.apply_shocks(frame(), {"Stocks": -0.2})
    assert out["Simulated_Value"].tolist()[:2] == [800.0, 500.0]
    result = scenarios.compare(out)
    assert result["delta"] == pytest.approx(-200.0)
    assert result["delta_pct"] == pytest.approx(-200 / 1500 * 100)


def test_total_loss_and_over_shock_clip_at_zero():
    out = scenarios.apply_shocks(frame(), {"Stocks": -1.0, "Crypto": -1.5})
    assert out["Simulated_Value"].iloc[:2].tolist() == [0.0, 0.0]


def test_nan_values_stay_nan_and_input_is_not_mutated():
    original = frame()
    out = scenarios.apply_shocks(original, {"ETFs": 0.5})
    assert np.isnan(out["Simulated_Value"].iloc[2])
    assert "Simulated_Value" not in original.columns


def test_compare_with_zero_original_does_not_divide_by_zero():
    df = pd.DataFrame({"Asset_Class": ["Stocks"], "Current_Value": [0.0]})
    result = scenarios.compare(scenarios.apply_shocks(df, {"Stocks": 0.1}))
    assert result["delta_pct"] == 0.0
