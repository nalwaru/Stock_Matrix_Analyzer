"""What-if stress testing: pure functions over the enriched frame."""
import pandas as pd


def apply_shocks(df: pd.DataFrame, shocks: dict[str, float]) -> pd.DataFrame:
    """Return a copy with Simulated_Value. Shocks are fractions (-0.2 = -20%)."""
    out = df.copy()
    factor = out["Asset_Class"].map(lambda cls: max(0.0, 1.0 + shocks.get(cls, 0.0)))
    out["Simulated_Value"] = out["Current_Value"] * factor
    return out


def compare(df: pd.DataFrame) -> dict:
    original = float(df["Current_Value"].sum())
    simulated = float(df["Simulated_Value"].sum())
    delta = simulated - original
    return {
        "original": original,
        "simulated": simulated,
        "delta": delta,
        "delta_pct": delta / original * 100 if original > 0 else 0.0,
    }
