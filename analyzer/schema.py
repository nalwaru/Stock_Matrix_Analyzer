"""CSV schema validation for portfolio uploads."""
from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = ["Ticker", "Shares", "Purchase_Price", "Purchase_Date", "Asset_Class"]


@dataclass
class ValidationResult:
    df: pd.DataFrame
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _csv_lines(index) -> str:
    # index is the original 0-based row index; +2 = header line + 1-based row.
    return ", ".join(str(int(i) + 2) for i in index)


def _clean_text(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "").astype(str).str.strip()


def validate(raw: pd.DataFrame) -> ValidationResult:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    empty_extras = [
        c for c in df.columns if c not in REQUIRED_COLUMNS and df[c].isna().all()
    ]
    df = df.drop(columns=empty_extras)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return ValidationResult(
            df.iloc[0:0],
            errors=[f"Missing required column(s): {', '.join(missing)}"],
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["Ticker"] = _clean_text(df["Ticker"]).str.upper()
    df["Asset_Class"] = _clean_text(df["Asset_Class"])
    df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce")
    df["Purchase_Price"] = pd.to_numeric(df["Purchase_Price"], errors="coerce")
    df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"], errors="coerce", format="mixed")

    checks = {
        "missing ticker": df["Ticker"] == "",
        "missing asset class": df["Asset_Class"] == "",
        "shares must be a positive number (zero, negative or non-numeric)": ~(df["Shares"] > 0),
        "purchase price must be a non-negative number": ~(df["Purchase_Price"] >= 0),
        "invalid purchase date": df["Purchase_Date"].isna(),
    }
    warnings: list[str] = []
    bad = pd.Series(False, index=df.index)
    for reason, mask in checks.items():
        if mask.any():
            label = "row" if mask.sum() == 1 else "rows"
            warnings.append(f"Dropped {label} {_csv_lines(df.index[mask])}: {reason}")
            bad |= mask

    df = df[~bad].reset_index(drop=True)
    errors = [] if len(df) else ["No valid holdings found in the file."]
    return ValidationResult(df, errors=errors, warnings=warnings)
