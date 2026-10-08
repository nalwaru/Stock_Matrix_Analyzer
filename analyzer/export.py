"""CSV export and print-friendly CSS."""
import pandas as pd

PRINT_CSS = """
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stChatInput"],
[data-baseweb="tab-list"],
.stButton,
.stDownloadButton { display: none !important; }
[data-baseweb="tab-panel"] { display: block !important; }
.block-container { padding-top: 1rem; max-width: 100%; }
"""


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    if "Purchase_Date" in out.columns:
        out["Purchase_Date"] = pd.to_datetime(out["Purchase_Date"]).dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False).encode("utf-8")
