"""Charts for analyst results. One axis per panel (no dual axes); colours are fixed per series."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from agent.analysts.technical import indicator_frame

MIN_SESSIONS = 20
# Categorical slots 1-5, 7, 8 (dark-surface steps) in fixed order; neutrals for context marks.
COLORS = {
    "Close": "#3987e5", "SMA20": "#d95926", "SMA50": "#199e70", "SMA200": "#c98500",
    "RSI14": "#d55181", "MACD": "#9085e9", "Signal": "#e66767",
}
NEUTRAL = "rgba(139,138,133,0.55)"
GRID = "rgba(128,128,128,0.18)"


def technical_figure(ticker: str, history: pd.DataFrame, sessions: int = 250) -> go.Figure | None:
    if history is None or history.empty or "Close" not in history or len(history["Close"].dropna()) < MIN_SESSIONS:
        return None
    f = indicator_frame(history.dropna(subset=["Close"])).tail(sessions)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.5, 0.14, 0.18, 0.18])

    def line(name: str, series: pd.Series, row: int, width: float = 1.6, **kw) -> None:
        if series.dropna().empty:
            return
        fig.add_trace(go.Scatter(x=series.index, y=series, name=name, mode="lines",
                                 line=dict(color=COLORS.get(name, NEUTRAL), width=width), **kw), row=row, col=1)

    if not f["BB_UP"].dropna().empty:
        fig.add_trace(go.Scatter(x=f.index, y=f["BB_UP"], name="Bollinger upper", mode="lines", showlegend=False,
                                 line=dict(color=NEUTRAL, width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=f.index, y=f["BB_LOW"], name="Bollinger band", mode="lines", fill="tonexty",
                                 fillcolor="rgba(139,138,133,0.12)", line=dict(color=NEUTRAL, width=1)), row=1, col=1)
    line("Close", f["Close"], 1, 2)
    for name in ("SMA20", "SMA50", "SMA200"):
        line(name, f[name], 1)
    if not f["Volume"].dropna().empty:
        fig.add_trace(go.Bar(x=f.index, y=f["Volume"], name="Volume", marker_color=NEUTRAL), row=2, col=1)
    line("RSI14", f["RSI14"], 3)
    line("MACD", f["MACD"], 4)
    line("Signal", f["MACD_SIGNAL"], 4)
    if not f["MACD_HIST"].dropna().empty:
        fig.add_trace(go.Bar(x=f.index, y=f["MACD_HIST"], name="Histogram", marker_color=NEUTRAL), row=4, col=1)

    for level in (30, 70):
        fig.add_hline(y=level, row=3, col=1, line=dict(color=NEUTRAL, width=1, dash="dash"),
                      annotation_text=str(level), annotation_position="right")
    last = f.index[-1]
    for name in ("Close", "SMA20", "SMA50", "SMA200"):  # selective direct labels on the last value
        value = f[name].iloc[-1]
        if pd.notna(value):
            fig.add_annotation(x=last, y=value, text=f"{name} {value:,.2f}", xanchor="left", showarrow=False,
                               xshift=6, font=dict(color=COLORS[name], size=11), row=1, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI(14)", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.update_layout(
        title=f"{ticker.upper()} technical chart (last {len(f)} sessions, Yahoo daily data)",
        height=820, hovermode="x unified", barmode="relative",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.045, x=0, traceorder="normal"),
        margin=dict(l=60, r=110, t=110, b=40),
    )
    return fig
