"""Stock Portfolio Analyzer: Streamlit entry point (UI wiring only)."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from analyzer import ai, export, market, portfolio, scenarios, schema

load_dotenv()

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_portfolio.csv"

st.set_page_config(page_title="Stock Portfolio Analyzer", page_icon="📈", layout="wide")
st.title("Stock Portfolio Analyzer")


def money(value: float) -> str:
    return f"-${abs(value):,.2f}" if value < 0 else f"${value:,.2f}"


def load_raw() -> pd.DataFrame:
    uploaded = st.sidebar.file_uploader("Upload portfolio CSV", type=["csv"])
    use_sample = st.sidebar.toggle("Use Sample Dataset", value=False)
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not read the CSV file: {exc}")
            st.stop()
    if use_sample:
        return pd.read_csv(SAMPLE_PATH)
    st.info(
        "Upload a CSV with columns "
        f"`{'`, `'.join(schema.REQUIRED_COLUMNS)}`, or switch on **Use Sample Dataset** "
        "in the sidebar."
    )
    st.stop()


def apply_filters(holdings: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    classes = sorted(holdings["Asset_Class"].unique())
    chosen = st.sidebar.multiselect("Asset Class", classes, default=classes)
    lo = holdings["Purchase_Date"].min().date()
    hi = holdings["Purchase_Date"].max().date()
    picked = st.sidebar.date_input(
        "Acquisition date range", value=(lo, hi), min_value=lo, max_value=hi
    )
    if not isinstance(picked, tuple) or len(picked) != 2:
        st.info("Pick an end date to apply the date range.")
        st.stop()
    start, end = picked
    mask = holdings["Asset_Class"].isin(chosen) & holdings["Purchase_Date"].dt.date.between(
        start, end
    )
    return holdings[mask]


# ---------- ingest + validate ----------
result = schema.validate(load_raw())
for message in result.warnings:
    st.warning(message)
if result.errors:
    for message in result.errors:
        st.error(message)
    st.stop()

holdings = result.df
filtered = apply_filters(holdings)
if filtered.empty:
    st.warning("No holdings match the current filters.")
    st.stop()

# ---------- market data ----------
all_tickers = tuple(sorted(holdings["Ticker"].unique()))
try:
    quotes = market.fetch_quotes(all_tickers)
except market.MarketDataError:  # defensive: the cached wrapper normally returns empty
    quotes = market.empty_quotes()
stale = bool(quotes["Price"].isna().all())
if stale:
    st.error("Live market data is unavailable. Showing purchase prices as stale values.")

enriched = portfolio.enrich(filtered, quotes, fallback_to_cost=stale)
unavailable = enriched.loc[enriched["Price_Status"] == "unavailable", "Ticker"].unique().tolist()
if unavailable:
    st.warning(f"Price unavailable, excluded from totals: {', '.join(unavailable)}")

# ---------- stress-test controls ----------
st.sidebar.header("What-if stress test")
shocks = {
    cls: st.sidebar.slider(f"{cls} shock (%)", -50, 50, 0, key=f"shock_{cls}") / 100
    for cls in sorted(enriched["Asset_Class"].unique())
}
simulated = scenarios.apply_shocks(enriched, shocks)

# ---------- KPIs ----------
s = portfolio.summary(enriched)
k1, k2, k3 = st.columns(3)
k1.metric("Total Portfolio Value", money(s["total_value"]), delta=money(s["total_gain"]))
k2.metric(
    "Total Unrealized P/L",
    f"{s['total_gain_pct']:.2f}%",
    delta=f"{s['total_gain_pct']:.2f}%",
)
k3.metric("Daily P&L", money(s["day_pnl"]), delta=f"{s['day_pnl_pct']:.2f}%")

tab_dash, tab_stress, tab_ai, tab_export = st.tabs(
    ["Dashboard", "Stress Test", "AI Insights", "Export"]
)

# ---------- dashboard ----------
with tab_dash:
    left, right = st.columns(2)
    alloc = portfolio.allocation(enriched)
    top = portfolio.top_holdings(enriched, 10)
    if alloc.empty:
        st.info("No priced holdings to chart.")
    else:
        left.plotly_chart(
            px.pie(alloc, names="Asset_Class", values="Current_Value", hole=0.5,
                   title="Allocation by Asset Class"),
            width="stretch",
        )
        right.plotly_chart(
            px.bar(top, x="Ticker", y="Current_Value", title="Top Holdings by Current Value"),
            width="stretch",
        )
    st.dataframe(
        enriched[[
            "Ticker", "Asset_Class", "Shares", "Purchase_Price", "Purchase_Date", "Price",
            "Current_Value", "Gain_Loss", "Gain_Loss_Pct", "Day_Change",
            "Week52_Low", "Week52_High", "Price_Status",
        ]],
        hide_index=True,
        width="stretch",
        column_config={
            "Purchase_Price": st.column_config.NumberColumn(format="$%.2f"),
            "Purchase_Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Current_Value": st.column_config.NumberColumn(format="$%.2f"),
            "Gain_Loss": st.column_config.NumberColumn(format="$%.2f"),
            "Gain_Loss_Pct": st.column_config.NumberColumn(format="%.2f%%"),
            "Day_Change": st.column_config.NumberColumn(format="$%.2f"),
            "Week52_Low": st.column_config.NumberColumn(format="$%.2f"),
            "Week52_High": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

# ---------- stress test ----------
with tab_stress:
    cmp = scenarios.compare(simulated)
    c1, c2, c3 = st.columns(3)
    c1.metric("Original Value", money(cmp["original"]))
    c2.metric("Simulated Value", money(cmp["simulated"]))
    c3.metric("Change", money(cmp["delta"]), delta=f"{cmp['delta_pct']:.2f}%")
    if not any(shocks.values()):
        st.caption("Move the shock sliders in the sidebar to simulate market moves by asset class.")

# ---------- AI ----------
with tab_ai:
    payload = json.dumps(portfolio.ai_summary(enriched), sort_keys=True)
    if not ai.is_configured():
        st.info(
            "AI features are disabled. Set `OPENAI_API_KEY` in your environment or a `.env` "
            "file (see `.env.example`) and restart the app."
        )
    else:
        if st.button("Generate AI Audit"):
            try:
                with st.spinner("Analyzing portfolio..."):
                    st.session_state["audit"] = (payload, ai.audit(payload, ai.get_model()))
            except ai.AIError as exc:
                st.error(f"{exc} You can retry.")
        saved = st.session_state.get("audit")
        if saved and saved[0] == payload:
            with st.expander("AI Portfolio Audit", expanded=True):
                st.markdown(saved[1])
                st.caption("Educational analysis, not financial advice.")
        elif saved:
            st.caption("Your filters changed since the last audit. Generate a new one.")

        st.subheader("Ask AI about your portfolio")
        chat = st.session_state.setdefault("chat", [])
        for msg in chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if question := st.chat_input("e.g. Which holding is my highest risk?"):
            with st.chat_message("user"):
                st.markdown(question)
            try:
                reply = ai.chat_reply(payload, chat, question, ai.get_model())
            except ai.AIError as exc:
                st.error(f"{exc} You can retry.")
            else:
                chat += [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": reply},
                ]
                with st.chat_message("assistant"):
                    st.markdown(reply)

# ---------- export ----------
with tab_export:
    st.download_button(
        "Download enriched portfolio (CSV)",
        data=export.to_csv_bytes(simulated),
        file_name="portfolio_enriched.csv",
        mime="text/csv",
    )
    if st.toggle("Print-friendly view"):
        st.markdown(f"<style>{export.PRINT_CSS}</style>", unsafe_allow_html=True)
        st.caption("Use your browser's Print / Save as PDF now.")
