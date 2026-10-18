"""Stock Portfolio Analyzer: Streamlit entry point (UI wiring only)."""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from analyzer import ai, export, loading, market, portfolio, scenarios, schema
from agent import graph as agent_graph
from agent import wiring as agent_wiring
from analyzer.avatars import ASSISTANT_AVATAR, AVATARS, USER_AVATAR
from agent.memo import idk_memo, markdown_safe, render_memo, visible_sources

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
    ["Dashboard", "Stress Test", "Ask AI", "Export"]
)

# ---------- dashboard ----------
with tab_dash:
    left, right = st.columns(2)
    alloc = portfolio.allocation(enriched)
    colors = portfolio.class_colors(holdings)  # from all holdings, so colours survive filtering
    top = portfolio.top_holdings(enriched, 10)
    if alloc.empty:
        st.info("No priced holdings to chart.")
    else:
        left.plotly_chart(
            px.pie(alloc, names="Asset_Class", values="Current_Value", hole=0.5,
                   title="Allocation by Asset Class", color="Asset_Class",
                   color_discrete_map=colors),
            width="stretch",
        )
        right.plotly_chart(
            px.bar(top, x="Ticker", y="Current_Value", title="Top Holdings by Current Value",
                   color="Asset_Class", color_discrete_map=colors),
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

# ---------- Ask AI ----------
@st.cache_resource
def pinecone_client():
    from pinecone import Pinecone

    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


@st.cache_resource
def yahoo_data():
    from agent.analysts.data import YahooData

    return YahooData()


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(markdown_safe(msg["content"]))
        for figure in msg.get("charts", []):
            st.plotly_chart(figure, width="stretch")
        if msg.get("sources"):
            with st.expander("Sources"):
                for p in msg["sources"]:
                    st.markdown(markdown_safe(
                        f"**[{p.ticker} {p.form}, {p.filing_date}, {p.section}]** ({p.source_url})"))
                    st.caption(markdown_safe(p.text[:600]))
        if msg.get("trace"):
            with st.expander("Agent trace"):
                for line in msg["trace"]:
                    st.markdown(f"- {markdown_safe(line)}")


def render_ask_ai(df: pd.DataFrame) -> None:
    st.subheader("Ask AI")
    if not ai.is_configured():
        st.info(
            "Ask AI is disabled. Set `OPENAI_API_KEY` in your environment or a `.env` "
            "file (see `.env.example`) and restart the app."
        )
        return
    st.caption(ai.disclosure())
    st.caption("Try: Which holding lost the most? How concentrated am I in crypto?")
    try:
        context = ai.build_context(df)
    except ai.AIError as exc:
        st.info(str(exc))
        return

    fingerprint = hashlib.sha256(context.encode()).hexdigest()
    if st.session_state.get("ask_ai_scope") != fingerprint:
        st.session_state["ask_ai_scope"] = fingerprint
        st.session_state["chat"] = []
    if st.button("Clear conversation", key="clear_ask_ai"):
        st.session_state["chat"] = []

    payload = json.dumps(portfolio.ai_summary(df), sort_keys=True)
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

    # Input stays at the top; the conversation renders below it, newest exchange first
    # (each exchange keeps question-then-answer order).
    chat = st.session_state.setdefault("chat", [])
    with st.container():
        question = st.chat_input("Ask about your portfolio", max_chars=ai.MAX_QUESTION)
    live = st.empty()  # holds the in-progress exchange; cleared once the answer is stored
    if question:
        result = None
        with live.container():
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(question)
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                bag_col, status_col = st.columns([1, 5], vertical_alignment="center")
                with bag_col:
                    st.markdown(loading.money_bag_html(), unsafe_allow_html=True)
                with status_col:
                    status = st.status("Routing your question...", expanded=True)
                try:
                    for node, update in agent_graph.run_stream(
                        question, list(chat), context, list(df["Ticker"].unique()),
                        agent_wiring.make_deps(
                            pinecone_client, os.getenv("RAG_NAMESPACE", "semantic"),
                            os.getenv("PINECONE_API_KEY", ""),
                            holdings=df, analyst_data=yahoo_data(),
                            on_step=lambda line: status.caption(markdown_safe(line)),
                        ),
                    ):
                        if node == "done":
                            result = update
                            break
                        step = agent_graph.NODE_LABELS.get(node, node)
                        status.write(f"Done: {step}")
                        for line in update.get("trace", []):
                            if node == "agent" and line.startswith("Agent tool:"):
                                continue  # already shown live through on_step
                            status.caption(markdown_safe(line))
                        status.update(label=f"Working... (last step: {step.lower()})")
                except Exception:  # noqa: BLE001 - run_stream never raises; last-resort guard
                    result = None
                if result is None:
                    result = {"memo": idk_memo(), "graded": [], "trace": ["Agent failed unexpectedly"]}
                status.update(label="Finished", state="complete", expanded=False)
        live.empty()
        chat += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": render_memo(result["memo"]),
             "sources": visible_sources(result), "trace": result.get("trace", []),
             "charts": [] if result["memo"].answer == ai.IDK_MESSAGE else result.get("charts", [])},
        ]
        st.session_state["chat"] = chat = chat[-2 * ai.MAX_CHAT_TURNS:]
    for i in range(len(chat) - 2, -1, -2):
        for msg in chat[i:i + 2]:
            render_message(msg)


with tab_ai:
    render_ask_ai(enriched)

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
