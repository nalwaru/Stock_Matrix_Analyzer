"""Structured outputs the agent nodes request from the model."""
from typing import Literal

from pydantic import BaseModel

Route = Literal["sec_filings", "live_price", "both", "portfolio_only", "analysis", "agent", "out_of_scope"]
Analysis = Literal["technical", "fundamental", "risk", "ownership"]


class RouteDecision(BaseModel):
    route: Route
    tickers: list[str]
    reason: str
    analyses: list[Analysis] = []
    wants_outlook: bool = False
    declined_part: str = ""


class PassageGrade(BaseModel):
    id: str
    relevant: bool
    reason: str


class GradeResult(BaseModel):
    grades: list[PassageGrade]


class RewriteResult(BaseModel):
    query: str


class Memo(BaseModel):
    answer: str
    key_figures: list[str]
    sources: list[str]
    unverified: list[str]
    outlook: list[str] = []  # reasons, each ending with a supplied [label]
    outlook_leaning: Literal["", "positive", "neutral", "cautious"] = ""
    declined: str = ""  # set from the gate, never by the model


class CalcCall(BaseModel):
    """One portfolio calculation; unused arguments are null."""
    tool: Literal["holding", "class_weight", "top_n", "total", "stress"]
    ticker: str | None = None
    asset_class: str | None = None
    metric: Literal["Current_Value", "Gain_Loss", "Gain_Loss_Pct", "Day_Change"] | None = None
    n: int | None = None
    shock_pct: float | None = None
    order: Literal["largest", "smallest"] | None = None


class ToolPlan(BaseModel):
    calc_calls: list[CalcCall]
