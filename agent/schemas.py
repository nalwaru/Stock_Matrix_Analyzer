"""Structured outputs the agent nodes request from the model."""
from typing import Literal

from pydantic import BaseModel

Route = Literal["sec_filings", "live_price", "both", "portfolio_only", "out_of_scope"]


class RouteDecision(BaseModel):
    route: Route
    tickers: list[str]
    reason: str


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
