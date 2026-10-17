"""Retrieval metrics over ranked chunk ids."""


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return 1.0 if gold & set(ranked[:k]) else 0.0


def mrr(ranked: list[str], gold: set[str]) -> float:
    for rank, item in enumerate(ranked, start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(gold & set(ranked[:k])) / len(gold) if gold else 0.0
