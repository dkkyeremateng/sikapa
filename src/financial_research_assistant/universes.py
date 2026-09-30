"""What the recommender chooses from, and how the book maps onto asset classes.

**Stocks**: the S&P 500 (fetched keyless, as the screener does) with the built-in
large-cap list as the fallback. **Funds**: the curated catalog in
``data/etf_catalog.json`` — about sixty liquid US-listed ETFs tagged by asset
class and sleeve. There is no keyless bulk ETF feed, so the universe is that
list, and every ideas sheet names it rather than implying a search of the whole
market. Bonds, real estate, commodities and crypto come in through their sleeves.

``allocation`` puts the current book into the same asset classes the profile's
target mix is written in, so a gap between the two is a like-for-like number.
"""

from __future__ import annotations

from typing import Any
import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def catalog() -> tuple[dict[str, Any], ...]:
    """The fund catalog (read once; it ships with the package)."""
    text = resources.files(__package__ or "financial_research_assistant").joinpath(
        "data/etf_catalog.json").read_text("utf-8")
    return tuple(json.loads(text)["funds"])


def fund(symbol: str) -> dict[str, Any] | None:
    sym = (symbol or "").strip().upper()
    return next((f for f in catalog() if f["symbol"] == sym), None)


def funds_in(asset_class: str) -> list[dict[str, Any]]:
    return [f for f in catalog() if f["asset_class"] == asset_class]


def stock_universe() -> tuple[list[str], str]:
    """``(symbols, label)``: the S&P 500, or the built-in large caps when the
    constituents can't be fetched (the label says which)."""
    from . import screener

    syms = screener._fetch_sp500()  # pyright: ignore[reportPrivateUsage]
    if syms:
        return syms, "S&P 500"
    return list(screener._DEFAULT_UNIVERSE), "built-in large-cap list"  # pyright: ignore[reportPrivateUsage]


def asset_class_of(symbol: str) -> str:
    """The asset class a holding counts toward. A catalog fund is what the catalog
    says; a coin is crypto; anything else held as a stock is US equity — the one
    assumption here, and a foreign ADR is the case it gets wrong."""
    sym = (symbol or "").strip().upper()
    f = fund(sym)
    if f:
        return str(f["asset_class"])
    if sym.endswith("-USD"):
        return "crypto"
    return "us_equity"


def allocation(book: dict[str, Any]) -> dict[str, float]:
    """Percent of the priced book in each asset class."""
    total = sum(h["value"] for h in book["holdings"]) or 0.0
    out: dict[str, float] = {}
    for h in book["holdings"]:
        cls = asset_class_of(h["symbol"])
        out[cls] = out.get(cls, 0.0) + (h["value"] / total * 100 if total else 0.0)
    return out


def gaps(book: dict[str, Any], targets: dict[str, Any], min_gap_pct: float = 3.0) -> list[tuple[str, float, float]]:
    """``(asset_class, target %, current %)`` for classes under target by at least
    ``min_gap_pct`` points, largest gap first."""
    current = allocation(book)
    rows = []
    for cls, target in targets.items():
        have = current.get(cls, 0.0)
        if float(target) - have >= min_gap_pct:
            rows.append((cls, float(target), have))
    return sorted(rows, key=lambda r: r[1] - r[2], reverse=True)
