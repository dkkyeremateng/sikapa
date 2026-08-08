"""Yahoo reference-data caching behavior.

`.info` is the slow quoteSummary round-trip that every valuation tool leans on,
so it is cached — but it also carries the spot price, which is why the cache has
to expire on the same clock the price series uses.
"""

import time

import pytest


class _Ticker:
    """Stands in for yfinance's Ticker; counts how often `.info` is reached."""

    def __init__(self, counter, payload):
        self._counter = counter
        self._payload = payload

    @property
    def info(self):
        self._counter.append(1)
        return dict(self._payload)


@pytest.fixture
def yahoo(monkeypatch):
    """A stubbed ticker with an empty cache, returning the fetch counter."""
    from financial_research_assistant import fundamentals

    fetches: list[int] = []
    monkeypatch.setattr(fundamentals, "_INFO_CACHE", {})
    monkeypatch.setattr(
        fundamentals, "_ticker", lambda sym: _Ticker(fetches, {"currentPrice": 100.0})
    )
    return fundamentals, fetches


def test_a_repeated_lookup_inside_the_window_is_served_from_the_cache(yahoo, monkeypatch):
    """The reason the cache exists: a tool that walks every holding must not pay
    a quoteSummary round-trip per mention of the same ticker."""
    fundamentals, fetches = yahoo
    monkeypatch.setenv("FINANCIAL_RESEARCH_PRICE_TTL", "900")

    assert fundamentals._fetch_info("AAPL")["currentPrice"] == 100.0
    fundamentals._fetch_info("AAPL")
    fundamentals._fetch_info("aapl")  # same ticker, different case
    assert len(fetches) == 1


def test_a_stale_snapshot_is_refetched_rather_than_quoted_as_current(yahoo, monkeypatch):
    """A session left open across a trading day would otherwise price an option
    off the morning's quote while the chart beside it showed the afternoon close.
    Past the TTL the snapshot is fetched again."""
    fundamentals, fetches = yahoo
    monkeypatch.setenv("FINANCIAL_RESEARCH_PRICE_TTL", "900")

    fundamentals._fetch_info("AAPL")
    info, _fetched_at = fundamentals._INFO_CACHE["AAPL"]
    fundamentals._INFO_CACHE["AAPL"] = (info, time.time() - 901)

    fundamentals._fetch_info("AAPL")
    assert len(fetches) == 2


def test_the_freshness_knob_can_disable_caching_entirely(yahoo, monkeypatch):
    """`FINANCIAL_RESEARCH_PRICE_TTL=0` means every call re-fetches — the same
    contract the price-series cache documents, now honored by both."""
    fundamentals, fetches = yahoo
    monkeypatch.setenv("FINANCIAL_RESEARCH_PRICE_TTL", "0")

    fundamentals._fetch_info("AAPL")
    fundamentals._fetch_info("AAPL")
    assert len(fetches) == 2


def test_a_failed_lookup_is_not_cached_as_an_answer(yahoo, monkeypatch):
    """An outage returns {} without poisoning the cache, so the next call gets a
    real chance rather than the outage being remembered for the whole session."""
    from financial_research_assistant import fundamentals as fund

    monkeypatch.setenv("FINANCIAL_RESEARCH_PRICE_TTL", "900")
    monkeypatch.setattr(fund, "_INFO_CACHE", {})

    def _down(_sym):
        raise RuntimeError("yahoo is down")

    monkeypatch.setattr(fund, "_ticker", _down)
    assert fund._fetch_info("AAPL") == {}
    assert "AAPL" not in fund._INFO_CACHE
