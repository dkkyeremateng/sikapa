from contextlib import asynccontextmanager


class RecordingWorker:
    """Stand-in for a running turn worker that records .cancel() calls."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class FakeTool:
    """Minimal stand-in for a loaded MCP tool (only ``.name`` matters here)."""

    def __init__(self, name: str) -> None:
        self.name = name


def fake_ibkr_tools_session(*tool_names):
    """A stand-in ibkr_tools_session async CM yielding the named fake tools."""

    @asynccontextmanager
    async def _cm():
        yield [FakeTool(n) for n in tool_names]

    return _cm


def install_fake_price_fetch(monkeypatch):
    import financial_research_assistant.tools as t

    def fake(sym, days):
        base = {"AAPL": 100.0, "MSFT": 50.0, "SPY": 400.0}.get(sym.upper(), 100.0)
        # a gently varying series so vol/drawdown/beta are well-defined
        return [(f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                 base * (1 + 0.02 * ((i % 7) - 3) / 100 + 0.0005 * i)) for i in range(60)]

    monkeypatch.setattr(t, "_fetch_daily", fake)


def install_stub_fx(monkeypatch, rates):
    """Stub the Yahoo FX fetch: rates maps currency -> USD-per-unit."""
    import financial_research_assistant.tools as t

    def fake(sym, days):
        # sym like "EURUSD=X" -> currency EUR
        ccy = sym.replace("USD=X", "")
        r = rates.get(ccy)
        return [("2025-01-01", r), ("2025-12-31", r)] if r else []

    monkeypatch.setattr(t, "_fetch_daily", fake)
