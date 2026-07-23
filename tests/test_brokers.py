"""Pluggable broker-provider registry tests.

Covers the abstraction that lets a *different* broker's read-only MCP server plug
in via config: the registry, the reusable read-only filter factory, runtime
selection, and a second broker mounting alongside IBKR through the shared
``broker_tools_session`` — with the trade-safety invariant intact.
"""

import json

import pytest

from financial_research_assistant import brokers
from .helpers.fakes import FakeTool as _FakeTool


# --- make_readonly_filter (the reusable per-broker policy) ------------------

def test_make_readonly_filter_keeps_reads_drops_writes():
    """A broker built from make_readonly_filter keeps its read-verb tools and
    drops anything not matching a read prefix — the default trade-safe shape."""
    filt = brokers.make_readonly_filter(("get_", "list_", "search_"))
    tools = [_FakeTool(n) for n in (
        "get_positions", "list_orders", "search_symbols",
        "place_order", "cancel_order", "modify_order",
    )]
    kept = {t.name for t in filt(tools)}
    assert kept == {"get_positions", "list_orders", "search_symbols"}


def test_make_readonly_filter_denylist_beats_allow():
    """The write denylist wins first: an order verb can never be re-exposed, even
    if it also matches a read prefix or is explicitly allow-listed."""
    filt = brokers.make_readonly_filter(
        ("get_",),
        write_deny=frozenset({"get_rich_place_order"}),
        allow=frozenset({"get_rich_place_order"}),
    )
    kept = {t.name for t in filt([_FakeTool("get_rich_place_order"), _FakeTool("get_quote")])}
    assert kept == {"get_quote"}


def test_make_readonly_filter_extra_allow_is_read_per_call():
    """extra_allow is a callable read on each filter call, so a runtime opt-in
    (like an env flag) takes effect without rebuilding the filter."""
    state = {"names": frozenset()}
    filt = brokers.make_readonly_filter(("get_",), extra_allow=lambda: state["names"])
    tools = [_FakeTool("authenticate"), _FakeTool("get_quote")]

    assert {t.name for t in filt(tools)} == {"get_quote"}
    state["names"] = frozenset({"authenticate"})
    assert {t.name for t in filt(tools)} == {"authenticate", "get_quote"}


# --- registry + selection --------------------------------------------------

def test_ibkr_is_registered_by_default():
    """IBKR ships as the reference provider."""
    assert "ibkr" in brokers.registered_broker_keys()


def test_register_and_active_selection(monkeypatch):
    """A newly registered provider is active by default; BROKER_PROVIDERS pins an
    explicit subset."""
    monkeypatch.setitem(
        brokers.BROKER_REGISTRY,
        "demo",
        brokers.BrokerProvider("demo", lambda: {"url": "u", "transport": "streamable_http"},
                               brokers.make_readonly_filter(("get_",))),
    )
    monkeypatch.delenv("BROKER_PROVIDERS", raising=False)
    keys = {p.key for p in brokers.active_broker_providers()}
    assert {"ibkr", "demo"} <= keys

    monkeypatch.setenv("BROKER_PROVIDERS", "demo")
    assert {p.key for p in brokers.active_broker_providers()} == {"demo"}

    monkeypatch.setenv("BROKER_PROVIDERS", "demo, nope")  # unknown key ignored
    assert {p.key for p in brokers.active_broker_providers()} == {"demo"}


def test_configured_brokers_reflects_env(monkeypatch):
    """configured_brokers only includes a broker whose config_from_env resolves."""
    import financial_research_assistant.tools as tools

    # IBKR configured via its env-backed config; late binding sees the patch.
    monkeypatch.setattr(
        tools, "_ibkr_server_config",
        lambda: {"command": "x", "args": [], "transport": "stdio"},
    )
    monkeypatch.delenv("BROKER_PROVIDERS", raising=False)
    got = brokers.configured_brokers()
    assert "ibkr" in got
    spec, filt = got["ibkr"]
    assert spec["transport"] == "stdio"
    # The bound filter is IBKR's strict allowlist: it drops a trade tool.
    assert {t.name for t in filt([_FakeTool("get_positions"), _FakeTool("place_order")])} == {
        "get_positions"
    }

    # Unconfigured -> absent.
    monkeypatch.setattr(tools, "_ibkr_server_config", lambda: None)
    assert "ibkr" not in brokers.configured_brokers()


# --- second broker mounts through the shared session -----------------------

@pytest.mark.asyncio
async def test_second_broker_mounts_alongside_ibkr(monkeypatch):
    """A hypothetical second broker registered with make_readonly_filter mounts
    through broker_tools_session next to IBKR: both brokers' read tools survive,
    both brokers' order tools are dropped, and every session closes on exit."""
    import financial_research_assistant.tools as tools

    # IBKR configured.
    monkeypatch.setattr(
        tools, "_ibkr_server_config",
        lambda: {"command": "x", "args": [], "transport": "stdio"},
    )
    # Register a second broker (Tradier-style: get_/list_ reads, explicit order
    # denylist) that is configured via its own env var.
    def _tradier_config():
        import os
        url = os.environ.get("TRADIER_MCP_URL")
        return {"url": url, "transport": "streamable_http"} if url else None

    monkeypatch.setitem(
        brokers.BROKER_REGISTRY,
        "tradier",
        brokers.BrokerProvider(
            "tradier", _tradier_config,
            brokers.make_readonly_filter(
                ("get_", "list_"),
                write_deny=frozenset({"place_order", "cancel_order"}),
            ),
        ),
    )
    monkeypatch.setenv("TRADIER_MCP_URL", "https://tradier.test/mcp")
    monkeypatch.delenv("BROKER_PROVIDERS", raising=False)
    monkeypatch.delenv("EXTRA_MCP_SERVERS", raising=False)

    closed = []

    class _Marker:
        def __init__(self, name):
            self.name = name

    class _SessionCtx:
        def __init__(self, name):
            self.name = name

        async def __aenter__(self):
            return _Marker(self.name)

        async def __aexit__(self, *a):
            closed.append(self.name)
            return False

    class _Client:
        def __init__(self, connections):
            self.connections = connections

        def session(self, name):
            return _SessionCtx(name)

    async def fake_load(session):
        if session.name == "ibkr":
            return [_FakeTool("get_positions"), _FakeTool("place_order")]
        return [_FakeTool("get_quotes"), _FakeTool("list_orders"), _FakeTool("place_order")]

    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", _Client)
    monkeypatch.setattr("langchain_mcp_adapters.tools.load_mcp_tools", fake_load)

    async with tools.broker_tools_session() as loaded:
        names = {t.name for t in loaded}
        # Reads from both brokers kept; order tools dropped on both sides.
        assert names == {"get_positions", "get_quotes", "list_orders"}
        assert "place_order" not in names
    assert set(closed) == {"ibkr", "tradier"}  # every session closed on exit


def test_broker_tools_session_alias():
    """ibkr_tools_session is retained as an alias of the registry-driven session
    (the stable seam graph.py imports and tests patch)."""
    import financial_research_assistant.tools as tools

    assert tools.ibkr_tools_session is tools.broker_tools_session
