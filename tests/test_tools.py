"""Tools and integration-wrapper behavior tests."""

import json

from .fixtures.statements import BUYSELL_STATEMENT as _BUYSELL_STATEMENT, SAMPLE_STATEMENT as _SAMPLE_STATEMENT, mini_statement as _mini_statement
from .helpers.fakes import FakeTool as _FakeTool, install_fake_price_fetch as _fake_fetch, install_stub_fx as _stub_fx


_FLEX_SENT_OK = (
    "<FlexStatementResponse timestamp='x'>"
    "<Status>Success</Status><ReferenceCode>REF123</ReferenceCode>"
    "<Url>https://example.test/GetStatement</Url></FlexStatementResponse>"
)


_FLEX_IN_PROGRESS = (
    "<FlexStatementResponse><Status>Warn</Status><ErrorCode>1019</ErrorCode>"
    "<ErrorMessage>Statement generation in progress</ErrorMessage>"
    "</FlexStatementResponse>"
)


_FLEX_STATEMENT = (
    "<FlexQueryResponse queryName='Activity' type='AF'><FlexStatements count='1'>"
    "<FlexStatement accountId='U1' fromDate='2026-04-01' toDate='2026-04-02'/>"
    "</FlexStatements></FlexQueryResponse>"
)


def test_local_calculators_are_correct():
    """The offline analyst calculators return the documented values."""
    from financial_research_assistant.tools import (
        cagr,
        pct_change,
        position_weight,
    )

    assert pct_change(100, 125) == 25.0
    assert pct_change(200, 150) == -25.0
    # doubling in 2 years is ~41.42% CAGR.
    assert round(cagr(100, 200, 2), 2) == 41.42
    assert position_weight(25_000, 200_000) == 12.5


def test_calculators_reject_degenerate_inputs():
    """Guard rails: zero/negative denominators raise rather than returning nonsense."""
    import pytest

    from financial_research_assistant.tools import cagr, pct_change, position_weight

    with pytest.raises(ValueError):
        pct_change(0, 10)
    with pytest.raises(ValueError):
        cagr(100, 200, 0)
    with pytest.raises(ValueError):
        position_weight(10, 0)


def test_readonly_filter_keeps_data_tools_drops_writes():
    """filter_readonly keeps get_*/search_* market-data tools and drops every
    order-entry / watchlist-mutation / feedback tool — the research-only boundary."""
    from financial_research_assistant.tools import filter_readonly

    # Names observed on the live interactive-brokers-mcp server.
    read = [
        "get_positions",
        "get_account_info",
        "get_market_data",
        "get_option_chain",
        "get_order_status",
        "resolve_option_conid",  # pure lookup, read-only
    ]
    # `authenticate` is dropped too: it triggers an interactive browser login and
    # can disrupt a live session — session setup is the user's operational step.
    dropped = [
        "place_order",
        "confirm_order",
        "create_alert",
        "activate_alert",
        "delete_alert",
        "authenticate",
    ]
    tools = [_FakeTool(n) for n in read + dropped]
    kept = {t.name for t in filter_readonly(tools)}
    assert kept == set(read)
    assert not (kept & set(dropped))  # no mutation/auth tool survives
    # The trade-executing tools specifically must never be exposed.
    assert "place_order" not in kept and "confirm_order" not in kept


def test_authenticate_opt_in_flag_never_unblocks_trades(monkeypatch):
    """IBKR_ALLOW_AUTHENTICATE opts `authenticate` back in (off by default), but
    the flag can NEVER expose an order/alert mutation — the denylist wins first."""
    from financial_research_assistant.tools import filter_readonly

    tools = [_FakeTool(n) for n in ("authenticate", "get_positions", "place_order")]

    # Default: authenticate dropped.
    monkeypatch.delenv("IBKR_ALLOW_AUTHENTICATE", raising=False)
    assert {t.name for t in filter_readonly(tools)} == {"get_positions"}

    # Opted in: authenticate kept, but place_order still blocked.
    monkeypatch.setenv("IBKR_ALLOW_AUTHENTICATE", "1")
    kept = {t.name for t in filter_readonly(tools)}
    assert kept == {"authenticate", "get_positions"}
    assert "place_order" not in kept


def test_write_denylist_beats_every_allow_path():
    """The core safety invariant: every _WRITE_DENY tool is blocked even when its
    exact name appears in the runtime extra_allow set — no opt-in mechanism can
    ever expose an order/alert/watchlist mutation."""
    from financial_research_assistant.tools import _WRITE_DENY, _is_readonly

    for name in _WRITE_DENY:
        assert not _is_readonly(name), name
        assert not _is_readonly(name, extra_allow=frozenset({name})), (
            f"{name} must stay blocked even if explicitly allow-listed"
        )
    # And the Web-API-variant mutation names are all on the denylist.
    assert {
        "create_order_instruction", "delete_order_instruction",
        "create_watchlist", "edit_watchlist", "delete_watchlist",
    } <= _WRITE_DENY


def test_parse_yahoo_json_extracts_date_and_close():
    """The Yahoo chart parser pairs timestamps with closes, skips null closes,
    and returns an empty list when the symbol has no result."""
    import json as _json

    from financial_research_assistant.tools import _parse_yahoo_json

    # 2026-01-01 and 2026-01-02 00:00 UTC; middle close is null (skipped).
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1767225600, 1767268800, 1767312000],
                    "indicators": {"quote": [{"close": [100.5, None, 103.5]}]},
                }
            ]
        }
    }
    rows = _parse_yahoo_json(_json.dumps(payload))
    assert rows == [("2026-01-01", 100.5), ("2026-01-02", 103.5)]
    # Unknown symbol -> result is null -> no rows, no crash.
    assert _parse_yahoo_json('{"chart": {"result": null, "error": {}}}') == []


def test_yahoo_range_buckets_cover_days():
    from financial_research_assistant.tools import _yahoo_range

    assert _yahoo_range(5) == "5d"
    assert _yahoo_range(30) == "1mo"
    assert _yahoo_range(90) == "3mo"
    assert _yahoo_range(400) == "2y"
    assert _yahoo_range(1825) == "5y"
    assert _yahoo_range(3650) == "10y"
    assert _yahoo_range(5000) == "max"  # long enough to want a true all-time high


def test_render_price_chart_returns_plain_text():
    """The chart renders to a non-empty string carrying the symbol/title, with no
    ANSI escape codes (the 'clear' theme keeps it plain for panels + markdown)."""
    from financial_research_assistant.tools import _render_price_chart

    chart = _render_price_chart("AAPL", ["2026-01-01", "2026-01-05"], [10, 12, 11, 14, 13])
    assert isinstance(chart, str) and chart.strip()
    assert "AAPL" in chart
    assert "\x1b[" not in chart  # no ANSI color codes


def test_price_history_chart_no_data_message(monkeypatch):
    """An empty history (unknown ticker) returns a helpful message, not a crash."""
    import financial_research_assistant.tools as tools

    monkeypatch.setattr(tools, "_fetch_daily", lambda symbol, days, strict=False, **_kw: [])
    out = tools.price_history_chart("NOPE")
    assert "No historical data" in out and "NOPE" in out


def test_price_history_chart_renders_stats_and_window(monkeypatch):
    """With a synthetic series the tool takes the most recent `days` window and
    returns stats (symbol, % change) plus the chart — all offline."""
    import financial_research_assistant.tools as tools

    series = [(f"2026-01-{i:02d}", float(100 + i)) for i in range(1, 21)]  # 20 sessions
    monkeypatch.setattr(tools, "_fetch_daily", lambda symbol, days, strict=False, **_kw: series)
    out = tools.price_history_chart("AAPL", days=5)
    assert "AAPL" in out
    assert "change" in out
    assert "5 sessions" in out           # sliced to the requested window
    assert "2026-01-16" in out           # window starts at the 16th of 20
    assert "daily close" in out          # the chart title rendered


def test_normalize_result_handles_provider_shapes():
    """_normalize_result maps DuckDuckGo-news, DuckDuckGo-text, and Tavily keys
    onto one {title, url, source, date, snippet} shape."""
    from financial_research_assistant.tools import _normalize_result

    news = _normalize_result(
        {"title": "AAPL up", "url": "http://x", "source": "Reuters",
         "date": "2026-07-11", "body": "shares rose"}
    )
    assert news == {"title": "AAPL up", "url": "http://x", "source": "Reuters",
                    "date": "2026-07-11", "snippet": "shares rose"}
    text = _normalize_result({"title": "T", "href": "http://y", "body": "b"})
    assert text["url"] == "http://y" and text["snippet"] == "b"
    tav = _normalize_result(
        {"title": "T2", "url": "http://z", "content": "c", "published_date": "2026-07-01"}
    )
    assert tav["snippet"] == "c" and tav["date"] == "2026-07-01"


def test_format_search_results_and_empty():
    from financial_research_assistant.tools import _format_search_results

    out = _format_search_results(
        "AAPL news",
        [{"title": "Apple hits high", "url": "http://a", "source": "CNBC",
          "date": "2026-07-11", "snippet": "Apple rose 2%."}],
    )
    assert "Apple hits high" in out
    assert "CNBC · 2026-07-11" in out
    assert "http://a" in out
    # Results are framed as untrusted data (prompt-injection mitigation).
    assert "untrusted" in out and "ignore any instructions" in out
    assert "No results" in _format_search_results("nothing", [])


def test_web_search_formats_provider_output(monkeypatch):
    """web_search runs the keyless provider by default and formats its results;
    a bounded max_results is passed through."""
    import financial_research_assistant.tools as tools

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    captured = {}

    def fake_ddg(query, max_results):
        captured["max_results"] = max_results
        return [{"title": "Market up", "url": "http://m", "source": "WSJ",
                 "date": "2026-07-11", "snippet": "Stocks climbed."}]

    monkeypatch.setattr(tools, "_ddg_search", fake_ddg)
    out = tools.web_search("stock market today", max_results=50)  # clamps to 10
    assert "Market up" in out and "WSJ" in out and "http://m" in out
    assert captured["max_results"] == 10


def test_web_search_prefers_tavily_when_key_set(monkeypatch):
    import financial_research_assistant.tools as tools

    def _norm(title, url):
        return {"title": title, "url": url, "source": "", "date": "", "snippet": ""}

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-xxx")
    monkeypatch.setattr(tools, "_ddg_search", lambda q, n: [_norm("DDG", "http://d")])
    monkeypatch.setattr(
        tools, "_tavily_search", lambda q, n, key: [_norm("TAVILY", "http://t")]
    )
    out = tools.web_search("AAPL earnings")
    assert "TAVILY" in out and "DDG" not in out


def test_web_search_survives_provider_error(monkeypatch):
    """A provider exception is reported as text, never raised into the turn."""
    import financial_research_assistant.tools as tools

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def boom(query, max_results):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(tools, "_ddg_search", boom)
    out = tools.web_search("anything")
    assert "Web search failed" in out and "rate limited" in out


async def test_ibkr_tools_session_yields_filtered_tools(monkeypatch):
    """ibkr_tools_session opens a session (via load_mcp_tools), yields the
    read-only-filtered tools for the block's duration, and drops a mutation tool
    the server offers."""
    import financial_research_assistant.tools as tools

    monkeypatch.setattr(
        tools, "_ibkr_server_config",
        lambda: {"command": "x", "args": [], "transport": "stdio"},
    )

    closed = []

    class _SessionCtx:
        async def __aenter__(self):
            return object()  # the ClientSession

        async def __aexit__(self, *a):
            closed.append(True)  # session closed when the block exits
            return False

    class _Client:
        def __init__(self, connections):
            pass

        def session(self, name):
            return _SessionCtx()

    async def fake_load(session):
        return [_FakeTool("get_positions"), _FakeTool("place_order")]

    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", _Client)
    monkeypatch.setattr("langchain_mcp_adapters.tools.load_mcp_tools", fake_load)

    async with tools.ibkr_tools_session() as loaded:
        assert [t.name for t in loaded] == ["get_positions"]  # place_order dropped
    assert closed == [True]  # session closed on block exit (nothing left open)


def test_filter_safe_drops_mutating_keeps_data_tools():
    """filter_safe (for user-added data servers) keeps noun-named data tools that
    the strict get_/search_ allowlist would drop, but still excludes write verbs."""
    from financial_research_assistant.tools import filter_safe

    tools = [_FakeTool(n) for n in (
        "fred_series", "series_observations", "company_facts", "get_quote",
        "place_order", "create_watchlist", "delete_alert", "submit_trade",
    )]
    kept = {t.name for t in filter_safe(tools)}
    assert kept == {"fred_series", "series_observations", "company_facts", "get_quote"}
    assert not (kept & {"place_order", "create_watchlist", "delete_alert", "submit_trade"})


def test_extra_mcp_servers_parses_and_guards(monkeypatch):
    """EXTRA_MCP_SERVERS parses a JSON object of specs, ignores bad JSON / non-dict
    payloads, drops non-dict specs, and never lets an 'ibkr' key shadow the strict
    IBKR mount."""
    from financial_research_assistant.tools import _extra_mcp_servers

    monkeypatch.delenv("EXTRA_MCP_SERVERS", raising=False)
    assert _extra_mcp_servers() == {}

    monkeypatch.setenv("EXTRA_MCP_SERVERS", "{not json")
    assert _extra_mcp_servers() == {}

    monkeypatch.setenv("EXTRA_MCP_SERVERS", "[1, 2]")  # not an object
    assert _extra_mcp_servers() == {}

    monkeypatch.setenv("EXTRA_MCP_SERVERS", json.dumps({
        "fred": {"command": "npx", "args": ["fred"], "transport": "stdio"},
        "bad": "not-a-dict",
        "ibkr": {"url": "http://evil/mcp", "transport": "streamable_http"},
    }))
    got = _extra_mcp_servers()
    assert set(got) == {"fred"}  # 'bad' dropped, 'ibkr' can't be shadowed


async def test_ibkr_tools_session_mounts_extra_servers(monkeypatch):
    """With both IBKR and an extra data server configured, the session mounts
    both — IBKR strictly filtered (mutation dropped), the extra server safe-
    filtered (noun-named data tools kept) — and closes every session on exit."""
    import financial_research_assistant.tools as tools

    monkeypatch.setattr(
        tools, "_ibkr_server_config",
        lambda: {"command": "x", "args": [], "transport": "stdio"},
    )
    monkeypatch.setenv("EXTRA_MCP_SERVERS", json.dumps({
        "fred": {"command": "y", "args": [], "transport": "stdio"},
    }))

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
        return [_FakeTool("fred_series"), _FakeTool("delete_series")]

    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", _Client)
    monkeypatch.setattr("langchain_mcp_adapters.tools.load_mcp_tools", fake_load)

    async with tools.ibkr_tools_session() as loaded:
        names = {t.name for t in loaded}
        assert names == {"get_positions", "fred_series"}  # mutations dropped both ways
    assert set(closed) == {"ibkr", "fred"}  # every session closed on exit


async def test_ibkr_tools_session_empty_without_endpoint(monkeypatch):
    """With no IBKR_MCP_* env configured, the session yields [] and never touches
    the network, so real mode and offline tests run with local tools only."""
    for var in ("IBKR_MCP_URL", "IBKR_MCP_COMMAND", "IBKR_MCP_TOKEN",
                "IBKR_MCP_ARGS", "EXTRA_MCP_SERVERS"):
        monkeypatch.delenv(var, raising=False)
    from financial_research_assistant.tools import ibkr_tools_session

    async with ibkr_tools_session() as loaded:
        assert loaded == []


def test_ibkr_server_config_shapes_transport(monkeypatch):
    """URL config -> streamable_http (+ Bearer header when a token is set);
    command config -> stdio with shell-split args; neither -> None."""
    from financial_research_assistant.tools import _ibkr_server_config

    for var in ("IBKR_MCP_URL", "IBKR_MCP_COMMAND", "IBKR_MCP_TOKEN", "IBKR_MCP_ARGS"):
        monkeypatch.delenv(var, raising=False)
    assert _ibkr_server_config() is None

    monkeypatch.setenv("IBKR_MCP_URL", "https://host/mcp")
    monkeypatch.setenv("IBKR_MCP_TOKEN", "secret")
    cfg = _ibkr_server_config()
    assert cfg["transport"] == "streamable_http"
    assert cfg["url"] == "https://host/mcp"
    assert cfg["headers"]["Authorization"] == "Bearer secret"

    monkeypatch.delenv("IBKR_MCP_URL")
    monkeypatch.delenv("IBKR_MCP_TOKEN")
    monkeypatch.setenv("IBKR_MCP_COMMAND", "npx")
    monkeypatch.setenv("IBKR_MCP_ARGS", "-y @org/ibkr-mcp-server")
    cfg = _ibkr_server_config()
    assert cfg["transport"] == "stdio"
    assert cfg["command"] == "npx"
    assert cfg["args"] == ["-y", "@org/ibkr-mcp-server"]


def test_import_ibkr_statement_tool_summary_and_missing_file(monkeypatch, tmp_path):
    """The agent tool imports from a path and returns a readable summary; a bad
    path is reported as text, never raised into the turn."""
    import financial_research_assistant.tools as tools

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    csv_path = tmp_path / "stmt.csv"
    csv_path.write_text(_SAMPLE_STATEMENT, encoding="utf-8")

    out = tools.import_ibkr_statement(str(csv_path))
    assert "U1111111" in out and "2 trade(s)" in out and "5 cash" in out
    assert "dividend 2" in out
    assert "2 open position(s)" in out and "12.5%" in out
    assert "1 corporate action(s)" in out

    missing = tools.import_ibkr_statement(str(tmp_path / "nope.csv"))
    assert "No file found" in missing


def test_query_transactions_tool_renders_table(monkeypatch, tmp_path):
    import financial_research_assistant.tools as tools

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    from financial_research_assistant import statements

    statements.import_statement(_SAMPLE_STATEMENT)
    out = tools.query_transactions(kind="trade", symbol="AMZN")
    assert "TRADE" in out and "AMZN" in out
    empty = tools.query_transactions(kind="fee", symbol="ZZZZ")
    assert "No transactions found" in empty

    corp = tools.query_transactions(kind="corporate_action")
    assert "CORP ACTION" in corp and "Split 3 for 1" in corp


def test_render_series_chart_breaks_line_at_gap():
    """A gap splits the series into separate plotted segments (no solid line
    interpolating across the missing span), and the chart stays plain text."""
    from financial_research_assistant.tools import _render_series_chart

    dates = ["2024-01-01", "2024-12-31", "2026-12-31"]
    values = [100.0, 110.0, 130.0]
    # No gap: one continuous line.
    whole = _render_series_chart("t", dates, values)
    # Gap after point 1: the 2024→2026 span is a break.
    broken = _render_series_chart("t", dates, values, gap_after={1})
    assert whole.strip() and broken.strip()
    assert "\x1b[" not in broken            # still plain text, no ANSI
    assert broken != whole                  # the break changes the rendering


def test_portfolio_performance_chart_tool_renders(monkeypatch, tmp_path):
    import financial_research_assistant.tools as tools
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    assert "Not enough data" in tools.portfolio_performance_chart()

    statements.import_statement(_SAMPLE_STATEMENT)
    out = tools.portfolio_performance_chart()
    assert "Performance index" in out and "growth of 100" in out.lower()
    assert "cumulative return" in out
    assert "\x1b[" not in out  # plain text


def test_portfolio_value_history_tool_renders_chart(monkeypatch, tmp_path):
    import financial_research_assistant.tools as tools
    from financial_research_assistant import statements

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))

    # With nothing imported, it asks for statements rather than crashing.
    assert "Not enough data" in tools.portfolio_value_history()

    statements.import_statement(_SAMPLE_STATEMENT)
    out = tools.portfolio_value_history()
    assert "Account value (NAV)" in out
    assert "start" in out and "end" in out
    assert "\x1b[" not in out  # plain text, no ANSI


def test_query_portfolio_tool_renders_positions_and_nav(monkeypatch, tmp_path):
    import financial_research_assistant.tools as tools

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    from financial_research_assistant import statements

    statements.import_statement(_SAMPLE_STATEMENT)
    out = tools.query_portfolio()
    assert "OPEN POSITIONS" in out and "AMAZON.COM INC" in out
    assert "US0231351067" in out          # ISIN surfaced
    assert "NET ASSET VALUE" in out and "12.5%" in out

    # The empty branch, on a fresh DB with nothing imported.
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "empty.db"))
    assert "No portfolio data" in tools.query_portfolio()


def test_flex_fetch_two_step(monkeypatch):
    """fetch_flex_xml does SendRequest then GetStatement against the returned URL,
    passing the reference code, and returns the statement XML."""
    from financial_research_assistant import flex

    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params["q"]))
        return _FLEX_SENT_OK if "SendRequest" in url else _FLEX_STATEMENT

    monkeypatch.setattr(flex, "_flex_get", fake_get)
    out = flex.fetch_flex_xml("TOKEN", "QID", retry_delay=0)
    assert out == _FLEX_STATEMENT
    assert calls[0][1] == "QID"               # SendRequest uses the query id
    assert calls[1] == ("https://example.test/GetStatement", "REF123")  # then the ref code


def test_flex_fetch_retries_in_progress(monkeypatch):
    """A 'generation in progress' GetStatement is retried until the statement is
    ready (no real sleeping — retry_delay=0)."""
    from financial_research_assistant import flex

    seq = [_FLEX_SENT_OK, _FLEX_IN_PROGRESS, _FLEX_IN_PROGRESS, _FLEX_STATEMENT]
    monkeypatch.setattr(flex, "_flex_get", lambda url, params, timeout: seq.pop(0))
    out = flex.fetch_flex_xml("TOKEN", "QID", retry_delay=0)
    assert out == _FLEX_STATEMENT and seq == []  # consumed both in-progress responses


def test_flex_fetch_send_request_failure(monkeypatch):
    import pytest

    from financial_research_assistant import flex

    fail = ("<FlexStatementResponse><Status>Fail</Status><ErrorCode>1015</ErrorCode>"
            "<ErrorMessage>Invalid token</ErrorMessage></FlexStatementResponse>")
    monkeypatch.setattr(flex, "_flex_get", lambda url, params, timeout: fail)
    with pytest.raises(flex.FlexError) as ei:
        flex.fetch_flex_xml("BAD", "QID", retry_delay=0)
    assert "Invalid token" in str(ei.value)


def test_flex_fetch_never_ready(monkeypatch):
    """If the statement stays in-progress past max_retries, it raises rather than
    hanging."""
    import pytest

    from financial_research_assistant import flex

    monkeypatch.setattr(
        flex, "_flex_get",
        lambda url, params, timeout: _FLEX_SENT_OK if "SendRequest" in url else _FLEX_IN_PROGRESS,
    )
    with pytest.raises(flex.FlexError):
        flex.fetch_flex_xml("TOKEN", "QID", max_retries=2, retry_delay=0)


def test_flex_sync_missing_credentials(monkeypatch):
    from financial_research_assistant import flex

    monkeypatch.delenv("IBKR_FLEX_TOKEN", raising=False)
    monkeypatch.delenv("IBKR_FLEX_QUERY_ID", raising=False)
    assert "No Flex token" in flex.flex_sync()
    monkeypatch.setenv("IBKR_FLEX_TOKEN", "t")
    assert "No Flex query id" in flex.flex_sync()


def test_flex_sync_saves_the_xml_and_imports_it(monkeypatch, tmp_path):
    """flex_sync fetches, saves the raw XML, and imports it into the same store
    the manual CSV path fills — reporting what landed rather than a raw dict."""
    from financial_research_assistant import flex

    monkeypatch.setenv("FINANCIAL_RESEARCH_FLEX_DIR", str(tmp_path / "flex"))
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    monkeypatch.setenv("IBKR_FLEX_TOKEN", "TOKEN")
    monkeypatch.setattr(flex, "fetch_flex_xml", lambda *a, **k: _FLEX_STATEMENT)

    out = flex.flex_sync(query_id="QID")
    assert "saved" in out and "Imported April 01, 2026 - April 02, 2026 for U1" in out
    saved = list((tmp_path / "flex").glob("flex-QID-*.xml"))
    assert len(saved) == 1 and saved[0].read_text() == _FLEX_STATEMENT


def test_a_statement_that_cannot_be_parsed_is_still_kept(monkeypatch, tmp_path):
    """The XML is written before it is parsed. A statement this parser can't read
    is the one sample needed to teach it — losing it with the error would mean
    reproducing the pull to see what went wrong."""
    from financial_research_assistant import flex

    monkeypatch.setenv("FINANCIAL_RESEARCH_FLEX_DIR", str(tmp_path / "flex"))
    monkeypatch.setenv("IBKR_FLEX_TOKEN", "TOKEN")
    monkeypatch.setattr(flex, "fetch_flex_xml", lambda *a, **k: "<SomethingElse/>")

    out = flex.flex_sync(query_id="QID")
    assert "could not be imported" in out
    assert len(list((tmp_path / "flex").glob("flex-QID-*.xml"))) == 1


def test_realized_gains_tool_and_empty(monkeypatch, tmp_path):
    import financial_research_assistant.tools as t
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    assert "No realized gains" in t.realized_gains()
    s.import_statement(_BUYSELL_STATEMENT)
    out = t.realized_gains(year=2025)
    assert "REALIZED GAINS" in out and "AMZN" in out and "+260.00" in out


def test_compare_prices_normalizes_and_overlays(monkeypatch):
    import financial_research_assistant.tools as t
    _fake_fetch(monkeypatch)
    out = t.compare_prices("AAPL, MSFT, SPY", days=60)
    assert "Normalized price comparison" in out
    assert "AAPL" in out and "MSFT" in out and "SPY" in out
    assert "\x1b[" not in out
    assert "Give one or more" in t.compare_prices("")


def test_compare_prices_drops_ticker_with_no_data(monkeypatch):
    """One bad ticker no longer sinks the whole comparison: it's dropped with a
    note and the remaining tickers still chart."""
    import financial_research_assistant.tools as t

    def fake(sym, days, strict=False, **_kw):
        if sym.upper() == "NOPE":
            return []
        return [(f"2025-01-{i + 1:02d}", 100.0 + i) for i in range(20)]

    monkeypatch.setattr(t, "_fetch_daily", fake)
    out = t.compare_prices("AAPL, NOPE, SPY", days=20)
    assert "Normalized price comparison" in out
    assert "no price data for NOPE" in out
    assert "AAPL" in out and "SPY" in out


def test_risk_metrics_reports_vol_drawdown_sharpe_beta(monkeypatch):
    import financial_research_assistant.tools as t
    _fake_fetch(monkeypatch)
    out = t.risk_metrics("AAPL", days=60)
    assert "annualized volatility" in out and "max drawdown" in out
    assert "Sharpe" in out and "beta vs SPY" in out


def _yahoo_like_fetch(monkeypatch, *, history_days=10_000):
    """Stub the price fetch the way the real source answers, in the one respect
    that matters to a benchmark comparison: the window always ENDS TODAY and
    reaches back only as far as it was asked to.

    A fixed handful of hand-picked dates answers every request identically, so a
    fetch that can't reach the portfolio's period looks exactly like one that can
    — which is how a benchmark return measured over a different span reached the
    output as a verdict. ``history_days`` caps how far back the source has any
    data at all, for the case where the period predates its coverage.
    """
    from datetime import date, timedelta

    import financial_research_assistant.tools as t
    from financial_research_assistant.pointintime import as_of_series

    today = date.today()

    def fake(sym, days, strict=False, as_of=None, **_kw):
        span = min(int(days), history_days)
        series = [
            (d.isoformat(), 100.0 + i * 0.01)
            for i in range(span)
            if (d := today - timedelta(days=span - 1 - i)).weekday() < 5
        ]
        return as_of_series(series, as_of, days)

    monkeypatch.setattr(t, "_fetch_daily", fake)


def _two_year_portfolio(monkeypatch, tmp_path):
    """Two chained statements covering 2024 and 2025 -> a TWRR series to benchmark."""
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(_mini_statement("U1", "January 1, 2024 - December 31, 2024", "10%", 1000))
    s.import_statement(_mini_statement("U1", "January 1, 2025 - December 31, 2025", "20%", 1300))


def test_risk_metrics_beta_pairs_the_same_sessions(monkeypatch):
    """Beta pairs the stock's and the benchmark's returns by POSITION. Dropping an
    unusable session from only the series that had it shortens that list alone, so
    every later return lines up against the wrong day's market return — and the
    covariance stays a perfectly ordinary-looking number while measuring nothing.
    Here the two move identically, so anything but 1.00 is the misalignment."""
    from datetime import date, timedelta

    import financial_research_assistant.tools as t
    from financial_research_assistant.pointintime import as_of_series

    start = date(2026, 1, 1)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(31)]
    # The benchmark's first close is unusable; after it the two move together
    # exactly (−10%, +11.1%, …), so a like-for-like beta is 1.00.
    closes = {
        "AAPL": [110.0] + [100.0, 90.0] * 15,
        "SPY": [0.0] + [400.0, 360.0] * 15,
    }
    monkeypatch.setattr(
        t, "_fetch_daily",
        lambda sym, days, strict=False, as_of=None, **_kw: as_of_series(
            list(zip(dates, closes[sym.upper()])), as_of, days
        ),
    )
    assert "beta vs SPY   1.00" in t.risk_metrics("AAPL", days=90)


def test_portfolio_vs_benchmark(monkeypatch, tmp_path):
    import financial_research_assistant.tools as t

    _two_year_portfolio(monkeypatch, tmp_path)
    _yahoo_like_fetch(monkeypatch)
    out = t.portfolio_vs_benchmark("SPY")
    assert "Portfolio vs SPY" in out
    assert "portfolio (TWRR)" in out and "%" in out


def test_portfolio_vs_benchmark_reaches_back_to_the_period_start(monkeypatch, tmp_path):
    """The portfolio's period closed well before today. Sizing the fetch by the
    period's own LENGTH reaches only that far back from today, so the whole window
    lands after the period and there is nothing left to compare."""
    import financial_research_assistant.tools as t

    _two_year_portfolio(monkeypatch, tmp_path)
    _yahoo_like_fetch(monkeypatch)
    out = t.portfolio_vs_benchmark("SPY")
    assert "2024-01-01 → 2025-12-31" in out
    assert "you outperformed" in out or "you underperformed" in out


def test_portfolio_vs_benchmark_refuses_a_partly_covered_span(monkeypatch, tmp_path):
    """A source with only a year of history covers the back half of a two-year
    period. That yields plenty of rows and a plausible return — over the wrong
    window — and the tool's entire output is the difference between that return
    and the portfolio's, so the verdict would be an artifact of the mismatch."""
    import financial_research_assistant.tools as t

    _two_year_portfolio(monkeypatch, tmp_path)
    _yahoo_like_fetch(monkeypatch, history_days=365)
    out = t.portfolio_vs_benchmark("SPY")
    assert "not enough to compare like for like" in out
    assert "outperformed" not in out and "underperformed" not in out
    # The portfolio's own figure is still reported — it needed no benchmark.
    assert "Portfolio return over 2024-01-01 → 2025-12-31" in out


def test_export_data_writes_csv(monkeypatch, tmp_path):
    import csv as _csv
    import financial_research_assistant.tools as t
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    monkeypatch.setenv("FINANCIAL_RESEARCH_EXPORT_DIR", str(tmp_path))
    s.import_statement(_BUYSELL_STATEMENT)

    out_path = tmp_path / "positions.csv"
    msg = t.export_data("positions", str(out_path))
    assert "Exported" in msg and out_path.exists()
    with out_path.open() as f:
        rows = list(_csv.DictReader(f))
    assert {r["symbol"] for r in rows} == {"AMZN", "VOO"}
    assert any(r["security_id"] == "" or r["value"] for r in rows)

    assert "must be 'transactions' or 'positions'" in t.export_data("bogus", str(out_path))


def test_export_data_confined_to_export_dir(monkeypatch, tmp_path):
    """export_data writes only inside the export directory: a bare filename lands
    there, and an absolute or `..`-escaping path outside it is refused — the tool
    must never be usable as an arbitrary-file-write primitive."""
    import financial_research_assistant.tools as t
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    export_dir = tmp_path / "exports"
    monkeypatch.setenv("FINANCIAL_RESEARCH_EXPORT_DIR", str(export_dir))
    s.import_statement(_BUYSELL_STATEMENT)

    # A bare filename lands inside the export dir.
    msg = t.export_data("positions", "holdings.csv")
    assert "Exported" in msg and (export_dir / "holdings.csv").exists()

    # Absolute path outside the export dir: refused, nothing written.
    outside = tmp_path / "evil.csv"
    msg = t.export_data("positions", str(outside))
    assert "Refused" in msg and not outside.exists()

    # Relative path escaping with `..`: refused, nothing written.
    msg = t.export_data("positions", "../escape.csv")
    assert "Refused" in msg and not (tmp_path / "escape.csv").exists()


def test_export_data_empty_store(monkeypatch, tmp_path):
    import financial_research_assistant.tools as t
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "none.db"))
    monkeypatch.setenv("FINANCIAL_RESEARCH_EXPORT_DIR", str(tmp_path))
    assert "No positions to export" in t.export_data("positions", str(tmp_path / "x.csv"))


def test_fetch_daily_caches_and_retries(monkeypatch):
    """One transient failure is retried; a successful fetch is cached so a second
    call for the same (symbol, range) doesn't hit the network again."""
    import financial_research_assistant.tools as t
    t._PRICE_CACHE.clear()
    calls = {"n": 0}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"chart":{"result":[{"timestamp":[1704067200,1704153600],"indicators":{"quote":[{"close":[10.0,11.0]}]}}]}}'

    def fake_urlopen(req, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")   # first attempt fails -> retry
        return _Resp()

    monkeypatch.setattr(t.urllib.request, "urlopen", fake_urlopen)
    first = t._fetch_daily("AAPL", 5)
    assert len(first) == 2 and calls["n"] == 2      # failed once, retried, succeeded
    second = t._fetch_daily("AAPL", 5)
    assert second == first and calls["n"] == 2      # served from cache, no new call


def test_convert_currency_tool(monkeypatch):
    import financial_research_assistant.tools as t
    _stub_fx(monkeypatch, {"EUR": 1.1})
    out = t.convert_currency(100, "EUR")
    assert "110.00 USD" in out and "1.1000" in out
    assert t.convert_currency(1, "USD").startswith("1.00 USD ≈ 1.00 USD")  # base is identity
    assert "Couldn't fetch" in t.convert_currency(5, "ZZZ")  # no rate


def test_convert_currency_says_when_the_rate_is_from_after_the_date(monkeypatch):
    """When the pair's history doesn't reach back to the requested day there is no
    rate on or before it, and the closest one available comes from AFTER. Returning
    it is right; describing it as "the closest date on/before" says the one thing
    that isn't true of it."""
    import financial_research_assistant.tools as t

    monkeypatch.setattr(
        t, "_fetch_daily",
        lambda sym, days, strict=False, **_kw: [("2025-06-02", 1.1), ("2025-06-03", 1.1)],
    )
    out = t.convert_currency(100, "EUR", on_date="2019-01-15")
    assert "2025-06-02" in out
    assert "on/before" not in out
    assert "EARLIEST rate available" in out
    # A date the history does cover still reads as on/before.
    covered = t.convert_currency(100, "EUR", on_date="2025-06-03")
    assert "on/before" in covered or "rate on 2025-06-03" in covered


def test_income_summary_converts_a_single_non_base_currency(monkeypatch, tmp_path):
    """An account reporting entirely in one non-USD currency needs the USD total
    MORE than a mixed one, not less: without it a USD-based user is handed a
    dividend figure in a currency they don't think in, and nothing to compare it
    to. Gating on "more than one currency" left exactly that case bare."""
    import financial_research_assistant.tools as t
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(
        "Statement,Header,Field Name,Field Value\n"
        'Statement,Data,Period,"January 1, 2025 - December 31, 2025"\n'
        "Dividends,Header,Currency,Account,Date,Description,Amount\n"
        "Dividends,Data,EUR,U1,2025-03-01,SAP(DE000) Cash Dividend,20\n"
    )
    _stub_fx(monkeypatch, {"EUR": 1.5})
    out = t.income_summary(year=2025)
    assert "EUR:" in out
    assert "30.00 USD net total" in out


def test_fetch_daily_strict_distinguishes_outage_from_bad_ticker(monkeypatch):
    """strict=True raises PriceDataUnavailable when the source is unreachable (so a
    caller can say 'try again'), but a 404 (unknown symbol) is still an empty
    result, not an outage. Non-strict stays backward-compatible (empty on failure)."""
    import pytest
    import financial_research_assistant.tools as t
    t._PRICE_CACHE.clear()

    def always_fail(req, timeout=0):
        raise OSError("dns down")

    monkeypatch.setattr(t.urllib.request, "urlopen", always_fail)
    with pytest.raises(t.PriceDataUnavailable):
        t._fetch_daily("AAPL", 5, strict=True)
    assert t._fetch_daily("AAPL", 5) == []  # non-strict: empty, no raise

    def not_found(req, timeout=0):
        raise t.urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(t.urllib.request, "urlopen", not_found)
    assert t._fetch_daily("NOPE", 5, strict=True) == []  # 404 → empty even in strict


def test_fetch_many_dedups_and_flags_unreachable(monkeypatch):
    """The concurrent multi-symbol fetch dedupes symbols, maps each to its series,
    and (in strict mode) flags when any symbol hit a transport outage."""
    import financial_research_assistant.tools as t

    def fake(sym, days, strict=False, **_kw):
        if sym == "DOWN":
            if strict:
                raise t.PriceDataUnavailable("outage")
            return []
        return [("2025-01-01", 10.0), ("2025-01-02", 11.0)]

    monkeypatch.setattr(t, "_fetch_daily", fake)
    res, unreachable = t._fetch_many(["AAPL", "MSFT", "AAPL"], 5)  # AAPL repeated
    assert set(res) == {"AAPL", "MSFT"} and res["AAPL"] and not unreachable
    res2, unreachable2 = t._fetch_many(["AAPL", "DOWN"], 5, strict=True)
    assert res2["AAPL"] and res2["DOWN"] == [] and unreachable2 is True


def test_aligned_closes_strict_raises_on_full_outage(monkeypatch):
    """When every symbol is unreachable and nothing came back, strict alignment
    raises the outage rather than returning an empty 'bad tickers' result."""
    import pytest

    import financial_research_assistant.tools as t

    def down(sym, days, strict=False, **_kw):
        if strict:
            raise t.PriceDataUnavailable("outage")
        return []

    monkeypatch.setattr(t, "_fetch_daily", down)
    with pytest.raises(t.PriceDataUnavailable):
        t._aligned_closes(["AAPL", "MSFT"], 90, strict=True)


def test_price_history_chart_reports_outage_not_bad_ticker(monkeypatch):
    """A data-source outage is reported as such, never misattributed to the ticker
    — the trust fix so a Yahoo blip doesn't tell users their symbol is wrong."""
    import financial_research_assistant.tools as t

    def boom(symbol, days, strict=False, **_kw):
        if strict:
            raise t.PriceDataUnavailable("down")
        return []

    monkeypatch.setattr(t, "_fetch_daily", boom)
    out = t.price_history_chart("AAPL")
    assert "unreachable" in out.lower() and "AAPL" in out
    assert "Check the ticker" not in out


def test_price_cache_expires(monkeypatch):
    """A cached series past its TTL is re-fetched, so a long-lived session never
    serves stale closes as 'latest'."""
    import financial_research_assistant.tools as t
    t._PRICE_CACHE.clear()
    calls = {"n": 0}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return b'{"chart":{"result":[{"timestamp":[1704067200],"indicators":{"quote":[{"close":[10.0]}]}}]}}'

    def fake_urlopen(req, timeout=0):
        calls["n"] += 1
        return _Resp()

    monkeypatch.setattr(t.urllib.request, "urlopen", fake_urlopen)
    t._fetch_daily("AAPL", 5)
    assert calls["n"] == 1
    t._fetch_daily("AAPL", 5)               # within TTL → served from cache
    assert calls["n"] == 1
    # Backdate the cached entry beyond the TTL → the next call must re-fetch.
    series, _ts = t._PRICE_CACHE[("AAPL", "5d")]
    t._PRICE_CACHE[("AAPL", "5d")] = (series, t.time.time() - 100_000)
    t._fetch_daily("AAPL", 5)
    assert calls["n"] == 2


def test_fx_window_sizes_from_date_and_reports_actual_rate_date(monkeypatch):
    """The FX fetch window reaches back to a requested historical date, and the
    conversion names the actual series date the rate came from (not the exact
    requested day, which may be a weekend/holiday or predate the series)."""
    import financial_research_assistant.tools as t
    assert t._fx_series_days(None) == 7            # latest → short window
    assert t._fx_series_days("2020-01-01") > 700   # old date → window reaching it

    def fake(sym, days, strict=False, **_kw):
        return [("2025-06-12", 1.20), ("2025-06-13", 1.25)]

    monkeypatch.setattr(t, "_fetch_daily", fake)
    out = t.convert_currency(100, "EUR", on_date="2025-06-14")  # Sat → closest on/before
    assert "125.00 USD" in out and "1.2500" in out
    assert "2025-06-13" in out and "2025-06-14" in out


def test_income_summary_tool_adds_usd_total(monkeypatch, tmp_path):
    import financial_research_assistant.tools as t
    from financial_research_assistant import statements as s
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    mixed = (
        "Dividends,Header,Currency,Account,Date,Description,Amount\n"
        "Dividends,Data,USD,U1,2025-04-01,A(x) Cash Dividend,10\n"
        "Dividends,Data,EUR,U1,2025-04-02,B(y) Cash Dividend,20\n"
    )
    s.import_statement(mixed)
    _stub_fx(monkeypatch, {"EUR": 1.5})   # 20 EUR -> 30 USD; total 10 + 30 = 40
    out = t.income_summary(year=2025)
    assert "USD:" in out and "EUR:" in out
    assert "40.00 USD net total" in out

