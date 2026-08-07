"""FRED macro data — alias resolution, CSV parsing, and the year-over-year forms.

Offline: the HTTP fetch is stubbed. The parsing cases are taken from what the real
endpoint actually returns (blank values on holidays, an HTML page for an unknown
id), verified against it while building this.
"""

from datetime import date

from financial_research_assistant import macro


def _stub(monkeypatch, payload):
    """Stub the CSV fetch. ``payload`` is the raw body FRED would return."""
    macro._CACHE.clear()

    class _Resp:
        def __init__(self, body): self._body = body.encode()
        def read(self): return self._body
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(macro.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp(payload))


def _csv(series_id, rows):
    head = f"observation_date,{series_id}\n"
    return head + "".join(f"{d},{v}\n" for d, v in rows)


# --- aliases --------------------------------------------------------------------


def test_plain_names_resolve_to_series_ids():
    """Nobody asks for DGS10; they ask about the 10-year."""
    assert macro.resolve("10y")[0] == "DGS10"
    assert macro.resolve("fed funds")[0] == "DFF"
    assert macro.resolve("core PCE")[0] == "PCEPILFE"
    assert macro.resolve("yield_curve")[0] == "T10Y2Y"
    assert macro.resolve("UNEMPLOYMENT")[0] == "UNRATE"


def test_a_raw_fred_id_passes_through():
    """The alias table is a convenience, not a whitelist — the whole library stays
    reachable."""
    assert macro.resolve("DGS10") == ("DGS10", "DGS10")
    assert macro.resolve("BAMLH0A0HYM2")[0] == "BAMLH0A0HYM2"


def test_an_unresolvable_name_returns_none():
    assert macro.resolve("some nonsense phrase") is None
    assert macro.resolve("") is None


def test_an_unknown_name_lists_what_is_available(monkeypatch):
    out = macro.macro_series("the vibes index")
    assert "Don't know a macro series" in out
    assert "unemployment" in out, "the refusal must name real options"


# --- parsing --------------------------------------------------------------------


def test_blank_values_are_dropped_not_read_as_zero(monkeypatch):
    """Daily series carry blank rows for market holidays. A blank is 'no
    observation'; reading it as 0.0 would put a fake crash in every chart."""
    _stub(monkeypatch, _csv("DGS10", [
        ("2026-07-01", "4.48"), ("2026-07-03", ""), ("2026-07-06", "4.48"),
    ]))
    rows = macro._fetch("DGS10", date(2026, 7, 1), None)
    assert rows == [("2026-07-01", 4.48), ("2026-07-06", 4.48)]


def test_a_dot_placeholder_is_dropped_too(monkeypatch):
    _stub(monkeypatch, _csv("X", [("2026-07-01", "."), ("2026-07-02", "1.5")]))
    assert macro._fetch("X", date(2026, 7, 1), None) == [("2026-07-02", 1.5)]


def test_an_unknown_series_returns_nothing_rather_than_garbage(monkeypatch):
    """FRED serves an HTML error page for an unknown id, not a 404 — so the CSV
    header is what's checked. Parsing HTML as CSV would yield junk rows."""
    _stub(monkeypatch, "<!DOCTYPE html><html><head><title>Error</title></head></html>")
    assert macro._fetch("NOTASERIES", date(2026, 1, 1), None) == []


def test_an_outage_is_distinct_from_an_empty_series(monkeypatch):
    """'the source is down' and 'that series has no data' need different answers."""
    macro._CACHE.clear()

    def boom(req, timeout=0):
        raise OSError("connection refused")

    monkeypatch.setattr(macro.urllib.request, "urlopen", boom)
    try:
        macro._fetch("DGS10", date(2026, 1, 1), None)
    except macro.MacroDataUnavailable as exc:
        assert "Couldn't reach FRED" in str(exc)
    else:
        raise AssertionError("an outage must raise, not return []")


def test_the_fetch_is_cached_within_a_run(monkeypatch):
    """A snapshot pulls a dozen series; FRED is a courtesy endpoint."""
    calls = []
    macro._CACHE.clear()

    class _Resp:
        def read(self): return _csv("X", [("2026-01-01", "1")]).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def counting(req, timeout=0):
        calls.append(req.full_url)
        return _Resp()

    monkeypatch.setattr(macro.urllib.request, "urlopen", counting)
    macro._fetch("X", date(2026, 1, 1), None)
    macro._fetch("X", date(2026, 1, 1), None)
    assert len(calls) == 1


# --- year over year -------------------------------------------------------------


def test_an_index_series_is_quoted_year_over_year(monkeypatch):
    """A CPI level of 332.6 says nothing; '+3.5% y/y' is what people mean by
    'CPI'."""
    _stub(monkeypatch, _csv("CPIAUCSL", [
        ("2025-06-01", "300.0"), ("2025-12-01", "305.0"), ("2026-06-01", "310.5"),
    ]))
    out = macro.macro_series("cpi", days=800)
    assert "year over year +3.5%" in out


def test_the_year_ago_lookup_is_by_anniversary_not_row_offset(monkeypatch):
    """These series run at wildly different frequencies — DGS10 daily, UNRATE
    monthly, GDPC1 quarterly — so a fixed row offset would mean a different span
    for each."""
    rows = [("2025-06-01", 100.0), ("2025-09-01", 150.0), ("2026-06-01", 110.0)]
    assert macro._year_ago(rows) == 100.0
    assert macro._yoy(rows) == 10.0


def test_a_window_too_short_for_a_year_reports_no_yoy():
    rows = [("2026-05-01", 100.0), ("2026-06-01", 110.0)]
    assert macro._year_ago(rows) is None
    assert macro._yoy(rows) is None


def test_a_leap_day_observation_has_an_anniversary():
    rows = [("2024-02-28", 100.0), ("2025-02-28", 110.0)]
    assert macro._year_ago(rows) == 100.0


# --- the tools ------------------------------------------------------------------


def test_a_series_reports_its_window_and_latest(monkeypatch):
    _stub(monkeypatch, _csv("DGS10", [
        ("2026-01-02", "4.20"), ("2026-06-01", "4.63"),
    ]))
    out = macro.macro_series("10y", days=365)
    assert "10-year Treasury yield" in out and "DGS10" in out
    assert "latest 4.63" in out and "change +0.43" in out


def test_a_dated_series_says_the_vintage_caveat(monkeypatch):
    """as_of bounds the OBSERVATION date, not the vintage — FRED revises, so a past
    period comes back as currently restated. Leaving that unsaid would be the same
    class of quiet wrongness the point-in-time work exists to prevent."""
    _stub(monkeypatch, _csv("DGS10", [("2025-01-02", "4.20"), ("2025-06-30", "4.63")]))
    out = macro.macro_series("10y", days=365, as_of="2025-06-30")
    assert "AS OF 2025-06-30" in out
    assert "not the data vintage" in out
    assert "not the data vintage" not in macro.macro_series("10y", days=365)


def test_a_bad_as_of_is_reported_by_the_tool(monkeypatch):
    _stub(monkeypatch, _csv("DGS10", [("2026-01-02", "4.20")]))
    assert "YYYY-MM-DD" in macro.macro_series("10y", as_of="last summer")


def test_an_empty_series_says_so_rather_than_charting_nothing(monkeypatch):
    _stub(monkeypatch, _csv("DGS10", []))
    assert "No observations" in macro.macro_series("10y")


def test_the_snapshot_covers_the_panel(monkeypatch):
    """Every line the snapshot promises must appear, or the 'backdrop' it claims to
    give has a hole the reader can't see."""
    def any_series(req, timeout=0):
        sid = req.full_url.split("id=")[1].split("&")[0]
        body = _csv(sid, [("2025-06-01", "100.0"), ("2026-06-01", "110.0")])

        class _Resp:
            def read(self): return body.encode()
            def __enter__(self): return self
            def __exit__(self, *a): return False

        return _Resp()

    macro._CACHE.clear()
    monkeypatch.setattr(macro.urllib.request, "urlopen", any_series)
    out = macro.macro_snapshot()
    for alias in macro._SNAPSHOT:
        _sid, label = macro.ALIASES[alias]
        assert label in out, f"{alias} missing from the snapshot"
    assert "+10.0% y/y" in out, "index series are quoted as a rate of change"


def test_the_snapshot_survives_a_total_outage(monkeypatch):
    macro._CACHE.clear()

    def boom(req, timeout=0):
        raise OSError("down")

    monkeypatch.setattr(macro.urllib.request, "urlopen", boom)
    out = macro.macro_snapshot()
    assert "Couldn't reach FRED" in out


def test_the_tools_are_registered():
    from financial_research_assistant import catalog

    names = {catalog.tool_name(t) for t in catalog.TOOLS}
    assert {"macro_series", "macro_snapshot"} <= names
