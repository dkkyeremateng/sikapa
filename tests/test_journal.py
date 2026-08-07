"""The thesis journal: recording a call, scoring it, and reading it back.

Fully offline — the price fetch is stubbed, and the autouse conftest fixture
points the store at a throwaway file. Nothing here reaches a model: scoring is two
price lookups and a subtraction, which is the point of it.
"""

from datetime import date, timedelta

from financial_research_assistant import journal


def _prices(monkeypatch, table):
    """Stub the daily-close fetch. ``table`` maps symbol -> [(date, close)]."""
    import financial_research_assistant.tools as t
    from financial_research_assistant.pointintime import as_of_series

    def fake(sym, days, strict=False, as_of=None, **_kw):
        return as_of_series(table.get(sym.upper(), []), as_of)

    monkeypatch.setattr(t, "_fetch_daily", fake)


_TODAY = date.today()
_OPENED = (_TODAY - timedelta(days=200)).isoformat()


def _series(start_price, end_price):
    """A two-point series: 200 days ago, and today."""
    return [(_OPENED, start_price), (_TODAY.isoformat(), end_price)]


def _record_then_move(monkeypatch, entry_price, exit_price, bench=(100.0, 100.0),
                      horizon=10, verdict="bullish", symbol="NVDA"):
    """Record a call at ``entry_price``, then let the market move to ``exit_price``.

    The two-step stub is what makes the arithmetic real: recording sees only the
    opening close, so ``entry_price`` is genuinely captured rather than assumed,
    and scoring then sees a later close. ``due`` is computed from the real clock,
    so the returned date is one day past the horizon — score at it.
    """
    _prices(monkeypatch, {symbol: [(_OPENED, entry_price)], "SPY": [(_OPENED, bench[0])]})
    journal.record_thesis(symbol, verdict, "Because reasons.", horizon_days=horizon)
    _prices(monkeypatch, {
        symbol: [(_OPENED, entry_price), (_TODAY.isoformat(), exit_price)],
        "SPY": [(_OPENED, bench[0]), (_TODAY.isoformat(), bench[1])],
    })
    return _TODAY + timedelta(days=horizon + 1)


# --- recording ------------------------------------------------------------------


def test_a_call_captures_the_price_itself(monkeypatch):
    """The entry price is fetched, never taken from the model. A call scored
    against a price the model typed is scored against its memory of a price."""
    _prices(monkeypatch, {"NVDA": _series(100.0, 120.0)})
    out = journal.record_thesis("NVDA", "bullish", "Data-centre demand still accelerating.")
    assert "[t1]" in out and "BULLISH NVDA" in out
    entry = journal.load_entries()[0]
    assert entry["entry_price"] == 120.0  # the latest close, not anything passed in
    assert entry["status"] == "open"
    assert entry["horizon_days"] == journal.DEFAULT_HORIZON_DAYS


def test_recording_says_it_is_not_a_recommendation(monkeypatch):
    _prices(monkeypatch, {"NVDA": _series(100.0, 120.0)})
    out = journal.record_thesis("NVDA", "bullish", "Because reasons.")
    assert "not" in out.lower() and "recommendation" in out.lower()


def test_a_call_with_no_price_is_refused(monkeypatch):
    """An entry with no price could never be scored, so recording it would only
    accumulate junk that looks like a track record."""
    _prices(monkeypatch, {})
    out = journal.record_thesis("NOPE", "bullish", "A thesis about nothing.")
    assert "could never be scored" in out
    assert journal.load_entries() == []


def test_a_call_needs_its_reasoning(monkeypatch):
    _prices(monkeypatch, {"NVDA": _series(100.0, 120.0)})
    out = journal.record_thesis("NVDA", "bullish", "  ")
    assert "reasoning" in out
    assert journal.load_entries() == []


def test_an_unknown_verdict_is_refused(monkeypatch):
    _prices(monkeypatch, {"NVDA": _series(100.0, 120.0)})
    out = journal.record_thesis("NVDA", "buy", "Strong quarter.")
    assert "bullish" in out and "bearish" in out
    assert journal.load_entries() == []


def test_the_horizon_is_bounded(monkeypatch):
    _prices(monkeypatch, {"NVDA": _series(100.0, 120.0)})
    journal.record_thesis("NVDA", "bullish", "x", horizon_days=100000)
    assert journal.load_entries()[0]["horizon_days"] == journal._MAX_HORIZON_DAYS


# --- scoring --------------------------------------------------------------------


def test_direction_decides_the_hit():
    assert journal._hit("bullish", 12.0) and not journal._hit("bullish", -3.0)
    assert journal._hit("bearish", -12.0) and not journal._hit("bearish", 3.0)
    assert journal._hit("neutral", 1.0) and not journal._hit("neutral", 30.0)


def test_the_return_and_alpha_are_computed_over_the_window(monkeypatch):
    """'Bullish, +30%' means something different when the index did +10%, so both
    figures and the difference between them are always recorded."""
    due = _record_then_move(monkeypatch, 100.0, 130.0, bench=(100.0, 110.0))
    scored = journal.score_due(now=due)
    assert len(scored) == 1
    assert scored[0]["change_pct"] == 30.0
    assert scored[0]["benchmark_change_pct"] == 10.0
    assert scored[0]["alpha_pct"] == 20.0
    assert scored[0]["hit"] is True
    assert scored[0]["status"] == "scored"


def test_a_call_can_be_right_and_still_lag_the_index(monkeypatch):
    """The case the benchmark exists for: direction was right, and it was worth
    nothing. The verdict is a hit; the alpha says the rest."""
    due = _record_then_move(monkeypatch, 100.0, 108.0, bench=(100.0, 114.0))
    scored = journal.score_due(now=due)
    assert scored[0]["hit"] is True
    assert scored[0]["alpha_pct"] == -6.0


def test_a_wrong_call_is_recorded_as_wrong(monkeypatch):
    due = _record_then_move(monkeypatch, 100.0, 70.0, bench=(100.0, 105.0))
    scored = journal.score_due(now=due)
    assert scored[0]["hit"] is False
    assert scored[0]["change_pct"] == -30.0
    told = journal.describe_outcome(scored[0])
    assert "was WRONG" in told
    assert "Because reasons." in told, "the reasoning must come back with the score"


def test_a_bearish_call_is_right_when_the_price_falls(monkeypatch):
    due = _record_then_move(monkeypatch, 100.0, 70.0, verdict="bearish")
    assert journal.score_due(now=due)[0]["hit"] is True


def test_a_call_not_yet_due_is_left_alone(monkeypatch):
    _prices(monkeypatch, {"NVDA": _series(100.0, 120.0), "SPY": _series(100.0, 105.0)})
    journal.record_thesis("NVDA", "bullish", "Demand.", horizon_days=90)
    assert journal.score_due(now=_TODAY + timedelta(days=10)) == []
    assert journal.load_entries()[0]["status"] == "open"


def test_a_price_outage_leaves_the_call_open_rather_than_scoring_it_a_miss(monkeypatch):
    """A data-source failure is not evidence about the call. Burning it as a miss
    would quietly corrupt the record every time Yahoo hiccups."""
    due = _record_then_move(monkeypatch, 100.0, 130.0)
    _prices(monkeypatch, {})  # source down
    assert journal.score_due(now=due) == []
    assert journal.load_entries()[0]["status"] == "open"


def test_scoring_is_idempotent(monkeypatch):
    """A second tick must not re-score and double-count a closed call."""
    due = _record_then_move(monkeypatch, 100.0, 130.0)
    assert len(journal.score_due(now=due)) == 1
    assert journal.score_due(now=due) == []
    assert len(journal.load_entries()) == 1


# --- reading it back ------------------------------------------------------------


def test_an_empty_journal_says_how_to_start():
    assert "No recorded calls" in journal.review_theses()


def test_the_review_separates_open_from_scored(monkeypatch):
    due = _record_then_move(monkeypatch, 100.0, 130.0)
    journal.score_due(now=due)
    _prices(monkeypatch, {"AAPL": _series(100.0, 120.0), "SPY": _series(100.0, 105.0)})
    journal.record_thesis("AAPL", "bearish", "Still open.", horizon_days=90)

    out = journal.review_theses()
    assert "Open calls (1)" in out and "Scored calls (1)" in out
    assert "Track record:" in out


def test_a_thin_record_says_it_is_thin(monkeypatch):
    """One scored call is not a hit rate. Reporting 100% without that caveat is
    the exact overclaim this feature could otherwise manufacture."""
    due = _record_then_move(monkeypatch, 100.0, 130.0)
    journal.score_due(now=due)
    out = journal.review_theses()
    assert "far too few to mean anything" in out


def test_the_review_always_disclaims_predictiveness(monkeypatch):
    _prices(monkeypatch, {"NVDA": _series(100.0, 120.0), "SPY": _series(100.0, 105.0)})
    journal.record_thesis("NVDA", "bullish", "Demand.", horizon_days=7)
    out = journal.review_theses()
    assert "not evidence about the next one" in out


def test_the_review_narrows_to_one_ticker(monkeypatch):
    _prices(monkeypatch, {
        "NVDA": _series(100.0, 120.0), "AAPL": _series(100.0, 120.0),
        "SPY": _series(100.0, 105.0),
    })
    journal.record_thesis("NVDA", "bullish", "One.")
    journal.record_thesis("AAPL", "bearish", "Two.")
    out = journal.review_theses("NVDA")
    assert "NVDA" in out and "AAPL" not in out


def test_a_scored_call_becomes_a_lesson(monkeypatch, tmp_path):
    """The outcome goes into the same store reflection uses, so the existing recall
    path surfaces it next time the ticker comes up — which is the whole point of
    scoring rather than merely recording."""
    monkeypatch.setenv("MEMORY_BACKEND", "local")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_USER", "journal-test")
    due = _record_then_move(monkeypatch, 100.0, 70.0)
    journal.score_due(now=due)

    from financial_research_assistant.memory import get_memory

    mem = get_memory()
    assert mem is not None
    lessons = [e for e in mem.all() if e.get("kind") == "lesson"]
    assert any("NVDA" in e["text"] and "WRONG" in e["text"] for e in lessons)


def test_scoring_works_with_memory_off(monkeypatch):
    """Memory is opt-in; the journal must not depend on it."""
    monkeypatch.delenv("MEMORY_BACKEND", raising=False)
    due = _record_then_move(monkeypatch, 100.0, 130.0)
    assert len(journal.score_due(now=due)) == 1


# --- wiring ---------------------------------------------------------------------


def test_review_is_gated_on_having_recorded_something(monkeypatch):
    """`record_thesis` is the bootstrap tool and stays bound; `review_theses` reads
    a store that is empty until the first call, so it is bound only once there is
    something to review."""
    from financial_research_assistant import catalog

    names = {catalog.tool_name(t) for t in catalog.active_tools()}
    assert "record_thesis" in names
    assert "review_theses" not in names

    _prices(monkeypatch, {"NVDA": _series(100.0, 120.0)})
    journal.record_thesis("NVDA", "bullish", "Demand.")
    names = {catalog.tool_name(t) for t in catalog.active_tools()}
    assert "review_theses" in names


async def test_a_scheduler_tick_scores_due_calls(monkeypatch):
    """Scoring rides the tick that already exists, and costs no model call."""
    from financial_research_assistant import scheduler

    _record_then_move(monkeypatch, 100.0, 130.0)
    # The horizon is clamped to a minimum, so age the entry rather than recording
    # a negative one — `run_due` scores against the real clock, not a passed date.
    items = journal.load_entries()
    items[0]["due"] = (_TODAY - timedelta(days=1)).isoformat()
    journal.save_entries(items)

    await scheduler.run_due(fake=True)
    scored = journal.load_entries()[0]
    assert scored["status"] == "scored"
    assert scored["change_pct"] == 30.0


async def test_a_scoring_failure_does_not_sink_the_tick(monkeypatch):
    from financial_research_assistant import scheduler

    def boom():
        raise RuntimeError("price source down")

    monkeypatch.setattr(journal, "score_due", boom)
    assert await scheduler.run_due(fake=True) == []  # returned normally, no raise
