"""Deep-research report persistence."""

import stat


def test_two_reports_in_the_same_second_both_survive(tmp_path, monkeypatch):
    """The filename stamp resolves to the second, and a parallel research run (or
    a retry) lands two reports inside one. Overwriting silently loses a finished
    analysis, so the second gets a counter."""
    from financial_research_assistant import research

    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    monkeypatch.setattr(
        research, "datetime", _FrozenClock("20260808-120000")
    )

    first = research.save_report("AAPL", "first report")
    second = research.save_report("AAPL", "second report")

    assert first != second
    assert first.read_text() == "first report"
    assert second.read_text() == "second report"


def test_a_saved_report_is_not_world_readable(tmp_path, monkeypatch):
    """A portfolio report names the holdings and their sizes, which puts it in the
    same class as every other store here: 0600."""
    from financial_research_assistant import research

    monkeypatch.setenv("FINANCIAL_RESEARCH_REPORTS_DIR", str(tmp_path))
    dest = research.save_report("portfolio", "# holdings\n")

    assert stat.S_IMODE(dest.stat().st_mode) == 0o600


class _FrozenClock:
    """A `datetime` stand-in whose `now()` always formats to the same stamp."""

    def __init__(self, stamp: str):
        self._stamp = stamp

    def now(self):
        return self

    def strftime(self, _fmt: str) -> str:
        return self._stamp
