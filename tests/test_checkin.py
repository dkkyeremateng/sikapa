"""The weekly check-in: measured against the profile and the journal, raised once.

Offline. The book, sectors and prices are faked; the profile, journal and the
check-in's own state are the real stores in the conftest's throwaway home.
"""

import asyncio
from datetime import date, datetime, timedelta, timezone

import pytest

from financial_research_assistant import (
    checkin, fundamentals, hooks, jobs, journal, profile, statements,
)

NOW = datetime(2026, 10, 26, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def world(monkeypatch):
    """AAPL 60%, UNH 15%, BND 25%; UNH up 20% since 1 Sep against SPY +2%."""
    positions = [
        {"symbol": "AAPL", "asset_category": "STK", "currency": "USD", "close_price": 100.0,
         "value": 6000.0},
        {"symbol": "UNH", "asset_category": "STK", "currency": "USD", "close_price": 120.0,
         "value": 1500.0},
        {"symbol": "BND", "asset_category": "STK", "currency": "USD", "close_price": 70.0,
         "value": 2500.0},
    ]
    monkeypatch.setattr(statements, "query_positions", lambda account=None: list(positions))
    monkeypatch.setattr(statements, "default_account", lambda: "U1")
    monkeypatch.setattr(statements, "list_imports", lambda: [
        {"account": "U1", "period": "September 26, 2026 - October 23, 2026"}])
    sectors = {"AAPL": "Technology", "UNH": "Healthcare"}
    monkeypatch.setattr(fundamentals, "_fetch_info",
                        lambda sym: {"sector": sectors[sym]} if sym in sectors else {})
    spot = {"UNH": 120.0, "SPY": 102.0, "AAPL": 100.0}
    monkeypatch.setattr(journal, "_spot", lambda sym: (spot.get(sym), "2026-10-23"))
    monkeypatch.setattr(journal, "_price_on",
                        lambda sym, when: (100.0, when.isoformat()) if sym == "SPY" else (None, ""))
    profile.save({"targets": {"us_equity": 60, "intl_equity": 20, "bonds": 20},
                  "max_position_pct": 12, "max_sector_pct": 35})
    return {"positions": positions, "spot": spot}


def _call(id_, sym, verdict, price, on):
    return {"id": id_, "symbol": sym, "verdict": verdict, "status": "open",
            "entry_price": price, "entry_date": on, "benchmark": "SPY"}


def test_what_breaks_the_profile_is_found(world):
    keys = {f.key: f.line for f in checkin.findings()}
    assert keys["cap:AAPL"] == "AAPL is 60.0% of the book, over your 12% position cap."
    assert "cap:UNH" in keys and "cap:BND" not in keys, "a bond fund is not a single stock"
    assert keys["sector:Technology"].startswith("Technology is 60.0% of the book, over your 35%")
    assert "sector:Healthcare" not in keys
    assert keys["mix:us_equity"].endswith("75% of the book against a 60% target (over by 15 points).")
    assert "under by 20 points" in keys["mix:intl_equity"]
    assert "mix:bonds" in keys, "25% against 20% is a 5-point drift"


def test_the_caps_and_drift_are_the_profiles(world):
    profile.save({"max_position_pct": 70, "max_sector_pct": 70,
                  "targets": {"us_equity": 75, "bonds": 25}})
    assert checkin.findings() == []


def test_a_call_losing_to_its_benchmark_is_raised_once_per_symbol(world):
    journal.save_entries([
        _call("t1", "UNH", "bearish", 100.0, "2026-09-01"),
        _call("t2", "UNH", "bearish", 110.0, "2026-09-15"),
        _call("t3", "AAPL", "bullish", 99.0, "2026-09-01"),  # +1% vs SPY +2%: close enough
    ])
    calls = [f for f in checkin.findings() if f.key.startswith("call:")]
    assert [f.key for f in calls] == ["call:t1"], "the earliest call, not each repeat"
    assert calls[0].line == ("Your bearish call on UNH (t1, 01 Sep at 100.00) is 18 points "
                             "behind: UNH +20.0% vs SPY +2.0% to 2026-10-23. "
                             "Made 2 times; this is the first.")


def test_how_far_behind_counts(world, monkeypatch):
    journal.save_entries([_call("t1", "UNH", "bearish", 100.0, "2026-09-01")])
    world["spot"]["UNH"] = 111.0   # +11% vs +2%: 9 points, under the bar
    assert not [f for f in checkin.findings() if f.key.startswith("call:")]
    monkeypatch.setattr(checkin, "BEHIND_PTS", 8.0)
    assert [f.key for f in checkin.findings() if f.key.startswith("call:")] == ["call:t1"]


def test_a_finding_is_raised_once_then_only_when_worse_or_overdue(world):
    first = checkin.to_raise(checkin.findings(), NOW)
    assert "cap:AAPL" in {f.key for f in first}
    assert checkin.to_raise(checkin.findings(), NOW + timedelta(days=7)) == [], "seen already"
    world["positions"][0]["value"] = 7000.0   # AAPL from 60% to ~64%
    again = checkin.to_raise(checkin.findings(), NOW + timedelta(days=14))
    keys = {f.key for f in again}
    assert "cap:AAPL" in keys, "60% → 64%: worse by more than its 1-point step"
    assert "mix:us_equity" not in keys, "75% → 77%: under its 3-point step"
    later = checkin.to_raise(checkin.findings(), NOW + timedelta(days=14 + checkin.REMIND_DAYS))
    assert "cap:AAPL" in {f.key for f in later}, "a standing breach is a monthly reminder"


def test_a_cleared_finding_is_news_if_it_comes_back(world):
    checkin.to_raise(checkin.findings(), NOW)
    profile.save({"max_position_pct": 70})
    checkin.to_raise(checkin.findings(), NOW + timedelta(days=7))
    profile.save({"max_position_pct": 12})
    back = checkin.to_raise(checkin.findings(), NOW + timedelta(days=8))
    assert "cap:AAPL" in {f.key for f in back}


def test_the_job_is_silent_when_there_is_nothing_new(world):
    handler = jobs.handler_for("checkin")
    res = asyncio.run(handler({"id": "s1"}, False))
    assert res.ok and res.notify and res.text.startswith("🧭 Worth a look this week:\n• ")
    quiet = asyncio.run(handler({"id": "s1"}, False))
    assert quiet.ok and not quiet.notify


def test_without_a_profile_only_the_calls_are_checked(world):
    profile.profile_file().unlink()
    journal.save_entries([_call("t1", "UNH", "bearish", 100.0, "2026-09-01")])
    assert [f.key for f in checkin.findings()] == ["call:t1"]


def test_it_is_scheduled_and_has_a_command(world):
    assert any(j.job == "checkin" and j.repeat == "weekly" for j in jobs.DEFAULT_JOBS)
    hooks.load_feature_modules()
    handler, _help = hooks.COMMANDS["checkin"]
    out = asyncio.run(handler("", False))
    assert out.startswith("🧭 Worth a look now:") and "cap:" not in out
    assert checkin.to_raise(checkin.findings(), NOW), "the command does not use up a finding"
