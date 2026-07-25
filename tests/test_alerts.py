"""Alert-rule store, evaluation, and digest integration — fully offline."""

import datetime as dt

import pytest

from financial_research_assistant import alerts


@pytest.fixture(autouse=True)
def _tmp_alerts(monkeypatch, tmp_path):
    """Isolate the alert store to a temp file so tests never touch the real one."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_ALERTS_FILE", str(tmp_path / "alerts.json"))


def test_add_list_remove_and_validation():
    assert "Added alert a1" in alerts.add_alert("AAPL", "drop", 5)
    assert "Added alert a2" in alerts.add_alert("*", "move", 8)
    assert "Added alert a3" in alerts.add_alert("TSLA", "below", 200)
    # synonyms map to canonical kinds ("up" -> rise)
    assert "rises" in alerts.add_alert("MSFT", "up", 3)

    out = alerts.list_alerts()
    assert "AAPL drops ≥ 5%" in out
    assert "any holding moves ≥ 8%" in out
    assert "TSLA price at or below 200" in out
    assert "MSFT rises ≥ 3%" in out

    # validation
    assert "Unknown alert kind" in alerts.add_alert("X", "frobnicate", 1)
    assert "needs a specific symbol" in alerts.add_alert("*", "below", 100)
    assert "positive threshold" in alerts.add_alert("X", "drop", 0)
    # earnings defaults to 7 days when no value is given
    assert "within 7 day(s)" in alerts.add_alert("NVDA", "earnings", 0)

    # remove by id, unknown id, and clear-all
    assert "Removed alert a1" in alerts.remove_alert("a1")
    assert "No alert with id" in alerts.remove_alert("zzz")
    assert "Removed all" in alerts.remove_alert("all")
    assert "No alert rules set" in alerts.list_alerts()


def test_evaluate_alerts_fires_the_right_rules(monkeypatch):
    import financial_research_assistant.fundamentals as fund
    import financial_research_assistant.tools as tools

    def fake_daily(sym, days, **kw):
        return {
            "AAPL": [("2026-01-01", 100.0), ("2026-01-05", 90.0)],   # -10%
            "MSFT": [("2026-01-01", 100.0), ("2026-01-05", 106.0)],  # +6%
            "TSLA": [("2026-01-05", 180.0)],
        }.get(sym, [])

    monkeypatch.setattr(tools, "_fetch_daily", fake_daily)
    monkeypatch.setattr(fund, "_fetch_calendar",
                        lambda sym: {"Earnings Date": [dt.date(2026, 1, 8)]} if sym == "NVDA" else {})

    alerts.add_alert("AAPL", "drop", 5)       # a1 fires (down 10%)
    alerts.add_alert("MSFT", "rise", 5)       # a2 fires (up 6%)
    alerts.add_alert("TSLA", "below", 200)    # a3 fires (180 <= 200), even if not held
    alerts.add_alert("NVDA", "earnings", 7)   # a4 fires (earnings in 3d)
    alerts.add_alert("AAPL", "rise", 5)       # a5 does NOT fire (AAPL is down)

    fired = "\n".join(alerts.evaluate_alerts(["AAPL", "MSFT"], 5, dt.date(2026, 1, 5)))
    assert "[a1]" in fired and "AAPL down -10.0%" in fired
    assert "[a2]" in fired and "MSFT up +6.0%" in fired
    assert "[a3]" in fired and "TSLA at 180.00" in fired
    assert "[a4]" in fired and "NVDA reports earnings 2026-01-08" in fired
    assert "[a5]" not in fired


def test_evaluating_rules_buffers_them_for_notification(monkeypatch):
    """Firing a rule also records it for out-of-band delivery, so an interface
    can raise a notification the moment it triggers. The buffer drains once."""
    import financial_research_assistant.fundamentals as fund
    import financial_research_assistant.tools as tools

    monkeypatch.setattr(tools, "_fetch_daily",
                        lambda sym, days, **kw: [("2026-01-01", 100.0), ("2026-01-05", 88.0)])
    monkeypatch.setattr(fund, "_fetch_calendar", lambda sym: {})
    alerts.drain_triggered()

    alerts.add_alert("AAPL", "drop", 5)
    alerts.evaluate_alerts(["AAPL"], 5, dt.date(2026, 1, 5))

    fired = alerts.drain_triggered()
    assert len(fired) == 1
    assert "AAPL down -12.0%" in fired[0]
    assert alerts.drain_triggered() == []  # drained, so it won't replay next turn


def test_alert_sound_is_on_unless_explicitly_silenced(monkeypatch):
    """The bell defaults ON — an alert you only catch while looking at the
    terminal defeats the point — and only an explicit falsy value turns it off."""
    monkeypatch.delenv("FINANCIAL_RESEARCH_ALERT_SOUND", raising=False)
    assert alerts.sound_enabled() is True

    for off in ("0", "false", "no", "off", "OFF", " Off "):
        monkeypatch.setenv("FINANCIAL_RESEARCH_ALERT_SOUND", off)
        assert alerts.sound_enabled() is False, off

    for on in ("1", "true", "yes", "on", ""):
        monkeypatch.setenv("FINANCIAL_RESEARCH_ALERT_SOUND", on)
        assert alerts.sound_enabled() is True, on


def test_alert_sound_file_prefers_an_explicit_override(monkeypatch, tmp_path):
    """FINANCIAL_RESEARCH_ALERT_SOUND doubles as a file override, while its
    on/off words stay plain toggles rather than being read as paths."""
    mine = tmp_path / "chime.aiff"
    mine.write_bytes(b"not really audio")

    monkeypatch.setenv("FINANCIAL_RESEARCH_ALERT_SOUND", str(mine))
    assert alerts.alert_sound_file() == str(mine)

    # A toggle word is not a path, so it falls through to the system default.
    monkeypatch.setenv("FINANCIAL_RESEARCH_ALERT_SOUND", "on")
    assert alerts.alert_sound_file() != "on"

    # A path that doesn't exist also falls through rather than being played.
    monkeypatch.setenv("FINANCIAL_RESEARCH_ALERT_SOUND", str(tmp_path / "missing.aiff"))
    assert alerts.alert_sound_file() != str(tmp_path / "missing.aiff")


def test_play_alert_sound_launches_a_player(monkeypatch, tmp_path,
                                            _never_actually_play_audio):
    """The happy path hands a real player command to the spawner."""
    sound = tmp_path / "ping.aiff"
    sound.write_bytes(b"x")
    monkeypatch.setenv("FINANCIAL_RESEARCH_ALERT_SOUND", str(sound))
    monkeypatch.setattr(alerts, "_player_cmd", lambda s: ["afplay", s])

    assert alerts.play_alert_sound() is True
    assert _never_actually_play_audio == [["afplay", str(sound)]]


def test_play_alert_sound_is_silent_when_it_cannot_play(monkeypatch,
                                                        _never_actually_play_audio):
    """Disabled, no sound file, or no player installed (a headless server) each
    return False without raising — the toast and 🔔 line still carried it."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_ALERT_SOUND", "0")
    assert alerts.play_alert_sound() is False

    monkeypatch.delenv("FINANCIAL_RESEARCH_ALERT_SOUND", raising=False)
    monkeypatch.setattr(alerts, "alert_sound_file", lambda: None)
    assert alerts.play_alert_sound() is False

    monkeypatch.setattr(alerts, "alert_sound_file", lambda: "/tmp/x.aiff")
    monkeypatch.setattr(alerts, "_player_cmd", lambda s: None)
    assert alerts.play_alert_sound() is False

    assert _never_actually_play_audio == []


def test_desktop_notification_passes_text_through_argv(monkeypatch,
                                                       _never_actually_play_audio):
    """The alert text must never be interpolated into the AppleScript source —
    a line carrying quotes (or AppleScript syntax) would break it or run as
    code. It goes through argv, so it survives verbatim."""
    monkeypatch.setattr(alerts.sys, "platform", "darwin")
    monkeypatch.setattr(alerts.shutil, "which", lambda n: f"/usr/bin/{n}")
    monkeypatch.delenv("FINANCIAL_RESEARCH_ALERT_DESKTOP", raising=False)

    nasty = 'CPRT at 27.94 — at/below your "100" level'
    assert alerts.notify_desktop(nasty) is True

    cmd = _never_actually_play_audio[0]
    assert cmd[0] == "osascript"
    assert nasty in cmd                      # verbatim, its own argv entry
    assert nasty not in cmd[2]               # and NOT inside the script source


def test_desktop_notification_can_be_silenced_and_degrades(monkeypatch,
                                                           _never_actually_play_audio):
    """Off by env var, and a no-op where the OS offers no notifier — neither
    raises, since the toast and 🔔 line already delivered the alert."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_ALERT_DESKTOP", "0")
    assert alerts.notify_desktop("AAPL down 6%") is False

    monkeypatch.delenv("FINANCIAL_RESEARCH_ALERT_DESKTOP", raising=False)
    monkeypatch.setattr(alerts.sys, "platform", "linux")
    monkeypatch.setattr(alerts.shutil, "which", lambda n: None)  # no notify-send
    assert alerts.notify_desktop("AAPL down 6%") is False

    assert _never_actually_play_audio == []


def test_desktop_notification_uses_notify_send_on_linux(monkeypatch,
                                                        _never_actually_play_audio):
    monkeypatch.setattr(alerts.sys, "platform", "linux")
    monkeypatch.setattr(alerts.shutil, "which",
                        lambda n: "/usr/bin/notify-send" if n == "notify-send" else None)
    monkeypatch.delenv("FINANCIAL_RESEARCH_ALERT_DESKTOP", raising=False)

    assert alerts.notify_desktop("TSLA at 195.00") is True
    assert _never_actually_play_audio[0][0] == "notify-send"
    assert "TSLA at 195.00" in _never_actually_play_audio[0]


def test_notification_buffer_is_bounded(monkeypatch):
    """Nothing guarantees a drain (the eval harness ignores alert events), so the
    buffer caps rather than growing for the life of the process."""
    alerts.drain_triggered()
    for i in range(alerts._fired.maxlen + 25):
        alerts._fired.append(f"alert {i}")

    fired = alerts.drain_triggered()
    assert len(fired) == alerts._fired.maxlen
    assert fired[-1] == f"alert {alerts._fired.maxlen + 24}"  # newest kept


def test_evaluate_star_applies_to_each_holding(monkeypatch):
    import financial_research_assistant.fundamentals as fund
    import financial_research_assistant.tools as tools

    monkeypatch.setattr(tools, "_fetch_daily",
                        lambda sym, days, **kw: [("2026-01-01", 100.0), ("2026-01-05", 88.0)])  # -12%
    monkeypatch.setattr(fund, "_fetch_calendar", lambda sym: {})
    alerts.add_alert("*", "move", 8)  # any holding moving ≥ 8%
    fired = alerts.evaluate_alerts(["AAPL", "MSFT"], 5, dt.date(2026, 1, 5))
    assert len(fired) == 2 and any("AAPL" in f for f in fired) and any("MSFT" in f for f in fired)


def test_digest_surfaces_triggered_alerts(monkeypatch, tmp_path):
    import financial_research_assistant.fundamentals as fund
    import financial_research_assistant.tools as tools
    from financial_research_assistant import monitor
    from financial_research_assistant import statements as s

    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    s.import_statement(
        "Open Positions,Header,DataDiscriminator,Asset Category,Currency,Symbol,"
        "Quantity,Mult,Cost Price,Cost Basis,Close Price,Value,Unrealized P/L,Code\n"
        "Open Positions,Data,Summary,Stocks,USD,AAPL,1,1,100,100,100,100,0,\n"
    )
    monkeypatch.setattr(tools, "_fetch_daily",
                        lambda sym, days, **kw: [("2026-01-01", 100.0), ("2026-01-05", 90.0)])
    monkeypatch.setattr(fund, "_fetch_calendar", lambda sym: {})

    # No rules yet → no alerts section at all.
    out0 = monitor.build_digest(today=dt.date(2026, 1, 5))
    assert "🔔" not in out0

    alerts.add_alert("AAPL", "drop", 5)
    out = monitor.build_digest(today=dt.date(2026, 1, 5))
    assert "## 🔔 Alerts triggered" in out
    assert "AAPL down -10.0%" in out


def test_alert_tools_registered():
    from financial_research_assistant.tools import TOOLS

    names = {getattr(t, "name", getattr(t, "__name__", "")) for t in TOOLS}
    assert {"add_alert", "list_alerts", "remove_alert"} <= names
