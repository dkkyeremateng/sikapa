"""The autonomy switch and quiet hours.

Three ways to tell an always-on agent to stand down, cheapest first:

- **Quiet** (``/quiet 2h``): keep working, but hold back pushes that can wait.
  Quiet hours (``FRA_QUIET_HOURS``, e.g. ``22:00-07:00``) do the same every night.
- **Pause** (``/pause``, ``--pause``, or ``FRA_AUTONOMY=off``): no autonomous work
  at all — no reports, no event analyses, no ideas. Chat still answers: a pause is
  "stop acting on your own", not "stop listening".
- **Budget** (``autonomy.over_budget``): once the day's or month's autonomous
  tokens are spent, the model is not called again until the next period; reports
  go out model-free and events as plain facts.

The switch lives in ``autonomy.json`` under the state directory, so it survives
a restart — a pause that a deploy silently undid would not be much of a pause.
"""

from __future__ import annotations

from typing import Any
import json
import os
import re
from datetime import datetime, time, timedelta

from . import hooks
from .storage import locked, read_json, state_file, write_private

_OFF = {"0", "off", "false", "no", "paused"}


def switch_file():
    return state_file("autonomy.json", "FRA_AUTONOMY_FILE")


def _read() -> dict[str, Any]:
    data = read_json(switch_file(), {})
    return data if isinstance(data, dict) else {}


def _write(update: dict[str, Any]) -> None:
    path = switch_file()
    with locked(path):
        data = _read()
        data.update(update)
        write_private(path, json.dumps(data, indent=2), prefix=".autonomy-")


def pause(reason: str = "") -> None:
    _write({"paused": True, "since": datetime.now().isoformat(timespec="seconds"),
            "reason": reason})


def resume() -> None:
    _write({"paused": False, "since": None, "reason": ""})


def paused_reason() -> str:
    """Why autonomous work is off right now, or "" when it is on."""
    if (os.environ.get("FRA_AUTONOMY") or "").strip().lower() in _OFF:
        return "autonomy is switched off (FRA_AUTONOMY)"
    data = _read()
    if data.get("paused"):
        since = str(data.get("since") or "")[:16].replace("T", " ")
        return "autonomy is paused" + (f" since {since}" if since else "") + " — /resume to restart"
    return ""


# --- quiet -----------------------------------------------------------------------


def set_quiet(minutes: float) -> datetime:
    until = datetime.now() + timedelta(minutes=minutes)
    _write({"quiet_until": until.isoformat(timespec="seconds")})
    return until


def clear_quiet() -> None:
    _write({"quiet_until": None})


def _parse_hours(raw: str) -> tuple[time, time] | None:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", raw or "")
    if not m:
        return None
    a, b, c, d = (int(x) for x in m.groups())
    if a > 23 or c > 23 or b > 59 or d > 59:
        return None
    return time(a, b), time(c, d)


def quiet_now(now: datetime | None = None) -> str:
    """Why a push that can wait should wait right now, or "" when it shouldn't.

    ``now`` is local time. Quiet hours may wrap midnight (22:00-07:00)."""
    now = now or datetime.now()
    until = str(_read().get("quiet_until") or "")
    if until:
        try:
            if now < datetime.fromisoformat(until):
                return f"quiet until {until[11:16]}"
        except ValueError:
            pass
    hours = _parse_hours(os.environ.get("FRA_QUIET_HOURS") or "")
    if hours:
        start, end = hours
        t = now.time()
        inside = (start <= t < end) if start <= end else (t >= start or t < end)
        if inside:
            return f"quiet hours ({start:%H:%M}-{end:%H:%M})"
    return ""


def _minutes(text: str) -> float | None:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(m|min|mins|minutes?|h|hr|hrs|hours?)?\s*", text or "")
    if not m:
        return None
    n = float(m.group(1))
    return n * 60 if (m.group(2) or "h").startswith("h") else n


# --- commands -----------------------------------------------------------------------


async def _cmd_pause(arg: str, _fake: bool) -> str:
    pause(arg)
    return ("Paused. No reports, event analyses or ideas until /resume. "
            "I still answer messages.")


async def _cmd_resume(_arg: str, _fake: bool) -> str:
    resume()
    reason = paused_reason()
    if reason:  # the env switch outranks the file
        return f"Resumed here, but {reason} — change it in the server env."
    return "Resumed. Scheduled reports and watchers are back on."


async def _cmd_quiet(arg: str, _fake: bool) -> str:
    text = (arg or "").strip().lower()
    if text in ("off", "0", "stop", "end"):
        clear_quiet()
        return "Quiet mode off."
    minutes = _minutes(text or "1h")
    if minutes is None or minutes <= 0:
        return "Usage: /quiet 2h (or 30m, or /quiet off)"
    until = set_quiet(minutes)
    return (f"Quiet until {until:%H:%M}: only urgent alerts will come through; "
            "everything else waits and is summarised afterwards.")


hooks.register_command("pause", _cmd_pause, "stop autonomous work (chat still answers)")
hooks.register_command("resume", _cmd_resume, "restart autonomous work")
hooks.register_command("quiet", _cmd_quiet, "hold non-urgent pushes: /quiet 2h, /quiet off")
