"""The US market's clock — when it is open, and whether a new session has closed.

Two questions the always-on service keeps asking. The event watchers poll more
often while the market is open. The daily report is written after the close,
and must not be written at all on a day with no session (a holiday), because
a "daily report" on Thanksgiving is yesterday's report with today's date on it.

**Holidays come from the data, not a calendar.** A hardcoded holiday list is
wrong the first time the exchange adds a day of mourning. The last session SPY
traded is the answer to "was there a session today?" — ``latest_session`` asks
the same keyless daily-close source everything else here uses. ``regular_hours``
only knows the weekday and the clock, so on a holiday it says "open" and the
watchers poll a quiet market a little more often than necessary; that is the
cheap side to be wrong on.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
OPEN = time(9, 30)
CLOSE = time(16, 0)

#: The symbol whose last daily close stands in for "the last session".
SESSION_SYMBOL = "SPY"


def ny_now(now: datetime | None = None) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(NEW_YORK)


def regular_hours(now: datetime | None = None) -> bool:
    """Weekday, between 09:30 and 16:00 New York time."""
    t = ny_now(now)
    return t.weekday() < 5 and OPEN <= t.time() < CLOSE


def latest_session(symbol: str = SESSION_SYMBOL) -> date | None:
    """The date of the most recent completed daily bar, or None when the source
    has nothing (down, or an unknown symbol)."""
    from .tools import _fetch_daily  # pyright: ignore[reportPrivateUsage]

    series = _fetch_daily(symbol, 10)
    if not series:
        return None
    try:
        return date.fromisoformat(series[-1][0])
    except ValueError:
        return None


def session_closed_today(now: datetime | None = None) -> bool | None:
    """Whether today (New York) had a session that has now closed. None when the
    price source can't say — the caller decides whether to go ahead blind."""
    today = ny_now(now).date()
    last = latest_session()
    if last is None:
        return None
    return last >= today and ny_now(now).time() >= CLOSE
