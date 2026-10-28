"""The weekly check-in — what the agent raises on its own, unasked.

Reports describe the book on a timetable and the watchers react to the market.
Neither asks whether the book still matches what the investor said they wanted,
or whether the calls in the journal are going wrong. The check-in does, once a
week, and sends a message only when something needs a decision:

- a single stock over the profile's position cap, or a sector over its cap;
- an asset class drifted ``DRIFT_PTS`` or more from its target;
- an open call losing to its benchmark by ``BEHIND_PTS`` or more since it was
  made (the earliest open call per symbol and direction, so repeats count once).

**Raised once, not every week.** A standing breach is a decision the investor
has already seen; repeating it every Monday teaches them to ignore the message.
A finding comes back only when it has got worse by its own step, or after
``REMIND_DAYS``. Model-free: the figures are computed here, never written by a
model. Upcoming earnings are left to the daily and weekly reports, which list them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
import asyncio
import json

from . import hooks, jobs
from .storage import read_json, state_file, write_private

DRIFT_PTS = 5.0
BEHIND_PTS = 10.0
REMIND_DAYS = 28


@dataclass
class Finding:
    key: str
    line: str
    value: float   # the size of the problem, in the finding's own unit
    step: float    # how much worse it must get to be raised again early


def state_path():
    return state_file("checkin-state.json", "FRA_CHECKIN_STATE")


# --- what is measured -----------------------------------------------------------------------


def _caps(book: dict[str, Any], prof: dict[str, Any],
          sectors: dict[str, float]) -> list[Finding]:
    from . import universes

    cap = float(prof["max_position_pct"])
    out = []
    for h in sorted(book["holdings"], key=lambda h: -h["weight_pct"]):
        sym = h["symbol"]
        if (universes.asset_class_of(sym) == "us_equity" and not universes.fund(sym)
                and h["weight_pct"] > cap):
            out.append(Finding(f"cap:{sym}", f"{sym} is {h['weight_pct']:.1f}% of the book, "
                               f"over your {cap:g}% position cap.", h["weight_pct"], 1.0))
    sector_cap = float(prof["max_sector_pct"])
    for sector, weight in sorted(sectors.items(), key=lambda kv: -kv[1]):
        if weight > sector_cap:
            out.append(Finding(f"sector:{sector}", f"{sector} is {weight:.1f}% of the book, "
                               f"over your {sector_cap:g}% sector cap.", weight, 2.0))
    return out


def _mix(book: dict[str, Any], prof: dict[str, Any]) -> list[Finding]:
    from . import profile, universes

    have = universes.allocation(book)
    out = []
    for cls in sorted(set(prof["targets"]) | set(have)):
        target = float(prof["targets"].get(cls, 0.0))
        actual = have.get(cls, 0.0)
        if abs(actual - target) >= DRIFT_PTS:
            name = profile.ASSET_CLASSES.get(cls, cls)
            side = "over" if actual > target else "under"
            out.append(Finding(f"mix:{cls}", f"{name[:1].upper()}{name[1:]}: {actual:.0f}% of "
                               f"the book against a {target:g}% target ({side} by "
                               f"{abs(actual - target):.0f} points).",
                               abs(actual - target), 3.0))
    return out


def _calls_behind() -> list[Finding]:
    """Open calls losing to their benchmark, the earliest per symbol and direction."""
    from . import journal

    earliest: dict[tuple[str, str], dict[str, Any]] = {}
    later: dict[tuple[str, str], int] = {}
    for e in journal.load_entries():
        if e.get("status") != "open" or e.get("verdict") not in ("bullish", "bearish"):
            continue
        k = (str(e["symbol"]), str(e["verdict"]))
        if k in earliest:
            later[k] = later.get(k, 0) + 1
            if str(e.get("entry_date")) >= str(earliest[k].get("entry_date")):
                continue
        earliest[k] = e
    out = []
    for (sym, verdict), e in earliest.items():
        entry = float(e.get("entry_price") or 0)
        try:
            opened = date.fromisoformat(str(e.get("entry_date"))[:10])
        except ValueError:
            continue
        price, priced_on = journal._spot(sym)  # pyright: ignore[reportPrivateUsage]
        bench = str(e.get("benchmark") or "SPY")
        b0, _ = journal._price_on(bench, opened)  # pyright: ignore[reportPrivateUsage]
        b1, _ = journal._spot(bench)  # pyright: ignore[reportPrivateUsage]
        if not (entry and price and b0 and b1):
            continue
        move, bench_move = (price / entry - 1) * 100, (b1 / b0 - 1) * 100
        behind = bench_move - move if verdict == "bullish" else move - bench_move
        if behind < BEHIND_PTS:
            continue
        repeats = later.get((sym, verdict), 0)
        out.append(Finding(
            f"call:{e.get('id')}",
            f"Your {verdict} call on {sym} ({e.get('id')}, {opened:%d %b} at {entry:,.2f}) is "
            f"{behind:.0f} points behind: {sym} {move:+.1f}% vs {bench} {bench_move:+.1f}% "
            f"to {priced_on}." + (f" Made {repeats + 1} times; this is the first." if repeats
                                  else ""),
            behind, 5.0))
    return out


def findings(book: dict[str, Any] | None = None) -> list[Finding]:
    """Everything worth raising today, before the once-only filter."""
    from . import periodic, profile, recommend

    book = book if book is not None else periodic.load_book()
    out: list[Finding] = []
    if profile.is_set() and book["holdings"]:
        prof = profile.load()
        sectors = recommend._sector_weights(book)  # pyright: ignore[reportPrivateUsage]
        out += _caps(book, prof, sectors) + _mix(book, prof)
    return out + _calls_behind()


# --- raised once ----------------------------------------------------------------------------


def to_raise(found: list[Finding], now: datetime | None = None) -> list[Finding]:
    """The findings that are new, worse by their step, or due a reminder; records
    them as raised."""
    now = now or datetime.now(timezone.utc)
    state = read_json(state_path(), {})
    state = state if isinstance(state, dict) else {}
    out = []
    for f in found:
        last = state.get(f.key)
        if isinstance(last, dict):
            try:
                age = now - datetime.fromisoformat(str(last.get("at")))
            except ValueError:
                age = timedelta(days=REMIND_DAYS)
            if age < timedelta(days=REMIND_DAYS) and f.value < float(last.get("value") or 0) + f.step:
                continue
        out.append(f)
        state[f.key] = {"at": now.isoformat(), "value": round(f.value, 2)}
    # A finding that has cleared is forgotten, so it is news again if it comes back.
    live = {f.key for f in found}
    state = {k: v for k, v in state.items() if k in live}
    write_private(state_path(), json.dumps(state, indent=1), prefix=".checkin-")
    return out


def message(raised: list[Finding]) -> str:
    return "🧭 Worth a look this week:\n" + "\n".join(f"• {f.line}" for f in raised)


# --- the job and the command ----------------------------------------------------------------


async def _job(task: dict[str, Any], fake: bool) -> jobs.JobResult:
    from . import guardrails

    if paused := guardrails.paused_reason():
        return jobs.JobResult(True, f"check-in skipped: {paused}", notify=False)
    raised = to_raise(await asyncio.to_thread(findings))
    if not raised:
        return jobs.JobResult(True, "check-in: nothing new to raise", notify=False)
    return jobs.JobResult(True, message(raised))


async def _cmd_checkin(_arg: str, _fake: bool) -> str:
    found = await asyncio.to_thread(findings)
    if not found:
        return "Nothing to raise: the book is inside your caps and targets, and no open call is far behind."
    return message(found).replace("this week", "now")


jobs.register_job("checkin", _job)
jobs.register_default_job(jobs.DefaultJob(
    "checkin", "[job] Weekly check-in", "monday 08:00", "weekly"))
hooks.register_command("checkin", _cmd_checkin,
                       "what needs a decision: caps, target mix, calls going wrong")
