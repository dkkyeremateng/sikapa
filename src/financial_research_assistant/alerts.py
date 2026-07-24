"""User-defined alert rules for the monitoring digest.

The digest (``monitor.build_digest``) already surfaces movers, upcoming earnings,
and ex-dividends over your holdings. This adds *personal thresholds* on top —
"tell me if AAPL drops 5%", "alert if any holding moves more than 8%", "notify me
if TSLA goes below 200", "flag NVDA earnings within a week" — so the digest
becomes a push you can act on, not just a status report.

Rules are stored as plain JSON at ``~/.financial-research-assistant/alerts.json``
(override with ``FINANCIAL_RESEARCH_ALERTS_FILE``), one object per rule:
``{"id", "symbol", "kind", "value"}``. ``symbol`` is a ticker, or ``"*"`` for every
holding. ``kind`` is one of:

- ``drop`` / ``rise`` / ``move`` — the price moved ≥ ``value`` percent over the
  digest's lookback window (down / up / either way).
- ``below`` / ``above`` — the latest price is at/through the ``value`` level
  (needs a specific symbol).
- ``earnings`` — the next earnings date is within ``value`` days.

Evaluation is model-free and reuses the same keyless fetch helpers the digest
uses (so it's exercised offline in tests and the price cache makes re-fetching
holdings cheap). ``monitor.build_digest`` calls ``evaluate_alerts`` and prepends a
"🔔 Alerts triggered" section when any fire.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

_KINDS = {"drop", "rise", "move", "below", "above", "earnings"}
# Symbol tokens that mean "every holding" rather than one ticker.
_ALL = {"*", "all", "any", "portfolio", "holdings", "everything"}
# Loose synonyms so the model's plain-English kind maps to a canonical one.
_KIND_SYNONYMS = {
    "falls": "drop", "fall": "drop", "down": "drop", "drops": "drop", "decline": "drop",
    "rises": "rise", "up": "rise", "gains": "rise", "gain": "rise", "climbs": "rise",
    "moves": "move", "changes": "move", "change": "move", "swing": "move",
    "under": "below", "less": "below", "beneath": "below",
    "over": "above", "greater": "above", "exceeds": "above",
    "reports": "earnings", "reporting": "earnings",
}
_DEFAULT_EARNINGS_DAYS = 7.0


def _alerts_file() -> Path:
    raw = os.environ.get("FINANCIAL_RESEARCH_ALERTS_FILE")
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    return Path.home() / ".financial-research-assistant" / "alerts.json"


def load_alerts() -> list[dict]:
    p = _alerts_file()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [r for r in data if isinstance(r, dict) and r.get("kind")] if isinstance(data, list) else []


def save_alerts(rules: list[dict]) -> None:
    p = _alerts_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rules, indent=2), encoding="utf-8")


def _next_id(rules: list[dict]) -> str:
    """A short stable id (``a1``, ``a2``, …), one past the current max."""
    n = 0
    for r in rules:
        rid = str(r.get("id", ""))
        if rid.startswith("a") and rid[1:].isdigit():
            n = max(n, int(rid[1:]))
    return f"a{n + 1}"


def _describe(r: dict) -> str:
    sym, kind, v = r.get("symbol", "*"), r.get("kind", ""), r.get("value", 0)
    who = "any holding" if sym in _ALL else sym
    return {
        "drop": f"{who} drops ≥ {v:g}% over the digest lookback",
        "rise": f"{who} rises ≥ {v:g}% over the digest lookback",
        "move": f"{who} moves ≥ {v:g}% either way over the digest lookback",
        "below": f"{sym} price at or below {v:g}",
        "above": f"{sym} price at or above {v:g}",
        "earnings": f"{who} reports earnings within {int(v)} day(s)",
    }.get(kind, f"{who} {kind} {v}")


def _canon(symbol: str, kind: str) -> tuple[str, str]:
    sym = (symbol or "").strip().upper() or "*"
    if sym.lower() in _ALL:
        sym = "*"
    k = (kind or "").strip().lower()
    return sym, _KIND_SYNONYMS.get(k, k)


# --- Model-facing tools ------------------------------------------------------

def add_alert(symbol: str, kind: str, value: float = 0.0) -> str:
    """Create a monitoring alert rule that the portfolio digest checks each run.

    ``symbol`` is a ticker (e.g. ``AAPL``), or ``*`` / "any" for every holding.
    ``kind`` is: ``drop`` / ``rise`` / ``move`` (``value`` = percent move over the
    digest's lookback window, down / up / either way); ``below`` / ``above``
    (``value`` = a price level — needs a specific symbol); or ``earnings``
    (``value`` = days — fire when the next earnings date is within that many days).
    Use when the user says 'tell me if / alert me when / notify me if X does Y' —
    e.g. add_alert('AAPL','drop',5), add_alert('*','move',8),
    add_alert('TSLA','below',200), add_alert('NVDA','earnings',7). The alert fires
    in the next `portfolio_digest` (or the `--digest` CLI). Manage with
    `list_alerts` and `remove_alert`."""
    sym, k = _canon(symbol, kind)
    if k not in _KINDS:
        return (f"Unknown alert kind {kind!r}. Use: drop, rise, move (percent), "
                f"below, above (price level), or earnings (days).")
    if k in ("below", "above") and sym == "*":
        return "A price-level alert (below/above) needs a specific symbol, not '*'/any."
    try:
        v = float(value or 0)
    except (TypeError, ValueError):
        v = 0.0
    if k == "earnings":
        v = v if v > 0 else _DEFAULT_EARNINGS_DAYS
    elif v <= 0:
        return ("Give a positive threshold — a percent for drop/rise/move, or a "
                "price for below/above.")
    rules = load_alerts()
    rule = {"id": _next_id(rules), "symbol": sym, "kind": k, "value": v}
    rules.append(rule)
    save_alerts(rules)
    return (f"Added alert {rule['id']}: {_describe(rule)}. It's checked whenever you "
            f"run the portfolio digest (`portfolio_digest` or the --digest CLI).")


def list_alerts() -> str:
    """List the monitoring alert rules currently set (each with its id), so you can
    review what the portfolio digest is watching for. Use for 'what alerts do I
    have / what am I being notified about'."""
    rules = load_alerts()
    if not rules:
        return ("No alert rules set. Add one with add_alert(symbol, kind, value) — "
                "e.g. add_alert('AAPL','drop',5) to be told if AAPL drops 5%.")
    return "Alert rules (checked by the portfolio digest):\n" + "\n".join(
        f"  [{r['id']}] {_describe(r)}" for r in rules
    )


def remove_alert(alert_id: str) -> str:
    """Remove a monitoring alert rule by its id (from `list_alerts`), or pass
    ``all`` to clear every rule. Use for 'stop alerting me about X / remove that
    alert / clear my alerts'."""
    aid = (alert_id or "").strip().lower()
    rules = load_alerts()
    if not rules:
        return "No alert rules to remove."
    if aid in ("all", "*", "everything"):
        save_alerts([])
        return f"Removed all {len(rules)} alert rule(s)."
    kept = [r for r in rules if str(r.get("id", "")).lower() != aid]
    if len(kept) == len(rules):
        return f"No alert with id {alert_id!r}. Use `list_alerts` to see the ids."
    save_alerts(kept)
    return f"Removed alert {alert_id}."


# --- Evaluation (called by the digest) ---------------------------------------

def _check_rule(sym: str, kind: str, v: float, lookback_days: int, today: date,
                fetch_daily, fetch_calendar) -> str | None:
    """Evaluate one rule against one symbol; return a trigger message or None."""
    if kind in ("drop", "rise", "move"):
        series = fetch_daily(sym, max(5, lookback_days))
        if len(series) < 2 or not series[0][1]:
            return None
        first, last = series[0][1], series[-1][1]
        chg = (last - first) / first * 100.0
        if kind == "drop" and chg <= -v:
            return f"{sym} down {chg:+.1f}% over ~{lookback_days}d (alert: drop ≥ {v:g}%) — now {last:.2f}"
        if kind == "rise" and chg >= v:
            return f"{sym} up {chg:+.1f}% over ~{lookback_days}d (alert: rise ≥ {v:g}%) — now {last:.2f}"
        if kind == "move" and abs(chg) >= v:
            return f"{sym} moved {chg:+.1f}% over ~{lookback_days}d (alert: move ≥ {v:g}%) — now {last:.2f}"
        return None
    if kind in ("below", "above"):
        series = fetch_daily(sym, 5)
        if not series:
            return None
        price = series[-1][1]
        if kind == "below" and price <= v:
            return f"{sym} at {price:.2f} — at/below your {v:g} level"
        if kind == "above" and price >= v:
            return f"{sym} at {price:.2f} — at/above your {v:g} level"
        return None
    if kind == "earnings":
        cal = fetch_calendar(sym)
        if not cal:
            return None
        from .monitor import _next_earnings

        ed = _next_earnings(cal)
        if ed and today <= ed and (ed.toordinal() - today.toordinal()) <= int(v):
            return f"{sym} reports earnings {ed.isoformat()} (within {int(v)}d)"
        return None
    return None


def evaluate_alerts(symbols_held: list[str], lookback_days: int, today: date) -> list[str]:
    """Check every stored rule and return ``["[id] message", …]`` for those that
    fire. A ``*`` rule applies to each held symbol; a symbol-specific rule applies
    to that symbol even if it isn't currently held (so watchlist alerts work)."""
    rules = load_alerts()
    if not rules:
        return []
    from .fundamentals import _fetch_calendar
    from .tools import _fetch_daily

    triggered: list[str] = []
    for r in rules:
        sym = r.get("symbol", "*")
        kind = r.get("kind", "")
        try:
            v = float(r.get("value", 0) or 0)
        except (TypeError, ValueError):
            continue
        targets = symbols_held if sym in _ALL else [sym]
        for t in targets:
            msg = _check_rule(t, kind, v, lookback_days, today, _fetch_daily, _fetch_calendar)
            if msg:
                triggered.append(f"[{r['id']}] {msg}")
    return triggered


ALERT_TOOLS = [add_alert, list_alerts, remove_alert]
