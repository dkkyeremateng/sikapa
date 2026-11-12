"""Model calls nobody asked for in the moment — the agent's autonomous spend.

A chat reply is paid for by a question. A daily report's commentary, an event
push's analysis and a recommendation run are not: the agent decides to make
them, on a schedule, while nobody watches. Every such call goes through ``ask``
so there is ONE place that knows what autonomous work costs, can refuse it once
a budget is spent, and can be switched off.

``ask`` returns None instead of raising when the call can't or mustn't be made
(no model, over budget, paused, provider error). Every caller has a model-free
fallback — a report without commentary, an event pushed as facts only — and
reaching it is a normal outcome, not an error.
"""

from __future__ import annotations

from typing import Any
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta

from . import guardrails, hooks
from .storage import append_jsonl, read_json, read_jsonl, state_file, write_private

#: Tiers: "quick" is the cheap model (QUICK_MODEL), "default" the primary one.
TIERS = ("quick", "default")


def usage_log():
    return state_file("autonomy-usage.jsonl", "FRA_AUTONOMY_USAGE_LOG")


def failure_file():
    return state_file("autonomy-failure.json", "FRA_AUTONOMY_FAILURE")


#: While unattended calls keep failing, the user hears about it this often.
_FAILURE_ALERT_EVERY = timedelta(hours=24)


def last_failure() -> dict[str, Any] | None:
    """The current run of failed autonomous calls, or None when the last call
    worked."""
    data = read_json(failure_file(), None)
    return data if isinstance(data, dict) and data.get("error") else None


async def _note_failure(exc: BaseException, purpose: str) -> None:
    """Make a failed call visible. It used to return None without a trace, so an
    expired key quietly turned every report into the commentary-free fallback and
    every event note into bare facts, with nothing in the log or /status. Logged
    each time, recorded for /status, and pushed once a day while it lasts."""
    from .errors import describe_error
    from .remedies import advice

    error = describe_error(exc)
    print(f"autonomy: {purpose or 'call'} failed: {error}", file=sys.stderr)
    now = datetime.now()
    prev = last_failure() or {}
    record = {"since": prev.get("since") or now.isoformat(timespec="seconds"),
              "at": now.isoformat(timespec="seconds"), "error": error[:300],
              "purpose": purpose, "alerted": prev.get("alerted") or ""}
    try:
        stale = (not record["alerted"]
                 or now - datetime.fromisoformat(record["alerted"]) >= _FAILURE_ALERT_EVERY)
    except ValueError:
        stale = True
    if stale:
        from . import channels

        fix = advice(error)
        text = (f"⚠ Unattended model calls are failing (since {record['since'][:16]}), so "
                f"reports go out without commentary and alerts without notes.\n{error[:300]}"
                + (f"\n\nWhat to do: {fix}" if fix else ""))
        try:
            await asyncio.to_thread(channels.deliver, text)
            record["alerted"] = now.isoformat(timespec="seconds")
        except Exception:  # noqa: BLE001 - the alert is best-effort; the log has it
            pass
    write_private(failure_file(), json.dumps(record, indent=1), prefix=".autonomy-")


def _note_success() -> None:
    try:
        failure_file().unlink()
    except FileNotFoundError:
        pass


def _int_env(name: str) -> int:
    raw = (os.environ.get(name) or "").strip().replace("_", "")
    return int(raw) if raw.isdigit() else 0


def daily_cap() -> int:
    """Token cap for autonomous work per local day (``FRA_AUTONOMY_DAILY_TOKENS``;
    0 = uncapped)."""
    return _int_env("FRA_AUTONOMY_DAILY_TOKENS")


def monthly_cap() -> int:
    """Token cap per calendar month (``FRA_AUTONOMY_MONTHLY_TOKENS``; 0 = uncapped)."""
    return _int_env("FRA_AUTONOMY_MONTHLY_TOKENS")


def spent(now: datetime | None = None) -> dict[str, int]:
    """Tokens used by autonomous work today and this month (local time)."""
    now = now or datetime.now()
    day, month = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")
    out = {"today": 0, "month": 0}
    for rec in read_jsonl(usage_log()):
        at = str(rec.get("at") or "")
        tokens = int(rec.get("tokens") or 0)
        if at.startswith(month):
            out["month"] += tokens
            if at.startswith(day):
                out["today"] += tokens
    return out


def record_usage(purpose: str, tokens_in: int, tokens_out: int, model: str = "") -> None:
    append_jsonl(usage_log(), {
        "at": datetime.now().isoformat(timespec="seconds"),
        "purpose": purpose,
        "model": model,
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "tokens": int(tokens_in) + int(tokens_out),
    })


def over_budget(now: datetime | None = None) -> str:
    """Why autonomous model calls are refused right now, or "" when they aren't."""
    used = spent(now)
    if daily_cap() and used["today"] >= daily_cap():
        return f"today's autonomous token budget is spent ({used['today']:,}/{daily_cap():,})"
    if monthly_cap() and used["month"] >= monthly_cap():
        return f"this month's autonomous token budget is spent ({used['month']:,}/{monthly_cap():,})"
    return ""


def blocked() -> str:
    """Why autonomous work may not call a model now ("" when it may). The pause
    switch and the budget both land here, so every caller checks one thing."""
    return guardrails.paused_reason() or over_budget()


def _usage_of(resp: Any) -> tuple[int, int]:
    meta = getattr(resp, "usage_metadata", None) or {}
    try:
        return int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0)
    except (TypeError, ValueError, AttributeError):
        return 0, 0


async def ask(
    system: str, user: str, *, tier: str = "quick", purpose: str = "", fake: bool = False,
    fake_reply: str = "",
) -> str | None:
    """One autonomous model call. Returns the reply text, or None when the call was
    not made (paused, over budget, no model) or failed — see the module docstring.

    ``fake`` returns ``fake_reply`` without a model, so every caller is testable
    offline and exercised by ``--fake`` runs.
    """
    if fake:
        return fake_reply or None
    if blocked():
        return None
    from langchain_core.messages import HumanMessage, SystemMessage

    from .messages import message_text
    from .llm import _make_llm, quick_llm, resolved_model  # pyright: ignore[reportPrivateUsage]

    try:
        llm = quick_llm() if tier == "quick" else _make_llm(None)
        resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the caller has a model-free fallback
        await _note_failure(exc, purpose or tier)
        return None
    if last_failure():
        _note_success()
    tin, tout = _usage_of(resp)
    model = os.environ.get("QUICK_MODEL", "") if tier == "quick" else ""
    try:
        record_usage(purpose or tier, tin, tout, model or resolved_model(None))
    except Exception:  # noqa: BLE001 - accounting must not lose a paid-for reply
        pass
    return message_text(resp).strip() or None


def budget_line() -> str:
    """The `/status` line: on, paused or over budget, and what has been spent."""
    used = spent()
    day_cap, month_cap = daily_cap(), monthly_cap()
    day = f"{used['today']:,}" + (f"/{day_cap:,}" if day_cap else "")
    month = f"{used['month']:,}" + (f"/{month_cap:,}" if month_cap else "")
    state = guardrails.paused_reason() or over_budget() or "on"
    quiet = guardrails.quiet_now()
    failing = last_failure()
    return (f"autonomy: {state} · tokens today {day}, this month {month}"
            + (f" · {quiet}" if quiet else "")
            + (f" · ⚠ model calls failing since {failing['since'][:16]}: "
               f"{failing['error'][:80]}" if failing else ""))


hooks.register_status_line("autonomy", budget_line)
