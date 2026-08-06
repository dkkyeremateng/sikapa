"""Telegram Bot API client — how scheduled work reaches you when the terminal is shut.

Two directions, deliberately asymmetric in what they trust:

**Outbound** (``send_message``) is the point of the module: a scheduled run
finishes hours after you closed the laptop, and its answer has to go somewhere
you'll actually see. Needs ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID``.

**Inbound** (``get_updates``) lets you message the bot and have the agent answer —
convenient, and a genuine attack surface: every message is untrusted text handed
to a tool-using agent that can read your imported statements and broker positions.
Three things keep that bounded:

1. **An allowlist is mandatory.** ``TELEGRAM_ALLOWED_CHAT_IDS`` must name the chats
   that may drive the agent; with it unset, ``inbound_enabled()`` is False and
   nothing is polled. Bot usernames are guessable and anyone can message a bot, so
   "poll everything" would mean a stranger driving your portfolio agent.
2. **Non-allowlisted messages are dropped before the model sees them** — not
   answered with a refusal, which would confirm the bot is live.
3. The read-only broker filter (``tools.py``) still applies, so an inbound message
   has exactly the reach an interactive turn has: it can read, never trade.

The bot token is a bearer credential in the URL path, so it must never reach a log
line: ``_safe`` masks it in every error string this module produces.

Standard library only (``urllib``) — the same choice ``flex.py`` and ``factors.py``
make, so there's no new dependency for a feature that is off by default.
"""

from __future__ import annotations

from typing import Any
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_API = "https://api.telegram.org"

#: Telegram rejects anything longer; we split rather than truncate, since the tail
#: of an analysis is usually the recommendation.
MAX_MESSAGE = 4096

#: Long-poll seconds. Telegram holds the connection open until a message arrives or
#: this elapses, so a watcher costs one idle request per interval instead of a busy
#: loop. Kept under the socket timeout below.
POLL_TIMEOUT = 25


def bot_token() -> str:
    return (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()


def default_chat_id() -> str:
    return (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()


def configured() -> bool:
    """True when outbound delivery is possible (token + a destination chat)."""
    return bool(bot_token() and default_chat_id())


def allowed_chat_ids() -> set[str]:
    """Chats permitted to drive the agent. Empty means inbound is off."""
    raw = (os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS") or "").strip()
    return {c.strip() for c in raw.split(",") if c.strip()}


def inbound_enabled() -> bool:
    """Inbound polling requires a token AND a non-empty allowlist.

    Not defaulting the allowlist to ``TELEGRAM_CHAT_ID``: that would turn inbound
    on for everyone who configured outbound, silently converting a notification
    channel into a remote-control channel. Opting in has to be a separate act.
    """
    return bool(bot_token() and allowed_chat_ids())


def is_allowed(chat_id: Any) -> bool:
    return str(chat_id) in allowed_chat_ids()


def _safe(text: str) -> str:
    """Mask the bot token, which sits in the URL path of every request.

    urllib puts the full URL in its exception text, so an unmasked error message
    would print a working credential into a cron log or a task's stored result.
    """
    token = bot_token()
    out = text or ""
    if token:
        out = out.replace(token, "***")
        # The token's own "<id>:<secret>" shape, in case only a part was echoed.
        head = token.split(":", 1)[0]
        if head and len(head) > 4:
            out = out.replace(head, "***")
    return out


def _call(method: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    """POST one Bot API method. Raises RuntimeError with a token-free message."""
    token = bot_token()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    url = f"{_API}/bot{token}/{method}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https)
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("description", "")
        except Exception:  # noqa: BLE001 - the error body is best-effort
            pass
        raise RuntimeError(_safe(f"telegram {method} failed: {exc.code} {detail}")) from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(_safe(f"telegram {method} failed: {exc}")) from None
    if not data.get("ok"):
        raise RuntimeError(_safe(f"telegram {method} rejected: {data.get('description', '')}"))
    return data


def _chunks(text: str, limit: int = MAX_MESSAGE) -> list[str]:
    """Split a long answer at line boundaries where possible.

    A market analysis routinely runs past 4096 characters. Splitting mid-word (or
    truncating) would cut the recommendation off the end, which is the one part
    the user scheduled the task for.
    """
    text = text or ""
    if len(text) <= limit:
        return [text] if text else []
    out: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n")
        if cut < limit // 2:  # no usable line break: fall back to a hard split
            cut = limit
        out.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        out.append(remaining)
    return out


def send_message(text: str, chat_id: str = "") -> bool:
    """Send ``text`` (split across messages if long). False if not configured.

    Sent as plain text, NOT Markdown: an analysis is full of ``*``, ``_`` and
    ``$`` from tickers and figures, and Telegram rejects the whole message when
    they don't parse as valid entities — losing the delivery to a formatting
    detail. A dropped report is worse than an unstyled one.
    """
    target = (chat_id or default_chat_id()).strip()
    if not bot_token() or not target:
        return False
    parts = _chunks(text)
    if not parts:
        return False
    for part in parts:
        _call("sendMessage", {
            "chat_id": target,
            "text": part,
            "disable_web_page_preview": True,
        })
    return True


# --- inbound -------------------------------------------------------------------


def _offset_file() -> Path:
    raw = os.environ.get("FINANCIAL_RESEARCH_TELEGRAM_STATE")
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    return Path.home() / ".financial-research-assistant" / "telegram-offset.json"


def _read_offset() -> int:
    try:
        return int(json.loads(_offset_file().read_text(encoding="utf-8")).get("offset", 0))
    except (OSError, ValueError, AttributeError, TypeError):
        return 0


def _write_offset(offset: int) -> None:
    """Persist the update cursor.

    Without it, a restarted watcher re-reads every message Telegram still holds and
    answers each one again — model calls for questions already answered.
    """
    path = _offset_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"offset": offset}), encoding="utf-8")
    except OSError:
        pass  # a lost cursor costs a duplicate answer, not a crash


def get_updates(timeout: int = POLL_TIMEOUT) -> list[dict[str, Any]]:
    """Fetch new messages from allowed chats. Returns ``[{chat_id, text, name}]``.

    The cursor advances past EVERY update, including the ones the allowlist drops —
    otherwise a message from a stranger is re-fetched forever and the queue never
    moves past it.
    """
    if not inbound_enabled():
        return []
    offset = _read_offset()
    payload: dict[str, Any] = {"timeout": timeout, "allowed_updates": ["message"]}
    if offset:
        payload["offset"] = offset
    data = _call("getUpdates", payload, timeout=timeout + 10)
    out: list[dict[str, Any]] = []
    highest = offset
    for upd in data.get("result", []):
        highest = max(highest, int(upd.get("update_id", 0)) + 1)
        msg = upd.get("message") or {}
        chat = (msg.get("chat") or {}).get("id")
        text = (msg.get("text") or "").strip()
        if not text or chat is None or not is_allowed(chat):
            continue
        frm = msg.get("from") or {}
        out.append({
            "chat_id": str(chat),
            "text": text,
            "name": frm.get("username") or frm.get("first_name") or str(chat),
        })
    if highest != offset:
        _write_offset(highest)
    return out
