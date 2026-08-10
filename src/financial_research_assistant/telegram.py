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

from collections.abc import Iterator
from typing import Any
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
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


# --- files ---------------------------------------------------------------------

#: Telegram's own limits. A photo over these is rejected or silently recompressed
#: to mush, so an oversized image is sent as a document instead — worse inline, but
#: it arrives intact and readable.
_PHOTO_MAX_BYTES = 10 * 1024 * 1024
_DOC_MAX_BYTES = 50 * 1024 * 1024
_PHOTO_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def _multipart(fields: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    """Encode one file plus text fields as multipart/form-data.

    Hand-rolled because this module is stdlib-only (like ``flex.py`` and
    ``factors.py``): adding ``requests`` for one upload would put a dependency in
    front of a feature that is off by default.
    """
    boundary = "----fra" + os.urandom(12).hex()
    out = bytearray()
    for key, value in fields.items():
        out += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
            f"{value}\r\n"
        ).encode("utf-8")
    out += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8")
    out += path.read_bytes()
    out += f"\r\n--{boundary}--\r\n".encode("utf-8")
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _upload(method: str, path: Path, fields: dict[str, str], file_field: str) -> bool:
    body, content_type = _multipart(fields, file_field, path)
    req = urllib.request.Request(
        f"{_API}/bot{bot_token()}/{method}",
        data=body,
        headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (https)
            return bool(json.loads(resp.read().decode("utf-8")).get("ok"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("description", "")
        except Exception:  # noqa: BLE001 - best-effort
            pass
        raise RuntimeError(_safe(f"telegram {method} failed: {exc.code} {detail}")) from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(_safe(f"telegram {method} failed: {exc}")) from None


def send_file(
    path: str, caption: str = "", chat_id: str = "", full_quality: bool = False
) -> bool:
    """Send a file. False when not configured or the file is missing/too big.

    ``full_quality`` decides how an IMAGE travels, and it matters more than any
    render setting: ``sendPhoto`` re-encodes to JPEG and downscales, which turns
    the body text of a dense report to mush no matter what resolution it was
    rendered at. ``sendDocument`` transfers the bytes untouched — the client still
    shows a tappable preview — so a text-bearing sheet goes that way and only
    genuinely photographic content should take the compressed path.
    """
    target = (chat_id or default_chat_id()).strip()
    src = Path(path)
    if not bot_token() or not target or not src.is_file():
        return False
    size = src.stat().st_size
    if size > _DOC_MAX_BYTES:
        return False
    as_photo = (
        not full_quality
        and src.suffix.lower() in _PHOTO_SUFFIXES
        and size <= _PHOTO_MAX_BYTES
    )
    fields = {"chat_id": target}
    if caption:
        # Telegram caps a caption at 1024 characters and rejects the whole upload
        # if it is longer, taking the file with it.
        fields["caption"] = caption[:1000]
    if as_photo:
        return _upload("sendPhoto", src, fields, "photo")
    return _upload("sendDocument", src, fields, "document")


# --- inbound -------------------------------------------------------------------


def _offset_file() -> Path:
    raw = os.environ.get("FINANCIAL_RESEARCH_TELEGRAM_STATE")
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    return Path.home() / ".financial-research-assistant" / "telegram-offset.json"


@contextmanager
def _inbox_lock() -> Iterator[bool]:
    """Hold the right to poll the inbox; yields whether we got it.

    Reading the cursor, fetching, and writing the cursor back is one
    read-modify-write over shared state, exactly like the task store — and it has
    the same two claimants, a cron ``--run-due`` and a ``--watch`` loop. Unguarded,
    both read the same offset, both receive the same update, and both run a full
    model turn on one message; the user gets two answers and pays twice.

    Non-blocking, unlike ``tasks._locked``: the lock is held across a long poll of
    up to ``POLL_TIMEOUT`` seconds, so waiting for it would stall a cron tick for
    half a minute to do work the other process is already doing. Losing the race
    means "someone else owns the inbox right now", and the correct response is to
    skip it this tick. POSIX-only; without ``fcntl`` this degrades to no locking
    rather than failing, the same trade ``tasks.py`` makes.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        yield True
        return
    path = _offset_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:  # pragma: no cover - an unwritable state dir
        yield True  # a missing lock file costs a duplicate answer, not a poll
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


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

    The whole read-fetch-write span runs under ``_inbox_lock``, so a second runner
    polling at the same moment gets nothing rather than a second copy of the same
    message. Returning ``[]`` when the lock is held is not a lost message: the
    process that owns the poll is answering it.
    """
    if not inbound_enabled():
        return []
    with _inbox_lock() as held:
        if not held:
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
