"""Pluggable delivery-channel registry — where background work sends its answer.

The scheduler runs a turn hours after you left the terminal, so "print it to
stdout" is the same as throwing it away. This module is the seam between *work
finished* and *user informed*, built the same way ``brokers.py`` handles market
data: each channel is one ``Channel`` entry declaring

1. a **key** — the name used in a task's ``channel`` field (``"telegram"``),
2. **configured** — whether its environment resolves, so a channel is *opt-in by
   the presence of its env vars* and an unconfigured one is simply absent, and
3. **send** — deliver one message, returning whether it went.

Adding a channel (email, Slack, ntfy, a webhook) means registering one provider
here; ``scheduler.py`` and ``tasks.py`` never learn its name.

Selecting channels at runtime:

- Unset ``NOTIFY_CHANNELS`` -> every registered channel whose env resolves.
- ``NOTIFY_CHANNELS=telegram,desktop`` -> pin an explicit subset.
- A task's own ``channel`` field overrides both, so one job can go to your phone
  while the rest stay local.

Delivery is best-effort by design: ``deliver`` never raises. A background run that
already produced its answer must not be recorded as failed — and retried, at the
cost of another model run — because a notification API was briefly down. The
failure is reported in the return value instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

#: A channel's sender: takes the message, returns whether it was delivered.
Sender = Callable[[str], bool]


#: A channel's file sender: (path, caption, full_quality) -> delivered?
#: ``full_quality`` asks the channel not to recompress — see telegram.send_file.
#: Optional — a channel that can only carry text simply leaves it None, and
#: ``deliver_file`` skips it rather than pretending.
FileSender = Callable[[str, str, bool], bool]


@dataclass(frozen=True)
class Channel:
    """One pluggable delivery target.

    ``configured`` reports whether this channel's environment is set up;
    ``send`` delivers a single message and returns success. ``send_file`` is
    optional: a rendered PDF has nowhere to go on a desktop-banner channel, and a
    channel that cannot carry a file must be skipped visibly rather than silently
    dropping it or stringifying it into a message.

    ``full_content`` is the same kind of declaration for text: a channel that
    truncates to a headline has shown the user that something happened, not the
    answer. Declared here rather than judged at the call site, so the scheduler can
    ask "did the analysis actually reach anyone?" without knowing any channel's
    name — the same reason ``send_file`` is a capability and not a lookup table.
    """

    key: str
    configured: Callable[[], bool]
    send: Sender
    #: Shown when explaining where a task's answer will go.
    label: str = ""
    send_file: FileSender | None = None
    #: False for banner-style channels that carry only an opening line.
    full_content: bool = True


CHANNEL_REGISTRY: dict[str, Channel] = {}


def register_channel(channel: Channel) -> None:
    """Add (or replace) a channel in the registry, keyed by ``channel.key``."""
    CHANNEL_REGISTRY[channel.key] = channel


def registered_keys() -> list[str]:
    return sorted(CHANNEL_REGISTRY)


def _selected() -> list[Channel]:
    """Channels permitted this run — every registered one, or the pinned subset."""
    sel = (os.environ.get("NOTIFY_CHANNELS") or "").strip()
    if not sel:
        return [CHANNEL_REGISTRY[k] for k in registered_keys()]
    keys = [k.strip().lower() for k in sel.split(",") if k.strip()]
    return [CHANNEL_REGISTRY[k] for k in keys if k in CHANNEL_REGISTRY]


def active_channels(prefer: str = "") -> list[Channel]:
    """The channels a message should go to.

    ``prefer`` is a task's own channel field: an explicit choice, honoured even if
    ``NOTIFY_CHANNELS`` doesn't list it (the user naming a channel on the task IS
    the selection), but still subject to being configured — sending to a channel
    with no token would just fail.
    """
    key = (prefer or "").strip().lower()
    if key:
        ch = CHANNEL_REGISTRY.get(key)
        return [ch] if ch is not None and ch.configured() else []
    return [c for c in _selected() if c.configured()]


def describe_targets(prefer: str = "") -> str:
    """Human phrase for where a message will go — used when confirming a schedule.

    Names the fallback honestly when nothing is configured: a user told "it will be
    sent to you" who has set up no channel would find the answer only in a log file.
    """
    chans = active_channels(prefer)
    if chans:
        return ", ".join(c.label or c.key for c in chans)
    if (prefer or "").strip():
        return (
            f"nothing — '{prefer}' is not configured, so the answer will only be "
            "written to the task log (see --tasks)"
        )
    return (
        "nothing yet — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (or "
        "NOTIFY_CHANNELS=desktop) to have it pushed; until then it is only in the "
        "task log (see --tasks)"
    )


def deliver(text: str, prefer: str = "") -> tuple[list[str], list[str]]:
    """Send ``text`` to the active channels. Returns ``(delivered, failed)`` keys.

    Never raises: see the module docstring — a delivery failure must not cost the
    completed work its "done" status.
    """
    delivered: list[str] = []
    failed: list[str] = []
    if not (text or "").strip():
        return delivered, failed
    for ch in active_channels(prefer):
        try:
            (delivered if ch.send(text) else failed).append(ch.key)
        except Exception:  # noqa: BLE001 - any channel failure is just a failed send
            failed.append(ch.key)
    return delivered, failed


def deliver_file(
    path: str, caption: str = "", prefer: str = "", full_quality: bool = False
) -> tuple[list[str], list[str]]:
    """Send a file to the active channels that can carry one.

    Returns ``(delivered, failed)``. A channel with no ``send_file`` is neither —
    it is simply not a target for a file, and reporting it as failed would read as
    an outage when it is a capability boundary.
    """
    delivered: list[str] = []
    failed: list[str] = []
    for ch in active_channels(prefer):
        if ch.send_file is None:
            continue
        try:
            (delivered if ch.send_file(path, caption, full_quality) else failed).append(ch.key)
        except Exception:  # noqa: BLE001 - a channel failure is never fatal here either
            failed.append(ch.key)
    return delivered, failed


def carried_full_text(delivered: list[str]) -> bool:
    """Whether any of these delivered-to channels carried the WHOLE message.

    The distinction the scheduler needs before it decides an answer has arrived.
    ``desktop`` is on by default and reports success as soon as the notifier is
    spawned, so a run whose Telegram delivery failed still came back with one
    "delivered" channel — and a full analysis was quietly reduced to a 200-character
    banner, with nothing parked for redelivery. A key that is no longer in the
    registry counts as full: truncation is a property a channel has to declare, and
    treating everything unrecognised as a banner would park answers that did arrive.
    """
    return any(
        CHANNEL_REGISTRY[k].full_content
        for k in delivered
        if k in CHANNEL_REGISTRY
    ) or any(k not in CHANNEL_REGISTRY for k in delivered)


def file_capable(prefer: str = "") -> list[str]:
    """Active channels that can carry a file — for telling the user where one can go."""
    return [c.key for c in active_channels(prefer) if c.send_file is not None]


# --- built-in channels ---------------------------------------------------------


def _telegram_send(text: str) -> bool:
    from . import telegram

    return telegram.send_message(text)


def _telegram_configured() -> bool:
    from . import telegram

    return telegram.configured()


def _telegram_send_file(path: str, caption: str, full_quality: bool = False) -> bool:
    from . import telegram

    return telegram.send_file(path, caption, full_quality=full_quality)


def _desktop_send(text: str) -> bool:
    """An OS banner. Truncated hard — a notification centre shows a line or two, and
    the full answer is in the task log and any other configured channel.

    Registered ``full_content=False`` for exactly that reason, and because the
    return value is optimistic anyway: it says a notifier process was started, not
    that anything appeared on screen."""
    from . import alerts

    head = (text or "").strip().splitlines()
    first = head[0] if head else ""
    return alerts.notify_desktop(first[:200])


def _desktop_configured() -> bool:
    """Enabled AND there is a notifier on this machine to raise the banner.

    Enabled alone used to be enough, and it is on by default — so on a headless
    server with no ``notify-send`` every delivery listed "desktop" as a failed
    channel, and every log line for a delivered answer read like an outage.
    """
    from . import alerts

    return alerts.desktop_enabled() and alerts._desktop_cmd("") is not None  # pyright: ignore[reportPrivateUsage]


def _stdout_send(text: str) -> bool:
    import sys

    print(text, file=sys.stdout, flush=True)
    return True


register_channel(Channel(
    "telegram", _telegram_configured, _telegram_send, "Telegram",
    send_file=_telegram_send_file,
))
register_channel(Channel(
    "desktop", _desktop_configured, _desktop_send, "desktop notification",
    full_content=False,
))
# stdout is never implicitly active: it's for `NOTIFY_CHANNELS=stdout` in a cron
# job that mails its own output. Left in the default set it would "deliver" every
# scheduled answer into a log nobody reads and report success for it.
register_channel(Channel(
    "stdout", lambda: (os.environ.get("NOTIFY_CHANNELS") or "").find("stdout") >= 0,
    _stdout_send, "stdout",
))
