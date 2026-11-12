"""Turning an exception into something a user can act on. Its own module so the
modules below the agent (autonomy, the scheduler) can use it without importing
the agent."""

from __future__ import annotations


def describe_error(exc: BaseException) -> str:
    """Flatten an exception to something a user can act on.

    The graph runs inside anyio task groups (the MCP session), so a failure
    anywhere arrives wrapped: ``str()`` gives "unhandled errors in a TaskGroup
    (1 sub-exception)", which is true and useless — the actual 404 or auth error
    is a leaf several levels down. Walk to the leaves and report those, keeping
    the type name because "NotFoundError" vs "AuthenticationError" is usually the
    whole diagnosis. Credentials are masked: an httpx error can carry the request
    headers.
    """
    from .auth import redact

    leaves: list[str] = []

    def walk(e: BaseException, depth: int = 0) -> None:
        subs = getattr(e, "exceptions", None)  # ExceptionGroup / anyio group
        if subs and depth < 5:
            for sub in subs:
                walk(sub, depth + 1)
            return
        text = str(e).strip()
        leaves.append(f"{type(e).__name__}: {text}" if text else type(e).__name__)

    walk(exc)
    # dict.fromkeys dedupes while keeping order — a fan-out often fails identically
    # in several tasks at once.
    return redact("; ".join(dict.fromkeys(leaves)) or type(exc).__name__)
