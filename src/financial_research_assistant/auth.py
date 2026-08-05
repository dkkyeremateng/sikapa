"""Local credential store — API keys and OAuth tokens, kept out of ``.env``.

``.env`` is a poor place for a key: it is world-readable, it is easy to commit by
accident, and ``main.py`` loads it into the process environment where every
subprocess inherits it. This module is the alternative — a single JSON file at
``~/.financial-research-assistant/auth.json``, written ``0600``, holding one
credential per *scope*.

Scopes mirror the model tiers the agent already has (``graph.quick_llm`` /
``subagents``), so a cheap local model and the primary agent can hold separate
credentials::

    {"version": 1,
     "credentials": {
       "default":  {"provider": "openrouter", "type": "oauth",
                    "access": "...", "refresh": "...", "expires": 1786000000000},
       "quick":    {"provider": "openai", "type": "api_key", "key": "$MY_KEY"},
       "subagent": {"provider": "openai", "type": "api_key", "key": "!op read op://…"}
     }}

A stored key may be a literal, ``$VAR`` (read from the environment at use time),
or ``!command`` (run a shell command and take its stdout — so the secret can live
in 1Password/``pass`` and never touch disk here). ``!command`` is EXECUTION from a
config file, so it is off unless ``FINANCIAL_RESEARCH_AUTH_ALLOW_EXEC=1``: this
file sits beside a SQLite database of brokerage conversation history, and a config
that silently runs shell is a bigger blast radius than a leaked key.

The store deliberately ranks ABOVE the environment in ``graph._make_llm`` — see the
resolution comment there. Nothing in this module reaches the network; OAuth flows
live in the provider registry and only hand their result here to be persisted.
"""

from __future__ import annotations

from typing import Any
import json
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

# The tiers `_make_llm` can be called for. "default" is also the fallback for any
# scope that has no credential of its own, matching the "unset = inherit the
# primary agent's config" rule the QUICK_*/SUBAGENT_* env overrides already follow.
DEFAULT_SCOPE = "default"
SCOPES = (DEFAULT_SCOPE, "quick", "subagent")

_VERSION = 1


def auth_file() -> Path:
    raw = os.environ.get("FINANCIAL_RESEARCH_AUTH_FILE")
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    return Path.home() / ".financial-research-assistant" / "auth.json"


def _exec_allowed() -> bool:
    return os.environ.get("FINANCIAL_RESEARCH_AUTH_ALLOW_EXEC", "").lower() in (
        "1", "true", "yes",
    )


@contextmanager
def _locked():
    """Serialize read-modify-write across processes.

    Two instances refreshing the same expired token concurrently would otherwise
    race and one would clobber the other's new refresh token, logging that
    instance out. POSIX-only; on a platform without ``fcntl`` this degrades to no
    locking rather than failing, which is the same trade the rest of the app makes
    for its other single-user JSON stores.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        yield
        return
    path = auth_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def load() -> dict[str, Any]:
    """The whole store, or an empty one. Never raises: an unreadable or corrupt
    file behaves like "no credentials" so a bad edit can't block startup — the same
    forgiving read ``alerts.load_alerts`` does.

    Returns the v2 shape — ``providers`` (every credential, keyed by provider) plus
    ``active`` (which provider each tier currently uses) — and a derived
    ``credentials`` view keyed by tier, which is what most callers want.
    """
    return _view(_read())


def _empty() -> dict[str, Any]:
    return {"version": _VERSION, "providers": {}, "active": {}}


def _anon_key(scope: str) -> str:
    """Storage key for a credential that names no provider (tests, hand-edits).
    Scoped so two anonymous credentials don't collide under one empty name."""
    return f"_scope:{scope}"


def _read() -> dict[str, Any]:
    """Normalized store, migrating the v1 layout on the way through.

    v1 keyed credentials by tier, so a tier could hold only one and signing in
    elsewhere destroyed the previous one. v2 keys them by provider and records
    which provider each tier points at, so several can be configured at once. The
    migration is read-time and lossless: an old file keeps working and is rewritten
    in the new shape the next time anything is stored.
    """
    path = auth_file()
    if not path.exists():
        return _empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _empty()
    if not isinstance(data, dict):
        return _empty()

    if isinstance(data.get("providers"), dict):
        providers = {k: v for k, v in data["providers"].items() if isinstance(v, dict)}
        active = data.get("active")
        active = {k: v for k, v in active.items() if isinstance(v, str)} if isinstance(active, dict) else {}
        return {"version": _VERSION, "providers": providers, "active": active}

    if isinstance(data.get("credentials"), dict):  # v1 → v2
        out = _empty()
        for scope, cred in data["credentials"].items():
            if not isinstance(cred, dict):
                continue
            key = cred.get("provider") or _anon_key(scope)
            out["providers"][key] = cred
            out["active"][scope] = key
        return out
    return _empty()


def _view(data: dict[str, Any]) -> dict[str, Any]:
    """Add the tier-keyed ``credentials`` view most callers read."""
    creds = {}
    for scope, name in data["active"].items():
        cred = data["providers"].get(name)
        if cred is not None:
            creds[scope] = cred
    return {**data, "credentials": creds}


def save(data: dict[str, Any]) -> None:
    """Write the store ``0600``, atomically.

    ``mkstemp`` in the destination directory gives a 0600 file to begin with, so
    the secret is never briefly world-readable the way ``write_text`` + ``chmod``
    would leave it; ``os.replace`` then swaps it in without a torn-write window.
    """
    path = auth_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # The tier-keyed view is derived; persisting it would let the two disagree.
    body = {k: v for k, v in data.items() if k != "credentials"}
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".auth-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(body, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get(scope: str = DEFAULT_SCOPE) -> dict[str, Any] | None:
    """The credential ``scope`` currently points at (no tier fallback)."""
    return load()["credentials"].get(scope)


def set_credential(scope: str, cred: dict[str, Any]) -> None:
    """Store ``cred`` and make ``scope`` use it.

    Stored under its provider name, so signing in to a second provider adds to the
    store rather than replacing what was there — switching back later needs no
    re-login.
    """
    with _locked():
        data = _read()
        key = cred.get("provider") or _anon_key(scope)
        data["providers"][key] = cred
        data["active"][scope] = key
        save(data)


def delete(scope: str = DEFAULT_SCOPE) -> bool:
    """Stop using ``scope``'s credential. True if it was pointing at one.

    The credential itself is kept so the tier can be pointed back at it; use
    ``forget`` to remove it from the store entirely.
    """
    with _locked():
        data = _read()
        existed = data["active"].pop(scope, None) is not None
        if existed:
            save(data)
        return existed


# --- multiple providers -------------------------------------------------------


def credentials() -> dict[str, dict[str, Any]]:
    """Every stored credential, keyed by provider name."""
    return _read()["providers"]


def providers() -> list[str]:
    """Configured provider names, anonymous ones excluded."""
    return sorted(n for n in _read()["providers"] if not n.startswith("_scope:"))


def active(scope: str = DEFAULT_SCOPE) -> str | None:
    """Which provider ``scope`` is using, following the ``default`` fallback."""
    data = _read()
    return data["active"].get(scope) or data["active"].get(DEFAULT_SCOPE)


def activate(provider: str, scope: str = DEFAULT_SCOPE) -> bool:
    """Point ``scope`` at an already-stored provider. False if it isn't stored."""
    with _locked():
        data = _read()
        if provider not in data["providers"]:
            return False
        data["active"][scope] = provider
        save(data)
        return True


def forget(provider: str) -> bool:
    """Remove a stored credential entirely, and any tier pointing at it."""
    with _locked():
        data = _read()
        if provider not in data["providers"]:
            return False
        del data["providers"][provider]
        data["active"] = {s: p for s, p in data["active"].items() if p != provider}
        save(data)
        return True


def resolve_value(raw: str) -> str | None:
    """Turn a stored key into the actual secret.

    ``$VAR`` reads the environment at use time (so rotating the env rotates the
    key), ``!cmd`` shells out when explicitly enabled, anything else is a literal.
    Returns None when the indirection resolves to nothing, so the caller falls
    through to the next source instead of sending an empty Authorization header.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.startswith("$"):
        return os.environ.get(raw[1:]) or None
    if raw.startswith("!"):
        if not _exec_allowed():
            return None
        try:
            out = subprocess.run(
                raw[1:], shell=True, capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() or None if out.returncode == 0 else None
    return raw


def effective_scope(scope: str = DEFAULT_SCOPE) -> str | None:
    """Which scope's credential ``scope`` actually uses, or None if there is none.

    Refreshing has to write back to the scope that really holds the credential,
    not the one that was asked for — otherwise a ``quick`` tier inheriting the
    primary credential would refresh it into a new ``quick`` entry and quietly
    stop inheriting.
    """
    creds = load()["credentials"]
    if scope in creds:
        return scope
    return DEFAULT_SCOPE if DEFAULT_SCOPE in creds else None


def resolve_key(scope: str = DEFAULT_SCOPE) -> str | None:
    """The API key for ``scope``, falling back to the ``default`` scope.

    OAuth credentials return their access token as-is; refreshing an expired one
    is the caller's job (it needs the provider registry, which this module
    deliberately doesn't import).
    """
    creds = load()["credentials"]
    cred = creds.get(scope) or creds.get(DEFAULT_SCOPE)
    if not cred:
        return None
    if cred.get("type") == "oauth":
        return cred.get("access") or None
    return resolve_value(cred.get("key", ""))


# Key shapes worth masking even when we don't hold the value ourselves — a tool
# result or a traceback can carry a key this process never stored.
_KEY_PATTERNS = (
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{12,}"),      # Anthropic (incl. sk-ant-oat01-)
    re.compile(r"\bsk-or-[A-Za-z0-9_\-]{12,}"),       # OpenRouter
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),          # OpenAI & compatible
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),      # GitHub
    # "Authorization: Bearer <tok>" as well as "x-api-key=<tok>". Two details:
    # the optional scheme word must be consumed too (else it gets masked and the
    # token doesn't), and the value stops at JSON punctuation — a greedy \S+ eats
    # the closing quote and braces, which would leave an exported .jsonl unparseable.
    re.compile(
        r"(?i)\b(bearer|x-api-key|authorization)([=:\"'\s]+)(?:bearer[\s\"']+)?[^\s\"',;}\])]+"
    ),
)
_MASK = "***"


#: (store mtime+size, the env vars we mask) -> the secrets to mask.
_Stamp = tuple[tuple[int, ...], tuple[tuple[str, str], ...]]
_secrets_cache: tuple[_Stamp, set[str]] | None = None


def _store_stamp() -> tuple[int, ...]:
    """Cheap identity for the store's current contents (mtime + size)."""
    try:
        st = auth_file().stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return ()


def _known_secrets() -> set[str]:
    """Literal secrets this process holds, cached against the store's mtime.

    ``redact`` runs on every traced tool call — twice, for arguments and result —
    so re-reading and re-parsing the store each time would put a file read on the
    hot path for a gain of nothing. The stamp means a fresh ``/login`` is still
    picked up on its next use.
    """
    global _secrets_cache
    stamp = (_store_stamp(), tuple(sorted(
        (v, os.environ.get(v) or "") for v in _ENV_SECRET_VARS
    )))
    if _secrets_cache is not None and _secrets_cache[0] == stamp:
        return _secrets_cache[1]
    found = _collect_secrets()
    _secrets_cache = (stamp, found)
    return found


_ENV_SECRET_VARS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY",
)


def _collect_secrets() -> set[str]:
    """Indirections (``$VAR`` / ``!cmd``) are resolved for masking too — a value
    we'd send is a value we must not print — except that ``!cmd`` is left alone,
    since running a shell command to redact a log line is a side effect no logger
    should cause."""
    out: set[str] = set()
    # Every stored credential, not just the active ones: an inactive provider's
    # key is still a secret sitting in this process's store.
    for cred in _read()["providers"].values():
        for field_name in ("access", "refresh"):
            val = cred.get(field_name)
            if isinstance(val, str) and val:
                out.add(val)
        raw = cred.get("key")
        if isinstance(raw, str) and raw and not raw.startswith("!"):
            resolved = resolve_value(raw)
            if resolved:
                out.add(resolved)
    for var in _ENV_SECRET_VARS:
        val = os.environ.get(var)
        if val:
            out.add(val)
    return {s for s in out if len(s) >= 8}  # a short "key" would mask real text


def redact(text: str) -> str:
    """Mask credentials in anything user-visible or persisted.

    Applied where a secret could otherwise be written down: trace span attributes,
    exported transcripts, and error text. Exact known values first, then shape
    patterns for keys this process never held.
    """
    if not text:
        return text
    for secret in sorted(_known_secrets(), key=len, reverse=True):
        text = text.replace(secret, _MASK)
    for pattern in _KEY_PATTERNS:
        text = pattern.sub(
            lambda m: (m.group(1) + m.group(2) + _MASK) if m.re.groups >= 2 else _MASK,
            text,
        )
    return text


def routing(scope: str = DEFAULT_SCOPE) -> tuple[str | None, str | None]:
    """``(model_provider, model)`` the stored credential is valid for.

    A key alone is not enough to build a client. An Anthropic token needs the
    ``anthropic`` integration and a Claude model name; handing it to ChatOpenAI
    with whatever ``OPENAI_MODEL`` happens to say produces a client that POSTs
    ``/v1/chat/completions`` at an API with no such route. So the credential
    carries its own routing and that routing travels with it.

    Falls back to what the provider declares, so a credential stored before these
    fields existed still routes correctly instead of silently doing nothing. The
    import is local: this module stays free of a load-time dependency on the OAuth
    registry, which imports it back.
    """
    creds = load()["credentials"]
    cred = creds.get(scope) or creds.get(DEFAULT_SCOPE) or {}
    if not cred:
        return None, None
    from . import oauth

    provider, models = oauth.routing_for(cred)
    return provider, (models[0] if models else None)


def model_config(model: str, scope: str = DEFAULT_SCOPE) -> dict[str, Any] | None:
    """The stored config for ``model`` — name plus any per-1M costs — or None if
    the credential for ``scope`` doesn't serve it."""
    creds = load()["credentials"]
    cred = creds.get(scope) or creds.get(DEFAULT_SCOPE) or {}
    if not cred:
        return None
    from . import oauth

    return oauth.model_config(cred, model)


def context_window(model: str = "", scope: str = DEFAULT_SCOPE) -> int | None:
    """The context window stored for ``scope``, in tokens, or None.

    When ``model`` is given it only applies if that model is one this credential
    serves — otherwise a window configured for a gateway would be claimed for an
    unrelated model that merely happened to be selected.
    """
    creds = load()["credentials"]
    cred = creds.get(scope) or creds.get(DEFAULT_SCOPE) or {}
    if not cred:
        return None
    from . import oauth

    if model:
        entry = oauth.model_config(cred, model)
        if entry is None:
            return None
    else:
        # No model named: the credential's default, i.e. its first model.
        entries = oauth.model_entries(cred)
        entry = entries[0] if entries else {}
    # The per-model window is the answer; the credential-wide one is only a
    # fallback for stores written before it moved into the model config.
    return entry.get("context_window") or oauth.context_window_of(cred)


def models(scope: str = DEFAULT_SCOPE) -> list[str]:
    """Every model the credential for ``scope`` offers; the first is its default.

    A credential holds a list because one key serves several models, and choosing
    between them should not need a re-login.
    """
    creds = load()["credentials"]
    cred = creds.get(scope) or creds.get(DEFAULT_SCOPE) or {}
    if not cred:
        return []
    from . import oauth

    return oauth.routing_for(cred)[1]


def base_url(scope: str = DEFAULT_SCOPE) -> str | None:
    """The endpoint stored alongside the key, if the provider pinned one.

    Logging in to a gateway has to carry its endpoint with it: a credential
    without one would be paired with whatever ``OPENAI_API_BASE`` happens to say,
    which is how you end up sending a gateway token to ``api.openai.com``.
    """
    creds = load()["credentials"]
    cred = creds.get(scope) or creds.get(DEFAULT_SCOPE) or {}
    return cred.get("base_url") or None


def describe(scope: str = DEFAULT_SCOPE) -> str:
    """One-line summary for ``/config`` — never includes the secret itself."""
    cred = get(scope)
    if not cred:
        return "none"
    provider = cred.get("provider", "?")
    if cred.get("type") == "oauth":
        exp = cred.get("expires")
        when = ""
        if isinstance(exp, (int, float)):
            import datetime

            when = " · expires " + datetime.datetime.fromtimestamp(
                exp / 1000
            ).strftime("%Y-%m-%d %H:%M")
        return f"{provider} (oauth){when}"
    return f"{provider} (api key)"
