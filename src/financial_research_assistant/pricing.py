"""Token pricing and context-window math — one source of truth.

The TUI status bar, the headless usage line, and the OpenTelemetry tracing
layer all read costs from here so a single price table stays consistent.

Prices are USD per 1M tokens (input, output); override either with
``OPENAI_INPUT_COST_PER_1M`` / ``OPENAI_OUTPUT_COST_PER_1M`` for a custom
gateway. Unknown models return ``None`` (no cost shown) rather than guessing.

To update prices or add models WITHOUT editing this code, drop a
``pricing.json`` data file (``FINANCIAL_RESEARCH_PRICING_FILE``, default
``~/.financial-research-assistant/pricing.json``) — see ``_load_overrides``; its
entries merge over the built-in tables. The per-model env vars above still win.

Cached input tokens are billed at their own rates — a cache READ at a fraction of
the input rate (``CACHE_DISCOUNT``), a cache WRITE at a premium on Anthropic
(``ANTHROPIC_CACHE_WRITE_MULTIPLIER``). Both counts are subsets of ``tokens_in``,
so ``cost_usd`` reprices those portions rather than adding to the total.
"""

from __future__ import annotations

from typing import Any
import json
import os
from pathlib import Path


def _pricing_file() -> Path:
    """Location of the optional pricing/context override file. Defaults to
    ``~/.financial-research-assistant/pricing.json``; override with
    ``FINANCIAL_RESEARCH_PRICING_FILE``."""
    raw = (os.environ.get("FINANCIAL_RESEARCH_PRICING_FILE") or "").strip()
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    return Path.home() / ".financial-research-assistant" / "pricing.json"


def _parse_pricing(raw: object) -> dict[str, tuple[float, float]]:
    """``{"model": [in, out]}`` → the rate table, dropping anything unusable.

    Typed ``object`` because the argument comes straight from a hand-edited JSON
    file: the key can be absent (None) or hold a string, a list, anything. The
    ``dict`` annotation this had was a promise the call site could not keep, and
    it hid a real crash — a non-dict section raised AttributeError out of
    ``_load_overrides``, which callers treat as always-succeeding.
    """
    out: dict[str, tuple[float, float]] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                out[str(k)] = (float(v[0]), float(v[1]))
            except (TypeError, ValueError):
                continue
    return out


def _parse_context(raw: object) -> dict[str, int]:
    """``{"model": tokens}`` → the context table, dropping anything unusable.
    Same tolerance as ``_parse_pricing``, and for the same reason."""
    out: dict[str, int] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n > 0:
            out[str(k)] = n
    return out


def _load_overrides() -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """User pricing/context overrides from the data file, so prices can be updated
    or a new model added WITHOUT editing this code. Returns ``(pricing, context)``
    keyed by model prefix (same prefix-match semantics as the built-in tables);
    file entries merge OVER the built-ins. Read fresh each call (the file is tiny
    and lookups aren't hot) so an edit takes effect without a restart. Shape::

        {"pricing": {"my-model": [1.0, 3.0]}, "context": {"my-model": 200000}}

    An absent or malformed file yields no overrides (built-in behavior)."""
    p = _pricing_file()
    try:
        if not p.exists():
            return {}, {}
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, {}
    if not isinstance(data, dict):
        return {}, {}
    return _parse_pricing(data.get("pricing")), _parse_context(data.get("context"))

# Known context windows (prefix match); everything else falls back to 128k.
# Non-OpenAI families are covered too so MODEL_PROVIDER=anthropic/google shows a
# sensible ctx%. (Claude's 1M-token beta isn't the default, so 200k is used.)
MODEL_CONTEXT: dict[str, int] = {
    "gpt-4.1": 1_047_576,
    "gpt-4o": 128_000,
    "o1": 200_000,
    "o3": 200_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5": 16_385,
    # Current Claude models carry a 1M window; older ones (and Haiku 4.5) stay at
    # 200k, which the bare "claude" fallback below covers. Longest prefix wins, so
    # these must be spelled out per model rather than collapsed to "claude-opus" —
    # Opus 4.5 and earlier are 200k, and would be wrong under a shared prefix.
    "claude-fable-5": 1_000_000,
    "claude-mythos-5": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude": 200_000,
    "gemini-1.5": 1_048_576,
    "gemini-2": 1_048_576,
    "llama-3": 128_000,
}

# USD per 1M tokens (input, output). Prefix match; longer keys are checked first
# so "claude-haiku" wins over a bare "claude" if one were added.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "o3-mini": (1.10, 4.40),
    "o1-mini": (1.10, 4.40),
    # The Opus tier repriced to $5/$25 with the 4.6 generation; the bare
    # "claude-opus" key below keeps the old $15/$75 for Opus 3, which is what it
    # was actually priced at. Longest prefix wins, so the specific keys take
    # precedence over it.
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-opus": (15.00, 75.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-haiku": (1.00, 5.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}

# Cached input tokens bill at this fraction of the input rate. The rate is
# provider-specific: OpenAI charges 0.25x for cached prompt reads on the gpt-4.1
# family, Anthropic ~0.1x for a cache read. Applying the OpenAI figure to an
# Anthropic run understates the saving from prompt caching by 2.5x, so
# `_cache_discount` picks per model. Kept as a module constant because it is the
# documented default and is referenced elsewhere.
CACHE_DISCOUNT = 0.25
ANTHROPIC_CACHE_DISCOUNT = 0.1


def _cache_discount(model: str) -> float:
    """Fraction of the input rate a cached token bills at, for ``model``."""
    return ANTHROPIC_CACHE_DISCOUNT if model.startswith("claude") else CACHE_DISCOUNT


# Anthropic bills a cache WRITE at 1.25x the input rate (5-minute TTL); OpenAI's
# compatible endpoints don't charge separately for one, so the write rate there is
# just the input rate. Seeds a credential's config at login AND backs `cost_usd`
# when the credential names no explicit write rate.
ANTHROPIC_CACHE_WRITE_MULTIPLIER = 1.25


def table_rates(model: str) -> tuple[float, float] | None:
    """The table's (input, output) per-1M rates, or None when nothing matches.

    Separate from ``rates`` because that one also consults the environment and the
    credential store; seeding a credential's config needs the table's own answer,
    not one derived from a credential that doesn't exist yet.
    """
    pricing_override, _ = _load_overrides()
    table = {**MODEL_PRICING, **pricing_override}
    for name in sorted(table, key=len, reverse=True):
        if model.startswith(name):
            return table[name]
    return None


def table_cache_rates(model: str, input_rate: float) -> tuple[float, float]:
    """(cache write, cache read) per-1M rates implied by the table for ``model``."""
    write = input_rate * ANTHROPIC_CACHE_WRITE_MULTIPLIER if model.startswith("claude") else input_rate
    return round(write, 6), round(input_rate * _cache_discount(model), 6)


def _stored_model_config(model: str) -> dict[str, Any] | None:
    """The active credential's config for ``model``, if it serves it."""
    try:
        from .auth import model_config

        return model_config(model)
    except Exception:  # a broken store must never take a turn down
        return None


def known_context(model: str) -> int | None:
    """The table's window for ``model``, or None when nothing matches.

    Distinct from ``context_cap``, which always answers — callers that need to
    know whether the answer is real (a login suggesting a default, say) cannot
    tell the 128k fallback from a genuine 128k entry.
    """
    _, context_override = _load_overrides()
    table = {**MODEL_CONTEXT, **context_override}
    for name in sorted(table, key=len, reverse=True):
        if model.startswith(name):
            return table[name]
    return None


def explain_context_cap(model: str) -> tuple[int, str]:
    """``(window, where it came from)`` — the same resolution ``context_cap`` does,
    with the winning source named.

    Exists because the window is resolvable from four places and a stale value in
    a losing one looks authoritative while doing nothing; "configured X, showing Y"
    is otherwise unanswerable without reading the code.
    """
    override = os.environ.get("OPENAI_CONTEXT_WINDOW")
    if override:
        try:
            n = int(override)
            if n > 0:
                return n, "env OPENAI_CONTEXT_WINDOW"
        except ValueError:
            pass
    stored = _stored_model_config(model)
    if stored and stored.get("context_window"):
        return int(stored["context_window"]), "auth.json model entry"
    known = known_context(model)
    if known:
        _, context_override = _load_overrides()
        where = "pricing.json" if any(
            model.startswith(k) for k in context_override
        ) else "built-in table"
        return known, where
    try:
        from .auth import context_window as _stored_window

        fallback = _stored_window(model)
    except Exception:
        fallback = None
    if fallback:
        return fallback, "auth.json credential-wide"
    return 128_000, "default fallback"


def context_cap(model: str) -> int:
    """The context window to divide by. See ``explain_context_cap`` for the
    resolution order and which source won."""
    return explain_context_cap(model)[0]


def context_pct(model: str, tokens_in: int) -> int:
    cap = context_cap(model)
    return min(100, round(tokens_in / cap * 100)) if cap else 0


def rates(model: str) -> tuple[float, float] | None:
    """(input, output) USD per 1M tokens, or None if the model is unknown.

    An env override (both vars set) wins over the built-in table so a custom
    gateway's pricing can be declared without editing code.
    """
    ci = os.environ.get("OPENAI_INPUT_COST_PER_1M")
    co = os.environ.get("OPENAI_OUTPUT_COST_PER_1M")
    if ci and co:
        try:
            return float(ci), float(co)
        except ValueError:
            pass
    # Rates recorded on the active credential for this exact model. Unlike the
    # context window — where the per-model table is authoritative and the store
    # only fills gaps — prices are the thing a gateway reseller changes, and the
    # user entered these deliberately for the model they are actually billed for.
    # Zero reads as "not set", not as free. The fields are written as 0 so the
    # shape to fill in is visible in auth.json, and a model whose price nobody has
    # entered must show no cost at all — a confident $0.00 on a gateway that bills
    # real money is worse than a blank.
    stored = _stored_model_config(model)
    if stored and stored.get("input_cost") and stored.get("output_cost"):
        return float(stored["input_cost"]), float(stored["output_cost"])
    # Data-file overrides merge over the built-ins (file wins on a shared prefix),
    # so updating a price or adding a model needs no code edit.
    return table_rates(model)


def cost_usd(
    model: str,
    tok_in: int,
    tok_out: int,
    tok_cache: int = 0,
    tok_cache_write: int = 0,
) -> float | None:
    """Turn cost in USD, pricing the cached portions of the input separately.

    ``tok_in`` is the total input; ``tok_cache`` (read from the cache) and
    ``tok_cache_write`` (written into it) are subsets of it, billed at their own
    rates — writing costs MORE than fresh input on Anthropic (~1.25x) while
    reading costs far less (~0.1x), so folding either into the input rate
    misreports a cache-heavy turn in both directions.

    Returns None when the model has no known/override pricing.
    """
    r = rates(model)
    if r is None:
        return None
    # Both cache figures are subsets of input, and a token is either read from the
    # cache or written to it — never both. Clamp so a provider reporting slightly
    # inconsistent counts can't drive fresh_in negative and refund the turn.
    cache = max(0, min(tok_cache, tok_in))
    write = max(0, min(tok_cache_write, tok_in - cache))
    fresh_in = tok_in - cache - write
    # Explicit rates on the credential beat the multipliers: a reseller's cache
    # pricing is not necessarily the upstream vendor's.
    stored = _stored_model_config(model) or {}
    read_rate = stored.get("cache_read_cost")
    read_rate = float(read_rate) if read_rate else r[0] * _cache_discount(model)
    write_rate = stored.get("cache_write_cost")
    write_rate = float(write_rate) if write_rate else table_cache_rates(model, r[0])[0]
    return (
        fresh_in * r[0] + cache * read_rate + write * write_rate + tok_out * r[1]
    ) / 1_000_000
