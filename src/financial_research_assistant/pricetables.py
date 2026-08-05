"""Published token prices and context windows: the static reference tables.

Separated from ``pricing.py`` because this half answers "what does the vendor
publish for this model", while that half answers "what will THIS install actually
be charged" — the distinction the existing ``table_rates`` vs ``rates`` and
``known_context`` vs ``context_cap`` pairs were already reaching for. Only the
second half consults the credential store, so keeping the tables here lets the
login flow read a model's defaults without ``oauth -> pricing -> auth -> oauth``
closing a loop.

Nothing here imports another module of this package. Keep it that way — that
property is the point.

Override or extend either table without editing code by dropping a
``pricing.json`` (``FINANCIAL_RESEARCH_PRICING_FILE``, default
``~/.financial-research-assistant/pricing.json``); see ``_load_overrides``.
"""

from __future__ import annotations

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
