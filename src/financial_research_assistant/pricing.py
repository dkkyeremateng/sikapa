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

Cached input tokens are billed at a fraction of the input rate
(``CACHE_DISCOUNT``); ``tokens_in`` already includes the cached tokens, so
``cost_usd`` discounts the cached portion rather than adding to it.
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


def _parse_pricing(raw: dict) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for k, v in (raw or {}).items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                out[str(k)] = (float(v[0]), float(v[1]))
            except (TypeError, ValueError):
                continue
    return out


def _parse_context(raw: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in (raw or {}).items():
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
    "claude-opus": (15.00, 75.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-haiku": (1.00, 5.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}

# Cached input tokens bill at this fraction of the input rate (OpenAI charges
# 0.25x for cached prompt reads on the gpt-4.1 family).
CACHE_DISCOUNT = 0.25


def context_cap(model: str) -> int:
    # An explicit window (e.g. for a custom gateway or an unlisted local model)
    # wins over the built-in table, so ctx% is right without editing code.
    override = os.environ.get("OPENAI_CONTEXT_WINDOW")
    if override:
        try:
            n = int(override)
            if n > 0:
                return n
        except ValueError:
            pass
    # Data-file overrides merge over the built-ins (file wins on a shared prefix),
    # so a new/updated context window needs no code edit. Longest prefix wins, so a
    # specific key ("gemini-2.5") beats a shorter one regardless of dict order.
    _, context_override = _load_overrides()
    table = {**MODEL_CONTEXT, **context_override}
    for name in sorted(table, key=len, reverse=True):
        if model.startswith(name):
            return table[name]
    return 128_000


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
    # Data-file overrides merge over the built-ins (file wins on a shared prefix),
    # so updating a price or adding a model needs no code edit.
    pricing_override, _ = _load_overrides()
    table = {**MODEL_PRICING, **pricing_override}
    for name in sorted(table, key=len, reverse=True):
        if model.startswith(name):
            return table[name]
    return None


def cost_usd(model: str, tok_in: int, tok_out: int, tok_cache: int = 0) -> float | None:
    """Turn cost in USD, discounting the cached portion of the input tokens.

    Returns None when the model has no known/override pricing.
    """
    r = rates(model)
    if r is None:
        return None
    cache = min(tok_cache, tok_in)  # cached tokens are a subset of input
    fresh_in = tok_in - cache
    return (
        fresh_in * r[0] + cache * r[0] * CACHE_DISCOUNT + tok_out * r[1]
    ) / 1_000_000
