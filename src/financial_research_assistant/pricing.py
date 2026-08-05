"""Token pricing and context-window math — one source of truth.

The TUI status bar, the headless usage line, and the OpenTelemetry tracing
layer all read costs from here so a single price table stays consistent.

Prices are USD per 1M tokens (input, output); override either with
``OPENAI_INPUT_COST_PER_1M`` / ``OPENAI_OUTPUT_COST_PER_1M`` for a custom
gateway. Unknown models return ``None`` (no cost shown) rather than guessing.

The published tables themselves live in ``pricetables.py``; this module is the
half that answers what THIS install will be charged, so it also consults the
credential store and the environment. The pairs are deliberate: ``table_rates``
vs ``rates``, ``known_context`` vs ``context_cap`` — the first of each is the
vendor's published figure, the second is what actually applies here.

To update prices or add models WITHOUT editing code, drop a ``pricing.json``
(``FINANCIAL_RESEARCH_PRICING_FILE``, default
``~/.financial-research-assistant/pricing.json``) — see
``pricetables._load_overrides``; its entries merge over the built-in tables. The
per-model env vars above still win.

Cached input tokens are billed at their own rates — a cache READ at a fraction of
the input rate, a cache WRITE at a premium on Anthropic. Both counts are subsets
of ``tokens_in``, so ``cost_usd`` reprices those portions rather than adding to
the total.
"""

from __future__ import annotations

from typing import Any
import os


from .pricetables import (
    _cache_discount,
    _load_overrides,
    known_context,
    table_cache_rates,
    table_rates,
)


def _stored_model_config(model: str) -> dict[str, Any] | None:
    """The active credential's config for ``model``, if it serves it."""
    try:
        from .auth import model_config

        return model_config(model)
    except Exception:  # a broken store must never take a turn down
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
