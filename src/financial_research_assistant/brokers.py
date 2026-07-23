"""Pluggable broker-provider registry for live market-data MCP servers.

The assistant loads a broker's read-only market-data tools from an MCP server per
turn (see ``tools.broker_tools_session``). This module makes *which* broker is
that simple, config-driven choice: each broker is one ``BrokerProvider`` entry
declaring

1. a **key** — the mount name (``"ibkr"``),
2. **config-from-env** — how to build its ``MultiServerMCPClient`` server spec
   from environment variables, returning ``None`` when the broker isn't
   configured (so a broker is *opt-in by the presence of its env vars*), and
3. a **read-only tool filter** — the safety boundary applied to that broker's
   loaded tools, so the model can read balances/positions/quotes but can never
   place, edit, or cancel an order.

Adding a broker means registering one provider here — no edits to ``graph.py``,
``adapter.py``, or the session lifecycle in ``tools.py``. Interactive Brokers is
registered below as the reference provider; its concrete policy still lives in
``tools.py`` (the security-critical, test-pinned code) and is reached here
through late-bound wrappers so operator monkeypatching and per-call env reads
keep working.

Selecting brokers at runtime:

- Unset ``BROKER_PROVIDERS`` -> every registered broker whose env resolves is
  mounted (today only IBKR's env is ever set).
- ``BROKER_PROVIDERS=ibkr`` (comma-separated) -> pin to an explicit subset, e.g.
  to run one broker at a time when several are configured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

# A loaded MCP tool exposes a ``.name``; a filter takes and returns a tool list.
ToolFilter = Callable[[list], list]


@dataclass(frozen=True)
class BrokerProvider:
    """One pluggable live-data broker.

    ``config_from_env`` returns a ``MultiServerMCPClient`` server spec (a dict) or
    ``None`` when the broker is not configured. ``filter_tools`` is the read-only
    boundary applied to that broker's loaded MCP tools.
    """

    key: str
    config_from_env: Callable[[], dict | None]
    filter_tools: ToolFilter


BROKER_REGISTRY: dict[str, BrokerProvider] = {}


def register_broker(provider: BrokerProvider) -> None:
    """Add (or replace) a provider in the registry, keyed by ``provider.key``."""
    BROKER_REGISTRY[provider.key] = provider


def registered_broker_keys() -> frozenset[str]:
    """Every registered broker key. Used to stop an ``EXTRA_MCP_SERVERS`` entry
    from shadowing a strictly-filtered broker mount."""
    return frozenset(BROKER_REGISTRY)


def active_broker_providers() -> list[BrokerProvider]:
    """Registered providers permitted this run. Unset ``BROKER_PROVIDERS`` -> all;
    otherwise the comma-separated subset (unknown keys ignored)."""
    sel = (os.environ.get("BROKER_PROVIDERS") or "").strip()
    if not sel:
        return list(BROKER_REGISTRY.values())
    keys = [k.strip() for k in sel.split(",") if k.strip()]
    return [BROKER_REGISTRY[k] for k in keys if k in BROKER_REGISTRY]


def configured_brokers() -> dict[str, tuple[dict, ToolFilter]]:
    """``{key: (server_spec, filter_fn)}`` for every active broker whose env
    resolves to a spec. Multiple brokers may mount at once; a broker with no env
    configured is simply absent."""
    out: dict[str, tuple[dict, ToolFilter]] = {}
    for p in active_broker_providers():
        spec = p.config_from_env()
        if spec is not None:
            out[p.key] = (spec, p.filter_tools)
    return out


def make_readonly_filter(
    read_prefixes: tuple[str, ...],
    *,
    write_deny: frozenset[str] = frozenset(),
    allow: frozenset[str] = frozenset(),
    extra_allow: Callable[[], frozenset[str]] = lambda: frozenset(),
) -> ToolFilter:
    """Build a strict read-only tool filter for a broker.

    A tool is kept iff its name is not in ``write_deny`` (the denylist always
    wins first, so no allow path can ever expose an order/alert mutation) AND it
    is either explicitly allowed (``allow`` or the per-call ``extra_allow()``
    opt-in set) or starts with one of ``read_prefixes``. This mirrors IBKR's
    policy in ``tools.py`` so a new broker gets the same trade-safe shape without
    reimplementing the invariant — pass the broker's own read verbs and the exact
    names of its order endpoints as ``write_deny``.
    """

    def _is_readonly(name: str, extra: frozenset[str]) -> bool:
        if name in write_deny:
            return False
        if name in allow or name in extra:
            return True
        return name.startswith(read_prefixes)

    def _filter(tools: list) -> list:
        extra = extra_allow()
        return [t for t in tools if _is_readonly(getattr(t, "name", ""), extra)]

    return _filter


# --- Registered providers --------------------------------------------------

# Interactive Brokers — the reference provider. Its concrete config + filter stay
# in tools.py (security-critical, test-pinned); reach them through the module
# object so a call-time ``monkeypatch.setattr(tools, ...)`` and the per-call
# ``IBKR_ALLOW_AUTHENTICATE`` read in ``filter_readonly`` are both honored. Never
# capture the function objects at import time.
def _ibkr_config_from_env() -> dict | None:
    from . import tools

    return tools._ibkr_server_config()


def _ibkr_filter(loaded: list) -> list:
    from . import tools

    return tools.filter_readonly(loaded)


register_broker(BrokerProvider("ibkr", _ibkr_config_from_env, _ibkr_filter))
