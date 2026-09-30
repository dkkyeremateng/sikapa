"""The investor profile — what "recommend" has to fit.

Without it the recommender can only say what looks good in general. With it, an
idea is judged against the book it would join: the target mix by asset class,
the position and sector caps, the exclusions, the account type. The profile is
``profile.json`` under the state directory (JSON rather than TOML so it reads on
Python 3.10, which has no ``tomllib``), set up with ``--profile``, edited by
hand, or changed from chat ("set my max position to 8%").

Every field has a default, and ``is_set()`` says whether the user has actually
written one — the ideas sheet says "generic mode" in its subtitle when they
haven't, because ideas that fit nobody in particular must not pass for advice.
"""

from __future__ import annotations

from typing import Any
import json
from datetime import datetime

from . import hooks
from .storage import locked, read_json, state_file, write_private

#: The asset classes the target mix is written in, and what each covers.
ASSET_CLASSES = {
    "us_equity": "US stocks and US equity ETFs",
    "intl_equity": "developed-market stocks outside the US",
    "em_equity": "emerging-market stocks",
    "bonds": "government and investment-grade bonds",
    "tips": "inflation-protected bonds",
    "reits": "real estate (REITs)",
    "commodities": "gold and broad commodities",
    "crypto": "bitcoin and ether funds",
    "cash": "T-bills and cash equivalents",
}

RISK_LEVELS = ("conservative", "moderate", "aggressive")

#: A reasonable default mix per risk level, used only in generic mode.
_DEFAULT_TARGETS = {
    "conservative": {"us_equity": 35, "intl_equity": 10, "bonds": 35, "tips": 10, "cash": 10},
    "moderate": {"us_equity": 55, "intl_equity": 15, "em_equity": 5, "bonds": 15,
                 "reits": 5, "commodities": 5},
    "aggressive": {"us_equity": 70, "intl_equity": 15, "em_equity": 10, "reits": 5},
}

DEFAULTS: dict[str, Any] = {
    "risk": "moderate",
    "horizon_years": 10,
    "targets": {},              # {asset_class: percent}; empty -> the risk level's default
    "max_position_pct": 10.0,   # no single stock above this share of the book
    "max_sector_pct": 35.0,     # no sector above this share
    "exclude": [],              # tickers never to recommend
    "exclude_sectors": [],      # sectors never to recommend (substring match)
    "account_type": "taxable",  # taxable | tax-advantaged
    "base_currency": "USD",
    "markets": ["US"],
    "max_expense_ratio_pct": 0.5,
    "allow_crypto": False,
    "watchlist": [],
    "themes": [],
}


def profile_file():
    return state_file("profile.json", "FRA_PROFILE_FILE")


def is_set() -> bool:
    return profile_file().exists()


def load() -> dict[str, Any]:
    """The profile, every field present (defaults filled in)."""
    stored = read_json(profile_file(), {})
    out = {**DEFAULTS, **(stored if isinstance(stored, dict) else {})}
    if not out.get("targets"):
        out["targets"] = dict(_DEFAULT_TARGETS.get(str(out["risk"]), _DEFAULT_TARGETS["moderate"]))
    return out


def save(values: dict[str, Any]) -> dict[str, Any]:
    """Merge ``values`` into the stored profile, validated. Raises ValueError."""
    path = profile_file()
    with locked(path):
        stored = read_json(path, {})
        stored = stored if isinstance(stored, dict) else {}
        merged = {**stored, **values}
        _validate({**DEFAULTS, **merged})
        merged["updated"] = datetime.now().isoformat(timespec="seconds")
        write_private(path, json.dumps(merged, indent=2), prefix=".profile-")
    return load()


def _validate(p: dict[str, Any]) -> None:
    if p["risk"] not in RISK_LEVELS:
        raise ValueError(f"risk must be one of {', '.join(RISK_LEVELS)}")
    targets = p.get("targets") or {}
    unknown = [k for k in targets if k not in ASSET_CLASSES]
    if unknown:
        raise ValueError(f"unknown asset class(es): {', '.join(unknown)} "
                         f"(use {', '.join(ASSET_CLASSES)})")
    if targets and abs(sum(float(v) for v in targets.values()) - 100) > 0.5:
        raise ValueError(f"target mix adds up to {sum(float(v) for v in targets.values()):g}%, "
                         "not 100%")
    for key in ("max_position_pct", "max_sector_pct", "max_expense_ratio_pct"):
        if not 0 < float(p[key]) <= 100:
            raise ValueError(f"{key} must be between 0 and 100")
    if p["account_type"] not in ("taxable", "tax-advantaged"):
        raise ValueError("account_type must be taxable or tax-advantaged")


# --- editing from chat and the phone -----------------------------------------------------

_LIST_FIELDS = ("exclude", "exclude_sectors", "markets", "watchlist", "themes")
_NUMBER_FIELDS = ("horizon_years", "max_position_pct", "max_sector_pct", "max_expense_ratio_pct")


def _coerce(field: str, value: str) -> Any:
    text = (value or "").strip()
    if field in _NUMBER_FIELDS:
        return float(text.rstrip("%"))
    if field == "allow_crypto":
        return text.lower() in ("1", "yes", "true", "on", "allow")
    if field in _LIST_FIELDS:
        items = [v.strip() for v in text.replace(";", ",").split(",") if v.strip()]
        return [i.upper() for i in items] if field in ("exclude", "watchlist") else items
    if field == "targets":
        out: dict[str, float] = {}
        for part in text.replace(";", ",").split(","):
            if not part.strip():
                continue
            name, _, pct = part.partition("=") if "=" in part else part.partition(":")
            out[name.strip().lower()] = float(pct.strip().rstrip("%"))
        return out
    return text.lower() if field in ("risk", "account_type") else text


def update_investor_profile(setting: str, value: str) -> str:
    """Change one setting of the investor profile the recommendations are fitted
    to. Use when the user states a preference: 'set my max position to 8%',
    'never recommend tobacco', 'add NVDA to my watchlist', 'I'm aggressive'.

    ``setting`` is one of: risk (conservative/moderate/aggressive), horizon_years,
    targets ('us_equity=60, intl_equity=20, bonds=20' — must total 100),
    max_position_pct, max_sector_pct, exclude (tickers), exclude_sectors,
    account_type (taxable/tax-advantaged), max_expense_ratio_pct, allow_crypto
    (yes/no), watchlist (tickers), themes. List settings REPLACE the list; pass the
    whole list. Returns the updated profile."""
    field = (setting or "").strip().lower()
    if field not in DEFAULTS:
        return f"Unknown setting {setting!r}. Settings: {', '.join(DEFAULTS)}."
    try:
        updated = save({field: _coerce(field, value)})
    except (ValueError, TypeError) as exc:
        return f"Not saved: {exc}."
    return "Profile updated.\n" + describe(updated)


def add_to_watchlist(symbols: str) -> str:
    """Add tickers to the watchlist the event watchers and the recommender follow
    (comma-separated). Use for 'watch NVDA', 'keep an eye on these'."""
    current = load()["watchlist"]
    new = [s.strip().upper() for s in symbols.replace(";", ",").split(",") if s.strip()]
    merged = list(dict.fromkeys([*current, *new]))
    save({"watchlist": merged})
    return f"Watchlist: {', '.join(merged) or '(empty)'}."


def describe(p: dict[str, Any] | None = None) -> str:
    p = p or load()
    mode = "" if is_set() else " (generic — not set up yet: --profile or tell me your preferences)"
    mix = ", ".join(f"{k} {float(v):g}%" for k, v in sorted(p["targets"].items(),
                                                             key=lambda kv: -float(kv[1])))
    lines = [
        f"Investor profile{mode}",
        f"  risk {p['risk']} · horizon {float(p['horizon_years']):g} years · {p['account_type']}",
        f"  target mix: {mix}",
        f"  caps: {float(p['max_position_pct']):g}% per stock, {float(p['max_sector_pct']):g}% per sector",
        f"  ETFs: expense ratio ≤ {float(p['max_expense_ratio_pct']):g}% · crypto "
        + ("allowed" if p["allow_crypto"] else "excluded"),
    ]
    if p["exclude"] or p["exclude_sectors"]:
        lines.append(f"  never: {', '.join([*p['exclude'], *p['exclude_sectors']])}")
    if p["watchlist"]:
        lines.append(f"  watchlist: {', '.join(p['watchlist'])}")
    return "\n".join(lines)


async def _cmd_profile(_arg: str, _fake: bool) -> str:
    return describe()


hooks.register_command("profile", _cmd_profile, "the investor profile ideas are fitted to")

PROFILE_TOOLS = [update_investor_profile, add_to_watchlist]
