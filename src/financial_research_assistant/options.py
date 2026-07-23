"""Options explainer — plain-language single-leg option economics (Yahoo, keyless).

Retail options tools (Robinhood/Public style) turn a cryptic contract like
``AAPL 250117C00200000`` into something a human can reason about: what it costs,
where it breaks even, the most you can make or lose, how far the stock has to move,
and how much of the premium is time value that decays. This tool does exactly that
— and, in keeping with the project's grounded-figures ethos, *the tool* computes
every number (breakeven, max profit/loss, intrinsic vs. extrinsic value, the move
required, the IV-implied move); the model only narrates.

Data source: ``yfinance`` option chains (free, no API key) — expiries via
``Ticker.options`` and the calls/puts grid via ``Ticker.option_chain(expiry)``.
The two network reaches go through mockable ``_fetch_*`` helpers that return plain
Python and swallow failures, so a Yahoo outage degrades to a friendly message and
the payoff math is unit-tested fully offline. (When a live IBKR session is mounted
its option tools are also available; this keyless tool is the always-on baseline.)

Scope is a SINGLE leg (the overwhelmingly common retail case), framed from the
long side with the short side's max-loss noted. Multi-leg spreads are out of scope
for v1.
"""

from __future__ import annotations

import math
from datetime import date

_CONTRACT_MULTIPLIER = 100  # US equity options: 1 contract = 100 shares


# --- Small helpers (self-contained) ------------------------------------------
def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _usd(v) -> str:
    n = _num(v)
    return f"${n:,.2f}" if n is not None else "n/a"


def _pct(v) -> str:
    n = _num(v)
    return f"{n * 100:.1f}%" if n is not None else "n/a"


def _norm_type(option_type: str) -> str:
    t = (option_type or "call").strip().lower()
    if t in ("c", "call", "calls"):
        return "call"
    if t in ("p", "put", "puts"):
        return "put"
    return "call"


# --- Pure math (no network; unit-tested) -------------------------------------
def _mid(bid, ask, last):
    """Best single premium estimate: bid/ask midpoint when both quote, else last."""
    b, a, l = _num(bid), _num(ask), _num(last)
    if b and a and b > 0 and a > 0:
        return (b + a) / 2.0
    for v in (l, a, b):
        if v and v > 0:
            return v
    return None


def _moneyness(spot, strike, is_call) -> tuple[str, float | None]:
    """(label, percent in/out of the money vs spot). ITM if the option has
    intrinsic value; OTM if not; ATM within 0.5% of spot."""
    s, k = _num(spot), _num(strike)
    if s is None or k is None or s <= 0:
        return "unknown", None
    pct = (s - k) / s if is_call else (k - s) / s  # >0 = in the money
    if abs((s - k) / s) < 0.005:
        return "at the money", pct
    return ("in the money" if pct > 0 else "out of the money"), pct


def _intrinsic(spot, strike, is_call) -> float:
    s, k = _num(spot) or 0.0, _num(strike) or 0.0
    return max(0.0, s - k) if is_call else max(0.0, k - s)


def _long_economics(is_call, spot, strike, premium) -> dict:
    """Economics of BUYING one contract (per share and per contract)."""
    k, prem = _num(strike) or 0.0, _num(premium) or 0.0
    intrinsic = _intrinsic(spot, strike, is_call)
    breakeven = (k + prem) if is_call else (k - prem)
    s = _num(spot)
    be_move = ((breakeven - s) / s) if s and s > 0 else None
    # Long call profit is unbounded (None); long put is capped at strike→0.
    max_profit = None if is_call else max(0.0, (k - prem)) * _CONTRACT_MULTIPLIER
    return {
        "intrinsic": intrinsic,
        "extrinsic": max(0.0, prem - intrinsic),
        "breakeven": breakeven,
        "breakeven_move_pct": be_move,
        "cost": prem * _CONTRACT_MULTIPLIER,        # debit paid to open
        "max_loss": prem * _CONTRACT_MULTIPLIER,    # long option: premium at risk
        "max_profit": max_profit,                   # None = theoretically unlimited
    }


def _implied_move(spot, iv, dte_days):
    """One–standard-deviation dollar move implied by IV over the days to expiry."""
    s, v = _num(spot), _num(iv)
    if s is None or v is None or v <= 0 or dte_days is None or dte_days < 0:
        return None
    return s * v * math.sqrt(dte_days / 365.0)


def _dte(expiry: str, today: date | None = None) -> int | None:
    """Calendar days from today to an ``YYYY-MM-DD`` expiry, or None."""
    try:
        y, m, d = (int(x) for x in expiry.split("-"))
        return ((today or date.today()) - date(y, m, d)).days * -1
    except (ValueError, AttributeError):
        return None


# --- Data gather (network; lazy-imported so tests monkeypatch the source) -----
def _ticker(symbol: str):
    import yfinance as yf

    return yf.Ticker(symbol.strip().upper())


def _fetch_expiries(symbol: str) -> list[str]:
    """Available expiration dates (``YYYY-MM-DD``), soonest first; ``[]`` on
    failure or an optionless ticker."""
    try:
        return list(_ticker(symbol).options or [])
    except Exception:  # noqa: BLE001 — network/parse failure degrades to no-data
        return []


def _fetch_chain(symbol: str, expiry: str) -> dict:
    """Calls/puts for one expiry as ``{"calls": [...], "puts": [...]}`` of plain
    dicts (strike/lastPrice/bid/ask/impliedVolatility/volume/openInterest/
    inTheMoney/contractSymbol); ``{}`` on failure."""
    try:
        chain = _ticker(symbol).option_chain(expiry)
    except Exception:  # noqa: BLE001
        return {}
    cols = ["contractSymbol", "strike", "lastPrice", "bid", "ask",
            "impliedVolatility", "volume", "openInterest", "inTheMoney"]

    def rows(df):
        if df is None or getattr(df, "empty", True):
            return []
        out = []
        for _, r in df.iterrows():
            out.append({c: r.get(c) for c in cols})
        return out

    return {"calls": rows(getattr(chain, "calls", None)),
            "puts": rows(getattr(chain, "puts", None))}


def _spot(symbol: str):
    from . import fundamentals

    return fundamentals._price(fundamentals._fetch_info(symbol))


# --- Selection ---------------------------------------------------------------
def _nearest_expiry(expiries: list[str], want: str) -> tuple[str, bool]:
    """Return (chosen expiry, exact?). An exact match wins; else the soonest
    expiry on/after the requested date; else the soonest available."""
    if want and want in expiries:
        return want, True
    if want:
        later = sorted(e for e in expiries if e >= want)
        return (later[0] if later else min(expiries)), False
    return min(expiries), True


def _nearest_strike(rows: list[dict], target) -> dict | None:
    """The chain row whose strike is closest to ``target`` (spot if no target)."""
    valid = [r for r in rows if _num(r.get("strike")) is not None]
    if not valid:
        return None
    t = _num(target)
    if t is None:
        return None
    return min(valid, key=lambda r: abs(_num(r["strike"]) - t))


# --- Rendering ---------------------------------------------------------------
def _chain_summary(sym, expiry, exact, rows, otype, spot, dte) -> str:
    """A near-the-money slice of one side's chain, when no strike was specified."""
    atm = _nearest_strike(rows, spot)
    order = sorted(rows, key=lambda r: _num(r.get("strike")) or 0.0)
    if atm is not None:
        i = order.index(atm)
        order = order[max(0, i - 3): i + 4]
    lines = [f"{sym} {otype}s expiring {expiry}"
             + ("" if exact else " (nearest available to your date)")
             + f" — spot {_usd(spot)}, {dte if dte is not None else '?'} days to expiry",
             f"{'strike':>9}{'last':>9}{'bid':>9}{'ask':>9}{'IV':>8}{'vol':>7}{'OI':>8}"]
    for r in order:
        lines.append(
            f"{_num(r.get('strike')) or 0:>9,.1f}"
            f"{_num(r.get('lastPrice')) or 0:>9,.2f}"
            f"{_num(r.get('bid')) or 0:>9,.2f}"
            f"{_num(r.get('ask')) or 0:>9,.2f}"
            f"{_pct(r.get('impliedVolatility')):>8}"
            f"{int(_num(r.get('volume')) or 0):>7}"
            f"{int(_num(r.get('openInterest')) or 0):>8}")
    lines.append("")
    lines.append("Pass a `strike` (and `option_type` call/put) to explain one contract's "
                 "cost, breakeven, max profit/loss and time value. Available expiries can "
                 "be listed by re-calling with a different date.")
    return "\n".join(lines)


def _contract_report(sym, expiry, exact, row, otype, spot, dte) -> str:
    is_call = otype == "call"
    strike = _num(row.get("strike"))
    premium = _mid(row.get("bid"), row.get("ask"), row.get("lastPrice"))
    iv = _num(row.get("impliedVolatility"))
    label, m_pct = _moneyness(spot, strike, is_call)
    econ = _long_economics(is_call, spot, strike, premium)
    imove = _implied_move(spot, iv, dte)

    head = (f"{sym} {_usd(strike)} {otype.upper()} expiring {expiry}"
            + ("" if exact else " (nearest available to your date)"))
    lines = [head, ""]
    lines.append(f"  Underlying spot     : {_usd(spot)}")
    lines.append(f"  Days to expiry      : {dte if dte is not None else 'n/a'}")
    money_detail = ""
    if m_pct is not None and label != "at the money":
        side = "ITM" if m_pct > 0 else "OTM"
        money_detail = f" ({_pct(abs(m_pct))} {side})"
    lines.append(f"  Moneyness           : {label}{money_detail}")
    lines.append(f"  Premium (mid/last)  : {_usd(premium)} per share "
                 f"→ {_usd(econ['cost'])} per contract (×{_CONTRACT_MULTIPLIER})")
    lines.append(f"  Bid / ask / last    : {_usd(row.get('bid'))} / {_usd(row.get('ask'))}"
                 f" / {_usd(row.get('lastPrice'))}")
    lines.append(f"  Implied volatility  : {_pct(iv)}"
                 + (f"  (≈ ±{_usd(imove)} 1-sigma move by expiry)" if imove is not None else ""))
    lines.append(f"  Volume / open int.  : {int(_num(row.get('volume')) or 0):,}"
                 f" / {int(_num(row.get('openInterest')) or 0):,}")
    lines.append("")
    lines.append("  Intrinsic value     : " + _usd(econ["intrinsic"]) + " per share")
    lines.append("  Time (extrinsic)    : " + _usd(econ["extrinsic"])
                 + " per share  ← decays to $0 by expiry")
    lines.append("")
    lines.append("If you BUY (long) this contract:")
    lines.append(f"  Cost / max loss     : {_usd(econ['cost'])} (the premium — capped)")
    mp = econ["max_profit"]
    max_profit_txt = ("theoretically unlimited (stock can keep rising)" if mp is None
                      else f"{_usd(mp)} (if the stock falls to $0)")
    lines.append("  Max profit          : " + max_profit_txt)
    lines.append(f"  Breakeven at expiry : {_usd(econ['breakeven'])}"
                 + (f"  ({_pct(econ['breakeven_move_pct'])} move from spot)"
                    if econ["breakeven_move_pct"] is not None else ""))
    lines.append("")
    lines.append(f"If you SELL (write) it instead, you collect {_usd(econ['cost'])} up front; "
                 + ("a short call has theoretically unlimited loss if the stock rises."
                    if is_call else
                    f"a short put risks up to {_usd((_num(strike) or 0) * _CONTRACT_MULTIPLIER - econ['cost'])} "
                    "if the stock goes to $0."))
    lines.append("")
    lines.append("Single-leg estimate at expiry (ignores early exercise, commissions, and "
                 "dividends); premium is the bid/ask midpoint. Options can expire worthless — "
                 "the whole premium is at risk. Data: Yahoo option chain (may be delayed).")
    return "\n".join(lines)


# --- The tool ----------------------------------------------------------------
def explain_option(symbol: str, expiry: str = "", strike: float = 0.0,
                   option_type: str = "call") -> str:
    """Explain a stock option in plain language with the numbers worked out. Given
    a ``symbol`` and (optionally) an ``expiry`` (``YYYY-MM-DD``), ``strike``, and
    ``option_type`` ('call' or 'put'), it pulls the live Yahoo option chain and
    reports the premium and per-contract cost, bid/ask/last, implied volatility and
    the IV-implied move, volume/open interest, the split of the premium into
    intrinsic vs. time (extrinsic) value, moneyness, and — for buying the contract —
    the breakeven, the % move required to reach it, the max loss (the premium), and
    the max profit; plus the short-side max loss. The tool does all the math.

    If you omit ``strike`` it prints a near-the-money slice of the chain for that
    expiry so you can pick one; if you omit ``expiry`` it uses the nearest one. Use
    for 'explain this option / what's the breakeven / how much can I lose on the
    AAPL 200 call / is this call worth it / how much is time value'. It's a
    single-leg explainer (not multi-leg spreads) and an estimate at expiry, not
    advice. Keyless (Yahoo); quotes may be delayed."""
    sym = symbol.strip().upper()
    otype = _norm_type(option_type)
    expiries = _fetch_expiries(sym)
    if not expiries:
        return (f"No options data found for {sym} — it may not have listed options, "
                f"or Yahoo returned nothing. (Only optionable US-listed names work.)")

    chosen, exact = _nearest_expiry(expiries, expiry.strip())
    chain = _fetch_chain(sym, chosen)
    rows = chain.get("calls" if otype == "call" else "puts") or []
    if not rows:
        return f"No {otype} contracts found for {sym} expiring {chosen}."

    spot = _spot(sym)
    dte = _dte(chosen)

    if not _num(strike):
        return _chain_summary(sym, chosen, exact, rows, otype, spot, dte)

    row = _nearest_strike(rows, strike)
    if row is None:
        return f"No {otype} strike near {_usd(strike)} for {sym} expiring {chosen}."
    return _contract_report(sym, chosen, exact, row, otype, spot, dte)


OPTIONS_TOOLS = [explain_option]
