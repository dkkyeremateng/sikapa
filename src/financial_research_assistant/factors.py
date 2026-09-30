"""Fama-French factor exposure — decompose a stock's or the portfolio's returns
into systematic style factors, a lens beyond single-factor beta and sector weights.

Regresses daily excess returns on the Fama-French factors from Ken French's Data
Library (free, keyless): market (Mkt-RF), size (SMB), value (HML), and optionally
profitability (RMW) and investment (CMA). The loadings say whether returns come
from a small/large-cap tilt, a value/growth tilt, etc.; the intercept is alpha
(return unexplained by the factors) and R² is how much of the variance the factors
explain.

The factor file download/parse goes through a mockable helper and is cached per
run, so tests exercise the regression offline with synthetic factors.
"""

from __future__ import annotations

from typing import Any
import io
import re
import time
import urllib.request
import zipfile
from datetime import date

# Ken French daily factor archives (keyless). The 5-factor set is a superset with
# RMW/CMA added; both carry Mkt-RF, SMB, HML, RF.
_FF_URLS = {
    False: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_Factors_daily_CSV.zip",
    True: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
}
_FF_UA = "Mozilla/5.0 (compatible; financial-research-assistant)"
_FF_CACHE: dict[bool, tuple[Any, ...]] = {}
_FF_AT: dict[bool, float] = {}
#: The library updates monthly; a day is plenty, and "for the life of the process"
#: was months for the always-on service.
_FF_TTL = 24 * 3600.0
_DATA_ROW = re.compile(r"^\s*(\d{8}),(.*)$")


def _parse_ff_csv(text: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Parse a Fama-French daily CSV into ``(factor_names, {date_iso: {name:
    decimal}})``. Factor names exclude RF (kept per-row for excess returns). Values
    in the file are percent; they're divided by 100 here."""
    lines = text.splitlines()
    header = next((ln for ln in lines if "Mkt-RF" in ln), "")
    cols = [c.strip() for c in header.split(",")[1:]]  # drop the leading date col
    if not cols:
        return [], {}
    by_date: dict[str, dict[str, Any]] = {}
    for ln in lines:
        m = _DATA_ROW.match(ln)
        if not m:
            continue
        ymd, rest = m.group(1), m.group(2)
        vals = [v.strip() for v in rest.split(",")]
        if len(vals) < len(cols):
            continue
        iso = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
        row = {}
        ok = True
        for name, v in zip(cols, vals):
            try:
                row[name] = float(v) / 100.0
            except ValueError:
                ok = False
                break
        if ok:
            by_date[iso] = row
    factor_names = [c for c in cols if c != "RF"]
    return factor_names, by_date


def _fetch_ff_factors(five_factor: bool = False) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Download + parse the Fama-French daily factors (cached per run). Returns
    ``([], {})`` on any failure so the caller shows a friendly message."""
    if five_factor in _FF_CACHE and time.time() - _FF_AT.get(five_factor, 0.0) < _FF_TTL:
        return _FF_CACHE[five_factor]
    try:
        req = urllib.request.Request(_FF_URLS[five_factor], headers={"User-Agent": _FF_UA})
        with urllib.request.urlopen(req, timeout=30.0) as resp:  # noqa: S310 (https)
            raw = resp.read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        member = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        text = zf.read(member).decode("utf-8", "replace") if member else ""
        parsed = _parse_ff_csv(text)
    except Exception:  # noqa: BLE001 — network/parse failure degrades to no-data
        parsed = ([], {})
    if parsed[1]:
        _FF_CACHE[five_factor] = parsed
        _FF_AT[five_factor] = time.time()
    elif five_factor in _FF_CACHE:
        return _FF_CACHE[five_factor]  # a failed refresh keeps the last good copy
    return parsed


def _ticker_returns(
    symbol: str, days: int, as_of: date | None = None
) -> list[tuple[str, float]]:
    """Daily ``(date, return)`` for one ticker (the return realized ON that date)."""
    from .tools import _fetch_daily

    series = _fetch_daily(symbol, days, as_of=as_of)
    out = []
    for i in range(1, len(series)):
        prev, (d, c) = series[i - 1][1], series[i]
        if prev:
            out.append((d, (c - prev) / prev))
    return out


def _portfolio_returns(
    days: int, account: str | None = None, as_of: date | None = None
) -> list[tuple[str, float]]:
    """Value-weighted daily portfolio returns from current holdings: each holding's
    daily return times its (base-currency value) weight, summed per day.

    ``as_of`` cuts the RETURN series, not the holdings — the weights are whatever
    is held now, because the statement store records positions as of its import,
    not a position history to reconstruct. So a dated portfolio regression answers
    "how would today's book have loaded on the factors back then", which is the
    same approximation `portfolio_risk` makes and states."""
    from . import statements
    from .fundamentals import _num
    from .tools import BASE_CURRENCY, _aligned_closes, _fx_rate

    positions = statements.query_positions(account=account)
    weights, total = {}, 0.0
    for p in positions:
        sym = (p.get("symbol") or "").upper()
        val = _num(p.get("value")) or 0.0
        r = _fx_rate((p.get("currency") or BASE_CURRENCY).upper()) or 1.0
        if sym and val > 0:
            weights[sym] = weights.get(sym, 0.0) + val * r
            total += val * r
    if not weights or total <= 0:
        return []
    syms = list(weights)
    dates, closes = _aligned_closes(syms, days, as_of=as_of)
    present = [s for s in syms if s in closes]
    if len(dates) < 2 or not present:
        return []
    wsum = sum(weights[s] for s in present) or 1.0
    out = []
    for i in range(1, len(dates)):
        r = 0.0
        for s in present:
            prev = closes[s][i - 1]
            if prev:
                r += (weights[s] / wsum) * (closes[s][i] - prev) / prev
        out.append((dates[i], r))
    return out


def _regress(returns: list[tuple[str, float]], five_factor: bool):
    """OLS of excess returns on the factors. Returns ``(alpha_daily, {factor:
    beta}, r2, n)`` or None if there's too little overlapping data."""
    import numpy as np

    names, factors = _fetch_ff_factors(five_factor)
    if not factors:
        return None
    xs, ys = [], []
    for d, ret in returns:
        f = factors.get(d)
        if f is None:
            continue
        xs.append([f[n] for n in names])
        ys.append(ret - f.get("RF", 0.0))
    if len(ys) < 30:
        return None
    x = np.array([[1.0, *row] for row in xs])
    y = np.array(ys)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    betas = {name: float(coef[i + 1]) for i, name in enumerate(names)}
    return float(coef[0]), betas, r2, len(ys)


def _tilt(beta: float, hi: str, lo: str) -> str:
    if beta > 0.1:
        return hi
    if beta < -0.1:
        return lo
    return "neutral"


def factor_exposure(symbol: str = "", days: int = 365, five_factor: bool = False,
                    as_of: str = "") -> str:
    """Fama-French factor exposure for a stock ``symbol`` (or your whole portfolio
    when ``symbol`` is empty), over the last ``days``. Regresses daily excess returns
    on the market, size (SMB), and value (HML) factors — plus profitability (RMW) and
    investment (CMA) when ``five_factor`` is set — reporting each loading, the
    annualized **alpha** (return not explained by the factors), and **R²** (variance
    explained). Reveals style tilts (small/large-cap, value/growth) that beta and
    sector weights don't. Use for 'factor exposure / style tilt / value or growth /
    is my alpha real / what drives my returns' questions. Data: Ken French Data
    Library (keyless). ``as_of`` (YYYY-MM-DD) ends the regression window at that
    date instead of today, for 'what was it loading on back then'."""
    from .pointintime import AsOfError, parse_as_of, window_note

    try:
        stamp = parse_as_of(as_of)
    except AsOfError as exc:
        return str(exc)
    subject = symbol.strip().upper() if symbol.strip() else "PORTFOLIO"
    returns = (
        _ticker_returns(subject, days, as_of=stamp) if symbol.strip()
        else _portfolio_returns(days, as_of=stamp)
    )
    if not returns:
        return (
            f"Not enough return data for {subject}"
            + (f" on or before {stamp.isoformat()}" if stamp else "")
            + ". "
            + ("Check the ticker." if symbol.strip()
               else "Import a portfolio with holdings first.")
        )
    res = _regress(returns, five_factor)
    if res is None:
        return (
            "Couldn't run the factor regression — either the Fama-French factor "
            "data was unavailable, or there wasn't enough overlapping history "
            "(need ~30+ common days). Try a longer window."
        )
    alpha_daily, betas, r2, n = res
    alpha_annual = alpha_daily * 252 * 100.0
    model = "5-factor" if five_factor else "3-factor"
    lines = [
        f"FAMA-FRENCH FACTOR EXPOSURE · {subject} · {model} · {n} days"
        f"{window_note(stamp, returns[-1][0])}:",
        f"  market (Mkt-RF)   {betas.get('Mkt-RF', 0.0):+.2f}",
        f"  size   (SMB)      {betas.get('SMB', 0.0):+.2f}  "
        f"({_tilt(betas.get('SMB', 0.0), 'small-cap tilt', 'large-cap tilt')})",
        f"  value  (HML)      {betas.get('HML', 0.0):+.2f}  "
        f"({_tilt(betas.get('HML', 0.0), 'value tilt', 'growth tilt')})",
    ]
    if five_factor:
        lines.append(
            f"  profit (RMW)      {betas.get('RMW', 0.0):+.2f}  "
            f"({_tilt(betas.get('RMW', 0.0), 'robust-profitability tilt', 'weak-profitability tilt')})"
        )
        lines.append(
            f"  invest (CMA)      {betas.get('CMA', 0.0):+.2f}  "
            f"({_tilt(betas.get('CMA', 0.0), 'conservative tilt', 'aggressive tilt')})"
        )
    lines.append(
        f"  alpha             {alpha_annual:+.1f}%/yr (return not explained by the factors)"
    )
    lines.append(f"  R²                {r2:.2f} (factors explain {r2 * 100:.0f}% of variance)")
    lines.append(
        "(Regression estimate over the window — loadings and alpha are noisy and "
        "not significance-tested; delayed data, not investment advice.)"
    )
    if stamp and not symbol.strip():
        # The returns are point-in-time; the WEIGHTS are not, and only the portfolio
        # branch has weights. Saying so is the difference between an approximation
        # and a misreading.
        lines.append(
            "note: the returns are as of the date above, but the weights are your "
            "CURRENT holdings — this is how today's book would have loaded back "
            "then, not what you actually held."
        )
    return "\n".join(lines)


FACTOR_TOOLS = [factor_exposure]
