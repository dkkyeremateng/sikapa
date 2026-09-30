"""Autonomous research ideas — stocks, ETFs, and the other asset classes through
their ETF sleeves — every one of them a scored call.

The pipeline is fixed steps with one model judgement in the middle (the choice
``research.py`` made: the orchestration decides what is gathered, the model only
judges it):

1. **Candidates, model-free.** Where the book is short of its target mix (the
   gap is filled with catalog ETFs); three stock screens over the S&P 500 or
   the large caps (momentum, quality, dividend); the holdings themselves
   (add / hold / trim); names the event watchers flagged; the watchlist. Then
   exclusions, a 30-day cooldown on anything already recommended, and a
   transparent score to rank them.
2. **Evidence, per candidate.** The stock or ETF brief, fundamentals, analyst
   view, risk figures, a news line — each item given an id.
3. **Judgement.** One model call per candidate argues both sides and returns a
   typed verdict: verdict, conviction 1-5, thesis, risks, horizon, the
   evidence it rests on, and what would prove it wrong. The schema has no field
   for a price or a return; the only number it may carry is an invalidation
   level, checked against the current price. Its text is held to the evidence
   the same way report commentary is held to the sheet.
4. **Fit, model-free.** Position and sector caps, overlap with what is already
   owned, cost, account type. What doesn't fit is dropped with the reason.
5. **The ledger.** Every non-neutral verdict is recorded in the journal
   (``source="recommender"``) against the benchmark of its asset class, so the
   existing tick scores it, and the monthly report shows whether conviction 5
   beats conviction 2.

Nothing here can place an order: no broker session is opened, and every tool an
unattended turn can bind is read-only (see ``test_recommend``). An idea is
research, delivered as research.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import asyncio
import json
import os
import re
import statistics
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

from . import hooks, jobs, periodic
from .storage import append_jsonl, read_jsonl, state_file

MODES = {"light": {"evaluate": 8, "deliver": 3}, "deep": {"evaluate": 15, "deliver": 8}}
COOLDOWN_DAYS = 30

#: verdict -> the journal's direction (None: not a call, not recorded)
_DIRECTION = {"buy-candidate": "bullish", "add": "bullish", "avoid": "bearish",
              "trim": "bearish", "watch": None, "hold": None}
NEW_VERDICTS = ("buy-candidate", "watch", "avoid")
HELD_VERDICTS = ("add", "hold", "trim")

_DISCLAIMER = ("Research idea, scored in the journal on {due}. Not an order and not "
               "personalised advice.")


@dataclass
class Candidate:
    symbol: str
    kind: str                 # stock | etf
    source: str               # gap | momentum | quality | dividend | holding | event | watchlist
    score: float              # 0-100, model-free
    reason: str               # why it is here, with its figures and windows
    held: bool = False
    asset_class: str = "us_equity"
    sleeve: str = ""
    benchmark: str = "SPY"
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass
class Idea:
    candidate: Candidate
    verdict: str
    conviction: int
    thesis: str
    key_risks: list[str]
    horizon_days: int
    evidence_ids: list[str]
    invalidation_price: float | None = None
    invalidation_event: str = ""
    evidence: list[tuple[str, str, str]] = field(default_factory=list)
    rank: int = 0
    journal_id: str = ""
    due: str = ""


def runs_file():
    return state_file("ideas-runs.jsonl", "FRA_IDEAS_RUNS")


def shadow() -> bool:
    return (os.environ.get("FRA_IDEAS_SHADOW") or "").strip().lower() in ("1", "true", "yes", "on")


# --- price figures (model-free) -----------------------------------------------------------


def price_facts(series: list[tuple[str, float]]) -> dict[str, Any] | None:
    """The figures a stock candidate is ranked and shown with, each with its window."""
    if len(series) < 210:
        return None
    closes = [c for _d, c in series]
    last_day, last = series[-1]
    six_back = series[-127] if len(series) >= 127 else series[0]
    year = series[-252:]
    high = max(c for _d, c in year)
    rets = [(closes[i] / closes[i - 1] - 1) for i in range(len(closes) - 126, len(closes))
            if closes[i - 1]]
    vol = statistics.pstdev(rets) * (252 ** 0.5) * 100 if len(rets) > 20 else None
    ma200 = sum(closes[-200:]) / 200
    return {
        "price": last, "price_date": last_day,
        "ret_6m_pct": (last / six_back[1] - 1) * 100, "ret_6m_from": six_back[0],
        "high_52w": high, "from_high_pct": (last / high - 1) * 100,
        "high_window": f"{year[0][0]} → {last_day}",
        "vol_6m_pct": vol, "ma200": ma200, "above_ma200": last > ma200,
    }


def _fmt_price_facts(pf: dict[str, Any]) -> str:
    return (f"{pf['ret_6m_pct']:+.1f}% over 6 months ({pf['ret_6m_from']} → {pf['price_date']}), "
            f"{pf['from_high_pct']:+.1f}% from its 52-week high, "
            + ("above" if pf["above_ma200"] else "below") + " its 200-day average")


# --- candidates -----------------------------------------------------------------------------


def _info(symbol: str) -> dict[str, Any]:
    from .fundamentals import _fetch_info  # pyright: ignore[reportPrivateUsage]

    return _fetch_info(symbol)


def _infos(symbols: list[str]) -> dict[str, dict[str, Any]]:
    syms = list(dict.fromkeys(symbols))
    if not syms:
        return {}
    with ThreadPoolExecutor(max_workers=min(8, len(syms))) as pool:
        return dict(zip(syms, pool.map(_info, syms)))


def _gap_candidates(book: dict[str, Any], prof: dict[str, Any]) -> list[Candidate]:
    """For each asset class under its target, the two cheapest large funds in it."""
    from . import etfs, universes

    out = []
    held_funds = {h["symbol"]: universes.fund(h["symbol"]) for h in book["holdings"]
                  if universes.fund(h["symbol"])}
    for cls, target, have in universes.gaps(book, prof["targets"]):
        if cls == "crypto" and not prof["allow_crypto"]:
            continue
        gap = target - have
        mine = [f for f in held_funds.values() if f and f["asset_class"] == cls]
        if mine:
            # Short of a class you already hold a fund for: the idea is to add to
            # that fund, not to buy a near-copy of it (BND held, AGG suggested).
            f = mine[0]
            out.append(Candidate(
                symbol=f["symbol"], kind="etf", source="gap", score=min(100.0, 45 + gap * 2),
                reason=(f"your book is {have:.1f}% {cls.replace('_', ' ')} against a {target:g}% "
                        f"target ({gap:.1f} points short), and you already hold {f['symbol']} "
                        f"for it"),
                held=True, asset_class=cls, sleeve=f["sleeve"], benchmark=f["benchmark"],
                facts={"gap_pct": gap, "target_pct": target, "have_pct": have},
            ))
            continue
        funds = [f for f in universes.funds_in(cls)
                 if not (f.get("taxable_only") and prof["account_type"] != "taxable")]
        infos = _infos([f["symbol"] for f in funds])
        ranked = []
        for f in funds:
            info = infos.get(f["symbol"]) or {}
            er = etfs.expense_ratio_pct(info)
            aum = info.get("totalAssets") or 0
            if er is not None and er > float(prof["max_expense_ratio_pct"]):
                continue
            ranked.append((er if er is not None else 9.9, -float(aum or 0), f, er))
        for er_sort, _aum, f, er in sorted(ranked, key=lambda r: (r[0], r[1]))[:2]:
            out.append(Candidate(
                symbol=f["symbol"], kind="etf", source="gap",
                score=min(100.0, 40 + gap * 2),
                reason=(f"your book is {have:.1f}% {cls.replace('_', ' ')} against a {target:g}% "
                        f"target ({gap:.1f} points short); {f['sleeve']}, expense ratio "
                        + (f"{er:.2f}%" if er is not None else "n/a")),
                asset_class=cls, sleeve=f["sleeve"], benchmark=f["benchmark"],
                facts={"gap_pct": gap, "target_pct": target, "have_pct": have,
                       "expense_ratio_pct": er},
            ))
    return out


def _momentum_candidates(universe: list[str], limit: int) -> list[Candidate]:
    """Strong, orderly trends: best 6-month return per unit of volatility, above
    the 200-day average and within 10% of the 52-week high."""
    series = periodic._fetch_many(universe, 400)  # pyright: ignore[reportPrivateUsage]
    rows = []
    for sym, s in series.items():
        pf = price_facts(s)
        if not pf or not pf["above_ma200"] or pf["from_high_pct"] < -10 or not pf["vol_6m_pct"]:
            continue
        rows.append((pf["ret_6m_pct"] / pf["vol_6m_pct"], sym, pf))
    rows.sort(reverse=True)
    out = []
    for i, (ratio, sym, pf) in enumerate(rows[:limit]):
        out.append(Candidate(
            symbol=sym, kind="stock", source="momentum", score=max(30.0, 80 - i * 4),
            reason=f"momentum screen: {_fmt_price_facts(pf)}; return/volatility {ratio:.2f}",
            facts={"price": pf},
        ))
    return out


def _fundamental_candidates(universe: list[str], limit: int) -> list[Candidate]:
    """Quality at a reasonable price, and dependable dividends — from Yahoo's
    fundamentals over the large-cap list (one lookup per name, so a bounded set)."""
    from .fundamentals import _num  # pyright: ignore[reportPrivateUsage]

    infos = _infos(universe)
    quality, dividend = [], []
    for sym, info in infos.items():
        fpe = _num(info.get("forwardPE"))
        margin = _num(info.get("profitMargins"))
        roe = _num(info.get("returnOnEquity"))
        growth = _num(info.get("revenueGrowth"))
        if fpe and 0 < fpe < 25 and margin and margin > 0.12 and roe and roe > 0.15 and (growth or 0) > 0:
            quality.append((roe / fpe, sym, fpe, margin, roe, growth))
        rate, price = _num(info.get("dividendRate")), _num(info.get("currentPrice"))
        payout = _num(info.get("payoutRatio"))
        if rate and price and payout is not None and 0 < payout < 0.7:
            yld = rate / price * 100
            if yld >= 2.5:
                dividend.append((yld, sym, payout))
    out = []
    for i, (_s, sym, fpe, margin, roe, growth) in enumerate(sorted(quality, reverse=True)[:limit]):
        out.append(Candidate(
            symbol=sym, kind="stock", source="quality", score=max(30.0, 75 - i * 4),
            reason=(f"quality screen (Yahoo, current): forward P/E {fpe:.1f}, net margin "
                    f"{margin * 100:.1f}%, return on equity {roe * 100:.1f}%, revenue growth "
                    f"{(growth or 0) * 100:+.1f}% year on year"),
        ))
    for i, (yld, sym, payout) in enumerate(sorted(dividend, reverse=True)[:limit]):
        out.append(Candidate(
            symbol=sym, kind="stock", source="dividend", score=max(30.0, 70 - i * 4),
            reason=(f"dividend screen (Yahoo, current): yield {yld:.2f}% on the current price, "
                    f"payout ratio {payout * 100:.0f}% of earnings"),
        ))
    return out


def _holding_candidates(book: dict[str, Any], prof: dict[str, Any], deep: bool) -> list[Candidate]:
    """The holdings, for add / hold / trim. Light mode only reviews the ones with
    something to say: over the position cap, or 20% off their high."""
    from . import universes

    series = periodic._fetch_many([h["symbol"] for h in book["holdings"]], 400)  # pyright: ignore[reportPrivateUsage]
    out = []
    for h in book["holdings"]:
        pf = price_facts(series.get(h["symbol"]) or [])
        over_cap = (universes.asset_class_of(h["symbol"]) == "us_equity"
                    and not universes.fund(h["symbol"])
                    and h["weight_pct"] > float(prof["max_position_pct"]))
        drawdown = pf is not None and pf["from_high_pct"] <= -20
        if not deep and not (over_cap or drawdown):
            continue
        why = [f"you hold it at {h['weight_pct']:.1f}% of the priced book"]
        if over_cap:
            why.append(f"above your {float(prof['max_position_pct']):g}% position cap")
        if pf:
            why.append(_fmt_price_facts(pf))
        f = universes.fund(h["symbol"])
        out.append(Candidate(
            symbol=h["symbol"], kind="etf" if f else "stock", source="holding",
            score=60 + (20 if over_cap else 0) + (10 if drawdown else 0),
            reason="; ".join(why), held=True,
            asset_class=universes.asset_class_of(h["symbol"]),
            sleeve=(f or {}).get("sleeve", ""), benchmark=(f or {}).get("benchmark", "SPY"),
            facts={"weight_pct": h["weight_pct"], "price": pf, "over_cap": over_cap},
        ))
    return out


def _event_candidates(held: set[str]) -> list[Candidate]:
    """Symbols the watchers flagged as high severity this week that aren't held
    (held ones are reviewed as holdings)."""
    from .storage import read_jsonl as _read

    since = (date.today() - timedelta(days=7)).isoformat()
    seen: dict[str, str] = {}
    for e in _read(state_file("events.jsonl", "FRA_EVENTS_LOG")):
        if (e.get("severity") == "high" and str(e.get("detected_at", ""))[:10] >= since
                and e.get("symbol") and e["symbol"] not in held):
            seen[e["symbol"]] = str(e.get("title"))
    return [Candidate(symbol=s, kind="stock", source="event", score=55,
                      reason=f"flagged this week: {t}") for s, t in seen.items()]


def _recently_recommended(days: int = COOLDOWN_DAYS) -> set[str]:
    from . import journal

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    return {str(e["symbol"]) for e in journal.load_entries()
            if e.get("source") == "recommender" and str(e.get("opened", "")) >= cutoff}


def candidates(book: dict[str, Any], prof: dict[str, Any], mode: str = "light",
               focus: str = "") -> tuple[list[Candidate], list[tuple[str, str]], str]:
    """``(kept, dropped, universe label)``: ranked candidates to evaluate, and
    every one filtered out with the reason."""
    from . import screener, universes

    deep = mode == "deep"
    held = {h["symbol"] for h in book["holdings"]}
    stocks, label = universes.stock_universe()
    cap = int(os.environ.get("FRA_IDEAS_UNIVERSE_MAX") or 520)
    pool: list[Candidate] = []
    pool += _gap_candidates(book, prof)
    pool += _momentum_candidates(stocks[:cap], 6 if deep else 4)
    pool += _fundamental_candidates(list(screener._DEFAULT_UNIVERSE), 4 if deep else 2)  # pyright: ignore[reportPrivateUsage]
    pool += _holding_candidates(book, prof, deep)
    pool += _event_candidates(held)
    pool += [Candidate(symbol=s, kind="etf" if universes.fund(s) else "stock", source="watchlist",
                       score=50, reason="on your watchlist")
             for s in prof.get("watchlist") or [] if s not in held]

    excluded = {s.upper() for s in prof.get("exclude") or []}
    cooling = _recently_recommended()
    dropped: list[tuple[str, str]] = []
    best: dict[str, Candidate] = {}
    for c in pool:
        sym = c.symbol.upper()
        why = ""
        if sym in excluded:
            why = "excluded in your profile"
        elif re.search(r"\.[A-Z]{1,3}$", sym):
            why = "not US-listed (markets: US)"
        elif (universes.asset_class_of(sym) == "crypto" and not prof["allow_crypto"]
              and not c.held):  # the rule stops new crypto ideas, not reviewing a held coin
            why = "crypto is excluded in your profile"
        elif sym in held and not c.held:
            why = "already held (reviewed as a holding instead)"
        elif sym in cooling and not c.held:
            why = f"recommended within the last {COOLDOWN_DAYS} days"
        elif focus and not _matches_focus(c, focus):
            why = f"outside the focus '{focus}'"
        if why:
            dropped.append((sym, why))
            continue
        prior = best.get(sym)
        if prior is None or c.score > prior.score:
            if prior is not None:
                c.reason = f"{c.reason}; also {prior.source}: {prior.reason}"
            best[sym] = c
    kept = sorted(best.values(), key=lambda c: c.score, reverse=True)
    return kept[: MODES[mode]["evaluate"]], dropped, label


def _matches_focus(c: Candidate, focus: str) -> bool:
    needle = focus.strip().lower()
    hay = " ".join([c.symbol, c.sleeve, c.asset_class, c.reason, c.source]).lower()
    if needle in hay:
        return True
    info = _info(c.symbol) if c.kind == "stock" else {}
    return needle in f"{info.get('sector', '')} {info.get('industry', '')}".lower()


# --- evidence -----------------------------------------------------------------------------------


def _cap(text: str, n: int = 1400) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def gather_evidence(c: Candidate, book: dict[str, Any]) -> list[tuple[str, str, str]]:
    """``[(id, label, text)]`` for one candidate. Each source is independent; one
    that fails is left out rather than failing the candidate."""
    from . import etfs, fundamentals, tools

    items: list[tuple[str, str]] = [("Why it is a candidate", c.reason)]

    def add(label: str, fn: Any) -> None:
        try:
            text = fn()
        except Exception as exc:  # noqa: BLE001
            text = f"(unavailable: {type(exc).__name__})"
        if text:
            items.append((label, _cap(str(text))))

    if c.kind == "etf":
        brief = etfs.build_etf_brief(c.symbol, book)
        c.facts["etf"] = brief["facts"]
        items.append(("Fund figures", "\n".join(brief["lines"])))
        add("Look-through", lambda: fundamentals.etf_exposure(c.symbol))
    else:
        add("Fundamentals", lambda: fundamentals.stock_fundamentals(c.symbol))
        add("Analysts", lambda: fundamentals.analyst_ratings(c.symbol))
        add("Risk (trailing year)", lambda: tools.risk_metrics(c.symbol, 365))

        def filings() -> str:
            from . import stocks

            b = stocks.build_stock_brief(c.symbol, quarters=4)
            return b["highlights"]

        add("Latest quarter (SEC filings)", filings)
    add("News", lambda: tools.web_search(f"{c.symbol} stock", max_results=3))
    return [(f"E{i + 1}", label, text) for i, (label, text) in enumerate(items)]


# --- judgement ------------------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are an investment analyst judging ONE idea for one investor. Argue the "
    "bull case and the bear case from the EVIDENCE only, then decide. Reply with a "
    "single JSON object and nothing else:\n"
    '{"verdict": one of VERDICTS, "conviction": 1-5, "thesis": "2-3 sentences, '
    'the reasoning", "key_risks": ["...", "..."], "horizon_days": 30-365, '
    '"evidence_ids": ["E1", ...], "invalidation": {"price": number or null, '
    '"event": "what would prove this wrong"}}\n'
    "Rules: use only figures that appear in the evidence (you may round them, never "
    "compute new ones); no price targets or expected returns; conviction 5 means "
    "the evidence is strong and consistent, 1 means it barely tips; the "
    "invalidation price, if any, is a level on the chart that would break the "
    "thesis (below the current price for a positive view, above it for a negative "
    "one)."
)


def _parse_verdict(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def validate_verdict(data: dict[str, Any], c: Candidate, evidence_ids: set[str]) -> tuple[dict[str, Any] | None, str]:
    """The typed verdict, or ``(None, why)``. Unknown keys are dropped, not kept:
    the only number the model may hand back is the invalidation level."""
    allowed = HELD_VERDICTS if c.held else NEW_VERDICTS
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in allowed:
        return None, f"verdict {verdict!r} is not one of {', '.join(allowed)}"
    try:
        conviction = int(data.get("conviction") or 0)
        horizon = int(data.get("horizon_days") or 90)
    except (TypeError, ValueError):
        return None, "conviction and horizon must be whole numbers"
    if not 1 <= conviction <= 5:
        return None, "conviction must be 1-5"
    thesis = str(data.get("thesis") or "").strip()
    if len(thesis) < 20:
        return None, "no thesis"
    risks = [str(r).strip() for r in (data.get("key_risks") or []) if str(r).strip()][:4]
    ids = [str(i) for i in (data.get("evidence_ids") or []) if str(i) in evidence_ids]
    if not ids:
        return None, "cites no evidence"
    inval = data.get("invalidation") or {}
    level = inval.get("price") if isinstance(inval, dict) else None
    price = ((c.facts.get("price") or {}).get("price")
             or (c.facts.get("etf") or {}).get("price"))
    try:
        level = float(level) if level is not None else None
    except (TypeError, ValueError):
        level = None
    if level is not None and price:
        positive = _DIRECTION.get(verdict) == "bullish"
        wrong_side = (positive and level >= price) or (not positive and level <= price)
        if wrong_side or abs(level / price - 1) > 0.5:
            level = None  # implausible: a level that is already broken, or far away
    return {
        "verdict": verdict, "conviction": conviction, "thesis": thesis, "key_risks": risks,
        "horizon_days": max(30, min(365, horizon)), "evidence_ids": ids,
        "invalidation_price": level,
        "invalidation_event": str((inval or {}).get("event") or "").strip()[:200]
        if isinstance(inval, dict) else "",
    }, ""


async def judge(c: Candidate, evidence: list[tuple[str, str, str]], fake: bool = False) -> tuple[dict[str, Any] | None, str]:
    """One model call per candidate; ``(verdict, "")`` or ``(None, why not)``."""
    from . import autonomy

    allowed = HELD_VERDICTS if c.held else NEW_VERDICTS
    system = _JUDGE_SYSTEM.replace("VERDICTS", " / ".join(allowed))
    body = "\n\n".join(f"[{i}] {label}\n{text}" for i, label, text in evidence)
    user = (f"IDEA: {c.symbol} ({c.kind}, {'held' if c.held else 'not held'}; "
            f"came from the {c.source} screen)\n\nEVIDENCE:\n{body}")
    fake_reply = json.dumps({
        "verdict": allowed[0], "conviction": 3,
        "thesis": f"{c.symbol} passed the {c.source} screen and the evidence is broadly supportive.",
        "key_risks": ["valuation", "market drawdown"], "horizon_days": 90,
        "evidence_ids": [evidence[0][0]], "invalidation": {"price": None, "event": "the screen reverses"},
    })
    ids = {i for i, _l, _t in evidence}
    sheet = periodic.Brief(kind="weekly", period="", label="", title=c.symbol, subtitle="",
                           highlights="", markdown=body, message=c.reason, facts=c.facts)
    last_why = "the model was unavailable"
    for attempt in range(2):
        prompt = user if attempt == 0 else (
            user + f"\n\nYour previous answer was rejected: {last_why}. Answer again.")
        text = await autonomy.ask(system, prompt, tier="default",
                                  purpose=f"ideas:{c.source}", fake=fake, fake_reply=fake_reply)
        if not text:
            return None, "the model was unavailable (budget, pause or provider)"
        data = _parse_verdict(text)
        if data is None:
            last_why = "it was not a JSON object"
            continue
        verdict, why = validate_verdict(data, c, ids)
        if verdict is None:
            last_why = why
            continue
        prose = " ".join([verdict["thesis"], *verdict["key_risks"], verdict["invalidation_event"]])
        bad = periodic.unsupported_figures(prose, sheet)
        if bad:
            last_why = f"it cited figures not in the evidence ({', '.join(bad[:4])})"
            continue
        return verdict, ""
    return None, last_why


# --- fit ------------------------------------------------------------------------------------------


def _sector_weights(book: dict[str, Any]) -> dict[str, float]:
    infos = _infos([h["symbol"] for h in book["holdings"]])
    out: dict[str, float] = {}
    for h in book["holdings"]:
        sector = str((infos.get(h["symbol"]) or {}).get("sector") or "")
        if sector:
            out[sector] = out.get(sector, 0.0) + h["weight_pct"]
    return out


def fit_check(idea: Idea, book: dict[str, Any], prof: dict[str, Any],
              sectors: dict[str, float]) -> str:
    """Why this idea doesn't fit the book, or ""."""
    c = idea.candidate
    direction = _DIRECTION.get(idea.verdict)
    if direction != "bullish":
        return ""  # a trim or an avoid never breaks a cap
    if c.kind == "stock":
        info = _info(c.symbol)
        sector = str(info.get("sector") or "")
        for banned in prof.get("exclude_sectors") or []:
            if banned and banned.lower() in f"{sector} {info.get('industry', '')}".lower():
                return f"in an excluded sector ({sector})"
        if sector and sectors.get(sector, 0.0) >= float(prof["max_sector_pct"]):
            return (f"{sector} is already {sectors[sector]:.1f}% of the book, at or over your "
                    f"{float(prof['max_sector_pct']):g}% sector cap")
        if c.held and (c.facts.get("weight_pct") or 0) >= float(prof["max_position_pct"]):
            return (f"already {c.facts['weight_pct']:.1f}% of the book, at or over your "
                    f"{float(prof['max_position_pct']):g}% position cap")
    else:
        etf = c.facts.get("etf") or {}
        er = etf.get("expense_ratio_pct")
        if er is not None and er > float(prof["max_expense_ratio_pct"]):
            return f"expense ratio {er:.2f}% is over your {float(prof['max_expense_ratio_pct']):g}% limit"
        ov = etf.get("overlap") or {}
        if not c.held and (ov.get("direct_pct", 0) >= 50 or max(ov.get("via_funds", {}).values(), default=0) >= 60):
            return "mostly what you already own (top-holdings overlap)"
    return ""


# --- the run -----------------------------------------------------------------------------------------


def _rank(ideas: list[Idea]) -> list[Idea]:
    def key(i: Idea) -> float:
        bonus = 10 if i.candidate.source == "gap" else 0
        return i.conviction * 15 + i.candidate.score * 0.25 + bonus

    return sorted(ideas, key=key, reverse=True)


async def run(mode: str = "light", fake: bool = False, focus: str = "",
              record: bool = True) -> dict[str, Any]:
    """One ideas run, end to end. Returns the run record (also appended to
    ``ideas-runs.jsonl``)."""
    from . import autonomy, journal, profile

    hooks.load_feature_modules()
    prof = profile.load()
    book = await asyncio.to_thread(periodic.load_book)
    kept, dropped, universe = await asyncio.to_thread(candidates, book, prof, mode, focus)
    run_id = f"r{datetime.now():%Y%m%d%H%M}-{uuid.uuid4().hex[:4]}"
    blocked = autonomy.blocked() if not fake else ""
    ideas: list[Idea] = []
    if not blocked:
        evidence = await asyncio.gather(*[asyncio.to_thread(gather_evidence, c, book) for c in kept])
        for c, ev in zip(kept, evidence):
            verdict, why = await judge(c, ev, fake=fake)
            if verdict is None:
                dropped.append((c.symbol, f"no verdict: {why}"))
                continue
            ideas.append(Idea(candidate=c, evidence=ev, **{
                k: verdict[k] for k in ("verdict", "conviction", "thesis", "key_risks",
                                        "horizon_days", "evidence_ids", "invalidation_price",
                                        "invalidation_event")}))
    sectors = await asyncio.to_thread(_sector_weights, book) if ideas else {}
    fitting: list[Idea] = []
    for idea in ideas:
        why = await asyncio.to_thread(fit_check, idea, book, prof, sectors)
        if why:
            dropped.append((idea.candidate.symbol, f"doesn't fit: {why}"))
        else:
            fitting.append(idea)
    ranked = _rank(fitting)
    delivered: list[Idea] = []
    filled: dict[str, str] = {}
    for idea in ranked:
        if idea.verdict in ("watch", "hold"):
            continue
        c = idea.candidate
        if c.source == "gap" and c.asset_class in filled:
            # One fund per gap: two near-identical funds are one idea, not two.
            dropped.append((c.symbol, f"same gap as {filled[c.asset_class]}, which ranks higher"))
            continue
        if c.source == "gap":
            filled[c.asset_class] = c.symbol
        delivered.append(idea)
    delivered = delivered[: MODES[mode]["deliver"]]
    watching = [i for i in ranked if i.verdict in ("watch", "hold")]
    for n, idea in enumerate(delivered, start=1):
        idea.rank = n
        direction = _DIRECTION.get(idea.verdict)
        if record and direction:
            try:
                entry = await asyncio.to_thread(
                    journal.record_call, idea.candidate.symbol, direction, idea.thesis,
                    idea.horizon_days, idea.candidate.benchmark, source="recommender",
                    run_id=run_id, conviction=idea.conviction, rank=n,
                    asset_class=idea.candidate.asset_class, preset=idea.candidate.source,
                    invalidation_price=idea.invalidation_price,
                    invalidation_event=idea.invalidation_event or None,
                    recommendation=idea.verdict, shadow=shadow() or None,
                )
                idea.journal_id, idea.due = entry["id"], entry["due"]
            except ValueError as exc:
                dropped.append((idea.candidate.symbol, f"not recorded: {exc}"))
    record_ = {
        "run_id": run_id, "at": datetime.now().isoformat(timespec="seconds"), "mode": mode,
        "focus": focus, "universe": universe, "generic": not profile.is_set(),
        "blocked": blocked, "shadow": shadow(),
        "ideas": [_idea_row(i) for i in delivered],
        "watching": [_idea_row(i) for i in watching],
        "dropped": dropped,
        "evaluated": [c.symbol for c in kept],
    }
    append_jsonl(runs_file(), record_)
    return record_


def _idea_row(i: Idea) -> dict[str, Any]:
    c = i.candidate
    return {
        "symbol": c.symbol, "kind": c.kind, "source": c.source, "asset_class": c.asset_class,
        "sleeve": c.sleeve, "benchmark": c.benchmark, "reason": c.reason, "held": c.held,
        "verdict": i.verdict, "conviction": i.conviction, "thesis": i.thesis,
        "key_risks": i.key_risks, "horizon_days": i.horizon_days,
        "invalidation_price": i.invalidation_price, "invalidation_event": i.invalidation_event,
        "evidence": [(eid, label) for eid, label, _t in i.evidence if eid in i.evidence_ids],
        "rank": i.rank, "journal_id": i.journal_id, "due": i.due,
        "facts": _card_facts(c),
    }


def _card_facts(c: Candidate) -> dict[str, Any]:
    pf = c.facts.get("price") or {}
    etf = c.facts.get("etf") or {}
    out: dict[str, Any] = {}
    if pf:
        out.update({k: pf.get(k) for k in ("price", "price_date", "ret_6m_pct", "ret_6m_from",
                                            "from_high_pct")})
    if etf:
        out.update({"price": etf.get("price"), "price_date": etf.get("price_date"),
                    "expense_ratio_pct": etf.get("expense_ratio_pct"),
                    "overlap_direct_pct": (etf.get("overlap") or {}).get("direct_pct"),
                    "correlation": (etf.get("correlation") or {}).get("corr")})
        one = (etf.get("returns") or {}).get("1y")
        if one:
            out.update({"ret_1y_pct": one["pct"], "ret_1y_window": f"{one['from']} → {one['to']}"})
    if "gap_pct" in c.facts:
        out["gap_pct"] = c.facts["gap_pct"]
    return out


# --- the sheet ----------------------------------------------------------------------------------------


def _card(row: dict[str, Any]) -> str:
    f = row["facts"]
    tiles = []
    if f.get("price") is not None:
        tiles.append(f"price {f['price']:,.2f} ({f.get('price_date')} close)")
    if f.get("ret_6m_pct") is not None:
        tiles.append(f"{f['ret_6m_pct']:+.1f}% since {f['ret_6m_from']}")
    if f.get("from_high_pct") is not None:
        tiles.append(f"{f['from_high_pct']:+.1f}% from the 52-week high")
    if f.get("ret_1y_pct") is not None:
        tiles.append(f"{f['ret_1y_pct']:+.1f}% total return {f['ret_1y_window']}")
    if f.get("expense_ratio_pct") is not None:
        tiles.append(f"expense ratio {f['expense_ratio_pct']:.2f}%")
    if f.get("overlap_direct_pct") is not None:
        tiles.append(f"{f['overlap_direct_pct']:.0f}% of its top holdings already owned")
    head = (f"## {row['rank']}. {row['symbol']} — {row['verdict']} "
            f"(conviction {row['conviction']}/5)")
    lines = [head, f"_{row['sleeve'] or row['asset_class'].replace('_', ' ')} · from the "
                   f"{row['source']} screen · scored against {row['benchmark']}_", ""]
    if tiles:
        lines.append("**Figures:** " + " · ".join(tiles))
    lines.append(f"**Why it came up:** {row['reason']}")
    lines.append(f"**Thesis:** {row['thesis']}")
    if row["key_risks"]:
        lines.append("**Risks:** " + "; ".join(row["key_risks"]))
    wrong = row["invalidation_event"] or ""
    if row["invalidation_price"]:
        wrong = (wrong + "; " if wrong else "") + f"a close through {row['invalidation_price']:,.2f}"
    if wrong:
        lines.append(f"**Would prove it wrong:** {wrong}")
    if row["evidence"]:
        lines.append("**Rests on:** " + ", ".join(f"{i} {label}" for i, label in row["evidence"]))
    if row["due"]:
        lines.append(f"_{_DISCLAIMER.format(due=row['due'])}_")
    return "\n".join(lines)


def open_ideas() -> list[dict[str, Any]]:
    """The recommender's calls still running, with the move since entry."""
    from . import journal

    rows = [e for e in journal.load_entries()
            if e.get("source") == "recommender" and e.get("status") == "open"]
    series = periodic._fetch_many([str(e["symbol"]) for e in rows], 10)  # pyright: ignore[reportPrivateUsage]
    for e in rows:
        s = series.get(str(e["symbol"])) or []
        if s and e.get("entry_price"):
            e["since_entry_pct"] = (s[-1][1] / float(e["entry_price"]) - 1) * 100
            e["priced_on"] = s[-1][0]
    return rows


def sheet(run_: dict[str, Any]) -> dict[str, str]:
    """``{title, subtitle, highlights, markdown, message}`` for one run."""
    ideas = run_["ideas"]
    mode = "Deep" if run_["mode"] == "deep" else "Weekly"
    title = f"{mode} Ideas — {run_['at'][:10]}"
    sub = [f"candidates from the {run_['universe']}, the fund catalog, your holdings and watchlist"]
    if run_["generic"]:
        sub.append("GENERIC MODE: no investor profile set — ideas fit no one in particular")
    if run_["shadow"]:
        sub.append("shadow run: recorded and scored, not sent")
    highlights = "\n".join(
        f"{r['symbol']} | {r['verdict']} {r['conviction']}/5 | {r['sleeve'] or r['source']}"
        for r in ideas[:6])
    brief = ["## In Brief\n" + "\n".join(
        f"- **{r['symbol']}** ({r['verdict']}, {r['conviction']}/5): {r['thesis'].split('. ')[0].rstrip('.')}."
        for r in ideas)] if ideas else []
    md = brief + [_card(r) for r in ideas] or [
        "## No ideas this run\n" + (f"Model judgement was skipped: {run_['blocked']}."
                                    if run_["blocked"] else
                                    "Nothing evaluated cleared the bar and fitted the book.")]
    if run_["watching"]:
        md.append("## Watching / holding\n" + "\n".join(
            f"- **{r['symbol']}:** {r['verdict']} ({r['conviction']}/5) — {r['thesis']}"
            for r in run_["watching"]))
    opened = open_ideas()
    if opened:
        md.append("## Open Ideas\n| Idea | Opened | Since entry | Scored |\n|---|---|---|---|")
        md += [f"| {e['symbol']} {e.get('recommendation', e['verdict'])} | "
               f"{str(e.get('opened'))[:10]} | "
               + (f"{e['since_entry_pct']:+.1f}%" if "since_entry_pct" in e else "n/a")
               + f" | {e.get('due')} |" for e in opened[:15]]
    if run_["dropped"]:
        # A table, not bullets: the cover summarises the first bullet lists it finds,
        # and "why UNH was dropped" is not an observation about the ideas.
        md.append("## Considered and Dropped\n| Name | Why not |\n|---|---|\n" + "\n".join(
            f"| {s} | {why.replace('|', '/')} |" for s, why in run_["dropped"][:25]))
    md.append("_Research ideas from delayed public data, each recorded in the journal and "
              "scored against its benchmark. Not orders, not personalised advice._")
    message = [f"💡 {mode} ideas · {run_['at'][:10]}"]
    message += [f"{r['rank']}. {r['symbol']} {r['verdict']} ({r['conviction']}/5): "
                f"{r['thesis'][:160]}" for r in ideas]
    if not ideas:
        message.append("No new ideas this run" + (f" ({run_['blocked']})" if run_["blocked"] else "."))
    if run_["generic"]:
        message.append("(Generic mode — set your profile with /profile or --profile.)")
    message.append("Full sheet attached. Research, not orders.")
    return {"title": title, "subtitle": " · ".join(sub), "highlights": highlights,
            "markdown": "\n\n".join(md), "message": "\n".join(message)}


def render_run(run_: dict[str, Any]) -> list[str]:
    from . import reports

    s = sheet(run_)
    try:
        paths = reports.render("", f"ideas-{run_['mode']}-{run_['at'][:10]}",
                               content={"title": s["title"], "subtitle": s["subtitle"],
                                        "highlights": s["highlights"], "markdown": s["markdown"],
                                        "eyebrow": "Research ideas"})
    except Exception:  # noqa: BLE001
        return []
    return [p for p in (paths.get("png"), paths.get("pdf")) if p]


def latest_run(mode: str = "") -> dict[str, Any] | None:
    runs = [r for r in read_jsonl(runs_file()) if not mode or r.get("mode") == mode]
    return runs[-1] if runs else None


# --- jobs, commands, the chat tool, report sections ---------------------------------------------------


def _job(mode: str) -> Any:
    async def job(task: dict[str, Any], fake: bool) -> jobs.JobResult:
        from . import guardrails

        paused = guardrails.paused_reason()
        if paused:
            return jobs.JobResult(True, f"ideas skipped: {paused}", notify=False)
        run_ = await run(mode, fake=fake)
        files = await asyncio.to_thread(render_run, run_)
        text = sheet(run_)["message"]
        # Shadow: recorded in the journal (so it is scored) and on disk, not sent.
        return jobs.JobResult(True, text, files=files, notify=not run_["shadow"])

    return job


jobs.register_job("ideas", _job("light"))
jobs.register_job("ideas-deep", _job("deep"))
jobs.register_default_job(jobs.DefaultJob(
    "ideas", "[job] Weekly ideas (light)", "sunday 18:00", "weekly"))
jobs.register_default_job(jobs.DefaultJob(
    "ideas-deep", "[job] Monthly ideas (deep)", "first weekday of the month 09:00",
    "monthly-first-weekday"))


async def recommend_ideas(focus: str = "", sleeve: str = "") -> str:
    """Research and recommend a few ideas NOW — stocks, ETFs, or another asset
    class through its ETF sleeve — fitted to the investor profile, each one
    recorded in the journal so it is scored later. Use for 'what should I look
    at / any ideas / recommend an ETF for bonds / what fills my international
    gap'. ``focus`` narrows it to a theme, sector or ticker ('semiconductors',
    'dividends'); ``sleeve`` to an asset class ('bonds', 'reits', 'intl_equity').
    Figures in the result are computed; present the ideas as research with their
    risks and invalidation, never as orders or personal advice."""
    run_ = await run("light", focus=(sleeve or focus or "").strip())
    s = sheet(run_)
    return s["message"].replace("Full sheet attached. ", "") + "\n\n" + s["markdown"][:6000]


def _make_tools() -> list[Any]:
    from langchain_core.tools import StructuredTool

    return [StructuredTool.from_function(coroutine=recommend_ideas, name="recommend_ideas")]


RECOMMEND_TOOLS = _make_tools()


async def _cmd_ideas(arg: str, fake: bool) -> str:
    if (arg or "").strip().lower() in ("now", "run", "new"):
        return await recommend_ideas()
    last = latest_run()
    if last is None:
        return "No ideas yet. /ideas now runs one."
    s = sheet(last)
    opened = open_ideas()
    tail = ""
    if opened:
        tail = "\nOpen ideas: " + ", ".join(
            f"{e['symbol']} {e['since_entry_pct']:+.1f}%" for e in opened if "since_entry_pct" in e)
    return s["message"].replace("Full sheet attached. ", "") + tail


hooks.register_command("ideas", _cmd_ideas, "the latest ideas and how open ones are doing: /ideas, /ideas now")


def _weekly_section(ctx: dict[str, Any]) -> str:
    last = latest_run("light")
    opened = open_ideas()
    if not last and not opened:
        return ""
    lines = ["## Ideas"]
    if last:
        lines.append(f"Latest run {last['at'][:10]}: " + (", ".join(
            f"{r['symbol']} {r['verdict']} ({r['conviction']}/5)" for r in last["ideas"])
            or "no new ideas"))
    for e in opened[:8]:
        if "since_entry_pct" in e:
            lines.append(f"- **{e['symbol']}** ({e.get('recommendation', e['verdict'])}, opened "
                         f"{str(e.get('opened'))[:10]}): {e['since_entry_pct']:+.1f}% since entry")
    return "\n".join(lines)


def _monthly_section(ctx: dict[str, Any]) -> str:
    last = latest_run("deep")
    if not last:
        return ""
    return ("## This Month's Deep Ideas\n" + "\n".join(
        f"- **{r['symbol']}** {r['verdict']} ({r['conviction']}/5): {r['thesis']}"
        for r in last["ideas"]) + "\n\n_The full ideas sheet was sent separately._")


periodic.register_section("weekly", _weekly_section)
periodic.register_section("monthly", _monthly_section)
