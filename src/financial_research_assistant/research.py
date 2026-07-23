"""Deep-research report generator — plan → parallel retrieval → synthesis.

Rather than a single tool call, this gathers a *breadth* of findings on a ticker
across the existing data tools (price history, fundamentals, analyst ratings,
earnings, risk, ETF look-through, news) — running them concurrently — then hands
the labeled findings to the model to synthesize a structured, cited markdown
report, saved to disk.

It's implemented as a deterministic orchestration rather than a model-planned
StateGraph: an equity report always wants the same sections, so a fixed
parallel-gather is more reliable (and fully testable) than letting the model plan
retrieval, while still following the plan→retrieve→synthesize shape. Two surfaces
share the gather core: the ``research_report`` tool returns the raw findings for
the chat agent to write up inline (no nested model call), and ``--research SYMBOL``
runs the full gather + a standalone synthesis + save.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path

# Report sections, each a (label, zero-arg callable) built for one ticker. Every
# callable returns the tool's plain-text output (or its own "no data" message), so
# a missing source degrades to a labeled gap rather than failing the report.
def _section_fns(symbol: str) -> dict:
    from . import tools
    from . import fundamentals

    sym = symbol.strip().upper()
    return {
        "Price history (6mo)": lambda: tools.price_history_chart(sym, days=180),
        "Fundamentals": lambda: fundamentals.stock_fundamentals(sym),
        "Analyst ratings": lambda: fundamentals.analyst_ratings(sym),
        "Earnings": lambda: fundamentals.earnings_calendar(sym),
        "Risk metrics (1y)": lambda: tools.risk_metrics(sym, days=365),
        "ETF look-through": lambda: fundamentals.etf_exposure(sym),
        "Recent news": lambda: tools.web_search(f"{sym} stock news latest", max_results=5),
    }


def _portfolio_section_fns(account: str = "") -> dict:
    from . import tools
    from . import analytics, fundamentals, monitor

    acct = account or ""
    return {
        "Allocation & concentration": lambda: tools.allocation(acct),
        "Look-through exposure": lambda: analytics.portfolio_lookthrough(acct),
        "Performance vs benchmark (SPY)": lambda: tools.portfolio_vs_benchmark(account=acct),
        "Income summary": lambda: tools.income_summary(account=acct),
        "Forward dividends": lambda: fundamentals.dividend_projection(acct),
        "Realized gains": lambda: tools.realized_gains(account=acct),
        "Upcoming (movers/earnings/ex-div)": lambda: monitor.build_digest(account or None),
    }


async def _gather(fns: dict, fake: bool, fake_label: str) -> list[tuple[str, str]]:
    """Run a label→callable map concurrently (each blocking fetch in a thread). A
    section that raises degrades to an 'unavailable' note. ``fake`` returns
    deterministic stubs with no network, for offline runs/tests."""
    if fake:
        return [(label, f"[fake {label.lower()} for {fake_label}]") for label in fns]
    results = await asyncio.gather(
        *[asyncio.to_thread(fn) for fn in fns.values()], return_exceptions=True
    )
    return [
        (label, res if isinstance(res, str) else f"(unavailable: {res})")
        for label, res in zip(fns, results)
    ]


def gather_sections_sync(symbol: str) -> list[tuple[str, str]]:
    """Run every ticker section retrieval sequentially (for the sync tool path). A
    section that raises becomes a labeled error note rather than aborting the set."""
    out = []
    for label, fn in _section_fns(symbol).items():
        try:
            out.append((label, fn()))
        except Exception as e:  # noqa: BLE001 — one bad source shouldn't sink the report
            out.append((label, f"(unavailable: {type(e).__name__}: {e})"))
    return out


async def gather_sections(symbol: str, fake: bool = False) -> list[tuple[str, str]]:
    """Concurrently gather the per-ticker research sections."""
    return await _gather(_section_fns(symbol), fake, symbol.strip().upper())


async def gather_portfolio_sections(
    account: str = "", fake: bool = False
) -> list[tuple[str, str]]:
    """Concurrently gather the portfolio-level research sections."""
    return await _gather(_portfolio_section_fns(account), fake, "portfolio")


RESEARCH_SYSTEM_PROMPT = (
    "You are an equity research analyst. Write a structured markdown research "
    "report on the given ticker using ONLY the tool findings provided — never "
    "invent figures or facts not present in them. Organize it as: a one-paragraph "
    "**Summary**, then **Valuation & Fundamentals**, **Analyst View**, "
    "**Earnings**, **Price & Risk**, **News & Catalysts**, **Risks / What to "
    "watch**, and a **Bottom line**. Cite the source section for figures (e.g. "
    "'(Fundamentals)', '(Analyst ratings)') and state 'as of' dates where the "
    "findings give them. If a section's data was unavailable, say so briefly rather "
    "than guessing. End with a one-line disclaimer that this is delayed data for "
    "research only, not investment advice."
)


RESEARCH_PORTFOLIO_PROMPT = (
    "You are a portfolio analyst. Write a structured markdown report on the "
    "investor's whole portfolio using ONLY the tool findings provided — never "
    "invent figures. Organize it as: a one-paragraph **Summary**, then "
    "**Allocation & Concentration**, **True Look-through Exposure** (call out any "
    "hidden single-stock concentration across ETFs), **Performance vs Benchmark**, "
    "**Income & Dividends**, **Realized Gains / Tax**, **What's Coming** (earnings, "
    "ex-dividends, movers), **Risks / What to watch**, and a **Bottom line**. Cite "
    "the source section for figures and state 'as of' dates where given. If a "
    "section's data was unavailable (e.g. too few statements imported), say so "
    "rather than guessing. End with a one-line disclaimer that this is delayed data "
    "for research only, not investment advice."
)


async def synthesize_report(
    subject: str,
    sections: list[tuple[str, str]],
    model: str | None = None,
    fake: bool = False,
    system_prompt: str | None = None,
    title: str | None = None,
    lessons: list[str] | None = None,
) -> str:
    """Synthesize the gathered findings into a cited markdown report via the model.
    ``subject`` labels the input to the model; ``title`` names the report heading
    (defaults to ``subject``); ``system_prompt`` selects the report shape (ticker vs
    portfolio). ``lessons`` are prior self-critique notes to apply this time (from
    ``reflection.recall_lessons``). ``fake`` returns a deterministic stub (no
    model/network)."""
    title = title or subject.strip().upper()
    if fake:
        body = "\n".join(f"- {label}: {text}" for label, text in sections)
        return f"# Research report: {title}\n\nFAKE-OK synthesized from:\n{body}"
    from langchain_core.messages import HumanMessage, SystemMessage

    from .graph import _make_llm

    findings = "\n\n".join(f"## {label}\n{text}" for label, text in sections)
    lesson_block = ""
    if lessons:
        bullets = "\n".join(f"- {ln}" for ln in lessons)
        lesson_block = (
            "\n\nLessons from prior research to apply this time (address these "
            f"gaps or note them explicitly):\n{bullets}"
        )
    llm = _make_llm(model)
    resp = await llm.ainvoke([
        SystemMessage(content=system_prompt or RESEARCH_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Subject: {subject}\n\nTool findings:\n{findings}{lesson_block}\n\nWrite the report."
        ),
    ])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def reports_dir() -> Path:
    """Directory saved research reports land in."""
    raw = os.environ.get("FINANCIAL_RESEARCH_REPORTS_DIR")
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    return Path.home() / ".financial-research-assistant" / "reports"


def save_report(symbol: str, report: str) -> Path:
    """Write ``report`` markdown to a timestamped file under ``reports_dir()``."""
    d = reports_dir()
    d.mkdir(parents=True, exist_ok=True)
    sym = "".join(c for c in symbol.upper() if c.isalnum()) or "report"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = d / f"research-{sym}-{stamp}.md"
    dest.write_text(report, encoding="utf-8")
    return dest


async def research_ticker(
    symbol: str, model: str | None = None, fake: bool = False
) -> dict:
    """Full pipeline for one ticker: recall prior lessons, gather findings
    concurrently, synthesize a cited markdown report applying those lessons, save
    it, then reflect to learn new lessons for next time. Returns
    ``{symbol, report, path, sections, lessons}`` (``lessons`` = ones stored this
    run; empty when long-term memory is off)."""
    from .reflection import recall_lessons, reflect

    sym = symbol.strip().upper()
    prior = recall_lessons(sym)
    sections = await gather_sections(sym, fake=fake)
    report = await synthesize_report(sym, sections, model=model, fake=fake, lessons=prior)
    path = save_report(sym, report)
    learned = await reflect(sym, sections, report, model=model, fake=fake)
    return {
        "symbol": sym, "report": report, "path": str(path),
        "sections": sections, "lessons": learned,
    }


async def research_portfolio(
    account: str = "", model: str | None = None, fake: bool = False
) -> dict:
    """Full pipeline for the whole portfolio: recall prior lessons, gather
    portfolio-level findings (allocation, look-through exposure, benchmark, income,
    dividends, realized gains, upcoming events) concurrently, synthesize a cited
    markdown report applying those lessons, save it, then reflect. Returns
    ``{symbol, report, path, sections, lessons}`` (``symbol`` = "PORTFOLIO")."""
    from .reflection import recall_lessons, reflect

    prior = recall_lessons("PORTFOLIO")
    sections = await gather_portfolio_sections(account, fake=fake)
    report = await synthesize_report(
        "the investor's portfolio", sections, model=model, fake=fake,
        system_prompt=RESEARCH_PORTFOLIO_PROMPT, title="Portfolio", lessons=prior,
    )
    path = save_report("portfolio", report)
    learned = await reflect("PORTFOLIO", sections, report, model=model, fake=fake)
    return {
        "symbol": "PORTFOLIO", "report": report, "path": str(path),
        "sections": sections, "lessons": learned,
    }


def research_report(symbol: str) -> str:
    """Gather a broad set of research findings on a stock ``symbol`` — price history,
    fundamentals, analyst ratings, earnings, risk metrics, ETF look-through (if a
    fund), and recent news — and return them as labeled sections for you to
    synthesize into a structured, cited research report in your answer. Use when the
    user wants a deep dive / full write-up / research report on a ticker (rather than
    a single quick figure). Data is delayed Yahoo/web research data — cite sections
    and 'as of' dates, and note it's not investment advice."""
    from .reflection import recall_lessons

    sym = symbol.strip().upper()
    sections = gather_sections_sync(symbol)
    header = (
        f"Research findings for {sym} — synthesize these into a "
        f"structured, cited markdown report (Summary, Valuation & Fundamentals, "
        f"Analyst View, Earnings, Price & Risk, News & Catalysts, Risks, Bottom "
        f"line); cite each section and its 'as of' dates; don't invent figures.\n"
    )
    # Apply anything learned from prior research on this ticker (self-critique
    # lessons from earlier runs); empty when long-term memory is off.
    prior = recall_lessons(sym)
    if prior:
        header += (
            "\nLessons from prior research on this ticker — address these gaps or "
            "note them explicitly:\n" + "\n".join(f"- {ln}" for ln in prior) + "\n"
        )
    return header + "\n\n".join(f"## {label}\n{text}" for label, text in sections)


def _recent_move(symbol: str, days: int) -> str | None:
    """Deterministic price-move summary for the last ``days`` sessions from Yahoo
    daily closes: the window change (first→last), the most recent single-session
    change, and the closing level. ``None`` when no price data is available."""
    from . import tools

    sym = symbol.strip().upper()
    series = tools._fetch_daily(sym, max(days + 5, 15))  # a little context beyond the window
    if not series or len(series) < 2:
        return None
    window = series[-(days + 1):] if len(series) > days else series
    (first_d, first_c), (last_d, last_c) = window[0], window[-1]
    win_chg = (last_c - first_c) / first_c * 100.0 if first_c else 0.0
    prev_c = series[-2][1]
    day_chg = (last_c - prev_c) / prev_c * 100.0 if prev_c else 0.0
    return (
        f"{sym} last close {last_c:.2f} on {last_d} "
        f"({day_chg:+.2f}% vs prior session). "
        f"Over the window {first_d} → {last_d} ({len(window) - 1} session(s)): "
        f"{win_chg:+.2f}% (from {first_c:.2f})."
    )


def explain_stock_move(symbol: str, days: int = 5) -> str:
    """Gather the evidence to explain WHY a stock moved recently — the measured
    price move (last session + over the last ``days`` sessions, from Yahoo daily
    closes), recent analyst upgrades/downgrades, and recent news headlines — and
    return them as labeled sections for you to synthesize into a SHORT, attributed
    explanation in your answer. Use for 'why is TICKER up/down (today)?', 'what's
    moving X', 'what happened to X' questions. Attribute the move to specific news
    items (cite them by their number) and rating changes; if the evidence doesn't
    clearly explain the move, say so rather than inventing a reason. Delayed
    Yahoo/web data — cite sources and 'as of' dates; not investment advice."""
    from . import tools
    from . import fundamentals

    sym = symbol.strip().upper()
    days = max(1, min(int(days or 5), 30))
    move = _recent_move(sym, days)
    if move is None:
        return (
            f"No recent price data for {sym!r}, so there's no measured move to "
            f"explain. Check the ticker (US symbols, or Yahoo suffixes like VOD.L)."
        )
    changes = fundamentals._fetch_rating_changes(sym, limit=5)
    if changes:
        change_lines = "\n".join(
            f"  {c['date']}  {c['firm'] or '(firm n/a)'}: "
            f"{(c['from'] + ' → ' + c['to']) if (c['from'] or c['to']) else c['action']}"
            f"  ({c['action']})"
            for c in changes
        )
    else:
        change_lines = "  (no recent analyst rating changes found)"
    news = tools.web_search(f"{sym} stock news why moving today", max_results=5)
    header = (
        f"Explain the recent move in {sym}. Using ONLY the findings below, write a "
        f"SHORT paragraph attributing the move to specific drivers — cite news "
        f"items by their number [n] and name any rating changes. If the news and "
        f"rating changes do NOT clearly explain the move, say the move isn't "
        f"clearly explained by the available evidence (do not invent a catalyst). "
        f"State the 'as of' dates and that it's delayed data, not advice.\n"
    )
    return (
        f"{header}\n## Price move\n{move}\n\n"
        f"## Recent analyst rating changes\n{change_lines}\n\n"
        f"## Recent news\n{news}"
    )


def bull_bear_debate(symbol: str) -> str:
    """Gather research findings on a stock ``symbol`` (price, fundamentals, analyst
    ratings, earnings, risk, ETF look-through, news) and return them framed for an
    adversarial **bull-vs-bear debate** — for you to write a steel-manned Bull case,
    a steel-manned Bear case, and a Verdict (which side the evidence better
    supports, a lean with rough confidence, the open questions, and what would
    change the conclusion). Use when the user wants the case for AND against a stock
    — 'bull vs bear', 'should I buy X', 'make the case for and against X', 'is X a
    buy or a trap'. Cite each claim to its section; don't invent figures; delayed
    data, not investment advice."""
    from .reflection import recall_lessons

    sym = symbol.strip().upper()
    sections = gather_sections_sync(symbol)
    header = (
        f"Construct an adversarial bull-vs-bear analysis of {sym} using ONLY the "
        f"findings below. Write THREE parts:\n"
        f"1. **Bull case** — the strongest, steel-manned reasons to be positive, "
        f"each citing its source section.\n"
        f"2. **Bear case** — the strongest, steel-manned reasons to be cautious or "
        f"negative, each citing its source section.\n"
        f"3. **Verdict** — which case the evidence better supports, a lean "
        f"(bullish / neutral / bearish) with a rough confidence, the key open "
        f"questions, and what specific evidence would change the conclusion.\n"
        f"Do not invent figures beyond the findings; state 'as of' dates where "
        f"given; end with a one-line 'delayed data, research only, not advice' note.\n"
    )
    prior = recall_lessons(sym)
    if prior:
        header += (
            "\nLessons from prior research on this ticker — address these gaps or "
            "note them explicitly:\n" + "\n".join(f"- {ln}" for ln in prior) + "\n"
        )
    return header + "\n" + "\n\n".join(f"## {label}\n{text}" for label, text in sections)


RESEARCH_TOOLS = [research_report, explain_stock_move, bull_bear_debate]
