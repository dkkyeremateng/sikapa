# pyright: reportImportCycles=false
# catalog <-> subagents is intrinsic, not incidental: a subagent's tool pool IS
# the catalog minus the dispatch tools, so `subagents` has to ask this module what
# the catalog holds while this module collects `SUBAGENT_TOOLS` from it. The
# back-reference is inside a function body and runs long after import.
"""The agent's tool catalog: the single ``TOOLS`` list and its capability gate.

Split out of ``tools.py`` to break an import cycle. ``tools.py`` used to be both
the base-tool module AND the place every feature module's ``*_TOOLS`` list was
collected, so it imported ``fundamentals``, ``research``, ``subagents``, … at the
bottom while each of those imported helpers back from ``tools`` — a cycle that
only worked because the back-imports sat inside function bodies. Aggregation
lives here instead, so the dependency runs one way: feature modules -> tools, and
catalog -> both.

Anything that needs the assembled toolset (``graph.py``, ``subagents.py``)
imports it from here. ``tools.py`` deliberately does NOT re-export these names:
an alias there would recreate the cycle, and — worse — a test patching
``tools.TOOLS`` would silently not affect the list the graph actually builds
from.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .tools import (
    _chart_tool,
    allocation,
    cagr,
    compare_prices,
    convert_currency,
    current_date,
    export_data,
    import_ibkr_statement,
    income_summary,
    pct_change,
    portfolio_performance_chart,
    portfolio_period_return,
    portfolio_review_brief,
    portfolio_value_history,
    render_review,
    render_stock_report,
    stock_brief,
    portfolio_vs_benchmark,
    position_weight,
    price_history_chart,
    query_portfolio,
    query_transactions,
    realized_gains,
    risk_metrics,
    web_search,
)


# The local tools the agent always has. `think` is added separately (only when
# reasoning is enabled) — see graph.build_graph(think=...). IBKR market-data
# tools are appended per turn via ibkr_tools_session().
#
# The four chart tools go through `_chart_tool` so the model gets only their
# summary stats and the chart art travels to the UI as an artifact — same
# functions, same output for direct callers, ~300 fewer tokens per call in the
# model's context (and in every later step of the turn that replays it).
TOOLS: list[Callable[..., Any]] = [
    current_date,
    pct_change,
    cagr,
    position_weight,
    _chart_tool(price_history_chart),
    web_search,
    import_ibkr_statement,
    query_transactions,
    query_portfolio,
    _chart_tool(portfolio_value_history),
    _chart_tool(portfolio_performance_chart),
    realized_gains,
    income_summary,
    portfolio_period_return,
    render_review,
    portfolio_review_brief,
    render_stock_report,
    stock_brief,
    allocation,
    _chart_tool(compare_prices),
    portfolio_vs_benchmark,
    risk_metrics,
    export_data,
    convert_currency,
]

# yfinance-backed reference tools (fundamentals, analyst ratings, earnings
# calendar, dividend projection). Imported here so the graph picks them up from
# the single TOOLS list; fundamentals.py imports back from this module lazily
# (inside function bodies), so there is no import cycle.
from .fundamentals import FUNDAMENTALS_TOOLS  # noqa: E402
from .monitor import MONITOR_TOOLS  # noqa: E402
from .analytics import ANALYTICS_TOOLS  # noqa: E402
from .research import RESEARCH_TOOLS  # noqa: E402
from .factors import FACTOR_TOOLS  # noqa: E402
from .screener import SCREENER_TOOLS  # noqa: E402
from .subagents import SUBAGENT_TOOLS  # noqa: E402
from .edgar import EDGAR_TOOLS  # noqa: E402
from .valuation import VALUATION_TOOLS  # noqa: E402
from .options import OPTIONS_TOOLS  # noqa: E402
from .documents import DOCUMENT_TOOLS  # noqa: E402
from .alerts import ALERT_TOOLS  # noqa: E402
from .tasks import TASK_TOOLS  # noqa: E402
from .reports import REPORT_TOOLS  # noqa: E402
from .journal import JOURNAL_TOOLS  # noqa: E402
from .macro import MACRO_TOOLS  # noqa: E402
from .transcripts import TRANSCRIPT_TOOLS  # noqa: E402

# `.extend` rather than `+=`: both mutate in place, but `+=` reads as a rebind of
# an upper-case (i.e. constant) name to a type checker, which flags every line.
TOOLS.extend(FUNDAMENTALS_TOOLS)
TOOLS.extend(MONITOR_TOOLS)
TOOLS.extend(ANALYTICS_TOOLS)
TOOLS.extend(RESEARCH_TOOLS)
TOOLS.extend(FACTOR_TOOLS)
TOOLS.extend(SCREENER_TOOLS)
TOOLS.extend(SUBAGENT_TOOLS)
TOOLS.extend(EDGAR_TOOLS)
TOOLS.extend(VALUATION_TOOLS)
TOOLS.extend(OPTIONS_TOOLS)
TOOLS.extend(DOCUMENT_TOOLS)
TOOLS.extend(ALERT_TOOLS)
TOOLS.extend(TASK_TOOLS)
TOOLS.extend(REPORT_TOOLS)
TOOLS.extend(JOURNAL_TOOLS)
TOOLS.extend(MACRO_TOOLS)
TOOLS.extend(TRANSCRIPT_TOOLS)


# --- Capability gating -----------------------------------------------------
#
# Roughly a third of the tool schemas above describe tools that read a local
# store which is usually EMPTY: the imported-statement database, ingested
# documents, saved alert rules. Bound anyway, they cost their full schema on
# every model call in every step of every turn, and the only thing the model can
# do with them is call one and get back "nothing imported yet".
#
# So each such group is bound only once its store actually holds data. The
# bootstrap tool of each group (`import_ibkr_statement`, `ingest_document`,
# `add_alert`) stays bound unconditionally — that is how the store gets its first
# row, and how a group turns itself on. Because the real graph is rebuilt once
# per turn, importing a statement mid-conversation makes the whole statements
# group appear on the very next turn.
#
# The matching system-prompt sections are gated on the same capability set (see
# graph.py), so the model is never told about a tool it wasn't given.

def _has_statements() -> bool:
    """True when the statements DB holds at least one import. Opened read-only by
    URI so the probe neither creates the file nor runs the schema migration (both
    of which ``statements._connect`` would do)."""
    import sqlite3

    from . import statements

    path = statements.db_path()
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return conn.execute("SELECT 1 FROM imports LIMIT 1").fetchone() is not None
        finally:
            conn.close()
    except sqlite3.Error:  # missing table, locked, corrupt — treat as "no data"
        return False


def _has_documents() -> bool:
    from . import documents

    try:
        return bool(documents._load_index())
    except OSError:
        return False


def _has_alerts() -> bool:
    from . import alerts

    try:
        return bool(alerts.load_alerts())
    except OSError:
        return False


def _has_tasks() -> bool:
    from . import tasks

    try:
        return bool(tasks.load_tasks())
    except OSError:
        return False


def _has_journal() -> bool:
    from . import journal

    try:
        return bool(journal.load_entries())
    except OSError:
        return False


def _has_transcripts() -> bool:
    """Gated on the KEY, not on a store. Transcripts are the one source here with
    no keyless provider, so without a key the tool can only ever report that —
    which the prompt says better, and for free, than a bound tool schema does."""
    from . import transcripts

    return transcripts.configured()


_CAPABILITY_PROBES = {
    "statements": _has_statements,
    "documents": _has_documents,
    "alerts": _has_alerts,
    "tasks": _has_tasks,
    "journal": _has_journal,
    "transcripts": _has_transcripts,
}

# Tools dropped when their capability is absent. Deliberately excluded from these
# sets (i.e. always bound):
#   - `import_ibkr_statement` / `ingest_document` / `add_alert` — the bootstrap
#     tools that create the data each group needs.
#   - `factor_exposure` — takes a `symbol` and works fine with no portfolio; it
#     only falls back to holdings when the symbol is omitted.
_GATED_TOOLS: dict[str, frozenset[str]] = {
    # Every one of these reads the local statements store (NOT the live broker
    # session), so with no import they can only report that there's no data.
    "statements": frozenset({
        "query_transactions", "query_portfolio", "portfolio_value_history",
        "portfolio_performance_chart", "realized_gains", "income_summary",
        "allocation", "export_data", "portfolio_vs_benchmark", "tax_loss_harvest",
        "portfolio_risk", "portfolio_lookthrough", "dividend_projection",
        "portfolio_digest",
    }),
    "documents": frozenset({"ask_document", "list_documents", "forget_document"}),
    "alerts": frozenset({"list_alerts", "remove_alert"}),
    # `schedule_task` is the bootstrap tool and stays bound: it is the only way to
    # act after the turn ends, so gating it on "a task already exists" would mean
    # the agent could never create its first one.
    "tasks": frozenset({"list_scheduled_tasks", "cancel_scheduled_task"}),
    # `record_thesis` is the bootstrap tool, for the same reason. `review_theses`
    # reads a store that is empty until the first call is logged, and its whole
    # value is reading back a record that exists.
    "journal": frozenset({"review_theses"}),
    # No key, no provider — unlike every other group here there is no bootstrap
    # tool that could turn it on, so the whole thing is dropped and the prompt
    # explains the keyless 8-K route instead.
    "transcripts": frozenset({"earnings_call_transcript"}),
}


def tool_name(t: Any) -> str:
    """A tool's bound name, whether it's a StructuredTool or a plain function."""
    return getattr(t, "name", None) or getattr(t, "__name__", "")


def capabilities() -> frozenset[str]:
    """Which optional local data sources currently hold data. A probe that raises
    is treated as "absent" — a broken store must degrade to fewer tools, never
    break graph construction."""
    active = set()
    for cap, probe in _CAPABILITY_PROBES.items():
        try:
            if probe():
                active.add(cap)
        except Exception:  # noqa: BLE001 — a bad probe must not break the graph
            continue
    return frozenset(active)


def active_tools(caps: frozenset[str] | None = None, pool: list[Any] | None = None) -> list[Any]:
    """``TOOLS`` minus the groups whose backing store is empty. ``caps`` lets a
    caller reuse an already-computed capability set (graph.py computes it once and
    passes it to both the toolset and the prompt); ``pool`` narrows the source list
    (subagents pass their own reduced pool)."""
    caps = capabilities() if caps is None else caps
    drop: set[str] = set()
    for cap, names in _GATED_TOOLS.items():
        if cap not in caps:
            drop |= names
    return [t for t in (TOOLS if pool is None else pool) if tool_name(t) not in drop]
