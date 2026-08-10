"""IBKR Flex Web Service sync — pull an Activity statement programmatically instead
of downloading the CSV by hand.

The Flex Web Service is a documented two-step, token-authenticated flow:

1. **SendRequest** — ``GET .../FlexWebService/SendRequest?t=<token>&q=<queryId>&v=3``
   returns a small ``<FlexStatementResponse>`` with a ``ReferenceCode`` and the
   ``GetStatement`` URL (or a ``Fail`` status).
2. **GetStatement** — ``GET <url>?t=<token>&q=<referenceCode>&v=3`` returns the
   statement XML (root ``<FlexQueryResponse>``), or a ``Warn`` "generation in
   progress" response to retry shortly.

This module implements that fetch (with the in-progress retry), saves the raw XML,
and maps it onto the same normalized dict the CSV importer produces — so Flex data
lands in the same store and is read by the same tools.

The mapping is not a rename. Flex uses its own field names (``cost`` for basis,
``ibCommission`` for commission, ``markPrice`` for the close), reports the cash
sections the CSV separates as one section discriminated by ``type``, and puts the
time-weighted return in ``ChangeInNAV`` rather than alongside NAV. With "Breakout by
Day" it also splits a single pull into a ``<FlexStatement>`` per business day, each
restating every position — so a naive read multiplies the portfolio by the number of
trading days. ``parse_flex_xml`` documents how each of those is resolved.

The XML is written to disk *before* it is parsed, so a statement this parser can't
read is kept as the sample needed to teach it.

Credentials: the **token** is read only from the ``IBKR_FLEX_TOKEN`` env var (a
secret — never passed through the model); the **query id** from
``IBKR_FLEX_QUERY_ID`` (or a CLI arg). Saved under
``~/.financial-research-assistant/flex`` (override ``FINANCIAL_RESEARCH_FLEX_DIR``).
"""

from __future__ import annotations

from typing import Any
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from .storage import write_private

_SEND_URL = (
    "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest"
)
_FLEX_VERSION = "3"
# IBKR error code returned by GetStatement while the statement is still being
# generated — the one condition we retry rather than fail on.
_IN_PROGRESS_CODE = "1019"


class FlexError(Exception):
    """A Flex Web Service request failed (bad token/query, IBKR error, or the
    statement never became ready)."""


def _flex_get(url: str, params: dict[str, Any], timeout: float) -> str:
    """HTTP GET returning decoded text. Errors never include the token."""
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": "financial-research-assistant"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https)
        return resp.read().decode("utf-8", "replace")


def _text(root: Any, tag: str) -> str:
    el = root.find(tag)
    return (el.text or "").strip() if el is not None and el.text else ""


def _parse_status(xml: str) -> dict[str, Any]:
    """Parse a ``<FlexStatementResponse>`` into
    ``{status, reference_code, url, error_code, error_message}`` (missing → "")."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise FlexError(f"unparseable Flex response: {e}") from e
    return {
        "status": _text(root, "Status"),
        "reference_code": _text(root, "ReferenceCode"),
        "url": _text(root, "Url"),
        "error_code": _text(root, "ErrorCode"),
        "error_message": _text(root, "ErrorMessage"),
    }


def _root_tag(xml: str) -> str:
    try:
        return ET.fromstring(xml).tag
    except ET.ParseError:
        return ""


def _is_in_progress(status: dict[str, Any]) -> bool:
    msg = status.get("error_message", "").lower()
    return (
        status.get("status") == "Warn"
        or status.get("error_code") == _IN_PROGRESS_CODE
        or "generation in progress" in msg
    )


def fetch_flex_xml(
    token: str,
    query_id: str,
    max_retries: int = 6,
    retry_delay: float = 5.0,
    timeout: float = 30.0,
) -> str:
    """Run the two-step Flex flow and return the statement XML string.

    Retries GetStatement up to ``max_retries`` times (``retry_delay`` s apart)
    while IBKR reports the statement is still generating. Raises ``FlexError`` on a
    missing credential, an IBKR error status, or a statement that never readies."""
    if not token:
        raise FlexError("no Flex token (set IBKR_FLEX_TOKEN)")
    if not query_id:
        raise FlexError("no Flex query id (set IBKR_FLEX_QUERY_ID)")

    sent = _parse_status(_flex_get(
        _SEND_URL, {"t": token, "q": query_id, "v": _FLEX_VERSION}, timeout
    ))
    if sent["status"] != "Success" or not sent["reference_code"] or not sent["url"]:
        raise FlexError(
            "SendRequest failed: "
            + (sent["error_message"] or sent["status"] or "unknown error")
        )

    for attempt in range(max_retries):
        data = _flex_get(
            sent["url"],
            {"t": token, "q": sent["reference_code"], "v": _FLEX_VERSION},
            timeout,
        )
        if _root_tag(data) != "FlexStatementResponse":
            return data  # the actual statement (root <FlexQueryResponse>)
        status = _parse_status(data)
        if _is_in_progress(status) and attempt < max_retries - 1:
            time.sleep(retry_delay)
            continue
        raise FlexError(
            "GetStatement failed: "
            + (status["error_message"] or status["status"] or "unknown error")
        )
    raise FlexError("statement not ready after retries — try again shortly")


def flex_dir() -> Path:
    """Directory saved Flex XML statements land in."""
    raw = os.environ.get("FINANCIAL_RESEARCH_FLEX_DIR")
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    return Path.home() / ".financial-research-assistant" / "flex"


def save_flex_xml(xml: str, query_id: str) -> Path:
    """Write the statement XML to a timestamped file under ``flex_dir()``.

    Written ``0600`` through ``storage.write_private``: a Flex statement is the
    most sensitive file this project keeps — every account number, position, and
    trade — so it gets the same treatment as the credential store rather than
    landing world-readable at the process umask.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_qid = "".join(c for c in query_id if c.isalnum()) or "query"
    dest = flex_dir() / f"flex-{safe_qid}-{stamp}.xml"
    write_private(dest, xml, prefix=".flex-")
    return dest


# --- XML -> the normalized statement dict ----------------------------------
#
# Flex names its fields differently from the Activity Statement CSV the manual
# importer reads (``cost`` not ``Basis``, ``ibCommission`` not ``Comm/Fee``,
# ``markPrice`` not ``Close Price``), so this module owns the translation and
# emits exactly the dict ``statements.store_statement`` consumes.

# CashTransaction/@type -> the ``kind`` the store uses. Keyed lowercase because
# the same concept is spelled differently across Flex versions ("Deposits/
# Withdrawals" vs "Deposits & Withdrawals"). An unmapped type is dropped rather
# than guessed at: a mis-binned row would silently distort income_summary.
_CASH_KINDS = {
    "dividends": "dividend",
    "payment in lieu of dividends": "dividend",
    "withholding tax": "withholding_tax",
    "871(m) withholding": "withholding_tax",
    "advisor fees": "fee",
    "broker fees": "fee",
    "other fees": "fee",
    "commission adjustments": "fee",
    "deposits/withdrawals": "deposit_withdrawal",
    "deposits & withdrawals": "deposit_withdrawal",
}

# EquitySummaryByReportDateInBase attribute -> the asset-class label the NAV
# table carries. Deliberately excludes ``total``: the store sums these rows to
# get account NAV, so carrying the broker's own total as a row would double it.
_NAV_CLASSES = (
    ("cash", "Cash"),
    ("stock", "Stocks"),
    ("options", "Options"),
    ("bonds", "Bonds"),
    ("commodities", "Commodities"),
    ("notes", "Notes"),
    ("dividendAccruals", "Dividend Accruals"),
    ("interestAccruals", "Interest Accruals"),
)


def _fnum(el: Any, *names: str) -> float:
    """First of ``names`` present on ``el`` as a float; 0.0 if absent or blank.

    Tolerant like the CSV importer's ``_to_float``: one unparseable cell must not
    abort a statement covering a year of activity."""
    for n in names:
        raw = (el.get(n) or "").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                return 0.0
    return 0.0


def _fstr(el: Any, *names: str) -> str:
    """First of ``names`` present on ``el`` as a stripped string, else ``""``."""
    for n in names:
        raw = (el.get(n) or "").strip()
        if raw:
            return raw
    return ""


def _flex_datetime(raw: str) -> str:
    """Normalize a Flex ``dateTime`` to the CSV importer's ``"YYYY-MM-DD, HH:MM:SS"``.

    Flex joins the two halves with whatever Date/Time Separator the query is
    configured for (a semicolon by default). Everything downstream slices the
    first ten characters, so only the date half has to be right — but emitting one
    shape means a row is indistinguishable whether it arrived by CSV or by Flex.
    """
    text = raw.strip()
    for sep in (";", " "):
        if sep in text:
            date_part, _, time_part = text.partition(sep)
            return f"{date_part.strip()}, {time_part.strip()}"
    return text


def _iso_period(start: str, end: str) -> str:
    """Render an ISO date range as the period string the store keys imports on.

    ``statements._period_bounds`` parses ``"%B %d, %Y"`` back out of this to date
    the NAV curve, so the format is load-bearing, not cosmetic."""
    try:
        s = datetime.strptime(start, "%Y-%m-%d").strftime("%B %d, %Y")
        e = datetime.strptime(end, "%Y-%m-%d").strftime("%B %d, %Y")
    except ValueError:
        return ""
    return f"{s} - {e}"


def _summary_only(elements: list[Any]) -> list[Any]:
    """Drop lot-level rows when a section carries both levels of detail.

    Selecting Open Positions at *both* Summary and Lot yields one SUMMARY row per
    holding plus one LOT row per tax lot. Both describe the same shares, so
    keeping both would double every position. When only one level is configured
    Flex omits the attribute entirely, which is why an absent ``levelOfDetail``
    counts as a keeper.
    """
    tagged = [e for e in elements if e.get("levelOfDetail")]
    if not tagged:
        return elements
    return [e for e in elements if (e.get("levelOfDetail") or "SUMMARY").upper() == "SUMMARY"]


def _stmt_account(stmt: Any) -> str:
    """The account a ``<FlexStatement>`` block is for."""
    return (stmt.get("accountId") or "").strip()


def _own_account(el: Any, statement_account: str) -> str:
    """The account a row belongs to, folding IBKR's segregated segments into the
    account the statement is for.

    Crypto executes in a Paxos segment reported as ``<accountId>-P``, and it never
    appears as a statement account of its own. Left alone, the holding shows up in
    the portfolio (positions are scoped by import, not by account) while the trade
    that opened it is filtered out of lot matching — so the coins are held with no
    cost basis and nothing says why. A genuinely different account, which shares no
    prefix, is left as it is.
    """
    raw = (el.get("accountId") or "").strip()
    if not raw:
        return statement_account
    if statement_account and raw.startswith(f"{statement_account}-"):
        return statement_account
    return raw


def _flex_trade(el: Any, statement_account: str = "") -> dict[str, Any]:
    """Map one Flex ``<Order>``/``<Trade>`` onto a normalized trade row."""
    qty = _fnum(el, "quantity")
    # Flex already signs quantity (a SELL arrives negative), but the sign is what
    # FIFO matching keys on, so it is re-derived from buySell rather than trusted:
    # an unsigned quantity would open a lot on a sale.
    if _fstr(el, "buySell").upper().startswith("SELL") and qty > 0:
        qty = -qty
    return {
        "asset_category": _fstr(el, "assetCategory"),
        "currency": _fstr(el, "currency"),
        "account": _own_account(el, statement_account),
        "symbol": _fstr(el, "symbol"),
        "datetime": _flex_datetime(_fstr(el, "dateTime") or _fstr(el, "tradeDate")),
        "quantity": qty,
        "trade_price": _fnum(el, "tradePrice"),
        "close_price": _fnum(el, "closePrice"),
        "proceeds": _fnum(el, "proceeds"),
        "comm_fee": _fnum(el, "ibCommission"),
        "basis": _fnum(el, "cost", "costBasis"),
        "realized_pl": _fnum(el, "fifoPnlRealized"),
        "mtm_pl": _fnum(el, "mtmPnl"),
        "code": _fstr(el, "notes", "code"),
    }


def _flex_cash(el: Any, statement_account: str = "") -> dict[str, Any] | None:
    """Map one ``<CashTransaction>`` onto a normalized cash row, or None when its
    type isn't one the store models."""
    kind = _CASH_KINDS.get(_fstr(el, "type").lower())
    if kind is None:
        return None
    return {
        "kind": kind,
        "currency": _fstr(el, "currency"),
        "account": _own_account(el, statement_account),
        "date": _flex_datetime(_fstr(el, "dateTime", "settleDate", "reportDate"))[:10],
        "description": _fstr(el, "description"),
        "amount": _fnum(el, "amount"),
        "code": _fstr(el, "code"),
    }


def _flex_position(el: Any) -> dict[str, Any]:
    """Map one ``<OpenPosition>`` onto a normalized position row."""
    return {
        "asset_category": _fstr(el, "assetCategory"),
        "currency": _fstr(el, "currency"),
        "symbol": _fstr(el, "symbol"),
        "quantity": _fnum(el, "position"),
        "mult": _fnum(el, "multiplier"),
        "cost_price": _fnum(el, "costBasisPrice"),
        "cost_basis": _fnum(el, "costBasisMoney"),
        "close_price": _fnum(el, "markPrice"),
        "value": _fnum(el, "positionValue"),
        "unrealized_pl": _fnum(el, "fifoPnlUnrealized"),
        "code": _fstr(el, "code"),
    }


def _flex_instrument(el: Any) -> dict[str, Any]:
    """Map one ``<SecurityInfo>`` onto a normalized instrument row."""
    return {
        "asset_category": _fstr(el, "assetCategory"),
        "symbol": _fstr(el, "symbol"),
        "description": _fstr(el, "description"),
        "conid": _fstr(el, "conid"),
        "security_id": _fstr(el, "isin", "securityID"),
        "underlying": _fstr(el, "underlyingSymbol"),
        "listing_exch": _fstr(el, "listingExchange"),
        "multiplier": _fstr(el, "multiplier"),
        "type": _fstr(el, "type", "subCategory"),
        "code": _fstr(el, "code"),
    }


def _flex_corporate_action(el: Any, statement_account: str = "") -> dict[str, Any]:
    """Map one ``<CorporateAction>`` onto a normalized corporate-action row.

    ``description`` is the load-bearing field: split detection reads the factor
    and the ticker straight out of its text."""
    return {
        "asset_category": _fstr(el, "assetCategory"),
        "currency": _fstr(el, "currency"),
        "account": _own_account(el, statement_account),
        "report_date": _flex_datetime(_fstr(el, "reportDate"))[:10],
        "datetime": _flex_datetime(_fstr(el, "dateTime")),
        "description": _fstr(el, "description"),
        "quantity": _fnum(el, "quantity"),
        "proceeds": _fnum(el, "proceeds"),
        "value": _fnum(el, "value"),
        "realized_pl": _fnum(el, "fifoPnlRealized", "realizedPnl"),
        "code": _fstr(el, "code"),
    }


def _chain_twr(statements: list[Any]) -> str:
    """Link every statement's TWR into one return for the whole span.

    With Breakout by Day the file holds one ``<ChangeInNAV>`` per business day,
    each carrying that day's time-weighted return as a percent. Returns compound,
    so they are chain-linked — summing them would understate a rising account and
    averaging them would ignore how many days each covers. Returns the CSV
    importer's percent-string shape, or ``""`` when no statement reports a TWR.
    """
    factor = 1.0
    seen = False
    for stmt in statements:
        for nav in stmt.iter("ChangeInNAV"):
            raw = (nav.get("twr") or "").strip()
            if not raw:
                continue
            try:
                factor *= 1 + float(raw) / 100
            except ValueError:
                continue
            seen = True
    return f"{(factor - 1) * 100}%" if seen else ""


def _nav_rows(statements: list[Any]) -> list[dict[str, Any]]:
    """Build per-asset-class NAV rows spanning every statement in the file.

    ``EquitySummaryByReportDateInBase`` is a dated snapshot, so the earliest one
    supplies each class's *prior* total and the latest its *current* — which is
    the shape the store keeps and what dates the two ends of the NAV curve.
    """
    snapshots = sorted(
        (e for s in statements for e in s.iter("EquitySummaryByReportDateInBase")),
        key=lambda e: e.get("reportDate") or "",
    )
    if not snapshots:
        return []
    first, last = snapshots[0], snapshots[-1]
    rows = []
    for attr, label in _NAV_CLASSES:
        prior = _fnum(first, attr)
        current = _fnum(last, attr)
        long_ = _fnum(last, f"{attr}Long")
        short = _fnum(last, f"{attr}Short")
        if not any((prior, current, long_, short)):
            continue  # a class the account has never held
        rows.append({
            "asset_class": label,
            "prior_total": prior,
            "current_long": long_ or current,
            "current_short": short,
            "current_total": current,
            "change": current - prior,
        })

    # Whatever the broker's own total exceeds these rows by is an asset class
    # this query didn't ask for — crypto is the common one, since the NAV section
    # offers no field for it yet folds it into `total`. Booking the remainder
    # keeps NAV equal to the broker's figure instead of quietly reporting a
    # smaller account, and it degrades gracefully as IBKR adds classes.
    prior_gap = _fnum(first, "total") - sum(r["prior_total"] for r in rows)
    current_gap = _fnum(last, "total") - sum(r["current_total"] for r in rows)
    if round(prior_gap, 2) or round(current_gap, 2):
        rows.append({
            "asset_class": "Other",
            "prior_total": prior_gap,
            "current_long": current_gap,
            "current_short": 0.0,
            "current_total": current_gap,
            "change": current_gap - prior_gap,
        })
    return rows


def parse_flex_xml(xml: str) -> dict[str, Any]:
    """Parse Flex XML into the internal statement dict ``store_statement`` consumes.

    One file becomes **one** import, even when Breakout by Day splits it into a
    ``<FlexStatement>`` per business day. Those daily statements repeat the whole
    position and security list every day, so:

    * trades, cash movements and corporate actions are **concatenated** — each
      day contributes only its own activity, so there is nothing to double;
    * positions and instruments come from the **last** statement alone, since
      only the closing snapshot describes what is held now;
    * NAV spans the whole file (earliest snapshot as prior, latest as current)
      and the daily TWRs are chain-linked into one figure for the period.

    Raises ``FlexError`` on XML that isn't a Flex statement, so a saved error
    page or a truncated download fails loudly instead of importing as empty.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise FlexError(f"unparseable Flex XML: {e}") from e
    if root.tag != "FlexQueryResponse":
        raise FlexError(
            f"not a Flex statement (root <{root.tag}>, expected <FlexQueryResponse>)"
        )

    statements = sorted(
        root.iter("FlexStatement"),
        key=lambda s: (s.get("toDate") or "", s.get("fromDate") or ""),
    )
    if not statements:
        raise FlexError("Flex statement contains no <FlexStatement> — check the query's period")
    last = statements[-1]

    trades = [
        _flex_trade(el, _stmt_account(stmt))
        for stmt in statements
        for tag in ("Order", "Trade")
        for el in _summary_only(list(stmt.iter(tag)))
    ]
    cash = [
        row
        for stmt in statements
        for el in stmt.iter("CashTransaction")
        if (row := _flex_cash(el, _stmt_account(stmt))) is not None
    ]
    corporate_actions = [
        _flex_corporate_action(el, _stmt_account(stmt))
        for stmt in statements
        for el in stmt.iter("CorporateAction")
    ]
    positions = [_flex_position(el) for el in _summary_only(list(last.iter("OpenPosition")))]

    # Instruments are a lookup table, not activity: the closing statement's copy
    # is the current one, and a symbol repeated across daily statements would
    # otherwise fan out the positions join into duplicate rows.
    instruments: dict[str, dict[str, Any]] = {}
    for el in last.iter("SecurityInfo"):
        row = _flex_instrument(el)
        key = row["conid"] or row["symbol"]
        if key:
            instruments.setdefault(key, row)

    accounts: list[str] = []
    for stmt in statements:
        acct = (stmt.get("accountId") or "").strip()
        if acct and acct not in accounts:
            accounts.append(acct)

    start = min((s.get("fromDate") or "") for s in statements)
    end = max((s.get("toDate") or "") for s in statements)

    return {
        "account": "+".join(accounts),
        "accounts": accounts,
        "period": _iso_period(start, end),
        "twrr": _chain_twr(statements),
        "trades": trades,
        "cash": cash,
        "positions": positions,
        "instruments": list(instruments.values()),
        "nav": _nav_rows(statements),
        "corporate_actions": corporate_actions,
    }


def flex_sync(
    query_id: str | None = None,
    token: str | None = None,
    retry_delay: float = 5.0,
) -> str:
    """Fetch the configured Flex statement and save its XML; returns a summary.

    ``token`` defaults to ``IBKR_FLEX_TOKEN`` (never taken from a model), ``query_id``
    to the arg or ``IBKR_FLEX_QUERY_ID``. Once ``parse_flex_xml`` is implemented this
    will also import into the store; today it saves the XML and points at the manual
    CSV path for querying."""
    token = (token or os.environ.get("IBKR_FLEX_TOKEN") or "").strip()
    qid = (query_id or os.environ.get("IBKR_FLEX_QUERY_ID") or "").strip()
    if not token:
        return (
            "No Flex token. Set IBKR_FLEX_TOKEN (IBKR → Reports → Flex Queries → "
            "Flex Web Service) to pull statements automatically. Manual import still "
            "works: import_ibkr_statement with a downloaded CSV."
        )
    if not qid:
        return "No Flex query id. Set IBKR_FLEX_QUERY_ID (or pass one) from your Flex Query config."
    try:
        xml = fetch_flex_xml(token, qid, retry_delay=retry_delay)
    except FlexError as e:
        return f"Flex sync failed: {e}"
    except Exception as e:  # network/other — never raise into a CLI/turn
        return f"Flex sync error: {type(e).__name__}: {e}"

    path = save_flex_xml(xml, qid)
    # The XML is saved before it is parsed, so a statement this parser can't yet
    # read is still on disk to look at rather than lost with the error.
    try:
        parsed = parse_flex_xml(xml)
    except FlexError as e:
        return (
            f"Fetched Flex statement for query {qid} ({len(xml):,} bytes) → saved {path}.\n"
            f"It could not be imported: {e}"
        )
    from . import statements

    summary = statements.store_statement(parsed)
    verb = "Re-imported" if summary["replaced"] else "Imported"
    return (
        f"Fetched Flex statement for query {qid} ({len(xml):,} bytes) → saved {path}.\n"
        f"{verb} {summary['period']} for {summary['account']}: "
        f"{summary['trades']} trades, {summary['cash']} cash movements, "
        f"{summary['positions']} positions, {summary['corporate_actions']} corporate actions"
        + (f", TWR {summary['twrr']}" if summary["twrr"] else "")
    )
