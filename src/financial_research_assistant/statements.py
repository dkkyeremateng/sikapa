"""Interactive Brokers activity-statement import + local store.

IBKR "Activity Statement" CSVs are a *stack of sections* in one file. Every row
begins with a section name and a row type, e.g.::

    Trades,Header,DataDiscriminator,Asset Category,Currency,...
    Trades,Data,Order,Stocks,USD,U1111111,AMZN,"2026-04-01, 12:14:23",5.25,...
    Trades,SubTotal,,Stocks,USD,AMZN,,,5.25,...
    Dividends,Header,Currency,Account,Date,Description,Amount
    Dividends,Data,USD,U1111111,2026-04-01,NVDA(...) Cash Dividend ...,0.1
    Dividends,Data,Total,,,,50.2

Within a section a ``Header`` row names the columns; the ``Data`` rows that
follow are the records. ``SubTotal``/``Total`` aggregate rows (and ``Data`` rows
whose first field is the literal ``Total``) are skipped so they never
double-count.

This module extracts the **transactions** — trades plus the four cash-flow
sections (Dividends, Withholding Tax, Fees, Deposits & Withdrawals) — and stores
them in a local SQLite database so the agent can import a statement once and
query it across turns.

Store location: ``~/.<package-name>/statements.db`` (override with
``FINANCIAL_RESEARCH_STATEMENTS_DB``; tests point it at a temp file). An import
is keyed by ``(account, period)``, so re-importing the same statement *replaces*
its rows instead of duplicating them — the operation is idempotent.
"""

from __future__ import annotations

from typing import Any
import csv
import io
import os
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path

# --- CSV parsing -----------------------------------------------------------

# The four cash-flow sections we ingest alongside Trades, mapped to the short
# ``kind`` stored in the DB. Each section's Header row names its own columns; we
# read the shared ones (currency, account, date, description, amount) by name so
# the code is robust to column-order differences between sections.
_CASH_SECTIONS = {
    "Dividends": "dividend",
    "Withholding Tax": "withholding_tax",
    "Fees": "fee",
    "Deposits & Withdrawals": "deposit_withdrawal",
}


def _to_float(value: str | None) -> float:
    """Parse an IBKR numeric cell (may be blank or comma-grouped) to float.

    Tolerant by design: a blank, footnoted, or otherwise non-numeric cell yields
    0.0 rather than raising, so one malformed value in a numeric column can't
    abort the import of the whole statement."""
    if value is None:
        return 0.0
    v = value.strip().replace(",", "").replace("%", "")
    if not v:
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0


def _read_rows(source: str | Path) -> list[list[str]]:
    """Read the statement CSV (a path, or the CSV text itself) into rows."""
    if isinstance(source, Path) or (
        "\n" not in source and len(source) < 4096 and Path(source).exists()
    ):
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        text = source
    return list(csv.reader(io.StringIO(text)))


def parse_statement(source: str | Path) -> dict[str, Any]:
    """Parse an IBKR activity-statement CSV into structured transactions.

    ``source`` may be a filesystem path or the raw CSV text. Returns::

        {
          "account": "U1111111" | "",
          "period": "April 1, 2026 - May 19, 2026" | "",
          "trades": [ {symbol, datetime, quantity, ...}, ... ],
          "cash":   [ {kind, currency, account, date, description, amount, code}, ...],
        }

    Only ``Data`` rows are kept; ``Header``/``SubTotal``/``Total`` rows and
    per-section ``Total`` data rows are dropped so nothing double-counts.
    """
    rows = _read_rows(source)
    headers: dict[str, list[str]] = {}  # section -> current column names
    trades: list[dict[str, Any]] = []
    cash: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    instruments: list[dict[str, Any]] = []
    nav: list[dict[str, Any]] = []
    corporate_actions: list[dict[str, Any]] = []
    accounts: list[str] = []  # distinct accounts seen, in first-seen order
    period = ""
    twrr = ""

    for row in rows:
        if len(row) < 2:
            continue
        section, row_type = row[0].strip(), row[1].strip()
        fields = row[2:]

        # Statement metadata: the reporting period.
        if section == "Statement" and row_type == "Data" and len(fields) >= 2:
            if fields[0].strip() == "Period":
                period = fields[1].strip()
            continue

        if row_type == "Header":
            headers[section] = [f.strip() for f in fields]
            continue
        if row_type != "Data":
            continue  # SubTotal / Total aggregate rows

        cols = headers.get(section)
        if not cols:
            continue
        record = {cols[i]: fields[i] for i in range(min(len(cols), len(fields)))}

        # A per-section "Total" row is a Data row whose first field is "Total".
        if fields and fields[0].strip() == "Total":
            continue

        acct = (record.get("Account") or "").strip()
        if acct and acct not in accounts:
            accounts.append(acct)

        if section == "Trades" and record.get("DataDiscriminator", "").strip() == "Order":
            trades.append({
                "asset_category": record.get("Asset Category", "").strip(),
                "currency": record.get("Currency", "").strip(),
                "account": record.get("Account", "").strip(),
                "symbol": record.get("Symbol", "").strip(),
                "datetime": record.get("Date/Time", "").strip(),
                "quantity": _to_float(record.get("Quantity")),
                "trade_price": _to_float(record.get("T. Price")),
                "close_price": _to_float(record.get("C. Price")),
                "proceeds": _to_float(record.get("Proceeds")),
                "comm_fee": _to_float(record.get("Comm/Fee")),
                "basis": _to_float(record.get("Basis")),
                "realized_pl": _to_float(record.get("Realized P/L")),
                "mtm_pl": _to_float(record.get("MTM P/L")),
                "code": record.get("Code", "").strip(),
            })
        elif section in _CASH_SECTIONS:
            cash.append({
                "kind": _CASH_SECTIONS[section],
                "currency": record.get("Currency", "").strip(),
                "account": record.get("Account", "").strip(),
                "date": record.get("Date") or record.get("Settle Date") or "",
                "description": record.get("Description", "").strip(),
                "amount": _to_float(record.get("Amount")),
                "code": record.get("Code", "").strip(),
            })
        elif section == "Open Positions" and record.get("DataDiscriminator", "").strip() == "Summary":
            positions.append({
                "asset_category": record.get("Asset Category", "").strip(),
                "currency": record.get("Currency", "").strip(),
                "symbol": record.get("Symbol", "").strip(),
                "quantity": _to_float(record.get("Quantity")),
                "mult": _to_float(record.get("Mult")),
                "cost_price": _to_float(record.get("Cost Price")),
                "cost_basis": _to_float(record.get("Cost Basis")),
                "close_price": _to_float(record.get("Close Price")),
                "value": _to_float(record.get("Value")),
                "unrealized_pl": _to_float(record.get("Unrealized P/L")),
                "code": record.get("Code", "").strip(),
            })
        elif section == "Financial Instrument Information" and record.get("Symbol"):
            instruments.append({
                "asset_category": record.get("Asset Category", "").strip(),
                "symbol": record.get("Symbol", "").strip(),
                "description": record.get("Description", "").strip(),
                "conid": record.get("Conid", "").strip(),
                "security_id": record.get("Security ID", "").strip(),  # ISIN
                "underlying": record.get("Underlying", "").strip(),
                "listing_exch": record.get("Listing Exch", "").strip(),
                "multiplier": record.get("Multiplier", "").strip(),
                "type": record.get("Type", "").strip(),
                "code": record.get("Code", "").strip(),
            })
        elif section == "Corporate Actions" and record.get("Description"):
            # Splits, mergers, symbol changes, spin-offs. Quantity is the share
            # delta; Proceeds/Value/Realized P/L are usually 0 for a split.
            corporate_actions.append({
                "asset_category": record.get("Asset Category", "").strip(),
                "currency": record.get("Currency", "").strip(),
                "account": record.get("Account", "").strip(),
                "report_date": record.get("Report Date", "").strip(),
                "datetime": record.get("Date/Time", "").strip(),
                "description": record.get("Description", "").strip(),
                "quantity": _to_float(record.get("Quantity")),
                "proceeds": _to_float(record.get("Proceeds")),
                "value": _to_float(record.get("Value")),
                "realized_pl": _to_float(record.get("Realized P/L")),
                "code": record.get("Code", "").strip(),
            })
        elif section == "Net Asset Value":
            # This section carries two sub-tables under separate Header rows: the
            # per-asset-class NAV breakdown, then a one-column "Time Weighted Rate
            # of Return". Route each row by which columns its current header has.
            if record.get("Asset Class"):
                nav.append({
                    "asset_class": record.get("Asset Class", "").strip(),
                    "prior_total": _to_float(record.get("Prior Total")),
                    "current_long": _to_float(record.get("Current Long")),
                    "current_short": _to_float(record.get("Current Short")),
                    "current_total": _to_float(record.get("Current Total")),
                    "change": _to_float(record.get("Change")),
                })
            elif "Time Weighted Rate of Return" in record:
                twrr = record["Time Weighted Rate of Return"].strip()

    for c in cash:
        c["date"] = c["date"].strip()
    # A consolidated statement can cover several accounts. Use the single account
    # as the import key when there's one; when there are several, use a combined
    # "U1+U2" key so the import is never silently mislabeled as just the first
    # account seen (which would let a different consolidated file for the same
    # period overwrite it). ``accounts`` carries the full list for callers.
    account = "+".join(accounts) if accounts else ""
    return {
        "account": account,
        "accounts": accounts,
        "period": period,
        "twrr": twrr,
        "trades": trades,
        "cash": cash,
        "positions": positions,
        "instruments": instruments,
        "nav": nav,
        "corporate_actions": corporate_actions,
    }


# --- SQLite store ----------------------------------------------------------

def db_path() -> Path:
    """Path to the statements SQLite database (parent created on first write).

    Defaults to ``~/.<package-name>/statements.db``; override with
    ``FINANCIAL_RESEARCH_STATEMENTS_DB`` (tests point it at a temp file)."""
    pkg = (__package__ or "financial_research_assistant").replace("_", "-")
    default = Path.home() / f".{pkg}" / "statements.db"
    return Path(os.environ.get("FINANCIAL_RESEARCH_STATEMENTS_DB") or default)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS imports (
    id       INTEGER PRIMARY KEY,
    account  TEXT NOT NULL,
    period   TEXT NOT NULL,
    twrr     TEXT,
    imported TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (account, period)
);
CREATE TABLE IF NOT EXISTS trades (
    id             INTEGER PRIMARY KEY,
    import_id      INTEGER NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
    asset_category TEXT, currency TEXT, account TEXT, symbol TEXT,
    datetime       TEXT, quantity REAL, trade_price REAL, close_price REAL,
    proceeds REAL, comm_fee REAL, basis REAL, realized_pl REAL, mtm_pl REAL,
    code TEXT
);
CREATE TABLE IF NOT EXISTS cash (
    id          INTEGER PRIMARY KEY,
    import_id   INTEGER NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
    kind TEXT, currency TEXT, account TEXT, date TEXT, description TEXT,
    amount REAL, code TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    id             INTEGER PRIMARY KEY,
    import_id      INTEGER NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
    asset_category TEXT, currency TEXT, symbol TEXT, quantity REAL, mult REAL,
    cost_price REAL, cost_basis REAL, close_price REAL, value REAL,
    unrealized_pl REAL, code TEXT
);
CREATE TABLE IF NOT EXISTS instruments (
    id             INTEGER PRIMARY KEY,
    import_id      INTEGER NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
    asset_category TEXT, symbol TEXT, description TEXT, conid TEXT,
    security_id TEXT, underlying TEXT, listing_exch TEXT, multiplier TEXT,
    type TEXT, code TEXT
);
CREATE TABLE IF NOT EXISTS nav (
    id            INTEGER PRIMARY KEY,
    import_id     INTEGER NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
    asset_class TEXT, prior_total REAL, current_long REAL, current_short REAL,
    current_total REAL, change REAL
);
CREATE TABLE IF NOT EXISTS corporate_actions (
    id             INTEGER PRIMARY KEY,
    import_id      INTEGER NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
    asset_category TEXT, currency TEXT, account TEXT, report_date TEXT,
    datetime TEXT, description TEXT, quantity REAL, proceeds REAL, value REAL,
    realized_pl REAL, code TEXT
);
"""


_SCHEMA_VERSION = 1

# Columns added to a table after its first shipped version. `CREATE TABLE IF NOT
# EXISTS` creates missing *tables* but never adds *columns* to an existing one,
# so a DB created before a column existed would raise "no such column" on insert.
# We backfill any missing column via ALTER on connect (idempotent). Future column
# additions go here so old stores upgrade in place instead of failing to import.
_ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "imports": [("twrr", "TEXT")],
}


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing DB up to the current schema: add any columns that were
    introduced after a table's original version. Cheap and idempotent."""
    for table, cols in _ADDED_COLUMNS.items():
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in cols:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def store_statement(parsed: dict[str, Any]) -> dict[str, Any]:
    """Persist a parsed statement, replacing any prior import of the same
    ``(account, period)``. Returns a summary dict of what was stored."""
    account = parsed.get("account") or "UNKNOWN"
    period = parsed.get("period") or "UNKNOWN"
    twrr = parsed.get("twrr") or ""
    trades = parsed.get("trades") or []
    cash = parsed.get("cash") or []
    positions = parsed.get("positions") or []
    instruments = parsed.get("instruments") or []
    nav = parsed.get("nav") or []
    corporate_actions = parsed.get("corporate_actions") or []

    conn = _connect()
    try:
        with conn:  # one transaction: delete-old + insert-new is atomic
            existing = conn.execute(
                "SELECT id FROM imports WHERE account = ? AND period = ?",
                (account, period),
            ).fetchone()
            replaced = existing is not None
            if replaced:
                conn.execute("DELETE FROM imports WHERE id = ?", (existing["id"],))
            cur = conn.execute(
                "INSERT INTO imports (account, period, twrr) VALUES (?, ?, ?)",
                (account, period, twrr),
            )
            import_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO trades (import_id, asset_category, currency, account, "
                "symbol, datetime, quantity, trade_price, close_price, proceeds, "
                "comm_fee, basis, realized_pl, mtm_pl, code) VALUES "
                "(:import_id, :asset_category, :currency, :account, :symbol, "
                ":datetime, :quantity, :trade_price, :close_price, :proceeds, "
                ":comm_fee, :basis, :realized_pl, :mtm_pl, :code)",
                [{"import_id": import_id, **t} for t in trades],
            )
            conn.executemany(
                "INSERT INTO cash (import_id, kind, currency, account, date, "
                "description, amount, code) VALUES (:import_id, :kind, :currency, "
                ":account, :date, :description, :amount, :code)",
                [{"import_id": import_id, **c} for c in cash],
            )
            conn.executemany(
                "INSERT INTO positions (import_id, asset_category, currency, symbol, "
                "quantity, mult, cost_price, cost_basis, close_price, value, "
                "unrealized_pl, code) VALUES (:import_id, :asset_category, :currency, "
                ":symbol, :quantity, :mult, :cost_price, :cost_basis, :close_price, "
                ":value, :unrealized_pl, :code)",
                [{"import_id": import_id, **p} for p in positions],
            )
            conn.executemany(
                "INSERT INTO instruments (import_id, asset_category, symbol, "
                "description, conid, security_id, underlying, listing_exch, "
                "multiplier, type, code) VALUES (:import_id, :asset_category, "
                ":symbol, :description, :conid, :security_id, :underlying, "
                ":listing_exch, :multiplier, :type, :code)",
                [{"import_id": import_id, **i} for i in instruments],
            )
            conn.executemany(
                "INSERT INTO nav (import_id, asset_class, prior_total, current_long, "
                "current_short, current_total, change) VALUES (:import_id, "
                ":asset_class, :prior_total, :current_long, :current_short, "
                ":current_total, :change)",
                [{"import_id": import_id, **n} for n in nav],
            )
            conn.executemany(
                "INSERT INTO corporate_actions (import_id, asset_category, currency, "
                "account, report_date, datetime, description, quantity, proceeds, "
                "value, realized_pl, code) VALUES (:import_id, :asset_category, "
                ":currency, :account, :report_date, :datetime, :description, "
                ":quantity, :proceeds, :value, :realized_pl, :code)",
                [{"import_id": import_id, **a} for a in corporate_actions],
            )
    finally:
        conn.close()

    # Group cash sums by (kind, currency) — summing amounts across currencies
    # would produce a meaningless combined figure (e.g. USD + EUR dividends).
    cash_by_kind: dict[str, dict[str, Any]] = {}
    for c in cash:
        agg = cash_by_kind.setdefault(c["kind"], {"count": 0, "amounts": {}})
        agg["count"] += 1
        ccy = c.get("currency") or "?"
        agg["amounts"][ccy] = agg["amounts"].get(ccy, 0.0) + c["amount"]
    accounts = parsed.get("accounts") or ([account] if account else [])
    return {
        "account": account,
        "accounts": accounts,
        "multi_account": len(accounts) > 1,
        "period": period,
        "twrr": twrr,
        "replaced": replaced,
        "trades": len(trades),
        "cash": len(cash),
        "cash_by_kind": cash_by_kind,
        "positions": len(positions),
        "instruments": len(instruments),
        "nav": len(nav),
        "corporate_actions": len(corporate_actions),
    }


# --- OFX / QFX parsing (cross-broker) --------------------------------------

# IBKR ships a bespoke multi-section CSV; most *other* brokers and banks export
# OFX/QFX (the Open Financial Exchange standard — Vanguard, E*TRADE, Schwab, …).
# ``parse_ofx`` maps an OFX investment (or plain bank) statement onto the SAME
# normalized dict ``store_statement`` consumes, so a different broker's export
# flows into the identical local store. Fields OFX doesn't carry (TWRR, NAV,
# corporate actions, cost basis, unrealized P/L) stay empty rather than invented.


class OfxSupportError(RuntimeError):
    """OFX/QFX import was requested but the optional ``ofxtools`` package (the
    ``[ofx]`` extra) is not installed. Distinct from a parse error so callers can
    render a clean install hint instead of a generic failure."""


def _looks_like_ofx(head: str) -> bool:
    """True if ``head`` (the start of a file/text) is an OFX/QFX document: the
    v1 SGML header token ``OFXHEADER`` or the ``<OFX>`` root tag (v1 or v2)."""
    upper = head.upper()
    return "OFXHEADER" in upper or "<OFX>" in upper


def _source_head(source: str | Path, n: int = 1024) -> tuple[str, str]:
    """Return ``(suffix, head_text)`` for a statement source, reusing the exact
    path-vs-inline rule from ``_read_rows``: a ``Path`` (or a short newline-free
    string naming an existing file) is read from disk; anything else is treated
    as the document text itself. ``suffix`` is the lowercased file extension when
    the source is a file, else ``""``."""
    if isinstance(source, Path) or (
        "\n" not in source and len(source) < 4096 and Path(source).exists()
    ):
        p = Path(source)
        return p.suffix.lower(), p.read_text(encoding="utf-8-sig", errors="replace")[:n]
    return "", str(source)[:n]


def _detect_format(source: str | Path) -> str:
    """Route a statement source to a parser key. ``.ofx``/``.qfx`` files and any
    source whose head carries an OFX header/root tag are ``"ofx"``; everything
    else defaults to ``"ibkr_csv"`` (preserving the original behavior, so every
    existing CSV import routes exactly as before)."""
    suffix, head = _source_head(source)
    if suffix in {".ofx", ".qfx"} or _looks_like_ofx(head):
        return "ofx"
    return "ibkr_csv"


def _ofx_num(value: Any) -> float:
    """OFX money/quantity fields arrive as ``Decimal`` (or ``None``); coerce to a
    plain float, treating missing as 0.0 — mirrors ``_to_float`` for the CSV path."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bank_cash(trn: Any, currency: str, account: str) -> dict[str, Any]:
    """Map an OFX ``STMTTRN`` (a cash movement — the payload of an INVBANKTRAN or
    a bank statement's banktranlist) to a normalized cash row: a fee for
    FEE/SRVCHG types, otherwise a deposit/withdrawal."""
    trntype = (trn.trntype or "").upper()
    dt = trn.dtposted
    return {
        "kind": "fee" if trntype in {"FEE", "SRVCHG"} else "deposit_withdrawal",
        "currency": currency,
        "account": account,
        "date": dt.strftime("%Y-%m-%d") if dt else "",
        "description": (trn.name or trn.memo or "").strip(),
        "amount": _ofx_num(trn.trnamt),
        "code": "",
    }


def parse_ofx(source: str | Path) -> dict[str, Any]:
    """Parse an OFX/QFX statement into the normalized transaction dict (the same
    shape ``parse_statement`` returns), so a non-IBKR broker's export can be
    stored and queried identically.

    Maps the OFX investment model: SECLIST -> instruments (+ a CUSIP->ticker
    resolver), BUY*/SELL* -> trades, INCOME(DIV) -> dividend cash, INVBANKTRAN ->
    fee/deposit cash, INVPOSLIST -> positions. A plain bank statement (STMTRS)
    degrades to its cash rows. Requires the optional ``ofxtools`` package.
    """
    try:
        from ofxtools.Parser import OFXTree  # pyright: ignore[reportMissingImports]  (optional extra)
    except ModuleNotFoundError as e:  # optional [ofx] extra not installed
        raise OfxSupportError(
            "OFX/QFX import needs the optional 'ofxtools' package. Install it "
            "with:  pip install 'financial-research-assistant[ofx]'"
        ) from e

    # ofxtools wants bytes: a filename on disk, else the document text encoded.
    suffix, _head = _source_head(source)
    tree = OFXTree()
    if suffix in {".ofx", ".qfx"} or isinstance(source, Path):
        tree.parse(str(source))
    else:
        tree.parse(io.BytesIO(str(source).encode("utf-8")))
    ofx = tree.convert()

    # Security master first: resolve trades/positions (which reference a SECID,
    # usually a CUSIP) to a ticker, and record each as an instrument row.
    instruments: list[dict[str, Any]] = []
    sec_map: dict[str, dict[str, Any]] = {}
    for sec in getattr(ofx, "securities", None) or []:
        info = getattr(sec, "secinfo", None)
        if info is None:
            continue
        uid = getattr(getattr(info, "secid", None), "uniqueid", "") or ""
        uidtype = getattr(getattr(info, "secid", None), "uniqueidtype", "") or ""
        ticker = (getattr(info, "ticker", "") or "").strip()
        name = (getattr(info, "secname", "") or "").strip()
        symbol = ticker or uid  # fall back to the CUSIP when no ticker is given
        if uid:
            sec_map[uid] = {"symbol": symbol, "name": name}
        instruments.append({
            "asset_category": type(sec).__name__.replace("INFO", "").title(),
            "symbol": symbol,
            "description": name,
            "conid": "",
            "security_id": uid,  # CUSIP/ISIN
            "underlying": "",
            "listing_exch": "",
            "multiplier": "",
            "type": uidtype,
            "code": "",
        })

    def _symbol(secid: Any) -> str:
        uid = getattr(secid, "uniqueid", "") or ""
        return (sec_map.get(uid) or {}).get("symbol") or uid

    trades: list[dict[str, Any]] = []
    cash: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    accounts: list[str] = []
    period = ""

    for stmt in getattr(ofx, "statements", None) or []:
        # ofxtools aggregates are list subclasses (falsy when empty) and their
        # ``__getattr__`` recurses on names outside the aggregate's schema — so
        # branch by statement class and touch only schema-valid attributes; never
        # ``A or B`` two aggregates, never ``getattr(agg, foreign_name, default)``.
        cls = type(stmt).__name__
        is_inv = cls == "INVSTMTRS"
        if not is_inv and cls != "STMTRS":
            continue  # unsupported statement type
        curdef = (stmt.curdef or "").strip()
        acct_from = stmt.invacctfrom if is_inv else stmt.bankacctfrom
        account = (getattr(acct_from, "acctid", "") or "").strip()
        if account and account not in accounts:
            accounts.append(account)

        tranlist = stmt.invtranlist if is_inv else None
        # Reporting period: format like the IBKR string ("April 1, 2026 - …") so
        # it chains through the same period-bounds parsing the NAV/perf tools use.
        start = tranlist.dtstart if tranlist is not None else None
        end = tranlist.dtend if tranlist is not None else None
        if start is None and end is None and is_inv:
            start = end = stmt.dtasof
        if start is not None and not period:
            period = " - ".join(
                dt.strftime("%B %d, %Y") for dt in (start, end) if dt is not None
            )

        for txn in (tranlist or []):
            cn = type(txn).__name__
            # BUY*/SELL* -> trades; REINVEST is a dividend reinvested into shares,
            # so map it as a buy only (adds the lot) — do NOT also emit a dividend,
            # which would double-count the cash.
            if cn.startswith(("BUY", "SELL", "REINV")):
                invtran = txn.invtran
                dttrade = invtran.dttrade
                units = abs(_ofx_num(txn.units))
                total = abs(_ofx_num(txn.total))
                comm = abs(_ofx_num(txn.commission))
                fees = abs(_ofx_num(txn.fees))
                is_buy = not cn.startswith("SELL")  # BUY* and REINV* add shares
                if cn.startswith("BUY"):
                    asset_cat = cn[3:].title()
                elif cn.startswith("SELL"):
                    asset_cat = cn[4:].title()
                else:
                    asset_cat = ""  # REINVEST doesn't encode the security type
                trades.append({
                    "asset_category": asset_cat,
                    "currency": curdef,
                    "account": account,
                    "symbol": _symbol(txn.secid),
                    "datetime": dttrade.strftime("%Y-%m-%d, %H:%M:%S") if dttrade else "",
                    # Sign by buy/sell TYPE, not the raw sign (brokers differ):
                    # buys add shares and spend cash; sells remove shares, add cash.
                    "quantity": units if is_buy else -units,
                    "trade_price": _ofx_num(txn.unitprice),
                    "close_price": 0.0,
                    "proceeds": -total if is_buy else total,
                    "comm_fee": -(comm + fees),  # IBKR stores fees negative
                    "basis": total,
                    "realized_pl": 0.0,
                    "mtm_pl": 0.0,
                    "code": "REINVEST" if cn.startswith("REINV") else "",
                })
            elif cn == "INCOME" and (txn.incometype or "").upper() == "DIV":
                invtran = txn.invtran
                dt = invtran.dttrade or invtran.dtsettle
                memo = (invtran.memo or "").strip()
                sec = sec_map.get(getattr(txn.secid, "uniqueid", ""), {})
                cash.append({
                    "kind": "dividend",
                    "currency": curdef,
                    "account": account,
                    "date": dt.strftime("%Y-%m-%d") if dt else "",
                    "description": memo or sec.get("name", ""),
                    "amount": _ofx_num(txn.total),
                    "code": "",
                })
            elif cn == "INVBANKTRAN":
                cash.append(_bank_cash(txn.stmttrn, curdef, account))

        # Plain bank statement (STMTRS): no invtranlist, a banktranlist of STMTTRN.
        banktranlist = stmt.banktranlist if cls == "STMTRS" else None
        for trn in (banktranlist or []):
            cash.append(_bank_cash(trn, curdef, account))

        invposlist = stmt.invposlist if is_inv else None
        for pos in (invposlist or []):
            postype = (pos.postype or "").upper()
            units = _ofx_num(pos.units)
            positions.append({
                "asset_category": type(pos).__name__.replace("POS", "").title(),
                "currency": curdef,
                "symbol": _symbol(pos.secid),
                "quantity": -abs(units) if postype == "SHORT" else units,
                "mult": 0.0,
                "cost_price": 0.0,   # not carried by standard INVPOS
                "cost_basis": 0.0,
                "close_price": _ofx_num(pos.unitprice),
                "value": _ofx_num(pos.mktval),
                "unrealized_pl": 0.0,
                "code": "",
            })

    account = "+".join(accounts) if accounts else ""
    return {
        "account": account,
        "accounts": accounts,
        "period": period,
        "twrr": "",
        "trades": trades,
        "cash": cash,
        "positions": positions,
        "instruments": instruments,
        "nav": [],
        "corporate_actions": [],
    }


# Statement-format registry: detector key -> parser producing the normalized
# dict. Extensible — a future Flex-XML parser registers here the same way.
_FORMAT_PARSERS = {
    "ibkr_csv": parse_statement,
    "ofx": parse_ofx,
}


def import_statement(source: str | Path) -> dict[str, Any]:
    """Parse ``source`` (a path or the document text) and store it, auto-detecting
    the format: IBKR Activity Statement CSV or a cross-broker OFX/QFX file."""
    parser = _FORMAT_PARSERS[_detect_format(source)]
    return store_statement(parser(source))


def _like_contains(term: str) -> str:
    """A ``%term%`` LIKE pattern with SQL wildcards in ``term`` escaped (pair
    with ``ESCAPE '\\'``), so a symbol containing ``%``/``_`` matches literally
    instead of acting as a wildcard."""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _txn_recency(row: dict[str, Any]) -> tuple[Any, ...]:
    """Sort key so a mixed trade/cash/corp-action list orders newest-first: newest
    import first, then latest row date (trades carry ``datetime``, cash ``date``,
    corporate actions ``report_date``) — all ISO/ISO-ish so string compare works."""
    d = row.get("datetime") or row.get("date") or row.get("report_date") or ""
    return (row.get("import_id") or 0, d)


def query_transactions(
    kind: str | None = None,
    symbol: str | None = None,
    limit: int = 100,
    account: str | None = None,
) -> list[dict[str, Any]]:
    """Return stored transactions, newest first (by import, then row date).

    ``kind`` filters the record type: ``trade``, one of the cash kinds
    (``dividend``, ``withholding_tax``, ``fee``, ``deposit_withdrawal``), or
    ``corporate_action``; None returns all of them. ``symbol`` filters trades by
    ticker and cash/corporate-action rows whose description mentions it.
    ``account`` optionally scopes to one account. ``limit`` caps the total row
    count **across all types by recency** (so a small limit no longer silently
    hides every cash/corporate-action row behind the trades).
    """
    if not db_path().exists():
        return []
    cash_kinds = {"dividend", "withholding_tax", "fee", "deposit_withdrawal"}
    cap = max(1, limit)
    conn = _connect()
    try:
        out: list[dict[str, Any]] = []
        if kind in (None, "trade"):
            clauses, params = [], []
            if symbol:
                clauses.append("UPPER(symbol) = ?")
                params.append(symbol.upper())
            if account:
                clauses.append("account = ?")
                params.append(account)
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            q = f"SELECT * FROM trades{where} ORDER BY import_id DESC, id DESC LIMIT ?"
            for r in conn.execute(q, [*params, cap]).fetchall():
                out.append({"kind": "trade", **{k: r[k] for k in r.keys()}})
        if kind is None or kind in cash_kinds:
            clauses, params = [], []
            if kind in cash_kinds:
                clauses.append("kind = ?")
                params.append(kind)
            if symbol:
                clauses.append("UPPER(description) LIKE ? ESCAPE '\\'")
                params.append(_like_contains(symbol.upper()))
            if account:
                clauses.append("account = ?")
                params.append(account)
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            q = f"SELECT * FROM cash{where} ORDER BY import_id DESC, id DESC LIMIT ?"
            for r in conn.execute(q, [*params, cap]).fetchall():
                out.append({k: r[k] for k in r.keys()})
        if kind in (None, "corporate_action"):
            clauses, params = [], []
            if symbol:
                clauses.append("UPPER(description) LIKE ? ESCAPE '\\'")
                params.append(_like_contains(symbol.upper()))
            if account:
                clauses.append("account = ?")
                params.append(account)
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            q = (f"SELECT * FROM corporate_actions{where} "
                 "ORDER BY import_id DESC, id DESC LIMIT ?")
            for r in conn.execute(q, [*params, cap]).fetchall():
                out.append({"kind": "corporate_action", **{k: r[k] for k in r.keys()}})
        # Merge the (already per-type capped) blocks by recency, then cap the
        # total — so the limit reflects the newest rows across every type.
        out.sort(key=_txn_recency, reverse=True)
        return out[:cap]
    finally:
        conn.close()


def list_accounts() -> list[str]:
    """Distinct accounts present in the store, newest-import first. (A combined
    ``U1+U2`` entry denotes a consolidated multi-account statement.)"""
    if not db_path().exists():
        return []
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT account FROM imports GROUP BY account ORDER BY MAX(id) DESC"
        ).fetchall()
        return [r["account"] for r in rows]
    finally:
        conn.close()


def _resolve_account(conn: sqlite3.Connection, account: str | None) -> str | None:
    """Which account a query scopes to: the one asked for, else the account of
    the newest import (so single-account stores behave exactly as before)."""
    if account:
        return account
    row = conn.execute(
        "SELECT account FROM imports ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row["account"] if row else None


def query_positions(symbol: str | None = None, account: str | None = None) -> list[dict[str, Any]]:
    """Open positions from the newest import **for one account**, each enriched
    with the instrument's description and ISIN (``security_id``). ``symbol``
    optionally filters to one ticker; ``account`` picks which account (default:
    the newest import's account — see ``list_accounts``)."""
    if not db_path().exists():
        return []
    conn = _connect()
    try:
        acct = _resolve_account(conn, account)
        if acct is None:
            return []
        newest = conn.execute(
            "SELECT MAX(id) AS m FROM imports WHERE account = ?", (acct,)
        ).fetchone()
        if not newest or newest["m"] is None:
            return []
        import_id = newest["m"]
        q = (
            "SELECT p.*, i.description, i.security_id, i.listing_exch, i.type "
            "FROM positions p LEFT JOIN instruments i "
            "ON i.import_id = p.import_id AND i.symbol = p.symbol "
            "WHERE p.import_id = ?"
        )
        params: list[Any] = [import_id]
        if symbol:
            q += " AND UPPER(p.symbol) = ?"
            params.append(symbol.upper())
        q += " ORDER BY p.value DESC"
        return [{k: r[k] for k in r.keys()} for r in conn.execute(q, params).fetchall()]
    finally:
        conn.close()


def query_nav(account: str | None = None) -> dict[str, Any]:
    """Net Asset Value breakdown (per asset class) and the time-weighted return
    from the newest import for one account: ``{"twrr": str, "rows": [...]}``.
    ``account`` defaults to the newest import's account."""
    if not db_path().exists():
        return {"twrr": "", "rows": []}
    conn = _connect()
    try:
        acct = _resolve_account(conn, account)
        if acct is None:
            return {"twrr": "", "rows": []}
        newest = conn.execute(
            "SELECT id, twrr FROM imports WHERE account = ? ORDER BY id DESC LIMIT 1",
            (acct,),
        ).fetchone()
        if not newest:
            return {"twrr": "", "rows": []}
        rows = conn.execute(
            "SELECT * FROM nav WHERE import_id = ? ORDER BY id ASC", (newest["id"],)
        ).fetchall()
        return {
            "twrr": newest["twrr"] or "",
            "rows": [{k: r[k] for k in r.keys()} for r in rows],
        }
    finally:
        conn.close()


def _period_bounds(period: str) -> tuple[str, str] | None:
    """Parse an IBKR period string like ``"January 1, 2024 - December 31, 2024"``
    into ISO ``(start, end)`` dates, or None if it can't be parsed."""
    if " - " not in period:
        return None
    left, right = period.split(" - ", 1)
    try:
        start = datetime.strptime(left.strip(), "%B %d, %Y").date().isoformat()
        end = datetime.strptime(right.strip(), "%B %d, %Y").date().isoformat()
    except ValueError:
        return None
    return start, end


def query_nav_history(account: str | None = None) -> list[dict[str, Any]]:
    """Total account NAV over time for one account, stitched from the NAV
    snapshot in every imported statement. Each statement contributes two dated
    points — its period-start NAV (sum of the asset-class *prior* totals) and its
    period-end NAV (sum of the *current* totals). Points are merged by date (a
    later import wins a tie) and returned oldest→newest as ``{date, nav}``.

    ``account`` defaults to the newest import's account, so mixing statements
    from two accounts doesn't blend their NAV into one nonsense series. This is a
    real, if coarse, account-value curve: import more statements (e.g. monthly)
    for finer resolution. Statements whose period can't be parsed are skipped."""
    if not db_path().exists():
        return []
    conn = _connect()
    try:
        acct = _resolve_account(conn, account)
        if acct is None:
            return []
        rows = conn.execute(
            "SELECT i.id, i.period, "
            "COALESCE(SUM(n.prior_total), 0) AS prior, "
            "COALESCE(SUM(n.current_total), 0) AS current "
            "FROM imports i JOIN nav n ON n.import_id = i.id "
            "WHERE i.account = ? "
            "GROUP BY i.id ORDER BY i.id ASC",
            (acct,),
        ).fetchall()
    finally:
        conn.close()
    points: dict[str, float] = {}
    for r in rows:  # oldest import first, so a newer import overwrites a tie
        bounds = _period_bounds(r["period"] or "")
        if not bounds:
            continue
        start, end = bounds
        points[start] = r["prior"]
        points[end] = r["current"]
    return [{"date": d, "nav": points[d]} for d in sorted(points)]


def _parse_pct(value: str) -> float | None:
    """Parse a percent string like ``"10.98645721%"`` into a float (10.98…), or
    None if blank/unparseable."""
    v = (value or "").strip().rstrip("%").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _select_non_overlapping(periods: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pick a non-overlapping subset of statement periods so their time-weighted
    returns can be chained without double-counting any span, and return
    ``(selected, dropped)``.

    Uses classic activity-selection (greedy by earliest end date), which
    maximizes the number of intervals kept — so if you've imported both a yearly
    statement and the twelve monthly ones it covers, the monthlies win (finer
    resolution) and the redundant annual is dropped. Two periods that merely abut
    (one ends the day before the next begins) do not overlap."""
    ordered = sorted(periods, key=lambda p: (p["end"], p["start"]))
    selected: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    coverage_end: str | None = None  # ISO dates compare correctly as strings
    for p in ordered:
        if coverage_end is None or p["start"] >= coverage_end:
            selected.append(p)
            coverage_end = p["end"]
        else:
            dropped.append(p)
    selected.sort(key=lambda p: p["start"])
    return selected, dropped


def query_performance_history(account: str | None = None) -> dict[str, Any]:
    """Deposit-independent performance index over time for one account.

    Raw NAV (``query_nav_history``) includes cash you deposited, so it can't tell
    growth from contributions. This instead compounds each imported statement's
    **time-weighted return** (TWRR) — which by construction removes the effect of
    deposits/withdrawals — into a ``growth of 100`` index: start at 100 on the
    earliest period start, then multiply by ``(1 + TWRR)`` for each chained
    statement.

    Robust to messy imports: overlapping statements are reconciled to a
    non-overlapping set (see ``_select_non_overlapping``) so no span is
    double-counted, and gaps between consecutive periods are detected and
    reported (they can't be filled without the missing statement). Returns::

        {
          "points":  [{date, index, cumulative_return_pct}, ...],  # oldest→newest
          "dropped": [period_str, ...],   # overlapping statements not chained
          "gaps":    [{"after": iso, "before": iso, "days": n}, ...],
        }

    Statements lacking a parseable period or TWRR are ignored."""
    empty = {"points": [], "dropped": [], "gaps": []}
    if not db_path().exists():
        return empty
    conn = _connect()
    try:
        acct = _resolve_account(conn, account)
        if acct is None:
            return empty
        rows = conn.execute(
            "SELECT period, twrr FROM imports WHERE account = ?", (acct,)
        ).fetchall()
    finally:
        conn.close()
    periods: list[dict[str, Any]] = []
    for r in rows:
        bounds = _period_bounds(r["period"] or "")
        pct = _parse_pct(r["twrr"] or "")
        if bounds and pct is not None:
            periods.append({"start": bounds[0], "end": bounds[1], "twrr": pct,
                            "period": r["period"]})
    if not periods:
        return empty

    selected, dropped = _select_non_overlapping(periods)

    # The points array is [first.start, selected[0].end, selected[1].end, ...],
    # so the end of selected[k] is at point index k+1. A gap between selected[k]
    # and selected[k+1] therefore sits after point index k+1 — recorded as
    # ``after_point`` so a renderer can break the line exactly there.
    gaps: list[dict[str, Any]] = []
    for k, (prev, nxt) in enumerate(zip(selected, selected[1:])):
        delta = (date.fromisoformat(nxt["start"]) - date.fromisoformat(prev["end"])).days
        if delta > 1:  # a day-adjacent boundary (Dec 31 → Jan 1) is not a gap
            gaps.append({"after": prev["end"], "before": nxt["start"],
                         "days": delta, "after_point": k + 1})

    points = [{"date": selected[0]["start"], "index": 100.0,
               "cumulative_return_pct": 0.0}]
    index = 100.0
    for p in selected:
        index *= 1.0 + p["twrr"] / 100.0
        points.append({
            "date": p["end"],
            "index": round(index, 4),
            "cumulative_return_pct": round(index - 100.0, 4),
        })
    return {
        "points": points,
        "dropped": [p["period"] for p in dropped],
        "gaps": gaps,
    }


# --- Analytics over stored data --------------------------------------------

def _days_between(start_iso: str, end_iso: str) -> int:
    """Whole days from ``start_iso`` to ``end_iso`` (both YYYY-MM-DD…); 0 if either
    can't be parsed."""
    try:
        return (date.fromisoformat(end_iso[:10]) - date.fromisoformat(start_iso[:10])).days
    except ValueError:
        return 0


def _trade_split_events(
    symbol: str | None = None, account: str | None = None
) -> list[tuple[Any, ...]]:
    """One chronological event stream of trades + share splits for the FIFO walk.

    Each event is ``(date, order, kind, payload)`` where ``order`` puts a split
    (0) before same-day trades (1), so a same-day sell matches post-split lots.
    ``payload`` is the trade row for ``"trade"`` and ``(symbol, factor)`` for
    ``"split"``. Shared by ``realized_gains`` and ``open_lots`` so the split
    handling lives in one place. Returns ``[]`` when the store is empty."""
    if not db_path().exists():
        return []
    conn = _connect()
    try:
        clauses, params = [], []
        if symbol:
            clauses.append("UPPER(symbol) = ?")
            params.append(symbol.upper())
        if account:
            clauses.append("account = ?")
            params.append(account)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT symbol, datetime, quantity, trade_price, proceeds, basis "
            f"FROM trades{where} ORDER BY datetime ASC, id ASC",
            params,
        ).fetchall()
        ca_clauses, ca_params = [], []
        if account:
            ca_clauses.append("account = ?")
            ca_params.append(account)
        ca_where = (" WHERE " + " AND ".join(ca_clauses)) if ca_clauses else ""
        ca_rows = conn.execute(
            f"SELECT report_date, datetime, description FROM corporate_actions{ca_where}",
            ca_params,
        ).fetchall()
    finally:
        conn.close()

    events: list[tuple[Any, ...]] = []
    for r in rows:
        events.append(((r["datetime"] or "")[:10], 1, "trade", r))
    for c in ca_rows:
        factor = _split_factor(c["description"])
        sym = _symbol_from_description(c["description"])
        if factor and sym != "?" and (symbol is None or sym == symbol.upper()):
            when = (c["datetime"] or c["report_date"] or "")[:10]
            events.append((when, 0, "split", (sym, factor)))
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def open_lots(
    symbol: str | None = None, account: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """The FIFO-**open** lots remaining after matching every sell against buys —
    i.e. the shares you still hold and what they cost.

    Walks the same trade+split stream as ``realized_gains`` but keeps the leftover
    (unsold) lots instead of recording gains. Returns ``{symbol: [{qty,
    cost_per_share, open_date}, …]}`` (oldest lot first), splits already applied so
    each lot's qty and cost/share are split-adjusted. ``symbol``/``account`` scope
    the input. The basis is commission-inclusive (same as ``realized_gains``)."""
    from collections import defaultdict, deque

    events = _trade_split_events(symbol, account)
    lots: dict[str, deque[Any]] = defaultdict(deque)
    for _when, _order, kind, payload in events:
        if kind == "split":
            sym, factor = payload
            for lot in lots.get(sym, ()):
                lot[0] *= factor
                lot[1] /= factor
            continue
        r = payload
        sym, qty = r["symbol"], r["quantity"]
        dt = (r["datetime"] or "")[:10]
        if qty > 0:
            cost_per_share = (r["basis"] / qty) if qty else r["trade_price"]
            lots[sym].append([qty, cost_per_share, dt])
        elif qty < 0:
            sell_qty = -qty
            while sell_qty > 1e-9 and lots[sym]:
                lot = lots[sym][0]
                take = min(sell_qty, lot[0])
                lot[0] -= take
                sell_qty -= take
                if lot[0] <= 1e-9:
                    lots[sym].popleft()
    return {
        s: [{"qty": l[0], "cost_per_share": l[1], "open_date": l[2]} for l in dq]
        for s, dq in lots.items() if dq
    }


def realized_gains(
    year: int | None = None,
    symbol: str | None = None,
    account: str | None = None,
) -> dict[str, Any]:
    """Realized gains computed by **FIFO lot matching** over stored trades.

    Walks every trade chronologically, opening a lot on each buy and matching
    sells against the oldest open lots. For each matched slice it records the
    gain ``(sell_price − lot_cost) × qty`` and classifies it **short-term**
    (held < 365 days) or **long-term** (≥ 365). Per-share amounts use the
    commission-inclusive ``basis``/``proceeds`` fields, so gains are net of
    commissions.

    ``year`` counts only realizations *sold* in that calendar year (lots may open
    earlier). ``symbol``/``account`` scope the input. Returns per-symbol and
    total short/long/realized, plus ``unmatched_proceeds`` for any sell with no
    open lot (e.g. the opening statement wasn't imported, or a short sale) so the
    number is never silently wrong.

    Note: FIFO here; IBKR's own default is also FIFO but the broker's realized
    figures can differ with wash-sale or specific-lot adjustments not in the
    basic Trades section."""
    from collections import defaultdict, deque

    events = _trade_split_events(symbol, account)
    lots: dict[str, deque[Any]] = defaultdict(deque)  # sym -> [ [qty, cost/sh, date], ...]
    per: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"realized": 0.0, "short_term": 0.0, "long_term": 0.0})
    unmatched = 0.0
    for _when, _order, kind, payload in events:
        if kind == "split":
            sym, factor = payload
            for lot in lots.get(sym, ()):  # more shares, proportionally lower cost/sh
                lot[0] *= factor
                lot[1] /= factor
            continue
        r = payload
        sym, qty = r["symbol"], r["quantity"]
        dt = (r["datetime"] or "")[:10]
        if qty > 0:  # buy → open a lot (basis is positive, commission-inclusive)
            cost_per_share = (r["basis"] / qty) if qty else r["trade_price"]
            lots[sym].append([qty, cost_per_share, dt])
        elif qty < 0:  # sell → match FIFO (proceeds positive, net of commission)
            sell_qty = -qty
            price_per_share = (r["proceeds"] / sell_qty) if sell_qty else r["trade_price"]
            counts = year is None or dt[:4] == str(year)
            while sell_qty > 1e-9 and lots[sym]:
                lot = lots[sym][0]
                take = min(sell_qty, lot[0])
                gain = (price_per_share - lot[1]) * take
                if counts:
                    bucket = "long_term" if _days_between(lot[2], dt) >= 365 else "short_term"
                    per[sym]["realized"] += gain
                    per[sym][bucket] += gain
                lot[0] -= take
                sell_qty -= take
                if lot[0] <= 1e-9:
                    lots[sym].popleft()
            if sell_qty > 1e-9 and counts:  # sold more than any open lot covered
                unmatched += price_per_share * sell_qty

    by_symbol = {s: v for s, v in per.items() if abs(v["realized"]) > 1e-9}
    return {
        "by_symbol": by_symbol,
        "total_realized": round(sum(v["realized"] for v in by_symbol.values()), 2),
        "short_term": round(sum(v["short_term"] for v in by_symbol.values()), 2),
        "long_term": round(sum(v["long_term"] for v in by_symbol.values()), 2),
        "unmatched_proceeds": round(unmatched, 2),
    }


_SYMBOL_RE = re.compile(r"^([A-Za-z0-9.]+)\(")


def _symbol_from_description(desc: str) -> str:
    """Pull the leading ticker from a cash-row description like
    ``"NVDA(US67066G1040) Cash Dividend …"`` → ``NVDA`` (``?`` if none)."""
    m = _SYMBOL_RE.match(desc or "")
    return m.group(1).upper() if m else "?"


def _split_factor(description: str) -> float | None:
    """Parse a share-split factor from a corporate-action description like
    ``"… Split 3 for 1 (…)"`` → 3.0, or ``"… Split 1 for 10 …"`` (reverse) → 0.1.
    Returns None if the description isn't a recognizable split."""
    m = re.search(r"Split\s+(\d+(?:\.\d+)?)\s+for\s+(\d+(?:\.\d+)?)", description or "",
                  re.IGNORECASE)
    if not m:
        return None
    num, den = float(m.group(1)), float(m.group(2))
    return (num / den) if den else None


def income_summary(year: int | None = None, account: str | None = None) -> dict[str, Any]:
    """Cash-income summary from stored dividends, withholding tax, and fees,
    grouped **by currency** (never blended). Returns per-currency
    ``{gross_dividends, withholding_tax, fees, net}`` plus a per-symbol dividend
    breakdown. ``year`` filters by the cash date; ``account`` scopes it."""
    if not db_path().exists():
        return {"by_currency": {}, "dividends_by_symbol": {}}
    conn = _connect()
    try:
        clauses, params = ["kind IN ('dividend','withholding_tax','fee')"], []
        if account:
            clauses.append("account = ?")
            params.append(account)
        where = " WHERE " + " AND ".join(clauses)
        rows = conn.execute(
            f"SELECT kind, currency, date, description, amount FROM cash{where}",
            params,
        ).fetchall()
    finally:
        conn.close()

    field = {"dividend": "gross_dividends", "withholding_tax": "withholding_tax",
             "fee": "fees"}
    by_ccy: dict[str, dict[str, Any]] = {}
    by_symbol: dict[str, float] = {}
    for r in rows:
        if year is not None and (r["date"] or "")[:4] != str(year):
            continue
        ccy = r["currency"] or "?"
        agg = by_ccy.setdefault(ccy, {"gross_dividends": 0.0, "withholding_tax": 0.0,
                                      "fees": 0.0, "net": 0.0})
        agg[field[r["kind"]]] += r["amount"]
        agg["net"] += r["amount"]  # withholding/fees are stored negative
        if r["kind"] == "dividend":
            sym = _symbol_from_description(r["description"])
            by_symbol[sym] = by_symbol.get(sym, 0.0) + r["amount"]
    for agg in by_ccy.values():
        for k in agg:
            agg[k] = round(agg[k], 2)
    return {
        "by_currency": by_ccy,
        "dividends_by_symbol": {s: round(a, 2) for s, a in sorted(
            by_symbol.items(), key=lambda kv: kv[1], reverse=True)},
    }


def allocation(account: str | None = None, fx: dict[str, Any] | None = None) -> dict[str, Any]:
    """Portfolio allocation & concentration from the newest import's open
    positions (for one account). Returns positions ranked by market value with
    each one's weight %% of the book, total value, the top-5 concentration %%, the
    largest single weight, and value/weight by asset category.

    ``fx`` optionally maps a position's currency → its rate to the base currency
    (e.g. ``{"EUR": 1.08}``); values are converted before weights are computed so
    a multi-currency book isn't summed across currencies. Rates are supplied by
    the caller (the tool layer fetches them) to keep this function offline/pure;
    absent, values are used as-is (correct for a single-currency book). The
    returned ``value``/``total_value`` are in base-currency terms when ``fx`` is
    given, and ``currencies_converted`` lists any non-base currencies seen."""
    positions = query_positions(account=account)
    fx = fx or {}
    converted: set[str] = set()

    def _base(v: float, ccy: str) -> float:
        rate = fx.get(ccy)
        if rate is not None and rate != 1.0:
            converted.add(ccy)
        return v * (rate if rate is not None else 1.0)

    total = sum(_base(p["value"], p.get("currency") or "") for p in positions)
    ranked: list[dict[str, Any]] = []
    by_category: dict[str, float] = {}
    for p in positions:
        val = _base(p["value"], p.get("currency") or "")
        w = (val / total * 100.0) if total else 0.0
        ranked.append({
            "symbol": p["symbol"],
            "description": p.get("description") or "",
            "value": round(val, 2),
            "weight_pct": round(w, 2),
            "unrealized_pl": round(p.get("unrealized_pl") or 0.0, 2),
        })
        cat = p.get("asset_category") or "?"
        by_category[cat] = by_category.get(cat, 0.0) + val
    ranked.sort(key=lambda r: r["value"], reverse=True)
    top5 = sum(r["weight_pct"] for r in ranked[:5])
    return {
        "total_value": round(total, 2),
        "positions": ranked,
        "largest_weight_pct": ranked[0]["weight_pct"] if ranked else 0.0,
        "top5_concentration_pct": round(top5, 2),
        "by_category": {c: {"value": round(v, 2),
                            "weight_pct": round(v / total * 100.0, 2) if total else 0.0}
                        for c, v in sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)},
        "currencies_converted": sorted(converted),
    }


def list_imports() -> list[dict[str, Any]]:
    """Every stored import (account, period, counts), newest first."""
    if not db_path().exists():
        return []
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT i.id, i.account, i.period, i.twrr, i.imported, "
            "(SELECT COUNT(*) FROM trades t WHERE t.import_id = i.id) AS trades, "
            "(SELECT COUNT(*) FROM cash c WHERE c.import_id = i.id) AS cash, "
            "(SELECT COUNT(*) FROM positions p WHERE p.import_id = i.id) AS positions, "
            "(SELECT COUNT(*) FROM instruments n WHERE n.import_id = i.id) AS instruments, "
            "(SELECT COUNT(*) FROM nav v WHERE v.import_id = i.id) AS nav, "
            "(SELECT COUNT(*) FROM corporate_actions a WHERE a.import_id = i.id) AS corporate_actions "
            "FROM imports i ORDER BY i.id DESC"
        ).fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]
    finally:
        conn.close()
