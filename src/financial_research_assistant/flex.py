"""IBKR Flex Web Service sync — pull an Activity statement programmatically instead
of downloading the CSV by hand.

The Flex Web Service is a documented two-step, token-authenticated flow:

1. **SendRequest** — ``GET .../FlexWebService/SendRequest?t=<token>&q=<queryId>&v=3``
   returns a small ``<FlexStatementResponse>`` with a ``ReferenceCode`` and the
   ``GetStatement`` URL (or a ``Fail`` status).
2. **GetStatement** — ``GET <url>?t=<token>&q=<referenceCode>&v=3`` returns the
   statement XML (root ``<FlexQueryResponse>``), or a ``Warn`` "generation in
   progress" response to retry shortly.

This module implements that fetch (with the in-progress retry) and saves the raw
XML. It deliberately does **not** yet parse the XML into the local store: the Flex
XML schema uses different field names than the CSV Activity Statement the importer
reads, and mapping it blind — without validating against a real statement — would
risk silently-wrong data. ``parse_flex_xml`` is a stub seam for that step; until
it's wired, ``--flex-sync`` fetches and saves the XML (which is exactly the sample
needed to finish the mapping), and the manual ``import_ibkr_statement`` CSV path is
unchanged.

Credentials: the **token** is read only from the ``IBKR_FLEX_TOKEN`` env var (a
secret — never passed through the model); the **query id** from
``IBKR_FLEX_QUERY_ID`` (or a CLI arg). Saved under
``~/.financial-research-assistant/flex`` (override ``FINANCIAL_RESEARCH_FLEX_DIR``).
"""

from __future__ import annotations

import os
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

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


def _flex_get(url: str, params: dict, timeout: float) -> str:
    """HTTP GET returning decoded text. Errors never include the token."""
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": "financial-research-assistant"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https)
        return resp.read().decode("utf-8", "replace")


def _text(root, tag: str) -> str:
    el = root.find(tag)
    return (el.text or "").strip() if el is not None and el.text else ""


def _parse_status(xml: str) -> dict:
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


def _is_in_progress(status: dict) -> bool:
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
    """Write the statement XML to a timestamped file under ``flex_dir()``."""
    d = flex_dir()
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_qid = "".join(c for c in query_id if c.isalnum()) or "query"
    dest = d / f"flex-{safe_qid}-{stamp}.xml"
    dest.write_text(xml, encoding="utf-8")
    return dest


def parse_flex_xml(xml: str) -> dict:  # noqa: ARG001 — seam, not yet implemented
    """Parse Flex XML into the internal statement dict (the shape
    ``statements.store_statement`` consumes). NOT YET IMPLEMENTED — pending
    validation against a real Flex statement, so the field mapping is correct
    rather than guessed. When wired, ``flex_sync`` will import automatically."""
    raise NotImplementedError(
        "Flex XML parsing is not wired yet — a real statement is needed to map its "
        "fields correctly. For now the XML is saved; import a CSV with "
        "import_ibkr_statement to populate the queryable store."
    )


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
    # Parser is a seam; try it, but today it's not implemented, so we save-only.
    try:
        parsed = parse_flex_xml(xml)
    except NotImplementedError:
        return (
            f"Fetched Flex statement for query {qid} ({len(xml):,} bytes) → saved {path}.\n"
            f"Automatic import into the queryable store isn't enabled yet (the XML "
            f"parser needs validating against a real statement). To query the data now, "
            f"import a downloaded CSV with import_ibkr_statement."
        )
    from . import statements

    summary = statements.store_statement(parsed)
    return f"Fetched and imported Flex statement for query {qid} → saved {path}. {summary}"
