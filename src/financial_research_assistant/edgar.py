"""SEC EDGAR filing-intelligence tools (keyless, free).

The single biggest gap versus every commercial financial-research agent (OpenBB,
AlphaSense, Bloomberg, Perplexity Finance) is **primary-document intelligence** —
answering from SEC filings, not just news. SEC EDGAR exposes all of it for free
with no API key:

- **Company facts (XBRL)** — as-reported financials (revenue, net income, margins,
  assets, cash, EPS) via ``data.sec.gov/api/xbrl/companyfacts`` — structured and
  reliable, no HTML parsing.
- **Submissions** — the recent-filings index (10-K/10-Q/8-K …) per company via
  ``data.sec.gov/submissions``; 8-K entries carry item codes (the catalyst type).
- **Full-text search** — keyword/phrase search across all filings since 2001 via
  ``efts.sec.gov/LATEST/search-index``, returning the exact filings that match.
- **The filing documents themselves** — fetched and reduced to text so an exact
  passage can be quoted and cited.

SEC's firewall 403s any request whose ``User-Agent`` lacks a contact email, and it
asks callers to stay under ~10 requests/second. Set ``SEC_EDGAR_UA`` to your own
"Name email" string; a working default is used otherwise. Every network access
goes through ``_fetch_json`` / ``_fetch_text``, which cache per-process and degrade
to empty on any failure, so an EDGAR outage returns a friendly "no data" rather
than aborting the turn — and the helpers are trivially monkeypatched in tests.
"""

from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
import urllib.request

# SEC endpoints (all keyless). CIK is zero-padded to 10 digits for the data.sec.gov
# JSON paths; the Archives document path uses the un-padded integer CIK.
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_FTS_URL = "https://efts.sec.gov/LATEST/search-index?{qs}"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

# SEC's firewall 403s requests whose User-Agent lacks a contact email, so the
# default embeds one on the reserved example.com documentation domain. SEC asks
# for a REAL contact — set SEC_EDGAR_UA (e.g. "Your Name you@domain.com") so heavy
# use is attributable to you.
_DEFAULT_UA = "financial-research-assistant admin@example.com"

_PER_SHARE = "USD/shares"  # XBRL unit key for per-share concepts (EPS)

# Same-process caches (cleared at process exit): the ticker→CIK map is a ~1MB file
# worth fetching once, and companyfacts/submissions are large per-company blobs.
_TICKER_CIK: dict[str, str] = {}
_JSON_CACHE: dict[str, dict] = {}


def _edgar_ua() -> str:
    return (os.environ.get("SEC_EDGAR_UA") or "").strip() or _DEFAULT_UA


def _request(url: str):
    return urllib.request.Request(url, headers={"User-Agent": _edgar_ua()})


def _fetch_json(url: str, timeout: float = 20.0) -> dict:
    """GET a JSON document from SEC with the required User-Agent, a same-process
    cache, and one retry; ``{}`` on any failure (network, non-JSON, unknown)."""
    if url in _JSON_CACHE:
        return _JSON_CACHE[url]
    for _ in range(2):  # one retry — SEC occasionally 5xx / rate-limits
        try:
            with urllib.request.urlopen(_request(url), timeout=timeout) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8", "replace"))
            if data:
                _JSON_CACHE[url] = data
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 — degrade to no-data, never abort the turn
            continue
    return {}


def _fetch_text(url: str, max_bytes: int = 6_000_000, timeout: float = 30.0) -> str:
    """GET a filing document as text (size-capped so a huge 10-K can't blow up
    memory), with the SEC User-Agent and one retry; ``""`` on any failure."""
    for _ in range(2):
        try:
            with urllib.request.urlopen(_request(url), timeout=timeout) as resp:  # noqa: S310
                raw = resp.read(max_bytes)
            return raw.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            continue
    return ""


def _load_ticker_map() -> dict[str, str]:
    """Ticker → zero-padded 10-digit CIK, from SEC's company_tickers.json (cached)."""
    if _TICKER_CIK:
        return _TICKER_CIK
    data = _fetch_json(_COMPANY_TICKERS_URL)
    for row in (data or {}).values():
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper()
        cik = row.get("cik_str")
        if ticker and cik is not None:
            _TICKER_CIK[ticker] = str(int(cik)).zfill(10)
    return _TICKER_CIK


def _cik_for(symbol: str) -> str | None:
    """Resolve a ticker to its 10-digit CIK, or ``None`` if unknown."""
    return _load_ticker_map().get(symbol.strip().upper())


def _filing_url(cik: str, accession: str, doc: str) -> str:
    """Build the public EDGAR Archives URL for a filing's primary document."""
    acc = accession.replace("-", "")
    try:
        cik_int = int(cik)
    except (TypeError, ValueError):
        return ""
    return _ARCHIVE_URL.format(cik=cik_int, acc=acc, doc=doc or "")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _money(v) -> str:
    """Compact money formatting: 383.29B / 12.3M / 1,234 / n/a."""
    n = _num(v)
    if n is None:
        return "n/a"
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{suffix}"
    return f"{n:,.0f}"


def _no_cik(sym: str) -> str:
    return (
        f"No SEC CIK found for {sym!r} — SEC EDGAR covers US-listed filers "
        f"(use the US ticker, e.g. AAPL). Foreign/ADR names may differ."
    )


# --- Recent filings (shared submissions accessor) --------------------------

def _fetch_submissions(cik: str) -> dict:
    return _fetch_json(_SUBMISSIONS_URL.format(cik=cik))


def _submission_recent(cik: str) -> tuple[str, list[dict]]:
    """(entity name, recent filings) from the submissions feed. Each filing is
    ``{date, form, accession, doc, desc, items}`` (``items`` = 8-K item codes)."""
    sub = _fetch_submissions(cik)
    recent = ((sub.get("filings") or {}).get("recent")) or {}
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    accns = recent.get("accessionNumber") or []
    docs = recent.get("primaryDocument") or []
    descs = recent.get("primaryDocDescription") or []
    items = recent.get("items") or []

    def at(seq, i):
        return seq[i] if i < len(seq) else ""

    out = [
        {"date": at(dates, i), "form": str(form), "accession": at(accns, i),
         "doc": at(docs, i), "desc": at(descs, i) or "", "items": at(items, i) or ""}
        for i, form in enumerate(forms)
    ]
    return sub.get("name") or "", out


def _filter_forms(filings: list[dict], ft: str, limit: int) -> list[dict]:
    """The first ``limit`` filings whose form matches ``ft`` (prefix, empty = any)."""
    out = []
    for f in filings:
        if ft and not f["form"].upper().startswith(ft):
            continue
        out.append(f)
        if len(out) >= limit:
            break
    return out


def sec_filings(symbol: str, form_type: str = "", limit: int = 10) -> str:
    """List a company's recent SEC filings from EDGAR (keyless) — form type, filing
    date, description, and a direct link to each document. ``form_type`` optionally
    filters to a form (e.g. ``"10-K"`` annual report, ``"10-Q"`` quarterly, ``"8-K"``
    material events, ``"4"`` insider trades) — empty = all recent forms. ``limit``
    caps the rows (default 10). Use for 'what has X filed / latest 10-K / recent
    8-Ks / annual report link / SEC filings for TICKER' questions. Pair with
    `sec_filing_search` (search inside filings) and `sec_financials` (the numbers)."""
    sym = symbol.strip().upper()
    cik = _cik_for(sym)
    if not cik:
        return _no_cik(sym)
    name, filings = _submission_recent(cik)
    if not filings:
        return f"No recent filings found for {sym} (CIK {int(cik)})."
    ft = form_type.strip().upper()
    rows = _filter_forms(filings, ft, max(1, min(int(limit or 10), 40)))
    if not rows:
        return f"No {ft or 'recent'} filings found for {name or sym} ({sym})."
    lines = [f"SEC filings · {name or sym} ({sym}, CIK {int(cik)})"
             + (f" · form {ft}" if ft else "")]
    for f in rows:
        tail = f"  {f['desc']}" if f["desc"] else ""
        lines.append(f"  {f['date']}  {f['form']:<8}{tail}\n      "
                     + _filing_url(cik, f["accession"], f["doc"]))
    lines.append("(Source: SEC EDGAR — primary filings, as filed.)")
    return "\n".join(lines)


# --- 8-K material events ---------------------------------------------------

# 8-K item codes → plain-English event type (the catalyst). Common items only;
# an unmapped code renders as its raw number.
_8K_ITEMS = {
    "1.01": "Entry into a Material Agreement",
    "1.02": "Termination of a Material Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition/Disposition of Assets",
    "2.02": "Results of Operations & Financial Condition (earnings)",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events Accelerating an Obligation",
    "2.05": "Costs Associated with Exit/Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting / Listing-Rule Failure",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Security Holders' Rights",
    "4.01": "Change in Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financials",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure/Election of Directors or Officers",
    "5.03": "Amendments to Articles / Bylaws",
    "5.07": "Submission of Matters to a Shareholder Vote",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements & Exhibits",
}


def _decode_items(codes: str) -> str:
    parts = [c.strip() for c in str(codes).split(",") if c.strip()]
    labels = [f"{c} {_8K_ITEMS[c]}" if c in _8K_ITEMS else c for c in parts]
    return "; ".join(labels)


def sec_material_events(symbol: str, limit: int = 10) -> str:
    """List a company's recent **8-K material-event filings** with the event type
    decoded from the SEC item codes — the catalyst monitor (earnings releases,
    M&A, executive departures, material agreements, impairments, guidance/other
    events), each with its filing date and a link. Keyless via SEC EDGAR. Use for
    'any material events / recent 8-Ks / what has X disclosed / material news /
    executive changes / did they file anything' questions about a US-listed
    company. For the numbers use `sec_financials`; to search inside filings use
    `sec_filing_search`."""
    sym = symbol.strip().upper()
    cik = _cik_for(sym)
    if not cik:
        return _no_cik(sym)
    name, filings = _submission_recent(cik)
    limit = max(1, min(int(limit or 10), 30))
    events = [f for f in filings if f["form"].upper().startswith("8-K")][:limit]
    if not events:
        return f"No recent 8-K material events found for {name or sym} ({sym})."
    lines = [f"SEC 8-K material events · {name or sym} ({sym}, CIK {int(cik)})"]
    for f in events:
        decoded = _decode_items(f["items"]) or f["desc"] or "(event type not specified)"
        lines.append(f"  {f['date']}  {f['form']:<6} {decoded}")
        lines.append("      " + _filing_url(cik, f["accession"], f["doc"]))
    lines.append("(Source: SEC EDGAR 8-K filings — item codes decoded. Open a link "
                 "to read the disclosure.)")
    return "\n".join(lines)


# --- As-reported financials (XBRL company facts) ---------------------------

# Curated key line items, each a fallback list of us-gaap XBRL tags (the same
# concept is tagged differently across filers/years — try each in order).
_KEY_CONCEPTS: list[tuple[str, list[str], str]] = [
    ("Revenue", ["RevenueFromContractWithCustomerExcludingAssessedTax",
                 "Revenues", "SalesRevenueNet"], "USD"),
    ("Gross profit", ["GrossProfit"], "USD"),
    ("Operating income", ["OperatingIncomeLoss"], "USD"),
    ("Net income", ["NetIncomeLoss"], "USD"),
    ("Diluted EPS", ["EarningsPerShareDiluted"], _PER_SHARE),
    ("Total assets", ["Assets"], "USD"),
    ("Total liabilities", ["Liabilities"], "USD"),
    ("Stockholders equity", ["StockholdersEquity"], "USD"),
    ("Cash & equivalents", ["CashAndCashEquivalentsAtCarryingValue"], "USD"),
]


def _fetch_company_facts(cik: str) -> dict:
    return _fetch_json(_COMPANY_FACTS_URL.format(cik=cik))


def _concept_units(facts: dict, tags: list[str], unit: str) -> list[dict] | None:
    """Return the raw unit entries for the first matching us-gaap tag, or None."""
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    for tag in tags:
        node = gaap.get(tag)
        if node:
            units = (node.get("units") or {}).get(unit)
            if units:
                return units
    return None


def _annual(entries: list[dict]) -> list[dict]:
    """The latest annual (10-K, full-year) value per fiscal year, newest first.
    Keeps one entry per fiscal year (the one with the latest period end), so a
    concept restated in a later filing doesn't produce duplicate years."""
    by_fy: dict = {}
    for e in entries:
        if not str(e.get("form") or "").startswith("10-K"):
            continue
        if e.get("fp") not in (None, "FY"):
            continue
        fy = e.get("fy")
        if fy is None:
            continue
        cur = by_fy.get(fy)
        if cur is None or str(e.get("end") or "") > str(cur.get("end") or ""):
            by_fy[fy] = e
    return [by_fy[k] for k in sorted(by_fy, reverse=True)]


def _fmt_val(val, unit: str) -> str:
    if unit == _PER_SHARE:
        n = _num(val)
        return f"{n:.2f}" if n is not None else "n/a"
    return _money(val)


def _annual_series(facts: dict, years: int) -> tuple[dict, list]:
    """Per-concept annual rows keyed by fiscal year, plus the newest ``years``
    fiscal years present across all concepts."""
    series: dict[str, dict] = {}
    fys: list = []
    for label, tags, unit in _KEY_CONCEPTS:
        units = _concept_units(facts, tags, unit)
        rows = {r.get("fy"): r for r in (_annual(units)[:years] if units else [])}
        series[label] = rows
        for fy in rows:
            if fy not in fys:
                fys.append(fy)
    return series, sorted(fys, reverse=True)[:years]


def _financials_summary(facts: dict, sym: str, name: str, cik: str, years: int) -> str:
    series, fys = _annual_series(facts, years)
    if not fys:
        return f"No annual (10-K) XBRL figures found for {sym} (CIK {int(cik)})."
    unit_by_label = {lbl: unit for lbl, _, unit in _KEY_CONCEPTS}
    col = 14
    lines = [f"SEC financials · {name} ({sym}, CIK {int(cik)}) — as-reported (10-K XBRL)",
             f"{'':<20}" + "".join(f"{('FY' + str(fy)):>{col}}" for fy in fys)]
    for label, _, _ in _KEY_CONCEPTS:
        rows, unit = series[label], unit_by_label[label]
        cells = "".join(
            f"{(_fmt_val(rows[fy].get('val'), unit) if fy in rows else '—'):>{col}}"
            for fy in fys
        )
        lines.append(f"{label:<20}{cells}")
    lines.append("")
    lines.append("Margins & growth:")
    lines.extend(_margin_lines(series, fys))
    lines.append("(Source: SEC EDGAR XBRL company facts — audited as-reported figures.)")
    return "\n".join(lines)


def _margin_line(series: dict, fy) -> str:
    rev = _num((series["Revenue"].get(fy) or {}).get("val"))
    gp = _num((series["Gross profit"].get(fy) or {}).get("val"))
    ni = _num((series["Net income"].get(fy) or {}).get("val"))
    parts = []
    if rev and gp is not None:
        parts.append(f"gross {gp / rev * 100:.1f}%")
    if rev and ni is not None:
        parts.append(f"net {ni / rev * 100:.1f}%")
    return f"  FY{fy}: {' · '.join(parts) if parts else 'n/a'}"


def _margin_lines(series: dict, fys: list) -> list[str]:
    rev_rows = series["Revenue"]
    out = [_margin_line(series, fy) for fy in fys]
    ordered = [fy for fy in fys if fy in rev_rows]
    for newer, older in zip(ordered, ordered[1:]):
        rn, ro = _num(rev_rows[newer].get("val")), _num(rev_rows[older].get("val"))
        if rn and ro:
            out.append(f"  revenue growth FY{older}→FY{newer}: {(rn - ro) / ro * 100:+.1f}%")
    return out


def sec_financials(symbol: str, concept: str = "", years: int = 4) -> str:
    """As-reported annual financials for a company from SEC XBRL data (keyless, no
    HTML parsing) — the real audited numbers from its 10-K filings. With no
    ``concept``, returns a summary across recent fiscal years: revenue, gross/
    operating/net income, diluted EPS, assets, liabilities, equity, cash — plus
    computed gross/net margins and revenue growth. Pass a ``concept`` (a us-gaap
    tag like ``NetIncomeLoss`` or ``Assets``) to get just that line's history.
    ``years`` caps how many fiscal years (default 4). Use for 'X's revenue /
    margins / net income / balance sheet / financials over time / as-reported
    numbers'. These are audited filing figures — more authoritative than the Yahoo
    `stock_fundamentals` snapshot; cite them to the 10-K."""
    sym = symbol.strip().upper()
    cik = _cik_for(sym)
    if not cik:
        return _no_cik(sym)
    facts = _fetch_company_facts(cik)
    if not facts or not facts.get("facts"):
        return f"No XBRL company facts found for {sym} (CIK {int(cik)})."
    name = facts.get("entityName") or sym
    years = max(1, min(int(years or 4), 10))
    if concept.strip():
        return _one_concept(facts, sym, name, concept.strip(), years)
    return _financials_summary(facts, sym, name, cik, years)


def _one_concept(facts: dict, sym: str, name: str, concept: str, years: int) -> str:
    """History for a single us-gaap concept tag (annual 10-K values)."""
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    node = gaap.get(concept)
    if not node:
        available = ", ".join(sorted(gaap)[:12])
        return (
            f"No us-gaap concept {concept!r} for {sym}. Examples available: "
            f"{available}… (use an exact XBRL tag like NetIncomeLoss, Assets)."
        )
    units = node.get("units") or {}
    unit_key = _PER_SHARE if _PER_SHARE in units else next(iter(units), "USD")
    rows = _annual(units.get(unit_key) or [])[:years]
    if not rows:
        return f"No annual (10-K) values for {concept} on {sym}."
    lines = [f"{name} ({sym}) · {concept} ({unit_key}), annual:"]
    for r in rows:
        lines.append(f"  FY{r.get('fy')} (ended {r.get('end')}): "
                     f"{_fmt_val(r.get('val'), unit_key)}")
    lines.append("(Source: SEC EDGAR XBRL company facts.)")
    return "\n".join(lines)


# --- Full-text search across filings ---------------------------------------

def _fts(query: str, forms: str, cik: str | None) -> dict:
    params = {"q": query}
    if forms.strip():
        params["forms"] = forms.strip()
    if cik:
        params["ciks"] = cik
    return _fetch_json(_FTS_URL.format(qs=urllib.parse.urlencode(params)))


def _fts_header(query: str, sym: str, forms: str, total) -> str:
    scope = f" · {sym}" if sym else ""
    form_note = f" · forms {forms}" if forms.strip() else ""
    total_note = f"  (~{total} total hits)" if total else ""
    return f"SEC full-text search · {query!r}{scope}{form_note}{total_note}"


def _format_fts_hit(hit: dict, fallback_cik: str | None) -> list[str]:
    src = hit.get("_source") or {}
    acc, _, doc = (hit.get("_id") or "").partition(":")
    ciks = src.get("ciks") or []
    cik0 = ciks[0] if ciks else (fallback_cik or "")
    names = src.get("display_names") or []
    who = names[0] if names else (src.get("entityName") or "")
    form = src.get("form") or (src.get("root_forms") or [""])[0]
    rows = [f"  {src.get('file_date') or ''}  {str(form):<8} {who}"]
    url = _filing_url(cik0, acc, doc) if (cik0 and acc and doc) else ""
    if url:
        rows.append(f"      {url}")
    return rows


def sec_filing_search(query: str, symbol: str = "", forms: str = "", limit: int = 5) -> str:
    """Full-text search across SEC filings (EDGAR, keyless) — find the exact filings
    that mention a phrase, so a claim can be cited to a primary source. ``query`` is
    a keyword or "quoted phrase"; ``symbol`` optionally restricts to one company;
    ``forms`` optionally restricts to a form (e.g. ``"10-K"``, ``"8-K"``).
    Returns the matching filings (company, form, date, document link). Use for
    'which filings mention X / find the 10-K that discusses <risk/segment/topic> /
    has TICKER disclosed <thing> in its filings' questions, then use
    `sec_filing_excerpt` (or open the link) to quote the exact language. Covers
    filings since 2001."""
    q = query.strip()
    if not q:
        return "Give a keyword or \"quoted phrase\" to search SEC filings for."
    sym = symbol.strip().upper()
    cik = _cik_for(sym) if sym else None
    if sym and not cik:
        return f"No SEC CIK found for {sym!r} — omit the symbol or use a US ticker."
    data = _fts(q, forms, cik)
    hits = ((data.get("hits") or {}).get("hits")) or []
    if not hits:
        scope = f" for {sym}" if sym else ""
        return f"No SEC filings{scope} matched {query!r}."
    total = ((data.get("hits") or {}).get("total") or {}).get("value")
    limit = max(1, min(int(limit or 5), 20))
    lines = [_fts_header(query, sym, forms, total)]
    for hit in hits[:limit]:
        lines.extend(_format_fts_hit(hit, cik))
    lines.append("(Source: SEC EDGAR full-text search. Use `sec_filing_excerpt` to "
                 "quote the exact text.)")
    return "\n".join(lines)


# --- Passage extraction from a filing document -----------------------------

def _html_to_text(raw: str) -> str:
    """Reduce filing HTML to plain text with paragraph breaks: drop script/style,
    turn block-close tags and <br> into newlines, strip the rest, unescape
    entities, and collapse intra-line whitespace."""
    raw = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)</(p|div|tr|li|h[1-6]|table|section)\s*>", "\n", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    lines = (re.sub(r"[ \t ]+", " ", ln).strip() for ln in raw.split("\n"))
    return "\n".join(ln for ln in lines if ln)


def _passages(text: str, query: str, top: int) -> list[str]:
    """Rank the document's paragraphs by how many distinct query terms they contain
    (ties broken by total occurrences); return the best ``top`` substantive ones."""
    terms = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
    if not terms:
        return []
    scored = []
    for para in text.split("\n"):
        if len(para) < 120:  # skip headers/labels/boilerplate one-liners
            continue
        low = para.lower()
        matched = sum(1 for t in terms if t in low)
        if matched:
            occ = sum(low.count(t) for t in terms)
            scored.append((matched, occ, para))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [p for _, _, p in scored[:top]]


def sec_filing_excerpt(symbol: str, query: str, form_type: str = "10-K",
                       max_passages: int = 3) -> str:
    """Fetch a company's most recent filing of a given type and return the passages
    that best match a query — the exact filing language to quote and cite (the
    grounding step that follows `sec_filing_search`). ``form_type`` picks the filing
    (default ``"10-K"``; e.g. ``"10-Q"``, ``"8-K"``). Use for 'what does X's 10-K
    say about <topic> / quote their disclosure on <risk/segment/guidance> / find the
    exact language on <thing>'. Returns the source filing (date + link) and the
    matching paragraphs. Keyless via SEC EDGAR; US-listed filers only."""
    sym = symbol.strip().upper()
    if not query.strip():
        return "Give a topic/phrase to find in the filing (e.g. 'supply chain risk')."
    cik = _cik_for(sym)
    if not cik:
        return _no_cik(sym)
    name, filings = _submission_recent(cik)
    ft = form_type.strip().upper() or "10-K"
    match = next((f for f in filings if f["form"].upper().startswith(ft)), None)
    if not match:
        return f"No recent {ft} filing found for {name or sym} ({sym})."
    url = _filing_url(cik, match["accession"], match["doc"])
    doc = _fetch_text(url)
    if not doc:
        return (f"Couldn't fetch the {ft} document for {sym} (it may be very large "
                f"or unavailable). Filing: {url}")
    passages = _passages(_html_to_text(doc), query, max(1, min(int(max_passages or 3), 5)))
    if not passages:
        return (f"No passage in {sym}'s {ft} ({match['date']}) matched {query!r}. "
                f"Try different terms. Filing: {url}")
    lines = [f"Excerpts · {name or sym} ({sym}) {ft} filed {match['date']} · matching {query!r}",
             f"  {url}", ""]
    for i, p in enumerate(passages, 1):
        lines.append(f"[{i}] {p[:800].strip()}" + ("…" if len(p) > 800 else ""))
        lines.append("")
    lines.append("(Source: SEC EDGAR filing text — quote these verbatim and cite the "
                 "filing + date. Delayed/as-filed; not advice.)")
    return "\n".join(lines).rstrip()


# --- Structured filing summary (fixed-slot tearsheet) ----------------------

# The fixed slots of a filing tearsheet, each with the query terms that locate the
# relevant passages in the document. A predictable schema (not free-form) is what
# makes summaries comparable and skimmable — the Bloomberg/AlphaSense pattern.
_SUMMARY_TOPICS: list[tuple[str, str]] = [
    ("Business & segments", "business segment products services markets operations"),
    ("Revenue & growth drivers", "revenue net sales growth increased driver demand"),
    ("Margins & profitability", "gross margin operating margin cost of sales profitability"),
    ("Outlook / guidance", "outlook expect future anticipate believe guidance fiscal"),
    ("Capital allocation", "dividend repurchase buyback capital expenditures returned shareholders"),
    ("Key risks", "risk adversely affect could harm uncertainty depend"),
]


def filing_summary(symbol: str, form_type: str = "10-K") -> str:
    """Build a structured **tearsheet** of a company's latest filing (default
    10-K): the relevant verbatim passages grouped under fixed slots — Business &
    segments, Revenue & growth drivers, Margins & profitability, Outlook/guidance,
    Capital allocation, Key risks — for you to synthesize into a fixed-slot summary
    (fill each slot from its passages, cite them, and write 'not disclosed' for a
    slot with no evidence). Keyless via SEC EDGAR. Use for 'summarize X's 10-K /
    give me a tearsheet on X's filing / what are the key points of their annual
    report'. For the raw numbers use `sec_financials`; for one specific topic use
    `sec_filing_excerpt`."""
    sym = symbol.strip().upper()
    cik = _cik_for(sym)
    if not cik:
        return _no_cik(sym)
    name, filings = _submission_recent(cik)
    ft = form_type.strip().upper() or "10-K"
    match = next((f for f in filings if f["form"].upper().startswith(ft)), None)
    if not match:
        return f"No recent {ft} filing found for {name or sym} ({sym})."
    url = _filing_url(cik, match["accession"], match["doc"])
    text = _html_to_text(_fetch_text(url))
    if not text:
        return f"Couldn't fetch the {ft} document for {sym}. Filing: {url}"
    lines = [
        f"Structured summary · {name or sym} ({sym}) {ft} filed {match['date']} — "
        f"synthesize a tearsheet: fill EACH slot below from its passages, cite them, "
        f"and write 'not disclosed' for any slot with no evidence. Delayed as-filed "
        f"data, not advice.",
        f"  {url}",
    ]
    found_any = False
    for slot, terms in _SUMMARY_TOPICS:
        passages = _passages(text, terms, 2)
        lines.append(f"\n## {slot}")
        if passages:
            found_any = True
            for p in passages:
                lines.append(f"- {p[:600].strip()}" + ("…" if len(p) > 600 else ""))
        else:
            lines.append("- (no matching passage found in this filing)")
    if not found_any:
        return (f"Fetched {sym}'s {ft} ({match['date']}) but couldn't locate the "
                f"usual sections in it. Filing: {url}")
    lines.append("\n(Source: SEC EDGAR filing text — quote verbatim and cite the filing + date.)")
    return "\n".join(lines)


# SEC EDGAR filing-intelligence tools, appended to tools.TOOLS.
EDGAR_TOOLS = [
    sec_filings,
    sec_material_events,
    sec_financials,
    sec_filing_search,
    sec_filing_excerpt,
    filing_summary,
]
