# Tools Reference

Every agent tool the assistant can call, grouped by area. **57 tools.**

You don't call these directly — you ask the assistant in plain English and it
picks the tool(s). Each entry below shows:

- **Signature** — the tool's name and arguments (with defaults). `0` / `""` for a
  numeric/string argument usually means "off / use default".
- **Ask** — an example natural-language request that routes to it.
- **Call** — the equivalent underlying tool call (what the model emits).

**Data sources & keys.** Almost everything is **keyless**: Yahoo Finance (quotes,
fundamentals, options), SEC EDGAR (filings/XBRL), Ken French (factors),
DuckDuckGo (news). Optional upgrades: `TAVILY_API_KEY` (better news),
`OPENAI_API_BASE`/`KEY` (embeddings for semantic document search), the
`[documents]` extra (PDF ingest), the `[ofx]` extra (OFX/QFX statements), and a
live **IBKR** MCP session (real-time quotes/positions — read-only). Portfolio
tools need a broker statement imported first.

---

## Table of contents

1. [Market data & charts](#1-market-data--charts)
2. [Quick calculators](#2-quick-calculators)
3. [Fundamentals & company analysis](#3-fundamentals--company-analysis)
4. [Valuation & options](#4-valuation--options)
5. [Deep research & debate](#5-deep-research--debate)
6. [SEC EDGAR filing intelligence](#6-sec-edgar-filing-intelligence)
7. [Stock screener](#7-stock-screener)
8. [Portfolio & statements](#8-portfolio--statements)
9. [Advanced portfolio analytics](#9-advanced-portfolio-analytics)
10. [Document upload (RAG)](#10-document-upload-rag)
11. [Subagent delegation](#11-subagent-delegation)

---

## 1. Market data & charts

#### `current_date()`
Current local date **and** wall-clock time with timezone — the "as of" stamp for any quote or note.
- **Ask:** "What's today's date and time?"
- **Call:** `current_date()`

#### `price_history_chart(symbol, days=90)`
Daily closing prices for one ticker as a terminal line chart with summary stats (first/last/high/low, % change). Keyless (Yahoo).
- **Ask:** "Show me Apple's price over the last 6 months."
- **Call:** `price_history_chart(symbol="AAPL", days=180)`

#### `compare_prices(symbols, days=180)`
Several stocks on one **normalized** chart (each rebased to 100), plus each ticker's total return — comparable regardless of share price.
- **Ask:** "Compare NVDA vs AMD vs the S&P over the past year."
- **Call:** `compare_prices(symbols="NVDA, AMD, SPY", days=365)`

#### `risk_metrics(symbol, days=365, benchmark="SPY")`
Per-ticker risk/return from daily returns: annualized volatility, max drawdown, Sharpe (rf=0), and beta vs a benchmark.
- **Ask:** "How risky is TSLA — volatility, drawdown, beta?"
- **Call:** `risk_metrics(symbol="TSLA", days=365, benchmark="SPY")`

#### `web_search(query, max_results=6)`
Recent news/headlines (title, source, date, snippet, URL). Keyless DuckDuckGo; uses Tavily when `TAVILY_API_KEY` is set.
- **Ask:** "Any recent news on Nvidia's data-center demand?"
- **Call:** `web_search(query="Nvidia data center demand", max_results=6)`

---

## 2. Quick calculators

#### `pct_change(start_price, end_price)`
Percent change between two prices (100→125 returns 25.0).
- **Ask:** "What's the percent gain from $100 to $125?"
- **Call:** `pct_change(start_price=100, end_price=125)`

#### `cagr(start_value, end_value, years)`
Compound annual growth rate (%) over a number of years.
- **Ask:** "If $10k grew to $18k over 5 years, what's the CAGR?"
- **Call:** `cagr(start_value=10000, end_value=18000, years=5)`

#### `position_weight(position_value, portfolio_value)`
A position's weight as a % of the portfolio ($25k in $200k → 12.5).
- **Ask:** "What percent of a $200k book is a $25k position?"
- **Call:** `position_weight(position_value=25000, portfolio_value=200000)`

#### `convert_currency(amount, from_currency, on_date="")`
Convert an amount to USD using market FX (Yahoo). `on_date` uses that day's historical rate.
- **Ask:** "What's €5,000 in USD?"
- **Call:** `convert_currency(amount=5000, from_currency="EUR")`

---

## 3. Fundamentals & company analysis

#### `stock_fundamentals(symbol)`
Snapshot: profile (sector/industry), valuation (market cap, P/E, EPS), price + 52-week range, dividend, beta, one-line analyst summary. Keyless (Yahoo).
- **Ask:** "Give me the fundamentals on Microsoft."
- **Call:** `stock_fundamentals(symbol="MSFT")`

#### `analyst_ratings(symbol)`
Buy/hold/sell consensus, price targets (low/mean/median/high) with implied upside, recent upgrades/downgrades.
- **Ask:** "What do analysts think of AMD? Price target?"
- **Call:** `analyst_ratings(symbol="AMD")`

#### `earnings_calendar(symbol)`
Next earnings date + EPS estimate, upcoming ex-dividend/pay dates, and recent quarters (estimate vs reported, surprise %).
- **Ask:** "When does Apple report next, and how have they done recently?"
- **Call:** `earnings_calendar(symbol="AAPL")`

#### `compare_stocks(symbols)`
2–4 stocks side by side: price, market cap, P/E, PEG, P/S, revenue growth, margin, EPS, yield, beta, analyst consensus + target. Keyless (Yahoo snapshot).
- **Ask:** "Compare AAPL, MSFT and NVDA — which is cheaper and growing faster?"
- **Call:** `compare_stocks(symbols="AAPL, MSFT, NVDA")`

#### `etf_exposure(symbol)`
Look through an ETF/fund to sector weightings and top holdings. Funds only.
- **Ask:** "What's inside VOO — sectors and top holdings?"
- **Call:** `etf_exposure(symbol="VOO")`

#### `dividend_projection(account="")`
Forward 12-month dividend income across your imported holdings — per-holding income, yield-on-cost, current yield, FX-converted total. **Needs an imported statement.**
- **Ask:** "How much dividend income should my portfolio generate next year?"
- **Call:** `dividend_projection()`

---

## 4. Valuation & options

#### `dcf_valuation(symbol, growth_rate=0, discount_rate=0, terminal_growth=0, years=0)`
Deterministic **two-stage DCF** intrinsic value. FCF history as-reported from the 10-K (SEC XBRL: operating cash flow − capex); net debt/shares/price from Yahoo. Shows every input, projected cash flows + present values, terminal value, intrinsic value/share, upside vs price, and a sensitivity grid. Rates accept `0.10` or `10`. Pass `0` to use defaults (growth = historical FCF CAGR, discount 9%, terminal 2.5%, 5y). A model, not a price target; declines for banks / pre-FCF companies.
- **Ask:** "Run a DCF on Apple — is it over- or undervalued?"
- **Call:** `dcf_valuation(symbol="AAPL")` · custom: `dcf_valuation(symbol="MSFT", discount_rate=10, terminal_growth=2.5, years=7)`

#### `explain_option(symbol, expiry="", strike=0, option_type="call")`
Plain-language single-leg option economics over the live Yahoo option chain: premium + per-contract cost, bid/ask/last, IV and the IV-implied move, intrinsic vs. time value, moneyness, breakeven (+ % move to reach it), max loss, max profit. Omit `strike` for a near-the-money chain slice; omit `expiry` for the nearest. Estimate at expiry, not advice.
- **Ask:** "Explain the Apple $320 call expiring next month — breakeven and max loss?"
- **Call:** `explain_option(symbol="AAPL", expiry="2026-08-21", strike=320, option_type="call")` · slice: `explain_option(symbol="AAPL", option_type="put")`

---

## 5. Deep research & debate

These **gather findings** and hand them to the assistant to synthesize (with citations) — they don't write the answer themselves.

#### `research_report(symbol)`
Broad gather on one ticker — price, fundamentals, analyst, earnings, risk, ETF look-through (if a fund), news — for a structured, cited deep-dive write-up.
- **Ask:** "Give me a full research report on Nvidia."
- **Call:** `research_report(symbol="NVDA")`

#### `explain_stock_move(symbol, days=5)`
Evidence for **why** a stock moved: the measured move (last session + window), recent rating changes, and news — to attribute it with numbered citations (or say it's unexplained).
- **Ask:** "Why is Tesla down today?"
- **Call:** `explain_stock_move(symbol="TSLA", days=5)`

#### `bull_bear_debate(symbol)`
Findings framed for an adversarial **bull vs bear** debate + a verdict (lean, rough confidence, open questions, what would change it).
- **Ask:** "Make the case for and against buying Meta."
- **Call:** `bull_bear_debate(symbol="META")`

---

## 6. SEC EDGAR filing intelligence

Primary-source, **audited/as-reported** data straight from SEC EDGAR — keyless. Cite claims to the filing + date. (Set `SEC_EDGAR_UA` to your name+email for heavy use.)

#### `sec_filings(symbol, form_type="", limit=10)`
Recent filings — form, date, description, direct document link. Filter by form (`10-K`, `10-Q`, `8-K`, `4`…).
- **Ask:** "List Apple's latest 10-K and recent 8-Ks."
- **Call:** `sec_filings(symbol="AAPL", form_type="10-K", limit=5)`

#### `sec_material_events(symbol, limit=10)`
Recent **8-K** filings with the event type decoded from item codes (earnings, M&A, exec departures, agreements, impairments…).
- **Ask:** "Any material events for Nvidia recently?"
- **Call:** `sec_material_events(symbol="NVDA", limit=10)`

#### `insider_transactions(symbol, limit=15)`
Recent **insider** activity from Form 4 ownership XML — separates open-market **buys (P)** and **sales (S)**, the conviction signals, from routine grants/exercises/tax-withholding, and reports the net. Buying is the rarer, stronger signal.
- **Ask:** "Are insiders buying or selling Nvidia?"
- **Call:** `insider_transactions(symbol="NVDA", limit=15)`

#### `sec_financials(symbol, concept="", years=4)`
As-reported annual financials from XBRL: revenue, gross/operating/net income, EPS, balance sheet, cash + computed margins & growth. Pass a `concept` (us-gaap tag) for one line's history.
- **Ask:** "What was Apple's revenue and net margin over the last 4 years, per their 10-K?"
- **Call:** `sec_financials(symbol="AAPL", years=4)` · one line: `sec_financials(symbol="AAPL", concept="NetIncomeLoss")`

#### `sec_quarterly_financials(symbol, concept="", quarters=8)`
The **quarterly** (10-Q) companion to `sec_financials` — the same line items across recent quarters (newest first), with per-quarter margins and revenue QoQ + YoY growth. Columns are labeled by period-end date; the fiscal-year-end quarter may be absent (the 10-K reports it as the full year).
- **Ask:** "How has Nvidia's revenue trended over the last four quarters?"
- **Call:** `sec_quarterly_financials(symbol="NVDA", quarters=8)`

#### `sec_filing_search(query, symbol="", forms="", limit=5)`
Full-text search across filings (since 2001) — find the filings that mention a phrase, so a claim is citable.
- **Ask:** "Which filings mention 'supply chain concentration' for Apple?"
- **Call:** `sec_filing_search(query="supply chain concentration", symbol="AAPL", forms="10-K")`

#### `sec_filing_excerpt(symbol, query, form_type="10-K", max_passages=3)`
Fetch the latest filing of a type and return the best-matching **verbatim passages** to quote — the grounding step after a search.
- **Ask:** "What does Apple's 10-K say about regulatory risk? Quote it."
- **Call:** `sec_filing_excerpt(symbol="AAPL", query="regulatory risk", form_type="10-K")`

#### `filing_summary(symbol, form_type="10-K")`
A structured **tearsheet**: verbatim passages grouped under fixed slots — Business, Revenue drivers, Margins, Outlook, Capital allocation, Key risks.
- **Ask:** "Summarize Microsoft's latest 10-K into a tearsheet."
- **Call:** `filing_summary(symbol="MSFT", form_type="10-K")`

#### `compare_sec_financials(symbols, concept="", years=3)`
Companies × metrics **matrix** from 10-K XBRL (audited — distinct from the Yahoo `compare_stocks`). 2–6 tickers. Pass a `concept` for one metric across companies over time.
- **Ask:** "Compare the audited revenue and margins of AAPL, MSFT and NVDA."
- **Call:** `compare_sec_financials(symbols="AAPL, MSFT, NVDA", years=3)` · one metric: `compare_sec_financials(symbols="AAPL, MSFT", concept="Net income")`

#### `filing_tone_trend(symbol, years=3)`
Negative-word **density** (Loughran-McDonald finance lexicon) across recent 10-Ks — is management's language getting more cautious? A heuristic, not a judgment. (Slower — one multi-MB fetch per year.)
- **Ask:** "Is Apple's 10-K tone getting more negative over time?"
- **Call:** `filing_tone_trend(symbol="AAPL", years=3)`

#### `sec_metric_rank(symbol, concept="Revenue", year=0)`
Where a company **ranks** on a metric among all SEC filers — its value, rank/percentile, and peer median.
- **Ask:** "Where does Apple's net income rank among all filers?"
- **Call:** `sec_metric_rank(symbol="AAPL", concept="Net income")`

---

## 7. Stock screener

#### `screen_stocks(symbols="", universe="", min_market_cap_b=0, max_market_cap_b=0, near_high_pct=0, near_high_within_days=10, high_lookback_days=1825, market_down_pct=0, market_symbol="SPY", min_earnings_beats=0, sector="", max_symbols=40)`
Screen a **candidate universe** against quantitative criteria and return passers with the measured figures. Universe (first match wins): explicit `symbols`, a named `universe` (`"sp500"` / `"largecap"`), or the built-in large-cap default. Any criterion set to `0`/`""` is off. Criteria: market-cap bounds, proximity-to-high (`near_high_pct` within `near_high_within_days`, high over `high_lookback_days`), relative strength on a down-market day (`market_down_pct`), EPS-beat streak (`min_earnings_beats`), `sector`. **Quantitative only** — it reminds you to confirm qualitative claims (guidance beats) per name.
- **Ask:** "Find large-caps (>$10B) within 1% of their high in the last two weeks that beat EPS the last 2 quarters."
- **Call:** `screen_stocks(min_market_cap_b=10, near_high_pct=1.0, near_high_within_days=10, min_earnings_beats=2)`
- **S&P 500:** `screen_stocks(universe="sp500", min_market_cap_b=50, max_symbols=500)`

---

## 8. Portfolio & statements

Import a broker statement once, then everything below reads from the local store.

#### `import_ibkr_statement(path)`
Import a broker statement — auto-detects an **IBKR Activity Statement CSV** or a cross-broker **OFX/QFX** file (needs the `[ofx]` extra). Extracts trades, cash flows, positions, instruments. Re-importing the same account+period replaces it.
- **Ask:** "Import my statement at ~/Downloads/activity.csv."
- **Call:** `import_ibkr_statement(path="~/Downloads/activity.csv")`

#### `query_transactions(kind="", symbol="", limit=100, account="")`
Read transactions back. `kind` ∈ trade / dividend / withholding_tax / fee / deposit_withdrawal / corporate_action.
- **Ask:** "Show my Amazon trades."
- **Call:** `query_transactions(kind="trade", symbol="AMZN")`

#### `query_portfolio(symbol="", account="")`
Portfolio snapshot from the newest import: positions (qty, cost basis, value, unrealized P/L, name, ISIN), NAV by asset class, time-weighted return.
- **Ask:** "What are my current positions?"
- **Call:** `query_portfolio()`

#### `portfolio_value_history(account="")`
Total account value (NAV) over time, stitched from imported statements' NAV snapshots. Includes deposits/withdrawals.
- **Ask:** "Chart my account value over time."
- **Call:** `portfolio_value_history()`

#### `portfolio_performance_chart(account="")`
Investment **performance** independent of deposits — each statement's TWRR compounded into a "growth of 100" index.
- **Ask:** "How are my investments actually doing, excluding deposits?"
- **Call:** `portfolio_performance_chart()`

#### `portfolio_vs_benchmark(benchmark="SPY", account="")`
Your deposit-independent performance (TWRR) vs a benchmark over the same span.
- **Ask:** "Am I beating the S&P?"
- **Call:** `portfolio_vs_benchmark(benchmark="SPY")`

#### `realized_gains(year=0, symbol="", account="")`
Realized capital gains via FIFO lot matching, split short-term vs long-term, net of commissions. `year` filters by realization year.
- **Ask:** "What were my realized gains in 2025?"
- **Call:** `realized_gains(year=2025)`

#### `income_summary(year=0, account="")`
Cash income: dividends, withholding tax, fees — netted, grouped by currency, plus dividends by symbol.
- **Ask:** "How much did I earn in dividends last year?"
- **Call:** `income_summary(year=2025)`

#### `allocation(account="")`
Allocation & concentration: each position's weight, largest position, top-5 concentration, breakdown by asset category.
- **Ask:** "How concentrated is my portfolio?"
- **Call:** `allocation()`

#### `export_data(kind, path, account="")`
Export to CSV for Excel/tax software. `kind` = `transactions` or `positions`. Writes only inside the export directory.
- **Ask:** "Export my transactions to a CSV."
- **Call:** `export_data(kind="transactions", path="2025-transactions.csv")`

---

## 9. Advanced portfolio analytics

#### `tax_loss_harvest(account="", min_loss=0, short_term_rate=0.35, long_term_rate=0.15)`
Open lots now below cost basis — the harvestable losses, split short/long term, estimated tax benefit, and **wash-sale flags** (bought the same symbol within 30 days). Read-only, not tax advice.
- **Ask:** "Which positions could I harvest for tax losses?"
- **Call:** `tax_loss_harvest(min_loss=100)`

#### `correlation_matrix(symbols="", days=180)`
Correlation of daily returns across tickers (or your holdings when empty) — a quick diversification read.
- **Ask:** "How correlated are my holdings?"
- **Call:** `correlation_matrix(symbols="AAPL, MSFT, SPY", days=180)`

#### `portfolio_lookthrough(account="")`
**True** exposure by looking through ETFs to their holdings — a real sector breakdown and single-stock exposure that surfaces hidden concentration (a name held directly *and* via ETFs).
- **Ask:** "What's my true tech exposure after looking through my ETFs?"
- **Call:** `portfolio_lookthrough()`

#### `factor_exposure(symbol="", days=365, five_factor=False)`
Fama-French factor exposure for a ticker (or the whole portfolio when empty): market/size(SMB)/value(HML) loadings — plus RMW/CMA in 5-factor — with annualized alpha and R². Keyless (Ken French).
- **Ask:** "What's my portfolio's value vs growth tilt — is my alpha real?"
- **Call:** `factor_exposure(five_factor=True)` · one ticker: `factor_exposure(symbol="IWM")`

#### `portfolio_risk(account="", days=365, benchmark="SPY")`
Whole-**portfolio** risk (the companion to per-ticker `risk_metrics`): value-weights your current holdings into one synthetic daily return series, then reports annualized volatility, max drawdown, Sharpe, and beta. Holdings held constant over the window; reports the covered share of portfolio value.
- **Ask:** "How risky is my overall portfolio — its volatility and drawdown?"
- **Call:** `portfolio_risk(days=365, benchmark="SPY")`

#### `portfolio_digest(account="", lookback_days=5, move_threshold=5.0, earnings_within=14, include_news=False)`
Monitoring digest over your holdings: any triggered **alert rules** (see below), price movers beyond ±threshold, upcoming earnings/ex-dividends, optionally a headline per mover.
- **Ask:** "Anything I should know about my portfolio this week?"
- **Call:** `portfolio_digest(lookback_days=5, move_threshold=5, include_news=True)`

#### `add_alert(symbol, kind, value=0)` · `list_alerts()` · `remove_alert(alert_id)`
Standing **alert rules** the digest checks each run. `kind`: `drop`/`rise`/`move` (percent move over the lookback), `below`/`above` (a price level), or `earnings` (days out). `symbol="*"` means any holding. Triggered alerts appear at the top of `portfolio_digest` and the `--digest` CLI, and are pushed as they fire — an OS notification + sound + toast + a 🔔 transcript line in the TUI, stderr in the headless CLI. `FINANCIAL_RESEARCH_ALERT_SOUND=0` silences the sound (or point it at an audio file to pick your own); `FINANCIAL_RESEARCH_ALERT_DESKTOP=0` turns off the OS banner.
- **Ask:** "Tell me if AAPL drops more than 5%." / "Alert me if any holding moves 8%." / "Notify me when NVDA reports within a week."
- **Call:** `add_alert(symbol="AAPL", kind="drop", value=5)` · `add_alert(symbol="*", kind="move", value=8)` · `list_alerts()` · `remove_alert(alert_id="a1")`

---

## 10. Document upload (RAG)

Bring your own file; answers are grounded in **your** document with `[doc · p.N]` citations. `.txt`/`.md`/`.html` work out of the box; `.pdf` needs the `[documents]` extra. Semantic search when an embeddings endpoint is set, else keyword.

#### `ingest_document(path)`
Load a local `.txt`/`.md`/`.html`/`.pdf` — chunked, embedded (if possible), stored locally. Re-ingesting a filename replaces it.
- **Ask:** "Read the PDF at ~/Downloads/acme-10k.pdf."
- **Call:** `ingest_document(path="~/Downloads/acme-10k.pdf")`

#### `ask_document(query, doc="", max_passages=4)`
Answer from ingested document(s), returning cited verbatim passages. `doc` scopes to one document; omit to search all.
- **Ask:** "What does the report say about free cash flow?"
- **Call:** `ask_document(query="free cash flow", doc="acme-10k")`

#### `list_documents()`
List ingested documents — id, chunk count, whether embedded.
- **Ask:** "What documents have I loaded?"
- **Call:** `list_documents()`

#### `forget_document(doc)`
Remove a document from the store.
- **Ask:** "Forget the acme-10k document."
- **Call:** `forget_document(doc="acme-10k")`

---

## 11. Subagent delegation

The assistant can spin up fresh **research subagents** (their own tool loop over public data) — they can't delegate further, trade, or touch the live account.

#### `dispatch_subagent(task)`
One focused, self-contained side-investigation.
- **Ask:** "Research Nvidia's latest quarter and guidance in depth."
- **Call:** `dispatch_subagent(task="Research NVDA's most recent quarter: revenue, guidance, and segment trends. Return a cited summary.")`

#### `dispatch_subagents(tasks, mode="parallel")`
Fan several tasks out — one per line (or `||`-separated). `parallel` for independent work; `sequence` feeds each subagent a digest of earlier results. Max 6 per call.
- **Ask:** "Research AAPL, MSFT and NVDA in parallel and compare them."
- **Call:** `dispatch_subagents(tasks="Research AAPL fundamentals and outlook\nResearch MSFT fundamentals and outlook\nResearch NVDA fundamentals and outlook", mode="parallel")`

---

*Generated from the live tool registry (`financial_research_assistant.tools.TOOLS`).
Data is delayed/reference and not investment advice.*
