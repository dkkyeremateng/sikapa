# Tools Reference

Every agent tool the assistant can call, grouped by area. **66 tools.**

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
12. [Scheduled work](#12-scheduled-work)
13. [Rendered reports](#13-rendered-reports)
14. [Thesis journal](#14-thesis-journal)
15. [Macro backdrop](#15-macro-backdrop)
16. [Earnings calls](#16-earnings-calls)

---

## Answering about a past date (`as_of`)

Every tool here reports what is true **now** unless told otherwise, and that is the
easiest way to get a confidently wrong answer: ask "was NVDA expensive in January
2025?" and a current-snapshot tool hands back today's P/E, which then gets written
into a report as though it were January's.

The sources do not all have history, so `as_of` means two different things:

**Honoured exactly** — `price_history_chart`, `compare_prices`, `risk_metrics`,
`correlation_matrix`, `factor_exposure`. These are computed from a daily-close
series, which can be cut at a date. Pass `as_of="2025-01-31"` and the window ends
there; the output says so and names the last session on or before it (so a weekend
or holiday is visible rather than silently shifted). `days` keeps its meaning —
`days=365, as_of=…` is the year ending at that date, the same span `days=365`
covers today.

**Refused, with a redirect** — `stock_fundamentals`, `compare_stocks`,
`analyst_ratings`, `etf_exposure`, `dcf_valuation`, `screen_stocks`,
`dividend_projection`. These read a
current Yahoo snapshot that keeps no history: there is no January 2025 version of
those figures to return. Passing `as_of` makes the tool say so and name a source
that *does* have history — usually the SEC tools, which are point-in-time by
construction (a 10-K covers a fiscal year and is not restated in place).

A tool that quietly ignored `as_of` would be worse than one that never had it, so
none of them do. Two redirects are worth knowing:

- `screen_stocks` declines rather than pretending — a screen against today's
  fundamentals is not a backtest.
- `dividend_projection` looks *forward* from today's holdings and today's declared
  rates, so it has no past version at all; the historical question is what was
  actually received, which `income_summary(year=…)` answers.

One caveat on `factor_exposure` without a symbol: the returns are cut to `as_of`,
but the weights are your **current** holdings, because the statement store records
positions as of its import rather than a position history. So a dated portfolio
regression says how today's book would have loaded back then — not what you held.
The output states this whenever both apply.

---

## 1. Market data & charts

#### `current_date()`
Current local date **and** wall-clock time with timezone — the "as of" stamp for any quote or note.
- **Ask:** "What's today's date and time?"
- **Call:** `current_date()`

#### `price_history_chart(symbol, days=90, as_of="")`
Daily closing prices for one ticker as a terminal line chart with summary stats (first/last/high/low, % change). Keyless (Yahoo).
- **Ask:** "Show me Apple's price over the last 6 months."
- **Call:** `price_history_chart(symbol="AAPL", days=180)`

#### `compare_prices(symbols, days=180, as_of="")`
Several stocks on one **normalized** chart (each rebased to 100), plus each ticker's total return — comparable regardless of share price.
- **Ask:** "Compare NVDA vs AMD vs the S&P over the past year."
- **Call:** `compare_prices(symbols="NVDA, AMD, SPY", days=365)`

#### `risk_metrics(symbol, days=365, benchmark="SPY", as_of="")`
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

#### `stock_fundamentals(symbol, as_of="")`
Snapshot: profile (sector/industry), valuation (market cap, P/E, EPS), price + 52-week range, dividend, beta, one-line analyst summary. Keyless (Yahoo).
- **Ask:** "Give me the fundamentals on Microsoft."
- **Call:** `stock_fundamentals(symbol="MSFT")`

#### `analyst_ratings(symbol, as_of="")`
Buy/hold/sell consensus, price targets (low/mean/median/high) with implied upside, recent upgrades/downgrades.
- **Ask:** "What do analysts think of AMD? Price target?"
- **Call:** `analyst_ratings(symbol="AMD")`

#### `earnings_calendar(symbol)`
Next earnings date + EPS estimate, upcoming ex-dividend/pay dates, and recent quarters (estimate vs reported, surprise %).
- **Ask:** "When does Apple report next, and how have they done recently?"
- **Call:** `earnings_calendar(symbol="AAPL")`

#### `compare_stocks(symbols, as_of="")`
2–4 stocks side by side: price, market cap, P/E, PEG, P/S, revenue growth, margin, EPS, yield, beta, analyst consensus + target. Keyless (Yahoo snapshot).
- **Ask:** "Compare AAPL, MSFT and NVDA — which is cheaper and growing faster?"
- **Call:** `compare_stocks(symbols="AAPL, MSFT, NVDA")`

#### `etf_exposure(symbol, as_of="")`
Look through an ETF/fund to sector weightings and top holdings. Funds only.
- **Ask:** "What's inside VOO — sectors and top holdings?"
- **Call:** `etf_exposure(symbol="VOO")`

#### `dividend_projection(account="", as_of="")`
Forward 12-month dividend income across your imported holdings — per-holding income, yield-on-cost, current yield, FX-converted total. **Needs an imported statement.**
- **Ask:** "How much dividend income should my portfolio generate next year?"
- **Call:** `dividend_projection()`

---

## 4. Valuation & options

#### `dcf_valuation(symbol, growth_rate=0, discount_rate=0, terminal_growth=0, years=0, as_of="")`
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

#### `screen_stocks(symbols="", universe="", min_market_cap_b=0, max_market_cap_b=0, near_high_pct=0, near_high_within_days=10, high_lookback_days=1825, market_down_pct=0, market_symbol="SPY", min_earnings_beats=0, sector="", max_symbols=40, as_of="")`
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

#### `correlation_matrix(symbols="", days=180, as_of="")`
Correlation of daily returns across tickers (or your holdings when empty) — a quick diversification read.
- **Ask:** "How correlated are my holdings?"
- **Call:** `correlation_matrix(symbols="AAPL, MSFT, SPY", days=180)`

#### `portfolio_lookthrough(account="")`
**True** exposure by looking through ETFs to their holdings — a real sector breakdown and single-stock exposure that surfaces hidden concentration (a name held directly *and* via ETFs).
- **Ask:** "What's my true tech exposure after looking through my ETFs?"
- **Call:** `portfolio_lookthrough()`

#### `factor_exposure(symbol="", days=365, five_factor=False, as_of="")`
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

## 12. Scheduled work

The assistant exists only for the turn you are in — it cannot wait, check back, or
follow up. These tools are how anything happens **later**: the work is stored, a
background runner (`--run-due` from cron, or a `--watch` loop) executes it as an
ordinary turn hours later, and the answer is **delivered to you** — Telegram by
default, or any registered channel. The system prompt makes `schedule_task`
mandatory for a request about the future, so the agent schedules rather than
promising to check back.

Standing *conditions* on a price or an earnings date are better as **alert rules**
(§9) — those are model-free and checked by the digest. A scheduled task is for work
needing judgment: reading results, comparing to consensus, writing a view.

#### `schedule_task(prompt, when, repeat="once", channel="")`
Queue work to run later. `when` takes `2026-08-14 09:00`, `tomorrow 9am`, `friday`,
`+2h`; `repeat` is `once`/`hourly`/`daily`/`weekdays`/`weekly`; `channel` names one
delivery channel (blank = every configured one). The `prompt` must be
**self-contained** — the run happens in a fresh session that cannot see this
conversation, so it names the ticker, the event and what to produce. Recurring tasks
reschedule from their due time (no drift); a failure retries and parks after three
attempts. If no runner has ticked recently the reply says so, rather than promising
a delivery nothing will make.
- **Ask:** "Monitor NOMD's earnings tomorrow and analyse the results." / "Every weekday at 08:30, brief me on overnight moves in my holdings."
- **Call:** `schedule_task(prompt="NOMD reports Q3 on Aug 13. Pull actual EPS/revenue vs consensus, margin trend and guidance, then give a buy/hold/sell with reasoning.", when="tomorrow 9am")`

#### `list_scheduled_tasks()`
What is queued: id, when it runs, what it will do, and the outcome of anything that
has already run (including an answer still waiting to be delivered).
- **Ask:** "What have you got scheduled for me?" / "What are you watching?"
- **Call:** `list_scheduled_tasks()`

#### `cancel_scheduled_task(task_id)`
Drop one by id (from `list_scheduled_tasks`), or `all` to clear every one.
- **Ask:** "Cancel the Friday one." / "Stop watching NOMD."
- **Call:** `cancel_scheduled_task(task_id="s1")`

**Delivery.** Channels are a registry (`channels.py`): **telegram**
(`TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`), **desktop** (an OS banner), and
**stdout** (only when named in `NOTIFY_CHANNELS`). `--notify-test` checks it end to
end. An answer that reaches no channel is parked and re-sent on later ticks — the
model call is already spent, so it is never thrown away over a brief outage. With
`TELEGRAM_ALLOWED_CHAT_IDS` set you can also message the bot and have the agent
answer (`/tasks`, `/cancel <id>`, or any prompt); that is off by default, and
messages from other chats are dropped unread.

---

## 13. Rendered reports

Everything else here returns text, which is right in a terminal and poor on a
phone. This turns a summary into a **typeset sheet** — a title, stat tiles, then
the body — rendered to **PNG + PDF** and delivered as a file through the same
channel registry (§12). It is the natural output for a scheduled task, whose
answer arrives on a phone.

#### `render_report(title, markdown, highlights="", subtitle="", deliver=True, allow_prose=False, theme="", output="")`
`markdown` is the body — headings, bold, lists, tables, and `>` for a warning
callout all render. `highlights` is the stat-tile row, **one per line** as
`label | value | note` (max 6); put the headline numbers there rather than
repeating them in the body. Files are saved under
`~/.financial-research-assistant/reports` (`FINANCIAL_RESEARCH_REPORTS_DIR`) and
the PNG + PDF are sent to every configured channel that can carry a file. They go
as **documents, not photos**: Telegram recompresses a photo and would soften the
body text, while a document transfers untouched and still previews. Rendered at 3x
and cropped to the measured content height (`FINANCIAL_RESEARCH_REPORT_SCALE`),
in a light or dark theme (`FINANCIAL_RESEARCH_REPORT_THEME` for the image,
`FINANCIAL_RESEARCH_REPORT_PDF_THEME` for the PDF, which stays light by default).
The delivered image is a **one-sheet infographic** of the whole report — stat tiles
plus the body's own series charted, from percentage lists **or markdown tables**
(top holdings, sector weights, movers, beat histories, scenario ladders) — at any
length; the PDF alongside it is the full document. A report with neither tiles nor
series sends page 1 instead. Rendering prefers
headless **Chrome/Chromium** (better typography) and falls back to a built-in
**fpdf2** renderer when no browser is installed, so it works everywhere including
containers; `pypdfium2` makes the cover image either way. The fallback bundles the
Inter typeface, so its output is identical on every machine.
- **Ask:** "Send me that as a PDF." / "Make an infographic of the FISV results and put it on Telegram." / "Every Friday, email me a one-pager on my holdings."
- **Call:** `render_report(title="FISV Q2 2026 — miss and guidance cut", highlights="Adjusted EPS | $1.84 | vs $1.91 consensus\nMean target | $66.62 | +26.5%", markdown="## Headline\n\nFiserv missed and **cut guidance**…")`

---

## 14. Thesis journal

The assistant's other learning layers check the *process* (was a source thin?) or a
fixed regression set. None of them ever checks whether its market calls were right,
so all three can be satisfied while it is consistently wrong. This closes that loop.

#### `record_thesis(symbol, verdict, thesis, horizon_days=90, benchmark="SPY")`
Log a directional view — `bullish` / `bearish` / `neutral` — with the price **captured
here from market data**, not reported by the model. A runner tick scores it once the
horizon passes: price then vs now, and the same window for the benchmark. Logging a
call is explicitly not making a recommendation.
- **Ask:** *(the assistant calls this itself whenever it takes a side)*
- **Call:** `record_thesis(symbol="NVDA", verdict="bullish", thesis="Data-centre demand still outrunning supply; guidance looks conservative.", horizon_days=90)`

#### `review_theses(symbol="")`
Open calls, how the scored ones turned out, and the running hit rate plus mean
performance against the benchmark. The assistant checks this before taking a fresh
view on a ticker it has covered, so a view it has already been wrong on gets said
out loud rather than quietly repeated.
- **Ask:** "How have your calls done?" / "What did you say about NVDA before?"
- **Call:** `review_theses(symbol="NVDA")`
- **CLI:** `financial-research-assistant --theses` (model-free)

**Scoring.** Direction decides the hit — that is what was actually claimed — and the
benchmark comparison rides alongside to answer whether being right was worth
anything (`bullish +8%` against an index that did `+14%` is a hit with negative
alpha). `neutral` counts as right when the move stays inside ±5%. Scoring is two
price lookups and a subtraction, so it is model-free, runs on the existing
`--run-due` / `--watch` tick, and cannot hallucinate. A call the price source can't
answer for stays **open** for a later tick rather than being burned as a miss.

**What it is not.** A handful of past calls on whatever you happened to ask about —
far too small a sample, and not a random one, to say anything about the next call.
Every surface reports it as calibration and says so.

---

## 15. Macro backdrop

Rates, inflation, employment and credit — the context that decides whether a single
company's numbers mean anything. **Keyless:** FRED's official API needs a key, but
the graph CSV endpoint behind their charts doesn't, and takes the same series ids.

#### `macro_snapshot(as_of="")`
One screen: policy rate, 2y/10y and the curve spread, CPI and core PCE (year over
year), breakeven inflation, unemployment, jobless claims, high-yield spreads, VIX
and oil — each with its move over the past year and its own observation date.
- **Ask:** "What's the macro backdrop right now?" / "Where are rates and inflation?"
- **Call:** `macro_snapshot()`

#### `macro_series(series, days=730, as_of="")`
One series in depth, charted, with year-over-year for index series where the level
alone is meaningless. Takes plain names — `fed funds`, `10y`, `cpi`, `core pce`,
`unemployment`, `claims`, `yield curve`, `breakeven`, `high yield`, `vix`, `oil`,
`gdp`, `mortgage` — or any raw FRED series id.
- **Ask:** "Chart the 10-year over the past two years." / "Is inflation cooling?"
- **Call:** `macro_series(series="10y", days=730)`

**One caveat, stated in the output.** `as_of` bounds the **observation** date, not
the data **vintage**. FRED revises — GDP and payrolls are restated for months — so
a past period comes back as *currently restated*, not the figure that was on the
screen then. True point-in-time vintages need ALFRED, a separate keyed service.

---

## 16. Earnings calls

#### `earnings_call_transcript(symbol, year=0, quarter=0, query="", max_passages=6)`
Management's prepared remarks **and the analyst Q&A** — the one thing the SEC suite
can't reach. Returns speaker-attributed passages matching `query` rather than the
whole 15,000-word call.
- **Ask:** "What did management say about margins on the last call?" / "What did analysts push on?"
- **Call:** `earnings_call_transcript(symbol="NVDA", query="China export restrictions")`

**This is the only keyed tool here.** Everything else works on a fresh clone;
transcripts have no free programmatic provider. Without
`EARNINGS_TRANSCRIPT_API_KEY` the tool is unbound and the assistant is told to fall
back to the **8-K earnings release** (`sec_material_events` + `sec_filing_excerpt`),
which carries the prepared commentary on the same quarter — but not the Q&A, and it
is told to say which it is quoting.

---

*Generated from the live tool registry (`financial_research_assistant.tools.TOOLS`).
Data is delayed/reference and not investment advice.*
