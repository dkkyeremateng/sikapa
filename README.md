# financial-research-assistant

A LangGraph financial research assistant for your brokerage account —
**Interactive Brokers (IBKR)** by default, with a **pluggable broker layer** for
others. It answers questions about markets, holdings, and instruments by calling
a broker's market-data tools over MCP — filtered to **read-only** — plus local
analyst calculators, a **price-history charting** tool, and a **web-search**
tool for stock/market news. It ships a Textual chat TUI, a headless CLI, an
offline deterministic fake mode, and a generic eval harness.

> ### ▶ [Watch the guided demo](https://claude.ai/code/artifact/b39ec2f9-1b91-4af5-996c-a351aa3c62a0)
> A self-playing terminal walkthrough of seven features — company comparison, DCF
> valuation, cited SEC-filing answers, the options explainer, document Q&A,
> parallel subagents, and background work delivered to your phone — plus a map of
> all 71 tools. Source: [`demo.html`](demo.html)
> (open it locally in any browser). See also the full [`Tools.md`](Tools.md) reference.

> **The broker is pluggable — two registry-driven seams.** A different broker's
> **live read-only MCP server** plugs into the
> [broker-provider registry](#pluggable-brokers) (`brokers.py`), and a different
> broker's **statement export** imports through a format dispatcher — IBKR
> Activity CSV or cross-broker **OFX/QFX** (Vanguard, E\*TRADE, Schwab, most
> banks). Both land in the same store and read back through the same tools.

> **Work that outlives the session.** A chat turn ends when its answer does, so
> *"monitor NOMD's earnings tomorrow and analyse the results"* used to be a promise
> nothing could keep. `schedule_task` queues it, a background runner
> (`--run-due` from cron, or `--watch`) executes it hours later with no terminal
> open, and the answer is **pushed to you** — Telegram by default, through a
> [pluggable channel registry](#delivery-channels). The system prompt makes calling
> it mandatory for anything in the future, so the agent schedules rather than
> promising to check back. See [Scheduled tasks](#scheduled-tasks-work-that-runs-later-and-finds-you).

> **News search.** `web_search` returns recent headlines (title, source, date,
> snippet, URL). Keyless by default (DuckDuckGo news); set `TAVILY_API_KEY` for
> higher-quality LLM-oriented results. News can be inaccurate or promotional — the
> agent is told to cite the source and date and cross-check figures against the
> market-data tools.

> **Where price history comes from.** This IBKR MCP server exposes only
> *real-time* quotes (no historical bars), so the `price_history_chart` tool
> pulls daily closes from **Yahoo Finance** (free, no API key) and renders them
> as a terminal line chart with **plotext**. The chart is plain text, so it shows
> in the headless CLI, the TUI tool panel, and markdown alike.

> **Statement import.** Point the agent at a downloaded broker statement and
> `import_ibkr_statement` parses it into a local SQLite store
> (`~/.financial-research-assistant/statements.db`, override with
> `FINANCIAL_RESEARCH_STATEMENTS_DB`). The format is **auto-detected**: an IBKR
> **Activity Statement CSV**, or a cross-broker **OFX/QFX** export (Vanguard,
> E\*TRADE, Schwab, most banks — needs the optional `[ofx]` extra, see below).
> Both land in the same store and read back through the same tools; OFX carries
> trades/cash/positions/instruments but not the TWRR or NAV breakdown, so those
> stay IBKR-rich. It extracts — and skips subtotal/total rows throughout:
> - **Transactions** — trades plus every cash-flow section (dividends,
>   withholding tax, fees, deposits & withdrawals). Read back with
>   `query_transactions` (filter by `kind`/`symbol`).
> - **Portfolio snapshot** — open positions (qty, cost basis, value, unrealized
>   P/L), the instrument reference table (full name + **ISIN**), and the
>   net-asset-value breakdown by asset class plus the time-weighted return. Read
>   back with `query_portfolio` (positions joined to their name/ISIN).
> - **Account value over time** — `portfolio_value_history` stitches the NAV
>   snapshot from every imported statement (each contributes its period start/end
>   total NAV) into one account-value line chart. Coarse with one statement, finer
>   the more you import — a real growth curve, not a single stock's price.
> - **Performance vs. contributions** — raw account value includes the cash you
>   deposited. `portfolio_performance_chart` instead compounds each statement's
>   **time-weighted return** into a deposit-independent "growth of 100" index, so
>   you see how the investments themselves performed. Overlapping statements are
>   reconciled to a non-overlapping set (finer periods win, so monthly beats
>   annual — no span is double-counted) and coverage gaps are detected, reported,
>   and drawn as a visible break in the line (never interpolated across). When a
>   live IBKR Portfolio Analyst session is connected, the agent
>   prefers its `get_pa_performance_all_periods` series (finer, up-to-date) over
>   these statement-stitched charts.
>
> Import is keyed by (account, period), so re-importing the same statement
> replaces rather than double-counts. Statements that *overlap* rather than match —
> a full year plus the monthly statements inside it, which is exactly what the NAV
> chart above asks you to import — are reconciled on read: each trade and corporate
> action is counted once no matter how many statements report it, so the FIFO
> figures (`realized_gains`, `tax_loss_harvest`, open lots) don't double and a
> split reported twice doesn't rescale your cost basis by the square of the factor.
> In the TUI, `/import PATH` loads a
> statement directly (no model call — works in `--fake` too); or just ask the
> agent in a normal message and it calls `import_ibkr_statement` for you.
> Consolidated multi-account and multi-currency statements are handled
> correctly: cash sums are grouped by currency (never blended), and every
> analytic/history query is scoped to one account (defaulting to the newest
> import's) so two accounts' data never gets chained into one nonsense series —
> pass an `account` argument to pick.

> **Company reference data (Yahoo, keyless).** Beyond real-time IBKR quotes and
> price charts, the agent pulls fundamentals via the `yfinance` library:
> - **`stock_fundamentals`** — valuation (market cap, trailing/forward P/E, EPS),
>   company profile (sector/industry), 52-week range, dividend rate & computed
>   yield, beta, and a one-line analyst consensus.
> - **`analyst_ratings`** — buy/hold/sell consensus and analyst count, price
>   targets (low/mean/median/high) with implied upside, and recent
>   upgrades/downgrades.
> - **`earnings_calendar`** — next earnings date + consensus EPS, upcoming
>   ex-dividend / pay dates, and recent estimate-vs-reported history with surprise %.
> - **`dividend_projection`** — forward 12-month dividend income across your
>   imported positions (share count × current declared rate), with per-holding
>   yield-on-cost and current yield, totaled in the base currency. Distinct from
>   `income_summary`, which reports dividends already *received*.
> - **`compare_stocks`** — 2–4 tickers side by side in one normalized metric table
>   (price, market cap, trailing/forward P/E, PEG, P/S, revenue growth, profit
>   margin, EPS, dividend yield, beta) plus the analyst consensus and mean target
>   with implied upside. For "AAPL vs MSFT", "which is cheaper / growing faster".
>
> These are delayed reference figures (verify before acting), and gracefully
> return "no data" on an unknown ticker or a Yahoo outage rather than failing.

> **SEC filing intelligence (`sec_financials`, `sec_filings`, `sec_filing_search`).**
> Primary-source company data straight from **SEC EDGAR** — keyless, free, and the
> audited numbers rather than a third-party snapshot:
> - **`sec_financials`** — as-reported annual financials from a company's **10-K
>   XBRL** data (revenue, gross/operating/net income, diluted EPS, assets,
>   liabilities, equity, cash) across recent fiscal years, with computed gross/net
>   margins and revenue growth. More authoritative than the Yahoo
>   `stock_fundamentals` snapshot — cite it to the 10-K. Pass a us-gaap `concept`
>   tag (e.g. `NetIncomeLoss`) for a single line's history.
> - **`sec_quarterly_financials`** — the **quarterly (10-Q)** companion to
>   `sec_financials`: the same line items across recent quarters (newest first),
>   with per-quarter margins and revenue **QoQ + YoY** growth. Columns are labeled by
>   period-end date; the fiscal-year-end quarter can be absent (the 10-K reports it
>   as the full year). For "last N quarters / quarterly revenue / trend by quarter".
> - **`sec_filings`** — a company's recent filings (10-K/10-Q/8-K/insider Form 4)
>   with the filing date, description, and a direct link to each document.
> - **`insider_transactions`** — recent **insider** activity parsed from **Form 4**
>   ownership XML: separates open-market **buys (P)** and **sales (S)** — the
>   conviction signals — from routine grants, option exercises, and tax-withholding,
>   and reports the net. For "are insiders buying/selling X". Buying is the rarer,
>   stronger signal; selling is often routine.
> - **`sec_material_events`** — recent **8-K** filings with the event type decoded
>   from the SEC item codes (earnings releases, M&A, executive departures, material
>   agreements, impairments) — a catalyst monitor with a link to each disclosure.
> - **`sec_filing_search`** — full-text search across **all filings since 2001**
>   for a phrase or topic, returning the exact matching filings (company, form,
>   date, link) so a claim can be **cited to a primary document**.
> - **`sec_filing_excerpt`** — fetches a company's latest filing (10-K/10-Q/8-K)
>   and returns the **verbatim passages** that match a topic — the exact language
>   to quote and cite (the grounding step after `sec_filing_search`), keyword-based
>   and keyless (no embeddings).
> - **`filing_summary`** — a structured **tearsheet** of a company's latest filing:
>   the relevant passages grouped under fixed slots (business & segments, revenue
>   drivers, margins, outlook/guidance, capital allocation, key risks) for a
>   predictable, comparable summary rather than free-form prose.
> - **`compare_sec_financials`** — several companies' **audited financials side by
>   side** in one matrix (companies × metrics) from 10-K XBRL — the as-reported
>   peer comparison, distinct from the Yahoo-snapshot `compare_stocks`. Pass a
>   `concept` for one metric across companies over several years.
> - **`filing_tone_trend`** — tracks the **tone** of a company's recent 10-Ks over
>   time: the negative-word density (a Loughran-McDonald finance-sentiment
>   heuristic) per filing, so you can see whether management's language is growing
>   more cautious. A lexicon signal, not a judgment — pair with `sec_filing_excerpt`.
> - **`sec_metric_rank`** — where a company **ranks** on a metric among *all* SEC
>   filers (e.g. "AAPL's net income is #4 of ~6,000"), with its value and the peer
>   median. Reports a rank/percentile — robust to the occasional filer error — not
>   a raw "biggest companies" leaderboard (whose extremes can be mis-scaled filings).
> - **`dcf_valuation`** — a deterministic two-stage **discounted-cash-flow** estimate
>   of intrinsic value. Free-cash-flow history is pulled *as-reported* from the 10-K
>   (XBRL operating cash flow − capex); net debt, shares and price come from Yahoo.
>   The tool does all the arithmetic and returns the projected cash flows and their
>   present values, the terminal value, intrinsic value per share, upside/(downside)
>   vs the current price, and a discount-rate × terminal-growth **sensitivity grid**.
>   Assumption knobs (`growth_rate`, `discount_rate`, `terminal_growth`, `years`)
>   default to sensible values — stage-1 growth is derived from the historical FCF
>   CAGR, measured over the fiscal years the history actually spans (a year the
>   filer didn't tag stretches the period, it doesn't shorten it). Omit a knob to
>   derive it; pass `0` to mean zero, which is a real assumption rather than a
>   request for the default. It's a *model, not a price target*, presented with its
>   assumptions, and it
>   declines for banks/insurers (no meaningful capex) and pre-FCF companies.
> - **`explain_option`** — turns an options contract into plain-language economics
>   over the live (keyless) Yahoo option chain. Pass a ticker and optionally an
>   `expiry`, `strike`, and `option_type` ('call'/'put'); the tool works out the
>   premium and per-contract cost, bid/ask/last, implied volatility and the
>   IV-implied move, the split of the premium into **intrinsic vs. time value**,
>   moneyness, and — for buying it — the **breakeven** (and % move to reach it),
>   **max loss** (the premium), and **max profit**, plus the short-side max loss.
>   Omit the strike to get a near-the-money slice of the chain to choose from. A
>   single-leg estimate at expiry, not advice or a spread builder.
> - **`ingest_document`** / **`ask_document`** / **`list_documents`** /
>   **`forget_document`** — bring your own file. Point `ingest_document` at a local
>   **.txt / .md / .html / .pdf** and it's chunked and stored locally (PDF needs the
>   optional `[documents]` extra); `ask_document` then retrieves the most relevant
>   passages **with `[doc · p.N]` citations** so the answer is grounded in *your*
>   document, not the model's memory. Semantic search when an embeddings endpoint is
>   configured (same `OPENAI_API_BASE`/`OPENAI_API_KEY` as the model), keyword
>   otherwise — so it works with zero config. This is for user-provided files; SEC
>   filings are fetched directly by the `sec_*` tools.
>
> `sec_filing_excerpt` and `filing_summary` rank passages by keyword by default;
> set **`SEC_EDGAR_SEMANTIC=1`** (needs an embeddings endpoint) to re-rank the
> keyword shortlist by embedding similarity, catching paraphrases keyword matching
> misses. It degrades to keyword if embeddings are unavailable, so the keyless path
> always works.
>
> These cover US-listed filers (identified by ticker → CIK). SEC asks callers to
> send a descriptive `User-Agent` with a contact email and to stay under ~10
> requests/second — a working default is used; set **`SEC_EDGAR_UA`** (e.g.
> `"Your Name you@domain.com"`) so heavy use is attributable to you. What EDGAR
> does *not* carry cheaply — verbatim earnings-call transcripts and sell-side
> research — stays out of scope (those need a paid feed).

> **Move attribution & bull-vs-bear (`explain_stock_move`, `bull_bear_debate`).**
> Two narrative research tools that gather evidence and hand it back for the agent
> to synthesize (no nested model call):
> - **`explain_stock_move`** — answers *"why is TICKER up/down today?"* by fusing
>   the measured price move (last session + over the last N sessions), recent
>   analyst upgrades/downgrades, and recent news into a short **attributed**
>   explanation — citing the news items by number, and saying plainly when the
>   evidence *doesn't* explain the move rather than inventing a catalyst.
> - **`bull_bear_debate`** — for *"should I buy X / make the case for and against
>   X"*: gathers the full research findings and frames them for a steel-manned
>   **Bull case**, **Bear case**, and a **Verdict** (which side the evidence better
>   supports, a lean with rough confidence, and what would change it). An
>   adversarial pass borrowed from multi-agent trading frameworks, kept advisory.

> **Stock screening (`screen_stocks`).** Find stocks meeting a set of conditions —
> e.g. *"large caps near their all-time highs that keep beating earnings, that held
> up on a down-market day."* There is no bulk market-data feed here (every source is
> per-ticker), so the screener evaluates a **candidate universe** and measures the
> criteria it can offline, running the per-symbol lookups concurrently. Choose the
> universe with `universe="sp500"` (the current S&P 500, fetched keyless from a
> constituents CSV — raise `max_symbols` toward ~500 for the whole set, since it's one
> Yahoo lookup per name and therefore slow), an explicit `symbols` list (an ETF's
> holdings from `etf_exposure`, a watchlist, an index), or neither for a built-in
> large-cap default. Criteria:
> - **market-cap bounds** (`min_market_cap_b` / `max_market_cap_b`, in USD billions);
> - **proximity to a high** (`near_high_pct`) — the closest the price came, in the
>   last `near_high_within_days` sessions, to its high over `high_lookback_days`
>   (pass a large value for a true **all-time** high) — optionally requiring that the
>   near-high day was a **down-market day** (`market_down_pct` vs. `market_symbol`,
>   default SPY), i.e. the stock showed relative strength;
> - an **EPS-beat streak** (`min_earnings_beats`) — consecutive most-recent reported
>   quarters that beat the analyst estimate;
> - a **sector** substring.
>
> It returns a ranked table of the passers with the measured figures. To screen a
> specific set, pass its tickers (an index, an ETF's holdings from `etf_exposure`, or
> a watchlist) in `symbols`. **Only the quantitative criteria above are screened** —
> qualitative conditions like *forward-guidance* beats or raised guidance aren't in
> any structured feed, so the tool flags them for per-name follow-up via
> `earnings_calendar` / `web_search` / `research_report` rather than silently
> treating them as met.

> **Subagent dispatch (`dispatch_subagent` / `dispatch_subagents`).** The agent can
> delegate work to fresh **research subagents** — each a full agent with its own
> model + tool loop — and gather their findings:
> - **`dispatch_subagent(task)`** — hand off one self-contained investigation (e.g.
>   *"research NVDA's latest quarter and guidance"*), keeping a big side-quest out
>   of the main thread.
> - **`dispatch_subagents(tasks, mode)`** — fan out several tasks (one per line).
>   `mode="parallel"` (default) runs them **concurrently** — ideal for independent
>   work like researching several tickers or pulling several data sources at once;
>   `mode="sequence"` runs them **one at a time and chains context forward**, feeding
>   each subagent a digest of the earlier results (e.g. survey a sector, then
>   deep-dive the standout).
>
> Subagents get the same delayed/public-data research tools the primary agent has,
> but **not** the dispatch tools themselves — that one-level cap is a hard recursion
> guard, so a subagent can never spawn more subagents. They also can't trade, touch
> the live IBKR account, or write long-term memory — nor take any other action that
> outlives the investigation: no scheduling work, recording a thesis, setting an
> alert, importing a statement, or pushing a report to your phone. A subagent is
> given a question to answer, and the answer comes back to the primary agent to act
> on; a vaguely-worded task must not be able to queue recurring model spend or
> deliver a half-finished file. Each run is time-bounded
> (`FINANCIAL_RESEARCH_SUBAGENT_TIMEOUT`, default 180s), a call fans out at most 6
> tasks, and a failed or timed-out subagent comes back as a labeled note rather than
> aborting the turn. Because a subagent sees only the task text it's given, the
> primary agent makes each task self-contained and then synthesizes the results into
> its own answer.

> **Portfolio analytics.** Over the imported data the agent can compute:
> - **`realized_gains`** — FIFO capital gains split short- vs long-term, net of
>   commissions and **adjusted for stock splits** (a 3-for-1 rescales open lots so
>   the cost basis stays correct); reports unmatched sells when an opening lot
>   wasn't imported.
> - **`income_summary`** — dividends, withholding tax, and fees netted, grouped by
>   currency, with a per-symbol dividend breakdown.
> - **`allocation`** — position weights, largest holding, top-5 concentration, and
>   a by-asset-category breakdown.
> - **`portfolio_vs_benchmark`** / **`compare_prices`** — benchmark your TWRR
>   against an index (e.g. SPY), or overlay several tickers rebased to 100.
> - **`risk_metrics`** — per-ticker annualized volatility, max drawdown, Sharpe,
>   and beta vs a benchmark.
> - **`portfolio_risk`** — the **whole-portfolio** companion to `risk_metrics`:
>   value-weights your current holdings into one synthetic daily return series, then
>   reports annualized volatility, max drawdown, Sharpe, and beta. Current holdings
>   are held constant over the window, and the covered share of portfolio value is
>   reported (unpriceable positions are excluded, not treated as risk-free).
> - **`factor_exposure`** — Fama-French factor regression for a ticker or the whole
>   portfolio: market, size (SMB), and value (HML) loadings — plus profitability
>   (RMW) and investment (CMA) in 5-factor mode — with annualized **alpha** and
>   **R²**. Reveals style tilts (small/large-cap, value/growth) and how much of your
>   return is unexplained factor-wise, a lens beyond single-factor beta. Factors
>   from the Ken French Data Library (keyless).
> - **`tax_loss_harvest`** — open FIFO lots now trading below cost, split
>   short-/long-term, with an estimated tax benefit and a **wash-sale** flag when
>   you bought the same symbol within the last 30 days. Read-only analysis; a loss
>   is only realized if you actually sell.
> - **`correlation_matrix`** — a matrix of daily-return correlations across your
>   holdings (or given tickers), so you can see what really moves together vs. what
>   diversifies.
> - **`etf_exposure`** — look through an ETF/fund to its sector weights and top
>   holdings (e.g. what's inside VOO, or how two ETFs overlap).
> - **`portfolio_lookthrough`** — your portfolio's *true* exposure after expanding
>   every ETF to its underlying holdings: a value-weighted sector breakdown, plus a
>   true single-stock exposure that surfaces **hidden concentration** — a name held
>   directly *and* inside several ETFs is combined (e.g. "AAPL 30.9% — held directly,
>   via SPY"), which naive per-position allocation misses. Sector exposure is
>   complete; single-stock look-through covers each fund's top holdings.
> - **`export_data`** — write positions or transactions to a CSV for Excel / an
>   accountant / tax software. Writes are confined to the export directory
>   (`~/.financial-research-assistant/exports`, override with
>   `FINANCIAL_RESEARCH_EXPORT_DIR`) so a tool call can never overwrite an
>   arbitrary file elsewhere on disk.
> - **`convert_currency`** — convert any amount to the base currency (USD) at
>   market FX rates. `income_summary` and `allocation` also fold non-USD amounts
>   into a USD-equivalent total (keeping per-currency detail), so a
>   multi-currency book/income stream reconciles into one figure. Set
>   `BASE_CURRENCY` if your IBKR base isn't USD.
>
> Historical prices and FX rates (Yahoo, keyless) are cached per run and retried
> once on a transient failure. The cache expires after 15 minutes
> (`FINANCIAL_RESEARCH_PRICE_TTL`, seconds; `0` disables caching), and the same
> clock governs the company snapshot behind `stock_fundamentals` / `compare_stocks`
> / `dcf_valuation` / `explain_option` — it carries the spot price, so a session
> left open across a trading day would otherwise quote an option's breakeven
> against the morning's price while the chart beside it showed the afternoon close.

> **Research-only.** The assistant loads only IBKR `get_*` / `search_*` tools
> (balances, positions, quotes, price history, contract/company lookups). Order
> entry and watchlist mutation are never exposed — it cannot place, modify, or
> cancel a trade. Figures can be delayed; verify before acting. Not investment
> advice.

## Layout

```
├── src/financial_research_assistant/
│   ├── events.py      # AgentEvent contract shared by all interfaces
│   ├── adapter.py     # run_turn(): framework -> AgentEvent stream
│   ├── graph.py       # build_graph / build_real_graph -> compiled ReAct agent
│   ├── auth.py        # 0600 credential store (providers x tiers) + secret redaction
│   ├── oauth.py       # UI-neutral login flows: PKCE, paste-a-key, provider registry
│   ├── codex_proxy.py # OpenAI-compatible shim over the ChatGPT Codex backend
│   ├── tools.py       # local calculators + read-only broker MCP loader/filter
│   ├── brokers.py     # pluggable broker-provider registry (IBKR + any other broker)
│   ├── fundamentals.py# yfinance reference tools (valuation/ratings/earnings/dividends/etf)
│   ├── analytics.py   # tax-loss-harvest, correlation, ETF look-through, portfolio_risk
│   ├── factors.py     # Fama-French factor exposure (market/size/value, alpha, R²)
│   ├── edgar.py       # SEC EDGAR intelligence: annual/quarterly financials, filings, insider Form 4, full-text (keyless)
│   ├── valuation.py   # deterministic two-stage DCF intrinsic valuation (dcf_valuation)
│   ├── options.py     # keyless options explainer — single-leg payoff economics (explain_option)
│   ├── documents.py   # document-upload RAG — ingest/ask local files with cited passages
│   ├── screener.py    # stock screener over a candidate universe (screen_stocks, S&P 500)
│   ├── subagents.py   # parallel/sequence research subagent dispatch (dispatch_subagent[s])
│   ├── research.py    # deep-research report (parallel gather → synthesize) + --research
│   ├── monitor.py     # portfolio monitoring digest (movers/earnings/ex-divs) + --digest
│   ├── alerts.py      # user-defined alert rules the digest checks (add/list/remove_alert)
│   ├── tasks.py       # scheduled-task store + schedule_task/list/cancel (work to run later)
│   ├── scheduler.py   # the runner: --run-due (cron) / --watch (loop) + Telegram inbox
│   ├── channels.py    # pluggable delivery-channel registry (telegram/desktop/stdout)
│   ├── reports.py     # render_report: markdown -> typeset PNG/PDF sheet (headless Chrome)
│   ├── telegram.py    # Bot API client — outbound delivery + allowlisted inbound
│   ├── flex.py        # IBKR Flex Web Service pull (--flex-sync) + XML → store
│   ├── statements.py  # IBKR CSV + OFX/QFX parsers, format dispatch, SQLite store
│   ├── tui.py         # Textual chat app
│   └── main.py        # CLI entry (argparse): TUI / --prompt / --fake / --login
├── tests/test_smoke.py
└── eval/              # evaluate.py + dataset.jsonl
```

## Setup

Uses [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Optional extras: `[ofx]` (import cross-broker OFX/QFX statements),
`[anthropic]`/`[google]`/`[groq]` (non-OpenAI model providers), `[tracing]`
(OpenTelemetry export) — e.g. `uv pip install -e ".[dev,ofx]"`.

> The virtualenv is **not** relocatable — if you move this folder, don't copy
> `.venv`; recreate it at the destination with the commands above (moving a venv
> can segfault native deps at runtime).

## Configure a model

Two ways: **`/login`** stores credentials in a `0600` file outside `.env` and is
the better default (see [Signing in](#signing-in-login)); or copy `.env.example`
to `.env` and fill it in (loaded via python-dotenv). A stored credential outranks
the environment, so the two never fight.

- **Cloud (OpenAI):** set `OPENAI_API_KEY`; optionally `OPENAI_MODEL`
  (default `gpt-4.1-mini`).
- **Local LLM (llama.cpp / Ollama / LM Studio):** set `OPENAI_API_BASE` to the
  server's OpenAI-compatible endpoint and `OPENAI_MODEL` to the served model id.
- **Other providers (Anthropic / Google / Groq):** set `MODEL_PROVIDER`
  (`anthropic`, `google_genai`, or `groq`) and that provider's key
  (`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`), and install the extra
  (`uv pip install -e '.[anthropic]'`). These route through LangChain's
  `init_chat_model`; each has a sensible default model (e.g. `claude-sonnet-5`)
  so `MODEL_PROVIDER` alone is enough. The default/`openai` path is unchanged.
  `/login anthropic-key` does the same thing without touching `.env`.
- **No model at all:** pass `--fake` — a deterministic in-process graph replies
  with a `FAKE-OK` marker, fully offline (no model, no IBKR).

**Quick / deep model tiering (optional).** Summarization is cheap work that
doesn't need the primary reasoning model, so set `QUICK_MODEL` (with optional
`QUICK_API_BASE` / `QUICK_API_KEY` / `QUICK_MODEL_PROVIDER`, same shape as the
`SUBAGENT_*` and primary vars) to route **context compaction summaries** to a
cheaper/faster model while the agent itself stays on the primary. Unset = the
primary model does it (unchanged). Subagents have their own `SUBAGENT_MODEL`
tier, so you can run the agent on a strong model, subagents on a mid one, and
summarization on a small one.

## Connect IBKR market data (MCP)

The assistant loads IBKR tools from an MCP server and keeps only the read-only
ones. Point it at your server in `.env` (either transport):

- **Remote / hosted (HTTP):** `IBKR_MCP_URL=https://your-ibkr-mcp-host/mcp`
  (optionally `IBKR_MCP_TOKEN` → sent as a Bearer header).
- **Local process (stdio):** `IBKR_MCP_COMMAND=npx` and
  `IBKR_MCP_ARGS=-y @your-org/ibkr-mcp-server`.

With neither set, the agent still runs — with the local calculators only. The
read-only filter (`tools.filter_readonly`) drops every order/watchlist-write and
feedback tool even if the server offers them.

**Additional MCP servers.** Set `EXTRA_MCP_SERVERS` to a JSON object of
`{name: server_spec}` (MultiServerMCPClient form) to mount extra read-only data
servers — FRED macro data, fundamentals, SEC filings — alongside IBKR. IBKR keeps
its strict allowlist (it can execute trades); extra servers are filtered with a
write-verb denylist (`tools.filter_safe`), so noun-named data tools survive while
anything that mutates state is dropped. A key matching a registered broker is
ignored so it can't shadow that broker's strictly-filtered mount.

### Pluggable brokers

IBKR is not hard-wired — it's the reference entry in a **broker-provider
registry** (`brokers.py`). A `BrokerProvider` declares three things: a `key`
(mount name), a `config_from_env` that builds its MCP server spec from
environment variables (returns `None` when unconfigured, so a broker is opt-in by
the presence of its env vars), and a `filter_tools` read-only policy. A broker
mounts through the same per-turn `broker_tools_session` as IBKR.

Adding a different broker is one registry entry — **no changes to the graph,
adapter, or session lifecycle**:

```python
# in brokers.py
def _tradier_config():
    url = os.environ.get("TRADIER_MCP_URL")
    return {"url": url, "transport": "streamable_http"} if url else None

register_broker(BrokerProvider(
    "tradier", _tradier_config,
    make_readonly_filter(("get_", "list_", "search_"),
                         write_deny=frozenset({"place_order", "cancel_order", "modify_order"})),
))
```

`make_readonly_filter` gives a new broker the same trade-safe shape as IBKR —
keep only read-verb-prefixed tools, and the `write_deny` denylist always wins
first so an order endpoint can never be exposed. A broker activates once its own
env vars are set; when several are configured, `BROKER_PROVIDERS=ibkr` (or a
comma-separated list) pins which mount, one at a time.

## Authenticating IBKR

Market-data and position reads need a **live, authenticated IB gateway session**.
The agent deliberately does **not** call the server's `authenticate` tool: on the
`interactive-brokers-mcp` server that opens an interactive browser login and can
disrupt an already-valid session. You establish the session once, then the agent
just reads against it. Two ways:

- **Interactive (browser):** start the MCP server / gateway and complete the
  login it opens (e.g. `https://localhost:5001`) with your IBKR credentials. Keep
  that session alive while you use the agent. Verified working: with a live
  session, account balances and summary return real data through the agent.
- **Headless (automation, recommended):** configure the server for
  non-interactive auth — `IB_HEADLESS_MODE=true`, `IB_USERNAME`,
  `IB_PASSWORD_AUTH`, and for fully hands-off 2FA `IB_TWO_FA_STRATEGY=totp` +
  `IB_TOTP_SECRET` — so no browser is needed. Use a **paper-trading** account
  first (`IB_PAPER_TRADING=true`). See `.env.example` for the full template.

If (and only if) your server exposes a **non-interactive** `authenticate`, you can
let the agent drive it by setting `IBKR_ALLOW_AUTHENTICATE=1`. This still never
exposes order/alert tools — `place_order` and friends stay blocked regardless.
Leave it unset for interactive/browser-login servers.

### Why you re-authenticate after every restart

`interactive-brokers-mcp` runs as a **stdio child of the agent** and manages a
separate IB Gateway. It's designed to reuse a healthy Gateway across restarts
(`ib-gateway/.runtime/gateway-session.json`), but when the agent exits it often
hard-kills the child before the "detach and leave the Gateway running" path runs,
so the session (or `gateway-session.lock`) goes stale and the next start
re-authenticates. IBKR also expires sessions on its own. There is **no HTTP/serve
transport** for this server (stdio only), so a persistent standalone server isn't
an option here — the practical fix is **headless auth with a TOTP secret** so each
restart re-auths silently in the background. Without `IB_TOTP_SECRET`, headless
mode still waits ~60s for you to approve 2FA. Quick check on shutdown: if the pid
in `gateway-session.json` is dead every time, the Gateway isn't surviving the
restart; if it's alive but you still re-auth, the IBKR session expired.

## Automatic context compaction (any interface)

`/compact` is a TUI command, but the same summarize-and-shrink runs in the shared
turn pipeline, so **headless runs, the eval harness, and any service embedding
`run_turn`** get it too. Set `AGENT_AUTO_COMPACT` to a fraction of the context
window (e.g. `AGENT_AUTO_COMPACT=0.85`) and `run_turn` compacts *before* a turn
once the previous turn's input reached that fraction — emitting a `status` event
("auto-compacted N older message(s)…"). Unset (the default) it never fires, so
behavior is unchanged. It applies per long-running process/session; a fresh
`--prompt` one-shot has no prior turn to measure.

## Deep-research reports (`--research`)

`--research SYMBOL` produces a structured, cited markdown research report on a
ticker. It follows a plan → **parallel retrieval** → synthesis shape: it gathers a
breadth of findings **concurrently** (price history, fundamentals, analyst ratings,
earnings, risk metrics, ETF look-through, and recent news), then the model
synthesizes them into a report (Summary, Valuation & Fundamentals, Analyst View,
Earnings, Price & Risk, News & Catalysts, Risks, Bottom line) with each figure
cited to its source section. The report is saved under
`~/.financial-research-assistant/reports` (override `FINANCIAL_RESEARCH_REPORTS_DIR`).

```bash
financial-research-assistant --research AAPL          # gather → synthesize → save + print
financial-research-assistant --research AAPL --fake   # offline stub (no model/network)
financial-research-assistant --research portfolio     # a report on your WHOLE portfolio
```

`--research portfolio` reports on the imported portfolio instead of one ticker,
gathering allocation, true look-through exposure, benchmark performance, income &
forward dividends, realized gains, and upcoming events, then synthesizing them the
same way.

In chat you can just ask for a deep dive on a ticker: the `research_report` tool
gathers the same findings and the agent writes the report inline (no extra model
round-trip). The synthesis uses ONLY the gathered findings — it's told not to
invent figures — and the report ends with a research-only, not-advice disclaimer.

## IBKR Flex Web Service sync (`--flex-sync`)

Instead of downloading an Activity Statement CSV by hand, `--flex-sync` pulls it
programmatically over IBKR's **Flex Web Service** — a two-step, token-authenticated
request (SendRequest → GetStatement, with an automatic retry while IBKR generates
the statement). Set it up in IBKR (Reports → Flex Queries → create an Activity Flex
Query and enable the Flex Web Service), then:

```bash
export IBKR_FLEX_TOKEN=...          # the Flex Web Service token (sensitive; env only)
export IBKR_FLEX_QUERY_ID=...       # your Activity Flex Query ID
financial-research-assistant --flex-sync
financial-research-assistant --flex-sync --flex-query-id 123456   # override the query
```

It's **model-free** (no API key) and cron-friendly, like `--digest`. The token is
read only from `IBKR_FLEX_TOKEN` and never passed through the model.

The statement XML is saved `0600` under `~/.financial-research-assistant/flex`
(override `FINANCIAL_RESEARCH_FLEX_DIR`) and then imported into the same SQLite
store the CSV path fills, so Flex data is queryable through the identical tools.
The raw file is written *before* it is parsed, so a statement the parser can't yet
read is kept rather than lost with the error. A saved `.xml` can also be re-imported
by hand with `import_ibkr_statement` — the format is detected from its
`<FlexQueryResponse>` root, not its extension.

### Configuring the Flex Query

Flex names its fields differently from the Activity Statement CSV, so the query has
to carry the sections the store reads. Create an **Activity Flex Query** with:

| Section | Level | Notes |
| --- | --- | --- |
| Trades | **Orders** | `cost`, `proceeds`, `ibCommission` drive every gain figure |
| Open Positions | **Summary** | Lot rows, if also selected, are ignored so shares aren't counted twice |
| Cash Transactions | **Detail** | plus the types: dividends, payment in lieu, withholding tax, other/broker fees, deposits & withdrawals |
| Corporate Actions | Detail | the *description* is what split detection reads |
| Net Asset Value (NAV) in Base | — | per-asset-class totals |
| Change in NAV | — | carries `TWR`, the only place a time-weighted return appears |
| Financial Instrument Information | — | conid/ISIN/exchange for the positions join |

Under General Configuration set **Date Format `yyyy-MM-dd`**. IBKR's `yyyyMMdd`
default is converted automatically; a day-first or month-first format is refused
outright, because `03/04/2026` is two different days depending on a setting the file
doesn't carry and guessing would misdate trades near the start of a month rather
than fail. `Period` should be **Last 365 Calendar Days** (Flex's cap for activity).

Getting a section wrong is not silent. A section that is selected but empty still
emits its container, so an absent one means the query never asked — and `--flex-sync`
names it:

```
warning: Corporate Actions not in the Flex query — splits will not adjust lots,
         silently distorting cost basis per share
```

That one matters most: without it, a year containing an unapplied 3-for-1 looks
exactly like a year with no splits, and that position's basis per share is trebled.

**Breakout by Day** is optional and expensive. With it on, one pull becomes a
`<FlexStatement>` per business day — each repeating the full position and security
list — which is why the parser concatenates trades and cash but takes positions,
instruments and the closing NAV from the last statement alone, and chain-links the
daily TWRs into one figure. One pull is always one import either way.

NAV is reconciled against the broker's own total, and any shortfall (crypto, which
IBKR folds into `total` without offering a field for it) is booked as an **Other**
asset class rather than dropped.

Lots opened before the 365-day window have no acquisition record. Sales of those
shares surface as `unmatched_proceeds`; shares still *held* are reported by
`statements.lot_coverage()` and noted by the `realized_gains` tool, since otherwise
the position renders normally while its cost basis is computed from whatever
fraction of the shares happens to be covered. Import older Activity Statement CSVs
(Reports → Statements → Activity supports a multi-year custom range) to close the
gap — the store keys imports by `(account, period)`, so overlapping windows dedupe
rather than double-count.

## Monitoring digest (cron-friendly, no model)

`--digest` prints a proactive "what changed / what's coming" report over your
imported holdings and exits — **no model call**, so it needs no API key and is
free to schedule:

```bash
financial-research-assistant --digest                 # newest import's account
financial-research-assistant --digest --account U123  # a specific account
```

It scans each holding for **price movers** (beyond ±5% over the lookback window),
**upcoming earnings** (next 14 days, with consensus EPS), and **upcoming
ex-dividend dates**, all from keyless Yahoo data — plus any of your own **alert
rules** that fire (see below). Run it from cron/launchd and pipe it to mail or a
push service (e.g. `ntfy`):

```cron
# 8am on weekdays: email the portfolio digest
0 8 * * 1-5  financial-research-assistant --digest | mail -s "Portfolio digest" you@example.com
```

The same report is available in-chat as the `portfolio_digest` tool (which also
takes `include_news=true` to attach a headline per mover), so you can just ask
*"anything I should know about my portfolio this week?"*.

**Standing alert rules.** Set your own thresholds and the digest checks them on
every run (in-chat or via `--digest`), surfacing a 🔔 section at the top when any
fire. Just say *"tell me if AAPL drops more than 5%"*, *"alert me if any holding
moves 8%"*, *"notify me if TSLA goes below 200"*, or *"flag NVDA earnings within a
week"* and the agent calls `add_alert(symbol, kind, value)` — where `kind` is
`drop`/`rise`/`move` (percent), `below`/`above` (a price level), or `earnings`
(days), and `symbol="*"` means any holding. Review them with `list_alerts` and drop
one with `remove_alert`. Rules are stored as plain JSON at
`~/.financial-research-assistant/alerts.json` (override with
`FINANCIAL_RESEARCH_ALERTS_FILE`).

A rule that fires is also pushed to you the moment it triggers, rather than only
appearing in the digest text: the TUI raises an **OS notification**, plays a sound,
shows a toast, and writes a 🔔 line into the transcript (tool panels are collapsed
by default, so the line is what survives a dismissed toast); the headless CLI
prints it to stderr and does the same sound and banner.

The OS banner uses built-in tooling — `osascript` on macOS, `notify-send` on Linux
— so there's nothing to install. It's the only delivery that reaches you with the
terminal buried behind other windows. On macOS the first one may not appear until
your terminal app is granted permission under System Settings › Notifications. Set
`FINANCIAL_RESEARCH_ALERT_DESKTOP=0` to turn banners off.

The sound is a real audio file (`afplay` on macOS, `paplay`/`aplay`/`ffplay`/`play`
on Linux) played alongside the terminal bell, because the bell on its own is just a
BEL byte that many terminals drop — VS Code's integrated terminal disables it by
default, and macOS Terminal/iTerm profiles often ship with the audible bell off.
Set `FINANCIAL_RESEARCH_ALERT_SOUND=0` to silence it and keep the toast and line,
or point it at an audio file to choose your own.

## Scheduled tasks (work that runs later and finds you)

A chat turn ends when the answer does. Ask *"monitor NOMD earnings tomorrow and
analyse the results"* and, without this, the best the agent could honestly do is
tell you to come back — anything else is a promise nothing keeps.

`schedule_task` is the fix: it stores the work, and a background runner executes
it and **pushes the answer to you**. Just ask in chat —

> *"monitor NOMD's earnings tomorrow and analyse the results"*
> *"every weekday at 08:30, brief me on overnight moves in my holdings"*
> *"check on Friday whether NVDA's 10-Q has landed"*

— and the agent queues it, tells you when it will run and where the answer will
go. The system prompt makes calling it mandatory for anything in the future: the
model is instructed never to write *"I will …"* about a later moment without
having scheduled it.

**The confirmation is enforced, not trusted.** A model that says "✓ Scheduled"
without calling the tool would leave you waiting for work that doesn't exist —
measured at roughly one real call in four on `claude-haiku-4-5`. So the answer's
claim is checked against the turn's actual tool calls: if a task was claimed but
never created, the details are recovered from the exchange and the task is created,
and the reply says so (`✅ Task created — [s1], running …`). If even that fails, the
claim is visibly retracted rather than left standing. Either way the reply and the
queue agree — check with `--tasks`.

From the shell, no model needed:

```bash
# WHEN|PROMPT[|REPEAT]  — when accepts '2026-08-14 09:00', 'tomorrow 9am',
# 'friday', '+2h'; repeat is once/hourly/daily/weekdays/weekly
financial-research-assistant --schedule 'tomorrow 9am|Analyse NOMD Q3 results vs consensus'
financial-research-assistant --schedule '08:30|Pre-market brief on my holdings|weekdays'
financial-research-assistant --tasks              # what's queued, with ids and outcomes
financial-research-assistant --unschedule s1      # or 'all'
```

### Running what's due

Three entry points over the same pass — pick whichever suits the machine:

```bash
financial-research-assistant --serve       # the always-on service (see deploy/README.md)
financial-research-assistant --run-due     # one pass, then exit (for cron/launchd)
financial-research-assistant --watch 60    # stay running, same pass every 60s
```

`--serve` is the one to run on a host that stays up. The job pass, the Telegram
inbox and the event watchers each get their own loop, so a ten-minute report
never leaves a phone message waiting. Chat and background work take separate
model lanes (`FRA_BACKGROUND_RUNS`, default 1), and a loop that crashes is
restarted with backoff while the others carry on. On SIGTERM it stops claiming
work and gives running turns `FRA_STOP_GRACE` seconds (60) to finish. A task
cut off after that hands its claim back, and an unanswered message was never
marked read, so both are picked up after the restart. A watchdog thread exits
the process if the event loop freezes or a pass outlives its job timeout
(`FRA_WATCHDOG_MINUTES`, `FRA_JOB_TIMEOUT_MINUTES`), so the supervisor restarts
it. `--status` shows what it is doing; `--status --check` is the exit-code form
for health checks.

`--run-due` is model-free until it actually finds work, so an empty tick costs
nothing and it's cheap to run often:

```cron
# every 15 minutes, run anything due
*/15 * * * *  cd /path/to/repo && .venv/bin/financial-research-assistant --run-due
```

<details>
<summary>launchd equivalent (macOS)</summary>

```xml
<!-- ~/Library/LaunchAgents/com.you.fra-scheduler.plist -->
<plist version="1.0"><dict>
  <key>Label</key><string>com.you.fra-scheduler</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/repo/.venv/bin/financial-research-assistant</string>
    <string>--run-due</string>
  </array>
  <key>WorkingDirectory</key><string>/path/to/repo</string>
  <key>StartInterval</key><integer>900</integer>
  <key>StandardErrorPath</key><string>/tmp/fra-scheduler.log</string>
</dict></plist>
```
`launchctl load ~/Library/LaunchAgents/com.you.fra-scheduler.plist`
</details>

A scheduled run is an ordinary turn — same tools, same read-only broker boundary,
same tracing and long-term memory — with its own session id **per run**
(`task-s1-20260930T091500`), so it starts from a clean conversation instead of
inheriting whatever the TUI was mid-thought about, or what yesterday's run of the
same task said. The one difference in tools: nobody is watching an unattended
turn, so the two tools that read a path on the server's disk
(`import_ibkr_statement`, `ingest_document`) are not bound for it.
A recurring task reschedules from its **due** time, not from when it finished, so a
daily 09:00 brief doesn't creep into the afternoon; a machine that was asleep for
three days resumes at the next real occurrence rather than firing three catch-ups.
A task that errors is retried on the next tick, then parked after three attempts so
a permanently-broken prompt stops billing model calls forever. A tick runs at most
10 tasks (`FINANCIAL_RESEARCH_TASK_BATCH`) and says how many it deferred, so a
backlog drains over consecutive ticks instead of firing everything at once.

A runner that is killed mid-task — SIGKILL, a laptop closing, an OOM — never gets
to record the outcome, so the task would otherwise sit claimed forever: skipped by
every later tick while `--tasks` still showed it as running. A claim older than 30
minutes (`FINANCIAL_RESEARCH_TASK_CLAIM_TIMEOUT`) is therefore treated as
abandoned and picked up again, counting as one of the three attempts so a prompt
that reliably kills its runner still parks instead of looping.

Recurrence follows the **wall clock**, not a fixed number of hours: a daily 09:00
brief stays at 09:00 through a daylight-saving change rather than drifting to
08:00 or 10:00. (`hourly` is the exception, and stays a real hour — there is no
time-of-day to preserve, and a wall-clock hour would skip a run at fall-back.)

A task can be pinned to a **zone** (`schedule_task(..., time_zone="America/New_York")`),
and then its wall clock is that zone's, whatever the host runs in. Anything tied
to the US close is pinned this way. A server in a zone without daylight saving
would otherwise run a "17:15" job at 16:15 New York all winter, before the close
it exists to report on. Repeats are `once`, `hourly`, `daily`, `weekdays`,
`weekly`, `monthly` and `monthly-first-weekday`. `monthly` keeps the day it was
first set for (Jan 31 → Feb 28 → Mar 31, not stuck on the 28th), and times like
`1st of the month 8am` or `first weekday of the month 08:00` parse directly.

A **recurring** task that fails all three attempts skips that run and stays
scheduled for the next one; it tells you once. Parking it, as a one-shot is
parked, would let one day's provider outage end a daily report for good.

**Jobs.** A task can run a registered job instead of a model turn on its text:
the reports below and the model-free Flex sync. A job's period comes from the
occurrence it was scheduled for, not when it ran, and delivered reports are
entered in a ledger keyed by period, so a retry or restart never sends the same
week twice. `--reports-setup` schedules the default jobs that apply to this
install and aren't already scheduled. It's safe to re-run, and the server runs
it on first start.

### Daily, weekly and monthly reports

Three scheduled reports, built as **jobs**. Code resolves the period, fetches
the data and computes every figure; the model only writes the commentary.
`--reports-setup` schedules them, together with a model-free Flex sync at 16:45
New York so each report reads fresh positions:

| Report | When | Covers | Model |
|---|---|---|---|
| Daily close | weekdays 17:15 New York | the session just closed: your holdings, indexes, sectors, rates/dollar/oil/gold/VIX, movers, earnings and ex-dividends in the next 7 days, open calls that have moved past ±10% | quick tier, ≤120 words |
| Weekly | Saturday 09:00, your zone | the week from the close before Monday to Friday's: what carried it (P/L by holding), vs the S&P 500, sectors, the account's time-weighted return where the Flex file reaches, next week's calendar, calls settled | primary, 3–5 bullets |
| Monthly | first weekday 08:00, your zone | last calendar month: the performance review, the track record of the agent's own ideas by conviction, plus the sections other features add | primary, 4–6 bullets |

Every figure is labelled with what it measured. "Your holdings −1.11%" is the
price move of the positions on file between two named closes. It excludes
cash, options and trades since the statement, and it isn't the account's
return, which comes from the IBKR file where that file covers the window. If
the file covers only part of a month, the title says so and the S&P beside it
covers the same days. Windows are labelled from the session they really start
on (the Friday before a Monday, not a Sunday with no close). The period comes
from the schedule, so a weekly report that runs late still covers its week.
The report ledger sends each period once, and a daily report on a market
holiday sends nothing.

**The commentary is checked.** Every number in it must round from a figure on
the sheet. The model may write "0.8%" for the sheet's 0.84%, but may not compute
a difference or an average. A draft that does is retried once with the
offending figures named, and if it still does, the report goes out without
commentary and says why.

On demand: `/report daily|weekly|monthly` from the phone,
`--report weekly [--no-deliver]` from the shell, or ask in chat ("send me
last month's report"), which calls the `periodic_report` tool. A daily, weekly
or monthly report assembled by hand through `render_report` is refused, the
same way a hand-built review is. Set `FRA_DAILY_PDF=1` to get the daily as a
PDF too.

### Guardrails on autonomous work

An agent that acts on its own needs a way to tell it to stop. There are three,
cheapest first:

- **Quiet** (`/quiet 2h`, `/quiet off`, or nightly via `FRA_QUIET_HOURS=22:00-07:00`):
  keep working, but hold back pushes that can wait.
- **Pause** (`/pause`, `--pause`; undo with `/resume` or `--resume-autonomy`;
  `FRA_AUTONOMY=off` in the env outranks both): no reports, no event analyses,
  no ideas. Chat still answers. The switch is stored in the state directory, so
  a restart doesn't undo it.
- **Budget** (`FRA_AUTONOMY_DAILY_TOKENS`, `FRA_AUTONOMY_MONTHLY_TOKENS`): every
  model call the agent makes on its own goes through one helper
  (`autonomy.ask`). Once the day's or month's tokens are spent, that helper
  stops calling the model, reports go out without commentary, and events go out
  as plain facts. Scheduled prompt tasks count toward the spend but aren't cut
  off, since you asked for them. Chat is never counted or capped.

Every job, task and phone turn is recorded in `runs.jsonl`: when, how long, the
tokens, the tools it called, where the answer went, and the error if it failed.
`/runs` shows the latest, `--reports` lists the reports sent and the recent runs,
and `/status` carries the autonomy line (on or paused, tokens today and this
month). A report built on positions more than `FRA_STALE_POSITIONS_DAYS` (5)
old says so on the sheet and in the message, because a stale book usually means
the Flex sync is failing.

### Delivery channels

Channels are a registry (`channels.py`) in the same shape as the
[pluggable brokers](#pluggable-brokers): each declares a key, whether its env
resolves, and how to send. Shipped: **telegram**, **desktop** (an OS banner) and
**stdout**. `NOTIFY_CHANNELS` pins a subset; unset means every configured one; a
task can name its own. Adding email or Slack is one `register_channel` call.

**Telegram.** Create a bot with [@BotFather](https://t.me/botfather), message it
once, and read your chat id from
`https://api.telegram.org/bot<TOKEN>/getUpdates`:

```bash
export TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
export TELEGRAM_CHAT_ID=987654321
financial-research-assistant --notify-test    # confirms it end-to-end
```

Long answers are split at line boundaries rather than truncated (the
recommendation is usually at the end), and sent as plain text — an analysis is
full of `*`, `_` and `$` from tickers and figures, and Telegram rejects a whole
message whose Markdown doesn't parse.

Delivery is best-effort: a channel that's down is reported, never fatal. Work that
finished must not be re-run — and re-billed — because a notification API blipped.
An answer that reached no channel **able to carry it** is parked on its task and
re-sent at the top of every later tick (before any new model call), for up to 8
attempts; `--tasks` shows it as *answer waiting to be delivered*. Redelivery costs
one HTTP request, which is why it retries far more patiently than a failed *run*
does.

"Able to carry it" is a property each channel declares (`Channel.full_content`),
because a desktop banner is a notification, not a delivery: it shows the first
couple of hundred characters and is gone. Counting one as success meant that when
Telegram blipped on a machine with banners enabled — the default — the analysis
was silently reduced to its own first paragraph and never re-sent. A banner now
tells you the answer is ready without settling the delivery, so the full text
still arrives once a channel that can carry it comes back.

A run only counts as done if it produced something that looks like an answer. A
reply that opens with "I don't have a record of that" or "could you clarify", or
that is under 40 characters, is treated as a failure and retried — otherwise a
confused non-answer arrives on your phone labelled as the analysis you asked for.
The check is deliberately shallow (opening lines, short replies only): judging
quality properly would mean a second model call per task. A retry that is still
pending isn't pushed to you; only the final verdict is, so one broken task costs
one notification rather than three.

### Reports as files, not walls of text

`render_report(title, markdown, highlights)` typesets a summary as a **PNG + PDF
sheet** — headline, stat tiles, body — and sends it through the same channels,
which is the difference between a scheduled analysis being read on a phone and
being scrolled past. Ask for *"a PDF"*, *"an infographic"*, or *"send me a one-pager"*, or let a
scheduled task produce one.

Per report, the agent can pass `theme="dark"` (image only — the PDF stays
print-friendly) and `output="image"` or `"pdf"` to send just one of the two files;
both fall back to the configured defaults. Operator-level, `FINANCIAL_RESEARCH_REPORT_RENDERER=chrome|fpdf2`
pins the engine, which is worth doing in a container or when reproducing a
rendering bug.

`highlights` is the stat-tile row, one per line as `label | value | note` (max 6).
The body is ordinary markdown: headings, tables, and `>` for a warning callout.
Files land in `~/.financial-research-assistant/reports`.

**Performance reviews are one call, not a prompt.** Ask in a sentence — *"produce
my year-to-date portfolio performance review and send it to Telegram"*, *"generate
a performance review for last month"* — and `render_review(period, observations,
stance)` computes every figure, renders the sheet and delivers it. The agent
supplies only `observations`: three to six bullets on what the numbers mean.

That split is the point. It has no parameter through which a figure could arrive,
so it cannot get one wrong. Three delivered sheets in a row printed `$7,410.61`
where the source said `$7,412.61` — each time because the model had rewritten a
figure by hand, and twice while under explicit instruction not to. Docstrings ask
for compliance; a signature removes the choice.

`portfolio_review_brief(period=...)` returns the same figures **without**
rendering, for shaping a report by hand or answering in chat.

`period` takes `ytd` (default), `last month`, `this month`, `last quarter`,
`Q2 2026`, `2025`, `last 90 days`, or `2026-01-01..2026-06-30`. Windows anchor on
the **statement's last session, not today**, so a file pulled on the 7th and read
on the 10th still means the same month by "this month"; a window the statement
cannot reach is refused rather than quietly answered for a different span, and one
it can only partly cover says so in its own title.

This exists because four generated reviews were each wrong in a different way.
Pasted into a prompt, the method was one dropped newline from vanishing — and the
reply looked finished either way. With the method intact, the figures still had to
be retyped, and one sheet printed a total two dollars off the line items directly
above it. A template that computes has neither failure available to it.

**Earnings write-ups are one call too, and every figure carries its window.**
*"Analyse the latest earnings report of FISV and share findings on Telegram"* goes
to `render_stock_report(symbol, observations, stance)`, which pulls the quarter
from SEC 10-Q XBRL and the price side from daily history. `stock_brief(symbol)`
returns the same figures without rendering.

What makes this more than a second copy of the review template is what the tiles
*say about themselves*. "Net margin 11.8% — **level, not a change**". "Max drawdown
−66.3% — **trailing 12 months**". "From the high −78.0% — **high since 2023-05-01**".

That labelling is the whole fix, and it came from a delivered sheet whose three
decline figures — −70%, −63% and −66.3% — read as three guesses at one number.
Only the last was tool-computed, and it was **right**: −66.3% is the trailing-year
drawdown. But the sheet never said "trailing year", and measured from its actual
high the stock was down 78%. A correct figure sat on the page answering a question
nobody had asked, and the prose drifted around it precisely because "the decline"
had no fixed window to be checked against. A number is not enough on its own.

The same sheet put revenue at $4.96B where the as-reported figure is $5.29B —
plausibly the adjusted, non-GAAP number a company headlines, which is a different
number. The sheet said neither, so the brief states the basis on the tile and says
in the body that adjusted figures are not carried.

Both guards refuse rather than warn. A hand-built earnings sheet — an earnings
word in the title, two of a quarter's figures in the tiles — is turned back with
the name of the tool that can build it properly. A peer comparison, valuation or
risk sheet is not an earnings write-up and still renders through `render_report`.

**The call gets a badge.** `stance="HOLD | trim 50% at $40–$42"` puts a
colour-coded pill under the title on both the image and the PDF — buy /
accumulate / overweight read positive, sell / reduce / trim / underweight
negative, hold / neutral / watch neutral. What to do about a stock is the
reader's first question, and a stance is the one thing on the sheet that is not a
figure, so it gets its own element rather than a stat tile. A stance written as a
heading (`## Rating: SELL`) is picked up automatically; a word the palette cannot
tone is refused outright and reported back, because badging the wrong colour on
this field is worse than leaving it off.

The badge is **this report's own call**, not the street's. When they disagree —
the interesting case — put the consensus in a tile
(`Analyst Consensus | BUY | 12 analysts, mean $47.28`) and the sheet shows both
instead of quietly picking one.

**A verdict written as a heading still reaches the cover.** A heading shaped
`## Fear Price: $32.00 – $38.00` — a fear price, fair value, or price target —
is promoted to a stat tile on both the image and the PDF, taking the last tile's
slot if all six are full. Without it the one number a report exists to produce
could sit on page 2 while the cover showed six context figures, which is exactly
what happened to an NVO sheet whose entire second half priced a fear zone the
image never mentioned. Promotion is deterministic — it re-reads the heading the
model wrote rather than asking a second model what mattered, so it costs nothing
and cannot invent a figure. It only fires when the text after the colon *is* a
figure, so `## Coverage: 12 analysts` stays prose; write the tile yourself in
`highlights` when you want a different label or note.

**Light or dark, per artifact.** `FINANCIAL_RESEARCH_REPORT_THEME=dark` renders
the **cover image** on a dark surface; the **PDF stays light** unless you also set
`FINANCIAL_RESEARCH_REPORT_PDF_THEME=dark`. They are separate on purpose: the image
is read on a phone, and a dark PDF lays down a full page of ink when printed.

Both themes are *selected*, not one flipped into the other — the dark row takes its
own steps from the same ramps and its diverging pair is re-validated against the
dark surface (CVD ΔE 19.2, contrast ≥ 3:1), because inverting light values fails
contrast on a dark background.

Two details decide whether it is actually readable. The sheet is rendered at 3x
(`FINANCIAL_RESEARCH_REPORT_SCALE`) and sized to its own measured height, so
there is no dead band and body text survives being pinched into. And it is sent as
a **document, not a photo**: Telegram's `sendPhoto` re-encodes to JPEG and
downscales, which turns dense body text to mush no matter what resolution it was
rendered at — as a document the bytes arrive untouched and the client still shows a
tappable preview.

**The PDF is the document; the image is an infographic of it.** Every report gets
one — the headline figures plus the report's own series, charted — at any length:
page 1 of a long report is the masthead and whatever happened to fit, and page 1 of
a short one is prose a picture summarises better. A report with neither stat tiles
nor series falls back to page 1, since a sheet with nothing to visualise adds
nothing. The PDF is always the full document, sized to its content when short and
paginated when long.

The charts are extracted deterministically from the body, with no second model
call: a run of list items under one heading becomes a series when at least three
carry a percentage — the shape of *top holdings*, *sector exposure* and *movers* —
**markdown tables chart too**, taking the first column as labels and the column
with the most single-number cells as values (preferring percentages, since
`$13.58` and `+15.3%` cannot share an axis), and a breakdown written *inside a
sentence* — `Geographic diversification (North America ~65%, Europe ~35%)` — is
picked up as a last resort, since narrative reports bury their only data that way.
Dollar series are labelled in dollars.

A report with none of those charts nothing, and that is usually the report's shape
rather than a bug: a pure narrative digest has no composition, ranking or history
to draw. The prompt pushes the model to look for one before writing prose. Signed values (`+28.6%` / `-20.7%`) render as diverging bars so gains and
losses read as opposites; unsigned ones as ranked magnitude bars. The palette is
the validated diverging pair from the project's data-viz reference, and the sheet
says which report it summarises so it never poses as the document itself.

**Two renderers, one content model.** Both consume the same markdown and
highlights:

| | Renderer | Notes |
|---|---|---|
| Preferred | **headless Chrome/Chromium** | The CSS template — better typography. Auto-detected on the usual macOS and Linux paths; override with `FINANCIAL_RESEARCH_CHROME`. |
| Fallback | **fpdf2** (built in) | Used when no browser is installed — including this project's own Docker image. Plainer, ~10x faster, always available. |

`pypdfium2` rasterises page 1 whichever renderer produced the PDF. With neither
renderer available the HTML is still written and its path returned — a finished
analysis is never lost to a rendering problem.

The fallback embeds **Inter** (bundled, SIL OFL 1.1, licence beside the files) so
a sheet renders identically on every machine — relying on system fonts made the
output depend on the host: Arial on macOS, DejaVu on Linux, transliterated ASCII
in a slim container. Override with `FINANCIAL_RESEARCH_REPORT_FONT`; if the
bundled font is ever missing it falls back to a system face, then to the built-in
PDF fonts, which are Latin-1 and *raise* on an em dash — hence the transliteration
of last resort.

Channels declare file support explicitly (`Channel.send_file`), so a
desktop-banner channel is skipped rather than being handed a PDF it can't show.

### Replying from your phone (opt-in)

With `TELEGRAM_ALLOWED_CHAT_IDS` set, `--serve` (and `--watch`) read the bot's
inbox: message it and the agent answers, keeping context per chat. `--run-due`
does not. Slash commands are handled without a model call: `/status`, `/tasks`,
`/cancel <id>`, `/help`, and the report, ideas and autonomy commands described
below. An unknown one gets the help text rather than a model turn. Anything else
is a prompt, so you can schedule from the phone too. A message is marked read
only once it has been answered, so a restart mid-answer answers it again rather
than dropping it.

This is **off unless you list chat ids** — it is not implied by having configured
outbound. Anyone can message a bot whose username they guess, and an inbound
message is untrusted text driving a tool-using agent that can read your imported
statements and positions. Messages from chats not on the list are dropped unread
(no reply, which would confirm the bot is live). The read-only broker filter still
applies, so the reach is exactly an interactive turn's: it can research and read
the account, never trade. The bot token never appears in an error or a log — it's a
bearer credential in the request URL, and it's masked on the way out.

## Durable conversation memory (survives restarts)

By default the graph checkpoints to an in-process `MemorySaver`, so conversation
state is lost when the process exits — `--resume NAME` replays a session's saved
*transcript* but the model starts with empty memory. Set
`FINANCIAL_RESEARCH_CHECKPOINT_DB` to a path (or `default` for
`~/.financial-research-assistant/checkpoints.db`) to checkpoint conversations to
SQLite with `AsyncSqliteSaver` instead: `--resume` then restores the model's
*actual* memory, and `/new` erases just that session's stored history. The
durable saver is opened and closed within each turn's own task (mirroring the MCP
session's task-affinity rule), so nothing dangles at shutdown. Unset (the
default) behavior is unchanged, and `--fake`/tests stay hermetic.

## Thesis journal (gets held to its calls)

The other learning layers check the *process* or a fixed regression set. None of
them ever checks whether the assistant's **market calls** were right — so all of
them can be green while it is consistently wrong.

When it takes a side — a bull/bear verdict, a DCF concluding under- or overvalued,
"this looks cheap" — it calls `record_thesis` with the reasoning and a horizon. The
entry price is captured from market data at that moment, not typed by the model. On
a later runner tick (`--run-due` / `--watch`) the call is scored: entry price vs the
price **at the horizon it named**, and the same window for the benchmark. The tick
that happens to run the scoring doesn't set the window — a scheduler that was off
for a month still scores a 90-day call over 90 days, so a call that was right at
its horizon can't be recorded as wrong because nobody was watching.

```
financial-research-assistant --theses          # every call, model-free
financial-research-assistant --theses NVDA     # just one ticker
```
```
Open calls (1) — not yet scored:
  [t2] BEARISH AAPL   from 231.40 on 2026-08-05 · scores 2026-11-03
Scored calls (1):
  ✓ [t1] BULLISH NVDA   +18.4% over 90d · +7.1% vs SPY
Track record: 1/1 directionally right (100%) · mean +7.1% vs benchmark
  (only 1 scored call(s) — far too few to mean anything; report it as such)
```

Four things worth knowing:

- **Scoring is model-free.** Two price lookups and a subtraction, so it rides the
  tick that already exists, costs nothing, and cannot hallucinate an outcome.
- **The benchmark always travels with the score.** `bullish +8%` against an index
  that did `+14%` is a hit with negative alpha, and the output says both.
- **A data outage never becomes a miss.** A call the price source can't answer for
  stays open for a later tick.
- **It feeds back in.** A scored outcome is filed as a `lesson` memory (when
  `MEMORY_BACKEND` is on), so the next time that ticker comes up the assistant reads
  what it said and how it went — and is told to say so rather than quietly repeat a
  view it has already been wrong about.

**This is not a track record to trade on.** It is a handful of past calls on
whatever you happened to ask about — not a random sample, and nowhere near large
enough to say anything about the next one. Every surface reports it as calibration
and says so; the point is to stop the assistant repeating a confident line it has
already got wrong, not to advertise a hit rate.

Override the store location with `FINANCIAL_RESEARCH_JOURNAL_FILE`
(default `~/.financial-research-assistant/journal.json`, written `0600`).

## Long-term memory (learns across conversations)

Durable *conversation* memory (above) makes one thread survive a restart.
**Long-term memory** is different: a curated store of durable *facts* about you —
risk tolerance, watchlist tickers, tax situation, how you like answers formatted —
that carries across *every* session, scoped by `MEMORY_USER` rather than the
session id. It's the piece that makes the agent feel like it learns.

Enable it with `MEMORY_BACKEND`:

| value | store |
| --- | --- |
| unset (default) | off — nothing persists across sessions; `--fake`/tests stay hermetic |
| `local` | zero-dependency JSONL under `MEMORY_DIR` (default `~/.agent-builder/memory`), one file per `MEMORY_USER`; **keyword** recall |
| `semantic` | the same JSONL store, but **embedding-based** recall — finds *related* memories, not just term-overlapping ones ("how aggressive should I be?" recalls "my risk tolerance is high"). Uses an OpenAI-compatible embeddings endpoint (honors `OPENAI_API_BASE`; `MEMORY_EMBED_MODEL`, default `text-embedding-3-small`); falls back to keyword when embeddings are unavailable |
| `mem0` | the [mem0](https://github.com/mem0ai/mem0) memory layer (optional dep) for semantic recall + automatic fact extraction + contradiction handling |

Semantic recall reaches every layer — facts, reflection lessons, and feedback
exemplars all rank through the backend, so switching to `semantic` upgrades
retrieval everywhere at once. `MEMORY_SIM_THRESHOLD` (default 0.25) tunes how
related a memory must be to surface.

Facts get in two ways, both deduped on write:

- **The model curates them.** With memory on, the agent gets `remember`,
  `recall`, `forget`, and `list_memories` tools and is told to store *lasting*
  facts (never transient quotes/prices). So *"from now on keep my answers brief"*
  or *"add NVDA to my watchlist"* sticks.
- **Auto-capture.** After each turn a message that reads as a durable statement
  (an explicit *"remember …"*, a preference, *"my goal is …"*) is stored
  automatically — passing questions and one-off figures are not, so the store
  stays clean instead of hoarding every exchange.

At the start of each turn the most relevant facts are recalled and injected into
the prompt, so the model answers with them in mind. Everything is inspectable and
editable from the CLI — no model needed:

```bash
MEMORY_BACKEND=local financial-research-assistant --memory                 # list stored facts
MEMORY_BACKEND=local financial-research-assistant --memory forget:watchlist # prune matching facts
```

**It also learns from its own work.** After a deep-research run (`--research`), a
reflection pass distills *lessons* — sections that came back thin or unavailable,
plus (with a model) a few process notes — and stores them as `[lesson]` memories.
The next research run on that ticker recalls them and addresses the known gaps
instead of repeating them. Set `RESEARCH_REFLECT=0` to skip the extra critique
call; the free deterministic gap-lessons still run.

**And it learns from your verdicts.** Rate a reply in the TUI with `/good [note]`
or `/bad [note]` — good answers are stored as `[exemplar]` memories, poor ones as
`[avoid]`. On a similar future question the relevant ones are injected as few-shot
guidance ("emulate these / avoid these"), so answers drift toward what you like.
No fine-tuning — it copies *approach*, never figures (numbers are always recomputed
from tools).

See [references/memory-guide.md](references/memory-guide.md) for the write policy,
PII/scoping cautions, the reflection layer, and how to plug in a richer backend.

## Usage

```bash
financial-research-assistant                        # chat TUI
financial-research-assistant --fake                 # chat TUI, offline fake model
financial-research-assistant --prompt "AAPL quote"  # headless: answer -> stdout
financial-research-assistant --prompt "hi" --fake   # headless, offline
financial-research-assistant --session work         # separate conversation thread
financial-research-assistant --resume work          # reopen session "work" (replays it)
financial-research-assistant --list-sessions        # list saved sessions and exit
financial-research-assistant --login                # list model providers, exit
financial-research-assistant --login anthropic-key  # store a credential, exit
financial-research-assistant --logout [PROVIDER]    # stop using / forget one, exit
MEMORY_BACKEND=local financial-research-assistant --memory   # list long-term memory, exit

# scheduled work (see "Scheduled tasks")
financial-research-assistant --schedule 'tomorrow 9am|Analyse NOMD Q3 vs consensus'
financial-research-assistant --tasks                # what's queued, with outcomes
financial-research-assistant --unschedule s1        # cancel one (or 'all')
financial-research-assistant --run-due              # run everything due, exit (cron)
financial-research-assistant --watch 60             # same pass every 60s, stay running
financial-research-assistant --notify-test          # check the delivery channels
python -m financial_research_assistant.main --prompt "hi" --fake   # module form
```

Example questions (with IBKR configured): *"What are my largest positions and
their weights?"*, *"Show AAPL's 1-year price history and its return."*,
*"What's the current quote for MSFT, as of when?"*, *"Search for SPX option
contracts expiring next month."*, *"What are MSFT's fundamentals and analyst
price target?"*, *"When does NVDA next report earnings?"*, *"Project my forward
dividend income and yield-on-cost."*

## TUI

The chat TUI is a small agent console: your message renders as a right-aligned
bubble (click to copy) and each reply as a left `● Agent` bubble. Every IBKR
tool call appears as a collapsible panel titled `🛠 [Agent] get_price_snapshot
✅ (1.2s)` (expand for args + result); the model's reasoning renders as a
collapsible `💭 thinking` panel (toggle with `/toggle_thinking`). A status bar
shows `model · provider · ctx N% of CAP · in/out tok · cache R +W w · $cost`, plus a
spinner + elapsed timer while a turn runs (cache reads and writes are listed
separately — they bill at very different rates). Type `/` for the command palette; **Esc** cancels a running
turn (a cancel that lands between a tool call and its result leaves the thread in
a state every provider rejects, so the next turn repairs it and says so rather
than erroring until you `/clear`); typing while busy **queues** the message; **Ctrl+O**/**Ctrl+T** collapse
all tool / thinking panels. `/compact` summarizes the older turns and rewrites
the thread so the running context (and `ctx %`) shrinks while recent turns stay
verbatim (this also happens automatically for **any** interface when
`AGENT_AUTO_COMPACT` is set — see below). Commands: `/new`, `/compact`, `/clear`, `/sessions`,
`/resume [NAME]`, `/models [NAME]`, `/login [PROVIDER]`, `/logout [tier|PROVIDER]`,
`/config`, `/toggle_thinking`, `/thinking on|off`, `/copy`,
`/good [note]`, `/bad [note]`, `/memory [forget TEXT]`,
`/export NAME.html|NAME.jsonl`, `/hotkeys`, `/theme`, `/help`, `/quit`. With
long-term memory on, `/good`/`/bad` rate the last answer so the agent learns from
it, and `/memory` reviews or prunes everything it has learned (facts, lessons,
feedback) without leaving the chat. `/models` lists every model from every
configured provider and switches provider + model together; `/config` shows
what's stored, which provider is active, and the effective context window per
model. (The
fake graph has no tools, so tool panels appear only against a real model + IBKR;
the reasoning panel shows in fake mode too.)

## Tests (offline)

```bash
uv run pytest tests/ -q      # or: python -m pytest tests/ -q  (venv activated)
```

Covers the adapter event round-trip, tool_start/tool_end pairing, the TUI
round-trip via the Textual pilot, checkpointer wiring, the local calculators,
the **read-only broker filter** (asserts no order/watchlist-write tool ever
survives — for IBKR and any registered broker), the **broker-provider registry**
(a second broker mounts alongside IBKR), and the **statement format dispatch**
(IBKR CSV vs OFX/QFX routing, plus the OFX importer when the `[ofx]` extra is
installed) — all without a network or API key.

## Eval

```bash
python eval/evaluate.py --fake    # offline plumbing run
python eval/evaluate.py           # against your configured real model + IBKR
python eval/ci_gate.py --min-score 0.8   # fail if the mean score regresses (CI)
```

Scorers: `contains` / `regex` (match stdout), `trajectory` (which tools were
called), `llm_judge`, and `rubric`. The judge scores with the **same configured
model as the agent** (via `graph._make_llm`), so it honors `OPENAI_API_BASE` (local
servers) and `MODEL_PROVIDER` (Anthropic/Google/…) — no bare `OPENAI_API_KEY`
required. Set `EVAL_JUDGE_MODEL` to override the judge model (e.g. a stronger
cross-family one) — leaving it unset means the model is grading itself, which the
harness now says out loud rather than letting it pass unnoticed. Results append to
`eval/results.jsonl`.

**Every item runs against its own empty memory store.** Long-term memory is on for
the run — recall and write-back are part of what's being measured — but it starts
blank and is thrown away afterwards. Sharing the developer's real store made the
gate dishonest in both directions: a regression could score full marks by recalling
the answer a previous run had stored, and each run quietly filed its eval questions
as facts about you. The same isolation runs per **A/B arm** and per `--repeat`, so
the candidate arm can't score well by remembering what the baseline just answered,
and repeats stay independent samples instead of echoes of run 1.

### `rubric` — grading the derivation, not the answer

The other four can't see *how* an answer was reached. `trajectory` checks a tool was
called but not with what arguments or to what end; `llm_judge` reads the final text
and nothing else. A financial answer can land on the right number from the wrong
source, over the wrong period, or with its assumptions unstated — and each of those
is invisible to a single "is this good" score that a capable model can talk its way
into. (The 2026 benchmark literature makes the same point: on BigFinanceBench the
best frontier agent scores 58.8% on derivation rubrics, and final-answer accuracy is
"a useful but lossy proxy".)

So a `rubric` item shows the judge the **tool trajectory** alongside the answer and
scores each dimension separately:

```json
{"query": "Was NVDA expensive in January 2025?",
 "eval_type": "rubric",
 "criteria": [
   {"dimension": "as_of",  "requirement": "The answer is about JANUARY 2025, not today…", "weight": 3.0},
   {"dimension": "source", "requirement": "Uses a source that actually has history…",     "weight": 2.0}]}
```

The per-dimension breakdown is persisted to `results.jsonl` and printed by the
improvement report, so a failure names *which* part of the reasoning broke. Two
deliberate choices: a dimension the judge omits scores **0**, not "skip" (dropping
it would shrink the denominator and quietly raise the mean), and derivation misses
are **reported, never auto-applied** — a requirement is prose written for a judge,
and turning it into a system-prompt rule is a judgement a human should make.

### A/B a code change against a commit (`eval/ab_compare.py`)

`--ab` below measures a candidate **prompt addendum** — extra text layered onto
the prompt. To measure an edit to the **code itself** (a trimmed system prompt, a
reworded tool description, changed tool gating), use `ab_compare.py`, which scores
the working tree against any git ref:

```bash
python eval/ab_compare.py                # working tree vs HEAD
python eval/ab_compare.py --items 10     # cheap subset first
python eval/ab_compare.py --repeat 3     # average out llm_judge noise
python eval/ab_compare.py --ref main --tolerance 0.01
```

The baseline arm is a throwaway `git worktree` at the ref, loaded by pointing the
child's `PYTHONPATH` at that checkout — so there is no frozen copy of the old
prompt to maintain and no environment variable that could override the system
prompt in production. Both arms get their **own `MEMORY_DIR`** (otherwise arm 1's
answers are recalled into arm 2's prompt) and an empty prompt addendum, and each
is preflighted with one cheap prompt so a misconfigured arm fails loudly instead
of scoring 0.00 on everything and looking like a catastrophic regression. Exits
non-zero if the mean drops past `--tolerance`, so it can gate a merge.

Most of the dataset is `trajectory` items, which assert *which tool* a query
should drive — exactly what a prompt or tool-description edit is most likely to
break.

### Eval-driven self-improvement (learns from its own scores)

The harness can close the loop — turn eval **failures** into a **proposed prompt
improvement**, measure it, and apply it only if it actually helps:

```bash
python eval/evaluate.py --diagnose          # run + diagnose misses → propose an addendum
python eval/evaluate.py --ab <candidate.txt>          # A/B it: baseline vs candidate mean
python eval/evaluate.py --ab <candidate.txt> --apply  # install it IF it beats baseline
```

`--diagnose` finds trajectory misses (the dataset says which tool a query should
have driven; the trace says which ran) and writes a concrete routing addendum plus
a review report under `~/.financial-research-assistant/eval/`. `--ab` runs the
dataset twice — clean baseline vs the candidate addendum — and reports the delta.
With `--apply`, the addendum is installed **only if** it raises the mean by
`--min-delta`.

Crucially for a money tool, a learned change is never baked into code: it's a
**reversible data file** (`prompt_addendum.txt`, appended to the system prompt at
`FINANCIAL_RESEARCH_PROMPT_ADDENDUM_FILE`), the proposals are deterministic and
auditable (no model rewrites the prompt), and deleting the file reverts. The
human stays in the loop — nothing changes behavior without an explicit `--apply`
on a measured improvement.

## Signing in (`/login`)

Credentials can live outside `.env`, in a `0600` store at
`~/.financial-research-assistant/auth.json`:

```bash
financial-research-assistant --login                          # list providers
financial-research-assistant --login anthropic-key            # paste a key (masked)
financial-research-assistant --login openrouter               # browser OAuth (PKCE)
financial-research-assistant --login groq-key --tier quick    # just the cheap tier
financial-research-assistant --logout                         # stop using it, keep it
financial-research-assistant --logout groq-key                # forget it for good
```

In the TUI: `/login [PROVIDER] [tier]` (bare opens a picker), `/logout`,
`/models` to switch, and `/config` to see what's stored — never the secret itself.

**Why prefer it over `.env`.** A key in `.env` becomes a process environment
variable that every subprocess inherits. The store is `0600`, per-provider, and
redacted from trace spans, exported transcripts, and error text.

### Several providers, several models

Credentials are keyed by provider; each **model tier** records which one it uses.
So `/login anthropic-key` then `/login groq-key` leaves both in place, and
`/models` lists every model from every configured provider. Selecting one
switches the active provider **and** the model together — a model only works with
the key from its own lane, and setting one without the other is what produces
`404 model: <name>`.

Tiers are `default`, `quick`, and `subagent`, matching `QUICK_*` / `SUBAGENT_*`.
A tier with nothing stored inherits `default`, so signing in once covers
everything.

`/logout [tier]` stops using a provider but keeps its credential, so switching
back needs no re-login. `/logout PROVIDER` removes one for good.

### What `/login` asks

The key (masked; `getpass` when headless), then — for `openai-key` — a base URL,
then the models this credential serves, then their context window. Blank accepts
the suggestion, so a known provider is a few Enters. The endpoint is asked before
the models because it decides which models exist.

```json
"anthropic-key": {
  "models": [
    {"name": "claude-haiku-4-5-20251001",
     "input_cost": 1.0, "output_cost": 5.0,
     "cache_write_cost": 1.25, "cache_read_cost": 0.1,
     "context_window": 200000}
  ]
}
```

Everything is **per model**, because one key serves models that differ: an
Anthropic key serves a 1M Sonnet and a 200k Haiku, so no credential-wide figure
could be right for both. Entries are seeded from the built-in tables at login;
the first model listed is that credential's default. Hand-editing one entry
corrects that model alone.

**Costs** are USD per 1M tokens. Models the tables cover are filled in
automatically — which is why non-OpenAI providers need no input. A model they've
never heard of, such as a gateway serving one under its own name, gets every
field written as `0` so the shape to fill in is visible in the file. Strings
(`"5.0"`) read the same as numbers.

> **`0` means "not set", not "free."** An unpriced model shows no cost at all: a
> confident `$0.00` on a gateway that bills real money is worse than a blank.
> Both `input_cost` and `output_cost` must be non-zero, or the pair falls back to
> the table rather than billing output at a guess.

Cache reads and writes are billed **and displayed** separately
(`cache 400000 +200000 w`) because they price in opposite directions — a write
costs ~1.25x fresh input on Anthropic while a read costs ~0.1x, so one combined
figure would hide which you paid for. Both are subsets of the input total, read
from `input_token_details.cache_read` / `.cache_creation`.

### Precedence

Two chains, each resolving to exactly one winner:

| | Credential |
|---|---|
| 1 | explicit CLI/API override |
| 2 | **the store** |
| 3 | environment (`OPENAI_API_KEY`, …) |

The store outranks the environment deliberately: `load_dotenv()` turns a stale
key in `.env` into a real environment variable, and ranked the other way it would
silently shadow a fresh `/login` with no error to go on. The endpoint and the key
always come from the *same* source, so a key is never paired with a mismatched
`OPENAI_API_BASE`.

| | Context window |
|---|---|
| 1 | `OPENAI_CONTEXT_WINDOW` (global; overrides everything) |
| 2 | the model entry in `auth.json` |
| 3 | the built-in per-model table (and `pricing.json`) |
| 4 | a credential-wide `context_window` (older stores) |
| 5 | 128k fallback |

`/config` prints the effective window per model **and where it came from**, which
is the fastest way to answer "I configured X but it shows Y". Note that
`OPENAI_CONTEXT_WINDOW` is global and beats per-model values — if you set it for
a gateway and later add per-model windows, unset it.

The window matters beyond the `ctx %` gauge: tool-result clearing and
auto-compaction both divide by it.

A stored key may be a literal, `$VAR`, or `!command` (take it from a command's
stdout, so it can live in 1Password/`pass`). `!command` is execution from a
config file and stays off unless `FINANCIAL_RESEARCH_AUTH_ALLOW_EXEC=1`.

### Providers

| Provider | Status |
|---|---|
| `anthropic-key`, `google-key`, `groq-key`, `openai-key` | **Works.** Paste a vendor API key. `openai-key` also accepts a gateway base URL. |
| `openrouter` | **Works.** PKCE mints a real API key billed from your credits; nothing expires, nothing to refresh. |
| `codex` | **Works, with caveats.** ChatGPT OAuth routed through a local shim (`codex_proxy.py`) that speaks the Codex request shape. Since 4 Apr 2026 third-party traffic bills as *overage*, not against your ChatGPT plan. |
| `anthropic` | **Haiku only.** Since 28 Apr 2026 Sonnet and Opus return a bare 429 (no `anthropic-ratelimit` headers — a hard block, not a quota) for third-party OAuth clients, and loading an extra-usage balance does **not** lift it: the gate is on a promotional-credit flag, not your balance. Use `anthropic-key` for Sonnet/Opus. |
| `google` | **Blocked.** Banned Feb 2026 with account suspensions (incl. paid Ultra); Code Assist stopped serving the individual / AI Pro / AI Ultra tiers on 18 Jun 2026. Use `google-key`. |

The blocked subscription flows are implemented and tested end to end — PKCE,
exchange, storage, refresh — so they work the moment enforcement changes, and
`/login` states the caveat before opening a browser. **None of them provides
flat-rate subscription inference**; for that, point `OPENAI_API_BASE` at a local
model or a flat-rate gateway.

Adding a provider is a registration in `oauth.py`, not an edit to the TUI — flows
receive UI-neutral callbacks (`on_auth`, `on_device_code`, `on_prompt`,
`on_secret`, `on_status`), so one implementation serves the TUI, the CLI, and
tests.

## Tracing (optional)

```bash
uv pip install -e '.[tracing]'
AGENT_TRACING=1 OTEL_EXPORTER_OTLP_ENDPOINT=<collector>/v1/traces \
  financial-research-assistant --prompt "..."
```

`tracing.py` emits OpenTelemetry GenAI spans (agent turn + per-tool) — no-op
unless enabled. Point `OTEL_EXPORTER_OTLP_ENDPOINT` at any OTLP collector
(Langfuse, LangSmith, Arize Phoenix, …); `OTEL_EXPORTER_OTLP_HEADERS` carries
auth. As an alternative, `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` uses
LangGraph's native LangSmith auto-instrumentation. See `.env.example` for the
full set of tracing variables.

## Deploy

**Always-on:** [`deploy/README.md`](deploy/README.md) is the runbook for running the
agent as a service on a host that doesn't sleep. It covers the host, secrets,
moving your existing state across, the systemd unit, the outside liveness check,
encrypted backups and updating. The short version: one container runs `--serve`,
publishes no ports, and keeps all its state in one mounted directory.

The `Dockerfile` installs from `uv.lock` (so the image runs the versions the suite
ran on) plus the extras named in `EXTRAS` (default `anthropic`):

```bash
docker build --build-arg EXTRAS=anthropic,tracing -t fra .
docker run --rm -e ANTHROPIC_API_KEY=... -e MODEL_PROVIDER=anthropic fra --prompt "AAPL quote"
docker run --rm fra --prompt "hi" --fake   # no key needed
```

## Customization points

- **Live-data broker** — brokers register in `brokers.py` via
  `register_broker(BrokerProvider(key, config_from_env, filter_tools))`; the
  per-turn `tools.broker_tools_session` mounts every configured one. Reuse
  `brokers.make_readonly_filter(read_prefixes, write_deny=…)` for a new broker's
  trade-safe filter; IBKR's own policy lives in `filter_readonly` / `_is_readonly`
  / `_WRITE_DENY`. Adding order entry would require a human-in-the-loop approval
  gate (`interrupt`) — intentionally omitted. See [Pluggable brokers](#pluggable-brokers).
- **Statement formats** — parsers register in `statements._FORMAT_PARSERS`
  (`ibkr_csv`, `ofx`); `_detect_format` sniffs the source and `import_statement`
  dispatches. Add a broker's bespoke export by writing a parser that returns the
  normalized dict `store_statement` consumes and registering it there.
- **Delivery channels** — channels register in `channels.py` via
  `register_channel(Channel(key, configured, send, label))`; `deliver()` fans out to
  every configured one (or the `NOTIFY_CHANNELS` subset, or a task's own choice).
  Adding email, Slack or ntfy is one entry — `scheduler.py` and `tasks.py` never
  learn its name. A `send` must return whether it went and may raise; `deliver`
  never propagates either, because finished work must not be re-run over a failed
  notification. See [Delivery channels](#delivery-channels).
- **Local calculators** — add typed, docstring'd functions to `tools.py` and
  list them in `TOOLS`.
- **System prompt / model** — `SYSTEM_PROMPT` and `_build_real_graph()` in
  `graph.py`. Provider selection is `_make_llm()` (OpenAI-compatible by default,
  `init_chat_model` for `MODEL_PROVIDER=anthropic|google_genai|groq`).
- **Persistence** — set `FINANCIAL_RESEARCH_CHECKPOINT_DB` to checkpoint
  conversations to SQLite (`AsyncSqliteSaver`) so they survive restarts;
  unset keeps the in-process `MemorySaver`. See "Durable conversation memory".
- **Interface** — everything consumes `adapter.run_turn`; add an API server or
  REPL beside `tui.py` without touching graph code.
