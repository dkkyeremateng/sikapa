# financial-research-assistant

A LangGraph financial research assistant for your brokerage account —
**Interactive Brokers (IBKR)** by default, with a **pluggable broker layer** for
others. It answers questions about markets, holdings, and instruments by calling
a broker's market-data tools over MCP — filtered to **read-only** — plus local
analyst calculators, a **price-history charting** tool, and a **web-search**
tool for stock/market news. It ships a Textual chat TUI, a headless CLI, an
offline deterministic fake mode, and a generic eval harness.

> **The broker is pluggable — two registry-driven seams.** A different broker's
> **live read-only MCP server** plugs into the
> [broker-provider registry](#pluggable-brokers) (`brokers.py`), and a different
> broker's **statement export** imports through a format dispatcher — IBKR
> Activity CSV or cross-broker **OFX/QFX** (Vanguard, E\*TRADE, Schwab, most
> banks). Both land in the same store and read back through the same tools.

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
> replaces rather than double-counts. In the TUI, `/import PATH` loads a
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
> - **`sec_filings`** — a company's recent filings (10-K/10-Q/8-K/insider Form 4)
>   with the filing date, description, and a direct link to each document.
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
> the live IBKR account, or write long-term memory. Each run is time-bounded
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
> once on a transient failure.

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
│   ├── tools.py       # local calculators + read-only broker MCP loader/filter
│   ├── brokers.py     # pluggable broker-provider registry (IBKR + any other broker)
│   ├── fundamentals.py# yfinance reference tools (valuation/ratings/earnings/dividends/etf)
│   ├── analytics.py   # tax-loss-harvest + correlation matrix + ETF look-through aggregate
│   ├── factors.py     # Fama-French factor exposure (market/size/value, alpha, R²)
│   ├── edgar.py       # SEC EDGAR filing intelligence (financials/filings/full-text search, keyless)
│   ├── screener.py    # stock screener over a candidate universe (screen_stocks, S&P 500)
│   ├── subagents.py   # parallel/sequence research subagent dispatch (dispatch_subagent[s])
│   ├── research.py    # deep-research report (parallel gather → synthesize) + --research
│   ├── monitor.py     # portfolio monitoring digest (movers/earnings/ex-divs) + --digest
│   ├── flex.py        # IBKR Flex Web Service pull (--flex-sync); XML parser is a seam
│   ├── statements.py  # IBKR CSV + OFX/QFX parsers, format dispatch, SQLite store
│   ├── tui.py         # Textual chat app
│   └── main.py        # CLI entry (argparse): TUI / --prompt / --fake / --session
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

Copy `.env.example` to `.env` and fill in (loaded via python-dotenv):

- **Cloud (OpenAI):** set `OPENAI_API_KEY`; optionally `OPENAI_MODEL`
  (default `gpt-4.1-mini`).
- **Local LLM (llama.cpp / Ollama / LM Studio):** set `OPENAI_API_BASE` to the
  server's OpenAI-compatible endpoint and `OPENAI_MODEL` to the served model id.
- **Other providers (Anthropic / Google / Groq):** set `MODEL_PROVIDER`
  (`anthropic`, `google_genai`, or `groq`) and that provider's key
  (`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`), and install the extra
  (`uv pip install -e '.[anthropic]'`). These route through LangChain's
  `init_chat_model`; each has a sensible default model (e.g. `claude-sonnet-4-5`)
  so `MODEL_PROVIDER` alone is enough. The default/`openai` path is unchanged.
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

> **Status.** Today `--flex-sync` fetches and **saves the statement XML** (under
> `~/.financial-research-assistant/flex`, override `FINANCIAL_RESEARCH_FLEX_DIR`).
> Automatic parsing into the queryable store isn't wired yet: the Flex XML schema
> differs from the CSV Activity Statement, and the field mapping needs validating
> against a real statement before it can be trusted. `parse_flex_xml` in
> [`flex.py`](src/financial_research_assistant/flex.py) is the seam for that step —
> once implemented, the same command imports automatically. Meanwhile, querying
> data still works via the manual CSV path (`import_ibkr_statement`), which is
> unchanged.

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
ex-dividend dates**, all from keyless Yahoo data. Run it from cron/launchd and
pipe it to mail or a push service (e.g. `ntfy`):

```cron
# 8am on weekdays: email the portfolio digest
0 8 * * 1-5  financial-research-assistant --digest | mail -s "Portfolio digest" you@example.com
```

The same report is available in-chat as the `portfolio_digest` tool (which also
takes `include_news=true` to attach a headline per mover), so you can just ask
*"anything I should know about my portfolio this week?"*.

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
MEMORY_BACKEND=local financial-research-assistant --memory   # list long-term memory, exit
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
shows `model · provider · ctx N% of CAP · in/out tok`, plus a spinner + elapsed timer
while a turn runs. Type `/` for the command palette; **Esc** cancels a running
turn; typing while busy **queues** the message; **Ctrl+O**/**Ctrl+T** collapse
all tool / thinking panels. `/compact` summarizes the older turns and rewrites
the thread so the running context (and `ctx %`) shrinks while recent turns stay
verbatim (this also happens automatically for **any** interface when
`AGENT_AUTO_COMPACT` is set — see below). Commands: `/new`, `/compact`, `/clear`, `/sessions`,
`/resume [NAME]`, `/model [NAME]`, `/config`, `/toggle_thinking`, `/copy`,
`/good [note]`, `/bad [note]`, `/memory [forget TEXT]`,
`/export NAME.html|NAME.jsonl`, `/hotkeys`, `/theme`, `/help`, `/quit`. With
long-term memory on, `/good`/`/bad` rate the last answer so the agent learns from
it, and `/memory` reviews or prunes everything it has learned (facts, lessons,
feedback) without leaving the chat. (The
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
called), and `llm_judge`. The judge scores with the **same configured model as the
agent** (via `graph._make_llm`), so it honors `OPENAI_API_BASE` (local servers) and
`MODEL_PROVIDER` (Anthropic/Google/…) — no bare `OPENAI_API_KEY` required. Set
`EVAL_JUDGE_MODEL` to override the judge model (e.g. a stronger cross-family one).
Results append to `eval/results.jsonl`.

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

A `Dockerfile` ships for headless/service use (the TUI is interactive):

```bash
docker build -t financial-research-assistant .
docker run --rm -e OPENAI_API_KEY=sk-... -e IBKR_MCP_URL=... \
  financial-research-assistant --prompt "AAPL quote"
docker run --rm financial-research-assistant --prompt "hi" --fake   # no key needed
```

The image runs `pip install .` (base deps only). Optional extras — `[ofx]` for
OFX/QFX statement import, `[tracing]`, or a model provider like `[anthropic]` —
aren't included; add them to the `Dockerfile`'s install line if you need them.

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
