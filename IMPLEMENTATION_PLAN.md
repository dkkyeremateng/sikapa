# Implementation plan — 2026-08-08 review findings

Source: full-codebase review (five subsystem passes, findings verified against the
code). Baseline at time of review: 895 passed / 2 skipped, clean working tree at
`89ecfeb`.

Work is sequenced into seven phases, each a separate branch merged independently.
Ordering favors user-visible correctness first (silently wrong financial numbers),
then leaks, then reliability, then eval integrity, then polish. Every phase lands
with tests that would have caught its bugs — most of the wrong-number class was
hidden by fakes that ignore the argument under test, so **Phase 0 upgrades the
shared fakes first** and is a prerequisite for Phase 1.

---

## Phase 0 — `test/fakes-model-the-knob` (prerequisite)

Make the shared test fakes honor the arguments under test, so the Phase 1 fixes
can be pinned by tests instead of re-hidden.

| # | File | Change |
|---|------|--------|
| 0.1 | `tests/helpers/fakes.py` — `install_stub_fx` | Generate the rate series from `days`/`as_of` (dates back from a fixed anchor), mirroring what `install_fake_price_fetch` already does. |
| 0.2 | `tests/test_tools.py:682` (portfolio_vs_benchmark fake) | Return dates derived from the requested `days`, not a fixed 2-point series, so a fetch that can't reach the portfolio window is detectable. |
| 0.3 | `tests/test_journal.py:18-21` (price fake) | Model the date argument: return a price *per requested date* so scoring at the wrong date produces a different number. Add a scoring-tick-runs-late case (see 1.3). |
| 0.4 | `tests/test_screener.py:125` (earnings-history fake) | Honor `limit`, and include upcoming (unreported) rows first the way yfinance does. |

**Acceptance:** suite still green; each upgraded fake, when paired with the
*current* buggy code, produces at least one failing test (commit those failing
tests alongside the Phase 1 fixes, not on this branch — this branch only upgrades
fakes used by already-correct paths so it merges green).

---

## Phase 1 — `fix/window-and-math-bugs` (wrong numbers)

The cluster that corrupts analyses and, via journal lessons, the agent's memory.

### 1.1 `days` ignored without `as_of` — `tools.py`, `factors.py`, `screener.py`
- **Bug:** `_fetch_daily` only trims to `days` on the `as_of` path (its docstring
  admits it, `tools.py:181-187`). Callers that consume the series whole compute
  over the entire Yahoo range bucket: `risk_metrics` (`tools.py:1191`),
  `factors._ticker_returns` (`factors.py:96`), `screener._near_high`
  (`screener.py:187-196`, `max(closes)` over the bucket).
- **Fix:** trim to the last `days` sessions on the no-`as_of` path too (make
  `as_of_series(series, None, days)` do the trim, or trim in `_fetch_daily`
  unconditionally). Audit every `_fetch_daily` caller for ones that *rely* on
  receiving the whole bucket (`compare_prices` slices itself; the cache already
  stores the full range, so slicing at return is safe).
- **Tests:** `risk_metrics(days=400)` against a fake with 730 days of data must
  report a ~400-session window (assert on the header dates); same shape for
  `factor_exposure` and `_near_high` (`high_lookback_days=400` must not see a
  700-day-old peak).

### 1.2 `portfolio_vs_benchmark` window anchored to today — `tools.py:1148`
- **Bug:** fetch sized by `span_days` of the statement period, but Yahoo windows
  end today, so a historical portfolio window falls partly/fully outside the
  fetched range → mismatched-comparison verdict with no warning.
- **Fix:** size the fetch from `(date.today() - start).days + margin`. After the
  `start <= d <= end` filter, verify `bench[0]` is within a few sessions of
  `start` (and `bench[-1]` near `end`); otherwise return the "couldn't fetch for
  that span" message instead of a verdict.
- **Tests:** portfolio period ending 400 days ago; fake returns only the last
  365 days → must refuse to compare, not declare out/under-performance.

### 1.3 Thesis journal scores late ticks over the wrong window — `journal.py:278`
- **Bug:** exit price fetched at `today`, not the horizon; a tick that runs late
  scores (and permanently records a lesson on) a longer window than claimed.
- **Fix:** `score_date = min(today, due_date)`; use it for both the exit price
  and the benchmark end (`journal.py:292`). `describe_outcome` then matches the
  recorded `horizon_days`.
- **Tests:** entry due 2026-05-01, scored 2026-08-08, fake prices diverge after
  the due date → score must reflect the due-date price. (Needs fake 0.3.)

### 1.4 FIFO double-counts overlapping imports; pools across accounts — `statements.py`
- **Bug A:** `_trade_split_events` (`statements.py:1129-1133`) reads trades from
  *all* imports; an annual + a monthly statement inside it stores each March
  trade twice → doubled `open_lots`, `realized_gains`, `tax_loss_harvest`.
- **Fix A:** reconcile overlapping imports to a non-overlapping set before the
  trade walk — reuse the finer-period-wins logic `query_performance_history`
  already applies to NAV (`statements.py:1035`), or dedupe trades on a natural
  key (account, symbol, datetime, qty, price, side).
- **Bug B:** `open_lots`/`realized_gains` with `account=None` pool lots across
  accounts by symbol (`statements.py:1159-1263`), unlike `query_positions`
  which defaults to the newest import's account.
- **Fix B:** default to the newest import's account, matching
  `query_positions`; keep an explicit `account="all"` escape hatch only if the
  output labels it clearly.
- **Tests:** overlapping annual+monthly fixture → each trade counted once;
  two-account fixture → a sell in U2 never consumes U1's lot.

### 1.5 DCF growth math — `valuation.py`
- **Bug A:** `_resolve_growth` uses `_cagr(oldest, base, len(history)-1)`
  (`valuation.py:209`) while `_fcf_history` skips years with missing OCF →
  undercounted periods, overstated growth.
- **Fix A:** `periods = history[0]["fy"] - history[-1]["fy"]`.
- **Bug B:** `if g:` treats `growth_rate=0` as unset (`valuation.py:204-206`);
  `_as_rate`'s strict `> 1.0` reads `growth_rate=1` as 100%.
- **Fix B:** distinguish None from 0 (`g is not None`); treat `>= 1.0` as
  percent (document that 100%+ growth must be passed as `1.0` fraction… simpler:
  values `>= 1` are percent, and reject absurd inputs with a clear message).
- **Tests:** gap-year FCF history (FY2025/24/22) → CAGR over 3 years not 2;
  `growth_rate=0` → zero growth honored; `growth_rate=1` → 1%.

### 1.6 Smaller math/window items (same branch, cheap)
- `tools.py:955` — `_fx_lookup` returns a rate *after* the requested date while
  the message claims "on/before": either say what it did or refuse.
- `tools.py:859` — `income_summary` skips the base-currency total for a
  single-non-USD-currency account: condition should be "any currency ≠ base".
- `tools.py:1221-1226` and `analytics.py:126-130` — zero-close guard drops rows
  independently per series, desynchronizing positional pairing (beta,
  correlation, portfolio_risk): drop the *date* from both series instead.
- `fundamentals.py:71` + `screener.py:219` — earnings-beat streak: fetch enough
  rows that upcoming (unreported) quarters don't consume the limit (filter to
  reported rows *before* applying `limit`).
- `options.py:172-180` — `_nearest_expiry`: pick the truly nearest expiry
  (before or after), not "first later, else earliest".

**Acceptance:** all new tests green; run the full suite; `--fake` smoke intact.

---

## Phase 2 — `fix/redaction-and-perms` (leaks; small, ship fast)

| # | File | Fix |
|---|------|-----|
| 2.1 | `tracing.py:100,143` | Wrap `gen_ai.input.messages` / `gen_ai.output.messages` in `auth.redact()` like every other attribute. Test: a prompt containing a `sk-…` key never reaches the span attributes unmasked. |
| 2.2 | `flex.py:151-159` | Write statement XML via `storage.write_private` (0600). Test: mode check on the written file. |
| 2.3 | `alerts.py:262-265, 329-332` | `save_alerts` → `write_private`; wrap load→mutate→save in the `_locked` pattern from `tasks.py`. Test: torn/concurrent write doesn't lose rules. |
| 2.4 | `storage.py:38-42` | `flush()` + `os.fsync()` before `os.replace` in `write_private`. |

---

## Phase 3 — `fix/oauth-and-proxy-hardening`

### 3.1 PKCE verifier sent as `state` — `oauth.py:369, 436, 444`
- Send `secrets.token_urlsafe(32)` as `state` for providers where
  `SEND_STATE=False`; keep verifier-as-state only for Anthropic (its protocol
  requires it). Thread the expected state into the loopback handler.

### 3.2 Loopback callback validates nothing — `oauth.py:104-125, 141-144`
- Compare returned `state` to the sent value; reject mismatches. Surface
  `error=access_denied` as a readable denial instead of the paste-the-code
  prompt.

### 3.3 Codex proxy is an open local endpoint — `codex_proxy.py:298-343, 394-406`
- Generate a per-process bearer token in `ensure_running`, inject it into the
  client config, require it on every request; additionally reject unexpected
  `Host` headers. Test: request without the token → 401.

### 3.4 Mid-stream upstream error corrupts the SSE stream — `codex_proxy.py:334-357`
- Once headers are sent, never call `_json_error`: emit the failure as a final
  SSE `data:` error chunk + `data: [DONE]`, then close. Handle
  `BrokenPipeError` by closing silently.

### 3.5 Refresh race — `oauth.py:968-994`
- Move the credential re-read *and* the refresh POST inside `_locked()`;
  re-check `needs_refresh` after acquiring the lock (double-checked locking) so
  the second waiter reuses the first's fresh token instead of re-refreshing.

---

## Phase 4 — `fix/scheduler-delivery-reliability`

### 4.1 Stranded "running" tasks — `tasks.py:522-548`
- In `claim_due`, treat `status=="running"` with `claimed_at` older than a
  timeout (default ~30 min, env-tunable) as claimable again; count it as an
  attempt so the three-strike parking still applies.
- Add per-task try/except in the scheduler batch loop (`scheduler.py:297-298`)
  so one task's `record_result`/delivery failure can't strand the rest of the
  claimed batch.

### 4.2 Desktop channel false "delivered" — `channels.py:187-216`
- A truncated banner is a *courtesy*, not delivery of the answer. Either make
  `notify_desktop` report real outcome where knowable, or — simpler and more
  honest — mark the desktop channel as not sufficient on its own: if the only
  successful channel(s) are banner-style truncating channels, still call
  `queue_delivery` so the full answer is re-sent when a full-content channel
  recovers. Test: telegram fails + desktop "succeeds" → answer parked.

### 4.3 Telegram poll offset race — `telegram.py:274-292`
- Guard `getUpdates`/offset read-write with an fcntl lock (same pattern as the
  task store) so cron + `--watch` can't both consume the same update.

### 4.4 Recurring tasks drift across DST — `tasks.py:378-401`
- Compute the next occurrence in local wall-clock time and convert back to UTC
  for storage, honoring the module's documented promise.

### 4.5 Digest duplicates — `monitor.py:211-213`
- De-duplicate the symbol list before scanning (preserve first-seen order).

---

## Phase 5 — `fix/adapter-and-subagents`

### 5.1 Tool-call stream ownership — `adapter.py:698-742`
- Gate the `tool_call_chunks` (AIMessageChunk) and whole-`tool_calls`
  (AIMessage) branches on the same `_from_own_node` test that already gates
  reasoning/tokens (usage stays unfiltered — subagent tokens are billed).
- Key `by_index` by `(checkpoint_ns, index)` so two concurrently streaming
  models can't cross-contaminate arg fragments.
- Test (fills the known gap): a scripted graph streaming two interleaved
  id-less tool-call chunk sequences from different namespaces → correctly
  paired panels; a subagent-namespace tool call → no event.

### 5.2 Memory failures must not kill turns — `adapter.py:903-1018`
- Wrap the pre-`try` recall (`:903-905`) so a backend failure yields a status
  event and proceeds without memories.
- Wrap auto-compaction (`:913-930`): on failure, emit a status event, proceed
  uncompacted, and *don't* leave `_last_input` re-triggering every turn.
- Wrap `mem.remember` (`:992-993`): a write failure after the answer is
  computed logs/emits status, never converts the turn into an error.
- Also add try/except around `_Mem0Adapter.search/remember`
  (`memory.py:419-460`) returning empty/no-op on backend errors.

### 5.3 Auto-compaction trigger uses the wrong number — `adapter.py:989-1012`
- Track the *last* model call's `input_tokens` (per-call deltas already exist
  at `:673-678`) and use it for `context_tokens` and `_last_input`, instead of
  the turn's summed input.

### 5.4 Subagent fallback pool includes mutators — `subagents.py:290-293`
- Build an explicit deny-set for subagents (`schedule_task`, `record_thesis`,
  `add_alert`/`remove_alert`, `render_report`, `import_ibkr_statement`,
  `ingest_document`?, memory tools) applied to the pool *before* keyword
  selection, so the no-match fallback is research-only. Test: a no-keyword task
  → pool contains no side-effecting tool.

### 5.5 Block-content flattening — `subagents.py:327-330`, `graph.py:682`
- Replace `str(content)` fallbacks with the `_chunk_text`-style text-block
  flattening (extract a shared helper) so Anthropic-style content lists don't
  leak reprs (including thinking blocks) into findings and compaction seeds.

### 5.6 Small adapter items
- `main.py:592-600` — write the `--trace` file (with an error marker) before
  returning on an error event.
- Session-switch memory wipe (`adapter.py:97-103`): at minimum emit a status
  event ("model changed — conversation memory reset") when a new checkpointer
  is created for an existing session; consider keying by session only.

---

## Phase 6 — `fix/eval-integrity`

### 6.1 Memory isolation — `eval/evaluate.py:280-284, 380-382`
- In `run_item`, unless the caller overrides, set `MEMORY_BACKEND=""` (or a
  per-run temp `MEMORY_DIR`) in the child env so eval runs neither read nor
  pollute the operator's real store. In `_run_ab`, give each arm its own temp
  `MEMORY_DIR` exactly as `ab_compare._arm_env` (`ab_compare.py:128-141`)
  already does. In `ab_compare`, give each *repeat* its own dir too
  (`:148-162`).

### 6.2 Judge parsing — `eval/evaluate.py:112-116, 188-223`
- Score parse: require a lone float (or take the last number); add tests for
  "scale of 0.0 to 1.0 … 0.9" and "8/10".
- Rubric: reject/uniquify duplicate dimension names so the weighted score can't
  exceed 1.0; warn loudly when `EVAL_JUDGE_MODEL` is unset (self-grading).
- Fake mode: have `ci_gate --fake` report how much of the score came from
  auto-passed judge/rubric items, so a judged-heavy dataset can't green a
  garbage run silently (`evaluate.py:89-90, 192-194`).

### 6.3 Robustness
- `score` (`evaluate.py:259-267`): validate criteria shape up front (before any
  subprocess runs) and score a malformed item 0 with a note instead of raising
  `KeyError` mid-run.
- `improve.diagnose` (`improve.py:82-86`): under-floor trajectory items with no
  missing tools fall through to the content-miss shape instead of vanishing.

---

## Phase 7 — `fix/tui-memory-polish` (lower urgency, batched)

- **TUI Esc**: `action_cancel_turn` closes the command palette first if open
  (`tui.py:698, 1664-1678`).
- **TUI worker crash**: add `except Exception` in `stream_response`
  (`tui.py:1788-1922`) rendering an error line instead of exiting the app.
- **`/new` stale rating targets**: reset `_last_answer`/`_last_user`
  (`tui.py:1155-1167`).
- **Reflection false gaps**: drop `"n/a"` from `_GAP_MARKERS`
  (`reflection.py:42`) or match it only as a whole section, not a field value.
- **Memory dedup**: scope `_is_dup` by kind and raise the threshold for
  templated lesson text (`memory.py:257-267`); an `avoid` must never be dropped
  as a dup of an `exemplar`.
- **OFX routing**: `parse_ofx` treats any existing path as a file
  (`statements.py:547-550`), reusing `_read_rows`'s rule.
- **"Total in USD" rows**: per-section total guard matches
  `startswith("Total")` (`statements.py:133`), with a multi-currency fixture.
- **`_INFO_CACHE` TTL**: give `.info` the same 15-min TTL as prices
  (`fundamentals.py:25-46`).
- **`edgar._fetch_json`**: only cache dicts; guard the cache-hit path
  (`edgar.py:69-80`).
- **Documents**: `_locked` around index read-modify-write
  (`documents.py:171-214`); merge keyword results for non-embedded records in
  `_rank` (`:228-240`, mirror `SemanticMemory.rank`'s top-up).
- **Reports**: pass theme/output as arguments instead of env mutation
  (`reports.py:957-973`); don't report "0 page(s)" without pypdfium2
  (`:752-758`); add a collision-proof suffix to report stems
  (`research.py:194-202`).
- **Misc**: `memory._load` reads with `encoding="utf-8"` (`memory.py:226`);
  amendment-vs-original filing pick (`edgar.py:834,888` — prefer the base form
  when the amendment is cover-page-only); auth v1→v2 migration tier collision
  (`auth.py:143-151`); `load_tasks` keeps (inert) malformed entries instead of
  deleting them on the next save (`tasks.py:116-129`); wrap blocking HTTP in
  `asyncio.to_thread` in the scheduler loop (`scheduler.py:146, 205, 317, 333`).

---

## Process

- One branch per phase, merged to `main` independently; full suite + `--fake`
  smoke before each merge.
- Every bug fix lands with the test that would have caught it (Phase 0 makes
  that possible for the window bugs).
- Phases 1-2 first (wrong numbers + leaks), then 3-4-5 in any order, then 6
  before the next `--ab --apply` run, then 7 opportunistically.
- Update `TODO.md` / the improvement-roadmap memory when phases land.
