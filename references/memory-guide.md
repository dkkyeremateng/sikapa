# Long-term memory — guide

This project ships an **optional** long-term (cross-session) memory layer in
[`memory.py`](../src/financial_research_assistant/memory.py). It is a no-op unless
`MEMORY_BACKEND` is set, so the default install, `--fake`, and the test suite all
stay hermetic.

Long-term memory is distinct from the two things that already existed:

| layer | scope | what it holds |
| --- | --- | --- |
| session checkpointer (`MemorySaver`) | one conversation, one process | the running message history |
| durable checkpointer (`AsyncSqliteSaver`, `FINANCIAL_RESEARCH_CHECKPOINT_DB`) | one conversation, across restarts | the running message history, on disk |
| **long-term memory (`MEMORY_BACKEND`)** | **the user, across every conversation** | **curated durable facts** |

## What gets remembered

Long-term memory is for **durable facts about the user or how they want to work**:
risk tolerance, watchlist tickers, tax situation, base currency, investing goals,
answer-formatting preferences. It is **not** for transient data — quotes, prices,
one-off calculations — which go stale and would poison recall.

Two write paths feed one store, both deduped on write:

1. **Model-curated (primary).** When memory is on, the agent is given the
   `remember`, `recall`, `forget`, and `list_memories` tools and is instructed to
   store lasting facts deliberately. This is the high-signal path.
2. **Auto-capture (fallback).** After each turn the adapter calls
   `remember(user_msg)`, which stores the message *only* if `_memorable()` judges
   it a durable statement (an explicit "remember …", a preference, "my goal is
   …") and never a passing question or a lone figure. This keeps the store clean
   rather than accumulating every raw exchange (the failure mode of the original
   naive version, which appended every Q&A).

At the start of a turn the adapter calls `recall(user_msg)` and injects the top
hits into the prompt (`inject()`), so the model answers with them in mind.

## Backends (`MEMORY_BACKEND`)

- **unset / `""`** — disabled (default). `get_memory()` returns `None`; the
  adapter and graph skip all memory work.
- **`local`** — `LocalMemory`, a zero-dependency per-user JSONL store under
  `MEMORY_DIR` (default `~/.financial-research-assistant/memory`), one file per `MEMORY_USER`.
  Retrieval is deterministic keyword overlap; writes dedup on near-identical text
  (Jaccard ≥ 0.85). Fully inspectable and offline.
- **`semantic`** — `SemanticMemory`, the **same JSONL store** as `local` (so
  `--memory` and everything else works unchanged), but each entry also carries an
  embedding and recall ranks by **cosine similarity** — it surfaces *related*
  memories, not just term-overlapping ones. Embeddings come from an
  OpenAI-compatible endpoint via `langchain-openai` (a core dep, so no extra
  install): honors `OPENAI_API_BASE`/`OPENAI_API_KEY`, model from
  `MEMORY_EMBED_MODEL` (default `text-embedding-3-small`), threshold from
  `MEMORY_SIM_THRESHOLD` (default 0.25). When embeddings are unavailable it saves
  without a vector and ranks by keyword, so it never hard-fails — a strict superset
  of `local`. Ranking is centralized in `Memory.rank`, so **all** recall paths
  (facts, reflection lessons, feedback exemplars) get semantic retrieval at once.
- **`mem0`** — the [mem0](https://github.com/mem0ai/mem0) universal memory layer
  (optional dependency). It does semantic recall, LLM-based salient-fact
  extraction, and contradiction handling that the local keyword store does not.
  `_Mem0Adapter` maps it onto the same contract; the curation methods
  (`forget`/`all`) are best-effort against mem0's API.

## Scoping and multi-user

Memory is keyed by `MEMORY_USER` (default `"default"`), **not** the session id —
that's what makes a fact learned in one chat available in the next. If you run the
assistant for more than one person from the same machine, set a distinct
`MEMORY_USER` per person so their facts never mix. The `local` backend writes one
file per sanitized user key.

## Cautions

- **PII / sensitive data.** Stored facts are written verbatim to disk (or to
  mem0). Don't enable long-term memory for accounts where persisting personal or
  financial details to `MEMORY_DIR` is unacceptable, and treat that directory as
  sensitive. The agent is told not to store transient market data, but it may
  store personal preferences the user states.
- **Untrusted content never writes memory.** Only the user's own messages and the
  model's deliberate `remember` calls write facts. Web-search results and other
  tool output are untrusted data (see the system prompt's SECURITY section) and
  must never be persisted as memory on their own.
- **Staleness / supersede-on-contradiction.** When a durable fact updates a
  *single-valued* attribute the user already stated — risk tolerance, base
  currency, name, tax bracket, and a few others in `_SINGLE_VALUED` — the stale
  value is **archived** (marked `superseded`, kept in the store) and the new one
  becomes active, so recall never surfaces the old value but the history is
  auditable. This is deliberately conservative: it fires only for
  `subject_key()`-recognized single-valued attributes and only when the value
  actually changed (a case/punctuation restatement is a no-op), so multi-value
  facts (holdings, watchlist, goals) always co-exist, and `lesson`/`exemplar`/
  `avoid` rows are never touched. Re-stating a previously-archived value
  reactivates it. The `remember` tool reports what it replaced.
  - **Archive tier.** `all()` returns only active entries by default (so recall,
    `list_memories`, and the model never see stale values); `all(include_superseded=True)`
    is the full audit view. `--memory` and the TUI `/memory` show archived values
    tagged `·superseded`; prune them with `forget` like anything else.
  - Turn the whole mechanism off with `MEMORY_SUPERSEDE=0` to keep every stated
    value. mem0 handles contradictions with its own LLM reconciliation instead.
- **Inspect and prune.** `financial-research-assistant --memory` lists everything
  stored; `--memory forget:TEXT` removes matching facts. No model required.

## Reflection — lessons from its own work (`reflection.py`)

Long-term *fact* memory learns about the user. **Reflection** is a second
self-learning layer that learns about the *work*: after the full research
pipeline (`research_ticker` / `research_portfolio`, i.e. `--research`) produces a
report, `reflect()` distills **lessons** and stores them in the same store as
`kind="lesson"` memories. On the next research run for the same subject,
`recall_lessons()` pulls them back and injects them into the synthesis prompt (and
into the `research_report` chat tool's findings), so the report addresses known
gaps instead of repeating them.

Two lesson sources:

- **Gap lessons** — deterministic and free: sections whose data came back
  unavailable or too thin become a lesson each. Always runs when memory is on, so
  it's fully testable offline.
- **Critique lessons** — one LLM call asking for a few terse *process* lessons
  about the report just written. Skipped in `--fake`, on error, and when
  `RESEARCH_REFLECT=0`.

Lessons are stamped with their subject (e.g. "When researching AAPL, …") so
keyword recall reliably finds them, and they show up as `[lesson]` in `--memory`.
Like everything else here, it's a no-op when `MEMORY_BACKEND` is unset.

## Feedback — learning from your verdicts (`feedback.py`)

The third self-learning layer. Rate an answer in the TUI with **`/good [note]`** or
**`/bad [note]`** and `record()` stores that exchange as an `exemplar` (good) or
`avoid` (bad) memory. On a later, similar question the adapter recalls the
relevant ones and injects them as **few-shot guidance** ahead of the prompt —
"answers the user rated good, emulate their depth/structure/format" and "answers
the user rated poor, avoid these" — so replies drift toward what you like. No
fine-tuning, just retrieval.

Notes:

- The few-shot block instructs the model to copy *approach*, not figures —
  numbers are always recomputed from tools, so a stale exemplar can't inject a
  stale price.
- Exemplar answers are capped (~700 chars) to keep the injected block affordable.
- Feedback rows show as `[exemplar]` / `[avoid]` in `--memory` and are prunable
  with `--memory forget:TEXT`.
- `recall_facts` (the plain fact injection) deliberately **excludes** the
  `lesson`/`exemplar`/`avoid` kinds, so each specially-framed kind is injected
  exactly once, by its own layer.

## Extending

To swap in a different store, implement the `Memory` protocol
(`save` / `search` / `forget` / `all` / `remember` / `recall`) and wire it into
`get_memory()`. Everything else — auto-inject, auto-capture, the four model tools,
and the `--memory` CLI — goes through that contract unchanged.
