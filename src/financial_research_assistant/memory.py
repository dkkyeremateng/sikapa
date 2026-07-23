"""Optional long-term (cross-session) memory. No-op unless MEMORY_BACKEND is set.

Session memory (the running conversation) already ships in the adapter. This
adds a curated store of durable facts that persists *across* sessions, scoped by
a stable user key (``MEMORY_USER``, default "default") rather than the
per-conversation session id — the piece that makes the agent feel like it
learns.

Two write paths feed one store:
  * **Model-curated** — the agent calls the ``remember`` tool when the user
    states a lasting preference, goal, constraint, or holding of interest. This
    is the primary, high-signal path.
  * **Auto-capture** — after each turn the adapter calls ``remember(user_msg)``,
    which stores the message ONLY when it reads as a durable statement (an
    explicit "remember …", a preference, "my goal is …"), never a passing
    question or a one-off quote. So the store stays clean instead of
    accumulating every raw exchange.

Both dedup on write, so the same fact isn't stored twice. The model can also
``recall`` on demand, ``list_memories`` to review, and ``forget`` when something
is no longer true — all deterministic and inspectable (see also ``--memory`` on
the CLI).

Backends (``MEMORY_BACKEND``):
  unset / ""  -> None (session-only; nothing persists across sessions). Default,
                 so ``--fake`` and the smoke tests stay hermetic.
  "local"     -> a zero-dependency JSONL store under ``MEMORY_DIR``
                 (default ~/.agent-builder/memory), one file per user.
  "mem0"      -> the mem0 universal memory layer (optional dep; see
                 references/memory-guide.md).

Backend contract (see the ``Memory`` protocol):
  save(text, kind)      -> store a durable fact (deduped); True if newly stored.
  search(query, k)      -> up to k stored facts most relevant to the query.
  forget(query)         -> remove matching facts; returns how many.
  all()                 -> every stored fact (for review).
  remember(user_msg)    -> auto-capture: save iff the message is durable.
  recall(query, k)      -> alias of search (used by the adapter's auto-inject).

Cautions (write policy, PII, user scoping): see references/memory-guide.md.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

_DEFAULT_DIR = Path.home() / ".agent-builder" / "memory"


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def _cmp(text: str) -> str:
    """Normalize for equality: lowercased, whitespace-collapsed, trailing
    punctuation stripped — so 'moderate' and 'Moderate.' compare equal."""
    return " ".join((text or "").split()).lower().rstrip(" .;,!?")


# --- Auto-capture salience --------------------------------------------------
# Only a user message that reads as a *durable statement* is auto-remembered, so
# the store doesn't fill with passing questions and one-off figures.

# An explicit request to store something — always memorable, even if phrased as a
# question ("could you remember that …?").
_STORE_VERB = re.compile(
    r"\b(remember|note that|make a note|keep in mind|from now on|"
    r"don'?t forget|for future reference)\b",
    re.I,
)
# A lasting fact about the user or how they want to work.
_DURABLE = re.compile(
    r"\b(i (prefer|like|want|own|hold|need|avoid)|i'?m interested|"
    r"my (name|goal|goals|risk|risk tolerance|favou?rite|watchlist|portfolio|"
    r"strategy|target|budget|horizon|tax|base currency)|call me|"
    r"always|never|watchlist)\b",
    re.I,
)
# "my <thing> is/are <value>" — a stated attribute.
_MY_IS = re.compile(r"\bmy\b[^?]*\b(is|are)\b", re.I)
# Interrogative opener: a question is asking, not telling.
_QUESTION_START = re.compile(
    r"^\s*(what|how|why|when|where|which|who|whose|is|are|do|does|did|can|"
    r"could|should|would|will|has|have)\b",
    re.I,
)
# Polite lead-in stripped from an auto-captured fact so it reads as a statement.
_PREFIX = re.compile(
    r"^\s*(please\s+)?(remember|note that|make a note that|keep in mind that|"
    r"for future reference,?|don'?t forget that)\b[:,]?\s*(that\b\s*)?",
    re.I,
)


def _memorable(msg: str) -> bool:
    """True if ``msg`` is a durable statement worth auto-remembering."""
    msg = (msg or "").strip()
    if not msg:
        return False
    if _STORE_VERB.search(msg):
        return True  # explicit intent wins, even if phrased as a question
    if not (_DURABLE.search(msg) or _MY_IS.search(msg)):
        return False
    # Durable-looking but interrogative → the user is asking, not telling.
    return not (msg.endswith("?") or bool(_QUESTION_START.match(msg)))


def _clean(msg: str) -> str:
    """Drop a leading "remember that …" lead-in so the stored fact is a clean
    statement; fall back to the original if stripping leaves nothing."""
    cleaned = _PREFIX.sub("", (msg or "").strip()).strip()
    return cleaned or (msg or "").strip()


# --- Memory hygiene: supersede-on-contradiction -----------------------------
# When a durable fact updates a SINGLE-VALUED attribute the user already stated
# (their risk tolerance, base currency, …), the new value replaces the old so
# recall never surfaces the stale one. Restricted to attributes that logically
# hold ONE value, so multi-value facts (holdings, watchlist, goals) are never
# clobbered — those legitimately co-exist.
_SINGLE_VALUED = {
    "risk tolerance", "risk appetite", "base currency", "reporting currency",
    "investment horizon", "time horizon", "investing style", "investment style",
    "name", "tax bracket", "tax rate", "retirement age", "target allocation",
    "budget", "monthly budget", "annual budget",
}
# "my <1-4 word attribute> is/are …" — bounded word groups with literal single
# spaces (input is whitespace-normalized first), so there's no char-class/\s
# overlap and matching stays linear.
_ATTR_RE = re.compile(r"\bmy ([a-z]+(?: [a-z]+){0,3}?) (?:is|are)\b", re.I)


def subject_key(text: str) -> str | None:
    """The single-valued attribute a statement sets, or None. E.g. "My risk
    tolerance is now high" → "risk tolerance"; "I own AAPL" or "My goal is to
    retire early" → None (multi-value / not a curated attribute), so they never
    supersede anything."""
    t = " ".join((text or "").split())
    if re.search(r"\bcall me\b", t, re.I):
        return "name"
    m = _ATTR_RE.search(t)
    if not m:
        return None
    attr = m.group(1).lower()
    return attr if attr in _SINGLE_VALUED else None


def _supersede_enabled() -> bool:
    """Whether supersede-on-contradiction runs (``MEMORY_SUPERSEDE``, default on).
    Set to 0/false/off to keep every stated value instead of replacing."""
    return (os.environ.get("MEMORY_SUPERSEDE") or "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


@runtime_checkable
class Memory(Protocol):
    def save(self, text: str, kind: str = "note") -> bool: ...
    def search(self, query: str, k: int = 3) -> list[str]: ...
    def forget(self, query: str) -> int: ...
    def all(self, include_superseded: bool = False) -> list[dict]: ...
    def remember(self, user_msg: str, answer: str = "") -> None: ...
    def recall(self, query: str, k: int = 3) -> list[str]: ...
    # Shared ranking primitive: rank a caller-supplied list of entries (already
    # filtered by kind) against a query, returning the top-k entries. Keyword for
    # LocalMemory, cosine for SemanticMemory — so every kind-scoped recall
    # (facts, lessons, feedback) gets the backend's retrieval, not just search().
    def rank(self, query: str, entries: list[dict], k: int) -> list[dict]: ...


def _keyword_rank(query: str, entries: list[dict], k: int) -> list[dict]:
    """Rank entries by query-term overlap (recency breaks ties), keeping only
    those with at least one shared term. The deterministic default retrieval."""
    q = _terms(query)
    scored = [(len(q & _terms(e["text"])), i, e) for i, e in enumerate(entries)]
    hits = [s for s in scored if s[0] > 0]
    hits.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [e for _, _, e in hits[:k]]


class LocalMemory:
    """Zero-dependency per-user JSONL store of durable facts.

    Each line is ``{"text", "kind", "ts"}``. Retrieval is deterministic keyword
    overlap and writes dedup on near-identical text — good enough to be useful
    and fully inspectable offline. For semantic recall, salient-fact extraction,
    and contradiction handling, swap in the mem0 backend (memory-guide.md)."""

    def __init__(self, root: Path, user: str) -> None:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", user)
        self._file = Path(root) / f"{safe}.jsonl"

    def _load(self) -> list[dict]:
        if not self._file.exists():
            return []
        out: list[dict] = []
        for line in self._file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue  # tolerate a partially-written/corrupt line
            if isinstance(d, dict) and d.get("text"):
                entry = {"text": d["text"], "kind": d.get("kind", "note"), "ts": d.get("ts", "")}
                if isinstance(d.get("vec"), list):  # semantic backend embedding
                    entry["vec"] = d["vec"]
                if d.get("superseded"):  # archived stale value (date it was replaced)
                    entry["superseded"] = d["superseded"]
                out.append(entry)
        return out

    def _write(self, entries: list[dict]) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text("".join(json.dumps(e) + "\n" for e in entries))

    def _is_dup(self, text: str, entries: list[dict]) -> bool:
        """True if ``text`` is an exact or near-duplicate (term Jaccard ≥ 0.85) of
        an existing entry."""
        nt, ntext = _terms(text), text.lower()
        for e in entries:
            if e["text"].lower() == ntext:
                return True
            et = _terms(e["text"])
            if nt and et and len(nt & et) / len(nt | et) >= 0.85:
                return True
        return False

    def _make_entry(self, text: str, kind: str) -> dict:
        """Build a stored entry. Subclasses override to attach extra fields (e.g.
        SemanticMemory adds an embedding)."""
        return {"text": text, "kind": kind or "note", "ts": date.today().isoformat()}

    def save(self, text: str, kind: str = "note") -> bool:
        """Store a durable fact. Returns False (without storing) when it's empty
        or a near-duplicate of something already known."""
        text = " ".join((text or "").split())
        if not text:
            return False
        entries = self._load()
        active = [e for e in entries if not e.get("superseded")]
        # A restatement of an active memory (ignoring case and trailing
        # punctuation) is always a no-op — so "moderate" and "moderate." don't
        # create a spurious update, while a genuinely different value still does.
        norm = _cmp(text)
        if any(_cmp(e["text"]) == norm for e in active):
            return False
        key = subject_key(text) if _supersede_enabled() and kind not in _SPECIAL_KINDS else None
        if key:
            # A CHANGED value for a single-valued attribute the user already
            # stated: ARCHIVE the stale entry (mark superseded, keep for audit) so
            # recall never surfaces the old value but the history is retained. Skip
            # near-duplicate dedup here — for a single-valued attribute any
            # different value (even 24% → 32%) is a legitimate update.
            stamp = date.today().isoformat()
            for e in entries:
                if (not e.get("superseded") and e.get("kind") not in _SPECIAL_KINDS
                        and subject_key(e["text"]) == key):
                    e["superseded"] = stamp
        elif self._is_dup(text, active):
            # Not a single-valued update: drop near-duplicates (holdings, notes, …).
            return False
        entries.append(self._make_entry(text, kind))
        self._write(entries)
        return True

    def rank(self, query: str, entries: list[dict], k: int) -> list[dict]:
        """Keyword-overlap ranking (the deterministic default)."""
        return _keyword_rank(query, entries, k)

    def search(self, query: str, k: int = 3) -> list[str]:
        # Rank over ACTIVE entries only — superseded (archived) values never recall.
        return [e["text"] for e in self.rank(query, self.all(), k)]

    def forget(self, query: str) -> int:
        """Remove facts matching ``query`` (substring, or all query terms
        present). Returns the number removed."""
        entries = self._load()
        q, ql = _terms(query), (query or "").strip().lower()
        if not ql:
            return 0
        keep, removed = [], 0
        for e in entries:
            et = _terms(e["text"])
            if (ql and ql in e["text"].lower()) or (q and q <= et):
                removed += 1
            else:
                keep.append(e)
        if removed:
            self._write(keep)
        return removed

    def all(self, include_superseded: bool = False) -> list[dict]:
        """Stored entries. By default returns only ACTIVE ones — so recall,
        `list_memories`, and the model never see archived stale values. Pass
        ``include_superseded=True`` for the full audit view (CLI/TUI inspection)."""
        entries = self._load()
        if include_superseded:
            return entries
        return [e for e in entries if not e.get("superseded")]

    # -- compatibility / auto paths used by the adapter --------------------
    def recall(self, query: str, k: int = 3) -> list[str]:
        return self.search(query, k)

    def remember(self, user_msg: str, answer: str = "") -> None:
        """Auto-capture: store ``user_msg`` only if it's a durable statement
        (``answer`` is ignored — answers go stale and are the noise the naive
        store accumulated)."""
        if _memorable(user_msg):
            self.save(_clean(user_msg), kind="note")


def _sim_threshold() -> float:
    """Minimum cosine similarity for a semantic hit (``MEMORY_SIM_THRESHOLD``,
    default 0.25) — keeps auto-injection from surfacing unrelated memories."""
    try:
        return float(os.environ.get("MEMORY_SIM_THRESHOLD") or 0.25)
    except ValueError:
        return 0.25


class SemanticMemory(LocalMemory):
    """LocalMemory with embedding-based recall (``MEMORY_BACKEND=semantic``).

    Same JSONL store and curation as LocalMemory, but each entry also carries a
    ``vec`` embedding, and ``rank`` scores by cosine similarity — so recall finds
    *related* memories, not just term-overlapping ones ('how aggressive should I
    be' recalls 'my risk tolerance is high'). Embeddings come from an
    OpenAI-compatible endpoint (see embeddings.py); when they're unavailable, both
    save and rank degrade gracefully to the keyword behavior, so nothing hard-fails
    and the store stays a superset of the local one."""

    def _make_entry(self, text: str, kind: str) -> dict:
        from . import embeddings

        entry = super()._make_entry(text, kind)
        vec = embeddings.embed_query(text)
        if vec:
            entry["vec"] = vec
        return entry

    def rank(self, query: str, entries: list[dict], k: int) -> list[dict]:
        from . import embeddings

        qvec = embeddings.embed_query(query)
        if qvec is None:  # embeddings unavailable → keyword ranking
            return _keyword_rank(query, entries, k)
        vecd, plain = [], []
        for e in entries:
            (vecd if isinstance(e.get("vec"), list) and len(e["vec"]) == len(qvec)
             else plain).append(e)
        threshold = _sim_threshold()
        scored = [(embeddings.cosine(qvec, e["vec"]), e) for e in vecd]
        sem = [e for s, e in sorted(scored, key=lambda t: t[0], reverse=True) if s >= threshold]
        if len(sem) >= k or not plain:
            return sem[:k]
        # Top up with keyword hits from any entries that lack a usable embedding.
        return (sem + _keyword_rank(query, plain, k - len(sem)))[:k]


class _Mem0Adapter:
    """Thin adapter so mem0 satisfies the same contract. Best-effort for the
    curation methods, since mem0 manages extraction/dedup internally."""

    def __init__(self, mem, user: str) -> None:
        self._mem = mem
        self._user = user

    def save(self, text: str, kind: str = "note") -> bool:
        text = " ".join((text or "").split())
        if not text:
            return False
        self._mem.add(text, user_id=self._user, metadata={"kind": kind})
        return True

    def search(self, query: str, k: int = 3) -> list[str]:
        hits = self._mem.search(query=query, user_id=self._user, limit=k)
        results = hits.get("results", hits) if isinstance(hits, dict) else hits
        return [h.get("memory", "") for h in results][:k]

    def forget(self, query: str) -> int:
        removed = 0
        try:
            hits = self._mem.search(query=query, user_id=self._user, limit=50)
            results = hits.get("results", hits) if isinstance(hits, dict) else hits
            for h in results:
                mid = h.get("id")
                if mid is not None:
                    self._mem.delete(memory_id=mid)
                    removed += 1
        except Exception:
            pass  # mem0 API drift must not break the forget tool
        return removed

    def all(self, include_superseded: bool = False) -> list[dict]:
        # mem0 manages its own reconciliation, so it has no superseded tier — the
        # flag is accepted for contract parity and ignored.
        try:
            hits = self._mem.get_all(user_id=self._user)
            results = hits.get("results", hits) if isinstance(hits, dict) else hits
            return [{"text": h.get("memory", ""), "kind": "note", "ts": ""} for h in results]
        except Exception:
            return []

    def recall(self, query: str, k: int = 3) -> list[str]:
        return self.search(query, k)

    def rank(self, query: str, entries: list[dict], k: int) -> list[dict]:
        # mem0 owns its own semantic index; for the kind-scoped recalls that pass
        # pre-filtered entries, fall back to keyword ranking over them.
        return _keyword_rank(query, entries, k)

    def remember(self, user_msg: str, answer: str = "") -> None:
        # mem0 extracts salient facts itself, so let it see the whole turn.
        self._mem.add(
            f"user: {user_msg}\nassistant: {answer}".strip(), user_id=self._user
        )


def _local_root() -> Path:
    """Resolve MEMORY_DIR (expanding ~ and $VARS) or the default dir."""
    raw = os.environ.get("MEMORY_DIR")
    return Path(os.path.expandvars(raw)).expanduser() if raw else _DEFAULT_DIR


def get_memory() -> Memory | None:
    """Return the configured long-term memory, or None when disabled (default)."""
    backend = (os.environ.get("MEMORY_BACKEND") or "").lower()
    if not backend:
        return None
    user = os.environ.get("MEMORY_USER") or "default"
    if backend == "local":
        return LocalMemory(_local_root(), user)
    if backend == "semantic":
        # Same JSONL store as local, with embedding-based recall (embeddings.py).
        return SemanticMemory(_local_root(), user)
    if backend == "mem0":
        try:
            from mem0 import Memory as _Mem0  # type: ignore[import-not-found]  # optional dep
        except Exception:
            return None
        return _Mem0Adapter(_Mem0(), user)  # type: ignore[return-value]
    return None


# Kinds that carry their own retrieval + prompt framing elsewhere (reflection
# lessons, feedback exemplars/avoids), so plain-fact recall must exclude them —
# otherwise they'd be injected twice, once unframed as a bare memory bullet.
_SPECIAL_KINDS = {"lesson", "exemplar", "avoid"}


def recall_facts(user_msg: str, k: int = 3) -> list[str]:
    """Durable *facts* relevant to ``user_msg`` for auto-injection each turn —
    everything except the specially-framed kinds. Empty when memory is off. Uses
    the same keyword-overlap ranking as ``search`` but reads whole entries so it
    can filter by kind (``search`` returns text only)."""
    mem = get_memory()
    if mem is None:
        return []
    try:
        entries = [e for e in mem.all() if e.get("kind") not in _SPECIAL_KINDS]
    except Exception:
        # A backend without an inspectable store (e.g. mem0) falls back to its own
        # recall over all kinds — acceptable; it manages its own relevance.
        return mem.recall(user_msg, k)
    return [e["text"] for e in mem.rank(user_msg, entries, k)]


def inject(memories: list[str], user_msg: str) -> str:
    """Prepend recalled memories to the user message as a context preamble."""
    if not memories:
        return user_msg
    block = "\n".join(f"- {m}" for m in memories)
    return f"Relevant memory from earlier conversations:\n{block}\n\n{user_msg}"


# --- Model-facing tools (added to the agent only when memory is enabled) -----
# Named for how the model should invoke them; each resolves the active backend
# per call via get_memory(), so they're stateless and safe to bind once.

_DISABLED = "Long-term memory is disabled."


def remember(fact: str, category: str = "note") -> str:
    """Save a durable fact about the user or their portfolio to long-term memory
    so it's available in future conversations.

    Use this when the user states a *lasting* preference, goal, constraint, or
    holding of interest — e.g. risk tolerance, a watchlist ticker, tax situation,
    base currency, or how they like answers formatted. Keep the fact concise and
    self-contained. Do NOT store transient data (quotes, prices, one-off
    calculations) — those go stale. ``category`` is a short free-form tag
    ('preference', 'holding', 'goal', 'fact'). Duplicates are ignored."""
    mem = get_memory()
    if mem is None:
        return _DISABLED
    # Detect a single-valued attribute this will supersede, so the reply can say
    # what it replaced (transparency — the model can tell the user).
    replaced = ""
    key = subject_key(fact) if _supersede_enabled() and category not in _SPECIAL_KINDS else None
    if key:
        try:
            prior = [
                e["text"] for e in mem.all()
                if e.get("kind") not in _SPECIAL_KINDS and subject_key(e["text"]) == key
            ]
        except Exception:
            prior = []
        replaced = f" (updated your {key}; was: {prior[0]})" if prior else ""
    if not mem.save(fact, category):
        return "Already knew that."
    return f"Remembered: {fact}{replaced}"


def recall(query: str) -> str:
    """Search long-term memory for durable facts relevant to ``query`` (things
    remembered in earlier conversations). Relevant memories are also surfaced
    automatically at the start of a turn; call this to look up more."""
    mem = get_memory()
    if mem is None:
        return _DISABLED
    hits = mem.search(query, k=5)
    return "\n".join(f"- {h}" for h in hits) if hits else "No relevant memories."


def forget(query: str) -> str:
    """Remove facts from long-term memory that match ``query`` (use when the user
    says something is no longer true). Returns how many were removed."""
    mem = get_memory()
    if mem is None:
        return _DISABLED
    n = mem.forget(query)
    return f"Forgot {n} memory item(s)." if n else "Nothing matched; nothing removed."


def list_memories() -> str:
    """List everything stored in long-term memory for this user, so you can
    review what you know before answering."""
    mem = get_memory()
    if mem is None:
        return _DISABLED
    entries = mem.all()
    if not entries:
        return "No memories stored yet."
    return "\n".join(f"- [{e.get('kind', 'note')}] {e['text']}" for e in entries)


def memory_tools() -> list:
    """The memory tools to bind to the agent — empty when memory is disabled, so
    the model is never given tools that would silently no-op."""
    return [remember, recall, forget, list_memories] if get_memory() is not None else []
