"""Local and semantic memory, reflection, and feedback tests."""


def test_memory_disabled_by_default(monkeypatch):
    """No MEMORY_BACKEND -> no long-term memory (session-only, hermetic)."""
    monkeypatch.delenv("MEMORY_BACKEND", raising=False)
    from financial_research_assistant.memory import get_memory

    assert get_memory() is None


def test_local_memory_recall_and_persist(monkeypatch, tmp_path):
    """The local backend persists a fact and recalls it by keyword overlap."""
    monkeypatch.setenv("MEMORY_BACKEND", "local")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_USER", "alice")
    from financial_research_assistant.memory import get_memory

    mem = get_memory()
    assert mem is not None
    mem.remember("My favorite color is teal", "Noted your favorite color.")
    assert any("teal" in h for h in mem.recall("what is my favorite color?"))
    assert mem.recall("unrelated stock market forecast") == []


async def test_run_turn_recalls_memory_across_sessions(monkeypatch, tmp_path):
    """Cross-session: a fact remembered in one session is recalled in another
    (memory is scoped by MEMORY_USER, not the session id)."""
    monkeypatch.setenv("MEMORY_BACKEND", "local")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_USER", "bob")
    from financial_research_assistant.adapter import run_turn

    async for _ in run_turn("Remember my project is codenamed Falcon", "sess-a", fake=True):
        pass
    events = [ev async for ev in run_turn("what is my project codename?", "sess-b", fake=True)]
    assert any(ev.kind == "status" and "recalled" in ev.text for ev in events)
    assert any(ev.kind == "final" for ev in events)


def test_mem0_backend_outage_degrades_instead_of_raising():
    """mem0 is a service (vector store plus its own extraction model), reached on
    the way into every turn and again on the way out. An exception from either
    escapes into the turn — recall runs before an answer exists, and the write runs
    after one has been produced and PAID for. Neither is worth a lost turn, which
    is the trade the adapter's other mem0 methods already make."""
    from financial_research_assistant.memory import _Mem0Adapter

    class _Down:
        def search(self, **_kw):
            raise ConnectionError("qdrant refused the connection")

        def add(self, *_a, **_kw):
            raise ConnectionError("qdrant refused the connection")

    mem = _Mem0Adapter(_Down(), "alice")
    assert mem.search("what is my risk tolerance?") == []
    assert mem.recall("what is my risk tolerance?") == []
    assert mem.remember("hi", "hello") is None  # a no-op, not a raise

    # A response the API drifted the shape of must not raise either.
    class _Drifted:
        def search(self, **_kw):
            return {"results": "not a list of hits"}

    assert _Mem0Adapter(_Drifted(), "alice").search("anything") == []


def _local_mem(monkeypatch, tmp_path, user="curator"):
    monkeypatch.setenv("MEMORY_BACKEND", "local")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_USER", user)
    from financial_research_assistant.memory import get_memory

    mem = get_memory()
    assert mem is not None
    return mem


def test_memory_save_dedups(monkeypatch, tmp_path):
    """save() stores once and refuses exact and near-duplicate re-writes."""
    mem = _local_mem(monkeypatch, tmp_path)
    assert mem.save("My risk tolerance is moderate", "preference") is True
    assert mem.save("My risk tolerance is moderate", "preference") is False # exact
    assert mem.save("my risk tolerance is moderate.", "preference") is False # near-dup
    assert len(mem.all()) == 1
    assert mem.save("I want to retire by 2045", "goal") is True # distinct fact stored
    assert len(mem.all()) == 2


def test_memory_forget_removes_matching(monkeypatch, tmp_path):
    """forget() prunes facts matching a substring and reports the count."""
    mem = _local_mem(monkeypatch, tmp_path)
    mem.save("Watchlist includes NVDA", "holding")
    mem.save("Watchlist includes TSLA", "holding")
    mem.save("Base currency is USD", "preference")
    assert mem.forget("watchlist") == 2
    remaining = [e["text"] for e in mem.all()]
    assert remaining == ["Base currency is USD"]
    assert mem.forget("nothing here") == 0


def test_subject_key_only_single_valued_attributes():
    """subject_key names a curated single-valued attribute, or None — so
    multi-value statements never trigger superseding."""
    from financial_research_assistant.memory import subject_key

    assert subject_key("My risk tolerance is now high") == "risk tolerance"
    assert subject_key("my base currency is USD") == "base currency"
    assert subject_key("Call me Kofi") == "name"
    assert subject_key("I own AAPL") is None # not "my … is"
    assert subject_key("My goal is to retire early") is None # goal is multi-value
    assert subject_key("My watchlist includes NVDA") is None # not a single-valued attr


def test_memory_supersedes_changed_single_valued_fact(monkeypatch, tmp_path):
    """A new value for a single-valued attribute replaces the stale one, while
    multi-value facts (holdings) co-exist untouched."""
    mem = _local_mem(monkeypatch, tmp_path)
    mem.save("My risk tolerance is aggressive", "preference")
    mem.save("I own AAPL", "holding")
    mem.save("I own MSFT", "holding")
    mem.save("My risk tolerance is conservative", "preference") # updates the attr
    texts = [e["text"] for e in mem.all()]
    assert "My risk tolerance is conservative" in texts
    assert "My risk tolerance is aggressive" not in texts # stale value dropped
    assert "I own AAPL" in texts and "I own MSFT" in texts # holdings kept
    # A second, unrelated goal is NOT superseded by the first (goal is multi-value).
    mem.save("My goal is to retire at 50", "goal")
    mem.save("My goal is to buy a house", "goal")
    goals = [t for t in (e["text"] for e in mem.all()) if t.startswith("My goal")]
    assert len(goals) == 2


def test_memory_supersede_can_be_disabled(monkeypatch, tmp_path):
    """MEMORY_SUPERSEDE=0 keeps every stated value (no replacement)."""
    monkeypatch.setenv("MEMORY_SUPERSEDE", "0")
    mem = _local_mem(monkeypatch, tmp_path)
    mem.save("My base currency is USD", "preference")
    mem.save("My base currency is EUR", "preference")
    texts = [e["text"] for e in mem.all()]
    assert "My base currency is USD" in texts and "My base currency is EUR" in texts


def test_supersede_never_touches_feedback_or_lessons(monkeypatch, tmp_path):
    """Superseding only affects non-special kinds; an exemplar/lesson mentioning
    the same attribute is never dropped."""
    mem = _local_mem(monkeypatch, tmp_path)
    mem.save("Q: my risk tolerance is high? A: noted", "exemplar")
    mem.save("My risk tolerance is high", "preference")
    mem.save("My risk tolerance is low", "preference") # supersedes the preference only
    kinds = sorted(e["kind"] for e in mem.all())
    assert kinds == ["exemplar", "preference"] # exemplar survived; one preference
    assert any(e["text"] == "My risk tolerance is low" for e in mem.all())


def test_remember_tool_reports_supersede(monkeypatch, tmp_path):
    """The remember tool tells the user what it replaced."""
    from financial_research_assistant import memory as m

    _local_mem(monkeypatch, tmp_path)
    m.remember("My risk tolerance is high", "preference")
    msg = m.remember("My risk tolerance is low", "preference")
    assert "updated your risk tolerance" in msg and "was: My risk tolerance is high" in msg


def test_superseded_value_archived_not_recalled(monkeypatch, tmp_path):
    """A superseded value is kept for audit (all(include_superseded=True)) but
    excluded from the active view, recall, and search — so it never influences
    answers."""
    from financial_research_assistant.memory import recall_facts

    mem = _local_mem(monkeypatch, tmp_path)
    mem.save("My base currency is USD", "preference")
    mem.save("My base currency is EUR", "preference") # supersedes USD
    active = [e["text"] for e in mem.all()]
    assert active == ["My base currency is EUR"] # stale hidden from active
    assert mem.search("base currency") == ["My base currency is EUR"]
    assert recall_facts("what currency do I use") == ["My base currency is EUR"]
    # ...but retained for audit, marked with the date it was replaced.
    audit = mem.all(include_superseded=True)
    archived = [e for e in audit if e.get("superseded")]
    assert len(archived) == 1 and archived[0]["text"] == "My base currency is USD"
    assert archived[0]["superseded"] # a date stamp


def test_restating_archived_value_reactivates_it(monkeypatch, tmp_path):
    """Re-stating a previously-superseded value makes it active again (and
    archives the current one) — the value isn't blocked by dedup against the
    archive."""
    mem = _local_mem(monkeypatch, tmp_path)
    mem.save("My risk tolerance is high", "preference")
    mem.save("My risk tolerance is low", "preference") # high archived
    mem.save("My risk tolerance is high", "preference") # reactivate high
    assert [e["text"] for e in mem.all()] == ["My risk tolerance is high"]


def test_list_memories_tool_hides_superseded(monkeypatch, tmp_path):
    """The model-facing list_memories tool shows only active memories — the model
    never reasons over archived stale values."""
    from financial_research_assistant import memory as m

    _local_mem(monkeypatch, tmp_path)
    m.get_memory().save("My tax bracket is 24%", "preference")
    m.get_memory().save("My tax bracket is 32%", "preference")
    listed = m.list_memories()
    assert "32%" in listed and "24%" not in listed


def test_memory_auto_capture_skips_questions(monkeypatch, tmp_path):
    """Auto-capture stores durable statements but not passing questions or
    one-off data — the store stays clean instead of hoarding every exchange."""
    mem = _local_mem(monkeypatch, tmp_path)
    mem.remember("What is AAPL's P/E ratio right now?", "12x") # a question → skip
    mem.remember("Show me my biggest position", "…") # a request → skip
    assert mem.all() == []
    mem.remember("I prefer concise answers with tables", "ok") # durable → stored
    mem.remember("From now on benchmark me against QQQ", "ok") # explicit → stored
    texts = [e["text"] for e in mem.all()]
    assert any("concise" in t for t in texts)
    assert any("QQQ" in t for t in texts)


def test_memory_auto_capture_skips_action_requests(monkeypatch, tmp_path):
    """An imperative 'I want a chart / I need a comparison' is a one-off task, not a
    durable fact — it must NOT be stored (and re-injected into later turns), even
    though the bare verb 'want'/'need' looks durable. A real goal or preference
    phrased with the same verb is still captured."""
    mem = _local_mem(monkeypatch, tmp_path)
    mem.remember("I want a chart of AAPL", "…") # task → skip
    mem.remember("I need a comparison of MSFT and GOOG", "…") # task → skip
    mem.remember("I'd like a DCF valuation of NVDA", "…") # task → skip
    mem.remember("I want the latest price of TSLA", "…") # task → skip
    assert mem.all() == []
    # ...but a genuine goal / preference with the same verb IS durable.
    mem.remember("I want to retire early", "ok")
    mem.remember("I need lower-risk holdings", "ok")
    texts = [e["text"] for e in mem.all()]
    assert any("retire early" in t for t in texts)
    assert any("lower-risk" in t for t in texts)


def test_memory_tools_gated_by_backend(monkeypatch, tmp_path):
    """The model-facing memory tools are present only when a backend is set."""
    from financial_research_assistant.memory import memory_tools

    monkeypatch.delenv("MEMORY_BACKEND", raising=False)
    assert memory_tools() == []
    _local_mem(monkeypatch, tmp_path)
    names = {t.__name__ for t in memory_tools()}
    assert names == {"remember", "recall", "forget", "list_memories"}


def test_memory_tool_functions_roundtrip(monkeypatch, tmp_path):
    """The remember/recall/list_memories/forget tools operate on the active
    backend and no-op cleanly when memory is disabled."""
    from financial_research_assistant import memory as m

    _local_mem(monkeypatch, tmp_path)
    assert "Remembered" in m.remember("My base currency is USD", "preference")
    assert m.remember("My base currency is USD") == "Already knew that." # dedup
    assert "USD" in m.recall("what currency do I use")
    assert "USD" in m.list_memories()
    assert "Forgot 1" in m.forget("base currency")
    assert m.list_memories() == "No memories stored yet."
    # Disabled backend → every tool reports it, nothing raises.
    monkeypatch.delenv("MEMORY_BACKEND", raising=False)
    assert m.remember("x") == m.recall("x") == m.list_memories() == "Long-term memory is disabled."


def test_build_real_graph_binds_memory_tools_when_enabled(monkeypatch, tmp_path):
    """With memory on, _build_real_graph binds the memory tools and adds the
    memory guidance; with it off, neither appears."""
    captured: dict = {}
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kw: object())
    monkeypatch.setattr(
        "langchain.agents.create_agent", lambda **kw: captured.update(kw) or kw
    )
    from financial_research_assistant.graph import _build_real_graph

    monkeypatch.delenv("MEMORY_BACKEND", raising=False)
    _build_real_graph(think=False)
    off_names = {getattr(t, "__name__", "") for t in captured["tools"]}
    assert "remember" not in off_names
    assert "LONG-TERM MEMORY" not in captured["system_prompt"]

    _local_mem(monkeypatch, tmp_path)
    _build_real_graph(think=False)
    on_names = {getattr(t, "__name__", "") for t in captured["tools"]}
    assert {"remember", "recall", "forget", "list_memories"} <= on_names
    assert "LONG-TERM MEMORY" in captured["system_prompt"]


# -- reflection / self-critique (self-learning layer #2) --------------------


def test_reflection_gap_detection_and_flag(monkeypatch):
    """_gap_lessons flags only sections whose data was thin/unavailable, and the
    LLM critique honors RESEARCH_REFLECT."""
    from financial_research_assistant import reflection

    lessons = reflection._gap_lessons(
        "AAPL", [("Fundamentals", "P/E 30"), ("Earnings", "No data available")]
    )
    assert len(lessons) == 1 and "'Earnings'" in lessons[0] and "AAPL" in lessons[0]
    monkeypatch.setenv("RESEARCH_REFLECT", "0")
    assert reflection._reflect_enabled() is False
    monkeypatch.setenv("RESEARCH_REFLECT", "1")
    assert reflection._reflect_enabled() is True


async def test_reflect_stores_and_recalls_lessons(monkeypatch, tmp_path):
    """reflect() persists gap lessons (kind='lesson', subject-scoped) and
    recall_lessons() finds them for the same subject only."""
    from financial_research_assistant import reflection

    mem = _local_mem(monkeypatch, tmp_path)
    sections = [("Fundamentals", "P/E 30"), ("ETF look-through", "(unavailable: not a fund)")]
    stored = await reflection.reflect("AAPL", sections, "report body", fake=True)
    assert any("ETF look-through" in s for s in stored)
    assert any(e["kind"] == "lesson" for e in mem.all())
    assert reflection.recall_lessons("AAPL") # recalled for its subject
    assert reflection.recall_lessons("TSLA") == [] # not for an unrelated one


async def test_reflect_is_noop_without_memory(monkeypatch):
    """With long-term memory off, reflection stores nothing and recalls nothing —
    --research behaves exactly as before."""
    monkeypatch.delenv("MEMORY_BACKEND", raising=False)
    from financial_research_assistant import reflection

    stored = await reflection.reflect("AAPL", [("X", "(unavailable)")], "r", fake=True)
    assert stored == []
    assert reflection.recall_lessons("AAPL") == []


async def test_prior_lessons_injected_into_synthesis(monkeypatch):
    """synthesize_report folds recalled lessons into the model prompt so the next
    report addresses known gaps."""
    from financial_research_assistant import research

    class _CapLLM:
        def __init__(self):
            self.seen = None

        async def ainvoke(self, messages):
            self.seen = messages
            return type("R", (), {"content": "REPORT"})()

    cap = _CapLLM()
    monkeypatch.setattr(
        "financial_research_assistant.llm._make_llm", lambda model=None: cap
    )
    out = await research.synthesize_report(
        "AAPL", [("Fundamentals", "P/E 30")], lessons=["Check AAPL ETF exposure next time"]
    )
    assert out == "REPORT"
    human = cap.seen[1].content
    assert "Lessons from prior research" in human and "ETF exposure" in human


def test_research_report_tool_applies_prior_lessons(monkeypatch, tmp_path):
    """The sync research_report tool prepends prior lessons for the ticker so the
    agent applies them when synthesizing inline."""
    from financial_research_assistant import research

    mem = _local_mem(monkeypatch, tmp_path)
    mem.save(
        "When researching MSFT, the 'Earnings' data was thin — check an alternate source.",
        kind="lesson",
    )
    monkeypatch.setattr(
        research, "_section_fns", lambda sym: {"Fundamentals": lambda: f"{sym} P/E 30"}
    )
    out = research.research_report("MSFT")
    assert "Lessons from prior research on this ticker" in out and "Earnings" in out


# -- feedback capture → few-shot (self-learning layer #3) -------------------


def test_feedback_record_and_recall(monkeypatch, tmp_path):
    """record() stores rated exchanges as exemplar/avoid memories; recall_feedback
    finds the relevant ones by keyword overlap and keeps the two kinds apart."""
    from financial_research_assistant import feedback

    _local_mem(monkeypatch, tmp_path)
    assert feedback.record(
        "How much dividend income did I get?", "You received $1,240 in 2025 …", good=True
    )
    assert feedback.record(
        "Give me AAPL analysis", "AAPL is a stock. The end.", good=False, note="too shallow"
    )
    exemplars, avoids = feedback.recall_feedback("what's my dividend income this year?")
    assert any("dividend income" in e for e in exemplars)
    assert avoids == [] # the 'avoid' is about AAPL, not dividends
    ex2, av2 = feedback.recall_feedback("analyze AAPL for me")
    assert any("AAPL" in a for a in av2) and ex2 == []


def test_feedback_noop_without_memory(monkeypatch):
    """With memory off, feedback records nothing and recalls nothing."""
    monkeypatch.delenv("MEMORY_BACKEND", raising=False)
    from financial_research_assistant import feedback

    assert feedback.record("q", "a", good=True) is False
    assert feedback.recall_feedback("q") == ([], [])


def test_recall_facts_excludes_special_kinds(monkeypatch, tmp_path):
    """recall_facts returns durable facts but never lesson/exemplar/avoid rows —
    those are injected with their own framing, so plain-fact recall must skip
    them to avoid double-injection."""
    from financial_research_assistant.memory import recall_facts

    mem = _local_mem(monkeypatch, tmp_path)
    mem.save("My base currency is USD", "preference")
    mem.save("When researching USD pairs, FX data was thin", "lesson")
    mem.save("Q: USD question\nA: a good USD answer", "exemplar")
    facts = recall_facts("what is my usd currency")
    assert facts == ["My base currency is USD"]


async def test_run_turn_injects_fewshot_feedback(monkeypatch, tmp_path):
    """A good-rated exchange becomes few-shot guidance injected into a later,
    similar turn (the fake graph echoes the injected content, so we can see it)."""
    monkeypatch.setenv("MEMORY_BACKEND", "local")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_USER", "rater")
    from financial_research_assistant import feedback
    from financial_research_assistant.adapter import run_turn

    feedback.record(
        "How much dividend income did I get?", "You received $1,240 across 2025 …", good=True
    )
    events = [ev async for ev in run_turn("what's my dividend income this year?", "s1", fake=True)]
    status = [ev.text for ev in events if ev.kind == "status"]
    final = next(ev.text for ev in events if ev.kind == "final")
    assert any("feedback" in s for s in status)
    assert "rated GOOD" in final # the few-shot framing reached the model prompt


# -- semantic recall (embedding-based memory) -------------------------------

# A deterministic offline stand-in for real embeddings: maps concept words to a
# shared dimension, so synonyms with NO shared keyword still get a high cosine.
_CONCEPTS = {
    0: {"risk", "aggressive", "tolerance", "volatile", "volatility"},
    1: {"dividend", "income", "yield", "distribution"},
    2: {"currency", "usd", "dollar", "fx"},
}


def _fake_embed(text: str):
    import re as _re

    words = set(_re.findall(r"[a-z]+", (text or "").lower()))
    return [1.0 if words & concept else 0.0 for concept in _CONCEPTS.values()]


def _semantic_mem(monkeypatch, tmp_path, embed=_fake_embed):
    monkeypatch.setenv("MEMORY_BACKEND", "semantic")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_USER", "sem")
    monkeypatch.setattr("financial_research_assistant.embeddings.embed_query", embed)
    from financial_research_assistant.memory import SemanticMemory, get_memory

    mem = get_memory()
    assert isinstance(mem, SemanticMemory)
    return mem


def test_semantic_recall_beats_keyword(monkeypatch, tmp_path):
    """Semantic recall finds a related memory with NO shared keyword (which the
    keyword store would miss), and drops unrelated ones below threshold."""
    mem = _semantic_mem(monkeypatch, tmp_path)
    mem.save("My risk tolerance is high", "preference")
    assert mem.all()[0].get("vec") # embedding was stored
    # "aggressive" shares no word with "risk tolerance", but the concept matches.
    assert mem.search("how aggressive should I be") == ["My risk tolerance is high"]
    # Unrelated concept → below the similarity threshold → nothing.
    assert mem.search("what's the weather today") == []


def test_semantic_falls_back_to_keyword_without_embeddings(monkeypatch, tmp_path):
    """When embeddings are unavailable (embed returns None), the semantic backend
    saves without a vector and ranks by keyword — never hard-failing."""
    mem = _semantic_mem(monkeypatch, tmp_path, embed=lambda t: None)
    mem.save("My base currency is USD", "preference")
    assert "vec" not in mem.all()[0]
    assert mem.search("what currency do I use") == ["My base currency is USD"]


def test_recall_facts_uses_semantic_ranking(monkeypatch, tmp_path):
    """The per-turn fact injection (recall_facts) goes through the backend's rank,
    so it benefits from semantic recall too."""
    _semantic_mem(monkeypatch, tmp_path)
    from financial_research_assistant.memory import recall_facts, get_memory

    get_memory().save("My risk tolerance is high", "preference")
    assert recall_facts("how aggressive should I be") == ["My risk tolerance is high"]


# -- dedup scoping ----------------------------------------------------------


def test_a_bad_rating_is_not_dropped_as_a_duplicate_of_the_good_one(monkeypatch, tmp_path):
    """/good then /bad on one answer stores the SAME Q/A text under two kinds. The
    correction used to be discarded as a near-duplicate, leaving the answer the
    user had just rejected in the store as an example to emulate."""
    from financial_research_assistant.feedback import record, recall_feedback

    mem = _local_mem(monkeypatch, tmp_path)
    q, a = "how risky is my portfolio?", "It is quite risky, broadly speaking."
    assert record(q, a, good=True) is True
    assert record(q, a, good=False, note="too vague") is True, "the correction is kept"

    kinds = [e["kind"] for e in mem.all()]
    assert kinds == ["avoid"], "and the verdict it reversed no longer recalls"
    exemplars, avoids = recall_feedback(q)
    assert exemplars == [] and len(avoids) == 1
    # The reversed verdict is archived, not deleted — the history is auditable.
    archived = [e for e in mem.all(include_superseded=True) if e.get("superseded")]
    assert [e["kind"] for e in archived] == ["exemplar"]


def test_two_gap_lessons_differing_only_by_section_are_both_kept(monkeypatch, tmp_path):
    """Gap lessons are one template with the section label substituted in, so
    nearly every term is boilerplate. At the free-form near-duplicate threshold the
    second section's lesson was swallowed by the first and only one gap per run was
    ever learned."""
    from financial_research_assistant import reflection

    mem = _local_mem(monkeypatch, tmp_path)
    lessons = reflection._gap_lessons(
        "AAPL", [("News", "(unavailable)"), ("Peers", "(unavailable)")]
    )
    assert len(lessons) == 2
    assert all(mem.save(lesson, kind="lesson") for lesson in lessons)
    assert len({e["text"] for e in mem.all()}) == 2
    # A genuine restatement of one of them is still a no-op.
    assert mem.save(lessons[0], kind="lesson") is False


def test_a_lesson_never_dedups_against_a_plain_fact(monkeypatch, tmp_path):
    """The kinds are separate channels with their own retrieval and framing, so a
    lesson must not be suppressed by a fact that happens to share its words."""
    mem = _local_mem(monkeypatch, tmp_path)
    text = "When researching AAPL, check the dividend history before writing"
    assert mem.save(text, kind="note") is True
    assert mem.save(text, kind="lesson") is True
    assert sorted(e["kind"] for e in mem.all()) == ["lesson", "note"]


def test_the_store_is_read_as_utf8_not_the_platform_default(monkeypatch, tmp_path):
    """The store is WRITTEN as UTF-8, so it must be read as UTF-8. Left to the
    platform default, a store holding "€" (a base currency, a European ticker)
    raised UnicodeDecodeError out of the read itself — outside the per-line guard —
    on every turn that recalled."""
    from pathlib import Path

    mem = _local_mem(monkeypatch, tmp_path)
    assert mem.save("My base currency is € (euro), never $", "preference") is True

    real_read = Path.read_text

    def platform_default_cannot_decode(self, *args, encoding=None, **kwargs):
        # Stands in for a machine whose default encoding is not UTF-8 — the knob
        # under test, so a read that doesn't name its encoding fails here.
        if encoding is None:
            raise UnicodeDecodeError("charmap", b"\x80", 0, 1, "undefined character")
        return real_read(self, *args, encoding=encoding, **kwargs)

    monkeypatch.setattr(Path, "read_text", platform_default_cannot_decode)
    assert any("€" in e["text"] for e in mem.all())


# -- reflection: a per-field n/a is not a missing section --------------------


def test_a_per_field_na_is_not_a_data_gap():
    """`fundamentals.py` prints "n/a" for ANY missing field, so as a substring
    marker it fired on healthy output: every company that pays no dividend taught a
    permanent "the fundamentals data was unavailable" lesson, which was then
    injected into every later report on it."""
    from financial_research_assistant import reflection

    healthy = (
        "AAPL · Apple Inc.\n"
        "  sector: Technology · industry: Consumer Electronics\n"
        "  P/E: 31.20 · dividend yield: n/a · payout ratio: n/a\n"
    )
    assert reflection._gap_lessons("AAPL", [("Fundamentals", healthy)]) == []
    # A section that is NOTHING but the marker is still a gap.
    assert len(reflection._gap_lessons("AAPL", [("Dividends", "n/a")])) == 1
    assert len(reflection._gap_lessons("AAPL", [("Earnings", "No data available")])) == 1
