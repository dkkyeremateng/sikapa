"""Subagent-dispatch tests.

The dispatch tools delegate to `run_subagent`, which builds and runs a real
ReAct subagent (model + tools). That seam is monkeypatched here so the fan-out /
sequencing / error-handling logic is exercised fully offline — no model, no
network. One structural test also confirms the recursion guard (a subagent's
toolset never contains the dispatch tools).
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from financial_research_assistant import subagents, tools


# --- pure helpers ----------------------------------------------------------

def test_parse_tasks_splits_lines_and_double_pipe():
    assert subagents._parse_tasks("a\n b \n\nc") == ["a", "b", "c"]
    assert subagents._parse_tasks("one || two || three") == ["one", "two", "three"]
    assert subagents._parse_tasks("   ") == []
    # A single multi-word line is one task (|| only splits when it's present).
    assert subagents._parse_tasks("research AAPL earnings") == ["research AAPL earnings"]


def test_final_text_prefers_last_ai_message():
    result = {"messages": [
        HumanMessage(content="do it"),
        AIMessage(content="intermediate"),
        AIMessage(content="final findings"),
    ]}
    assert subagents._final_text(result) == "final findings"
    assert subagents._final_text({"messages": []}) == ""


def test_subagent_toolset_excludes_dispatch_tools():
    """Recursion guard: a subagent must not receive the dispatch tools, so it can
    never spawn further subagents."""
    names = {getattr(t, "name", None) or getattr(t, "__name__", "")
             for t in subagents._subagent_tools()}
    assert not (names & {"dispatch_subagent", "dispatch_subagents"})
    # …but it does get the ordinary research tools.
    assert "stock_fundamentals" in names and "price_history_chart" in names


def test_subagent_prompt_guards_untrusted_content_and_file_paths():
    """The subagent inherits the untrusted-data rule AND the file-path clause —
    it holds filesystem tools (import/export/ingest), so a path must come from its
    task, never from web/tool content it fetches."""
    prompt = subagents.SUBAGENT_SYSTEM_PROMPT
    assert "UNTRUSTED DATA" in prompt
    assert "file path" in prompt.lower() and "never from web or tool content" in prompt


# --- single dispatch -------------------------------------------------------

async def test_dispatch_subagent_runs_one_and_returns_findings(monkeypatch):
    async def fake_run(task, model=None):
        return f"FINDINGS for: {task}"

    monkeypatch.setattr(subagents, "run_subagent", fake_run)
    out = await subagents._dispatch_subagent("Research AAPL valuation")
    assert out == "FINDINGS for: Research AAPL valuation"


async def test_dispatch_subagent_rejects_empty_task():
    out = await subagents._dispatch_subagent("   ")
    assert "Give a task" in out


async def test_dispatch_subagent_reports_failure_gracefully(monkeypatch):
    async def boom(task, model=None):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(subagents, "run_subagent", boom)
    out = await subagents._dispatch_subagent("anything")
    assert "subagent failed" in out and "model exploded" in out


async def test_dispatch_subagent_reports_timeout(monkeypatch):
    async def slow(task, model=None):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(subagents, "run_subagent", slow)
    out = await subagents._dispatch_subagent("anything")
    assert "timed out" in out


# --- parallel / sequence fan-out -------------------------------------------

async def test_dispatch_subagents_parallel_runs_all_concurrently(monkeypatch):
    order = []

    async def fake_run(task, model=None):
        order.append(("start", task))
        await asyncio.sleep(0)  # yield so all three interleave
        return f"result::{task}"

    monkeypatch.setattr(subagents, "run_subagent", fake_run)
    out = await subagents._dispatch_subagents("AAPL\nMSFT\nNVDA", mode="parallel")
    assert "mode=parallel" in out
    for sym in ("AAPL", "MSFT", "NVDA"):
        assert f"result::{sym}" in out
    assert "DISPATCHED 3 subagent(s)" in out


async def test_dispatch_subagents_sequence_chains_context_forward(monkeypatch):
    seen = []

    async def fake_run(task, model=None):
        seen.append(task)
        return f"[out for {task.splitlines()[0]}]"

    monkeypatch.setattr(subagents, "run_subagent", fake_run)
    out = await subagents._dispatch_subagents(
        "survey the sector\ndeep-dive the standout", mode="sequence"
    )
    assert "mode=sequence" in out
    # The second subagent's task must carry a digest of the first's result.
    assert len(seen) == 2
    assert "Context from earlier subagents" in seen[1]
    assert "survey the sector" in seen[1]  # earlier task label in the digest


async def test_dispatch_subagents_caps_task_count(monkeypatch):
    async def fake_run(task, model=None):
        return "ok"

    monkeypatch.setattr(subagents, "run_subagent", fake_run)
    many = "\n".join(f"task {i}" for i in range(9))  # 9 > _MAX_TASKS (6)
    out = await subagents._dispatch_subagents(many, mode="parallel")
    assert "DISPATCHED 6 subagent(s)" in out
    assert "3 task(s) beyond" in out


async def test_dispatch_subagents_empty_prompts_for_tasks():
    out = await subagents._dispatch_subagents("", mode="parallel")
    assert "No tasks given" in out


async def test_one_bad_task_does_not_sink_the_batch(monkeypatch):
    async def flaky(task, model=None):
        if "bad" in task:
            raise RuntimeError("nope")
        return f"ok::{task}"

    monkeypatch.setattr(subagents, "run_subagent", flaky)
    out = await subagents._dispatch_subagents("good one\nbad two\ngood three", mode="parallel")
    assert "ok::good one" in out and "ok::good three" in out
    assert "subagent failed" in out  # the bad one is a labeled note, not a crash


# --- registration ----------------------------------------------------------

def test_subagent_model_env_resolution(monkeypatch):
    """SUBAGENT_MODEL sets the subagent model; an explicit arg overrides it; unset
    -> None (falls back to the primary model in _make_llm)."""
    monkeypatch.delenv("SUBAGENT_MODEL", raising=False)
    assert subagents._subagent_model() is None
    monkeypatch.setenv("SUBAGENT_MODEL", "  cheap-mini  ")
    assert subagents._subagent_model() == "cheap-mini"


def test_build_subagent_passes_env_model_to_llm(monkeypatch):
    """_build_subagent resolves SUBAGENT_MODEL and hands it to _make_llm; an
    explicit model argument wins over the env."""
    captured = {}

    def fake_make_llm(model=None, **kw):
        captured["model"] = model
        captured.update(kw)
        return object()

    def fake_create_agent(model, tools, system_prompt, checkpointer):
        return "AGENT"

    monkeypatch.setattr("financial_research_assistant.graph._make_llm", fake_make_llm)
    monkeypatch.setattr("langchain.agents.create_agent", fake_create_agent)

    monkeypatch.setenv("SUBAGENT_MODEL", "env-model")
    subagents._build_subagent()
    assert captured["model"] == "env-model"

    subagents._build_subagent(model="explicit-model")
    assert captured["model"] == "explicit-model"  # explicit arg wins


def test_subagent_openai_compatible_overrides(monkeypatch):
    """SUBAGENT_API_BASE / SUBAGENT_API_KEY / SUBAGENT_MODEL_PROVIDER are read into
    _make_llm overrides so a subagent can use its OWN OpenAI-compatible endpoint;
    each unset value is None so _make_llm falls back to the primary agent's config."""
    monkeypatch.delenv("SUBAGENT_API_BASE", raising=False)
    monkeypatch.delenv("SUBAGENT_API_KEY", raising=False)
    monkeypatch.delenv("SUBAGENT_MODEL_PROVIDER", raising=False)
    assert subagents._subagent_llm_overrides() == {
        "provider": None, "base_url": None, "api_key": None,
    }

    monkeypatch.setenv("SUBAGENT_API_BASE", "http://localhost:11434/v1")
    monkeypatch.setenv("SUBAGENT_API_KEY", "local-key")
    monkeypatch.setenv("SUBAGENT_MODEL_PROVIDER", "openai")
    assert subagents._subagent_llm_overrides() == {
        "provider": "openai",
        "base_url": "http://localhost:11434/v1",
        "api_key": "local-key",
    }

    captured = {}

    def fake_make_llm(model=None, **kw):
        captured["model"] = model
        captured.update(kw)
        return object()

    monkeypatch.setattr("financial_research_assistant.graph._make_llm", fake_make_llm)
    monkeypatch.setattr("langchain.agents.create_agent",
                        lambda **kw: "AGENT")
    monkeypatch.setenv("SUBAGENT_MODEL", "qwen2.5-7b-instruct")
    subagents._build_subagent()
    assert captured == {
        "model": "qwen2.5-7b-instruct",
        "provider": "openai",
        "base_url": "http://localhost:11434/v1",
        "api_key": "local-key",
        "scope": "subagent",
    }


def test_make_llm_openai_compatible_override_reaches_chatopenai(monkeypatch):
    """_make_llm forwards base_url/api_key overrides to ChatOpenAI (the same
    OpenAI-compatible path the primary agent uses), preferring them over env."""
    from financial_research_assistant import graph

    seen = {}
    monkeypatch.setattr(
        "langchain_openai.ChatOpenAI",
        lambda **kw: seen.update(kw) or object(),
    )
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_BASE", "http://primary/v1")
    graph._make_llm("m", base_url="http://sub/v1", api_key="sub-key")
    assert seen["base_url"] == "http://sub/v1"  # override wins over OPENAI_API_BASE
    assert seen["model"] == "m"


def test_make_llm_overrides_reach_non_openai_provider(monkeypatch):
    """Regression: the non-OpenAI path used to accept base_url/api_key and drop
    them, so SUBAGENT_*/QUICK_* were silent no-ops for anthropic/google/groq."""
    from financial_research_assistant import graph

    seen = {}

    def fake_init(model, **kw):
        seen.update(kw, model=model)
        return object()

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init)
    graph._make_llm(
        "claude-sonnet-5",
        provider="anthropic",
        base_url="http://gateway/v1",
        api_key="gw-key",
    )
    assert seen == {
        "model": "claude-sonnet-5",
        "model_provider": "anthropic",
        "base_url": "http://gateway/v1",
        "api_key": "gw-key",
    }


def test_make_llm_non_openai_without_overrides_is_unchanged(monkeypatch):
    """No override => the original env-only call, so existing setups are untouched."""
    from financial_research_assistant import graph

    seen = {}
    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model",
        lambda model, **kw: seen.update(kw, model=model) or object(),
    )
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    graph._make_llm()
    assert seen == {"model": "claude-sonnet-5", "model_provider": "anthropic"}


def test_make_llm_non_openai_falls_back_when_kwargs_rejected(monkeypatch):
    """A provider integration that doesn't take base_url/api_key must not take the
    turn down — fall back to the env-only construction instead of raising."""
    from financial_research_assistant import graph

    calls = []

    def picky_init(model, **kw):
        calls.append(kw)
        if "base_url" in kw:
            raise TypeError("unexpected keyword argument 'base_url'")
        return "LLM"

    monkeypatch.setattr("langchain.chat_models.init_chat_model", picky_init)
    assert graph._make_llm("m", provider="groq", base_url="http://x/v1") == "LLM"
    assert len(calls) == 2 and "base_url" not in calls[1]


def test_dispatch_tools_registered_with_async_impl():
    by_name = {t.name: t for t in subagents.SUBAGENT_TOOLS}
    assert set(by_name) == {"dispatch_subagent", "dispatch_subagents"}
    assert all(t.coroutine is not None for t in by_name.values())
    names = {getattr(t, "name", None) or getattr(t, "__name__", "") for t in tools.TOOLS}
    assert {"dispatch_subagent", "dispatch_subagents"} <= names
