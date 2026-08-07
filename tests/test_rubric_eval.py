"""The rubric eval type: grading the derivation, not just the answer.

Offline — the judge model is stubbed. What matters here is the SCORING contract,
because every failure mode it has inflates scores rather than lowering them, and
an eval that silently over-reports is worse than no eval.
"""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def ev():
    """The eval harness, loaded by path (it's a script dir, not a package)."""
    eval_dir = Path(__file__).resolve().parent.parent / "eval"
    spec = importlib.util.spec_from_file_location("evaluate", eval_dir / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_judge(monkeypatch, ev, reply):
    """Stand in for the judge model with a fixed reply."""
    class _Resp:
        content = reply

    class _LLM:
        def invoke(self, _messages):
            return _Resp()

    from financial_research_assistant import llm

    monkeypatch.setattr(llm, "_make_llm", lambda *a, **k: _LLM())


_CRITERIA = [
    {"dimension": "source", "requirement": "uses as-reported filings", "weight": 2.0},
    {"dimension": "period", "requirement": "covers the right period", "weight": 1.0},
]


# --- parsing the judge's reply --------------------------------------------------


def test_scores_are_read_per_dimension(ev):
    got = ev._parse_rubric('{"source": 1.0, "period": 0.5}', ["source", "period"])
    assert got == {"source": 1.0, "period": 0.5}


def test_a_missing_dimension_scores_zero_rather_than_being_dropped(ev):
    """Dropping it would shrink the weighted denominator and RAISE the mean — a
    dimension the judge never assessed would become a free pass."""
    got = ev._parse_rubric('{"source": 1.0}', ["source", "period"])
    assert got == {"source": 1.0, "period": 0.0}


def test_scores_are_clamped(ev):
    got = ev._parse_rubric('{"source": 4.2, "period": -3}', ["source", "period"])
    assert got == {"source": 1.0, "period": 0.0}


def test_prose_around_the_json_is_tolerated(ev):
    got = ev._parse_rubric(
        'Here is my assessment:\n```json\n{"source": 0.25, "period": 1}\n```\nDone.',
        ["source", "period"],
    )
    assert got == {"source": 0.25, "period": 1.0}


def test_an_unparseable_reply_scores_zero_not_full_marks(ev):
    """The safe direction is down. A judge that answered nothing usable has told
    us nothing, and treating that as a pass would quietly inflate every run."""
    assert ev._parse_rubric("I could not assess this.", ["source"]) == {"source": 0.0}
    assert ev._parse_rubric("", ["source"]) == {"source": 0.0}


# --- weighting ------------------------------------------------------------------


def test_the_score_is_weighted_by_dimension(monkeypatch, ev):
    _stub_judge(monkeypatch, ev, '{"source": 1.0, "period": 0.0}')
    value, breakdown = ev._rubric_judge("q", "an answer", [], _CRITERIA, fake=False)
    assert value == pytest.approx(2.0 / 3.0), "source is weighted 2, period 1"
    assert [b["dimension"] for b in breakdown] == ["source", "period"]
    assert breakdown[0]["score"] == 1.0 and breakdown[1]["score"] == 0.0


def test_the_breakdown_carries_the_requirement(monkeypatch, ev):
    """It is what the improvement report prints, so a reader sees WHAT failed
    rather than a bare dimension name."""
    _stub_judge(monkeypatch, ev, '{"source": 0.0, "period": 1.0}')
    _value, breakdown = ev._rubric_judge("q", "a", [], _CRITERIA, fake=False)
    assert breakdown[0]["requirement"] == "uses as-reported filings"


def test_a_model_failure_scores_zero_and_does_not_raise(monkeypatch, ev):
    from financial_research_assistant import llm

    def boom(*_a, **_k):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(llm, "_make_llm", boom)
    value, breakdown = ev._rubric_judge("q", "a", [], _CRITERIA, fake=False)
    assert value == 0.0 and breakdown == []


# --- what the judge is shown ----------------------------------------------------


def test_the_judge_sees_the_tool_trajectory(monkeypatch, ev):
    """The whole reason `rubric` exists: `llm_judge` reads the final text and
    nothing else, so it cannot tell a grounded figure from an asserted one."""
    seen = {}

    class _Resp:
        content = '{"source": 1.0, "period": 1.0}'

    class _LLM:
        def invoke(self, messages):
            seen["prompt"] = messages[0].content
            return _Resp()

    from financial_research_assistant import llm

    monkeypatch.setattr(llm, "_make_llm", lambda *a, **k: _LLM())
    ev._rubric_judge(
        "was NVDA cheap in Jan 2025?", "It was.",
        [{"name": "sec_financials"}, {"name": "price_history_chart"}],
        _CRITERIA, fake=False,
    )
    assert "sec_financials, price_history_chart" in seen["prompt"]
    assert "uses as-reported filings" in seen["prompt"], "requirements must reach it"
    assert "was NVDA cheap in Jan 2025?" in seen["prompt"]


def test_a_run_with_no_tools_says_none(monkeypatch, ev):
    seen = {}

    class _Resp:
        content = "{}"

    class _LLM:
        def invoke(self, messages):
            seen["prompt"] = messages[0].content
            return _Resp()

    from financial_research_assistant import llm

    monkeypatch.setattr(llm, "_make_llm", lambda *a, **k: _LLM())
    ev._rubric_judge("q", "an ungrounded answer", [], _CRITERIA, fake=False)
    assert "(none)" in seen["prompt"]


# --- integration with score() and the improvement loop --------------------------


def test_score_dispatches_rubric_and_fills_the_detail(monkeypatch, ev):
    _stub_judge(monkeypatch, ev, '{"source": 1.0, "period": 1.0}')
    item = {"query": "q", "eval_type": "rubric", "criteria": _CRITERIA}
    detail = {}
    value = ev.score(item, "an answer", [], fake=False, detail=detail)
    assert value == 1.0
    assert [d["dimension"] for d in detail["rubric"]] == ["source", "period"]


def test_detail_is_optional_so_existing_callers_are_unaffected(monkeypatch, ev):
    _stub_judge(monkeypatch, ev, '{"source": 1.0, "period": 1.0}')
    item = {"query": "q", "eval_type": "rubric", "criteria": _CRITERIA}
    assert ev.score(item, "an answer", [], fake=False) == 1.0


def test_fake_mode_never_calls_a_model(monkeypatch, ev):
    """`--fake` must stay hermetic — the harness self-test runs with no keys."""
    from financial_research_assistant import llm

    def explode(*_a, **_k):
        raise AssertionError("the rubric judge called a model in fake mode")

    monkeypatch.setattr(llm, "_make_llm", explode)
    value, breakdown = ev._rubric_judge("q", "some output", [], _CRITERIA, fake=True)
    assert value == 1.0 and len(breakdown) == 2
    assert ev._rubric_judge("q", "", [], _CRITERIA, fake=True)[0] == 0.0


def test_a_rubric_miss_names_the_failing_dimensions():
    """The diagnosis has to say WHICH part of the derivation failed — that is the
    difference between a rubric result and an opaque low score."""
    from financial_research_assistant import improve

    d = improve.diagnose(
        {"query": "was NVDA cheap in Jan 2025?", "eval_type": "rubric",
         "criteria": _CRITERIA},
        score=0.33, tools=["stock_fundamentals"], answer="It was cheap.",
        rubric=[
            {"dimension": "as_of", "score": 0.0, "requirement": "answer about January"},
            {"dimension": "source", "score": 1.0, "requirement": "as-reported"},
        ],
    )
    assert d is not None and d["kind"] == "derivation"
    assert [f["dimension"] for f in d["failed"]] == ["as_of"], "only the failures"
    assert d["called"] == ["stock_fundamentals"]


def test_a_passing_rubric_item_is_not_diagnosed():
    from financial_research_assistant import improve

    assert improve.diagnose(
        {"query": "q", "eval_type": "rubric", "criteria": _CRITERIA},
        score=1.0, tools=[], answer="fine", rubric=[],
    ) is None


def test_the_report_prints_the_failing_dimensions():
    from financial_research_assistant import improve

    d = {"kind": "derivation", "query": "was NVDA cheap in Jan 2025?",
         "eval_type": "rubric", "score": 0.33, "called": ["stock_fundamentals"],
         "failed": [{"dimension": "as_of", "score": 0.0,
                     "requirement": "the answer is about January 2025"}]}
    report = improve.render_report([d], "", "2026-08-07")
    assert "Derivation misses" in report
    assert "`as_of`" in report and "about January 2025" in report
    # It must be clear these are NOT auto-applied — the requirement text is prose
    # for a judge, not a prompt rule.
    assert "not auto-applied" in report


def test_derivation_misses_never_become_an_automatic_prompt_rule():
    from financial_research_assistant import improve

    d = {"kind": "derivation", "query": "q", "failed": [
        {"dimension": "as_of", "score": 0.0, "requirement": "be about January"}]}
    assert improve.propose_addendum([d]) == ""
