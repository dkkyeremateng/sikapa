"""Integrity of the eval harness itself — the thing that says whether the agent
is any good.

Every bug covered here is a bug in the measurement rather than in the agent, and
they share a shape: each one moves a score for a reason that has nothing to do
with the answer. A judge reply parsed off by a digit, an item graded against
memory of a previous run, a dataset typo that aborts the run or scores as a free
mark — none of them are visible in the number they produce, which is exactly why
they need tests. A wrong score is worse than no score: it gets acted on.

Everything here is offline; the agent subprocess and the judge model are stubbed.
"""

import argparse
import importlib.util
import subprocess
import types
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def _load(name: str, monkeypatch=None):
    """Load one of the eval scripts by path (that directory is not a package)."""
    if monkeypatch is not None:  # ci_gate/ab_compare do `import evaluate`
        monkeypatch.syspath_prepend(str(EVAL_DIR))
    spec = importlib.util.spec_from_file_location(name, EVAL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ev():
    """A FRESH copy of the harness per test, so its once-per-process warning
    flag doesn't depend on which test ran first."""
    return _load("evaluate")


def _stub_judge(monkeypatch, reply: str) -> None:
    """Stand in for the judge model with a fixed reply."""
    class _Resp:
        content = reply

    class _LLM:
        def invoke(self, _messages):
            return _Resp()

    from financial_research_assistant import llm

    monkeypatch.setattr(llm, "_make_llm", lambda *a, **k: _LLM())


# --- reading a score out of the judge's reply -----------------------------------


def test_a_restated_scale_does_not_score_the_scale(ev):
    """The reported false FAILURE: reading the first number scored this 0.0, the
    lowest possible mark, for a reply that awarded 0.9."""
    assert ev._parse_score("On a scale of 0.0 to 1.0, I'd give this 0.9") == 0.9


def test_a_ratio_is_divided_out_not_clamped(ev):
    """The reported false PASS: reading the first number scored this 8, which
    clamped to a perfect 1.0 — the judge said 0.8."""
    assert ev._parse_score("8/10") == pytest.approx(0.8)
    assert ev._parse_score("Overall: 8 / 10 — decent but shallow") == pytest.approx(0.8)
    assert ev._parse_score("0.9 out of 1.0") == pytest.approx(0.9)


def test_a_percentage_is_read_as_a_fraction(ev):
    assert ev._parse_score("85%") == pytest.approx(0.85)
    assert ev._parse_score("I'd say 0.7 (70%)") == pytest.approx(0.7)


def test_a_bare_float_is_taken_at_face_value(ev):
    assert ev._parse_score("0.85") == 0.85
    assert ev._parse_score(" 1.0 \n") == 1.0
    assert ev._parse_score("0") == 0.0


def test_an_unreadable_reply_scores_zero(ev, capsys):
    """The safe direction. A reply with no rating in it is not evidence of
    quality, and a bare number off the 0-1 scale has nothing to anchor it."""
    assert ev._parse_score("I am unable to assess this output.") == 0.0
    assert ev._parse_score("") == 0.0
    assert ev._parse_score("7") == 0.0, "no scale to read a bare 7 against"
    assert "no readable" in capsys.readouterr().err


def test_llm_judge_uses_the_same_parse(monkeypatch, ev):
    _stub_judge(monkeypatch, "Thinking on a 0.0 to 1.0 scale, I give it 0.4.")
    assert ev._llm_judge("q", "out", [{"answer": "x"}], fake=False) == 0.4


# --- who is doing the grading ---------------------------------------------------


def test_an_unset_judge_model_warns_that_the_run_is_self_grading(monkeypatch, ev, capsys):
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    assert ev._judge_model_name() is None
    err = capsys.readouterr().err
    assert "EVAL_JUDGE_MODEL" in err and "self-grading" in err
    # Once per process, not once per item: a 50-item run must stay readable.
    ev._judge_model_name()
    assert capsys.readouterr().err == ""


def test_a_configured_judge_model_is_used_and_says_nothing(monkeypatch, ev, capsys):
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "some-other-model")
    assert ev._judge_model_name() == "some-other-model"
    assert capsys.readouterr().err == ""


# --- dataset shape --------------------------------------------------------------


_GOOD = {"query": "q", "eval_type": "contains", "criteria": [{"answer": "ok"}]}


def test_a_well_formed_item_validates(ev):
    assert ev.validate_item(_GOOD) is None
    assert ev.validate_item({"query": "q", "eval_type": "trajectory",
                             "criteria": [{"tool": "t", "weight": 2}]}) is None


@pytest.mark.parametrize("item, expected", [
    ({"query": "q", "eval_type": "contains", "criteria": [{"weight": 1.0}]}, "answer"),
    ({"query": "q", "eval_type": "contains", "criteria": [{"answer": " "}]}, "answer"),
    ({"query": "q", "eval_type": "trajectory", "criteria": [{"answer": "x"}]}, "tool"),
    ({"query": "q", "eval_type": "regex", "criteria": [{"pattern": "("}]}, "compile"),
    ({"query": "q", "eval_type": "vibes", "criteria": [{"answer": "x"}]}, "eval_type"),
    ({"query": "q", "eval_type": "contains", "criteria": []}, "non-empty"),
    ({"query": "", "eval_type": "contains", "criteria": [{"answer": "x"}]}, "query"),
    ({"query": "q", "eval_type": "contains",
      "criteria": [{"answer": "x", "weight": "two"}]}, "not a number"),
    ({"query": "q", "eval_type": "contains",
      "criteria": [{"answer": "x", "weight": -1}]}, "negative"),
])
def test_unscoreable_items_are_named(ev, item, expected):
    problem = ev.validate_item(item)
    assert problem and expected in problem


def test_the_shipped_dataset_is_scoreable(ev):
    """The check is worthless if it can't pass on the real dataset."""
    items = ev.load_dataset(ev.DATASET)
    assert [ev.validate_item(i) for i in items] == [None] * len(items)


def test_a_malformed_item_is_zeroed_before_any_agent_runs(ev, monkeypatch):
    """It used to raise KeyError from inside score() — after the subprocess had
    already run, and with every earlier result still unpersisted, so one typo
    threw away the whole run."""
    def boom(*_a, **_k):
        raise AssertionError("spent a subprocess on an item that cannot be scored")

    monkeypatch.setattr(ev, "run_item", boom)
    bad = {"query": "q", "eval_type": "contains", "criteria": [{"weight": 1.0}]}
    results = ev.run_all([bad], fake=True, timeout=1.0, verbose=False)
    assert results[0]["score"] == 0.0
    assert "not scored" in results[0]["note"] and "answer" in results[0]["note"]
    assert results[0]["query"] == "q" and results[0]["eval_type"] == "contains"


def test_the_rest_of_the_dataset_still_runs(ev, monkeypatch):
    """One bad line costs one item, not the run — and the zero stays IN the mean,
    since dropping it would raise the score of a run containing a broken item."""
    ran = []

    def fake_run_item(item, fake, timeout, env=None):
        ran.append(item["query"])
        return {"query": item["query"], "eval_type": item["eval_type"], "score": 1.0,
                "time_taken": 0.0, "timed_out": False, "tools": [], "_answer": "ok"}

    monkeypatch.setattr(ev, "run_item", fake_run_item)
    bad = {"query": "bad", "eval_type": "contains", "criteria": [{"weight": 1.0}]}
    results = ev.run_all([bad, {**_GOOD, "query": "good"}], fake=True, timeout=1.0,
                         verbose=False)
    assert ran == ["good"]
    assert [r["score"] for r in results] == [0.0, 1.0]
    assert ev._mean(results) == 0.5


def test_score_returns_zero_rather_than_raising_on_a_bad_criterion(ev):
    """Defence in depth behind validate_item — and note each of these scores as a
    MISS: an empty substring and an empty pattern both match anything."""
    assert ev.score({"query": "q", "eval_type": "contains",
                     "criteria": [{"weight": 1.0}]}, "anything", [], fake=True) == 0.0
    assert ev.score({"query": "q", "eval_type": "regex",
                     "criteria": [{"weight": 1.0}]}, "anything", [], fake=True) == 0.0
    assert ev.score({"query": "q", "eval_type": "trajectory",
                     "criteria": [{"weight": 1.0}]}, "", [{"name": "x"}],
                    fake=True) == 0.0
    assert ev.score({"query": "q", "eval_type": "vibes",
                     "criteria": [{"answer": "x"}]}, "anything", [], fake=True) == 0.0


# --- memory isolation -----------------------------------------------------------


def _stub_subprocess(monkeypatch, ev, stdout: str = "ok", seen: dict | None = None):
    """Replace the agent subprocess, recording the environment it was handed."""
    def fake_run(cmd, **kwargs):
        env = kwargs["env"]
        if seen is not None:
            seen["env"] = env
            memory_dir = env.get("MEMORY_DIR")
            seen["existed"] = bool(memory_dir) and Path(memory_dir).is_dir()
        return types.SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(ev, "subprocess", types.SimpleNamespace(
        run=fake_run, TimeoutExpired=subprocess.TimeoutExpired))


def test_an_item_runs_against_its_own_empty_memory_store(monkeypatch, ev, tmp_path):
    """The whole point of the gate is that it measures the agent. Inheriting the
    operator's store lets a previous run's stored answer be recalled into the
    prompt, so a regressed agent passes on its own history."""
    monkeypatch.setenv("MEMORY_BACKEND", "local")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path / "operator"))
    monkeypatch.setenv("MEMORY_USER", "operator")
    seen: dict = {}
    _stub_subprocess(monkeypatch, ev, seen=seen)

    ev.run_item(_GOOD, fake=True, timeout=5.0)

    child_dir = seen["env"]["MEMORY_DIR"]
    assert child_dir != str(tmp_path / "operator"), "the real store must not be used"
    assert seen["existed"], "the child needs a real directory to write into"
    assert not (tmp_path / "operator").exists(), "and must not create the real one"
    # The backend stays on, so recall and write-back are still exercised — the
    # eval covers the agent the operator actually runs, from an empty store.
    assert seen["env"]["MEMORY_BACKEND"] == "local"
    assert not Path(child_dir).exists(), "the throwaway store is cleaned up after"


def test_a_backend_that_ignores_memory_dir_is_switched_off(monkeypatch, ev):
    """mem0 keeps its own store, so redirecting MEMORY_DIR would isolate nothing
    and the eval would still read and pollute the operator's real memory."""
    monkeypatch.setenv("MEMORY_BACKEND", "mem0")
    seen: dict = {}
    _stub_subprocess(monkeypatch, ev, seen=seen)
    ev.run_item(_GOOD, fake=True, timeout=5.0)
    assert seen["env"]["MEMORY_BACKEND"] == ""


def test_a_caller_that_isolates_memory_itself_is_left_alone(monkeypatch, ev, tmp_path):
    """ab_compare hands each arm its own store; run_item must not second-guess it."""
    monkeypatch.setenv("MEMORY_BACKEND", "local")
    seen: dict = {}
    _stub_subprocess(monkeypatch, ev, seen=seen)
    arm = tmp_path / "arm-a"
    arm.mkdir()
    ev.run_item(_GOOD, fake=True, timeout=5.0, env={"MEMORY_DIR": str(arm)})
    assert seen["env"]["MEMORY_DIR"] == str(arm)
    assert arm.exists(), "the caller's directory is not deleted underneath it"


def test_ab_arms_never_share_a_memory_store(monkeypatch, ev, tmp_path):
    """Sharing one, the baseline answers every question and the candidate recalls
    those answers into its prompt — so the candidate clears --min-delta on the
    baseline's work and --apply installs an addendum that did nothing."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_EVAL_DIR", str(tmp_path))
    envs: list[dict] = []

    def fake_run_all(items, fake, timeout, env=None):
        envs.append(env)
        return [{"score": 1.0, "tools": [], "_answer": "a"}]

    monkeypatch.setattr(ev, "run_all", fake_run_all)
    args = argparse.Namespace(fake=True, timeout=1.0, min_delta=0.01, apply=False)
    ev._run_ab([_GOOD], args, "CANDIDATE ADDENDUM")

    assert len(envs) == 2
    assert envs[0]["MEMORY_DIR"] != envs[1]["MEMORY_DIR"]
    assert envs[0]["FINANCIAL_RESEARCH_PROMPT_ADDENDUM"] == ""
    assert envs[1]["FINANCIAL_RESEARCH_PROMPT_ADDENDUM"] == "CANDIDATE ADDENDUM"


def test_ab_compare_repeats_are_independent_samples(monkeypatch, tmp_path):
    """--repeat exists to measure judge noise; sharing a store across repeats
    makes run 2 recall run 1 and hides the very spread it was added to show."""
    ab = _load("ab_compare", monkeypatch)
    arm = {"MEMORY_DIR": str(tmp_path / "candidate"),
           "FINANCIAL_RESEARCH_PROMPT_ADDENDUM": ""}
    (tmp_path / "candidate").mkdir()
    envs = [ab._repeat_env(arm, r) for r in (1, 2, 3)]
    dirs = [e["MEMORY_DIR"] for e in envs]
    assert len(set(dirs)) == 3
    assert all(Path(d).is_dir() for d in dirs)
    assert all(e["FINANCIAL_RESEARCH_PROMPT_ADDENDUM"] == "" for e in envs)


# --- what --fake actually measures ----------------------------------------------


def test_fake_mode_flags_judged_items_as_ungraded(monkeypatch, ev):
    """Offline there is no judge, so any non-empty output takes full marks —
    including an error message. The flag is how the caller can say so."""
    seen: dict = {}
    _stub_subprocess(monkeypatch, ev, stdout="Error: tool unavailable", seen=seen)
    item = {"query": "q", "eval_type": "llm_judge", "criteria": [{"answer": "x"}]}
    record = ev.run_item(item, fake=True, timeout=5.0)
    assert record["score"] == 1.0 and record["auto_pass"] is True


def test_a_genuinely_graded_item_is_not_flagged(monkeypatch, ev):
    _stub_subprocess(monkeypatch, ev, stdout="ok")
    assert "auto_pass" not in ev.run_item(_GOOD, fake=True, timeout=5.0)
    detail: dict = {}
    _stub_judge(monkeypatch, "1.0")
    ev.score({"query": "q", "eval_type": "llm_judge", "criteria": [{"answer": "x"}]},
             "an answer", [], fake=False, detail=detail)
    assert "auto_pass" not in detail, "a real judge graded it"


def test_the_gate_reports_how_much_of_the_score_was_auto_passed(monkeypatch, capsys):
    """Otherwise a judged-heavy dataset greens `ci_gate --fake` while every answer
    is an error string, and the PASS reads as evidence of quality."""
    gate = _load("ci_gate", monkeypatch)
    results = [
        {"score": 1.0, "auto_pass": True, "query": "judged"},
        {"score": 1.0, "auto_pass": True, "query": "judged"},
        {"score": 0.0, "query": "graded"},
        {"score": 0.5, "query": "graded"},
    ]
    gate._report_auto_passes(results, mean=0.625)
    out = capsys.readouterr().out
    assert "2/4" in out and "0.500" in out, "the auto-passed share of the mean"
    assert "0.250" in out, "and the mean over what was actually graded"


def test_a_fully_graded_run_says_nothing(monkeypatch, capsys):
    gate = _load("ci_gate", monkeypatch)
    gate._report_auto_passes([{"score": 1.0}, {"score": 0.5}], mean=0.75)
    assert capsys.readouterr().out == ""


# --- diagnosis coverage ---------------------------------------------------------


def test_a_failing_trajectory_item_that_called_everything_is_still_reported():
    """An `ordered` item whose calls came in the wrong sequence has no missing
    tool to write a routing rule about — and used to vanish from BOTH sections of
    the report, leaving a mean that didn't match the failures listed under it."""
    from financial_research_assistant import improve

    item = {"query": "compare NVDA to AMD", "eval_type": "trajectory",
            "ordered": True, "criteria": [{"tool": "sec_financials"},
                                          {"tool": "stock_quote"}]}
    d = improve.diagnose(item, 0.0, ["stock_quote", "sec_financials"], "an answer",
                         floor=1.0)
    assert d is not None, "a failure must appear somewhere in the report"
    assert d["kind"] == "content", "nothing to route — report, don't auto-apply"
    assert d["criteria"] == ["sec_financials", "stock_quote"], "names what it wanted"
    report = improve.render_report([d], "", "2026-08-08")
    assert "compare NVDA to AMD" in report and "Content misses" in report
    assert improve.propose_addendum([d]) == "", "still never an automatic rule"


def test_a_passing_trajectory_item_is_still_silent():
    from financial_research_assistant import improve

    item = {"query": "q", "eval_type": "trajectory", "criteria": [{"tool": "a"}]}
    assert improve.diagnose(item, 1.0, ["a"], "", floor=1.0) is None


def test_a_missing_tool_is_still_a_routing_miss():
    from financial_research_assistant import improve

    item = {"query": "q", "eval_type": "trajectory", "criteria": [{"tool": "a"}]}
    d = improve.diagnose(item, 0.0, ["b"], "", floor=1.0)
    assert d["kind"] == "routing" and d["expected"] == ["a"]
