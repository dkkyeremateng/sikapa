#!/usr/bin/env python3
"""Generic eval harness (stdlib; llm_judge scores with the app's configured model).

Reads eval/dataset.jsonl — one JSON object per line:
    {"query": str,
     "eval_type": "contains" | "regex" | "trajectory" | "llm_judge" | "rubric",
     "criteria": [ ... ],           # shape depends on eval_type (see below)
     "ordered": bool}               # trajectory only, optional

Scoring strategies:
  contains    criteria: [{"answer": str, "weight": float}]   substring match
  regex       criteria: [{"pattern": str, "weight": float}]  re.search match
  trajectory  criteria: [{"tool": str, "weight": float}]     tool was called
              (set item "ordered": true to require the listed order as a
              subsequence of the actual calls)
  llm_judge   criteria: [{"answer": str, "weight": float}]   a stronger judge
              model rates how well the output meets each described criterion
  rubric      criteria: [{"dimension": str, "requirement": str, "weight": float}]
              grades the DERIVATION, not just the answer

`rubric` exists because the other four cannot see how an answer was reached.
`trajectory` checks a tool was called but not with what arguments or to what end;
`llm_judge` reads the final text and nothing else. A financial answer can land on
the right number from the wrong source, over the wrong period, or with its
assumptions unstated — and each of those is invisible to a single "is this answer
good" score, which a capable model can talk its way into. So `rubric` shows the
judge the TOOL TRAJECTORY alongside the answer and scores each dimension
separately, and the per-dimension breakdown is persisted to results.jsonl. That
breakdown is the point: it names which part of the reasoning failed, which is the
only form `improve.diagnose` can act on.

For each item it runs the agent CLI in a subprocess (capturing stdout for the
answer and a --trace JSON file for the tool trajectory), scores it, appends a
result line to eval/results.jsonl, and prints a summary. Importable: ci_gate.py
calls run_all() to fail CI on a score regression.

Every item runs against a throwaway long-term memory store (see ``run_item``), so
a score measures what the agent can work out now rather than what a previous run
left lying around for it to recall.

Usage:
    python eval/evaluate.py            # real model (needs API config)
    python eval/evaluate.py --fake     # offline deterministic model
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PKG = "financial_research_assistant"
EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "dataset.jsonl"
RESULTS = EVAL_DIR / "results.jsonl"


def load_dataset(path: Path) -> list[dict]:
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _ordered_subsequence(expected: list[str], actual: list[str]) -> bool:
    """True if every name in ``expected`` appears in ``actual`` in that order."""
    it = iter(actual)
    return all(any(name == a for a in it) for name in expected)


#: Set once the self-grading warning has been printed, so a 50-item run says it
#: loudly rather than 50 times (nobody reads the 40th copy).
_SELF_GRADING_WARNED = False


def _judge_model_name() -> str | None:
    """The judge model (``EVAL_JUDGE_MODEL``), or None to use the agent's own.

    Warns — once per process — when there is no override, because that case is
    **self-grading**: the same weights that wrote the answer decide whether the
    answer is good, and a model finds its own reasoning more convincing than a
    stranger's. The scores still track change run-to-run, which is what a
    regression gate needs, but they are not an independent measure of quality and
    should not be quoted as one. A cross-family judge is a one-variable fix, so
    the warning names the variable.
    """
    global _SELF_GRADING_WARNED
    name = os.environ.get("EVAL_JUDGE_MODEL") or None
    if name is None and not _SELF_GRADING_WARNED:
        _SELF_GRADING_WARNED = True
        print(
            "  WARNING: EVAL_JUDGE_MODEL is unset — judged items are graded by the "
            "SAME model being evaluated (self-grading). Scores are a relative signal "
            "only; set EVAL_JUDGE_MODEL to a different (ideally stronger, "
            "cross-family) model for an independent grade.",
            file=sys.stderr,
        )
    return name


#: One number in a judge reply together with whatever gives it its scale: a
#: trailing "%", or an "out of"/"/" denominator. Both show up whenever the judge
#: ignores "return only a float", and both change what the digits mean.
_SCORE_TOKEN = re.compile(
    r"(?P<num>\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*(?P<pct>%)|\s*(?:/|out of)\s*(?P<den>\d+(?:\.\d+)?|\.\d+))?",
    re.IGNORECASE,
)


def _parse_score(content: str) -> float:
    """Read a 0.0–1.0 rating out of a judge's reply; 0.0 when there isn't one.

    This parse *is* the item's score — nothing downstream can recover from it
    getting the wrong number — and reading the FIRST number in the reply is wrong
    in both directions. "On a scale of 0.0 to 1.0, I'd give this 0.9" scores
    **0.0**, failing an answer the judge passed; "8/10" scores **8**, which clamps
    to **1.0** and passes an answer the judge failed. A gate built on either is
    measuring the judge's prose style.

    So the LAST number wins — a reply that restates the scale states it before its
    verdict, never after — and it is read with its scale: "8/10" is 0.8 and "85%"
    is 0.85, which is what the judge meant in each case. A bare number outside
    [0, 1] has no scale to read it against and is not treated as a rating; that
    and a reply with no number at all score 0.0, the safe direction, since an
    unusable judgement is not evidence of quality.
    """
    matches = list(_SCORE_TOKEN.finditer(content))
    if matches:
        last = matches[-1]
        value = float(last.group("num"))
        if last.group("pct"):
            return max(0.0, min(1.0, value / 100.0))
        if last.group("den"):
            denominator = float(last.group("den"))
            if denominator > 0:
                return max(0.0, min(1.0, value / denominator))
        elif 0.0 <= value <= 1.0:
            return value
    print(f"  (judge reply carried no readable 0.0-1.0 score; scoring 0.0: "
          f"{content.strip()[:120]!r})", file=sys.stderr)
    return 0.0


def _llm_judge(query: str, output: str, criteria: list[dict], fake: bool) -> float:
    """Rate output against criteria with a judge model. Returns 0.0–1.0.

    Offline (``fake``): no network — return a deterministic 1.0 if the agent
    produced any output, so the harness self-test exercises the path without
    keys.

    Real mode: score with the app's **configured** model via the same
    ``llm._make_llm`` the agent uses — so it honors ``OPENAI_API_BASE`` (local
    servers, whose key is a dummy), ``MODEL_PROVIDER`` (Anthropic/Google/…), and
    the ``OPENAI_*`` contract, instead of requiring a bare ``OPENAI_API_KEY``.
    ``EVAL_JUDGE_MODEL`` overrides the judge model (e.g. a stronger cross-family
    one); unset, the configured agent model is used.
    """
    if fake:
        return 1.0 if output.strip() else 0.0
    try:
        from langchain_core.messages import HumanMessage

        from financial_research_assistant.llm import _make_llm
    except Exception as e:  # package/langchain not importable
        print(f"  (llm_judge needs the package + langchain; scoring 0.0: {e})", file=sys.stderr)
        return 0.0
    wants = "\n".join(f"- {c.get('answer', '')}" for c in criteria)
    prompt = (
        f"Query:\n{query}\n\nAgent output:\n{output}\n\n"
        f"Rate 0.0-1.0 how well the output satisfies ALL of these criteria:\n"
        f"{wants}\n\nReturn ONLY a float between 0.0 and 1.0."
    )
    try:
        llm = _make_llm(_judge_model_name())
        resp = llm.invoke([HumanMessage(content=prompt)])
    except Exception as e:  # model/endpoint error — score 0.0, don't crash the run
        print(f"  (llm_judge model call failed; scoring 0.0: {e})", file=sys.stderr)
        return 0.0
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return _parse_score(content)


#: Asked of the judge for a `rubric` item. Two things separate it from
#: `llm_judge`: it sees the TOOL TRAJECTORY, and it scores each dimension
#: separately.
#:
#: Both matter for the same reason. A financial answer can land on the right
#: number from the wrong source, over the wrong period, or with the assumptions
#: left unstated — and a single "how good is this answer" score hides all three
#: behind one number that a stronger model can talk its way into. Grading the
#: derivation per dimension says WHICH part failed, which is the only form the
#: improvement loop can act on.
_RUBRIC_PROMPT = """You are grading the DERIVATION behind a financial research \
answer, not just its conclusion.

QUERY:
{query}

TOOLS THE AGENT ACTUALLY CALLED (in order):
{trajectory}

THE AGENT'S ANSWER:
{answer}

Score EACH dimension below from 0.0 to 1.0:
{dimensions}

Grade what the evidence shows, not what sounds plausible. A correct-looking final
number derived from the wrong source, the wrong period, or an unstated assumption
scores LOW on that dimension even when the number happens to be right. If the
trajectory shows no tool call was made for something the answer asserts as fact,
that is not grounded.

Return ONLY a JSON object mapping each dimension key to its score, e.g.
{{"source": 1.0, "period": 0.5}}. No prose, no code fence."""


def _parse_rubric(raw: str, keys: list[str]) -> dict[str, float]:
    """Pull ``{key: score}`` out of the judge's reply, clamped to [0, 1].

    Missing keys score 0.0 rather than being skipped: a judge that omits a
    dimension has not assessed it, and silently dropping it would raise the
    weighted mean by shrinking the denominator — turning an unanswered dimension
    into a free pass.
    """
    found: dict[str, float] = {}
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
        except ValueError:
            data = {}
        if isinstance(data, dict):
            for key in keys:
                try:
                    found[key] = max(0.0, min(1.0, float(data[key])))
                except (KeyError, TypeError, ValueError):
                    continue
    return {key: found.get(key, 0.0) for key in keys}


def _rubric_dimensions(criteria: list[dict]) -> list[str]:
    """Dimension keys for a rubric item, guaranteed distinct.

    Two criteria carrying the same ``dimension`` is a dataset typo, but it breaks
    the scoring in the one direction an eval must never break: the weighted mean
    sums the numerator once per criterion while the weights collapse into a single
    dict entry, so a duplicated dimension is counted twice against a denominator
    that counted it once — and the item can score **above 1.0**, dragging the run
    mean up past a gate floor. It is also unanswerable as written, since the judge
    replies with a JSON object and an object cannot hold two values under one key.
    Suffixing the repeat keeps both requirements graded and keeps the result in
    [0, 1]; the warning says the dataset needs fixing.
    """
    keys: list[str] = []
    seen: dict[str, int] = {}
    for i, c in enumerate(criteria):
        raw = str(c.get("dimension") or f"d{i}")
        seen[raw] = n = seen.get(raw, 0) + 1
        if n == 1:
            keys.append(raw)
            continue
        key = f"{raw}__{n}"
        print(f"  (duplicate rubric dimension {raw!r} — grading the repeat as "
              f"{key!r}; fix the dataset)", file=sys.stderr)
        keys.append(key)
    return keys


def _rubric_judge(
    query: str, output: str, tools: list[dict], criteria: list[dict], fake: bool
) -> tuple[float, list[dict]]:
    """Grade the derivation dimension by dimension.

    Returns ``(weighted_score, breakdown)``. The breakdown is the point — it is
    what makes a rubric failure actionable instead of just low.
    """
    keys = _rubric_dimensions(criteria)
    weights = {k: max(0.0, float(c.get("weight", 1.0))) for k, c in zip(keys, criteria)}
    if fake:
        # Offline: no network. Award the dimensions only if the agent produced
        # anything at all, mirroring `_llm_judge`'s self-test behaviour.
        got = 1.0 if output.strip() else 0.0
        scores = {k: got for k in keys}
    else:
        try:
            from langchain_core.messages import HumanMessage

            from financial_research_assistant.llm import _make_llm
        except Exception as e:
            print(f"  (rubric needs the package + langchain; scoring 0.0: {e})",
                  file=sys.stderr)
            return 0.0, []
        called = [str(t.get("name", "")) for t in tools]
        prompt = _RUBRIC_PROMPT.format(
            query=query,
            trajectory=", ".join(called) or "(none)",
            answer=output.strip()[:6000],
            dimensions="\n".join(
                f'- "{k}": {c.get("requirement", "")}' for k, c in zip(keys, criteria)
            ),
        )
        try:
            llm = _make_llm(_judge_model_name())
            resp = llm.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            print(f"  (rubric model call failed; scoring 0.0: {e})", file=sys.stderr)
            return 0.0, []
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        scores = _parse_rubric(content, keys)

    total = sum(weights.values()) or 1.0
    value = sum(scores[k] * weights[k] for k in keys) / total
    breakdown = [
        {"dimension": k, "score": round(scores[k], 3), "weight": weights[k],
         "requirement": c.get("requirement", "")}
        for k, c in zip(keys, criteria)
    ]
    return value, breakdown


#: Strategies that need a model to grade them — and that `--fake` therefore
#: cannot grade at all.
_JUDGED_TYPES = ("llm_judge", "rubric")

#: The criterion field each strategy is actually scored on. A criterion missing
#: it isn't a hard criterion, it's an unscoreable one.
_CRITERION_FIELD = {
    "contains": "answer",
    "regex": "pattern",
    "trajectory": "tool",
    "llm_judge": "answer",
    "rubric": "requirement",
}


def validate_item(item: dict) -> str | None:
    """Say why a dataset item cannot be scored, or None when it can.

    Run over the whole dataset before the first subprocess starts, because the
    cost of finding out late is out of all proportion to the mistake: scoring
    reads the criterion fields, and a single typo'd key used to raise out of
    ``score`` *after* that item's agent run had already been paid for — with
    every earlier result still unwritten, since results are persisted only at the
    end. One bad line in the dataset threw away an entire live run and left
    nothing behind to look at. Checking the shape costs nothing and turns that
    into a named zero for the one item that deserves it.
    """
    if not isinstance(item, dict):
        return "item is not a JSON object"
    if not str(item.get("query") or "").strip():
        return "no 'query' to run"
    eval_type = item.get("eval_type")
    if eval_type not in _CRITERION_FIELD:
        return (f"unknown eval_type {eval_type!r} — expected one of "
                f"{', '.join(sorted(_CRITERION_FIELD))}")
    criteria = item.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        return f"{eval_type} needs a non-empty 'criteria' list"
    field = _CRITERION_FIELD[eval_type]
    for i, c in enumerate(criteria):
        if not isinstance(c, dict):
            return f"criteria[{i}] is not a JSON object"
        value = c.get(field)
        # Empty is rejected as firmly as missing: an empty substring and an empty
        # regex both match anything, so the typo would score as a free mark.
        if not isinstance(value, str) or not value.strip():
            return f"criteria[{i}] has no non-empty {field!r}"
        if eval_type == "regex":
            try:
                re.compile(value)
            except re.error as e:
                return f"criteria[{i}] pattern does not compile: {e}"
        weight = c.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            return f"criteria[{i}] weight {weight!r} is not a number"
        if weight < 0:
            return f"criteria[{i}] weight {weight} is negative"
    return None


def _unscoreable_result(item: dict, problem: str) -> dict:
    """A zero-scored record for an item the dataset describes too badly to run.

    Scored zero rather than skipped: dropping it would shrink the denominator of
    the run mean, so a broken item would quietly RAISE the score of the run that
    contains it. The note rides along into results.jsonl so the zero is
    identifiable as a dataset fault rather than an agent failure.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": str(item.get("query", "")) if isinstance(item, dict) else "",
        "eval_type": (item.get("eval_type", "") if isinstance(item, dict) else ""),
        "score": 0.0,
        "time_taken": 0.0,
        "timed_out": False,
        "tools": [],
        "note": f"not scored: {problem}",
        "_answer": "",
    }


def score(
    item: dict, stdout: str, tools: list[dict], fake: bool,
    detail: dict | None = None,
) -> float:
    """Score one item. ``detail``, when given, is populated with any structured
    breakdown the strategy produced (currently the rubric's per-dimension scores),
    so the caller can persist it without changing this function's return type.

    Criteria are read leniently here because the shape was already checked by
    ``validate_item`` before any subprocess ran: reaching this point with a
    malformed criterion would mean throwing away a completed agent run over a
    dataset typo, so a missing field scores as a miss rather than raising.
    """
    eval_type, criteria = item.get("eval_type", ""), item.get("criteria", [])
    if fake and eval_type in _JUDGED_TYPES and detail is not None and stdout.strip():
        # Offline, both judged strategies award full marks for ANY non-empty
        # output — including "Error: tool unavailable". Flagging it lets the
        # caller say how much of the mean is this rather than measured quality.
        detail["auto_pass"] = True
    if eval_type == "rubric":
        value, breakdown = _rubric_judge(item.get("query", ""), stdout, tools,
                                         criteria, fake)
        if detail is not None and breakdown:
            detail["rubric"] = breakdown
        return value
    if eval_type == "llm_judge":
        return _llm_judge(item.get("query", ""), stdout, criteria, fake)

    total = sum(c.get("weight", 1.0) for c in criteria) or 1.0
    if eval_type == "trajectory":
        called = [str(t.get("name", "")) for t in tools]
        if item.get("ordered"):
            expected = [c.get("tool", "") for c in criteria]
            if not _ordered_subsequence(expected, called):
                return 0.0
        got = sum(c.get("weight", 1.0) for c in criteria if c.get("tool") in called)
        return got / total

    got = 0.0
    for c in criteria:
        if eval_type == "contains":
            # As with the empty regex below: every string contains "", so a
            # criterion missing its answer must be a miss, never a free mark.
            want = str(c.get("answer") or "")
            hit = bool(want) and want.lower() in stdout.lower()
        elif eval_type == "regex":
            # An empty pattern matches everything, so a criterion missing its
            # pattern would score as a hit — a typo cannot be a free mark.
            pattern = str(c.get("pattern") or "")
            hit = bool(pattern) and re.search(pattern, stdout) is not None
        else:
            return 0.0
        if hit:
            got += c.get("weight", 1.0)
    return got / total


#: Environment keys that decide where long-term memory is read and written. A
#: caller that sets any of them has taken charge of isolation itself (ab_compare
#: gives each arm its own store), so ``run_item`` leaves its choice alone.
_MEMORY_ENV = ("MEMORY_BACKEND", "MEMORY_DIR", "MEMORY_USER")

#: Backends whose store follows ``MEMORY_DIR``. Anything else keeps its own
#: location and can only be isolated by switching it off.
_REDIRECTABLE_BACKENDS = ("", "local", "semantic")


def _hermetic_memory(memory_dir: str) -> dict:
    """Child-env overlay pointing long-term memory at a throwaway directory.

    An eval subprocess otherwise inherits the operator's real store, and that
    leaks in both directions. Reading it is the worse one: the agent recalls a
    previous run's stored answer straight into its prompt (graph.py injects
    recalled memories), so the gate can PASS an agent that has genuinely
    regressed — it is grading recall of an answer it was handed, not the
    reasoning the dataset asks about, and the more often the gate runs the more
    thoroughly it is fooled. Writing it is the ruder one: every gate run files
    the eval's questions and answers into a personal store meant for the
    operator's own portfolio.

    Redirecting beats disabling where it's possible, because the recall and
    write-back paths stay exercised — the eval keeps covering the agent the
    operator actually runs, just from an empty store each time.
    """
    overlay = {"MEMORY_DIR": memory_dir}
    if (os.environ.get("MEMORY_BACKEND") or "").lower() not in _REDIRECTABLE_BACKENDS:
        # A backend that keeps its own store (mem0) ignores MEMORY_DIR entirely,
        # so switching it off is the only isolation available.
        overlay["MEMORY_BACKEND"] = ""
    return overlay


def run_item(item: dict, fake: bool, timeout: float, env: dict | None = None) -> dict:
    """Run one eval item in a subprocess and score it. ``env`` overlays extra
    environment variables onto the child (used to A/B a prompt addendum). The
    returned dict carries the persisted fields plus the called-tool names and, in
    ``_answer``, the raw stdout — the underscore-prefixed key is stripped before
    results are written (it's for in-process diagnosis only).

    Unless ``env`` says otherwise, the child gets its own empty long-term memory
    store for the duration of the item (see ``_hermetic_memory``)."""
    overrides = env or {}
    memory_dir = (
        None if any(k in overrides for k in _MEMORY_ENV)
        else tempfile.mkdtemp(prefix="fra-eval-mem-")
    )
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        trace_path = tf.name
    try:
        cmd = [sys.executable, "-m", f"{PKG}.main", "--prompt", item["query"],
               "--session", "eval", "--trace", trace_path]
        if fake:
            cmd.append("--fake")
        child_env = {
            **os.environ,
            **(_hermetic_memory(memory_dir) if memory_dir else {}),
            **overrides,
        }
        start = time.perf_counter()
        # A single slow item (a heavyweight tool like research_report, or a stuck
        # endpoint) must score 0 for that item, not abort the whole run — so a
        # timeout is caught here and treated as a failure like a non-zero exit.
        stdout = ""
        timed_out = False
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=child_env
            )
            stdout = proc.stdout
            returncode = proc.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            returncode = None
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        elapsed = time.perf_counter() - start
        tools: list[dict] = []
        try:
            with open(trace_path) as f:
                tools = json.load(f).get("tools", [])
        except (OSError, json.JSONDecodeError):
            pass
        detail: dict = {}
        value = (
            0.0 if (timed_out or returncode != 0)
            else score(item, stdout, tools, fake, detail=detail)
        )
    finally:
        Path(trace_path).unlink(missing_ok=True)
        if memory_dir:
            shutil.rmtree(memory_dir, ignore_errors=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": item["query"],
        "eval_type": item["eval_type"],
        "score": round(value, 4),
        "time_taken": round(elapsed, 3),
        "timed_out": timed_out,
        "tools": [str(t.get("name", "")) for t in tools],
        "_answer": stdout,
    }
    # Persisted (no underscore): the per-dimension scores are the diagnostic value
    # of a rubric run, and a mean alone would throw them away.
    record.update(detail)
    return record


def run_all(
    items: list[dict], fake: bool, timeout: float, verbose: bool = True,
    env: dict | None = None,
) -> list[dict]:
    """Run and score every item. Items the dataset describes unscoreably are
    reported and zeroed here, before the first (expensive) subprocess starts."""
    problems = {i: p for i, item in enumerate(items) if (p := validate_item(item))}
    if problems:
        # Always announced, verbose or not: a dataset fault silently scoring 0
        # would read as an agent regression, which is the wrong thing to go and
        # debug.
        print(f"{len(problems)} dataset item(s) cannot be scored and count as 0:",
              file=sys.stderr)
        for i, problem in problems.items():
            print(f"  item {i + 1}: {problem}", file=sys.stderr)
    results = []
    for i, item in enumerate(items, 1):
        problem = problems.get(i - 1)
        result = (_unscoreable_result(item, problem) if problem
                  else run_item(item, fake, timeout, env=env))
        results.append(result)
        if verbose:
            print(f"[{i}/{len(items)}] score={result['score']:.2f} "
                  f"time={result['time_taken']:.2f}s  {result['query'][:60]}")
    return results


def _mean(results: list[dict]) -> float:
    return sum(r["score"] for r in results) / len(results) if results else 0.0


def _persist(results: list[dict]) -> None:
    """Append results to RESULTS, dropping in-process-only underscore keys."""
    with RESULTS.open("a") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")


def _diagnose_run(items, results, floor):
    """Diagnose under-performing items and propose an addendum (pure package
    logic). Returns (diagnoses, addendum)."""
    from financial_research_assistant import improve

    diagnoses = []
    for item, r in zip(items, results):
        d = improve.diagnose(
            item, r["score"], r.get("tools", []), r.get("_answer", ""), floor,
            rubric=r.get("rubric"),
        )
        if d:
            diagnoses.append(d)
    return diagnoses, improve.propose_addendum(diagnoses)


def _ab_arm_env(name: str, root: Path, addendum: str) -> dict:
    """Environment overlay for one A/B arm: its own memory store, plus exactly the
    addendum under test.

    The private store is what makes the two arms independent samples rather than
    a relay. Sharing one, the baseline arm answers every question and writes what
    it learned; the candidate arm then recalls those answers into its prompt and
    scores higher for reasons that have nothing to do with the addendum. With
    ``--apply`` that difference installs a candidate that changed nothing. (This
    mirrors ``ab_compare._arm_env``, which isolates its arms for the same reason.)
    """
    mem = root / name
    mem.mkdir(parents=True, exist_ok=True)
    return {
        "MEMORY_DIR": str(mem),
        "FINANCIAL_RESEARCH_PROMPT_ADDENDUM": addendum,
    }


def _run_ab(items, args, candidate: str) -> int:
    """A/B a candidate prompt addendum: baseline (no addendum) vs candidate, over
    the dataset. Writes a report; with --apply, installs the addendum only if it
    scored at least --min-delta better. Never applies on a tie or regression."""
    from financial_research_assistant import improve
    from financial_research_assistant.graph import prompt_addendum_path

    memory_root = Path(tempfile.mkdtemp(prefix="fra-ab-mem-"))
    try:
        print("baseline run (no addendum)…", file=sys.stderr)
        base = run_all(items, args.fake, args.timeout,
                       env=_ab_arm_env("baseline", memory_root, ""))
        print("candidate run (with addendum)…", file=sys.stderr)
        cand = run_all(items, args.fake, args.timeout,
                       env=_ab_arm_env("candidate", memory_root, candidate))
    finally:
        shutil.rmtree(memory_root, ignore_errors=True)
    bmean, cmean = _mean(base), _mean(cand)
    stamp = datetime.now(timezone.utc).isoformat()
    report = improve.render_report([], candidate, stamp, mean=cmean, ab=(bmean, cmean))
    improve.improve_dir().mkdir(parents=True, exist_ok=True)
    report_path = improve.improve_dir() / "ab-report.md"
    report_path.write_text(report, encoding="utf-8")
    apply = improve.ab_decision(bmean, cmean, args.min_delta)
    print(f"\nbaseline {bmean:.3f} → candidate {cmean:.3f} "
          f"(delta {cmean - bmean:+.3f}); report {report_path}")
    if args.apply and apply:
        dest = prompt_addendum_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(candidate, encoding="utf-8")
        print(f"APPLIED — wrote addendum to {dest} (delete it to revert)")
    elif args.apply:
        print("not applied: candidate did not beat baseline by --min-delta")
    else:
        print("measure-only (pass --apply to install when improved)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score the agent against eval/dataset.jsonl"
    )
    parser.add_argument("--fake", action="store_true",
                        help="run the agent with its offline fake model")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="per-item subprocess timeout in seconds")
    parser.add_argument("--diagnose", action="store_true",
                        help="diagnose failures and write a proposed prompt addendum "
                             "for review (does not change behavior)")
    parser.add_argument("--floor", type=float, default=1.0,
                        help="with --diagnose: score below which an item is a failure")
    parser.add_argument("--ab", metavar="FILE",
                        help="A/B a candidate addendum FILE (baseline vs candidate "
                             "over the dataset) and write a report")
    parser.add_argument("--apply", action="store_true",
                        help="with --ab: install the addendum if it beats baseline "
                             "by --min-delta (reversible; deletes to revert)")
    parser.add_argument("--min-delta", type=float, default=0.01,
                        help="with --ab --apply: minimum mean-score gain to install")
    args = parser.parse_args()

    items = load_dataset(args.dataset)
    if not items:
        print("dataset is empty", file=sys.stderr)
        return 1

    if args.ab:
        candidate = Path(args.ab).read_text(encoding="utf-8").strip()
        return _run_ab(items, args, candidate)

    results = run_all(items, args.fake, args.timeout)
    _persist(results)
    mean = _mean(results)
    print(f"\n{len(results)} items | mean score {mean:.2f} | "
          f"results appended to {RESULTS}")

    if args.diagnose:
        diagnoses, addendum = _diagnose_run(items, results, args.floor)
        from financial_research_assistant import improve

        stamp = datetime.now(timezone.utc).isoformat()
        report = improve.render_report(diagnoses, addendum, stamp, mean=mean)
        improve.improve_dir().mkdir(parents=True, exist_ok=True)
        report_path = improve.improve_dir() / "diagnosis-report.md"
        report_path.write_text(report, encoding="utf-8")
        cand_path = improve.improve_dir() / "candidate-addendum.txt"
        cand_path.write_text(addendum, encoding="utf-8")
        print(f"diagnosed {len(diagnoses)} failure(s); report {report_path}")
        if addendum:
            print(f"proposed addendum → {cand_path}\n  A/B it: "
                  f"python eval/evaluate.py --ab {cand_path} [--apply]")
        else:
            print("no actionable routing gaps found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
