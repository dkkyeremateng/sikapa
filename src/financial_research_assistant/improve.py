"""Eval-driven self-improvement — the agent learns from its own regression scores.

The fourth self-learning layer, and the only one that touches *core* behavior
rather than adding retrieval. It closes the loop around the eval harness
(``eval/evaluate.py``):

    run dataset → diagnose failures → propose a prompt addendum →
    A/B measure it against the dataset → apply only if it scores better.

Safety-first for a financial agent (see references/memory-guide.md and the
security notes): a proposed change is **never** silently baked into the code. It's
emitted as a reversible DATA file (``graph.prompt_addendum_path()``) that a human
reviews; ``--apply`` writes it only when the A/B delta is positive, and deleting
the file reverts. The proposals themselves are DETERMINISTIC (derived from which
expected tool a trajectory eval shows was missed), so they're auditable — no model
rewrites the prompt.

This module is pure/importable (diagnosis rules, proposal, the apply decision);
the subprocess orchestration lives in ``eval/evaluate.py`` which calls in here.
"""

from __future__ import annotations

from typing import Any
import os
from pathlib import Path


def improve_dir() -> Path:
    """Directory eval improvement reports + candidate addenda are written to.
    ``FINANCIAL_RESEARCH_EVAL_DIR`` overrides the default."""
    raw = (os.environ.get("FINANCIAL_RESEARCH_EVAL_DIR") or "").strip()
    if raw:
        return Path(os.path.expandvars(raw)).expanduser()
    return Path.home() / ".financial-research-assistant" / "eval"


#: A rubric dimension at or below this scored badly enough to be worth naming.
#: Not 0.0: a dimension the judge gave 0.3 was assessed and largely failed, and
#: waiting for a clean zero would hide most real derivation faults.
_RUBRIC_FAIL = 0.5


def diagnose(
    item: dict[str, Any], score: float, tools: list[str], answer: str,
    floor: float = 1.0, rubric: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Diagnose one under-performing eval item, or return None if it met ``floor``.

    Trajectory misses are the actionable signal: the dataset says which tool the
    query should have driven, and the trace says which tools actually ran, so a
    miss names exactly what routing guidance to add. Content misses
    (contains/regex/llm_judge) are reported for human review but not turned into
    automatic prompt rules — too vague to apply safely. A failing trajectory item
    that called every expected tool is reported the same way: nothing is missing
    to route, but a failure that appears in no section of the report is worse than
    a vague one.

    A rubric miss sits between the two. It is more specific than a content miss —
    the breakdown names WHICH dimension of the derivation failed and carries that
    dimension's stated requirement — but the requirement is prose written for a
    judge, not a prompt rule, so it is reported rather than auto-applied. Turning
    "should state its discount-rate assumption" into a system-prompt line is a
    judgement call, and the whole design of this loop is that a human makes those.
    """
    if score >= floor:
        return None
    eval_type = item.get("eval_type", "")
    query = item.get("query", "")
    criteria = item.get("criteria", [])
    if eval_type == "rubric":
        failed = [
            {"dimension": d.get("dimension", ""), "score": d.get("score", 0.0),
             "requirement": d.get("requirement", "")}
            for d in (rubric or []) if float(d.get("score", 0.0)) <= _RUBRIC_FAIL
        ]
        return {
            "kind": "derivation",
            "query": query,
            "eval_type": eval_type,
            "score": round(score, 4),
            "failed": failed,
            "called": list(tools),
        }
    if eval_type == "trajectory":
        expected = [c.get("tool", "") for c in criteria if c.get("tool")]
        missing = [t for t in expected if t not in tools]
        if missing:
            return {
                "kind": "routing",
                "query": query,
                "eval_type": eval_type,
                "score": round(score, 4),
                "expected": missing,
                "called": list(tools),
            }
        # Every expected tool ran and the item still scored under the floor —
        # an `ordered` item whose calls came in the wrong sequence, typically.
        # There is no missing tool to write a routing rule about, but the item
        # DID fail, and returning None here dropped it from both sections of the
        # report: the run looked cleaner than it was, and the only trace of the
        # failure was a mean that didn't match the listed misses. It falls
        # through to the content shape, which reports without auto-applying.
    return {
        "kind": "content",
        "query": query,
        "eval_type": eval_type,
        "score": round(score, 4),
        "criteria": [
            c.get("answer") or c.get("pattern") or c.get("tool")
            or c.get("requirement") or ""
            for c in criteria
        ],
        "answer_snippet": (answer or "").strip()[:200],
    }


ADDENDUM_HEADER = "Additional tool-routing guidance (learned from evaluation runs):"


def propose_addendum(diagnoses: list[dict[str, Any]]) -> str:
    """Build a candidate system-prompt addendum from routing diagnoses — one hint
    per missed tool (deduped), each anchored to an example query. Returns "" when
    there's nothing actionable (no routing misses)."""
    lines: list[str] = []
    seen: set[str] = set()
    for d in diagnoses:
        if d.get("kind") != "routing":
            continue
        for tool in d.get("expected", []):
            if tool in seen:
                continue
            seen.add(tool)
            lines.append(
                f'For requests like "{d["query"]}", use the `{tool}` tool.'
            )
    if not lines:
        return ""
    return ADDENDUM_HEADER + "\n" + "\n".join(f"- {ln}" for ln in lines)


def ab_decision(
    baseline_mean: float, candidate_mean: float, min_delta: float = 0.01
) -> bool:
    """Whether to APPLY a candidate addendum: it must raise the mean score by at
    least ``min_delta``. A tie or regression is never applied."""
    return (candidate_mean - baseline_mean) >= min_delta


def render_report(
    diagnoses: list[dict[str, Any]],
    addendum: str,
    stamp: str,
    mean: float | None = None,
    ab: tuple[float, float] | None = None,
) -> str:
    """Render a human-review markdown report of the failures, the proposed
    addendum, and (when A/B'd) the measured deltas. ``stamp`` is supplied by the
    caller so this stays pure/deterministic."""
    routing = [d for d in diagnoses if d.get("kind") == "routing"]
    derivation = [d for d in diagnoses if d.get("kind") == "derivation"]
    content = [d for d in diagnoses if d.get("kind") == "content"]
    out = [f"# Eval improvement report — {stamp}", ""]
    if mean is not None:
        out.append(f"Mean score this run: **{mean:.3f}**")
        out.append("")
    if ab is not None:
        base, cand = ab
        verdict = "APPLY" if ab_decision(base, cand) else "keep baseline"
        out += [
            "## A/B of the proposed addendum",
            f"- baseline mean:  **{base:.3f}**",
            f"- candidate mean: **{cand:.3f}**",
            f"- delta: **{cand - base:+.3f}** → **{verdict}**",
            "",
        ]
    if routing:
        out.append("## Routing misses (actionable)")
        for d in routing:
            out.append(
                f"- score {d['score']:.2f} — expected `{'`, `'.join(d['expected'])}`, "
                f"called {d['called'] or '[]'} — {d['query']}"
            )
        out.append("")
    if derivation:
        out.append("## Derivation misses (review only, not auto-applied)")
        out.append("")
        out.append(
            "The answer may have been right; these are the dimensions of HOW it "
            "got there that scored poorly."
        )
        for d in derivation:
            out.append(f"- score {d['score']:.2f} — {d['query']}")
            for f in d.get("failed", []):
                out.append(
                    f"    - `{f['dimension']}` {f['score']:.2f} — {f['requirement']}"
                )
            out.append(f"    - called: {d['called'] or '[]'}")
        out.append("")
    if content:
        out.append("## Content misses (review only, not auto-applied)")
        for d in content:
            out.append(f"- score {d['score']:.2f} ({d['eval_type']}) — {d['query']}")
        out.append("")
    out.append("## Proposed prompt addendum")
    out.append("")
    out.append("```\n" + (addendum or "(nothing actionable)") + "\n```")
    out.append("")
    out.append(
        "_Review before applying. It applies as reversible data via "
        "`FINANCIAL_RESEARCH_PROMPT_ADDENDUM_FILE`; delete that file to revert._"
    )
    return "\n".join(out)
