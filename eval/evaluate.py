#!/usr/bin/env python3
"""Generic eval harness (stdlib; llm_judge scores with the app's configured model).

Reads eval/dataset.jsonl — one JSON object per line:
    {"query": str,
     "eval_type": "contains" | "regex" | "trajectory" | "llm_judge",
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

For each item it runs the agent CLI in a subprocess (capturing stdout for the
answer and a --trace JSON file for the tool trajectory), scores it, appends a
result line to eval/results.jsonl, and prints a summary. Importable: ci_gate.py
calls run_all() to fail CI on a score regression.

Usage:
    python eval/evaluate.py            # real model (needs API config)
    python eval/evaluate.py --fake     # offline deterministic model
"""

import argparse
import json
import os
import re
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


def _llm_judge(query: str, output: str, criteria: list[dict], fake: bool) -> float:
    """Rate output against criteria with a judge model. Returns 0.0–1.0.

    Offline (``fake``): no network — return a deterministic 1.0 if the agent
    produced any output, so the harness self-test exercises the path without
    keys.

    Real mode: score with the app's **configured** model via the same
    ``graph._make_llm`` the agent uses — so it honors ``OPENAI_API_BASE`` (local
    servers, whose key is a dummy), ``MODEL_PROVIDER`` (Anthropic/Google/…), and
    the ``OPENAI_*`` contract, instead of requiring a bare ``OPENAI_API_KEY``.
    ``EVAL_JUDGE_MODEL`` overrides the judge model (e.g. a stronger cross-family
    one); unset, the configured agent model is used.
    """
    if fake:
        return 1.0 if output.strip() else 0.0
    try:
        from langchain_core.messages import HumanMessage

        from financial_research_assistant.graph import _make_llm
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
        llm = _make_llm(os.environ.get("EVAL_JUDGE_MODEL") or None)
        resp = llm.invoke([HumanMessage(content=prompt)])
    except Exception as e:  # model/endpoint error — score 0.0, don't crash the run
        print(f"  (llm_judge model call failed; scoring 0.0: {e})", file=sys.stderr)
        return 0.0
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    # Robust parse: pull the first number out of the reply, then clamp to [0, 1].
    m = re.search(r"\d+(?:\.\d+)?|\.\d+", content)
    if not m:
        return 0.0
    try:
        return max(0.0, min(1.0, float(m.group(0))))
    except ValueError:
        return 0.0


def score(item: dict, stdout: str, tools: list[dict], fake: bool) -> float:
    eval_type, criteria = item["eval_type"], item.get("criteria", [])
    if eval_type == "llm_judge":
        return _llm_judge(item["query"], stdout, criteria, fake)

    total = sum(c.get("weight", 1.0) for c in criteria) or 1.0
    if eval_type == "trajectory":
        called = [str(t.get("name", "")) for t in tools]
        if item.get("ordered"):
            expected = [c["tool"] for c in criteria]
            if not _ordered_subsequence(expected, called):
                return 0.0
        got = sum(c.get("weight", 1.0) for c in criteria if c["tool"] in called)
        return got / total

    got = 0.0
    for c in criteria:
        if eval_type == "contains":
            hit = c["answer"].lower() in stdout.lower()
        elif eval_type == "regex":
            hit = re.search(c["pattern"], stdout) is not None
        else:
            raise ValueError(f"unknown eval_type: {eval_type!r}")
        if hit:
            got += c.get("weight", 1.0)
    return got / total


def run_item(item: dict, fake: bool, timeout: float, env: dict | None = None) -> dict:
    """Run one eval item in a subprocess and score it. ``env`` overlays extra
    environment variables onto the child (used to A/B a prompt addendum). The
    returned dict carries the persisted fields plus the called-tool names and, in
    ``_answer``, the raw stdout — the underscore-prefixed key is stripped before
    results are written (it's for in-process diagnosis only)."""
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        trace_path = tf.name
    try:
        cmd = [sys.executable, "-m", f"{PKG}.main", "--prompt", item["query"],
               "--session", "eval", "--trace", trace_path]
        if fake:
            cmd.append("--fake")
        child_env = {**os.environ, **(env or {})}
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
        value = 0.0 if (timed_out or returncode != 0) else score(item, stdout, tools, fake)
    finally:
        Path(trace_path).unlink(missing_ok=True)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": item["query"],
        "eval_type": item["eval_type"],
        "score": round(value, 4),
        "time_taken": round(elapsed, 3),
        "timed_out": timed_out,
        "tools": [str(t.get("name", "")) for t in tools],
        "_answer": stdout,
    }


def run_all(
    items: list[dict], fake: bool, timeout: float, verbose: bool = True,
    env: dict | None = None,
) -> list[dict]:
    results = []
    for i, item in enumerate(items, 1):
        result = run_item(item, fake, timeout, env=env)
        results.append(result)
        if verbose:
            print(f"[{i}/{len(items)}] score={result['score']:.2f} "
                  f"time={result['time_taken']:.2f}s  {item['query'][:60]}")
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
        d = improve.diagnose(item, r["score"], r.get("tools", []), r.get("_answer", ""), floor)
        if d:
            diagnoses.append(d)
    return diagnoses, improve.propose_addendum(diagnoses)


def _run_ab(items, args, candidate: str) -> int:
    """A/B a candidate prompt addendum: baseline (no addendum) vs candidate, over
    the dataset. Writes a report; with --apply, installs the addendum only if it
    scored at least --min-delta better. Never applies on a tie or regression."""
    from financial_research_assistant import improve
    from financial_research_assistant.graph import prompt_addendum_path

    print("baseline run (no addendum)…", file=sys.stderr)
    base = run_all(items, args.fake, args.timeout, env={"FINANCIAL_RESEARCH_PROMPT_ADDENDUM": ""})
    print("candidate run (with addendum)…", file=sys.stderr)
    cand = run_all(items, args.fake, args.timeout, env={"FINANCIAL_RESEARCH_PROMPT_ADDENDUM": candidate})
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
