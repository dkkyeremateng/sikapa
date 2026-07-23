#!/usr/bin/env python3
"""Fail CI when the eval score regresses below a threshold.

Runs the dataset through the same harness evaluate.py uses, then exits non-zero
if the mean score is under the floor. Wire it into a GitHub Action (see
.github/workflows/eval.yml) so a prompt/tool/model change that drops quality
blocks the merge.

Usage:
    python eval/ci_gate.py --min-score 0.8            # real model quality gate
    python eval/ci_gate.py --fake --min-score 0.5     # only for a fake-passable set

The floor also reads EVAL_MIN_SCORE if --min-score is omitted (default 0.8).
Note: --fake can't answer content evals, so a content dataset scores low
offline by design — run the gate live, and use `evaluate.py --fake` for the
offline plumbing check (see .github/workflows/eval.yml).
"""

import argparse
import os
import sys

from evaluate import DATASET, load_dataset, run_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate CI on the eval mean score")
    parser.add_argument("--fake", action="store_true",
                        help="run the offline fake model (plumbing gate)")
    parser.add_argument("--min-score", type=float,
                        default=float(os.environ.get("EVAL_MIN_SCORE", "0.8")),
                        help="minimum acceptable mean score (default 0.8)")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    items = load_dataset(DATASET)
    if not items:
        print("dataset is empty", file=sys.stderr)
        return 1

    results = run_all(items, args.fake, args.timeout)
    mean = sum(r["score"] for r in results) / len(results)
    ok = mean >= args.min_score
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] mean score {mean:.3f} (floor {args.min_score:.3f}) "
          f"over {len(results)} items")
    if not ok:
        worst = sorted(results, key=lambda r: r["score"])[:3]
        for r in worst:
            print(f"  low: {r['score']:.2f}  {r['query'][:60]}", file=sys.stderr)
        _emit_diagnosis(items, results, mean)
    return 0 if ok else 1


def _emit_diagnosis(items, results, mean) -> None:
    """On a gate failure, diagnose the misses and surface a proposed prompt
    addendum (reusing the run we already did — no re-run). Best-effort: never let
    diagnosis turn a clean FAIL into a crash."""
    try:
        from datetime import datetime, timezone

        from financial_research_assistant import improve

        diagnoses = [
            d for item, r in zip(items, results)
            if (d := improve.diagnose(item, r["score"], r.get("tools", []),
                                      r.get("_answer", ""), floor=1.0))
        ]
        if not diagnoses:
            return
        addendum = improve.propose_addendum(diagnoses)
        stamp = datetime.now(timezone.utc).isoformat()
        report = improve.render_report(diagnoses, addendum, stamp, mean=mean)
        improve.improve_dir().mkdir(parents=True, exist_ok=True)
        path = improve.improve_dir() / "ci-diagnosis-report.md"
        path.write_text(report, encoding="utf-8")
        print("\n--- eval diagnosis (regression) ---", file=sys.stderr)
        print(report, file=sys.stderr)
        print(f"\n(report also written to {path})", file=sys.stderr)
        if addendum:
            print("Try it:  python eval/evaluate.py --ab <(printf '%s' \"$ADDENDUM\") "
                  "--apply   # applies only if it beats baseline", file=sys.stderr)
    except Exception as e:  # diagnosis must never mask the real gate failure
        print(f"(diagnosis skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
