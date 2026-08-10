#!/usr/bin/env python3
"""A/B the working tree against a committed baseline on the eval dataset.

Built for the question "did trimming the system prompt cost us anything?", but
it is not specific to that change: it compares **whatever is in the working
tree** against **whatever is at a git ref**, so any prompt, tool-description, or
routing edit can be validated the same way before it lands.

How the two arms are isolated:

* **candidate** — the working tree, imported through the editable install.
* **baseline**  — a throwaway ``git worktree`` checked out at ``--ref``
  (default ``HEAD``), imported by pointing the child's ``PYTHONPATH`` at that
  checkout's ``src/``. PYTHONPATH is searched before the editable install's
  ``.pth`` entry, so the subprocess loads the baseline package instead.

Nothing about the baseline lives in the working tree — no frozen copy of the old
prompt to maintain, and no environment variable that could override the system
prompt in production. The comparison is exactly "these edits vs. that commit".

Confounds the harness controls for, so the two arms stay comparable:

* **long-term memory** — each arm gets its own ``MEMORY_DIR``. Otherwise the
  first arm's answers are saved and then recalled *into the second arm's
  prompt*, which is precisely the leak an A/B must not have.
* **the learned prompt addendum** — forced empty, so a previously-accepted
  addendum on this machine can't mask a regression in the base prompt.

Usage:
    python eval/ab_compare.py                      # working tree vs HEAD
    python eval/ab_compare.py --ref main           # vs another ref
    python eval/ab_compare.py --items 10           # cheap subset first
    python eval/ab_compare.py --repeat 3           # average out judge noise
    python eval/ab_compare.py --fake               # plumbing check, no network

Exit status is 1 if the candidate regresses by more than ``--tolerance`` (mean
score), so this can gate a merge the way ci_gate.py does.
"""

import argparse
import contextlib
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from evaluate import DATASET, load_dataset, run_all

REPO = Path(__file__).resolve().parent.parent


def _load_repo_env() -> None:
    """Load the repo's ``.env`` into *this* process.

    The agent itself runs in a subprocess and loads ``.env`` on its own, but
    ``llm_judge`` items are scored here in the parent — and an unconfigured
    parent silently scores every judged item 0.0 in both arms, which reads as a
    tie rather than as a broken scorer. Missing python-dotenv or a missing file
    is not fatal: an already-exported environment works fine.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO / ".env")


def _git(*args: str, cwd: Path | None = None) -> str:
    """Run a git command, returning stdout. Raises on failure."""
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd or REPO),
        capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip()


def _resolve(ref: str) -> str:
    try:
        return _git("rev-parse", "--short", ref)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"cannot resolve ref {ref!r}: {e.stderr.strip()}") from e


def _seed_worktree_config(worktree: Path) -> None:
    """Copy the repo's ``.env`` into the baseline worktree.

    ``main.py`` calls ``load_dotenv()`` with no argument, which resolves relative
    to *that module's own path* — so once the baseline package is imported from
    the worktree, dotenv looks for ``.env`` beside the worktree, not the repo.
    ``.env`` is gitignored and therefore absent from a fresh checkout, so without
    this the baseline arm starts with no API key and every item fails in under a
    second. Copying it also keeps the arms honest: both must run against the same
    endpoint, model, and keys for the comparison to mean anything.
    """
    src = REPO / ".env"
    if src.is_file():
        shutil.copy2(src, worktree / ".env")


def _preflight(label: str, env: dict, timeout: float = 90.0) -> None:
    """Run one trivial prompt through an arm and fail loudly if it doesn't answer.

    Learned the hard way: a misconfigured arm doesn't error, it just scores 0.00
    on every item in well under a second, and the run looks like a catastrophic
    regression instead of a broken harness. Checking one cheap invocation up
    front turns that into an immediate, readable failure.
    """
    from evaluate import run_item

    probe = {"query": "Reply with the single word OK.",
             "eval_type": "contains", "criteria": [{"answer": "", "weight": 1.0}]}
    result = run_item(probe, fake=False, timeout=timeout, env=env)
    answer = (result.get("_answer") or "").strip()
    if result["timed_out"] or not answer:
        raise SystemExit(
            f"\nPREFLIGHT FAILED for the {label} arm — it produced no answer "
            f"in {result['time_taken']:.1f}s.\n"
            f"The comparison would be meaningless, so nothing was run.\n"
            f"Check that the arm can reach the model endpoint and read its "
            f"credentials."
        )
    print(f"preflight {label}: OK ({result['time_taken']:.1f}s)")


def _arm_env(name: str, memory_root: Path, extra: dict | None = None) -> dict:
    """Environment overlay for one arm: an isolated memory store, no learned
    addendum, plus anything arm-specific (the baseline's PYTHONPATH)."""
    mem = memory_root / name
    mem.mkdir(parents=True, exist_ok=True)
    return {
        # Per-arm memory: without this, arm 1 writes exchanges that arm 2 then
        # recalls into its prompt, and the arms are no longer independent. Each
        # --repeat then gets a subdirectory of this one (see _repeat_env).
        "MEMORY_DIR": str(mem),
        # Operator-accepted learned guidance would otherwise ride along on both
        # arms and could paper over a base-prompt regression.
        "FINANCIAL_RESEARCH_PROMPT_ADDENDUM": "",
        **(extra or {}),
    }


def _mean(results: list[dict]) -> float:
    return sum(r["score"] for r in results) / len(results) if results else 0.0


def _repeat_env(env: dict, run: int) -> dict:
    """The arm's environment with a memory store private to this repeat.

    ``--repeat`` exists to average out judge noise, and averaging only helps if
    the runs are independent samples of it. Sharing one store across repeats
    makes run 2 recall run 1's answers into its prompt: the runs agree with each
    other more than the arm deserves, the spread shrinks, and the number that was
    supposed to expose noise hides it instead.
    """
    root = env.get("MEMORY_DIR")
    if not root:
        return dict(env)
    mem = Path(root) / f"run{run}"
    mem.mkdir(parents=True, exist_ok=True)
    return {**env, "MEMORY_DIR": str(mem)}


def _run_arm(
    label: str, items: list[dict], fake: bool, timeout: float,
    repeat: int, env: dict,
) -> list[float]:
    """Run the dataset ``repeat`` times for one arm; return the per-item mean
    score across repeats (index-aligned with ``items``)."""
    runs: list[list[dict]] = []
    for r in range(1, repeat + 1):
        suffix = f" (run {r}/{repeat})" if repeat > 1 else ""
        print(f"\n=== {label}{suffix} ===", flush=True)
        runs.append(run_all(items, fake, timeout, env=_repeat_env(env, r)))
    return [
        statistics.fmean([run[i]["score"] for run in runs])
        for i in range(len(items))
    ]


def _report(items: list[dict], base: list[float], cand: list[float],
            tolerance: float, base_ref: str) -> bool:
    """Print the comparison and return True when the candidate is acceptable."""
    base_mean, cand_mean = statistics.fmean(base), statistics.fmean(cand)
    delta = cand_mean - base_mean

    moved = [
        (c - b, items[i]["query"], b, c)
        for i, (b, c) in enumerate(zip(base, cand))
        if abs(c - b) > 1e-9
    ]
    moved.sort()

    print("\n" + "=" * 72)
    print(f"baseline ({base_ref})   mean {base_mean:.4f}")
    print(f"candidate (working tree) mean {cand_mean:.4f}")
    print(f"delta                        {delta:+.4f}")

    if moved:
        regressions = [m for m in moved if m[0] < 0]
        improvements = [m for m in moved if m[0] > 0]
        print(f"\n{len(moved)} item(s) moved "
              f"({len(regressions)} worse, {len(improvements)} better):")
        for d, query, b, c in moved:
            mark = "WORSE" if d < 0 else "better"
            print(f"  {mark:>6}  {b:.2f} -> {c:.2f} ({d:+.2f})  {query[:52]}")
    else:
        print("\nno per-item score changed")

    ok = delta >= -tolerance
    print("\n" + ("=" * 72))
    if ok:
        print(f"[PASS] within tolerance ({tolerance:.3f}) — the trim looks safe to keep")
    else:
        print(f"[FAIL] regressed by {-delta:.4f}, beyond tolerance {tolerance:.3f}")
        print("       inspect the WORSE items above: a dropped disambiguation cue")
        print("       usually shows up as the wrong tool being chosen.")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B the working tree against a git ref on the eval dataset"
    )
    parser.add_argument("--ref", default="HEAD",
                        help="baseline git ref to compare against (default HEAD)")
    parser.add_argument("--fake", action="store_true",
                        help="offline deterministic model (plumbing check only)")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--items", type=int, default=0,
                        help="run only the first N dataset items (0 = all)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run each arm N times and average (smooths judge noise)")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="allowed mean-score drop before failing (default 0.02)")
    args = parser.parse_args()

    # A full run is 90 agent invocations and can take the better part of an hour.
    # Redirected to a file, stdout is block-buffered and per-item progress stays
    # invisible until the process ends, which makes a long run indistinguishable
    # from a hung one. Line buffering makes `tail -f` work.
    with contextlib.suppress(AttributeError):  # not a TextIOWrapper (rare)
        sys.stdout.reconfigure(line_buffering=True)

    _load_repo_env()  # so llm_judge items can be scored in this process

    items = load_dataset(DATASET)
    if args.items > 0:
        items = items[: args.items]
    if not items:
        raise SystemExit("dataset is empty")

    base_sha = _resolve(args.ref)
    print(f"baseline  : {args.ref} ({base_sha})")
    print("candidate : working tree")
    print(f"dataset   : {len(items)} item(s) x {args.repeat} run(s) x 2 arms "
          f"= {len(items) * args.repeat * 2} agent invocations")

    tmp = Path(tempfile.mkdtemp(prefix="fra-ab-"))
    worktree = tmp / "baseline"
    memory_root = tmp / "memory"
    try:
        # A detached worktree at the baseline ref. This works with a dirty main
        # working tree, which is the whole point — the edits under test stay
        # exactly where they are.
        _git("worktree", "add", "--detach", str(worktree), base_sha)

        base_src = worktree / "src"
        if not base_src.is_dir():
            raise SystemExit(f"{base_sha} has no src/ directory — wrong ref?")
        _seed_worktree_config(worktree)

        base_env = _arm_env("baseline", memory_root, {"PYTHONPATH": str(base_src)})
        # No PYTHONPATH: the editable install resolves to the working tree.
        cand_env = _arm_env("candidate", memory_root)

        # Prove both arms can actually answer before spending a full run on them.
        if not args.fake:
            _preflight("baseline", base_env)
            _preflight("candidate", cand_env)

        base_scores = _run_arm(
            f"BASELINE {base_sha}", items, args.fake, args.timeout, args.repeat,
            base_env,
        )
        cand_scores = _run_arm(
            "CANDIDATE working tree", items, args.fake, args.timeout, args.repeat,
            cand_env,
        )
        ok = _report(items, base_scores, cand_scores, args.tolerance, base_sha)
        return 0 if ok else 1
    finally:
        # Remove the worktree through git so its administrative entry goes too;
        # fall back to a plain delete if git can't (already-removed, etc.).
        try:
            _git("worktree", "remove", "--force", str(worktree))
        except subprocess.CalledProcessError:
            pass
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(["git", "worktree", "prune"], cwd=str(REPO),
                       capture_output=True, check=False)


if __name__ == "__main__":
    sys.exit(main())
