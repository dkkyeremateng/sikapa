"""Feedback capture — the agent learns from your verdicts on its answers.

The third self-learning layer. When you rate a reply (``/good`` / ``/bad`` in the
TUI), ``record()`` stores that exchange in the long-term memory store as an
``exemplar`` (well-received) or ``avoid`` (poorly-received) memory. On a later,
similar question the adapter recalls the relevant ones and injects them as
**few-shot guidance** — "answers the user rated good, emulate them" and "answers
the user rated poor, avoid these" — so behavior drifts toward what you like
without any fine-tuning.

Like the fact memory (``memory.py``) and reflection (``reflection.py``) layers,
this is gated on ``MEMORY_BACKEND`` and is a no-op when memory is disabled.
Retrieval reuses the same deterministic keyword-overlap ranking, filtered to the
feedback kinds, so it stays inspectable (``--memory`` shows ``[exemplar]`` /
``[avoid]`` rows).
"""

from __future__ import annotations

from .memory import get_memory

_ANSWER_CAP = 700  # keep exemplars compact so the few-shot block stays affordable


def record(question: str, answer: str, good: bool, note: str = "") -> bool:
    """Store a rated exchange as an ``exemplar`` (good) or ``avoid`` (bad) memory.
    Returns False when memory is disabled, the exchange is empty, or it's a
    near-duplicate of one already stored. ``note`` optionally captures *why*."""
    mem = get_memory()
    if mem is None:
        return False
    q = " ".join((question or "").split())
    a = " ".join((answer or "").split())
    if not q and not a:
        return False
    if len(a) > _ANSWER_CAP:
        a = a[: _ANSWER_CAP - 1] + "…"
    text = f"Q: {q}\nA: {a}"
    if note:
        text += f"\nNote: {note}"
    return mem.save(text, kind="exemplar" if good else "avoid")


def _recall_kind(user_msg: str, kind: str, k: int) -> list[str]:
    mem = get_memory()
    if mem is None:
        return []
    try:
        entries = [e for e in mem.all() if e.get("kind") == kind]
    except Exception:
        return []
    # Route through the backend's ranking (keyword or semantic).
    return [e["text"] for e in mem.rank(user_msg, entries, k)]


def recall_feedback(user_msg: str, k: int = 2) -> tuple[list[str], list[str]]:
    """Return ``(exemplars, avoids)`` relevant to ``user_msg`` — the highest
    keyword-overlap rated exchanges. Empty lists when memory is off or nothing
    applies."""
    return _recall_kind(user_msg, "exemplar", k), _recall_kind(user_msg, "avoid", k)


def format_fewshot(exemplars: list[str], avoids: list[str]) -> str:
    """Build a few-shot preamble from recalled feedback, or "" if there's none.
    Framed so the model treats these as guidance on style/depth/format, not as new
    facts to repeat."""
    if not exemplars and not avoids:
        return ""
    blocks = []
    if exemplars:
        blocks.append(
            "Answers the user rated GOOD on similar past questions — emulate their "
            "depth, structure, and format (do NOT reuse their figures; recompute "
            "from tools):\n\n" + "\n\n".join(exemplars)
        )
    if avoids:
        blocks.append(
            "Answers the user rated POOR on similar past questions — avoid these "
            "mistakes:\n\n" + "\n\n".join(avoids)
        )
    return (
        "Guidance from earlier feedback on similar questions:\n\n"
        + "\n\n".join(blocks)
        + "\n\n---\n\n"
    )
