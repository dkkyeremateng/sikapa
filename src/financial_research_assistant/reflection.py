"""Self-critique / reflection — the agent learns from its own research.

After the full research pipeline produces a report, ``reflect()`` looks back at
what it just did and distills **lessons** — process notes for next time: which
sources came back thin or unavailable, what to check, how to present it better.
Those lessons are written into the same long-term store as ``kind="lesson"``
memories, and ``recall_lessons()`` pulls the relevant ones back in on the next
research run so the report explicitly addresses known gaps instead of repeating
them.

This is the second self-learning layer on top of the curated fact memory
(``memory.py``): fact memory learns *about the user*; reflection learns *about the
work*. Both are gated on ``MEMORY_BACKEND`` being set — with memory off there is
nowhere to store lessons, so reflection is a no-op and ``--research`` behaves
exactly as before.

Two lesson sources:
  * **Gap lessons (deterministic, free).** Scan the gathered sections for
    unavailable/thin data and record a lesson per gap. No model, always runs when
    memory is on — so it's fully testable offline.
  * **Critique lessons (LLM, one call).** When a model is configured (not fake),
    ask it for a few terse process lessons about the report it just wrote. Gated
    additionally by ``RESEARCH_REFLECT`` (default on; set to ``0`` to skip the
    extra call).
"""

from __future__ import annotations

import os

from .memory import get_memory

# Substrings (case-insensitive) that mark a section as thin/unavailable. These
# match the "no data" shapes the gather + underlying tools actually emit, and each
# only appears in a sentence ABOUT missing data — never inside a healthy result.
_GAP_MARKERS = (
    "unavailable",
    "not enough",
    "no data",
    "not available",
    "couldn't",
    "could not",
)

#: ``n/a`` is deliberately NOT one of them. It is how ``fundamentals.py`` renders
#: any single missing FIELD (``_fmt``/``_money``), so as a substring it fires on
#: perfectly good output: every company that pays no dividend, or whose sector is
#: blank, would permanently teach "the fundamentals data was unavailable" and have
#: that lesson injected into every later report on it. It means a gap only when it
#: IS the whole section — the tool returned the marker and nothing else.
_NA_TRIM = " \t\n.-—·:"


def _is_gap(text: str) -> bool:
    """Whether a gathered section came back unavailable or too thin."""
    low = (text or "").strip().lower()
    if any(m in low for m in _GAP_MARKERS):
        return True
    return low.strip(_NA_TRIM) == "n/a"


def _reflect_enabled() -> bool:
    """Whether the LLM critique call runs (default on; ``RESEARCH_REFLECT=0`` off).
    Gap lessons don't depend on this — they're free."""
    return (os.environ.get("RESEARCH_REFLECT") or "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _gap_lessons(subject: str, sections: list[tuple[str, str]]) -> list[str]:
    """One lesson per section whose data came back unavailable or too thin."""
    subj = subject.strip().upper()
    lessons = []
    for label, text in sections:
        if _is_gap(text):
            lessons.append(
                f"When researching {subj}, the '{label}' data was unavailable or "
                f"too thin — try an alternate source or note the gap explicitly."
            )
    return lessons


CRITIQUE_SYSTEM_PROMPT = (
    "You are reviewing a research report you just produced, to improve your own "
    "process. Return 1-3 TERSE lessons for next time — data that was thin or "
    "missing, a check you should have run, or a clearer way to present it. Focus "
    "on repeatable PROCESS lessons, not facts about the subject. One lesson per "
    "line, no numbering, no preamble. If the report is solid, return nothing."
)


async def _critique_lessons(
    subject: str,
    sections: list[tuple[str, str]],
    report: str,
    model: str | None,
) -> list[str]:
    """Ask the model for a few process lessons about the report it just wrote.
    Returns [] on any error — reflection must never break a research run."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from .llm import quick_llm

        findings = "\n\n".join(f"## {label}\n{text}" for label, text in sections)
        # Self-critique is a cheap, high-volume, low-reasoning task (distilling a few
        # terse process notes, not the report itself) — a natural fit for the 'quick'
        # model tier. Falls back to the primary model when QUICK_MODEL is unset, so
        # default behavior is unchanged.
        llm = quick_llm(model)
        resp = await llm.ainvoke([
            SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Subject: {subject}\n\nTool findings I had:\n{findings}\n\n"
                    f"Report I wrote:\n{report}\n\nLessons for next time:"
                )
            ),
        ])
        from .graph import message_text

        # A block-list reply stringified would split into "lessons" made of dict
        # syntax and the model's own chain-of-thought — and be stored as memory.
        content = message_text(resp)
    except Exception:
        return []  # a flaky/unconfigured model must not sink the pipeline
    subj = subject.strip().upper()
    lessons = []
    for line in content.splitlines():
        line = line.strip().lstrip("-*•0123456789. ").strip()
        if len(line) > 4:
            # Stamp with the subject so it recalls on the next run for this subject.
            lessons.append(f"From researching {subj}: {line}")
    return lessons[:3]


async def reflect(
    subject: str,
    sections: list[tuple[str, str]],
    report: str,
    model: str | None = None,
    fake: bool = False,
) -> list[str]:
    """Distill lessons from a just-produced report and persist them (as
    ``kind="lesson"`` memories) for future runs. No-op when long-term memory is
    disabled. Returns the lessons stored this run (deduped by the store)."""
    mem = get_memory()
    if mem is None:
        return []  # nowhere to learn to
    lessons = _gap_lessons(subject, sections)
    if not fake and _reflect_enabled():
        lessons += await _critique_lessons(subject, sections, report, model)
    stored = [lesson for lesson in lessons if mem.save(lesson, kind="lesson")]
    return stored


def recall_lessons(subject: str, k: int = 5) -> list[str]:
    """Prior lessons relevant to ``subject``, most-relevant first. Empty when
    memory is off or nothing applies. Filters the store to ``kind="lesson"`` and
    ranks by keyword overlap with the subject (lessons are stamped with their
    subject, so this reliably finds them)."""
    mem = get_memory()
    if mem is None:
        return []
    try:
        entries = [e for e in mem.all() if e.get("kind") == "lesson"]
    except Exception:
        return []
    # Route through the backend's ranking (keyword or semantic) so lessons recall
    # the same way facts and feedback do.
    return [e["text"] for e in mem.rank(subject, entries, k)]
