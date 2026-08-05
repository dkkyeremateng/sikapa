"""Session transcript persistence.

This scaffold is single-agent and conversational — there is no per-session
workspace folder, so a "session" is simply a JSONL transcript of (query,
answer) turns kept under the home dir. That record lets a session be listed
(``/sessions``, ``--list-sessions``) and its conversation replayed on resume
(``/resume NAME``, ``--resume NAME``).

Store location: ``~/.<package-name>/sessions/<session_id>.jsonl`` (derived from
the package, e.g. ``~/.financial-research-assistant/sessions/``). Set
``FINANCIAL_RESEARCH_SESSIONS_DIR`` to redirect it (tests point it at a temp dir
so they never touch the real home). Note: replaying a transcript restores the
*visible* conversation, not the model's in-process memory — the graph's
MemorySaver is always fresh per process.
"""

from __future__ import annotations

from typing import Any
import json
import os
import time
from pathlib import Path


def store_dir() -> Path:
    """Directory holding every session transcript (created on first write).

    Defaults to a per-app dir derived from the package name
    (``~/.<package-name>/sessions``), so a renamed agent keeps its own store
    instead of sharing the scaffold's; override with the
    ``FINANCIAL_RESEARCH_SESSIONS_DIR`` env var."""
    pkg = (__package__ or "financial_research_assistant").replace("_", "-")
    default = Path.home() / f".{pkg}" / "sessions"
    return Path(os.environ.get("FINANCIAL_RESEARCH_SESSIONS_DIR") or default)


def _transcript_path(session_id: str) -> Path:
    return store_dir() / f"{session_id}.jsonl"


def log_turn(session_id: str, query: str, answer: str) -> None:
    """Append one (query, answer) turn to the session's transcript."""
    d = store_dir()
    d.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "query": query, "answer": answer}
    with _transcript_path(session_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def read_transcript(session_id: str) -> list[dict[str, Any]]:
    """Every recorded turn, oldest first; tolerant of partial/corrupt lines."""
    path = _transcript_path(session_id)
    if not path.exists():
        return []
    turns: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return turns


def list_sessions() -> list[dict[str, Any]]:
    """Saved sessions under the store, most-recently-modified first.

    Each entry: ``{name, turns, mtime}`` where ``name`` is the session id
    (the transcript filename without its ``.jsonl`` suffix) — the value
    ``--session``/``--resume``/``/resume`` take."""
    base = store_dir()
    if not base.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(base.glob("*.jsonl")):
        if not p.is_file():
            continue
        out.append({
            "name": p.stem,
            "turns": len(read_transcript(p.stem)),
            "mtime": p.stat().st_mtime,
        })
    out.sort(key=lambda s: s["mtime"], reverse=True)
    return out


def session_exists(name: str) -> bool:
    return _transcript_path(name).exists()
