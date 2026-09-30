"""Reading a chat message's text, whatever shape the provider sent it in.

Its own module, with no imports from this package, so anything can use it — the
autonomous-call helper included — without pulling in the graph and, through it,
the whole tool catalog.
"""

from __future__ import annotations

from typing import Any


def message_text(message: Any) -> str:
    """The readable text of a chat message, whichever shape the provider used.

    A string is the common case. Anthropic-style providers instead report content
    as a LIST of typed blocks — text, thinking, tool_use, and whatever a future
    model adds — and ``str()`` on that list yields a Python repr: the answer
    wrapped in dict syntax, with the model's raw chain-of-thought and tool
    arguments alongside it. That repr then leaks wherever the text was headed (a
    subagent's findings, a compaction seed, the answer read back from state), so
    the text blocks are picked out and concatenated instead. Non-text blocks are
    dropped rather than summarized: they are the model's working, and every caller
    here wants what it SAID.
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(block.get("text", ""))
    return "".join(parts)
