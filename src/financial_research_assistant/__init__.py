# pyright: reportImportCycles=false
# The remaining cycles through this file are the `from . import x` idiom, not a
# design flaw: that form imports the PACKAGE and then the submodule, so every one
# of them is an edge back to here. The alternative — `from .x import name` — binds
# the callee at import time, which would silently break the monkeypatching the
# test suite is built on (`monkeypatch.setattr(statements, "db_path", ...)` would
# no longer be seen by the module that already grabbed the name). Nothing here is
# imported at module scope, so none of it is a cycle at runtime.
"""Financial research assistant: LangGraph ReAct agent over read-only IBKR MCP
market-data tools + Textual TUI + headless CLI + eval harness.

``run_turn`` is exported lazily (PEP 562). Importing it eagerly here pulled the
whole graph — langchain, langgraph, yfinance, every tool module — into memory the
moment ANY submodule ran ``from . import auth``, since that statement executes
this file first. It also made a real cycle out of every such lookup: package ->
adapter -> graph -> catalog -> tools -> back into the half-initialised package.
Deferring it keeps ``from financial_research_assistant import run_turn`` working
while making a plain ``from . import auth`` cost nothing.
"""

from typing import Any

# Deliberately no `if TYPE_CHECKING: from .adapter import run_turn` either: a
# type-only import is still an edge to a checker, so it would keep reporting the
# very cycle this file exists to remove. The costs are that `run_turn` resolves
# through `__getattr__` as Any rather than its real signature — import it from
# `.adapter` directly where the signature matters — and that a checker can't see
# it in `__all__`, hence the ignore.
__all__ = ["ask", "run_turn"]  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    if name == "run_turn":
        from .adapter import run_turn

        return run_turn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


async def ask(message: str, session_id: str = "default", fake: bool = False) -> str:
    """Library facade: run one turn and return the final answer text."""
    from .adapter import run_turn

    async for ev in run_turn(message, session_id, fake=fake):
        if ev.kind == "final":
            return ev.text
        if ev.kind == "error":
            raise RuntimeError(ev.text)
    raise RuntimeError("stream ended without final event")
