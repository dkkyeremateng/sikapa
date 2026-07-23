"""Financial research assistant: LangGraph ReAct agent over read-only IBKR MCP
market-data tools + Textual TUI + headless CLI + eval harness."""

from .adapter import run_turn

__all__ = ["run_turn", "ask"]


async def ask(message: str, session_id: str = "default", fake: bool = False) -> str:
    """Library facade: run one turn and return the final answer text."""
    async for ev in run_turn(message, session_id, fake=fake):
        if ev.kind == "final":
            return ev.text
        if ev.kind == "error":
            raise RuntimeError(ev.text)
    raise RuntimeError("stream ended without final event")
