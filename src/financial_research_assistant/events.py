from dataclasses import dataclass


def format_duration(seconds: float, *, precise: bool = True) -> str:
    """Human-readable elapsed time. Under a minute it stays in seconds
    (``"0.4s"``, or ``"3s"`` when ``precise=False`` for live tick timers); at 60s
    and over it switches to minutes + seconds (``"1m 05s"``, ``"2m 34s"``)."""
    if seconds >= 60:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}m {secs:02d}s"
    return f"{seconds:.1f}s" if precise else f"{int(seconds)}s"


@dataclass
class AgentEvent:
    """One UI event. ``kind``/``text`` are the minimal contract every
    interface renders; the structured fields power rich TUIs (collapsible
    tool panels, per-call durations).

    kinds:
      status      app-level line
      reasoning   streamed model thinking; rendered in a collapsible 💭 panel
      tool_start  a tool call began (agent, tool, detail=args JSON)
      tool_end    a tool call finished (duration, ok, detail=result snippet)
      token       streamed answer text (unused in fake mode)
      alert       a user alert rule fired; surfaced out-of-band (TUI toast)
      usage       cumulative token counts for the turn (tokens_in/out/cache)
      final       the answer; always the last event of a successful run
      error       fatal; always the last event of a failed run
    """

    kind: str
    text: str
    agent: str = ""      # emitting agent; this scaffold is single-agent
    tool: str = ""       # tool name for tool_* events
    detail: str = ""     # args JSON (tool_start) / result snippet (tool_end)
    duration: float = 0.0
    ok: bool = True
    call_id: str = ""    # pairs tool_start with tool_end
    depth: int = 0       # always 0 here (single agent); indent hint for TUIs
    tokens_in: int = 0   # prompt tokens for the turn (usage events)
    tokens_out: int = 0  # completion tokens for the turn (usage events)
    tokens_cache: int = 0  # cache-read prompt tokens (when the provider reports it)
    # Cache-WRITE prompt tokens: the portion of this call's input that was written
    # into the prompt cache. Billed at a different rate from both fresh input and a
    # cache read (Anthropic charges ~1.25x input to write, ~0.1x to read), so it is
    # counted separately rather than folded into tokens_in. Like tokens_cache it is
    # a SUBSET of tokens_in, not an addition to it.
    tokens_cache_write: int = 0
    # Snapshot of the CURRENT context size (this turn's total input tokens), as
    # opposed to the additive per-call ``tokens_in`` delta. Drives the footer's
    # "ctx %" gauge, which must reflect how full the window is *now* — not the
    # cumulative session spend. -1 = "this event carries no context snapshot".
    # Set to the turn's input total on the final usage event, and to 0 on the
    # auto-compaction status event so the gauge drops the instant context shrinks.
    context_tokens: int = -1
