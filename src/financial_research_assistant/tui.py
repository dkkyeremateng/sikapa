"""Textual TUI — full agent console. Launched by ``code-review-agent`` with no
arguments.

Layout (modeled on modern agent consoles):
- Config bar (top): endpoint · model · mode — visible at a glance.
- Activity log: each tool call is a collapsible panel titled
  ``Agent · tool ✓ (1.2s)``, expandable to the call's args and result
  snippet; the model's reasoning shows as a collapsible 💭 panel (toggle with
  ``/toggle_thinking``). Status/final/error render as plain lines.
- Command palette: type ``/`` for slash commands (↑/↓ select, Tab/→ fill,
  Enter run). ↑/↓ with the palette closed walk prompt history.
- Status bar (above the input): ``model · provider · ctx N% of CAP · in/out tok · $cost
  · thinking on/off`` plus a live spinner + elapsed timer while a turn runs and a
  queued-message count. The ``thinking`` indicator updates live on toggle.
- Footer: key hints.

Features: live token streaming into the answer bubble (with a run duration;
re-renders are throttled to ~10 Hz so long replies stay cheap), prompt history
(↑/↓), Esc cancels the running turn, a message queue (Enter while busy queues;
Alt+↑ restores), Ctrl+O/Ctrl+T collapse all tool/thinking panels (click a
collapsed panel to expand it), PgUp/PgDn·Shift+↑/↓·wheel scroll the log —
streaming auto-follow pauses while you're scrolled up and resumes at the bottom —
modal ``/resume`` and ``/model`` pickers, and ``/copy``, ``/export``,
``/hotkeys``, ``/theme`` commands. This scaffold is single-agent
and conversational (no workspace), so ``/new`` starts a *new conversation* — a
fresh session id with cleared memory — and sessions persist a transcript so
``/sessions`` lists them and ``/resume NAME`` replays a past conversation.

``--once "query"`` auto-submits one query and exits after the final event.
All text renders through rich.text.Text — never markup — so bracketed
content ([2026-07-09], [/path]) can never crash or vanish.
"""

from __future__ import annotations

import html as _html
import json
import os
import time
from typing import cast
from uuid import uuid4

from rich.text import Text
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Collapsible, Footer, Header, Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from . import alerts, sessions
from .adapter import compact_session, reset_session, run_turn
from .events import AgentEvent, format_duration
from .pricing import context_cap as _context_cap
from .pricing import context_pct as _context_pct
from .pricing import cost_usd as _cost
from .tracing import traced

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Single source of truth: the palette, /help, and dispatch all read this.
_COMMANDS: list[tuple[str, str]] = [
    ("/new", "start a new conversation"),
    ("/compact", "summarize older turns to shrink the context"),
    ("/clear", "clear the transcript view"),
    ("/sessions", "list saved sessions"),
    ("/resume", "resume a saved session (/resume NAME, or pick from a list)"),
    ("/model", "switch the model for the next query (/model NAME, or a picker; /model default clears)"),
    ("/config", "show endpoint, model, mode"),
    ("/import", "import a broker statement — IBKR CSV or OFX/QFX — into the store (/import PATH)"),
    ("/toggle_thinking", "enable/disable the reasoning trace (💭 panels)"),
    ("/thinking", "turn reasoning on or off (/thinking on|off, or toggle)"),
    ("/copy", "copy the last answer to the clipboard"),
    ("/good", "rate the last answer good — learn from it (/good [note]; needs memory on)"),
    ("/bad", "rate the last answer poor — avoid it (/bad [note]; needs memory on)"),
    ("/memory", "review or prune long-term memory (/memory, or /memory forget TEXT)"),
    ("/export", "export the transcript (/export NAME.html|NAME.jsonl, or a full path)"),
    ("/hotkeys", "list keyboard shortcuts"),
    ("/theme", "toggle light / dark theme"),
    ("/help", "list commands"),
    ("/quit", "exit"),
]

_HOTKEYS: list[tuple[str, str]] = [
    ("Enter", "submit query (queues if a turn is running)"),
    ("Esc", "cancel the running turn"),
    ("↑ / ↓", "prompt history (palette navigation when it is open)"),
    ("Tab / →", "accept the palette completion"),
    ("Alt+↑", "restore the last queued message to the input"),
    ("PgUp / PgDn", "scroll the log a page"),
    ("Shift+↑ / ↓", "scroll the log a line"),
    ("Ctrl+Home/End", "scroll to top / bottom"),
    ("Ctrl+O", "collapse / expand all tool panels"),
    ("Ctrl+T", "collapse / expand all thinking panels"),
    ("⌥/Shift + drag", "select text natively (mouse captured); copy with Ctrl/Cmd+C"),
    ("F2", "release the mouse for plain-drag selection (pauses scroll; F2 restores)"),
    ("Ctrl+Q", "quit  (Ctrl+C shows a 'Press Ctrl+Q to quit' reminder, never exits)"),
]

_READY = "Ready. Type a message, or / for commands (/help lists them)."

# Oldest log widgets are trimmed past this count so week-long sessions don't
# accumulate thousands of live widgets (the transcript file keeps full history).
_LOG_CAP = 600


def _provider_label() -> str:
    """Human label for the active provider: the explicit MODEL_PROVIDER, else
    'openai-compatible' when a custom base URL is set, else 'openai'."""
    prov = (os.environ.get("MODEL_PROVIDER") or "").strip().lower()
    if prov and prov != "openai":
        return prov
    return "openai-compatible" if os.environ.get("OPENAI_API_BASE") else "openai"


def _model_provider(fake: bool, override: str | None = None) -> tuple[str, str]:
    if fake:
        return "scripted-fake", "offline"
    from .adapter import _resolved_model

    return _resolved_model(override), _provider_label()


def _config_line(fake: bool, override: str | None = None) -> Text:
    from .adapter import _resolved_model

    if fake:
        endpoint, model = "offline", "scripted-fake"
    else:
        endpoint = os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1"
        model = _resolved_model(override)
    t = Text()
    t.append(" Endpoint: ", style="dim")
    t.append(endpoint, style="bold blue")
    t.append("  Model: ", style="dim")
    t.append(model, style="bold green")
    t.append("  Mode: ", style="dim")
    t.append("FAKE" if fake else "live", style="bold red" if fake else "bold green")
    return t


def _fmt_tokens(n: int) -> str:
    """Compact context-window size: 128000 -> '128k', 1047576 -> '1M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
    if n >= 1_000:
        return f"{round(n / 1_000)}k"
    return str(n)


def _footer_line(
    model: str,
    provider: str,
    tokens_in: int,
    tokens_out: int,
    tokens_cache: int = 0,
    ctx_pct: int | None = None,
    thinking: bool | None = None,
) -> Text:
    t = Text()
    t.append(model, style="bold")
    t.append(" · ", style="dim")
    t.append(provider)
    if ctx_pct is not None:
        t.append(" · ", style="dim")
        t.append(f"ctx {ctx_pct}% of {_fmt_tokens(_context_cap(model))}")
    t.append(" · ", style="dim")
    t.append(f"in {tokens_in} / out {tokens_out} tok", style="dim")
    if tokens_cache:
        t.append(" · ", style="dim")
        t.append(f"cache {tokens_cache}", style="dim")
    cost = _cost(model, tokens_in, tokens_out, tokens_cache)
    if cost is not None:
        t.append(" · ", style="dim")
        t.append(f"${cost:.4f}", style="dim")
    if thinking is not None:
        t.append(" · ", style="dim")
        t.append("thinking ", style="dim")
        t.append("on" if thinking else "off", style="green" if thinking else "red")
    return t


def _export_html(turns: list[dict], title: str) -> str:
    rows = []
    for turn in turns:
        q = _html.escape(turn.get("query", ""))
        a = _html.escape(turn.get("answer", ""))
        ts = _html.escape(turn.get("ts", ""))
        rows.append(
            f'<section class="turn"><p class="q"><b>you</b> {q}'
            f'<span class="ts">{ts}</span></p><pre class="a">{a}</pre></section>'
        )
    body = "\n".join(rows)
    t = _html.escape(title)
    return (
        "<!doctype html><meta charset=utf-8><title>" + t + "</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:48rem;margin:2rem "
        "auto;padding:0 1rem;line-height:1.5}.q{color:#0066cc}.ts{color:#999;"
        "font-size:.8em;margin-left:.5rem}.a{background:#f4f4f4;padding:.75rem;"
        "border-radius:.4rem;white-space:pre-wrap;overflow-x:auto}</style>"
        f"<h1>{t}</h1>{body}"
    )


class CommandInput(Input):
    """Input with a slash-command palette (↑/↓ select, Tab/→ fill, Enter run)
    and prompt history (↑/↓ when the palette is closed)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_index: int = -1

    def _open_palette(self) -> OptionList | None:
        try:
            ol = self.app.query_one("#command-list", OptionList)
        except Exception:
            return None
        return ol if ol.display else None

    def on_key(self, event: events.Key) -> None:
        if event.key == "alt+up":
            app = cast("AgentApp", self.app)
            if getattr(app, "_queue", None):
                self.value = app._queue.pop()
                app._render_statusbar()
                app._line("↩ restored queued message to the input", "dim")
                event.stop()
                event.prevent_default()
            return
        ol = self._open_palette()
        if ol is not None and ol.option_count:
            if event.key == "up":
                ol.highlighted = max(0, (ol.highlighted or 0) - 1)
                event.stop()
                event.prevent_default()
            elif event.key == "down":
                cur = ol.highlighted
                ol.highlighted = 0 if cur is None else min(ol.option_count - 1, cur + 1)
                event.stop()
                event.prevent_default()
            elif event.key in ("tab", "right"):
                self._accept(ol)  # fill; stop() keeps Tab from moving focus
                event.stop()
                event.prevent_default()
            # enter: let Input submit naturally; on_input_submitted expands the
            # highlighted command from the palette (see AgentApp).
            return
        if event.key == "up":
            if self._history and self._history_index == -1:
                self._history_index = len(self._history) - 1
                self.value = self._history[self._history_index]
            elif self._history and self._history_index > 0:
                self._history_index -= 1
                self.value = self._history[self._history_index]
            else:
                return
            event.stop()
            event.prevent_default()
        elif event.key == "down":
            if self._history_index != -1 and self._history_index < len(self._history) - 1:
                self._history_index += 1
                self.value = self._history[self._history_index]
            elif self._history_index == len(self._history) - 1:
                self._history_index = -1
                self.value = ""
            else:
                return
            event.stop()
            event.prevent_default()

    def _accept(self, ol: OptionList) -> None:
        if ol.highlighted is None:
            return
        cmd = ol.get_option_at_index(ol.highlighted).id or ""
        self.value = cmd
        self.cursor_position = len(cmd)

    def record_history(self, val: str) -> None:
        if val and (not self._history or self._history[-1] != val):
            self._history.append(val)
        self._history_index = -1


class ToolPanel(Collapsible):
    """One tool call rendered like the reference console: the title carries
    ``🛠 [agent] tool <spinner> → ✅/❌ (Xs)`` with a live braille spinner while
    the call runs, and the body holds Arguments + Result logs. Self-times so the
    duration ticks without extra events. RichLogs use ``markup=False`` so
    bracketed args/results can never raise ``MarkupError``."""

    DOTS = _SPINNER

    def __init__(self, ev: AgentEvent) -> None:
        self.tool_name = ev.tool or "tool"
        self.agent = ev.agent or "Agent"
        self._done = ev.kind == "tool_end"
        self._frame = 0
        self._start = time.monotonic()
        self._pending_args = ev.detail if ev.kind == "tool_start" else ""
        self.args_log = RichLog(wrap=True, markup=False, highlight=False, min_width=20, classes="tool-log")
        self.args_log.border_title = "Arguments"
        self.result_log = RichLog(wrap=True, markup=False, highlight=False, min_width=20, classes="tool-log")
        self.result_log.border_title = "Result"
        css = "subagent-tool" if ev.depth > 0 else "orchestrator-tool"
        super().__init__(self.args_log, self.result_log, title=self._fmt_title(self.DOTS[0]), collapsed=True, classes=css)
        # A blank line above and below sets the tool call apart from the
        # surrounding answer text (depth adds left indent for sub-agent tools).
        self.styles.margin = (1, 2, 1, 2 + ev.depth * 4)

    def _fmt_title(self, status: str) -> str:
        # Plain 🛠 (no VARIATION SELECTOR-16): its Rich cell width (1) matches how
        # terminals paint it, so the title has no width drift. Adding VS16 makes
        # Rich reserve 2 cells while the terminal paints 1, leaving a stray
        # uninitialized cell (a black rectangle) at the end of the line. The \[
        # keeps Rich markup from parsing the label bracket as a tag.
        return f"🛠 \\[{self.agent}] {self.tool_name} {status}"

    def on_mount(self) -> None:
        if self._pending_args:
            self.args_log.write(self._pending_args)
        self._timer = self.set_interval(0.1, self._tick)

    def _tick(self) -> None:
        if self._done:
            self._timer.stop()
            return
        self._frame = (self._frame + 1) % len(self.DOTS)
        self.title = self._fmt_title(
            f"{self.DOTS[self._frame]} ({format_duration(time.monotonic() - self._start)})"
        )

    def finish(self, ev: AgentEvent) -> None:
        self._done = True
        timer = getattr(self, "_timer", None)
        if timer is not None:
            timer.stop()
        self.result_log.clear()
        self.result_log.write(ev.detail or "(no result)")
        icon = "✅" if ev.ok else "❌"
        self.title = self._fmt_title(f"{icon} ({format_duration(ev.duration)})")

    def abort(self, note: str = "interrupted") -> None:
        """Stop the spinner and mark the call as never completed — the turn
        errored or was cancelled before its result arrived. Without this the
        self-timer keeps ticking forever on an orphaned panel."""
        if self._done:
            return
        self._done = True
        timer = getattr(self, "_timer", None)
        if timer is not None:
            timer.stop()
        dt = time.monotonic() - self._start
        self.result_log.write(f"({note})")
        self.title = self._fmt_title(f"⚠ {note} ({format_duration(dt)})")

    def on_click(self) -> None:
        # Clicking anywhere on a collapsed panel expands it (the title click
        # already toggles; this makes the whole row a hit target).
        if self.collapsed:
            self.collapsed = False


class ThinkingPanel(Collapsible):
    """A collapsible 💭 panel holding one agent's reasoning, styled like a
    ToolPanel (accent bar, margin, padded title) but with a grey bar so thinking
    still reads apart from tool calls."""

    def __init__(self, ev: AgentEvent) -> None:
        agent = ev.agent or "Agent"
        body = Static(Text(ev.text or "(no detail)", style="italic dim"), classes="thinking-body")
        # Reasoning arrives as one complete event (not streamed), so the thought
        # is done the moment its panel mounts — the ✓ says so, vs. the live
        # "● Agent ⠋" processing spinner that means the turn is still working.
        # The think tool carries a duration (like a tool call); native/fake
        # reasoning has none, so the (Xs) is shown only when it's tracked.
        dur = f" ({format_duration(ev.duration)})" if ev.duration else ""
        super().__init__(
            body, title=f"💭 \\[{agent}] thinking ✅{dur}", collapsed=True, classes="thinking-panel"
        )
        self.styles.margin = (1, 2, 1, 2 + ev.depth * 4)

    def on_click(self) -> None:
        if self.collapsed:
            self.collapsed = False


class UserMessageWidget(Static):
    """Full-width transcript bar for the user's message (click to copy), styled
    like the research console: a bold ``User (Click to Copy):`` label inline
    before the query, on a flat dark background. Rendered as ``Text`` (never
    markup) so a pasted path or diff with brackets can't crash the render."""

    def __init__(self, query: str) -> None:
        self._query = query
        super().__init__(
            Text.assemble(("User (Click to Copy): ", "bold"), query),
            classes="user-bubble",
        )
        self.tooltip = "Click to copy"

    def on_click(self) -> None:
        try:
            self.app.copy_to_clipboard(self._query)
            self.app.notify("copied prompt to clipboard")
        except Exception as e:  # noqa: BLE001 - clipboard is best-effort
            self.app.notify(f"copy failed: {e}", severity="error")


class _FlushCodeBlock(CodeBlock):
    """A fenced code block rendered with **no background box** and no wrapping —
    so a monospace table sits flush on the app's own background instead of inside a
    themed rectangle, with its column alignment preserved (wrapping would mangle
    it)."""

    def __rich_console__(self, console, options):  # noqa: D401
        yield Syntax(
            str(self.text).rstrip(),
            self.lexer_name,
            theme=self.theme,
            background_color="default",  # transparent → the app background shows
            word_wrap=False,
            padding=0,
        )


class _ReplyMarkdown(Markdown):
    """Markdown for agent replies: same as Rich's, but fenced/indented code
    blocks render without the default background box (see ``_FlushCodeBlock``)."""

    elements = {
        **Markdown.elements,
        "fence": _FlushCodeBlock,
        "code_block": _FlushCodeBlock,
    }


class AgentMessageWidget(Static):
    """Full-width transcript entry for an agent reply, styled like the research
    console: a bold ``Author:`` header with an inline ``(Xs)`` run duration,
    then the reply. Header and body live in one Markdown block so a single
    newline renders as a soft break — the reply starts on the header line.

    A ``live=True`` bubble (token streaming) throttles its re-renders: re-parsing
    the whole reply as Markdown on every token is O(n²) in the answer length, so
    appends only mark the bubble dirty and a ~10 Hz timer flushes them. ``text``
    always updates immediately; only the visual render is deferred. The app calls
    ``finalize()`` when the bubble is complete (turn ended, or output moved past
    it to a tool/thinking panel), which flushes and stops the timer."""

    def __init__(self, text: str = "", author: str = "Agent", *, live: bool = False) -> None:
        super().__init__(classes="agent-msg")
        self.author = author
        self.text = text
        self.duration: float | None = None
        self._live = live
        self._dirty = False
        self._render_timer = None
        self._update_content()

    def on_mount(self) -> None:
        if self._live:
            self._render_timer = self.set_interval(0.1, self._flush)

    def on_unmount(self) -> None:
        if self._render_timer is not None:
            self._render_timer.stop()
            self._render_timer = None

    def _update_content(self) -> None:
        dur = f" ({format_duration(self.duration)})" if self.duration is not None else ""
        head = f"**{self.author}:**{dur}"
        self.update(_ReplyMarkdown(f"{head}\n{self.text}" if self.text else head))

    def _flush(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self._update_content()
        # The flush is what grows the bubble (appends are deferred), so following
        # the stream happens here — and only while the user is at the bottom.
        parent = self.parent
        if getattr(self.app, "_follow", True) and isinstance(parent, VerticalScroll):
            parent.scroll_end(animate=False)

    def append_text(self, new_text: str) -> None:
        self.text += new_text
        if self._render_timer is not None:
            self._dirty = True  # rendered by the ~10 Hz throttle timer
        else:
            self._update_content()

    def finalize(self) -> None:
        """Flush any pending text and stop the render throttle — the bubble
        won't grow further."""
        if self._render_timer is not None:
            self._render_timer.stop()
            self._render_timer = None
        self._dirty = False
        self._update_content()

    def set_duration(self, seconds: float) -> None:
        self.duration = seconds
        self.finalize()


class ProcessingWidget(Static):
    """Inline ``● Agent <spinner> (Xs)`` shown while the agent works before its
    first visible output; replaced by that output (or an error mark)."""

    DOTS = _SPINNER

    def __init__(self, agent: str = "Agent") -> None:
        super().__init__(classes="agent-msg")
        self.agent = agent
        self._frame = 0
        self._start = time.monotonic()

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._tick)
        self._tick()

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(self.DOTS)
        dt = time.monotonic() - self._start
        self.update(Text.assemble(
            (f"{self.agent}: ", "bold"),
            (f"{self.DOTS[self._frame]} ({format_duration(dt)})", "dim"),
        ))

    def stop(self) -> None:
        timer = getattr(self, "_timer", None)
        if timer is not None:
            timer.stop()
        self.remove()

    def mark_error(self, msg: str) -> None:
        timer = getattr(self, "_timer", None)
        if timer is not None:
            timer.stop()
        self.update(Text.assemble(
            (f"{self.agent}: ", "bold"), (f"✖ {msg}", "red"),
        ))


class SelectScreen(ModalScreen[str | None]):
    """Modal list picker: ↑/↓ + Enter select, Esc cancel; dismiss(id|None)."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str, items: list[tuple[str, str]]) -> None:
        super().__init__()
        self._title = title
        self._items = items

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Static(self._title, id="modal-title")
            yield OptionList(id="modal-list")

    def on_mount(self) -> None:
        ol = self.query_one("#modal-list", OptionList)
        for id_, label in self._items:
            ol.add_option(Option(label, id=id_))
        if self._items:
            ol.highlighted = 0
        ol.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ModelScreen(ModalScreen[str | None]):
    """Modal single-field prompt to set the model name; Esc cancels."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, current: str) -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Static(
                "Set model for the next query (Enter apply, Esc cancel; "
                "'default' clears the override)",
                id="modal-title",
            )
            yield Input(value=self._current, id="modal-input")

    def on_mount(self) -> None:
        self.query_one("#modal-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)


class AgentApp(App):
    TITLE = "Financial Research Assistant"
    SUB_TITLE = "IBKR market data · research-only"
    # Textual's own palette (Ctrl+P) would swallow our command keys; ours is the
    # only palette. The mouse is captured by default so scroll-wheel and clicks
    # work. Native text selection then uses the terminal's modifier drag (⌥ on
    # macOS, Shift on many Linux terminals) — OR press F2 to release the mouse so
    # a plain drag selects (scroll pauses until F2 again). In-app selection stays
    # off; copying is always the terminal's own (Ctrl/Cmd+C).
    ENABLE_COMMAND_PALETTE = False
    ALLOW_SELECT = False
    BINDINGS = [
        # Quit is Ctrl+Q (inherited from App with priority). We deliberately do
        # NOT bind Ctrl+C to quit: Textual's built-in `ctrl+c → help_quit` then
        # stays active, so a reflexive Ctrl+C shows "Press Ctrl+Q to quit"
        # instead of closing the agent.
        ("ctrl+q", "quit", "Quit"),
        ("f2", "toggle_mouse", "Select mode"),
        ("escape", "cancel_turn", "Cancel"),
        ("ctrl+o", "toggle_tools", "Tools"),
        ("ctrl+t", "toggle_thinking_collapse", "Thinking"),
        # Scroll the log without stealing ↑/↓ (those walk prompt history).
        ("pageup", "scroll_log_page_up", "Scroll up"),
        ("pagedown", "scroll_log_page_down", "Scroll down"),
        ("shift+up", "scroll_log_up", "Scroll up"),
        ("shift+down", "scroll_log_down", "Scroll down"),
        ("ctrl+home", "scroll_log_home", "Top"),
        ("ctrl+end", "scroll_log_end", "Bottom"),
    ]
    CSS = """
    #config { dock: top; height: 1; background: $surface; color: $text; }
    #log { height: 1fr; border: round $accent; padding: 0 1; }
    /* Theme every scrollbar to blend with the accent border instead of the
       default black track + blue thumb: thin, dark track, accent-colored grip. */
    #log, #command-list, RichLog {
        scrollbar-size-vertical: 1;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        scrollbar-color: $accent 40%;
        scrollbar-color-hover: $accent 70%;
        scrollbar-color-active: $accent;
    }
    #command-list { height: auto; max-height: 12; border: round $accent; }
    #prompt-row { dock: bottom; height: auto; padding: 0 1; background: $surface; border: round $accent; }
    #prompt-marker { width: 2; height: 3; color: $accent; text-style: bold; content-align: center middle; }
    #prompt { border: none; background: transparent; padding: 1 0; height: 3; }
    #prompt:focus { border: none; }
    /* Flat full-width transcript bar (research-console style), not a chat pill.
       Vertical margin on BOTH sides (not just top) so a turn's bubble/reply is
       always separated from an adjacent status line — which has no margin of its
       own — instead of butting right against it. */
    .user-bubble { width: 1fr; height: auto; padding: 1 2; margin: 1 0 1 0;
                   color: white; background: #303030; }
    .user-bubble:hover { background: #3a3a3a; color: #aaffaa; }
    .agent-msg { margin: 1 0 1 0; padding: 0 1; color: $text; }
    .subagent-status { height: auto; }
    .orchestrator-tool { border-left: vkey blue; }
    .subagent-tool { border-left: vkey purple; }
    /* Pad the tool-call title: horizontal so it isn't flush against the accent
       bar, vertical so the row is taller with the label centered. */
    .orchestrator-tool > CollapsibleTitle,
    .subagent-tool > CollapsibleTitle,
    .thinking-panel > CollapsibleTitle { padding: 1 1; }
    /* Thinking panels get the same treatment as tool panels — accent bar, margin
       (set inline), padded title, same title weight — with only a grey bar to
       tell reasoning apart from a tool call. */
    .thinking-panel { border-left: vkey #6b6b6b; }
    .thinking-body { margin: 0 1; height: auto; }
    /* Taller than a typical tool-result panel: a code review's read_source
       results are whole files, so give expanded panels more room before the
       inner scrollbar kicks in. overflow-y: auto (RichLog defaults to `scroll`)
       shows the vertical scrollbar ONLY when content overflows — otherwise the
       always-on thumb fills the track and reads as a black square at the right
       edge; overflow-x: hidden drops the horizontal bar (content wraps). */
    RichLog.tool-log { height: auto; max-height: 32; margin: 0 1; border: solid $panel;
                       overflow-y: auto; overflow-x: hidden; }
    /* No dock: it flows just below the log and above the input. (Docking it
       bottom made it collide with the Footer on the same row, hiding it — and
       with it the model + token counts.) Height 3 + middle align gives the
       status line room to breathe. */
    #statusbar { height: 3; background: $surface; color: $text; padding: 0 1;
                 content-align: left middle; }
    /* Inset the key-hints so they aren't flush against the terminal edges. */
    Footer { padding: 0 2; }
    Collapsible { border: none; padding: 0; }
    CollapsibleTitle { padding: 0; }
    /* No blue focus box on panel titles — they are click/keyboard targets. */
    CollapsibleTitle:focus { background: transparent; text-style: none; }
    SelectScreen, ModelScreen { align: center middle; }
    #modal { width: 70%; max-width: 100; height: auto; max-height: 80%;
             border: round $accent; background: $surface; padding: 1 2; }
    #modal-title { height: auto; margin-bottom: 1; color: $text; }
    #modal-list { height: auto; max-height: 20; }
    """

    def __init__(
        self,
        fake: bool = False,
        once: str | None = None,
        session_id: str = "tui",
        think: bool = True,
    ):
        super().__init__()
        self.fake = fake
        self.once = once
        self.session_id = session_id
        self.model_override: str | None = None
        self.show_thinking = think
        self._exit_code = 0
        self._pending: dict[str, ToolPanel] = {}
        self._tok_in = 0    # cumulative session input tokens (drives $cost)
        self._tok_out = 0
        self._tok_cache = 0
        self._turn_in = 0   # this turn's input total — the current context size (drives ctx%)
        self._ctx_pct = 0
        self._busy = False
        self._turn_started = 0.0
        self._spin = 0
        self._queue: list[str] = []
        self._last_answer = ""
        self._last_user = ""  # last submitted query, for /good and /bad feedback
        self._tools_collapsed = True
        self._think_collapsed = True
        self._turn_worker = None
        self._compacting = False  # /compact runs under _busy but on its own worker
        self._processing: ProcessingWidget | None = None
        self._answer: AgentMessageWidget | None = None  # live-streamed reply
        # Auto-scroll follows the stream only while the user is at the bottom of
        # the log; scrolling up pauses it, scrolling back down resumes it.
        self._follow = True

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(_config_line(self.fake, self.model_override), id="config")
        yield VerticalScroll(id="log")
        palette = OptionList(id="command-list")
        palette.display = False
        palette.can_focus = False
        yield palette
        model, provider = _model_provider(self.fake, self.model_override)
        yield Static(
            _footer_line(model, provider, 0, 0, thinking=self.show_thinking),
            id="statusbar",
        )
        with Horizontal(id="prompt-row"):
            yield Static("❯", id="prompt-marker")
            yield CommandInput(
                placeholder="Ask the agent… (/ for commands; ↑ recalls history)",
                id="prompt",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(CommandInput).focus()
        self._render_statusbar()
        self._line(_READY, "dim")
        self.set_interval(0.1, self._tick)
        if sessions.read_transcript(self.session_id):
            self._replay_transcript(self.session_id)
        if self.once:
            self._start_turn(self.once)

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        """Keep the prompt focused on the main screen. Clicking a collapsible
        panel (or otherwise moving focus) would otherwise steal focus from the
        input; bounce it straight back so typing always lands in the prompt —
        text selection is terminal-native (F2), so it never touches focus.
        Modals (ModelScreen / SelectScreen) manage their own focus, so we leave
        those alone."""
        self.call_after_refresh(self._refocus_prompt)

    def _refocus_prompt(self) -> None:
        if isinstance(self.screen, (ModelScreen, SelectScreen)):
            return  # a modal is active; don't fight its own input/list focus
        prompt = self.query_one(CommandInput)
        if self.focused is not prompt:
            prompt.focus()

    # -- helpers ------------------------------------------------------------

    def _logview(self) -> VerticalScroll:
        return self.query_one("#log", VerticalScroll)

    def _line(self, text: str, style: str = "", indent: int = 0) -> None:
        log = self._logview()
        log.mount(Static(Text(" " * indent + text, style=style)))
        self._trim_log()
        # Command output while idle always jumps into view; during a turn the
        # follow flag decides, so a user reading scrollback isn't yanked down.
        if self._follow or not self._busy:
            log.scroll_end(animate=False)

    def _notify_alert(self, text: str) -> None:
        """Surface a fired alert rule: bell, toast, and a transcript line.

        Each reaches a different kind of inattention. The bell carries when the
        terminal isn't even on screen; the toast lands while the user is reading
        something else; the line is what makes it durable, since toasts
        auto-dismiss and the digest that evaluated the rule renders in a tool
        panel that is collapsed by default.

        The line is written first and unguarded — it's the one that must always
        land. Bell and toast are best-effort and independently guarded: they go
        through the driver and the screen, so a failure in either (a detached
        driver during teardown, say) must neither sink the turn nor suppress the
        other."""
        self._line(f"🔔 {text}", "bold yellow")

        def best_effort(fn) -> None:
            try:
                fn()
            except Exception:  # noqa: BLE001 - the transcript line already landed
                pass

        if alerts.sound_enabled():
            best_effort(self.bell)
        best_effort(
            lambda: self.notify(
                text, title="🔔 Alert triggered", severity="warning", timeout=10
            )
        )

    def _trim_log(self) -> None:
        # Removal is async — the _trimming mark keeps a burst of mounts from
        # re-removing widgets whose removal is still pending.
        log = self._logview()
        kids = [w for w in log.children if not getattr(w, "_trimming", False)]
        excess = len(kids) - _LOG_CAP
        if excess > 0:
            for w in kids[:excess]:
                w._trimming = True
                w.remove()

    def _log_pinned(self) -> bool:
        """True while the log is scrolled to (within a row of) the bottom."""
        log = self._logview()
        return log.scroll_offset.y >= log.max_scroll_y - 1

    def _sync_follow(self) -> None:
        """Re-derive the follow flag from geometry after a user scroll:
        following resumes at the bottom, pauses anywhere above it."""
        self._follow = self._log_pinned()

    def _follow_end(self, log: VerticalScroll) -> None:
        if self._follow:
            log.scroll_end(animate=False)

    def _mount_stream(self, widget) -> None:
        """Mount a widget produced by the running turn, trimming the oldest
        log entries and following the stream only while the user is at the
        bottom."""
        log = self._logview()
        log.mount(widget)
        self._trim_log()
        self._follow_end(log)

    def _close_answer(self) -> None:
        """Close the live answer bubble (flush pending text, stop its render
        throttle) so the next output starts a NEW bubble below whatever mounts
        next."""
        if self._answer is not None:
            self._answer.finalize()
            self._answer = None

    def _busy_guard(self, what: str) -> bool:
        """True (after printing a refusal) while a turn or compaction runs —
        session-mutating commands would corrupt the in-flight turn's state."""
        if self._busy:
            self._line(f"busy — wait for the current turn before {what}", "dim red")
            return True
        return False

    def _stop_processing(self) -> None:
        if self._processing is not None:
            self._processing.stop()
            self._processing = None

    def _set_thinking(self, on: bool) -> None:
        """Enable/disable reasoning: gates the 💭 trace and (in real mode) whether
        the agent gets the `think` tool. Applies to the next query."""
        self.show_thinking = on
        self._render_statusbar()  # footer thinking indicator updates live
        state = "ON" if on else "OFF"
        self._line(f"reasoning is now {state} — applies to the next query", "dim")

    def _show_processing(self) -> None:
        """Show the inline ``Agent ⠋ (Xs)`` spinner at the tail whenever a turn
        is still running and nothing else is streaming — e.g. between a finished
        tool call and the next output. The next streamed event calls
        ``_stop_processing`` before mounting, so it never lingers out of order."""
        if self._busy and self._processing is None:
            self._processing = ProcessingWidget("Agent")
            self._mount_stream(self._processing)

    def _status_widget(self, text: str, style: str, depth: int) -> Static:
        w = Static(Text(text, style=style), classes="subagent-status")
        if depth > 0:
            w.styles.margin = (0, 2, 0, 2 + depth * 4)
            w.styles.border_left = ("vkey", "purple")
        return w

    def _render_statusbar(self) -> None:
        model, provider = _model_provider(self.fake, self.model_override)
        ctx = self._ctx_pct if (self._tok_in or self._busy) else None
        t = _footer_line(
            model, provider, self._tok_in, self._tok_out, self._tok_cache, ctx,
            thinking=self.show_thinking,
        )
        if self._busy:
            t.append("   ", style="dim")
            t.append(_SPINNER[self._spin % len(_SPINNER)], style="bold yellow")
            t.append(
                f" working {format_duration(time.time() - self._turn_started, precise=False)}",
                style="yellow",
            )
        if self._queue:
            t.append(f"   ⏳ {len(self._queue)} queued", style="dim")
        try:
            self.query_one("#statusbar", Static).update(t)
        except Exception:
            pass

    def _tick(self) -> None:
        if self._busy:
            self._spin += 1
            self._render_statusbar()

    # -- input --------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        # Only the main prompt drives the palette (modal inputs have their own).
        if not isinstance(event.input, CommandInput):
            return
        try:
            palette = self.query_one("#command-list", OptionList)
        except Exception:  # palette not on the active screen (modal / teardown)
            return
        val = event.value
        palette.clear_options()
        if val.startswith("/") and " " not in val:
            matches = [(c, d) for c, d in _COMMANDS if c.startswith(val.lower())]
            if matches:
                for cmd, desc in matches:
                    palette.add_option(Option(Text(f"{cmd}   {desc}", style="dim"), id=cmd))
                palette.highlighted = 0
                palette.display = True
                return
        palette.display = False

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Modal inputs (ModelScreen) handle their own submit.
        if not isinstance(event.input, CommandInput):
            return
        msg = event.value.strip()
        try:
            palette = self.query_one("#command-list", OptionList)
        except Exception:
            palette = None
        # Enter with the command palette open submits the highlighted command
        # (so "/n"+Enter runs /new, like "/n"+Tab+Enter).
        if (
            palette is not None
            and msg.startswith("/")
            and palette.display
            and palette.option_count
        ):
            hl = palette.get_option_at_index(palette.highlighted or 0).id or ""
            if hl.startswith(msg.lower()):
                msg = hl
        event.input.record_history(msg)
        event.input.clear()
        if palette is not None:
            palette.display = False
        if not msg:
            return
        if msg.startswith("/"):
            self._command(msg)
            return
        if self._busy:
            self._queue.append(msg)
            self._line(f"⏳ queued: {msg}  (delivered after the current turn)", "dim yellow")
            self._render_statusbar()
            return
        self._start_turn(msg)

    def _command(self, raw: str) -> None:
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd == "/quit":
            self.exit(return_code=self._exit_code)
        elif cmd == "/new":
            # /new, /clear and /resume mutate the session/log a running turn is
            # still streaming into (worst case: its transcript entry would land
            # in the NEW session) — refuse while busy, like /compact.
            if not self._busy_guard("/new"):
                self._new_conversation()
        elif cmd == "/compact":
            self._compact()
        elif cmd == "/clear":
            if not self._busy_guard("/clear"):
                self._logview().remove_children()
                self._pending.clear()
                self._follow = True
                self._line(_READY, "dim")
        elif cmd == "/help":
            self._line("commands:", "bold")
            for c, d in _COMMANDS:
                self._line(f"  {c:<18} {d}", "dim")
        elif cmd == "/config":
            self._show_config()
        elif cmd == "/import":
            self._import(arg)
        elif cmd == "/toggle_thinking":
            self._set_thinking(not self.show_thinking)
        elif cmd == "/thinking":
            low = arg.lower()
            if low in ("on", "enable", "true"):
                self._set_thinking(True)
            elif low in ("off", "disable", "false"):
                self._set_thinking(False)
            else:
                self._set_thinking(not self.show_thinking)
        elif cmd == "/sessions":
            self._show_sessions()
        elif cmd == "/resume":
            if self._busy_guard("/resume"):
                pass
            elif arg:
                self._resume(arg)
            else:
                self._pick_session()
        elif cmd == "/model":
            if arg:
                self._set_model(arg)
            else:
                self._pick_model()
        elif cmd == "/copy":
            self._copy_last()
        elif cmd in ("/good", "/bad"):
            self._rate_last(cmd == "/good", arg)
        elif cmd == "/memory":
            self._memory(arg)
        elif cmd == "/export":
            self._export(arg)
        elif cmd == "/hotkeys":
            self._line("keyboard shortcuts:", "bold")
            for k, d in _HOTKEYS:
                self._line(f"  {k:<10} {d}", "dim")
        elif cmd == "/theme":
            self._cycle_theme()
        else:
            self._line(f"unknown command {cmd} — /help lists commands", "dim red")

    # -- commands -----------------------------------------------------------

    def _new_conversation(self) -> None:
        # Fresh conversation: drop the old thread's cached graph so its memory
        # is discarded, switch to a fresh session id (whose graph is built
        # clean on the next turn), and clear the view.
        reset_session(self.session_id)
        self.session_id = uuid4().hex[:8]
        reset_session(self.session_id)
        self._pending.clear()
        self._reset_tokens()
        self._follow = True
        self._logview().remove_children()
        self._line(f"new conversation: {self.session_id}", "dim")
        self._line(_READY, "dim")

    def _reset_tokens(self) -> None:
        """Zero the running token/ctx counters so the status bar reflects a fresh
        or just-compacted context (they rebuild on the next turn's usage event)."""
        self._tok_in = self._tok_out = self._tok_cache = 0
        self._turn_in = 0
        self._ctx_pct = 0
        self._render_statusbar()

    def _compact(self) -> None:
        """Summarize older turns and rewrite the thread so the context shrinks.
        Runs the model off the UI thread; refuses while a turn is in flight."""
        if self._busy_guard("/compact"):
            return
        self.compact_conversation()

    @work(exclusive=True)
    async def compact_conversation(self) -> None:
        self._busy = True
        self._compacting = True  # lets Esc tell compaction apart from a turn
        self._render_statusbar()
        self._line("• compacting conversation…", "dim")
        try:
            res = await compact_session(
                self.session_id, fake=self.fake, model=self.model_override,
                think=self.show_thinking,
            )
        except Exception as e:  # never crash the UI on a summarize failure
            self._line(f"compact failed: {e}", "bold red")
            return
        finally:
            self._busy = False
            self._compacting = False
            self._render_statusbar()
            self.call_after_refresh(self._drain_queue)
        if res["removed"] == 0:
            self._line("nothing to compact yet — not enough history", "dim")
            return
        # Context shrank: reset the running counters so ctx% reflects the compacted
        # thread; it repopulates on the next turn.
        self._reset_tokens()
        self._line(
            f"compacted {res['removed']} message(s) → kept {res['kept']}; "
            "context reset",
            "dim",
        )
        if res.get("summary"):
            self._line(f"  summary: {res['summary'][:200]}", "dim")

    def _show_config(self) -> None:
        endpoint = "offline" if self.fake else (
            os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1"
        )
        model, _ = _model_provider(self.fake, self.model_override)
        self._line("configuration:", "bold")
        self._line(f"  endpoint   {endpoint}", "dim")
        self._line(f"  model      {model}", "dim")
        self._line(f"  mode       {'FAKE' if self.fake else 'live'}", "dim")
        self._line(f"  session    {self.session_id}", "dim")

    def _import(self, arg: str) -> None:
        """Import a broker statement — IBKR CSV or OFX/QFX, auto-detected —
        straight into the store: no model call, so it works identically in live
        and --fake mode. Parses in a background thread so a large statement
        can't freeze the UI, then prints the summary of what was stored."""
        path = arg.strip().strip('"').strip("'")
        if not path:
            self._line(
                "usage: /import PATH   (a broker statement — IBKR CSV or OFX/QFX)",
                "dim red",
            )
            return
        # Path() alone does not expand ~ or env vars; do it here so a typed
        # "~/Downloads/stmt.csv" resolves instead of erroring as not-found.
        path = os.path.expanduser(os.path.expandvars(path))
        self._line(f"• importing {path} …", "dim")
        self.run_import(path)

    @work(thread=True, group="import")
    def run_import(self, path: str) -> None:
        # Own worker group: the turn workers are exclusive in "default", and
        # starting one must not cancel a half-done import.
        from .tools import import_ibkr_statement

        summary = import_ibkr_statement(path)
        self.call_from_thread(self._show_import_result, summary)

    def _show_import_result(self, summary: str) -> None:
        failed = summary.startswith(("No file", "Could not", "OFX/QFX import needs"))
        style = "dim red" if failed else "dim"
        for line in summary.splitlines():
            self._line(f"  {line}", style)

    def _show_sessions(self) -> None:
        found = sessions.list_sessions()
        if not found:
            self._line("no saved sessions", "dim")
            return
        self._line("saved sessions:", "bold")
        for s in found:
            marker = " ←" if s["name"] == self.session_id else ""
            self._line(f"  {s['name']:<24} {s['turns']} turns{marker}", "dim")
        self._line("resume one with /resume NAME (or /resume for a picker)", "dim")

    def _pick_session(self) -> None:
        found = sessions.list_sessions()
        if not found:
            self._line("no saved sessions", "dim")
            return
        items = [(s["name"], f"{s['name']}   ·   {s['turns']} turns") for s in found]
        self.push_screen(
            SelectScreen("Resume session (↑/↓ Enter, Esc cancel)", items), self._on_pick_session
        )

    def _on_pick_session(self, name: str | None) -> None:
        if name:
            self._resume(name)

    def _resume(self, arg: str) -> None:
        if not arg:
            self._line("usage: /resume NAME (see /sessions)", "dim red")
            return
        if not sessions.session_exists(arg):
            self._line(f"no such session: {arg} (see /sessions)", "dim red")
            return
        # Resume restores the visible transcript; model memory is always fresh,
        # so drop cached graphs for both the old and resumed ids. Token/cost
        # counters belong to the old conversation — reset them like /new does.
        reset_session(self.session_id)
        self.session_id = arg
        reset_session(self.session_id)
        self._pending.clear()
        self._reset_tokens()
        self._follow = True
        self._logview().remove_children()
        self._line(f"resumed session: {self.session_id}", "dim")
        self._replay_transcript(self.session_id)

    def _pick_model(self) -> None:
        cur, _ = _model_provider(self.fake, self.model_override)
        self.push_screen(ModelScreen(cur), self._on_pick_model)

    def _on_pick_model(self, model: str | None) -> None:
        if model:
            self._set_model(model)

    def _set_model(self, model: str) -> None:
        if model.lower() in ("default", "none", "-"):
            self.model_override = None
            note = "model override cleared — using the configured default"
        else:
            self.model_override = model
            note = f"model set to {model} (applies to the next query)"
        self.query_one("#config", Static).update(
            _config_line(self.fake, self.model_override)
        )
        self._render_statusbar()
        self._line(note, "dim")

    def _copy_last(self) -> None:
        if not self._last_answer:
            self._line("nothing to copy yet", "dim")
            return
        self.copy_to_clipboard(self._last_answer)
        self._line("copied last answer to the clipboard", "dim")

    def _rate_last(self, good: bool, note: str) -> None:
        """Record feedback on the last answer so the agent learns from it. Stores
        it as an exemplar (good) / avoid (bad) memory that seeds future few-shot
        guidance on similar questions. No-op with a hint when memory is off."""
        if not self._last_answer:
            self._line("nothing to rate yet — ask something first", "dim red")
            return
        from .feedback import record
        from .memory import get_memory

        if get_memory() is None:
            self._line(
                "long-term memory is off — set MEMORY_BACKEND=local to use /good and /bad",
                "dim red",
            )
            return
        if record(self._last_user, self._last_answer, good, note):
            if good:
                self._line("noted 👍 — I'll emulate answers like the last one", "dim green")
            else:
                self._line("noted 👎 — I'll avoid answers like the last one", "dim yellow")
        else:
            self._line("already recorded that one", "dim")

    def _memory(self, arg: str) -> None:
        """Review or prune long-term memory in-session: `/memory` lists what's
        been learned (facts, lessons, feedback) grouped by kind; `/memory forget
        TEXT` removes matching items. The interactive complement to /good and /bad
        (and the CLI's --memory)."""
        from .memory import get_memory

        mem = get_memory()
        if mem is None:
            self._line(
                "long-term memory is off — set MEMORY_BACKEND=local to enable it",
                "dim red",
            )
            return
        arg = arg.strip()
        if arg.lower().startswith("forget"):
            text = arg[len("forget"):].strip().lstrip(":").strip()
            if not text:
                self._line("usage: /memory forget <text>", "dim red")
                return
            n = mem.forget(text)
            self._line(
                f"forgot {n} memory item(s)" if n else "nothing matched; nothing removed",
                "dim green" if n else "dim",
            )
            return
        entries = mem.all(include_superseded=True)
        if not entries:
            self._line("no long-term memories stored yet", "dim")
            return
        active = [e for e in entries if not e.get("superseded")]
        archived = [e for e in entries if e.get("superseded")]

        def _key(e):  # kind label, marking archived stale values
            k = e.get("kind", "note")
            return f"{k}·superseded" if e.get("superseded") else k

        groups: dict[str, list] = {}
        for e in active + archived:  # active first, archived tier last
            groups.setdefault(_key(e), []).append(e)
        self._line(
            f"long-term memory · {len(active)} active"
            + (f" · {len(archived)} archived" if archived else "") + ":",
            "bold",
        )
        shown, limit = 0, 60
        for kind in sorted(groups, key=lambda g: ("·superseded" in g, g)):
            self._line(f"  [{kind}] ({len(groups[kind])})", "dim")
            for e in groups[kind]:
                if shown >= limit:
                    break
                text = " ".join(e["text"].split())
                if len(text) > 100:
                    text = text[:99] + "…"
                self._line(f"    • {text}", "dim")
                shown += 1
            if shown >= limit:
                self._line(
                    f"  … {len(entries) - shown} more — use the CLI `--memory` to see all",
                    "dim",
                )
                break
        self._line("archived values are kept for audit but never recalled; "
                   "prune with /memory forget <text>", "dim")

    def _export(self, arg: str) -> None:
        turns = sessions.read_transcript(self.session_id)
        if not turns:
            self._line("nothing to export yet", "dim")
            return
        name = os.path.expanduser(os.path.expandvars(arg or "export.html"))
        store = sessions.store_dir()
        # pathlib: an absolute NAME overrides the store dir entirely, and a
        # relative one may point into a subdir — both are deliberate.
        dest = store / name
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".jsonl"):
                dest.write_text(
                    "\n".join(json.dumps(t) for t in turns) + "\n", encoding="utf-8"
                )
            else:
                dest.write_text(_export_html(turns, self.session_id), encoding="utf-8")
        except OSError as e:
            self._line(f"export failed: {e}", "dim red")
            return
        self._line(f"exported {len(turns)} turn(s) → {dest}", "dim")

    def _cycle_theme(self) -> None:
        order = ["textual-dark", "textual-light"]
        try:
            cur = getattr(self, "theme", None)
            self.theme = order[(order.index(cur) + 1) % 2] if cur in order else order[1]
            self._line(f"theme: {self.theme}", "dim")
        except Exception:
            self.dark = not getattr(self, "dark", True)
            self._line(f"theme: {'dark' if self.dark else 'light'}", "dim")

    def _replay_transcript(self, session_id: str) -> None:
        turns = sessions.read_transcript(session_id)
        if not turns:
            return
        self._line(f"— replaying {len(turns)} past turn(s) —", "dim")
        for turn in turns:
            log = self._logview()
            log.mount(UserMessageWidget(turn.get("query", "")))
            log.mount(AgentMessageWidget(turn.get("answer", "") or ""))
            log.scroll_end(animate=False)
        # The replayed conversation's last exchange is on screen — make /copy,
        # /good and /bad act on it instead of claiming nothing happened yet.
        self._last_user = turns[-1].get("query", "")
        self._last_answer = turns[-1].get("answer", "") or ""

    # -- actions ------------------------------------------------------------

    def action_cancel_turn(self) -> None:
        if not self._busy:
            return
        if self._compacting:
            # Compaction rewrites the thread at the end of its run; killing it
            # mid-summarize would be safe but pointless — it has no partial
            # output to save. More importantly it is NOT the turn worker, so
            # don't claim "turn cancelled" while it keeps running.
            self._line("compacting — can't cancel; it finishes on its own", "dim")
            return
        self._stop_processing()
        worker = self._turn_worker
        if worker is not None:
            worker.cancel()
            self._line("⏹ turn cancelled", "dim red")

    def action_toggle_mouse(self) -> None:
        """Toggle terminal mouse capture. Captured (default): scroll-wheel and
        clicks work, and native selection needs a modifier drag (⌥ / Shift).
        Released: the app stops grabbing the mouse, so a plain drag selects text
        and Ctrl/Cmd+C copies via the terminal — but scroll-wheel pauses until you
        toggle back. Degrades gracefully if the driver lacks these hooks."""
        driver = getattr(self, "_driver", None)
        disable = getattr(driver, "_disable_mouse_support", None)
        enable = getattr(driver, "_enable_mouse_support", None)
        if not callable(disable) or not callable(enable):
            self._line("mouse toggle isn't supported by this terminal", "yellow")
            return
        self._mouse_released = not getattr(self, "_mouse_released", False)
        try:
            if self._mouse_released:
                disable()
                self._line(
                    "mouse released — drag selects text; Ctrl/Cmd+C copies. F2 restores scroll",
                    "yellow",
                )
            else:
                enable()
                self._line("mouse captured — scroll & click active", "dim")
        except Exception as e:  # noqa: BLE001 - best-effort terminal control
            self._line(f"mouse toggle failed: {e}", "red")

    def action_toggle_tools(self) -> None:
        self._tools_collapsed = not self._tools_collapsed
        for panel in self.query(ToolPanel):
            panel.collapsed = self._tools_collapsed
        state = "collapsed" if self._tools_collapsed else "expanded"
        self._line(f"tool panels {state}", "dim")

    def action_toggle_thinking_collapse(self) -> None:
        self._think_collapsed = not self._think_collapsed
        for panel in self.query(ThinkingPanel):
            panel.collapsed = self._think_collapsed
        state = "collapsed" if self._think_collapsed else "expanded"
        self._line(f"thinking panels {state}", "dim")

    # -- scrolling ----------------------------------------------------------
    # ↑/↓ are reserved for prompt history, so scrolling the log is on
    # PgUp/PgDn, Shift+↑/↓, Ctrl+Home/End, and the mouse wheel. Every user
    # scroll re-derives the follow flag: away from the bottom pauses streaming
    # auto-scroll, back at the bottom resumes it. Even with animate=False the
    # new offset only lands on the next refresh, so the check is deferred.

    def _scrolled(self) -> None:
        self.call_after_refresh(self._sync_follow)

    def action_scroll_log_page_up(self) -> None:
        self._logview().scroll_page_up(animate=False)
        self._scrolled()

    def action_scroll_log_page_down(self) -> None:
        self._logview().scroll_page_down(animate=False)
        self._scrolled()

    def action_scroll_log_up(self) -> None:
        self._logview().scroll_up(animate=False)
        self._scrolled()

    def action_scroll_log_down(self) -> None:
        self._logview().scroll_down(animate=False)
        self._scrolled()

    def action_scroll_log_home(self) -> None:
        self._logview().scroll_home(animate=False)
        self._scrolled()

    def action_scroll_log_end(self) -> None:
        self._logview().scroll_end(animate=False)
        self._follow = True

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self._logview().scroll_up(animate=False)
        self._scrolled()
        event.stop()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self._logview().scroll_down(animate=False)
        self._scrolled()
        event.stop()

    # -- run ----------------------------------------------------------------

    def _start_turn(self, msg: str) -> None:
        log = self._logview()
        log.mount(UserMessageWidget(msg))
        self._processing = ProcessingWidget("Agent")
        log.mount(self._processing)
        self._follow = True  # a fresh submit implies watching the reply
        log.scroll_end(animate=False)
        self._busy = True
        self._turn_started = time.time()
        # Start a fresh context measurement for this turn; ctx% keeps showing the
        # last known value (no flicker to 0%) until this turn's usage arrives.
        self._turn_in = 0
        self._render_statusbar()
        # The session id is snapshotted into the worker so the finished turn is
        # always logged to the conversation it STARTED in, whatever happens to
        # self.session_id while it runs.
        self._turn_worker = self.stream_response(msg, self.session_id)

    def _drain_queue(self) -> None:
        if self._queue and not self._busy:
            self._start_turn(self._queue.pop(0))

    @work(exclusive=True)
    async def stream_response(self, msg: str, session_id: str) -> None:
        # ``session_id`` is the conversation this turn belongs to, snapshotted
        # at submit time — never re-read self.session_id here.
        log = self._logview()
        started = time.monotonic()
        self._answer = None  # created lazily on the first token
        self._last_user = msg  # remember the query so /good and /bad can rate it
        final_text = ""
        turn_errored = False
        model, _ = _model_provider(self.fake, self.model_override)
        try:
            async for ev in traced(
                run_turn(
                    msg, session_id, fake=self.fake, model=self.model_override,
                    think=self.show_thinking,
                ),
                user_msg=msg, session_id=session_id, model=model, fake=self.fake,
            ):
                if ev.kind == "token":
                    # Stream tokens live into a growing answer bubble (the
                    # bubble throttles its own re-renders and follow-scrolls).
                    if self._answer is None:
                        self._stop_processing()
                        self._answer = AgentMessageWidget("", live=True)
                        self._mount_stream(self._answer)
                    self._answer.append_text(ev.text)
                elif ev.kind == "reasoning":
                    if self.show_thinking:
                        self._stop_processing()
                        # Close the current answer bubble so output after this
                        # thought starts a NEW bubble BELOW the panel — a thought
                        # that preceded the output stays above it (chronological),
                        # same as tool panels.
                        self._close_answer()
                        self._mount_stream(ThinkingPanel(ev))
                    self._show_processing()  # keep a spinner while it works on
                elif ev.kind == "usage":
                    # Usage arrives as per-call deltas (live). The in/out/cache
                    # counters are CUMULATIVE session totals (they drive $cost).
                    # ctx% is different: it must track the CURRENT context size, so
                    # it derives from this turn's input only (_turn_in), which the
                    # final usage event reconciles to the authoritative total.
                    self._tok_in += ev.tokens_in
                    self._tok_out += ev.tokens_out
                    self._tok_cache += ev.tokens_cache
                    self._turn_in += ev.tokens_in
                    if ev.context_tokens >= 0:
                        self._turn_in = ev.context_tokens
                    model, _ = _model_provider(self.fake, self.model_override)
                    self._ctx_pct = _context_pct(model, self._turn_in)
                    self._render_statusbar()
                elif ev.kind == "status":
                    # Auto-compaction shrinks the context mid-turn (context_tokens=0);
                    # reset the gauge immediately so the footer reflects it now.
                    if ev.context_tokens >= 0:
                        self._turn_in = ev.context_tokens
                        model, _ = _model_provider(self.fake, self.model_override)
                        self._ctx_pct = _context_pct(model, self._turn_in)
                        self._render_statusbar()
                    self._line(f"• {ev.text}", "dim")
                elif ev.kind == "tool_start":
                    self._stop_processing()
                    # Close the current answer bubble so any text streamed after
                    # this tool call starts a NEW bubble *below* the panel. Keeps
                    # the transcript chronological: a tool that finished before the
                    # answer stays above it, instead of the answer floating to the
                    # top because it reused the pre-tool bubble.
                    self._close_answer()
                    panel = ToolPanel(ev)
                    panel.collapsed = self._tools_collapsed
                    self._pending[ev.call_id] = panel
                    self._mount_stream(panel)
                elif ev.kind == "tool_end":
                    panel = self._pending.pop(ev.call_id, None)
                    if panel is None:
                        panel = ToolPanel(ev)
                        panel.collapsed = self._tools_collapsed
                        self._mount_stream(panel)
                    panel.finish(ev)
                    self._follow_end(log)
                    self._show_processing()  # working on the next step
                elif ev.kind == "alert":
                    self._notify_alert(ev.text)
                elif ev.kind == "final":
                    self._stop_processing()
                    final_text = ev.text or (self._answer.text if self._answer else "")
                    self._last_answer = final_text
                    dt = time.monotonic() - started
                    if self._answer is not None:
                        # Streamed already — possibly across several bubbles split
                        # by tool calls. Those bubbles collectively hold the full
                        # answer, so only stamp the duration on the last one.
                        # Overwriting it with the full text would duplicate any
                        # pre-tool segment that lives in an earlier bubble.
                        self._answer.set_duration(dt)
                    else:
                        # No tokens streamed (fake mode / non-streaming endpoint):
                        # mount the answer read back from graph state.
                        w = AgentMessageWidget(final_text)
                        w.set_duration(dt)
                        self._mount_stream(w)
                    self._follow_end(log)
                elif ev.kind == "error":
                    turn_errored = True
                    if self._processing is not None:
                        self._processing.mark_error(ev.text)
                        self._processing = None
                    else:
                        self._line(f"error  {ev.text}", "bold red")
                    self._exit_code = 1
            # Per-turn verdict: a clean turn resets a sticky error exit code, so
            # /quit after later successes exits 0 (and --once keeps its signal).
            self._exit_code = 1 if turn_errored else 0
            if final_text:
                try:
                    sessions.log_turn(session_id, msg, final_text)
                except OSError as e:  # disk full/permissions must not kill the app
                    self._line(f"could not save the turn to the transcript: {e}", "dim red")
            if self.once:
                self.exit(return_code=self._exit_code)
        finally:
            self._stop_processing()
            # Any tool panel still awaiting its result is orphaned (the turn
            # errored, was cancelled, or ended mid-call). Stop its spinner so it
            # doesn't tick forever, and clear so the next turn starts clean.
            for panel in self._pending.values():
                panel.abort()
            self._pending.clear()
            self._close_answer()  # flushes throttled text on cancel/error too
            self._busy = False
            self._turn_worker = None
            self._render_statusbar()
            self.call_after_refresh(self._drain_queue)
