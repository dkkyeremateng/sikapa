"""Textual UI tests — no network, no API keys required."""

import json
import time
from typing import cast

from textual.containers import VerticalScroll
from textual.widgets import Input, OptionList, Static
from textual.worker import Worker

from financial_research_assistant import sessions
from .fixtures.statements import SAMPLE_STATEMENT
from .helpers.fakes import RecordingWorker
from .helpers.tui import log_text, statusbar_text
from financial_research_assistant.events import AgentEvent
from financial_research_assistant.tui import (
    AgentApp,
    AgentMessageWidget,
    CommandInput,
    ModelScreen,
    ProcessingWidget,
    SelectScreen,
    ThinkingPanel,
    ToolPanel,
    _COMMANDS,
    _HOTKEYS,
    _READY,
    _SPINNER,
    _context_pct,
    _footer_line,
    _model_provider,
)


async def test_tui_command_palette(monkeypatch, tmp_path):
    """Typing "/" opens the docked #command-list listing every command; "/c"
    narrows and Tab fills the highlighted "/clear"; Enter runs it, collapsing a
    populated log back to just the ready line. A non-slash prefix never opens
    the palette."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    app = AgentApp(fake=True, session_id="palette")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        palette = app.query_one("#command-list", OptionList)
        log = app.query_one("#log", VerticalScroll)

        # A completed turn leaves the log populated (something worth clearing).
        box.value = "hello"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(list(log.query(Static))) > 1

        # "/" opens the palette with one option per command.
        box.focus()
        await pilot.press("slash")
        await pilot.pause()
        assert palette.display
        assert palette.option_count == len(_COMMANDS)

        # "/cl" narrows to /clear (highlighted 0); Tab fills it into the input.
        # ("/c" alone is ambiguous now that /compact also starts with c.)
        await pilot.press("c", "l")
        await pilot.pause()
        await pilot.press("tab")
        assert box.value == "/clear"

        # Enter runs /clear → the log collapses to a single ready line.
        await pilot.press("enter")
        await pilot.pause()
        lines = [str(w.render()) for w in log.query(Static)]
        assert lines == [_READY]

        # Negative: a non-slash prefix leaves the palette hidden.
        await pilot.press(*"hi")
        await pilot.pause()
        assert not palette.display


async def test_tui_history_recall(monkeypatch, tmp_path):
    """With the palette closed, ↑ recalls the last submitted prompt into the
    (cleared) input."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    app = AgentApp(fake=True, session_id="history")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        box.value = "recall me later"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert box.value == ""  # input clears on submit

        box.focus()
        await pilot.press("up")
        await pilot.pause()
        assert box.value == "recall me later"


async def test_tui_reasoning_panel_and_toggle(monkeypatch, tmp_path):
    """A fake run mounts a collapsed 💭 ThinkingPanel; after /toggle_thinking
    turns reasoning off, the next run mounts no new panel."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    app = AgentApp(fake=True, session_id="thinking")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        log = app.query_one("#log", VerticalScroll)

        box.value = "think it over"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        panels = list(log.query(ThinkingPanel))
        assert len(panels) == 1
        assert "💭" in panels[0].title

        # Hide reasoning, then run again — no additional panel appears.
        box.focus()
        box.value = "/toggle_thinking"
        await pilot.press("enter")
        await pilot.pause()
        assert app.show_thinking is False
        before = len(list(log.query(ThinkingPanel)))

        box.value = "and again"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(list(log.query(ThinkingPanel))) == before


async def test_tui_help_lists_every_command(monkeypatch, tmp_path):
    """/help prints every command name and its description from _COMMANDS."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    app = AgentApp(fake=True, session_id="help")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        box.value = "/help"
        await pilot.press("enter")
        await pilot.pause()
        text = log_text(app)
        for cmd, desc in _COMMANDS:
            assert cmd in text
            assert desc in text


async def test_tui_config_shows_configured_model(monkeypatch, tmp_path):
    """/config surfaces the model the environment configured (not a default)."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_MODEL", "sentinel-model-x7")

    # Live mode reflects the configured model; /config only reads config and
    # never contacts a model, so it stays offline. (Fake mode deliberately
    # reports the "scripted-fake" marker instead — covered elsewhere.)
    app = AgentApp(fake=False, session_id="config")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        box.value = "/config"
        await pilot.press("enter")
        await pilot.pause()
        # The model appears in the log only because /config printed it.
        assert "sentinel-model-x7" in log_text(app)


async def test_tui_import_command_imports_statement(monkeypatch, tmp_path):
    """/import PATH parses a statement straight into the store (no model call;
    runs on a background thread worker) and prints the summary; a bad path is
    reported, not raised."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))
    csv_path = tmp_path / "stmt.csv"
    csv_path.write_text(SAMPLE_STATEMENT, encoding="utf-8")

    app = AgentApp(fake=True, session_id="imp")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        box.value = f"/import {csv_path}"
        await pilot.press("enter")
        await app.workers.wait_for_complete()  # parse runs off the UI thread
        await pilot.pause()
        text = log_text(app)
        assert "U1111111" in text and "2 trade(s)" in text

        # It actually landed in the store.
        from financial_research_assistant import statements
        assert len(statements.list_imports()) == 1

        # A missing path is reported inline, never raised.
        box.value = f"/import {tmp_path / 'nope.csv'}"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "No file found" in log_text(app)


async def test_tui_import_command_requires_path(monkeypatch, tmp_path):
    """/import with no argument prints a usage hint instead of importing."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    monkeypatch.setenv("FINANCIAL_RESEARCH_STATEMENTS_DB", str(tmp_path / "s.db"))

    app = AgentApp(fake=True, session_id="imp2")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        box.value = "/import"
        await pilot.press("enter")
        await pilot.pause()
        assert "usage: /import PATH" in log_text(app)


async def test_tui_new_conversation_resets(monkeypatch, tmp_path):
    """/new switches to a fresh session id and clears the previous turn's log
    down to the new-conversation notice and the ready line."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    app = AgentApp(fake=True, session_id="original-sess")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        log = app.query_one("#log", VerticalScroll)

        box.value = "leave a trace"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "FAKE-OK" in log_text(app)
        old_id = app.session_id

        box.focus()
        box.value = "/new"
        await pilot.press("enter")
        await pilot.pause()
        assert app.session_id != old_id  # fresh conversation id
        lines = [str(w.render()) for w in log.query(Static)]
        assert _READY in lines
        assert any("new conversation" in ln for ln in lines)
        assert not any("FAKE-OK" in ln for ln in lines)  # prior turn cleared


async def test_tui_transcript_persists_and_resume_replays(monkeypatch, tmp_path):
    """A completed turn is persisted to the session store; /sessions lists it
    and /resume NAME clears the view and replays that session's transcript."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    app = AgentApp(fake=True, session_id="alpha")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        log = app.query_one("#log", VerticalScroll)

        box.value = "remember alpha"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        # The turn was persisted to the tmp store for this session id.
        turns = sessions.read_transcript("alpha")
        assert len(turns) == 1
        assert turns[0]["query"] == "remember alpha"
        assert "FAKE-OK" in turns[0]["answer"]

        # /sessions lists the saved session by name.
        box.focus()
        box.value = "/sessions"
        await pilot.press("enter")
        await pilot.pause()
        assert "alpha" in log_text(app)

        # A separate pre-saved session replays on /resume, replacing the view.
        sessions.log_turn("beta", "hello from beta", "beta answer 42")
        box.focus()
        box.value = "/resume beta"
        await pilot.press("enter")
        await pilot.pause()
        assert app.session_id == "beta"
        replay = log_text(app)
        assert "hello from beta" in replay  # replayed query
        assert "beta answer 42" in replay  # replayed answer
        assert "remember alpha" not in replay  # old session's log was cleared

# ===========================================================================
# New TUI console features
# ===========================================================================


# -- pure helpers -----------------------------------------------------------


def test_format_duration_switches_to_minutes_over_60s():
    """Under a minute stays in seconds; 60s and over switches to minutes+seconds.
    ``precise=False`` drops the decimal for live tick timers."""
    from financial_research_assistant.events import format_duration

    assert format_duration(0.7) == "0.7s"
    assert format_duration(45.2) == "45.2s"
    assert format_duration(59.9) == "59.9s"
    assert format_duration(60) == "1m 00s"
    assert format_duration(65.3) == "1m 05s"
    assert format_duration(154) == "2m 34s"
    # precise=False: integer seconds under a minute (live "working Ns"), minutes over.
    assert format_duration(3, precise=False) == "3s"
    assert format_duration(90, precise=False) == "1m 30s"


def test_context_pct_uses_model_window_and_clamps():
    """_context_pct scales tokens against the model's context window, matching
    a known model by prefix, falling back to 128k for unknown models, and
    clamping to 100."""
    # gpt-4o has a 128k window: 64k prompt tokens is exactly half.
    assert _context_pct("gpt-4o", 64_000) == 50
    # a variant id still matches its family window by prefix.
    assert _context_pct("gpt-4o-mini", 128_000) == 100
    # a smaller-window model reaches 100% far sooner.
    assert _context_pct("gpt-3.5-turbo", 16_385) == 100
    # a 200k-window model: 100k tokens is half.
    assert _context_pct("o1-preview", 100_000) == 50
    # unknown model falls back to the 128k window (32k/128k = 25%),
    # NOT to gpt-3.5's 16k window (which would clamp to 100).
    assert _context_pct("mystery-model-9000", 32_000) == 25
    # over the window clamps to 100, never above.
    assert _context_pct("gpt-4o", 500_000) == 100


def test_footer_line_exact_rendering():
    """_footer_line renders model · provider · [ctx N% ·] in/out tok [· cache N],
    including the segments only when they carry information. Uses an unpriced
    model so no cost segment appears (cost is covered separately)."""
    m = "local-model"  # not in _MODEL_PRICING → no $cost segment
    base = _footer_line(m, "openai", 10, 20)
    assert base.plain == "local-model · openai · in 10 / out 20 tok"

    # unknown model → default 128k cap shown alongside the percentage
    with_ctx = _footer_line(m, "openai", 10, 20, 0, 50)
    assert with_ctx.plain == "local-model · openai · ctx 50% of 128k · in 10 / out 20 tok"

    with_cache = _footer_line(m, "openai", 10, 20, 5)
    assert with_cache.plain == "local-model · openai · in 10 / out 20 tok · cache 5"

    full = _footer_line(m, "openai", 10, 20, 5, 50)
    assert full.plain == "local-model · openai · ctx 50% of 128k · in 10 / out 20 tok · cache 5"


def test_footer_line_cost_and_thinking_segments():
    """A priced model appends a $cost segment; passing ``thinking`` appends a
    live on/off indicator. An unpriced model shows no cost."""
    # gpt-4o is priced (2.50 / 10.00 per 1M): (10*2.5 + 20*10)/1e6 = $0.0002.
    priced = _footer_line("gpt-4o", "openai", 10, 20)
    assert "$0.0002" in priced.plain
    assert _footer_line("local-model", "openai", 10, 20, thinking=None).plain.count("$") == 0

    on = _footer_line("local-model", "openai", 0, 0, thinking=True)
    assert on.plain.endswith("thinking on")
    off = _footer_line("local-model", "openai", 0, 0, thinking=False)
    assert off.plain.endswith("thinking off")


def test_cost_helper_prices_known_and_env_override(monkeypatch):
    """_cost prices a known model and returns None for unknown ones; env vars
    override the built-in table for a custom gateway."""
    from financial_research_assistant.tui import _cost

    assert _cost("gpt-4o", 1_000_000, 0) == 2.50
    assert _cost("mystery-model", 1_000_000, 1_000_000) is None
    monkeypatch.setenv("OPENAI_INPUT_COST_PER_1M", "1.00")
    monkeypatch.setenv("OPENAI_OUTPUT_COST_PER_1M", "3.00")
    assert _cost("mystery-model", 1_000_000, 1_000_000) == 4.00


def test_model_provider_fake_is_offline_scripted():
    """In fake mode _model_provider always reports the offline scripted marker,
    ignoring any override."""
    assert _model_provider(True) == ("scripted-fake", "offline")
    assert _model_provider(True, "gpt-4o") == ("scripted-fake", "offline")


def test_model_provider_override_beats_env(monkeypatch):
    """A live-mode override wins over the environment model; with no override
    the environment model is used."""
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    assert _model_provider(False, "override-model")[0] == "override-model"
    assert _model_provider(False)[0] == "env-model"


# -- status bar -------------------------------------------------------------


async def test_statusbar_reports_model_and_zero_tokens_after_turn(monkeypatch, tmp_path):
    """After one fake turn the status bar shows the model, provider, and the
    all-zero fake-mode token counts (ctx segment dropped once idle)."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="statusbar")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "hi there"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert statusbar_text(app) == "scripted-fake · offline · in 0 / out 0 tok · thinking on"


async def test_statusbar_shows_spinner_and_elapsed_while_busy(monkeypatch, tmp_path):
    """While busy, _render_statusbar appends a spinner glyph and the elapsed
    'working Ns' since the turn started."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="busy")
    async with app.run_test() as pilot:
        app._busy = True
        app._turn_started = time.time() - 3
        app._render_statusbar()
        text = statusbar_text(app)
        assert "working 3s" in text
        assert _SPINNER[app._spin % len(_SPINNER)] in text


# -- message queue ----------------------------------------------------------


async def test_busy_submit_enqueues_instead_of_starting(monkeypatch, tmp_path):
    """Submitting a query while a turn is running enqueues it (growing _queue,
    logging a 'queued' line, showing 'N queued' in the bar) instead of starting
    a new turn."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="queue")
    async with app.run_test() as pilot:
        app._busy = True  # a turn is (notionally) already running
        box = app.query_one(CommandInput)
        box.value = "second question"
        await pilot.press("enter")
        await pilot.pause()
        assert app._queue == ["second question"]
        text = log_text(app)
        assert "queued" in text
        assert "second question" in text
        assert "FAKE-OK" not in text  # no turn actually started
        assert "1 queued" in statusbar_text(app)


# -- cancel -----------------------------------------------------------------


async def test_cancel_turn_cancels_worker_and_logs(monkeypatch, tmp_path):
    """Esc while busy cancels the running worker and logs a cancelled line."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="cancel")
    async with app.run_test() as pilot:
        worker = RecordingWorker()
        app._busy = True
        app._turn_worker = cast(Worker, worker)
        app.action_cancel_turn()
        await pilot.pause()
        assert worker.cancelled is True
        assert "cancelled" in log_text(app)


async def test_cancel_turn_is_noop_when_idle(monkeypatch, tmp_path):
    """With no turn running, cancel does nothing — neither cancels a worker nor
    logs a cancelled line."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="cancel-idle")
    async with app.run_test() as pilot:
        worker = RecordingWorker()
        app._busy = False
        app._turn_worker = cast(Worker, worker)
        app.action_cancel_turn()
        await pilot.pause()
        assert worker.cancelled is False
        assert "cancelled" not in log_text(app)


# -- collapse toggles -------------------------------------------------------


async def test_toggle_collapse_flips_tool_and_thinking_panels(monkeypatch, tmp_path):
    """Ctrl+O / Ctrl+T actions flip the collapsed state of every tool /
    thinking panel independently."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="collapse")
    async with app.run_test() as pilot:
        tool = ToolPanel(
            AgentEvent(
                "tool_start", "Agent · word_count", tool="word_count",
                detail="{}", call_id="c1",
            )
        )
        think = ThinkingPanel(AgentEvent("reasoning", "pondering", agent="Agent"))
        log = app.query_one("#log", VerticalScroll)
        await log.mount(tool)
        await log.mount(think)
        await pilot.pause()
        assert tool.collapsed is True and think.collapsed is True

        # Toggling tools expands tool panels but leaves thinking untouched.
        app.action_toggle_tools()
        await pilot.pause()
        assert tool.collapsed is False
        assert think.collapsed is True

        # Toggling thinking expands thinking panels.
        app.action_toggle_thinking_collapse()
        await pilot.pause()
        assert think.collapsed is False

        # Toggling tools again collapses them back.
        app.action_toggle_tools()
        await pilot.pause()
        assert tool.collapsed is True


# -- /model modal -----------------------------------------------------------


async def test_model_command_opens_modal_and_sets_override(monkeypatch, tmp_path):
    """/model with no arg opens the ModelScreen picker; typing a name + Enter
    dismisses it and sets model_override."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="model-modal")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        box.value = "/model"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ModelScreen)

        modal_input = app.screen.query_one("#modal-input", Input)
        modal_input.value = "gpt-4o-custom"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, ModelScreen)  # dismissed
        assert app.model_override == "gpt-4o-custom"


async def test_model_command_with_arg_sets_override_directly(monkeypatch, tmp_path):
    """/model NAME sets model_override without opening the picker."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="model-arg")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        box.value = "/model gpt-4o-direct"
        await pilot.press("enter")
        await pilot.pause()
        assert app.model_override == "gpt-4o-direct"
        assert not isinstance(app.screen, ModelScreen)  # no picker needed
        assert "gpt-4o-direct" in log_text(app)


# -- /resume modal ----------------------------------------------------------


async def test_resume_without_arg_opens_picker_escape_dismisses(monkeypatch, tmp_path):
    """With a saved session present, /resume (no arg) opens the SelectScreen
    picker; Escape dismisses it without switching sessions."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    sessions.log_turn("saved-one", "hello saved", "saved answer")
    app = AgentApp(fake=True, session_id="resume-modal")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        box.value = "/resume"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, SelectScreen)

        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, SelectScreen)  # dismissed
        assert app.session_id == "resume-modal"          # unchanged on cancel


# -- /copy /export /hotkeys /theme ------------------------------------------


def test_bindings_quit_on_ctrl_q_not_ctrl_c():
    """In-app selection off; quit is Ctrl+Q; F2 toggles mouse. Ctrl+C is NOT
    bound to quit — leaving Textual's built-in `help_quit` warning active."""
    assert AgentApp.ALLOW_SELECT is False
    actions = {b[0]: b[1] for b in AgentApp.BINDINGS}
    assert actions.get("ctrl+q") == "quit"
    assert "ctrl+c" not in actions      # not overridden → reflexive Ctrl+C warns
    assert actions.get("f2") == "toggle_mouse"


async def test_ctrl_c_warns_instead_of_quitting(monkeypatch, tmp_path):
    """Pressing Ctrl+C must NOT close the agent — it triggers Textual's
    help_quit (a 'Press Ctrl+Q to quit' notice); Ctrl+Q is the real quit."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="ctrlc")
    async with app.run_test() as pilot:
        # Ctrl+Q is an active, priority quit binding (inherited from App).
        assert any(
            k == "ctrl+q" and b.binding.action in ("quit", "app.quit")
            for k, b in app.active_bindings.items()
        )
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running  # reflexive Ctrl+C did not exit the app


async def test_prompt_stays_focused_after_clicking_panel(monkeypatch, tmp_path):
    """The prompt keeps focus even after clicking a collapsible panel (which
    would otherwise focus the panel title), so typing always lands in the input."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="focus-keep")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        assert app.focused is box
        tool = ToolPanel(
            AgentEvent("tool_start", "Agent · x", tool="x", detail="{}", call_id="c1")
        )
        await app.query_one("#log", VerticalScroll).mount(tool)
        await pilot.pause()
        await pilot.click(tool)
        await pilot.pause()
        await pilot.pause()
        assert app.focused is box  # focus bounced back to the prompt


async def test_model_modal_input_keeps_focus(monkeypatch, tmp_path):
    """The focus guard must NOT steal focus from a modal — the /model picker's
    own input stays focused so typing goes to it."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="focus-modal")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "/model"
        await pilot.press("enter")
        await pilot.pause()
        modal_input = app.screen.query_one("#modal-input", Input)
        await pilot.pause()
        assert app.focused is modal_input  # modal keeps its own focus


async def test_mouse_captured_by_default_and_f2_toggles(monkeypatch, tmp_path):
    """Mouse is captured at startup (so scroll-wheel/click work); F2 releases it
    for plain-drag native selection, and F2 again re-captures for scrolling."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="mouse-toggle")
    async with app.run_test() as pilot:
        calls: list = []
        monkeypatch.setattr(
            app._driver, "_disable_mouse_support",
            lambda: calls.append("off"), raising=False,
        )
        monkeypatch.setattr(
            app._driver, "_enable_mouse_support",
            lambda: calls.append("on"), raising=False,
        )
        assert getattr(app, "_mouse_released", False) is False  # captured by default

        app.action_toggle_mouse()          # release for selection
        await pilot.pause()
        assert app._mouse_released is True and calls == ["off"]

        app.action_toggle_mouse()          # re-capture (restore scroll)
        await pilot.pause()
        assert app._mouse_released is False and calls == ["off", "on"]


async def test_copy_reports_no_answer_then_copies_after_turn(monkeypatch, tmp_path):
    """/copy with no answer yet reports there is nothing to copy; after a turn
    it copies the last answer and says so."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="copy")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        box.value = "/copy"
        await pilot.press("enter")
        await pilot.pause()
        assert "nothing to copy yet" in log_text(app)

        box.value = "give me an answer"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        box.value = "/copy"
        await pilot.press("enter")
        await pilot.pause()
        assert "copied last answer to the clipboard" in log_text(app)


async def test_export_writes_html_and_jsonl_to_store(monkeypatch, tmp_path):
    """/export NAME.html and /export NAME.jsonl write the transcript to the
    session store, serializing the turn's query and answer."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="export")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        box.value = "capture this turn"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        box.value = "/export out.html"
        await pilot.press("enter")
        await pilot.pause()
        html_path = tmp_path / "out.html"
        assert html_path.exists()
        html = html_path.read_text(encoding="utf-8")
        assert "capture this turn" in html  # query serialized
        assert "FAKE-OK" in html             # answer serialized

        box.value = "/export dump.jsonl"
        await pilot.press("enter")
        await pilot.pause()
        jsonl_path = tmp_path / "dump.jsonl"
        assert jsonl_path.exists()
        rows = [
            json.loads(ln)
            for ln in jsonl_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["query"] == "capture this turn"
        assert "FAKE-OK" in rows[0]["answer"]


async def test_hotkeys_lists_every_shortcut(monkeypatch, tmp_path):
    """/hotkeys prints every key and description from _HOTKEYS."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="hotkeys")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        box.value = "/hotkeys"
        await pilot.press("enter")
        await pilot.pause()
        text = log_text(app)
        for key, desc in _HOTKEYS:
            assert key in text
            assert desc in text


async def test_theme_command_switches_theme(monkeypatch, tmp_path):
    """/theme changes the active theme and reports the new one."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="theme")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        before = app.theme
        box.value = "/theme"
        await pilot.press("enter")
        await pilot.pause()
        assert app.theme != before          # a real theme transition
        assert "theme:" in log_text(app)    # and it reports the new theme


# -- adapter usage event ----------------------------------------------------












# -- thinking toggle (real, end-to-end) -------------------------------------




async def test_toggle_thinking_updates_footer_live(monkeypatch, tmp_path):
    """/toggle_thinking flips the footer indicator immediately (before any new
    turn) and stops new reasoning panels from mounting."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="footer-think")
    async with app.run_test() as pilot:
        assert statusbar_text(app).endswith("thinking on")
        box = app.query_one(CommandInput)
        box.value = "/toggle_thinking"
        await pilot.press("enter")
        await pilot.pause()
        assert app.show_thinking is False
        assert statusbar_text(app).endswith("thinking off")


async def test_no_thinking_app_starts_with_reasoning_off(monkeypatch, tmp_path):
    """AgentApp(think=False) starts with the footer off and mounts no 💭 panel
    for a turn."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="no-think", think=False)
    async with app.run_test() as pilot:
        assert statusbar_text(app).endswith("thinking off")
        app.query_one(CommandInput).value = "hi"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        log = app.query_one("#log", VerticalScroll)
        assert not list(log.query(ThinkingPanel))


# -- live streaming + run duration ------------------------------------------


async def test_streamed_tokens_render_live_with_duration(monkeypatch, tmp_path):
    """token events stream into a single growing AgentMessageWidget, and the
    final event stamps a run duration on its header."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    async def streaming(*args, **kwargs):
        yield AgentEvent("token", "Hello ")
        yield AgentEvent("token", "world")
        yield AgentEvent("usage", "", tokens_in=1, tokens_out=2)
        yield AgentEvent("final", "Hello world")

    monkeypatch.setattr("financial_research_assistant.tui.run_turn", streaming)
    app = AgentApp(fake=True, session_id="stream")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "stream please"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        bubbles = list(app.query(AgentMessageWidget))
        assert len(bubbles) == 1                 # one bubble, not one-per-token
        assert bubbles[0].text == "Hello world"  # tokens accumulated live
        assert bubbles[0].duration is not None    # duration stamped at final


async def test_working_indicator_re_armed_between_tool_and_answer(monkeypatch, tmp_path):
    """A working spinner must be visible whenever the agent is busy — including
    the gap between a finished tool call and the next output — and must not
    linger once the turn completes."""
    import asyncio

    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    gate = asyncio.Event()

    async def streaming(*args, **kwargs):
        yield AgentEvent("tool_start", "Agent · read_source", tool="read_source",
                         detail="{}", call_id="c1")
        yield AgentEvent("tool_end", "Agent · read_source", tool="read_source",
                         detail="ok", call_id="c1", ok=True, duration=0.1)
        await gate.wait()  # agent is "working" on the next step here
        yield AgentEvent("usage", "", tokens_in=1, tokens_out=1)
        yield AgentEvent("final", "the review")

    monkeypatch.setattr("financial_research_assistant.tui.run_turn", streaming)
    app = AgentApp(fake=True, session_id="working")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "review"
        await pilot.press("enter")
        # Advance until the worker suspends at the gate (tool done, spinner re-armed).
        for _ in range(40):
            await pilot.pause()
            if list(app.query(ToolPanel)) and list(app.query(ProcessingWidget)):
                break
        assert app._busy
        assert len(list(app.query(ToolPanel))) == 1        # the tool finished
        assert len(list(app.query(ProcessingWidget))) == 1  # spinner re-armed in the gap

        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not app._busy
        assert list(app.query(ProcessingWidget)) == []      # no lingering spinner
        assert any(w.text == "the review" for w in app.query(AgentMessageWidget))


async def test_tool_panels_stay_above_post_tool_answer(monkeypatch, tmp_path):
    """Chronology regression: text streamed before a tool call stays in a bubble
    ABOVE the tool panel, and text streamed after it starts a NEW bubble BELOW —
    the post-tool answer must not float to the top by reusing the pre-tool
    bubble, and final must not duplicate the pre-tool text into the last bubble."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    async def streaming(*args, **kwargs):
        yield AgentEvent("token", "reading files ")
        yield AgentEvent("tool_start", "Agent · read_source", tool="read_source",
                         detail="{}", call_id="c1")
        yield AgentEvent("tool_end", "Agent · read_source", tool="read_source",
                         detail="ok", call_id="c1", ok=True, duration=0.4)
        yield AgentEvent("token", "the review")
        yield AgentEvent("usage", "", tokens_in=1, tokens_out=2)
        yield AgentEvent("final", "reading files the review")  # full concatenation

    monkeypatch.setattr("financial_research_assistant.tui.run_turn", streaming)
    app = AgentApp(fake=True, session_id="order")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "review"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        log = app.query_one("#log", VerticalScroll)
        ordered = [w for w in log.children if isinstance(w, (AgentMessageWidget, ToolPanel))]
        assert [type(w).__name__ for w in ordered] == [
            "AgentMessageWidget", "ToolPanel", "AgentMessageWidget",
        ]  # pre-tool text, then the panel, then the post-tool answer — in order
        assert ordered[0].text == "reading files "  # pre-tool bubble stays above
        assert ordered[2].text == "the review"       # post-tool answer, not duplicated
        assert ordered[2].duration is not None        # duration stamped on final


# -- regression: exit code + env resolution ---------------------------------
async def test_usage_deltas_accumulate_live_in_statusbar(monkeypatch, tmp_path):
    """Per-call usage deltas accumulate into the running in/out totals shown in
    the status bar (live), rather than the bar reflecting only the last delta."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    async def streaming(*args, **kwargs):
        yield AgentEvent("usage", "", tokens_in=100, tokens_out=20)  # model call 1
        yield AgentEvent("token", "hi")
        yield AgentEvent("usage", "", tokens_in=150, tokens_out=35)  # model call 2
        yield AgentEvent("usage", "", tokens_in=0, tokens_out=0)     # reconciliation
        yield AgentEvent("final", "hi")

    monkeypatch.setattr("financial_research_assistant.tui.run_turn", streaming)
    app = AgentApp(fake=True, session_id="usage-live")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "go"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._tok_in == 250   # 100 + 150 (+ 0)
        assert app._tok_out == 55   # 20 + 35
        assert "in 250 / out 55 tok" in statusbar_text(app)


async def test_ctx_pct_tracks_current_context_not_cumulative(monkeypatch, tmp_path):
    """ctx% must reflect the CURRENT context size (this turn's input), not the
    cumulative session input — otherwise it pegs at 100% forever. Two turns whose
    input is identical must show the same ctx%, even though cumulative in-tokens
    doubled."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "100000")  # clean 100k window

    async def streaming(*args, **kwargs):
        yield AgentEvent("usage", "", tokens_in=25_000, tokens_out=10)
        yield AgentEvent("usage", "", tokens_in=0, tokens_out=0, context_tokens=25_000)
        yield AgentEvent("final", "ok")

    monkeypatch.setattr("financial_research_assistant.tui.run_turn", streaming)
    app = AgentApp(fake=True, session_id="ctx-current")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "one"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._ctx_pct == 25          # 25k / 100k
        app.query_one(CommandInput).value = "two"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._tok_in == 50_000       # cumulative doubled...
        assert app._ctx_pct == 25          # ...but ctx% stayed with the current context


async def test_ctx_pct_resets_on_auto_compaction(monkeypatch, tmp_path):
    """An auto-compaction status (context_tokens=0) drops the ctx% gauge the instant
    context shrinks — mid-turn — instead of the footer staying pinned at its old
    value until the turn's usage arrives."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_CONTEXT_WINDOW", "100000")

    async def streaming(*args, **kwargs):
        # First a normal turn drives ctx% up.
        yield AgentEvent("usage", "", tokens_in=90_000, tokens_out=5, context_tokens=90_000)
        yield AgentEvent("final", "big")

    monkeypatch.setattr("financial_research_assistant.tui.run_turn", streaming)
    app = AgentApp(fake=True, session_id="ctx-compact")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "grow"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._ctx_pct == 90

        # Next turn auto-compacts before running: the status snapshot resets ctx%.
        async def compacting(*args, **kwargs):
            yield AgentEvent("status", "auto-compacted 12 older message(s)", context_tokens=0)
            yield AgentEvent("final", "small")

        monkeypatch.setattr("financial_research_assistant.tui.run_turn", compacting)
        app.query_one(CommandInput).value = "again"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._ctx_pct == 0
        assert "ctx 0% of 100k" in statusbar_text(app)


# -- think tool → reasoning routing -----------------------------------------






async def test_pending_tool_panel_aborted_on_error(monkeypatch, tmp_path):
    """A turn that errors while a tool call is outstanding (no tool_end) must
    finalize the panel — stop its spinner — instead of ticking forever."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    async def streaming(*args, **kwargs):
        yield AgentEvent("tool_start", "Agent · read_source", tool="read_source",
                         detail="{}", call_id="c1")
        yield AgentEvent("error", "boom")  # errors before the tool_end arrives

    monkeypatch.setattr("financial_research_assistant.tui.run_turn", streaming)
    app = AgentApp(fake=True, session_id="err-tool")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "review"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        panels = list(app.query(ToolPanel))
        assert len(panels) == 1
        assert panels[0]._done is True           # spinner stopped, not orphaned
        assert "interrupted" in panels[0].title   # shows the aborted status
        assert app._pending == {}                 # cleared for the next turn
        assert app._exit_code == 1                # error propagated




async def test_thinking_panel_ordered_above_post_think_output(monkeypatch, tmp_path):
    """A thought that precedes output stays ABOVE it: text before a reasoning
    event is one bubble, the 💭 panel comes next, and text after starts a NEW
    bubble below — the thought never sinks under output it came before."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    async def streaming(*args, **kwargs):
        yield AgentEvent("token", "let me look. ")
        yield AgentEvent("reasoning", "read a.py first", agent="Agent")
        yield AgentEvent("token", "the review")
        yield AgentEvent("final", "let me look. the review")

    monkeypatch.setattr("financial_research_assistant.tui.run_turn", streaming)
    app = AgentApp(fake=True, session_id="think-order")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "review"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        log = app.query_one("#log", VerticalScroll)
        ordered = [w for w in log.children if isinstance(w, (AgentMessageWidget, ThinkingPanel))]
        assert [type(w).__name__ for w in ordered] == [
            "AgentMessageWidget", "ThinkingPanel", "AgentMessageWidget",
        ]
        assert ordered[0].text == "let me look. "  # pre-think bubble stays above
        assert ordered[2].text == "the review"      # post-think bubble below panel
        assert "✅" in ordered[1].title               # thinking marked done




async def test_thinking_panel_shows_duration_only_when_tracked(monkeypatch, tmp_path):
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="t-dur")
    async with app.run_test() as pilot:
        log = app.query_one("#log", VerticalScroll)
        p = ThinkingPanel(AgentEvent("reasoning", "plan", agent="Agent", duration=0.7))
        await log.mount(p)
        await pilot.pause()
        assert "(0.7s)" in p.title and "✅" in p.title
        p2 = ThinkingPanel(AgentEvent("reasoning", "plan", agent="Agent"))  # no duration
        await log.mount(p2)
        await pilot.pause()
        assert "(0" not in p2.title and "✅" in p2.title  # no duration segment
async def test_tui_good_command_records_feedback(monkeypatch, tmp_path):
    """The /good TUI command stores the last exchange as an exemplar memory."""
    monkeypatch.setenv("MEMORY_BACKEND", "local")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_USER", "tuiuser")
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path / "sess"))
    from financial_research_assistant.memory import get_memory

    app = AgentApp(fake=True, session_id="fb-tui")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "what are my top positions?"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one(CommandInput).value = "/good clear and well-structured"
        await pilot.press("enter")
        await pilot.pause()
    exemplars = [e for e in get_memory().all() if e["kind"] == "exemplar"]
    assert len(exemplars) == 1
    assert "top positions" in exemplars[0]["text"]
    assert "Note: clear and well-structured" in exemplars[0]["text"]


async def test_tui_memory_command_lists_and_prunes(monkeypatch, tmp_path):
    """/memory lists stored memories (no crash) and /memory forget TEXT prunes
    the matching ones from the store."""
    monkeypatch.setenv("MEMORY_BACKEND", "local")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_USER", "memtui")
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path / "sess"))
    from financial_research_assistant.memory import get_memory

    mem = get_memory()
    mem.save("My base currency is USD", "preference")
    mem.save("Watchlist includes NVDA", "holding")

    app = AgentApp(fake=True, session_id="mem-tui")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "/memory"
        await pilot.press("enter")
        await pilot.pause()
        assert len(get_memory().all()) == 2  # listing doesn't mutate
        app.query_one(CommandInput).value = "/memory forget watchlist"
        await pilot.press("enter")
        await pilot.pause()
    remaining = [e["text"] for e in get_memory().all()]
    assert remaining == ["My base currency is USD"]
# -- busy guards + session integrity ----------------------------------------


async def test_session_commands_refused_while_busy(monkeypatch, tmp_path):
    """/new, /clear and /resume while a turn is running are refused (like
    /compact): they mutate the session/log the running turn is streaming into —
    worst case its transcript entry would land in the WRONG session."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    sessions.log_turn("elsewhere", "q", "a")
    app = AgentApp(fake=True, session_id="busy-guard")
    async with app.run_test() as pilot:
        app._busy = True  # a turn is (notionally) running
        box = app.query_one(CommandInput)
        for cmd in ("/new", "/clear", "/resume elsewhere"):
            box.value = cmd
            await pilot.press("enter")
            await pilot.pause()
        assert app.session_id == "busy-guard"        # /new and /resume refused
        text = log_text(app)
        assert text.count("busy — wait") == 3        # each command told the user
        assert text.count(_READY) == 1               # /clear refused: log intact


async def test_turn_logs_to_the_session_it_started_in(monkeypatch, tmp_path):
    """The transcript entry goes to the session that STARTED the turn, even if
    self.session_id changes mid-turn (belt-and-braces under the busy guard)."""
    import asyncio

    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    gate = asyncio.Event()

    async def streaming(*args, **kwargs):
        await gate.wait()
        yield AgentEvent("final", "late answer")

    monkeypatch.setattr("financial_research_assistant.tui.run_turn", streaming)
    app = AgentApp(fake=True, session_id="origin")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "slow question"
        await pilot.press("enter")
        await pilot.pause()
        app.session_id = "hijacked"  # simulate any future mid-turn switch
        gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert [t["answer"] for t in sessions.read_transcript("origin")] == ["late answer"]
    assert sessions.read_transcript("hijacked") == []


async def test_cancel_during_compaction_cancels_nothing(monkeypatch, tmp_path):
    """Esc during /compact must not claim '⏹ turn cancelled' (compaction runs on
    its own worker; _turn_worker may be a stale handle from a finished turn) —
    it notes that compaction can't be cancelled and cancels nothing."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="compact-esc")
    async with app.run_test() as pilot:
        worker = RecordingWorker()
        app._busy = True
        app._compacting = True
        app._turn_worker = cast(Worker, worker)  # stale handle from an old turn
        app.action_cancel_turn()
        await pilot.pause()
        assert worker.cancelled is False
        assert "turn cancelled" not in log_text(app)
        assert "compacting" in log_text(app)


async def test_turn_worker_cleared_when_turn_ends(monkeypatch, tmp_path):
    """The worker handle is dropped at turn end so a later Esc can never
    'cancel' a finished worker."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="worker-clear")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "hi"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._turn_worker is None


async def test_exit_code_resets_after_a_clean_turn(monkeypatch, tmp_path):
    """One errored turn must not make /quit exit non-zero forever: a later
    clean turn resets the exit code."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    async def erroring(*args, **kwargs):
        yield AgentEvent("error", "boom")

    monkeypatch.setattr("financial_research_assistant.tui.run_turn", erroring)
    app = AgentApp(fake=True, session_id="exit-reset")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "fail"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._exit_code == 1

        async def fine(*args, **kwargs):
            yield AgentEvent("final", "all good")

        monkeypatch.setattr("financial_research_assistant.tui.run_turn", fine)
        app.query_one(CommandInput).value = "succeed"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._exit_code == 0


# -- /resume state restoration ----------------------------------------------


async def test_resume_resets_tokens_and_restores_rating_state(monkeypatch, tmp_path):
    """/resume starts the resumed conversation with fresh token/cost counters
    (like /new) and points /copy·/good·/bad at the replayed last exchange."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    sessions.log_turn("beta", "hello from beta", "beta answer 42")
    app = AgentApp(fake=True, session_id="alpha2")
    async with app.run_test() as pilot:
        app._tok_in, app._tok_out, app._tok_cache = 500, 60, 7
        box = app.query_one(CommandInput)
        box.value = "/resume beta"
        await pilot.press("enter")
        await pilot.pause()
        assert (app._tok_in, app._tok_out, app._tok_cache) == (0, 0, 0)
        assert app._last_user == "hello from beta"
        assert app._last_answer == "beta answer 42"


# -- feedback + import styling ----------------------------------------------


async def test_tui_bad_command_reports_thumbs_down(monkeypatch, tmp_path):
    """/bad acknowledges with 👎/'avoid' — not the /good 👍 message."""
    monkeypatch.setenv("MEMORY_BACKEND", "local")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("MEMORY_USER", "tuibad")
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path / "sess"))

    app = AgentApp(fake=True, session_id="fb-bad")
    async with app.run_test() as pilot:
        app.query_one(CommandInput).value = "how risky is my portfolio?"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.query_one(CommandInput).value = "/bad too vague"
        await pilot.press("enter")
        await pilot.pause()
        text = log_text(app)
        assert "noted 👎" in text and "avoid" in text
        assert "noted 👍" not in text


async def test_import_missing_dep_hint_styled_as_error(monkeypatch, tmp_path):
    """The OFX missing-dependency hint renders in the error style like the other
    import failures (it starts with neither 'No file' nor 'Could not')."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="imp-ofx")
    async with app.run_test():
        seen: list[tuple[str, str]] = []
        app._line = lambda text, style="", indent=0: seen.append((text, style))

        app._show_import_result(
            "OFX/QFX import needs the optional 'ofxtools' package. Install it "
            "with: pip install 'financial-research-assistant[ofx]'"
        )
        assert seen and all(style == "dim red" for _, style in seen)
        assert any("ofxtools" in text for text, _ in seen)

        seen.clear()
        app._show_import_result("Imported 2 trade(s) for U123")  # success stays dim
        assert seen and all(style == "dim" for _, style in seen)


# -- late tool_end fallback ---------------------------------------------------


async def test_late_tool_end_panel_honors_expand_all(monkeypatch, tmp_path):
    """A tool_end with no matching tool_start (fallback panel) still honors the
    Ctrl+O expand-all preference instead of always mounting collapsed."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    async def streaming(*args, **kwargs):
        yield AgentEvent("tool_end", "Agent · x", tool="x", detail="ok",
                        call_id="ghost", ok=True, duration=0.1)
        yield AgentEvent("final", "done")

    monkeypatch.setattr("financial_research_assistant.tui.run_turn", streaming)
    app = AgentApp(fake=True, session_id="late-end")
    async with app.run_test() as pilot:
        app._tools_collapsed = False  # user pressed Ctrl+O: expand all
        app.query_one(CommandInput).value = "go"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        panels = list(app.query(ToolPanel))
        assert len(panels) == 1
        assert panels[0].collapsed is False


# -- /model override clearing -------------------------------------------------


async def test_model_default_clears_override(monkeypatch, tmp_path):
    """/model default resets the override back to the configured model."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="model-clear")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        box.value = "/model gpt-4o-custom"
        await pilot.press("enter")
        await pilot.pause()
        assert app.model_override == "gpt-4o-custom"

        box.value = "/model default"
        await pilot.press("enter")
        await pilot.pause()
        assert app.model_override is None
        assert "override cleared" in log_text(app)


# -- follow-aware autoscroll + render throttle + log cap ----------------------


async def test_autoscroll_pauses_when_scrolled_up_and_resumes_at_bottom(
    monkeypatch, tmp_path
):
    """Scrolling away from the bottom pauses streaming auto-follow; scrolling
    back to the bottom resumes it. A scroll gesture that never leaves the bottom
    (nothing to scroll yet) must NOT pause following."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="follow")
    async with app.run_test() as pilot:
        assert app._follow is True
        app.action_scroll_log_page_up()   # log fits on screen: still pinned
        await pilot.pause()               # (offset lands on the next refresh)
        assert app._follow is True

        for i in range(120):              # overflow the viewport
            app._line(f"line {i}")
        await pilot.pause()
        app.action_scroll_log_page_up()
        await pilot.pause()
        assert app._follow is False       # user moved away from the bottom
        app.action_scroll_log_end()
        assert app._follow is True        # back at the bottom → follow again


async def test_live_bubble_throttles_render_but_text_is_immediate(
    monkeypatch, tmp_path
):
    """A live bubble defers re-renders to its ~10 Hz flush timer (O(n²) guard)
    while .text updates immediately; finalize/set_duration stop the timer. A
    non-live (replayed) bubble renders synchronously and never ticks."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    app = AgentApp(fake=True, session_id="throttle")
    async with app.run_test() as pilot:
        log = app.query_one("#log", VerticalScroll)
        w = AgentMessageWidget("", live=True)
        await log.mount(w)
        await pilot.pause()
        w.append_text("hello")
        assert w.text == "hello"          # data immediate…
        assert w._dirty is True           # …render deferred to the throttle
        await pilot.pause(0.25)           # ≥ one 0.1s flush tick
        assert w._dirty is False          # flushed
        w.set_duration(1.2)
        assert w._render_timer is None    # finalized: no more ticking

        w2 = AgentMessageWidget("replay answer")
        await log.mount(w2)
        await pilot.pause()
        assert w2._render_timer is None   # replayed bubbles never tick


async def test_log_trims_oldest_widgets_past_cap(monkeypatch, tmp_path):
    """The log keeps at most _LOG_CAP widgets — the oldest are dropped (the
    transcript file keeps full history)."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    monkeypatch.setattr("financial_research_assistant.tui._LOG_CAP", 20)
    app = AgentApp(fake=True, session_id="trim")
    async with app.run_test() as pilot:
        for i in range(30):
            app._line(f"row {i}")
        await pilot.pause()
        log = app.query_one("#log", VerticalScroll)
        assert len(log.children) <= 21
        text = log_text(app)
        assert "row 29" in text           # newest kept
        assert "row 0" not in text        # oldest trimmed


async def test_export_accepts_full_path(monkeypatch, tmp_path):
    """/export with a full path writes there (parents created) instead of the
    session store."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path / "store"))
    app = AgentApp(fake=True, session_id="exp-path")
    async with app.run_test() as pilot:
        box = app.query_one(CommandInput)
        box.value = "capture me"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        dest = tmp_path / "elsewhere" / "deep" / "out.html"
        box.value = f"/export {dest}"
        await pilot.press("enter")
        await pilot.pause()
        assert dest.exists()
        assert "capture me" in dest.read_text(encoding="utf-8")


def test_reply_markdown_code_block_has_no_background():
    """Agent-reply code blocks (where a monospace table is pasted) render with NO
    background box, so the table sits flush on the app background — while the table
    text and 🟢/🔴 dots are preserved."""
    import io
    import re

    from rich.console import Console
    from rich.markdown import Markdown

    from financial_research_assistant.tui import _ReplyMarkdown

    md = "**Agent:**\n```\nTicker  Value  Status\nVOO     100    🟢\n```"

    def render(cls):
        buf = io.StringIO()
        Console(file=buf, force_terminal=True, color_system="truecolor", width=60).print(cls(md))
        return buf.getvalue()

    # Default Markdown paints a code-block background; _ReplyMarkdown must not.
    assert re.search(r"48;[25];", render(Markdown))          # baseline has bg
    flush = render(_ReplyMarkdown)
    assert not re.search(r"48;[25];", flush)                 # ours has none
    assert "VOO" in flush and "🟢" in flush                   # content preserved


async def test_tui_alert_event_toasts_and_leaves_a_transcript_line(monkeypatch, tmp_path):
    """A fired alert raises a toast AND writes a durable 🔔 line. Both matter:
    the toast lands while the user is reading something else, and the line
    survives it — the digest that evaluated the rule sits in a tool panel that
    is collapsed by default, so a dismissed toast would leave nothing visible."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    app = AgentApp(fake=True, session_id="alerting")
    async with app.run_test() as pilot:
        raised: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            type(app), "notify", lambda self, msg, **kw: raised.append((msg, kw))
        )

        app._notify_alert("AAPL down 6.2% — now 180.10")
        await pilot.pause()

        assert raised == [(
            "AAPL down 6.2% — now 180.10",
            {"title": "🔔 Alert triggered", "severity": "warning", "timeout": 10},
        )]
        assert "🔔 AAPL down 6.2% — now 180.10" in log_text(app)


async def test_tui_alert_line_survives_a_failing_toast(monkeypatch, tmp_path):
    """A toast is best-effort: if notify() raises, the alert must still reach the
    transcript rather than taking the turn down with it."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))

    def boom(self, msg, **kw):
        raise RuntimeError("no screen")

    app = AgentApp(fake=True, session_id="alerting-broken")
    async with app.run_test() as pilot:
        monkeypatch.setattr(type(app), "notify", boom)
        app._notify_alert("TSLA at 195.00")
        await pilot.pause()
        assert "🔔 TSLA at 195.00" in log_text(app)


async def test_tui_alert_makes_a_sound_and_can_be_silenced(monkeypatch, tmp_path,
                                                           _never_actually_play_audio):
    """A fired alert both rings the bell and plays a sound file — the bell alone
    is a BEL byte that many terminals drop. FINANCIAL_RESEARCH_ALERT_SOUND=0
    silences both while the toast and the transcript line still land."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    monkeypatch.delenv("FINANCIAL_RESEARCH_ALERT_SOUND", raising=False)

    app = AgentApp(fake=True, session_id="alert-bell")
    async with app.run_test() as pilot:
        rings: list[int] = []
        toasts: list[str] = []
        monkeypatch.setattr(type(app), "bell", lambda self: rings.append(1))
        monkeypatch.setattr(type(app), "notify", lambda self, msg, **kw: toasts.append(msg))

        app._notify_alert("AAPL down 6.2%")
        await pilot.pause()
        assert rings == [1]
        assert len(_never_actually_play_audio) == 1  # a player was launched

        monkeypatch.setenv("FINANCIAL_RESEARCH_ALERT_SOUND", "0")
        app._notify_alert("TSLA at 195.00")
        await pilot.pause()
        assert rings == [1]                          # no new ring
        assert len(_never_actually_play_audio) == 1  # and no new sound
        assert toasts == ["AAPL down 6.2%", "TSLA at 195.00"]  # toast unaffected
        assert "🔔 TSLA at 195.00" in log_text(app)


async def test_tui_alert_survives_a_failing_bell(monkeypatch, tmp_path):
    """Bell and toast are independently best-effort: a bell that raises must
    neither sink the turn nor swallow the toast that follows it."""
    monkeypatch.setenv("FINANCIAL_RESEARCH_SESSIONS_DIR", str(tmp_path))
    monkeypatch.delenv("FINANCIAL_RESEARCH_ALERT_SOUND", raising=False)

    def boom(self):
        raise RuntimeError("no driver")

    app = AgentApp(fake=True, session_id="alert-bell-broken")
    async with app.run_test() as pilot:
        toasts: list[str] = []
        monkeypatch.setattr(type(app), "bell", boom)
        monkeypatch.setattr(type(app), "notify", lambda self, msg, **kw: toasts.append(msg))

        app._notify_alert("NVDA reports earnings 2026-01-08")
        await pilot.pause()

        assert toasts == ["NVDA reports earnings 2026-01-08"]
        assert "🔔 NVDA reports earnings 2026-01-08" in log_text(app)
