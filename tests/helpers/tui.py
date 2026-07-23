from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

from financial_research_assistant.tui import AgentMessageWidget


def log_text(app) -> str:
    """The activity log rendered as one plain string (no config bar).

    AgentMessageWidget is a Static subclass whose renderable is a rich Group
    (no ``.plain``/``.visual.plain``), so scrape its ``.text`` directly instead
    of stringifying the Group; every other Static (including the user echo's
    ``.user-bubble`` child) renders to a plain string.
    """
    log = app.query_one("#log", VerticalScroll)
    parts = []
    for w in log.query(Static):
        if isinstance(w, AgentMessageWidget):
            parts.append(w.text)
        else:
            parts.append(str(w.render()))
    return "\n".join(parts)


def statusbar_text(app) -> str:
    """Plain text of the status bar, whose renderable is always a rich Text."""
    r = app.query_one("#statusbar", Static).render()
    return r.plain if isinstance(r, Text) else str(r)
