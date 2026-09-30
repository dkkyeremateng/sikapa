"""Registries the always-on service reads and feature modules fill.

The service (``scheduler.serve``) runs loops, answers slash commands and prints a
status; the features that provide those — reports, the event watchers, the
recommender, the autonomy switch — live in their own modules. Registering through
this dependency-free module keeps the arrow one-way: features import ``hooks``,
the scheduler imports ``hooks``, and neither has to import the other.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import asyncio
import importlib

#: A model-free chat command: ``(argument, fake) -> reply``.
CommandHandler = Callable[[str, bool], Awaitable[str]]

#: name -> (handler, one-line help)
COMMANDS: dict[str, tuple[CommandHandler, str]] = {}

#: An extra service loop: ``(stop, fake) -> None``, returning when ``stop`` is set.
ServiceLoop = Callable[[asyncio.Event, bool], Awaitable[None]]
SERVICE_LOOPS: dict[str, ServiceLoop] = {}

#: A line for `/status`: ``() -> str`` ("" to show nothing).
STATUS_LINES: dict[str, Callable[[], str]] = {}

#: Modules that register into the above when imported.
FEATURE_MODULES = ("jobs", "periodic", "guardrails", "autonomy", "profile", "watchers",
                   "recommend")


def register_command(name: str, handler: CommandHandler, help_line: str) -> None:
    COMMANDS[name.lower().lstrip("/")] = (handler, help_line)


def register_service_loop(name: str, loop: ServiceLoop) -> None:
    SERVICE_LOOPS[name] = loop


def register_status_line(name: str, fn: Callable[[], str]) -> None:
    STATUS_LINES[name] = fn


def load_feature_modules() -> None:
    """Import every feature module that exists, so its registrations are in.

    A module that isn't there yet is skipped; one that fails to import for any
    other reason raises, because a half-registered service is worse than one that
    refuses to start."""
    for name in FEATURE_MODULES:
        full = f"{__package__}.{name}"
        try:
            importlib.import_module(full)
        except ModuleNotFoundError as exc:
            if exc.name != full:
                raise
