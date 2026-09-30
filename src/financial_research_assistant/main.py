"""Entry point.

- ``financial-research-assistant``               -> Textual chat TUI
- ``financial-research-assistant --prompt "hi"`` -> headless one-shot (answer on
                                       stdout, status/reasoning lines on stderr)
- ``--once "hi"``                   -> TUI smoke mode: auto-submit one query
                                       and exit after the answer
- ``--fake``                        -> offline deterministic model (no API key)
- ``--session ID``                  -> conversation id (default "cli")
- ``--resume NAME``                 -> resume a saved session (replays it in
                                       the TUI; alias of --session for a saved
                                       one in headless)
- ``--list-sessions``               -> list saved sessions and exit
"""

from typing import Any
import argparse
import asyncio
import json
import sys

from dotenv import load_dotenv


def cli() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="financial-research-assistant",
        description=(
            "Financial research assistant over read-only IBKR market data: "
            "Textual TUI by default, headless with --prompt."
        ),
    )
    parser.add_argument(
        "--prompt", help="run one turn headlessly and print the answer"
    )
    parser.add_argument(
        "--once", help="TUI smoke mode: auto-submit one query and exit"
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="use the offline deterministic fake model",
    )
    parser.add_argument(
        "--session", default="cli", help='session id (default "cli")'
    )
    parser.add_argument(
        "--resume",
        metavar="NAME",
        help="resume a saved session (replay its transcript in the TUI)",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="list saved sessions and exit",
    )
    parser.add_argument(
        "--digest",
        action="store_true",
        help=(
            "print a portfolio monitoring digest (price movers, upcoming earnings "
            "and ex-dividends over your imported holdings) and exit — no model, "
            "cron-friendly"
        ),
    )
    parser.add_argument(
        "--account",
        default="",
        help="with --digest: scope to one account (default: newest import's)",
    )
    parser.add_argument(
        "--run-due",
        action="store_true",
        help=(
            "run every scheduled task that is due now, push each answer to the "
            "configured channels, then exit — the cron/launchd entry point. Does "
            "not read the Telegram inbox (--watch and --serve do)"
        ),
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "run the always-on service: scheduled jobs, the Telegram inbox and the "
            "watchers side by side, until SIGTERM/Ctrl-C. See deploy/README.md"
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help=(
            "show whether the always-on service is running and what is next, then "
            "exit; with --check, print nothing and exit 1 unless it is healthy"
        ),
    )
    parser.add_argument(
        "--reports-setup",
        action="store_true",
        help=(
            "schedule the default jobs (Flex sync, daily/weekly/monthly reports, "
            "ideas) that aren't already scheduled, then exit; safe to re-run"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="with --status: exit code only (for container health checks)",
    )
    parser.add_argument(
        "--watch",
        nargs="?",
        const=60.0,
        type=float,
        metavar="SECONDS",
        help=(
            "stay running and do what --run-due does every SECONDS (default 60) — "
            "for a machine where a cron entry is more trouble than a process"
        ),
    )
    parser.add_argument(
        "--schedule",
        metavar="WHEN|PROMPT",
        help=(
            "queue a task and exit, e.g. --schedule 'tomorrow 9am|Analyse NOMD Q3 "
            "results vs consensus'. Add a third field to repeat: "
            "'08:30|Pre-market brief|weekdays'; no model needed"
        ),
    )
    parser.add_argument(
        "--tasks",
        action="store_true",
        help="list scheduled tasks (with ids and outcomes) and exit; no model",
    )
    parser.add_argument(
        "--unschedule",
        metavar="ID",
        help="cancel a scheduled task by id (or 'all') and exit; no model",
    )
    parser.add_argument(
        "--theses",
        nargs="?",
        const="",
        metavar="SYMBOL",
        help=(
            "show the recorded directional calls and how the scored ones turned "
            "out, then exit; optionally narrowed to one ticker. Scoring itself "
            "happens on a runner tick (--run-due/--watch); no model"
        ),
    )
    parser.add_argument(
        "--notify-test",
        action="store_true",
        help=(
            "send a test message to the configured delivery channels and exit — "
            "checks TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID without scheduling anything"
        ),
    )
    parser.add_argument(
        "--flex-sync",
        action="store_true",
        help=(
            "pull an IBKR Activity statement via the Flex Web Service and save its "
            "XML, then exit — no model; token from IBKR_FLEX_TOKEN, query from "
            "IBKR_FLEX_QUERY_ID or --flex-query-id"
        ),
    )
    parser.add_argument(
        "--flex-query-id",
        default="",
        help="with --flex-sync: the Flex Query ID (overrides IBKR_FLEX_QUERY_ID)",
    )
    parser.add_argument(
        "--research",
        metavar="SYMBOL",
        help=(
            "generate a deep-research markdown report on SYMBOL (gathers price, "
            "fundamentals, analyst, earnings, risk, news; synthesizes and saves it), "
            "then exit. Needs a model (or --fake for an offline stub)"
        ),
    )
    parser.add_argument(
        "--memory",
        nargs="?",
        const="list",
        metavar="forget:TEXT",
        help=(
            "inspect long-term memory and exit: bare --memory lists stored facts; "
            "--memory forget:TEXT removes facts matching TEXT. Needs MEMORY_BACKEND "
            "set (e.g. MEMORY_BACKEND=local); no model"
        ),
    )
    parser.add_argument(
        "--login",
        nargs="?",
        const="",
        metavar="PROVIDER",
        help=(
            "sign in to a model provider and exit — stores the credential in "
            "~/.financial-research-assistant/auth.json (0600), which outranks the "
            "key in .env. Bare --login lists the providers. Pair with --tier to "
            "give the cheap/subagent model its own credential; no model needed"
        ),
    )
    parser.add_argument(
        "--logout",
        nargs="?",
        const="",
        metavar="PROVIDER",
        help=(
            "stop using the credential for --tier and exit; it stays stored, so "
            "switching back needs no re-login. --logout PROVIDER removes that "
            "provider's credential entirely"
        ),
    )
    parser.add_argument(
        "--tier",
        default="default",
        metavar="TIER",
        help='with --login/--logout: which model tier ("default", "quick", "subagent")',
    )
    parser.add_argument(
        "--no-thinking",
        dest="think",
        action="store_false",
        help="disable the model's reasoning trace (💭 panels)",
    )
    parser.add_argument(
        "--trace",
        metavar="FILE",
        help="with --prompt: write the tool trajectory + usage as JSON to FILE",
    )
    args = parser.parse_args()

    # Model-free early-exit subcommands (list/digest/memory/flex/research). Each
    # returns an exit code when it handled the run; None means "fall through to the
    # interactive/headless chat path below".
    rc = _run_subcommand(args)
    if rc is not None:
        sys.exit(rc)

    session_id = args.resume or args.session

    # Resuming replays the saved transcript, but the model's actual memory is only
    # restored when durable checkpointing is on. Nudge the user to the env var
    # rather than silently persisting sensitive conversations to disk by default.
    if args.resume:
        from .adapter import durable_checkpoints_enabled

        if not durable_checkpoints_enabled():
            print(
                "note: resuming replays the transcript, but the model's memory of the "
                "conversation isn't restored. Set FINANCIAL_RESEARCH_CHECKPOINT_DB=1 to "
                "persist and restore full conversation state across restarts.",
                file=sys.stderr,
            )

    if args.prompt is not None:
        sys.exit(
            asyncio.run(
                _headless(args.prompt, session_id, args.fake, args.think, args.trace)
            )
        )

    from .tui import AgentApp

    app = AgentApp(
        fake=args.fake, once=args.once, session_id=session_id, think=args.think
    )
    app.run()
    sys.exit(app.return_code or 0)


def _run_subcommand(args: argparse.Namespace) -> int | None:
    """Dispatch the model-free early-exit subcommands. Returns an exit code if one
    ran, or None to continue to the chat (TUI/headless) path."""
    if args.list_sessions:
        from . import sessions as _sessions

        found = _sessions.list_sessions()
        if not found:
            print("no saved sessions")
        for s in found:
            print(f"{s['name']:<28} {s['turns']} turns")
        return 0

    if args.digest:
        # Deterministic, model-free portfolio digest for cron/launchd. Works
        # offline against imported statements (needs no API key, no --fake).
        from .monitor import build_digest

        print(build_digest(account=args.account or None))
        return 0

    if args.memory is not None:
        return _handle_memory(args.memory)

    # Scheduled tasks. The store subcommands are model-free; --run-due/--watch need
    # a model only once they find work to run (an empty tick costs nothing).
    if args.tasks:
        from . import tasks as _tasks

        print(_tasks.list_scheduled_tasks())
        return 0

    if args.unschedule:
        from . import tasks as _tasks

        print(_tasks.cancel_scheduled_task(args.unschedule))
        return 0

    if args.theses is not None:
        from . import journal as _journal

        print(_journal.review_theses(args.theses))
        return 0

    if args.schedule:
        return _handle_schedule(args.schedule)

    if args.notify_test:
        from . import channels

        delivered, failed = channels.deliver(
            "🤖 Test message from the financial research assistant. "
            "Scheduled work will arrive here."
        )
        if delivered:
            print(f"sent to: {', '.join(delivered)}")
        if failed:
            print(f"failed on: {', '.join(failed)}", file=sys.stderr)
        if not delivered and not failed:
            print(
                "no delivery channel is configured — set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID (see .env.example), or NOTIFY_CHANNELS=desktop",
                file=sys.stderr,
            )
        return 0 if delivered else 1

    if args.run_due or args.watch is not None:
        return _handle_scheduler(args)

    if args.serve:
        from . import scheduler

        asyncio.run(scheduler.serve(fake=args.fake))
        return 0

    if args.reports_setup:
        from . import jobs

        print("Default jobs:")
        print(jobs.describe_setup(jobs.ensure_default_jobs()))
        return 0

    if args.status:
        from . import scheduler

        status = scheduler.service_status()
        if args.check:
            return 0 if status["healthy"] else 1
        print(scheduler.format_status(status))
        return 0

    if args.flex_sync:
        # Model-free IBKR Flex Web Service pull (token from IBKR_FLEX_TOKEN). Saves
        # the statement XML; cron-friendly. Manual CSV import is unaffected.
        from .flex import flex_sync

        print(flex_sync(query_id=args.flex_query_id or None))
        return 0

    if args.research:
        return _handle_research(args)

    if args.login is not None:
        return _handle_login(args.login, args.tier)

    if args.logout is not None:
        return _handle_logout(args.tier, args.logout)

    return None


def _handle_schedule(spec: str) -> int:
    """``--schedule 'WHEN|PROMPT[|REPEAT]'``.

    Pipe-separated rather than three flags: the whole point is pasting one line
    into a shell or a cron entry, and 'when' and 'what' belong together. A prompt
    containing a pipe still works — only the first and last fields are split off.
    """
    from . import channels, tasks

    parts = [p.strip() for p in spec.split("|")]
    if len(parts) < 2:
        print(
            "usage: --schedule 'WHEN|PROMPT[|REPEAT]'  e.g. "
            "--schedule 'tomorrow 9am|Analyse NOMD Q3 results vs consensus'",
            file=sys.stderr,
        )
        return 2
    when, prompt = parts[0], parts[1]
    repeat = "once"
    if len(parts) > 2 and parts[-1].lower() in tasks.REPEATS:
        repeat, prompt = parts[-1].lower(), "|".join(parts[1:-1])
    elif len(parts) > 2:
        prompt = "|".join(parts[1:])
    try:
        task = tasks.add_task(prompt, when, repeat)
    except ValueError as exc:
        print(f"could not schedule that: {exc}", file=sys.stderr)
        return 2
    from datetime import datetime

    due = datetime.fromisoformat(task["due"]).astimezone()
    print(f"scheduled [{task['id']}] for {due:%Y-%m-%d %H:%M} local"
          + ("" if repeat == "once" else f", repeating {repeat}"))
    print(f"  delivery: {channels.describe_targets(task['channel'])}")
    print("  run it with: --run-due (from cron), or leave --watch running")
    return 0


def _handle_scheduler(args: argparse.Namespace) -> int:
    """``--run-due`` (one pass) and ``--watch`` (a loop over the same pass)."""
    from . import scheduler

    try:
        if args.watch is not None:
            asyncio.run(scheduler.watch(float(args.watch), fake=args.fake))
            return 0
        results = asyncio.run(scheduler.run_due(fake=args.fake))
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
        return 130
    if not results:
        # Silent on stdout: this runs every few minutes from cron, and a line per
        # empty tick would bury the real output in the mail spool.
        print("nothing due", file=sys.stderr)
        return 0
    failed = [r for r in results if not r["ok"]]
    print(
        f"ran {len(results)} task(s), {len(failed)} failed", file=sys.stderr
    )
    return 1 if failed else 0


def _handle_login(provider: str, tier: str) -> int:
    """Run a provider's OAuth flow from the terminal. The flow itself is shared
    with the TUI — only these callbacks differ, which is the point of keeping them
    UI-neutral."""
    import webbrowser

    from . import auth, oauth

    if tier not in auth.SCOPES:
        print(f"unknown tier: {tier} (one of {', '.join(auth.SCOPES)})", file=sys.stderr)
        return 2
    if not provider:
        print("providers:")
        for name, label in oauth.choices():
            print(f"  {name:<14} {label}")
        print("\nsign in with: --login openrouter")
        return 0
    if oauth.get(provider) is None:
        print(
            f"unknown provider: {provider} (try {', '.join(oauth.names())})",
            file=sys.stderr,
        )
        return 2

    def on_auth(url: str) -> None:
        # Print before opening: on a headless box there is no browser to open and
        # the URL is the only way through.
        print(f"opening {url}", file=sys.stderr)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import getpass

    cb = oauth.LoginCallbacks(
        on_auth=on_auth,
        on_status=lambda s: print(s, file=sys.stderr),
        on_device_code=lambda code, url: print(f"enter code {code} at {url}", file=sys.stderr),
        on_prompt=lambda q: input(q),
        # No echo, and it stays out of shell history and any terminal capture.
        on_secret=lambda q: getpass.getpass(q),
    )
    try:
        oauth.login(provider, cb, scope=tier)
    except oauth.LoginError as exc:
        print(f"login failed: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nlogin cancelled", file=sys.stderr)
        return 1
    print(f"signed in to {provider} ({tier} tier)")
    return 0


def _handle_logout(tier: str, provider: str = "") -> int:
    from . import auth

    if provider:
        if auth.forget(provider):
            print(f"forgot the {provider} credential")
            return 0
        print(
            f"no stored credential for {provider} "
            f"(configured: {', '.join(auth.providers()) or 'none'})",
            file=sys.stderr,
        )
        return 2
    if tier not in auth.SCOPES:
        print(f"unknown tier: {tier} (one of {', '.join(auth.SCOPES)})", file=sys.stderr)
        return 2
    if auth.delete(tier):
        # The credential is kept so the tier can be pointed back at it.
        print(f"signed out ({tier} tier) — credential kept; --login or /models switches back")
        return 0
    print(f"no credential in use for the {tier} tier")
    return 0


def _handle_research(args: argparse.Namespace) -> int:
    """Deep-research report: gather → synthesize → save → reflect, then print.
    Needs a model (real synthesis); ``--fake`` produces an offline stub. The
    special value "portfolio" reports on the whole imported portfolio."""
    from .research import research_portfolio, research_ticker

    if args.research.strip().lower() in ("portfolio", "all"):
        print("Researching your portfolio — gathering sources…", file=sys.stderr)
        res = asyncio.run(research_portfolio(fake=args.fake))
    else:
        print(f"Researching {args.research.upper()} — gathering sources…", file=sys.stderr)
        res = asyncio.run(research_ticker(args.research, fake=args.fake))
    print(res["report"])
    print(f"\n[saved to {res['path']}]", file=sys.stderr)
    if res.get("lessons"):
        # Self-critique notes stored for next time (only when memory is on).
        print("[learned for next time:]", file=sys.stderr)
        for lesson in res["lessons"]:
            print(f"  • {lesson}", file=sys.stderr)
    return 0


def _handle_memory(value: str) -> int:
    """Handle ``--memory``: list stored facts, or ``forget:TEXT`` to prune them.
    Returns a process exit code."""
    from .memory import get_memory

    mem = get_memory()
    if mem is None:
        print(
            "Long-term memory is disabled. Set MEMORY_BACKEND=local (and optionally "
            "MEMORY_USER / MEMORY_DIR) to enable it.",
            file=sys.stderr,
        )
        return 1
    if value.lower().startswith("forget:"):
        n = mem.forget(value.split(":", 1)[1].strip())
        print(f"Forgot {n} memory item(s).")
        return 0
    entries = mem.all(include_superseded=True)
    if not entries:
        print("(no memories stored)")
        return 0
    for e in entries:
        ts = f" ({e['ts']})" if e.get("ts") else ""
        # Archived stale values (superseded by a newer one) are shown for audit but
        # never recalled.
        tag = f"[{e.get('kind', 'note')}·superseded]" if e.get("superseded") else f"[{e.get('kind', 'note')}]"
        print(f"{tag}{ts} {e['text']}")
    return 0


def _write_trace(
    trace_path: str,
    prompt: str,
    answer: str,
    tools: list[dict[str, Any]],
    usage: dict[str, Any],
    error: str = "",
) -> None:
    """Write the machine-readable trajectory for eval_type "trajectory"
    (evaluate.py).

    Written on the failing path too, carrying an ``error`` marker. An eval harness
    reads this file by the path it passed in, so a run that dies without writing it
    either raises FileNotFoundError (reported as a harness bug rather than a failed
    run) or — when the path is reused across runs, which is how a sweep works —
    silently hands back the PREVIOUS run's trajectory and scores that instead."""
    payload: dict[str, Any] = {
        "query": prompt, "answer": answer, "tools": tools, "usage": usage,
    }
    if error:
        payload["error"] = error
    with open(trace_path, "w") as f:
        json.dump(payload, f)


async def _headless(
    prompt: str,
    session_id: str,
    fake: bool,
    think: bool = True,
    trace_path: str | None = None,
) -> int:
    from . import alerts, sessions
    from .adapter import run_turn
    from .pricing import cost_usd
    from .tracing import traced

    from .adapter import _resolved_model

    model = "scripted-fake" if fake else _resolved_model(None)
    final = ""
    tools: list[dict[str, Any]] = []  # ordered tool trajectory for --trace / trajectory eval
    usage: dict[str, Any] = {"tokens_in": 0, "tokens_out": 0, "tokens_cache": 0,
                             "tokens_cache_write": 0, "cost_usd": None}
    async for ev in traced(
        run_turn(prompt, session_id, fake=fake, think=think),
        user_msg=prompt, session_id=session_id, model=model, fake=fake,
    ):
        if ev.kind == "status":
            print(f"• {ev.text}", file=sys.stderr)
        elif ev.kind == "reasoning":
            print(f"💭 {ev.text}", file=sys.stderr)
        elif ev.kind == "tool_start":
            print(f"▶ {ev.text}", file=sys.stderr)
        elif ev.kind == "tool_end":
            print(f"• {ev.text}", file=sys.stderr)
            tools.append({"name": ev.tool, "ok": ev.ok, "duration": round(ev.duration, 3)})
        elif ev.kind == "alert":
            print(f"🔔 {ev.text}", file=sys.stderr)
            # Only ring an interactive terminal: piped/redirected stderr is a log
            # file or a CI transcript, where a stray BEL byte is just noise. The
            # played sound has no such problem, so it isn't gated on the tty.
            if alerts.sound_enabled() and sys.stderr.isatty():
                print("\a", end="", file=sys.stderr, flush=True)
            alerts.play_alert_sound()
            alerts.notify_desktop(ev.text)
        elif ev.kind == "usage" and (
            ev.tokens_in or ev.tokens_out or ev.tokens_cache or ev.tokens_cache_write
        ):
            # Usage streams as per-call deltas; accumulate so the trace + line
            # report the turn total, not just the last call.
            usage["tokens_in"] += ev.tokens_in
            usage["tokens_out"] += ev.tokens_out
            usage["tokens_cache"] += ev.tokens_cache
            usage["tokens_cache_write"] += ev.tokens_cache_write
            cost = cost_usd(
                model, usage["tokens_in"], usage["tokens_out"],
                usage["tokens_cache"], usage["tokens_cache_write"],
            )
            usage["cost_usd"] = round(cost, 6) if cost is not None else None
            cache = f" / cache {usage['tokens_cache']}" if usage["tokens_cache"] else ""
            if usage["tokens_cache_write"]:
                cache += f" / cache-write {usage['tokens_cache_write']}"
            dollars = f" ≈ ${cost:.4f}" if cost is not None else ""
            print(
                f"• tokens: in {usage['tokens_in']} / out {usage['tokens_out']}{cache}{dollars}",
                file=sys.stderr,
            )
        elif ev.kind == "final":
            final = ev.text
        elif ev.kind == "error":
            print(f"error: {ev.text}", file=sys.stderr)
            if trace_path is not None:
                _write_trace(trace_path, prompt, final, tools, usage, error=ev.text)
            return 1
    if final:
        sessions.log_turn(session_id, prompt, final)
    if trace_path is not None:
        _write_trace(trace_path, prompt, final, tools, usage)
    print(final)
    return 0


if __name__ == "__main__":
    cli()
