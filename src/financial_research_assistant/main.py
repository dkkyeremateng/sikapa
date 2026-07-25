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

import argparse
import asyncio
import json
import os
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


def _run_subcommand(args) -> int | None:
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

    if args.flex_sync:
        # Model-free IBKR Flex Web Service pull (token from IBKR_FLEX_TOKEN). Saves
        # the statement XML; cron-friendly. Manual CSV import is unaffected.
        from .flex import flex_sync

        print(flex_sync(query_id=args.flex_query_id or None))
        return 0

    if args.research:
        return _handle_research(args)

    return None


def _handle_research(args) -> int:
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
    tools: list[dict] = []  # ordered tool trajectory for --trace / trajectory eval
    usage = {"tokens_in": 0, "tokens_out": 0, "tokens_cache": 0, "cost_usd": None}
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
        elif ev.kind == "usage" and (ev.tokens_in or ev.tokens_out or ev.tokens_cache):
            # Usage streams as per-call deltas; accumulate so the trace + line
            # report the turn total, not just the last call.
            usage["tokens_in"] += ev.tokens_in
            usage["tokens_out"] += ev.tokens_out
            usage["tokens_cache"] += ev.tokens_cache
            cost = cost_usd(
                model, usage["tokens_in"], usage["tokens_out"], usage["tokens_cache"]
            )
            usage["cost_usd"] = round(cost, 6) if cost is not None else None
            cache = f" / cache {usage['tokens_cache']}" if usage["tokens_cache"] else ""
            dollars = f" ≈ ${cost:.4f}" if cost is not None else ""
            print(
                f"• tokens: in {usage['tokens_in']} / out {usage['tokens_out']}{cache}{dollars}",
                file=sys.stderr,
            )
        elif ev.kind == "final":
            final = ev.text
        elif ev.kind == "error":
            print(f"error: {ev.text}", file=sys.stderr)
            return 1
    if final:
        sessions.log_turn(session_id, prompt, final)
    if trace_path is not None:
        # Machine-readable trajectory for eval_type "trajectory" (evaluate.py).
        with open(trace_path, "w") as f:
            json.dump({"query": prompt, "answer": final, "tools": tools, "usage": usage}, f)
    print(final)
    return 0


if __name__ == "__main__":
    cli()
