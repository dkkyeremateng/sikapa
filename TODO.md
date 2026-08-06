# TODO

Open items from the scheduled-tasks + delivery-channels work (2026-08-06).
Unchecked boxes are pending; what shipped is described in README → "Scheduled
tasks" and in the git history.

## Blocking — nothing runs scheduled work across a reboot

- [ ] **Pick a permanent runner.** Right now the queue only drains while
      `financial-research-assistant --watch 60` is running in a terminal, which dies
      on reboot or when the window closes. The launchd agent is written and
      installed at `~/Library/LaunchAgents/com.teckdroids.fra-scheduler.plist` but
      **unloaded**, because it fails on this machine:

      ```
      PermissionError: [Errno 1] Operation not permitted: .../.venv/pyvenv.cfg
      ```

      That is macOS TCC, not file permissions — the same file reads fine from a
      shell. The repo lives under `~/Documents`, which `launchd`-spawned processes
      cannot read without an explicit grant. Three ways out:

      1. **Move the repo out of `~/Documents`** (recommended, e.g. `~/Dev/`), then
         `uv venv && uv sync` (the console scripts hard-code absolute paths), update
         the two paths in the plist, and `launchctl load -w …`. Nothing outside the
         venv references the old path — checked `.env`, the config files and the git
         remotes — but anything *outside* this repo (editor workspaces, shell
         aliases) would need updating.
      2. **Grant Full Disk Access** to
         `/Users/teckdroids/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3`
         in System Settings › Privacy & Security, then load the agent. One GUI step.
         Broad (every uv-python script gains Documents access) and brittle: a uv
         Python upgrade changes that path and the grant silently stops applying.
      3. **Keep `--watch`** and accept restarting it after every reboot.

- [x] **Restart the running watcher.** ~~The process running as of 17:2x predates the
      task-framing fix.~~ Done 2026-08-06: restarted at 18:31:38 (PID 28492), which
      postdates every change, so it carries the task framing, the non-answer gate,
      the redelivery outbox and the batch-cap logging. Verified: heartbeat fresh,
      `runner_is_live()` True (so `schedule_task` no longer warns), queue empty and
      nothing parked awaiting delivery.

      Note this has to be repeated after any change to `scheduler.py`, `tasks.py` or
      `channels.py` — a long-lived `--watch` process keeps running the code it
      started with. The launchd agent would not have this problem: it starts a fresh
      process per tick.

## Ship

- [x] **Commit and merge.** ~~The whole feature is uncommitted.~~ Done 2026-08-06:
      `254bd20` on `feat/scheduled-tasks`, merged to `main` as `85975a6` (`--no-ff`),
      15 files / +2,668 lines — `tasks.py`, `channels.py`, `telegram.py`,
      `scheduler.py`, plus edits to `catalog.py`, `graph.py`, `main.py`,
      `conftest.py`, `README.md`, `.env.example`, `TODO.md` and four test modules.
      633 tests pass, pyright clean on merged `main`; working tree clean. Local only
      — this repo has no remote configured, so nothing is pushed.

## Optional / not yet enabled

- [ ] **Inbound Telegram.** `TELEGRAM_ALLOWED_CHAT_IDS` is still commented out in
      `.env`, so the bot delivers but cannot be messaged. Uncomment it with your own
      chat id to reply from the phone (`/tasks`, `/cancel <id>`, or any prompt).
      Deliberately opt-in: an inbound message is untrusted text driving a tool-using
      agent that can read imported statements and positions. The read-only broker
      filter still applies — it can research and read the account, never trade.

## Unverified

- [ ] **A real model calling `schedule_task`.** The rule making it mandatory for
      anything in the future is a system-prompt change, and the scripted fake model
      ignores prompts — so it has never been observed end to end. Ask the live agent
      to "monitor X's earnings tomorrow" and confirm it calls the tool instead of
      promising to check back.
- [ ] **The task-framing fix under a real model.** Same reason. The regression it
      fixes is real and was observed (task s1 replied "I don't have a record of that
      research" and that non-answer was delivered as a success), but the fix itself
      has only been verified by test.

## Known gaps in the feature

All three closed 2026-08-06 (644 tests, pyright clean).

- [x] ~~A task is judged successful whenever the model returns *any* text.~~ A reply
      that opens with "I don't have a record of that" / "could you clarify", or that
      is under 40 characters, now counts as a failure and retries — the exact
      regression seen on task s1. The check is deterministic and deliberately
      shallow (opening lines and short replies only): judging quality properly still
      means a second model call per task, which remains unbuilt. **Residual risk:** a
      false positive costs one retry; a real report that merely *mentions* a missing
      record later is untouched, and there is a test pinning that.
      Also: a retry that is still pending is no longer pushed to you, so one broken
      task costs one notification instead of three.
- [x] ~~A tick caps at 10 tasks.~~ It still does — that is the right behaviour — but
      the cap is no longer silent: the runner logs `+N deferred to the next tick`,
      and `FINANCIAL_RESEARCH_TASK_BATCH` widens it.
- [x] ~~Delivery is best-effort and never retried.~~ An answer that reached no
      channel is parked on its task and re-sent at the top of every later tick,
      before any new model call, for up to 8 attempts (redelivery is free — the
      model call is already spent). `--tasks` shows it as *answer waiting to be
      delivered*; after 8 attempts it gives up and leaves the answer in the log.
