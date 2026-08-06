# TODO

Open items from the scheduled-tasks + delivery-channels work (2026-08-06).
Everything below is *pending*; what shipped is described in README → "Scheduled
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

- [ ] **Restart the running watcher** (Ctrl-C, then `--watch 60`) so it picks up the
      task-framing fix. The process running as of 2026-08-06 17:2x started before it.
      Do it when `--tasks` shows nothing waiting, so no task is mid-run.

## Ship

- [ ] **Commit and merge.** The whole feature is uncommitted: `tasks.py`,
      `channels.py`, `telegram.py`, `scheduler.py`, plus edits to `catalog.py`,
      `graph.py`, `main.py`, `conftest.py`, `README.md`, `.env.example` and four test
      modules. 633 tests pass, pyright clean.

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

## Known gaps in the feature (design notes, not bugs)

- A task is judged successful whenever the model returns *any* text — there is no
  quality check, so a confused-but-non-empty answer counts as done. Detecting that
  would need a judge call per task; deliberately not built.
- `--run-due` runs due tasks sequentially and caps a tick at 10, so a large backlog
  drains over several ticks rather than firing everything at once.
- Delivery is best-effort: a channel that is down is reported, never retried. The
  answer stays in the task record and falls back to stdout.
