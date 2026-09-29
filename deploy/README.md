# Running the agent always-on

One container runs `financial-research-assistant --serve` on a host that doesn't
sleep. It works through the scheduled jobs, answers Telegram, and runs the event
watchers, all at the same time. It has no open ports. Everything it knows lives
in one data directory, which is backed up nightly with encryption.

```
            outbound only
  ┌──────────────┐  ─────►  Telegram (long-poll), model API, Yahoo, SEC, IBKR Flex
  │ fra (docker) │
  │  --serve     │  ◄──  /srv/fra/data  (statements, journal, memory, tasks, reports)
  └──────────────┘  ◄──  /etc/fra/env   (secrets, 0600)
        ▲
   systemd: restart on exit, and on a hang (the service exits itself)
```

## 1. Host

Any always-on Linux box will do: a small VPS (1–2 vCPU, 2 GB) or a home server.
The model runs remotely, so the host only has to run Python. Everything below
assumes Ubuntu/Debian with Docker.

```bash
sudo apt-get update && sudo apt-get install -y docker.io age sqlite3 git rsync
sudo systemctl enable --now docker
# SSH keys only; nothing else needs to be reachable.
sudo ufw default deny incoming && sudo ufw allow OpenSSH && sudo ufw enable
```

## 2. Code

This repo has no git remote by default, so copy it over from the Mac (include
`.git`, since the deploy script tags images with the commit):

```bash
# on the Mac
rsync -az --delete --exclude .venv --exclude __pycache__ \
    ./ server:/opt/fra/
```

If you add a private remote later, `deploy/deploy.sh` pulls from it instead.

## 3. Secrets

```bash
sudo mkdir -p /etc/fra
sudo cp /opt/fra/deploy/server.env.example /etc/fra/env
sudo chmod 600 /etc/fra/env && sudo chown root:root /etc/fra/env
sudoedit /etc/fra/env
```

At minimum set the model key, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
`TELEGRAM_ALLOWED_CHAT_IDS` (your own chat id, which turns on two-way chat),
`IBKR_FLEX_TOKEN` / `IBKR_FLEX_QUERY_ID`, and `TZ`.

**Model credential.** Use an API key on the server. The `/login` OAuth session
stored in `auth.json` does work if you copy it over, but an unattended box has
nobody to sign in again if the refresh ever fails.

## 4. Data, and moving your existing state

```bash
sudo mkdir -p /srv/fra/data && sudo chown -R 10001:10001 /srv/fra/data && sudo chmod 700 /srv/fra/data
```

To carry over what the Mac already knows (imported statements, the thesis
journal, memory, alert rules, the investor profile), do this once, **with the
Mac's scheduler stopped** so the two don't both run the same jobs:

```bash
# on the Mac
rsync -az --exclude auth.json --exclude 'auth.lock' --exclude '*.lock' \
    ~/.financial-research-assistant/ server:/tmp/fra-data/
# on the server
sudo rsync -a /tmp/fra-data/ /srv/fra/data/ && sudo chown -R 10001:10001 /srv/fra/data && rm -rf /tmp/fra-data
```

From then on **the server's directory is the source of truth.** Run the TUI
against it (section 8) rather than against a second copy on the Mac.

## 5. First start

```bash
cd /opt/fra
sudo docker build --build-arg EXTRAS=anthropic -t fra:latest .
sudo cp deploy/systemd/fra.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now fra
journalctl -u fra -f            # watch it come up
sudo docker exec fra financial-research-assistant --status
sudo docker exec fra financial-research-assistant --reports-setup   # the default report jobs
```

Then send `/status` to your bot from your phone.

## 6. Knowing when it's down

A process that has died can't report its own death, so liveness is checked
from outside:

1. Create a check at a dead-man service such as [healthchecks.io](https://healthchecks.io)
   with a 10-minute period and 5-minute grace, alerting you by email or push.
2. Put its ping URL in `/etc/fra/env` as `FRA_HEALTHCHECK_URL` and restart.

The service pings it at most every 5 minutes from the job loop, so pings stop
when the process is dead or its loop is stuck. A stuck loop also makes the
service exit by itself after `FRA_WATCHDOG_MINUTES` (default 10), and systemd
restarts it.

## 7. Backups

```bash
age-keygen -o fra-backup.key       # do this on the MAC; keep the key off the server
# copy the "public key: age1..." line into /etc/fra/backup.env on the server:
sudo tee /etc/fra/backup.env >/dev/null <<'EOF'
FRA_BACKUP_AGE_RECIPIENT=age1...
FRA_BACKUP_DIR=/srv/fra/backups
FRA_BACKUP_KEEP=14
EOF
sudo chmod 600 /etc/fra/backup.env
sudo cp deploy/systemd/fra-backup.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now fra-backup.timer
sudo systemctl start fra-backup.service && ls -l /srv/fra/backups   # test it now
```

Sync `/srv/fra/backups` somewhere off the box (object storage, the Mac). To
restore: `age -d -i fra-backup.key fra-….tar.gz.age | tar -xz -C /srv/fra/data`.

## 8. Using it

- **Phone:** message the bot. `/status`, `/report daily`, `/ideas`, `/pause`,
  `/resume`, `/quiet 2h`, `/tasks`, `/cancel <id>`, or just ask a question.
- **Terminal:** `ssh -t server sudo docker exec -it fra financial-research-assistant`
  opens the TUI inside the running container, against the live state.

## 9. Updating

```bash
# on the Mac, after merging to main
rsync -az --delete --exclude .venv --exclude __pycache__ ./ server:/opt/fra/
ssh server /opt/fra/deploy/deploy.sh
```

`deploy.sh` rebuilds the image, restarts the service (a running turn gets 90 s
to finish), and fails loudly if it doesn't come back healthy.

## Mac fallback

If the host has to be the Mac, use `deploy/launchd/com.fra.agent.plist.template`
(instructions inside). Move the repo out of `~/Documents` first, and keep the
Mac awake on power. It works, but reboots, a closed lid or lost Wi-Fi all
pause the agent, and a daily report written for the US close arrives whenever
the Mac next wakes.
