#!/usr/bin/env bash
# Pull, rebuild, restart. Run on the server from the checkout (/opt/fra).
#
# Every deploy restarts the service on purpose: a long-lived process keeps
# running the code it started with, so an edit that is merged but not restarted
# is an edit that isn't live (TODO.md learned this the hard way with --watch).
set -euo pipefail

cd "$(dirname "$0")/.."
BRANCH="${FRA_BRANCH:-main}"
EXTRAS="${FRA_EXTRAS:-anthropic}"

# With a remote, pull it. Without one (this repo has none by default), the
# checkout is whatever was rsynced here — see deploy/README.md, "Updating".
if git remote get-url origin >/dev/null 2>&1; then
    git fetch --quiet origin "$BRANCH"
    git checkout --quiet "$BRANCH"
    git pull --quiet --ff-only origin "$BRANCH"
fi

docker build --quiet --build-arg EXTRAS="$EXTRAS" -t fra:latest -t "fra:$(git rev-parse --short HEAD)" .

# The restart waits for a running turn to finish (see fra.service ExecStop).
sudo systemctl restart fra
sleep 20
if ! docker exec fra financial-research-assistant --status --check; then
    echo "service did not come up healthy — check: journalctl -u fra -n 100" >&2
    exit 1
fi
echo "deployed $(git rev-parse --short HEAD)"
