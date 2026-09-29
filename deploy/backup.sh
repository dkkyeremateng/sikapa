#!/usr/bin/env bash
# Nightly encrypted backup of the agent's state directory.
#
# statements.db, the thesis journal and long-term memory cannot be regenerated —
# the journal in particular is the only record of what the agent predicted and
# when, which is the whole basis of its track record. So the backup is not
# optional, and it is never written in the clear: the directory also holds the
# provider credential store and prompts that quote position sizes.
#
# Needs `age` (https://age-encryption.org) and, in /etc/fra/backup.env:
#   FRA_BACKUP_AGE_RECIPIENT=age1...        public key; keep the private key OFF this server
#   FRA_BACKUP_DIR=/srv/fra/backups         where archives go (sync it off-box)
#   FRA_BACKUP_KEEP=14                      how many archives to keep
set -euo pipefail

DATA="${FRA_DATA_DIR:-/srv/fra/data}"
OUT="${FRA_BACKUP_DIR:-/srv/fra/backups}"
KEEP="${FRA_BACKUP_KEEP:-14}"
RECIPIENT="${FRA_BACKUP_AGE_RECIPIENT:?set FRA_BACKUP_AGE_RECIPIENT (an age public key) — refusing to write an unencrypted backup}"

command -v age >/dev/null || { echo "age is not installed" >&2; exit 1; }
mkdir -p "$OUT"
chmod 700 "$OUT"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
target="$OUT/fra-$stamp.tar.gz.age"

# SQLite files are copied through `.backup` when sqlite3 is available, so an
# archive taken mid-write is still a consistent database.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp -a "$DATA/." "$tmp/"
if command -v sqlite3 >/dev/null; then
    for db in "$DATA"/*.db; do
        [ -e "$db" ] || continue
        sqlite3 "$db" ".backup '$tmp/$(basename "$db")'"
    done
fi

tar -C "$tmp" -czf - . | age -r "$RECIPIENT" -o "$target"
chmod 600 "$target"

# Retention: newest $KEEP archives.
ls -1t "$OUT"/fra-*.tar.gz.age | tail -n +"$((KEEP + 1))" | xargs -r rm -f
echo "backup written: $target"
