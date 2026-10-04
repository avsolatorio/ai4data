#!/usr/bin/env bash
# Wait until a log reports BACKFILL-DONE, then run the given command. Usage: scripts/after_backfill.sh <log> <command...>
LOG=${1:?log}; shift
until grep -q BACKFILL-DONE "$LOG" 2>/dev/null; do sleep 120; done
cd "$(dirname "$0")/.."
exec "$@"
