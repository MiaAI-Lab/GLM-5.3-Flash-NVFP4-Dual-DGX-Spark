#!/usr/bin/env bash
# stop-tp4.sh — stop the EXPERIMENTAL TP=4 cluster started by start-tp4.sh
#
# Removes glm53-flash-tp4-head on this machine and glm53-flash-tp4-w{1,2,3}
# on the three workers. Does NOT touch the 2-node ./start.sh containers
# (glm53-flash-head / glm53-flash-worker).
#
# Equivalent to: ./start-tp4.sh stop
# Needs the same .env.tp4 (or WORKER{1,2,3}_SSH) as start-tp4.sh.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/start-tp4.sh" stop
