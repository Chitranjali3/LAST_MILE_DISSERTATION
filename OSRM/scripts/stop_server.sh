#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Stopping OSRM server..."

compose_bin=(docker compose)
if ! docker compose version &>/dev/null; then
    compose_bin=(docker-compose)
fi

"${compose_bin[@]}" -f "$PROJECT_ROOT/docker-compose.yml" down

echo "OSRM server stopped."
