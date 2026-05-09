#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
export OSRM_MAP_STEM="${OSRM_MAP_STEM:-odisha}"
OSRM_FILE="$DATA_DIR/${OSRM_MAP_STEM}.osrm"

if [ ! -f "$OSRM_FILE" ]; then
    echo "Error: OSRM data file not found: $OSRM_FILE"
    echo "Please run build_osrm.sh first (same OSRM_MAP_STEM)."
    exit 1
fi

export OSRM_HOST_PORT="${OSRM_HOST_PORT:-5001}"
echo "Starting OSRM server (container :5000 → host :${OSRM_HOST_PORT})..."
echo "Data directory: $DATA_DIR"
echo "OSRM map stem: $OSRM_MAP_STEM"

compose_bin=(docker compose)
if ! docker compose version &>/dev/null; then
    compose_bin=(docker-compose)
fi

"${compose_bin[@]}" -f "$PROJECT_ROOT/docker-compose.yml" up -d

echo "Waiting for server to be ready..."
sleep 5

MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # /health is not implemented by osrm-routed; /nearest is, and a 200 on
    # this endpoint proves the server is bound and the dataset is loaded.
    if curl -fs "http://localhost:${OSRM_HOST_PORT}/nearest/v1/driving/0,0" > /dev/null 2>&1; then
        echo "OSRM server is ready!"
        echo "Base URL for Last-Mile: http://localhost:${OSRM_HOST_PORT}"
        exit 0
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep 1
done

echo "Warning: Server may not be fully ready. Check logs with: docker compose logs"
