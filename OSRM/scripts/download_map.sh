#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
# This script always downloads from Geofabrik. The default ``odisha`` build is
# produced by clipping ``eastern-zone-latest`` (Geofabrik does not publish a
# standalone Odisha extract); see ``./scripts/clip_to_bbox.sh``. So when the
# requested stem is ``odisha`` we fetch its source PBF instead.
MAP_STEM="${OSRM_MAP_STEM:-odisha}"
DOWNLOAD_STEM="$MAP_STEM"
if [ "$DOWNLOAD_STEM" = "odisha" ]; then
    DOWNLOAD_STEM="eastern-zone-latest"
fi

if [ "$DOWNLOAD_STEM" = "india-latest" ]; then
    MAP_URL="https://download.geofabrik.de/asia/india-latest.osm.pbf"
else
    MAP_URL="https://download.geofabrik.de/asia/india/${DOWNLOAD_STEM}.osm.pbf"
fi

MAP_FILE="$DATA_DIR/${DOWNLOAD_STEM}.osm.pbf"

mkdir -p "$DATA_DIR"

if [ -f "$MAP_FILE" ]; then
    echo "Map file already exists: $MAP_FILE"
    echo "Skipping download."
    exit 0
fi

echo "Downloading OSM extract (${MAP_STEM}) from Geofabrik..."
echo "URL: $MAP_URL"
echo "Destination: $MAP_FILE"

if command -v wget &> /dev/null; then
    wget -O "$MAP_FILE" "$MAP_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$MAP_FILE" "$MAP_URL"
else
    echo "Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

if [ ! -f "$MAP_FILE" ]; then
    echo "Error: Download failed. Map file not found."
    exit 1
fi

if head -c 50 "$MAP_FILE" | grep -q '<!DOCTYPE\|<html'; then
    echo "Error: Download returned HTML (wrong URL or mirror page). Removing file."
    rm -f "$MAP_FILE"
    exit 1
fi

echo "Download completed successfully."
echo "Map file: $MAP_FILE"
ls -lh "$MAP_FILE"
