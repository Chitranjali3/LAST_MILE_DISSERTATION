#!/bin/bash
# Clip a Geofabrik OSM extract to a smaller bounding box using osmium-tool.
#
# Why this exists:
#   On Apple Silicon, OSRM's official Docker image is x86_64 only and runs
#   under emulation. A full state/zone (e.g. eastern-zone-latest, ~242 MB,
#   39M nodes) routinely OOM-kills `osrm-extract`/`osrm-partition`. Clipping
#   to just the area you actually route in (Odisha by default) shrinks the
#   PBF to ~25-35 MB and keeps the build comfortably under memory limits.
#
# Inputs (env vars, all optional):
#   OSRM_SOURCE_STEM   default: eastern-zone-latest
#       Source PBF stem; expects ``data/${OSRM_SOURCE_STEM}.osm.pbf`` to exist
#       (download with ``./scripts/download_map.sh``).
#   OSRM_MAP_STEM      default: odisha
#       Output PBF stem; produces ``data/${OSRM_MAP_STEM}.osm.pbf``.
#   OSRM_BBOX          default: 81.30,17.78,87.60,22.60   (Odisha)
#       comma-separated min_lon,min_lat,max_lon,max_lat
#
# Behavior: idempotent — skips work if the output already exists.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

SOURCE_STEM="${OSRM_SOURCE_STEM:-eastern-zone-latest}"
OUT_STEM="${OSRM_MAP_STEM:-odisha}"
BBOX="${OSRM_BBOX:-81.30,17.78,87.60,22.60}"

SOURCE_PBF="$DATA_DIR/${SOURCE_STEM}.osm.pbf"
OUT_PBF="$DATA_DIR/${OUT_STEM}.osm.pbf"

if [ ! -f "$SOURCE_PBF" ]; then
    echo "Error: source PBF not found: $SOURCE_PBF"
    echo "Run ./scripts/download_map.sh (with OSRM_MAP_STEM=$SOURCE_STEM) first."
    exit 1
fi

if [ -f "$OUT_PBF" ]; then
    echo "Output PBF already exists: $OUT_PBF"
    echo "Skipping clip. Delete the file to force a rebuild."
    exit 0
fi

run_osmium() {
    if command -v osmium >/dev/null 2>&1; then
        osmium "$@"
    else
        # Fall back to a docker image with osmium-tool when not installed
        # natively. iboates/osmium is a small, popular wrapper.
        docker run --rm -v "$DATA_DIR:/data" -w /data iboates/osmium osmium "$@"
    fi
}

echo "Clipping ${SOURCE_STEM}.osm.pbf -> ${OUT_STEM}.osm.pbf"
echo "BBox (min_lon,min_lat,max_lon,max_lat): $BBOX"
run_osmium extract --bbox="$BBOX" --strategy=complete_ways \
    "$SOURCE_PBF" -o "$OUT_PBF"

echo "Done."
ls -lh "$OUT_PBF"
