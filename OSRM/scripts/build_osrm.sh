#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
MAP_STEM="${OSRM_MAP_STEM:-odisha}"
MAP_FILE="$DATA_DIR/${MAP_STEM}.osm.pbf"
OSRM_FILE="$DATA_DIR/${MAP_STEM}.osrm"
OSRM_PARTITION="$DATA_DIR/${MAP_STEM}.osrm.partition"
# Newer osrm-backend emits `.osrm.mldgr` after customize; older builds used `.osrm.mld`.
OSRM_MLD="$DATA_DIR/${MAP_STEM}.osrm.mldgr"

# If the chosen PBF doesn't exist yet but a source PBF (eastern-zone by
# default) does, transparently bbox-clip it via osmium so smaller/regional
# builds work end-to-end without hand-running clip_to_bbox.sh.
if [ ! -f "$MAP_FILE" ]; then
    SOURCE_STEM="${OSRM_SOURCE_STEM:-eastern-zone-latest}"
    SOURCE_PBF="$DATA_DIR/${SOURCE_STEM}.osm.pbf"
    if [ "$MAP_STEM" != "$SOURCE_STEM" ] && [ -f "$SOURCE_PBF" ]; then
        echo "Map file $MAP_FILE not found; clipping from $SOURCE_PBF via osmium..."
        OSRM_SOURCE_STEM="$SOURCE_STEM" OSRM_MAP_STEM="$MAP_STEM" \
            "$SCRIPT_DIR/clip_to_bbox.sh"
    fi
fi

if [ ! -f "$MAP_FILE" ]; then
    echo "Error: Map file not found: $MAP_FILE"
    echo "Run ./scripts/download_map.sh and (if clipping) ./scripts/clip_to_bbox.sh first."
    exit 1
fi

echo "Building OSRM MLD pipeline..."
echo "Map stem: $MAP_STEM"
echo "Map file: $MAP_FILE"

OSRM_IMAGE="osrm/osrm-backend:latest"
# Lower thread count helps avoid OOM when running linux/amd64 OSRM under QEMU on Apple Silicon.
THREADS="${OSRM_BUILD_THREADS:-2}"
echo "Using $THREADS thread(s) for extract/partition/customize (set OSRM_BUILD_THREADS to override)."

echo "Step 1: Extracting road network..."
docker run --rm \
    -v "$DATA_DIR:/data" \
    "$OSRM_IMAGE" \
    osrm-extract -t "$THREADS" -p /opt/car.lua "/data/${MAP_STEM}.osm.pbf"

if [ ! -f "$OSRM_FILE" ]; then
    echo "Error: osrm-extract failed. Output file not found."
    exit 1
fi

echo "Step 2: Partitioning graph..."
docker run --rm \
    -v "$DATA_DIR:/data" \
    "$OSRM_IMAGE" \
    osrm-partition -t "$THREADS" "/data/${MAP_STEM}.osrm"

if [ ! -f "$OSRM_PARTITION" ]; then
    echo "Error: osrm-partition failed. Output file not found."
    exit 1
fi

echo "Step 3: Customizing graph..."
docker run --rm \
    -v "$DATA_DIR:/data" \
    "$OSRM_IMAGE" \
    osrm-customize -t "$THREADS" "/data/${MAP_STEM}.osrm"

if [ ! -f "$OSRM_MLD" ]; then
    echo "Error: osrm-customize failed. Output file not found."
    exit 1
fi

echo "OSRM MLD build completed successfully!"
echo "Generated files:"
ls -lh "$DATA_DIR"/*.osrm*

