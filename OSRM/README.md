# OSRM Local Development Environment

Docker-based OSRM (Open Source Routing Machine) setup for local development and testing.

The default build target is **`odisha`** — a bbox-clipped subset of Geofabrik's
`eastern-zone-latest` extract. We clip first because:

- Geofabrik does not publish a standalone Odisha-only extract.
- The full eastern-zone (~242 MB, 39 M nodes) routinely OOM-kills `osrm-extract`
  / `osrm-partition` on Apple Silicon (`osrm/osrm-backend` is `linux/amd64`-only
  and runs under emulation).
- After clipping, the Odisha PBF is ~37 MB and the entire MLD pipeline finishes
  in roughly **1 minute** with peak RAM ~330 MB during partition.

## Prerequisites

- Docker and Docker Compose installed
- Bash shell (macOS / Linux)
- Python 3.6+ with `requests` library (only for the Python API smoke tests)
- `wget` or `curl` (for downloading map data)
- `osmium-tool` for the bbox clip step (`brew install osmium-tool` on macOS).
  If unavailable, the script falls back to the `iboates/osmium` Docker image.

## Project Structure

```
OSRM/
 ├── data/                    # Map data and OSRM files (gitignored large binaries)
 ├── scripts/
 │    ├── download_map.sh     # Download source PBF (default: eastern-zone-latest)
 │    ├── clip_to_bbox.sh     # Clip a source PBF to a bounding box (default: Odisha)
 │    ├── build_osrm.sh       # Build OSRM MLD pipeline (auto-clips if needed)
 │    ├── run_server.sh       # Start OSRM server
 │    └── stop_server.sh      # Stop OSRM server
 ├── tests/
 │    └── test_osrm_api.py    # API smoke test suite (Bhubaneswar coords)
 ├── docker-compose.yml       # Pinned platform: linux/amd64
 └── README.md
```

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `OSRM_MAP_STEM` | `odisha` | Base name for `.osm.pbf` / `.osrm` files under `data/`. The compose file and all scripts use this stem to find the routing dataset. |
| `OSRM_SOURCE_STEM` | `eastern-zone-latest` | When building/clipping, the upstream Geofabrik PBF we clip from. |
| `OSRM_BBOX` | `81.30,17.78,87.60,22.60` | Odisha bounding box (`min_lon,min_lat,max_lon,max_lat`). |
| `OSRM_BUILD_THREADS` | `2` | Threads for `osrm-extract`/`partition`/`customize`. Drop to `1` if Docker keeps OOM-killing the build. |
| `OSRM_HOST_PORT` | `5001` | Host port mapped to the container’s `5000` (macOS often reserves `5000` for AirPlay). |

## Quick Start

### One-Command Build

```bash
./scripts/download_map.sh && ./scripts/build_osrm.sh
```

`build_osrm.sh` transparently calls `clip_to_bbox.sh` when the chosen
`${OSRM_MAP_STEM}.osm.pbf` is missing but `${OSRM_SOURCE_STEM}.osm.pbf` exists.

### One-Command Run

```bash
./scripts/run_server.sh
```

### One-Command Test

```bash
python tests/test_osrm_api.py
```

## Detailed Setup Instructions

### 1. Download Map Data

Downloads the source PBF from Geofabrik. With the default `OSRM_MAP_STEM=odisha`,
the script automatically fetches `eastern-zone-latest.osm.pbf`:

```bash
./scripts/download_map.sh
```

Idempotent — skips download if the file already exists.

### 2. (Optional) Clip to a Bounding Box

```bash
./scripts/clip_to_bbox.sh
```

Outputs `data/${OSRM_MAP_STEM}.osm.pbf` (default `data/odisha.osm.pbf`, ~37 MB).
You can also skip this — `build_osrm.sh` will run it automatically when needed.

### 3. Build OSRM Data

Runs the MLD (Multi-Level Dijkstra) pipeline:

- `osrm-extract`: extracts the road network from OSM data
- `osrm-partition`: partitions the graph into cells
- `osrm-customize`: precomputes per-cell metrics

```bash
./scripts/build_osrm.sh
```

On Apple Silicon under x86 emulation: ~1 min for the Odisha clip, ~30+ min and
prone to OOM for the full eastern-zone.

### 4. Start Server

```bash
./scripts/run_server.sh
```

Defaults to `http://localhost:5001` (see `OSRM_HOST_PORT`).

### 5. Run Tests

```bash
python tests/test_osrm_api.py
```

If `requests` is missing:

```bash
pip install requests
python tests/test_osrm_api.py
```

### 6. Stop Server

```bash
./scripts/stop_server.sh
```

## API Examples

Coordinates below use **Odisha / Bhubaneswar-area** samples (lon,lat per OSRM).

### Health Check

```bash
curl http://localhost:5001/health
```

### Route API

```bash
curl "http://localhost:5001/route/v1/driving/85.8245,20.2961;85.8345,20.3061?overview=false"
```

### Table API

```bash
curl "http://localhost:5001/table/v1/driving/85.8245,20.2961;85.8345,20.3061"
```

### Nearest API

```bash
curl "http://localhost:5001/nearest/v1/driving/85.8245,20.2961"
```

## Targeting a different region

Set `OSRM_MAP_STEM`, `OSRM_SOURCE_STEM`, and `OSRM_BBOX` together. Example: build
a tiny **Goa** dataset out of `western-zone-latest`:

```bash
OSRM_MAP_STEM=goa \
OSRM_SOURCE_STEM=western-zone-latest \
OSRM_BBOX=73.65,14.85,74.35,15.85 \
./scripts/download_map.sh && ./scripts/build_osrm.sh && ./scripts/run_server.sh
```

## Troubleshooting

### Map download fails

- Check internet connection
- Verify Geofabrik URL is accessible
- Ensure sufficient disk space

### Clip fails

- Install osmium-tool natively: `brew install osmium-tool`
- Or ensure Docker can pull `iboates/osmium`

### Build fails (`Killed` / exit 137 mid-`osrm-partition`)

- This is usually OOM under x86 emulation. Either:
  - Stick to the small `odisha` default (recommended).
  - Bump Docker Desktop → Settings → Resources → Memory to 8 GB+.
  - Enable Docker Desktop → Settings → General → "Use Rosetta for x86_64/amd64 emulation".
  - Lower threads further: `OSRM_BUILD_THREADS=1 ./scripts/build_osrm.sh`.

### Server won't start

- Check if the host port (`OSRM_HOST_PORT`, default `5001`) is already in use
- Verify OSRM data files exist in `data/` for the same `OSRM_MAP_STEM` you built
- Check Docker logs: `docker compose logs osrm`

### Tests fail

- Ensure server is running: `curl http://localhost:5001/health`
- Install Python requests: `pip install requests`
- Check server logs for errors

## Notes

- All scripts use relative paths and are portable.
- `docker-compose.yml` pins `platform: linux/amd64` to silence the
  platform-mismatch warning on Apple Silicon.
- Server uses the MLD algorithm for fast routing.
