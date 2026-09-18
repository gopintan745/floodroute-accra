"""
Download the raw DEM (elevation) tile for the study area from OpenTopography's
Global Datasets API. This is the input to build_flood_layer.py's flood-risk
heuristic — low-lying areas near drainage paths get a higher flood-risk score.

Provider: OpenTopography Global Datasets API, serving Copernicus GLO-30 (30m)
by default (demtype=COP30). Free personal API key required.

Auth: read from the OPENTOPOGRAPHY_API_KEY environment variable, NOT from
config.yaml — same reasoning as the Mapbox token in sample_traffic.py.
    export OPENTOPOGRAPHY_API_KEY="your_key_here"

Get a free key: https://portal.opentopography.org/requestService?service=api

Input:  config.yaml study_area.bbox, data_sources.dem.demtype
Output: data/raw/dem/{demtype}_{study_area_name}.tif

Usage:
    python -m src.data_pipeline.fetch_dem
    (run from the repo root, with config/config.yaml already created and
    OPENTOPOGRAPHY_API_KEY set)

Note: this script needs network access to portal.opentopography.org. It
cannot be run in a sandboxed environment without that access — run it on
your own machine.
"""

import argparse
import os
import re
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "dem"
API_URL = "https://portal.opentopography.org/API/globaldem"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Copy config/config.example.yaml to "
            "config/config.yaml first (it already has the Kaneshie bbox and demtype filled in)."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_api_key() -> str:
    """Read the OpenTopography API key from the environment. Fail loudly
    here rather than sending an unauthenticated request that will 401/403."""
    key = os.environ.get("OPENTOPOGRAPHY_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENTOPOGRAPHY_API_KEY not set. Get a free key at "
            "https://portal.opentopography.org/requestService?service=api "
            "then run: export OPENTOPOGRAPHY_API_KEY='your_key_here'"
        )
    return key


def slugify(name: str) -> str:
    """Turn a study-area name into a filesystem-safe slug for the output filename."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "study_area"


def fetch_dem(bbox: tuple[float, float, float, float], demtype: str, api_key: str) -> bytes:
    """Request a GeoTIFF DEM tile covering the given (south, west, north, east) bbox."""
    south, west, north, east = bbox
    params = {
        "demtype": demtype,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    response = requests.get(API_URL, params=params, timeout=60)
    if response.status_code != 200:
        # OpenTopography returns error details as plain text/XML in the body,
        # not JSON — surface it directly rather than a generic HTTP error.
        raise RuntimeError(
            f"OpenTopography request failed ({response.status_code}): {response.text[:500]}"
        )
    content_type = response.headers.get("Content-Type", "")
    if "tif" not in content_type and "octet-stream" not in content_type:
        # A 200 with the wrong content type usually means an error page was
        # returned anyway (e.g. malformed bbox) — catch it before writing
        # a "DEM" file that's actually an HTML error page.
        raise RuntimeError(
            f"Unexpected response Content-Type '{content_type}', expected GeoTIFF. "
            f"First 300 bytes: {response.content[:300]!r}"
        )
    return response.content


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the DEM tile for the study area.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--demtype", default=None, help="Override config.yaml's data_sources.dem.demtype")
    args = parser.parse_args()

    config = load_config(args.config)
    bbox = tuple(config["study_area"]["bbox"])
    area_name = config["study_area"].get("name", "study_area")
    demtype = args.demtype or config["data_sources"]["dem"].get("demtype", "COP30")
    api_key = get_api_key()

    print(f"Fetching {demtype} DEM for '{area_name}' — bbox={bbox} ...")
    tif_bytes = fetch_dem(bbox, demtype, api_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{demtype.lower()}_{slugify(area_name)}.tif"
    out_path.write_bytes(tif_bytes)
    print(f"Saved DEM -> {out_path} ({len(tif_bytes) / 1024:.1f} KB)")
    print("Next: build_flood_layer.py uses this as its elevation input.")


if __name__ == "__main__":
    main()