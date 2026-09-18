"""
Download historical daily rainfall for the study area from Open-Meteo's
free Historical Weather (archive) API — no signup, no API key.

Open-Meteo's archive is ERA5-reanalysis-based, so it returns one time series
per queried point rather than a satellite raster. That's an acceptable
simplification here: the Kaneshie bbox is small enough (~3km) that even a
gridded product like CHIRPS (0.05° = ~5.5km cells) would only cover a
handful of cells anyway, so a single point at the bbox centroid is a
reasonable representative rainfall signal for this study area, not a
meaningful loss of resolution.

IMPORTANT: Open-Meteo's archive API REQUIRES both start_date AND end_date.
If end_date is not provided, it returns "Bad Request". This script now
defaults end_date to yesterday (the most recent available archive day).

Input:  config.yaml study_area.bbox, data_sources.rainfall.{start_date,end_date}
Output: data/raw/rainfall/{study_area_slug}_precipitation.json

Usage:
    python -m src.data_pipeline.fetch_rainfall

Note: needs network access to archive-api.open-meteo.com. Cannot be run
in a sandboxed environment without that access — run it on your own machine.
"""

import argparse
import re
from datetime import date, timedelta
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "rainfall"
API_URL = "https://archive-api.open-meteo.com/v1/archive"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Copy config/config.example.yaml to "
            "config/config.yaml first (it already has the Kaneshie bbox and date range filled in)."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "study_area"


def bbox_centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """(south, west, north, east) -> (lat, lon) centroid."""
    south, west, north, east = bbox
    return (south + north) / 2, (west + east) / 2


def get_default_end_date() -> str:
    """Get yesterday's date as ISO string (Open-Meteo's most recent available day)."""
    yesterday = date.today() - timedelta(days=1)
    return yesterday.isoformat()


def fetch_rainfall(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """Query Open-Meteo's archive API for daily precipitation at a point.
    Both start_date and end_date are required (ISO format: YYYY-MM-DD)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "Africa/Accra",
    }

    response = requests.get(API_URL, params=params, timeout=60)
    try:
        data = response.json()
    except ValueError as e:
        raise RuntimeError(
            f"Open-Meteo returned a non-JSON response ({response.status_code}): "
            f"{response.text[:300]!r}"
        ) from e
    if response.status_code != 200 or "daily" not in data:
        # Open-Meteo returns error detail as JSON with a "reason" field,
        # even on 400s — surface it directly.
        reason = data.get("reason", response.text[:500])
        raise RuntimeError(f"Open-Meteo request failed ({response.status_code}): {reason}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical rainfall for the study area.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = load_config(args.config)
    bbox = tuple(config["study_area"]["bbox"])
    area_name = config["study_area"].get("name", "study_area")
    rainfall_cfg = config["data_sources"]["rainfall"]
    start_date = rainfall_cfg["start_date"]
    end_date = rainfall_cfg.get("end_date")

    # Default end_date to yesterday if not provided (Open-Meteo requires it)
    if end_date is None:
        end_date = get_default_end_date()
        print(f"No end_date in config, defaulting to yesterday: {end_date}")

    lat, lon = bbox_centroid(bbox)
    print(f"Fetching daily precipitation for '{area_name}' centroid ({lat:.4f}, {lon:.4f}), "
          f"{start_date} -> {end_date} ...")

    data = fetch_rainfall(lat, lon, start_date, end_date)
    num_days = len(data["daily"]["time"])
    print(f"Received {num_days} daily records.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{slugify(area_name)}_precipitation.json"
    out_path.write_text(__import__("json").dumps(data, indent=2))
    print(f"Saved -> {out_path}")
    print("Next: build_flood_layer.py combines this with the DEM to compute per-edge flood risk.")


if __name__ == "__main__":
    main()