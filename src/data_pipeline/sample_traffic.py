"""
Sample traffic conditions for the study area over time, as a real-data proxy
layer (mirrors the approach used in comparable data-scarce-city studies:
lean on an existing traffic-inference API rather than building sensors).

Input:  config.yaml data_sources.traffic (API key, interval, duration)
Output: data/raw/traffic_samples/*.json  (raw API responses, timestamped)
        -> aggregated later into data/processed/traffic_profile.parquet

Usage:
    python -m src.data_pipeline.sample_traffic --config config/config.yaml

IMPORTANT: this script calls out to Mapbox Directions API over the network.
It cannot be run in a sandboxed environment without that access — run it
on your own machine.
"""

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dotenv
import requests
import yaml

dotenv.load_dotenv()  # read .env for local dev convenience

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "traffic_samples"
MAPBOX_DIRECTIONS_URL = "https://api.mapbox.com/directions/v5/mapbox"

# Pre-defined representative road segments for the Kaneshie study area.
# Each segment is an origin-destination pair covering key arterials and
# feeders. Coordinates are (lon, lat) for Mapbox API.
SEGMENTS = [
    # Kaneshie First Light - major E-W arterial
    {"id": "kaneshie_first_light", "origin": [-0.235, 5.562], "destination": [-0.215, 5.562]},
    # Graphic Road - major N-S arterial
    {"id": "graphic_road", "origin": [-0.225, 5.572], "destination": [-0.225, 5.552]},
    # Awudome Road - connector
    {"id": "awudome_road", "origin": [-0.238, 5.565], "destination": [-0.222, 5.565]},
    # Kaneshie Market area loop
    {"id": "kaneshie_market_loop", "origin": [-0.230, 5.560], "destination": [-0.220, 5.568]},
    # Bubuashie - residential feeder
    {"id": "bubuashie_feeder", "origin": [-0.240, 5.558], "destination": [-0.230, 5.558]},
]


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Copy config/config.example.yaml to "
            "config/config.yaml first (it already has the Kaneshie bbox and traffic config filled in)."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_access_token() -> str:
    """Get the Mapbox access token from the environment or config.

    Priority order:
    1. MAPBOX_ACCESS_TOKEN environment variable
    2. config.yaml data_sources.traffic.mapbox_access_token
    """
    # Check environment first (from .env)
    token = os.environ.get("MAPBOX_ACCESS_TOKEN")
    if token:
        return token.strip()

    # Fall back to config file
    config = load_config()
    token = config["data_sources"]["traffic"].get("mapbox_access_token")
    if token and token != "TODO":
        return token.strip()

    raise RuntimeError(
        "Mapbox access token not found. Set MAPBOX_ACCESS_TOKEN in .env "
        "or add it to config/config.yaml under data_sources.traffic.mapbox_access_token"
    )


def build_directions_url(origin: list[float], destination: list[float], profile: str, token: str) -> str:
    """Build the Mapbox Directions API URL for a single origin-destination pair."""
    coords = f"{origin[0]},{origin[1]};{destination[0]},{destination[1]}"
    return (
        f"{MAPBOX_DIRECTIONS_URL}/{profile}/{coords}"
        f"?alternatives=false&annotations=duration,distance&overview=simplified&access_token={token}"
    )


def sample_once(segments: list[dict[str, Any]] | None = None, profile: str = "driving-traffic") -> dict:
    """Query current traffic duration for a list of road segments.

    Args:
        segments: List of segment dicts with 'id', 'origin', 'destination'.
                  Defaults to the module-level SEGMENTS.
        profile: Mapbox routing profile ('driving-traffic', 'driving', 'walking').

    Returns:
        Dict with timestamp, profile, and per-segment results including
        raw Mapbox response for each segment.
    """
    if segments is None:
        segments = SEGMENTS

    token = get_access_token()
    timestamp = datetime.now(timezone.utc).isoformat()

    results = {
        "timestamp": timestamp,
        "profile": profile,
        "segments": {},
    }

    for seg in segments:
        seg_id = seg["id"]
        url = build_directions_url(seg["origin"], seg["destination"], profile, token)

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract key fields for quick access; keep raw response too
            routes = data.get("routes", [])
            if routes:
                route = routes[0]  # best/only route
                duration = route.get("duration")  # seconds
                distance = route.get("distance")  # meters
                results["segments"][seg_id] = {
                    "origin": seg["origin"],
                    "destination": seg["destination"],
                    "duration_seconds": duration,
                    "distance_meters": distance,
                    "duration_minutes": round(duration / 60, 2) if duration else None,
                    "raw_response": data,
                }
            else:
                results["segments"][seg_id] = {
                    "origin": seg["origin"],
                    "destination": seg["destination"],
                    "error": "No routes returned",
                    "raw_response": data,
                }

        except requests.RequestException as e:
            results["segments"][seg_id] = {
                "origin": seg["origin"],
                "destination": seg["destination"],
                "error": str(e),
                "raw_response": None,
            }

        # Small delay to respect rate limits
        time.sleep(0.1)

    return results


def save_sample(sample_data: dict) -> Path:
    """Save a single sample to a timestamped JSON file in raw/traffic_samples/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse timestamp for filename
    ts = datetime.fromisoformat(sample_data["timestamp"].replace("Z", "+00:00"))
    filename = ts.strftime("traffic_%Y%m%d_%H%M%S.json")
    out_path = OUTPUT_DIR / filename

    import json
    out_path.write_text(json.dumps(sample_data, indent=2))
    return out_path


def run_sampling_schedule(interval_minutes: int, days: int, profile: str = "driving-traffic") -> None:
    """Run sample_once on a schedule and append results to raw/traffic_samples/.

    Args:
        interval_minutes: Minutes between samples.
        days: How many days to run (converted to number of iterations).
        profile: Mapbox routing profile to use.
    """
    total_iterations = int((days * 24 * 60) / interval_minutes)
    print(
        f"Starting traffic sampling: {total_iterations} iterations "
        f"every {interval_minutes} minutes for {days} day(s)..."
    )

    for i in range(total_iterations):
        iteration_start = time.time()

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Sample {i + 1}/{total_iterations}...")
        sample = sample_once(profile=profile)
        out_path = save_sample(sample)
        print(f"  Saved -> {out_path.name}")

        # Calculate sleep time to maintain exact interval
        elapsed = time.time() - iteration_start
        sleep_time = max(0, interval_minutes * 60 - elapsed)
        if i < total_iterations - 1:  # don't sleep after the last iteration
            time.sleep(sleep_time)

    print("Sampling complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample traffic conditions via Mapbox Directions API.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single sample and exit (default: run scheduled sampling)."
    )
    parser.add_argument(
        "--interval", type=int,
        help="Sampling interval in minutes (overrides config)."
    )
    parser.add_argument(
        "--days", type=int,
        help="Number of days to sample (overrides config)."
    )
    parser.add_argument(
        "--profile", type=str, default="driving-traffic",
        help="Mapbox routing profile (driving-traffic, driving, walking)."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    traffic_cfg = config["data_sources"]["traffic"]

    interval = args.interval or traffic_cfg.get("sample_interval_minutes", 60)
    days = args.days or traffic_cfg.get("sample_days", 14)
    profile = args.profile or traffic_cfg.get("profile", "driving-traffic")

    if args.once:
        print("Running single traffic sample...")
        sample = sample_once(profile=profile)
        out_path = save_sample(sample)
        print(f"Saved -> {out_path}")
    else:
        run_sampling_schedule(interval_minutes=interval, days=days, profile=profile)


if __name__ == "__main__":
    main()
