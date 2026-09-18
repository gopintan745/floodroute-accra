"""
Sample traffic conditions for the study area over time, as a real-data proxy
layer (mirrors the approach used in comparable data-scarce-city studies:
lean on an existing traffic-inference API rather than building sensors).

Input:  config.yaml data_sources.traffic (API key, interval, duration)
Output: data/raw/traffic_samples/*.json  (raw API responses, timestamped)
        -> aggregated later into data/processed/traffic_profile.parquet

TODO:
- Pick a fixed set of representative origin-destination pairs / road segments
  covering the study area
- Call the Google Maps Directions API with departure_time=now on a schedule
  (respect free-tier rate limits — this should run as a scheduled/cron job
  over days-to-weeks, not all at once)
- Save raw responses as-is (timestamped) — do NOT aggregate at collection
  time, so you can change your aggregation logic later without re-collecting
- Where API sampling is impractical, fall back to a synthetic time-of-day
  congestion profile (see build_graph_costs.py) rather than blocking on data
"""

import requests  # noqa: F401

def get_access_token() -> str:
    """Get the Mapbox access token from the environment or config."""
    raise NotImplementedError("TODO")

def sample_once(segments: list) -> dict:
    """Query current traffic duration for a list of road segments. TODO."""
    raise NotImplementedError("TODO")


def run_sampling_schedule(interval_minutes: int, days: int) -> None:
    """Run sample_once on a schedule and append results to raw/traffic_samples/."""
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    raise NotImplementedError("TODO: wire up CLI / scheduling")
