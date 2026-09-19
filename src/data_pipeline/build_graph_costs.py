"""
Merge road graph + flood-risk layer + traffic profile + road-quality into a
single graph with all per-edge attributes the environment needs, plus a
separate traffic profile table for runtime lookup.

IMPORTANT DESIGN NOTE: unlike flood_susceptibility and road_quality_score
(both static per-edge attributes), traffic is inherently TIME-VARYING and
is NOT baked into the graph as a fixed number. Instead, this script:
  1. Normalizes each edge's OSM 'highway' tag into a single clean
     'highway_class' attribute (raw OSM tags are sometimes lists)
  2. Builds/saves a separate traffic profile table keyed by
     (highway_class, hour_of_day, is_weekend) -> multiplier
The environment (accra_routing_env.py / hazard_simulator.py) looks up the
multiplier at each simulated step using the edge's highway_class and the
episode's current simulated time — this keeps traffic properly separable
from the static hazard layers, per the same principle used for flood risk
vs. rainfall climatology in build_flood_layer.py.

Two traffic profile sources, blended when both are available:
  - Synthetic (this script): time-of-day multiplier per road class, based
    on typical urban congestion patterns. Always available, always a
    reasonable fallback.
  - Real (from sample_traffic.py's collected Mapbox samples): overrides
    the synthetic multiplier wherever real samples exist. NOTE: the
    blending function here (blend_traffic_profiles) is a scaffold pending
    the actual schema of the collected samples — fill in
    _load_real_traffic_samples() once that schema is confirmed.

Inputs:
    data/processed/road_graph.graphml           (from fetch_osm.py)
    data/processed/flood_risk.tif                (from build_flood_layer.py)
    data/raw/traffic_samples/*                   (from sample_traffic.py, optional)
Outputs:
    data/processed/road_graph_full.graphml       (all static attributes attached)
    data/processed/traffic_profile.parquet       (synthetic, or blended if real data present)

Usage:
    python -m src.data_pipeline.build_graph_costs
    (pure computation + reads local files — no network access needed)
"""

import argparse
import json
from pathlib import Path
import yaml
import networkx as nx
import osmnx as ox
import pandas as pd

from .build_flood_layer import DEFAULT_OUTPUT_TIF, sample_risk_along_edges

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
INPUT_GRAPHML = REPO_ROOT / "data" / "processed" / "road_graph.graphml"
TRAFFIC_SAMPLES_DIR = REPO_ROOT / "data" / "raw" / "traffic_samples"
OUTPUT_GRAPHML = REPO_ROOT / "data" / "processed" / "road_graph_full.graphml"
OUTPUT_TRAFFIC_PROFILE = REPO_ROOT / "data" / "processed" / "traffic_profile.parquet"

HOURS = list(range(24))
WEEKEND_OPTIONS = [False, True]

def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Copy config/config.example.yaml to "
            "config/config.yaml first."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)

# --------------------------------------------------------------------------
# Road quality (OSM-tag heuristic)
# --------------------------------------------------------------------------

def _first_or_list(value):
    """OSM tags are sometimes a single string, sometimes a list (when
    multiple ways got merged into one graph edge). Normalize to a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def road_quality_score(surface, highway, smoothness, rq_cfg: dict) -> float:
    """Score an edge's road quality in [0, 1] from OSM tags. When a tag is
    list-valued (multiple merged ways), the WORST-scoring value is used —
    a single bad patch matters to a driver regardless of how good the rest
    of the edge is.

    Config keys (from config.yaml road_quality section):
    - highway_base_scores: base score by highway class
    - surface_multipliers: multiplier applied to base if surface tag present
    - default_highway_score: used if highway tag missing/unrecognized
    - smoothness_scores: direct score if smoothness tag present
    - smoothness_blend_weight: weight on smoothness_score when tag is present
    """
    highway_scores = rq_cfg.get("highway_base_scores", {})
    surface_mults = rq_cfg.get("surface_multipliers", {})
    smoothness_scores = rq_cfg.get("smoothness_scores", {})
    default_score = rq_cfg.get("default_highway_score", 0.5)
    smoothness_weight = rq_cfg.get("smoothness_blend_weight", 0.6)

    surface_vals = _first_or_list(surface)
    highway_vals = _first_or_list(highway)
    smoothness_vals = _first_or_list(smoothness)

    # Base score: prefer highway class; fall back to default if no recognized highway.
    highway_candidates = [highway_scores[h] for h in highway_vals if h in highway_scores]
    if highway_candidates:
        base_score = min(highway_candidates)
    else:
        base_score = default_score

    # Apply surface multiplier if surface tag present and recognized.
    surface_candidates = [surface_mults[s] for s in surface_vals if s in surface_mults]
    if surface_candidates:
        # Use the WORST (lowest) surface multiplier - a bad surface drags down the score
        base_score *= min(surface_candidates)

    smoothness_candidates = [smoothness_scores[s] for s in smoothness_vals if s in smoothness_scores]
    if smoothness_candidates:
        smoothness_score = min(smoothness_candidates)
        return smoothness_weight * smoothness_score + (1 - smoothness_weight) * base_score

    return base_score


def compute_road_quality(graph: "nx.MultiDiGraph", rq_cfg: dict) -> "nx.MultiDiGraph":
    """Attach 'road_quality_score' to every edge."""
    for u, v, k, data in graph.edges(keys=True, data=True):
        data["road_quality_score"] = road_quality_score(
            data.get("surface"), data.get("highway"), data.get("smoothness"), rq_cfg
        )
    return graph


# --------------------------------------------------------------------------
# Highway class normalization (needed for both road quality and traffic lookup)
# --------------------------------------------------------------------------

def normalize_highway_class(graph: "nx.MultiDiGraph", known_classes: list) -> "nx.MultiDiGraph":
    """Attach a single clean 'highway_class' string per edge, taking the
    first recognized value from a possibly-list raw 'highway' tag, or
    'unclassified' if nothing recognized."""
    for u, v, k, data in graph.edges(keys=True, data=True):
        vals = _first_or_list(data.get("highway"))
        recognized = [val for val in vals if val in known_classes]
        data["highway_class"] = recognized[0] if recognized else "unclassified"
    return graph


# --------------------------------------------------------------------------
# Synthetic traffic profile
# --------------------------------------------------------------------------

def build_synthetic_traffic_profile(traffic_cfg: dict) -> pd.DataFrame:
    """Build a (highway_class, hour, is_weekend) -> multiplier table from
    typical urban peak/off-peak patterns. This is the fallback used when no
    real traffic samples are available for a given class/hour combination."""
    morning_lo, morning_hi = traffic_cfg["peak_hours_morning"]
    evening_lo, evening_hi = traffic_cfg["peak_hours_evening"]
    weekday_peak = traffic_cfg["weekday_peak_multiplier"]
    weekend_factor = traffic_cfg["weekend_peak_multiplier_factor"]
    offpeak = traffic_cfg["offpeak_multiplier"]

    def is_peak_hour(hour: int) -> bool:
        return (morning_lo <= hour <= morning_hi) or (evening_lo <= hour <= evening_hi)

    rows = []
    for highway_class, peak_mult in weekday_peak.items():
        weekend_peak_mult = 1 + (peak_mult - 1) * weekend_factor
        for hour in HOURS:
            for is_weekend in WEEKEND_OPTIONS:
                if is_peak_hour(hour):
                    multiplier = weekend_peak_mult if is_weekend else peak_mult
                else:
                    multiplier = offpeak
                rows.append({
                    "highway_class": highway_class,
                    "hour": hour,
                    "is_weekend": is_weekend,
                    "multiplier": multiplier,
                    "source": "synthetic",
                })
    return pd.DataFrame(rows)


def _load_real_traffic_samples(graph: "nx.MultiDiGraph") -> "pd.DataFrame | None":
    """Parse raw sample_traffic.py JSON files into a
    (highway_class, hour, is_weekend, multiplier) table.

    Two-step approach:
      1. Compute each segment's congestion multiplier directly from
         Mapbox's own response: duration_seconds (current, traffic-aware)
         / raw_response.routes[0].duration_typical (Mapbox's own
         historical/free-flow baseline). This means a real multiplier can
         be computed from a SINGLE sample — no need to collect for weeks
         before Mapbox's reference value exists, since Mapbox already
         provides one per request.
      2. Segment names (e.g. "kaneshie_first_light") don't carry road-class
         info themselves, so each segment's midpoint is snapped to the
         nearest edge in `graph` to borrow that edge's highway_class label
         (graph must already have highway_class set via
         normalize_highway_class before this is called).

    Multiple samples landing in the same (highway_class, hour, is_weekend)
    bucket are averaged. Assumes timestamps are UTC — Africa/Accra has no
    DST and sits at UTC+0 year-round, so no timezone conversion is needed.
    """
    if not TRAFFIC_SAMPLES_DIR.exists():
        return None
    sample_files = list(TRAFFIC_SAMPLES_DIR.glob("*.json"))
    if not sample_files:
        return None

    rows = []
    for path in sample_files:
        with open(path) as f:
            data = json.load(f)
        ts = pd.to_datetime(data["timestamp"])
        hour = ts.hour
        is_weekend = ts.dayofweek >= 5

        for name, seg in data.get("segments", {}).items():
            duration = seg.get("duration_seconds")
            routes = seg.get("raw_response", {}).get("routes", [])
            duration_typical = routes[0].get("duration_typical") if routes else None
            if not duration or not duration_typical:
                continue  # skip malformed/incomplete segment samples

            multiplier = duration / duration_typical

            lon = (seg["origin"][0] + seg["destination"][0]) / 2
            lat = (seg["origin"][1] + seg["destination"][1]) / 2
            try:
                u, v, k = ox.distance.nearest_edges(graph, X=lon, Y=lat)
            except Exception:
                continue  # segment midpoint falls outside the study-area graph

            highway_class = graph.edges[u, v, k].get("highway_class", "unclassified")
            rows.append({
                "highway_class": highway_class, "hour": hour, "is_weekend": bool(is_weekend),
                "multiplier": multiplier, "segment": name, "sample_file": path.name,
            })

    if not rows:
        return None

    raw_df = pd.DataFrame(rows)
    aggregated = (
        raw_df.groupby(["highway_class", "hour", "is_weekend"])
        .agg(multiplier=("multiplier", "mean"), num_real_samples=("multiplier", "count"))
        .reset_index()
    )
    return aggregated


def blend_traffic_profiles(synthetic_df: pd.DataFrame, real_df: "pd.DataFrame | None") -> pd.DataFrame:
    """Real samples override the synthetic multiplier wherever they exist
    for a given (highway_class, hour, is_weekend); synthetic fills every
    gap. Keeping both means coverage never has holes even with a small
    real sample. num_real_samples is carried through (0 for synthetic-only
    rows) so it's visible later how much real evidence backs each bucket."""
    if real_df is None or real_df.empty:
        out = synthetic_df.copy()
        out["num_real_samples"] = 0
        return out

    merged = synthetic_df.merge(
        real_df, on=["highway_class", "hour", "is_weekend"], how="left", suffixes=("_synthetic", "_real")
    )
    merged["multiplier"] = merged["multiplier_real"].combine_first(merged["multiplier_synthetic"])
    merged["source"] = merged["multiplier_real"].notna().map({True: "real", False: "synthetic"})
    merged["num_real_samples"] = merged["num_real_samples"].fillna(0).astype(int)
    return merged[["highway_class", "hour", "is_weekend", "multiplier", "source", "num_real_samples"]]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge all layers into the full study-area graph.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--synthetic-only", action="store_true",
                         help="Skip attempting to load/blend real traffic samples.")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Loading graph <- {INPUT_GRAPHML}")
    graph = ox.load_graphml(INPUT_GRAPHML)

    print(f"Sampling flood risk <- {DEFAULT_OUTPUT_TIF}")
    graph = sample_risk_along_edges(graph, DEFAULT_OUTPUT_TIF)

    print("Computing road quality from OSM tags")
    graph = compute_road_quality(graph, config["road_quality"])

    known_classes = list(config["traffic_synthetic"]["weekday_peak_multiplier"].keys())
    graph = normalize_highway_class(graph, known_classes)

    OUTPUT_GRAPHML.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, OUTPUT_GRAPHML)
    print(f"Saved full graph -> {OUTPUT_GRAPHML}")

    print("Building synthetic traffic profile")
    synthetic_profile = build_synthetic_traffic_profile(config["traffic_synthetic"])

    real_profile = None if args.synthetic_only else _load_real_traffic_samples(graph)
    if real_profile is not None:
        print(f"Blending {len(real_profile)} real traffic sample rows into the synthetic profile")
    profile = blend_traffic_profiles(synthetic_profile, real_profile)

    profile.to_parquet(OUTPUT_TRAFFIC_PROFILE, index=False)
    print(f"Saved traffic profile -> {OUTPUT_TRAFFIC_PROFILE} "
          f"({(profile['source'] == 'real').sum()} real rows, "
          f"{(profile['source'] == 'synthetic').sum()} synthetic rows)")


if __name__ == "__main__":
    main()