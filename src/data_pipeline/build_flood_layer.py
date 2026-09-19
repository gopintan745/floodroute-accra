"""
Build a per-edge (or per-cell, then sampled per-edge) flood-risk layer from
elevation and rainfall data.

Input:  data/raw/dem/*, data/raw/rainfall/*
Output: data/processed/flood_risk.tif  (raster)
        + a per-edge flood-risk column merged in build_graph_costs.py
        + data/processed/rainfall_climatology.json (monthly heavy-rain-day frequency)

Design decisions documented in docs/design_decisions.md
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, calculate_default_transform
import yaml
import networkx as nx

try:
    import osmnx as ox
    import networkx as nx
except ImportError:
    ox = None
    nx = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
DEFAULT_OUTPUT_TIF = REPO_ROOT / "data" / "processed" / "flood_risk.tif"
DEFAULT_CLIMATOLOGY_JSON = REPO_ROOT / "data" / "processed" / "rainfall_climatology.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flood-triggering threshold (mm/day) — consistent with load_rainfall_data
FLOOD_THRESHOLD_MM = 50.0


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Copy config/config.example.yaml to "
            "config/config.yaml first."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_study_area_bbox(config: dict) -> tuple[float, float, float, float]:
    """Extract bounding box as (south, west, north, east)."""
    return tuple(config["study_area"]["bbox"])


def find_dem_file(dem_dir: Path) -> Path | None:
    """Find a DEM file in the raw data directory."""
    # Look for common DEM file extensions
    for ext in ["*.tif", "*.tiff", "*.dem"]:
        files = list(dem_dir.glob(ext))
        if files:
            return files[0]
    return None


def find_rainfall_file(rainfall_dir: Path) -> Path | None:
    """Find a rainfall data file in the raw data directory."""
    for ext in ["*.json", "*.nc", "*.tif", "*.tiff", "*.hdf", "*.he5"]:
        files = list(rainfall_dir.glob(ext))
        if files:
            return files[0]
    return None


def load_and_clip_dem(dem_path: Path, bbox: tuple[float, float, float, float],
                       target_resolution: float = 30.0) -> tuple[np.ndarray, rasterio.Affine, dict]:
    """
    Load DEM and clip to study area bounding box.

    Args:
        dem_path: Path to DEM file
        bbox: (south, west, north, east) in degrees
        target_resolution: Target resolution in meters (approx 30m for SRTM)

    Returns:
        (dem_array, transform, profile) clipped to bbox
    """
    south, west, north, east = bbox

    with rasterio.open(dem_path) as src:
        # Calculate the window that covers our bbox
        # rasterio uses (west, south, east, north) for bounds
        window = src.window(west, south, east, north)
        window = window.round_offsets().round_shape()

        # Read the data
        dem_data = src.read(1, window=window)
        transform = src.window_transform(window)

        # Update profile for output
        profile = src.profile.copy()
        profile.update({
            "height": dem_data.shape[0],
            "width": dem_data.shape[1],
            "transform": transform,
            "compress": "lzw",
        })

        # Handle nodata
        nodata = src.nodata
        if nodata is not None:
            dem_data = np.where(dem_data == nodata, np.nan, dem_data)

        return dem_data, transform, profile


def compute_topographic_wetness_index(dem: np.ndarray, transform: rasterio.Affine,
                                       cell_size_m: float = 30.0) -> np.ndarray:
    """
    Compute a simplified Topographic Wetness Index (TWI) proxy.

    TWI = ln(a / tan(beta)) where a is specific catchment area and
    beta is local slope. For a coarse v1, we use a simplified version:
    - Low elevation areas get higher flood risk
    - Flat areas (low slope) get higher flood risk
    - This is a heuristic proxy, not full hydrological modeling

    Returns normalized risk 0-1.
    """
    # Smooth the DEM with a NumPy-only 3x3 mean filter to avoid requiring SciPy.
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    padded = np.pad(dem_filled, 1, mode="edge")
    dem_smooth = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0

    # Compute slope using central differences.
    dy, dx = np.gradient(dem_smooth, cell_size_m, cell_size_m)
    slope = np.sqrt(dx**2 + dy**2)
    slope = np.maximum(slope, 0.001)  # Avoid division by zero

    # Specific catchment area proxy: use inverse of slope as a crude
    # approximation (flatter areas accumulate more water)
    sca_proxy = 1.0 / slope

    # TWI-like index
    twi = np.log(sca_proxy / np.tan(slope + 1e-6))

    # Normalize to 0-1 range using robust percentiles
    valid = np.isfinite(twi)
    if not np.any(valid):
        return np.zeros_like(twi)

    p5, p95 = np.nanpercentile(twi[valid], [5, 95])
    twi_norm = np.clip((twi - p5) / (p95 - p5 + 1e-6), 0, 1)

    # Also incorporate absolute elevation (lower = higher risk)
    elev_valid = np.isfinite(dem)
    if np.any(elev_valid):
        elev_min, elev_max = np.nanpercentile(dem[elev_valid], [5, 95])
        elev_norm = np.clip((elev_max - dem) / (elev_max - elev_min + 1e-6), 0, 1)
        # Combine: 70% TWI, 30% elevation
        risk = 0.7 * twi_norm + 0.3 * elev_norm
    else:
        risk = twi_norm

    return np.clip(risk, 0, 1)


def load_rainfall_data(rainfall_path: Path, bbox: tuple[float, float, float, float],
                        dem_shape: tuple[int, int], dem_transform: rasterio.Affine) -> np.ndarray:
    """
    Load rainfall data and resample to match DEM grid.

    Supports:
    - JSON from Open-Meteo Historical Weather API (point time series)
    - NetCDF (CHIRPS/GPM), GeoTIFF, and other raster formats

    Returns rainfall intensity normalized to 0-1.
    """
    south, west, north, east = bbox

    # First, try to load as JSON (Open-Meteo format)
    try:
        import json
        with open(rainfall_path) as f:
            data = json.load(f)

        # Check if it's Open-Meteo format
        if "daily" in data and "precipitation_sum" in data.get("daily", {}):
            precip = data["daily"]["precipitation_sum"]
            times = data["daily"].get("time", [])

            if not precip:
                logger.warning("Open-Meteo data has no precipitation values, using synthetic")
                return create_synthetic_rainfall(dem_shape)

            # Convert to numpy array
            precip_arr = np.array(precip, dtype=np.float32)
            precip_arr = np.nan_to_num(precip_arr, nan=0.0)

            # Design decision: Use 95th percentile of daily precipitation as the
            # representative "heavy rain" intensity for this location.
            # This captures extreme events better than mean, and is more stable than max.
            # (See design_decisions.md for the single-point justification)
            rep_precip_mm = float(np.percentile(precip_arr, 95))

            logger.info(f"Open-Meteo rainfall: {len(precip_arr)} days, "
                        f"95th percentile = {rep_precip_mm:.1f} mm/day")

            # Normalize using sigmoid around 50mm/day flood threshold
            flood_threshold_mm = 50.0
            rain_norm = 1.0 / (1.0 + np.exp(-(rep_precip_mm - flood_threshold_mm) / 20.0))
            rain_norm = np.clip(rain_norm, 0, 1)

            # Open-Meteo gives a point time series, not a grid.
            # Since the study area is small (~3km), create a uniform field.
            # This is intentional per design decision: single centroid is representative.
            return np.full(dem_shape, rain_norm, dtype=np.float32)

    except json.JSONDecodeError:
        # Not JSON, fall through to raster handling
        pass
    except Exception as e:
        logger.warning(f"Failed to load Open-Meteo JSON: {e}")

    # Try raster formats (NetCDF, GeoTIFF) using xarray/rasterio
    try:
        import xarray as xr
    except ImportError:
        logger.warning("xarray not available, using synthetic rainfall")
        return create_synthetic_rainfall(dem_shape)

    try:
        with xr.open_dataset(rainfall_path) as ds:
            # Find the rainfall variable (common names)
            rain_var = None
            for var_name in ["precipitation", "rainfall", "precip", "pr", "pcp"]:
                if var_name in ds.data_vars:
                    rain_var = ds[var_name]
                    break

            if rain_var is None:
                # Try first data variable
                rain_var = list(ds.data_vars.values())[0]

            # Handle time dimension - take mean over time if present
            if "time" in rain_var.dims:
                rain_data = rain_var.mean(dim="time").values
            else:
                rain_data = rain_var.values

            # Get coordinate arrays
            if "lon" in ds.coords and "lat" in ds.coords:
                lons = ds["lon"].values
                lats = ds["lat"].values
            elif "longitude" in ds.coords and "latitude" in ds.coords:
                lons = ds["longitude"].values
                lats = ds["latitude"].values
            elif "x" in ds.coords and "y" in ds.coords:
                lons = ds["x"].values
                lats = ds["y"].values
            else:
                logger.warning("Could not find coordinate variables in rainfall data")
                return create_synthetic_rainfall(dem_shape)

            # Create a rasterio dataset in memory for reprojection
            # Determine source transform and CRS
            if len(lons) == rain_data.shape[-1] and len(lats) == rain_data.shape[-2]:
                # Regular grid
                if lons[1] > lons[0]:
                    lon_res = lons[1] - lons[0]
                else:
                    lon_res = lons[0] - lons[1]
                if lats[1] > lats[0]:
                    lat_res = lats[1] - lats[0]
                else:
                    lat_res = lats[0] - lats[1]

                src_transform = from_bounds(lons.min(), lats.min(), lons.max(), lats.max(),
                                             rain_data.shape[-1], rain_data.shape[-2])
                src_crs = "EPSG:4326"  # Assume WGS84 for CHIRPS/GPM
            else:
                logger.warning("Irregular rainfall grid, using synthetic")
                return create_synthetic_rainfall(dem_shape)

            # Handle NaN values
            rain_data = np.nan_to_num(rain_data.squeeze(), nan=0.0)

            # Reproject to match DEM grid
            dst_shape = dem_shape
            dst_transform = dem_transform
            dst_crs = "EPSG:4326"  # We work in lat/lon for consistency

            dst_rain = np.zeros(dst_shape, dtype=np.float32)

            reproject(
                source=rain_data.astype(np.float32),
                destination=dst_rain,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

            # Normalize rainfall to 0-1 using a flood-triggering threshold
            # CHIRPS is in mm/day; >50mm/day is heavy rain, >100mm/day is extreme
            # Use a sigmoid-like normalization around 50mm/day threshold
            flood_threshold_mm = 50.0
            rain_norm = 1.0 / (1.0 + np.exp(-(dst_rain - flood_threshold_mm) / 20.0))

            return np.clip(rain_norm, 0, 1)

    except Exception as e:
        logger.warning(f"Failed to load rainfall data: {e}, using synthetic")
        return create_synthetic_rainfall(dem_shape)


def create_synthetic_rainfall(shape: tuple[int, int]) -> np.ndarray:
    """Create a synthetic rainfall field for testing."""
    # Create a gradient with some noise
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    # Base gradient (e.g., higher rainfall in north)
    base = 0.3 + 0.4 * (y / h)
    # Add some spatial variation
    noise = np.random.RandomState(42).rand(h, w) * 0.3
    return np.clip(base + noise, 0, 1)


def compute_rainfall_climatology(rainfall_path: Path) -> dict:
    """
    Compute monthly climatology of heavy-rain-day frequency from Open-Meteo data.

    Returns a dict with:
    - monthly_heavy_rain_frequency: list of 12 floats (0-1), probability of a day
      exceeding FLOOD_THRESHOLD_MM in each month (Jan=0, Dec=11)
    - monthly_mean_precip: list of 12 floats, mean daily precip (mm) per month
    - monthly_max_precip: list of 12 floats, max daily precip (mm) per month
    - total_days: int, total days in record
    - heavy_rain_days_total: int, total days exceeding threshold
    - record_start: str, first date in record
    - record_end: str, last date in record
    """
    import json
    from collections import defaultdict

    with open(rainfall_path) as f:
        data = json.load(f)

    if "daily" not in data or "precipitation_sum" not in data["daily"]:
        raise ValueError("Rainfall file does not contain Open-Meteo daily precipitation data")

    precip = data["daily"]["precipitation_sum"]
    times = data["daily"]["time"]

    if not precip or not times:
        raise ValueError("Empty precipitation or time arrays")

    # Group by month
    monthly_precip = defaultdict(list)
    for date_str, p in zip(times, precip):
        if p is None:
            continue
        month = int(date_str.split("-")[1]) - 1  # 0-indexed
        monthly_precip[month].append(float(p))

    # Compute statistics per month
    monthly_heavy_rain_frequency = []
    monthly_mean_precip = []
    monthly_max_precip = []

    for month in range(12):
        vals = monthly_precip.get(month, [])
        if vals:
            heavy_count = sum(1 for v in vals if v >= FLOOD_THRESHOLD_MM)
            freq = heavy_count / len(vals)
            mean_precip = sum(vals) / len(vals)
            max_precip = max(vals)
        else:
            freq = 0.0
            mean_precip = 0.0
            max_precip = 0.0

        monthly_heavy_rain_frequency.append(round(freq, 4))
        monthly_mean_precip.append(round(mean_precip, 2))
        monthly_max_precip.append(round(max_precip, 2))

    # Overall stats
    all_precip = [float(p) for p in precip if p is not None]
    heavy_total = sum(1 for p in all_precip if p >= FLOOD_THRESHOLD_MM)

    climatology = {
        "monthly_heavy_rain_frequency": monthly_heavy_rain_frequency,
        "monthly_mean_precip_mm": monthly_mean_precip,
        "monthly_max_precip_mm": monthly_max_precip,
        "total_days": len(all_precip),
        "heavy_rain_days_total": heavy_total,
        "overall_heavy_rain_frequency": round(heavy_total / len(all_precip), 4) if all_precip else 0.0,
        "record_start": times[0],
        "record_end": times[-1],
        "flood_threshold_mm": FLOOD_THRESHOLD_MM,
    }

    logger.info(f"Rainfall climatology: {heavy_total}/{len(all_precip)} heavy rain days "
                f"({climatology['overall_heavy_rain_frequency']:.1%})")
    logger.info(f"Monthly heavy-rain freq: {[f'{f:.1%}' for f in monthly_heavy_rain_frequency]}")

    return climatology


def save_rainfall_climatology(climatology: dict, output_path: Path) -> None:
    """Save rainfall climatology to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(climatology, f, indent=2)
    logger.info(f"Saved rainfall climatology to {output_path}")


def combine_flood_risk(topo_risk: np.ndarray, rainfall_risk: np.ndarray,
                        topo_weight: float = 0.6, rain_weight: float = 0.4) -> np.ndarray:
    """
    Combine topographic and rainfall risk into final flood risk score.

    Design decision (2026-09-18): Weight topography more heavily (0.6) because
    the DEM is static and reliable, while rainfall is a proxy from satellite
    with coarser resolution and temporal averaging. The 0.6/0.4 split means
    persistent low-lying areas are flagged as risky even in dry periods, while
    rainfall amplifies risk during wet periods.
    """
    combined = topo_weight * topo_risk + rain_weight * rainfall_risk
    return np.clip(combined, 0, 1)


def save_flood_risk_raster(risk_array: np.ndarray, transform: rasterio.Affine,
                            profile: dict, output_path: Path) -> None:
    """Save flood risk raster to GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile.update({
        "dtype": "float32",
        "count": 1,
        "nodata": -9999,
        "compress": "lzw",
    })

    # Replace NaN with nodata
    risk_out = np.where(np.isfinite(risk_array), risk_array, -9999).astype(np.float32)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(risk_out, 1)

    logger.info(f"Saved flood risk raster to {output_path}")


def compute_flood_risk_raster(config: dict | None = None,
                               dem_path: Path | None = None,
                               rainfall_path: Path | None = None,
                               output_path: Path = DEFAULT_OUTPUT_TIF) -> np.ndarray:
    """
    Combine DEM + rainfall into a flood-risk raster.

    Args:
        config: Configuration dict (loads from default if None)
        dem_path: Path to DEM file (auto-detects if None)
        rainfall_path: Path to rainfall file (auto-detects if None)
        output_path: Output GeoTIFF path

    Returns:
        Flood risk array (0-1)
    """
    if config is None:
        config = load_config()

    bbox = get_study_area_bbox(config)

    # Auto-detect input files if not provided
    if dem_path is None:
        dem_path = find_dem_file(REPO_ROOT / "data" / "raw" / "dem")
        if dem_path is None:
            logger.warning("No DEM file found in data/raw/dem/, creating synthetic DEM")
            dem_array, transform, profile = create_synthetic_dem(bbox)
        else:
            logger.info(f"Using DEM: {dem_path}")
            dem_array, transform, profile = load_and_clip_dem(dem_path, bbox)
    else:
        dem_array, transform, profile = load_and_clip_dem(dem_path, bbox)

    if rainfall_path is None:
        rainfall_path = find_rainfall_file(REPO_ROOT / "data" / "raw" / "rainfall")
        if rainfall_path is None:
            logger.warning("No rainfall file found in data/raw/rainfall/, using synthetic")
            rainfall_risk = create_synthetic_rainfall(dem_array.shape)
        else:
            logger.info(f"Using rainfall data: {rainfall_path}")
            rainfall_risk = load_rainfall_data(rainfall_path, bbox, dem_array.shape, transform)
    else:
        rainfall_risk = load_rainfall_data(rainfall_path, bbox, dem_array.shape, transform)

    # Compute topographic flood risk
    logger.info("Computing topographic wetness index...")
    topo_risk = compute_topographic_wetness_index(dem_array, transform)

    # Combine
    logger.info("Combining topographic and rainfall risk...")
    flood_risk = combine_flood_risk(topo_risk, rainfall_risk)

    # Save
    save_flood_risk_raster(flood_risk, transform, profile, output_path)

    return flood_risk


def create_synthetic_dem(bbox: tuple[float, float, float, float],
                          resolution_deg: float = 0.00027) -> tuple[np.ndarray, rasterio.Affine, dict]:
    """Create a synthetic DEM for testing when no real data is available."""
    south, west, north, east = bbox

    height = int((north - south) / resolution_deg)
    width = int((east - west) / resolution_deg)

    # Create synthetic elevation: lower in center (simulating floodplain)
    y, x = np.mgrid[0:height, 0:width]
    y_norm = y / height
    x_norm = x / width

    # Base elevation ~30m (Accra coastal plain)
    base_elev = 30.0
    # Depression in center
    center_y, center_x = 0.5, 0.5
    dist_from_center = np.sqrt((y_norm - center_y)**2 + (x_norm - center_x)**2)
    elevation = base_elev - 15 * np.exp(-dist_from_center**2 / 0.1)
    # Add noise
    elevation += np.random.RandomState(42).randn(height, width) * 2.0

    transform = from_bounds(west, south, east, north, width, height)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999,
    }

    return elevation.astype(np.float32), transform, profile


def sample_risk_along_edges(graph, risk_raster_path: Path | str, sample_distance_m: float = 10.0) :
    """
    Attach a flood_risk attribute to each graph edge by sampling the raster along the edge geometry.

    Args:
        graph: NetworkX MultiDiGraph with edge geometries
        risk_raster_path: Path to flood risk GeoTIFF
        sample_distance_m: Distance between sample points along edge (meters)

    Returns:
        Graph with flood_risk attribute on each edge (mean of samples)
    """
    if ox is None:
        raise ImportError("osmnx and networkx required for sampling")

    with rasterio.open(risk_raster_path) as src:
        risk_data = src.read(1)
        transform = src.transform
        nodata = src.nodata

    # Convert sample distance from meters to degrees (approximate at Accra latitude)
    # 1 degree ≈ 111km at equator, but varies with latitude
    # At ~5.5°N, 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(5.5°) ≈ 110.5km
    sample_distance_deg = sample_distance_m / 111000.0

    for u, v, key, data in graph.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            # No geometry - assign nodata
            data["flood_risk"] = -9999
            continue

        # Sample points along the line
        length = geom.length
        if length == 0:
            data["flood_risk"] = -9999
            continue

        num_samples = max(2, int(length / sample_distance_deg) + 1)
        distances = np.linspace(0, length, num_samples)
        points = [geom.interpolate(d) for d in distances]

        # Sample raster at each point
        risks = []
        for pt in points:
            # Convert point coordinates to pixel indices
            col, row = ~transform * (pt.x, pt.y)
            col, row = int(col), int(row)

            if 0 <= row < risk_data.shape[0] and 0 <= col < risk_data.shape[1]:
                val = risk_data[row, col]
                if nodata is None or val != nodata:
                    risks.append(val)

        if risks:
            data["flood_risk"] = float(np.mean(risks))
        else:
            data["flood_risk"] = -9999

    # Replace nodata with NaN for easier handling downstream
    for u, v, key, data in graph.edges(keys=True, data=True):
        if data.get("flood_risk", -9999) == -9999:
            data["flood_risk"] = np.nan

    logger.info(f"Sampled flood risk for {graph.number_of_edges()} edges")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build flood risk layer from DEM and rainfall data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dem", type=Path, help="Path to DEM file")
    parser.add_argument("--rainfall", type=Path, help="Path to rainfall data file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_TIF)
    parser.add_argument("--graph", type=Path, help="Path to road graph GraphML for edge sampling")
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY_JSON,
                        help="Output path for rainfall climatology JSON")
    parser.add_argument("--climatology-only", action="store_true",
                        help="Only compute rainfall climatology, skip flood risk raster")
    args = parser.parse_args()

    config = load_config(args.config)

    # Determine rainfall file path
    rainfall_path = args.rainfall
    if rainfall_path is None:
        rainfall_path = find_rainfall_file(REPO_ROOT / "data" / "raw" / "rainfall")

    # Compute rainfall climatology if rainfall data is available
    if rainfall_path and rainfall_path.exists():
        logger.info(f"Computing rainfall climatology from {rainfall_path}")
        climatology = compute_rainfall_climatology(rainfall_path)
        save_rainfall_climatology(climatology, args.climatology)

    if args.climatology_only:
        logger.info("Climatology-only mode, skipping flood risk raster")
        return

    # Compute flood risk raster
    flood_risk = compute_flood_risk_raster(
        config=config,
        dem_path=args.dem,
        rainfall_path=args.rainfall,
        output_path=args.output,
    )

    logger.info(f"Flood risk raster computed: shape={flood_risk.shape}, "
                f"min={np.nanmin(flood_risk):.3f}, max={np.nanmax(flood_risk):.3f}, "
                f"mean={np.nanmean(flood_risk):.3f}")

    # Optionally sample along graph edges
    if args.graph and args.graph.exists():
        logger.info(f"Loading graph from {args.graph}")
        graph = ox.load_graphml(args.graph)
        graph = sample_risk_along_edges(graph, args.output)

        # Save updated graph
        output_graph = args.graph.with_stem(args.graph.stem + "_with_flood")
        ox.save_graphml(graph, output_graph)
        logger.info(f"Saved graph with flood risk to {output_graph}")

    # Print summary stats for design_decisions.md documentation
    print("\n=== Flood Risk Layer Summary ===")
    print(f"Output: {args.output}")
    print(f"Shape: {flood_risk.shape}")
    print(f"Min: {np.nanmin(flood_risk):.4f}")
    print(f"Max: {np.nanmax(flood_risk):.4f}")
    print(f"Mean: {np.nanmean(flood_risk):.4f}")
    print(f"Std: {np.nanstd(flood_risk):.4f}")
    print(f"Percentiles (10, 25, 50, 75, 90): "
          f"{np.nanpercentile(flood_risk, [10, 25, 50, 75, 90])}")


if __name__ == "__main__":
    main()
