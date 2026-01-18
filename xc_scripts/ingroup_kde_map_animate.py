"""
Visualize an ingroup's geographic distribution with KDE isoclines on a map.

Edit the paths and settings below, then run:
    python xc_scripts/ingroup_kde_map_animate.py
    python xc_scripts/ingroup_kde_map_animate.py --animate
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from bokeh.io import output_file, show
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    Label,
    Range1d,
    Slider,
)
from bokeh.plotting import figure
from KDEpy import FFTKDE
from pyproj import Transformer
from skimage import measure
from xyzservices import providers as xyz_providers

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
METADATA_CSV = Path("/Volumes/Z Slim/zslim_birdcluster/embeddings/chloris_chloris/metadata.csv")
INFERENCE_CSV = Path("/Volumes/Z Slim/zslim_birdcluster/embeddings/chloris_chloris/inference.csv")
OUTPUT_HTML = Path("ingroup_kde_map.html")
ANIMATION_OUTPUT_HTML = Path("ingroup_kde_map_animated.html")

FILTER_ONE_PER_RECORDIST = True

NUM_ISOCLINES = 8
FILL_ISOCLINES = True
HDR_MIN_PROB = 0.1
HDR_MAX_PROB = 0.8

DATE_FILTER_MODE = "recent"  # "all", "recent", or "exclude_recent"
DATE_FILTER_YEARS = 11
DATE_PARSE_FORMAT = "%m/%d/%Y"
DATE_PARSE_DAYFIRST = False
DATE_PARSE_FALLBACK = True

KDE_GRID_SIZE = 256
BANDWIDTH_METHOD = "knn"  # "cv", "scott", "knn", or "manual" (after scaling)
BANDWIDTH_MANUAL = None  # Example: 0.2
BANDWIDTH_KNN_K = 5  # Kth neighbor distance used for "knn" bandwidth.
BANDWIDTH_KNN_QUANTILE = 0.5  # Quantile of kth distances (0-1).
BANDWIDTH_KNN_SCALE = 0.42  # Multiplier for the "knn" bandwidth.
BANDWIDTH_GRID_SIZE = 20
BANDWIDTH_CV_FOLDS = 5
BANDWIDTH_SEARCH_LOG_MIN = -1.0
BANDWIDTH_SEARCH_LOG_MAX = 1.0
BANDWIDTH_SAMPLE_SIZE = 5000
RANDOM_SEED = 7

AUTO_ZOOM = False
EUROPE_BOUNDS = {
    "lon_min": -12.0,
    "lon_max": 35.0,
    "lat_min": 20.0,
    "lat_max": 78.0,
}

PLOT_WIDTH = 1100
PLOT_HEIGHT = 700
POINT_SIZE =8
POINT_ALPHA = 0.7
MAX_ABS_LAT = 85.0
MIN_POINTS_FOR_KDE = 5
YEAR_LABEL_LAT = 70.0

ANIMATION_WINDOW_MONTHS = 12  # 1-year window per frame.
ANIMATION_STEP_MONTHS = 1  # Advance one month per frame.
ANIMATION_PLAY_INTERVAL_MS = 100  # Playback interval for the HTML animation.


@dataclass(frozen=True)
class Isopleth:
    """Container for a single KDE isopleth."""

    probability: float
    level: float
    paths: list[np.ndarray]


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file or raise a helpful error."""
    if not path.exists():
        raise SystemExit(f"Missing CSV file: {path}")
    return pd.read_csv(path)


def normalize_xcid(value: object) -> str:
    """Normalize XCIDs to stable string keys."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric = int(float(text))
        return str(numeric)
    except ValueError:
        return text


def normalize_clip_index(value: object) -> str:
    """Normalize clip indices to zero-padded 2-digit strings."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.isdigit():
        return f"{int(text):02d}"
    digits = re.sub(r"\D", "", text)
    if digits:
        return f"{int(digits):02d}"
    return text


def parse_date_column(series: pd.Series) -> tuple[pd.Series, int]:
    """Parse date strings using configured formats."""
    raw = series.astype(str).str.strip()
    raw = raw.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    if DATE_PARSE_FORMAT:
        parsed = pd.to_datetime(raw, format=DATE_PARSE_FORMAT, errors="coerce")
    else:
        parsed = pd.to_datetime(raw, errors="coerce", dayfirst=DATE_PARSE_DAYFIRST)

    if DATE_PARSE_FALLBACK:
        missing = parsed.isna() & raw.notna()
        if missing.any():
            fallback = pd.to_datetime(
                raw[missing], errors="coerce", dayfirst=DATE_PARSE_DAYFIRST
            )
            parsed.loc[missing] = fallback

    invalid = parsed.isna() & raw.notna()
    return parsed, int(invalid.sum())


def prepare_inference_table(path: Path) -> pd.DataFrame:
    """Load inference.csv and extract xcid/clip_index from filenames."""
    df = read_csv(path)
    if "filename" not in df.columns:
        raise SystemExit("Inference CSV must contain a 'filename' column.")

    filenames = df["filename"].astype(str).map(lambda value: Path(value).name)
    extracted = filenames.str.extract(
        r"_(?P<xcid>\d+)_(?P<clip_index>\d{2})(?:\.[^.]+)?$"
    )
    missing = extracted["xcid"].isna().sum()
    if missing:
        print(
            f"[WARN] {missing} rows in inference.csv could not parse xcid/clip_index."
        )

    df = df.join(extracted)
    df = df.dropna(subset=["xcid", "clip_index"]).copy()
    df["xcid"] = df["xcid"].astype(str)
    df["clip_index"] = df["clip_index"].astype(str).str.zfill(2)

    if "logits" not in df.columns:
        print(
            "[WARN] Inference CSV missing 'logits'; recordist filter will be skipped."
        )
        df["logits"] = np.nan
    else:
        df["logits"] = pd.to_numeric(df["logits"], errors="coerce")
    return df


def prepare_metadata_table(path: Path) -> pd.DataFrame:
    """Load metadata.csv and normalize key columns."""
    df = read_csv(path)
    required = {"xcid", "clip_index", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            "Metadata CSV missing required columns: " + ", ".join(sorted(missing))
        )
    df = df.copy()
    df["xcid_norm"] = df["xcid"].apply(normalize_xcid)
    df["clip_index_norm"] = df["clip_index"].apply(normalize_clip_index)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    if "date" in df.columns:
        parsed, invalid_count = parse_date_column(df["date"])
        df["date_dt"] = parsed
        if invalid_count:
            print(f"[WARN] {invalid_count} metadata rows have invalid dates.")
    return df


def merge_inference_metadata(
    inference_df: pd.DataFrame, metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """Join inference rows to metadata coordinates."""
    merged = inference_df.merge(
        metadata_df,
        how="left",
        left_on=["xcid", "clip_index"],
        right_on=["xcid_norm", "clip_index_norm"],
        suffixes=("", "_meta"),
    )
    missing_coords = merged["lat"].isna() | merged["lon"].isna()
    if missing_coords.any():
        print(
            f"[WARN] {missing_coords.sum()} inference rows missing lat/lon matches."
        )
    return merged


def apply_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows based on DATE_FILTER_MODE relative to the latest date."""
    mode = DATE_FILTER_MODE.strip().lower()
    if mode == "all":
        return df
    if mode not in {"recent", "exclude_recent"}:
        raise SystemExit(
            "DATE_FILTER_MODE must be 'all', 'recent', or 'exclude_recent'."
        )
    if DATE_FILTER_YEARS <= 0:
        raise SystemExit("DATE_FILTER_YEARS must be > 0.")

    if "date_dt" in df.columns:
        date_series = df["date_dt"]
    elif "date" in df.columns:
        parsed, invalid_count = parse_date_column(df["date"])
        if invalid_count:
            print(f"[WARN] {invalid_count} metadata rows have invalid dates.")
        date_series = parsed
    else:
        print("[WARN] No date column available; skipping date filter.")
        return df

    if date_series.notna().sum() == 0:
        print("[WARN] No valid dates; skipping date filter.")
        return df

    reference_date = date_series.max()
    cutoff = reference_date - pd.DateOffset(years=DATE_FILTER_YEARS)

    if mode == "recent":
        mask = date_series >= cutoff
    else:
        mask = date_series < cutoff

    filtered = df.loc[mask].copy()
    dropped_invalid = date_series.isna().sum()
    if dropped_invalid:
        print(
            f"[WARN] Dropped {dropped_invalid} rows without valid dates for filtering."
        )
    print(
        f"[INFO] Date filter '{mode}' kept {len(filtered)} of {len(df)} rows "
        f"(cutoff={cutoff.date()}, reference={reference_date.date()})."
    )
    return filtered


def filter_top_logit_per_recordist(
    df: pd.DataFrame, *, log: bool = True
) -> pd.DataFrame:
    """Keep the highest-logit entry per recordist when enabled."""
    if not FILTER_ONE_PER_RECORDIST:
        return df
    if "recordist" not in df.columns:
        if log:
            print("[WARN] No 'recordist' column in metadata; skipping filter.")
        return df
    if "logits" not in df.columns:
        if log:
            print("[WARN] No 'logits' column in inference; skipping filter.")
        return df

    df = df.copy()
    recordist = df["recordist"].astype(str).str.strip()
    recordist_lower = recordist.str.lower()
    has_recordist = (
        recordist_lower.ne("")
        & recordist_lower.ne("nan")
        & recordist_lower.ne("none")
    )
    df.loc[has_recordist, "recordist"] = recordist[has_recordist]
    df["logits"] = pd.to_numeric(df["logits"], errors="coerce")

    with_recordist = df[has_recordist].sort_values("logits", ascending=False)
    filtered = with_recordist.drop_duplicates("recordist", keep="first")
    without_recordist = df[~has_recordist]
    combined = pd.concat([filtered, without_recordist], ignore_index=True)
    if log:
        print(
            f"[INFO] Recordist filter kept {len(combined)} of {len(df)} rows."
        )
    return combined


def prepare_points(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return filtered dataframe plus lon/lat arrays."""
    mask = df["lon"].notna() & df["lat"].notna()
    filtered = df.loc[mask].copy()
    lon = filtered["lon"].to_numpy(dtype=float)
    lat = filtered["lat"].to_numpy(dtype=float)
    valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= MAX_ABS_LAT)
    if not valid.all():
        filtered = filtered.loc[valid].copy()
        lon = lon[valid]
        lat = lat[valid]
    return filtered, lon, lat


def standardize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center and scale points to stabilize bandwidth selection."""
    center = np.mean(points, axis=0)
    scale = np.std(points, axis=0, ddof=1)
    scale[scale == 0] = 1.0
    standardized = (points - center) / scale
    return standardized, center, scale


def estimate_scott_bandwidth(points: np.ndarray) -> float:
    """Scott's rule bandwidth for standardized points."""
    n_samples, n_dims = points.shape
    if n_samples <= 1:
        return 1.0
    return float(n_samples ** (-1.0 / (n_dims + 4)))


def estimate_neighbor_scale(
    points: np.ndarray, n_neighbors: int, quantile: float
) -> float:
    """Estimate typical spacing using k-nearest-neighbor distances."""
    if len(points) < 2:
        return 1.0
    try:
        from sklearn.neighbors import NearestNeighbors

        n_neighbors = max(1, min(n_neighbors, len(points) - 1))
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn.fit(points)
        distances, _ = nn.kneighbors(points)
        kth_distances = distances[:, n_neighbors]
        return float(np.quantile(kth_distances, quantile))
    except Exception:  # noqa: BLE001
        return float(np.mean(np.std(points, axis=0)))


def select_optimal_bandwidth(points: np.ndarray) -> float:
    """Select a CV bandwidth using KDE on standardized data."""
    if len(points) < 5:
        return estimate_scott_bandwidth(points)

    sample = points
    if len(points) > BANDWIDTH_SAMPLE_SIZE:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(points), size=BANDWIDTH_SAMPLE_SIZE, replace=False)
        sample = points[idx]

    base = estimate_neighbor_scale(
        sample,
        n_neighbors=BANDWIDTH_KNN_K,
        quantile=BANDWIDTH_KNN_QUANTILE,
    )
    if not np.isfinite(base) or base <= 0:
        base = estimate_scott_bandwidth(sample)

    bandwidths = base * np.logspace(
        BANDWIDTH_SEARCH_LOG_MIN, BANDWIDTH_SEARCH_LOG_MAX, BANDWIDTH_GRID_SIZE
    )

    try:
        from sklearn.model_selection import GridSearchCV
        from sklearn.neighbors import KernelDensity

        cv_folds = min(BANDWIDTH_CV_FOLDS, len(sample))
        cv_folds = max(cv_folds, 2)
        grid = GridSearchCV(
            KernelDensity(kernel="gaussian"),
            {"bandwidth": bandwidths},
            cv=cv_folds,
            n_jobs=-1,
        )
        grid.fit(sample)
        best = float(grid.best_params_["bandwidth"])
        return max(best, 1e-3)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Bandwidth selection failed ({exc}); using Scott's rule.")
        return estimate_scott_bandwidth(points)


def select_bandwidth(points: np.ndarray) -> float:
    """Select a bandwidth based on the configured method."""
    method = BANDWIDTH_METHOD.strip().lower()
    if method == "manual":
        if BANDWIDTH_MANUAL is None:
            raise SystemExit(
                "BANDWIDTH_MANUAL must be set when BANDWIDTH_METHOD='manual'."
            )
        if BANDWIDTH_MANUAL <= 0:
            raise SystemExit("BANDWIDTH_MANUAL must be > 0.")
        return float(BANDWIDTH_MANUAL)
    if method == "scott":
        return estimate_scott_bandwidth(points)
    if method == "knn":
        if BANDWIDTH_KNN_K < 1:
            raise SystemExit("BANDWIDTH_KNN_K must be >= 1.")
        if not 0 <= BANDWIDTH_KNN_QUANTILE <= 1:
            raise SystemExit("BANDWIDTH_KNN_QUANTILE must be between 0 and 1.")
        scale = estimate_neighbor_scale(
            points,
            n_neighbors=BANDWIDTH_KNN_K,
            quantile=BANDWIDTH_KNN_QUANTILE,
        )
        scale *= BANDWIDTH_KNN_SCALE
        if not np.isfinite(scale) or scale <= 0:
            print("[WARN] KNN bandwidth invalid; falling back to Scott's rule.")
            return estimate_scott_bandwidth(points)
        return float(scale)
    if method == "cv":
        return select_optimal_bandwidth(points)
    raise SystemExit(f"Unknown BANDWIDTH_METHOD: {BANDWIDTH_METHOD}")


def compute_kde(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute KDE grid values for projected points."""
    standardized, center, scale = standardize_points(points)
    bandwidth = select_bandwidth(standardized)
    kde = FFTKDE(bw=bandwidth, kernel="gaussian")
    grid_scaled, values = kde.fit(standardized).evaluate(grid_points=KDE_GRID_SIZE)
    grid = grid_scaled * scale + center
    return grid, values, bandwidth


def reshape_kde_arrays(
    grid: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape KDE grid output into axis arrays."""
    total_points = len(values)
    grid_dim = int(np.sqrt(total_points))
    if grid_dim * grid_dim != total_points:
        raise ValueError("Unexpected KDE grid size.")
    coords = grid.reshape(grid_dim, grid_dim, 2)
    values = values.reshape(grid_dim, grid_dim)
    x_axis = coords[:, 0, 0]
    y_axis = coords[0, :, 1]
    return x_axis, y_axis, values


def compute_hdr_levels(
    values: np.ndarray, probabilities: Sequence[float]
) -> dict[float, float | None]:
    """Convert HDR probabilities into density thresholds."""
    flat = values.flatten()
    total = float(flat.sum())
    if total <= 0:
        return {prob: None for prob in probabilities}
    order = np.argsort(flat)[::-1]
    sorted_vals = flat[order]
    cumulative = np.cumsum(sorted_vals)
    levels: dict[float, float | None] = {}
    for prob in probabilities:
        target = prob * total
        idx = int(np.searchsorted(cumulative, target))
        idx = min(idx, len(sorted_vals) - 1)
        levels[prob] = float(sorted_vals[idx])
    return levels


def contour_paths_from_values(
    x_axis: np.ndarray, y_axis: np.ndarray, values: np.ndarray, level: float
) -> list[np.ndarray]:
    """Extract contour paths at a given density level."""
    contours = measure.find_contours(values, level=level)
    if not contours:
        return []
    grid_indices = np.arange(len(x_axis))
    col_indices = np.arange(len(y_axis))
    paths: list[np.ndarray] = []
    for contour in contours:
        rows = contour[:, 0]
        cols = contour[:, 1]
        xs = np.interp(rows, grid_indices, x_axis)
        ys = np.interp(cols, col_indices, y_axis)
        paths.append(np.column_stack((xs, ys)))
    return paths


def close_paths(paths: Iterable[np.ndarray]) -> list[np.ndarray]:
    """Ensure contour paths are closed for polygon filling."""
    closed: list[np.ndarray] = []
    for path in paths:
        if len(path) < 3:
            continue
        if not np.allclose(path[0], path[-1]):
            path = np.vstack([path, path[0]])
        closed.append(path)
    return closed


def build_isopleths(
    grid: np.ndarray, values: np.ndarray, num_levels: int
) -> list[Isopleth]:
    """Build isopleths for the requested number of levels."""
    if num_levels <= 0:
        return []
    x_axis, y_axis, reshaped_values = reshape_kde_arrays(grid, values)
    probabilities = np.linspace(HDR_MIN_PROB, HDR_MAX_PROB, num_levels)
    levels = compute_hdr_levels(reshaped_values, probabilities)

    isopleths: list[Isopleth] = []
    for prob in probabilities:
        level = levels.get(prob)
        if level is None:
            continue
        paths = contour_paths_from_values(x_axis, y_axis, reshaped_values, level)
        if not paths:
            continue
        isopleths.append(
            Isopleth(probability=prob, level=level, paths=close_paths(paths))
        )
    return isopleths


def project_isopleths(
    isopleths: Iterable[Isopleth], transformer: Transformer
) -> list[Isopleth]:
    """Project isopleth paths to a new CRS."""
    projected: list[Isopleth] = []
    for iso in isopleths:
        projected_paths: list[np.ndarray] = []
        for path in iso.paths:
            x_vals, y_vals = transformer.transform(path[:, 0], path[:, 1])
            projected_paths.append(np.column_stack((x_vals, y_vals)))
        projected.append(
            Isopleth(
                probability=iso.probability,
                level=iso.level,
                paths=projected_paths,
            )
        )
    return projected


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color to RGB."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Iterable[float]) -> str:
    """Convert RGB floats to a hex color."""
    clamped = [max(0, min(255, int(round(channel)))) for channel in rgb]
    return "#{:02x}{:02x}{:02x}".format(*clamped)


def build_heat_palette(num_colors: int) -> list[str]:
    """Create a blue-green-yellow-red palette of the requested length."""
    anchors = ["#0b1d8b", "#00a878", "#f4d03f", "#e53935"]
    if num_colors <= 1:
        return [anchors[-1]]

    anchor_rgb = np.array([hex_to_rgb(color) for color in anchors], dtype=float)
    anchor_pos = np.linspace(0, 1, len(anchors))
    positions = np.linspace(0, 1, num_colors)

    colors: list[str] = []
    for pos in positions:
        idx = int(np.searchsorted(anchor_pos, pos)) - 1
        idx = max(0, min(idx, len(anchors) - 2))
        frac = (pos - anchor_pos[idx]) / (anchor_pos[idx + 1] - anchor_pos[idx])
        rgb = anchor_rgb[idx] + frac * (anchor_rgb[idx + 1] - anchor_rgb[idx])
        colors.append(rgb_to_hex(rgb))
    return colors


def compute_bounds(
    points_x: np.ndarray,
    points_y: np.ndarray,
    isopleths: Iterable[Isopleth],
) -> tuple[Range1d, Range1d]:
    """Compute plot bounds from points and isopleths."""
    all_x = points_x.tolist()
    all_y = points_y.tolist()
    for iso in isopleths:
        for path in iso.paths:
            all_x.extend(path[:, 0].tolist())
            all_y.extend(path[:, 1].tolist())

    if AUTO_ZOOM and all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        pad_x = max((max_x - min_x) * 0.08, 1000.0)
        pad_y = max((max_y - min_y) * 0.08, 1000.0)
        return (
            Range1d(min_x - pad_x, max_x + pad_x),
            Range1d(min_y - pad_y, max_y + pad_y),
        )

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    lon_min = EUROPE_BOUNDS["lon_min"]
    lon_max = EUROPE_BOUNDS["lon_max"]
    lat_min = EUROPE_BOUNDS["lat_min"]
    lat_max = EUROPE_BOUNDS["lat_max"]
    x_vals, y_vals = transformer.transform(
        [lon_min, lon_max], [lat_min, lat_max]
    )
    return Range1d(min(x_vals), max(x_vals)), Range1d(min(y_vals), max(y_vals))


def project_points(
    lon: np.ndarray, lat: np.ndarray, transformer: Transformer
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat arrays into the target CRS."""
    x_vals, y_vals = transformer.transform(lon, lat)
    return np.asarray(x_vals), np.asarray(y_vals)


def build_point_source_data(
    points_df: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    x_merc: np.ndarray,
    y_merc: np.ndarray,
) -> dict[str, list]:
    """Assemble the scatter data for Bokeh point rendering."""
    recordist_series = points_df.get("recordist")
    if recordist_series is None:
        recordist_values = [""] * len(lon)
    else:
        recordist_values = recordist_series.fillna("").astype(str).tolist()

    filename_series = points_df.get("filename")
    if filename_series is None:
        filename_values = [""] * len(lon)
    else:
        filename_values = filename_series.fillna("").astype(str).tolist()

    logits_series = points_df.get("logits")
    if logits_series is None:
        logits_values = [np.nan] * len(lon)
    else:
        logits_values = pd.to_numeric(logits_series, errors="coerce").to_list()

    return {
        "x": x_merc.tolist(),
        "y": y_merc.tolist(),
        "lon": lon.tolist(),
        "lat": lat.tolist(),
        "recordist": recordist_values,
        "logits": logits_values,
        "filename": filename_values,
    }


def align_isopleth_paths(
    isopleths: Iterable[Isopleth], probabilities: Sequence[float]
) -> list[list[np.ndarray]]:
    """Align isopleth paths to the requested probability order."""
    lookup = {
        round(float(iso.probability), 6): iso.paths for iso in isopleths
    }
    aligned: list[list[np.ndarray]] = []
    for prob in probabilities:
        aligned.append(lookup.get(round(float(prob), 6), []))
    return aligned


def build_isopleth_source_data(
    paths: Iterable[np.ndarray],
) -> dict[str, list[list[float]]]:
    """Convert isopleth paths to patch source data."""
    xs_list = [path[:, 0].tolist() for path in paths]
    ys_list = [path[:, 1].tolist() for path in paths]
    return {"xs": xs_list, "ys": ys_list}


def empty_isopleth_sources(
    num_levels: int,
) -> list[dict[str, list[list[float]]]]:
    """Create empty isopleth sources for frames without KDE data."""
    return [{"xs": [], "ys": []} for _ in range(num_levels)]


def plot_map(
    points_df: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    isopleths: list[Isopleth],
    bandwidth: float,
    output_html: Path | None = None,
    title: str | None = None,
) -> None:
    """Render the map, points, and KDE isopleths."""
    point_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_merc, y_merc = project_points(lon, lat, point_transformer)

    contour_transformer = Transformer.from_crs("EPSG:3035", "EPSG:3857", always_xy=True)
    projected_isopleths = project_isopleths(isopleths, contour_transformer)

    x_range, y_range = compute_bounds(
        np.array(x_merc), np.array(y_merc), projected_isopleths
    )

    plot_title = title or "Ingroup KDE Isoclines (Web Mercator)"
    plot = figure(
        title=plot_title,
        x_axis_type="mercator",
        y_axis_type="mercator",
        match_aspect=True,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        tools="pan,wheel_zoom,reset,save",
        x_range=x_range,
        y_range=y_range,
    )
    plot.add_tile(xyz_providers.CartoDB.Positron)

    palette = build_heat_palette(len(projected_isopleths))
    ordered_isopleths = sorted(projected_isopleths, key=lambda iso: iso.level)

    if FILL_ISOCLINES:
        for iso, color in zip(ordered_isopleths, palette):
            xs_list = [path[:, 0].tolist() for path in iso.paths]
            ys_list = [path[:, 1].tolist() for path in iso.paths]
            if xs_list:
                plot.patches(
                    xs_list,
                    ys_list,
                    fill_color=color,
                    fill_alpha=0.25,
                    line_width=0,
                )
    else:
        for iso, color in zip(ordered_isopleths, palette):
            xs_list = [path[:, 0].tolist() for path in iso.paths]
            ys_list = [path[:, 1].tolist() for path in iso.paths]
            if xs_list:
                plot.multi_line(
                    xs=xs_list,
                    ys=ys_list,
                    line_color=color,
                    line_width=2,
                    legend_label=f"HDR {iso.probability:.0%}",
                )

    source = ColumnDataSource(
        build_point_source_data(points_df, lon, lat, x_merc, y_merc)
    )
    point_renderer = plot.scatter(
        x="x",
        y="y",
        size=POINT_SIZE,
        color="white",
        alpha=POINT_ALPHA,
        line_color="black",
        line_width=0.5,
        source=source,
    )
    hover = HoverTool(
        renderers=[point_renderer],
        tooltips=[
            ("lon, lat", "@lon{0.000}, @lat{0.000}"),
            ("recordist", "@recordist"),
            ("logits", "@logits{0.000}"),
            ("filename", "@filename"),
        ],
    )
    plot.add_tools(hover)

    if not FILL_ISOCLINES:
        plot.legend.location = "top_left"
        plot.legend.click_policy = "hide"

    output_path = output_html or OUTPUT_HTML
    output_file(output_path, title=plot_title)
    show(plot)

    print(
        f"[INFO] Rendered {len(lon)} points with bandwidth={bandwidth:.4f} "
        f"(method={BANDWIDTH_METHOD})."
    )
    print(f"[INFO] Output written to {output_path}.")


def prepare_base_dataframe() -> pd.DataFrame:
    """Load data, merge inference/metadata, and apply the date filter."""
    inference_df = prepare_inference_table(INFERENCE_CSV)
    metadata_df = prepare_metadata_table(METADATA_CSV)
    merged = merge_inference_metadata(inference_df, metadata_df)
    return apply_date_filter(merged)


def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a parsed datetime column exists for animation windows."""
    if "date_dt" in df.columns:
        return df
    if "date" not in df.columns:
        raise SystemExit("Animation requires a 'date' column in metadata.")
    parsed, invalid_count = parse_date_column(df["date"])
    df = df.copy()
    df["date_dt"] = parsed
    if invalid_count:
        print(f"[WARN] {invalid_count} metadata rows have invalid dates.")
    return df


def build_monthly_windows(
    date_series: pd.Series, window_months: int, step_months: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Build monthly sliding windows for the animation."""
    if window_months <= 0:
        raise SystemExit("ANIMATION_WINDOW_MONTHS must be > 0.")
    if step_months <= 0:
        raise SystemExit("ANIMATION_STEP_MONTHS must be > 0.")
    if date_series.empty:
        return []

    min_date = date_series.min()
    max_date = date_series.max()
    if pd.isna(min_date) or pd.isna(max_date):
        return []

    start = min_date.to_period("M").to_timestamp()
    end = (max_date + pd.offsets.MonthBegin(1)).to_period("M").to_timestamp()
    last_start = end - pd.DateOffset(months=window_months)

    frame_starts: list[pd.Timestamp] = []
    if last_start < start:
        frame_starts = [start]
    else:
        current = start
        step = pd.DateOffset(months=step_months)
        while current <= last_start:
            frame_starts.append(current)
            current += step

    return [
        (frame_start, frame_start + pd.DateOffset(months=window_months))
        for frame_start in frame_starts
    ]


def format_frame_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Format a 1-year window label for display."""
    end_label = (end - pd.DateOffset(days=1)).strftime("%Y-%m")
    return f"{start.strftime('%Y-%m')} to {end_label}"


def build_animation_frames(
    df: pd.DataFrame,
    windows: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
    probabilities: Sequence[float],
    point_transformer: Transformer,
    equal_area: Transformer,
    contour_transformer: Transformer,
) -> list[dict[str, object]]:
    """Compute per-frame KDEs and sources for the animation."""
    date_series = df["date_dt"]
    frames: list[dict[str, object]] = []

    for start, end in windows:
        mask = (date_series >= start) & (date_series < end)
        frame_df = df.loc[mask].copy()
        frame_df = filter_top_logit_per_recordist(frame_df, log=False)
        points_df, lon, lat = prepare_points(frame_df)
        x_merc, y_merc = project_points(lon, lat, point_transformer)
        point_data = build_point_source_data(points_df, lon, lat, x_merc, y_merc)

        if len(lon) < MIN_POINTS_FOR_KDE:
            isopleth_sources = empty_isopleth_sources(len(probabilities))
            bandwidth = float("nan")
        else:
            x_eq, y_eq = equal_area.transform(lon, lat)
            points_eq = np.column_stack((x_eq, y_eq))
            grid, values, bandwidth = compute_kde(points_eq)
            isopleths = build_isopleths(grid, values, NUM_ISOCLINES)
            projected_isopleths = project_isopleths(
                isopleths, contour_transformer
            )
            aligned_paths = align_isopleth_paths(
                projected_isopleths, probabilities
            )
            isopleth_sources = [
                build_isopleth_source_data(paths) for paths in aligned_paths
            ]

        frame_label = format_frame_label(start, end)
        if np.isfinite(bandwidth):
            bandwidth_text = (
                f"Bandwidth: {bandwidth:.4f} (method={BANDWIDTH_METHOD})"
            )
        else:
            bandwidth_text = "Bandwidth: n/a"

        frames.append(
            {
                "points": point_data,
                "isopleths": isopleth_sources,
                "title": f"Ingroup KDE (Window {frame_label})",
                "frame_label": f"Window: {frame_label}",
                "bandwidth_text": bandwidth_text,
                "year_text": f"{start.year}",
            }
        )

    return frames


def plot_animated_map(
    frames: list[dict[str, object]],
    probabilities: Sequence[float],
    x_range: Range1d,
    y_range: Range1d,
    output_html: Path,
) -> None:
    """Render an animated KDE map with a month-by-month slider."""
    if not frames:
        raise SystemExit("No animation frames to render.")

    first_frame = frames[0]
    palette = build_heat_palette(len(probabilities))
    render_order = list(range(len(probabilities) - 1, -1, -1))

    plot = figure(
        title=first_frame["title"],
        x_axis_type="mercator",
        y_axis_type="mercator",
        match_aspect=True,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        tools="pan,wheel_zoom,reset,save",
        x_range=x_range,
        y_range=y_range,
    )
    plot.add_tile(xyz_providers.CartoDB.Positron)

    point_source = ColumnDataSource(first_frame["points"])
    point_renderer = plot.scatter(
        x="x",
        y="y",
        size=POINT_SIZE,
        color="white",
        alpha=POINT_ALPHA,
        line_color="black",
        line_width=0.5,
        source=point_source,
    )
    hover = HoverTool(
        renderers=[point_renderer],
        tooltips=[
            ("lon, lat", "@lon{0.000}, @lat{0.000}"),
            ("recordist", "@recordist"),
            ("logits", "@logits{0.000}"),
            ("filename", "@filename"),
        ],
    )
    plot.add_tools(hover)

    iso_sources: list[ColumnDataSource] = []
    for draw_idx, source_idx in enumerate(render_order):
        prob = probabilities[source_idx]
        source = ColumnDataSource(first_frame["isopleths"][source_idx])
        iso_sources.append(source)
        if FILL_ISOCLINES:
            plot.patches(
                "xs",
                "ys",
                source=source,
                fill_color=palette[draw_idx],
                fill_alpha=0.25,
                line_width=0,
            )
        else:
            plot.multi_line(
                xs="xs",
                ys="ys",
                source=source,
                line_color=palette[draw_idx],
                line_width=2,
                legend_label=f"HDR {prob:.0%}",
            )

    if not FILL_ISOCLINES:
        plot.legend.location = "top_left"
        plot.legend.click_policy = "hide"

    frame_div = Div(text=first_frame["frame_label"])
    bandwidth_div = Div(text=first_frame["bandwidth_text"])
    x_start = x_range.start
    x_end = x_range.end
    y_start = y_range.start
    y_end = y_range.end
    if None in (x_start, x_end, y_start, y_end):
        label_x = 0.0
        label_y = 0.0
    else:
        x_pad = float(x_end - x_start) * 0.02
        label_x = float(x_start) + x_pad
        label_transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )
        _, label_y = label_transformer.transform(0.0, YEAR_LABEL_LAT)

    year_label = Label(
        x=label_x,
        y=label_y,
        x_units="data",
        y_units="data",
        text=first_frame["year_text"],
        text_font_size="32pt",
        text_font_style="bold",
        text_color="#111111",
        text_align="left",
        text_baseline="top",
        background_fill_color="white",
        background_fill_alpha=0.7,
    )
    plot.add_layout(year_label)
    slider = Slider(
        start=0, end=len(frames) - 1, value=0, step=1, title="Frame"
    )

    slider_callback = CustomJS(
        args=dict(
            frames=frames,
            point_source=point_source,
            iso_sources=iso_sources,
            render_order=render_order,
            plot=plot,
            frame_div=frame_div,
            bandwidth_div=bandwidth_div,
            year_label=year_label,
        ),
        code="""
            const frame = frames[cb_obj.value];
            point_source.data = frame.points;
            point_source.change.emit();
            for (let i = 0; i < iso_sources.length; i++) {
                const source_idx = render_order[i];
                iso_sources[i].data = frame.isopleths[source_idx];
                iso_sources[i].change.emit();
            }
            plot.title.text = frame.title;
            frame_div.text = frame.frame_label;
            bandwidth_div.text = frame.bandwidth_text;
            year_label.text = frame.year_text;
        """,
    )
    slider.js_on_change("value", slider_callback)

    play_button = Button(label="Play", button_type="success", width=60)
    play_callback = CustomJS(
        args=dict(
            slider=slider,
            play_button=play_button,
            interval_ms=ANIMATION_PLAY_INTERVAL_MS,
        ),
        code="""
            if (play_button.label === "Play") {
                play_button.label = "Pause";
                play_button._interval = setInterval(function() {
                    let next = slider.value + 1;
                    if (next > slider.end) {
                        next = slider.start;
                    }
                    slider.value = next;
                }, interval_ms);
            } else {
                play_button.label = "Play";
                if (play_button._interval != null) {
                    clearInterval(play_button._interval);
                    play_button._interval = null;
                }
            }
        """,
    )
    play_button.js_on_click(play_callback)

    layout = column(
        plot,
        row(play_button, slider),
        row(frame_div, bandwidth_div),
    )
    output_file(output_html, title="Ingroup KDE Animation")
    show(layout)

    print(f"[INFO] Animated output written to {output_html}.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize an ingroup's geographic distribution with KDE isoclines."
        )
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help=(
            "Render an animated KDE with 12 monthly frames per year, "
            "each frame covering a 1-year window."
        ),
    )
    return parser


def run_static_map() -> None:
    """Render a static KDE map."""
    dated = prepare_base_dataframe()
    filtered = filter_top_logit_per_recordist(dated)
    points_df, lon, lat = prepare_points(filtered)

    if len(lon) < MIN_POINTS_FOR_KDE:
        print("[WARN] Not enough points for KDE; showing points only.")
        bandwidth = float("nan")
        isopleths: list[Isopleth] = []
    else:
        equal_area = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
        x_eq, y_eq = equal_area.transform(lon, lat)
        points_eq = np.column_stack((x_eq, y_eq))
        grid, values, bandwidth = compute_kde(points_eq)
        isopleths = build_isopleths(grid, values, NUM_ISOCLINES)

    plot_map(points_df, lon, lat, isopleths, bandwidth)


def run_animated_map() -> None:
    """Render an animated KDE map with month-by-month frames."""
    dated = prepare_base_dataframe()
    dated = ensure_date_column(dated)
    valid_mask = dated["date_dt"].notna()
    if valid_mask.sum() == 0:
        raise SystemExit("Animation requires valid dates in the metadata.")
    if not valid_mask.all():
        dropped = int((~valid_mask).sum())
        print(f"[WARN] Dropping {dropped} rows without valid dates.")
        dated = dated.loc[valid_mask].copy()

    windows = build_monthly_windows(
        dated["date_dt"], ANIMATION_WINDOW_MONTHS, ANIMATION_STEP_MONTHS
    )
    if not windows:
        raise SystemExit("No animation windows could be built from the data.")

    probabilities = np.linspace(HDR_MIN_PROB, HDR_MAX_PROB, NUM_ISOCLINES).tolist()
    point_transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )
    equal_area = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    contour_transformer = Transformer.from_crs(
        "EPSG:3035", "EPSG:3857", always_xy=True
    )

    _, all_lon, all_lat = prepare_points(dated)
    if len(all_lon) == 0:
        raise SystemExit("No valid coordinates available for animation.")
    all_x, all_y = project_points(all_lon, all_lat, point_transformer)
    x_range, y_range = compute_bounds(all_x, all_y, [])

    print(
        f"[INFO] Building {len(windows)} frames "
        f"({ANIMATION_WINDOW_MONTHS} months per frame, "
        f"step={ANIMATION_STEP_MONTHS} month)."
    )

    frames = build_animation_frames(
        dated,
        windows,
        probabilities,
        point_transformer,
        equal_area,
        contour_transformer,
    )
    plot_animated_map(
        frames, probabilities, x_range, y_range, ANIMATION_OUTPUT_HTML
    )


def main() -> None:
    """Parse CLI arguments and render the requested output."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.animate:
        run_animated_map()
    else:
        run_static_map()


if __name__ == "__main__":
    main()
