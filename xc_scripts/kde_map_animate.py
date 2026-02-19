"""
Visualize an ingroup's geographic distribution with KDE isoclines on a map.

Edit the paths and settings below, then run:
    python xc_scripts/kde_map_animate.py
    python xc_scripts/kde_map_animate.py --animate
    python xc_scripts/kde_map_animate.py --interactive

Interactive mode starts a local Bokeh server with a KDE scale input and
"Recalculate KDE" button for the static map.
"""

from __future__ import annotations

import argparse
import base64
import html
import math
import re
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from bokeh import palettes as bokeh_palettes
from bokeh.io import output_file, show
from bokeh.events import DocumentReady
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    Label,
    LayoutDOM,
    Range1d,
    RangeSlider,
    Slider,
    Spinner,
)
from bokeh.plotting import figure
from bokeh.server.server import Server
from KDEpy import FFTKDE
from pyproj import Transformer
from skimage import measure

try:
    from xyzservices import providers as xyz_providers
except Exception:  # noqa: BLE001
    xyz_providers = None

try:
    from bokeh.sampledata.world_cities import data as world_cities_data
except Exception:  # noqa: BLE001
    world_cities_data = None

try:
    from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter
except Exception:  # noqa: BLE001
    binary_closing = None
    binary_fill_holes = None
    gaussian_filter = None

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
METADATA_CSV = Path("/Volumes/Z Slim/zslim_birdcluster/embeddings/emberiza_calandra/metadata.csv")
INFERENCE_CSV = Path("/Volumes/Z Slim/zslim_birdcluster/embeddings/emberiza_calandra/inference.csv")
OUTPUT_HTML = Path("ingroup_kde_map.html")
ANIMATION_OUTPUT_HTML = Path("ingroup_kde_map_animated.html")

FILTER_ONE_PER_RECORDIST = True

NUM_ISOCLINES = 5
FILL_ISOCLINES = True
HDR_MIN_PROB = 0.1
HDR_MAX_PROB = 0.7
ISOCLINE_PROB_ROUND_STEP = 0.05 # Set <= 0 to disable interior rounding.

DATE_FILTER_MODE = (
    "range"  # "all", "recent", "exclude_recent", or "range"
)
DATE_FILTER_YEARS = 3
DATE_RANGE_START = "2014-01-01"  # Used when mode="range".
DATE_RANGE_END = "2026-11-12"  # Used when mode="range".
# Examples: "2014", "2014-01", "2014-01-01"
DATE_PARSE_FORMAT = "%m/%d/%Y"
DATE_PARSE_DAYFIRST = False
DATE_PARSE_FALLBACK = True

KDE_GRID_SIZE = 256
BANDWIDTH_METHOD = "knn"  # "cv", "scott", "knn", or "manual" (after scaling)
BANDWIDTH_MANUAL = None  # Example: 0.2
BANDWIDTH_KNN_K = 5  # Kth neighbor distance used for "knn" bandwidth.
BANDWIDTH_KNN_QUANTILE = 0.5  # Quantile of kth distances (0-1).
BANDWIDTH_KNN_SCALE = 0.8  # Multiplier for the "knn" bandwidth.
BANDWIDTH_GRID_SIZE = 20
BANDWIDTH_CV_FOLDS = 5
BANDWIDTH_SEARCH_LOG_MIN = -1.0
BANDWIDTH_SEARCH_LOG_MAX = 1.0
BANDWIDTH_SAMPLE_SIZE = 5000
RANDOM_SEED = 7

AUTO_ZOOM = False
EUROPE_BOUNDS = {
    "lon_min": -20.0,
    "lon_max": 60.0,
    "lat_min": 28.0,
    "lat_max": 72.0,
}

PLOT_WIDTH = 2400
PLOT_HEIGHT = 2400
PLAYLIST_WIDTH = 860
POINT_SIZE = 15
POINT_ALPHA = 0.7
POINT_LINE_WIDTH = 2.5
ISOCLINE_FILL_ALPHA = 0.4
ISOCLINE_FILL_MODE = "bands"  # "bands" (non-overlapping) or "stacked"
ISOCLINE_STROKE_WIDTH = 1  # Set to 0 to remove isocline outlines.
ISOCLINE_STROKE_COLOR = "#000000"
ISOCLINE_STROKE_ALPHA = 1
ISOCLINE_LINE_WIDTH = 10
ISOCLINE_LINE_ALPHA = 0.6
ISOCLINE_COLOR_MODE = "colormap"  # "anchors" or "colormap"
ISOCLINE_COLOR_ANCHORS = ["#0b1d8b", "#00a878", "#f4d03f", "#e53935"]
ISOCLINE_COLORMAP_NAME = "viridis"  # Example: "viridis", "magma", "Viridis256"
ISOCLINE_COLORMAP_REVERSE = True  # True maps first isocline to colormap max.
ISOCLINE_STROKE_COLOR_MODE = "colormap"  # "single", "match_fill", "anchors", or "colormap"
ISOCLINE_STROKE_COLOR_ANCHORS = ["#0b1d8b", "#00a878", "#f4d03f", "#e53935"]
ISOCLINE_STROKE_COLORMAP_NAME = "viridis"
ISOCLINE_STROKE_COLORMAP_REVERSE = True
WATER_BACKGROUND_COLOR = "#dcecf7"
LAND_FILL_COLOR = "#F9EC9C"
LAND_FILL_ALPHA = 1.0
MAP_BASE_PROVIDER = "Esri.WorldPhysical"
MAP_BASE_RETINA = False
MAP_BASE_ALPHA = 1.0
MAP_TITLE_FONT_SIZE = "20pt"
MAP_TITLE_FONT_STYLE = "italic"
MAP_TITLE_TEXT = "Emberiza calandra"
MAP_AXIS_MAJOR_LABEL_FONT_SIZE = "28pt"
MAP_AXIS_LABEL_FONT_SIZE = "32pt"
MAP_LEGEND_LOCATION = "top_right"
MAP_LEGEND_FONT_SIZE = "28pt"
MAP_LEGEND_GLYPH_WIDTH = 56
MAP_LEGEND_GLYPH_HEIGHT = 30
MAP_LEGEND_LABEL_STANDOFF = 12
MAP_LEGEND_SPACING = 10
LAND_MASK_LON_BINS = 720
LAND_MASK_LAT_BINS = 360
LAND_MASK_SMOOTH_SIGMA = 1.6
LAND_MASK_THRESHOLD_RATIO = 0.006
LAND_MASK_CLOSING_ITERATIONS = 2
LAND_MASK_MIN_POINTS = 90
LAND_MASK_MIN_AREA_DEG2 = 80.0
MAP_TILE_FALLBACK_PROVIDER = "CartoDB.PositronNoLabels"
MAP_TILE_FALLBACK_RETINA = True
MAP_TILE_FALLBACK_ALPHA = 1.0
MAX_ABS_LAT = 85.0
MIN_POINTS_FOR_KDE = 5
YEAR_LABEL_LAT = 70.0

MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
HISTOGRAM_HEIGHT = 240
HISTOGRAM_BAR_COLOR = "#3b6ea5"
HISTOGRAM_ALPHA = 0.85
HISTOGRAM_Y_PADDING = 1.1
YEARLY_HEIGHT = 190

ANIMATION_WINDOW_MONTHS = 12  # 1-year window per frame.
ANIMATION_STEP_MONTHS = 1  # Advance one month per frame.
ANIMATION_PLAY_INTERVAL_MS = 100  # Playback interval for the HTML animation.

# Media playlist configuration (interactive mode only)
SPECTROGRAM_IMAGE_FORMAT = "png"
INLINE_SPECTROGRAMS = False
SPECTROGRAM_BASE_URL: str | None = None
SPECTROGRAM_HOST = "127.0.0.1"
SPECTROGRAM_PORT = 8766
SPECTROGRAM_AUTO_SERVE = True
SPECTROGRAMS_DIR: Path | None = None

AUDIO_BASE_URL: str | None = None
AUDIO_HOST = "127.0.0.1"
AUDIO_PORT = 8765
AUDIO_AUTO_SERVE = True
AUDIO_DIR: Path | None = None

PLAYLIST_MAX_ITEMS: int | None = None  # Set to cap playlist size.


@dataclass(frozen=True)
class Isopleth:
    """Container for a single KDE isopleth."""

    probability: float
    level: float
    paths: list[np.ndarray]


_STATIC_HTTP_SERVERS: dict[str, dict[str, Any]] = {}
_LAND_PATCH_CACHE: tuple[list[list[float]], list[list[float]]] | None = None
_LAND_PATCH_WARNING_SHOWN = False


def detect_species_slug(path: Path) -> str:
    """Infer the species slug from a CSV path."""
    if path and path.parent.name:
        return path.parent.name
    return "unknown_species"


def format_species_name(species_slug: str) -> str:
    """Format slug-like species names for display."""
    cleaned = re.sub(r"[^a-zA-Z0-9_ -]", " ", species_slug or "").strip()
    tokens = [token for token in re.split(r"[_\s-]+", cleaned) if token]
    if not tokens:
        return "Unknown species"
    genus = tokens[0].capitalize()
    epithet = [token.lower() for token in tokens[1:]]
    return " ".join([genus, *epithet]).strip()


def build_map_title_text(species_slug: str) -> str:
    """Build a consistent species-aware map title."""
    configured = str(MAP_TITLE_TEXT or "").strip()
    if configured:
        return configured
    return format_species_name(species_slug)


def detect_root_path(path: Path) -> Path:
    """Infer the dataset root by walking up to an 'embeddings' directory."""
    for parent in path.parents:
        if parent.name == "embeddings":
            return parent.parent
    return path.parent


def start_static_file_server(
    *,
    label: str,
    directory: Path,
    host: str,
    port: int,
    log_requests: bool = False,
) -> str | None:
    """Start or reuse a background HTTP server to expose local files."""
    existing = _STATIC_HTTP_SERVERS.get(label)
    if existing:
        return existing["base_url"]

    if not directory.exists():
        print(f"[WARN] {label} directory missing, cannot auto-serve: {directory}")
        return None

    class _StaticHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:
            if log_requests:
                super().log_message(format, *args)

    server = ThreadingHTTPServer((host, port), _StaticHandler)
    base_url = f"http://{host}:{port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"[INFO] {label} server started at {base_url} serving {directory}")
    _STATIC_HTTP_SERVERS[label] = {
        "server": server,
        "thread": thread,
        "host": host,
        "port": port,
        "directory": directory,
        "base_url": base_url,
    }
    return base_url

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


def parse_date_range_endpoint(
    value: object, *, is_end: bool
) -> tuple[pd.Timestamp | None, bool]:
    """Parse a range endpoint and flag whether a time component was provided."""
    if value is None:
        return None, False
    try:
        if pd.isna(value):
            return None, False
    except TypeError:
        pass

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return None, False

    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        if is_end:
            return pd.Timestamp(year=year, month=12, day=31), False
        return pd.Timestamp(year=year, month=1, day=1), False

    if re.fullmatch(r"\d{4}-\d{2}", text):
        parsed = pd.to_datetime(f"{text}-01", errors="coerce")
        if pd.isna(parsed):
            return None, False
        if is_end:
            parsed = parsed + pd.offsets.MonthEnd(0)
        return pd.Timestamp(parsed), False

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=DATE_PARSE_DAYFIRST)
    if pd.isna(parsed):
        return None, False

    has_time = bool(re.search(r"\d{2}:\d{2}", text)) or "T" in text
    return pd.Timestamp(parsed), has_time


def format_timestamp(value: pd.Timestamp) -> str:
    """Format timestamps with time if present."""
    if value is pd.NaT:
        return "NaT"
    if value.hour or value.minute or value.second or value.microsecond:
        return value.isoformat(sep=" ")
    return str(value.date())


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
    """Filter rows based on DATE_FILTER_MODE settings."""
    mode = DATE_FILTER_MODE.strip().lower()
    if mode == "all":
        return df
    if mode not in {"recent", "exclude_recent", "range"}:
        raise SystemExit(
            "DATE_FILTER_MODE must be 'all', 'recent', 'exclude_recent', or 'range'."
        )

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

    if mode in {"recent", "exclude_recent"}:
        if DATE_FILTER_YEARS <= 0:
            raise SystemExit("DATE_FILTER_YEARS must be > 0.")
        reference_date = date_series.max()
        cutoff = reference_date - pd.DateOffset(years=DATE_FILTER_YEARS)

        if mode == "recent":
            mask = date_series >= cutoff
        else:
            mask = date_series < cutoff
    else:
        start, start_has_time = parse_date_range_endpoint(
            DATE_RANGE_START, is_end=False
        )
        end, end_has_time = parse_date_range_endpoint(
            DATE_RANGE_END, is_end=True
        )
        if start is None and end is None:
            raise SystemExit(
                "DATE_RANGE_START or DATE_RANGE_END must be set when "
                "DATE_FILTER_MODE='range'."
            )
        if start is not None and end is not None and end < start:
            raise SystemExit("DATE_RANGE_END must be >= DATE_RANGE_START.")

        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= date_series >= start
        if end is not None:
            if end_has_time:
                mask &= date_series <= end
            else:
                mask &= date_series < (end + pd.Timedelta(days=1))

    filtered = df.loc[mask].copy()
    dropped_invalid = date_series.isna().sum()
    if dropped_invalid:
        print(
            f"[WARN] Dropped {dropped_invalid} rows without valid dates for filtering."
        )
    if mode in {"recent", "exclude_recent"}:
        print(
            f"[INFO] Date filter '{mode}' kept {len(filtered)} of {len(df)} rows "
            f"(cutoff={cutoff.date()}, reference={reference_date.date()})."
        )
    else:
        range_bits: list[str] = []
        if start is not None:
            label = (
                format_timestamp(start) if start_has_time else str(start.date())
            )
            range_bits.append(f"start={label}")
        if end is not None:
            label = format_timestamp(end) if end_has_time else str(end.date())
            range_bits.append(f"end={label}")
        range_text = ", ".join(range_bits) if range_bits else "no range"
        print(
            f"[INFO] Date filter 'range' kept {len(filtered)} of {len(df)} rows "
            f"({range_text})."
        )
        filtered_dates = date_series[mask].dropna()
        if filtered_dates.empty:
            print("[WARN] Date filter 'range' left no valid dates.")
        else:
            earliest = filtered_dates.min()
            latest = filtered_dates.max()
            print(
                f"[INFO] Date span in range: {format_timestamp(earliest)} "
                f"to {format_timestamp(latest)}."
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


def select_bandwidth(
    points: np.ndarray, *, knn_scale_override: float | None = None
) -> float:
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
        scale_multiplier = (
            BANDWIDTH_KNN_SCALE
            if knn_scale_override is None
            else float(knn_scale_override)
        )
        scale *= scale_multiplier
        if not np.isfinite(scale) or scale <= 0:
            print("[WARN] KNN bandwidth invalid; falling back to Scott's rule.")
            return estimate_scott_bandwidth(points)
        return float(scale)
    if method == "cv":
        return select_optimal_bandwidth(points)
    raise SystemExit(f"Unknown BANDWIDTH_METHOD: {BANDWIDTH_METHOD}")


def compute_kde(
    points: np.ndarray, *, knn_scale_override: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute KDE grid values for projected points."""
    standardized, center, scale = standardize_points(points)
    bandwidth = select_bandwidth(
        standardized, knn_scale_override=knn_scale_override
    )
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
    probabilities = get_isocline_probabilities(num_levels)
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


def interpolate_hex_palette(colors: Sequence[str], num_colors: int) -> list[str]:
    """Interpolate a list of hex colors to the requested palette length."""
    if num_colors <= 0:
        return []
    if not colors:
        return ["#000000"] * num_colors
    if len(colors) == 1:
        return [colors[0]] * num_colors
    if num_colors == 1:
        return [colors[-1]]

    anchor_rgb = np.array([hex_to_rgb(color) for color in colors], dtype=float)
    anchor_pos = np.linspace(0, 1, len(colors))
    positions = np.linspace(0, 1, num_colors)

    palette: list[str] = []
    for pos in positions:
        idx = int(np.searchsorted(anchor_pos, pos)) - 1
        idx = max(0, min(idx, len(colors) - 2))
        frac = (pos - anchor_pos[idx]) / (anchor_pos[idx + 1] - anchor_pos[idx])
        rgb = anchor_rgb[idx] + frac * (anchor_rgb[idx + 1] - anchor_rgb[idx])
        palette.append(rgb_to_hex(rgb))
    return palette


def resolve_colormap_colors(colormap_name: str) -> list[str] | None:
    """Resolve a named Bokeh colormap into a list of hex colors."""
    requested = re.sub(r"\s+", "", str(colormap_name or "")).strip()
    if not requested:
        return None

    exact_size_match = re.fullmatch(r"([a-zA-Z_]+)(\d+)", requested)
    if exact_size_match:
        family_name = exact_size_match.group(1)
        size = int(exact_size_match.group(2))
        for known_family, sizes in bokeh_palettes.all_palettes.items():
            if known_family.lower() == family_name.lower() and size in sizes:
                return list(sizes[size])

    for known_family, sizes in bokeh_palettes.all_palettes.items():
        if known_family.lower() == requested.lower():
            return list(sizes[max(sizes.keys())])

    for attr in dir(bokeh_palettes):
        if attr.lower() != requested.lower():
            continue
        value = getattr(bokeh_palettes, attr)
        if isinstance(value, (list, tuple)):
            return [str(color) for color in value]

    return None


def build_heat_palette(num_colors: int) -> list[str]:
    """Create an isocline palette from configured anchors or a named colormap."""
    default_anchors = ["#0b1d8b", "#00a878", "#f4d03f", "#e53935"]
    mode = str(ISOCLINE_COLOR_MODE).strip().lower()

    if mode == "colormap":
        colors = resolve_colormap_colors(ISOCLINE_COLORMAP_NAME)
        if colors is None:
            print(
                "[WARN] Unknown ISOCLINE_COLORMAP_NAME; falling back to "
                "ISOCLINE_COLOR_ANCHORS."
            )
            anchors = [
                str(color).strip()
                for color in ISOCLINE_COLOR_ANCHORS
                if str(color).strip()
            ]
            if len(anchors) < 2:
                anchors = default_anchors
            return interpolate_hex_palette(anchors, num_colors)

        if ISOCLINE_COLORMAP_REVERSE:
            colors = list(reversed(colors))
        return interpolate_hex_palette(colors, num_colors)

    if mode not in {"anchors", "colormap"}:
        print("[WARN] ISOCLINE_COLOR_MODE must be 'anchors' or 'colormap'; using anchors.")

    anchors = [
        str(color).strip() for color in ISOCLINE_COLOR_ANCHORS if str(color).strip()
    ]
    if len(anchors) < 2:
        print("[WARN] ISOCLINE_COLOR_ANCHORS needs >=2 colors; using defaults.")
        anchors = default_anchors
    return interpolate_hex_palette(anchors, num_colors)


def build_isocline_stroke_palette(
    num_colors: int, fill_palette: Sequence[str] | None = None
) -> list[str]:
    """Create a stroke palette for isocline outlines."""
    if num_colors <= 0:
        return []
    mode = str(ISOCLINE_STROKE_COLOR_MODE).strip().lower()

    if mode == "single":
        return [ISOCLINE_STROKE_COLOR] * num_colors
    if mode == "match_fill":
        if fill_palette is not None and len(fill_palette) == num_colors:
            return [str(color) for color in fill_palette]
        return build_heat_palette(num_colors)
    if mode == "anchors":
        anchors = [
            str(color).strip()
            for color in ISOCLINE_STROKE_COLOR_ANCHORS
            if str(color).strip()
        ]
        if len(anchors) < 2:
            print(
                "[WARN] ISOCLINE_STROKE_COLOR_ANCHORS needs >=2 colors; using "
                "ISOCLINE_STROKE_COLOR."
            )
            return [ISOCLINE_STROKE_COLOR] * num_colors
        return interpolate_hex_palette(anchors, num_colors)
    if mode == "colormap":
        colors = resolve_colormap_colors(ISOCLINE_STROKE_COLORMAP_NAME)
        if colors is None:
            print(
                "[WARN] Unknown ISOCLINE_STROKE_COLORMAP_NAME; using "
                "ISOCLINE_STROKE_COLOR."
            )
            return [ISOCLINE_STROKE_COLOR] * num_colors
        if ISOCLINE_STROKE_COLORMAP_REVERSE:
            colors = list(reversed(colors))
        return interpolate_hex_palette(colors, num_colors)

    print(
        "[WARN] ISOCLINE_STROKE_COLOR_MODE must be 'single', 'match_fill', "
        "'anchors', or 'colormap'; using ISOCLINE_STROKE_COLOR."
    )
    return [ISOCLINE_STROKE_COLOR] * num_colors


def use_band_fill_mode() -> bool:
    """Return whether filled isoclines should be rendered as non-overlapping bands."""
    mode = str(ISOCLINE_FILL_MODE).strip().lower()
    if mode == "bands":
        return True
    if mode == "stacked":
        return False
    print("[WARN] ISOCLINE_FILL_MODE must be 'bands' or 'stacked'; using 'bands'.")
    return True


def polygon_area_xy(path: np.ndarray) -> float:
    """Return absolute polygon area in projected coordinates."""
    if len(path) < 3:
        return 0.0
    x_vals = path[:, 0]
    y_vals = path[:, 1]
    return 0.5 * float(
        np.abs(
            np.dot(x_vals, np.roll(y_vals, -1))
            - np.dot(y_vals, np.roll(x_vals, -1))
        )
    )


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test."""
    x_coord = float(point[0])
    y_coord = float(point[1])
    inside = False
    for idx in range(len(polygon) - 1):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[idx + 1]
        if (y1 > y_coord) == (y2 > y_coord):
            continue
        denom = y2 - y1
        if abs(denom) < 1e-12:
            continue
        x_intersection = (x2 - x1) * (y_coord - y1) / denom + x1
        if x_coord < x_intersection:
            inside = not inside
    return inside


def find_interior_seed_point(path: np.ndarray) -> np.ndarray:
    """Return a point that is likely inside a closed polygon path."""
    if len(path) < 4:
        return np.array([np.nan, np.nan], dtype=float)

    candidate = np.mean(path[:-1], axis=0)
    if point_in_polygon(candidate, path):
        return candidate

    min_x = float(np.min(path[:, 0]))
    max_x = float(np.max(path[:, 0]))
    min_y = float(np.min(path[:, 1]))
    max_y = float(np.max(path[:, 1]))

    for grid_size in (7, 11, 15):
        xs = np.linspace(min_x, max_x, grid_size)
        ys = np.linspace(min_y, max_y, grid_size)
        for x_coord in xs:
            for y_coord in ys:
                probe = np.array([x_coord, y_coord], dtype=float)
                if point_in_polygon(probe, path):
                    return probe

    return candidate


def build_region_polygons(
    paths: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Group contour rings into shell+hole polygons using containment depth."""
    valid_paths = [
        path for path in paths if len(path) >= 4 and polygon_area_xy(path) > 0
    ]
    if not valid_paths:
        return []

    areas = [polygon_area_xy(path) for path in valid_paths]
    seeds = [find_interior_seed_point(path) for path in valid_paths]
    parents: list[int | None] = [None] * len(valid_paths)

    for idx, seed in enumerate(seeds):
        best_parent_idx: int | None = None
        best_parent_area = float("inf")
        for candidate_idx, candidate_path in enumerate(valid_paths):
            if candidate_idx == idx:
                continue
            candidate_area = areas[candidate_idx]
            if candidate_area <= areas[idx]:
                continue
            if not point_in_polygon(seed, candidate_path):
                continue
            if candidate_area < best_parent_area:
                best_parent_idx = candidate_idx
                best_parent_area = candidate_area
        parents[idx] = best_parent_idx

    depth_cache: dict[int, int] = {}

    def depth_for(index: int) -> int:
        cached = depth_cache.get(index)
        if cached is not None:
            return cached
        parent_idx = parents[index]
        depth = 0 if parent_idx is None else depth_for(parent_idx) + 1
        depth_cache[index] = depth
        return depth

    depths = [depth_for(idx) for idx in range(len(valid_paths))]
    holes_by_shell: dict[int, list[np.ndarray]] = {
        idx: [] for idx, depth in enumerate(depths) if depth % 2 == 0
    }

    for idx, depth in enumerate(depths):
        if depth % 2 == 0:
            continue
        shell_idx = parents[idx]
        while shell_idx is not None and depths[shell_idx] % 2 == 1:
            shell_idx = parents[shell_idx]
        if shell_idx is not None:
            holes_by_shell.setdefault(shell_idx, []).append(valid_paths[idx])

    shell_indices = sorted(
        (idx for idx, depth in enumerate(depths) if depth % 2 == 0),
        key=lambda idx: areas[idx],
        reverse=True,
    )
    return [
        (valid_paths[shell_idx], holes_by_shell.get(shell_idx, []))
        for shell_idx in shell_indices
    ]


def source_data_to_paths(source_data: dict[str, list[list[float]]]) -> list[np.ndarray]:
    """Convert isopleth source dictionaries into closed numpy paths."""
    xs_list = source_data.get("xs", [])
    ys_list = source_data.get("ys", [])
    paths: list[np.ndarray] = []
    for xs, ys in zip(xs_list, ys_list):
        if len(xs) < 3 or len(xs) != len(ys):
            continue
        path = np.column_stack(
            (
                np.asarray(xs, dtype=float),
                np.asarray(ys, dtype=float),
            )
        )
        if not np.allclose(path[0], path[-1]):
            path = np.vstack([path, path[0]])
        paths.append(path)
    return paths


def build_band_multipolygon_source_data(
    outer_paths: Sequence[np.ndarray], inner_paths: Sequence[np.ndarray]
) -> dict[str, list]:
    """Build MultiPolygons source data for one isocline band."""
    outer_polygons = build_region_polygons(outer_paths)
    if not outer_polygons:
        return {"xs": [], "ys": []}

    outer_shells = [shell for shell, _ in outer_polygons]
    assigned_holes = [list(holes) for _, holes in outer_polygons]
    outer_areas = [polygon_area_xy(path) for path in outer_shells]
    inner_shells = [shell for shell, _ in build_region_polygons(inner_paths)]

    for inner_path in inner_shells:
        inner_seed = find_interior_seed_point(inner_path)
        container_candidates: list[tuple[float, int]] = []
        for outer_idx, outer_path in enumerate(outer_shells):
            if not point_in_polygon(inner_seed, outer_path):
                continue
            if any(
                point_in_polygon(inner_seed, hole_path)
                for hole_path in assigned_holes[outer_idx]
            ):
                continue
            container_candidates.append((outer_areas[outer_idx], outer_idx))
        if not container_candidates:
            continue
        _, best_outer_idx = min(container_candidates)
        assigned_holes[best_outer_idx].append(inner_path)

    multi_xs: list[list[list[list[float]]]] = []
    multi_ys: list[list[list[list[float]]]] = []
    for outer_path, holes in zip(outer_shells, assigned_holes):
        polygon_x_rings = [outer_path[:, 0].tolist()]
        polygon_y_rings = [outer_path[:, 1].tolist()]
        for hole_path in holes:
            polygon_x_rings.append(hole_path[:, 0].tolist())
            polygon_y_rings.append(hole_path[:, 1].tolist())
        multi_xs.append([polygon_x_rings])
        multi_ys.append([polygon_y_rings])

    return {"xs": multi_xs, "ys": multi_ys}


def build_band_sources_from_ordered_paths(
    ordered_paths: Sequence[Sequence[np.ndarray]],
) -> list[dict[str, list]]:
    """Build non-overlapping band source data from outer-to-inner paths."""
    if not ordered_paths:
        return []
    bands: list[dict[str, list]] = []
    for idx, outer_paths in enumerate(ordered_paths):
        inner_paths = ordered_paths[idx + 1] if idx + 1 < len(ordered_paths) else []
        bands.append(build_band_multipolygon_source_data(outer_paths, inner_paths))
    return bands


def build_band_sources_from_level_sources(
    level_sources: Sequence[dict[str, list[list[float]]]],
    render_order: Sequence[int],
) -> list[dict[str, list]]:
    """Build non-overlapping band source data from aligned level sources."""
    ordered_paths = [source_data_to_paths(level_sources[idx]) for idx in render_order]
    return build_band_sources_from_ordered_paths(ordered_paths)


def get_isocline_probabilities(num_levels: int) -> np.ndarray:
    """Return the HDR probabilities used to generate isoclines."""
    if num_levels <= 0:
        return np.array([], dtype=float)
    probabilities = np.linspace(HDR_MIN_PROB, HDR_MAX_PROB, num_levels, dtype=float)
    round_step = float(ISOCLINE_PROB_ROUND_STEP)
    if round_step <= 0 or num_levels <= 2:
        return probabilities

    interior = np.round(probabilities[1:-1] / round_step) * round_step
    if probabilities[-1] >= probabilities[0]:
        interior = np.clip(interior, probabilities[0], probabilities[-1])
        interior = np.maximum.accumulate(interior)
    else:
        interior = np.clip(interior, probabilities[-1], probabilities[0])
        interior = np.minimum.accumulate(interior)

    probabilities[1:-1] = interior
    return probabilities


def log_isocline_percentages(num_levels: int) -> None:
    """Print exact percentages for the configured isocline levels."""
    probabilities = get_isocline_probabilities(num_levels)
    if probabilities.size == 0:
        print("[INFO] No isoclines configured (NUM_ISOCLINES <= 0).")
        return
    percent_text = ", ".join(f"{probability * 100:.2f}%" for probability in probabilities)
    print(f"[INFO] Isoclines represent HDR levels: {percent_text}")


def polygon_area_degrees(lon: np.ndarray, lat: np.ndarray) -> float:
    """Approximate polygon area in lon/lat degree space."""
    if len(lon) < 3 or len(lat) < 3:
        return 0.0
    return 0.5 * float(
        np.abs(np.dot(lon, np.roll(lat, -1)) - np.dot(lat, np.roll(lon, -1)))
    )


def build_land_patches_mercator() -> tuple[list[list[float]], list[list[float]]]:
    """Create simplified global land patches from city density."""
    if (
        world_cities_data is None
        or gaussian_filter is None
        or binary_closing is None
        or binary_fill_holes is None
    ):
        return [], []

    lon = pd.to_numeric(world_cities_data["lng"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(world_cities_data["lat"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= MAX_ABS_LAT)
    if not valid.any():
        return [], []
    lon = lon[valid]
    lat = lat[valid]

    lon_edges = np.linspace(-180.0, 180.0, LAND_MASK_LON_BINS + 1)
    lat_edges = np.linspace(-MAX_ABS_LAT, MAX_ABS_LAT, LAND_MASK_LAT_BINS + 1)
    density, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])
    smoothed = gaussian_filter(
        density,
        sigma=LAND_MASK_SMOOTH_SIGMA,
        mode=("nearest", "wrap"),
    )
    max_density = float(smoothed.max())
    if max_density <= 0:
        return [], []

    threshold = max_density * LAND_MASK_THRESHOLD_RATIO
    land_mask = smoothed >= threshold
    land_mask = binary_closing(land_mask, iterations=LAND_MASK_CLOSING_ITERATIONS)
    land_mask = binary_fill_holes(land_mask)

    contours = measure.find_contours(land_mask.astype(float), level=0.5)
    if not contours:
        return [], []

    lon_axis = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    lat_axis = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    row_axis = np.arange(len(lat_axis))
    col_axis = np.arange(len(lon_axis))
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    xs_list: list[list[float]] = []
    ys_list: list[list[float]] = []
    for contour in contours:
        if len(contour) < LAND_MASK_MIN_POINTS:
            continue
        rows = contour[:, 0]
        cols = contour[:, 1]
        path_lat = np.interp(rows, row_axis, lat_axis)
        path_lon = np.interp(cols, col_axis, lon_axis)
        if polygon_area_degrees(path_lon, path_lat) < LAND_MASK_MIN_AREA_DEG2:
            continue
        if not np.allclose(path_lon[0], path_lon[-1]) or not np.allclose(
            path_lat[0], path_lat[-1]
        ):
            path_lon = np.append(path_lon, path_lon[0])
            path_lat = np.append(path_lat, path_lat[0])
        path_x, path_y = transformer.transform(path_lon, path_lat)
        xs_list.append(path_x.astype(float).tolist())
        ys_list.append(path_y.astype(float).tolist())

    return xs_list, ys_list


def get_land_patches_mercator() -> tuple[list[list[float]], list[list[float]]]:
    """Return cached land patches in Web Mercator coordinates."""
    global _LAND_PATCH_CACHE
    if _LAND_PATCH_CACHE is None:
        _LAND_PATCH_CACHE = build_land_patches_mercator()
    return _LAND_PATCH_CACHE


def resolve_tile_provider(path: str) -> Any | None:
    """Resolve a dotted xyzservices provider path."""
    if xyz_providers is None or not path:
        return None
    provider: Any = xyz_providers
    for part in path.split("."):
        provider = getattr(provider, part, None)
        if provider is None:
            return None
    return provider


def style_map_figure(plot: Any) -> None:
    """Apply shared background styling and basemap."""
    global _LAND_PATCH_WARNING_SHOWN
    plot.grid.visible = False
    plot.background_fill_color = WATER_BACKGROUND_COLOR
    plot.border_fill_color = WATER_BACKGROUND_COLOR
    plot.axis.major_label_text_font_size = MAP_AXIS_MAJOR_LABEL_FONT_SIZE
    plot.axis.axis_label_text_font_size = MAP_AXIS_LABEL_FONT_SIZE
    plot.title.text_font_size = MAP_TITLE_FONT_SIZE
    plot.title.text_font_style = MAP_TITLE_FONT_STYLE

    provider = resolve_tile_provider(MAP_BASE_PROVIDER)
    if provider is not None:
        tile_renderer = plot.add_tile(provider, retina=MAP_BASE_RETINA)
        tile_renderer.alpha = MAP_BASE_ALPHA
        return

    land_xs, land_ys = get_land_patches_mercator()
    if land_xs:
        plot.patches(
            land_xs,
            land_ys,
            fill_color=LAND_FILL_COLOR,
            fill_alpha=LAND_FILL_ALPHA,
            line_color=None,
        )
        return

    fallback_provider = resolve_tile_provider(MAP_TILE_FALLBACK_PROVIDER)
    if fallback_provider is not None:
        tile_renderer = plot.add_tile(
            fallback_provider, retina=MAP_TILE_FALLBACK_RETINA
        )
        tile_renderer.alpha = MAP_TILE_FALLBACK_ALPHA
        if not _LAND_PATCH_WARNING_SHOWN:
            print(
                "[WARN] Primary basemap unavailable; using fallback basemap "
                f"'{MAP_TILE_FALLBACK_PROVIDER}'."
            )
            _LAND_PATCH_WARNING_SHOWN = True
        return

    if not _LAND_PATCH_WARNING_SHOWN:
        print(
            "[WARN] No basemap available (primary/fallback); rendering water-only background."
        )
        _LAND_PATCH_WARNING_SHOWN = True


def style_map_legend(plot: Any) -> None:
    """Apply shared legend styling."""
    for legend in plot.legend:
        legend.location = MAP_LEGEND_LOCATION
        legend.click_policy = "hide"
        legend.label_text_font_size = MAP_LEGEND_FONT_SIZE
        legend.glyph_width = MAP_LEGEND_GLYPH_WIDTH
        legend.glyph_height = MAP_LEGEND_GLYPH_HEIGHT
        legend.label_standoff = MAP_LEGEND_LABEL_STANDOFF
        legend.spacing = MAP_LEGEND_SPACING


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


def format_bandwidth_text(bandwidth: float) -> str:
    """Format the bandwidth display text."""
    if np.isfinite(bandwidth):
        return f"Bandwidth: {bandwidth:.4f} (method={BANDWIDTH_METHOD})"
    return "Bandwidth: n/a"


def extract_logit_range(values: Iterable[object]) -> tuple[float, float] | None:
    """Return (min, max) logits from a sequence, or None if unavailable."""
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = np.isfinite(series)
    if not finite.any():
        return None
    min_val = float(series[finite].min())
    max_val = float(series[finite].max())
    return min_val, max_val


def infer_logit_step(logit_min: float, logit_max: float) -> float:
    """Pick a reasonable slider step for the logit range."""
    span = logit_max - logit_min
    if span <= 0:
        return 0.01
    if span <= 1:
        return 0.01
    if span <= 5:
        return 0.05
    if span <= 20:
        return 0.1
    if span <= 50:
        return 0.25
    return 0.5


def build_spectrogram_url(
    *,
    filename: str,
    spectrogram_dir: Path,
    image_format: str,
    base_url: str | None,
    inline: bool,
) -> tuple[str, bool]:
    """Build a spectrogram URL (or inline data URI) if available."""
    if not filename:
        return "", False
    image_name = f"{Path(filename).stem}.{image_format}"
    image_path = spectrogram_dir / image_name
    if not image_path.exists():
        return "", False
    if inline:
        try:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            return f"data:image/{image_format};base64,{encoded}", True
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to inline {image_name}: {exc}")
            return "", False
    if base_url:
        return f"{base_url.rstrip('/')}/{image_name}", True
    return "", False


def build_media_url(base_url: str | None, species_slug: str, filename: str) -> str:
    """Construct a media URL, handling base URLs with or without species suffixes."""
    if not base_url or not filename:
        return ""
    normalized = base_url.rstrip("/")
    suffix = f"/{species_slug}"
    if normalized.endswith(suffix):
        return f"{normalized}/{filename}"
    return f"{normalized}{suffix}/{filename}"


def coerce_clip_index(value: object) -> int | None:
    """Return clip index as an int when possible."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None


def build_audio_filename(entry: dict[str, Any], species_slug: str) -> str:
    """Derive an audio filename from available metadata."""
    for key in ("clip_file", "file_path", "filepath", "path"):
        value = entry.get(key)
        if value:
            return Path(str(value)).name

    filename_value = entry.get("filename")
    if filename_value:
        candidate = Path(str(filename_value)).name
        if candidate and Path(candidate).suffix.lower() in {
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
        }:
            return candidate

    xcid = entry.get("xcid")
    clip_index = coerce_clip_index(entry.get("clip_index"))
    if xcid and clip_index is not None:
        return f"{species_slug}_{str(xcid).strip()}_{clip_index:02d}.wav"
    return ""


def build_playlist_html(
    entries: list[dict[str, Any]],
    *,
    spectrogram_dir: Path,
    image_format: str,
    base_url: str | None,
    inline: bool,
    audio_base_url: str | None,
    species_slug: str,
) -> str:
    """Build HTML for the playlist panel."""
    if not entries:
        return "<i>No rendered points.</i>"

    rows: list[str] = []
    for idx, entry in enumerate(entries, start=1):
        filename = html.escape(str(entry.get("filename", "") or "").strip())
        recordist = html.escape(str(entry.get("recordist", "") or "").strip())
        date = html.escape(str(entry.get("date", "") or "").strip())
        xcid = html.escape(str(entry.get("xcid", "") or "").strip())
        clip_index = html.escape(str(entry.get("clip_index", "") or "").strip())
        logit_text = ""
        raw_logit = entry.get("logits")
        if raw_logit is not None and not (
            isinstance(raw_logit, float) and np.isnan(raw_logit)
        ):
            try:
                logit_value = float(raw_logit)
            except (TypeError, ValueError):
                logit_value = None
            if logit_value is not None and np.isfinite(logit_value):
                logit_text = f"logit: {logit_value:.3f}"

        meta_bits = [f"<b>#{idx}</b>"]
        if filename:
            meta_bits.append(filename)
        if recordist:
            meta_bits.append(f"rec: {recordist}")
        if date:
            meta_bits.append(f"date: {date}")
        if xcid:
            meta_bits.append(f"xcid: {xcid}")
        if clip_index:
            meta_bits.append(f"clip: {clip_index}")
        if logit_text:
            meta_bits.append(logit_text)

        audio_url = ""
        if audio_base_url:
            audio_filename = build_audio_filename(entry, species_slug)
            if audio_filename:
                audio_url = build_media_url(
                    audio_base_url, species_slug, audio_filename
                )
        if audio_url:
            audio_html = (
                f"<audio controls preload='none' src='{html.escape(audio_url)}'>"
                "Your browser does not support audio."
                "</audio>"
            )
        else:
            audio_html = "<div style='color:#888;'>No audio</div>"

        url, exists = build_spectrogram_url(
            filename=entry.get("filename", "") or "",
            spectrogram_dir=spectrogram_dir,
            image_format=image_format,
            base_url=base_url,
            inline=inline,
        )
        if exists:
            img_html = (
                f"<img src='{html.escape(url)}' "
                "style='width: 780px; max-width: 100%; border: 1px solid #ddd;'>"
            )
        else:
            img_html = "<div style='color:#888;'>No spectrogram</div>"

        source_index = entry.get("source_index")
        if source_index is None:
            data_attr = ""
            cursor_style = ""
            highlight_button = ""
        else:
            data_attr = f" data-playlist-index='{int(source_index)}'"
            cursor_style = "cursor: pointer;"
            highlight_button = (
                "<button style='margin-right:6px; padding:2px 6px;' "
                f"onclick='if(window._kde_selectPoint){{window._kde_selectPoint({int(source_index)})}}"
                "return false;'>Highlight</button>"
            )

        rows.append(
            f"<div style='margin-bottom: 14px; {cursor_style}'{data_attr}>"
            "<div style='margin-bottom: 4px; display:flex; align-items:center; "
            "gap:6px; flex-wrap:wrap;'>"
            f"{highlight_button}<div>{' | '.join(meta_bits)}</div>"
            "</div>"
            f"{audio_html}"
            f"<div style='margin-top: 6px;'>{img_html}</div>"
            "</div>"
        )

    return (
        "<div style='overflow-y: auto; height: 680px; padding-right: 6px;'>"
        + "".join(rows)
        + "</div>"
    )


def build_playlist_entries_from_source(
    source_data: dict[str, list],
    order: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Convert ColumnDataSource data into an ordered list of entry dicts."""
    if not source_data or "x" not in source_data:
        return []
    size = len(source_data["x"])
    if size == 0:
        return []
    if order is None:
        indices = range(size)
    else:
        indices = [idx for idx in order if 0 <= idx < size]
    entries: list[dict[str, Any]] = []
    for idx in indices:
        entry = {key: values[idx] for key, values in source_data.items()}
        entry["source_index"] = int(idx)
        entries.append(entry)
    return entries


def compute_isopleth_sources(
    points_eq: np.ndarray,
    probabilities: Sequence[float],
    contour_transformer: Transformer,
    *,
    knn_scale_override: float | None = None,
) -> tuple[list[dict[str, list[list[float]]]], list[Isopleth], float]:
    """Compute projected isopleth sources for plotting."""
    if len(points_eq) < MIN_POINTS_FOR_KDE:
        return empty_isopleth_sources(len(probabilities)), [], float("nan")

    grid, values, bandwidth = compute_kde(
        points_eq, knn_scale_override=knn_scale_override
    )
    isopleths = build_isopleths(grid, values, NUM_ISOCLINES)
    projected_isopleths = project_isopleths(isopleths, contour_transformer)
    aligned_paths = align_isopleth_paths(projected_isopleths, probabilities)
    source_data = [
        build_isopleth_source_data(paths) for paths in aligned_paths
    ]
    return source_data, projected_isopleths, bandwidth


def plot_map(
    points_df: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    isopleths: list[Isopleth],
    bandwidth: float,
    output_html: Path | None = None,
    title: str | None = None,
    species_slug: str = "unknown_species",
) -> None:
    """Render the map, points, and KDE isopleths."""
    point_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_merc, y_merc = project_points(lon, lat, point_transformer)

    contour_transformer = Transformer.from_crs("EPSG:3035", "EPSG:3857", always_xy=True)
    projected_isopleths = project_isopleths(isopleths, contour_transformer)

    x_range, y_range = compute_bounds(
        np.array(x_merc), np.array(y_merc), projected_isopleths
    )

    plot_title = title or build_map_title_text(species_slug)
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
    style_map_figure(plot)

    palette = build_heat_palette(len(projected_isopleths))
    stroke_palette = build_isocline_stroke_palette(
        len(projected_isopleths), fill_palette=palette
    )
    ordered_isopleths = sorted(projected_isopleths, key=lambda iso: iso.level)
    legend_added = False

    if FILL_ISOCLINES:
        if use_band_fill_mode():
            ordered_paths = [iso.paths for iso in ordered_isopleths]
            band_sources = build_band_sources_from_ordered_paths(ordered_paths)
            for iso, band_source, fill_color, stroke_color in zip(
                ordered_isopleths, band_sources, palette, stroke_palette
            ):
                if band_source["xs"]:
                    plot.multi_polygons(
                        xs=band_source["xs"],
                        ys=band_source["ys"],
                        fill_color=fill_color,
                        fill_alpha=ISOCLINE_FILL_ALPHA,
                        line_color=stroke_color,
                        line_alpha=ISOCLINE_STROKE_ALPHA,
                        line_width=max(0.0, ISOCLINE_STROKE_WIDTH),
                        legend_label=f"{iso.probability:.0%} isocline",
                    )
                    legend_added = True
        else:
            for iso, fill_color, stroke_color in zip(
                ordered_isopleths, palette, stroke_palette
            ):
                xs_list = [path[:, 0].tolist() for path in iso.paths]
                ys_list = [path[:, 1].tolist() for path in iso.paths]
                if xs_list:
                    plot.patches(
                        xs_list,
                        ys_list,
                        fill_color=fill_color,
                        fill_alpha=ISOCLINE_FILL_ALPHA,
                        line_color=stroke_color,
                        line_alpha=ISOCLINE_STROKE_ALPHA,
                        line_width=max(0.0, ISOCLINE_STROKE_WIDTH),
                        legend_label=f"{iso.probability:.0%} isocline",
                    )
                    legend_added = True
    else:
        for iso, stroke_color in zip(ordered_isopleths, stroke_palette):
            xs_list = [path[:, 0].tolist() for path in iso.paths]
            ys_list = [path[:, 1].tolist() for path in iso.paths]
            if xs_list:
                plot.multi_line(
                    xs=xs_list,
                    ys=ys_list,
                    line_color=stroke_color,
                    line_width=ISOCLINE_LINE_WIDTH,
                    line_alpha=ISOCLINE_LINE_ALPHA,
                    legend_label=f"{iso.probability:.0%} isocline",
                )
                legend_added = True

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
        line_width=POINT_LINE_WIDTH,
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

    if legend_added:
        style_map_legend(plot)

    output_path = output_html or OUTPUT_HTML
    output_file(output_path, title=plot_title)
    show(plot)

    print(
        f"[INFO] Rendered {len(lon)} points with bandwidth={bandwidth:.4f} "
        f"(method={BANDWIDTH_METHOD})."
    )
    print(f"[INFO] Output written to {output_path}.")


def build_static_interactive_layout(
    points_df: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    points_eq: np.ndarray,
    logit_range: tuple[float, float] | None = None,
    spectrogram_dir: Path | None = None,
    spectrogram_base_url: str | None = None,
    spectrogram_image_format: str = "png",
    inline_spectrograms: bool = False,
    audio_base_url: str | None = None,
    species_slug: str = "unknown_species",
    playlist_max_items: int | None = None,
) -> tuple[LayoutDOM, ColumnDataSource, Div]:
    """Build the interactive static KDE layout with controls."""
    point_transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )
    contour_transformer = Transformer.from_crs(
        "EPSG:3035", "EPSG:3857", always_xy=True
    )
    if spectrogram_dir is None:
        spectrogram_dir = Path(".")
    x_merc, y_merc = project_points(lon, lat, point_transformer)

    probabilities = get_isocline_probabilities(NUM_ISOCLINES).tolist()
    source_data, projected_isopleths, bandwidth = compute_isopleth_sources(
        points_eq,
        probabilities,
        contour_transformer,
        knn_scale_override=BANDWIDTH_KNN_SCALE,
    )
    x_range, y_range = compute_bounds(
        np.array(x_merc), np.array(y_merc), projected_isopleths
    )

    plot = figure(
        title=build_map_title_text(species_slug),
        x_axis_type="mercator",
        y_axis_type="mercator",
        match_aspect=True,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        tools="pan,wheel_zoom,tap,reset,save",
        x_range=x_range,
        y_range=y_range,
    )
    style_map_figure(plot)

    palette = build_heat_palette(len(probabilities))
    stroke_palette = build_isocline_stroke_palette(
        len(probabilities), fill_palette=palette
    )
    use_band_fill = FILL_ISOCLINES and use_band_fill_mode()
    render_order = list(range(len(probabilities) - 1, -1, -1))
    band_source_data = (
        build_band_sources_from_level_sources(source_data, render_order)
        if use_band_fill
        else []
    )
    iso_sources: list[ColumnDataSource] = []
    legend_added = False
    for draw_idx, source_idx in enumerate(render_order):
        initial_source_data = (
            band_source_data[draw_idx] if use_band_fill else source_data[source_idx]
        )
        source = ColumnDataSource(initial_source_data)
        iso_sources.append(source)
        if FILL_ISOCLINES:
            if use_band_fill:
                plot.multi_polygons(
                    xs="xs",
                    ys="ys",
                    source=source,
                    fill_color=palette[draw_idx],
                    fill_alpha=ISOCLINE_FILL_ALPHA,
                    line_color=stroke_palette[draw_idx],
                    line_alpha=ISOCLINE_STROKE_ALPHA,
                    line_width=max(0.0, ISOCLINE_STROKE_WIDTH),
                    legend_label=f"{probabilities[source_idx]:.0%} isocline",
                )
                legend_added = True
            else:
                plot.patches(
                    "xs",
                    "ys",
                    source=source,
                    fill_color=palette[draw_idx],
                    fill_alpha=ISOCLINE_FILL_ALPHA,
                    line_color=stroke_palette[draw_idx],
                    line_alpha=ISOCLINE_STROKE_ALPHA,
                    line_width=max(0.0, ISOCLINE_STROKE_WIDTH),
                    legend_label=f"{probabilities[source_idx]:.0%} isocline",
                )
                legend_added = True
        else:
            plot.multi_line(
                xs="xs",
                ys="ys",
                source=source,
                line_color=stroke_palette[draw_idx],
                line_width=ISOCLINE_LINE_WIDTH,
                line_alpha=ISOCLINE_LINE_ALPHA,
                legend_label=f"{probabilities[source_idx]:.0%} isocline",
            )
            legend_added = True

    if legend_added:
        style_map_legend(plot)

    base_data = build_point_source_data(points_df, lon, lat, x_merc, y_merc)
    base_arrays = {key: np.asarray(values) for key, values in base_data.items()}
    point_logits = pd.to_numeric(points_df.get("logits"), errors="coerce").to_numpy()
    has_point_logits = np.isfinite(point_logits).any()
    slider_bounds = logit_range or extract_logit_range(point_logits)
    slider_enabled = has_point_logits and slider_bounds is not None

    if slider_bounds is None:
        slider_min, slider_max = 0.0, 1.0
    else:
        slider_min, slider_max = slider_bounds
        if not np.isfinite(slider_min) or not np.isfinite(slider_max):
            slider_min, slider_max = 0.0, 1.0
            slider_enabled = False
        elif slider_max <= slider_min:
            slider_max = slider_min + 1.0

    logit_slider = RangeSlider(
        title="Logit range",
        start=slider_min,
        end=slider_max,
        value=(slider_min, slider_max),
        step=infer_logit_step(slider_min, slider_max),
        width=320,
        disabled=not slider_enabled,
    )

    if "date_dt" in points_df.columns:
        date_series = points_df["date_dt"]
    elif "date" in points_df.columns:
        date_series = points_df["date"]
    else:
        date_series = pd.Series([pd.NaT] * len(points_df))

    def build_histogram_data(mask: np.ndarray) -> dict[str, list]:
        counts = compute_monthly_histogram(date_series[mask])
        return {"month": MONTH_LABELS, "count": counts.astype(int).tolist()}

    hist_max = int(compute_monthly_histogram(date_series).max())
    hist_max = max(hist_max, 1)

    yearly_all = compute_yearly_histogram(date_series)
    if yearly_all.empty:
        year_labels = ["n/a"]
        year_max = 1
    else:
        year_labels = [str(year) for year in yearly_all.index]
        year_max = max(int(yearly_all.max()), 1)

    def build_yearly_data(mask: np.ndarray) -> dict[str, list]:
        if year_labels == ["n/a"]:
            return {"year": year_labels, "count": [0]}
        counts = compute_yearly_histogram(date_series[mask])
        count_map = {int(year): int(count) for year, count in counts.items()}
        values = [int(count_map.get(int(year), 0)) for year in year_labels]
        return {"year": year_labels, "count": values}

    playlist_title = Div(text="<b>Playlist</b>")
    playlist_div = Div(text="<i>No rendered points.</i>", width=PLAYLIST_WIDTH)

    point_source = ColumnDataSource(base_data)
    point_renderer = plot.scatter(
        x="x",
        y="y",
        size=POINT_SIZE,
        color="white",
        alpha=POINT_ALPHA,
        line_color="black",
        line_width=POINT_LINE_WIDTH,
        selection_fill_color="black",
        selection_line_color="black",
        selection_line_width=2,
        selection_alpha=1.0,
        nonselection_alpha=0.2,
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

    bandwidth_div = Div(text=format_bandwidth_text(bandwidth))
    status_div = Div(text="")
    count_div = Div(text="")
    if not slider_enabled:
        status_div.text = "No logits available for filtering; showing all points."
    if not np.isfinite(bandwidth):
        status_div.text = "Not enough points for KDE; showing points only."
    scale_spinner = Spinner(
        title="BANDWIDTH_KNN_SCALE",
        value=BANDWIDTH_KNN_SCALE,
        step=0.05,
        width=180,
    )
    recalc_button = Button(label="Recalculate KDE", button_type="primary")

    def build_logit_mask() -> np.ndarray:
        if not slider_enabled:
            return np.ones(len(point_logits), dtype=bool)
        min_val, max_val = logit_slider.value
        finite = np.isfinite(point_logits)
        return finite & (point_logits >= min_val) & (point_logits <= max_val)

    hist_source = ColumnDataSource(build_histogram_data(build_logit_mask()))
    hist_plot = figure(
        title="Monthly observation counts",
        x_range=MONTH_LABELS,
        y_range=(0, hist_max * HISTOGRAM_Y_PADDING),
        width=PLOT_WIDTH,
        height=HISTOGRAM_HEIGHT,
        tools="",
        toolbar_location=None,
    )
    hist_plot.vbar(
        x="month",
        top="count",
        width=0.9,
        source=hist_source,
        color=HISTOGRAM_BAR_COLOR,
        alpha=HISTOGRAM_ALPHA,
    )
    hist_plot.xgrid.grid_line_color = None
    hist_plot.yaxis.axis_label = "Observations"
    hist_plot.xaxis.axis_label = "Month"

    year_source = ColumnDataSource(build_yearly_data(build_logit_mask()))
    year_plot = figure(
        title="Yearly sample size",
        x_range=year_labels,
        y_range=(0, year_max * HISTOGRAM_Y_PADDING),
        width=int((PLOT_WIDTH + PLAYLIST_WIDTH) / 3),
        height=YEARLY_HEIGHT,
        tools="",
        toolbar_location=None,
    )
    year_plot.line(
        x="year",
        y="count",
        source=year_source,
        line_color=HISTOGRAM_BAR_COLOR,
        line_width=2,
        alpha=HISTOGRAM_ALPHA,
    )
    year_plot.circle(
        x="year",
        y="count",
        source=year_source,
        size=6,
        color=HISTOGRAM_BAR_COLOR,
        alpha=HISTOGRAM_ALPHA,
    )
    year_plot.xgrid.grid_line_color = None
    year_plot.yaxis.axis_label = "Observations"
    year_plot.xaxis.axis_label = "Year"

    def update_points(*, update_playlist: bool = True) -> np.ndarray:
        mask = build_logit_mask()
        point_source.data = {
            key: values[mask].tolist() for key, values in base_arrays.items()
        }
        point_source.selected.indices = []
        count_div.text = f"Points shown: {int(mask.sum())} of {len(point_logits)}"
        if update_playlist:
            data = point_source.data
            logits = np.asarray(data.get("logits", []), dtype=float)
            if logits.size == 0:
                entries = []
            else:
                finite = np.isfinite(logits)
                sort_key = np.where(finite, -logits, np.inf)
                order = np.argsort(sort_key)
                if playlist_max_items is not None:
                    order = order[:playlist_max_items]
                entries = build_playlist_entries_from_source(data, order)
            playlist_div.text = build_playlist_html(
                entries,
                spectrogram_dir=spectrogram_dir,
                image_format=spectrogram_image_format,
                base_url=spectrogram_base_url,
                inline=inline_spectrograms,
                audio_base_url=audio_base_url,
                species_slug=species_slug,
            )
        return mask

    update_points(update_playlist=True)

    def on_logit_change(attr: str, old: object, new: object) -> None:
        update_points(update_playlist=False)
        status_div.text = "Logit filter updated. Press Recalculate KDE to refresh playlist."

    logit_slider.on_change("value", on_logit_change)

    def on_selection_change(attr: str, old: list[int], new: list[int]) -> None:
        if not new:
            return
        selected_idx = new[0]
        data = point_source.data
        if not data or "x" not in data:
            return
        xs = np.asarray(data.get("x", []), dtype=float)
        ys = np.asarray(data.get("y", []), dtype=float)
        if selected_idx >= len(xs):
            return
        valid = np.isfinite(xs) & np.isfinite(ys)
        if not valid[selected_idx]:
            return
        dist = np.full(xs.shape, np.inf, dtype=float)
        dx = xs[valid] - xs[selected_idx]
        dy = ys[valid] - ys[selected_idx]
        dist[valid] = dx * dx + dy * dy
        order = np.argsort(dist)
        if playlist_max_items is not None:
            order = order[:playlist_max_items]
        entries = build_playlist_entries_from_source(data, order)
        playlist_div.text = build_playlist_html(
            entries,
            spectrogram_dir=spectrogram_dir,
            image_format=spectrogram_image_format,
            base_url=spectrogram_base_url,
            inline=inline_spectrograms,
            audio_base_url=audio_base_url,
            species_slug=species_slug,
        )
        status_div.text = "Playlist sorted by distance to selected point."

    point_source.selected.on_change("indices", on_selection_change)

    clear_button = Button(label="Clear highlight", button_type="default")

    def clear_selection() -> None:
        point_source.selected.indices = []
        update_points(update_playlist=True)
        status_div.text = "Highlight cleared. Playlist reset to logit order."

    clear_button.on_click(clear_selection)

    def recalc_kde() -> None:
        raw_value = scale_spinner.value
        if raw_value is None:
            status_div.text = "Scale value is required."
            return
        try:
            scale_value = float(raw_value)
        except (TypeError, ValueError):
            status_div.text = "Scale must be numeric."
            return
        if not np.isfinite(scale_value) or scale_value <= 0:
            status_div.text = "Scale must be a positive number."
            return

        status_div.text = "Recalculating KDE..."
        mask = update_points(update_playlist=True)
        filtered_eq = points_eq[mask]
        updated_data, _, updated_bw = compute_isopleth_sources(
            filtered_eq,
            probabilities,
            contour_transformer,
            knn_scale_override=scale_value,
        )
        rendered_data = (
            build_band_sources_from_level_sources(updated_data, render_order)
            if use_band_fill
            else [updated_data[source_idx] for source_idx in render_order]
        )
        for draw_idx, data in enumerate(rendered_data):
            iso_sources[draw_idx].data = data
        bandwidth_div.text = format_bandwidth_text(updated_bw)
        hist_source.data = build_histogram_data(mask)
        year_source.data = build_yearly_data(mask)
        if np.isfinite(updated_bw):
            status_div.text = (
                f"Updated KDE with scale {scale_value:.3f} "
                f"using {int(mask.sum())} points."
            )
        else:
            status_div.text = (
                "Not enough points for KDE; showing points only."
            )

    recalc_button.on_click(recalc_kde)

    controls = column(
        row(recalc_button, clear_button, scale_spinner, bandwidth_div),
        row(logit_slider, count_div),
        status_div,
    )
    map_panel = column(plot, hist_plot, controls)
    playlist_panel = column(playlist_title, playlist_div, width=PLAYLIST_WIDTH)
    layout = column(row(map_panel, playlist_panel), year_plot)
    return layout, point_source, playlist_div


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


def compute_monthly_histogram(date_series: pd.Series) -> np.ndarray:
    """Return counts per calendar month, aggregating across years."""
    if date_series.empty:
        return np.zeros(12, dtype=int)
    parsed = pd.to_datetime(
        date_series, errors="coerce", dayfirst=DATE_PARSE_DAYFIRST
    )
    months = parsed.dropna().dt.month.astype(int)
    if months.empty:
        return np.zeros(12, dtype=int)
    counts = np.bincount(months, minlength=13)[1:13]
    return counts


def compute_yearly_histogram(date_series: pd.Series) -> pd.Series:
    """Return counts per year for the provided dates."""
    if date_series.empty:
        return pd.Series(dtype=int)
    parsed = pd.to_datetime(
        date_series, errors="coerce", dayfirst=DATE_PARSE_DAYFIRST
    )
    years = parsed.dropna().dt.year.astype(int)
    if years.empty:
        return pd.Series(dtype=int)
    return years.value_counts().sort_index()


def build_animation_frames(
    df: pd.DataFrame,
    windows: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
    probabilities: Sequence[float],
    point_transformer: Transformer,
    equal_area: Transformer,
    contour_transformer: Transformer,
    species_slug: str,
) -> list[dict[str, object]]:
    """Compute per-frame KDEs, sources, and monthly histograms."""
    date_series = df["date_dt"]
    frames: list[dict[str, object]] = []
    render_order = list(range(len(probabilities) - 1, -1, -1))

    for start, end in windows:
        mask = (date_series >= start) & (date_series < end)
        frame_df = df.loc[mask].copy()
        frame_df = filter_top_logit_per_recordist(frame_df, log=False)
        points_df, lon, lat = prepare_points(frame_df)
        x_merc, y_merc = project_points(lon, lat, point_transformer)
        point_data = build_point_source_data(points_df, lon, lat, x_merc, y_merc)
        month_counts = (
            compute_monthly_histogram(points_df["date_dt"])
            if "date_dt" in points_df.columns
            else np.zeros(12, dtype=int)
        )
        histogram = {
            "month": MONTH_LABELS,
            "count": month_counts.astype(int).tolist(),
        }
        hist_max = int(month_counts.max()) if month_counts.size else 0

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
        isopleth_bands = build_band_sources_from_level_sources(
            isopleth_sources, render_order
        )

        frame_label = format_frame_label(start, end)
        bandwidth_text = format_bandwidth_text(bandwidth)

        frames.append(
            {
                "points": point_data,
                "isopleths": isopleth_sources,
                "isopleth_bands": isopleth_bands,
                "title": build_map_title_text(species_slug),
                "frame_label": f"Window: {frame_label}",
                "bandwidth_text": bandwidth_text,
                "year_text": f"{start.year}",
                "histogram": histogram,
                "hist_max": hist_max,
            }
        )

    return frames


def plot_animated_map(
    frames: list[dict[str, object]],
    probabilities: Sequence[float],
    x_range: Range1d,
    y_range: Range1d,
    output_html: Path,
    species_slug: str,
) -> None:
    """Render an animated KDE map with a slider and monthly histogram."""
    if not frames:
        raise SystemExit("No animation frames to render.")

    first_frame = frames[0]
    palette = build_heat_palette(len(probabilities))
    stroke_palette = build_isocline_stroke_palette(
        len(probabilities), fill_palette=palette
    )
    use_band_fill = FILL_ISOCLINES and use_band_fill_mode()
    render_order = list(range(len(probabilities) - 1, -1, -1))
    hist_max = max(
        (max(frame["histogram"]["count"]) for frame in frames),
        default=0,
    )
    hist_max = max(hist_max, 1)

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
    style_map_figure(plot)

    point_source = ColumnDataSource(first_frame["points"])
    point_renderer = plot.scatter(
        x="x",
        y="y",
        size=POINT_SIZE,
        color="white",
        alpha=POINT_ALPHA,
        line_color="black",
        line_width=POINT_LINE_WIDTH,
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
    legend_added = False
    for draw_idx, source_idx in enumerate(render_order):
        prob = probabilities[source_idx]
        source_payload = (
            first_frame["isopleth_bands"][draw_idx]
            if use_band_fill
            else first_frame["isopleths"][source_idx]
        )
        source = ColumnDataSource(source_payload)
        iso_sources.append(source)
        if FILL_ISOCLINES:
            if use_band_fill:
                plot.multi_polygons(
                    xs="xs",
                    ys="ys",
                    source=source,
                    fill_color=palette[draw_idx],
                    fill_alpha=ISOCLINE_FILL_ALPHA,
                    line_color=stroke_palette[draw_idx],
                    line_alpha=ISOCLINE_STROKE_ALPHA,
                    line_width=max(0.0, ISOCLINE_STROKE_WIDTH),
                    legend_label=f"{prob:.0%} isocline",
                )
                legend_added = True
            else:
                plot.patches(
                    "xs",
                    "ys",
                    source=source,
                    fill_color=palette[draw_idx],
                    fill_alpha=ISOCLINE_FILL_ALPHA,
                    line_color=stroke_palette[draw_idx],
                    line_alpha=ISOCLINE_STROKE_ALPHA,
                    line_width=max(0.0, ISOCLINE_STROKE_WIDTH),
                    legend_label=f"{prob:.0%} isocline",
                )
                legend_added = True
        else:
            plot.multi_line(
                xs="xs",
                ys="ys",
                source=source,
                line_color=stroke_palette[draw_idx],
                line_width=ISOCLINE_LINE_WIDTH,
                line_alpha=ISOCLINE_LINE_ALPHA,
                legend_label=f"{prob:.0%} isocline",
            )
            legend_added = True

    if legend_added:
        style_map_legend(plot)

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
    hist_source = ColumnDataSource(first_frame["histogram"])
    hist_plot = figure(
        title="Monthly observation counts",
        x_range=MONTH_LABELS,
        y_range=(0, hist_max * HISTOGRAM_Y_PADDING),
        width=PLOT_WIDTH,
        height=HISTOGRAM_HEIGHT,
        tools="",
        toolbar_location=None,
    )
    hist_plot.vbar(
        x="month",
        top="count",
        width=0.9,
        source=hist_source,
        color=HISTOGRAM_BAR_COLOR,
        alpha=HISTOGRAM_ALPHA,
    )
    hist_plot.xgrid.grid_line_color = None
    hist_plot.yaxis.axis_label = "Observations"
    hist_plot.xaxis.axis_label = "Month"
    slider = Slider(
        start=0, end=len(frames) - 1, value=0, step=1, title="Frame"
    )

    slider_callback = CustomJS(
        args=dict(
            frames=frames,
            point_source=point_source,
            iso_sources=iso_sources,
            render_order=render_order,
            use_band_fill=use_band_fill,
            plot=plot,
            frame_div=frame_div,
            bandwidth_div=bandwidth_div,
            year_label=year_label,
            hist_source=hist_source,
        ),
        code="""
            const frame = frames[cb_obj.value];
            point_source.data = frame.points;
            point_source.change.emit();
            for (let i = 0; i < iso_sources.length; i++) {
                if (use_band_fill) {
                    iso_sources[i].data = frame.isopleth_bands[i];
                } else {
                    const source_idx = render_order[i];
                    iso_sources[i].data = frame.isopleths[source_idx];
                }
                iso_sources[i].change.emit();
            }
            plot.title.text = frame.title;
            frame_div.text = frame.frame_label;
            bandwidth_div.text = frame.bandwidth_text;
            year_label.text = frame.year_text;
            hist_source.data = frame.histogram;
            hist_source.change.emit();
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
        hist_plot,
        row(play_button, slider),
        row(frame_div, bandwidth_div),
    )
    output_file(output_html, title=f"{build_map_title_text(species_slug)} animation")
    show(layout)

    print(f"[INFO] Animated output written to {output_html}.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize an ingroup's geographic distribution with KDE isoclines."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--animate",
        action="store_true",
        help=(
            "Render an animated KDE with 12 monthly frames per year, "
            "each frame covering a 1-year window."
        ),
    )
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Launch a local Bokeh server for the static map with KDE controls."
        ),
    )
    return parser


def run_static_map() -> None:
    """Render a static KDE map."""
    log_isocline_percentages(NUM_ISOCLINES)
    species_slug = detect_species_slug(INFERENCE_CSV)
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

    plot_map(
        points_df,
        lon,
        lat,
        isopleths,
        bandwidth,
        species_slug=species_slug,
    )


def run_interactive_map() -> None:
    """Launch a Bokeh server with interactive KDE controls."""
    log_isocline_percentages(NUM_ISOCLINES)
    inference_df = prepare_inference_table(INFERENCE_CSV)
    logit_range = extract_logit_range(inference_df["logits"])
    metadata_df = prepare_metadata_table(METADATA_CSV)
    merged = merge_inference_metadata(inference_df, metadata_df)
    dated = apply_date_filter(merged)
    filtered = filter_top_logit_per_recordist(dated)
    points_df, lon, lat = prepare_points(filtered)

    if len(lon) == 0:
        raise SystemExit("No valid coordinates available for plotting.")

    data_root = detect_root_path(INFERENCE_CSV)
    species_slug = detect_species_slug(INFERENCE_CSV)
    spectrogram_dir = (
        SPECTROGRAMS_DIR
        if SPECTROGRAMS_DIR is not None
        else data_root / "spectrograms" / species_slug
    )
    audio_dir = AUDIO_DIR if AUDIO_DIR is not None else data_root / "clips"

    spectrogram_base = SPECTROGRAM_BASE_URL
    if spectrogram_base is None and SPECTROGRAM_AUTO_SERVE:
        spectrogram_base = start_static_file_server(
            label=f"Spectrograms ({species_slug})",
            directory=spectrogram_dir,
            host=SPECTROGRAM_HOST,
            port=SPECTROGRAM_PORT,
        )

    audio_base = AUDIO_BASE_URL
    if audio_base is None and AUDIO_AUTO_SERVE:
        audio_generated = start_static_file_server(
            label=f"Audio ({species_slug})",
            directory=audio_dir,
            host=AUDIO_HOST,
            port=AUDIO_PORT,
        )
        if audio_generated:
            audio_base = audio_generated.rstrip("/")

    equal_area = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x_eq, y_eq = equal_area.transform(lon, lat)
    points_eq = np.column_stack((x_eq, y_eq))

    def modify_doc(doc) -> None:
        layout, point_source, _ = build_static_interactive_layout(
            points_df,
            lon,
            lat,
            points_eq,
            logit_range=logit_range,
            spectrogram_dir=spectrogram_dir,
            spectrogram_base_url=spectrogram_base,
            spectrogram_image_format=SPECTROGRAM_IMAGE_FORMAT,
            inline_spectrograms=INLINE_SPECTROGRAMS,
            audio_base_url=audio_base,
            species_slug=species_slug,
            playlist_max_items=PLAYLIST_MAX_ITEMS,
        )
        doc.add_root(layout)
        doc.title = build_map_title_text(species_slug)
        doc.js_on_event(
            DocumentReady,
            CustomJS(
                args=dict(source=point_source),
                code="""
                    if (!window._kde_playlist_listener_attached) {
                        window._kde_playlist_listener_attached = true;
                        window._kde_selectPoint = (idx) => {
                            if (idx === null || idx === undefined) {
                                return;
                            }
                            const safeIdx = Number(idx);
                            if (!Number.isFinite(safeIdx)) {
                                return;
                            }
                            source.selected.indices = [safeIdx];
                            source.selected.change.emit();
                            source.change.emit();
                        };
                        document.addEventListener("click", (event) => {
                            const target = event.target.closest("[data-playlist-index]");
                            if (!target) {
                                return;
                            }
                            const idx = Number(target.dataset.playlistIndex);
                            if (!Number.isFinite(idx)) {
                                return;
                            }
                            window._kde_selectPoint(idx);
                        });
                    }
                """,
            ),
        )

    server = Server({"/": modify_doc}, num_procs=1)
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


def run_animated_map() -> None:
    """Render an animated KDE map with month-by-month frames."""
    log_isocline_percentages(NUM_ISOCLINES)
    species_slug = detect_species_slug(INFERENCE_CSV)
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

    probabilities = get_isocline_probabilities(NUM_ISOCLINES).tolist()
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
        species_slug,
    )
    plot_animated_map(
        frames, probabilities, x_range, y_range, ANIMATION_OUTPUT_HTML, species_slug
    )


def main() -> None:
    """Parse CLI arguments and render the requested output."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.animate:
        run_animated_map()
    elif args.interactive:
        run_interactive_map()
    else:
        run_static_map()


if __name__ == "__main__":
    main()
