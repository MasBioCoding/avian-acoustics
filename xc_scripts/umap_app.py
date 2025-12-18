"""
Bokeh application for UMAP visualization of bird audio embeddings.

To run:
1. Save this file as 'xc_scripts/umap_app.py'
2. Run the Bokeh app:
    cd /path/to/birdnet_data_pipeline
    for me: cd /Users/masjansma/Desktop/birdnetcluster1folder/birdnet_data_pipeline
    bokeh serve xc_scripts/umap_app.py --session-token-expiration 1800 --keep-alive 60000 --websocket-max-message-size 200000000 --show --args --config xc_configs/config_limosa_limosa.yaml
    bokeh serve xc_scripts/umap_app.py --session-token-expiration 1800 --keep-alive 60000 --websocket-max-message-size 200000000 --show --args --config xc_configs/config_chloris_chloris.yaml
    bokeh serve xc_scripts/umap_app.py --session-token-expiration 1800 --keep-alive 60000 --websocket-max-message-size 200000000 --show --args --config xc_configs/config_regulus_ignicapilla.yaml
    bokeh serve xc_scripts/umap_app.py --session-token-expiration 1800 --keep-alive 60000 --websocket-max-message-size 200000000 --show --args --config xc_configs/config_regulus_regulus.yaml
    bokeh serve xc_scripts/umap_app.py --session-token-expiration 1800 --keep-alive 60000 --websocket-max-message-size 200000000 --show --args --config xc_configs/config_curruca_communis.yaml
    bokeh serve xc_scripts/umap_app.py --session-token-expiration 1800 --keep-alive 60000 --websocket-max-message-size 200000000 --show --args --config xc_configs/config_carduelis_carduelis.yaml
    bokeh serve xc_scripts/umap_app.py --session-token-expiration 1800 --keep-alive 60000 --websocket-max-message-size 200000000 --show --args --config xc_configs/config_acrocephalus_scirpaceus.yaml
    bokeh serve xc_scripts/umap_app.py --session-token-expiration 1800 --keep-alive 60000 --websocket-max-message-size 200000000 --show --args --config xc_configs/config_phylloscopus_trochilus.yaml

    bokeh serve xc_scripts/umap_app.py --session-token-expiration 1800 --keep-alive 60000 --websocket-max-message-size 200000000 --show --args --config xc_configs/config_prunella_modularis.yaml
    bokeh serve xc_scripts/umap_app.py --session-token-expiration 1800 --keep-alive 60000 --websocket-max-message-size 200000000 --show --args --config xc_configs/config_phylloscopus_trochilus.yaml
    bokeh serve xc_scripts/umap_app.py --session-token-expiration 1800 --keep-alive 60000 --websocket-max-message-size 200000000 --show --args --config xc_configs/config_linaria_cannabina.yaml

    
    """

import argparse
import base64
import errno
import json
import sys
from datetime import datetime
import yaml
from pathlib import Path
from typing import Any, Optional, Tuple
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from importlib_metadata import metadata
import pandas as pd
import numpy as np
import umap
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
import traceback
import colorsys
from collections import Counter

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, Button, Div, DateRangeSlider, HoverTool, BoxSelectTool,
    LassoSelectTool, PolySelectTool,
    TapTool, Toggle, Select, CheckboxGroup, CustomJS, RangeSlider, CDSView, BooleanFilter,
    Spinner, Arrow, NormalHead, Range1d
)
from bokeh.events import ButtonClick
from bokeh.plotting import figure
import hdbscan

SELECTION_PALETTE = [
    "#4477AA",  # Blue
    "#EE6677",  # Red
    "#228833",  # Green
    "#CCBB44",  # Yellow
    "#66CCEE",  # Cyan
    "#AA3377",  # Magenta
    "#BBBBBB",  # Gray
    "#000000",  # Black
    "#FFA500",  # Orange
    "#00CED1",  # Dark turquoise
    "#6A5ACD",  # Slate blue
    "#D2691E",  # Chocolate
    "#FF1493",  # Deep pink
    "#40E0D0",  # Turquoise
    "#FFD700",  # Gold
    "#7CFC00",  # Lawn green
]
SELECTION_UNASSIGNED_COLOR = "#bdbdbd"

EARTH_RADIUS_KM = 6371.0088
GEOSHIFT_MAX_RECORDINGS_PER_CLUSTER = 3
GEOSHIFT_CLUSTER_DIAMETER_KM = 2.0
GEOSHIFT_CLUSTER_RADIUS_KM = GEOSHIFT_CLUSTER_DIAMETER_KM / 2.0
GEOSHIFT_NORTH_COLOR = "#1f78b4"
GEOSHIFT_SOUTH_COLOR = "#d62728"
REGION_UNASSIGNED_COLOR = "#c0c0c0"
REGION_DEFAULT_COLOR = "#005f73"

def normalize_clip_index(value: Any) -> Optional[int]:
    """Convert a clip index-like value to an integer, returning None when invalid."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        # Non-numeric types may raise here; fall through to parsing logic.
        pass

    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None

print("=" * 80)
print("STARTING BOKEH SERVER APP")
print("=" * 80)

_STATIC_HTTP_SERVERS: dict[str, dict[str, Any]] = {}
APP_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------


def load_config(config_path: Optional[Path] = None):
    """Load configuration from file or use defaults"""
    
    # Default configuration with analysis parameters
    default_config = {
        "species": {
            "scientific_name": "Emberiza citrinella",
            "common_name": "Yellowhammer",
            "slug": "emberiza_citrinella"
        },
        "paths": {
            "root": "/Volumes/Z Slim/zslim_birdcluster"
        },
        "audio": {
            "auto_serve": True,
            "host": "127.0.0.1",
            "port": 8765,
            "base_url": None
        },
        "analysis": {
            "umap_n_neighbors": 10,
            "umap_min_dist": 0.0,
            "umap_n_components": 2,
            "point_size": 10,
            "point_alpha": 0.3
        },
        "spectrograms": {
            "auto_serve": True,
            "host": "127.0.0.1",
            "port": 8766,
            "base_url": None,
            "image_format": "png",
            "inline": None
        }
    }
    
    if config_path:
        requested_path = config_path
        resolved_path = config_path if config_path.is_absolute() else (Path.cwd() / config_path)
        if not resolved_path.exists():
            raise SystemExit(
                f"Config file '{requested_path}' not found. Use '--config xc_configs/<name>.yaml'."
            )
        with open(resolved_path) as f:
            config = yaml.safe_load(f)
        # Merge with defaults
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
            elif isinstance(default_config[key], dict):
                for subkey in default_config[key]:
                    if subkey not in config[key]:
                        config[key][subkey] = default_config[key][subkey]
        return config
    
    return default_config

def start_static_file_server(
    *,
    label: str,
    directory: Path,
    host: str,
    port: int,
    log_requests: bool = False,
) -> Optional[tuple[str, bool]]:
    """Start or reuse a background HTTP server to expose local files.

    Returns:
        A tuple of (base_url, started_now) if a usable server is available, or None.
    """

    existing = _STATIC_HTTP_SERVERS.get(label)
    if existing:
        return existing["base_url"], existing["started"]

    if not directory.exists():
        print(f"  {label} directory missing, cannot auto-serve: {directory}")
        return None

    class _StaticHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format: str, *args) -> None:  # type: ignore[override]
            if log_requests:
                super().log_message(format, *args)

    url_host = "localhost" if host in ("0.0.0.0", "") else host
    base_url = f"http://{url_host}:{port}"

    try:
        server = ThreadingHTTPServer((host, port), _StaticHandler)
        server.daemon_threads = True
        server.allow_reuse_address = True
    except OSError as exc:
        reuse_codes = {
            getattr(errno, "EADDRINUSE", None),
            getattr(errno, "WSAEADDRINUSE", None),
            48,  # macOS
            98,  # Linux
            10048,  # Windows
        }
        if exc.errno in reuse_codes:
            print(f"  {label} server already active on {host}:{port}; reusing {base_url}")
            _STATIC_HTTP_SERVERS[label] = {
                "server": None,
                "thread": None,
                "host": host,
                "port": port,
                "directory": directory,
                "base_url": base_url,
                "started": False,
            }
            return base_url, False

        print(f"  ERROR: Could not start {label.lower()} server on {host}:{port} ({exc})")
        return None

    thread = Thread(
        target=server.serve_forever,
        name=f"{label.replace(' ', '')}HTTPServer",
        daemon=True,
    )
    thread.start()

    print(f"  {label} server started at {base_url} serving {directory}")
    _STATIC_HTTP_SERVERS[label] = {
        "server": server,
        "thread": thread,
        "host": host,
        "port": port,
        "directory": directory,
        "base_url": base_url,
        "started": True,
    }
    return base_url, True

# Parse command line arguments to get config file
parser = argparse.ArgumentParser(description="UMAP Visualization App")
parser.add_argument("--config", type=Path, help="Path to config.yaml file")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Set paths based on config
ROOT_PATH = Path(config["paths"]["root"]).expanduser()
SPECIES_SLUG = config["species"]["slug"]
EMBEDDINGS_FILE = ROOT_PATH / "embeddings" / SPECIES_SLUG / "embeddings.csv"
METADATA_FILE = ROOT_PATH / "embeddings" / SPECIES_SLUG / "metadata.csv"
CLIPS_DIR = ROOT_PATH / "clips" / SPECIES_SLUG
XC_GROUPS_ROOT = ROOT_PATH / "xc_groups"
SPECIES_GROUPS_DIR = XC_GROUPS_ROOT / SPECIES_SLUG
VOCAL_TYPES_DIR = SPECIES_GROUPS_DIR / "vocal_types"
DIALECTS_DIR = SPECIES_GROUPS_DIR / "dialects"
ANNOTATE_ONE_DIR = SPECIES_GROUPS_DIR / "annotate_1"
ANNOTATE_TWO_DIR = SPECIES_GROUPS_DIR / "annotate_2"
GROUP_TABLE_SUFFIX = ".csv"
REGION_BOUNDARIES_ROOT = ROOT_PATH / "region_boundaries"
REGION_SPECIES_DIR = REGION_BOUNDARIES_ROOT / SPECIES_SLUG
REGION_VARIANTS: dict[str, dict[str, str]] = {
    "course": {"label": "Region (course)", "filename": "regions_course.json"},
    "fine": {"label": "Region (fine)", "filename": "regions_fine.json"},
    "custom": {"label": "Region (custom)", "filename": "regions_custom.json"},
}


def region_file_path(key: str) -> Path:
    """Return the boundary JSON path for a given region variant key."""

    filename = REGION_VARIANTS.get(key, {}).get("filename") or f"regions_{key}.json"
    return REGION_SPECIES_DIR / filename
_audio_cfg = config.get("audio", {}) or {}
_audio_base = _audio_cfg.get("base_url")
AUDIO_BASE_URL = (
    str(_audio_base).rstrip("/")
    if isinstance(_audio_base, str) and _audio_base
    else None
)
AUDIO_HOST = str(_audio_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
AUDIO_PORT = int(_audio_cfg.get("port", 8765))
AUDIO_AUTO_SERVE_REQUESTED = bool(_audio_cfg.get("auto_serve", True))
AUDIO_SERVER_STARTED = False
AUDIO_SERVER_REUSED = False

if AUDIO_AUTO_SERVE_REQUESTED:
    generated_audio_url = start_static_file_server(
        label=f"Audio ({SPECIES_SLUG})",
        directory=CLIPS_DIR,
        host=AUDIO_HOST,
        port=AUDIO_PORT,
        log_requests=bool(_audio_cfg.get("log_requests", False)),
    )
    if generated_audio_url:
        audio_base, audio_started_now = generated_audio_url
        if audio_started_now:
            AUDIO_SERVER_STARTED = True
        else:
            AUDIO_SERVER_REUSED = True
        if AUDIO_BASE_URL is None:
            AUDIO_BASE_URL = audio_base.rstrip("/")

_spectro_cfg = config.get("spectrograms", {}) or {}
SPECTROGRAM_IMAGE_FORMAT = str(_spectro_cfg.get("image_format", "png")).lower()
SPECTROGRAMS_DIR = ROOT_PATH / "spectrograms" / SPECIES_SLUG
_spectro_inline_flag = _spectro_cfg.get("inline")
_spectro_base = _spectro_cfg.get("base_url")
SPECTROGRAM_BASE_URL = (
    _spectro_base.rstrip("/") if isinstance(_spectro_base, str) and _spectro_base else None
)
SPECTROGRAM_HOST = str(_spectro_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
SPECTROGRAM_PORT = int(_spectro_cfg.get("port", 8766))
SPECTROGRAM_AUTO_SERVE_REQUESTED = bool(_spectro_cfg.get("auto_serve", True))
SPECTROGRAM_SERVER_STARTED = False
SPECTROGRAM_SERVER_REUSED = False

if _spectro_inline_flag is True:
    SPECTROGRAM_AUTO_SERVE_REQUESTED = False

if SPECTROGRAM_BASE_URL is None and SPECTROGRAM_AUTO_SERVE_REQUESTED:
    generated_url = start_static_file_server(
        label=f"Spectrogram ({SPECIES_SLUG})",
        directory=SPECTROGRAMS_DIR,
        host=SPECTROGRAM_HOST,
        port=SPECTROGRAM_PORT,
        log_requests=bool(_spectro_cfg.get("log_requests", False)),
    )
    if generated_url:
        spectro_base, spectro_started_now = generated_url
        if spectro_started_now:
            SPECTROGRAM_SERVER_STARTED = True
            SPECTROGRAM_BASE_URL = spectro_base.rstrip("/")
        else:
            SPECTROGRAM_SERVER_REUSED = True
            SPECTROGRAM_BASE_URL = spectro_base.rstrip("/")
    else:
        _spectro_inline_flag = True  # Fallback if server failed

if _spectro_inline_flag is None:
    INLINE_SPECTROGRAMS = SPECTROGRAM_BASE_URL is None
else:
    INLINE_SPECTROGRAMS = bool(_spectro_inline_flag)

# UMAP parameters from config
ANALYSIS_PARAMS = config.get("analysis", {})

print(f"Configuration loaded:")
print(f"  Species: {config['species']['scientific_name']} ({config['species']['common_name']})")
print(f"  Embeddings: {EMBEDDINGS_FILE}")
print(f"  Metadata: {METADATA_FILE}")
print(f"  Clips: {CLIPS_DIR}")
if AUDIO_BASE_URL:
    print(f"  Audio base URL: {AUDIO_BASE_URL}")
else:
    print("  Audio base URL: <none>")
print(f"  Spectrograms: {SPECTROGRAMS_DIR}")
if SPECTROGRAM_BASE_URL:
    print(f"  Spectrogram base URL: {SPECTROGRAM_BASE_URL}")
else:
    print("  Spectrogram base URL: <none>")
if AUDIO_SERVER_STARTED:
    print(f"  Audio server: auto-started on {AUDIO_HOST}:{AUDIO_PORT}")
elif AUDIO_SERVER_REUSED:
    print(f"  Audio server: reusing existing service on {AUDIO_HOST}:{AUDIO_PORT}")
elif AUDIO_AUTO_SERVE_REQUESTED and AUDIO_BASE_URL:
    print("  Audio server: supplied base URL in use (auto-start unavailable)")
elif AUDIO_AUTO_SERVE_REQUESTED:
    print("  Audio server: requested but not running (see messages above)")
else:
    print("  Audio server: disabled")
if SPECTROGRAM_SERVER_STARTED:
    print(f"  Spectrogram server: auto-started on {SPECTROGRAM_HOST}:{SPECTROGRAM_PORT}")
elif SPECTROGRAM_SERVER_REUSED:
    print(f"  Spectrogram server: reusing existing service on {SPECTROGRAM_HOST}:{SPECTROGRAM_PORT}")
elif SPECTROGRAM_AUTO_SERVE_REQUESTED:
    print("  Spectrogram server: requested but not running (see messages above)")
else:
    print("  Spectrogram server: disabled")
print(f"  Spectrogram inline delivery: {'enabled' if INLINE_SPECTROGRAMS else 'disabled'}")
print(f"  Spectrogram image format: .{SPECTROGRAM_IMAGE_FORMAT}")
print(f"  Analysis parameters:")
print(f"    UMAP neighbors: {ANALYSIS_PARAMS.get('umap_n_neighbors', 10)}")
print(f"    UMAP min_dist: {ANALYSIS_PARAMS.get('umap_min_dist', 0.0)}")
print(f"    UMAP n_components: {ANALYSIS_PARAMS.get('umap_n_components', 2)}")
print(f"    Point size: {ANALYSIS_PARAMS.get('point_size', 10)}")
print(f"    Point alpha: {ANALYSIS_PARAMS.get('point_alpha', 0.3)}")

# -----------------------------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------------------------

class AppState:
    """Manages the application state"""
    def __init__(self):
        self.original_embeddings = None
        self.original_meta = None
        self.current_embeddings = None
        self.current_meta = None
        self.current_indices = None
        self.mapper = None
        self.projection = None
        self.is_zoomed = False  # Simple flag instead of complex tracking
        self.current_umap_params = {  # Track current UMAP parameters
            'n_neighbors': ANALYSIS_PARAMS.get('umap_n_neighbors', 10),
            'min_dist': ANALYSIS_PARAMS.get('umap_min_dist', 0.0)
        }
        
state = AppState()
print("AppState initialized")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def time_to_color(time_str):
    """Convert time string (HH:MM) to a color representing day/night cycle"""
    try:
        if pd.isna(time_str) or time_str == '':
            return "#808080"  # Gray for unknown
        
        # Parse time
        parts = str(time_str).split(':')
        if len(parts) < 2:
            return "#808080"
        
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        
        # Convert to fraction of day (0-1)
        time_fraction = (hour + minute/60) / 24
        
        # Create a color gradient for day/night cycle
        # Night (0:00-4:00): Dark blue to purple
        # Dawn (4:00-8:00): Purple to orange
        # Day (8:00-16:00): Orange to yellow to light blue
        # Dusk (16:00-20:00): Light blue to purple
        # Night (20:00-24:00): Purple to dark blue
        
        if hour < 4:  # Night
            hue = 0.65  # Blue
            saturation = 0.8
            lightness = 0.2 + (hour/4) * 0.2
        elif hour < 8:  # Dawn
            hue = 0.75 - ((hour-4)/4) * 0.15  # Purple to orange
            saturation = 0.7
            lightness = 0.3 + ((hour-4)/4) * 0.3
        elif hour < 12:  # Morning
            hue = 0.15 - ((hour-8)/4) * 0.05  # Orange to yellow
            saturation = 0.6
            lightness = 0.6
        elif hour < 16:  # Afternoon
            hue = 0.10 + ((hour-12)/4) * 0.45  # Yellow to light blue
            saturation = 0.5
            lightness = 0.65
        elif hour < 20:  # Dusk
            hue = 0.55 + ((hour-16)/4) * 0.20  # Light blue to purple
            saturation = 0.6
            lightness = 0.5 - ((hour-16)/4) * 0.2
        else:  # Night
            hue = 0.75 - ((hour-20)/4) * 0.10  # Purple to blue
            saturation = 0.8
            lightness = 0.3 - ((hour-20)/4) * 0.1
            
        # Convert HSL to RGB to hex
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
        
        return hex_color
        
    except:
        return "#808080"  # Gray for any error

def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    """Convert hex color to an RGB tuple."""

    color = color.lstrip('#')
    return tuple(int(color[i:i+2], 16) for i in range(0, 6, 2))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert an RGB tuple to a hex color string."""

    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def _interpolate_color(start_color: str, end_color: str, fraction: float) -> str:
    """Linearly interpolate between two hex colors."""

    start_rgb = _hex_to_rgb(start_color)
    end_rgb = _hex_to_rgb(end_color)
    interpolated = tuple(
        int(round(s + (e - s) * fraction)) for s, e in zip(start_rgb, end_rgb)
    )
    return _rgb_to_hex(interpolated)


def _gradient_by_series(values: pd.Series) -> list[str]:
    """Return gradient colors (yellow→green→blue) for a numeric Series."""

    numeric_values = pd.to_numeric(values, errors='coerce')
    valid = numeric_values.notna()

    if not valid.any():
        return ["#999999"] * len(values)

    min_val = float(numeric_values[valid].min())
    max_val = float(numeric_values[valid].max())

    if np.isclose(max_val, min_val):
        normalized = pd.Series([0.5] * len(values))
    else:
        normalized = (numeric_values - min_val) / (max_val - min_val)

    colors: list[str] = []
    for value, is_valid in zip(normalized, valid):
        if not is_valid:
            colors.append("#999999")
            continue

        val = float(np.clip(value, 0.0, 1.0))
        if val <= 0.5:
            local_fraction = val / 0.5 if val > 0 else 0.0
            colors.append(_interpolate_color('#ffff00', '#00ff00', local_fraction))
        else:
            local_fraction = (val - 0.5) / 0.5
            colors.append(_interpolate_color('#00ff00', '#0000ff', local_fraction))

    return colors


def lonlat_to_web_mercator(lon_values, lat_values):
    """Convert longitude/latitude pairs to Web Mercator coordinates."""
    lon_series = pd.Series(lon_values)
    lat_series = pd.Series(lat_values)
    lon_arr = pd.to_numeric(lon_series, errors='coerce').to_numpy(dtype=float)
    lat_arr = pd.to_numeric(lat_series, errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(lon_arr) & np.isfinite(lat_arr)

    x = np.full(lon_arr.shape, np.nan, dtype=float)
    y = np.full(lat_arr.shape, np.nan, dtype=float)
    if valid.any():
        k = 6378137.0
        lon_rad = np.deg2rad(lon_arr[valid])
        lat_rad = np.deg2rad(lat_arr[valid])
        x[valid] = k * lon_rad
        y[valid] = k * np.log(np.tan((np.pi / 4.0) + (lat_rad / 2.0)))
    return x, y


def web_mercator_to_lonlat(x_values, y_values):
    """Convert Web Mercator coordinates back to longitude/latitude pairs."""
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    lon = (x_arr / 6378137.0) * (180.0 / np.pi)
    lat = (2.0 * np.arctan(np.exp(y_arr / 6378137.0)) - (np.pi / 2.0)) * (180.0 / np.pi)
    return lon, lat


def latitude_to_colors(latitudes: pd.Series) -> list[str]:
    """Return gradient colors (yellow→green→blue) based on latitude values."""

    return _gradient_by_series(latitudes)


def longitude_to_colors(longitudes: pd.Series) -> list[str]:
    """Return gradient colors (yellow→green→blue) based on longitude values."""

    return _gradient_by_series(longitudes)


def _clean_region_color(value: Any, fallback: str = REGION_DEFAULT_COLOR) -> str:
    """Return a sanitized hex color (#RRGGBB) or the provided fallback."""

    try:
        text = str(value).strip()
    except Exception:
        text = ""
    if not text:
        return fallback
    if text.startswith("#"):
        text = text.lstrip("#")
    if len(text) not in (3, 6):
        return fallback
    try:
        int(text, 16)
    except ValueError:
        return fallback
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    return f"#{text.lower()}"


def _clean_region_name(name_value: Any, fallback: str) -> str:
    """Return a trimmed region name or the provided fallback if blank."""

    try:
        text = str(name_value).strip()
    except Exception:
        text = ""
    return text or fallback


def load_region_polygons(path: Path) -> list[dict[str, Any]]:
    """Load saved region polygons for the current species."""

    if not path.exists():
        return []

    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"  Warning: failed to read region file {path}: {exc}")
        return []

    if not isinstance(payload, list):
        print(f"  Warning: region file {path} is not a list.")
        return []

    regions: list[dict[str, Any]] = []
    for region in payload:
        if not isinstance(region, dict):
            continue
        lon_values = region.get("lon") or region.get("lons") or []
        lat_values = region.get("lat") or region.get("lats") or []
        try:
            lon_list = [float(val) for val in lon_values]
            lat_list = [float(val) for val in lat_values]
        except Exception:
            continue
        if len(lon_list) < 3 or len(lat_list) < 3 or len(lon_list) != len(lat_list):
            continue
        default_name = f"Region {len(regions) + 1}"
        regions.append({
            "lon": lon_list,
            "lat": lat_list,
            "name": _clean_region_name(region.get("name") or region.get("label"), default_name),
            "color": _clean_region_color(region.get("color"), REGION_DEFAULT_COLOR),
        })

    return regions


def save_region_polygons(
    xs_list: list[Any],
    ys_list: list[Any],
    names_list: Optional[list[Any]] = None,
    colors_list: Optional[list[Any]] = None,
    *,
    destination: Path,
) -> Optional[Path]:
    """Persist drawn regions to disk as lon/lat JSON, including region names and colors."""

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"  ERROR: could not prepare region directory {destination.parent}: {exc}")
        return None

    regions: list[dict[str, Any]] = []
    count = min(len(xs_list), len(ys_list))

    for idx in range(count):
        xs = xs_list[idx]
        ys = ys_list[idx]
        if xs is None or ys is None:
            continue
        try:
            xs_arr = np.asarray(xs, dtype=float)
            ys_arr = np.asarray(ys, dtype=float)
        except Exception:
            continue
        if xs_arr.size < 3 or ys_arr.size < 3 or xs_arr.size != ys_arr.size:
            continue

        lon, lat = web_mercator_to_lonlat(xs_arr, ys_arr)
        raw_name = None
        if names_list is not None and idx < len(names_list):
            raw_name = names_list[idx]
        region_name = _clean_region_name(
            raw_name,
            fallback=f"Region {len(regions) + 1}",
        )
        raw_color = None
        if colors_list is not None and idx < len(colors_list):
            raw_color = colors_list[idx]
        region_color = _clean_region_color(raw_color, REGION_DEFAULT_COLOR)
        regions.append(
            {
                "lon": [float(val) for val in lon],
                "lat": [float(val) for val in lat],
                "name": region_name,
                "color": region_color,
            }
        )

    if not regions:
        print("  No valid regions to save.")
        return None

    try:
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(regions, handle, indent=2)
    except Exception as exc:
        print(f"  ERROR: failed to write regions to {destination}: {exc}")
        return None

    return destination


def update_stats_display(metadata, stats_div, zoom_level, date_slider=None, source=None):
    """Update the statistics display with UMAP and selection info"""
    dates = pd.to_datetime(metadata['date'], errors='coerce')
    valid_dates = dates[dates.notna()]
    
    # Season counts for UMAP data
    months = dates.dt.month
    spring_count = months.isin([4, 5]).sum()
    summer_count = months.isin([6, 7, 8]).sum()
    autumn_count = months.isin([9, 10, 11]).sum()
    winter_count = months.isin([12, 1, 2, 3]).sum()
    
    # Date range of UMAP data
    if len(valid_dates) > 0:
        date_min = valid_dates.min().strftime('%Y-%m-%d')
        date_max = valid_dates.max().strftime('%Y-%m-%d')
        date_range_str = f"{date_min} to {date_max}"
    else:
        date_range_str = "No valid dates"
    
    # Get currently visible points if source is provided
    visible_info = ""
    if source is not None and 'alpha' in source.data:
        visible_count = sum(1 for a in source.data['alpha'] if a > 0)
        visible_info = f" (Visible: {visible_count})"
    
    # Get date slider range if provided
    slider_info = ""
    if date_slider is not None:
        try:
            start_date = pd.Timestamp(date_slider.value[0], unit='ms').strftime('%Y-%m-%d')
            end_date = pd.Timestamp(date_slider.value[1], unit='ms').strftime('%Y-%m-%d')
            slider_info = f"<br><b>Date Filter:</b> {start_date} to {end_date}"
        except:
            pass
        
    # Include species name in the display
    species_info = f"{config['species']['scientific_name']}"
    if config['species'].get('common_name'):
        species_info += f" ({config['species']['common_name']})"
    
    stats_html = f"""
    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #dee2e6;">
        <b>Current UMAP Statistics</b> (Zoom Level: {zoom_level})<br>
        <div style="display: flex; gap: 20px; margin-top: 8px;">
            <div><b>Total Points:</b> {len(metadata)}{visible_info}</div>
            <div><b>UMAP Date Range:</b> {date_range_str}</div>
            <div style="display: flex; gap: 10px;">
                <span style="color: #88CC88;">Spring: {spring_count}</span>
                <span style="color: #F6C667;">Summer: {summer_count}</span>
                <span style="color: #D98859;">Autumn: {autumn_count}</span>
                <span style="color: #6BAED6;">Winter: {winter_count}</span>
            </div>
        </div>
        {slider_info}
    </div>
    """
    stats_div.text = stats_html

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def load_data():
    """Load embeddings and metadata"""
    print("\n" + "-" * 40)
    print("LOADING DATA...")
    
    try:
        # Check if files exist
        if not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        if not METADATA_FILE.exists():
            raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
        
        # Load embeddings
        print("  Loading embeddings CSV...")
        embeddings_df = pd.read_csv(EMBEDDINGS_FILE)
        print(f"  Loaded {len(embeddings_df)} rows")
        
        print("  Parsing embedding strings...")
        embeddings = embeddings_df['embedding'].apply(lambda x: np.fromstring(x, sep=',')).values
        embeddings_array = np.stack(embeddings)
        print(f"  Embeddings array shape: {embeddings_array.shape}")
        
        # Load metadata
        print("  Loading metadata CSV...")
        metadata = pd.read_csv(METADATA_FILE)
        print(f"  Metadata shape: {metadata.shape}")
        
        # Add audio URLs based on clip filenames
        if AUDIO_BASE_URL:
            if 'clip_file' not in metadata.columns:
                # If no clip_file column, create from xcid and clip_index
                metadata['audio_url'] = metadata.apply(
                    lambda row: f"{AUDIO_BASE_URL}/{SPECIES_SLUG}_{row['xcid']}_{row['clip_index']:02d}.wav",
                    axis=1
                )
            else:
                # Use the actual filename from clip_file
                metadata['audio_url'] = metadata['clip_file'].apply(
                    lambda f: f"{AUDIO_BASE_URL}/{Path(f).name}"
                )
        else:
            metadata['audio_url'] = [''] * len(metadata)

        # Resolve spectrogram URLs if images exist on disk
        def _clip_basename(row: pd.Series) -> str:
            try:
                if 'clip_file' in metadata.columns and pd.notna(row['clip_file']):
                    return Path(row['clip_file']).stem
                clip_index = int(row['clip_index'])
            except Exception:
                clip_index = int(row.get('clip_index', 0))
            return f"{SPECIES_SLUG}_{row['xcid']}_{clip_index:02d}"

        spectrogram_urls: list[str] = []
        spectrogram_exists: list[bool] = []
        spectrogram_data_uri: list[str] = []
        missing_count = 0
        inline_attempts = 0
        inline_failures = 0

        def _inline_spectrogram(path: Path) -> str:
            """Return a base64 data URI for the spectrogram image."""
            nonlocal inline_failures
            try:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                return f"data:image/{SPECTROGRAM_IMAGE_FORMAT};base64,{encoded}"
            except FileNotFoundError:
                return ""
            except Exception as exc:
                inline_failures += 1
                print(f"  Warning: failed to inline {path.name}: {exc}")
                return ""

        for _, meta_row in metadata.iterrows():
            base_name = _clip_basename(meta_row)
            filename = f"{base_name}.{SPECTROGRAM_IMAGE_FORMAT}"
            image_path = SPECTROGRAMS_DIR / filename
            if image_path.exists():
                url = (
                    f"{SPECTROGRAM_BASE_URL}/{filename}"
                    if SPECTROGRAM_BASE_URL
                    else ""
                )
                spectrogram_urls.append(url)
                spectrogram_exists.append(True)
                if INLINE_SPECTROGRAMS:
                    inline_attempts += 1
                    spectrogram_data_uri.append(_inline_spectrogram(image_path))
                else:
                    spectrogram_data_uri.append("")
            else:
                spectrogram_urls.append("")
                spectrogram_exists.append(False)
                missing_count += 1
                spectrogram_data_uri.append("")

        metadata['spectrogram_url'] = spectrogram_urls
        metadata['spectrogram_exists'] = spectrogram_exists
        metadata['spectrogram_data_uri'] = spectrogram_data_uri
        print(
            f"  Spectrogram availability: {len(metadata) - missing_count}/{len(metadata)} "
            f"(directory {SPECTROGRAMS_DIR})"
        )
        if INLINE_SPECTROGRAMS:
            print(
                f"  Spectrogram inline payloads: {inline_attempts - inline_failures}/"
                f"{inline_attempts} embedded"
            )
        
        # Store in state
        state.original_embeddings = embeddings_array
        state.original_meta = metadata.copy()
        state.current_embeddings = embeddings_array
        state.current_meta = metadata.copy()
        state.current_indices = np.arange(len(embeddings_array))
        
        print(f"Successfully loaded {len(embeddings_array)} embeddings")
        return embeddings_array, metadata
        
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        traceback.print_exc()
        raise

def compute_initial_umap(embeddings_array, metadata, n_neighbors=None, min_dist=None):
    """Compute initial UMAP projection"""
    print("\n" + "-" * 40)
    print("COMPUTING INITIAL UMAP...")
    
    try:
        # Only compute if we haven't already
        if state.projection is None:
            # UMAP projection with parameters from config or user input
            print("  Fitting UMAP...")
            
            # Use provided parameters or fall back to config/defaults
            if n_neighbors is None:
                n_neighbors = ANALYSIS_PARAMS.get("umap_n_neighbors", 10)
            if min_dist is None:
                min_dist = ANALYSIS_PARAMS.get("umap_min_dist", 0.0)
            n_components = ANALYSIS_PARAMS.get("umap_n_components", 2)
            
            print(f"    n_neighbors: {n_neighbors}")
            print(f"    min_dist: {min_dist}")
            print(f"    n_components: {n_components}")
            
            state.mapper = umap.UMAP(
                n_neighbors=n_neighbors, 
                min_dist=min_dist, 
                n_components=n_components, 
                n_jobs=1
            )
            state.projection = state.mapper.fit_transform(embeddings_array)
            print(f"  UMAP projection shape: {state.projection.shape}")
            print("UMAP computation complete")
        else:
            print("  Using existing UMAP projection")
            
        return state.projection, metadata
        
    except Exception as e:
        print(f"  ERROR computing UMAP: {e}")
        traceback.print_exc()
        raise


# -----------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------

def prepare_hover_data(
    metadata: pd.DataFrame,
    projection: np.ndarray,
    point_alpha: Optional[float] = None,
) -> dict[str, list]:
    """Prepare data for hover tooltips and interactions"""
    print("\n" + "-" * 40)
    print("PREPARING HOVER DATA...")

    try:
        effective_alpha = (
            float(point_alpha)
            if point_alpha is not None
            else float(ANALYSIS_PARAMS.get("point_alpha", 0.3))
        )
        effective_alpha = max(min(effective_alpha, 1.0), 0.01)

        # Parse dates
        print("  Parsing dates...")
        dates = pd.to_datetime(pd.Series(metadata['date']), errors='coerce')
        months = dates.dt.month
        valid = dates.notna()
        ts_ms = (dates.astype('int64', copy=False) // 10**6).astype('float64')
        ts_ms[~valid] = np.nan
        
        # Define seasons
        print("  Assigning seasons...")
        SPRING_MONTHS = {4, 5}
        SUMMER_MONTHS = {6, 7, 8}
        AUTUMN_MONTHS = {9, 10, 11}
        WINTER_MONTHS = {12, 1, 2, 3}
        
        is_spring = months.isin(SPRING_MONTHS).fillna(False)
        is_summer = months.isin(SUMMER_MONTHS).fillna(False)
        is_autumn = months.isin(AUTUMN_MONTHS).fillna(False)
        is_winter = months.isin(WINTER_MONTHS).fillna(False)
        
        # Assign season labels
        season_arr = np.array(['Unknown'] * len(metadata), dtype=object)
        season_arr[is_spring] = 'Spring'
        season_arr[is_summer] = 'Summer'
        season_arr[is_autumn] = 'Autumn'
        season_arr[is_winter] = 'Winter'
        
        # Add to metadata for consistency
        metadata['season'] = season_arr
        
        # Season visibility flags        
        season_on = [True] * len(metadata)

        # Convert coordinates to Web Mercator for map
        print("  Converting coordinates to Web Mercator...")
        def lonlat_to_mercator_arrays(lon_deg, lat_deg):
            lon = pd.to_numeric(lon_deg, errors='coerce')
            lat = pd.to_numeric(lat_deg, errors='coerce')
            k = 6378137.0
            x = lon * (np.pi/180.0) * k
            y = np.log(np.tan((np.pi/4.0) + (lat * (np.pi/180.0) / 2.0))) * k
            return x, y
        
        x3857, y3857 = lonlat_to_mercator_arrays(metadata['lon'], metadata['lat'])
        
        # Color palettes
        print("  Assigning colors...")
        season_palette = {
            "Spring": "#88CC88",
            "Summer": "#F6C667", 
            "Autumn": "#D98859",
            "Winter": "#6BAED6",
            "Unknown": "#999999"
        }
        
        # Season colors
        season_colors = [season_palette.get(s, "#999999") for s in season_arr]
        
        # Sex colors
        sex_palette = {"M": "#4B8BBE", "F": "#F07C7C", "?": "#A0A0A0"}
        sex_colors = [sex_palette.get(str(s).upper(), "#A0A0A0") for s in metadata['sex']]
        sex_labels = [str(s) if pd.notna(s) else "?" for s in metadata['sex']]
        
        # Type colors (song, call, etc.)
        type_palette = {
            "S": "#FF6B6B",  # Song - Red
            "C": "#4ECDC4",  # Call - Teal
            "F": "#FFE66D",  # Flight - Yellow
            "A": "#95E77E",  # Alarm - Green
            "?": "#A0A0A0"   # Unknown - Gray
        }
        type_colors = [type_palette.get(str(t).upper(), "#A0A0A0") for t in metadata['type']]
        type_labels = [str(t) if pd.notna(t) else "?" for t in metadata['type']]
        
        # Time of day colors (continuous gradient)
        time_colors = [time_to_color(t) for t in metadata['time']]

        # Latitude gradient colors (yellow -> green -> blue)
        latitude_colors = latitude_to_colors(metadata['lat'])
        longitude_colors = longitude_to_colors(metadata['lon'])
        
        # Parse time to hours for filtering
        time_hours = []
        for t in metadata['time']:
            try:
                if pd.notna(t) and t != '':
                    parts = str(t).split(':')
                    hour = float(parts[0]) + (float(parts[1])/60 if len(parts) > 1 else 0)
                    time_hours.append(hour)
                else:
                    time_hours.append(-1)  # Invalid time
            except:
                time_hours.append(-1)
        
        # Initial alpha (visibility)
        earliest_date = dates[valid].min() if valid.any() else pd.Timestamp("2000-01-01")
        alpha = np.where(valid & (dates >= earliest_date), effective_alpha, 0.0)
        
        # Clip indices (needed for saving/loading selection groups)
        if 'clip_index' in metadata.columns:
            clip_index_values = [
                normalize_clip_index(value)
                for value in metadata['clip_index'].tolist()
            ]
        else:
            clip_index_values = [None] * len(metadata)

        # Build data dictionary
        print("  Building data dictionary...")
        data: dict[str, list] = {
            'x': projection[:, 0].tolist(),
            'y': projection[:, 1].tolist(),
            'xcid': metadata['xcid'].tolist(),
            'sex': sex_labels,
            'type': type_labels,
            'cnt': metadata['country'].tolist(),
            'lat': metadata['lat'].tolist(),
            'lon': metadata['lon'].tolist(),
            'x3857': x3857.tolist(),
            'y3857': y3857.tolist(),
            'alt': metadata['alt'].tolist(),
            'date': metadata['date'].tolist(),
            'time': metadata['time'].tolist(),
            'time_hour': time_hours,
            'also': metadata['also'].tolist(),
            'rmk': metadata['remarks'].tolist(),
            'month': months.tolist(),
            'ts': ts_ms.tolist(),
            'valid_date': valid.tolist(),
            'hdbscan_on': [True] * len(metadata),
            'dedupe_on': [True] * len(metadata),
            'audio_url': metadata['audio_url'].tolist() if 'audio_url' in metadata else [''] * len(metadata),
            'spectrogram_url': metadata['spectrogram_url'].tolist() if 'spectrogram_url' in metadata else [''] * len(metadata),
            'spectrogram_exists': metadata['spectrogram_exists'].tolist() if 'spectrogram_exists' in metadata else [False] * len(metadata),
            'spectrogram_data_uri': metadata['spectrogram_data_uri'].tolist() if 'spectrogram_data_uri' in metadata else [''] * len(metadata),
            'season': season_arr.tolist(),
            'season_on': season_on,
            'season_color': season_colors,
            'sex_color': sex_colors,
            'type_color': type_colors,
            'time_color': time_colors,
            'lat_color': latitude_colors,
            'lon_color': longitude_colors,
            'region_label': ["Unassigned"] * len(metadata),
            'region_color': [REGION_UNASSIGNED_COLOR] * len(metadata),
            'region_on': [True] * len(metadata),
            'region_id': [-1] * len(metadata),
            'selection_group': [-1] * len(metadata),
            'selection_group_label': [-1] * len(metadata),
            'selection_color': [SELECTION_UNASSIGNED_COLOR] * len(metadata),
            'selection_on': [True] * len(metadata),
            'active_color': season_colors,  # Start with season colors
            'alpha': alpha.tolist(),
            'alpha_base': alpha.tolist(),
            'sex_on': [True] * len(metadata),
            'type_on': [True] * len(metadata),
            'time_on': [True] * len(metadata),
            'original_index': np.arange(len(metadata)).tolist(),
            # fields used to draw highlight outlines on hover over playlist items
            'hl_alpha': [0.0] * len(metadata),
            'hl_width': [0.0] * len(metadata),
            'clip_index': clip_index_values,
        }

        print(f"Data dictionary created with {len(data)} keys")
        return data
        
    except Exception as e:
        print(f"  ERROR preparing data: {e}")
        traceback.print_exc()
        raise

# -----------------------------------------------------------------------------
# CREATE PLOTS
# -----------------------------------------------------------------------------

def create_umap_plot(source):
    """Create the main UMAP scatter plot"""
    print("\n" + "-" * 40)
    print("CREATING UMAP PLOT...")
    
    point_size = ANALYSIS_PARAMS.get("point_size", 10)
    point_alpha = ANALYSIS_PARAMS.get("point_alpha", 0.3)
    
    p = figure(
        title="UMAP Projection",
        width=600, height=600,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        background_fill_color="white",
        border_fill_color="#ffe88c"
    )
    
    # Add selection tools (box default plus lasso/polygon for freeform filtering)
    box_select = BoxSelectTool()
    lasso_select = LassoSelectTool()
    poly_select = PolySelectTool()
    p.add_tools(box_select, lasso_select, poly_select)
    p.toolbar.active_drag = box_select
    
    # Filter selections to only include visible points
    selection_filter_callback = CustomJS(args=dict(source=source), code="""
        const indices = source.selected.indices;
        if (!indices || indices.length === 0) {
            return;
        }

        const alpha = source.data['alpha'];
        const filtered = [];
        
        for (let i of indices) {
            if (alpha[i] > 0) {
                filtered.push(i);
            }
        }
        
        if (filtered.length !== indices.length) {
            source.selected.indices = filtered;
        }
    """)
    source.selected.js_on_change('indices', selection_filter_callback)
    
    # Create a CDSView with boolean filter based on alpha
    view_filter = BooleanFilter(booleans=[a > 0 for a in source.data['alpha']])
    view = CDSView(filter=view_filter)
    
    # Single scatter renderer using the view
    # Use dynamic line properties to allow highlighting individual points.
    # The 'hl_alpha' and 'hl_width' columns will control the outline of each point.
    scatter = p.scatter('x', 'y', source=source, view=view,
                        size=point_size,
                        fill_color={'field':'active_color'},
                        line_color='black',
                        line_alpha={'field':'hl_alpha'},
                        line_width={'field':'hl_width'},
                        alpha='alpha',
                        hover_line_color="black",
                        hover_alpha=1.0,
                        hover_line_width=1.5)
        
    # Hover tool - will only work on visible points (alpha > 0)
    hover = HoverTool(
        renderers=[scatter],
        tooltips=[
            ("xcid", "@xcid"),
            ("season", "@season"),
            ("date", "@date"),
            ("time", "@time"),
            ("sex", "@sex"),
            ("type", "@type"),
            ("lat, lon", "@lat, @lon")
        ]
    )
    
    p.add_tools(hover)
    p.add_tools(TapTool())
    
    print("  UMAP plot created")
    return p, hover, view

def create_map_plot(source):
    """Create the geographic map plot"""
    print("\n" + "-" * 40)
    print("CREATING MAP PLOT...")
    
    # Europe bounds
    lon_min, lon_max = -25.0, 40.0
    lat_min, lat_max = 34.0, 72.0
    
    def lonlat_to_mercator(lon, lat):
        k = 6378137.0
        x = lon * (np.pi/180.0) * k
        y = np.log(np.tan((np.pi/4.0) + (lat * (np.pi/180.0) / 2.0))) * k
        return x, y
    
    x0, y0 = lonlat_to_mercator(lon_min, lat_min)
    x1, y1 = lonlat_to_mercator(lon_max, lat_max)
    
    map_fig = figure(
        title="Recording locations (Europe)",
        width=600, height=600,
        x_axis_label="Longitude", y_axis_label="Latitude",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        match_aspect=True,
        background_fill_color="#ffe88c",
        border_fill_color="#ffe88c"
    )

    # Mirror selection tools from the UMAP plot for consistent interactions
    map_box_select = BoxSelectTool()
    map_lasso_select = LassoSelectTool()
    map_poly_select = PolySelectTool()
    map_fig.add_tools(map_box_select, map_lasso_select, map_poly_select)
    map_fig.toolbar.active_drag = map_box_select

    try:
        map_fig.add_tile("CartoDB Positron", retina=True)
    except:
        pass
    
    map_fig.x_range.start, map_fig.x_range.end = x0, x1
    map_fig.y_range.start, map_fig.y_range.end = y0, y1
    
    hv_line_width = 1.5
    
    # Create a CDSView with boolean filter based on alpha
    view_filter = BooleanFilter(booleans=[a > 0 for a in source.data['alpha']])
    view = CDSView(filter=view_filter)
    
    # Single scatter renderer using the view
    map_scatter = map_fig.scatter('x3857','y3857', source=source, view=view,
                                 size=8,  # Slightly bigger points on map
                                 fill_color={'field':'active_color'},
                                 line_color='black',
                                 line_alpha={'field':'hl_alpha'},
                                 line_width={'field':'hl_width'},
                                 alpha='alpha',
                                 hover_line_color="black",
                                 hover_alpha=1.0,
                                 hover_line_width=1.5)
    
    # Store view reference

    # Hover tool
    hover_map = HoverTool(
        renderers=[map_scatter],
        tooltips=[
            ("xcid", "@xcid"),
            ("season", "@season"),
            ("date", "@date"),
            ("time", "@time"),
            ("lat, lon", "@lat, @lon"),
            ("alt", "@alt")
        ]
    )
        
    map_fig.add_tools(hover_map)
    map_fig.add_tools(TapTool())
    
    print("  Map plot created")
    return map_fig, hover_map, view # Return views too

# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------

def create_app():
    """Create the Bokeh application"""
    print("\n" + "=" * 80)
    print("CREATING BOKEH APPLICATION")
    print("=" * 80)
    
    try:
        # Load data
        embeddings_array, metadata = load_data()
        projection, metadata = compute_initial_umap(embeddings_array, metadata)
        
        # Prepare data
        point_alpha_default = max(
            min(float(ANALYSIS_PARAMS.get('point_alpha', 0.3)), 1.0), 0.01
        )
        data = prepare_hover_data(
            metadata, projection, point_alpha=point_alpha_default
        )
        source = ColumnDataSource(data=data)
        # Assign a name to the data source so it can be retrieved from JavaScript.
        source.name = 'source'
        print(f"\nColumnDataSource created with {len(source.data['x'])} points")
        
        # Create plots - now they return season views
        umap_plot, umap_hover, umap_view = create_umap_plot(source)
        map_plot, map_hover, map_view = create_map_plot(source)

        active_region_key = next(iter(REGION_VARIANTS))
        region_contexts: dict[str, dict[str, Any]] = {}
        region_widget_groups: list[list[Any]] = []
        region_label_to_key: dict[str, str] = {
            meta.get("label", key.title()): key for key, meta in REGION_VARIANTS.items()
        }

        def _make_region_context(key: str, label: str) -> dict[str, Any]:
            """Create ColumnDataSources, renderer, and widgets for a region variant."""

            source = ColumnDataSource(
                data={
                    'xs': [],
                    'ys': [],
                    'fill_alpha': [],
                    'name': [],
                    'color': [],
                }
            )
            source.name = f"region_source_{key}"
            renderer = map_plot.patches(
                'xs',
                'ys',
                source=source,
                fill_color='color',
                fill_alpha='fill_alpha',
                line_color='color',
                line_alpha=0.9,
                line_width=2,
                visible=False,
                name=f"region_renderer_{key}",
            )
            draft_source = ColumnDataSource(data={'x': [], 'y': []})
            draft_line = map_plot.line(
                'x',
                'y',
                source=draft_source,
                line_color="#222222",
                line_dash="dashed",
                line_width=2,
                alpha=0.7,
                visible=False,
            )
            draft_points = map_plot.circle(
                'x',
                'y',
                source=draft_source,
                size=6,
                color="#222222",
                alpha=0.9,
                visible=False,
            )

            saved_regions = load_region_polygons(region_file_path(key))
            if saved_regions:
                xs_saved: list[list[float]] = []
                ys_saved: list[list[float]] = []
                fill_alpha_saved: list[float] = []
                names_saved: list[str] = []
                colors_saved: list[str] = []
                for idx, region in enumerate(saved_regions):
                    lon_vals = region.get("lon", [])
                    lat_vals = region.get("lat", [])
                    if len(lon_vals) < 3 or len(lat_vals) < 3 or len(lon_vals) != len(lat_vals):
                        continue
                    x_vals, y_vals = lonlat_to_web_mercator(lon_vals, lat_vals)
                    xs_saved.append([float(val) for val in x_vals])
                    ys_saved.append([float(val) for val in y_vals])
                    fill_alpha_saved.append(0.2)
                    default_name = f"Region {len(names_saved) + 1}"
                    names_saved.append(
                        _clean_region_name(region.get("name") or region.get("label"), default_name)
                    )
                    colors_saved.append(_clean_region_color(region.get("color"), REGION_DEFAULT_COLOR))
                source.data = {
                    'xs': xs_saved,
                    'ys': ys_saved,
                    'fill_alpha': fill_alpha_saved,
                    'name': names_saved,
                    'color': colors_saved,
                }

            help_div = Div(
                text=(
                    f"<b>{label}:</b> use the polygon draw tool on the map to outline areas. "
                    "Name each region and pick a hex color when closing the polygon; save to persist them, "
                    "load to restore, and clear to start fresh."
                ),
                width=300,
                visible=False,
                styles={'font-size': '11px', 'color': '#444'}
            )
            draw_toggle = Toggle(
                label=f"Draw {label.lower()}",
                active=False,
                button_type="primary",
                width=160,
                visible=False,
                name=f"region_draw_{key}",
            )
            cancel_btn = Button(
                label="Cancel draft",
                button_type="default",
                width=160,
                visible=False,
                name=f"region_cancel_{key}",
            )
            status_div = Div(
                text="<i>No regions loaded.</i>",
                width=300,
                visible=False,
                styles={'font-size': '11px', 'color': '#666'},
                name=f"region_status_{key}",
            )
            checks = CheckboxGroup(
                labels=["Unassigned (0)"],
                active=[0],
                visible=False,
                name=f"region_checks_{key}",
            )
            save_btn = Button(
                label="Save regions",
                button_type="success",
                width=180,
                visible=False,
                name=f"region_save_{key}",
            )
            load_btn = Button(
                label="Load regions",
                button_type="default",
                width=180,
                visible=False,
                name=f"region_load_{key}",
            )
            clear_btn = Button(
                label="Clear regions",
                button_type="default",
                width=180,
                visible=False,
                name=f"region_clear_{key}",
            )

            widgets = [
                help_div,
                draw_toggle,
                cancel_btn,
                status_div,
                save_btn,
                load_btn,
                clear_btn,
                checks,
            ]
            region_widget_groups.append(widgets)

            ctx = {
                "key": key,
                "label": label,
                "file_path": region_file_path(key),
                "source": source,
                "renderer": renderer,
                "draft_source": draft_source,
                "draft_line": draft_line,
                "draft_points": draft_points,
                "help_div": help_div,
                "draw_toggle": draw_toggle,
                "cancel_btn": cancel_btn,
                "status_div": status_div,
                "checks": checks,
                "save_btn": save_btn,
                "load_btn": load_btn,
                "clear_btn": clear_btn,
                "widgets": widgets,
            }
            region_contexts[key] = ctx
            return ctx

        for region_key, meta in REGION_VARIANTS.items():
            _make_region_context(region_key, meta.get("label", region_key.title()))

        # Overlays for geographic shift visualisation
        north_shift_source = ColumnDataSource(data={
            'x': [],
            'y': [],
            'year': [],
            'lat': [],
            'lon': [],
            'count': [],
        })
        south_shift_source = ColumnDataSource(data={
            'x': [],
            'y': [],
            'year': [],
            'lat': [],
            'lon': [],
            'count': [],
        })
        north_arrow_source = ColumnDataSource(data={
            'x_start': [],
            'y_start': [],
            'x_end': [],
            'y_end': [],
        })
        south_arrow_source = ColumnDataSource(data={
            'x_start': [],
            'y_start': [],
            'x_end': [],
            'y_end': [],
        })
        north_line_source = ColumnDataSource(data={
            'year': [],
            'lat': [],
        })
        south_line_source = ColumnDataSource(data={
            'year': [],
            'lat': [],
        })

        north_shift_renderer = map_plot.cross(
            'x',
            'y',
            size=18,
            line_color=GEOSHIFT_NORTH_COLOR,
            line_width=2,
            source=north_shift_source,
            alpha=0.9
        )
        south_shift_renderer = map_plot.cross(
            'x',
            'y',
            size=18,
            line_color=GEOSHIFT_SOUTH_COLOR,
            line_width=2,
            source=south_shift_source,
            alpha=0.9
        )

        north_shift_hover = HoverTool(
            renderers=[north_shift_renderer],
            tooltips=[
                ("Year", "@year"),
                ("North mean lat", "@lat{0.00}°"),
                ("North mean lon", "@lon{0.00}°"),
                ("Samples", "@count"),
            ]
        )
        south_shift_hover = HoverTool(
            renderers=[south_shift_renderer],
            tooltips=[
                ("Year", "@year"),
                ("South mean lat", "@lat{0.00}°"),
                ("South mean lon", "@lon{0.00}°"),
                ("Samples", "@count"),
            ]
        )
        map_plot.add_tools(north_shift_hover)
        map_plot.add_tools(south_shift_hover)

        north_shift_arrows = Arrow(
            end=NormalHead(size=6, line_color=GEOSHIFT_NORTH_COLOR, fill_color=GEOSHIFT_NORTH_COLOR),
            x_start='x_start',
            y_start='y_start',
            x_end='x_end',
            y_end='y_end',
            line_color=GEOSHIFT_NORTH_COLOR,
            line_width=1.5,
            source=north_arrow_source
        )
        south_shift_arrows = Arrow(
            end=NormalHead(size=6, line_color=GEOSHIFT_SOUTH_COLOR, fill_color=GEOSHIFT_SOUTH_COLOR),
            x_start='x_start',
            y_start='y_start',
            x_end='x_end',
            y_end='y_end',
            line_color=GEOSHIFT_SOUTH_COLOR,
            line_width=1.5,
            source=south_arrow_source
        )
        map_plot.add_layout(north_shift_arrows)
        map_plot.add_layout(south_shift_arrows)

        shared_lat_range = Range1d(0, 1)

        north_trend_fig = figure(
            title="North extreme mean latitude",
            width=320,
            height=220,
            toolbar_location=None,
            x_axis_label="Year",
            y_axis_label="Latitude (°)"
        )
        north_trend_fig.y_range = shared_lat_range
        north_trend_fig.line('year', 'lat', source=north_line_source, line_color=GEOSHIFT_NORTH_COLOR, line_width=2)
        north_trend_fig.circle('year', 'lat', source=north_line_source, size=8, color=GEOSHIFT_NORTH_COLOR)

        south_trend_fig = figure(
            title="South extreme mean latitude",
            width=320,
            height=220,
            toolbar_location=None,
            x_axis_label="Year",
            y_axis_label="Latitude (°)"
        )
        south_trend_fig.y_range = shared_lat_range
        south_trend_fig.line('year', 'lat', source=south_line_source, line_color=GEOSHIFT_SOUTH_COLOR, line_width=2)
        south_trend_fig.circle('year', 'lat', source=south_line_source, size=8, color=GEOSHIFT_SOUTH_COLOR)

        def refresh_alpha(
            data_override: Optional[dict[str, list[Any]]] = None,
            *,
            update_source: bool = True,
        ) -> dict[str, list[Any]]:
            """Recalculate alpha values based on current visibility flags."""

            data_dict = dict(source.data) if data_override is None else data_override
            alpha_base_raw = data_dict.get('alpha_base', [])
            alpha_base = list(alpha_base_raw)
            n = len(alpha_base)

            alpha_values = list(data_dict.get('alpha', []))
            if len(alpha_values) < n:
                alpha_values.extend([0.0] * (n - len(alpha_values)))
            elif len(alpha_values) > n:
                alpha_values = alpha_values[:n]

            def _normalize(values, default_value: bool) -> list[bool]:
                if values is None:
                    arr = [default_value] * n
                else:
                    arr = list(values)
                    if len(arr) < n:
                        arr.extend([default_value] * (n - len(arr)))
                    elif len(arr) > n:
                        arr = arr[:n]
                return [bool(item) for item in arr]

            hdbscan_on = _normalize(data_dict.get('hdbscan_on'), True)
            sex_on = _normalize(data_dict.get('sex_on'), True)
            type_on = _normalize(data_dict.get('type_on'), True)
            time_on = _normalize(data_dict.get('time_on'), True)
            season_on = _normalize(data_dict.get('season_on'), True)
            selection_on = _normalize(data_dict.get('selection_on'), True)
            dedupe_on = _normalize(data_dict.get('dedupe_on'), True)
            region_on = _normalize(data_dict.get('region_on'), True)

            for idx in range(n):
                if (
                    hdbscan_on[idx]
                    and sex_on[idx]
                    and type_on[idx]
                    and time_on[idx]
                    and season_on[idx]
                    and selection_on[idx]
                    and dedupe_on[idx]
                    and region_on[idx]
                ):
                    alpha_values[idx] = alpha_base[idx] if idx < len(alpha_base) else 0.0
                else:
                    alpha_values[idx] = 0.0

            data_dict['alpha'] = alpha_values
            visibility_flags = [value > 0 for value in alpha_values]
            if hasattr(umap_view, 'filter') and umap_view.filter is not None:
                umap_view.filter.booleans = list(visibility_flags)
            if hasattr(map_view, 'filter') and map_view.filter is not None:
                map_view.filter.booleans = list(visibility_flags)

            if update_source:
                source.data = data_dict
            return data_dict
        
        # --- CREATE ALL WIDGETS ---
        print("\n" + "-" * 40)
        print("CREATING WIDGETS...")
        
        # Statistics display
        stats_div = Div(width=1400, height=60,
                       styles={'margin-bottom': '10px'})
        update_stats_display(metadata, stats_div, 0)  # Initial stats
        
        # Zoom controls (should this be at top or bottom of widget order? what about with callback order?)
        zoom_button = Button(label="Zoom to Selection", button_type="success", width=150)
        reset_button = Button(label="Reset to Full Dataset", button_type="warning", width=150)
        zoom_status = Div(text="Viewing: Full dataset", width=400, height=30,
                         styles={'padding':'4px', 'background':'#f0f0f0', 'border-radius':'4px'})
        
        # Get unique values for categorical filters
        unique_seasons = sorted(set(source.data['season']))
        unique_sex = sorted(set(source.data['sex']))
        unique_type = sorted(set(source.data['type']))

        
        # --- CHECKBOXES and TIMERANGE SLIDER---
        # Season checkboxes
        season_checks = CheckboxGroup(
            labels=unique_seasons,
            active=list(range(len(unique_seasons))),  # All active by default
            visible=True,  # Hidden by default
            name="season_checks"
        )

        # Sex checkboxes
        sex_checks = CheckboxGroup(
            labels=unique_sex,
            active=list(range(len(unique_sex))),
            visible=False,  # Hidden by default
            name="sex_checks"
        )
        
        # Type checkboxes
        type_checks = CheckboxGroup(
            labels=unique_type,
            active=list(range(len(unique_type))),
            visible=False,  # Hidden by default
            name="type_checks"
        )

        selection_help_div = Div(
            text=(
                "<b>Selection groups:</b> use the box or lasso selection tools on either plot "
                "to highlight points, then click <i>Create group from selection</i> to save them. "
                "Select a group in the checklist and click <i>Save group to vocal types</i>, "
                "<i>Save group to dialects</i>, <i>Save group to annotate 1</i>, or "
                "<i>Save group to annotate 2</i> to export it. Use the "
                "<i>Load</i> buttons to recreate saved vocal type, dialect, or annotate groups from disk. "
                "Use the checkboxes to toggle visibility of saved groups."
            ),
            width=300,
            visible=False,
            styles={'font-size': '11px', 'color': '#444'}
        )
        selection_status_div = Div(
            text="<i>No active selection.</i>",
            width=300,
            visible=False,
            styles={'font-size': '11px', 'color': '#666'}
        )
        selection_create_btn = Button(
            label="Create group from selection",
            button_type="primary",
            width=220,
            visible=False,
            disabled=True,
        )
        selection_save_vocal_btn = Button(
            label="Save group to vocal types",
            button_type="success",
            width=220,
            visible=False,
            disabled=True,
        )
        selection_save_dialect_btn = Button(
            label="Save group to dialects",
            button_type="success",
            width=220,
            visible=False,
            disabled=True,
        )
        selection_save_annotate1_btn = Button(
            label="Save group to annotate 1",
            button_type="success",
            width=220,
            visible=False,
            disabled=True,
        )
        selection_save_annotate2_btn = Button(
            label="Save group to annotate 2",
            button_type="success",
            width=220,
            visible=False,
            disabled=True,
        )
        selection_load_vocal_btn = Button(
            label="Load vocal types",
            button_type="default",
            width=220,
            visible=False,
        )
        selection_load_dialect_btn = Button(
            label="Load dialects",
            button_type="default",
            width=220,
            visible=False,
        )
        selection_load_annotate1_btn = Button(
            label="Load annotate 1",
            button_type="default",
            width=220,
            visible=False,
        )
        selection_load_annotate2_btn = Button(
            label="Load annotate 2",
            button_type="default",
            width=220,
            visible=False,
        )
        description_request_source = ColumnDataSource(data={
            'kind': [],
            'description': [],
            'nonce': []
        })
        annotation_request_source = ColumnDataSource(data={
            'index': [],
            'group': [],
            'nonce': []
        })
        selection_checks = CheckboxGroup(
            labels=["Unassigned (0)"],
            active=[0],
            visible=False,
            name="selection_checks"
        )
        selection_clear_btn = Button(
            label="Clear selection groups",
            button_type="default",
            width=200,
            visible=False
        )

        # Time range slider (0-24 hours) (no check for len(unique times), necessary?( tested and does indeed cause the same bug, maybe fix later))
        time_range_slider = RangeSlider(
            start=0, end=24, value=(0, 24), step=0.5,
            title="Time of day (hours)",
            visible=False,  # Hidden by default
            width=300
        )
        
        # Color selector - expanded options
        color_select = Select(
            title="Color by",
            value="Season",
            options=[
                "Season",
                "HDBSCAN",
                "Sex",
                "Type",
                "Time of Day",
                "Latitude",
                "Longitude",
                "Region (course)",
                "Region (fine)",
                "Region (custom)",
                "Selection",
            ],
        )
        
        def _on_color_select_change(attr: str, old: Any, new: Any) -> None:
            region_key = _region_key_from_mode(new)
            if region_key:
                _set_active_region(region_key)
                apply_regions_from_source(region_contexts[region_key], activate=True)
            else:
                _set_active_region(active_region_key)

        color_select.on_change("value", _on_color_select_change)
        
        # --- SLIDERS ---
        # Bounds slider (first create valid range)
        dates = pd.to_datetime(pd.Series(source.data['date']), errors='coerce')
        valid = dates.notna()

        if valid.any():
            # Use ACTUAL min/max from the data
            start_dt = dates[valid].min()
            end_dt = dates[valid].max()
        else:
            # Fallback if no valid dates
            start_dt = pd.Timestamp("2000-01-01")
            end_dt = pd.Timestamp("2025-01-01")

        # Bounds control slider - this controls the range of the main slider
        date_bounds_slider = DateRangeSlider(
            title="Adjust date slider range (zoom timeline)",
            start=start_dt,  # Actual earliest recording
            end=end_dt,      # Actual latest recording
            value=(start_dt, end_dt),  # Start with full range
            step=1, 
            width=1400
        )

        # Main date filter slider
        date_slider = DateRangeSlider(
            title="Filter recordings between",
            start=start_dt, end=end_dt,
            value=(start_dt, end_dt),
            step=1, width=1400
        )
        
        # --- OTHER WIDGETS ---
        # Hover toggle
        hover_toggle = Toggle(label="Show hover info", active=True)
        
        # Audio controls
        test_audio_btn = Button(label="Test audio server", button_type="primary")
        audio_status = Div(text="Audio server: <i>not tested</i>", width=500, height=50,
                          styles={'border':'1px solid #ddd','padding':'6px','background':'#fff'})
        
        # Playlist panel
        playlist_panel = Div(width=420, height=320,
                           styles={'border':'1px solid #ddd','padding':'8px','background':'#fff'})
        playlist_panel.text = "<i>Click a point in either plot to list nearby recordings...</i>"
        playlist_help_div = Div(
            text=(
                "<b>Annotation requires no pre-existing groups.</b><br>"
                "Click a playlist row, then press 1-9 to assign it to a group."
            ),
            width=420,
            styles={'font-size': '11px', 'color': '#444', 'margin-bottom': '6px'}
        )
        
        # Alpha controls for point transparency
        alpha_toggle = Toggle(label="Full opacity", active=False, width=120)
        point_alpha_spinner = Spinner(
            title="Point alpha",
            low=0.05,
            high=1.0,
            step=0.05,
            value=point_alpha_default,
            width=120,
        )

        dedupe_toggle = Toggle(
            label="Filter local duplicates",
            active=False,
            width=220,
            button_type="default",
        )
        dedupe_distance_spinner = Spinner(
            title="Spatial radius (km)",
            low=0.1,
            high=500.0,
            step=0.1,
            value=2.0,
            width=220,
        )
        dedupe_days_spinner = Spinner(
            title="Temporal window (days)",
            low=0.0,
            high=60.0,
            step=0.5,
            value=1.0,
            width=220,
        )
        dedupe_umap_spinner = Spinner(
            title="UMAP distance",
            low=0.0,
            high=10.0,
            step=0.05,
            value=0.2,
            width=220,
        )
        dedupe_status_div = Div(
            text="<i>Duplicate filter inactive.</i>",
            width=240,
            styles={
                'border': '1px solid #ddd',
                'padding': '6px',
                'background': '#fff',
                'border-radius': '4px',
                'margin-top': '4px'
            }
        )

        def compute_local_duplicate_mask(
            distance_km: float,
            day_window: float,
            umap_threshold: float,
        ) -> np.ndarray:
            """Return a boolean mask marking which points to keep after de-duplication."""

            data = source.data
            n_points = len(data.get('x', []))
            if n_points == 0:
                return np.ones(0, dtype=bool)

            lat_series = pd.to_numeric(pd.Series(data.get('lat', [])), errors='coerce')
            lon_series = pd.to_numeric(pd.Series(data.get('lon', [])), errors='coerce')
            x_series = pd.to_numeric(pd.Series(data.get('x', [])), errors='coerce')
            y_series = pd.to_numeric(pd.Series(data.get('y', [])), errors='coerce')
            date_series = pd.to_datetime(pd.Series(data.get('date', [])), errors='coerce')

            valid_mask = (
                lat_series.notna()
                & lon_series.notna()
                & x_series.notna()
                & y_series.notna()
                & date_series.notna()
            )
            valid_indices = np.flatnonzero(valid_mask.to_numpy(dtype=bool))
            keep_mask = np.ones(n_points, dtype=bool)
            if len(valid_indices) <= 1:
                return keep_mask

            lat_valid = lat_series.iloc[valid_indices].to_numpy(dtype=float)
            lon_valid = lon_series.iloc[valid_indices].to_numpy(dtype=float)
            coords_rad = np.column_stack((np.deg2rad(lat_valid), np.deg2rad(lon_valid)))

            try:
                tree = BallTree(coords_rad, metric='haversine')
            except ValueError:
                return keep_mask

            distance_radians = max(distance_km, 0.0) / EARTH_RADIUS_KM
            neighbors = tree.query_radius(coords_rad, r=distance_radians)

            umap_coords = np.column_stack((
                x_series.iloc[valid_indices].to_numpy(dtype=float),
                y_series.iloc[valid_indices].to_numpy(dtype=float),
            ))
            date_values = date_series.iloc[valid_indices]
            date_ns = date_values.to_numpy(dtype='datetime64[ns]')
            date_ns_int = date_ns.astype('int64')

            max_time_delta = pd.to_timedelta(max(day_window, 0.0), unit='D').to_numpy()
            umap_limit_sq = max(umap_threshold, 0.0) ** 2

            subset_size = len(valid_indices)
            parent = list(range(subset_size))
            rank = [0] * subset_size

            def find_root(idx: int) -> int:
                while parent[idx] != idx:
                    parent[idx] = parent[parent[idx]]
                    idx = parent[idx]
                return idx

            def union(idx_a: int, idx_b: int) -> None:
                root_a = find_root(idx_a)
                root_b = find_root(idx_b)
                if root_a == root_b:
                    return
                if rank[root_a] < rank[root_b]:
                    parent[root_a] = root_b
                elif rank[root_a] > rank[root_b]:
                    parent[root_b] = root_a
                else:
                    parent[root_b] = root_a
                    rank[root_a] += 1

            for local_idx, neighbor_indices in enumerate(neighbors):
                for neighbor_local in neighbor_indices:
                    if neighbor_local == local_idx or neighbor_local < local_idx:
                        continue
                    time_diff = np.abs(date_ns[local_idx] - date_ns[neighbor_local])
                    if time_diff > max_time_delta:
                        continue
                    delta = umap_coords[local_idx] - umap_coords[neighbor_local]
                    if (delta[0] * delta[0] + delta[1] * delta[1]) > umap_limit_sq:
                        continue
                    union(local_idx, neighbor_local)

            components: dict[int, list[int]] = {}
            for local_idx in range(subset_size):
                root = find_root(local_idx)
                components.setdefault(root, []).append(local_idx)

            for members in components.values():
                if len(members) <= 1:
                    continue
                best_local = min(
                    members,
                    key=lambda idx_local: (
                        int(date_ns_int[idx_local]),
                        valid_indices[idx_local],
                    ),
                )
                for idx_local in members:
                    global_idx = valid_indices[idx_local]
                    keep_mask[global_idx] = idx_local == best_local

            return keep_mask

        def apply_dedupe_filter() -> None:
            """Apply or clear the duplicate filter based on current widget values."""

            data = source.data
            n_points = len(data.get('x', []))
            if n_points == 0:
                dedupe_status_div.text = "<i>No data available to filter.</i>"
                return

            def _clean_value(widget, default: float, *, minimum: float | None = None) -> float:
                try:
                    value = float(widget.value)
                except (TypeError, ValueError):
                    value = default
                if not np.isfinite(value):
                    value = default
                if minimum is not None:
                    value = max(value, minimum)
                widget.value = value
                return value

            distance_val = _clean_value(dedupe_distance_spinner, 2.0, minimum=0.0)
            days_val = _clean_value(dedupe_days_spinner, 1.0, minimum=0.0)
            umap_val = _clean_value(dedupe_umap_spinner, 0.2, minimum=0.0)

            if not dedupe_toggle.active:
                reset_data = dict(source.data)
                reset_data['dedupe_on'] = [True] * n_points
                refresh_alpha(reset_data)
                dedupe_status_div.text = "<i>Duplicate filter inactive.</i>"
                return

            mask = compute_local_duplicate_mask(distance_val, days_val, umap_val)
            duplicates_removed = int(mask.size - np.count_nonzero(mask))
            updated = dict(source.data)
            updated['dedupe_on'] = mask.tolist()
            refresh_alpha(updated)

            if mask.size == 0:
                dedupe_status_div.text = "<i>No data available to evaluate.</i>"
                return

            remaining = int(np.count_nonzero(mask))
            if duplicates_removed > 0:
                dedupe_status_div.text = (
                    f"Kept {remaining} of {mask.size} points "
                    f"(removed {duplicates_removed}) within "
                    f"{distance_val:g} km / {days_val:g} day / UMAP ≤ {umap_val:g}."
                )
            else:
                dedupe_status_div.text = (
                    "No points met the spatial, temporal, and UMAP similarity thresholds."
                )

        def _on_dedupe_update(attr: str, old: Any, new: Any) -> None:
            apply_dedupe_filter()

        dedupe_toggle.on_change('active', _on_dedupe_update)
        dedupe_distance_spinner.on_change('value', _on_dedupe_update)
        dedupe_days_spinner.on_change('value', _on_dedupe_update)
        dedupe_umap_spinner.on_change('value', _on_dedupe_update)

        def get_spinner_alpha() -> float:
            """Return the current spinner alpha, clamped to (0, 1]."""
            try:
                value = float(point_alpha_spinner.value)
            except (TypeError, ValueError):
                value = point_alpha_default
            return max(min(value, 1.0), 0.01)

        region_update_in_progress = False
        region_normalizing = False

        def _region_key_from_mode(mode: str) -> Optional[str]:
            """Map a Color-by label to a region variant key, if applicable."""

            if not mode:
                return None
            if mode in region_contexts:
                return mode
            return region_label_to_key.get(str(mode), None)

        def _set_active_region(key: str) -> None:
            """Mark the given region context as active and toggle renderers."""

            nonlocal active_region_key
            if key not in region_contexts:
                return
            active_region_key = key
            for ctx in region_contexts.values():
                is_active = ctx["key"] == active_region_key and _region_key_from_mode(color_select.value) == ctx["key"]
                ctx["renderer"].visible = is_active
                ctx["draft_line"].visible = is_active and ctx["draw_toggle"].active
                ctx["draft_points"].visible = is_active and ctx["draw_toggle"].active

        def _normalize_region_source(ctx: dict[str, Any]) -> None:
            """Keep a region source columns in sync to avoid renderer errors."""

            nonlocal region_normalizing
            if region_normalizing:
                return
            region_normalizing = True
            try:
                data = dict(ctx["source"].data)
                xs = list(data.get('xs', []))
                ys = list(data.get('ys', []))
                n = min(len(xs), len(ys))
                xs = xs[:n]
                ys = ys[:n]

                fill_alpha = list(data.get('fill_alpha', []))
                if len(fill_alpha) < n:
                    fill_alpha.extend([0.2] * (n - len(fill_alpha)))
                elif len(fill_alpha) > n:
                    fill_alpha = fill_alpha[:n]

                raw_names = list(data.get('name', []))
                if len(raw_names) < n:
                    raw_names.extend([""] * (n - len(raw_names)))
                elif len(raw_names) > n:
                    raw_names = raw_names[:n]
                names = [
                    _clean_region_name(raw_names[idx], f"Region {idx + 1}")
                    for idx in range(n)
                ]

                raw_colors = list(data.get('color', []))
                if len(raw_colors) < n:
                    raw_colors.extend([REGION_DEFAULT_COLOR] * (n - len(raw_colors)))
                elif len(raw_colors) > n:
                    raw_colors = raw_colors[:n]
                colors = [
                    _clean_region_color(raw_colors[idx], REGION_DEFAULT_COLOR)
                    for idx in range(n)
                ]

                ctx["source"].data = {
                    'xs': xs,
                    'ys': ys,
                    'fill_alpha': fill_alpha,
                    'name': names,
                    'color': colors,
                }
            finally:
                region_normalizing = False

        def apply_regions_from_source(
            ctx: dict[str, Any],
            status_message: Optional[str] = None,
            *,
            activate: bool = True,
        ) -> None:
            """Assign region labels/colors based on the provided region context."""

            nonlocal region_update_in_progress
            if region_update_in_progress:
                return
            region_update_in_progress = True
            try:
                if activate:
                    _set_active_region(ctx["key"])

                data = dict(source.data)
                x_points = np.asarray(data.get('x3857', []), dtype=float)
                y_points = np.asarray(data.get('y3857', []), dtype=float)
                n_points = len(x_points)

                raw_xs = list(ctx["source"].data.get('xs', []))
                raw_ys = list(ctx["source"].data.get('ys', []))
                raw_names = list(ctx["source"].data.get('name', []))
                raw_colors = list(ctx["source"].data.get('color', []))

                xs: list[list[float]] = []
                ys: list[list[float]] = []
                names: list[str] = []
                colors: list[str] = []

                for idx in range(min(len(raw_xs), len(raw_ys))):
                    try:
                        xs_vals = [float(val) for val in raw_xs[idx]]
                        ys_vals = [float(val) for val in raw_ys[idx]]
                    except Exception:
                        continue
                    if len(xs_vals) < 3 or len(ys_vals) < 3 or len(xs_vals) != len(ys_vals):
                        continue

                    xs.append(xs_vals)
                    ys.append(ys_vals)
                    fallback_name = f"Region {len(names) + 1}"
                    name_value = raw_names[idx] if idx < len(raw_names) else ""
                    names.append(_clean_region_name(name_value, fallback_name))
                    color_value = raw_colors[idx] if idx < len(raw_colors) else REGION_DEFAULT_COLOR
                    colors.append(_clean_region_color(color_value, REGION_DEFAULT_COLOR))

                region_labels = ["Unassigned"] * n_points
                region_colors = [REGION_UNASSIGNED_COLOR] * n_points
                region_ids = [-1] * n_points

                def point_in_poly(px_val: float, py_val: float, poly_x: list[float], poly_y: list[float]) -> bool:
                    inside = False
                    last = len(poly_x) - 1
                    for i, (x_i, y_i) in enumerate(zip(poly_x, poly_y)):
                        x_j = poly_x[last]
                        y_j = poly_y[last]
                        last = i
                        try:
                            intersects = (y_i > py_val) != (y_j > py_val) and (
                                px_val
                                < (x_j - x_i) * (py_val - y_i) / ((y_j - y_i) if (y_j - y_i) != 0 else 1e-9) + x_i
                            )
                        except Exception:
                            intersects = False
                        if intersects:
                            inside = not inside
                    return inside

                region_names = names
                region_color_lookup = colors
                for region_idx, (poly_x, poly_y) in enumerate(zip(xs, ys)):
                    name_val = region_names[region_idx]
                    color_val = region_color_lookup[region_idx] if region_idx < len(region_color_lookup) else REGION_DEFAULT_COLOR
                    for point_idx, (px_val, py_val) in enumerate(zip(x_points, y_points)):
                        if not (np.isfinite(px_val) and np.isfinite(py_val)):
                            continue
                        if point_in_poly(float(px_val), float(py_val), poly_x, poly_y):
                            region_labels[point_idx] = name_val
                            region_colors[point_idx] = color_val
                            region_ids[point_idx] = region_idx

                counts = Counter(region_labels)
                checkbox_labels = [f"Unassigned ({counts.get('Unassigned', 0)})"] + [
                    f"{name} ({counts.get(name, 0)})" for name in region_names
                ]
                checks = ctx["checks"]
                checks.labels = checkbox_labels
                if not checks.active or max(checks.active, default=-1) >= len(checkbox_labels):
                    checks.active = list(range(len(checkbox_labels)))

                active_labels = {
                    checkbox_labels[idx].rsplit(" (", 1)[0]
                    for idx in (checks.active or [])
                    if 0 <= idx < len(checkbox_labels)
                }
                if not active_labels:
                    active_labels = {label.rsplit(" (", 1)[0] for label in checkbox_labels}

                region_on = [label in active_labels for label in region_labels]
                adjusted_fill_alpha = [
                    0.25 if name in active_labels else 0.05
                    for name in region_names
                ]

                nonlocal region_normalizing
                region_normalizing = True
                try:
                    ctx["source"].data = {
                        'xs': xs,
                        'ys': ys,
                        'fill_alpha': adjusted_fill_alpha,
                        'name': region_names,
                        'color': region_color_lookup,
                    }
                finally:
                    region_normalizing = False

                if _region_key_from_mode(color_select.value) == ctx["key"]:
                    data['region_label'] = region_labels
                    data['region_color'] = region_colors
                    data['region_id'] = region_ids
                    data['region_on'] = region_on
                    if _region_key_from_mode(color_select.value) == ctx["key"]:
                        data['active_color'] = list(region_colors)
                    refresh_alpha(data)

                status_text = status_message
                if status_text is None:
                    if region_names:
                        status_text = f"{ctx['label']}: {', '.join(region_names)}"
                    else:
                        status_text = "<i>No regions loaded.</i>"
                ctx["status_div"].text = status_text
            finally:
                region_update_in_progress = False

        def apply_active_region(status_message: Optional[str] = None) -> None:
            """Apply the region assignments for the currently selected variant."""

            key = _region_key_from_mode(color_select.value) or active_region_key
            ctx = region_contexts.get(key)
            if not ctx:
                return
            _set_active_region(key)
            apply_regions_from_source(ctx, status_message=status_message, activate=True)

        def clear_regions(ctx: dict[str, Any]) -> None:
            """Clear all drawn regions and reset assignments."""
            ctx["source"].data = {'xs': [], 'ys': [], 'fill_alpha': [], 'name': [], 'color': []}
            ctx["draft_source"].data = {'x': [], 'y': []}
            checks = ctx["checks"]
            checks.labels = ["Unassigned (0)"]
            checks.active = [0]
            apply_regions_from_source(ctx, status_message="<i>Regions cleared.</i>")

        def load_regions_from_disk(ctx: dict[str, Any]) -> None:
            """Load polygons from the region boundaries directory for a context."""
            regions = load_region_polygons(ctx["file_path"])
            if not regions:
                clear_regions(ctx)
                ctx["status_div"].text = (
                    f"<i>No region polygons found for {SPECIES_SLUG} ({ctx['file_path'].name}).</i>"
                )
                return

            xs_saved: list[list[float]] = []
            ys_saved: list[list[float]] = []
            fill_alpha_saved: list[float] = []
            names_saved: list[str] = []
            colors_saved: list[str] = []
            for region in regions:
                lon_vals = region.get("lon", [])
                lat_vals = region.get("lat", [])
                if len(lon_vals) < 3 or len(lat_vals) < 3 or len(lon_vals) != len(lat_vals):
                    continue
                x_vals, y_vals = lonlat_to_web_mercator(lon_vals, lat_vals)
                xs_saved.append([float(val) for val in x_vals])
                ys_saved.append([float(val) for val in y_vals])
                fill_alpha_saved.append(0.2)
                default_name = f"Region {len(names_saved) + 1}"
                names_saved.append(
                    _clean_region_name(region.get("name") or region.get("label"), default_name)
                )
                colors_saved.append(_clean_region_color(region.get("color"), REGION_DEFAULT_COLOR))

            ctx["source"].data = {
                'xs': xs_saved,
                'ys': ys_saved,
                'fill_alpha': fill_alpha_saved,
                'name': names_saved,
                'color': colors_saved,
            }
            apply_regions_from_source(
                ctx,
                f"Loaded {len(xs_saved)} region{'s' if len(xs_saved) != 1 else ''} "
                f"from {ctx['file_path'].name}.",
            )

        def save_regions_to_disk(ctx: dict[str, Any]) -> None:
            """Save current polygons for the context to its region boundaries file."""
            xs_list = ctx["source"].data.get('xs', [])
            ys_list = ctx["source"].data.get('ys', [])
            names_list = ctx["source"].data.get('name', [])
            colors_list = ctx["source"].data.get('color', [])
            saved_path = save_region_polygons(
                xs_list,
                ys_list,
                names_list,
                colors_list,
                destination=ctx["file_path"],
            )
            if saved_path is None:
                ctx["status_div"].text = "<i>No valid regions to save.</i>"
                return
            apply_regions_from_source(
                ctx,
                f"Saved regions to {saved_path.name}.",
            )

        def _on_region_source_change(ctx: dict[str, Any], attr: str, old: dict, new: dict) -> None:
            _normalize_region_source(ctx)
            apply_regions_from_source(ctx)

        def _on_cancel_draft(ctx: dict[str, Any]) -> None:
            ctx["draft_source"].data = {'x': [], 'y': []}
            ctx["draw_toggle"].active = False

        def _on_draw_toggle_change(ctx: dict[str, Any], attr: str, old: Any, new: Any) -> None:
            if new:
                for other_key, other_ctx in region_contexts.items():
                    if other_key != ctx["key"] and other_ctx["draw_toggle"].active:
                        other_ctx["draw_toggle"].active = False
                _set_active_region(ctx["key"])
            ctx["draft_line"].visible = (
                ctx["key"] == active_region_key
                and ctx["draw_toggle"].active
                and _region_key_from_mode(color_select.value) == ctx["key"]
            )
            ctx["draft_points"].visible = ctx["draft_line"].visible

        for context in region_contexts.values():
            context["source"].on_change('data', lambda attr, old, new, c=context: _on_region_source_change(c, attr, old, new))
            context["checks"].on_change('active', lambda attr, old, new, c=context: apply_regions_from_source(c))
            context["load_btn"].on_click(lambda c=context: load_regions_from_disk(c))
            context["save_btn"].on_click(lambda c=context: save_regions_to_disk(c))
            context["clear_btn"].on_click(lambda c=context: clear_regions(c))
            context["cancel_btn"].on_click(lambda c=context: _on_cancel_draft(c))
            context["draw_toggle"].on_change('active', lambda attr, old, new, c=context: _on_draw_toggle_change(c, attr, old, new))

        # Activate default region variant on startup.
        _set_active_region(active_region_key)
        apply_regions_from_source(region_contexts[active_region_key], status_message=None)
        region_widgets_flat = [widget for group in region_widget_groups for widget in group]
        region_widget_sets = {key: ctx["widgets"] for key, ctx in region_contexts.items()}

        geoshift_percent_input = Spinner(
            title="Extreme percentile",
            low=1,
            high=50,
            step=1,
            value=10,
            width=140,
        )
        geoshift_button = Button(
            label="Calculate geographic shift",
            button_type="primary",
            width=220,
        )
        clear_geoshift_button = Button(
            label="Clear geographic shift",
            button_type="default",
            width=220,
        )
        geoshift_summary_div = Div(
            text='<i>Click "Calculate geographic shift" to analyse the visible points.</i>',
            width=280,
            styles={
                'border': '1px solid #ddd',
                'padding': '6px',
                'background': '#fff',
                'border-radius': '4px',
                'margin-top': '6px'
            }
        )

        def _visible_mask_from_view() -> np.ndarray:
            """Return a boolean mask for currently visible points using the map view."""
            try:
                if hasattr(map_view, "filter") and map_view.filter is not None:
                    booleans = getattr(map_view.filter, "booleans", None)
                    if booleans is not None and len(booleans) == len(source.data.get('xcid', [])):
                        return np.array(booleans, dtype=bool)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[GEOSHIFT] Failed to read map_view filter: {exc}")
            alpha_values = np.asarray(source.data.get('alpha', []), dtype=float)
            return alpha_values > 0

        def calculate_geographic_shift() -> None:
            """Compute yearly geographic extremes for the currently visible points."""

            def reset_sources(message: str) -> None:
                geoshift_summary_div.text = message
                empty_shift = {'x': [], 'y': [], 'year': [], 'lat': [], 'lon': [], 'count': []}
                north_shift_source.data = dict(empty_shift)
                south_shift_source.data = dict(empty_shift)
                empty_arrows = {'x_start': [], 'y_start': [], 'x_end': [], 'y_end': []}
                north_arrow_source.data = dict(empty_arrows)
                south_arrow_source.data = dict(empty_arrows)
                north_line_source.data = {'year': [], 'lat': []}
                south_line_source.data = {'year': [], 'lat': []}
                shared_lat_range.start = 0
                shared_lat_range.end = 1

            def build_arrow_data(x_vals: list[float], y_vals: list[float]) -> dict[str, list[float]]:
                x_start: list[float] = []
                y_start: list[float] = []
                x_end: list[float] = []
                y_end: list[float] = []
                for idx in range(len(x_vals) - 1):
                    xs, ys = x_vals[idx], y_vals[idx]
                    xe, ye = x_vals[idx + 1], y_vals[idx + 1]
                    if not (np.isfinite(xs) and np.isfinite(ys) and np.isfinite(xe) and np.isfinite(ye)):
                        continue
                    x_start.append(float(xs))
                    y_start.append(float(ys))
                    x_end.append(float(xe))
                    y_end.append(float(ye))
                return {'x_start': x_start, 'y_start': y_start, 'x_end': x_end, 'y_end': y_end}

            try:
                try:
                    percentile_value = float(geoshift_percent_input.value or 10)
                except (TypeError, ValueError):
                    percentile_value = 10.0
                percentile_value = max(1.0, min(50.0, percentile_value))
                geoshift_percent_input.value = int(round(percentile_value))
                percentile = geoshift_percent_input.value

                alpha_values = np.asarray(source.data.get('alpha', []), dtype=float)
                if alpha_values.size == 0:
                    reset_sources("<i>No data available to compute geographic shift.</i>")
                    return

                visible_mask = _visible_mask_from_view()
                if visible_mask.size != alpha_values.size:
                    reset_sources("<i>No data available to compute geographic shift.</i>")
                    return

                if not visible_mask.any():
                    reset_sources("<i>No visible points under current filters.</i>")
                    return

                lat_series = pd.to_numeric(pd.Series(source.data.get('lat', [])), errors='coerce')
                lon_series = pd.to_numeric(pd.Series(source.data.get('lon', [])), errors='coerce')
                date_series = pd.to_datetime(pd.Series(source.data.get('date', [])), errors='coerce')

                visible_df = pd.DataFrame({
                    'lat': lat_series,
                    'lon': lon_series,
                    'date': date_series
                })
                visible_df = visible_df[visible_mask]
                visible_df = visible_df.dropna(subset=['lat', 'lon', 'date'])
                if visible_df.empty:
                    reset_sources("<i>No visible points have valid coordinates and dates.</i>")
                    return

                slider_value = getattr(date_slider, 'value', None)
                if slider_value and len(slider_value) == 2:
                    slider_start = pd.Timestamp(slider_value[0], unit='ms')
                    slider_end = pd.Timestamp(slider_value[1], unit='ms')
                else:
                    slider_start = visible_df['date'].min()
                    slider_end = visible_df['date'].max()

                if pd.isna(slider_start) or pd.isna(slider_end):
                    reset_sources("<i>Unable to read selected date range.</i>")
                    return

                slider_start = slider_start.normalize()
                slider_end = slider_end.normalize()
                slider_end_inclusive = slider_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

                visible_df = visible_df[
                    (visible_df['date'] >= slider_start) & (visible_df['date'] <= slider_end_inclusive)
                ]
                if visible_df.empty:
                    reset_sources("<i>No visible points fall inside the selected date window.</i>")
                    return

                years: list[int] = []
                north_lats: list[float] = []
                north_lons: list[float] = []
                north_counts: list[int] = []
                south_lats: list[float] = []
                south_lons: list[float] = []
                south_counts: list[int] = []
                lat_samples: list[float] = []

                for year in range(slider_start.year, slider_end.year + 1):
                    year_start = pd.Timestamp(year=year, month=1, day=1)
                    year_end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)

                    if year_start < slider_start or year_end > slider_end_inclusive:
                        continue

                    year_subset = visible_df[
                        (visible_df['date'] >= year_start) & (visible_df['date'] <= year_end)
                    ].copy()
                    if len(year_subset) < 2:
                        continue

                    # Limit dense clusters: max N recordings inside 2 km diameter (≈1 km radius).
                    try:
                        coords = np.radians(year_subset[['lat', 'lon']].to_numpy(dtype=float))
                        if len(coords):
                            db = DBSCAN(
                                eps=GEOSHIFT_CLUSTER_RADIUS_KM / EARTH_RADIUS_KM,
                                min_samples=1,
                                metric='haversine'
                            )
                            cluster_labels = db.fit_predict(coords)
                            year_subset['_cluster'] = cluster_labels
                            year_subset = (
                                year_subset.sort_values(['_cluster', 'date'])
                                .groupby('_cluster', sort=False)
                                .head(GEOSHIFT_MAX_RECORDINGS_PER_CLUSTER)
                                .drop(columns='_cluster')
                            )
                    except Exception as cluster_exc:  # pragma: no cover - defensive
                        print(f"[GEOSHIFT] Cluster capping failed for {year}: {cluster_exc}")

                    if len(year_subset) < 2:
                        continue

                    max_non_overlap = len(year_subset) // 2
                    if max_non_overlap == 0:
                        continue

                    count_extreme = max(1, int(np.floor(len(year_subset) * (percentile / 100.0))))
                    count_extreme = min(count_extreme, max_non_overlap)
                    if count_extreme == 0:
                        continue

                    ordered = year_subset.sort_values('lat', kind='mergesort')
                    south_slice = ordered.iloc[:count_extreme]
                    north_slice = ordered.iloc[-count_extreme:]

                    south_mean_lat = float(south_slice['lat'].mean())
                    south_mean_lon = float(south_slice['lon'].mean())
                    north_mean_lat = float(north_slice['lat'].mean())
                    north_mean_lon = float(north_slice['lon'].mean())

                    years.append(year)
                    north_lats.append(north_mean_lat)
                    north_lons.append(north_mean_lon)
                    north_counts.append(int(len(north_slice)))
                    south_lats.append(south_mean_lat)
                    south_lons.append(south_mean_lon)
                    south_counts.append(int(len(south_slice)))
                    lat_samples.extend(south_slice['lat'].tolist())
                    lat_samples.extend(north_slice['lat'].tolist())

                if not years:
                    reset_sources("<i>No full calendar years inside the selected range have enough visible data.</i>")
                    return

                north_x, north_y = lonlat_to_web_mercator(north_lons, north_lats)
                south_x, south_y = lonlat_to_web_mercator(south_lons, south_lats)

                north_shift_source.data = {
                    'x': [float(val) for val in north_x],
                    'y': [float(val) for val in north_y],
                    'year': [str(year) for year in years],
                    'lat': [float(val) for val in north_lats],
                    'lon': [float(val) for val in north_lons],
                    'count': [int(val) for val in north_counts],
                }
                south_shift_source.data = {
                    'x': [float(val) for val in south_x],
                    'y': [float(val) for val in south_y],
                    'year': [str(year) for year in years],
                    'lat': [float(val) for val in south_lats],
                    'lon': [float(val) for val in south_lons],
                    'count': [int(val) for val in south_counts],
                }

                if len(years) > 1:
                    north_arrow_source.data = build_arrow_data(list(north_x), list(north_y))
                    south_arrow_source.data = build_arrow_data(list(south_x), list(south_y))
                else:
                    empty_arrow = {'x_start': [], 'y_start': [], 'x_end': [], 'y_end': []}
                    north_arrow_source.data = dict(empty_arrow)
                    south_arrow_source.data = dict(empty_arrow)

                north_line_source.data = {
                    'year': [int(year) for year in years],
                    'lat': [float(val) for val in north_lats],
                }
                south_line_source.data = {
                    'year': [int(year) for year in years],
                    'lat': [float(val) for val in south_lats],
                }
                if lat_samples:
                    lat_min = float(np.nanmin(lat_samples))
                    lat_max = float(np.nanmax(lat_samples))
                    if np.isfinite(lat_min) and np.isfinite(lat_max):
                        if np.isclose(lat_min, lat_max):
                            lat_min -= 0.5
                            lat_max += 0.5
                        shared_lat_range.start = lat_min
                        shared_lat_range.end = lat_max

                north_shifts = [north_lats[i + 1] - north_lats[i] for i in range(len(north_lats) - 1)]
                south_shifts = [south_lats[i + 1] - south_lats[i] for i in range(len(south_lats) - 1)]
                avg_north_shift = float(np.mean(north_shifts)) if north_shifts else None
                avg_south_shift = float(np.mean(south_shifts)) if south_shifts else None

                def compute_trend(year_vals: list[int], lat_vals: list[float]) -> Optional[float]:
                    """Return slope in degrees/year for best-fit line."""
                    if len(year_vals) < 2:
                        return None
                    try:
                        slope, _ = np.polyfit(year_vals, lat_vals, 1)
                        return float(slope)
                    except Exception:
                        return None

                north_slope = compute_trend(years, north_lats)
                south_slope = compute_trend(years, south_lats)

                rows_html = "".join(
                    f"<tr>"
                    f"<td style='padding:2px 4px;'>{year}</td>"
                    f"<td style='padding:2px 4px; color:{GEOSHIFT_NORTH_COLOR};'>"
                    f"{north_lats[idx]:.2f}&deg; (n={north_counts[idx]})</td>"
                    f"<td style='padding:2px 4px; color:{GEOSHIFT_SOUTH_COLOR};'>"
                    f"{south_lats[idx]:.2f}&deg; (n={south_counts[idx]})</td>"
                    f"</tr>"
                    for idx, year in enumerate(years)
                )
                table_html = (
                    "<table style='width:100%; border-collapse:collapse; margin-top:6px;'>"
                    "<thead><tr style='text-align:left;'>"
                    "<th style='padding:2px 4px;'>Year</th>"
                    "<th style='padding:2px 4px;'>North mean lat</th>"
                    "<th style='padding:2px 4px;'>South mean lat</th>"
                    "</tr></thead>"
                    f"<tbody>{rows_html}</tbody></table>"
                )

                summary_lines = [
                    f"<b>Geographic shift</b> ({percentile}% extremes, "
                    f"≤{GEOSHIFT_MAX_RECORDINGS_PER_CLUSTER} recs / {GEOSHIFT_CLUSTER_DIAMETER_KM:.1f} km diameter)",
                    f"Full years analysed: {', '.join(str(year) for year in years)}",
                ]
                if avg_north_shift is not None:
                    summary_lines.append(
                        f"Avg north lat shift: <span style='color:{GEOSHIFT_NORTH_COLOR};'>"
                        f"{avg_north_shift:+.2f}&deg;/year</span> (n={len(north_shifts)})"
                    )
                else:
                    summary_lines.append("Avg north lat shift: n/a")
                if avg_south_shift is not None:
                    summary_lines.append(
                        f"Avg south lat shift: <span style='color:{GEOSHIFT_SOUTH_COLOR};'>"
                        f"{avg_south_shift:+.2f}&deg;/year</span> (n={len(south_shifts)})"
                    )
                else:
                    summary_lines.append("Avg south lat shift: n/a")
                if north_slope is not None:
                    summary_lines.append(
                        f"North trend slope: <span style='color:{GEOSHIFT_NORTH_COLOR};'>"
                        f"{north_slope:+.2f}&deg;/year</span>"
                    )
                else:
                    summary_lines.append("North trend slope: n/a")
                if south_slope is not None:
                    summary_lines.append(
                        f"South trend slope: <span style='color:{GEOSHIFT_SOUTH_COLOR};'>"
                        f"{south_slope:+.2f}&deg;/year</span>"
                    )
                else:
                    summary_lines.append("South trend slope: n/a")

                geoshift_summary_div.text = "<br>".join(summary_lines) + table_html

            except Exception as exc:  # pragma: no cover - defensive
                reset_sources(
                    f"<span style='color:#b00;'>Geographic shift failed: {exc}</span>"
                )
                traceback.print_exc()

        def clear_geographic_shift() -> None:
            reset_sources('<i>Geographic shift cleared.</i>')

        # UMAP parameter inputs
        umap_params_div = Div(text="<b>UMAP Parameters:</b>", width=200)
        umap_neighbors_input = Spinner(
            title="Nearest neighbors", 
            low=2, high=100, step=1, 
            value=ANALYSIS_PARAMS.get('umap_n_neighbors', 10),
            width=150
        )
        umap_mindist_input = Spinner(
            title="Min distance", 
            low=0.0, high=1.0, step=0.01, 
            value=ANALYSIS_PARAMS.get('umap_min_dist', 0.0),
            width=150
        )
        
        # HDBSCAN parameters box
        hdbscan_params_div = Div(text="<b>HDBSCAN Parameters:</b>", width=200)
        hdbscan_min_cluster_size = Spinner(
            title="Min cluster size",
            low=2, high=500, step=1,
            value=15,
            width=150
        )
        hdbscan_min_samples = Spinner(
            title="Min samples",
            low=1, high=100, step=1,
            value=5,
            width=150
        )
        hdbscan_apply_btn = Button(
            label="Apply HDBSCAN",
            button_type="primary",
            width=150
        )
        hdbscan_stats_div = Div(width=300, height=30, text="<i>HDBSCAN not yet computed</i>")

        # HDBSCAN checkboxes (initially hidden)
        hdbscan_checks = CheckboxGroup(
            labels=[],
            active=[],
            visible=False,
            name="hdbscan_checks"
        )

        SelectionGroup = dict[str, int | str]
        selection_groups: list[SelectionGroup] = []
        next_selection_id = 0
        suppress_selection_status_update = False

        def _group_label_value(group: SelectionGroup, fallback: int) -> int:
            """Return the display label for a selection group."""
            try:
                label_val = int(group.get('label', fallback))
                if label_val > 0:
                    return label_val
            except (TypeError, ValueError):
                pass
            return fallback

        def _next_group_label() -> int:
            """Pick the smallest positive group label not already in use."""
            existing = {
                _group_label_value(group, int(group.get('id', -1)) + 1)
                for group in selection_groups
            }
            candidate = 1
            while candidate in existing:
                candidate += 1
            return candidate

        def _group_color_for_id(group_id: int) -> str:
            """Return the color associated with a group id."""
            for group in selection_groups:
                try:
                    if int(group.get('id')) == group_id:
                        return str(group.get('color'))
                except (TypeError, ValueError):
                    continue
            return SELECTION_UNASSIGNED_COLOR

        def _group_label_for_id(group_id: int) -> int:
            """Return the display label for a group id."""
            for group in selection_groups:
                try:
                    if int(group.get('id')) == group_id:
                        return _group_label_value(group, group_id + 1)
                except (TypeError, ValueError):
                    continue
            return group_id + 1

        def _find_group_by_label(label: int) -> Optional[SelectionGroup]:
            """Find a group matching the provided label, if it exists."""
            for group in selection_groups:
                if _group_label_value(group, -1) == label:
                    return group
            return None

        def pick_group_color() -> str:
            """Return a high-contrast color that is not already in use."""
            used_colors = {str(group['color']).lower() for group in selection_groups}
            for color in SELECTION_PALETTE:
                if color.lower() not in used_colors:
                    return color
            return SELECTION_PALETTE[next_selection_id % len(SELECTION_PALETTE)]

        def update_selection_widgets() -> None:
            nonlocal selection_groups
            """Update widget labels, visibility, and active states for selections."""
            data_dict = source.data
            assignments = data_dict.get('selection_group', [])
            selection_flags = data_dict.get('selection_on', [])
            counts = Counter(assignments)

            selection_groups = [
                group for group in selection_groups
                if counts.get(int(group['id']), 0) > 0
            ]
            selection_groups = sorted(
                selection_groups,
                key=lambda group: _group_label_value(
                    group,
                    int(group.get('id', -1)) + 1,
                )
            )

            labels: list[str] = []
            tags: list[int] = []
            active_indices: list[int] = []

            def group_is_active(group_id: int) -> bool:
                return any(
                    assignments[idx] == group_id and selection_flags[idx]
                    for idx in range(len(assignments))
                )

            unassigned_count = counts.get(-1, 0)
            labels.append(f"Unassigned ({unassigned_count})")
            tags.append(-1)
            if group_is_active(-1) or (not selection_groups and unassigned_count > 0):
                active_indices.append(0)

            for group in selection_groups:
                group_id = int(group['id'])
                count = counts.get(group_id, 0)
                label_value = _group_label_value(group, len(labels) + 1)
                labels.append(f"Group {label_value} ({count})")
                tags.append(group_id)
                if group_is_active(group_id):
                    active_indices.append(len(labels) - 1)

            selection_checks.labels = labels
            selection_checks.tags = tags
            selection_checks.active = sorted(set(active_indices))
            is_selection_mode = color_select.value == "Selection"
            selection_checks.visible = is_selection_mode
            selection_help_div.visible = is_selection_mode
            for btn in (
                selection_save_vocal_btn,
                selection_save_dialect_btn,
                selection_save_annotate1_btn,
                selection_save_annotate2_btn,
                selection_load_vocal_btn,
                selection_load_dialect_btn,
                selection_load_annotate1_btn,
                selection_load_annotate2_btn,
            ):
                btn.visible = is_selection_mode
            for save_btn in (
                selection_save_vocal_btn,
                selection_save_dialect_btn,
                selection_save_annotate1_btn,
                selection_save_annotate2_btn,
            ):
                save_btn.disabled = not selection_groups
            if not is_selection_mode:
                selection_status_div.text = ""
            elif not selection_status_div.text:
                selection_status_div.text = "<i>No active selection.</i>"
            selection_create_btn.visible = is_selection_mode
            if not is_selection_mode:
                selection_create_btn.disabled = True
            selection_status_div.visible = is_selection_mode
            has_assigned = any(group_id != -1 for group_id in assignments)
            selection_clear_btn.visible = is_selection_mode and has_assigned

        def clear_selection_groups() -> None:
            nonlocal selection_groups, next_selection_id
            """Clear all selection groups and restore base colors and visibility."""
            selection_groups = []
            next_selection_id = 0
            current = dict(source.data)
            total = len(current.get('selection_group', []))
            current['selection_group'] = [-1] * total
            current['selection_group_label'] = [-1] * total
            current['selection_color'] = [SELECTION_UNASSIGNED_COLOR for _ in range(total)]
            current['selection_on'] = [True] * total
            selection_status_div.text = "<i>No active selection.</i>"
            selection_create_btn.disabled = True
            current = refresh_alpha(current, update_source=False)
            if color_select.value == "Selection":
                current['active_color'] = list(current['selection_color'])
            source.data = current
            source.selected.indices = []
            update_selection_widgets()

        def assign_new_group(
            indices: list[int],
            *,
            label: Optional[int] = None,
        ) -> tuple[int, int]:
            """Assign the provided indices to a newly created selection group."""
            nonlocal selection_groups, next_selection_id
            assignments = list(source.data.get('selection_group', []))
            total = len(assignments)
            if total == 0:
                return -1, 0

            valid_indices = sorted({
                idx for idx in indices
                if isinstance(idx, int) and 0 <= idx < total
            })
            if not valid_indices:
                return -1, 0

            colors = list(source.data.get('selection_color', []))
            selection_flags = list(source.data.get('selection_on', []))
            labels = list(source.data.get('selection_group_label', []))
            if len(colors) < total:
                colors.extend([SELECTION_UNASSIGNED_COLOR] * (total - len(colors)))
            if len(selection_flags) < total:
                selection_flags.extend([True] * (total - len(selection_flags)))
            if len(labels) < total:
                labels.extend([-1] * (total - len(labels)))

            group_id = next_selection_id
            next_selection_id += 1
            group_color = pick_group_color()
            requested_label = label if label is not None else _next_group_label()
            group_label = (
                requested_label
                if requested_label > 0 and not _find_group_by_label(requested_label)
                else _next_group_label()
            )

            for idx in valid_indices:
                assignments[idx] = group_id
                colors[idx] = group_color
                selection_flags[idx] = True
                labels[idx] = group_label

            for idx in range(total):
                if assignments[idx] == -1:
                    colors[idx] = SELECTION_UNASSIGNED_COLOR
                    labels[idx] = -1

            selection_groups.append({'id': group_id, 'color': group_color, 'label': group_label})

            updated = dict(source.data)
            updated['selection_group'] = assignments
            updated['selection_color'] = colors
            updated['selection_on'] = selection_flags
            updated['selection_group_label'] = labels
            if color_select.value == "Selection":
                updated['active_color'] = list(colors)

            source.data = updated
            update_selection_widgets()
            return group_id, len(valid_indices)

        def assign_indices_to_existing_group(
            group_id: int,
            indices: list[int],
        ) -> int:
            """Assign the provided indices to an existing selection group."""
            assignments = list(source.data.get('selection_group', []))
            total = len(assignments)
            if total == 0:
                return 0

            valid_indices = sorted({
                idx for idx in indices
                if isinstance(idx, int) and 0 <= idx < total
            })
            if not valid_indices:
                return 0

            colors = list(source.data.get('selection_color', []))
            selection_flags = list(source.data.get('selection_on', []))
            labels = list(source.data.get('selection_group_label', []))
            if len(colors) < total:
                colors.extend([SELECTION_UNASSIGNED_COLOR] * (total - len(colors)))
            if len(selection_flags) < total:
                selection_flags.extend([True] * (total - len(selection_flags)))
            if len(labels) < total:
                labels.extend([-1] * (total - len(labels)))

            group_color = _group_color_for_id(group_id)
            group_label = _group_label_for_id(group_id)

            for idx in valid_indices:
                assignments[idx] = group_id
                colors[idx] = group_color
                selection_flags[idx] = True
                labels[idx] = group_label

            for idx in range(total):
                if assignments[idx] == -1:
                    colors[idx] = SELECTION_UNASSIGNED_COLOR
                    labels[idx] = -1

            updated = dict(source.data)
            updated['selection_group'] = assignments
            updated['selection_color'] = colors
            updated['selection_on'] = selection_flags
            updated['selection_group_label'] = labels
            if color_select.value == "Selection":
                updated['active_color'] = list(colors)

            source.data = updated
            update_selection_widgets()
            return len(valid_indices)

        def create_selection_group(selected_indices: list[int]) -> None:
            nonlocal selection_groups, next_selection_id, suppress_selection_status_update
            """Create a new selection group using the currently visible selection."""
            if color_select.value != "Selection":
                return
            if not selected_indices:
                selection_status_div.text = "<i>No points selected.</i>"
                selection_create_btn.disabled = True
                return

            alpha_values = source.data.get('alpha', [])
            visible_selected = [
                idx
                for idx in selected_indices
                if 0 <= idx < len(alpha_values) and alpha_values[idx] > 0
            ]
            if not visible_selected:
                selection_status_div.text = (
                    "<i>Selection contains no visible points; adjust filters.</i>"
                )
                selection_create_btn.disabled = True
                return

            group_id, assigned_count = assign_new_group(visible_selected)
            if group_id == -1 or assigned_count == 0:
                selection_status_div.text = "<i>Could not create a group from the current selection.</i>"
                selection_create_btn.disabled = True
                return

            selection_status_div.text = (
                f"Created selection group with {assigned_count} visible points."
            )
            selection_create_btn.disabled = True
            suppress_selection_status_update = True
            source.selected.indices = []

        def handle_selection_change(attr: str, old: list[int], new: list[int]) -> None:
            nonlocal suppress_selection_status_update
            if suppress_selection_status_update and not new:
                suppress_selection_status_update = False
                selection_create_btn.disabled = True
                return
            suppress_selection_status_update = False
            is_selection_mode = color_select.value == "Selection"
            selection_create_btn.visible = is_selection_mode
            if not is_selection_mode:
                selection_create_btn.disabled = True
                return

            alpha_values = source.data.get('alpha', [])
            visible_selected = [
                idx
                for idx in new
                if 0 <= idx < len(alpha_values) and alpha_values[idx] > 0
            ]
            selection_create_btn.disabled = len(visible_selected) == 0

            if visible_selected:
                count = len(visible_selected)
                plural = "s" if count != 1 else ""
                selection_status_div.text = (
                    f"Selected {count} visible point{plural}. Click \"Create group from selection\" to save."
                )
            elif new:
                selection_status_div.text = "<i>Selection contains no visible points; adjust filters.</i>"
            else:
                selection_status_div.text = "<i>No active selection.</i>"

        update_selection_widgets()
        selection_clear_btn.on_click(clear_selection_groups)
        source.selected.on_change('indices', handle_selection_change)
        def on_create_selection() -> None:
            """Persist the current selection as a group when requested by the user."""
            create_selection_group(list(source.selected.indices))

        selection_create_btn.on_click(on_create_selection)

        def _active_selection_group_ids() -> list[int]:
            """Return the list of non-negative group ids currently checked in the widget."""
            tags = list(getattr(selection_checks, 'tags', []) or [])
            active_indices = list(getattr(selection_checks, 'active', []) or [])
            group_ids: list[int] = []
            for idx in active_indices:
                if 0 <= idx < len(tags):
                    try:
                        group_id = int(tags[idx])
                    except (TypeError, ValueError):
                        continue
                    if group_id != -1:
                        group_ids.append(group_id)
            return group_ids

        def save_selection_group_to_table(
            kind_label: str,
            destination_dir: Path,
            description_text: str,
        ) -> None:
            """Persist the currently highlighted selection group to a CSV table."""
            if color_select.value != "Selection":
                return
            if not selection_groups:
                selection_status_div.text = "<i>No groups are available to save.</i>"
                return
            selected_ids = _active_selection_group_ids()
            if not selected_ids:
                selection_status_div.text = "<i>Select exactly one group in the list to save.</i>"
                return
            if len(selected_ids) > 1:
                selection_status_div.text = "<i>Please select only one group before saving.</i>"
                return

            target_group = selected_ids[0]
            assignments = list(source.data.get('selection_group', []))
            target_indices = sorted([
                idx for idx, group_id in enumerate(assignments)
                if group_id == target_group
            ])
            if not target_indices:
                selection_status_div.text = "<i>The selected group contains no samples.</i>"
                return

            xcid_values = list(source.data.get('xcid', []))
            clip_indices = source.data.get('clip_index')
            if clip_indices is None:
                selection_status_div.text = (
                    "<span style='color:#b00;'>Clip index data is unavailable; cannot save group.</span>"
                )
                return
            clip_list = list(clip_indices)

            rows: list[dict[str, Any]] = []
            skipped = 0
            for idx in target_indices:
                if idx >= len(xcid_values) or idx >= len(clip_list):
                    continue
                clip_int = normalize_clip_index(clip_list[idx])
                if clip_int is None:
                    skipped += 1
                    continue
                rows.append({
                    'xcid': xcid_values[idx],
                    'clip_index': clip_int,
                })

            if not rows:
                selection_status_div.text = (
                    "<span style='color:#b00;'>No valid clip indices found for this group; nothing was saved.</span>"
                )
                return

            try:
                destination_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                selection_status_div.text = (
                    f"<span style='color:#b00;'>Failed to prepare {kind_label} directory: {exc}</span>"
                )
                return

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            base_name = f"group_{target_group}_{timestamp}"
            output_path = destination_dir / f"{base_name}{GROUP_TABLE_SUFFIX}"
            counter = 1
            while output_path.exists():
                output_path = destination_dir / f"{base_name}_{counter}{GROUP_TABLE_SUFFIX}"
                counter += 1

            description_to_write = description_text if description_text is not None else ""

            try:
                pd.DataFrame(rows).to_csv(output_path, index=False)
                description_path = output_path.with_suffix(".txt")
                with open(description_path, "w", encoding="utf-8") as desc_file:
                    desc_file.write(description_to_write.strip() or "No description provided.")
            except Exception as exc:  # pragma: no cover - file write failure
                selection_status_div.text = (
                    f"<span style='color:#b00;'>Failed to save group artifacts: {exc}</span>"
                )
                return

            try:
                relative_display = output_path.relative_to(ROOT_PATH)
            except ValueError:
                try:
                    relative_display = output_path.relative_to(APP_ROOT)
                except ValueError:
                    relative_display = output_path

            message = (
                f"Saved {len(rows)} samples to {relative_display} "
                f"for {kind_label} (description stored beside CSV)."
            )
            if skipped:
                message += f" Skipped {skipped} sample(s) without clip indices."
            selection_status_div.text = message

        def load_groups_from_tables(kind_label: str, tables_dir: Path) -> None:
            """Load all group tables for the current species and recreate them as selections."""
            if not tables_dir.exists():
                selection_status_div.text = (
                    f"<i>No {kind_label} folder found at {tables_dir}.</i>"
                )
                return

            table_paths = sorted(
                path for path in tables_dir.iterdir()
                if path.is_file() and path.suffix.lower() == GROUP_TABLE_SUFFIX
            )
            if not table_paths:
                selection_status_div.text = (
                    f"<i>No {GROUP_TABLE_SUFFIX} {kind_label} tables found for {SPECIES_SLUG}.</i>"
                )
                return

            clip_indices = source.data.get('clip_index')
            if clip_indices is None:
                selection_status_div.text = (
                    "<span style='color:#b00;'>Clip index data is unavailable; cannot load tables.</span>"
                )
                return

            xcid_values = list(source.data.get('xcid', []))
            clip_list = list(clip_indices)
            dataset_lookup: dict[tuple[str, int], list[int]] = {}
            for idx, (xcid_value, clip_value) in enumerate(zip(xcid_values, clip_list)):
                clip_int = normalize_clip_index(clip_value)
                if clip_int is None:
                    continue
                key = (str(xcid_value), clip_int)
                dataset_lookup.setdefault(key, []).append(idx)

            if not dataset_lookup:
                selection_status_div.text = (
                    "<span style='color:#b00;'>Current dataset is missing clip indices; cannot load tables.</span>"
                )
                return

            candidate_groups: list[tuple[Path, list[int]]] = []
            load_notes: list[str] = []
            for table_path in table_paths:
                try:
                    table_df = pd.read_csv(table_path)
                except Exception as exc:
                    load_notes.append(f"{table_path.name}: failed to read ({exc})")
                    continue

                lower_map = {col.lower(): col for col in table_df.columns}
                if 'xcid' not in lower_map or 'clip_index' not in lower_map:
                    load_notes.append(
                        f"{table_path.name}: missing required 'xcid' or 'clip_index' columns; skipped."
                    )
                    continue

                x_series = table_df[lower_map['xcid']]
                clip_series = table_df[lower_map['clip_index']]
                indices: list[int] = []
                seen_local: set[int] = set()
                for x_value, clip_value in zip(x_series, clip_series):
                    clip_int = normalize_clip_index(clip_value)
                    if clip_int is None or pd.isna(x_value):
                        continue
                    key = (str(x_value), clip_int)
                    matches = dataset_lookup.get(key, [])
                    for idx in matches:
                        if idx not in seen_local:
                            seen_local.add(idx)
                            indices.append(idx)

                if not indices:
                    load_notes.append(f"{table_path.name}: no matching samples in current dataset.")
                    continue

                candidate_groups.append((table_path, indices))

            if not candidate_groups:
                status = (
                    f"<i>No {kind_label} groups were loaded; no matching samples were found.</i>"
                )
                if load_notes:
                    status += "<br>" + "<br>".join(load_notes)
                selection_status_div.text = status
                return

            clear_selection_groups()

            overlapping_indices: set[int] = set()
            summaries: list[str] = []
            total_assigned = 0

            for table_path, indices in candidate_groups:
                current_assignments = list(source.data.get('selection_group', []))
                overlapping = {
                    idx for idx in indices
                    if 0 <= idx < len(current_assignments) and current_assignments[idx] != -1
                }
                overlapping_indices.update(overlapping)

                _, assigned_count = assign_new_group(indices)
                if assigned_count == 0:
                    continue
                total_assigned += assigned_count
                summaries.append(f"{table_path.stem} ({assigned_count})")

            summary_parts = [
                f"Loaded {len(summaries)} {kind_label} table{'s' if len(summaries) != 1 else ''} "
                f"({total_assigned} sample{'s' if total_assigned != 1 else ''})."
            ]
            if summaries:
                summary_parts.append("Tables: " + ", ".join(summaries))
            if overlapping_indices:
                summary_parts.append(
                    f"<span style='color:#b35;'>Warning: {len(overlapping_indices)} sample"
                    f"{'s' if len(overlapping_indices) != 1 else ''} were present in multiple tables; "
                    "later tables overwrote earlier assignments.</span>"
                )
            if load_notes:
                summary_parts.append("<br>".join(load_notes))

            selection_status_div.text = "<br>".join(summary_parts)

        selection_save_vocal_btn.js_on_event(
            ButtonClick,
            CustomJS(args=dict(target=description_request_source), code="""
                const desc = window.prompt("Enter a description for this vocal type group:");
                if (desc === null) {
                    return;
                }
                target.data = {
                    kind: ['vocal'],
                    description: [desc],
                    nonce: [Date.now()]
                };
            """)
        )
        selection_save_dialect_btn.js_on_event(
            ButtonClick,
            CustomJS(args=dict(target=description_request_source), code="""
                const desc = window.prompt("Enter a description for this dialect group:");
                if (desc === null) {
                    return;
                }
                target.data = {
                    kind: ['dialect'],
                    description: [desc],
                    nonce: [Date.now()]
                };
            """)
        )
        selection_save_annotate1_btn.js_on_event(
            ButtonClick,
            CustomJS(args=dict(target=description_request_source), code="""
                const desc = window.prompt("Enter a description for this annotate 1 group:");
                if (desc === null) {
                    return;
                }
                target.data = {
                    kind: ['annotate 1'],
                    description: [desc],
                    nonce: [Date.now()]
                };
            """)
        )
        selection_save_annotate2_btn.js_on_event(
            ButtonClick,
            CustomJS(args=dict(target=description_request_source), code="""
                const desc = window.prompt("Enter a description for this annotate 2 group:");
                if (desc === null) {
                    return;
                }
                target.data = {
                    kind: ['annotate 2'],
                    description: [desc],
                    nonce: [Date.now()]
                };
            """)
        )
        selection_load_vocal_btn.on_click(
            lambda: load_groups_from_tables("vocal types", VOCAL_TYPES_DIR)
        )
        selection_load_dialect_btn.on_click(
            lambda: load_groups_from_tables("dialects", DIALECTS_DIR)
        )
        selection_load_annotate1_btn.on_click(
            lambda: load_groups_from_tables("annotate 1", ANNOTATE_ONE_DIR)
        )
        selection_load_annotate2_btn.on_click(
            lambda: load_groups_from_tables("annotate 2", ANNOTATE_TWO_DIR)
        )

        def handle_annotation_request(attr: str, old: dict, new: dict) -> None:
            """Assign the selected playlist entry to the requested group label."""
            indices = new.get('index') or []
            groups = new.get('group') or []
            if not indices or not groups:
                return

            try:
                target_index = int(indices[0])
                target_label = int(groups[0])
            except (TypeError, ValueError):
                annotation_request_source.data = {'index': [], 'group': [], 'nonce': []}
                return

            assignments = list(source.data.get('selection_group', []))
            if target_index < 0 or target_index >= len(assignments):
                annotation_request_source.data = {'index': [], 'group': [], 'nonce': []}
                return

            target_label = max(1, min(target_label, 9))
            existing_group = _find_group_by_label(target_label)
            assigned = 0

            if existing_group:
                assigned = assign_indices_to_existing_group(
                    int(existing_group['id']),
                    [target_index],
                )
                if assigned:
                    selection_status_div.text = (
                        f"Annotated 1 sample to Group {target_label}."
                    )
            else:
                _, assigned = assign_new_group(
                    [target_index],
                    label=target_label,
                )
                if assigned:
                    selection_status_div.text = (
                        f"Created Group {target_label} and annotated 1 sample."
                    )

            if not assigned:
                selection_status_div.text = "<i>Annotation failed; no valid target.</i>"

            annotation_request_source.data = {'index': [], 'group': [], 'nonce': []}

        def handle_description_request(attr: str, old: dict, new: dict) -> None:
            kinds = new.get('kind') or []
            descriptions = new.get('description') or []
            if not kinds or not descriptions:
                return

            kind_value = str(kinds[0]).strip().lower()
            description_text = str(descriptions[0])
            directory_map = {
                'vocal': ("vocal types", VOCAL_TYPES_DIR),
                'vocal types': ("vocal types", VOCAL_TYPES_DIR),
                'dialect': ("dialects", DIALECTS_DIR),
                'dialects': ("dialects", DIALECTS_DIR),
                'annotate 1': ("annotate 1", ANNOTATE_ONE_DIR),
                'annotate1': ("annotate 1", ANNOTATE_ONE_DIR),
                'annotate 2': ("annotate 2", ANNOTATE_TWO_DIR),
                'annotate2': ("annotate 2", ANNOTATE_TWO_DIR),
            }
            mapped = directory_map.get(kind_value)
            if not mapped:
                selection_status_div.text = (
                    f"<span style='color:#b00;'>Unknown save target '{kind_value}'.</span>"
                )
                description_request_source.data = {'kind': [], 'description': [], 'nonce': []}
                return

            label, directory = mapped
            save_selection_group_to_table(label, directory, description_text)
            description_request_source.data = {'kind': [], 'description': [], 'nonce': []}

        annotation_request_source.on_change('data', handle_annotation_request)
        description_request_source.on_change('data', handle_description_request)

        print("  All widgets created")
        
        # --- SETUP CALLBACKS ---
        print("\n" + "-" * 40)
        print("SETTING UP CALLBACKS...")
        
        # --- CHECKBOX CALLBACKS ---
        # Season checkbox callback - exactly like cluster callback
        season_callback = CustomJS(args=dict(src=source, cb=season_checks,
                                   umap_view=umap_view, map_view=map_view), code="""
            const d = src.data;
            const labs = cb.labels;  // Read current labels from widget
            const active_indices = cb.active;
            
            // Build set of active labels
            const active = new Set();
            for (let idx of active_indices) {
                if (idx < labs.length) {
                    active.add(labs[idx]);
                }
            }
            
            const season = d['season'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const hdbscan_on = d['hdbscan_on'] || new Array(season.length).fill(true);  // ADD THIS
            const sex_on = d['sex_on'] || new Array(season.length).fill(true);
            const type_on = d['type_on'] || new Array(season.length).fill(true);
            const time_on = d['time_on'] || new Array(season.length).fill(true);
            const selection_on = d['selection_on'] || new Array(season.length).fill(true);
            const dedupe_on = d['dedupe_on'] || new Array(season.length).fill(true);
            const region_on = d['region_on'] || new Array(season.length).fill(true);

            const n = season.length;
            for (let i = 0; i < n; i++) {
                const season_visible = active.has(String(season[i]));
                d['season_on'][i] = season_visible;

                if (hdbscan_on[i] && sex_on[i] && type_on[i] &&
                    time_on[i] && selection_on[i] && dedupe_on[i] && region_on[i] && season_visible) {
                    alpha[i] = alpha_base[i];
                } else {
                    alpha[i] = 0.0;
                }
            }
            
            // Update views at the end
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }
            src.change.emit();
        """)
        season_checks.js_on_change('active', season_callback)
        
        # Sex checkbox callback - reads labels dynamically
        sex_callback = CustomJS(args=dict(src=source, cb=sex_checks,
                                   umap_view=umap_view, map_view=map_view), code="""
            const d = src.data;
            const labs = cb.labels;  // Read current labels from widget
            const active_indices = cb.active;
            
            // Build set of active labels
            const active = new Set();
            for (let idx of active_indices) {
                if (idx < labs.length) {
                    active.add(labs[idx]);
                }
            }
            
            const sex = d['sex'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const hdbscan_on = d['hdbscan_on'] || new Array(sex.length).fill(true);
            const type_on = d['type_on'] || new Array(sex.length).fill(true);
            const time_on = d['time_on'] || new Array(sex.length).fill(true);
            const season_on = d['season_on'] || new Array(sex.length).fill(true);
            const selection_on = d['selection_on'] || new Array(sex.length).fill(true);
            const dedupe_on = d['dedupe_on'] || new Array(sex.length).fill(true);
            const region_on = d['region_on'] || new Array(sex.length).fill(true);


            const n = sex.length;
            for (let i = 0; i < n; i++) {
                const sex_visible = active.has(String(sex[i]));
                d['sex_on'][i] = sex_visible;

                if (hdbscan_on[i] && sex_visible && type_on[i] &&
                    time_on[i] && season_on[i] && selection_on[i] && dedupe_on[i] && region_on[i]) {
                    alpha[i] = alpha_base[i];
                } else {
                    alpha[i] = 0.0;
                }
            }
            
            // Update views at the end
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }
            src.change.emit();
        """)
        sex_checks.js_on_change('active', sex_callback)

        # Type checkbox callback - reads labels dynamically
        type_callback = CustomJS(args=dict(src=source, cb=type_checks,
                                   umap_view=umap_view, map_view=map_view), code="""
            const d = src.data;
            const labs = cb.labels;  // Read current labels from widget
            const active_indices = cb.active;
            
            // Build set of active labels
            const active = new Set();
            for (let idx of active_indices) {
                if (idx < labs.length) {
                    active.add(labs[idx]);
                }
            }
            
            const type = d['type'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const hdbscan_on = d['hdbscan_on'] || new Array(type.length).fill(true);
            const sex_on = d['sex_on'] || new Array(type.length).fill(true);
            const time_on = d['time_on'] || new Array(type.length).fill(true);
            const season_on = d['season_on'] || new Array(type.length).fill(true);
            const selection_on = d['selection_on'] || new Array(type.length).fill(true);
            const dedupe_on = d['dedupe_on'] || new Array(type.length).fill(true);
            const region_on = d['region_on'] || new Array(type.length).fill(true);

            const n = type.length;
            for (let i = 0; i < n; i++) {
                const type_visible = active.has(String(type[i]));
                d['type_on'][i] = type_visible;

                if (hdbscan_on[i] && sex_on[i] && type_visible &&
                    time_on[i] && season_on[i] && selection_on[i] && dedupe_on[i] && region_on[i]) {
                    alpha[i] = alpha_base[i];
                } else {
                    alpha[i] = 0.0;
                }
            }
            
            // Update views at the end
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }
            src.change.emit();
        """)
        type_checks.js_on_change('active', type_callback)
        
        # Time range slider callback (what would happen if selection contains invalid times?, like with checkboxes should we read dynamically?)
        time_range_callback = CustomJS(args=dict(src=source, slider=time_range_slider,
                                   umap_view=umap_view, map_view=map_view), code="""
            const d = src.data;
            const time_hour = d['time_hour'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            
            // Add fallback initialization for all _on fields
            const hdbscan_on = d['hdbscan_on'] || new Array(time_hour.length).fill(true);
            const sex_on = d['sex_on'] || new Array(time_hour.length).fill(true);
            const type_on = d['type_on'] || new Array(time_hour.length).fill(true);
            const season_on = d['season_on'] || new Array(time_hour.length).fill(true);
            const selection_on = d['selection_on'] || new Array(time_hour.length).fill(true);
            const dedupe_on = d['dedupe_on'] || new Array(time_hour.length).fill(true);
            const region_on = d['region_on'] || new Array(time_hour.length).fill(true);

            const min_hour = slider.value[0];
            const max_hour = slider.value[1];
            
            const n = time_hour.length;
            for (let i = 0; i < n; i++) {
                const hour = time_hour[i];
                // Allow invalid times (-1) or times within range
                const time_visible = (hour < 0) || (hour >= min_hour && hour <= max_hour);
                d['time_on'][i] = time_visible;
                
                // Alpha is visible only if ALL filters allow it
                if (hdbscan_on[i] && sex_on[i] && type_on[i] &&
                    time_visible && season_on[i] && selection_on[i] && dedupe_on[i] && region_on[i]) {
                    alpha[i] = alpha_base[i];
                } else {
                    alpha[i] = 0.0;
                }
            }
            
            // Update views at the end
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }
            src.change.emit();
        """)
        time_range_slider.js_on_change('value', time_range_callback)

        selection_callback = CustomJS(args=dict(
            src=source,
            cb=selection_checks,
            umap_view=umap_view,
            map_view=map_view,
            clear_btn=selection_clear_btn,
            sel=color_select
        ), code="""
            const d = src.data;
            const assignments = d['selection_group'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const selection_on = d['selection_on'] || new Array(assignments.length).fill(true);
            const tags = cb.tags || [];
            const active_indices = new Set(cb.active);
            const active_groups = new Set();
            for (let i = 0; i < tags.length; i++) {
                if (active_indices.has(i)) {
                    active_groups.add(tags[i]);
                }
            }

            const hdbscan_on = d['hdbscan_on'] || new Array(assignments.length).fill(true);
            const sex_on = d['sex_on'] || new Array(assignments.length).fill(true);
            const type_on = d['type_on'] || new Array(assignments.length).fill(true);
            const time_on = d['time_on'] || new Array(assignments.length).fill(true);
            const season_on = d['season_on'] || new Array(assignments.length).fill(true);
            const dedupe_on = d['dedupe_on'] || new Array(assignments.length).fill(true);
            const region_on = d['region_on'] || new Array(assignments.length).fill(true);

            const n = assignments.length;
            let hasAssigned = false;
            for (let i = 0; i < n; i++) {
                const group_id = assignments[i];
                if (group_id !== -1) {
                    hasAssigned = true;
                }
                const selection_visible = active_groups.has(group_id);
                selection_on[i] = selection_visible;
                if (selection_visible && hdbscan_on[i] && sex_on[i] && type_on[i] &&
                    time_on[i] && season_on[i] && dedupe_on[i] && region_on[i] && alpha_base[i] > 0) {
                    alpha[i] = alpha_base[i];
                } else {
                    alpha[i] = 0.0;
                }
            }

            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }

            clear_btn.visible = hasAssigned && sel.value === "Selection";
            d['selection_on'] = selection_on;
            src.change.emit();
        """)
        selection_checks.js_on_change('active', selection_callback)
        
        color_callback = CustomJS(args=dict(
            src=source,
            sel=color_select,
            season_checks=season_checks,
            hdbscan_checks=hdbscan_checks,  # Add this
            sex_checks=sex_checks,
            type_checks=type_checks,
            time_slider=time_range_slider,
            selection_checks=selection_checks,
            selection_help=selection_help_div,
            selection_clear=selection_clear_btn,
            selection_status=selection_status_div,
            selection_create=selection_create_btn,
            selection_save_vocal=selection_save_vocal_btn,
            selection_save_dialect=selection_save_dialect_btn,
            selection_save_annotate1=selection_save_annotate1_btn,
            selection_save_annotate2=selection_save_annotate2_btn,
            selection_load_vocal=selection_load_vocal_btn,
            selection_load_dialect=selection_load_dialect_btn,
            selection_load_annotate1=selection_load_annotate1_btn,
            selection_load_annotate2=selection_load_annotate2_btn,
            region_widgets=region_widgets_flat,
            region_widget_sets=region_widget_sets,
            region_label_map=region_label_to_key
        ), code="""
            const d = src.data;
            const mode = sel.value;

            // Hide all filter widgets first
            season_checks.visible = false;
            hdbscan_checks.visible = false;
            sex_checks.visible = false;
            type_checks.visible = false;
            time_slider.visible = false;
            selection_checks.visible = false;
            selection_help.visible = false;
            selection_clear.visible = false;
            selection_status.visible = false;
            selection_status.text = "";
            selection_create.visible = false;
            selection_create.disabled = true;
            selection_save_vocal.visible = false;
            selection_save_vocal.disabled = true;
            selection_save_dialect.visible = false;
            selection_save_dialect.disabled = true;
            selection_save_annotate1.visible = false;
            selection_save_annotate1.disabled = true;
            selection_save_annotate2.visible = false;
            selection_save_annotate2.disabled = true;
            selection_load_vocal.visible = false;
            selection_load_dialect.visible = false;
            selection_load_annotate1.visible = false;
            selection_load_annotate2.visible = false;
            for (const key in region_widget_sets) {
                const widgets = region_widget_sets[key] || [];
                for (const w of widgets) {
                    if (w.visible !== undefined) {
                        w.visible = false;
                    }
                }
            }

            // Map mode to color column and show appropriate widget
            let from_col;
            const regionKey = region_label_map[mode] || null;
            switch(mode) {
                case "Season":
                    from_col = "season_color";
                    season_checks.visible = true;
                    break;
                case "HDBSCAN":
                    if (d['hdbscan_color']) {
                        from_col = "hdbscan_color";
                        hdbscan_checks.visible = true;
                    } else {
                        alert("HDBSCAN not yet computed. Please click 'Apply HDBSCAN' first.");
                        return;
                    }
                    break;
                case "Sex":
                    from_col = "sex_color";
                    sex_checks.visible = true;
                    break;
                case "Type":
                    from_col = "type_color";
                    type_checks.visible = true;
                    break;
                case "Time of Day":
                    from_col = "time_color";
                    time_slider.visible = true;
                    break;
                case "Latitude":
                    from_col = "lat_color";
                    break;
                case "Longitude":
                    from_col = "lon_color";
                    break;
                case "Region (course)":
                case "Region (fine)":
                case "Region (custom)":
                    from_col = "region_color";
                    if (regionKey && region_widget_sets[regionKey]) {
                        for (const w of region_widget_sets[regionKey]) {
                            if (w.visible !== undefined) {
                                w.visible = true;
                            }
                        }
                    }
                    break;
                case "Selection":
                    from_col = "selection_color";
                    selection_checks.visible = true;
                    selection_help.visible = true;
                    selection_status.visible = true;
                    selection_create.visible = true;
                    selection_save_vocal.visible = true;
                    selection_save_dialect.visible = true;
                    selection_save_annotate1.visible = true;
                    selection_save_annotate2.visible = true;
                    selection_load_vocal.visible = true;
                    selection_load_dialect.visible = true;
                    selection_load_annotate1.visible = true;
                    selection_load_annotate2.visible = true;
                    selection_load_vocal.disabled = false;
                    selection_load_dialect.disabled = false;
                    selection_load_annotate1.disabled = false;
                    selection_load_annotate2.disabled = false;
                    let hasVisibleSelection = false;
                    const selected = src.selected.indices ?? [];
                    const alpha = d['alpha'] || [];
                    for (const idx of selected) {
                        if (Number.isInteger(idx) && idx >= 0 && idx < alpha.length && alpha[idx] > 0) {
                            hasVisibleSelection = true;
                            break;
                        }
                    }
                    selection_create.disabled = !hasVisibleSelection;
                    if (!hasVisibleSelection && selected.length === 0) {
                        selection_status.text = "<i>No active selection.</i>";
                    }
                    let hasGroups = false;
                    const selectionAssignments = d['selection_group'];
                    for (let i = 0; i < selectionAssignments.length; i++) {
                        if (selectionAssignments[i] !== -1) {
                            hasGroups = true;
                            break;
                        }
                    }
                    selection_clear.visible = hasGroups;
                    selection_save_vocal.disabled = !hasGroups;
                    selection_save_dialect.disabled = !hasGroups;
                    selection_save_annotate1.disabled = !hasGroups;
                    selection_save_annotate2.disabled = !hasGroups;
                    break;
            }

            // Safety: do nothing if we couldn't map a color column.
            if (!from_col || !d[from_col]) {
                return;
            }
            
            // Update colors
            const n = d['active_color'].length;
            for (let i = 0; i < n; i++) {
                d['active_color'][i] = d[from_col][i];
            }
            src.change.emit();
        """)
        color_select.js_on_change('value', color_callback)

        # --- SLIDER CALLBACKS ---
        # Bounds slider callback - update main slider range
        bounds_callback = CustomJS(args=dict(
            bounds=date_bounds_slider,
            main=date_slider,
            source=source
        ), code="""
            // Get new bounds from the bounds slider
            const new_start = bounds.value[0];
            const new_end = bounds.value[1];
            
            // Update the main slider's range
            main.start = new_start;
            main.end = new_end;
            
            // Ensure current value stays within new bounds
            const current_start = main.value[0];
            const current_end = main.value[1];
            
            // Constrain current selection to new bounds if needed
            const adjusted_start = Math.max(current_start, new_start);
            const adjusted_end = Math.min(current_end, new_end);
            
            // Only update value if it changed
            if (adjusted_start != current_start || adjusted_end != current_end) {
                main.value = [adjusted_start, adjusted_end];
            }
            
            // Optional: Update the main slider's title to show the range
            const start_date = new Date(new_start).toISOString().split('T')[0];
            const end_date = new Date(new_end).toISOString().split('T')[0];
            main.title = `Filter recordings between (${start_date} to ${end_date})`;
        """)
        date_bounds_slider.js_on_change('value', bounds_callback)

        # Also add a reset button for the bounds slider
        reset_bounds_btn = Button(label="Reset timeline range", button_type="default", width=150)
        reset_bounds_callback = CustomJS(args=dict(
            bounds=date_bounds_slider,
            main=date_slider,
            source=source
        ), code="""
            // Find actual data bounds
            const dates = source.data['ts'];
            let data_min = Infinity;
            let data_max = -Infinity;
            
            for (let ts of dates) {
                if (!Number.isNaN(ts)) {
                    if (ts < data_min) data_min = ts;
                    if (ts > data_max) data_max = ts;
                }
            }
            
            // Reset both sliders to full data range
            bounds.value = [data_min, data_max];
            main.start = data_min;
            main.end = data_max;
            main.title = "Filter recordings between";
        """)
        reset_bounds_btn.js_on_click(reset_bounds_callback)
                
        # Date slider callback - update stats when slider changes
        date_callback = CustomJS(args=dict(src=source, s=date_slider, stats=stats_div,
                                   umap_view=umap_view, map_view=map_view,
                                   alpha_spinner=point_alpha_spinner,
                                   toggle=alpha_toggle), code="""
            const d = src.data;
            const ts = d['ts'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const hdbscan_on = d['hdbscan_on'] || new Array(ts.length).fill(true);
            const sex_on = d['sex_on'] || new Array(ts.length).fill(true);
            const type_on = d['type_on'] || new Array(ts.length).fill(true);
            const time_on = d['time_on'] || new Array(ts.length).fill(true);
            const season_on = d['season_on'] || new Array(ts.length).fill(true);
            const selection_on = d['selection_on'] || new Array(ts.length).fill(true);
            const dedupe_on = d['dedupe_on'] || new Array(ts.length).fill(true);
            const region_on = d['region_on'] || new Array(ts.length).fill(true);
            
            const cut0 = Number(s.value[0]);
            const cut1 = Number(s.value[1]);
            const spinner_val_raw = alpha_spinner.value ?? 0.3;
            const spinner_val = Math.max(Math.min(spinner_val_raw, 1.0), 0.01);
            const base_alpha = toggle.active ? 1.0 : spinner_val;

            let visible_count = 0;
            for (let i = 0; i < ts.length; i++) {
                let base = Number.isNaN(ts[i]) ? 0.0 :
                          (ts[i] >= cut0 && ts[i] <= cut1) ? base_alpha : 0.0;
                alpha_base[i] = base;

                // Final alpha depends on ALL filters
                if (hdbscan_on[i] && sex_on[i] && type_on[i] &&
                    time_on[i] && season_on[i] && selection_on[i] && dedupe_on[i] && region_on[i] && base > 0) {
                    alpha[i] = base;
                } else {
                    alpha[i] = 0.0;
                }
                
                if (alpha[i] > 0) visible_count++;
            }
            
            // Update stats display to show visible count
            const start_date = new Date(cut0).toISOString().split('T')[0];
            const end_date = new Date(cut1).toISOString().split('T')[0];
            
            // Keep existing HTML but update the visible count
            let html = stats.text;
            if (html.includes('(Visible:')) {
                html = html.replace(/\(Visible: \d+\)/, `(Visible: ${visible_count})`);
            } else {
                html = html.replace(/Total Points:<\/b> (\d+)/, 
                                  `Total Points:</b> $1 (Visible: ${visible_count})`);
            }
            
            // Update date filter line
            if (html.includes('Date Filter:')) {
                html = html.replace(/Date Filter:<\/b> [\d-]+ to [\d-]+/, 
                                  `Date Filter:</b> ${start_date} to ${end_date}`);
            } else {
                html = html.replace('</div>', 
                                  `<br><b>Date Filter:</b> ${start_date} to ${end_date}</div>`);
            }
            
            stats.text = html;
            
            
            // Update views at the end
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < ts.length; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < ts.length; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }
            src.change.emit();
        """)
        date_slider.js_on_change('value', date_callback)
        
        # --- OTHER CALLBACKS ---
        # Hover toggle callback
        hover_toggle_callback = CustomJS(args=dict(
            h_u=umap_hover,
            h_m=map_hover,
            shift_hovers=[north_shift_hover, south_shift_hover],
            toggle=hover_toggle
        ), code="""
            const showInfo = toggle.active;
            if (!window.original_umap_tooltips) {
                window.original_umap_tooltips = h_u ? h_u.tooltips : null;
                window.original_map_tooltips = h_m ? h_m.tooltips : null;
            }
            if (!window.original_shift_tooltips_list) {
                window.original_shift_tooltips_list = [];
                for (const tool of shift_hovers) {
                    window.original_shift_tooltips_list.push(tool ? tool.tooltips : null);
                }
            }
            if (h_u) h_u.tooltips = showInfo ? window.original_umap_tooltips : null;
            if (h_m) h_m.tooltips = showInfo ? window.original_map_tooltips : null;
            if (shift_hovers && window.original_shift_tooltips_list) {
                for (let i = 0; i < shift_hovers.length; i++) {
                    const tool = shift_hovers[i];
                    if (!tool) continue;
                    const original = window.original_shift_tooltips_list[i];
                    tool.tooltips = showInfo ? original : null;
                }
            }
        """)
        hover_toggle.js_on_change('active', hover_toggle_callback)
        
        # Audio test callback
        test_audio_callback = CustomJS(args=dict(src=source, status=audio_status, base=AUDIO_BASE_URL), code="""
            let testUrl = "";
            const d = src.data;
            if (d['audio_url']) {
                for (let i = 0; i < d['audio_url'].length; i++) {
                    const u = d['audio_url'][i];
                    if (u && typeof u === 'string' && u.trim().length) { 
                        testUrl = u; 
                        break; 
                    }
                }
            }
            if (!testUrl && base) testUrl = String(base);
            
            if (!testUrl) {
                status.text = "Audio server: No URL available to test.";
                return;
            }
            
            testUrl += (testUrl.includes("?") ? "&" : "?") + "t=" + Date.now();
            status.text = "Audio server: testing...";
            
            const a = new Audio();
            let done = false;
            const finish = (ok, msg) => {
                if (done) return; 
                done = true;
                status.text = ok ? "Audio server: reachable" : "Audio server: " + msg;
            };
            
            a.oncanplaythrough = () => finish(true);
            a.onerror = () => finish(false, "unreachable");
            a.src = testUrl;
            a.load();
        """)
        test_audio_btn.js_on_click(test_audio_callback)
        geoshift_button.on_click(calculate_geographic_shift)
        clear_geoshift_button.on_click(clear_geographic_shift)
        
        
        # Playlist callback - FIXED to only include visible points
        # Track plot clicks
        umap_plot.js_on_event('tap', CustomJS(code="""
            window._ctx = 'umap';
        """))
        map_plot.js_on_event('tap', CustomJS(code="""
            window._ctx = 'map';
        """))
        for context in region_contexts.values():
            map_plot.js_on_event('tap', CustomJS(args=dict(
                draw_toggle=context["draw_toggle"],
                draft=context["draft_source"],
                regions=context["source"],
                status=context["status_div"],
                main_source=source,
                region_default_color=REGION_DEFAULT_COLOR,
            ), code="""
                if (!draw_toggle.active) {
                    return;
                }
                const x = cb_obj.x;
                const y = cb_obj.y;
                if (!isFinite(x) || !isFinite(y)) {
                    return;
                }
                if (main_source) {
                    main_source.selected.indices = [];
                }
                const CLOSE_THRESH = 25000;  // meters in web mercator approx
                const draftData = draft.data;
                let xs = Array.isArray(draftData.x) ? draftData.x.slice() : [];
                let ys = Array.isArray(draftData.y) ? draftData.y.slice() : [];

                if (xs.length === 0) {
                    xs.push(x);
                    ys.push(y);
                    draft.data = {x: xs, y: ys};
                    if (status) status.text = "Drawing region: 1 point.";
                    return;
                }

                const dx = x - xs[0];
                const dy = y - ys[0];
                const dist = Math.hypot(dx, dy);

                if (dist < CLOSE_THRESH && xs.length >= 3) {
                    const polyX = xs.concat([xs[0]]);
                    const polyY = ys.concat([ys[0]]);
                    const rdata = Object.assign({}, regions.data);
                    rdata.xs = (rdata.xs || []).slice();
                    rdata.ys = (rdata.ys || []).slice();
                    rdata.fill_alpha = (rdata.fill_alpha || []).slice();
                    rdata.name = (rdata.name || []).slice();
                    rdata.color = (rdata.color || []).slice();
                    const idx = rdata.xs.length;
                    const defaultName = `Region ${idx + 1}`;
                    let name = defaultName;
                    try {
                        const response = window.prompt("Name this region:", defaultName);
                        if (response !== null) {
                            const trimmed = String(response).trim();
                            if (trimmed.length > 0) {
                                name = trimmed;
                            }
                        }
                    } catch (err) {
                        // Ignore prompt errors and fall back to the default name.
                    }
                    const defaultColor = (regions && regions.data && regions.data.color && regions.data.color[idx])
                        ? regions.data.color[idx]
                        : region_default_color;
                    let color = defaultColor;
                    try {
                        const colorResp = window.prompt("Enter color hex (e.g. #005f73):", defaultColor);
                        if (colorResp !== null) {
                            const trimmed = String(colorResp).trim();
                            if (trimmed.length > 0) {
                                color = trimmed.startsWith("#") ? trimmed : `#${trimmed}`;
                            }
                        }
                    } catch (err) {
                        // Ignore prompt errors and fall back to the default color.
                    }

                    rdata.xs.push(polyX);
                    rdata.ys.push(polyY);
                    rdata.fill_alpha.push(0.2);
                    rdata.name.push(name);
                    rdata.color.push(color);

                    regions.data = rdata;
                    draft.data = {x: [], y: []};
                    draw_toggle.active = false;
                    if (status) status.text = `Added ${name} with ${polyX.length - 1} vertices.`;
                    return;
                }

                xs.push(x);
                ys.push(y);
                draft.data = {x: xs, y: ys};
                if (status) status.text = `Drawing region: ${xs.length} points (click near first point to close).`;
            """))
        
        playlist_callback = CustomJS(args=dict(
            src=source,
            pane=playlist_panel,
            annotate=annotation_request_source
        ), code="""
            const d = src.data;
            const inds = src.selected.indices;
            if (!inds.length) return;
            const i = inds[0];
            const selectionGroups = d['selection_group'] || [];
            const selectionLabels = d['selection_group_label'] || [];
            const selectionColors = d['selection_color'] || [];
            const DEFAULT_GROUP_COLOR = "#bdbdbd";

            function normalizeHex(color) {
                if (typeof color !== 'string') return null;
                const trimmed = color.trim();
                if (/^#([0-9a-f]{6}|[0-9a-f]{3})$/i.test(trimmed)) {
                    return trimmed;
                }
                return null;
            }

            function hexToRgb(color) {
                const hex = normalizeHex(color);
                if (!hex) return null;
                let clean = hex.slice(1);
                if (clean.length === 3) {
                    clean = clean.split('').map((ch) => ch + ch).join('');
                }
                const r = parseInt(clean.slice(0, 2), 16);
                const g = parseInt(clean.slice(2, 4), 16);
                const b = parseInt(clean.slice(4, 6), 16);
                if ([r, g, b].some((v) => Number.isNaN(v))) return null;
                return { r, g, b };
            }

            function tintedBackground(color) {
                const rgb = hexToRgb(color) || hexToRgb(DEFAULT_GROUP_COLOR);
                if (!rgb) {
                    return "rgba(189, 189, 189, 0.12)";
                }
                return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.14)`;
            }

            function rowColorInfo(idx) {
                const groupId = Number.isInteger(selectionGroups[idx]) ? selectionGroups[idx] : -1;
                const baseColor = (selectionColors[idx] && typeof selectionColors[idx] === 'string')
                    ? selectionColors[idx]
                    : DEFAULT_GROUP_COLOR;
                const labelValue = Number.isInteger(selectionLabels[idx])
                    ? selectionLabels[idx]
                    : (groupId >= 0 ? groupId + 1 : -1);
                return {
                    groupId,
                    displayLabel: labelValue,
                    groupColor: baseColor || DEFAULT_GROUP_COLOR,
                    groupBg: tintedBackground(baseColor || DEFAULT_GROUP_COLOR),
                };
            }

            function groupLabel(labelValue) {
                return labelValue >= 1 ? `Group ${labelValue}` : "Unassigned";
            }

            if (annotate && !window._bn_annotation_listener_attached) {
                window._bn_annotation_listener_attached = true;
                document.addEventListener('keydown', function(ev) {
                    if (!ev || typeof ev.key !== 'string') {
                        return;
                    }
                    if (!/^[1-9]$/.test(ev.key)) {
                        return;
                    }
                    const active = document.activeElement;
                    if (active && ['INPUT', 'TEXTAREA', 'SELECT'].includes(active.tagName)) {
                        return;
                    }
                    if (window._sel_playlist_idx === null || typeof window._sel_playlist_idx === 'undefined') {
                        return;
                    }
                    annotate.data = {
                        index: [window._sel_playlist_idx],
                        group: [parseInt(ev.key, 10)],
                        nonce: [Date.now()]
                    };
                });
            }

            // Reset playlist-based highlight when building a new playlist
            window._hover_playlist_idx = null;
            window._sel_playlist_idx = null;
            window._sel_playlist_el = null;
            const N_all = d['hl_alpha'].length;
            for (let k = 0; k < N_all; k++) {
                d['hl_alpha'][k] = 0;
                d['hl_width'][k] = 0;
            }
            src.change.emit();
            
            // Check if the clicked point is visible
            if (d['alpha'][i] <= 0) {
                pane.text = "<i>Selected point is hidden by current filters</i>";
                return;
            }
            
            const KM_RADIUS = 15.0;
            const UMAP_RADIUS = 0.4;
            
            const toRad = (deg) => deg * Math.PI / 180.0;
            function hav_km(lat1, lon1, lat2, lon2) {
                if (!isFinite(lat1) || !isFinite(lon1) || 
                    !isFinite(lat2) || !isFinite(lon2)) return Infinity;
                const R = 6371.0088;
                const dlat = toRad(lat2 - lat1);
                const dlon = toRad(lon2 - lon1);
                const a = Math.sin(dlat/2)**2 +
                        Math.cos(toRad(lat1))*Math.cos(toRad(lat2))*Math.sin(dlon/2)**2;
                return 2 * R * Math.asin(Math.sqrt(a));
            }
            
            const ctx = window._ctx || 'umap';
            const N = d['xcid'].length;
            const items = [];
            let centerInfo = "";
            
            if (ctx === 'map') {
                const lat0 = Number(d['lat'][i]);
                const lon0 = Number(d['lon'][i]);
                centerInfo = `Map click – ${KM_RADIUS} km radius`;
                
                for (let j = 0; j < N; j++) {
                    // Only include visible points
                    if (d['alpha'][j] <= 0) continue;
                    
                    const km = hav_km(lat0, lon0, Number(d['lat'][j]), Number(d['lon'][j]));
                    if (km <= KM_RADIUS) items.push([j, km]);
                }
                items.sort((a,b) => a[1]-b[1]);
            } else {
                const x0 = Number(d['x'][i]);
                const y0 = Number(d['y'][i]);
                centerInfo = `UMAP click – radius ${UMAP_RADIUS}`;
                
                for (let j = 0; j < N; j++) {
                    // Only include visible points
                    if (d['alpha'][j] <= 0) continue;
                    
                    const dx = Number(d['x'][j]) - x0;
                    const dy = Number(d['y'][j]) - y0;
                    const r = Math.hypot(dx, dy);
                    if (isFinite(r) && r <= UMAP_RADIUS) items.push([j, r]);
                }
                items.sort((a,b) => a[1]-b[1]);
            }
            
            if (!items.length) {
                pane.text = `<b>${centerInfo}</b><br>No visible neighbors found.`;
                return;
            }
            
            let html = `
                <style>
                    .playlist-row {
                        cursor: pointer;
                        margin: 4px 0;
                        padding: 4px 0;
                        border-bottom: 1px solid #eee;
                        border-left: 6px solid var(--group-color, ${DEFAULT_GROUP_COLOR});
                        background: var(--group-bg, #fff);
                        transition: background-color 120ms ease, border-color 120ms ease, outline 120ms ease;
                    }
                    .playlist-row:hover {
                        outline: 2px solid #ff8c00;
                        outline-offset: 1px;
                        background: #fff9e6 !important;
                        border-left-color: #ff8c00 !important;
                    }
                    .playlist-row-selected {
                        outline: 2px solid #ff8c00;
                        outline-offset: 1px;
                        background: #ffe5b3 !important;
                        border-left-color: #ff8c00 !important;
                    }
                    .playlist-row .group-pill {
                        display: inline-block;
                        min-width: 72px;
                        padding: 3px 6px;
                        margin-right: 6px;
                        border-radius: 10px;
                        background: var(--group-color, ${DEFAULT_GROUP_COLOR});
                        color: #000;
                        font-size: 10px;
                        line-height: 1.2;
                        text-align: center;
                        font-weight: 600;
                        box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
                    }
                </style>
                <b>${centerInfo}</b> – ${items.length} visible recordings<br>
                <div style="max-height:280px; overflow:auto; margin-top:8px;">
            `;
            
            for (const [j, dist] of items) {
                const xcid = d['xcid'][j];
                const date = d['date'][j];
                const url = d['audio_url'] ? d['audio_url'][j] : "";
                const spectro = d['spectrogram_url'] ? d['spectrogram_url'][j] : "";
                const spectroInline = d['spectrogram_data_uri'] ? d['spectrogram_data_uri'][j] : "";
                const spectroSrc = spectroInline || spectro;
                const spectroHtml = spectroSrc
                    ? `<img src="${spectroSrc}" alt="Spectrogram for ${xcid}" style="max-width:160px; border:1px solid #ccc; border-radius:4px;">`
                    : `<div style="color:#888;"><small>No spectrogram</small></div>`;
                const { displayLabel, groupColor, groupBg } = rowColorInfo(j);
                const groupText = groupLabel(displayLabel);
                
                // Each recording row becomes sensitive to mouse enter/leave events.
                // Hover highlights the corresponding point; click pins the highlight.
                html += `<div class="playlist-row"
                        style="--group-color:${groupColor}; --group-bg:${groupBg};"
                        onmouseenter="
                            var src = Bokeh.documents[0].get_model_by_name('source');
                            var d = src.data;
                            var N = d['hl_alpha'].length;
                            window._hover_playlist_idx = ${j};
                            for (var k = 0; k < N; k++) {
                                var sel = (window._sel_playlist_idx === k);
                                var hov = (window._hover_playlist_idx === k);
                                var on = sel || hov;
                                d['hl_alpha'][k] = on ? 1 : 0;
                                d['hl_width'][k] = on ? 3 : 0;
                            }
                            src.change.emit();
                        "
                        onmouseleave="
                            var src = Bokeh.documents[0].get_model_by_name('source');
                            var d = src.data;
                            var N = d['hl_alpha'].length;
                            window._hover_playlist_idx = null;
                            for (var k = 0; k < N; k++) {
                                var sel = (window._sel_playlist_idx === k);
                                var on = sel;
                                d['hl_alpha'][k] = on ? 1 : 0;
                                d['hl_width'][k] = on ? 3 : 0;
                            }
                            src.change.emit();
                        "
                        onclick="
                            var src = Bokeh.documents[0].get_model_by_name('source');
                            var d = src.data;
                            var N = d['hl_alpha'].length;

                            // Update row selection styling
                            if (window._sel_playlist_el && window._sel_playlist_el !== this) {
                                window._sel_playlist_el.classList.remove('playlist-row-selected');
                            }
                            this.classList.add('playlist-row-selected');
                            window._sel_playlist_el = this;

                            // Pin the selected index
                            window._sel_playlist_idx = ${j};

                            // Recompute highlight to combine selection + current hover
                            for (var k = 0; k < N; k++) {
                                var sel = (window._sel_playlist_idx === k);
                                var hov = (window._hover_playlist_idx === k);
                                var on = sel || hov;
                                d['hl_alpha'][k] = on ? 1 : 0;
                                d['hl_width'][k] = on ? 3 : 0;
                            }
                            src.change.emit();
                        "
                    >
                    
                    <div style="display:flex; align-items:flex-start; gap:10px;">
                        <div class="group-pill" title="${groupText}">${groupText}</div>
                        <button onclick="(function(u){
                            if(!u) return;
                            if(window._BN_audio) window._BN_audio.pause();
                            window._BN_audio = new Audio(u);
                            window._BN_audio.play();
                        })('${url}')">Play</button>
                        <div style="min-width:120px;">
                            <b>${xcid}</b><br>
                            <small>${date}</small>
                        </div>
                        ${spectroHtml}
                    </div>
                </div>`;
            }
            
            html += '</div>';
            pane.text = html;
        """)
        
        source.selected.js_on_change('indices', playlist_callback)
        
        # Alpha toggle callback - instant change
        alpha_toggle_callback = CustomJS(args=dict(src=source, toggle=alpha_toggle,
                                          alpha_spinner=point_alpha_spinner,
                                          umap_view=umap_view, map_view=map_view), code="""
            const d = src.data;
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const spinner_val_raw = alpha_spinner.value ?? 0.3;
            const spinner_val = Math.max(Math.min(spinner_val_raw, 1.0), 0.01);
            const new_alpha = toggle.active ? 1.0 : spinner_val;

            // Update base alpha for all visible points
            for (let i = 0; i < alpha_base.length; i++) {
                if (alpha_base[i] > 0) {
                    alpha_base[i] = new_alpha;
                    // Also update current alpha if it's visible
                    if (alpha[i] > 0) {
                        alpha[i] = new_alpha;
                    }
                }
            }

            const booleans = Array.from(alpha, (value) => value > 0);
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                umap_view.filter.booleans = booleans.slice();
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                map_view.filter.booleans = booleans.slice();
            }
            src.change.emit();
        """)
        alpha_toggle.js_on_change('active', alpha_toggle_callback)

        point_alpha_spinner_callback = CustomJS(args=dict(
            src=source,
            spinner=point_alpha_spinner,
            toggle=alpha_toggle,
            umap_view=umap_view,
            map_view=map_view,
        ), code="""
            if (toggle.active) {
                return;
            }

            const d = src.data;
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const spinner_val_raw = spinner.value ?? 0.3;
            const spinner_val = Math.max(Math.min(spinner_val_raw, 1.0), 0.01);
            const n = alpha_base.length;

            for (let i = 0; i < n; i++) {
                if (alpha_base[i] > 0) {
                    alpha_base[i] = spinner_val;
                    if (alpha[i] > 0) {
                        alpha[i] = spinner_val;
                    }
                }
            }

            const booleans = Array.from(alpha, (value) => value > 0);
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                umap_view.filter.booleans = booleans.slice();
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                map_view.filter.booleans = booleans.slice();
            }
            src.change.emit();
        """)
        point_alpha_spinner.js_on_change('value', point_alpha_spinner_callback)
        
        # Zoom and reset callbacks
        # Define zoom functions inline in create_app scope
        def zoom_to_visible():
            """Recompute UMAP on only the currently visible points"""
            print("\n[ZOOM] Computing UMAP on visible points...")
            
            # Get selected indices that are also visible
            selected = source.selected.indices
            if not selected:
                zoom_status.text = "No points selected"
                return
                
            # Filter to only visible selected points
            visible_selected = [i for i in selected if source.data['alpha'][i] > 0]
            
            if len(visible_selected) < 10:
                zoom_status.text = f"Need at least 10 visible points (found {len(visible_selected)})"
                return
            
            print(f"[ZOOM] Selected {len(selected)} points, {len(visible_selected)} are visible")
            
            # Get the original data indices for the visible selected points
            actual_indices = [state.current_indices[i] for i in visible_selected]
            
            # Get subset of original data
            subset_embeddings = state.original_embeddings[actual_indices]
            subset_meta = state.original_meta.iloc[actual_indices].copy().reset_index(drop=True)
            
            print(f"[ZOOM] Computing UMAP on {len(actual_indices)} points from original data")
            
            # Recompute UMAP with user-specified parameters
            n_neighbors = min(umap_neighbors_input.value, len(actual_indices) - 1)
            min_dist = umap_mindist_input.value
            
            print(f"[ZOOM] Using UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")
            
            subset_mapper = umap.UMAP(
                n_neighbors=n_neighbors, 
                min_dist=min_dist, 
                n_components=2, 
                n_jobs=1
            )
            new_projection = subset_mapper.fit_transform(subset_embeddings)
            
            # Update state (not sure yet how indices play into this)
            state.current_embeddings = subset_embeddings
            state.current_meta = subset_meta
            state.current_indices = np.array(actual_indices)
            state.projection = new_projection
            state.is_zoomed = True
            
            # Update visualization
            base_alpha = 1.0 if alpha_toggle.active else get_spinner_alpha()
            new_data = prepare_hover_data(
                subset_meta, new_projection, point_alpha=base_alpha
            )
            
            # Get new unique values             
            new_unique_seasons = sorted(set(new_data['season']))
            new_unique_sex = sorted(set(new_data['sex']))
            new_unique_type = sorted(set(new_data['type']))
            
            print(f"[ZOOM] New seasons: {new_unique_seasons}")
            print(f"[ZOOM] New sex values: {new_unique_sex}")
            print(f"[ZOOM] New type values: {new_unique_type}")
            
            # Reset all filter states to True
            new_data['season_on'] = [True] * len(new_data['x'])
            new_data['hdbscan_on'] = [True] * len(new_data['x'])
            new_data['sex_on'] = [True] * len(new_data['x'])
            new_data['type_on'] = [True] * len(new_data['x'])
            new_data['time_on'] = [True] * len(new_data['x'])
            
            # If HDBSCAN was computed, recompute for zoomed data
            if 'hdbscan_str' in source.data:
                # Recompute HDBSCAN on zoomed projection
                compute_hdbscan_clustering()  # This will update the zoomed data
            
            # When we zoom, and we recreate widgets, should this source data be changed before their creation? Does this currently happen?
            source.data = new_data
            clear_selection_groups()
            apply_dedupe_filter()
            apply_active_region()
            
            # Update filter widgets with new unique values and reset to all active
            season_checks.labels = new_unique_seasons
            season_checks.active = list(range(len(new_unique_seasons)))
            
            sex_checks.labels = new_unique_sex
            sex_checks.active = list(range(len(new_unique_sex)))
            
            type_checks.labels = new_unique_type
            type_checks.active = list(range(len(new_unique_type)))

            print(f"[ZOOM] Updated widgets - seasons: {len(new_unique_seasons)}, sex: {len(new_unique_sex)}, type: {len(new_unique_type)}")

            # Update displays
            update_stats_display(subset_meta, stats_div, 1, source=source)
            zoom_status.text = f"Viewing: Zoomed subset ({len(actual_indices)} points)"
            print(f"[ZOOM] Complete. Showing {len(actual_indices)} points")
        
        def reset_to_full_dataset():
            """Simple reset - reload everything from scratch"""
            print("\n[RESET] Resetting to full dataset...")
            
            # Reset state to original
            state.current_embeddings = state.original_embeddings
            state.current_meta = state.original_meta.copy()
            state.current_indices = np.arange(len(state.original_embeddings))
            state.is_zoomed = False
            
            # Force recomputation by clearing the projection first
            state.projection = None  # This ensures compute_initial_umap will actually compute
            
            # Recompute fresh UMAP on full dataset with user-specified parameters
            n_neighbors = umap_neighbors_input.value
            min_dist = umap_mindist_input.value
            
            print(f"[RESET] Using UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")
            
            state.projection, state.current_meta = compute_initial_umap(
                state.original_embeddings, 
                state.current_meta,
                n_neighbors=n_neighbors,
                min_dist=min_dist
            )
            
            # Prepare full data with original factors
            base_alpha = 1.0 if alpha_toggle.active else get_spinner_alpha()
            new_data = prepare_hover_data(
                state.current_meta, state.projection, point_alpha=base_alpha
            )
            
            # Get unique values for the full dataset
            original_unique_seasons = sorted(set(new_data['season']))
            original_unique_sex = sorted(set(new_data['sex']))
            original_unique_type = sorted(set(new_data['type']))
            original_unique_seasons = sorted(set(new_data['season']))

            print(f"[RESET] Original seasons: {original_unique_seasons}")
            print(f"[RESET] Original sex values: {original_unique_sex}")
            print(f"[RESET] Original type values: {original_unique_type}")
            
            # Reset all filter states to True for full dataset
            new_data['season_on'] = [True] * len(new_data['x'])
            new_data['hdbscan_on'] = [True] * len(new_data['x'])
            new_data['sex_on'] = [True] * len(new_data['x'])
            new_data['type_on'] = [True] * len(new_data['x'])
            new_data['time_on'] = [True] * len(new_data['x'])
            
            # same comment as in zoom_to_visible
            source.data = new_data
            clear_selection_groups()
            apply_dedupe_filter()
            apply_active_region()
            
            # Reset filter widgets with original values and set all active
            season_checks.labels = original_unique_seasons
            season_checks.active = list(range(len(original_unique_seasons)))

            sex_checks.labels = original_unique_sex
            sex_checks.active = list(range(len(original_unique_sex)))
            
            type_checks.labels = original_unique_type
            type_checks.active = list(range(len(original_unique_type)))
            
            # Reset time range slider to full range
            time_range_slider.value = (0, 24)

            print(f"[RESET] Reset widgets - seasons: {len(original_unique_seasons)}, sex: {len(original_unique_sex)}, type: {len(original_unique_type)}")

            # Update displays
            update_stats_display(state.current_meta, stats_div, 0, source=source)
            zoom_status.text = "Viewing: Full dataset"
            print(f"[RESET] Complete - showing {len(state.current_meta)} points")
        
        # Simple zoom callbacks
        def on_zoom_click():
            selected = source.selected.indices
            if selected:
                # Only zoom on selected AND visible points
                visible_selected = [i for i in selected if source.data['alpha'][i] > 0]
                if len(visible_selected) < 10:
                    zoom_status.text = "Need at least 10 visible points selected"
                    source.selected.indices = []
                    return
                
                zoom_to_visible()
                source.selected.indices = []
        
        def on_reset_click():
            reset_to_full_dataset()
        
        zoom_button.on_click(on_zoom_click)
        reset_button.on_click(on_reset_click)
        
        def compute_hdbscan_clustering():
            """Compute HDBSCAN clustering on current projection"""
            print("\n[HDBSCAN] Computing clusters...")
            
            min_cluster_size = hdbscan_min_cluster_size.value
            min_samples = hdbscan_min_samples.value
            
            print(f"  min_cluster_size: {min_cluster_size}")
            print(f"  min_samples: {min_samples}")
            
            # Run HDBSCAN on current projection
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            
            cluster_labels = clusterer.fit_predict(state.projection)
            
            # HDBSCAN uses -1 for noise points
            cluster_labels_str = [str(label) if label >= 0 else "Noise" for label in cluster_labels]
            
            # Calculate metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = sum(1 for label in cluster_labels if label == -1)
            
            print(f"  Found {n_clusters} clusters")
            print(f"  Noise points: {n_noise} ({100*n_noise/len(cluster_labels):.1f}%)")
            
            # Update the source data
            current_data = dict(source.data)
            current_data['hdbscan'] = cluster_labels.tolist()
            current_data['hdbscan_str'] = cluster_labels_str
            
            # Create color mapping for HDBSCAN
            unique_hdbscan = sorted(set(cluster_labels_str))
            hdbscan_palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", 
                            "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
            
            hdbscan_color_map = {}
            for i, label in enumerate(unique_hdbscan):
                if label == "Noise":
                    hdbscan_color_map[label] = "#CCCCCC"  # Gray for noise
                else:
                    hdbscan_color_map[label] = hdbscan_palette[i % len(hdbscan_palette)]
            
            current_data['hdbscan_color'] = [hdbscan_color_map[str(c)] for c in cluster_labels_str]
            current_data['hdbscan_on'] = [True] * len(cluster_labels)
            
            # Update HDBSCAN checkboxes
            hdbscan_checks.labels = unique_hdbscan
            hdbscan_checks.active = list(range(len(unique_hdbscan)))
            
            source.data = current_data
            apply_dedupe_filter()
            
            # Update stats
            hdbscan_stats_div.text = f"<b>HDBSCAN:</b> {n_clusters} clusters, {n_noise} noise points ({100*n_noise/len(cluster_labels):.1f}%)"

        # HDBSCAN checkbox callback
        hdbscan_callback = CustomJS(args=dict(src=source, cb=hdbscan_checks,
                                umap_view=umap_view, map_view=map_view), code="""
            const d = src.data;
            const labs = cb.labels;
            const active_indices = cb.active;
            
            // Build set of active labels
            const active = new Set();
            for (let idx of active_indices) {
                if (idx < labs.length) {
                    active.add(labs[idx]);
                }
            }
            
            const hdbscan = d['hdbscan_str'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const season_on = d['season_on'] || new Array(hdbscan.length).fill(true);
            const sex_on = d['sex_on'] || new Array(hdbscan.length).fill(true);
            const type_on = d['type_on'] || new Array(hdbscan.length).fill(true);
            const time_on = d['time_on'] || new Array(hdbscan.length).fill(true);
            const selection_on = d['selection_on'] || new Array(hdbscan.length).fill(true);
            const dedupe_on = d['dedupe_on'] || new Array(hdbscan.length).fill(true);
            const region_on = d['region_on'] || new Array(hdbscan.length).fill(true);

            const n = hdbscan.length;
            for (let i = 0; i < n; i++) {
                const hdbscan_visible = active.has(String(hdbscan[i]));
                d['hdbscan_on'][i] = hdbscan_visible;

                if (season_on[i] && sex_on[i] && type_on[i] &&
                    time_on[i] && selection_on[i] && dedupe_on[i] && region_on[i] && hdbscan_visible) {
                    alpha[i] = alpha_base[i];
                } else {
                    alpha[i] = 0.0;
                }
            }
            
            // Update views
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }
            src.change.emit();
        """)
        hdbscan_checks.js_on_change('active', hdbscan_callback)

        # Add callback for apply button
        hdbscan_apply_btn.on_click(compute_hdbscan_clustering)
        
        print("  All callbacks set up")
        
        # --- LAYOUT ---
        print("\n" + "-" * 40)
        print("CREATING LAYOUT...")
        
        # Create UMAP parameters box
        umap_params_box = column(
            umap_params_div,
            umap_neighbors_input,
            umap_mindist_input,
            styles={'border': '1px solid #ddd', 'padding': '10px', 'border-radius': '5px'}
        )
        
        # Create HDBSCAN parameters box
        hdbscan_params_box = column(
            hdbscan_params_div,
            hdbscan_min_cluster_size,
            hdbscan_min_samples,
            hdbscan_apply_btn,
            hdbscan_stats_div,
            styles={'border': '1px solid #ddd', 'padding': '10px', 'border-radius': '5px'}
        )
        
        zoom_controls = row(
            zoom_button,
            reset_button,
            alpha_toggle,
            point_alpha_spinner,
            zoom_status,
        )

        dedupe_controls = column(
            dedupe_toggle,
            dedupe_distance_spinner,
            dedupe_days_spinner,
            dedupe_umap_spinner,
            dedupe_status_div,
            width=260,
            styles={
                'border': '1px solid #ddd',
                'padding': '10px',
                'border-radius': '5px'
            }
        )

        # Create a column that contains all filter widgets
        filter_widgets = column(
            *region_widgets_flat,
            selection_help_div,
            selection_status_div,
            selection_create_btn,
            selection_save_vocal_btn,
            selection_save_dialect_btn,
            selection_save_annotate1_btn,
            selection_save_annotate2_btn,
            selection_load_vocal_btn,
            selection_load_dialect_btn,
            selection_load_annotate1_btn,
            selection_load_annotate2_btn,
            selection_checks,
            selection_clear_btn,
            season_checks,
            hdbscan_checks,
            sex_checks,
            type_checks,
            dedupe_controls,
            time_range_slider
        )
        geoshift_box = column(
            geoshift_percent_input,
            geoshift_button,
            clear_geoshift_button,
            geoshift_summary_div,
            width=300,
            styles={'border': '1px solid #ddd', 'padding': '10px', 'border-radius': '5px'}
        )
        geoshift_plot_column = column(
            north_trend_fig,
            south_trend_fig,
            width=340,
            styles={'border': '1px solid #ddd', 'padding': '10px', 'border-radius': '5px'}
        )
        geoshift_layout = row(geoshift_box, geoshift_plot_column)
        controls = row(color_select, filter_widgets, hover_toggle, test_audio_btn, umap_params_box, hdbscan_params_box, geoshift_layout)
        playlist_column = column(
            playlist_help_div,
            playlist_panel,
            width=440,
        )

        plots = row(umap_plot, map_plot, playlist_column)
        date_bounds_controls = row(date_bounds_slider, reset_bounds_btn)
        
        layout = column(
            stats_div,  # Statistics at the top
            zoom_controls,
            controls,
            audio_status,
            date_bounds_controls,  # Timeline zoom control
            date_slider,  # Main date filter
            plots,
            sizing_mode="stretch_width"
        )
        
        print("  Layout created successfully")
        print("\n" + "=" * 80)
        print("BOKEH APP READY!")
        print("=" * 80)
        
        return layout
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        error_div = Div(text=f"<h2>Error: {str(e)}</h2>", width=800, height=600)
        return column(error_div)

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

doc = curdoc()
doc.title = f"UMAP {config['species']['common_name'] or config['species']['scientific_name']} Visualization"
layout = create_app()
doc.add_root(layout)

print("\nServer ready at: http://localhost:5006/umap_yellowhammer_app")

#   bokeh serve --show umap_yellowhammer_app.py
