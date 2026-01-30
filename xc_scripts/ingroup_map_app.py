"""
Interactive ingroup explorer with a Mercator map and spectrogram previews.

Run (from repository root):
    bokeh serve xc_scripts/ingroup_map_app.py --show --args \
        --config xc_configs/config_chloris_chloris.yaml \
        --ingroup-size 300

Optional overrides:
    --ingroup-csv /path/to/ingroup_energy_with_meta.csv
    --topk-csv /path/to/topk_table_meta_annotated.csv
    --spectrogram-dir /path/to/spectrograms/<species_slug>
    --top-n 50
    --sort desc

Config tweaks (config yaml):
    map:
        auto_zoom: false  # Keep map extent fixed when switching ingroups
    
    For me:
    
    bokeh serve xc_scripts/ingroup_map_app.py --show --args \
    --config xc_configs_perch/config_chloris_chloris.yaml \
    --ingroup-csv "/Users/masjansma/Desktop/temp_edvecs/ingroup_energy_with_meta.csv" \
    --topk-csv "/Users/masjansma/Desktop/temp_edvecs/topk_table_meta_annotated.csv" \
    --spectrogram-dir "/Volumes/Z Slim/zslim_birdcluster/spectrograms/chloris_chloris"
    
        bokeh serve xc_scripts/ingroup_map_app.py --show --args \
    --config xc_configs_perch/config_carduelis_carduelis.yaml \
    --ingroup-csv "/Users/masjansma/Desktop/temp_edvecs/ingroup_energy_with_meta.csv" \
    --topk-csv "/Users/masjansma/Desktop/temp_edvecs/topk_table_meta_annotated.csv" \
    --spectrogram-dir "/Volumes/Z Slim/zslim_birdcluster/spectrograms/carduelis_carduelis"
    
"""

from __future__ import annotations

import argparse
import ast
import base64
import html
import json
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    Div,
    HoverTool,
    HTMLTemplateFormatter,
    NumberFormatter,
    Select,
    Spinner,
    TableColumn,
)
from bokeh.plotting import figure

print("=" * 80)
print("STARTING BOKEH SERVER APP - INGROUP MAP VIEWER")
print("=" * 80)

_STATIC_HTTP_SERVERS: dict[str, dict[str, Any]] = {}


def load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load configuration from file or use defaults."""
    default_config: dict[str, Any] = {
        "species": {
            "scientific_name": "Unknown",
            "common_name": "Unknown",
            "slug": "unknown_species",
        },
        "paths": {
            "root": str(Path.cwd()),
        },
        "spectrograms": {
            "auto_serve": True,
            "host": "127.0.0.1",
            "port": 8766,
            "base_url": None,
            "image_format": "png",
            "inline": False,
        },
        "audio": {
            "auto_serve": True,
            "host": "127.0.0.1",
            "port": 8765,
            "base_url": None,
        },
        "map": {
            "auto_zoom": False,
        },
    }

    if config_path is None:
        return default_config

    requested_path = config_path
    resolved_path = (
        config_path if config_path.is_absolute() else (Path.cwd() / config_path)
    )
    if not resolved_path.exists():
        raise SystemExit(
            "Config file "
            f"'{requested_path}' not found. Use '--config xc_configs/<name>.yaml'."
        )

    with resolved_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merged[key] = _merge(base[key], value)
            else:
                merged[key] = value
        return merged

    return _merge(default_config, config)


def start_static_file_server(
    *,
    label: str,
    directory: Path,
    host: str,
    port: int,
    log_requests: bool = False,
) -> Optional[tuple[str, bool]]:
    """Start or reuse a background HTTP server to expose local files."""
    existing = _STATIC_HTTP_SERVERS.get(label)
    if existing:
        return existing["base_url"], existing["started"]

    if not directory.exists():
        print(f"  {label} directory missing, cannot auto-serve: {directory}")
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


def normalize_identifier(value: Any) -> str:
    """Normalize identifiers to stable string keys."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return text


def resolve_csv_path(
    *,
    cli_value: Optional[Path],
    config_paths: dict[str, Any],
    root_path: Path,
    species_slug: str,
    filename: str,
    arg_label: str,
    config_key: str,
) -> Path:
    """Resolve a CSV path from CLI, config, or known locations."""
    if cli_value:
        return cli_value.expanduser()

    config_value = config_paths.get(config_key)
    if not config_value:
        config_value = config_paths.get(filename) or config_paths.get(
            filename.replace(".csv", "")
        )
    if config_value:
        return Path(config_value).expanduser()

    candidates = [
        root_path / "analysis" / species_slug / filename,
        root_path / "analysis" / filename,
        root_path / "embeddings" / species_slug / filename,
        root_path / species_slug / filename,
        root_path / filename,
        Path.cwd() / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise SystemExit(
        f"Could not find {filename}. Provide --{arg_label} or set "
        f"paths.{config_key} in the config."
    )


def resolve_directory_path(
    *,
    cli_value: Optional[Path],
    config_paths: dict[str, Any],
    root_path: Path,
    species_slug: str,
    key: str,
    fallback: Path,
) -> Path:
    """Resolve a directory path from CLI, config, or fallback."""
    if cli_value:
        return cli_value.expanduser()
    config_value = config_paths.get(key)
    if config_value:
        return Path(config_value).expanduser()
    return fallback


def normalize_ingroup_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ingroup metadata column names and types."""
    columns = list(df.columns)
    if not columns:
        raise ValueError("Ingroup CSV has no columns.")

    lower_map = {str(col).lower(): col for col in columns}

    def _pick(name: str, index: int) -> str:
        return lower_map.get(name) or columns[index]

    rename_map: dict[str, str] = {}
    for expected, idx in [
        ("source_windowid", 0),
        ("ingroup_size", 1),
        ("outgroup_size", 2),
        ("energy_distance", 3),
    ]:
        if expected in lower_map:
            rename_map[lower_map[expected]] = expected
        elif idx < len(columns):
            rename_map[columns[idx]] = expected
        else:
            raise ValueError(f"Missing expected column for {expected}.")

    df = df.rename(columns=rename_map)
    df["source_windowid"] = df["source_windowid"].apply(normalize_identifier)
    df["ingroup_size"] = (
        pd.to_numeric(df["ingroup_size"], errors="coerce").fillna(0).astype(int)
    )
    df["outgroup_size"] = (
        pd.to_numeric(df["outgroup_size"], errors="coerce").fillna(0).astype(int)
    )
    df["energy_distance"] = pd.to_numeric(df["energy_distance"], errors="coerce")
    return df


def parse_rank_cell(value: Any) -> Optional[dict[str, Any]]:
    """Parse a rank cell into a metadata dict."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        cleaned = text.replace("nan", "None").replace("NaN", "None")
        try:
            parsed = ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            return None
    if isinstance(parsed, dict):
        return parsed
    return None


def is_pruned(meta: Optional[dict[str, Any]]) -> bool:
    """Return True when metadata marks the entry as pruned."""
    if not meta:
        return False
    raw = meta.get("pruned")
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    text = str(raw).strip().lower()
    return text in {"true", "1", "yes", "y"}


def extract_source_id(meta: Optional[dict[str, Any]]) -> Optional[str]:
    """Extract a stable source identifier from rank metadata."""
    if not meta:
        return None
    for key in ("source_windowid", "windowid", "window_id", "id"):
        if key in meta:
            value = normalize_identifier(meta[key])
            if value:
                return value
    return None


def rank_sort_key(name: str) -> int:
    """Sort rank columns by numeric suffix."""
    match = re.search(r"rank_(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def lonlat_to_web_mercator(
    lon_values: list[float], lat_values: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert longitude/latitude pairs to Web Mercator coordinates."""
    lon_series = pd.to_numeric(pd.Series(lon_values), errors="coerce")
    lat_series = pd.to_numeric(pd.Series(lat_values), errors="coerce")
    lon_arr = lon_series.to_numpy(dtype=float)
    lat_arr = lat_series.to_numpy(dtype=float)
    valid = np.isfinite(lon_arr) & np.isfinite(lat_arr)
    x = np.full(lon_arr.shape, np.nan, dtype=float)
    y = np.full(lat_arr.shape, np.nan, dtype=float)
    if np.any(valid):
        k = 6378137.0
        lon_rad = np.deg2rad(lon_arr[valid])
        lat_rad = np.deg2rad(lat_arr[valid])
        x[valid] = k * lon_rad
        y[valid] = k * np.log(np.tan((np.pi / 4.0) + (lat_rad / 2.0)))
    return x, y


def build_spectrogram_url(
    *,
    filename: str,
    spectrogram_dir: Path,
    image_format: str,
    base_url: Optional[str],
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
        except Exception as exc:
            print(f"  Warning: failed to inline {image_name}: {exc}")
            return "", False
    if base_url:
        return f"{base_url.rstrip('/')}/{image_name}", True
    return "", False


def build_media_url(base_url: Optional[str], species_slug: str, filename: str) -> str:
    """Construct a media URL, handling base URLs with or without species suffixes."""
    if not base_url or not filename:
        return ""
    normalized = base_url.rstrip("/")
    suffix = f"/{species_slug}"
    if normalized.endswith(suffix):
        return f"{normalized}/{filename}"
    return f"{normalized}{suffix}/{filename}"


def coerce_clip_index(value: Any) -> Optional[int]:
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


def build_metadata_html(
    entries: list[dict[str, Any]],
    *,
    spectrogram_dir: Path,
    image_format: str,
    base_url: Optional[str],
    inline: bool,
    audio_base_url: Optional[str],
    species_slug: str,
    spectrogram_urls: Optional[list[str]] = None,
) -> str:
    """Build HTML for the metadata list, reusing spectrogram URLs if provided."""
    if not entries:
        return "<i>No ingroup entries available.</i>"

    rows: list[str] = []
    for idx, entry in enumerate(entries, start=1):
        filename = html.escape(str(entry.get("filename", "") or "").strip())
        recordist = html.escape(str(entry.get("recordist", "") or "").strip())
        date = html.escape(str(entry.get("date", "") or "").strip())
        xcid = html.escape(str(entry.get("xcid", "") or "").strip())
        clip_index = html.escape(str(entry.get("clip_index", "") or "").strip())

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

        audio_url = ""
        if audio_base_url:
            audio_filename = build_audio_filename(entry, species_slug)
            if audio_filename:
                audio_url = build_media_url(
                    audio_base_url, species_slug, audio_filename
                )
        if audio_url:
            safe_audio = json.dumps(audio_url)
            play_button = (
                "<button style='margin-right:6px; padding:2px 6px;' "
                f"onclick='(function(u){{"
                "if(!u){return;}"
                "if(window._ingroup_audio){window._ingroup_audio.pause();}"
                "window._ingroup_audio=new Audio(u);"
                "window._ingroup_audio.play();"
                f"}})({safe_audio})'>Play</button>"
            )
        else:
            play_button = (
                "<button style='margin-right:6px; padding:2px 6px;' "
                "disabled title='Audio not available'>Play</button>"
            )

        url = ""
        exists = False
        if spectrogram_urls is not None and idx - 1 < len(spectrogram_urls):
            url = spectrogram_urls[idx - 1]
            exists = bool(url)
        else:
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
                "style='width: 260px; max-width: 100%; border: 1px solid #ddd;'>"
            )
        else:
            img_html = "<div style='color:#888;'>No spectrogram</div>"

        rows.append(
            "<div style='margin-bottom: 12px;'>"
            "<div style='margin-bottom: 4px; display:flex; align-items:center; "
            "gap:6px; flex-wrap:wrap;'>"
            f"{play_button}<div>{' | '.join(meta_bits)}</div>"
            "</div>"
            f"{img_html}"
            "</div>"
        )

    return "<div style='overflow-y: auto; height: 680px;'>" + "".join(rows) + "</div>"


def empty_points_data() -> dict[str, list[Any]]:
    """Return an empty payload for the points source."""
    return {
        "x": [],
        "y": [],
        "lat": [],
        "lon": [],
        "rank": [],
        "filename": [],
        "recordist": [],
        "date": [],
        "spectrogram_url": [],
    }


def update_map_range(map_fig: Any, x_values: np.ndarray, y_values: np.ndarray) -> None:
    """Auto-zoom the map to the current points."""
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(valid):
        return
    x_valid = x_values[valid]
    y_valid = y_values[valid]
    x_min, x_max = float(np.min(x_valid)), float(np.max(x_valid))
    y_min, y_max = float(np.min(y_valid)), float(np.max(y_valid))
    if np.isclose(x_min, x_max):
        x_min -= 50000
        x_max += 50000
    if np.isclose(y_min, y_max):
        y_min -= 50000
        y_max += 50000
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05
    map_fig.x_range.start = x_min - pad_x
    map_fig.x_range.end = x_max + pad_x
    map_fig.y_range.start = y_min - pad_y
    map_fig.y_range.end = y_max + pad_y


def set_initial_map_range(map_fig: Any, config_data: dict[str, Any]) -> None:
    """Set an initial map range using config bounds when available."""
    bbox = (config_data.get("xeno_canto", {}) or {}).get("bounding_box", {}) or {}
    lon_min = float(bbox.get("lon_min", -180.0))
    lon_max = float(bbox.get("lon_max", 180.0))
    lat_min = float(bbox.get("lat_min", -60.0))
    lat_max = float(bbox.get("lat_max", 80.0))
    lat_min = max(min(lat_min, 85.0), -85.0)
    lat_max = max(min(lat_max, 85.0), -85.0)
    x_vals, y_vals = lonlat_to_web_mercator([lon_min, lon_max], [lat_min, lat_max])
    if np.isfinite(x_vals).all() and np.isfinite(y_vals).all():
        map_fig.x_range.start = float(min(x_vals))
        map_fig.x_range.end = float(max(x_vals))
        map_fig.y_range.start = float(min(y_vals))
        map_fig.y_range.end = float(max(y_vals))


parser = argparse.ArgumentParser(description="Ingroup map viewer")
parser.add_argument("--config", type=Path, help="Path to config.yaml file")
parser.add_argument(
    "--ingroup-csv", type=Path, help="Path to ingroup_energy_with_metadata.csv"
)
parser.add_argument(
    "--topk-csv", type=Path, help="Path to topk_table_meta_annotated.csv"
)
parser.add_argument(
    "--spectrogram-dir",
    type=Path,
    help="Override spectrogram directory (default: <root>/spectrograms/<slug>)",
)
parser.add_argument(
    "--ingroup-size",
    type=int,
    default=300,
    help="Ingroup size to filter on",
)
parser.add_argument(
    "--top-n",
    type=int,
    default=50,
    help="Number of ingroups to display (0 shows all)",
)
parser.add_argument(
    "--sort",
    choices=["desc", "asc"],
    default="desc",
    help="Sort order for energy distance",
)
args = parser.parse_args()

config = load_config(args.config)
paths_cfg = config.get("paths", {}) or {}
ROOT_PATH = Path(paths_cfg.get("root", Path.cwd())).expanduser()
SPECIES_SLUG = str(config.get("species", {}).get("slug", "unknown_species"))

INGROUP_CSV = resolve_csv_path(
    cli_value=args.ingroup_csv,
    config_paths=paths_cfg,
    root_path=ROOT_PATH,
    species_slug=SPECIES_SLUG,
    filename="ingroup_energy_with_metadata.csv",
    arg_label="ingroup-csv",
    config_key="ingroup_csv",
)
TOPK_CSV = resolve_csv_path(
    cli_value=args.topk_csv,
    config_paths=paths_cfg,
    root_path=ROOT_PATH,
    species_slug=SPECIES_SLUG,
    filename="topk_table_meta_annotated.csv",
    arg_label="topk-csv",
    config_key="topk_csv",
)

SPECTROGRAMS_DIR = resolve_directory_path(
    cli_value=args.spectrogram_dir,
    config_paths=paths_cfg,
    root_path=ROOT_PATH,
    species_slug=SPECIES_SLUG,
    key="spectrograms_dir",
    fallback=ROOT_PATH / "spectrograms" / SPECIES_SLUG,
)

spectro_cfg = config.get("spectrograms", {}) or {}
SPECTROGRAM_IMAGE_FORMAT = str(spectro_cfg.get("image_format", "png")).lstrip(".")
INLINE_SPECTROGRAMS = bool(spectro_cfg.get("inline", False))
SPECTROGRAM_BASE_URL = spectro_cfg.get("base_url")
audio_cfg = config.get("audio", {}) or {}
_audio_base = audio_cfg.get("base_url")
AUDIO_BASE_URL = (
    str(_audio_base).rstrip("/")
    if isinstance(_audio_base, str) and _audio_base
    else None
)
AUDIO_HOST = str(audio_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
AUDIO_PORT = int(audio_cfg.get("port", 8765))
AUDIO_AUTO_SERVE_REQUESTED = bool(audio_cfg.get("auto_serve", True))
map_cfg = config.get("map", {}) or {}
AUTO_ZOOM = bool(map_cfg.get("auto_zoom", False))

if not SPECTROGRAM_BASE_URL and bool(spectro_cfg.get("auto_serve", True)):
    generated = start_static_file_server(
        label=f"Spectrograms ({SPECIES_SLUG})",
        directory=SPECTROGRAMS_DIR,
        host=str(spectro_cfg.get("host", "127.0.0.1")),
        port=int(spectro_cfg.get("port", 8766)),
        log_requests=bool(spectro_cfg.get("log_requests", False)),
    )
    if generated:
        SPECTROGRAM_BASE_URL = generated[0]

if AUDIO_BASE_URL is None and AUDIO_AUTO_SERVE_REQUESTED:
    generated_audio_url = start_static_file_server(
        label=f"Audio ({SPECIES_SLUG})",
        directory=ROOT_PATH / "clips",
        host=AUDIO_HOST,
        port=AUDIO_PORT,
        log_requests=bool(audio_cfg.get("log_requests", False)),
    )
    if generated_audio_url:
        audio_base, _ = generated_audio_url
        AUDIO_BASE_URL = audio_base.rstrip("/")

print("Configuration loaded:")
print(f"  Root: {ROOT_PATH}")
print(f"  Species slug: {SPECIES_SLUG}")
print(f"  Ingroup CSV: {INGROUP_CSV}")
print(f"  Top-k CSV: {TOPK_CSV}")
print(f"  Spectrograms: {SPECTROGRAMS_DIR}")
print(f"  Audio base URL: {AUDIO_BASE_URL}")

if not INGROUP_CSV.exists():
    raise SystemExit(f"Ingroup CSV not found: {INGROUP_CSV}")
if not TOPK_CSV.exists():
    raise SystemExit(f"Top-k CSV not found: {TOPK_CSV}")

ingroup_df = normalize_ingroup_df(pd.read_csv(INGROUP_CSV))
topk_df = pd.read_csv(TOPK_CSV)

rank_columns = sorted(
    [col for col in topk_df.columns if str(col).lower().startswith("rank_")],
    key=rank_sort_key,
)
if not rank_columns:
    raise SystemExit("Top-k table has no rank_* columns.")

source_lookup: dict[str, int] = {}
source_meta_lookup: dict[str, dict[str, Any]] = {}

for idx, raw in enumerate(topk_df[rank_columns[0]].tolist()):
    meta = parse_rank_cell(raw)
    source_id = extract_source_id(meta)
    if not source_id:
        continue
    source_lookup[source_id] = idx
    if meta:
        source_meta_lookup[source_id] = meta

if "filename" in ingroup_df.columns:
    ingroup_df["source_filename"] = ingroup_df["filename"]
else:
    ingroup_df["source_filename"] = ingroup_df["source_windowid"].map(
        lambda sid: source_meta_lookup.get(sid, {}).get("filename", "")
    )

available_sizes = sorted(ingroup_df["ingroup_size"].unique().tolist())
initial_size = args.ingroup_size
if available_sizes and initial_size not in available_sizes:
    initial_size = available_sizes[0]

table_source = ColumnDataSource(
    data={
        "source_windowid": [],
        "source_filename": [],
        "ingroup_size": [],
        "outgroup_size": [],
        "energy_distance": [],
    }
)

points_source = ColumnDataSource(
    data=empty_points_data()
)

status_div = Div(text="<i>Select an ingroup to display its map and metadata.</i>")
stats_div = Div(text="")
metadata_div = Div(
    text="<i>No ingroup selected.</i>",
    width=360,
)
SPECTROGRAM_TEMPLATE = """
<div style="display:flex; align-items:center; justify-content:center;">
<% if (value) { %>
  <img src="<%= value %>" style="width: 110px; max-height: 60px; border: 1px solid #ddd;">
<% } else { %>
  <span style="color:#888;">No spectrogram</span>
<% } %>
</div>
"""
spectrogram_formatter = HTMLTemplateFormatter(template=SPECTROGRAM_TEMPLATE)
metadata_columns = [
    TableColumn(field="rank", title="Rank", width=60),
    TableColumn(field="filename", title="Filename", width=180),
    TableColumn(field="recordist", title="Recordist", width=120),
    TableColumn(field="date", title="Date", width=90),
    TableColumn(
        field="spectrogram_url",
        title="Spectrogram",
        formatter=spectrogram_formatter,
        width=140,
    ),
]
metadata_table = DataTable(
    source=points_source,
    columns=metadata_columns,
    width=360,
    height=220,
    row_height=70,
    selectable=True,
    index_position=None,
)


def refresh_ingroup_table() -> None:
    """Refresh the ingroup list based on filters."""
    size_value = int(size_spinner.value)
    sort_desc = sort_select.value == "desc"
    top_n_value = int(top_n_spinner.value)

    filtered = ingroup_df[ingroup_df["ingroup_size"] == size_value].copy()
    filtered = filtered.sort_values(
        "energy_distance", ascending=not sort_desc, kind="mergesort"
    )
    if top_n_value > 0:
        filtered = filtered.head(top_n_value)

    table_source.data = {
        "source_windowid": filtered["source_windowid"].tolist(),
        "source_filename": filtered["source_filename"].fillna("").tolist(),
        "ingroup_size": filtered["ingroup_size"].tolist(),
        "outgroup_size": filtered["outgroup_size"].tolist(),
        "energy_distance": filtered["energy_distance"].tolist(),
    }

    stats_div.text = (
        f"<b>{len(filtered)}</b> ingroups "
        f"(size={size_value}, sort={sort_select.value})"
    )
    table_source.selected.indices = []
    points_source.data = empty_points_data()
    points_source.selected.indices = []
    metadata_div.text = "<i>No ingroup selected.</i>"


def build_ingroup_entries(source_id: str, max_entries: int) -> list[dict[str, Any]]:
    """Load ranked entries for a specific ingroup."""
    row_index = source_lookup.get(source_id)
    if row_index is None:
        return []
    row = topk_df.loc[row_index, rank_columns]
    entries: list[dict[str, Any]] = []
    for col in rank_columns:
        if len(entries) >= max_entries:
            break
        meta = parse_rank_cell(row[col])
        if meta and not is_pruned(meta):
            entries.append(meta)
    return entries


def update_map_and_metadata(source_id: str, max_entries: int) -> None:
    """Update map points and metadata list for a selected ingroup."""
    entries = build_ingroup_entries(source_id, max_entries)
    if not entries:
        status_div.text = f"<b>No entries found for source {html.escape(source_id)}</b>"
        points_source.data = empty_points_data()
        points_source.selected.indices = []
        metadata_div.text = "<i>No ingroup entries available.</i>"
        return

    lat_values: list[float] = []
    lon_values: list[float] = []
    ranks: list[int] = []
    filenames: list[str] = []
    recordists: list[str] = []
    dates: list[str] = []
    spectrogram_urls: list[str] = []

    for idx, entry in enumerate(entries, start=1):
        lat_values.append(entry.get("lat"))
        lon_values.append(entry.get("lon"))
        ranks.append(idx)
        filenames.append(str(entry.get("filename", "") or ""))
        recordists.append(str(entry.get("recordist", "") or ""))
        dates.append(str(entry.get("date", "") or ""))
        url, exists = build_spectrogram_url(
            filename=entry.get("filename", "") or "",
            spectrogram_dir=SPECTROGRAMS_DIR,
            image_format=SPECTROGRAM_IMAGE_FORMAT,
            base_url=SPECTROGRAM_BASE_URL,
            inline=INLINE_SPECTROGRAMS,
        )
        spectrogram_urls.append(url if exists else "")

    x_vals, y_vals = lonlat_to_web_mercator(lon_values, lat_values)
    points_source.data = {
        "x": x_vals.tolist(),
        "y": y_vals.tolist(),
        "lat": lat_values,
        "lon": lon_values,
        "rank": ranks,
        "filename": filenames,
        "recordist": recordists,
        "date": dates,
        "spectrogram_url": spectrogram_urls,
    }
    points_source.selected.indices = []
    metadata_div.text = build_metadata_html(
        entries,
        spectrogram_dir=SPECTROGRAMS_DIR,
        image_format=SPECTROGRAM_IMAGE_FORMAT,
        base_url=SPECTROGRAM_BASE_URL,
        inline=INLINE_SPECTROGRAMS,
        audio_base_url=AUDIO_BASE_URL,
        species_slug=SPECIES_SLUG,
        spectrogram_urls=spectrogram_urls,
    )
    status_div.text = (
        f"<b>Ingroup {html.escape(source_id)}</b>: {len(entries)} entries"
    )
    if AUTO_ZOOM:
        update_map_range(map_plot, x_vals, y_vals)


def on_table_selection(attr: str, old: list[int], new: list[int]) -> None:
    """Handle ingroup selection changes."""
    if not new:
        status_div.text = "<i>Select an ingroup to display its map and metadata.</i>"
        points_source.data = empty_points_data()
        points_source.selected.indices = []
        metadata_div.text = "<i>No ingroup selected.</i>"
        return

    if len(new) > 1:
        table_source.selected.indices = [new[0]]
        return

    idx = new[0]
    source_id = table_source.data["source_windowid"][idx]
    ingroup_size = int(table_source.data["ingroup_size"][idx])
    update_map_and_metadata(str(source_id), ingroup_size)


size_low = min(available_sizes) if available_sizes else 1
size_high = max(available_sizes) if available_sizes else max(1, initial_size)
size_value = initial_size if size_low <= initial_size <= size_high else size_low

size_spinner = Spinner(
    title="Ingroup size",
    low=size_low,
    high=size_high,
    step=1,
    value=size_value,
    width=180,
)
size_spinner.on_change("value", lambda attr, old, new: refresh_ingroup_table())

top_n_high = max(len(ingroup_df), args.top_n, 1)
top_n_value = min(max(args.top_n, 0), top_n_high)
top_n_spinner = Spinner(
    title="Top N",
    low=0,
    high=top_n_high,
    step=1,
    value=top_n_value,
    width=180,
)
top_n_spinner.on_change("value", lambda attr, old, new: refresh_ingroup_table())

sort_select = Select(
    title="Sort by energy distance",
    value=args.sort,
    options=["desc", "asc"],
    width=180,
)
sort_select.on_change("value", lambda attr, old, new: refresh_ingroup_table())

table_columns = [
    TableColumn(field="source_windowid", title="Source ID"),
    TableColumn(field="source_filename", title="Source filename"),
    TableColumn(field="ingroup_size", title="Ingroup"),
    TableColumn(field="outgroup_size", title="Outgroup"),
    TableColumn(
        field="energy_distance",
        title="Energy distance",
        formatter=NumberFormatter(format="0.000"),
    ),
]

ingroup_table = DataTable(
    source=table_source,
    columns=table_columns,
    width=520,
    height=520,
    selectable=True,
    index_position=None,
)
table_source.selected.on_change("indices", on_table_selection)

map_plot = figure(
    title="Ingroup map",
    x_axis_type="mercator",
    y_axis_type="mercator",
    width=620,
    height=520,
    tools="pan,wheel_zoom,reset,save",
    active_scroll="wheel_zoom",
)
set_initial_map_range(map_plot, config)
map_plot.add_tile("CartoDB Positron", retina=True)

map_plot.scatter(
    "x",
    "y",
    source=points_source,
    size=8,
    color="#2a9d8f",
    alpha=0.8,
    selection_color="#e76f51",
    selection_alpha=1.0,
    selection_line_color="#e76f51",
    selection_line_width=2,
    nonselection_alpha=0.2,
)

map_hover = HoverTool(
    tooltips=[
        ("rank", "@rank"),
        ("filename", "@filename"),
        ("recordist", "@recordist"),
        ("date", "@date"),
        ("lat, lon", "@lat, @lon"),
    ]
)
map_plot.add_tools(map_hover)

controls = column(size_spinner, top_n_spinner, sort_select, stats_div, status_div)
table_panel = column(Div(text="<b>Top ingroups</b>"), controls, ingroup_table)

metadata_panel = column(
    Div(text="<b>Ingroup metadata</b>"),
    metadata_table,
    metadata_div,
    width=380,
)
layout = row(table_panel, map_plot, metadata_panel, sizing_mode="scale_both")

curdoc().add_root(layout)
curdoc().title = f"Ingroup map - {SPECIES_SLUG}"

refresh_ingroup_table()
