"""
Interactive ingroup explorer with a Mercator map and spectrogram previews.

Run (from repository root):
    bokeh serve xc_scripts/ingroup_map_app.py --show --args \
        --config xc_configs/config_chloris_chloris.yaml \
        --ingroup-size 300

Optional overrides:
    --ingroup-csv /path/to/ingroup_energy_with_meta.csv
    --topk-csv /path/to/topk_table_meta_annotated.csv
    --inference-csv /path/to/inference.csv
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

Inference mode:
    bokeh serve xc_scripts/ingroup_map_app.py --show --args \
    --config xc_configs_perch/config_carduelis_carduelis.yaml \
    --inference-csv "/path/to/inference.csv" \
    --topk-csv "/Users/masjansma/Desktop/temp_edvecs/topk_table_meta_annotated.csv" \
    --spectrogram-dir "/Volumes/Z Slim/zslim_birdcluster/spectrograms/carduelis_carduelis"
    
    bokeh serve xc_scripts/ingroup_map_app_inferencecheck.py --show --args \
    --config xc_configs_perch/config_chloris_chloris.yaml \
    --inference-csv "/Volumes/Z Slim/zslim_birdcluster/embeddings/chloris_chloris/inference.csv" \
    --topk-csv "/Users/masjansma/Desktop/temp_edvecs/topk_table_meta_annotated.csv" \
    --spectrogram-dir "/Volumes/Z Slim/zslim_birdcluster/spectrograms/chloris_chloris"
    
"""

from __future__ import annotations

import argparse
import ast
import base64
import html
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
    RadioButtonGroup,
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


def resolve_optional_csv_path(
    *,
    cli_value: Optional[Path],
    config_paths: dict[str, Any],
    root_path: Path,
    species_slug: str,
    filename: str,
    arg_label: str,
    config_key: str,
) -> Optional[Path]:
    """Resolve a CSV path from CLI, config, or known locations when present."""
    if cli_value:
        resolved = cli_value.expanduser()
        if not resolved.exists():
            raise SystemExit(f"{arg_label} CSV not found: {resolved}")
        return resolved

    config_value = config_paths.get(config_key)
    if not config_value:
        config_value = config_paths.get(filename) or config_paths.get(
            filename.replace(".csv", "")
        )
    if config_value:
        resolved = Path(config_value).expanduser()
        if not resolved.exists():
            raise SystemExit(f"{arg_label} CSV not found: {resolved}")
        return resolved

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

    return None


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


def normalize_filename_key(value: Any) -> str:
    """Normalize filenames to stable lookup keys."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return Path(text).name.lower()


def normalize_inference_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize inference CSV column names and types."""
    columns = list(df.columns)
    if not columns:
        raise ValueError("Inference CSV has no columns.")
    lower_map = {str(col).lower(): col for col in columns}
    logit_column = None
    for candidate in ("logit_score", "logits"):
        if candidate in lower_map:
            logit_column = candidate
            break
    missing = [col for col in ("filename",) if col not in lower_map]
    if logit_column is None:
        missing.append("logit_score")
    if missing:
        raise ValueError(
            "Inference CSV missing required columns: " + ", ".join(missing)
        )
    rename_map = {
        lower_map["filename"]: "filename",
        lower_map[logit_column]: "logit_score",
    }
    df = df.rename(columns=rename_map)
    df["filename"] = df["filename"].apply(
        lambda value: str(value).strip() if pd.notna(value) else ""
    )
    df = df[df["filename"].astype(bool)].copy()
    df["logit_score"] = pd.to_numeric(df["logit_score"], errors="coerce")
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


def extract_meta_filename(meta: dict[str, Any]) -> str:
    """Extract a filename or file path from metadata when available."""
    for key in ("filename", "file_path", "filepath", "path"):
        value = meta.get(key)
        if value:
            return str(value)
    return ""


def is_finite_number(value: Any) -> bool:
    """Return True when value can be parsed into a finite float."""
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def has_coordinates(meta: dict[str, Any]) -> bool:
    """Return True when metadata includes finite lat/lon values."""
    return is_finite_number(meta.get("lat")) and is_finite_number(meta.get("lon"))


def build_metadata_lookup(
    topk_df: pd.DataFrame, rank_columns: list[str]
) -> dict[str, dict[str, Any]]:
    """Build a filename-keyed metadata lookup from top-k rank cells."""
    lookup: dict[str, dict[str, Any]] = {}
    for col in rank_columns:
        for raw in topk_df[col].tolist():
            meta = parse_rank_cell(raw)
            if not meta:
                continue
            filename = extract_meta_filename(meta)
            key = normalize_filename_key(filename)
            if not key:
                continue
            existing = lookup.get(key)
            if existing is None or (
                not has_coordinates(existing) and has_coordinates(meta)
            ):
                lookup[key] = meta
    return lookup


def build_inference_metadata(
    inference_df: pd.DataFrame,
    metadata_lookup: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, int]:
    """Attach metadata to inference rows using filename lookup."""
    rows: list[dict[str, Any]] = []
    missing = 0
    for _, row in inference_df.iterrows():
        filename = row["filename"]
        lookup_key = normalize_filename_key(filename)
        meta = metadata_lookup.get(lookup_key)
        if meta is None:
            missing += 1
            meta = {}
        rows.append(
            {
                "filename": filename or str(meta.get("filename", "") or ""),
                "logit_score": row["logit_score"],
                "recordist": str(meta.get("recordist", "") or ""),
                "date": str(meta.get("date", "") or ""),
                "lat": meta.get("lat"),
                "lon": meta.get("lon"),
                "xcid": str(meta.get("xcid", "") or ""),
                "clip_index": str(meta.get("clip_index", "") or ""),
            }
        )
    return pd.DataFrame(rows), missing


def apply_recordist_limit(
    entries_df: pd.DataFrame, max_per_recordist: Optional[int]
) -> pd.DataFrame:
    """Limit to the top-N logit scores per recordist when requested."""
    if entries_df.empty or not max_per_recordist:
        return entries_df.copy()
    working = entries_df.copy()
    working["_recordist"] = (
        working["recordist"].fillna("").astype(str).str.strip()
    )
    working["_score"] = pd.to_numeric(
        working["logit_score"], errors="coerce"
    ).fillna(-np.inf)
    working = working.sort_values(
        ["_recordist", "_score"], ascending=[True, False], kind="mergesort"
    )
    limited = working.groupby("_recordist", sort=False).head(max_per_recordist)
    return limited.drop(columns=["_recordist", "_score"])


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


def build_metadata_html(
    entries: list[dict[str, Any]],
    *,
    spectrogram_dir: Path,
    image_format: str,
    base_url: Optional[str],
    inline: bool,
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
        logit_text = ""
        raw_logit = entry.get("logit_score")
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
            f"<div style='margin-bottom: 4px;'>{' | '.join(meta_bits)}</div>"
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
        "logit_score": [],
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
    "--inference-csv",
    type=Path,
    help="Path to inference.csv with filename/logit_score columns",
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

INFERENCE_CSV = resolve_optional_csv_path(
    cli_value=args.inference_csv,
    config_paths=paths_cfg,
    root_path=ROOT_PATH,
    species_slug=SPECIES_SLUG,
    filename="inference.csv",
    arg_label="inference-csv",
    config_key="inference_csv",
)
USE_INFERENCE = INFERENCE_CSV is not None

INGROUP_CSV = None
if not USE_INFERENCE:
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

print("Configuration loaded:")
print(f"  Root: {ROOT_PATH}")
print(f"  Species slug: {SPECIES_SLUG}")
if USE_INFERENCE:
    print(f"  Inference CSV: {INFERENCE_CSV}")
else:
    print(f"  Ingroup CSV: {INGROUP_CSV}")
print(f"  Top-k CSV: {TOPK_CSV}")
print(f"  Spectrograms: {SPECTROGRAMS_DIR}")

if USE_INFERENCE:
    if INFERENCE_CSV is None or not INFERENCE_CSV.exists():
        raise SystemExit("Inference CSV not found. Provide --inference-csv.")
else:
    if INGROUP_CSV is None or not INGROUP_CSV.exists():
        raise SystemExit(f"Ingroup CSV not found: {INGROUP_CSV}")
if not TOPK_CSV.exists():
    raise SystemExit(f"Top-k CSV not found: {TOPK_CSV}")

topk_df = pd.read_csv(TOPK_CSV)
inference_df: pd.DataFrame | None = None
ingroup_df: pd.DataFrame | None = None
if USE_INFERENCE:
    inference_df = normalize_inference_df(pd.read_csv(INFERENCE_CSV))
else:
    ingroup_df = normalize_ingroup_df(pd.read_csv(INGROUP_CSV))

rank_columns = sorted(
    [col for col in topk_df.columns if str(col).lower().startswith("rank_")],
    key=rank_sort_key,
)
if not rank_columns:
    raise SystemExit("Top-k table has no rank_* columns.")

source_lookup: dict[str, int] = {}
source_meta_lookup: dict[str, dict[str, Any]] = {}

if not USE_INFERENCE:
    for idx, raw in enumerate(topk_df[rank_columns[0]].tolist()):
        meta = parse_rank_cell(raw)
        source_id = extract_source_id(meta)
        if not source_id:
            continue
        source_lookup[source_id] = idx
        if meta:
            source_meta_lookup[source_id] = meta

    if ingroup_df is None:
        raise SystemExit("Ingroup data could not be loaded.")
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

PLAYLIST_LIMIT = 1000
inference_entries: pd.DataFrame | None = None
inference_missing_meta = 0
if USE_INFERENCE:
    if inference_df is None:
        raise SystemExit("Inference data could not be loaded.")
    metadata_lookup = build_metadata_lookup(topk_df, rank_columns)
    inference_entries, inference_missing_meta = build_inference_metadata(
        inference_df, metadata_lookup
    )

table_source = ColumnDataSource(
    data={
        "source_windowid": [],
        "source_filename": [],
        "ingroup_size": [],
        "outgroup_size": [],
        "energy_distance": [],
    }
)

points_source = ColumnDataSource(data=empty_points_data())

status_text = (
    "<i>Select an ingroup to display its map and metadata.</i>"
    if not USE_INFERENCE
    else "<i>Preparing inference view...</i>"
)
metadata_text = (
    "<i>No ingroup selected.</i>"
    if not USE_INFERENCE
    else "<i>No inference entries available.</i>"
)
status_div = Div(text=status_text)
stats_div = Div(text="")
metadata_div = Div(text=metadata_text, width=360)
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
    TableColumn(
        field="logit_score",
        title="Logit",
        formatter=NumberFormatter(format="0.000"),
        width=70,
    ),
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
    logit_scores: list[float] = []
    spectrogram_urls: list[str] = []

    for idx, entry in enumerate(entries, start=1):
        lat_values.append(entry.get("lat"))
        lon_values.append(entry.get("lon"))
        ranks.append(idx)
        filenames.append(str(entry.get("filename", "") or ""))
        recordists.append(str(entry.get("recordist", "") or ""))
        dates.append(str(entry.get("date", "") or ""))
        logit_scores.append(np.nan)
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
        "logit_score": logit_scores,
        "spectrogram_url": spectrogram_urls,
    }
    points_source.selected.indices = []
    metadata_div.text = build_metadata_html(
        entries,
        spectrogram_dir=SPECTROGRAMS_DIR,
        image_format=SPECTROGRAM_IMAGE_FORMAT,
        base_url=SPECTROGRAM_BASE_URL,
        inline=INLINE_SPECTROGRAMS,
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


def recordist_limit_value() -> Optional[int]:
    """Return the current recordist limit, if any."""
    if recordist_filter.active is None:
        return None
    label = recordist_filter.labels[recordist_filter.active]
    return int(label) if label.isdigit() else None


def update_inference_view() -> None:
    """Update map points and metadata for inference entries."""
    if inference_entries is None or inference_entries.empty:
        status_div.text = "<i>No inference entries available.</i>"
        points_source.data = empty_points_data()
        points_source.selected.indices = []
        metadata_div.text = "<i>No inference entries available.</i>"
        return

    max_per_recordist = recordist_limit_value()
    filtered = apply_recordist_limit(inference_entries, max_per_recordist)
    if filtered.empty:
        status_div.text = "<i>No inference entries after recordist filter.</i>"
        points_source.data = empty_points_data()
        points_source.selected.indices = []
        metadata_div.text = "<i>No inference entries available.</i>"
        return

    score_series = pd.to_numeric(filtered["logit_score"], errors="coerce").fillna(
        -np.inf
    )
    sorted_df = filtered.assign(_score=score_series).sort_values(
        "_score", ascending=False, kind="mergesort"
    )
    sorted_df = sorted_df.drop(columns=["_score"]).reset_index(drop=True)

    lat_values = sorted_df["lat"].tolist()
    lon_values = sorted_df["lon"].tolist()
    x_vals, y_vals = lonlat_to_web_mercator(lon_values, lat_values)

    spectrogram_urls = [""] * len(sorted_df)
    playlist_df = sorted_df.head(PLAYLIST_LIMIT)
    playlist_urls: list[str] = []
    for idx, row in playlist_df.iterrows():
        url, exists = build_spectrogram_url(
            filename=row["filename"] or "",
            spectrogram_dir=SPECTROGRAMS_DIR,
            image_format=SPECTROGRAM_IMAGE_FORMAT,
            base_url=SPECTROGRAM_BASE_URL,
            inline=INLINE_SPECTROGRAMS,
        )
        resolved = url if exists else ""
        spectrogram_urls[idx] = resolved
        playlist_urls.append(resolved)

    points_source.data = {
        "x": x_vals.tolist(),
        "y": y_vals.tolist(),
        "lat": lat_values,
        "lon": lon_values,
        "rank": list(range(1, len(sorted_df) + 1)),
        "filename": sorted_df["filename"].astype(str).tolist(),
        "recordist": sorted_df["recordist"].astype(str).tolist(),
        "date": sorted_df["date"].astype(str).tolist(),
        "logit_score": sorted_df["logit_score"].tolist(),
        "spectrogram_url": spectrogram_urls,
    }
    points_source.selected.indices = []
    metadata_div.text = build_metadata_html(
        playlist_df.to_dict(orient="records"),
        spectrogram_dir=SPECTROGRAMS_DIR,
        image_format=SPECTROGRAM_IMAGE_FORMAT,
        base_url=SPECTROGRAM_BASE_URL,
        inline=INLINE_SPECTROGRAMS,
        spectrogram_urls=playlist_urls,
    )

    filter_bits = []
    if max_per_recordist:
        filter_bits.append(f"max per recordist={max_per_recordist}")
    if inference_missing_meta:
        filter_bits.append(f"missing metadata={inference_missing_meta}")
    filter_text = f" ({', '.join(filter_bits)})" if filter_bits else ""
    stats_div.text = (
        f"<b>{len(sorted_df)}</b> inference entries "
        f"(playlist={len(playlist_df)}){filter_text}"
    )
    status_div.text = "<i>Inference map updated.</i>"

    if AUTO_ZOOM:
        update_map_range(map_plot, x_vals, y_vals)


if not USE_INFERENCE:
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
else:
    recordist_filter = RadioButtonGroup(
        labels=["All", "1", "2", "3", "4", "5"],
        active=0,
        width=260,
    )
    recordist_filter.on_change(
        "active", lambda attr, old, new: update_inference_view()
    )

map_plot = figure(
    title="Inference map" if USE_INFERENCE else "Ingroup map",
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
        ("logit", "@logit_score{0.000}"),
        ("lat, lon", "@lat, @lon"),
    ]
)
map_plot.add_tools(map_hover)

if USE_INFERENCE:
    controls = column(
        Div(text="<b>Max logit per recordist</b>"),
        recordist_filter,
        stats_div,
        status_div,
    )
    table_panel = column(Div(text="<b>Inference entries</b>"), controls, width=520)
    metadata_title = "<b>Inference metadata</b>"
else:
    controls = column(size_spinner, top_n_spinner, sort_select, stats_div, status_div)
    table_panel = column(Div(text="<b>Top ingroups</b>"), controls, ingroup_table)
    metadata_title = "<b>Ingroup metadata</b>"

metadata_panel = column(
    Div(text=metadata_title),
    metadata_table,
    metadata_div,
    width=380,
)
layout = row(table_panel, map_plot, metadata_panel, sizing_mode="scale_both")

curdoc().add_root(layout)
curdoc().title = (
    f"Inference map - {SPECIES_SLUG}"
    if USE_INFERENCE
    else f"Ingroup map - {SPECIES_SLUG}"
)

if USE_INFERENCE:
    update_inference_view()
else:
    refresh_ingroup_table()
