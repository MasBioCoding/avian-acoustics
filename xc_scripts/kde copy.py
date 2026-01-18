#!/usr/bin/env python3
"""
Utility helpers for working toward 2D kernel density estimates of bird recording sites.

To run the script:
    cd /Users/masjansma/Desktop/birdnetcluster1folder/birdnet_data_pipeline
    python xc_scripts/kde.py --config xc_configs/config_limosa_limosa.yaml
    python xc_scripts/kde.py --config xc_configs/config_chloris_chloris.yaml
    python xc_scripts/kde.py --config xc_configs/config_regulus_ignicapilla.yaml
    python xc_scripts/kde.py --config xc_configs/config_regulus_regulus.yaml
    python xc_scripts/kde.py --config xc_configs/config_curruca_communis.yaml
    python xc_scripts/kde.py --config xc_configs/config_carduelis_carduelis.yaml
    python xc_scripts/kde.py --config xc_configs/config_acrocephalus_scirpaceus.yaml
    python xc_scripts/kde.py --config xc_configs/config_phylloscopus_trochilus.yaml

For now the script focuses on loading the saved selection groups (vocal types and dialects)
so that we can validate the data sources before computing any KDE overlays.
"""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from KDEpy import FFTKDE
from matplotlib import path as mpl_path
from pyproj import Transformer
from skimage import measure
import yaml
from collections import defaultdict
import math
import re
from bokeh.layouts import column, row
from bokeh.models import (
    BoxAnnotation,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    Legend,
    LegendItem,
    PanTool,
    Range1d,
    RangeSlider,
    Slider,
    Toggle,
    WheelZoomTool,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure, show
from xyzservices import providers as xyz_providers

ISOPLETH_LEVELS: tuple[float, float] = (0.5, 0.95)
ANIMATION_START_YEAR = 2015
ANIMATION_END_YEAR = 2025
WINDOW_MONTHS = 12
CENTROID_PLOTS_PER_ROW = 2


@dataclass
class GroupTable:
    """Container for a loaded group table."""

    group_type: str
    file_path: Path
    data: pd.DataFrame
    description: Optional[str] = None
    projected_points: Optional[np.ndarray] = None
    projection_center: Optional[np.ndarray] = None
    projection_scale: Optional[np.ndarray] = None
    kde_grid: Optional[np.ndarray] = None
    kde_values: Optional[np.ndarray] = None
    kde_resolution: Optional[int] = None
    bandwidth_scalar: Optional[float] = None
    isopleths: List[Dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Inspect selection group tables for KDE prep.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a species config file (e.g., xc_configs/config_limosa_limosa.yaml)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of rows to display from each table preview.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip launching the Bokeh visualization window.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_group_directories(config: Mapping) -> Dict[str, Path]:
    """Extract configured directories for vocal-type and dialect groups."""
    paths_cfg = config.get("paths", {})
    resolved: Dict[str, Path] = {}
    mapping = {
        "vocal_types": paths_cfg.get("vocal_type_groups"),
        "dialects": paths_cfg.get("dialect_groups"),
    }
    for group_type, raw_path in mapping.items():
        if raw_path:
            resolved[group_type] = Path(raw_path).expanduser()
    return resolved


def discover_group_tables(group_type: str, directory: Path) -> List[Path]:
    """Return all CSV files available for a group type."""
    if not directory.exists():
        print(f"[WARN] {group_type}: directory missing -> {directory}")
        return []
    csv_files = sorted(
        path for path in directory.glob("*.csv") if not path.name.startswith("._")
    )
    if not csv_files:
        print(f"[WARN] {group_type}: no CSV tables found in {directory}")
    return csv_files


def load_group_table(group_type: str, table_path: Path) -> GroupTable:
    """Load an individual group CSV file."""
    df = pd.read_csv(table_path)
    return GroupTable(group_type=group_type, file_path=table_path, data=df)


def load_group_description(table: GroupTable) -> GroupTable:
    """Load a group's description text from a sibling .txt file, if present."""
    desc_path = table.file_path.with_suffix(".txt")
    if not desc_path.exists():
        print(f"[WARN] {table.file_path.name}: description not found ({desc_path}).")
        table.description = None
        return table
    try:
        contents = desc_path.read_text(encoding="utf-8").strip()
        table.description = contents or None
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to read description for {table.file_path.name}: {exc}")
        table.description = None
    return table


def load_all_group_tables(group_dirs: Mapping[str, Path]) -> List[GroupTable]:
    """Load every CSV found beneath the configured group directories."""
    loaded: List[GroupTable] = []
    for group_type, directory in group_dirs.items():
        for csv_file in discover_group_tables(group_type, directory):
            try:
                table = load_group_table(group_type, csv_file)
                loaded.append(load_group_description(table))
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Failed to load {csv_file}: {exc}")
    return loaded


def resolve_metadata_path(config: Mapping) -> Path:
    """Build the metadata.csv path for the configured species."""
    paths_cfg = config.get("paths", {})
    species_cfg = config.get("species", {})
    root = paths_cfg.get("root")
    slug = species_cfg.get("slug")
    if not root or not slug:
        raise SystemExit(
            "Config must provide both 'paths.root' and 'species.slug' to locate metadata."
        )
    metadata_path = Path(root).expanduser() / "embeddings" / slug / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return metadata_path


def load_metadata_table(metadata_path: Path) -> pd.DataFrame:
    """Load the metadata CSV for the species."""
    return pd.read_csv(metadata_path)


def build_metadata_lookup(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce metadata to the fields we need for spatial summaries."""
    required_columns = ["xcid", "lat", "lon", "date"]
    missing = [col for col in required_columns if col not in metadata_df.columns]
    if missing:
        raise ValueError(
            f"Metadata file is missing required columns: {', '.join(missing)}"
        )
    lookup = (
        metadata_df[required_columns]
        .drop_duplicates(subset="xcid", keep="last")
        .set_index("xcid")
    )
    return lookup


def attach_metadata_to_group(table: GroupTable, metadata_lookup: pd.DataFrame) -> GroupTable:
    """Append latitude/longitude/date columns to a group table via xcid."""
    if metadata_lookup.empty:
        print("[WARN] Metadata lookup empty; skipping enrichment.")
        return table
    if "xcid" not in table.data.columns:
        print(f"[WARN] Table missing 'xcid' column: {table.file_path}")
        return table
    merged = table.data.merge(
        metadata_lookup, how="left", left_on="xcid", right_index=True
    )
    missing_coords = merged["lat"].isna().sum()
    if missing_coords:
        print(
            f"[WARN] {missing_coords} rows in {table.file_path.name} lack metadata matches."
        )
    table.data = merged
    return table


def build_equal_area_transformer(target_crs: str = "EPSG:3035") -> Transformer:
    """Prepare a transformer that projects lon/lat into an equal-area CRS."""
    print(f"[INFO] Using equal-area target CRS: {target_crs}")
    return Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)


def project_group_coordinates(
    table: GroupTable, transformer: Transformer
) -> GroupTable:
    """Project each row's lon/lat into equal-area coordinates for KDE."""
    if not {"lon", "lat"}.issubset(table.data.columns):
        print(f"[WARN] Skipping projection for {table.file_path.name}; missing lon/lat.")
        return table
    valid_mask = table.data["lon"].notna() & table.data["lat"].notna()
    if not valid_mask.any():
        print(f"[WARN] {table.file_path.name}: no valid coordinates to project.")
        return table
    lon = table.data.loc[valid_mask, "lon"].to_numpy()
    lat = table.data.loc[valid_mask, "lat"].to_numpy()
    try:
        x, y = transformer.transform(lon, lat)
        table.projected_points = np.column_stack([x, y])
        print(
            f"[INFO] {table.file_path.name}: projected {table.projected_points.shape[0]} points."
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to project coordinates for {table.file_path.name}: {exc}")
        table.projected_points = None
    return table


def estimate_scalar_bandwidth(points: np.ndarray) -> float:
    """Estimate an isotropic bandwidth using Scott's rule on standardized data."""
    n_samples, n_dims = points.shape
    if n_samples <= 1:
        return 1.0
    return float(n_samples ** (-1.0 / (n_dims + 4)))


def standardize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center and scale points per dimension to stabilize KDE bandwidths."""
    center = np.mean(points, axis=0)
    scale = np.std(points, axis=0, ddof=1)
    scale[scale == 0] = 1.0
    standardized = (points - center) / scale
    return standardized, center, scale


def compute_group_kde(
    table: GroupTable,
    grid_size: int = 256,
    bandwidth_override: Optional[float] = None,
) -> GroupTable:
    """Compute FFT-based KDE for the group's projected points."""
    points = table.projected_points
    table.bandwidth_scalar = None
    if points is None or len(points) < 5:
        print(
            f"[WARN] {table.file_path.name}: insufficient points ({0 if points is None else len(points)}) for KDE."
        )
        return table
    try:
        standardized, center, scale = standardize_points(points)
        if bandwidth_override is not None:
            bandwidth = float(bandwidth_override)
        else:
            bandwidth = estimate_scalar_bandwidth(standardized)
        bandwidth = max(bandwidth, 1e-3)
        print(
            f"[INFO] {table.file_path.name}: bandwidth scalar -> {bandwidth:.4f}"
        )
        table.bandwidth_scalar = bandwidth
        kde = FFTKDE(bw=bandwidth, kernel="triweight")
        grid_scaled, values = kde.fit(standardized).evaluate(grid_points=grid_size)
        grid = grid_scaled * scale + center
        table.kde_grid = grid
        table.kde_values = values
        table.projection_center = center
        table.projection_scale = scale
        print(
            f"[INFO] {table.file_path.name}: KDE computed with {grid.shape[0]} grid samples."
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] KDE computation failed for {table.file_path.name}: {exc}")
        table.kde_grid = None
        table.kde_values = None
        table.bandwidth_scalar = None
    return table


def log_kde_status(table: GroupTable) -> None:
    """Summarize KDE availability for the group."""
    if table.kde_grid is not None and table.kde_values is not None:
        print(
            f"[OK] {table.file_path.name}: KDE ready "
            f"(grid={table.kde_grid.shape[0]}, min={np.min(table.kde_values):.3e}, "
            f"max={np.max(table.kde_values):.3e})."
        )
    else:
        print(f"[WARN] {table.file_path.name}: KDE not available.")


def reshape_kde_arrays(table: GroupTable) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return grid-aligned coordinate and density arrays."""
    if table.kde_grid is None or table.kde_values is None:
        return None
    total_points = len(table.kde_values)
    grid_dim = int(np.sqrt(total_points))
    if grid_dim * grid_dim != total_points:
        print(f"[WARN] {table.file_path.name}: Unexpected grid size for KDE.")
        return None
    coords = table.kde_grid.reshape(grid_dim, grid_dim, 2)
    values = table.kde_values.reshape(grid_dim, grid_dim)
    x_axis = coords[:, 0, 0]
    y_axis = coords[0, :, 1]
    table.kde_resolution = grid_dim
    return x_axis, y_axis, values


def compute_hdr_levels(values: np.ndarray, probabilities: Sequence[float]) -> Dict[float, Optional[float]]:
    """Compute density thresholds corresponding to the requested HDR probabilities."""
    flat = values.flatten()
    total = float(flat.sum())
    if total <= 0:
        return {prob: None for prob in probabilities}
    order = np.argsort(flat)[::-1]
    sorted_vals = flat[order]
    cumulative = np.cumsum(sorted_vals)
    levels: Dict[float, Optional[float]] = {}
    for prob in probabilities:
        target = prob * total
        idx = int(np.searchsorted(cumulative, target))
        idx = min(idx, len(sorted_vals) - 1)
        levels[prob] = float(sorted_vals[idx])
    return levels


def contour_paths_from_values(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    level: float,
) -> List[np.ndarray]:
    """Extract contour paths at a given density level."""
    contours = measure.find_contours(values, level=level)
    if not contours:
        return []
    grid_indices = np.arange(len(x_axis))
    col_indices = np.arange(len(y_axis))
    paths: List[np.ndarray] = []
    for contour in contours:
        rows = contour[:, 0]
        cols = contour[:, 1]
        xs = np.interp(rows, grid_indices, x_axis)
        ys = np.interp(cols, col_indices, y_axis)
        paths.append(np.column_stack((xs, ys)))
    return paths


def extract_isopleths(
    table: GroupTable,
    probabilities: Sequence[float] = ISOPLETH_LEVELS,
    log_paths: bool = True,
) -> GroupTable:
    """Compute HDR isopleths for a table's KDE result."""
    reshaped = reshape_kde_arrays(table)
    if reshaped is None:
        table.isopleths = []
        return table
    x_axis, y_axis, values = reshaped
    levels = compute_hdr_levels(values, probabilities)
    isopleths: List[Dict[str, Any]] = []
    for prob, level in levels.items():
        if level is None:
            continue
        paths = contour_paths_from_values(x_axis, y_axis, values, level)
        if not paths:
            if log_paths:
                print(
                    f"[WARN] {table.file_path.name}: No contour paths identified for {prob:.0%} HDR."
                )
            continue
        if log_paths:
            print(
                f"[INFO] {table.file_path.name}: extracted {len(paths)} path(s) for {int(prob*100)}% HDR."
            )
        isopleths.append({"prob": prob, "level": level, "paths": paths})
    table.isopleths = isopleths
    return table


def ensure_datetime_column(table: GroupTable) -> None:
    """Add a parsed datetime column if missing."""
    if "date_dt" not in table.data.columns:
        table.data["date_dt"] = pd.to_datetime(table.data.get("date"), errors="coerce")


def generate_frame_periods(start_year: int, end_year: int) -> List[pd.Period]:
    """Return a list of monthly periods between start and end years (inclusive)."""
    periods: List[pd.Period] = []
    start = pd.Period(f"{start_year}-01", freq="M")
    end = pd.Period(f"{end_year}-10", freq="M")
    current = start
    while current <= end:
        periods.append(current)
        current += 1
    return periods


def _empty_frame_line_payload() -> Dict[str, List[Any]]:
    return {
        "xs": [],
        "ys": [],
        "line_color": [],
        "line_width": [],
        "line_dash": [],
        "probability": [],
        "label": [],
    }


def _empty_frame_point_payload() -> Dict[str, List[Any]]:
    return {
        "x": [],
        "y": [],
        "lon": [],
        "lat": [],
        "label": [],
        "color": [],
    }


def _empty_static_payload() -> Dict[str, List[Any]]:
    return {
        "xs": [],
        "ys": [],
        "line_width": [],
        "line_dash": [],
        "probability": [],
        "label": [],
    }


def concise_group_label(label: str) -> str:
    """Shorten a group label by dropping trailing timestamp-like suffixes."""
    return re.sub(r"(_\d{6,})+$", "", label)


def concise_display_label(label: str) -> str:
    """Shorten a label while preserving any suffix (e.g., '(points)')."""
    base, sep, suffix = label.partition(" ")
    short_base = concise_group_label(base)
    return f"{short_base}{sep}{suffix}" if sep else short_base


def build_description_html(group_tables: Iterable[GroupTable]) -> str:
    """Construct HTML for a summary of group descriptions grouped by type."""
    grouped: Dict[str, List[tuple[str, str]]] = defaultdict(list)
    for table in group_tables:
        label = concise_group_label(table.file_path.stem)
        desc = table.description or "No description available."
        grouped[table.group_type].append((label, desc))

    if not grouped:
        return "<em>No descriptions available for the loaded groups.</em>"

    type_labels = {
        "vocal_types": "Vocal types",
        "dialects": "Dialects",
    }
    ordered_types = [
        group_type for group_type in ("vocal_types", "dialects") if group_type in grouped
    ] + [group_type for group_type in grouped.keys() if group_type not in {"vocal_types", "dialects"}]

    sections: List[str] = [
        "<div style='font-size:14px;font-weight:600;margin-bottom:6px;'>Group descriptions</div>"
    ]
    for group_type in ordered_types:
        entries = sorted(grouped.get(group_type, []), key=lambda item: item[0])
        if not entries:
            continue
        header = type_labels.get(group_type, group_type.replace("_", " ").title())
        sections.append(
            f"<div style='font-weight:600;margin-bottom:4px;'>{html.escape(header)}</div>"
        )
        list_items = []
        for label, desc in entries:
            safe_desc = html.escape(desc).replace("\n", "<br>")
            safe_label = html.escape(label)
            list_items.append(
                f"<li style='margin-bottom:4px;'><span style='font-weight:600;'>{safe_label}</span>: {safe_desc}</li>"
            )
        sections.append(
            "<ul style='margin:0 0 10px 18px;padding:0;'>"
            + "".join(list_items)
            + "</ul>"
        )
    return "".join(sections)


def build_centroid_plot_payload(
    entries: Sequence[tuple[str, Optional[float], Optional[float]]],
    km_scale: float = 1000.0,
) -> Dict[str, List[Optional[float]]]:
    """Convert centroid track entries to relative displacement payload."""
    frames = [frame for frame, _, _ in entries]
    first_valid = next(((x, y) for _, x, y in entries if x is not None and y is not None), None)
    base_x = first_valid[0] if first_valid else None
    base_y = first_valid[1] if first_valid else None
    rel_x: List[Optional[float]] = []
    rel_y: List[Optional[float]] = []
    for _, x_val, y_val in entries:
        if base_x is None or base_y is None or x_val is None or y_val is None:
            rel_x.append(None)
            rel_y.append(None)
            continue
        rel_x.append((x_val - base_x) / km_scale)
        rel_y.append((y_val - base_y) / km_scale)
    return {"x": rel_x, "y": rel_y, "frame": frames}


def _empty_centroid_payload(frame_labels: Sequence[str]) -> Dict[str, List[Optional[float]]]:
    return {
        "x": [None] * len(frame_labels),
        "y": [None] * len(frame_labels),
        "frame": list(frame_labels),
    }


def polygon_centroid(path_points: np.ndarray) -> tuple[float, float]:
    """Compute centroid of a polygon defined by path points."""
    if len(path_points) < 3:
        return float(np.mean(path_points[:, 0])), float(np.mean(path_points[:, 1]))
    x = path_points[:, 0]
    y = path_points[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    if np.isclose(area, 0.0):
        return float(np.mean(x)), float(np.mean(y))
    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return float(cx), float(cy)


def polygon_area_sqkm(path_points: np.ndarray) -> float:
    """Return polygon area in square kilometers (expects projected meters)."""
    if len(path_points) < 3:
        return 0.0
    x = path_points[:, 0]
    y = path_points[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area_m2 = 0.5 * abs(cross.sum())
    return area_m2 / 1_000_000.0


def build_static_isopleth_payload(
    table: GroupTable,
    contour_transformer: Transformer,
    bandwidth_override: Optional[float] = None,
) -> tuple[Dict[str, Any], List[float], List[float]]:
    """Return multiline payload and flat coordinate lists for a group's isopleths."""
    isopleths = table.isopleths
    if bandwidth_override is not None:
        temp_table = GroupTable(
            group_type=table.group_type,
            file_path=table.file_path,
            data=table.data.copy(),
        )
        temp_table.projected_points = table.projected_points
        temp_table.projection_center = table.projection_center
        temp_table.projection_scale = table.projection_scale
        temp_table = compute_group_kde(
            temp_table,
            bandwidth_override=bandwidth_override,
        )
        temp_table = extract_isopleths(temp_table, log_paths=False)
        isopleths = temp_table.isopleths

    xs_collection: List[List[float]] = []
    ys_collection: List[List[float]] = []
    widths: List[float] = []
    dashes: List[str] = []
    probs: List[str] = []
    labels: List[str] = []
    flat_x: List[float] = []
    flat_y: List[float] = []
    label = f"{table.group_type}:{table.file_path.stem}"

    for iso in isopleths:
        prob = float(iso["prob"])
        pct_label = f"{int(prob * 100)}%"
        line_width = 3.0
        line_dash = "dotted" if np.isclose(prob, 0.95) else "solid"
        for path in iso["paths"]:
            if len(path) == 0:
                continue
            x_merc, y_merc = contour_transformer.transform(path[:, 0], path[:, 1])
            xs_collection.append(x_merc.tolist())
            ys_collection.append(y_merc.tolist())
            widths.append(line_width)
            dashes.append(line_dash)
            probs.append(pct_label)
            labels.append(label)
            flat_x.extend(x_merc.tolist())
            flat_y.extend(y_merc.tolist())

    payload = {
        "xs": xs_collection,
        "ys": ys_collection,
        "line_width": widths,
        "line_dash": dashes,
        "probability": probs,
        "label": labels,
    }
    return payload, flat_x, flat_y


def compute_weighted_bandwidth_average(
    bandwidth_series: Mapping[str, Sequence[Optional[float]]],
    sample_series: Mapping[str, Sequence[int]],
    target_group: str = "dialects",
    exclude_prefixes: Optional[Sequence[str]] = ("inter",),
) -> Optional[float]:
    """Return the sample-weighted average bandwidth scalar for a specific group type."""
    weighted_sum = 0.0
    total_weight = 0.0
    target_prefix = f"{target_group}:"
    for label, bandwidths in bandwidth_series.items():
        if not label.startswith(target_prefix):
            continue
        _, _, remainder = label.partition(":")
        remainder_lower = remainder.lower()
        if exclude_prefixes and any(
            remainder_lower.startswith(prefix.lower()) for prefix in exclude_prefixes
        ):
            continue
        counts = sample_series.get(label)
        if counts is None:
            continue
        length = min(len(bandwidths), len(counts))
        for idx in range(length):
            bandwidth = bandwidths[idx]
            count = counts[idx]
            if bandwidth is None or count is None or count <= 0:
                continue
            weighted_sum += bandwidth * count
            total_weight += count
    if total_weight <= 0.0:
        return None
    return weighted_sum / total_weight


def compute_monthly_sample_series(
    group_tables: Iterable[GroupTable],
    frame_periods: Sequence[pd.Period],
) -> Dict[str, List[int]]:
    """Return per-month sample counts (no rolling window) for each group."""
    period_index = {period: idx for idx, period in enumerate(frame_periods)}
    series: Dict[str, List[int]] = {}
    for table in group_tables:
        label = f"{table.group_type}:{table.file_path.stem}"
        ensure_datetime_column(table)
        dt = table.data.get("date_dt")
        if dt is None:
            continue
        periods = dt.dt.to_period("M")
        valid = periods.notna()
        if not valid.any():
            continue
        lon_mask = table.data.get("lon").notna() if "lon" in table.data else valid
        lat_mask = table.data.get("lat").notna() if "lat" in table.data else valid
        coord_mask = lon_mask & lat_mask
        mask = valid & coord_mask
        if not mask.any():
            continue
        subset_periods = periods[mask]
        counts = [0] * len(frame_periods)
        value_counts = subset_periods.value_counts()
        for period_value, count in value_counts.items():
            idx = period_index.get(period_value)
            if idx is None:
                continue
            counts[idx] = int(count)
        series[label] = counts
    return series


def build_sliding_window_payloads(
    group_tables: Iterable[GroupTable],
    frame_periods: Sequence[pd.Period],
    color_map: Mapping[str, str],
    eq_area_transformer: Transformer,
    contour_to_merc: Transformer,
    point_transformer: Transformer,
    centroid_hdr: float = 0.5,
    forced_bandwidth: Optional[float] = None,
) -> tuple[
    List[str],
    Dict[str, List[Dict[str, List[Any]]]],
    Dict[str, List[Dict[str, List[Any]]]],
    Dict[str, List[tuple[str, Optional[float], Optional[float]]]],
    List[int],
    Dict[str, List[int]],
    Dict[str, List[Optional[float]]],
    Dict[str, List[Optional[float]]],
]:
    """Compute per-frame KDE isopleths, point clouds, and centroid tracks."""
    frame_labels: List[str] = []
    group_line_frames: Dict[str, List[Dict[str, List[Any]]]] = defaultdict(list)
    group_point_frames: Dict[str, List[Dict[str, List[Any]]]] = defaultdict(list)
    centroid_tracks: Dict[str, List[tuple[str, Optional[float], Optional[float]]]] = defaultdict(list)
    total_sample_counts: List[int] = []
    group_sample_counts: Dict[str, List[int]] = defaultdict(list)
    group_bandwidth_series: Dict[str, List[Optional[float]]] = defaultdict(list)
    group_isopleth_area_series: Dict[str, List[Optional[float]]] = defaultdict(list)

    for current_period in frame_periods:
        frame_label = current_period.strftime("%Y-%m")
        start_period = current_period - (WINDOW_MONTHS - 1)
        frame_labels.append(frame_label)
        frame_total = 0

        for table in group_tables:
            label = f"{table.group_type}:{table.file_path.stem}"
            color = color_map.get(label, "#666666")
            line_payload = _empty_frame_line_payload()
            point_payload = _empty_frame_point_payload()
            bandwidth_value: Optional[float] = None
            area_value: Optional[float] = None

            ensure_datetime_column(table)
            dt = table.data.get("date_dt")
            if dt is None:
                centroid_tracks[label].append((frame_label, None, None))
                group_sample_counts[label].append(0)
                group_bandwidth_series[label].append(None)
                group_isopleth_area_series[label].append(None)
                group_line_frames[label].append(line_payload)
                group_point_frames[label].append(point_payload)
                continue

            periods = dt.dt.to_period("M")
            valid = periods.notna()
            if not valid.any():
                centroid_tracks[label].append((frame_label, None, None))
                group_sample_counts[label].append(0)
                group_bandwidth_series[label].append(None)
                group_isopleth_area_series[label].append(None)
                group_line_frames[label].append(line_payload)
                group_point_frames[label].append(point_payload)
                continue

            mask = valid & (periods >= start_period) & (periods <= current_period)
            subset = table.data.loc[mask].copy()
            lon_lat = subset[["lon", "lat"]].dropna()
            best_centroid: Optional[tuple[float, float]] = None
            group_count = len(lon_lat)
            frame_total += group_count
            bandwidth_value: Optional[float] = None
            area_value = None

            if len(lon_lat) >= 5:
                temp_table = GroupTable(
                    group_type=table.group_type,
                    file_path=table.file_path,
                    data=subset,
                )
                temp_table = project_group_coordinates(temp_table, eq_area_transformer)
                temp_table = compute_group_kde(
                    temp_table,
                    bandwidth_override=forced_bandwidth,
                )
                bandwidth_value = (
                    forced_bandwidth
                    if forced_bandwidth is not None and temp_table.bandwidth_scalar is not None
                    else temp_table.bandwidth_scalar
                )
                temp_table = extract_isopleths(temp_table, log_paths=False)

                for iso in temp_table.isopleths:
                    prob = float(iso["prob"])
                    pct_label = f"{int(prob * 100)}%"
                    line_width = 3.0
                    line_dash = "dotted" if np.isclose(prob, 0.95) else "solid"
                    for path in iso["paths"]:
                        if len(path) == 0:
                            continue
                        x_merc, y_merc = contour_to_merc.transform(path[:, 0], path[:, 1])
                        line_payload["xs"].append(x_merc.tolist())
                        line_payload["ys"].append(y_merc.tolist())
                        line_payload["line_color"].append(color)
                        line_payload["line_width"].append(line_width)
                        line_payload["line_dash"].append(line_dash)
                        line_payload["probability"].append(pct_label)
                        line_payload["label"].append(label)

                    if np.isclose(prob, centroid_hdr) and iso["paths"]:
                        proj_x, proj_y = eq_area_transformer.transform(
                            lon_lat["lon"].to_numpy(), lon_lat["lat"].to_numpy()
                        )
                        proj_points = np.column_stack((proj_x, proj_y))
                        best_count = -1
                        for path in iso["paths"]:
                            if len(path) == 0:
                                continue
                            poly = mpl_path.Path(path)
                            count = (
                                int(poly.contains_points(proj_points).sum())
                                if proj_points.size > 0
                                else 0
                            )
                            if count > best_count:
                                best_count = count
                                best_centroid = polygon_centroid(path)

                    if np.isclose(prob, centroid_hdr) and iso["paths"]:
                        area_total = 0.0
                        for path in iso["paths"]:
                            if len(path) < 3:
                                continue
                            area_total += polygon_area_sqkm(path)
                        if area_total > 0.0:
                            area_value = area_total

            if not lon_lat.empty:
                x_merc, y_merc = point_transformer.transform(
                    lon_lat["lon"].to_numpy(), lon_lat["lat"].to_numpy()
                )
                point_payload["x"].extend(x_merc.tolist())
                point_payload["y"].extend(y_merc.tolist())
                point_payload["lon"].extend(lon_lat["lon"].tolist())
                point_payload["lat"].extend(lon_lat["lat"].tolist())
                point_payload["label"].extend([label] * len(lon_lat))
                point_payload["color"].extend([color] * len(lon_lat))
                if best_centroid is None:
                    proj_x, proj_y = eq_area_transformer.transform(
                        lon_lat["lon"].to_numpy(), lon_lat["lat"].to_numpy()
                    )
                    if proj_x.size > 0:
                        best_centroid = (float(np.mean(proj_x)), float(np.mean(proj_y)))
            else:
                best_centroid = None

            centroid_tracks[label].append(
                (
                    frame_label,
                    best_centroid[0] if best_centroid else None,
                    best_centroid[1] if best_centroid else None,
                )
            )
            group_sample_counts[label].append(group_count)
            group_bandwidth_series[label].append(bandwidth_value)
            group_isopleth_area_series[label].append(area_value)

            group_line_frames[label].append(line_payload)
            group_point_frames[label].append(point_payload)

        total_sample_counts.append(frame_total)

    return (
        frame_labels,
        group_line_frames,
        group_point_frames,
        centroid_tracks,
        total_sample_counts,
        group_sample_counts,
        group_bandwidth_series,
        group_isopleth_area_series,
    )


def render_isopleth_map(group_tables: Iterable[GroupTable]) -> None:
    """Render KDE isopleths on a Bokeh Web Mercator map."""
    palette = Category10[10]
    color_index = 0
    color_map: Dict[str, str] = {}
    equal_area_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    contour_transformer = Transformer.from_crs("EPSG:3035", "EPSG:3857", always_xy=True)
    point_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    plot = figure(
        title="KDE Isopleths (Web Mercator)",
        match_aspect=True,
        x_axis_type="mercator",
        y_axis_type="mercator",
        width=1000,
        height=650,
        tools="pan,wheel_zoom,reset,save",
    )
    plot.add_tile(xyz_providers.CartoDB.Positron)

    all_x: List[float] = []
    all_y: List[float] = []
    hover_renderers: List[Any] = []
    point_renderers: List[Any] = []
    legend_renderers: Dict[str, List[Any]] = defaultdict(list)
    group_line_sources: Dict[str, ColumnDataSource] = {}
    mode_static_lines: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    empty_static_payload = _empty_static_payload()

    for table in group_tables:
        label = f"{table.group_type}:{table.file_path.stem}"
        color = color_map.get(label)
        if color is None:
            color = palette[color_index % len(palette)]
            color_map[label] = color
            color_index += 1

        payload, xs_flat, ys_flat = build_static_isopleth_payload(
            table,
            contour_transformer,
        )
        mode_static_lines["per_group"][label] = payload
        all_x.extend(xs_flat)
        all_y.extend(ys_flat)

        source = ColumnDataSource(payload if payload["xs"] else _empty_static_payload())

        renderer = plot.multi_line(
            xs="xs",
            ys="ys",
            line_color=color,
            line_width="line_width",
            line_dash="line_dash",
            name=label,
            source=source,
        )
        hover_renderers.append(renderer)
        legend_renderers[label].append(renderer)
        group_line_sources[label] = source

        # Add points for this group (toggle separately)
        if table.projected_points is not None:
            lon_lat = table.data[["lon", "lat"]].dropna()
            if not lon_lat.empty:
                x_merc, y_merc = point_transformer.transform(
                    lon_lat["lon"].to_numpy(), lon_lat["lat"].to_numpy()
                )
                all_x.extend(x_merc.tolist())
                all_y.extend(y_merc.tolist())
                point_source = ColumnDataSource(
                    {
                        "x": x_merc,
                        "y": y_merc,
                        "lon": lon_lat["lon"].to_numpy(),
                        "lat": lon_lat["lat"].to_numpy(),
                        "label": [label] * len(x_merc),
                    }
                )
                point_renderer = plot.scatter(
                    x="x",
                    y="y",
                    size=5,
                    color=color,
                    alpha=0.6,
                    marker="circle",
                    name=f"{label} (points)",
                    source=point_source,
                )
                point_renderers.append(point_renderer)
                legend_renderers[f"{label} (points)"].append(point_renderer)

    # Build sliding-window payloads
    frame_periods = generate_frame_periods(ANIMATION_START_YEAR, ANIMATION_END_YEAR)
    monthly_sample_series = compute_monthly_sample_series(group_tables, frame_periods)
    (
        frame_labels,
        group_line_frames,
        group_point_frames,
        centroid_tracks,
        total_sample_counts,
        group_sample_counts,
        group_bandwidth_series,
        group_isopleth_area_series,
    ) = build_sliding_window_payloads(
        group_tables,
        frame_periods,
        color_map,
        equal_area_transformer,
        contour_transformer,
        point_transformer,
    )
    mode_line_frames: Dict[str, Dict[str, List[Dict[str, List[Any]]]]] = {
        "per_group": group_line_frames
    }
    mode_point_frames: Dict[str, Dict[str, List[Dict[str, List[Any]]]]] = {
        "per_group": group_point_frames
    }
    mode_centroid_tracks: Dict[str, Dict[str, List[tuple[str, Optional[float], Optional[float]]]]] = {
        "per_group": centroid_tracks
    }
    mode_bandwidth_series: Dict[str, Dict[str, List[Optional[float]]]] = {
        "per_group": group_bandwidth_series
    }
    mode_area_series: Dict[str, Dict[str, List[Optional[float]]]] = {
        "per_group": group_isopleth_area_series
    }

    weighted_dialect_bandwidth = compute_weighted_bandwidth_average(
        group_bandwidth_series,
        group_sample_counts,
        target_group="dialects",
    )
    if weighted_dialect_bandwidth is not None:
        print(
            "[INFO] Weighted bandwidth scalar (dialects, sample-weighted): "
            f"{weighted_dialect_bandwidth:.4f}"
        )
    else:
        print("[WARN] Dialect-weighted bandwidth scalar could not be computed (no data).")

    weighted_vocal_bandwidth = compute_weighted_bandwidth_average(
        group_bandwidth_series,
        group_sample_counts,
        target_group="vocal_types",
    )
    if weighted_vocal_bandwidth is not None:
        print(
            "[INFO] Weighted bandwidth scalar (vocal types, sample-weighted): "
            f"{weighted_vocal_bandwidth:.4f}"
        )
    else:
        print("[WARN] Vocal-type-weighted bandwidth scalar could not be computed (no data).")

    def register_unified_mode(mode_key: str, bandwidth_value: float) -> None:
        mode_static_lines[mode_key] = {}
        for table in group_tables:
            label = f"{table.group_type}:{table.file_path.stem}"
            payload, xs_flat, ys_flat = build_static_isopleth_payload(
                table,
                contour_transformer,
                bandwidth_override=bandwidth_value,
            )
            mode_static_lines[mode_key][label] = payload
            all_x.extend(xs_flat)
            all_y.extend(ys_flat)

        (
            frame_labels_mode,
            unified_line_frames,
            unified_point_frames,
            unified_centroid_tracks,
            _,
            _,
            unified_bandwidth_series,
            unified_area_series,
        ) = build_sliding_window_payloads(
            group_tables,
            frame_periods,
            color_map,
            equal_area_transformer,
            contour_transformer,
            point_transformer,
            forced_bandwidth=bandwidth_value,
        )
        if frame_labels_mode != frame_labels:
            print(
                f"[WARN] {mode_key} frame labels differ from default; using default ordering."
            )
        mode_line_frames[mode_key] = unified_line_frames
        mode_point_frames[mode_key] = unified_point_frames
        mode_centroid_tracks[mode_key] = unified_centroid_tracks
        mode_bandwidth_series[mode_key] = unified_bandwidth_series
        mode_area_series[mode_key] = unified_area_series

    if weighted_dialect_bandwidth is not None:
        register_unified_mode("dialects_weighted", weighted_dialect_bandwidth)
    if weighted_vocal_bandwidth is not None:
        register_unified_mode("vocal_types_weighted", weighted_vocal_bandwidth)

    mode_static_lines = {key: value for key, value in mode_static_lines.items()}
    mode_centroid_payloads: Dict[str, Dict[str, Dict[str, List[Optional[float]]]]] = {}
    for mode_name, tracks in mode_centroid_tracks.items():
        payloads: Dict[str, Dict[str, List[Optional[float]]]] = {}
        for label, entries in tracks.items():
            payloads[label] = build_centroid_plot_payload(entries)
        mode_centroid_payloads[mode_name] = payloads

    # Extend bounds with frame data across modes
    for frames_by_label in mode_line_frames.values():
        for payloads in frames_by_label.values():
            for payload in payloads:
                for xs in payload["xs"]:
                    all_x.extend(xs)
                for ys in payload["ys"]:
                    all_y.extend(ys)
    for frames_by_label in mode_point_frames.values():
        for payloads in frames_by_label.values():
            for payload in payloads:
                all_x.extend(payload["x"])
                all_y.extend(payload["y"])

    group_line_sources: Dict[str, ColumnDataSource] = {}
    group_point_sources: Dict[str, ColumnDataSource] = {}

    default_line_frames = mode_line_frames.get("per_group", {})
    for label, frames in default_line_frames.items():
        color = color_map.get(label, "#666666")
        initial_data = frames[0] if frames else _empty_frame_line_payload()
        src = ColumnDataSource(initial_data)
        renderer = plot.multi_line(
            xs="xs",
            ys="ys",
            line_color="line_color" if "line_color" in initial_data else color,
            line_width="line_width",
            line_dash="line_dash",
            alpha=0.75,
            source=src,
        )
        hover_renderers.append(renderer)
        legend_renderers[f"{label} (window isopleths)"].append(renderer)
        group_line_sources[label] = src

    default_point_frames = mode_point_frames.get("per_group", {})
    for label, frames in default_point_frames.items():
        color = color_map.get(label, "#666666")
        initial_data = frames[0] if frames else _empty_frame_point_payload()
        src = ColumnDataSource(initial_data)
        renderer = plot.scatter(
            x="x",
            y="y",
            size=4,
            color="color" if "color" in initial_data else color,
            alpha=0.5,
            marker="circle",
            source=src,
        )
        point_renderers.append(renderer)
        legend_renderers[f"{label} (window points)"].append(renderer)
        group_point_sources[label] = src

    if not all_x or not all_y:
        print("No isopleths to visualize.")
        return

    margin_x = max(1.0, 0.05 * (max(all_x) - min(all_x)))
    margin_y = max(1.0, 0.05 * (max(all_y) - min(all_y)))
    plot.x_range.start = min(all_x) - margin_x
    plot.x_range.end = max(all_x) + margin_x
    plot.y_range.start = min(all_y) - margin_y
    plot.y_range.end = max(all_y) + margin_y

    if hover_renderers:
        hover = HoverTool(
            tooltips=[("Group", "@label"), ("HDR level", "@probability")],
            renderers=hover_renderers,
        )
        plot.add_tools(hover)
    if point_renderers:
        points_hover = HoverTool(
            tooltips=[("Group", "@label"), ("lon", "@lon{0.00}"), ("lat", "@lat{0.00}")],
            renderers=point_renderers,
        )
        plot.add_tools(points_hover)

    if legend_renderers:
        legend = Legend(
            items=[
                LegendItem(label=concise_display_label(label), renderers=renders)
                for label, renders in legend_renderers.items()
                if renders
            ],
            click_policy="hide",
            spacing=6,
        )
        plot.add_layout(legend, "right")

    sample_fig = None
    monthly_fig = None
    bandwidth_fig = None
    area_fig = None
    sample_highlight: Optional[BoxAnnotation] = None
    monthly_highlight: Optional[BoxAnnotation] = None
    bandwidth_highlight: Optional[BoxAnnotation] = None
    area_highlight: Optional[BoxAnnotation] = None
    sample_xrange: Optional[Range1d] = None
    x_indices: List[int] = []
    sample_sources: Dict[str, ColumnDataSource] = {}
    monthly_sources: Dict[str, ColumnDataSource] = {}
    bandwidth_sources: Dict[str, ColumnDataSource] = {}
    area_sources: Dict[str, ColumnDataSource] = {}
    initial_high = 0
    initial_low = 0
    if frame_labels:
        initial_high = min(len(frame_labels) - 1, WINDOW_MONTHS - 1)
        initial_low = max(0, initial_high - (WINDOW_MONTHS - 1))
        x_indices = list(range(len(frame_labels)))
        sample_xrange = Range1d(0, max(len(frame_labels) - 1, 1))
        sample_fig = figure(
            title="Sample counts per frame",
            width=600,
            height=250,
            x_range=sample_xrange,
            tools="pan,wheel_zoom,reset,save",
        )
        sample_legend_items = []
        for label, counts in group_sample_counts.items():
            if not counts:
                continue
            src = ColumnDataSource({"x": x_indices, "y": counts})
            legend_label = concise_group_label(label)
            renderer = sample_fig.line(
                "x",
                "y",
                color=color_map.get(label, "#666666"),
                line_width=1.5,
                source=src,
            )
            sample_sources[label] = src
            sample_legend_items.append(
                LegendItem(label=f"{legend_label}", renderers=[renderer])
            )
        if sample_legend_items:
            sample_legend = Legend(items=sample_legend_items, click_policy="hide", spacing=6)
            sample_fig.add_layout(sample_legend, "right")
        tick_step = max(1, len(frame_labels) // 12)
        label_overrides = {
            idx: label for idx, label in enumerate(frame_labels) if tick_step == 1 or idx % tick_step == 0
        }
        sample_fig.xaxis.major_label_overrides = label_overrides
        sample_highlight = BoxAnnotation(
            left=initial_low,
            right=initial_high,
            fill_alpha=0.15,
            fill_color="#F9E79F",
            line_color="#B9770E",
            line_width=2,
        )
        sample_fig.add_layout(sample_highlight)

        has_bandwidth_values = any(
            any(val is not None for val in series)
            for series in group_bandwidth_series.values()
        )
        if has_bandwidth_values:
            bandwidth_fig = figure(
                title=f"Bandwidth per frame ({WINDOW_MONTHS}-month window)",
                width=600,
                height=250,
                x_range=sample_xrange,
                tools="pan,wheel_zoom,reset,save",
            )
            bandwidth_fig.yaxis.axis_label = "Scott bandwidth scalar"
            bandwidth_fig.xaxis.major_label_overrides = label_overrides
            bandwidth_highlight = BoxAnnotation(
                left=initial_low,
                right=initial_high,
                fill_alpha=0.15,
                fill_color="#F9E79F",
                line_color="#B9770E",
                line_width=2,
            )
            bandwidth_fig.add_layout(bandwidth_highlight)
            bandwidth_legend_items: List[LegendItem] = []
            for label, values in group_bandwidth_series.items():
                color = color_map.get(label, "#666666")
                legend_label = concise_group_label(label)
                src = ColumnDataSource({"x": x_indices, "y": values})
                renderer = bandwidth_fig.line(
                    "x",
                    "y",
                    color=color,
                    line_width=1.5,
                    source=src,
                )
                bandwidth_sources[label] = src
                bandwidth_legend_items.append(
                    LegendItem(label=f"{legend_label}", renderers=[renderer])
                )
            if bandwidth_legend_items:
                bandwidth_legend = Legend(
                    items=bandwidth_legend_items,
                    click_policy="hide",
                    spacing=6,
                )
                bandwidth_fig.add_layout(bandwidth_legend, "right")

        def _series_has_values(series_map: Mapping[str, Sequence[Optional[float]]]) -> bool:
            return any(
                any(val is not None for val in series)
                for series in series_map.values()
            )

        has_area_values = _series_has_values(mode_area_series.get("per_group", {}))
        if not has_area_values:
            for area_map in mode_area_series.values():
                if _series_has_values(area_map):
                    has_area_values = True
                    break
        if has_area_values:
            area_fig = figure(
                title="50% isopleth area per frame",
                width=600,
                height=250,
                x_range=sample_xrange,
                tools="pan,wheel_zoom,reset,save",
            )
            area_fig.yaxis.axis_label = "Area (km²)"
            area_fig.xaxis.major_label_overrides = label_overrides
            area_highlight = BoxAnnotation(
                left=initial_low,
                right=initial_high,
                fill_alpha=0.15,
                fill_color="#F9E79F",
                line_color="#B9770E",
                line_width=2,
            )
            area_fig.add_layout(area_highlight)
            area_legend_items: List[LegendItem] = []
            area_labels = sorted(
                {
                    label
                    for area_map in mode_area_series.values()
                    for label in area_map.keys()
                }
            )
            for label in area_labels:
                values = group_isopleth_area_series.get(label) or [None] * len(x_indices)
                color = color_map.get(label, "#666666")
                legend_label = concise_group_label(label)
                src = ColumnDataSource({"x": x_indices, "y": values})
                renderer = area_fig.line(
                    "x",
                    "y",
                    color=color,
                    line_width=1.5,
                    source=src,
                )
                area_sources[label] = src
                area_legend_items.append(
                    LegendItem(label=f"{legend_label}", renderers=[renderer])
                )
            if area_legend_items:
                area_legend = Legend(
                    items=area_legend_items,
                    click_policy="hide",
                    spacing=6,
                )
                area_fig.add_layout(area_legend, "right")

        has_monthly_values = any(
            any(count > 0 for count in counts)
            for counts in monthly_sample_series.values()
        )
        if has_monthly_values:
            monthly_fig = figure(
                title="Monthly sample counts (per dialect/vocal type)",
                width=600,
                height=250,
                x_range=sample_xrange,
                tools="pan,wheel_zoom,reset,save",
            )
            monthly_fig.yaxis.axis_label = "Monthly samples"
            monthly_fig.xaxis.major_label_overrides = label_overrides
            monthly_highlight = BoxAnnotation(
                left=initial_low,
                right=initial_high,
                fill_alpha=0.15,
                fill_color="#F9E79F",
                line_color="#B9770E",
                line_width=2,
            )
            monthly_fig.add_layout(monthly_highlight)
            monthly_legend_items: List[LegendItem] = []
            for label, counts in monthly_sample_series.items():
                if not any(count > 0 for count in counts):
                    continue
                color = color_map.get(label, "#666666")
                legend_label = concise_group_label(label)
                src = ColumnDataSource({"x": x_indices, "y": counts})
                renderer = monthly_fig.line(
                    "x",
                    "y",
                    color=color,
                    line_width=1.5,
                    source=src,
                )
                monthly_sources[label] = src
                monthly_legend_items.append(
                    LegendItem(label=f"{legend_label}", renderers=[renderer])
                )
            if monthly_legend_items:
                monthly_legend = Legend(
                    items=monthly_legend_items,
                    click_policy="hide",
                    spacing=6,
                )
                monthly_fig.add_layout(monthly_legend, "right")

    movement_plots: List[Any] = []
    movement_sources: Dict[str, ColumnDataSource] = {}
    movement_labels = sorted(
        {
            label
            for payloads in mode_centroid_payloads.values()
            for label in payloads.keys()
        }
    )
    for label in movement_labels:
        has_data = any(
            any(val is not None for val in payload.get("x", []))
            for payload in (
                mode_centroid_payloads[mode].get(label, {})
                for mode in mode_centroid_payloads.keys()
            )
        )
        if not has_data:
            continue
        color = color_map.get(label, "#333333")
        default_payload = mode_centroid_payloads.get("per_group", {}).get(
            label, _empty_centroid_payload(frame_labels)
        )
        move_source = ColumnDataSource(default_payload)
        move_fig = figure(
            title=f"{label} centroid shift",
            width=250,
            height=250,
            x_range=Range1d(-500, 500),
            y_range=Range1d(-500, 500),
            tools="pan,wheel_zoom,reset,save",
            match_aspect=True,
        )
        move_fig.line("x", "y", source=move_source, color=color, line_width=2)
        move_fig.scatter("x", "y", source=move_source, color=color, size=6)
        move_fig.xaxis.axis_label = "East-West displacement (km)"
        move_fig.yaxis.axis_label = "North-South displacement (km)"
        move_fig.add_tools(
            HoverTool(
                tooltips=[("Frame", "@frame"), ("ΔE (km)", "@x{0.0}"), ("ΔN (km)", "@y{0.0}")]
            )
        )
        movement_plots.append(move_fig)
        movement_sources[label] = move_source

    movement_layout = None
    if movement_plots:
        rows: List[Any] = []
        for i in range(0, len(movement_plots), CENTROID_PLOTS_PER_ROW):
            rows.append(row(*movement_plots[i : i + CENTROID_PLOTS_PER_ROW]))
        movement_layout = column(*rows)

    charts_column: Optional[Any] = None
    chart_stack: List[Any] = []
    if sample_fig:
        chart_stack.append(sample_fig)
    if monthly_fig:
        chart_stack.append(monthly_fig)
    if bandwidth_fig:
        chart_stack.append(bandwidth_fig)
    if area_fig:
        chart_stack.append(area_fig)
    if chart_stack:
        charts_column = column(*chart_stack)

    description_html = build_description_html(group_tables)
    description_div = Div(
        text=description_html,
        width=plot.width,
        styles={
            "border": "1px solid #d9d9d9",
            "padding": "8px",
            "background-color": "#fafafa",
            "margin-top": "6px",
            "max-height": "240px",
            "overflow-y": "auto",
        },
    )
    map_column = column(plot, description_div)

    mode_state_source = ColumnDataSource({"mode": ["per_group"]})
    layout = map_column
    if frame_labels:
        range_slider = RangeSlider(
            start=0,
            end=len(frame_labels) - 1,
            value=(0, len(frame_labels) - 1),
            step=1,
            title="Frame window (index)",
            width=600,
        )
        num_chunks = math.ceil(len(frame_labels) / WINDOW_MONTHS) if WINDOW_MONTHS > 0 else 0
        initial_chunk = initial_low // WINDOW_MONTHS if WINDOW_MONTHS > 0 else 0
        chunk_slider: Optional[Slider] = None
        window_slider = RangeSlider(
            start=0,
            end=len(frame_labels) - 1,
            value=(initial_low, initial_high),
            step=1,
            title=f"{WINDOW_MONTHS}-month window ending (Year-Month)",
            width=600,
            bar_color="#F4D03F",
        )
        if num_chunks > 0:
            chunk_slider = Slider(
                start=0,
                end=max(num_chunks - 1, 0),
                value=min(initial_chunk, max(num_chunks - 1, 0)),
                step=1,
                title="Block window (non-overlap index)",
                width=600,
            )
            chunk_callback = CustomJS(
                args=dict(
                    chunk=chunk_slider,
                    slider=window_slider,
                    window_months=WINDOW_MONTHS,
                ),
                code="""
const span = window_months;
if (!span || span <= 0) { return; }
let idx = Math.round(chunk.value);
const minIdx = chunk.start ?? 0;
const maxIdx = chunk.end ?? idx;
idx = Math.max(minIdx, Math.min(maxIdx, idx));
let low = idx * span;
const minLow = slider.start;
let maxLow = slider.end - (span - 1);
if (!isFinite(maxLow)) {
    maxLow = slider.end;
}
if (maxLow < minLow) {
    maxLow = minLow;
}
low = Math.max(minLow, Math.min(maxLow, low));
let high = low + span - 1;
if (high > slider.end) {
    high = slider.end;
    low = Math.max(slider.start, high - (span - 1));
}
if (low < slider.start) {
    low = slider.start;
    high = Math.min(slider.end, low + span - 1);
}
slider.value = [low, high];
"""
            )
            chunk_slider.js_on_change("value", chunk_callback)
        base_title = "KDE Isopleths (Web Mercator)"
        plot.title.text = f"{base_title} – {frame_labels[initial_high]}"
        empty_line_payload = _empty_frame_line_payload()
        empty_point_payload = _empty_frame_point_payload()
        empty_static_payload = _empty_static_payload()
        slider_callback = CustomJS(
            args=dict(
                slider=window_slider,
                labels=frame_labels,
                title=plot.title,
                base_title=base_title,
                group_line_sources=group_line_sources,
                group_point_sources=group_point_sources,
                mode_line_frames=mode_line_frames,
                mode_point_frames=mode_point_frames,
                empty_line=empty_line_payload,
                empty_point=empty_point_payload,
                window_months=WINDOW_MONTHS,
                sample_box=sample_highlight,
                monthly_box=monthly_highlight,
                bandwidth_box=bandwidth_highlight,
                area_box=area_highlight,
                mode_source=mode_state_source,
            ),
            code="""
const total = labels.length;
if (!total) { return; }
const span = window_months;
const modeData = mode_source.data.mode || ["per_group"];
const mode = modeData[0] || "per_group";
const lineFrameMap = mode_line_frames[mode] || {};
const pointFrameMap = mode_point_frames[mode] || {};
let high = Math.round(slider.value[1]);
let low = high - (span - 1);
if (low < slider.start) {
    low = slider.start;
    high = low + span - 1;
}
if (high > slider.end) {
    high = slider.end;
    low = Math.max(slider.start, high - (span - 1));
}
slider.value = [low, high];
const idx = Math.max(0, Math.min(total - 1, high));
if (sample_box) {
    sample_box.left = low;
    sample_box.right = high;
}
if (monthly_box) {
    monthly_box.left = low;
    monthly_box.right = high;
}
if (bandwidth_box) {
    bandwidth_box.left = low;
    bandwidth_box.right = high;
}
if (area_box) {
    area_box.left = low;
    area_box.right = high;
}
for (const label in group_line_sources) {
    const frames = lineFrameMap[label] || [];
    const data = frames[idx] || empty_line;
    const source = group_line_sources[label];
    source.data = data;
    source.change.emit();
}
for (const label in group_point_sources) {
    const frames = pointFrameMap[label] || [];
    const data = frames[idx] || empty_point;
    const source = group_point_sources[label];
    source.data = data;
    source.change.emit();
}
title.text = `${base_title} – ${labels[idx]}`;
""",
        )
        window_slider.js_on_change("value", slider_callback)

        range_callback = CustomJS(
            args=dict(
                slider=window_slider,
                sample_range=sample_xrange,
                window_months=WINDOW_MONTHS,
                max_index=len(frame_labels) - 1,
            ),
            code="""
const span = window_months;
let low = Math.round(cb_obj.value[0]);
let high = Math.round(cb_obj.value[1]);
if (high - low + 1 < span) {
    high = Math.min(cb_obj.end, low + span - 1);
    low = Math.max(cb_obj.start, high - span + 1);
    cb_obj.value = [low, high];
}
slider.start = cb_obj.value[0];
slider.end = cb_obj.value[1];
let newHigh = Math.min(slider.end, Math.max(slider.start + span - 1, slider.value[1]));
let newLow = newHigh - (span - 1);
if (newLow < slider.start) {
    newLow = slider.start;
    newHigh = Math.min(slider.end, newLow + span - 1);
}
slider.value = [newLow, newHigh];
if (sample_range) {
    sample_range.start = low;
    sample_range.end = high;
}
""",
        )
        range_slider.js_on_change("value", range_callback)

        dialect_toggle = Toggle(
            label="Unified bandwidth (weighted average of dialects)",
            button_type="success" if weighted_dialect_bandwidth is not None else "default",
            width=320,
            active=False,
        )
        if weighted_dialect_bandwidth is None:
            dialect_toggle.disabled = True
            dialect_info = Div(
                text="Unified bandwidth unavailable (insufficient dialect coverage).",
                width=320,
            )
        else:
            dialect_info = Div(
                text=f"Weighted scalar: {weighted_dialect_bandwidth:.4f}",
                width=320,
            )

        vocal_toggle = Toggle(
            label="Unified bandwidth (weighted average of vocal types)",
            button_type="success" if weighted_vocal_bandwidth is not None else "default",
            width=320,
            active=False,
        )
        if weighted_vocal_bandwidth is None:
            vocal_toggle.disabled = True
            vocal_info = Div(
                text="Unified bandwidth unavailable (insufficient vocal-type coverage).",
                width=320,
            )
        else:
            vocal_info = Div(
                text=f"Weighted scalar: {weighted_vocal_bandwidth:.4f}",
                width=320,
            )

        empty_series = [None] * len(x_indices)
        empty_centroid_js = _empty_centroid_payload(frame_labels)
        toggle_callback = CustomJS(
            args=dict(
                dialect_toggle=dialect_toggle,
                vocal_toggle=vocal_toggle,
                slider=window_slider,
                labels=frame_labels,
                title=plot.title,
                base_title=base_title,
                group_line_sources=group_line_sources,
                group_point_sources=group_point_sources,
                mode_static_lines=mode_static_lines,
                mode_line_frames=mode_line_frames,
                mode_point_frames=mode_point_frames,
                empty_static=empty_static_payload,
                empty_line=empty_line_payload,
                empty_point=empty_point_payload,
                window_months=WINDOW_MONTHS,
                sample_box=sample_highlight,
                monthly_box=monthly_highlight,
                bandwidth_box=bandwidth_highlight,
                area_box=area_highlight,
                mode_source=mode_state_source,
                bandwidth_sources=bandwidth_sources,
                area_sources=area_sources,
                movement_sources=movement_sources,
                mode_bandwidth_values=mode_bandwidth_series,
                mode_area_values=mode_area_series,
                mode_centroid_payloads=mode_centroid_payloads,
                x_indices=x_indices,
                empty_series=empty_series,
                empty_centroid=empty_centroid_js,
            ),
            code="""
const dialectMode = "dialects_weighted";
const vocalMode = "vocal_types_weighted";
const hasDialect = !!mode_line_frames[dialectMode];
const hasVocal = !!mode_line_frames[vocalMode];
if (dialect_toggle && !hasDialect) {
    dialect_toggle.active = false;
}
if (vocal_toggle && !hasVocal) {
    vocal_toggle.active = false;
}
let mode = "per_group";
const dialectActive = !!(dialect_toggle && dialect_toggle.active);
const vocalActive = !!(vocal_toggle && vocal_toggle.active);
if (dialectActive && vocalActive) {
    if (cb_obj === vocal_toggle && hasVocal) {
        mode = vocalMode;
        if (dialect_toggle) { dialect_toggle.active = false; }
    } else if (cb_obj === dialect_toggle && hasDialect) {
        mode = dialectMode;
        if (vocal_toggle) { vocal_toggle.active = false; }
    } else if (hasVocal) {
        mode = vocalMode;
        if (dialect_toggle) { dialect_toggle.active = false; }
    } else if (hasDialect) {
        mode = dialectMode;
        if (vocal_toggle) { vocal_toggle.active = false; }
    }
} else if (vocalActive && hasVocal) {
    mode = vocalMode;
    if (dialect_toggle) { dialect_toggle.active = false; }
} else if (dialectActive && hasDialect) {
    mode = dialectMode;
    if (vocal_toggle) { vocal_toggle.active = false; }
} else {
    if (dialect_toggle) { dialect_toggle.active = false; }
    if (vocal_toggle) { vocal_toggle.active = false; }
}
const modeData = mode_source.data.mode || ["per_group"];
modeData[0] = mode;
mode_source.change.emit();
const span = window_months;
const total = labels.length;
if (!total) { return; }
let high = Math.round(slider.value[1]);
let low = high - (span - 1);
if (low < slider.start) {
    low = slider.start;
    high = low + span - 1;
}
if (high > slider.end) {
    high = slider.end;
    low = Math.max(slider.start, high - (span - 1));
}
slider.value = [low, high];
const idx = Math.max(0, Math.min(total - 1, high));
const staticData = mode_static_lines[mode] || {};
for (const label in group_line_sources) {
    const src = group_line_sources[label];
    const data = staticData[label];
    if (data && data.xs && data.xs.length) {
        src.data = data;
    } else {
        src.data = empty_static;
    }
    src.change.emit();
}
const lineFrameMap = mode_line_frames[mode] || {};
const pointFrameMap = mode_point_frames[mode] || {};
if (sample_box) {
    sample_box.left = low;
    sample_box.right = high;
}
if (monthly_box) {
    monthly_box.left = low;
    monthly_box.right = high;
}
if (bandwidth_box) {
    bandwidth_box.left = low;
    bandwidth_box.right = high;
}
if (area_box) {
    area_box.left = low;
    area_box.right = high;
}
for (const label in group_line_sources) {
    const frames = lineFrameMap[label] || [];
    const data = frames[idx] || empty_line;
    const source = group_line_sources[label];
    source.data = data;
    source.change.emit();
}
for (const label in group_point_sources) {
    const frames = pointFrameMap[label] || [];
    const data = frames[idx] || empty_point;
    const source = group_point_sources[label];
    source.data = data;
    source.change.emit();
}
const titleLabel = labels[idx] || "";
title.text = `${base_title} – ${titleLabel}`;
for (const label in bandwidth_sources) {
    const src = bandwidth_sources[label];
    const modeValues = mode_bandwidth_values[mode] || {};
    const values = modeValues[label];
    const y = Array.isArray(values) && values.length ? values : empty_series;
    src.data = {x: x_indices, y: y};
    src.change.emit();
}
for (const label in area_sources) {
    const src = area_sources[label];
    const modeValues = mode_area_values[mode] || {};
    const values = modeValues[label];
    const y = Array.isArray(values) && values.length ? values : empty_series;
    src.data = {x: x_indices, y: y};
    src.change.emit();
}
const centroidMode = mode_centroid_payloads[mode] || {};
for (const label in movement_sources) {
    const src = movement_sources[label];
    const payload = centroidMode[label];
    src.data = payload || empty_centroid;
    src.change.emit();
}
""",
        )
        dialect_toggle.js_on_change("active", toggle_callback)
        vocal_toggle.js_on_change("active", toggle_callback)

        slider_widgets = [range_slider]
        if chunk_slider is not None:
            slider_widgets.append(chunk_slider)
        slider_widgets.append(window_slider)
        slider_column = column(*slider_widgets)
        toggle_column = column(
            dialect_toggle,
            dialect_info,
            vocal_toggle,
            vocal_info,
        )
        controls_row = row(slider_column, toggle_column)
        map_section: Any = row(map_column, charts_column) if charts_column else map_column
        components: List[Any] = [controls_row, map_section]
        if movement_layout is not None:
            components.append(movement_layout)
        layout = column(*components) if len(components) > 1 else map_section
    else:
        map_section = row(map_column, charts_column) if charts_column else map_column
        components = [map_section]
        if movement_layout is not None:
            components.append(movement_layout)
        layout = column(*components) if len(components) > 1 else map_section

    show(layout)


def verify_spatiotemporal_completeness(table: GroupTable) -> None:
    """Report whether each group's rows have lat/lon/date populated."""
    required_columns = ["lat", "lon", "date"]
    for column in required_columns:
        if column not in table.data.columns:
            print(f"[WARN] {table.file_path.name}: missing '{column}' column.")
            return
    missing_mask = table.data[required_columns].isna().any(axis=1)
    missing_count = int(missing_mask.sum())
    if missing_count:
        print(
            f"[WARN] {table.file_path.name}: {missing_count} rows missing lat/lon/date values."
        )
    else:
        print(f"[OK] {table.file_path.name}: all rows include lat, lon, and date.")


def print_table_preview(group_tables: Iterable[GroupTable], head_rows: int = 5) -> None:
    """Print shape information and the first few rows for each loaded table."""
    tables = list(group_tables)
    for table in tables:
        df = table.data
        print("=" * 80)
        print(
            f"{table.group_type.upper()} | {table.file_path.name} | "
            f"{df.shape[0]} rows x {df.shape[1]} columns"
        )
        if df.empty:
            print("(empty table)")
            continue
        preview = df.head(head_rows)
        print(preview.to_string(index=False))
    if not tables:
        print("No group tables were loaded. Check that the config paths are correct.")


def main() -> None:
    """Entry point for the CLI."""
    args = parse_args()
    config = load_config(args.config)
    group_dirs = get_group_directories(config)
    if not group_dirs:
        raise SystemExit(
            "Config does not define 'paths.vocal_type_groups' or 'paths.dialect_groups'."
        )
    metadata_path = resolve_metadata_path(config)
    metadata_df = load_metadata_table(metadata_path)
    metadata_lookup = build_metadata_lookup(metadata_df)
    group_tables = load_all_group_tables(group_dirs)
    group_tables = [
        attach_metadata_to_group(table, metadata_lookup) for table in group_tables
    ]
    for table in group_tables:
        verify_spatiotemporal_completeness(table)
    transformer = build_equal_area_transformer()
    group_tables = [
        project_group_coordinates(table, transformer) for table in group_tables
    ]
    group_tables = [
        compute_group_kde(table) for table in group_tables
    ]
    for table in group_tables:
        log_kde_status(table)
    group_tables = [
        extract_isopleths(table) for table in group_tables
    ]
    if not args.no_plot:
        render_isopleth_map(group_tables)
    print_table_preview(group_tables, head_rows=args.head)


if __name__ == "__main__":
    main()
