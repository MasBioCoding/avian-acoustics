"""
Prototype Bokeh application for selecting species before building a
parallel-coordinate plot with optional HDBSCAN coloring plus a
playlist preview with audio and spectrograms.

Run with:
    bokeh serve --show xc_scripts/pcp.py --args --config xc_configs/config.yaml
"""

from __future__ import annotations

import argparse
import base64
import errno
import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
import hdbscan
import numpy as np
import pandas as pd
import umap
import yaml
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    BoxSelectTool,
    Button,
    CheckboxGroup,
    ColumnDataSource,
    CustomJS,
    Div,
    Select,
    Spinner,
    TapTool,
    TextInput,
    Toggle,
)
from bokeh.plotting import figure

APP_BACKGROUND_COLOR = "#fff0e9"
CARD_BACKGROUND_COLOR = "#ceddbb"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "xc_configs" / "config.yaml"
CATEGORY_FOLDERS = {
    "Vocal types": "vocal_types",
    "Dialects": "dialects",
    "Annotate 1": "annotate_1",
    "Annotate 2": "annotate_2",
    "PCP groups": "pcp_groups",
}
ACTIVE_PCP_GROUPS_LABEL = "active pcp_groups"
COLOR_PALETTE = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
]
HDBSCAN_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc949",
    "#af7aa1",
    "#ff9da7",
]
HDBSCAN_NOISE_LABEL = "Noise"
HDBSCAN_NOISE_COLOR = "#8c8c8c"
DEFAULT_UMAP_COMPONENTS = 6
DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.0
DEFAULT_UMAP_METRIC = "euclidean"
DEFAULT_UMAP_SEED = 42
PCP_BASE_COLOR = "#4477AA"
PCP_OTHER_COLOR = "#b0b0b0"
GRADIENT_LOW = "#2b83ba"
GRADIENT_HIGH = "#d7191c"
GRADIENT_MISSING = "#b0b0b0"
DEFAULT_PCP_ALPHA = 0.55
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 15
DEFAULT_HDBSCAN_MIN_SAMPLES = 5
DEFAULT_SPECTROGRAM_IMAGE_FORMAT = "png"
_STATIC_HTTP_SERVERS: dict[str, dict[str, Any]] = {}


def load_config(config_path: Path | None) -> dict[str, Any]:
    """Load configuration from YAML, falling back to minimal defaults when missing."""

    resolved_path = config_path or DEFAULT_CONFIG_PATH
    if not resolved_path.is_absolute():
        resolved_path = (Path.cwd() / resolved_path).resolve()

    config: dict[str, Any] = {}
    try:
        with resolved_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        print(f"Config not found at {resolved_path}; using defaults.")

    paths_config = config.setdefault("paths", {})
    paths_config.setdefault("root", str(Path.cwd()))
    return config


def start_static_file_server(
    *,
    label: str,
    directory: Path,
    host: str,
    port: int,
    log_requests: bool = False,
) -> tuple[str, bool] | None:
    """Start or reuse a background HTTP server to expose local files."""

    existing = _STATIC_HTTP_SERVERS.get(label)
    if existing:
        return existing["base_url"], existing["started"]

    if not directory.exists():
        print(f"  {label} directory missing, cannot auto-serve: {directory}")
        return None

    class _StaticHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:  # type: ignore[override]
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


def build_media_url(base_url: str | None, species_slug: str, filename: str) -> str:
    """Construct a media URL, handling base URLs with or without species suffixes."""

    if not base_url:
        return ""
    normalized = base_url.rstrip("/")
    suffix = f"/{species_slug}"
    if normalized.endswith(suffix):
        return f"{normalized}/{filename}"
    return f"{normalized}{suffix}/{filename}"


def _coerce_clip_index(value: Any) -> int | None:
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


def add_media_columns(
    metadata_df: pd.DataFrame,
    *,
    species_slug: str,
    audio_base_url: str | None,
    spectrogram_base_url: str | None,
    spectrogram_dir: Path,
    spectrogram_image_format: str,
    inline_spectrograms: bool,
) -> pd.DataFrame:
    """Return metadata with audio/spectrogram URLs for playlist rendering."""

    metadata = metadata_df.copy()
    row_count = len(metadata)
    if row_count == 0:
        metadata["audio_url"] = []
        metadata["spectrogram_url"] = []
        metadata["spectrogram_exists"] = []
        metadata["spectrogram_data_uri"] = []
        return metadata

    has_clip_file = "clip_file" in metadata.columns
    has_file_path = "file_path" in metadata.columns
    has_xcid = "xcid" in metadata.columns
    has_clip_index = "clip_index" in metadata.columns

    def _audio_filename(row: pd.Series) -> str:
        if has_clip_file and pd.notna(row.get("clip_file")):
            return Path(str(row["clip_file"])).name
        if has_file_path and pd.notna(row.get("file_path")):
            return Path(str(row["file_path"])).name
        if has_xcid and has_clip_index:
            clip_index = _coerce_clip_index(row.get("clip_index"))
            xcid = row.get("xcid")
            if clip_index is None or pd.isna(xcid):
                return ""
            return f"{species_slug}_{str(xcid).strip()}_{clip_index:02d}.wav"
        return ""

    if audio_base_url:
        audio_files = metadata.apply(_audio_filename, axis=1)
        metadata["audio_url"] = [
            build_media_url(audio_base_url, species_slug, filename)
            if filename
            else ""
            for filename in audio_files
        ]
    else:
        metadata["audio_url"] = [""] * row_count

    def _spectrogram_basename(row: pd.Series) -> str:
        if has_clip_file and pd.notna(row.get("clip_file")):
            return Path(str(row["clip_file"])).stem
        if has_file_path and pd.notna(row.get("file_path")):
            return Path(str(row["file_path"])).stem
        if has_xcid and has_clip_index:
            clip_index = _coerce_clip_index(row.get("clip_index"))
            xcid = row.get("xcid")
            if clip_index is None or pd.isna(xcid):
                return ""
            return f"{species_slug}_{str(xcid).strip()}_{clip_index:02d}"
        return ""

    spectrogram_urls: list[str] = []
    spectrogram_exists: list[bool] = []
    spectrogram_data_uri: list[str] = []
    image_format = spectrogram_image_format.lower().lstrip(".")

    def _inline_spectrogram(path: Path) -> str:
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:image/{image_format};base64,{encoded}"
        except FileNotFoundError:
            return ""
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: failed to inline {path.name}: {exc}")
            return ""

    for _, row in metadata.iterrows():
        base_name = _spectrogram_basename(row)
        if not base_name:
            spectrogram_urls.append("")
            spectrogram_exists.append(False)
            spectrogram_data_uri.append("")
            continue
        filename = f"{base_name}.{image_format}"
        image_path = spectrogram_dir / filename
        if image_path.exists():
            spectrogram_urls.append(
                build_media_url(spectrogram_base_url, species_slug, filename)
                if spectrogram_base_url
                else ""
            )
            spectrogram_exists.append(True)
            if inline_spectrograms:
                spectrogram_data_uri.append(_inline_spectrogram(image_path))
            else:
                spectrogram_data_uri.append("")
        else:
            spectrogram_urls.append("")
            spectrogram_exists.append(False)
            spectrogram_data_uri.append("")

    metadata["spectrogram_url"] = spectrogram_urls
    metadata["spectrogram_exists"] = spectrogram_exists
    metadata["spectrogram_data_uri"] = spectrogram_data_uri
    return metadata


def list_species_slugs(groups_root: Path) -> list[str]:
    """Return sorted species slugs discovered in the xc_groups directory."""

    if not groups_root.exists():
        print(f"No xc_groups directory found at {groups_root}.")
        return []

    return sorted(entry.name for entry in groups_root.iterdir() if entry.is_dir())


def collect_group_entries(
    category_root: Path, *, label_from_stem: bool = False
) -> list[dict[str, Any]]:
    """Gather group entries for a given category directory."""

    if not category_root.exists():
        return []

    stems = {p.stem for p in category_root.glob("*.txt")}
    stems.update(p.stem for p in category_root.glob("*.csv"))

    def count_csv_rows(csv_path: Path) -> int:
        try:
            with csv_path.open("r", encoding="utf-8") as handle:
                return max(sum(1 for _ in handle) - 1, 0)
        except OSError:
            return 0

    entries: list[dict[str, str]] = []
    for idx, stem in enumerate(sorted(stems), start=1):
        desc_path = category_root / f"{stem}.txt"
        csv_path = category_root / f"{stem}.csv"
        description = ""
        if desc_path.exists():
            try:
                description = desc_path.read_text(encoding="utf-8").strip()
            except OSError:
                description = ""

        entry_count = count_csv_rows(csv_path) if csv_path.exists() else 0
        label_value = stem if label_from_stem else f"group_{idx}"
        entries.append(
            {
                "label": label_value,
                "stem": stem,
                "description": description,
                "csv_path": str(csv_path) if csv_path.exists() else "",
                "entry_count": entry_count,
            }
        )

    return entries


def collect_groups_for_species(
    groups_root: Path, species_slug: str
) -> dict[str, list[dict[str, Any]]]:
    """Return category -> group entries for a species slug."""

    species_root = groups_root / species_slug
    grouped: dict[str, list[dict[str, Any]]] = {}
    for category_label, folder_name in CATEGORY_FOLDERS.items():
        grouped[category_label] = collect_group_entries(
            species_root / folder_name,
            label_from_stem=folder_name == "pcp_groups",
        )
    return grouped


def format_descriptions(groups_by_category: dict[str, list[dict[str, Any]]]) -> str:
    """Build HTML for the description panel."""

    sections: list[str] = []
    for category_label, items in groups_by_category.items():
        if not items:
            continue

        entries_html = []
        for item in items:
            if item["description"]:
                description_html = html.escape(item["description"]).replace("\n", "<br>")
            else:
                description_html = "<em>No description available.</em>"
            entries_html.append(
                f"<li><strong>{html.escape(item['label'])}</strong>: {description_html}</li>"
            )

        section_html = (
            f"<div class='desc-category'>"
            f"<div class='desc-title'>{html.escape(category_label)}</div>"
            f"<ul class='desc-list'>{''.join(entries_html)}</ul>"
            f"</div>"
        )
        sections.append(section_html)

    if not sections:
        return "<em>No group descriptions found for this species.</em>"

    return "<div class='desc-wrapper'>" + "".join(sections) + "</div>"


def extract_key_from_path(path_str: str) -> tuple[str, int] | None:
    """Derive (xcid, clip_index) from a file path or name."""

    try:
        stem = Path(str(path_str)).stem
    except Exception:
        return None
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    xcid_candidate = parts[-2].strip()
    clip_candidate = parts[-1].strip()
    if not xcid_candidate:
        return None
    try:
        clip_idx = int(clip_candidate.lstrip("0") or "0")
    except ValueError:
        return None
    return xcid_candidate, clip_idx


def keys_from_dataframe(df: pd.DataFrame) -> set[tuple[str, int]]:
    """Extract unique (xcid, clip_index) keys from embeddings/metadata DataFrame."""

    if {"xcid", "clip_index"}.issubset(df.columns):
        clean_df = df.dropna(subset=["xcid", "clip_index"]).copy()
        clean_df["xcid"] = clean_df["xcid"].astype(str).str.strip()
        clean_df["clip_index"] = pd.to_numeric(clean_df["clip_index"], errors="coerce")
        clean_df = clean_df.dropna(subset=["clip_index"])
        clean_df["clip_index"] = clean_df["clip_index"].astype(int)
        return set(zip(clean_df["xcid"], clean_df["clip_index"]))

    if "file_path" in df.columns:
        keys: set[tuple[str, int]] = set()
        for path_value in df["file_path"].dropna():
            key = extract_key_from_path(path_value)
            if key:
                keys.add(key)
        return keys

    return set()


def append_key_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the DataFrame with a __key__ column for (xcid, clip_index)."""

    working = df.copy()
    if {"xcid", "clip_index"}.issubset(working.columns):
        xcid_series = working["xcid"].astype(str).str.strip()
        clip_series = pd.to_numeric(working["clip_index"], errors="coerce")
        mask = clip_series.notna() & xcid_series.astype(bool)
        working = working.loc[mask].copy()
        working["xcid"] = xcid_series.loc[mask]
        working["clip_index"] = clip_series.loc[mask].astype(int)
        working["__key__"] = list(zip(working["xcid"], working["clip_index"]))
        return working

    if "file_path" in working.columns:
        working["__key__"] = working["file_path"].apply(extract_key_from_path)
        working = working.dropna(subset=["__key__"])
        return working

    working["__key__"] = None
    return working


def key_to_str(key: tuple[str, int] | None) -> str:
    """Convert a (xcid, clip_index) tuple to a compact string."""

    if not key or len(key) != 2:
        return ""
    return f"{key[0]}_{key[1]}"


def sanitize_group_name(raw_name: str) -> str:
    """Normalize a user-supplied group name into a safe filename stem."""

    if not raw_name:
        return ""
    stripped = raw_name.strip()
    if not stripped:
        return ""
    cleaned_chars: list[str] = []
    for char in stripped:
        if char.isascii() and (char.isalnum() or char in {"-", "_"}):
            cleaned_chars.append(char)
        elif char in {" ", "."}:
            cleaned_chars.append("_")
        else:
            cleaned_chars.append("_")
    return "".join(cleaned_chars).strip("_")


def format_pcp_group_description(
    description: str,
    umap_params: dict[str, Any] | None,
    hdbscan_params: dict[str, Any] | None,
) -> str:
    """Create the .txt contents for a saved PCP group."""

    lines: list[str] = ["Description:"]
    desc_clean = (description or "").strip()
    lines.append(desc_clean or "No description provided.")
    lines.append("")
    lines.append("UMAP parameters:")
    if umap_params:
        lines.append(f"- n_components: {umap_params.get('n_components', 'n/a')}")
        lines.append(f"- n_neighbors: {umap_params.get('n_neighbors', 'n/a')}")
        lines.append(f"- min_dist: {umap_params.get('min_dist', 'n/a')}")
        lines.append(f"- metric: {umap_params.get('metric', 'n/a')}")
        lines.append(f"- seed: {umap_params.get('seed', 'n/a')}")
    else:
        lines.append("- not run yet")
    lines.append("")
    lines.append("HDBSCAN parameters:")
    if hdbscan_params:
        lines.append(
            f"- min_cluster_size: {hdbscan_params.get('min_cluster_size', 'n/a')}"
        )
        lines.append(f"- min_samples: {hdbscan_params.get('min_samples', 'n/a')}")
    else:
        lines.append("- not run yet")
    return "\n".join(lines).strip() + "\n"


def _hex_component(value: float) -> str:
    return f"{int(max(0, min(255, round(value)))):02x}"


def interpolate_color(start_hex: str, end_hex: str, fraction: float) -> str:
    """Linearly interpolate between two hex colors."""

    fraction = max(0.0, min(1.0, fraction))
    sh = start_hex.lstrip("#")
    eh = end_hex.lstrip("#")
    sr, sg, sb = int(sh[0:2], 16), int(sh[2:4], 16), int(sh[4:6], 16)
    er, eg, eb = int(eh[0:2], 16), int(eh[2:4], 16), int(eh[4:6], 16)
    r = sr + (er - sr) * fraction
    g = sg + (eg - sg) * fraction
    b = sb + (eb - sb) * fraction
    return f"#{_hex_component(r)}{_hex_component(g)}{_hex_component(b)}"


def embedding_matrix_from_dataframe(embedding_df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """Convert embedding columns to a matrix and return indices that were used."""

    if "embedding" in embedding_df.columns:
        arrays: list[np.ndarray] = []
        valid_indices: list[int] = []
        expected_length: int | None = None
        for idx, value in embedding_df["embedding"].items():
            vector = np.fromstring(str(value), sep=",")
            if vector.size == 0:
                continue
            if expected_length is None:
                expected_length = vector.size
            if vector.size != expected_length:
                raise ValueError("Inconsistent embedding lengths detected in CSV.")
            arrays.append(vector)
            valid_indices.append(idx)

        if not arrays:
            raise ValueError("No usable embedding vectors found.")

        return np.vstack(arrays), valid_indices

    embedding_cols = [col for col in embedding_df.columns if col.startswith("embedding_")]
    if embedding_cols:
        return embedding_df[embedding_cols].to_numpy(), list(embedding_df.index)

    raise ValueError("Embedding columns not found in embeddings.csv.")


def create_layout(*, species_options: list[str], groups_root: Path) -> None:
    """Compose and register the Bokeh document layout."""

    current_species_slug: str | None = None
    current_embeddings_path: Path | None = None
    current_umap_projection: np.ndarray | None = None
    current_umap_metadata: pd.DataFrame | None = None
    current_hdbscan_labels: list[str] | None = None
    current_hdbscan_color_map: dict[str, str] = {}
    current_point_base: dict[str, list[Any]] | None = None
    current_point_dims = 0
    active_pcp_groups: list[dict[str, Any]] = []
    next_pcp_group_id = 1
    last_umap_params: dict[str, Any] | None = None
    last_hdbscan_params: dict[str, Any] | None = None
    last_save_nonce = ""

    safe_options = species_options or ["No species found"]
    species_select = Select(
        title="Species slug",
        value=safe_options[0],
        options=safe_options,
        width=260,
    )
    load_button = Button(label="Load", button_type="primary", width=110)

    header = Div(
        text="Load species",
        styles={
            "font-size": "16px",
            "font-weight": "600",
            "margin-bottom": "8px",
            "color": "#3a3426",
        },
    )

    card = column(
        header,
        row(species_select, load_button, sizing_mode="fixed"),
        width=560,
        sizing_mode="fixed",
        css_classes=["control-card"],
        styles={
            "background-color": CARD_BACKGROUND_COLOR,
            "border-radius": "12px",
            "padding": "12px 14px",
            "box-shadow": "0 3px 10px rgba(0, 0, 0, 0.08)",
        },
    )

    description = Div(
        text=(
            "Select a species slug from xc_groups and load it to view available group "
            "selections and their descriptions."
        ),
        styles={"margin-top": "12px", "color": "#4a452c"},
        width=520,
    )

    location_hint = Div(
        text=f"Looking for species under: <code>{groups_root}</code>",
        styles={"margin-top": "6px", "color": "#5a543a"},
        width=520,
    )

    calculate_button = Button(label="Calculate", button_type="primary", width=100)

    umap_components_spinner = Spinner(
        title="Number of dimensions",
        low=2,
        high=310,
        step=1,
        value=DEFAULT_UMAP_COMPONENTS,
        width=100,
    )
    umap_neighbors_spinner = Spinner(
        title="Number of neighbors",
        low=2,
        high=300,
        step=1,
        value=DEFAULT_UMAP_NEIGHBORS,
        width=100,
    )
    umap_min_dist_spinner = Spinner(
        title="Minimum distance",
        low=0,
        high=10,
        step=0.1,
        value=DEFAULT_UMAP_MIN_DIST,
        width=100,
    )
    umap_metric_input = TextInput(
        title="Metric",
        value=DEFAULT_UMAP_METRIC,
        width=120,
    )
    umap_seed_spinner = Spinner(
        title="Seed",
        low=0,
        high=2_147_483_647,
        step=1,
        value=DEFAULT_UMAP_SEED,
        width=120,
    )
    missing_metadata_warning = Div(
        text="",
        render_as_text=False,
        visible=False,
        styles={
            "background-color": "#fff4ce",
            "color": "#5a4a00",
            "padding": "6px 8px",
            "border-radius": "8px",
            "margin-top": "6px",
            "border": "1px solid #e6c200",
            "font-size": "12px",
        },
        width=240,
    )
    umap_status_box = Div(
        text="<em>UMAP will run on selected groups or all recordings.</em>",
        render_as_text=False,
        styles={"color": "#2d2616", "margin-top": "6px"},
        width=240,
    )

    umap_parameters_panel = column(
        Div(
            text="UMAP parameters",
            styles={
                "font-size": "15px",
                "font-weight": "600",
                "margin-bottom": "6px",
                "color": "#2d2616",
            },
        ),
        umap_components_spinner,
        umap_neighbors_spinner,
        umap_min_dist_spinner,
        umap_metric_input,
        umap_seed_spinner,
        calculate_button,
        missing_metadata_warning,
        umap_status_box,
        width=260,
        sizing_mode="fixed",
        css_classes=["control-card"],
        styles={
            "background-color": CARD_BACKGROUND_COLOR,
            "border-radius": "12px",
            "padding": "12px 14px",
            "box-shadow": "0 3px 10px rgba(0, 0, 0, 0.08)",
        },
    )

    hdbscan_min_cluster_size_spinner = Spinner(
        title="Min cluster size",
        low=2,
        high=500,
        step=1,
        value=DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
        width=120,
    )
    hdbscan_min_samples_spinner = Spinner(
        title="Min samples",
        low=1,
        high=300,
        step=1,
        value=DEFAULT_HDBSCAN_MIN_SAMPLES,
        width=120,
    )
    hdbscan_compute_button = Button(label="Compute", button_type="primary", width=100)
    hdbscan_status_box = Div(
        text="<em>Run UMAP first, then compute clusters.</em>",
        render_as_text=False,
        styles={"color": "#2d2616", "margin-top": "6px"},
        width=240,
    )
    hdbscan_panel = column(
        Div(
            text="HDBSCAN",
            styles={
                "font-size": "15px",
                "font-weight": "600",
                "margin-bottom": "6px",
                "color": "#2d2616",
            },
        ),
        hdbscan_min_cluster_size_spinner,
        hdbscan_min_samples_spinner,
        hdbscan_compute_button,
        hdbscan_status_box,
        width=260,
        sizing_mode="fixed",
        css_classes=["control-card"],
        styles={
            "background-color": CARD_BACKGROUND_COLOR,
            "border-radius": "12px",
            "padding": "12px 14px",
            "box-shadow": "0 3px 10px rgba(0, 0, 0, 0.08)",
        },
    )

    umap_checkbox_groups: dict[str, CheckboxGroup] = {}
    umap_items_by_category: dict[str, list[dict[str, Any]]] = {}
    color_checkbox_groups: dict[str, CheckboxGroup] = {}
    color_items_by_category: dict[str, list[dict[str, Any]]] = {}
    checkbox_sections_holder = row(
        sizing_mode="scale_width",
        styles={"gap": "6px", "flex-wrap": "wrap", "align-items": "flex-start"},
    )
    color_sections_holder = row(
        sizing_mode="scale_width",
        styles={"gap": "6px", "flex-wrap": "wrap", "align-items": "flex-start"},
    )
    hdbscan_clusters_label = Div(
        text="HDBSCAN clusters",
        visible=False,
        styles={
            "font-weight": "600",
            "margin-top": "8px",
            "margin-bottom": "2px",
            "color": "#211b10",
        },
        width=200,
    )
    hdbscan_checklist = CheckboxGroup(
        labels=[],
        active=[],
        visible=False,
        width=200,
    )
    descriptions_box = Div(
        text="<em>Load a species to view descriptions.</em>",
        render_as_text=False,
        styles={"color": "#3a3426"},
        width=520,
    )
    embeddings_counts_box = Div(
        text="<em>Press Calculate to load embeddings counts.</em>",
        render_as_text=False,
        styles={"color": "#3a3426"},
        width=520,
    )
    reconciliation_box = Div(
        text="<em>Press Calculate for reconciliation stats.</em>",
        render_as_text=False,
        styles={"color": "#3a3426"},
        width=520,
    )

    umap_title = Div(
        text="Calculate UMAP for:",
        styles={
            "font-size": "15px",
            "font-weight": "600",
            "margin-bottom": "6px",
            "color": "#2d2616",
        },
    )

    color_title = Div(
        text="Color by:",
        styles={
            "font-size": "15px",
            "font-weight": "600",
            "margin-bottom": "6px",
            "color": "#2d2616",
        },
    )

    all_toggle_label = Div(
        text="All recordings",
        styles={
            "font-size": "13px",
            "font-weight": "600",
            "margin-right": "6px",
            "color": "#2d2616",
        },
        width=100,
    )
    all_toggle_count = Div(
        text="(0)",
        styles={"font-size": "12px", "color": "#2d2616", "margin-right": "6px"},
        width=60,
    )
    all_toggle = Toggle(label="Select", button_type="success", width=90)
    color_assignments_box = Div(
        text="<em>Select up to 5 groups to assign colors.</em>",
        render_as_text=False,
        styles={"color": "#3a3426", "margin-top": "6px"},
        width=500,
    )
    color_mode_select = Select(
        title="Color mode",
        value="Groups",
        options=["Groups", "Latitude", "Longitude", "Altitude", "HDBSCAN"],
        width=160,
    )
    active_pcp_groups_header = Div(
        text=ACTIVE_PCP_GROUPS_LABEL,
        styles={
            "font-weight": "600",
            "margin-top": "6px",
            "margin-bottom": "2px",
            "color": "#211b10",
        },
    )
    active_pcp_groups_placeholder = Div(
        text="<em>No active groups.</em>",
        render_as_text=False,
        styles={"color": "#4a452c"},
    )
    active_pcp_groups_checkbox = CheckboxGroup(labels=[], active=[], width=150)
    active_pcp_groups_section = column(
        active_pcp_groups_header,
        active_pcp_groups_placeholder,
        spacing=2,
        width=150,
        sizing_mode="fixed",
    )
    create_group_button = Button(
        label="Create group from selection", button_type="primary", width=220
    )
    clear_groups_button = Button(label="Clear groups", button_type="warning", width=220)
    save_group_button = Button(label="Save group", button_type="success", width=220)
    annotation_status_box = Div(
        text="<em>Select points with box select to create a group.</em>",
        render_as_text=False,
        styles={"color": "#2d2616", "margin-top": "6px"},
        width=220,
    )
    pcp_status_box = Div(
        text="<em>Calculate to generate the parallel coordinates plot.</em>",
        render_as_text=False,
        styles={"color": "#2d2616", "margin-top": "6px"},
        width=1120,
    )
    pcp_source = ColumnDataSource(
        data={
            "xs": [],
            "ys": [],
            "color": [],
            "key": [],
            "alpha": [],
            "hdbscan_label": [],
            "playlist_active": [],
            "xcid": [],
            "clip_index": [],
            "date": [],
            "audio_url": [],
            "spectrogram_url": [],
            "spectrogram_data_uri": [],
        }
    )
    pcp_points_source = ColumnDataSource(
        data={
            "x": [],
            "y": [],
            "dim": [],
            "record_index": [],
            "color": [],
            "alpha": [],
            "playlist_active": [],
        }
    )
    save_request_source = ColumnDataSource(
        data={"name": [], "description": [], "nonce": []}
    )
    pcp_plot = figure(
        height=620,
        sizing_mode="stretch_width",
        toolbar_location="above",
        tools="pan,wheel_zoom,reset",
        background_fill_color="#f7f4ed",
    )
    pcp_plot.multi_line(
        xs="xs",
        ys="ys",
        line_color="color",
        line_alpha="alpha",
        line_width=1.2,
        source=pcp_source,
    )
    tap_tool = TapTool()
    pcp_plot.add_tools(tap_tool)
    box_select_tool = BoxSelectTool()
    pcp_plot.add_tools(box_select_tool)
    pcp_plot.toolbar.active_tap = tap_tool
    pcp_plot.toolbar.active_drag = box_select_tool
    pcp_plot.circle(
        x="x",
        y="y",
        size=5,
        fill_color="color",
        line_color="color",
        fill_alpha="alpha",
        line_alpha="alpha",
        source=pcp_points_source,
    )
    pcp_plot.xaxis.axis_label = "UMAP dimensions"
    pcp_plot.yaxis.axis_label = "UMAP value"
    pcp_plot.grid.minor_grid_line_color = None
    pcp_plot.toolbar.autohide = True
    pcp_plot.xaxis.ticker = []

    pcp_panel = column(
        Div(
            text="Parallel coordinates (UMAP dimensions)",
            styles={
                "font-size": "15px",
                "font-weight": "600",
                "margin-bottom": "6px",
                "color": "#2d2616",
            },
        ),
        pcp_status_box,
        pcp_plot,
        sizing_mode="stretch_width",
        width=1200,
        css_classes=["control-card"],
        styles={
            "background-color": CARD_BACKGROUND_COLOR,
            "border-radius": "12px",
            "padding": "12px 14px",
            "box-shadow": "0 3px 10px rgba(0, 0, 0, 0.08)",
        },
    )

    playlist_title = Div(
        text="Playlist",
        styles={
            "font-size": "15px",
            "font-weight": "600",
            "margin-bottom": "6px",
            "color": "#2d2616",
        },
    )
    playlist_panel = Div(
        text="<em>Click a point on a dimension axis to list nearby recordings.</em>",
        render_as_text=False,
        width=1000,
        height=1000,
        styles={
            "border": "1px solid #d6d0c2",
            "border-radius": "8px",
            "padding": "8px",
            "background-color": "#fffaf2",
            "overflow-y": "auto",
        },
    )
    playlist_column = column(
        playlist_title,
        playlist_panel,
        width=1100,
        sizing_mode="fixed",
        css_classes=["control-card"],
        styles={
            "background-color": CARD_BACKGROUND_COLOR,
            "border-radius": "12px",
            "padding": "12px 14px",
            "box-shadow": "0 3px 10px rgba(0, 0, 0, 0.08)",
        },
    )

    def on_all_toggle(active: bool) -> None:
        for cb in umap_checkbox_groups.values():
            cb.active = []
            cb.disabled = bool(active)

    all_toggle.on_click(on_all_toggle)

    def on_checkbox_change(attr: str, old: list[int], new: list[int]) -> None:
        if new and all_toggle.active:
            all_toggle.active = False
            for cb in umap_checkbox_groups.values():
                cb.disabled = False

    def on_color_mode_change(attr: str, old: str, new: str) -> None:
        update_color_assignments()
        apply_color_selection()
    color_mode_select.on_change("value", on_color_mode_change)

    def update_color_assignments() -> None:
        mode = color_mode_select.value
        if mode == "HDBSCAN":
            if not current_hdbscan_labels:
                color_assignments_box.text = (
                    "<em>Compute HDBSCAN to enable cluster coloring.</em>"
                )
                hdbscan_checklist.visible = False
                hdbscan_clusters_label.visible = False
                return

            hdbscan_checklist.visible = True
            hdbscan_clusters_label.visible = True
            labels = hdbscan_checklist.labels or []
            active_indices = set(hdbscan_checklist.active)
            html_entries: list[str] = []
            for idx, label in enumerate(labels):
                color = current_hdbscan_color_map.get(label, PCP_OTHER_COLOR)
                color_box = (
                    f"<span style='display:inline-block;width:14px;height:14px;"
                    f"background:{color};margin-right:6px;border-radius:3px;"
                    f"border:1px solid #d2d2d2;'></span>"
                )
                visibility_note = "" if idx in active_indices else " (hidden)"
                html_entries.append(
                    f"<div style='margin-bottom:4px;'>{color_box}"
                    f"{html.escape(label)}{visibility_note}</div>"
                )

            if html_entries:
                color_assignments_box.text = "".join(html_entries)
            else:
                color_assignments_box.text = (
                    "<em>Toggle clusters with the checklist.</em>"
                )
            return

        hdbscan_checklist.visible = False
        hdbscan_clusters_label.visible = False
        if mode != "Groups":
            color_assignments_box.text = "<em>Using gradient color mode.</em>"
            return

        active_entries = collect_color_entries()
        html_entries: list[str] = []
        for idx, (cat, entry) in enumerate(active_entries):
            label = entry.get("label") or entry.get("stem") or f"group_{idx + 1}"
            color = COLOR_PALETTE[idx] if idx < len(COLOR_PALETTE) else "#b0b0b0"
            color_box = (
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background:{color};margin-right:6px;border-radius:3px;'></span>"
            )
            html_entries.append(
                f"<div style='margin-bottom:4px;'>{color_box}"
                f"{html.escape(cat)} / {html.escape(label)}</div>"
            )

        if html_entries:
            color_assignments_box.text = "".join(html_entries)
        else:
            color_assignments_box.text = "<em>Select up to 5 groups to assign colors.</em>"

    def on_hdbscan_checklist_change(
        attr: str, old: list[int], new: list[int]
    ) -> None:
        update_color_assignments()
        apply_color_selection()

    hdbscan_checklist.on_change("active", on_hdbscan_checklist_change)

    def on_active_pcp_groups_change(
        attr: str, old: list[int], new: list[int]
    ) -> None:
        update_color_assignments()
        apply_color_selection()

    active_pcp_groups_checkbox.on_change("active", on_active_pcp_groups_change)

    def refresh_active_pcp_groups_panel(select_new: bool = False) -> None:
        """Sync the active PCP group checklist with the current group list."""

        if not active_pcp_groups:
            active_pcp_groups_checkbox.labels = []
            active_pcp_groups_checkbox.active = []
            active_pcp_groups_section.children = [
                active_pcp_groups_header,
                active_pcp_groups_placeholder,
            ]
            save_group_button.disabled = True
            clear_groups_button.disabled = True
            return

        labels = [group.get("label", "group") for group in active_pcp_groups]
        active_pcp_groups_checkbox.labels = labels
        active_indices = set(active_pcp_groups_checkbox.active)
        if select_new:
            active_indices.add(len(labels) - 1)
        active_pcp_groups_checkbox.active = sorted(
            idx for idx in active_indices if 0 <= idx < len(labels)
        )
        active_pcp_groups_section.children = [
            active_pcp_groups_header,
            active_pcp_groups_checkbox,
        ]
        save_group_button.disabled = False
        clear_groups_button.disabled = False

    refresh_active_pcp_groups_panel()

    def collect_selected_entries() -> list[tuple[str, dict[str, Any]]]:
        selected_entries: list[tuple[str, Any]] = []
        for category_label, checkbox in umap_checkbox_groups.items():
            items = umap_items_by_category.get(category_label, [])
            for idx in checkbox.active:
                if 0 <= idx < len(items):
                    selected_entries.append((category_label, items[idx]))
        return selected_entries

    def collect_color_entries() -> list[tuple[str, dict[str, Any]]]:
        """Return (category, entry) pairs chosen for color-by selections."""

        selected_entries: list[tuple[str, Any]] = []
        for category_label, checkbox in color_checkbox_groups.items():
            items = color_items_by_category.get(category_label, [])
            for idx in checkbox.active:
                if 0 <= idx < len(items):
                    selected_entries.append((category_label, items[idx]))
        for idx in active_pcp_groups_checkbox.active:
            if 0 <= idx < len(active_pcp_groups):
                selected_entries.append(
                    (ACTIVE_PCP_GROUPS_LABEL, active_pcp_groups[idx])
                )
        return selected_entries

    def clear_hdbscan_results() -> None:
        """Reset HDBSCAN results and related UI state."""

        nonlocal current_hdbscan_labels, current_hdbscan_color_map
        current_hdbscan_labels = None
        current_hdbscan_color_map = {}
        hdbscan_checklist.labels = []
        hdbscan_checklist.active = []
        hdbscan_checklist.visible = False
        hdbscan_clusters_label.visible = False
        hdbscan_status_box.text = "<em>Run UMAP first, then compute clusters.</em>"
        if pcp_source.data.get("xs"):
            current_data = pcp_source.data
            point_count = len(current_data.get("key", []))
            playlist_active = current_data.get(
                "playlist_active", [True] * point_count
            )
            pcp_source.data = {
                "xs": current_data.get("xs", []),
                "ys": current_data.get("ys", []),
                "color": current_data.get("color", []),
                "key": current_data.get("key", []),
                "alpha": current_data.get(
                    "alpha", [DEFAULT_PCP_ALPHA] * point_count
                ),
                "hdbscan_label": [""] * point_count,
                "playlist_active": playlist_active,
                "xcid": current_data.get("xcid", []),
                "clip_index": current_data.get("clip_index", []),
                "date": current_data.get("date", []),
                "audio_url": current_data.get("audio_url", []),
                "spectrogram_url": current_data.get("spectrogram_url", []),
                "spectrogram_data_uri": current_data.get("spectrogram_data_uri", []),
            }
            update_point_styles(
                current_data.get("color", []),
                current_data.get("alpha", []),
                playlist_active,
            )
        update_color_assignments()

    def update_point_styles(
        colors: list[str],
        alpha: list[float],
        playlist_active: list[bool],
    ) -> None:
        nonlocal current_point_base, current_point_dims

        if not current_point_base or current_point_dims <= 0:
            pcp_points_source.data = {
                "x": [],
                "y": [],
                "dim": [],
                "record_index": [],
                "color": [],
                "alpha": [],
                "playlist_active": [],
            }
            return

        point_colors = np.repeat(colors, current_point_dims).tolist()
        point_alpha = np.repeat(alpha, current_point_dims).tolist()
        point_active = np.repeat(playlist_active, current_point_dims).tolist()
        updated = dict(current_point_base)
        updated["color"] = point_colors
        updated["alpha"] = point_alpha
        updated["playlist_active"] = point_active
        pcp_points_source.data = updated

    def apply_color_selection() -> None:
        """Recolor PCP lines based on current color-by selections."""

        if current_umap_metadata is None:
            return
        if not pcp_source.data.get("xs"):
            return

        mode = color_mode_select.value
        keys = pcp_source.data.get("key", [])
        labels = pcp_source.data.get("hdbscan_label", [])
        point_count = len(keys)
        if len(labels) != point_count:
            labels = [""] * point_count
        base_alpha = [DEFAULT_PCP_ALPHA] * point_count

        def _update_source(
            colors: list[str],
            alpha: list[float] | None = None,
            playlist_active: list[bool] | None = None,
        ) -> None:
            if alpha is None:
                alpha = base_alpha
            if playlist_active is None:
                playlist_active = [True] * point_count
            current_data = pcp_source.data
            pcp_source.data = {
                "xs": current_data.get("xs", []),
                "ys": current_data.get("ys", []),
                "color": colors,
                "key": keys,
                "alpha": alpha,
                "hdbscan_label": labels,
                "playlist_active": playlist_active,
                "xcid": current_data.get("xcid", []),
                "clip_index": current_data.get("clip_index", []),
                "date": current_data.get("date", []),
                "audio_url": current_data.get("audio_url", []),
                "spectrogram_url": current_data.get("spectrogram_url", []),
                "spectrogram_data_uri": current_data.get("spectrogram_data_uri", []),
            }
            update_point_styles(colors, alpha, playlist_active)

        def _apply_group_colors() -> tuple[list[str], list[bool]]:
            color_entries = collect_color_entries()
            if not color_entries:
                return [PCP_BASE_COLOR] * point_count, [False] * point_count

            def _entry_key_strings(entry: dict[str, Any]) -> set[str]:
                key_set = entry.get("key_set")
                if isinstance(key_set, set):
                    return key_set
                if isinstance(key_set, list):
                    return {str(key) for key in key_set if key}

                csv_path = Path(entry.get("csv_path") or "")
                if not csv_path.exists():
                    return set()
                try:
                    group_df = pd.read_csv(csv_path)
                except Exception:
                    return set()
                group_keys_df = append_key_column(group_df)
                key_strings: set[str] = set()
                for key in group_keys_df["__key__"]:
                    key_str = key_to_str(key)
                    if key_str:
                        key_strings.add(key_str)
                return key_strings

            key_to_color: dict[str, str] = {}
            for color_idx, (_category_label, entry) in enumerate(color_entries):
                if color_idx >= len(COLOR_PALETTE):
                    break
                color_value = COLOR_PALETTE[color_idx]
                for key_str in _entry_key_strings(entry):
                    key_to_color[key_str] = color_value

            if not key_to_color:
                return [PCP_BASE_COLOR] * point_count, [False] * point_count
            colors = [key_to_color.get(key, PCP_OTHER_COLOR) for key in keys]
            playlist_active = [key in key_to_color for key in keys]
            return colors, playlist_active

        def _apply_gradient(field_candidates: list[str]) -> list[str]:
            field_name = None
            for candidate in field_candidates:
                if candidate in current_umap_metadata.columns:
                    field_name = candidate
                    break
                # try case-insensitive match
                for col in current_umap_metadata.columns:
                    if col.lower() == candidate.lower():
                        field_name = col
                        break
                if field_name:
                    break
            if field_name is None:
                return [PCP_OTHER_COLOR] * point_count

            series = pd.to_numeric(current_umap_metadata[field_name], errors="coerce")
            valid_mask = series.notna()
            if not valid_mask.any():
                return [PCP_OTHER_COLOR] * point_count

            values = series[valid_mask]
            mean = float(values.mean())
            std = float(values.std())
            if std <= 0:
                std = 1.0

            key_to_value = {
                key_to_str(k): float(v)
                for k, v in zip(current_umap_metadata["__key__"], series)
            }

            colors: list[str] = []
            for key in keys:
                val = key_to_value.get(key)
                if val is None or np.isnan(val):
                    colors.append(GRADIENT_MISSING)
                    continue
                z = (val - mean) / std
                z = max(-3.0, min(3.0, z))
                frac = (z + 3.0) / 6.0
                colors.append(interpolate_color(GRADIENT_LOW, GRADIENT_HIGH, frac))
            return colors

        if mode == "Latitude":
            colors = _apply_gradient(["lat", "latitude", "Lat", "Latitude"])
            _update_source(colors, base_alpha, [True] * point_count)
            return
        if mode == "Longitude":
            colors = _apply_gradient(["lon", "longitude", "Lon", "Longitude"])
            _update_source(colors, base_alpha, [True] * point_count)
            return
        if mode == "Altitude":
            colors = _apply_gradient(["alt", "altitude", "Altitude"])
            _update_source(colors, base_alpha, [True] * point_count)
            return
        if mode == "HDBSCAN":
            if not labels or current_hdbscan_labels is None:
                _update_source([PCP_BASE_COLOR] * point_count, base_alpha, [False] * point_count)
                return

            checklist_labels = hdbscan_checklist.labels or []
            if hdbscan_checklist.active:
                active_labels = {
                    checklist_labels[idx]
                    for idx in hdbscan_checklist.active
                    if 0 <= idx < len(checklist_labels)
                }
            else:
                active_labels = set()

            colors = [
                current_hdbscan_color_map.get(label, PCP_OTHER_COLOR)
                for label in labels
            ]
            alpha = [
                DEFAULT_PCP_ALPHA if label in active_labels else 0.0
                for label in labels
            ]
            playlist_active = [label in active_labels for label in labels]
            _update_source(colors, alpha, playlist_active)
            return
        colors, playlist_active = _apply_group_colors()
        _update_source(colors, base_alpha, playlist_active)

    def selected_record_indices() -> list[int]:
        """Return unique record indices from the current point selection."""

        selected_points = list(pcp_points_source.selected.indices)
        if not selected_points:
            return []
        record_indices = pcp_points_source.data.get("record_index", [])
        if not record_indices:
            return []
        unique_indices: set[int] = set()
        for idx in selected_points:
            if 0 <= idx < len(record_indices):
                try:
                    unique_indices.add(int(record_indices[idx]))
                except (TypeError, ValueError):
                    continue
        return sorted(unique_indices)

    def create_group_from_selection() -> None:
        """Create an active PCP group from the current box selection."""

        nonlocal active_pcp_groups, next_pcp_group_id
        if not pcp_source.data.get("key"):
            annotation_status_box.text = (
                "<em>Run Calculate to populate the plot before creating groups.</em>"
            )
            return

        record_indices = selected_record_indices()
        if not record_indices:
            annotation_status_box.text = (
                "<em>No points selected. Use box select to choose points first.</em>"
            )
            return

        keys = pcp_source.data.get("key", [])
        key_set = {
            keys[idx]
            for idx in record_indices
            if 0 <= idx < len(keys) and keys[idx]
        }
        if not key_set:
            annotation_status_box.text = (
                "<em>Selected points did not map to valid recordings.</em>"
            )
            return

        group_label = f"group_{next_pcp_group_id}"
        active_pcp_groups.append(
            {
                "id": next_pcp_group_id,
                "label": group_label,
                "key_set": key_set,
            }
        )
        next_pcp_group_id += 1
        refresh_active_pcp_groups_panel(select_new=True)
        annotation_status_box.text = (
            f"Created {html.escape(group_label)} with {len(key_set)} recordings."
        )
        update_color_assignments()
        apply_color_selection()

    def clear_active_pcp_groups(show_status: bool = True) -> None:
        """Clear all active PCP groups and reset group-related UI."""

        nonlocal active_pcp_groups, next_pcp_group_id
        active_pcp_groups = []
        next_pcp_group_id = 1
        refresh_active_pcp_groups_panel()
        if show_status:
            annotation_status_box.text = "<em>Cleared active PCP groups.</em>"
        update_color_assignments()
        apply_color_selection()

    def save_active_pcp_groups(group_name: str, description_text: str) -> None:
        """Persist the selected active PCP groups to disk."""

        if current_species_slug is None:
            annotation_status_box.text = "<em>Load a species before saving groups.</em>"
            return
        if not active_pcp_groups:
            annotation_status_box.text = "<em>No active groups to save.</em>"
            return

        active_indices = active_pcp_groups_checkbox.active
        if not active_indices:
            annotation_status_box.text = (
                "<em>Select at least one active PCP group to save.</em>"
            )
            return

        selected_key_set: set[str] = set()
        for idx in active_indices:
            if 0 <= idx < len(active_pcp_groups):
                key_set = active_pcp_groups[idx].get("key_set", set())
                if isinstance(key_set, set):
                    selected_key_set.update(key_set)
                elif isinstance(key_set, list):
                    selected_key_set.update(str(key) for key in key_set if key)

        if not selected_key_set:
            annotation_status_box.text = (
                "<em>The selected active groups contain no recordings.</em>"
            )
            return

        key_values = list(pcp_source.data.get("key", []))
        xcid_values = list(pcp_source.data.get("xcid", []))
        clip_values = list(pcp_source.data.get("clip_index", []))
        if not key_values or not xcid_values or not clip_values:
            annotation_status_box.text = (
                "<em>PCP data is missing xcid/clip_index; run Calculate first.</em>"
            )
            return

        rows: list[dict[str, Any]] = []
        skipped = 0
        seen_keys: set[str] = set()
        for idx, key in enumerate(key_values):
            if key not in selected_key_set or key in seen_keys:
                continue
            seen_keys.add(key)
            if idx >= len(xcid_values) or idx >= len(clip_values):
                continue
            xcid_value = xcid_values[idx]
            if xcid_value is None or pd.isna(xcid_value):
                skipped += 1
                continue
            xcid_str = str(xcid_value).strip()
            if not xcid_str:
                skipped += 1
                continue
            clip_idx = _coerce_clip_index(clip_values[idx])
            if clip_idx is None:
                skipped += 1
                continue
            rows.append({"xcid": xcid_str, "clip_index": clip_idx})

        if not rows:
            annotation_status_box.text = (
                "<em>No valid xcid/clip_index rows were found to save.</em>"
            )
            return

        base_stem = sanitize_group_name(Path(str(group_name)).stem)
        if not base_stem:
            annotation_status_box.text = "<em>Please provide a non-empty group name.</em>"
            return

        output_dir = groups_root / current_species_slug / "pcp_groups"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            annotation_status_box.text = (
                f"<span style='color:#b00;'>Failed to prepare group folder: "
                f"{html.escape(str(exc))}</span>"
            )
            return

        csv_path = output_dir / f"{base_stem}.csv"
        txt_path = output_dir / f"{base_stem}.txt"
        counter = 1
        while csv_path.exists() or txt_path.exists():
            csv_path = output_dir / f"{base_stem}_{counter}.csv"
            txt_path = output_dir / f"{base_stem}_{counter}.txt"
            counter += 1

        description_payload = format_pcp_group_description(
            description_text, last_umap_params, last_hdbscan_params
        )

        try:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            txt_path.write_text(description_payload, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            annotation_status_box.text = (
                f"<span style='color:#b00;'>Failed to save group: "
                f"{html.escape(str(exc))}</span>"
            )
            return

        try:
            relative_display = csv_path.relative_to(root_path)
        except ValueError:
            relative_display = csv_path

        message = (
            f"Saved {len(rows)} recordings to {html.escape(str(relative_display))} "
            "with description saved alongside the CSV."
        )
        if skipped:
            message += f" Skipped {skipped} row(s) without clip indices."
        annotation_status_box.text = message

    def on_save_request(
        attr: str, old: dict[str, list[str]], new: dict[str, list[str]]
    ) -> None:
        """Handle save requests issued from the JS prompt."""

        nonlocal last_save_nonce
        nonce_list = new.get("nonce", [])
        if not nonce_list:
            return
        nonce = str(nonce_list[0])
        if not nonce or nonce == last_save_nonce:
            return
        last_save_nonce = nonce
        name_list = new.get("name", [])
        desc_list = new.get("description", [])
        group_name = name_list[0] if name_list else ""
        description_text = desc_list[0] if desc_list else ""
        save_active_pcp_groups(group_name, description_text)

    create_group_button.on_click(create_group_from_selection)
    clear_groups_button.on_click(clear_active_pcp_groups)

    save_group_button.js_on_click(
        CustomJS(
            args=dict(save_source=save_request_source),
            code="""
            const name = window.prompt("Enter a name for this PCP group:");
            if (name === null) {
                return;
            }
            const trimmed = name.trim();
            const desc = window.prompt("Enter a description for this PCP group:");
            if (desc === null) {
                return;
            }
            save_source.data = {
                name: [trimmed],
                description: [desc],
                nonce: [String(Date.now())],
            };
            save_source.change.emit();
            """,
        )
    )
    save_request_source.on_change("data", on_save_request)

    def update_pcp_plot(projection: np.ndarray, metadata_df: pd.DataFrame) -> None:
        """Populate the parallel coordinates plot from UMAP output."""

        nonlocal current_point_base, current_point_dims

        if projection.size == 0:
            pcp_source.data = {
                "xs": [],
                "ys": [],
                "color": [],
                "key": [],
                "alpha": [],
                "hdbscan_label": [],
                "playlist_active": [],
                "xcid": [],
                "clip_index": [],
                "date": [],
                "audio_url": [],
                "spectrogram_url": [],
                "spectrogram_data_uri": [],
            }
            pcp_points_source.data = {
                "x": [],
                "y": [],
                "dim": [],
                "record_index": [],
                "color": [],
                "alpha": [],
                "playlist_active": [],
            }
            current_point_base = None
            current_point_dims = 0
            pcp_status_box.text = "<em>No points available for the plot.</em>"
            return

        dims = projection.shape[1]
        record_count = projection.shape[0]
        xs = [list(range(dims)) for _ in range(record_count)]

        sigma = projection.std(axis=0)
        safe_sigma = np.where(sigma > 0, sigma, 1.0)
        z_scores = (projection - projection.mean(axis=0)) / safe_sigma

        ys = z_scores.tolist()
        keys = [key_to_str(key) for key in metadata_df["__key__"]]
        colors = [PCP_BASE_COLOR] * record_count
        alphas = [DEFAULT_PCP_ALPHA] * record_count
        playlist_active = [True] * record_count

        if {"xcid", "clip_index"}.issubset(metadata_df.columns):
            xcid_values = (
                metadata_df["xcid"].fillna("").astype(str).tolist()
            )
            clip_index_series = pd.to_numeric(
                metadata_df["clip_index"], errors="coerce"
            )
            clip_index_values = [
                int(val) if not pd.isna(val) else None
                for val in clip_index_series
            ]
        else:
            xcid_values = []
            clip_index_values = []
            for key in metadata_df["__key__"]:
                if isinstance(key, tuple) and len(key) == 2:
                    xcid_values.append(str(key[0]))
                    clip_index_values.append(int(key[1]))
                else:
                    xcid_values.append("")
                    clip_index_values.append(None)

        if "date" in metadata_df.columns:
            date_values = metadata_df["date"].fillna("").astype(str).tolist()
        else:
            date_values = [""] * record_count

        if "audio_url" in metadata_df.columns:
            audio_values = metadata_df["audio_url"].fillna("").tolist()
        else:
            audio_values = [""] * record_count

        if "spectrogram_url" in metadata_df.columns:
            spectrogram_values = metadata_df["spectrogram_url"].fillna("").tolist()
        else:
            spectrogram_values = [""] * record_count

        if "spectrogram_data_uri" in metadata_df.columns:
            spectrogram_inline = metadata_df["spectrogram_data_uri"].fillna("").tolist()
        else:
            spectrogram_inline = [""] * record_count

        pcp_source.data = {
            "xs": xs,
            "ys": ys,
            "color": colors,
            "key": keys,
            "alpha": alphas,
            "hdbscan_label": [""] * record_count,
            "playlist_active": playlist_active,
            "xcid": xcid_values,
            "clip_index": clip_index_values,
            "date": date_values,
            "audio_url": audio_values,
            "spectrogram_url": spectrogram_values,
            "spectrogram_data_uri": spectrogram_inline,
        }
        current_point_dims = dims
        dim_indices = np.tile(np.arange(dims), record_count)
        record_indices = np.repeat(np.arange(record_count), dims)
        current_point_base = {
            "x": dim_indices.tolist(),
            "y": z_scores.reshape(-1).tolist(),
            "dim": dim_indices.tolist(),
            "record_index": record_indices.tolist(),
        }
        update_point_styles(colors, alphas, playlist_active)
        pcp_plot.xaxis.ticker = list(range(dims))
        pcp_plot.xaxis.major_label_overrides = {
            idx: f"Dim {idx + 1} (σ={sigma[idx]:.3g})" for idx in range(dims)
        }
        pcp_status_box.text = (
            f"Parallel coordinates plotted for {projection.shape[0]} points "
            f"across {dims} dimensions (z-score scaled per dimension)."
        )

    def rebuild_checkboxes(groups_by_category: dict[str, list[dict[str, Any]]]) -> None:
        umap_checkbox_groups.clear()
        umap_items_by_category.clear()
        sections = []

        for category_label, folder_name in CATEGORY_FOLDERS.items():
            items = groups_by_category.get(category_label, [])
            cat_header = Div(
                text=category_label,
                styles={
                    "font-weight": "600",
                    "margin-bottom": "2px",
                    "color": "#211b10",
                },
            )
            if items:
                labels = [f"{item['label']} ({item['entry_count']})" for item in items]
                checkbox = CheckboxGroup(labels=labels, active=[], width=150)
                checkbox.on_change("active", on_checkbox_change)
                umap_checkbox_groups[category_label] = checkbox
                umap_items_by_category[category_label] = items
                sections.append(
                    column(
                        cat_header,
                        checkbox,
                        spacing=2,
                        width=150,
                        sizing_mode="fixed",
                    )
                )
            else:
                sections.append(
                    column(
                        cat_header,
                        Div(
                            text="<em>No groups found.</em>",
                            render_as_text=False,
                            styles={"color": "#4a452c"},
                        ),
                        spacing=4,
                    )
                )

        if sections:
            checkbox_sections_holder.children = sections
        else:
            checkbox_sections_holder.children = [
                Div(
                    text="No group folders found for this species.",
                    styles={"color": "#4a452c"},
                )
            ]

    def rebuild_color_checkboxes(groups_by_category: dict[str, list[dict[str, Any]]]) -> None:
        color_checkbox_groups.clear()
        color_items_by_category.clear()
        sections: list[Any] = []

        def on_color_change(attr: str, old: list[int], new: list[int]) -> None:
            update_color_assignments()
            apply_color_selection()

        for category_label in CATEGORY_FOLDERS.keys():
            items = groups_by_category.get(category_label, [])
            cat_header = Div(
                text=category_label,
                styles={
                    "font-weight": "600",
                    "margin-bottom": "2px",
                    "color": "#211b10",
                },
            )
            if items:
                labels = [item["label"] for item in items]
                checkbox = CheckboxGroup(labels=labels, active=[], width=150)
                checkbox.on_change("active", on_color_change)
                color_checkbox_groups[category_label] = checkbox
                color_items_by_category[category_label] = items
                sections.append(
                    column(
                        cat_header,
                        checkbox,
                        spacing=2,
                        width=150,
                        sizing_mode="fixed",
                    )
                )
            else:
                sections.append(
                    column(
                        cat_header,
                        Div(
                            text="<em>No groups found.</em>",
                            render_as_text=False,
                            styles={"color": "#4a452c"},
                        ),
                        spacing=4,
                        width=150,
                        sizing_mode="fixed",
                    )
                )

        if sections:
            color_sections_holder.children = sections
        else:
            color_sections_holder.children = [
                Div(
                    text="No group folders found for this species.",
                    styles={"color": "#4a452c"},
                )
            ]
        update_color_assignments()

    def load_species_groups() -> None:
        nonlocal current_species_slug, current_embeddings_path, current_umap_projection
        nonlocal current_umap_metadata, current_point_base, current_point_dims
        species_slug = species_select.value
        groups_by_category = collect_groups_for_species(groups_root, species_slug)
        rebuild_checkboxes(groups_by_category)
        rebuild_color_checkboxes(groups_by_category)
        descriptions_box.text = format_descriptions(groups_by_category)
        if all_toggle.active:
            on_all_toggle(all_toggle.active)
        embeddings_counts_box.text = "<em>Press Calculate to load embeddings counts.</em>"
        reconciliation_box.text = "<em>Press Calculate for reconciliation stats.</em>"
        current_species_slug = species_slug
        current_embeddings_path = root_path / "embeddings" / species_slug / "embeddings.csv"
        metadata_path = root_path / "embeddings" / species_slug / "metadata.csv"
        metadata_total = 0
        metadata_unique = 0
        if metadata_path.exists():
            try:
                metadata_df = pd.read_csv(
                    metadata_path,
                    usecols=lambda c: c in {"xcid", "clip_index", "file_path"},
                )
                metadata_total = len(metadata_df)
                metadata_unique = len(keys_from_dataframe(metadata_df))
            except Exception:
                metadata_total = 0
                metadata_unique = 0
        all_toggle_count.text = f"(raw: {metadata_total}, unique: {metadata_unique})"
        current_umap_projection = None
        current_umap_metadata = None
        clear_active_pcp_groups(show_status=False)
        annotation_status_box.text = (
            "<em>Select points with box select to create a group.</em>"
        )
        pcp_source.data = {
            "xs": [],
            "ys": [],
            "color": [],
            "key": [],
            "alpha": [],
            "hdbscan_label": [],
            "playlist_active": [],
            "xcid": [],
            "clip_index": [],
            "date": [],
            "audio_url": [],
            "spectrogram_url": [],
            "spectrogram_data_uri": [],
        }
        pcp_points_source.data = {
            "x": [],
            "y": [],
            "dim": [],
            "record_index": [],
            "color": [],
            "alpha": [],
            "playlist_active": [],
        }
        current_point_base = None
        current_point_dims = 0
        pcp_status_box.text = "<em>Calculate to generate the parallel coordinates plot.</em>"
        playlist_panel.text = (
            "<em>Click a point on a dimension axis to list nearby recordings.</em>"
        )
        color_mode_select.value = "Groups"
        clear_hdbscan_results()

    load_button.on_click(load_species_groups)

    def format_embeddings_counts_html(counts_by_category: dict[str, list[tuple[str, int]]]) -> str:
        sections: list[str] = []
        for category_label, entries in counts_by_category.items():
            entry_items = "".join(
                f"<li><strong>{html.escape(label)}</strong>: {count}</li>"
                for label, count in entries
            )
            sections.append(
                f"<div class='desc-category'>"
                f"<div class='desc-title'>{html.escape(category_label)}</div>"
                f"<ul class='desc-list'>{entry_items or '<li><em>No selections.</em></li>'}</ul>"
                f"</div>"
            )
        if not sections:
            return "<em>No counts to show.</em>"
        return "<div class='desc-wrapper'>" + "".join(sections) + "</div>"

    def compute_embeddings_counts() -> None:
        if current_species_slug is None or current_embeddings_path is None:
            embeddings_counts_box.text = "<em>Please load a species first.</em>"
            return

        selected_entries = collect_selected_entries()
        if not selected_entries and not all_toggle.active:
            embeddings_counts_box.text = (
                "<em>No groups selected. Choose groups and press Calculate.</em>"
            )
            return

        if not current_embeddings_path.exists():
            embeddings_counts_box.text = (
                f"<em>Embeddings file not found at "
                f"{html.escape(str(current_embeddings_path))}.</em>"
            )
            return

        embeddings_total = 0
        embedding_unique = 0
        try:
            embeddings_df = pd.read_csv(
                current_embeddings_path,
                usecols=lambda c: c in {"xcid", "clip_index", "file_path"},
            )
            embeddings_total = len(embeddings_df)
            embedding_unique = len(keys_from_dataframe(embeddings_df))
        except Exception as exc:  # noqa: BLE001
            embeddings_counts_box.text = (
                f"<em>Failed to read embeddings: {html.escape(str(exc))}</em>"
            )
            return

        embedding_keys = keys_from_dataframe(embeddings_df)
        if not embedding_keys:
            embeddings_counts_box.text = (
                "<em>Embeddings file does not contain usable xcid/clip_index "
                "or file_path columns.</em>"
            )
            return

        metadata_path = root_path / "embeddings" / current_species_slug / "metadata.csv"
        metadata_keys: set[tuple[str, int]] = set()
        metadata_total = 0
        metadata_unique = 0
        if metadata_path.exists():
            try:
                metadata_df = pd.read_csv(
                    metadata_path,
                    usecols=lambda c: c in {"xcid", "clip_index", "file_path"},
                )
                metadata_total = len(metadata_df)
                metadata_keys = keys_from_dataframe(metadata_df)
                metadata_unique = len(metadata_keys)
            except Exception as exc:  # noqa: BLE001
                reconciliation_box.text = (
                    f"<em>Failed to read metadata: {html.escape(str(exc))}</em>"
                )
        else:
            reconciliation_box.text = "<em>Metadata file not found.</em>"

        counts_by_category: dict[str, list[tuple[str, int]]] = {
            k: [] for k in CATEGORY_FOLDERS.keys()
        }

        for category_label, entry in selected_entries:
            csv_path = Path(entry.get("csv_path") or "")
            if not csv_path.exists():
                counts_by_category[category_label].append((entry["label"], 0))
                continue

            try:
                group_df = pd.read_csv(csv_path)
            except Exception:  # noqa: BLE001
                counts_by_category[category_label].append((entry["label"], 0))
                continue

            group_keys = keys_from_dataframe(group_df)
            matched = embedding_keys & group_keys
            counts_by_category[category_label].append((entry["label"], len(matched)))

        counts_by_category["All embeddings"] = [
            ("All embeddings (raw)", embeddings_total),
            ("All embeddings (unique)", embedding_unique),
        ]
        counts_by_category["All recordings"] = [
            ("All recordings (raw)", metadata_total),
            ("All recordings (unique)", metadata_unique),
        ]
        embeddings_counts_box.text = format_embeddings_counts_html(counts_by_category)

        if metadata_total == 0 or not metadata_keys:
            # Leave reconciliation_box as-is if earlier error set a message.
            if (
                "Failed to read metadata" not in reconciliation_box.text
                and "not found" not in reconciliation_box.text
            ):
                reconciliation_box.text = "<em>No metadata available for reconciliation.</em>"
        else:
            missing_meta = len(embedding_keys - metadata_keys)
            missing_emb = len(metadata_keys - embedding_keys)
            reconciliation_box.text = (
                f"Embeddings without metadata: {missing_meta}<br>"
                f"Metadata without embeddings: {missing_emb}"
            )

    def compute_umap_projection() -> None:
        nonlocal current_umap_projection, current_umap_metadata, last_umap_params
        clear_hdbscan_results()
        if current_species_slug is None or current_embeddings_path is None:
            umap_status_box.text = "<em>Please load a species first.</em>"
            missing_metadata_warning.visible = False
            return

        selected_entries = collect_selected_entries()
        if not selected_entries and not all_toggle.active:
            umap_status_box.text = "<em>Select groups or enable All recordings.</em>"
            missing_metadata_warning.visible = False
            return

        if not current_embeddings_path.exists():
            umap_status_box.text = (
                f"<em>Embeddings file not found at "
                f"{html.escape(str(current_embeddings_path))}.</em>"
            )
            missing_metadata_warning.visible = False
            return

        metadata_path = root_path / "embeddings" / current_species_slug / "metadata.csv"
        if not metadata_path.exists():
            umap_status_box.text = "<em>Metadata file not found; cannot run UMAP.</em>"
            missing_metadata_warning.visible = False
            return

        try:
            embeddings_df = pd.read_csv(current_embeddings_path)
        except Exception as exc:  # noqa: BLE001
            umap_status_box.text = (
                f"<em>Failed to read embeddings: {html.escape(str(exc))}</em>"
            )
            missing_metadata_warning.visible = False
            return

        try:
            metadata_df = pd.read_csv(metadata_path)
        except Exception as exc:  # noqa: BLE001
            umap_status_box.text = (
                f"<em>Failed to read metadata: {html.escape(str(exc))}</em>"
            )
            missing_metadata_warning.visible = False
            return

        embeddings_with_keys = append_key_column(embeddings_df)
        metadata_with_keys = append_key_column(metadata_df)
        embedding_keys = set(embeddings_with_keys["__key__"])
        metadata_keys = set(metadata_with_keys["__key__"])

        if not embedding_keys:
            umap_status_box.text = "<em>No usable keys found in embeddings.csv.</em>"
            missing_metadata_warning.visible = False
            return

        if not metadata_keys:
            umap_status_box.text = "<em>No usable metadata entries found.</em>"
            missing_metadata_warning.visible = False
            return

        selected_keys: set[tuple[str, int]] = set()
        if all_toggle.active:
            selected_keys = set(embedding_keys)
        else:
            for _category_label, entry in selected_entries:
                csv_path = Path(entry.get("csv_path") or "")
                if not csv_path.exists():
                    continue
                try:
                    group_df = pd.read_csv(csv_path)
                except Exception:
                    continue
                group_keys_df = append_key_column(group_df)
                selected_keys.update(
                    key for key in group_keys_df["__key__"] if key is not None
                )

        if not selected_keys:
            umap_status_box.text = "<em>No recordings matched the selected groups.</em>"
            missing_metadata_warning.visible = False
            return

        selected_embedding_keys = selected_keys & embedding_keys
        missing_meta_count = len(selected_embedding_keys - metadata_keys)
        usable_keys = selected_embedding_keys & metadata_keys

        if not usable_keys:
            umap_status_box.text = (
                "<em>No overlap between selected embeddings and available metadata.</em>"
            )
            missing_metadata_warning.visible = missing_meta_count > 0
            if missing_meta_count:
                missing_metadata_warning.text = (
                    f"<strong>Warning:</strong> Omitted {missing_meta_count} "
                    "embeddings due to missing metadata."
                )
            return

        selected_embeddings = (
            embeddings_with_keys[embeddings_with_keys["__key__"].isin(usable_keys)]
            .drop_duplicates(subset="__key__")
            .reset_index(drop=True)
        )
        metadata_lookup = (
            metadata_with_keys.drop_duplicates(subset="__key__", keep="first")
            .set_index("__key__", drop=False)
        )
        try:
            selected_metadata = metadata_lookup.loc[
                selected_embeddings["__key__"]
            ].reset_index(drop=True)
        except KeyError:
            umap_status_box.text = "<em>Failed to align metadata with embeddings.</em>"
            missing_metadata_warning.visible = False
            return

        try:
            embedding_matrix, valid_indices = embedding_matrix_from_dataframe(
                selected_embeddings
            )
        except ValueError as exc:
            umap_status_box.text = f"<em>{html.escape(str(exc))}</em>"
            missing_metadata_warning.visible = False
            return

        if len(valid_indices) != len(selected_embeddings):
            selected_embeddings = (
                selected_embeddings.iloc[valid_indices].reset_index(drop=True)
            )
            selected_metadata = (
                selected_metadata.iloc[valid_indices].reset_index(drop=True)
            )

        selected_metadata = add_media_columns(
            selected_metadata,
            species_slug=current_species_slug,
            audio_base_url=AUDIO_BASE_URL,
            spectrogram_base_url=SPECTROGRAM_BASE_URL,
            spectrogram_dir=root_path / "spectrograms" / current_species_slug,
            spectrogram_image_format=SPECTROGRAM_IMAGE_FORMAT,
            inline_spectrograms=INLINE_SPECTROGRAMS,
        )

        if embedding_matrix.size == 0:
            umap_status_box.text = "<em>No embeddings available after filtering.</em>"
            missing_metadata_warning.visible = False
            return

        try:
            n_components = int(umap_components_spinner.value or DEFAULT_UMAP_COMPONENTS)
            n_neighbors = int(umap_neighbors_spinner.value or DEFAULT_UMAP_NEIGHBORS)
            min_dist = float(umap_min_dist_spinner.value or DEFAULT_UMAP_MIN_DIST)
            metric = umap_metric_input.value.strip() or DEFAULT_UMAP_METRIC
            seed = int(umap_seed_spinner.value or DEFAULT_UMAP_SEED)
        except (TypeError, ValueError):
            umap_status_box.text = "<em>Invalid UMAP parameter values.</em>"
            missing_metadata_warning.visible = False
            return

        last_umap_params = {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "seed": seed,
        }

        try:
            mapper = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=seed,
            )
            projection = mapper.fit_transform(embedding_matrix)
        except Exception as exc:  # noqa: BLE001
            umap_status_box.text = f"<em>UMAP failed: {html.escape(str(exc))}</em>"
            missing_metadata_warning.visible = False
            return

        missing_metadata_warning.visible = missing_meta_count > 0
        if missing_meta_count:
            missing_metadata_warning.text = (
                f"<strong>Warning:</strong> Omitted {missing_meta_count} "
                "embeddings due to missing metadata."
            )
        else:
            missing_metadata_warning.text = ""

        current_umap_projection = projection
        current_umap_metadata = selected_metadata
        update_pcp_plot(projection, selected_metadata)
        apply_color_selection()
        playlist_panel.text = (
            "<em>Click a point on a dimension axis to list nearby recordings.</em>"
        )
        umap_status_box.text = (
            f"UMAP completed for {projection.shape[0]} points "
            f"({projection.shape[1]} dimensions)."
        )
        hdbscan_status_box.text = (
            "<em>HDBSCAN ready: adjust parameters and press Compute.</em>"
        )

    def compute_hdbscan_clusters() -> None:
        """Compute HDBSCAN labels for the current UMAP projection."""

        nonlocal current_hdbscan_labels, current_hdbscan_color_map, last_hdbscan_params
        if current_umap_projection is None or current_umap_metadata is None:
            hdbscan_status_box.text = "<em>Run UMAP before computing HDBSCAN.</em>"
            return
        if current_umap_projection.size == 0:
            hdbscan_status_box.text = "<em>No UMAP points available for clustering.</em>"
            return

        try:
            min_cluster_size = int(
                hdbscan_min_cluster_size_spinner.value
                or DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE
            )
            min_samples = int(
                hdbscan_min_samples_spinner.value or DEFAULT_HDBSCAN_MIN_SAMPLES
            )
        except (TypeError, ValueError):
            hdbscan_status_box.text = "<em>Invalid HDBSCAN parameter values.</em>"
            return

        last_hdbscan_params = {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
        }

        hdbscan_compute_button.label = "Computing..."
        hdbscan_compute_button.disabled = True
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
            labels_array = clusterer.fit_predict(current_umap_projection)
        except Exception as exc:  # noqa: BLE001
            hdbscan_status_box.text = (
                f"<em>HDBSCAN failed: {html.escape(str(exc))}</em>"
            )
            return
        finally:
            hdbscan_compute_button.label = "Compute"
            hdbscan_compute_button.disabled = False

        if labels_array.size == 0:
            hdbscan_status_box.text = "<em>HDBSCAN returned no labels.</em>"
            return

        labels = [
            str(label) if label >= 0 else HDBSCAN_NOISE_LABEL
            for label in labels_array
        ]
        current_hdbscan_labels = labels
        unique_labels = sorted(
            set(labels), key=lambda lbl: (lbl != HDBSCAN_NOISE_LABEL, lbl)
        )
        color_map: dict[str, str] = {}
        color_idx = 0
        for label in unique_labels:
            if label == HDBSCAN_NOISE_LABEL:
                color_map[label] = HDBSCAN_NOISE_COLOR
            else:
                color_map[label] = HDBSCAN_PALETTE[color_idx % len(HDBSCAN_PALETTE)]
                color_idx += 1
        current_hdbscan_color_map = color_map

        hdbscan_checklist.labels = unique_labels
        hdbscan_checklist.active = list(range(len(unique_labels)))
        hdbscan_checklist.visible = True
        hdbscan_clusters_label.visible = True

        current_data = pcp_source.data
        point_count = len(current_data.get("xs", []))
        if point_count != len(labels):
            hdbscan_status_box.text = (
                "<em>HDBSCAN labels did not match the plotted points.</em>"
            )
            return

        pcp_source.data = {
            "xs": current_data.get("xs", []),
            "ys": current_data.get("ys", []),
            "color": current_data.get("color", []),
            "key": current_data.get("key", []),
            "alpha": [DEFAULT_PCP_ALPHA] * point_count,
            "hdbscan_label": labels,
            "playlist_active": current_data.get(
                "playlist_active", [True] * point_count
            ),
            "xcid": current_data.get("xcid", []),
            "clip_index": current_data.get("clip_index", []),
            "date": current_data.get("date", []),
            "audio_url": current_data.get("audio_url", []),
            "spectrogram_url": current_data.get("spectrogram_url", []),
            "spectrogram_data_uri": current_data.get("spectrogram_data_uri", []),
        }
        apply_color_selection()
        update_color_assignments()

        cluster_count = len(
            [lbl for lbl in unique_labels if lbl != HDBSCAN_NOISE_LABEL]
        )
        noise_count = labels.count(HDBSCAN_NOISE_LABEL)
        total = len(labels)
        noise_pct = (100.0 * noise_count / total) if total else 0.0
        hdbscan_status_box.text = (
            f"<strong>HDBSCAN:</strong> {cluster_count} clusters, {noise_count} noise "
            f"({noise_pct:.1f}% of {total} points)"
        )

    def on_calculate_click() -> None:
        calculate_button.label = "Calculating..."
        calculate_button.disabled = True
        umap_status_box.text = "<em>Calculating UMAP...</em>"
        missing_metadata_warning.visible = False
        try:
            compute_embeddings_counts()
            compute_umap_projection()
        finally:
            calculate_button.label = "Calculate"
            calculate_button.disabled = False

    calculate_button.on_click(on_calculate_click)
    hdbscan_compute_button.on_click(compute_hdbscan_clusters)

    playlist_callback = CustomJS(
        args=dict(
            points=pcp_points_source,
            records=pcp_source,
            pane=playlist_panel,
            color_mode=color_mode_select,
        ),
        code="""
        const p = points.data;
        const r = records.data;
        const inds = points.selected.indices;
        if (!inds || inds.length === 0) {
            return;
        }

        const idx = inds[0];
        const dim = p['dim'][idx];
        const recordIdx = p['record_index'][idx];
        const mode = color_mode.value || 'Groups';
        const useFilter = (mode === 'Groups' || mode === 'HDBSCAN');
        const playlistActive = r['playlist_active'] || [];

        if (useFilter && playlistActive.length && !playlistActive[recordIdx]) {
            pane.text = `<b>Dim ${dim + 1}</b><br><em>Selected point is not in the active ${mode.toLowerCase()} groups.</em>`;
            return;
        }

        const ys = r['ys'] || [];
        const keys = r['key'] || [];
        const colors = r['color'] || [];
        const dates = r['date'] || [];
        const xcids = r['xcid'] || [];
        const audioUrls = r['audio_url'] || [];
        const spectroUrls = r['spectrogram_url'] || [];
        const spectroInline = r['spectrogram_data_uri'] || [];
        const hdbLabels = r['hdbscan_label'] || [];

        const y0 = Number(p['y'][idx]);
        const items = [];
        for (let j = 0; j < ys.length; j++) {
            if (useFilter && playlistActive.length && !playlistActive[j]) {
                continue;
            }
            const row = ys[j];
            if (!row || row.length <= dim) {
                continue;
            }
            const y = Number(row[dim]);
            if (!isFinite(y)) {
                continue;
            }
            const dist = Math.abs(y - y0);
            items.push({ index: j, dist: dist, value: y });
        }

        items.sort((a, b) => a.dist - b.dist);

        if (!items.length) {
            pane.text = `<b>Dim ${dim + 1}</b><br><em>No recordings matched the active groups.</em>`;
            return;
        }

        function escapeHtml(text) {
            if (text === null || text === undefined) {
                return '';
            }
            return String(text)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }

        function formatNumber(value) {
            return Number.isFinite(value) ? value.toFixed(3) : 'n/a';
        }

        function normalizeHex(color) {
            if (typeof color !== 'string') {
                return null;
            }
            const trimmed = color.trim();
            if (/^#([0-9a-f]{6}|[0-9a-f]{3})$/i.test(trimmed)) {
                return trimmed;
            }
            return null;
        }

        function hexToRgb(color) {
            const hex = normalizeHex(color);
            if (!hex) {
                return null;
            }
            let clean = hex.slice(1);
            if (clean.length === 3) {
                clean = clean.split('').map((ch) => ch + ch).join('');
            }
            const rVal = parseInt(clean.slice(0, 2), 16);
            const gVal = parseInt(clean.slice(2, 4), 16);
            const bVal = parseInt(clean.slice(4, 6), 16);
            if ([rVal, gVal, bVal].some((val) => Number.isNaN(val))) {
                return null;
            }
            return { r: rVal, g: gVal, b: bVal };
        }

        function tintedBackground(color) {
            const rgb = hexToRgb(color) || { r: 176, g: 176, b: 176 };
            const boost = 40;
            const rVal = Math.min(255, rgb.r + boost);
            const gVal = Math.min(255, rgb.g + boost);
            const bVal = Math.min(255, rgb.b + boost);
            return `rgba(${rVal}, ${gVal}, ${bVal}, 0.22)`;
        }

        const dimLabel = dim + 1;
        const headerValue = formatNumber(y0);
        let html = `<div class="playlist-header"><b>Dim ${dimLabel}</b> (value ${headerValue}) - ${items.length} recordings</div>`;
        html += '<div class="playlist-scroll">';

        for (const item of items) {
            const j = item.index;
            const color = colors[j] || '#b0b0b0';
            const background = tintedBackground(color);
            const date = escapeHtml(dates[j] || '');
            const xcid = escapeHtml(xcids[j] || '');
            const keyFallback = escapeHtml(keys[j] || '');
            const displayId = xcid || keyFallback || `record ${j + 1}`;
            const audioUrl = audioUrls[j] || '';
            const spectroSrc = spectroInline[j] || spectroUrls[j] || '';
            const valueText = formatNumber(item.value);
            const distText = formatNumber(item.dist);
            const labelRaw = hdbLabels[j] || '';
            let labelText = '';
            if (mode === 'HDBSCAN') {
                labelText = labelRaw === 'Noise' ? 'Noise' : `Cluster ${labelRaw || 'Unknown'}`;
            }
            const badgeHtml = labelText
                ? `<div class="playlist-badge">${escapeHtml(labelText)}</div>`
                : '';
            const spectroHtml = spectroSrc
                ? `<img src="${spectroSrc}" alt="Spectrogram ${displayId}" class="playlist-spectrogram">`
                : `<div class="playlist-missing">No spectrogram</div>`;

            html += `<div class="playlist-row" style="--group-color:${color}; background:${background};">
                    <div class="playlist-row-main">
                        <button class="playlist-play" onclick="(function(u){
                            if (!u) { return; }
                            if (window._pcp_audio) { window._pcp_audio.pause(); }
                            window._pcp_audio = new Audio(u);
                            window._pcp_audio.play();
                        })('${audioUrl}')">Play</button>
                        <div class="playlist-meta">
                            <div class="playlist-id"><b>${displayId}</b></div>
                            <div class="playlist-date">${date || 'Unknown date'}</div>
                            <div class="playlist-value">Dim ${dimLabel}: ${valueText} (delta ${distText})</div>
                            ${badgeHtml}
                        </div>
                        ${spectroHtml}
                    </div>
                </div>`;
        }

        html += '</div>';
        pane.text = html;
        """,
    )

    pcp_points_source.selected.js_on_change("indices", playlist_callback)

    checkbox_panel = column(
        umap_title,
        row(all_toggle_label, all_toggle_count, all_toggle, sizing_mode="fixed"),
        checkbox_sections_holder,
        width=520,
        sizing_mode="fixed",
        css_classes=["control-card"],
        styles={
            "background-color": CARD_BACKGROUND_COLOR,
            "border-radius": "12px",
            "padding": "12px 14px",
            "box-shadow": "0 3px 10px rgba(0, 0, 0, 0.08)",
        },
    )

    color_panel = column(
        color_title,
        color_mode_select,
        color_sections_holder,
        active_pcp_groups_section,
        hdbscan_clusters_label,
        hdbscan_checklist,
        color_assignments_box,
        width=520,
        sizing_mode="fixed",
        css_classes=["control-card"],
        styles={
            "background-color": CARD_BACKGROUND_COLOR,
            "border-radius": "12px",
            "padding": "12px 14px",
            "box-shadow": "0 3px 10px rgba(0, 0, 0, 0.08)",
        },
    )

    description_panel = column(
        Div(
            text="Group descriptions",
            styles={
                "font-size": "15px",
                "font-weight": "600",
                "margin-bottom": "6px",
                "color": "#2d2616",
            },
        ),
        descriptions_box,
        width=520,
        sizing_mode="fixed",
        css_classes=["control-card"],
        styles={
            "background-color": CARD_BACKGROUND_COLOR,
            "border-radius": "12px",
            "padding": "12px 14px",
            "box-shadow": "0 3px 10px rgba(0, 0, 0, 0.08)",
        },
    )

    annotation_panel = column(
        Div(
            text="Annotation",
            styles={
                "font-size": "15px",
                "font-weight": "600",
                "margin-bottom": "6px",
                "color": "#2d2616",
            },
        ),
        create_group_button,
        clear_groups_button,
        save_group_button,
        annotation_status_box,
        width=260,
        sizing_mode="fixed",
        css_classes=["control-card"],
        styles={
            "background-color": CARD_BACKGROUND_COLOR,
            "border-radius": "12px",
            "padding": "12px 14px",
            "box-shadow": "0 3px 10px rgba(0, 0, 0, 0.08)",
        },
    )

    counts_panel = column(
        Div(
            text="Embeddings per group",
            styles={
                "font-size": "15px",
                "font-weight": "600",
                "margin-bottom": "6px",
                "color": "#2d2616",
            },
        ),
        embeddings_counts_box,
        Div(
            text="Reconciliation (embeddings vs metadata)",
            styles={
                "font-size": "15px",
                "font-weight": "600",
                "margin-top": "10px",
                "margin-bottom": "6px",
                "color": "#2d2616",
            },
        ),
        reconciliation_box,
        width=520,
        sizing_mode="fixed",
        css_classes=["control-card"],
        styles={
            "background-color": CARD_BACKGROUND_COLOR,
            "border-radius": "12px",
            "padding": "12px 14px",
            "box-shadow": "0 3px 10px rgba(0, 0, 0, 0.08)",
        },
    )

    style_injector = Div(
        text=f"""
<style>
html, body {{
    margin: 0;
    padding: 0;
    background-color: {APP_BACKGROUND_COLOR};
    min-height: 120vh;
    font-family: "Helvetica Neue", Arial, sans-serif;
}}
.bk-root {{
    background-color: {APP_BACKGROUND_COLOR};
    min-height: 120vh;
    padding: 18px 22px 140px 22px;
    box-sizing: border-box;
}}
.control-card {{
    background-color: {CARD_BACKGROUND_COLOR};
}}
.desc-wrapper {{
    display: flex;
    flex-direction: column;
    gap: 10px;
    color: #2f2818;
}}
.desc-category {{
    border-radius: 8px;
    padding: 6px 8px;
}}
.desc-title {{
    font-weight: 600;
    margin-bottom: 4px;
    color: #241d10;
}}
.desc-list {{
    margin: 0 0 0 16px;
    padding: 0;
}}
.desc-list li {{
    margin-bottom: 4px;
}}
.playlist-header {{
    font-weight: 600;
    margin-bottom: 6px;
    color: #2d2616;
}}
.playlist-scroll {{
    display: flex;
    flex-direction: column;
    gap: 8px;
}}
.playlist-row {{
    border-left: 6px solid var(--group-color, #b0b0b0);
    border-radius: 6px;
    padding: 6px;
    background: #fff;
}}
.playlist-row:hover {{
    outline: 2px solid #f2b100;
    outline-offset: 1px;
}}
.playlist-row-main {{
    display: flex;
    gap: 10px;
    align-items: flex-start;
}}
.playlist-play {{
    font-size: 11px;
    height: 24px;
}}
.playlist-meta {{
    min-width: 140px;
    font-size: 11px;
    color: #2d2616;
}}
.playlist-date {{
    color: #5a543a;
    font-size: 10px;
}}
.playlist-value {{
    color: #2d2616;
    font-size: 10px;
    margin-top: 2px;
}}
.playlist-badge {{
    display: inline-block;
    margin-top: 4px;
    padding: 2px 6px;
    border-radius: 10px;
    background: var(--group-color, #b0b0b0);
    font-size: 10px;
    font-weight: 600;
    color: #1e1e1e;
}}
.playlist-spectrogram {{
    max-width: 320px;
    border: 1px solid #cfc7b8;
    border-radius: 4px;
}}
.playlist-missing {{
    color: #777777;
    font-size: 10px;
    min-width: 120px;
}}
</style>
""",
        render_as_text=False,
    )

    top_row_left = row(
        card,
        checkbox_panel,
        color_panel,
        sizing_mode="scale_width",
        styles={"gap": "12px", "flex-wrap": "wrap"},
    )

    second_row = row(
        umap_parameters_panel,
        hdbscan_panel,
        annotation_panel,
        counts_panel,
        description_panel,
        sizing_mode="scale_width",
        styles={"gap": "12px", "flex-wrap": "wrap"},
    )

    left_stack = column(
        top_row_left,
        second_row,
        sizing_mode="scale_width",
        styles={"gap": "12px"},
    )

    main_row = row(
        left_stack,
        playlist_column,
        sizing_mode="scale_width",
        styles={"gap": "12px", "align-items": "flex-start"},
    )

    layout = column(
        style_injector,
        description,
        location_hint,
        main_row,
        pcp_panel,
        sizing_mode="scale_width",
        width=1200,
        styles={
            "background-color": APP_BACKGROUND_COLOR,
            "padding": "4px",
            "min-height": "100vh",
        },
    )
    curdoc().add_root(layout)
    curdoc().title = "Parallel Coordinates Prototype"


parser = argparse.ArgumentParser(description="Parallel coordinate prototype app.")
parser.add_argument(
    "--config",
    type=Path,
    default=DEFAULT_CONFIG_PATH,
    help="Path to config.yaml (defaults to xc_configs/config.yaml).",
)
args = parser.parse_args()

config = load_config(args.config)
root_path = Path(config["paths"]["root"]).expanduser()

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

if AUDIO_BASE_URL is None and AUDIO_AUTO_SERVE_REQUESTED:
    generated_audio_url = start_static_file_server(
        label="Audio (PCP)",
        directory=root_path / "clips",
        host=AUDIO_HOST,
        port=AUDIO_PORT,
        log_requests=bool(_audio_cfg.get("log_requests", False)),
    )
    if generated_audio_url:
        audio_base, _ = generated_audio_url
        AUDIO_BASE_URL = audio_base.rstrip("/")

_spectro_cfg = config.get("spectrograms", {}) or {}
_spectro_base = _spectro_cfg.get("base_url")
SPECTROGRAM_BASE_URL = (
    str(_spectro_base).rstrip("/")
    if isinstance(_spectro_base, str) and _spectro_base
    else None
)
SPECTROGRAM_HOST = str(_spectro_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
SPECTROGRAM_PORT = int(_spectro_cfg.get("port", 8766))
SPECTROGRAM_AUTO_SERVE_REQUESTED = bool(_spectro_cfg.get("auto_serve", True))
_spectro_inline_flag = _spectro_cfg.get("inline")
SPECTROGRAM_IMAGE_FORMAT = str(
    _spectro_cfg.get("image_format", DEFAULT_SPECTROGRAM_IMAGE_FORMAT)
).lower()

if _spectro_inline_flag is True:
    SPECTROGRAM_AUTO_SERVE_REQUESTED = False

if SPECTROGRAM_BASE_URL is None and SPECTROGRAM_AUTO_SERVE_REQUESTED:
    generated_spectro_url = start_static_file_server(
        label="Spectrograms (PCP)",
        directory=root_path / "spectrograms",
        host=SPECTROGRAM_HOST,
        port=SPECTROGRAM_PORT,
        log_requests=bool(_spectro_cfg.get("log_requests", False)),
    )
    if generated_spectro_url:
        spectro_base, _ = generated_spectro_url
        SPECTROGRAM_BASE_URL = spectro_base.rstrip("/")
    else:
        _spectro_inline_flag = True

if _spectro_inline_flag is None:
    INLINE_SPECTROGRAMS = SPECTROGRAM_BASE_URL is None
else:
    INLINE_SPECTROGRAMS = bool(_spectro_inline_flag)

groups_root = root_path / "xc_groups"
species_options = list_species_slugs(groups_root)
create_layout(species_options=species_options, groups_root=groups_root)
