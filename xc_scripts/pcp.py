"""
Prototype Bokeh application for selecting species before building a
parallel-coordinate plot.

Run with:
    bokeh serve --show xc_scripts/pcp.py --args --config xc_configs/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import html
import numpy as np
import pandas as pd
import umap
import yaml
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CheckboxGroup,
    ColumnDataSource,
    Div,
    Select,
    Spinner,
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
}
COLOR_PALETTE = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
]
DEFAULT_UMAP_COMPONENTS = 6
DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.0
DEFAULT_UMAP_METRIC = "euclidean"
PCP_BASE_COLOR = "#4477AA"
PCP_OTHER_COLOR = "#b0b0b0"
GRADIENT_LOW = "#2b83ba"
GRADIENT_HIGH = "#d7191c"
GRADIENT_MISSING = "#b0b0b0"


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


def list_species_slugs(groups_root: Path) -> list[str]:
    """Return sorted species slugs discovered in the xc_groups directory."""

    if not groups_root.exists():
        print(f"No xc_groups directory found at {groups_root}.")
        return []

    return sorted(entry.name for entry in groups_root.iterdir() if entry.is_dir())


def collect_group_entries(category_root: Path) -> list[dict[str, Any]]:
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
        entries.append(
            {
                "label": f"group_{idx}",
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
    grouped: dict[str, list[dict[str, str]]] = {}
    for category_label, folder_name in CATEGORY_FOLDERS.items():
        grouped[category_label] = collect_group_entries(species_root / folder_name)
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
        width=460,
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
        high=30,
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
        options=["Groups", "Latitude", "Longitude", "Altitude"],
        width=160,
    )
    pcp_status_box = Div(
        text="<em>Calculate to generate the parallel coordinates plot.</em>",
        render_as_text=False,
        styles={"color": "#2d2616", "margin-top": "6px"},
        width=1120,
    )
    pcp_source = ColumnDataSource(
        data={"xs": [], "ys": [], "color": [], "key": []}
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
        line_alpha=0.55,
        line_width=1.2,
        source=pcp_source,
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
        if color_mode_select.value != "Groups":
            color_assignments_box.text = "<em>Using gradient color mode.</em>"
            return

        active_labels: list[tuple[str, str]] = []
        for category_label, cb in color_checkbox_groups.items():
            for idx in cb.active:
                if 0 <= idx < len(cb.labels):
                    active_labels.append((category_label, cb.labels[idx]))

        html_entries: list[str] = []
        for idx, (cat, label) in enumerate(active_labels):
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
        return selected_entries

    def apply_color_selection() -> None:
        """Recolor PCP lines based on current color-by selections."""

        if current_umap_projection is None or current_umap_metadata is None:
            return
        if not pcp_source.data.get("xs"):
            return

        mode = color_mode_select.value
        keys = pcp_source.data.get("key", [])

        def _apply_group_colors() -> list[str]:
            color_entries = collect_color_entries()
            if not color_entries:
                return [PCP_BASE_COLOR] * len(keys)

            key_to_color: dict[str, str] = {}
            for color_idx, (_category_label, entry) in enumerate(color_entries):
                if color_idx >= len(COLOR_PALETTE):
                    break
                csv_path = Path(entry.get("csv_path") or "")
                if not csv_path.exists():
                    continue
                try:
                    group_df = pd.read_csv(csv_path)
                except Exception:
                    continue
                group_keys_df = append_key_column(group_df)
                color_value = COLOR_PALETTE[color_idx]
                for key in group_keys_df["__key__"]:
                    key_str = key_to_str(key)
                    if key_str:
                        key_to_color[key_str] = color_value

            if not key_to_color:
                return [PCP_BASE_COLOR] * len(keys)
            return [key_to_color.get(key, PCP_OTHER_COLOR) for key in keys]

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
                return [PCP_OTHER_COLOR] * len(keys)

            series = pd.to_numeric(current_umap_metadata[field_name], errors="coerce")
            valid_mask = series.notna()
            if not valid_mask.any():
                return [PCP_OTHER_COLOR] * len(keys)

            values = series[valid_mask]
            mean = float(values.mean())
            std = float(values.std())
            if std <= 0:
                std = 1.0

            key_to_value = {
                key_to_str(k): float(v) for k, v in zip(current_umap_metadata["__key__"], series)
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
        elif mode == "Longitude":
            colors = _apply_gradient(["lon", "longitude", "Lon", "Longitude"])
        elif mode == "Altitude":
            colors = _apply_gradient(["alt", "altitude", "Altitude"])
        else:
            colors = _apply_group_colors()

        pcp_source.data = {
            "xs": pcp_source.data.get("xs", []),
            "ys": pcp_source.data.get("ys", []),
            "color": colors,
            "key": keys,
        }

    def update_pcp_plot(projection: np.ndarray, metadata_df: pd.DataFrame) -> None:
        """Populate the parallel coordinates plot from UMAP output."""

        if projection.size == 0:
            pcp_source.data = {"xs": [], "ys": [], "color": [], "key": []}
            pcp_status_box.text = "<em>No points available for the plot.</em>"
            return

        dims = projection.shape[1]
        xs = [list(range(dims)) for _ in range(projection.shape[0])]

        sigma = projection.std(axis=0)
        safe_sigma = np.where(sigma > 0, sigma, 1.0)
        z_scores = (projection - projection.mean(axis=0)) / safe_sigma

        ys = z_scores.tolist()
        keys = [key_to_str(key) for key in metadata_df["__key__"]]
        colors = [PCP_BASE_COLOR] * projection.shape[0]

        pcp_source.data = {"xs": xs, "ys": ys, "color": colors, "key": keys}
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
        nonlocal current_species_slug, current_embeddings_path, current_umap_projection, current_umap_metadata
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
        pcp_source.data = {"xs": [], "ys": [], "color": [], "key": []}
        pcp_status_box.text = "<em>Calculate to generate the parallel coordinates plot.</em>"
        color_mode_select.value = "Groups"

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
        nonlocal current_umap_projection, current_umap_metadata
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

        if embedding_matrix.size == 0:
            umap_status_box.text = "<em>No embeddings available after filtering.</em>"
            missing_metadata_warning.visible = False
            return

        try:
            n_components = int(umap_components_spinner.value or DEFAULT_UMAP_COMPONENTS)
            n_neighbors = int(umap_neighbors_spinner.value or DEFAULT_UMAP_NEIGHBORS)
            min_dist = float(umap_min_dist_spinner.value or DEFAULT_UMAP_MIN_DIST)
            metric = umap_metric_input.value.strip() or DEFAULT_UMAP_METRIC
        except (TypeError, ValueError):
            umap_status_box.text = "<em>Invalid UMAP parameter values.</em>"
            missing_metadata_warning.visible = False
            return

        try:
            mapper = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
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
        umap_status_box.text = (
            f"UMAP completed for {projection.shape[0]} points "
            f"({projection.shape[1]} dimensions)."
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
</style>
""",
        render_as_text=False,
    )

    top_row = row(
        card,
        checkbox_panel,
        color_panel,
        sizing_mode="scale_width",
        styles={"gap": "12px", "flex-wrap": "wrap"},
    )

    second_row = row(
        umap_parameters_panel,
        counts_panel,
        description_panel,
        sizing_mode="scale_width",
        styles={"gap": "12px", "flex-wrap": "wrap"},
    )

    layout = column(
        style_injector,
        description,
        location_hint,
        top_row,
        second_row,
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
groups_root = root_path / "xc_groups"
species_options = list_species_slugs(groups_root)
create_layout(species_options=species_options, groups_root=groups_root)
