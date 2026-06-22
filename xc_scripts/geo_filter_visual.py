"""
Visualize a geographic bounding-box filter on the same basemap used by
kde_map_animate.py.

This renders the configured lat/lon bounding box as a highlighted "filter"
rectangle over the Esri.WorldPhysical basemap, titled "Western Palearctic".

Run:
    python xc_scripts/geo_filter_visual.py
    python xc_scripts/geo_filter_visual.py --no-show
    python xc_scripts/geo_filter_visual.py --savepng
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bokeh.io import output_file, save, show
from bokeh.models import Range1d
from bokeh.plotting import figure
from pyproj import Transformer

# Reuse the exact basemap (tiles, water background, land fallback) from the KDE
# map script so this visualization sits on the same geography.
from kde_map_animate import style_map_figure

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TITLE_TEXT = "Western Palearctic"

# Geographic filter bounding box (degrees).
FILTER_LAT_MIN = 30.0
FILTER_LAT_MAX = 82.0
FILTER_LON_MIN = -35.0
FILTER_LON_MAX = 45.0

# Margin of surrounding geography shown around the filter box so the selection
# reads as a filter rather than a plain crop. Set to 0.0 to plot the bounding
# box exactly.
VIEW_PADDING_DEG = 0
MAX_ABS_LAT = 84.0  # Web Mercator becomes singular near the poles.

# Filter-box overlay styling.
DRAW_FILTER_BOX = False
FILTER_BOX_FILL_COLOR = "#73D055FF"
FILTER_BOX_FILL_ALPHA = 0.18
FILTER_BOX_LINE_COLOR = "#73D055FF"
FILTER_BOX_LINE_WIDTH = 4
FILTER_BOX_LINE_ALPHA = 0.95

# Rendering scale. The basemap is a tiled raster, so its sharpness in the PNG
# is governed by the plot's pixel size: a larger plot makes Bokeh fetch
# higher-zoom (sharper) tiles. kde_map_animate.py renders at 2400px, hence its
# crisp maps. Fonts and line widths scale with RENDER_SCALE, so the layout and
# proportions stay identical while the map gets sharper. Bump this for more
# resolution. At BASE_PLOT_WIDTH = 900, RENDER_SCALE = 3 -> ~2700px wide.
RENDER_SCALE = 3
BASE_PLOT_WIDTH = 900
PLOT_WIDTH = BASE_PLOT_WIDTH * RENDER_SCALE

# Fonts (point sizes at RENDER_SCALE = 1) mirror the sample-size plots in
# kde_map_animate.py:
# - bold title at MONTHLY_SAMPLE_TITLE_FONT_SIZE
# - axis labels at MONTHLY_SAMPLE_AXIS_LABEL_FONT_SIZE
# - major tick labels at MONTHLY_SAMPLE_AXIS_MAJOR_LABEL_FONT_SIZE
TITLE_FONT_PT = 60
TITLE_FONT_STYLE = "bold"
AXIS_LABEL_FONT_PT = 40
AXIS_MAJOR_LABEL_FONT_PT = 30
AXIS_LABEL_FONT_STYLE = "normal"

OUTPUT_HTML = Path("geo_filter_western_palearctic.html")
OUTPUT_PNG = Path("geo_filter_western_palearctic.png")


def scaled_pt(base_pt: float) -> str:
    """Scale a base point size by RENDER_SCALE for a Bokeh font-size string."""
    return f"{base_pt * RENDER_SCALE:g}pt"


def lonlat_to_mercator(
    lon: list[float], lat: list[float]
) -> tuple[list[float], list[float]]:
    """Project lon/lat degrees to Web Mercator (EPSG:3857)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform(lon, lat)
    return list(xs), list(ys)


def build_view_ranges() -> tuple[Range1d, Range1d]:
    """Return Web Mercator x/y ranges for the padded filter box."""
    lat_min = max(-MAX_ABS_LAT, FILTER_LAT_MIN - VIEW_PADDING_DEG)
    lat_max = min(MAX_ABS_LAT, FILTER_LAT_MAX + VIEW_PADDING_DEG)
    lon_min = FILTER_LON_MIN - VIEW_PADDING_DEG
    lon_max = FILTER_LON_MAX + VIEW_PADDING_DEG
    xs, ys = lonlat_to_mercator([lon_min, lon_max], [lat_min, lat_max])
    return Range1d(min(xs), max(xs)), Range1d(min(ys), max(ys))


def compute_plot_height(x_range: Range1d, y_range: Range1d, width: int) -> int:
    """Size the figure to the view aspect so the map is not distorted."""
    x_span = float(x_range.end - x_range.start)
    y_span = float(y_range.end - y_range.start)
    if x_span <= 0:
        return width
    return max(1, int(round(width * y_span / x_span)))


def build_plot() -> figure:
    """Build the geography-filter figure."""
    x_range, y_range = build_view_ranges()
    height = compute_plot_height(x_range, y_range, PLOT_WIDTH)

    plot = figure(
        title=TITLE_TEXT,
        x_axis_type="mercator",
        y_axis_type="mercator",
        match_aspect=True,
        width=PLOT_WIDTH,
        height=height,
        x_range=x_range,
        y_range=y_range,
        tools="pan,wheel_zoom,reset,save",
    )

    # Same basemap as kde_map_animate.py (Esri.WorldPhysical tiles + water bg).
    style_map_figure(plot)

    if DRAW_FILTER_BOX:
        (box_x0, box_x1), (box_y0, box_y1) = lonlat_to_mercator(
            [FILTER_LON_MIN, FILTER_LON_MAX], [FILTER_LAT_MIN, FILTER_LAT_MAX]
        )
        plot.quad(
            left=box_x0,
            right=box_x1,
            bottom=box_y0,
            top=box_y1,
            fill_color=FILTER_BOX_FILL_COLOR,
            fill_alpha=FILTER_BOX_FILL_ALPHA,
            line_color=FILTER_BOX_LINE_COLOR,
            line_width=FILTER_BOX_LINE_WIDTH * RENDER_SCALE,
            line_alpha=FILTER_BOX_LINE_ALPHA,
        )

    # Title and axis fonts mirror the sample-size plots in kde_map_animate.py.
    # Applied after style_map_figure so they override its map-scale font sizes.
    plot.title.text_font_size = scaled_pt(TITLE_FONT_PT)
    plot.title.text_font_style = TITLE_FONT_STYLE

    plot.xaxis.axis_label = "Longitude (°)"
    plot.yaxis.axis_label = "Latitude (°)"
    plot.xaxis.axis_label_text_font_size = scaled_pt(AXIS_LABEL_FONT_PT)
    plot.yaxis.axis_label_text_font_size = scaled_pt(AXIS_LABEL_FONT_PT)
    plot.xaxis.axis_label_text_font_style = AXIS_LABEL_FONT_STYLE
    plot.yaxis.axis_label_text_font_style = AXIS_LABEL_FONT_STYLE
    plot.xaxis.major_label_text_font_size = scaled_pt(AXIS_MAJOR_LABEL_FONT_PT)
    plot.yaxis.major_label_text_font_size = scaled_pt(AXIS_MAJOR_LABEL_FONT_PT)

    return plot


def export_png_image(plot: figure, path: Path) -> None:
    """Export the figure to PNG (requires a Selenium webdriver)."""
    try:
        from bokeh.io import export_png

        export_png(plot, filename=str(path))
        print(f"[INFO] Saved PNG to {path}.")
    except Exception as exc:  # noqa: BLE001
        print(
            f"[WARN] PNG export failed ({exc}); a Selenium webdriver "
            "(geckodriver/chromedriver) is required for export_png."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a geographic bounding-box filter visualization."
    )
    parser.add_argument(
        "--savepng", action="store_true", help="Also export a PNG image."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Write the HTML without opening a browser.",
    )
    args = parser.parse_args()

    plot = build_plot()
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    output_file(OUTPUT_HTML, title=TITLE_TEXT)

    if args.no_show:
        save(plot)
        print(f"[INFO] Saved HTML to {OUTPUT_HTML}.")
    else:
        show(plot)
        print(f"[INFO] Rendered HTML to {OUTPUT_HTML}.")

    if args.savepng:
        export_png_image(plot, OUTPUT_PNG)


if __name__ == "__main__":
    main()
