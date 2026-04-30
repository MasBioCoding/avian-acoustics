#!/usr/bin/env python3
"""Review annotation labels against locally generated spectrogram PNG files.

The script joins an annotation CSV to an ingroup metadata CSV, derives the
spectrogram PNG filename from each metadata ``filename`` value, and starts a
small local web server with a scrollable positive/negative review page.

Typical usage:
    python verify_annotation.py \
        --annotations /path/to/annotations.csv \
        --meta /path/to/embeddings/<species>/ingroup_energy_with_meta.csv \
        --label chirp_staple \
        --open
        
    python verify_annotation.py \
        --annotations /Volumes/Z Slim/zslim_birdcluster/embeddings/phylloscopus_collybita/annotations.csv \
        --meta /Volumes/Z Slim/zslim_birdcluster/embeddings/phylloscopus_collybita/ingroup_energy_with_meta.csv \
        --label song_updown \
        --open


If ``--spectrogram-dir`` is omitted, the script tries to infer it from a
metadata path shaped like ``<data-root>/embeddings/<species>/...`` by using
``<data-root>/spectrograms/<species>``.
"""

from __future__ import annotations

import argparse
import csv
import html
import mimetypes
import shutil
import sys
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

POSITIVE = "positive"
NEGATIVE = "negative"
OTHER = "other"
REVIEW_TYPES = {POSITIVE, NEGATIVE}


@dataclass(frozen=True)
class AnnotationRow:
    """One selected annotation row from the annotation CSV."""

    line_number: int
    annotation_id: str
    window_id: str
    label: str
    raw_label_type: str
    label_type: str
    provenance: str
    timestamp: str


@dataclass(frozen=True)
class MetadataLoadResult:
    """Metadata rows keyed by normalized source window ID."""

    rows_by_window_id: dict[str, dict[str, str]]
    duplicate_window_ids: int


@dataclass(frozen=True)
class SpectrogramEntry:
    """Display-ready annotation joined to an optional metadata row."""

    annotation: AnnotationRow
    metadata: dict[str, str] | None
    wav_filename: str
    png_filename: str
    image_exists: bool

    @property
    def label_type(self) -> str:
        """Return the normalized positive/negative label type."""

        return self.annotation.label_type

    @property
    def missing_reason(self) -> str:
        """Explain why an entry cannot show a spectrogram image."""

        if self.metadata is None:
            return f"No metadata row for window_id {self.annotation.window_id}"
        if not self.wav_filename:
            return "Metadata row has no filename value"
        if not self.png_filename:
            return "Could not derive PNG filename"
        if not self.image_exists:
            return f"Missing image: {self.png_filename}"
        return ""


@dataclass(frozen=True)
class BuildStats:
    """Counts collected while preparing the review page."""

    skipped_duplicate_annotations: int
    skipped_unreviewable_label_types: int


def normalize_window_id(value: str | None) -> str:
    """Normalize CSV window IDs so integer-like values match across files."""

    cleaned = (value or "").strip()
    if cleaned.endswith(".0") and cleaned[:-2].isdigit():
        return cleaned[:-2]
    return cleaned


def normalize_label_type(value: str | None) -> str:
    """Normalize common positive/negative label type encodings."""

    cleaned = (value or "").strip().lower()
    if not cleaned:
        return OTHER

    if "positive" in cleaned or cleaned in {"1", "true", "yes", "pos", "p"}:
        return POSITIVE
    if "negative" in cleaned or cleaned in {"0", "false", "no", "neg", "n"}:
        return NEGATIVE
    return OTHER


def require_columns(
    fieldnames: list[str] | None, required: set[str], csv_path: Path
) -> None:
    """Exit with a useful error when a CSV is missing required columns."""

    existing = set(fieldnames or [])
    missing = sorted(required - existing)
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"{csv_path} is missing required column(s): {joined}")


def read_annotations(
    annotations_path: Path, label: str | None
) -> tuple[list[AnnotationRow], dict[str, dict[str, int]]]:
    """Read annotation rows and return rows matching ``label`` plus label counts."""

    selected_rows: list[AnnotationRow] = []
    label_counts: dict[str, dict[str, int]] = {}

    with annotations_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(
            reader.fieldnames, {"window_id", "label", "label_type"}, annotations_path
        )

        for line_number, row in enumerate(reader, start=2):
            row_label = (row.get("label") or "").strip()
            label_type = normalize_label_type(row.get("label_type"))
            counts = label_counts.setdefault(
                row_label, {POSITIVE: 0, NEGATIVE: 0, OTHER: 0}
            )
            counts[label_type] = counts.get(label_type, 0) + 1

            if label is not None and row_label != label:
                continue

            selected_rows.append(
                AnnotationRow(
                    line_number=line_number,
                    annotation_id=(row.get("id") or "").strip(),
                    window_id=normalize_window_id(row.get("window_id")),
                    label=row_label,
                    raw_label_type=(row.get("label_type") or "").strip(),
                    label_type=label_type,
                    provenance=(row.get("provenance") or "").strip(),
                    timestamp=(row.get("timestamp") or "").strip(),
                )
            )

    return selected_rows, label_counts


def read_metadata(metadata_path: Path) -> MetadataLoadResult:
    """Read ingroup metadata rows keyed by ``source_window_id``."""

    rows_by_window_id: dict[str, dict[str, str]] = {}
    duplicate_window_ids = 0

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(
            reader.fieldnames, {"source_window_id", "filename"}, metadata_path
        )

        for row in reader:
            window_id = normalize_window_id(row.get("source_window_id"))
            if not window_id:
                continue
            if window_id in rows_by_window_id:
                duplicate_window_ids += 1
                continue
            rows_by_window_id[window_id] = {
                key: value or "" for key, value in row.items()
            }

    return MetadataLoadResult(
        rows_by_window_id=rows_by_window_id,
        duplicate_window_ids=duplicate_window_ids,
    )


def png_name_from_wav(filename: str) -> str:
    """Derive the expected spectrogram PNG basename from a WAV filename."""

    if not filename:
        return ""
    return Path(filename).with_suffix(".png").name


def infer_spectrogram_dir(metadata_path: Path) -> Path | None:
    """Infer ``<data-root>/spectrograms/<species>`` from an embeddings CSV path."""

    resolved = metadata_path.expanduser().resolve()
    parts = resolved.parts
    for index, part in enumerate(parts):
        if part not in {"embeddings", "embeddings_birdnet"}:
            continue
        if index + 1 >= len(parts):
            continue
        data_root = Path(*parts[:index])
        species_slug = parts[index + 1]
        return data_root / "spectrograms" / species_slug
    return None


def build_entries(
    annotations: list[AnnotationRow],
    metadata: MetadataLoadResult,
    spectrogram_dir: Path,
    allow_duplicates: bool,
) -> tuple[list[SpectrogramEntry], BuildStats]:
    """Join selected annotations to metadata and prepare display entries."""

    entries: list[SpectrogramEntry] = []
    seen: set[tuple[str, str]] = set()
    skipped_duplicate_annotations = 0
    skipped_unreviewable_label_types = 0

    for annotation in annotations:
        if annotation.label_type not in REVIEW_TYPES:
            skipped_unreviewable_label_types += 1
            continue

        dedupe_key = (annotation.window_id, annotation.label_type)
        if not allow_duplicates and dedupe_key in seen:
            skipped_duplicate_annotations += 1
            continue
        seen.add(dedupe_key)

        metadata_row = metadata.rows_by_window_id.get(annotation.window_id)
        wav_filename = (metadata_row or {}).get("filename", "")
        png_filename = png_name_from_wav(wav_filename)
        image_exists = bool(png_filename) and (spectrogram_dir / png_filename).is_file()

        entries.append(
            SpectrogramEntry(
                annotation=annotation,
                metadata=metadata_row,
                wav_filename=Path(wav_filename).name if wav_filename else "",
                png_filename=png_filename,
                image_exists=image_exists,
            )
        )

    return entries, BuildStats(
        skipped_duplicate_annotations=skipped_duplicate_annotations,
        skipped_unreviewable_label_types=skipped_unreviewable_label_types,
    )


def sort_entries(
    entries: list[SpectrogramEntry], sort_by: str
) -> list[SpectrogramEntry]:
    """Return entries sorted by the requested display key."""

    if sort_by == "annotation":
        return entries
    if sort_by == "filename":
        return sorted(entries, key=lambda entry: entry.png_filename)
    if sort_by == "window_id":
        return sorted(
            entries, key=lambda entry: sortable_window_id(entry.annotation.window_id)
        )
    raise ValueError(f"Unknown sort key: {sort_by}")


def sortable_window_id(value: str) -> tuple[int, int | str]:
    """Sort integer-looking window IDs numerically and other IDs lexically."""

    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def limit_entries(
    entries: list[SpectrogramEntry], limit_per_type: int | None
) -> list[SpectrogramEntry]:
    """Limit displayed entries per positive/negative type when requested."""

    if limit_per_type is None:
        return entries

    counters = {POSITIVE: 0, NEGATIVE: 0}
    limited: list[SpectrogramEntry] = []
    for entry in entries:
        if counters[entry.label_type] >= limit_per_type:
            continue
        counters[entry.label_type] += 1
        limited.append(entry)
    return limited


def count_entries(entries: list[SpectrogramEntry]) -> dict[str, int]:
    """Count entries by normalized label type and missing image status."""

    counts = {
        POSITIVE: 0,
        NEGATIVE: 0,
        "positive_missing": 0,
        "negative_missing": 0,
    }
    for entry in entries:
        counts[entry.label_type] += 1
        if not entry.image_exists:
            counts[f"{entry.label_type}_missing"] += 1
    return counts


def html_escape(value: Any) -> str:
    """Escape text for HTML element content."""

    return html.escape(str(value), quote=False)


def html_attr(value: Any) -> str:
    """Escape text for HTML attribute values."""

    return html.escape(str(value), quote=True)


def render_detail(label: str, value: str) -> str:
    """Render one compact metadata field if ``value`` is present."""

    if not value:
        return ""
    return (
        '<div class="detail">'
        f"<dt>{html_escape(label)}</dt>"
        f"<dd>{html_escape(value)}</dd>"
        "</div>"
    )


def render_entry(entry: SpectrogramEntry) -> str:
    """Render one spectrogram review card."""

    metadata = entry.metadata or {}
    annotation = entry.annotation
    search_text = " ".join(
        value
        for value in [
            annotation.window_id,
            annotation.label,
            annotation.raw_label_type,
            annotation.provenance,
            annotation.timestamp,
            entry.wav_filename,
            entry.png_filename,
            metadata.get("xcid", ""),
            metadata.get("recordist", ""),
            metadata.get("date", ""),
            metadata.get("lat", ""),
            metadata.get("lon", ""),
        ]
        if value
    ).lower()
    exists_value = "true" if entry.image_exists else "false"
    missing_class = "" if entry.image_exists else " is-missing"
    image_markup = (
        f'<img src="/spectrogram/{quote(entry.png_filename)}" '
        f'alt="{html_attr(entry.png_filename)}" loading="lazy" decoding="async">'
        if entry.image_exists
        else f'<div class="missing-box">{html_escape(entry.missing_reason)}</div>'
    )
    title = entry.png_filename or f"window_id {annotation.window_id}"

    details = "".join(
        [
            render_detail("window", annotation.window_id),
            render_detail("annotation", annotation.annotation_id),
            render_detail("wav", entry.wav_filename),
            render_detail("xcid", metadata.get("xcid", "")),
            render_detail("clip", metadata.get("clip_index", "")),
            render_detail("recordist", metadata.get("recordist", "")),
            render_detail("date", metadata.get("date", "")),
            render_detail("lat", metadata.get("lat", "")),
            render_detail("lon", metadata.get("lon", "")),
            render_detail("by", annotation.provenance),
        ]
    )

    return f"""
<article class="entry {html_attr(entry.label_type)}{missing_class}"
    data-group="{html_attr(entry.label_type)}"
    data-exists="{exists_value}"
    data-search="{html_attr(search_text)}">
  <div class="image-wrap">{image_markup}</div>
  <div class="entry-body">
    <h3>{html_escape(title)}</h3>
    <dl>{details}</dl>
  </div>
</article>
"""


def render_section(entries: list[SpectrogramEntry], label_type: str) -> str:
    """Render the positive or negative scrollable column."""

    section_entries = [entry for entry in entries if entry.label_type == label_type]
    title = label_type.capitalize()
    cards = "\n".join(render_entry(entry) for entry in section_entries)
    if not cards:
        cards = '<p class="empty">No entries.</p>'

    return f"""
<section class="panel {html_attr(label_type)}" id="{html_attr(label_type)}">
  <div class="panel-head">
    <h2>{title}</h2>
    <span>
      <strong data-count-for="{html_attr(label_type)}">{len(section_entries)}</strong>
    </span>
  </div>
  <div class="entry-list">{cards}</div>
</section>
"""


def build_page(
    entries: list[SpectrogramEntry],
    label: str,
    annotations_path: Path,
    metadata_path: Path,
    spectrogram_dir: Path,
    stats: BuildStats,
    metadata: MetadataLoadResult,
) -> str:
    """Build the complete HTML review page."""

    counts = count_entries(entries)
    total_missing = counts["positive_missing"] + counts["negative_missing"]
    sections = "\n".join(
        [
            render_section(entries, POSITIVE),
            render_section(entries, NEGATIVE),
        ]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Annotation verification: {html_escape(label)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7f9;
      --surface: #ffffff;
      --text: #1d2733;
      --muted: #65758a;
      --line: #d9e0e8;
      --positive: #13795b;
      --negative: #b42318;
      --focus: #2457c5;
      --missing-bg: #fff4d6;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      min-height: 100%;
    }}
    body {{
      margin: 0;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.35;
    }}
    header.topbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: grid;
      gap: 10px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(8px);
    }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 20px; font-weight: 700; }}
    .summary, .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      align-items: center;
    }}
    .summary span {{
      color: var(--muted);
      white-space: nowrap;
    }}
    .summary strong {{ color: var(--text); }}
    input[type="search"] {{
      width: min(520px, 100%);
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #ffffff;
      color: var(--text);
      font: inherit;
    }}
    input[type="search"]:focus {{
      outline: 2px solid rgba(36, 87, 197, 0.25);
      border-color: var(--focus);
    }}
    label.toggle {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
    }}
    .jump {{
      color: var(--focus);
      text-decoration: none;
      font-weight: 600;
    }}
    main {{
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
      padding: 14px;
    }}
    .panel {{
      min-height: 0;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
    }}
    .panel-head {{
      position: sticky;
      top: 0;
      z-index: 1;
      display: flex;
      justify-content: space-between;
      align-items: center;
      min-height: 42px;
      padding: 9px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }}
    .panel-head h2 {{ font-size: 16px; }}
    .panel.positive .panel-head {{ border-top: 4px solid var(--positive); }}
    .panel.negative .panel-head {{ border-top: 4px solid var(--negative); }}
    .entry-list {{
      height: calc(100% - 42px);
      overflow-y: auto;
      padding: 10px;
    }}
    .entry {{
      overflow: hidden;
      margin-bottom: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
    }}
    .entry.positive {{ border-left: 4px solid var(--positive); }}
    .entry.negative {{ border-left: 4px solid var(--negative); }}
    .entry.is-missing {{ background: #fffaf0; }}
    .image-wrap {{
      min-height: 108px;
      display: grid;
      place-items: center;
      background: #151923;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .missing-box {{
      width: 100%;
      min-height: 108px;
      display: grid;
      place-items: center;
      padding: 16px;
      background: var(--missing-bg);
      color: #674d00;
      text-align: center;
      overflow-wrap: anywhere;
    }}
    .entry-body {{ padding: 9px 10px 10px; }}
    .entry-body h3 {{
      margin-bottom: 8px;
      font-size: 14px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }}
    dl {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px 10px;
      margin: 0;
    }}
    .detail {{ min-width: 0; }}
    dt {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    .empty {{
      margin: 8px;
      color: var(--muted);
    }}
    [hidden] {{ display: none !important; }}
    @media (max-width: 900px) {{
      body {{
        height: auto;
        overflow: auto;
      }}
      main {{
        flex: none;
        grid-template-columns: 1fr;
        height: auto;
      }}
      .panel {{ max-height: none; }}
      .entry-list {{
        height: auto;
        overflow-y: visible;
      }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <h1>Annotation verification</h1>
    <div class="summary">
      <span>Label <strong>{html_escape(label)}</strong></span>
      <span>Positive <strong>{counts[POSITIVE]}</strong></span>
      <span>Negative <strong>{counts[NEGATIVE]}</strong></span>
      <span>Missing images <strong data-missing-count>{total_missing}</strong></span>
      <span>
        Skipped duplicates <strong>{stats.skipped_duplicate_annotations}</strong>
      </span>
      <span>
        Skipped other types <strong>{stats.skipped_unreviewable_label_types}</strong>
      </span>
      <span>
        Duplicate metadata IDs <strong>{metadata.duplicate_window_ids}</strong>
      </span>
    </div>
    <div class="summary">
      <span>Annotations <strong>{html_escape(annotations_path)}</strong></span>
      <span>Metadata <strong>{html_escape(metadata_path)}</strong></span>
      <span>Spectrograms <strong>{html_escape(spectrogram_dir)}</strong></span>
    </div>
    <div class="controls">
      <input id="filter" type="search"
        placeholder="Filter window, filename, xcid, recordist, date">
      <label class="toggle">
        <input id="showMissing" type="checkbox" checked> Show missing
      </label>
      <a class="jump" href="#positive">Positive</a>
      <a class="jump" href="#negative">Negative</a>
    </div>
  </header>
  <main>{sections}</main>
  <script>
    const filterInput = document.getElementById("filter");
    const showMissingInput = document.getElementById("showMissing");
    const countTargets = {{
      positive: document.querySelector('[data-count-for="positive"]'),
      negative: document.querySelector('[data-count-for="negative"]'),
    }};
    const missingTarget = document.querySelector("[data-missing-count]");

    function applyFilters() {{
      const term = filterInput.value.trim().toLowerCase();
      const showMissing = showMissingInput.checked;
      const visible = {{ positive: 0, negative: 0 }};
      let missing = 0;

      document.querySelectorAll(".entry").forEach((entry) => {{
        const isMissing = entry.dataset.exists !== "true";
        const matchesTerm = term === "" || entry.dataset.search.includes(term);
        const shouldShow = matchesTerm && (showMissing || !isMissing);
        entry.hidden = !shouldShow;
        if (shouldShow) {{
          visible[entry.dataset.group] += 1;
          if (isMissing) missing += 1;
        }}
      }});

      countTargets.positive.textContent = visible.positive;
      countTargets.negative.textContent = visible.negative;
      missingTarget.textContent = missing;
    }}

    filterInput.addEventListener("input", applyFilters);
    showMissingInput.addEventListener("change", applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


def make_handler(
    page_html: str,
    spectrogram_dir: Path,
    log_requests: bool,
) -> type[BaseHTTPRequestHandler]:
    """Create an HTTP handler serving the review page and selected PNG files."""

    encoded_page = page_html.encode("utf-8")
    root = spectrogram_dir.resolve()

    class VerifyAnnotationHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Serve the HTML page or a spectrogram image."""

            self._handle_request(send_body=True)

        def do_HEAD(self) -> None:
            """Return headers for the HTML page or a spectrogram image."""

            self._handle_request(send_body=False)

        def log_message(self, format: str, *args: Any) -> None:
            if log_requests:
                super().log_message(format, *args)

        def _handle_request(self, send_body: bool) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_bytes(
                    encoded_page,
                    content_type="text/html; charset=utf-8",
                    send_body=send_body,
                )
                return

            if parsed.path.startswith("/spectrogram/"):
                requested_name = unquote(parsed.path.removeprefix("/spectrogram/"))
                image_path = self._resolve_image_path(requested_name)
                if image_path is None or not image_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "Spectrogram not found")
                    return
                self._send_file(image_path, send_body=send_body)
                return

            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def _resolve_image_path(self, requested_name: str) -> Path | None:
            if not requested_name or requested_name != Path(requested_name).name:
                return None
            if "\\" in requested_name:
                return None
            candidate = (root / requested_name).resolve()
            if candidate.parent != root:
                return None
            return candidate

        def _send_bytes(
            self, payload: bytes, content_type: str, send_body: bool
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if send_body:
                self.wfile.write(payload)

        def _send_file(self, file_path: Path, send_body: bool) -> None:
            content_type = (
                mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            )
            stat = file_path.stat()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(stat.st_size))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            if send_body:
                with file_path.open("rb") as handle:
                    shutil.copyfileobj(handle, self.wfile)

    return VerifyAnnotationHandler


def serve_page(
    page_html: str,
    spectrogram_dir: Path,
    host: str,
    port: int,
    log_requests: bool,
) -> tuple[ThreadingHTTPServer, str]:
    """Start a local HTTP server and return it with its public URL."""

    class ReusableThreadingHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True

    handler = make_handler(page_html, spectrogram_dir, log_requests)
    server = ReusableThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    url_host = "localhost" if host in {"0.0.0.0", ""} else host
    url = f"http://{url_host}:{server.server_address[1]}"
    return server, url


def print_label_counts(label_counts: dict[str, dict[str, int]]) -> None:
    """Print available annotation labels with positive/negative counts."""

    print("Available labels:")
    for label, counts in sorted(label_counts.items()):
        print(
            f"  {label}: "
            f"{counts.get(POSITIVE, 0)} positive, "
            f"{counts.get(NEGATIVE, 0)} negative, "
            f"{counts.get(OTHER, 0)} other"
        )


def print_summary(
    entries: list[SpectrogramEntry],
    label: str,
    spectrogram_dir: Path,
    stats: BuildStats,
    metadata: MetadataLoadResult,
) -> None:
    """Print the selected review set summary."""

    counts = count_entries(entries)
    print(f"Label: {label}")
    print(f"Spectrogram directory: {spectrogram_dir}")
    print(
        "Selected entries: "
        f"{counts[POSITIVE]} positive, {counts[NEGATIVE]} negative"
    )
    print(
        "Missing images: "
        f"{counts['positive_missing']} positive, {counts['negative_missing']} negative"
    )
    print(f"Skipped duplicate annotations: {stats.skipped_duplicate_annotations}")
    print(f"Skipped other label types: {stats.skipped_unreviewable_label_types}")
    print(
        "Duplicate metadata source_window_id rows ignored: "
        f"{metadata.duplicate_window_ids}"
    )


def path_from_fragments(
    value: list[str] | str | Path | None, default: Path | None = None
) -> Path | None:
    """Build a path from one or more shell-split path fragments."""

    if value is None:
        return default
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    if not value:
        return default
    return Path(" ".join(value))


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Serve a fast browser-based positive/negative spectrogram review page "
            "for one annotation label class."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        metavar="ANNOTATIONS",
        help="Annotation CSV with window_id, label, and label_type columns.",
    )
    parser.add_argument(
        "--meta",
        "--metadata",
        dest="metadata",
        nargs="+",
        metavar="METADATA",
        help="ingroup_energy_with_meta.csv with source_window_id and filename columns.",
    )
    parser.add_argument(
        "--spectrogram-dir",
        nargs="+",
        metavar="SPECTROGRAM_DIR",
        help=(
            "Directory containing spectrogram PNG files. Inferred from --meta "
            "when possible."
        ),
    )
    parser.add_argument(
        "--label",
        "--label-class",
        dest="label",
        help="Annotation label class to review. Omit to list available labels.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the local review server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Port for the local review server. Use 0 to choose a free port.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("annotation", "window_id", "filename"),
        default="annotation",
        help="Display order for entries.",
    )
    parser.add_argument(
        "--limit-per-type",
        type=int,
        help="Optional maximum number of positive and negative entries to display.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Show duplicate annotations for the same window_id and label_type.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the review URL in the default browser.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and exit without starting the server.",
    )
    parser.add_argument(
        "--log-requests",
        action="store_true",
        help="Print HTTP request logs while serving.",
    )
    return parser


def resolve_inputs(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[Path, Path, Path]:
    """Validate and resolve input paths from parsed arguments."""

    annotations_path = path_from_fragments(args.annotations, Path("annotations.csv"))
    if annotations_path is None:
        parser.error("--annotations is required")
    annotations_path = annotations_path.expanduser()
    if not annotations_path.is_file():
        parser.error(f"Annotation CSV not found: {annotations_path}")

    if args.metadata is None:
        parser.error("--meta is required when --label is set")

    metadata_path = path_from_fragments(args.metadata)
    if metadata_path is None:
        parser.error("--meta is required when --label is set")
    metadata_path = metadata_path.expanduser()
    if not metadata_path.is_file():
        parser.error(f"Metadata CSV not found: {metadata_path}")

    spectrogram_dir = path_from_fragments(args.spectrogram_dir)
    spectrogram_dir = spectrogram_dir.expanduser() if spectrogram_dir else None
    if spectrogram_dir is None:
        spectrogram_dir = infer_spectrogram_dir(metadata_path)
    if spectrogram_dir is None:
        parser.error(
            "Could not infer --spectrogram-dir from --meta; pass it explicitly"
        )
    if not spectrogram_dir.is_dir():
        parser.error(f"Spectrogram directory not found: {spectrogram_dir}")

    return annotations_path, metadata_path, spectrogram_dir


def main(argv: list[str] | None = None) -> int:
    """Run the annotation verification server."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.limit_per_type is not None and args.limit_per_type < 1:
        parser.error("--limit-per-type must be at least 1")
    if args.port < 0 or args.port > 65535:
        parser.error("--port must be between 0 and 65535")

    annotations_path = path_from_fragments(args.annotations, Path("annotations.csv"))
    if annotations_path is None:
        parser.error("--annotations is required")
    annotations_path = annotations_path.expanduser()
    if not annotations_path.is_file():
        parser.error(f"Annotation CSV not found: {annotations_path}")

    selected_annotations, label_counts = read_annotations(annotations_path, args.label)
    if args.label is None:
        print_label_counts(label_counts)
        return 0
    if not selected_annotations:
        print(f"No annotations found for label: {args.label}", file=sys.stderr)
        print_label_counts(label_counts)
        return 1

    annotations_path, metadata_path, spectrogram_dir = resolve_inputs(args, parser)
    metadata = read_metadata(metadata_path)
    entries, stats = build_entries(
        selected_annotations,
        metadata,
        spectrogram_dir,
        allow_duplicates=args.allow_duplicates,
    )
    entries = sort_entries(entries, args.sort_by)
    entries = limit_entries(entries, args.limit_per_type)

    print_summary(entries, args.label, spectrogram_dir, stats, metadata)
    if args.dry_run:
        return 0

    page_html = build_page(
        entries=entries,
        label=args.label,
        annotations_path=annotations_path,
        metadata_path=metadata_path,
        spectrogram_dir=spectrogram_dir,
        stats=stats,
        metadata=metadata,
    )
    server, url = serve_page(
        page_html=page_html,
        spectrogram_dir=spectrogram_dir,
        host=args.host,
        port=args.port,
        log_requests=args.log_requests,
    )

    print(f"Review server: {url}")
    print("Press Ctrl-C to stop.")
    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review server.")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
