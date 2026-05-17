#!/usr/bin/env python3
"""Serve a lightweight spectrogram annotation UI for inference evaluation.

The app samples rows from an ``inference.csv`` file, maps each WAV filename to
the matching spectrogram PNG, and starts a local browser UI for positive/negative
annotation. It persists labels to a small CSV so an interrupted session can be
resumed.

Typical usage:
    python evaluation.py --open

    python evaluation.py \
        --species-slug prunella_modularis \
        --inference-name song_trill \
        --mode random \
        --logit-min 0 \
        --logit-max 1 \
        --sample-size 120 \
        --open

    python evaluation.py \
        --species-slug prunella_modularis \
        --inference-name song_trill \
        --mode stratified \
        --bins "0:0.25:30,0.25:0.50:30,0.50:0.75:30,0.75:1.00:30" \
        --open

By default, paths are derived from ``BIRDCLUSTER_DATA_ROOT`` or
``/Volumes/Z Slim/zslim_birdcluster``:
    agile_inferences/<species_slug>/<inference_name>/inference.csv
    spectrograms/<species_slug>/
    eval_annotations/<species_slug>/<inference_name>.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import random
import re
import shutil
import sys
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import unquote, urlparse

DEFAULT_DATA_ROOT = Path(
    os.environ.get("BIRDCLUSTER_DATA_ROOT", "/Volumes/Z Slim/zslim_birdcluster")
)
DEFAULT_SPECIES_SLUG = os.environ.get("BIRDCLUSTER_SPECIES_SLUG", "prunella_modularis")
DEFAULT_INFERENCE_NAME = os.environ.get("BIRDCLUSTER_INFERENCE_NAME", "chirps_straight")
DEFAULT_BINS = "0:0.25,0.25:0.50,0.50:0.60,0.60:1.00"
ANNOTATION_VALUES = {"positive", "negative"}
ANNOTATION_FIELDNAMES = [
    "row_key",
    "annotation",
    "annotated_at",
    "filename",
    "png_filename",
    "score",
    "label",
    "source_idx",
    "source_line",
    "window_start",
    "window_end",
    "stratum",
    "weight",
    "inference_csv",
]
FLOAT_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*-\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)


@dataclass(frozen=True)
class BinSpec:
    """One score bin used for stratified sampling."""

    name: str
    lower: float
    upper: float
    target: int

    def contains(self, score: float, *, include_upper: bool) -> bool:
        """Return whether ``score`` falls in this bin."""

        if include_upper:
            return self.lower <= score <= self.upper
        return self.lower <= score < self.upper


@dataclass
class InferenceItem:
    """One sampled inference row prepared for browser display."""

    row_key: str
    line_number: int
    filename: str
    png_filename: str
    score: float
    stratum: str
    metadata: dict[str, str]
    image_exists: bool = False
    stratum_population: int = 0
    stratum_sampled: int = 0
    weight: float = 1.0


@dataclass
class Bucket:
    """Reservoir-sampled rows for one sampling stratum."""

    target: int
    population: int = 0
    sample: list[InferenceItem] = field(default_factory=list)


@dataclass
class LoadStats:
    """Counts collected while streaming the inference CSV."""

    total_rows: int = 0
    valid_score_rows: int = 0
    missing_score_rows: int = 0
    missing_filename_rows: int = 0
    candidate_rows: int = 0
    missing_images: int = 0
    score_min: float | None = None
    score_max: float | None = None

    def update_score_range(self, score: float) -> None:
        """Update the observed score range."""

        if self.score_min is None or score < self.score_min:
            self.score_min = score
        if self.score_max is None or score > self.score_max:
            self.score_max = score


@dataclass
class SampleResult:
    """Sampled items plus diagnostics."""

    items: list[InferenceItem]
    stats: LoadStats
    bucket_summaries: list[dict[str, Any]]


@dataclass
class AppState:
    """Mutable server state shared across HTTP requests."""

    items: list[InferenceItem]
    stats: LoadStats
    bucket_summaries: list[dict[str, Any]]
    annotations: dict[str, dict[str, str]]
    annotations_path: Path
    inference_csv: Path
    spectrogram_dir: Path
    species_slug: str
    inference_name: str
    sampling_mode: str
    threshold: float
    verbose: bool
    lock: Lock = field(default_factory=Lock)

    @property
    def items_by_key(self) -> dict[str, InferenceItem]:
        """Return sampled items keyed by stable row key."""

        return {item.row_key: item for item in self.items}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the annotation server."""

    parser = argparse.ArgumentParser(
        description=(
            "Sample an inference.csv file and serve a fast local spectrogram "
            "annotation UI."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Root containing agile_inferences/ and spectrograms/. Defaults to "
            "BIRDCLUSTER_DATA_ROOT or /Volumes/Z Slim/zslim_birdcluster."
        ),
    )
    parser.add_argument(
        "--species-slug",
        default=DEFAULT_SPECIES_SLUG,
        help="Species folder slug, for example prunella_modularis.",
    )
    parser.add_argument(
        "--inference-name",
        default=DEFAULT_INFERENCE_NAME,
        help="Inference run folder under agile_inferences/<species_slug>/.",
    )
    parser.add_argument(
        "--inference-csv",
        type=Path,
        default=None,
        help="Explicit inference CSV path. Overrides --data-root and --inference-name.",
    )
    parser.add_argument(
        "--spectrogram-dir",
        type=Path,
        default=None,
        help=(
            "Explicit spectrogram PNG directory. Defaults to "
            "spectrograms/<species_slug>."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Annotation CSV path. Defaults to "
            "<data-root>/eval_annotations/<species_slug>/<inference_name>.csv."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("random", "stratified"),
        default="random",
        help="Sampling protocol to use.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=120,
        help="Rows to sample in random mode. Use 0 to include all matching rows.",
    )
    parser.add_argument(
        "--logit-min",
        type=float,
        default=None,
        help="Minimum score for random sampling. Omit for no lower bound.",
    )
    parser.add_argument(
        "--logit-max",
        type=float,
        default=None,
        help="Maximum score for random sampling. Omit for no upper bound.",
    )
    parser.add_argument(
        "--bins",
        default=DEFAULT_BINS,
        help=(
            "Comma-separated stratified bins. Use lower:upper or lower:upper:n, "
            "for example '0:0.25:25,0.25:0.5:25'."
        ),
    )
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=30,
        help="Default sample count per stratified bin when a bin omits :n.",
    )
    parser.add_argument(
        "--score-column",
        default="logits",
        help="Numeric score column used for sampling and threshold precision.",
    )
    parser.add_argument(
        "--filename-column",
        default="filename",
        help="Column containing WAV filenames that map to spectrogram PNG names.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional filter for rows where the CSV label column matches this value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Initial UI threshold for precision calculation.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP host for the local annotation UI.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8790,
        help="HTTP port for the local annotation UI. Use 0 for any free port.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the annotation UI in the default browser.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep sampled rows grouped by stratum instead of shuffling display order.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log HTTP requests.",
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve inference, spectrogram, and annotation output paths."""

    data_root = args.data_root.expanduser()
    inference_csv = args.inference_csv
    if inference_csv is None:
        inference_csv = (
            data_root
            / "agile_inferences"
            / args.species_slug
            / args.inference_name
            / "inference.csv"
        )
    inference_csv = inference_csv.expanduser()

    spectrogram_dir = args.spectrogram_dir
    if spectrogram_dir is None:
        spectrogram_dir = data_root / "spectrograms" / args.species_slug
    spectrogram_dir = spectrogram_dir.expanduser()

    annotations_path = args.output
    if annotations_path is None:
        annotations_path = (
            data_root
            / "eval_annotations"
            / args.species_slug
            / f"{args.inference_name}.csv"
        )
    annotations_path = annotations_path.expanduser()

    return inference_csv, spectrogram_dir, annotations_path


def require_readable_file(path: Path, label: str) -> None:
    """Exit with a clear error when a required file is missing."""

    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def require_directory(path: Path, label: str) -> None:
    """Exit with a clear error when a required directory is missing."""

    if not path.is_dir():
        raise SystemExit(f"{label} not found: {path}")


def parse_bins(value: str, default_target: int) -> list[BinSpec]:
    """Parse CLI bin specs into ordered ``BinSpec`` values."""

    if default_target < 0:
        raise SystemExit("--samples-per-bin must be >= 0")

    bins: list[BinSpec] = []
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            continue

        target = default_target
        range_part = token
        if "=" in token:
            range_part, target_part = token.rsplit("=", 1)
            target = parse_nonnegative_int(target_part, token)
        elif token.count(":") == 2:
            lower_part, upper_part, target_part = token.split(":")
            range_part = f"{lower_part}:{upper_part}"
            target = parse_nonnegative_int(target_part, token)

        lower: float
        upper: float
        if ":" in range_part:
            lower_part, upper_part = range_part.split(":", 1)
            lower = parse_float(lower_part, token)
            upper = parse_float(upper_part, token)
        else:
            match = FLOAT_RE.match(range_part)
            if not match:
                raise SystemExit(
                    "Could not parse bin "
                    f"'{token}'. Use lower:upper, lower-upper, or lower:upper:n."
                )
            lower = parse_float(match.group(1), token)
            upper = parse_float(match.group(2), token)

        if upper <= lower:
            raise SystemExit(f"Bin upper bound must exceed lower bound: {token}")
        bins.append(
            BinSpec(
                name=f"{lower:g}-{upper:g}",
                lower=lower,
                upper=upper,
                target=target,
            )
        )

    if not bins:
        raise SystemExit("--bins did not contain any valid bins")
    return bins


def parse_float(value: str, context: str) -> float:
    """Parse a float for CLI configuration."""

    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid float in '{context}': {value}") from exc


def parse_nonnegative_int(value: str, context: str) -> int:
    """Parse a nonnegative integer for CLI configuration."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer in '{context}': {value}") from exc
    if parsed < 0:
        raise SystemExit(f"Sample count must be >= 0 in '{context}'")
    return parsed


def row_key_from(row: dict[str, str], line_number: int) -> str:
    """Build a stable key for one inference row."""

    identity_parts = [
        row.get("idx", ""),
        row.get("filename", ""),
        row.get("window_start", ""),
        row.get("window_end", ""),
        row.get("label", ""),
    ]
    identity = "\x1f".join(str(part).strip() for part in identity_parts)
    if not identity.strip("\x1f"):
        identity = f"line:{line_number}"
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]


def png_name_from_wav(filename: str) -> str:
    """Derive the expected spectrogram PNG basename from a WAV filename."""

    cleaned = Path(filename).name
    if not cleaned:
        return ""
    return Path(cleaned).with_suffix(".png").name


def row_to_item(
    *,
    row: dict[str, str],
    line_number: int,
    filename_column: str,
    score_column: str,
    score: float,
    stratum: str,
) -> InferenceItem:
    """Convert a CSV row to a sampled item."""

    filename = Path(row.get(filename_column, "")).name
    metadata = {key: value or "" for key, value in row.items()}
    metadata["source_line"] = str(line_number)
    return InferenceItem(
        row_key=row_key_from(row, line_number),
        line_number=line_number,
        filename=filename,
        png_filename=png_name_from_wav(filename),
        score=score,
        stratum=stratum,
        metadata=metadata,
    )


def add_to_reservoir(bucket: Bucket, item: InferenceItem, rng: random.Random) -> None:
    """Add ``item`` to a fixed-size reservoir sample."""

    bucket.population += 1
    if bucket.target == 0:
        bucket.sample.append(item)
        return

    if len(bucket.sample) < bucket.target:
        bucket.sample.append(item)
        return

    replacement_index = rng.randrange(bucket.population)
    if replacement_index < bucket.target:
        bucket.sample[replacement_index] = item


def score_in_random_range(
    score: float, logit_min: float | None, logit_max: float | None
) -> bool:
    """Return whether a score should be considered by random sampling."""

    if logit_min is not None and score < logit_min:
        return False
    if logit_max is not None and score > logit_max:
        return False
    return True


def find_bin(score: float, bins: list[BinSpec]) -> BinSpec | None:
    """Return the first configured bin containing ``score``."""

    for index, bin_spec in enumerate(bins):
        if bin_spec.contains(score, include_upper=index == len(bins) - 1):
            return bin_spec
    return None


def require_columns(
    fieldnames: list[str] | None, required: set[str], csv_path: Path
) -> None:
    """Exit when a CSV lacks columns required by this script."""

    existing = set(fieldnames or [])
    missing = sorted(required - existing)
    if missing:
        raise SystemExit(
            f"{csv_path} is missing required column(s): {', '.join(missing)}"
        )


def sample_inference(
    *,
    inference_csv: Path,
    spectrogram_dir: Path,
    mode: str,
    score_column: str,
    filename_column: str,
    sample_size: int,
    logit_min: float | None,
    logit_max: float | None,
    bins: list[BinSpec],
    label_filter: str | None,
    seed: int,
    shuffle_items: bool,
) -> SampleResult:
    """Stream the inference CSV and return a deterministic sample."""

    if sample_size < 0:
        raise SystemExit("--sample-size must be >= 0")

    rng = random.Random(seed)
    stats = LoadStats()
    bucket_map: dict[str, Bucket]
    if mode == "random":
        bucket_map = {"random": Bucket(target=sample_size)}
    else:
        bucket_map = {
            bin_spec.name: Bucket(target=bin_spec.target) for bin_spec in bins
        }

    required_columns = {score_column, filename_column}
    if label_filter is not None:
        required_columns.add("label")

    with inference_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, required_columns, inference_csv)

        for line_number, row in enumerate(reader, start=2):
            stats.total_rows += 1
            if label_filter is not None and row.get("label", "") != label_filter:
                continue

            filename = (row.get(filename_column) or "").strip()
            if not filename:
                stats.missing_filename_rows += 1
                continue

            try:
                score = float((row.get(score_column) or "").strip())
            except ValueError:
                stats.missing_score_rows += 1
                continue

            stats.valid_score_rows += 1
            stats.update_score_range(score)

            if mode == "random":
                if not score_in_random_range(score, logit_min, logit_max):
                    continue
                bucket = bucket_map["random"]
                item = row_to_item(
                    row=row,
                    line_number=line_number,
                    filename_column=filename_column,
                    score_column=score_column,
                    score=score,
                    stratum="random",
                )
                add_to_reservoir(bucket, item, rng)
                stats.candidate_rows += 1
                continue

            bin_spec = find_bin(score, bins)
            if bin_spec is None:
                continue
            bucket = bucket_map[bin_spec.name]
            item = row_to_item(
                row=row,
                line_number=line_number,
                filename_column=filename_column,
                score_column=score_column,
                score=score,
                stratum=bin_spec.name,
            )
            add_to_reservoir(bucket, item, rng)
            stats.candidate_rows += 1

    items: list[InferenceItem] = []
    bucket_summaries: list[dict[str, Any]] = []
    for name, bucket in bucket_map.items():
        sampled_count = len(bucket.sample)
        weight = bucket.population / sampled_count if sampled_count else 0.0
        for item in bucket.sample:
            item.stratum_population = bucket.population
            item.stratum_sampled = sampled_count
            item.weight = weight
            item.image_exists = (spectrogram_dir / item.png_filename).is_file()
            if not item.image_exists:
                stats.missing_images += 1
        bucket_summaries.append(
            {
                "name": name,
                "population": bucket.population,
                "sampled": sampled_count,
                "target": bucket.target,
                "weight": weight,
            }
        )
        items.extend(bucket.sample)

    if shuffle_items:
        rng.shuffle(items)

    return SampleResult(items=items, stats=stats, bucket_summaries=bucket_summaries)


def load_annotations(path: Path) -> dict[str, dict[str, str]]:
    """Load existing annotations from disk, keyed by row key."""

    if not path.is_file():
        return {}

    annotations: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "row_key" not in reader.fieldnames:
            return {}
        for row in reader:
            row_key = (row.get("row_key") or "").strip()
            annotation = (row.get("annotation") or "").strip().lower()
            if row_key and annotation in ANNOTATION_VALUES:
                annotations[row_key] = {key: value or "" for key, value in row.items()}
    return annotations


def save_annotations(path: Path, annotations: dict[str, dict[str, str]]) -> None:
    """Write annotations to disk atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    rows = sorted(
        annotations.values(),
        key=lambda row: (row.get("filename", ""), row.get("row_key", "")),
    )
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in ANNOTATION_FIELDNAMES}
            )
    temporary_path.replace(path)


def annotation_record(
    *,
    item: InferenceItem,
    annotation: str,
    inference_csv: Path,
) -> dict[str, str]:
    """Build a persistable CSV annotation record."""

    return {
        "row_key": item.row_key,
        "annotation": annotation,
        "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filename": item.filename,
        "png_filename": item.png_filename,
        "score": f"{item.score:.10g}",
        "label": item.metadata.get("label", ""),
        "source_idx": item.metadata.get("idx", ""),
        "source_line": str(item.line_number),
        "window_start": item.metadata.get("window_start", ""),
        "window_end": item.metadata.get("window_end", ""),
        "stratum": item.stratum,
        "weight": f"{item.weight:.10g}",
        "inference_csv": str(inference_csv),
    }


def annotations_for_items(
    items_by_key: dict[str, InferenceItem],
    annotations: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Return browser annotations for the currently sampled items."""

    return {
        row_key: row.get("annotation", "")
        for row_key, row in annotations.items()
        if row_key in items_by_key
    }


def state_payload(state: AppState) -> dict[str, Any]:
    """Serialize the current server state for the browser."""

    items_by_key = state.items_by_key
    with state.lock:
        annotations = annotations_for_items(items_by_key, state.annotations)

    scores = [item.score for item in state.items]
    sample_min = min(scores) if scores else None
    sample_max = max(scores) if scores else None

    return {
        "species_slug": state.species_slug,
        "inference_name": state.inference_name,
        "sampling_mode": state.sampling_mode,
        "threshold": state.threshold,
        "annotations_path": str(state.annotations_path),
        "inference_csv": str(state.inference_csv),
        "spectrogram_dir": str(state.spectrogram_dir),
        "stats": {
            "total_rows": state.stats.total_rows,
            "valid_score_rows": state.stats.valid_score_rows,
            "missing_score_rows": state.stats.missing_score_rows,
            "missing_filename_rows": state.stats.missing_filename_rows,
            "candidate_rows": state.stats.candidate_rows,
            "missing_images": state.stats.missing_images,
            "score_min": state.stats.score_min,
            "score_max": state.stats.score_max,
            "sample_min": sample_min,
            "sample_max": sample_max,
        },
        "buckets": state.bucket_summaries,
        "annotations": annotations,
        "items": [item_payload(item) for item in state.items],
    }


def item_payload(item: InferenceItem) -> dict[str, Any]:
    """Serialize one item for browser rendering."""

    metadata = item.metadata
    return {
        "row_key": item.row_key,
        "filename": item.filename,
        "png_filename": item.png_filename,
        "image_exists": item.image_exists,
        "score": item.score,
        "stratum": item.stratum,
        "weight": item.weight,
        "line_number": item.line_number,
        "meta": {
            "idx": metadata.get("idx", ""),
            "project": metadata.get("project", ""),
            "label": metadata.get("label", ""),
            "window_start": metadata.get("window_start", ""),
            "window_end": metadata.get("window_end", ""),
        },
    }


def json_bytes(payload: Any) -> bytes:
    """Serialize JSON using compact separators."""

    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


class EvaluationHandler(BaseHTTPRequestHandler):
    """HTTP handler for the local annotation app."""

    server: "EvaluationHTTPServer"

    def log_message(self, format_string: str, *args: Any) -> None:
        """Suppress request logs unless verbose mode is enabled."""

        if self.server.state.verbose:
            super().log_message(format_string, *args)

    def do_GET(self) -> None:
        """Handle browser and image requests."""

        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.send_html(build_html())
            return
        if path == "/api/state":
            self.send_json(state_payload(self.server.state))
            return
        if path.startswith("/spectrogram/"):
            filename = unquote(path.removeprefix("/spectrogram/"))
            self.send_spectrogram(filename)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        """Handle annotation updates."""

        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/save-annotations":
            self.save_current_annotations()
            return

        if path == "/api/load-annotations":
            self.load_saved_annotations()
            return

        if path != "/api/annotate":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            body = self.read_json_body()
            row_key = str(body.get("row_key", "")).strip()
            annotation = str(body.get("annotation", "")).strip().lower()
            payload = self.update_annotation(row_key, annotation)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except KeyError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        except OSError as exc:
            self.send_json(
                {"error": f"Could not save annotations: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self.send_json(payload)

    def save_current_annotations(self) -> None:
        """Persist the current in-memory annotations to the configured CSV."""

        state = self.server.state
        try:
            with state.lock:
                save_annotations(state.annotations_path, state.annotations)
                saved_total = len(state.annotations)
        except OSError as exc:
            self.send_json(
                {"error": f"Could not save annotations: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self.send_json(
            {
                "ok": True,
                "path": str(state.annotations_path),
                "saved_total": saved_total,
            }
        )

    def load_saved_annotations(self) -> None:
        """Load annotations from the configured CSV into memory."""

        state = self.server.state
        if not state.annotations_path.is_file():
            self.send_json(
                {"error": f"Annotation CSV not found: {state.annotations_path}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return

        try:
            loaded_annotations = load_annotations(state.annotations_path)
        except OSError as exc:
            self.send_json(
                {"error": f"Could not load annotations: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        items_by_key = state.items_by_key
        with state.lock:
            state.annotations = loaded_annotations
            sample_annotations = annotations_for_items(items_by_key, state.annotations)

        self.send_json(
            {
                "ok": True,
                "path": str(state.annotations_path),
                "loaded_total": len(loaded_annotations),
                "loaded_sample": len(sample_annotations),
                "annotations": sample_annotations,
            }
        )

    def read_json_body(self) -> dict[str, Any]:
        """Read and decode a JSON request body."""

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Request body is empty")
        if length > 20_000:
            raise ValueError("Request body is too large")
        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request JSON must be an object")
        return payload

    def update_annotation(self, row_key: str, annotation: str) -> dict[str, Any]:
        """Apply one annotation update and persist it to disk."""

        if annotation not in ANNOTATION_VALUES and annotation != "clear":
            raise ValueError("annotation must be positive, negative, or clear")

        state = self.server.state
        items_by_key = state.items_by_key
        if row_key not in items_by_key:
            raise KeyError(f"Unknown row_key: {row_key}")

        with state.lock:
            if annotation == "clear":
                state.annotations.pop(row_key, None)
            else:
                state.annotations[row_key] = annotation_record(
                    item=items_by_key[row_key],
                    annotation=annotation,
                    inference_csv=state.inference_csv,
                )
            save_annotations(state.annotations_path, state.annotations)

        return {
            "ok": True,
            "row_key": row_key,
            "annotation": "" if annotation == "clear" else annotation,
        }

    def send_spectrogram(self, filename: str) -> None:
        """Serve one PNG from the configured spectrogram directory."""

        safe_name = Path(filename).name
        if not safe_name or safe_name != filename:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid filename")
            return

        image_path = self.server.state.spectrogram_dir / safe_name
        if not image_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return

        content_type = (
            mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(image_path.stat().st_size))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        with image_path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def send_html(self, text: str) -> None:
        """Send an HTML response."""

        payload = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Send a JSON response."""

        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class EvaluationHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server carrying app state."""

    state: AppState


def build_html() -> str:
    """Return the annotation UI HTML."""

    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Inference Evaluation</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #637083;
      --border: #d8dde6;
      --positive: #147a4b;
      --positive-bg: #e5f5ed;
      --negative: #a12b2b;
      --negative-bg: #fae8e8;
      --focus: #285ea8;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      font-size: 14px;
    }

    header {
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(8px);
    }

    .topbar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      gap: 16px;
      align-items: center;
      padding: 14px 18px 12px;
    }

    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }

    .subtitle {
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }

    .metrics {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }

    .metric {
      min-width: 96px;
      padding: 8px 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
    }

    .metric strong {
      display: block;
      font-size: 17px;
      line-height: 1.1;
    }

    .metric span {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
    }

    .controls {
      display: grid;
      grid-template-columns: minmax(220px, 420px) minmax(160px, 1fr) auto;
      gap: 14px;
      align-items: center;
      padding: 0 18px 14px;
    }

    .threshold-control {
      display: grid;
      grid-template-columns: 1fr 92px;
      gap: 8px;
      align-items: center;
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--focus);
    }

    input[type="number"], select {
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 6px 8px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }

    main {
      padding: 18px;
    }

    .details {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      margin-bottom: 14px;
      color: var(--muted);
      font-size: 12px;
    }

    .details div {
      overflow-wrap: anywhere;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 12px;
      align-items: start;
    }

    .card {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }

    .card.positive {
      border-color: rgba(20, 122, 75, 0.7);
      box-shadow: 0 0 0 2px rgba(20, 122, 75, 0.12);
    }

    .card.negative {
      border-color: rgba(161, 43, 43, 0.7);
      box-shadow: 0 0 0 2px rgba(161, 43, 43, 0.12);
    }

    .image-wrap {
      min-height: 144px;
      background: #edf0f4;
      display: flex;
      align-items: center;
      justify-content: center;
      border-bottom: 1px solid var(--border);
    }

    .image-wrap img {
      display: block;
      width: 100%;
      height: auto;
    }

    .missing {
      padding: 22px;
      color: var(--muted);
      text-align: center;
      font-size: 12px;
    }

    .body {
      padding: 10px;
    }

    .filename {
      font-weight: 700;
      overflow-wrap: anywhere;
      line-height: 1.25;
    }

    .meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 5px 10px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }

    .meta span {
      overflow-wrap: anywhere;
    }

    .buttons {
      display: grid;
      grid-template-columns: 1fr 1fr 40px;
      gap: 7px;
      margin-top: 10px;
    }

    button {
      min-height: 34px;
      border: 1px solid var(--border);
      border-radius: 7px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    button:hover {
      border-color: var(--focus);
    }

    button[data-value="positive"].selected {
      border-color: var(--positive);
      background: var(--positive-bg);
      color: var(--positive);
    }

    button[data-value="negative"].selected {
      border-color: var(--negative);
      background: var(--negative-bg);
      color: var(--negative);
    }

    button.clear {
      color: var(--muted);
      font-weight: 600;
    }

    .annotation-actions {
      display: grid;
      grid-template-columns: 72px 72px;
      gap: 7px;
      align-items: center;
      justify-content: end;
    }

    .status {
      grid-column: 1 / -1;
      min-height: 16px;
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }

    .error {
      display: none;
      margin: 0 18px 12px;
      padding: 8px 10px;
      border: 1px solid #c23b3b;
      border-radius: 8px;
      background: #fff0f0;
      color: #8b1c1c;
    }

    @media (max-width: 760px) {
      .topbar,
      .controls {
        grid-template-columns: 1fr;
      }

      .metrics {
        justify-content: stretch;
      }

      .metric {
        flex: 1 1 120px;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <h1>Inference Evaluation</h1>
        <div class="subtitle" id="subtitle">Loading...</div>
      </div>
      <div class="metrics">
        <div class="metric"><strong id="precision">-</strong><span>Precision</span></div>
        <div class="metric"><strong id="weightedPrecision">-</strong><span>Weighted precision</span></div>
        <div class="metric"><strong id="predicted">0</strong><span>Predicted positive</span></div>
        <div class="metric"><strong id="positiveCount">0</strong><span>Positive labels</span></div>
        <div class="metric"><strong id="negativeCount">0</strong><span>Negative labels</span></div>
        <div class="metric"><strong id="remainingCount">0</strong><span>Remaining</span></div>
      </div>
    </div>
    <div class="controls">
      <label class="threshold-control">
        <input id="thresholdRange" type="range" step="0.001">
        <input id="thresholdNumber" type="number" step="0.001">
      </label>
      <select id="displayFilter" aria-label="Display filter">
        <option value="all">Show all sampled rows</option>
        <option value="unannotated">Show unannotated rows</option>
        <option value="annotated">Show annotated rows</option>
        <option value="predicted">Show sampled rows at or above threshold</option>
      </select>
      <div class="annotation-actions">
        <button id="saveAnnotations" type="button">Save</button>
        <button id="loadAnnotations" type="button">Load</button>
        <div class="status" id="annotationStatus"></div>
      </div>
    </div>
    <div class="error" id="errorBox"></div>
  </header>
  <main>
    <section class="details" id="details"></section>
    <section class="grid" id="grid"></section>
  </main>
  <script>
    const state = {
      items: [],
      annotations: {},
      metadata: {},
      threshold: 0
    };

    const els = {
      subtitle: document.getElementById("subtitle"),
      precision: document.getElementById("precision"),
      weightedPrecision: document.getElementById("weightedPrecision"),
      predicted: document.getElementById("predicted"),
      positiveCount: document.getElementById("positiveCount"),
      negativeCount: document.getElementById("negativeCount"),
      remainingCount: document.getElementById("remainingCount"),
      thresholdRange: document.getElementById("thresholdRange"),
      thresholdNumber: document.getElementById("thresholdNumber"),
      displayFilter: document.getElementById("displayFilter"),
      saveAnnotations: document.getElementById("saveAnnotations"),
      loadAnnotations: document.getElementById("loadAnnotations"),
      annotationStatus: document.getElementById("annotationStatus"),
      details: document.getElementById("details"),
      grid: document.getElementById("grid"),
      errorBox: document.getElementById("errorBox")
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function scoreText(value) {
      if (!Number.isFinite(value)) return "-";
      return value.toFixed(4);
    }

    function percentText(value) {
      if (!Number.isFinite(value)) return "-";
      return `${(value * 100).toFixed(1)}%`;
    }

    function setError(message) {
      els.errorBox.style.display = message ? "block" : "none";
      els.errorBox.textContent = message || "";
    }

    function setStatus(message) {
      els.annotationStatus.textContent = message || "";
    }

    async function loadState() {
      const response = await fetch("/api/state");
      if (!response.ok) {
        throw new Error(`Could not load state: ${response.status}`);
      }
      const payload = await response.json();
      state.items = payload.items;
      state.annotations = payload.annotations || {};
      state.metadata = payload;
      state.threshold = payload.threshold;
      configureThreshold(payload);
      renderHeader(payload);
      renderGrid();
      updateMetrics();
    }

    function configureThreshold(payload) {
      const stats = payload.stats || {};
      const minScore = Number.isFinite(stats.score_min) ? stats.score_min : -1;
      const maxScore = Number.isFinite(stats.score_max) ? stats.score_max : 1;
      const rangeMin = Math.min(minScore, payload.threshold);
      const rangeMax = Math.max(maxScore, payload.threshold);
      const step = Math.max((rangeMax - rangeMin) / 1000, 0.001);
      els.thresholdRange.min = String(rangeMin);
      els.thresholdRange.max = String(rangeMax);
      els.thresholdRange.step = String(step);
      els.thresholdNumber.min = String(rangeMin);
      els.thresholdNumber.max = String(rangeMax);
      els.thresholdNumber.step = String(step);
      els.thresholdRange.value = String(payload.threshold);
      els.thresholdNumber.value = String(payload.threshold);
    }

    function renderHeader(payload) {
      els.subtitle.textContent =
        `${payload.species_slug} / ${payload.inference_name} ` +
        `(${payload.sampling_mode}, ${payload.items.length} sampled)`;

      const stats = payload.stats || {};
      const details = [
        ["Inference CSV", payload.inference_csv],
        ["Spectrograms", payload.spectrogram_dir],
        ["Annotations", payload.annotations_path],
        ["Rows", `${stats.total_rows} total, ${stats.candidate_rows} candidates`],
        ["Score range", `${scoreText(stats.score_min)} to ${scoreText(stats.score_max)}`],
        ["Missing sampled images", stats.missing_images]
      ];

      els.details.innerHTML = details
        .map(([label, value]) => `<div><strong>${escapeHtml(label)}:</strong> ${escapeHtml(value)}</div>`)
        .join("");
    }

    function itemAnnotation(item) {
      return state.annotations[item.row_key] || "";
    }

    function renderGrid() {
      const filter = els.displayFilter.value;
      const threshold = Number(els.thresholdNumber.value);
      const cards = state.items
        .filter((item) => {
          const annotation = itemAnnotation(item);
          if (filter === "annotated") return Boolean(annotation);
          if (filter === "unannotated") return !annotation;
          if (filter === "predicted") return item.score >= threshold;
          return true;
        })
        .map(cardHtml)
        .join("");
      els.grid.innerHTML = cards || `<div class="missing">No rows match this view.</div>`;
    }

    function cardHtml(item) {
      const annotation = itemAnnotation(item);
      const image = item.image_exists
        ? `<img loading="lazy" src="/spectrogram/${encodeURIComponent(item.png_filename)}" alt="">`
        : `<div class="missing">Missing image<br>${escapeHtml(item.png_filename)}</div>`;
      const meta = item.meta || {};
      const windowText = [meta.window_start, meta.window_end].filter(Boolean).join(" - ");
      return `
        <article class="card ${escapeHtml(annotation)}" data-row-key="${escapeHtml(item.row_key)}">
          <div class="image-wrap">${image}</div>
          <div class="body">
            <div class="filename">${escapeHtml(item.filename)}</div>
            <div class="meta">
              <span><strong>Score</strong> ${scoreText(item.score)}</span>
              <span><strong>Stratum</strong> ${escapeHtml(item.stratum)}</span>
              <span><strong>Label</strong> ${escapeHtml(meta.label || "-")}</span>
              <span><strong>Window</strong> ${escapeHtml(windowText || "-")}</span>
              <span><strong>Idx</strong> ${escapeHtml(meta.idx || "-")}</span>
              <span><strong>Weight</strong> ${scoreText(item.weight)}</span>
            </div>
            <div class="buttons">
              <button type="button" data-row-key="${escapeHtml(item.row_key)}" data-value="positive"
                class="${annotation === "positive" ? "selected" : ""}">Positive</button>
              <button type="button" data-row-key="${escapeHtml(item.row_key)}" data-value="negative"
                class="${annotation === "negative" ? "selected" : ""}">Negative</button>
              <button type="button" data-row-key="${escapeHtml(item.row_key)}" data-value="clear"
                class="clear">Clear</button>
            </div>
          </div>
        </article>`;
    }

    async function annotate(rowKey, value) {
      setError("");
      const response = await fetch("/api/annotate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({row_key: rowKey, annotation: value})
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `Annotation failed: ${response.status}`);
      }
      if (payload.annotation) {
        state.annotations[rowKey] = payload.annotation;
      } else {
        delete state.annotations[rowKey];
      }
      updateCard(rowKey);
      updateMetrics();
      if (els.displayFilter.value !== "all") {
        renderGrid();
      }
    }

    async function saveAnnotations() {
      setError("");
      setStatus("Saving...");
      const response = await fetch("/api/save-annotations", {method: "POST"});
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `Save failed: ${response.status}`);
      }
      setStatus(`Saved ${payload.saved_total} annotations`);
    }

    async function loadAnnotations() {
      setError("");
      setStatus("Loading...");
      const response = await fetch("/api/load-annotations", {method: "POST"});
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `Load failed: ${response.status}`);
      }
      state.annotations = payload.annotations || {};
      renderGrid();
      updateMetrics();
      setStatus(`Loaded ${payload.loaded_sample} annotations for this sample`);
    }

    function updateCard(rowKey) {
      const card = els.grid.querySelector(`[data-row-key="${CSS.escape(rowKey)}"]`);
      if (!card) return;
      const annotation = state.annotations[rowKey] || "";
      card.classList.remove("positive", "negative");
      if (annotation) card.classList.add(annotation);
      card.querySelectorAll("button[data-value]").forEach((button) => {
        const value = button.getAttribute("data-value");
        button.classList.toggle("selected", value === annotation);
      });
    }

    function updateMetrics() {
      const threshold = Number(els.thresholdNumber.value);
      let positive = 0;
      let negative = 0;
      let predicted = 0;
      let truePositive = 0;
      let weightedPredicted = 0;
      let weightedTruePositive = 0;

      for (const item of state.items) {
        const annotation = itemAnnotation(item);
        if (annotation === "positive") positive += 1;
        if (annotation === "negative") negative += 1;
        if (!annotation || item.score < threshold) continue;
        predicted += 1;
        weightedPredicted += item.weight || 1;
        if (annotation === "positive") {
          truePositive += 1;
          weightedTruePositive += item.weight || 1;
        }
      }

      const annotated = positive + negative;
      els.positiveCount.textContent = String(positive);
      els.negativeCount.textContent = String(negative);
      els.remainingCount.textContent = String(state.items.length - annotated);
      els.predicted.textContent = String(predicted);
      els.precision.textContent = predicted ? percentText(truePositive / predicted) : "-";
      els.weightedPrecision.textContent = weightedPredicted
        ? percentText(weightedTruePositive / weightedPredicted)
        : "-";
    }

    els.thresholdRange.addEventListener("input", () => {
      els.thresholdNumber.value = els.thresholdRange.value;
      updateMetrics();
      if (els.displayFilter.value === "predicted") renderGrid();
    });

    els.thresholdNumber.addEventListener("input", () => {
      els.thresholdRange.value = els.thresholdNumber.value;
      updateMetrics();
      if (els.displayFilter.value === "predicted") renderGrid();
    });

    els.displayFilter.addEventListener("change", renderGrid);

    els.saveAnnotations.addEventListener("click", () => {
      saveAnnotations().catch((error) => {
        setStatus("");
        setError(error.message);
      });
    });

    els.loadAnnotations.addEventListener("click", () => {
      loadAnnotations().catch((error) => {
        setStatus("");
        setError(error.message);
      });
    });

    els.grid.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-row-key]");
      if (!button) return;
      annotate(button.dataset.rowKey, button.dataset.value).catch((error) => {
        setError(error.message);
      });
    });

    loadState().catch((error) => {
      setError(error.message);
    });
  </script>
</body>
</html>
"""


def print_startup_summary(state: AppState, url: str) -> None:
    """Print useful run information before serving forever."""

    print("Inference evaluation server")
    print(f"  URL: {url}")
    print(f"  Inference CSV: {state.inference_csv}")
    print(f"  Spectrogram dir: {state.spectrogram_dir}")
    print(f"  Annotation CSV: {state.annotations_path}")
    print(f"  Sampled rows: {len(state.items)}")
    print(f"  Candidate rows: {state.stats.candidate_rows}")
    print(f"  Missing sampled images: {state.stats.missing_images}")
    print("  Buckets:")
    for bucket in state.bucket_summaries:
        print(
            "    "
            f"{bucket['name']}: sampled {bucket['sampled']} / "
            f"{bucket['population']} candidates"
        )
    print("Press Ctrl+C to stop.")


def main() -> None:
    """Run the annotation server."""

    args = parse_args()
    inference_csv, spectrogram_dir, annotations_path = resolve_inputs(args)
    require_readable_file(inference_csv, "Inference CSV")
    require_directory(spectrogram_dir, "Spectrogram directory")

    bins = (
        parse_bins(args.bins, args.samples_per_bin)
        if args.mode == "stratified"
        else []
    )
    result = sample_inference(
        inference_csv=inference_csv,
        spectrogram_dir=spectrogram_dir,
        mode=args.mode,
        score_column=args.score_column,
        filename_column=args.filename_column,
        sample_size=args.sample_size,
        logit_min=args.logit_min,
        logit_max=args.logit_max,
        bins=bins,
        label_filter=args.label,
        seed=args.seed,
        shuffle_items=not args.no_shuffle,
    )
    if not result.items:
        raise SystemExit(
            "No rows were sampled. Check --mode, --logit-min/--logit-max, "
            "--bins, or --label."
        )

    annotations = load_annotations(annotations_path)
    state = AppState(
        items=result.items,
        stats=result.stats,
        bucket_summaries=result.bucket_summaries,
        annotations=annotations,
        annotations_path=annotations_path,
        inference_csv=inference_csv,
        spectrogram_dir=spectrogram_dir,
        species_slug=args.species_slug,
        inference_name=args.inference_name,
        sampling_mode=args.mode,
        threshold=args.threshold,
        verbose=args.verbose,
    )

    server = EvaluationHTTPServer((args.host, args.port), EvaluationHandler)
    server.state = state
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print_startup_summary(state, url)

    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
