#!/usr/bin/env python3
"""
Remove rows from a target CSV when the filename appears in a reference CSV.

Both CSV files must contain a "filename" column (case-insensitive matching is
supported via a flag). The script writes a new trimmed CSV and prints a summary
of how many rows were removed.

Usage:
    python xc_scripts/trim_target_by_reference.py \
        --target "/Volumes/Z Slim/zslim_birdcluster/agile_inferences/chloris_chloris/sweeps/inference.csv" \
        --reference "/Volumes/Z Slim/zslim_birdcluster/agile_inferences/chloris_chloris/upsweep/inference.csv"

    python xc_scripts/trim_target_by_reference.py \
        --target /path/to/target.csv \
        --reference /path/to/reference.csv \
        --output /path/to/target_trimmed.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class TrimStats:
    """Summary statistics for a trim operation."""

    reference_rows: int
    reference_unique: int
    reference_missing: int
    target_rows: int
    target_missing: int
    removed_rows: int
    kept_rows: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Remove target CSV rows when filename also appears in reference CSV."
        )
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Path to the target CSV (rows removed when filename is shared).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to the reference CSV containing filenames to remove.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional output path for the trimmed target CSV. Defaults to "
            "<target_stem>_trimmed.csv alongside the target."
        ),
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding to use when reading and writing CSV files.",
    )
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        help="Match filenames case-insensitively.",
    )
    parser.add_argument(
        "--basename-only",
        action="store_true",
        help="Match filenames using only the basename (strip directory paths).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file if it already exists.",
    )
    return parser.parse_args(argv)


def default_output_path(target_path: Path) -> Path:
    """Build the default output path for a trimmed target CSV."""

    suffix = target_path.suffix or ".csv"
    return target_path.with_name(f"{target_path.stem}_trimmed{suffix}")


def normalize_filename(
    value: object, *, case_insensitive: bool, basename_only: bool
) -> str:
    """Normalize a filename for matching."""

    text = str(value).strip()
    if not text:
        return ""
    if basename_only:
        text = Path(text).name
    if case_insensitive:
        text = text.lower()
    return text


def resolve_filename_column(fieldnames: Optional[Iterable[str]]) -> str:
    """Find the filename column name in a CSV header."""

    if not fieldnames:
        raise SystemExit("CSV is missing a header row with column names.")
    lower_map: dict[str, str] = {}
    for name in fieldnames:
        if name is None:
            continue
        key = name.strip().lower()
        if key and key not in lower_map:
            lower_map[key] = name
    if "filename" not in lower_map:
        raise SystemExit("CSV must include a 'filename' column.")
    return lower_map["filename"]


def load_reference_filenames(
    reference_path: Path,
    *,
    encoding: str,
    case_insensitive: bool,
    basename_only: bool,
) -> tuple[set[str], int, int]:
    """Load filenames from the reference CSV."""

    if not reference_path.exists():
        raise SystemExit(f"Reference CSV not found: {reference_path}")
    with reference_path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        filename_col = resolve_filename_column(reader.fieldnames)
        filenames: set[str] = set()
        total_rows = 0
        missing = 0
        for row in reader:
            total_rows += 1
            normalized = normalize_filename(
                row.get(filename_col, ""),
                case_insensitive=case_insensitive,
                basename_only=basename_only,
            )
            if normalized:
                filenames.add(normalized)
            else:
                missing += 1
    return filenames, total_rows, missing


def trim_target_csv(
    target_path: Path,
    output_path: Path,
    reference_filenames: set[str],
    *,
    encoding: str,
    case_insensitive: bool,
    basename_only: bool,
    overwrite: bool,
) -> tuple[int, int, int, int]:
    """Write the trimmed target CSV and return row statistics."""

    if not target_path.exists():
        raise SystemExit(f"Target CSV not found: {target_path}")
    if output_path.exists() and not overwrite:
        raise SystemExit(
            f"Output already exists: {output_path}. Use --overwrite to replace."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_rows = 0
    missing = 0
    removed = 0
    kept = 0
    with target_path.open("r", encoding=encoding, newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        if not reader.fieldnames:
            raise SystemExit("Target CSV is missing a header row.")
        filename_col = resolve_filename_column(reader.fieldnames)
        with output_path.open("w", encoding=encoding, newline="") as sink_handle:
            writer = csv.DictWriter(sink_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                target_rows += 1
                normalized = normalize_filename(
                    row.get(filename_col, ""),
                    case_insensitive=case_insensitive,
                    basename_only=basename_only,
                )
                if not normalized:
                    missing += 1
                    writer.writerow(row)
                    kept += 1
                    continue
                if normalized in reference_filenames:
                    removed += 1
                    continue
                writer.writerow(row)
                kept += 1
    return target_rows, missing, removed, kept


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the trim workflow."""

    args = parse_args(argv)
    target_path: Path = args.target
    reference_path: Path = args.reference
    output_path: Path = args.output or default_output_path(target_path)

    reference_filenames, reference_rows, reference_missing = load_reference_filenames(
        reference_path,
        encoding=args.encoding,
        case_insensitive=args.case_insensitive,
        basename_only=args.basename_only,
    )
    target_rows, target_missing, removed, kept = trim_target_csv(
        target_path,
        output_path,
        reference_filenames,
        encoding=args.encoding,
        case_insensitive=args.case_insensitive,
        basename_only=args.basename_only,
        overwrite=args.overwrite,
    )

    stats = TrimStats(
        reference_rows=reference_rows,
        reference_unique=len(reference_filenames),
        reference_missing=reference_missing,
        target_rows=target_rows,
        target_missing=target_missing,
        removed_rows=removed,
        kept_rows=kept,
    )

    print("Trim complete.")
    print(f"Reference rows: {stats.reference_rows}")
    print(f"Reference unique filenames: {stats.reference_unique}")
    print(f"Reference rows missing filename: {stats.reference_missing}")
    print(f"Target rows: {stats.target_rows}")
    print(f"Target rows missing filename: {stats.target_missing}")
    print(f"Target rows removed: {stats.removed_rows}")
    print(f"Target rows kept: {stats.kept_rows}")
    print(f"Wrote trimmed CSV: {output_path}")


if __name__ == "__main__":
    main()
