#!/usr/bin/env python3
"""
List clip filenames for metadata rows outside a bounding box and optionally
move those files into a separate folder.

This script reads a metadata.csv file containing xcid, clip_index, lat, and lon,
filters rows outside the requested bounding box, and builds clip filenames in
the format: <slug>_<xcid>_<clip_index>.wav. By default it prints the filenames
to stdout (or writes to --output). When --clips-dir is provided, it will move
the matching files into a "outside_bbox" subfolder (or --prune-dir if set).

Usage:
    python xc_scripts/filter_metadata_bbox.py --metadata /path/to/metadata.csv \
        --config xc_configs_perch/config_chloris_chloris.yaml
        
    python xc_scripts/filter_metadata_bbox.py --metadata "/Volumes/Z Slim/zslim_birdcluster/embeddings/carduelis_carduelis/metadata.csv" \
        --config xc_configs_perch/config_carduelis_carduelis.yaml \
        --clips-dir "/Volumes/Z Slim/zslim_birdcluster/clips/carduelis_carduelis"
        
    python xc_scripts/filter_metadata_bbox.py --metadata "/Volumes/Z Slim/zslim_birdcluster/embeddings/linaria_cannabina/metadata.csv" \
        --config "xc_configs_perch/config_linaria_cannabina.yaml" \
        --clips-dir "/Volumes/Z Slim/zslim_birdcluster/clips/linaria_cannabina"
        
    python xc_scripts/filter_metadata_bbox.py --metadata "/Volumes/Z Slim/zslim_birdcluster/embeddings/curruca_communis/metadata.csv" \
        --config "xc_configs_perch/config_curruca_communis.yaml" \
        --clips-dir "/Volumes/Z Slim/zslim_birdcluster/clips/curruca_communis"

    python xc_scripts/filter_metadata_bbox.py --metadata "/Volumes/Z Slim/zslim_birdcluster/embeddings/emberiza_calandra/metadata.csv" \
        --config "xc_configs_perch/config_emberiza_calandra.yaml" \
        --clips-dir "/Volumes/Z Slim/zslim_birdcluster/clips/emberiza_calandra"
        
    python xc_scripts/filter_metadata_bbox.py --metadata "/Volumes/Z Slim/zslim_birdcluster/embeddings/phylloscopus_collybita/metadata.csv" \
        --config "xc_configs_perch/config_phylloscopus_collybita.yaml" \
        --clips-dir "/Volumes/Z Slim/zslim_birdcluster/clips/phylloscopus_collybita"
        
    python xc_scripts/filter_metadata_bbox.py --metadata "/Volumes/Z Slim/zslim_birdcluster/embeddings/prunella_modularis/metadata.csv" \
        --config "xc_configs_perch/config_prunella_modularis.yaml" \
        --clips-dir "/Volumes/Z Slim/zslim_birdcluster/clips/prunella_modularis"
        
    python xc_scripts/filter_metadata_bbox.py --metadata "/Volumes/Z Slim/zslim_birdcluster/embeddings/phylloscopus_trochilus/metadata.csv" \
        --config "xc_configs_perch/config_phylloscopus_trochilus.yaml" \
        --clips-dir "/Volumes/Z Slim/zslim_birdcluster/clips/phylloscopus_trochilus"
        
        
        
        
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import yaml


@dataclass(frozen=True)
class BoundingBox:
    """Geographic bounding box in decimal degrees."""

    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

    def contains(self, lat: float, lon: float) -> bool:
        """Return True when the coordinate is inside the bounding box."""

        return (
            self.lat_min <= lat <= self.lat_max
            and self.lon_min <= lon <= self.lon_max
        )


DEFAULT_BBOX = BoundingBox(
    lat_min=30.0,
    lon_min=-35.0,
    lat_max=82.0,
    lon_max=45.0,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Output filenames for metadata rows that fall outside a bounding box."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to metadata.csv containing xcid, clip_index, lat, lon.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Config YAML used to derive species slug (species.slug).",
    )
    parser.add_argument(
        "--slug",
        help="Species slug to use in output filenames (overrides --config).",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=DEFAULT_BBOX.lat_min,
        help=f"Minimum latitude (default: {DEFAULT_BBOX.lat_min}).",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=DEFAULT_BBOX.lon_min,
        help=f"Minimum longitude (default: {DEFAULT_BBOX.lon_min}).",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=DEFAULT_BBOX.lat_max,
        help=f"Maximum latitude (default: {DEFAULT_BBOX.lat_max}).",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=DEFAULT_BBOX.lon_max,
        help=f"Maximum longitude (default: {DEFAULT_BBOX.lon_max}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write filtered filenames (one per line).",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        help=(
            "Directory containing clip files to move when pruning "
            "(optional)."
        ),
    )
    parser.add_argument(
        "--prune-dir",
        type=Path,
        help=(
            "Directory to move files into when pruning. Defaults to "
            "<clips-dir>/outside_bbox."
        ),
    )
    return parser.parse_args(argv)


def _parse_float(value: object) -> Optional[float]:
    """Parse a float from a CSV cell."""

    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_int(value: object) -> Optional[int]:
    """Parse an int from a CSV cell."""

    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _load_slug(config_path: Path) -> str:
    """Load the species slug from a config YAML."""

    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    species = config.get("species", {}) if isinstance(config, dict) else {}
    slug = species.get("slug")
    if slug:
        return str(slug).strip()
    scientific_name = species.get("scientific_name", "")
    if scientific_name:
        return str(scientific_name).lower().replace(" ", "_").strip()
    raise SystemExit(
        "Config missing species.slug and species.scientific_name; "
        "provide --slug instead."
    )


def _resolve_slug(config_path: Optional[Path], slug: Optional[str]) -> str:
    """Resolve slug from CLI input or config."""

    if slug:
        return slug.strip()
    if config_path:
        return _load_slug(config_path)
    raise SystemExit("Provide --config or --slug to build filenames.")


def _validate_bbox(bbox: BoundingBox) -> BoundingBox:
    """Validate bounding box bounds."""

    if bbox.lat_min > bbox.lat_max:
        raise SystemExit("lat_min must be <= lat_max.")
    if bbox.lon_min > bbox.lon_max:
        raise SystemExit("lon_min must be <= lon_max.")
    return bbox


def _collect_outside_bbox(
    metadata_path: Path, slug: str, bbox: BoundingBox
) -> tuple[list[str], int]:
    """Return filenames for rows outside the bounding box."""

    if not metadata_path.exists():
        raise SystemExit(f"metadata.csv not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"No header row found in {metadata_path}")

        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        required = {"xcid", "clip_index", "lat", "lon"}
        missing = sorted(required.difference(field_map.keys()))
        if missing:
            raise SystemExit(
                f"metadata.csv missing required columns: {', '.join(missing)}"
            )

        outside_files: list[str] = []
        skipped_rows = 0
        for row in reader:
            xcid = str(row.get(field_map["xcid"], "")).strip()
            clip_index = _parse_int(row.get(field_map["clip_index"]))
            lat = _parse_float(row.get(field_map["lat"]))
            lon = _parse_float(row.get(field_map["lon"]))

            if not xcid or clip_index is None:
                skipped_rows += 1
                continue

            inside = lat is not None and lon is not None and bbox.contains(lat, lon)
            if not inside:
                outside_files.append(f"{slug}_{xcid}_{clip_index:02d}.wav")

    return outside_files, skipped_rows


def _write_output(lines: list[str], output_path: Optional[Path]) -> None:
    """Write output lines to a file or stdout."""

    if output_path is None:
        for line in lines:
            print(line)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    output_path.write_text(content, encoding="utf-8")


def _prune_files(
    clip_names: list[str],
    clips_dir: Path,
    prune_dir: Optional[Path],
) -> tuple[int, int]:
    """Move clip files from clips_dir into prune_dir."""

    if not clips_dir.exists():
        raise SystemExit(f"Clips directory not found: {clips_dir}")

    target_dir = prune_dir or (clips_dir / "outside_bbox")
    target_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    missing = 0
    for name in clip_names:
        source = clips_dir / name
        if not source.exists():
            missing += 1
            continue
        destination = target_dir / name
        shutil.move(str(source), str(destination))
        moved += 1

    return moved, missing


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""

    args = parse_args(argv)
    slug = _resolve_slug(args.config, args.slug)
    bbox = _validate_bbox(
        BoundingBox(
            lat_min=args.lat_min,
            lon_min=args.lon_min,
            lat_max=args.lat_max,
            lon_max=args.lon_max,
        )
    )
    outside_files, skipped_rows = _collect_outside_bbox(
        args.metadata, slug, bbox
    )
    _write_output(outside_files, args.output)
    if args.clips_dir:
        moved, missing = _prune_files(
            outside_files, args.clips_dir, args.prune_dir
        )
        target_dir = args.prune_dir or (args.clips_dir / "outside_bbox")
        print(f"Moved {moved} files into {target_dir}.", file=sys.stderr)
        if missing:
            print(
                f"Missing {missing} files in clips directory.",
                file=sys.stderr,
            )
    if skipped_rows:
        print(
            f"Skipped {skipped_rows} rows missing xcid/clip_index.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
