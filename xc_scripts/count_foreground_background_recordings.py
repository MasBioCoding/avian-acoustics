#!/usr/bin/env python3
"""
Count Xeno-Canto foreground and background recordings for a species list.

The default run matches this repository request:
- species:
  - Emberiza citrinella
  - Emberiza calandra
  - Phylloscopus collybita
  - Phylloscopus trochilus
  - Prunella modularis
- date interval: [2014-01-01, 2026-01-01)
- bounding box: lat 30..82, lon -35..45
- recording length: 5..200 seconds

Foreground counts are recordings where the target species is the main species.
Background counts are recordings where the target species is listed in `also`
and is not the main species.

Usage:
    python xc_scripts/count_foreground_background_recordings.py
    python xc_scripts/count_foreground_background_recordings.py --out /Users/masjansma/Desktop/scriptie/results/filter_logic/xc_counts.csv
    python xc_scripts/count_foreground_background_recordings.py \
        --species "Emberiza citrinella" \
        --species "Prunella modularis"
"""

from __future__ import annotations

import argparse
import calendar
import csv
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import requests


DEFAULT_SPECIES: tuple[str, ...] = (
    "Emberiza citrinella",
    "Emberiza calandra",
    "Phylloscopus collybita",
    "Phylloscopus trochilus",
    "Prunella modularis",
)

DEFAULT_START_DATE = "2014-01-01"
DEFAULT_END_DATE = "2026-01-01"
DEFAULT_API_KEY_ENV_VAR = "XENO_CANTO_API_KEY"
DEFAULT_BASE_URL = "https://xeno-canto.org/api/3/recordings"
DEFAULT_MIN_LENGTH = 5.0
DEFAULT_MAX_LENGTH = 200.0


@dataclass(frozen=True)
class BoundingBox:
    """Geographic bounding box for Xeno-Canto box queries."""

    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float

    def to_query(self) -> str:
        """Return the Xeno-Canto box query fragment."""
        return (
            f"box:{self.lat_min},"
            f"{self.lon_min},"
            f"{self.lat_max},"
            f"{self.lon_max}"
        )


@dataclass(frozen=True)
class SpeciesCount:
    """Foreground/background count summary for one species."""

    scientific_name: str
    foreground_count: int
    background_count: int
    unique_recordists: int


class XenoCantoClient:
    """Minimal Xeno-Canto API v3 client with pagination support."""

    def __init__(
        self,
        api_key: str,
        *,
        per_page: int = 500,
        timeout: int = 30,
        pause: float = 0.05,
    ) -> None:
        self.api_key = api_key
        self.per_page = max(50, min(500, per_page))
        self.timeout = timeout
        self.pause = pause
        self.session = requests.Session()

    def search(self, query: str) -> Iterable[Dict[str, Any]]:
        """Yield recordings for a fully assembled XC query."""
        page = 1
        total_pages: Optional[int] = None

        while True:
            response = self.session.get(
                DEFAULT_BASE_URL,
                params={
                    "query": query,
                    "page": page,
                    "per_page": self.per_page,
                    "key": self.api_key,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            recordings = data.get("recordings", []) or []
            if not recordings:
                break

            if total_pages is None:
                total_pages = int(data.get("numPages", 1) or 1)
                total_records = int(data.get("numRecordings", 0) or 0)
                print(
                    f"  Query matched {total_records} recordings across "
                    f"{total_pages} pages."
                )

            yield from recordings

            if total_pages is not None and page >= total_pages:
                break

            page += 1
            time.sleep(self.pause)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Count Xeno-Canto foreground/background recordings for one or more "
            "species within a date interval and bounding box."
        )
    )
    parser.add_argument(
        "--species",
        action="append",
        dest="species",
        help=(
            "Scientific name to count. Repeat for multiple species. Defaults to "
            "the five requested species."
        ),
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Start of interval, inclusive. Default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help=f"End of interval, exclusive. Default: {DEFAULT_END_DATE}",
    )
    parser.add_argument("--latmin", type=float, default=30.0, help="Minimum latitude")
    parser.add_argument("--latmax", type=float, default=82.0, help="Maximum latitude")
    parser.add_argument("--lonmin", type=float, default=-35.0, help="Minimum longitude")
    parser.add_argument("--lonmax", type=float, default=45.0, help="Maximum longitude")
    parser.add_argument(
        "--key",
        help="Xeno-Canto API key. Defaults to the XENO_CANTO_API_KEY env var.",
    )
    parser.add_argument(
        "--api-key-env-var",
        default=DEFAULT_API_KEY_ENV_VAR,
        help=(
            "Environment variable for the XC API key. "
            f"Default: {DEFAULT_API_KEY_ENV_VAR}"
        ),
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=500,
        help="Results per page, clamped to the XC API v3 range 50-500.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.05,
        help="Pause in seconds between paginated requests.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional CSV output path for the final counts.",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=DEFAULT_MIN_LENGTH,
        help=f"Minimum recording length in seconds. Default: {DEFAULT_MIN_LENGTH}",
    )
    parser.add_argument(
        "--max-length",
        type=float,
        default=DEFAULT_MAX_LENGTH,
        help=f"Maximum recording length in seconds. Default: {DEFAULT_MAX_LENGTH}",
    )
    return parser.parse_args()


def resolve_api_key(args: argparse.Namespace) -> str:
    """Resolve the API key from args or environment."""
    if args.key:
        return args.key

    env_key = os.getenv(args.api_key_env_var)
    if env_key:
        return env_key

    raise SystemExit(
        "Xeno-Canto API key is required. Pass --key or set "
        f"{args.api_key_env_var} in your environment."
    )


def build_species_query(species: str) -> str:
    """Build a tag-based species query for XC API v3."""
    parts = species.replace("_", " ").strip().split()
    if len(parts) < 2:
        raise ValueError(
            f"Species must include at least genus and species, got: {species!r}"
        )

    query_parts = [f"gen:{parts[0]}", f"sp:{parts[1]}"]
    if len(parts) > 2:
        ssp_value = " ".join(parts[2:])
        query_parts.append(
            f'ssp:"{ssp_value}"' if " " in ssp_value else f"ssp:{ssp_value}"
        )
    return " ".join(query_parts)


def normalize_species_name(name: str) -> str:
    """Normalize species text for exact-ish scientific-name comparisons."""
    return " ".join(name.strip().lower().split())


def parse_interval_bound(value: str) -> date:
    """Parse an ISO date bound."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid ISO date {value!r}: {exc}") from exc


def parse_partial_date_bounds(value: str) -> Optional[tuple[date, date]]:
    """Return earliest/latest possible dates for partial XC date strings.

    Xeno-Canto sometimes stores dates with unknown month/day as `00`.
    This function interprets:
    - YYYY as YYYY-01-01 .. YYYY-12-31
    - YYYY-MM as first .. last day of that month
    - YYYY-MM-00 as first .. last day of that month
    - YYYY-00-00 as first .. last day of that year
    """
    text = value.strip()
    if not text:
        return None

    parts = text.split("-")
    if not parts:
        return None

    try:
        year = int(parts[0])
    except ValueError:
        return None

    month: Optional[int] = None
    day: Optional[int] = None

    if len(parts) >= 2 and parts[1] and parts[1] != "00":
        try:
            month = int(parts[1])
        except ValueError:
            return None

    if len(parts) >= 3 and parts[2] and parts[2] != "00":
        try:
            day = int(parts[2])
        except ValueError:
            return None

    try:
        earliest_month = month or 1
        earliest_day = day or 1
        earliest = date(year, earliest_month, earliest_day)

        latest_month = month or 12
        latest_day = day or calendar.monthrange(year, latest_month)[1]
        latest = date(year, latest_month, latest_day)
    except ValueError:
        return None

    return earliest, latest


def overlaps_interval(
    recorded_date: str,
    start_date: date,
    end_date: date,
) -> bool:
    """Return True when a recording date overlaps the requested interval."""
    bounds = parse_partial_date_bounds(recorded_date)
    if bounds is None:
        return False

    earliest, latest = bounds
    return earliest < end_date and latest >= start_date


def extract_also_field(record: Dict[str, Any]) -> str:
    """Normalize the XC `also` field to one lowercased string."""
    also_value = record.get("also", "")
    if isinstance(also_value, list):
        also_value = ", ".join(str(item) for item in also_value)
    return str(also_value).lower()


def normalize_recordist_name(value: Any) -> str:
    """Normalize a recordist name for uniqueness checks."""
    return " ".join(str(value).strip().lower().split())


def build_length_query(min_length: float, max_length: float) -> str:
    """Return an XC recording-length query fragment."""
    return f"len:{min_length:g}-{max_length:g}"


def count_species(
    client: XenoCantoClient,
    species: str,
    bbox: BoundingBox,
    start_date: date,
    end_date: date,
    min_length: float,
    max_length: float,
) -> SpeciesCount:
    """Count foreground and background recordings for one species."""
    species_query = build_species_query(species)
    normalized_target = normalize_species_name(species)
    bbox_query = bbox.to_query()
    length_query = build_length_query(min_length, max_length)

    foreground_ids: set[str] = set()
    background_ids: set[str] = set()
    unique_recordists: set[str] = set()

    foreground_query = f"{species_query} {bbox_query} {length_query}"
    print(f"\nCounting foreground recordings for {species}...")
    for record in client.search(foreground_query):
        xcid = str(record.get("id", "")).strip()
        if not xcid:
            continue
        if overlaps_interval(str(record.get("date", "")), start_date, end_date):
            foreground_ids.add(xcid)
            recordist = normalize_recordist_name(record.get("rec", ""))
            if recordist:
                unique_recordists.add(recordist)

    background_query = f'also:"{species}" {bbox_query} {length_query}'
    print(f"Counting background recordings for {species}...")
    for record in client.search(background_query):
        xcid = str(record.get("id", "")).strip()
        if not xcid:
            continue
        if not overlaps_interval(str(record.get("date", "")), start_date, end_date):
            continue

        main_species = normalize_species_name(
            f"{record.get('gen', '')} {record.get('sp', '')}"
        )
        if main_species == normalized_target:
            continue

        also_field = extract_also_field(record)
        if normalized_target not in also_field:
            continue

        background_ids.add(xcid)
        recordist = normalize_recordist_name(record.get("rec", ""))
        if recordist:
            unique_recordists.add(recordist)

    return SpeciesCount(
        scientific_name=species,
        foreground_count=len(foreground_ids),
        background_count=len(background_ids),
        unique_recordists=len(unique_recordists),
    )


def print_results(
    counts: Sequence[SpeciesCount],
    *,
    start_date: date,
    end_date: date,
    bbox: BoundingBox,
    min_length: float,
    max_length: float,
) -> None:
    """Print a small results table."""
    print("\n" + "=" * 84)
    print(
        "Xeno-Canto counts "
        f"for recordings in [{start_date.isoformat()}, {end_date.isoformat()}) "
        f"within box {bbox.lat_min},{bbox.lon_min},{bbox.lat_max},{bbox.lon_max} "
        f"and length {min_length:g}-{max_length:g}s"
    )
    print("=" * 84)
    print(
        f"{'Scientific name':<28}"
        f"{'Foreground':>14}"
        f"{'Background':>14}"
        f"{'Total':>14}"
        f"{'Unique recs':>14}"
    )
    print("-" * 84)

    total_foreground = 0
    total_background = 0
    for row in counts:
        total = row.foreground_count + row.background_count
        total_foreground += row.foreground_count
        total_background += row.background_count
        print(
            f"{row.scientific_name:<28}"
            f"{row.foreground_count:>14}"
            f"{row.background_count:>14}"
            f"{total:>14}"
            f"{row.unique_recordists:>14}"
        )

    print("-" * 84)
    print(
        f"{'ALL SPECIES':<28}"
        f"{total_foreground:>14}"
        f"{total_background:>14}"
        f"{(total_foreground + total_background):>14}"
        f"{'n/a':>14}"
    )


def write_csv(path: Path, counts: Sequence[SpeciesCount]) -> None:
    """Write results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scientific_name",
                "foreground_count",
                "background_count",
                "total_count",
                "unique_recordists",
            ]
        )
        for row in counts:
            writer.writerow(
                [
                    row.scientific_name,
                    row.foreground_count,
                    row.background_count,
                    row.foreground_count + row.background_count,
                    row.unique_recordists,
                ]
            )


def main() -> None:
    """Run the counting workflow."""
    args = parse_args()
    api_key = resolve_api_key(args)
    start_date = parse_interval_bound(args.start_date)
    end_date = parse_interval_bound(args.end_date)

    if start_date >= end_date:
        raise SystemExit("--start-date must be earlier than --end-date.")
    if args.min_length < 0:
        raise SystemExit("--min-length must be non-negative.")
    if args.min_length > args.max_length:
        raise SystemExit("--min-length must be less than or equal to --max-length.")

    bbox = BoundingBox(
        lat_min=args.latmin,
        lon_min=args.lonmin,
        lat_max=args.latmax,
        lon_max=args.lonmax,
    )

    species_list = tuple(args.species or DEFAULT_SPECIES)
    client = XenoCantoClient(
        api_key=api_key,
        per_page=args.per_page,
        timeout=args.timeout,
        pause=args.pause,
    )

    counts = [
        count_species(
            client=client,
            species=species,
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            min_length=args.min_length,
            max_length=args.max_length,
        )
        for species in species_list
    ]

    print_results(
        counts,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    if args.out:
        write_csv(args.out, counts)
        print(f"\nSaved CSV to: {args.out}")


if __name__ == "__main__":
    main()
