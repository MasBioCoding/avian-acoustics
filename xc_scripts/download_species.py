#!/usr/bin/env python3
"""
Download bird recordings from Xeno-Canto for a specified species.
Includes both foreground (main species) and background recordings.

Uses the Xeno-Canto API v3 (https://xeno-canto.org/explore/api),
which requires an API key. Set it via the `XENO_CANTO_API_KEY`
environment variable or the `xeno_canto.api_key` config field.
to set, use: export XENO_CANTO_API_KEY="your-key-here" in your shell (terminal).
Geographic filtering is supported via `xeno_canto.bounding_box`
using Xeno-Canto's `box:lat_min,lon_min,lat_max,lon_max` query.

cd /path/to/birdnet_data_pipeline
for me: /Users/masjansma/Desktop/birdnetcluster1folder/birdnet_data_pipeline

Usage:
    python xc_scripts/download_species.py --config xc_configs/config_regulus_ignicapilla.yaml
    python xc_scripts/download_species.py --config xc_configs/config_regulus_regulus.yaml
    python xc_scripts/download_species.py --config xc_configs/config_prunella_modularis.yaml
    python xc_scripts/download_species.py --config xc_configs/config_curruca_communis.yaml
    python xc_scripts/download_species.py --config xc_configs/config_acrocephalus_scirpaceus.yaml
    python xc_scripts/download_species.py --config xc_configs/config_phylloscopus_trochilus.yaml
    python xc_scripts/download_species.py --config xc_configs_perch/config_phylloscopus_collybita.yaml
    python xc_scripts/download_species.py --config xc_configs/config_carduelis_carduelis.yaml
    python xc_scripts/download_species.py --config xc_configs/config_linaria_cannabina.yaml
    python xc_scripts/download_species.py --config xc_configs/config_emberiza_calandra.yaml
    python xc_scripts/download_species.py --config xc_configs/config_turdus_iliacus.yaml

    

    python xc_scripts/download_species.py --species "Emberiza citrinella" --max-recordings 1000
    python xc_scripts/download_species.py --config xc_configs/config_limosa_limosa.yaml
    python xc_scripts/download_species.py --config xc_configs/config_emberiza_citrinella.yaml
    python xc_scripts/download_species.py --config xc_configs/config_sylvia_atricapilla.yaml
    python xc_scripts/download_species.py --config xc_configs/config_turdus_merula.yaml
    python xc_scripts/download_species.py --config xc_configs/config_parus_major.yaml
    python xc_scripts/download_species.py --config xc_configs/config_passer_montanus.yaml
    python xc_scripts/download_species.py --config xc_configs/config_passer_domesticus.yaml
    python xc_scripts/download_species.py --config xc_configs/config_strix_aluco.yaml
    python xc_scripts/download_species.py --config xc_configs/config_asio_otus.yaml
    python xc_scripts/download_species.py --config xc_configs/config_chloris_chloris.yaml
    python xc_scripts/download_species.py --config xc_configs/config_phylloscopus_collybita.yaml
    python xc_scripts/download_species.py --config xc_configs/config_phylloscopus_trochilus.yaml
    python xc_scripts/download_species.py --config xc_configs/config_acrocephalus_scirpaceus.yaml
    python xc_scripts/download_species.py --config xc_configs/config_curruca_communis.yaml
    python xc_scripts/download_species.py --config xc_configs/config_cettia_cetti.yaml
    python xc_scripts/download_species.py --config xc_configs/config.yaml
    
    python xc_scripts/download_species.py --config xc_configs/config_fringilla_coelebs.yaml
    python xc_scripts/download_species.py --config xc_configs/config_prunella_modularis.yaml
    python xc_scripts/download_species.py --config xc_configs/config_carduelis_carduelis.yaml
    python xc_scripts/download_species.py --config xc_configs/config_emberiza_calandra.yaml
    python xc_scripts/download_species.py --config xc_configs/config_erithacus_rubecula.yaml
    python xc_scripts/download_species.py --config xc_configs/config_turdus_migratorius.yaml
    python xc_scripts/download_species.py --config xc_configs/config_turdus_philomelos.yaml
    python xc_scripts/download_species.py --config xc_configs/config_sturnus_vulgaris.yaml
    python xc_scripts/download_species.py --config xc_configs/config_troglodytes_troglodytes.yaml
    python xc_scripts/download_species.py --config xc_configs/config_regulus_ignicapilla.yaml
    python xc_scripts/download_species.py --config xc_configs/config_regulus_regulus.yaml
    python xc_scripts/download_species.py --config xc_configs/config_linaria_cannabina.yaml
    python xc_scripts/download_species.py --config xc_configs/config_turdus_iliacus.yaml

"""


import argparse
import csv
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import requests
import yaml


@dataclass
class RecordingMetadata:
    """Metadata for a single recording."""

    xcid: str
    species_sci: str
    species_common: str
    is_background: bool  # True if our species is in background
    main_species: str  # The actual main species of the recording
    date: str
    time: str
    country: str
    location: str
    lat: float
    lon: float
    alt: str
    recordist: str
    quality: str
    length: str
    file_url: str
    file_name: str
    type: str  # song, call, etc
    sex: str
    stage: str  # adult, juvenile, etc
    also: str  # other species heard
    remarks: str

    @classmethod
    def from_xc_record(
        cls, rec: Dict[str, Any], target_species: str, is_background: bool = False
    ) -> "RecordingMetadata":
        """Create metadata from a Xeno-Canto API response."""
        also_field = rec.get("also", "")
        if isinstance(also_field, list):
            also_field = ", ".join(also_field)

        return cls(
            xcid=rec.get("id", ""),
            species_sci=f"{rec.get('gen', '')} {rec.get('sp', '')}".strip(),
            species_common=rec.get("en", ""),
            is_background=is_background,
            main_species=(
                f"{rec.get('gen', '')} {rec.get('sp', '')}".strip()
                if is_background
                else target_species
            ),
            date=rec.get("date", ""),
            time=rec.get("time", ""),
            country=rec.get("cnt", ""),
            location=rec.get("loc", ""),
            lat=_safe_float(rec.get("lat")),
            lon=_safe_float(rec.get("lon", rec.get("lng"))),
            alt=rec.get("alt", ""),
            recordist=rec.get("rec", ""),
            quality=rec.get("q", ""),
            length=rec.get("length", ""),
            file_url=normalize_url(rec.get("file", "")),
            file_name=rec.get("file-name", ""),
            type=rec.get("type", ""),
            sex=rec.get("sex", ""),
            stage=rec.get("stage", ""),
            also=also_field,
            remarks=rec.get("rmk", ""),
        )


class XenoCantoCrawler:
    """Handle Xeno-Canto API interactions."""

    def __init__(
        self,
        api_key: str,
        per_page: int = 200,
        request_timeout: int = 15,
        request_pause: float = 0.5,
    ):
        self.base_url = "https://xeno-canto.org/api/3/recordings"
        self.api_key = api_key
        self.per_page = self._coerce_per_page(per_page)
        self.timeout = request_timeout
        self.pause = request_pause
        self.session = requests.Session()

    def search_foreground(
        self, species: str, extra_query: Optional[str] = None
    ) -> Iterable[Dict[str, Any]]:
        """Search for recordings where species is the main subject."""
        query_parts = [self._build_species_query(species)]
        if extra_query:
            query_parts.append(extra_query)
        query = " ".join(query_parts)

        yield from self._paginate_search(query)

    def search_background(
        self, species: str, extra_query: Optional[str] = None
    ) -> Iterable[Dict[str, Any]]:
        """Search for recordings where species appears in 'also' field."""
        query_parts = [f'also:"{species}"']
        if extra_query:
            query_parts.append(extra_query)
        query = " ".join(query_parts)

        print(f"  Background query: {query}")
        yield from self._paginate_search(query)

    def _paginate_search(self, query: str) -> Iterable[Dict[str, Any]]:
        """Paginate through search results."""
        page = 1
        total_pages = None

        while True:
            params = {
                "query": query,
                "page": page,
                "per_page": self.per_page,
                "key": self.api_key,
            }

            try:
                response = self.session.get(
                    self.base_url, params=params, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                break

            recordings = data.get("recordings", [])
            if not recordings:
                break

            # Update total pages on first request
            if total_pages is None:
                total_pages = int(data.get("numPages", 1) or 1)
                total_recs = int(data.get("numRecordings", 0) or 0)
                print(
                    f"  Found {total_recs} recordings across "
                    f"{total_pages} pages (per_page={self.per_page})"
                )

            yield from recordings

            if page >= total_pages:
                break

            page += 1
            time.sleep(self.pause)

    @staticmethod
    def _coerce_per_page(per_page: Any) -> int:
        """Clamp per_page to the API v3 supported range."""
        try:
            per_page_int = int(per_page)
        except (TypeError, ValueError):
            per_page_int = 200
        return max(50, min(500, per_page_int))

    @staticmethod
    def _build_species_query(species: str) -> str:
        """Build a tag-only query for a species (genus/species[/ssp])."""
        parts = species.replace("_", " ").strip().split()
        if not parts:
            raise ValueError("Species name must not be empty.")

        gen = parts[0]
        tokens = [f"gen:{gen}"]

        if len(parts) >= 2:
            tokens.append(f"sp:{parts[1]}")
        if len(parts) >= 3:
            ssp_value = " ".join(parts[2:])
            if " " in ssp_value:
                tokens.append(f'ssp:"{ssp_value}"')
            else:
                tokens.append(f"ssp:{ssp_value}")

        return " ".join(tokens)


class SpeciesDownloader:
    """Main downloader class."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.species = config["species"]["scientific_name"]
        self.common_name = config["species"].get("common_name", "")
        self.slug = config["species"].get("slug", self.species.lower().replace(" ", "_"))

        # Setup paths
        self.root = Path(config["paths"]["root"]).expanduser()
        self.download_dir = self.root / "xc_downloads" / self.slug
        self.download_dir.mkdir(parents=True, exist_ok=True)

        xeno_cfg = config.get("xeno_canto", {})
        api_key = self._resolve_api_key(xeno_cfg)

        self.crawler = XenoCantoCrawler(
            api_key=api_key,
            per_page=xeno_cfg.get("per_page", 200),
            request_timeout=xeno_cfg.get("timeout", 15),
            request_pause=xeno_cfg.get("pause", 0.5),
        )

    def download(self) -> None:
        """Main download process."""
        xeno_cfg = self.config["xeno_canto"]
        max_recordings = xeno_cfg.get("max_recordings", 1000)
        include_background = xeno_cfg.get("include_background", True)
        bounding_box = self._validate_bounding_box(xeno_cfg.get("bounding_box"))
        extra_query = self._build_extra_query(
            xeno_cfg.get("extra_query"), bounding_box
        )

        print(f"\n{'='*60}")
        print(f"Downloading recordings for: {self.species}")
        if self.common_name:
            print(f"Common name: {self.common_name}")
        print(f"Output directory: {self.download_dir}")
        print(f"Max recordings: {max_recordings}")
        print(f"Include background: {include_background}")
        if extra_query:
            print(f"Extra filters: {extra_query}")
        if bounding_box:
            print(
                "Bounding box (lat, lon): "
                f"{bounding_box['lat_min']} to {bounding_box['lat_max']}, "
                f"{bounding_box['lon_min']} to {bounding_box['lon_max']}"
            )
        print(f"{'='*60}\n")

        # Track what we've downloaded
        existing_ids = self._get_existing_ids()
        all_metadata = []
        downloaded_count = 0

        # Download foreground recordings
        print("Searching for foreground recordings...")
        for rec in self.crawler.search_foreground(self.species, extra_query):
            if downloaded_count >= max_recordings:
                break

            metadata = RecordingMetadata.from_xc_record(
                rec, self.species, is_background=False
            )

            if metadata.xcid in existing_ids:
                all_metadata.append(metadata)  # Still track metadata
                continue

            if self._download_file(metadata):
                all_metadata.append(metadata)
                downloaded_count += 1
                existing_ids.add(metadata.xcid)

                if downloaded_count % 10 == 0:
                    print(f"  Downloaded {downloaded_count}/{max_recordings} recordings...")

        # Download background recordings if requested
        if include_background and downloaded_count < max_recordings:
            print(f"\nSearching for background recordings (where {self.species} is also heard)...")

            for rec in self.crawler.search_background(self.species, extra_query):
                if downloaded_count >= max_recordings:
                    break

                # For background recordings, the main species is different
                metadata = RecordingMetadata.from_xc_record(
                    rec, self.species, is_background=True
                )

                # Skip if we already have this recording
                if metadata.xcid in existing_ids:
                    continue

                # Verify our species is actually in the 'also' field
                also_text = metadata.also if isinstance(metadata.also, str) else str(metadata.also)
                if not also_text or self.species.lower() not in also_text.lower():
                    continue

                if self._download_file(metadata):
                    all_metadata.append(metadata)
                    downloaded_count += 1
                    existing_ids.add(metadata.xcid)

                    if downloaded_count % 10 == 0:
                        print(f"  Downloaded {downloaded_count}/{max_recordings} recordings...")

        # Save metadata
        self._save_metadata(all_metadata)

        print(f"\n{'='*60}")
        print(f"Download complete!")
        print(f"Total recordings: {len(all_metadata)}")
        print(f"Foreground: {sum(1 for m in all_metadata if not m.is_background)}")
        print(f"Background: {sum(1 for m in all_metadata if m.is_background)}")
        print(f"Metadata saved to: {self.download_dir / 'metadata.csv'}")
        print(f"{'='*60}\n")

    def _resolve_api_key(self, xeno_cfg: Dict[str, Any]) -> str:
        """Fetch the API key from env or config."""
        env_var = xeno_cfg.get("api_key_env_var", "XENO_CANTO_API_KEY")
        env_key = os.getenv(env_var)
        if env_key:
            return env_key

        config_key = xeno_cfg.get("api_key")
        if config_key:
            return config_key

        raise SystemExit(
            "Xeno-Canto API key is required for v3. "
            f"Set the environment variable {env_var} or define "
            "`xeno_canto.api_key` in your config file."
        )

    def _get_existing_ids(self) -> Set[str]:
        """Get IDs of already downloaded recordings."""
        existing = set()

        # Check for existing audio files
        for ext in ['.mp3', '.wav', '.ogg', '.m4a']:
            for f in self.download_dir.glob(f"*{ext}"):
                # Extract ID from filename (format: slug_xcid.ext)
                parts = f.stem.split('_')
                if len(parts) >= 2:
                    xcid = parts[-1]
                    existing.add(xcid)

        return existing

    def _download_file(self, metadata: RecordingMetadata) -> bool:
        """Download a single recording."""
        try:
            # Determine file extension
            ext = Path(metadata.file_name).suffix if metadata.file_name else '.mp3'
            if not ext:
                ext = Path(metadata.file_url).suffix or '.mp3'

            # Create filename
            filename = f"{self.slug}_{metadata.xcid}{ext}"
            filepath = self.download_dir / filename

            # Skip if exists
            if filepath.exists():
                return True

            # Download with streaming
            response = requests.get(metadata.file_url, stream=True, timeout=30)
            response.raise_for_status()

            # Save to file
            with open(filepath, 'wb') as f:
                shutil.copyfileobj(response.raw, f)

            return True

        except Exception as e:
            print(f"  Error downloading XC{metadata.xcid}: {e}")
            return False

    def _save_metadata(self, metadata_list: List[RecordingMetadata]) -> None:
        """Save metadata to CSV."""
        if not metadata_list:
            return

        csv_path = self.download_dir / "metadata.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            # Convert dataclasses to dicts
            rows = [asdict(m) for m in metadata_list]

            # Write CSV
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def _build_extra_query(
        self,
        base_query: Optional[str],
        bounding_box: Optional[Dict[str, float]],
    ) -> Optional[str]:
        """Combine base extra_query with optional bounding box filter."""
        parts: List[str] = []

        if base_query:
            parts.append(base_query)
        if bounding_box:
            parts.append(self._format_bounding_box(bounding_box))

        if parts:
            return " ".join(parts)
        return None

    @staticmethod
    def _validate_bounding_box(
        bounding_box: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, float]]:
        """Validate and coerce bounding box coordinates from config."""
        if not bounding_box:
            return None

        try:
            lat_min = float(bounding_box["lat_min"])
            lon_min = float(bounding_box["lon_min"])
            lat_max = float(bounding_box["lat_max"])
            lon_max = float(bounding_box["lon_max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                "Invalid bounding_box config. Expected keys 'lat_min', 'lon_min', "
                f"'lat_max', 'lon_max' with numeric values. Error: {exc}"
            )

        return {
            "lat_min": lat_min,
            "lon_min": lon_min,
            "lat_max": lat_max,
            "lon_max": lon_max,
        }

    @staticmethod
    def _format_bounding_box(bounding_box: Dict[str, float]) -> str:
        """Return Xeno-Canto 'box' query string."""
        return (
            f"box:{bounding_box['lat_min']},"
            f"{bounding_box['lon_min']},"
            f"{bounding_box['lat_max']},"
            f"{bounding_box['lon_max']}"
        )


def normalize_url(url: str) -> str:
    """Normalize URL (handle protocol-relative URLs)."""
    if url.startswith("//"):
        return "https:" + url
    return url


def _safe_float(value: Any) -> float:
    """Convert API number fields to float, defaulting to 0.0 on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults."""

    # Default configuration
    default_config = {
        "species": {
            "scientific_name": "Emberiza citrinella",
            "common_name": "Yellowhammer",
            "slug": "emberiza_citrinella"
        },
        "paths": {
            "root": "/Volumes/Z Slim/zslim_birdcluster"
        },
        "xeno_canto": {
            "max_recordings": 1000,
            "include_background": True,
            "extra_query": "len:3-200",
            "bounding_box": {
                "lat_min": 15.0,
                "lon_min": -35.0,
                "lat_max": 82.0,
                "lon_max": 90.0,
            },
            "timeout": 15,
            "pause": 0.5,
            "per_page": 200,
            "api_key": None,
            "api_key_env_var": "XENO_CANTO_API_KEY",
        }
    }

    if config_path:
        requested_path = config_path
        resolved_path = (
            config_path if config_path.is_absolute() else (Path.cwd() / config_path)
        )
        if not resolved_path.exists():
            raise SystemExit(
                f"Config file '{requested_path}' not found. Use '--config xc_configs/<name>.yaml'."
            )
        with open(resolved_path) as f:
            config = yaml.safe_load(f)
        # Merge with defaults
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
            elif isinstance(default_config[key], dict):
                for subkey in default_config[key]:
                    if subkey not in config[key]:
                        config[key][subkey] = default_config[key][subkey]
        return config
    
    return default_config


def main() -> None:
    """Parse arguments and start the download process."""
    parser = argparse.ArgumentParser(
        description="Download bird recordings from Xeno-Canto"
    )
    parser.add_argument("--config", type=Path, help="Path to config.yaml file")
    parser.add_argument(
        "--species",
        type=str,
        help="Scientific name (e.g., 'Emberiza citrinella')",
    )
    parser.add_argument(
        "--common-name",
        type=str,
        help="Common name (e.g., 'Yellowhammer')",
    )
    parser.add_argument(
        "--max-recordings", type=int, help="Maximum recordings to download"
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Include background recordings",
    )
    parser.add_argument("--root", type=Path, help="Root directory for downloads")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.species:
        config["species"]["scientific_name"] = args.species
        config["species"]["slug"] = args.species.lower().replace(" ", "_")
    if args.common_name:
        config["species"]["common_name"] = args.common_name
    if args.max_recordings:
        config["xeno_canto"]["max_recordings"] = args.max_recordings
    if args.include_background:
        config["xeno_canto"]["include_background"] = True
    if args.root:
        config["paths"]["root"] = str(args.root)
    
    # Run downloader
    downloader = SpeciesDownloader(config)
    downloader.download()


if __name__ == "__main__":
    main()
