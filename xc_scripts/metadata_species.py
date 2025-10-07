#!/usr/bin/env python3
"""
Rebuild metadata for already downloaded Xeno-Canto recordings.
This script recreates the metadata.csv file based on existing downloaded files,
without re-downloading anything.

Usage:
    python xc_scripts/metadata_species.py --config xc_configs/config_emberiza_citrinella.yaml
    python xc_scripts/metadata_species.py --species "Emberiza citrinella"
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Set
from dataclasses import dataclass, asdict
import requests
import yaml
from tqdm import tqdm


@dataclass
class RecordingMetadata:
    """Metadata for a single recording"""
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
    def from_xc_record(cls, rec: Dict, target_species: str, is_background: bool = False):
        """Create from Xeno-Canto API response"""
        # Handle 'also' field which can be a list or string
        also_field = rec.get("also", "")
        if isinstance(also_field, list):
            also_field = ", ".join(also_field)
        
        return cls(
            xcid=rec.get("id", ""),
            species_sci=f"{rec.get('gen', '')} {rec.get('sp', '')}".strip(),
            species_common=rec.get("en", ""),
            is_background=is_background,
            main_species=f"{rec.get('gen', '')} {rec.get('sp', '')}".strip() if is_background else target_species,
            date=rec.get("date", ""),
            time=rec.get("time", ""),
            country=rec.get("cnt", ""),
            location=rec.get("loc", ""),
            lat=float(rec.get("lat", 0)) if rec.get("lat") else 0,
            lon=float(rec.get("lng", rec.get("lon", 0))) if rec.get("lng") or rec.get("lon") else 0,
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
            remarks=rec.get("rmk", "")
        )


class XenoCantoCrawler:
    """Handle Xeno-Canto API interactions"""
    
    def __init__(self, request_timeout: int = 15, request_pause: float = 0.01):
        self.base_url = "https://xeno-canto.org/api/2/recordings"
        self.timeout = request_timeout
        self.pause = request_pause
        self.session = requests.Session()
    
    def search_foreground(self, species: str, extra_query: Optional[str] = None) -> Iterable[Dict]:
        """Search for recordings where species is the main subject"""
        query_parts = [species]
        if extra_query:
            query_parts.append(extra_query)
        query = " ".join(query_parts).replace(" ", "+")
        
        yield from self._paginate_search(query)
    
    def search_background(self, species: str, extra_query: Optional[str] = None) -> Iterable[Dict]:
        """Search for recordings where species appears in 'also' field"""
        query_parts = [f'also:"{species}"']
        if extra_query:
            query_parts.append(extra_query)
        query = " ".join(query_parts)
        
        print(f"  Background query: {query}")
        yield from self._paginate_search(query)
    
    def _paginate_search(self, query: str) -> Iterable[Dict]:
        """Paginate through search results"""
        page = 1
        total_pages = None
        
        while True:
            encoded_query = requests.utils.quote(query, safe='+')
            url = f"{self.base_url}?query={encoded_query}&page={page}"
            
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                print(f"Error fetching page {page}: {e}")
                break
            
            recordings = data.get("recordings", [])
            if not recordings:
                break
            
            if total_pages is None:
                total_pages = int(data.get("numPages", 1))
                total_recs = int(data.get("numRecordings", 0))
                print(f"  Found {total_recs} recordings across {total_pages} pages")
            
            yield from recordings
            
            if page >= total_pages:
                break
            
            page += 1
            time.sleep(self.pause)


class MetadataRebuilder:
    """Rebuild metadata for existing downloads"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.species = config["species"]["scientific_name"]
        self.common_name = config["species"].get("common_name", "")
        self.slug = config["species"].get("slug", self.species.lower().replace(" ", "_"))
        
        # Setup paths
        self.root = Path(config["paths"]["root"]).expanduser()
        self.download_dir = self.root / "xc_downloads" / self.slug
        
        if not self.download_dir.exists():
            raise ValueError(f"Download directory does not exist: {self.download_dir}")
        
        self.crawler = XenoCantoCrawler(
            request_timeout=config.get("xeno_canto", {}).get("timeout", 15),
            request_pause=config.get("xeno_canto", {}).get("pause", 0.5)
        )
    
    def rebuild_metadata(self):
        """Main rebuild process"""
        include_background = self.config["xeno_canto"].get("include_background", True)
        extra_query = self.config["xeno_canto"].get("extra_query", None)
        
        print(f"\n{'='*60}")
        print(f"Rebuilding metadata for: {self.species}")
        if self.common_name:
            print(f"Common name: {self.common_name}")
        print(f"Directory: {self.download_dir}")
        print(f"Include background: {include_background}")
        if extra_query:
            print(f"Extra filters: {extra_query}")
        print(f"{'='*60}\n")
        
        # Get existing file IDs
        existing_ids = self._get_existing_ids()
        print(f"Found {len(existing_ids)} existing recordings")
        
        if not existing_ids:
            print("No existing recordings found!")
            return
        
        # Collect metadata for existing files
        all_metadata = []
        processed_ids = set()
        
        # Search foreground recordings
        print("\nSearching for foreground recording metadata...")
        for rec in self.crawler.search_foreground(self.species, extra_query):
            xcid = rec.get("id", "")
            
            if xcid in existing_ids and xcid not in processed_ids:
                metadata = RecordingMetadata.from_xc_record(rec, self.species, is_background=False)
                all_metadata.append(metadata)
                processed_ids.add(xcid)
                
                if len(processed_ids) % 50 == 0:
                    print(f"  Processed {len(processed_ids)}/{len(existing_ids)} recordings...")
            
            # Stop if we've found all our files
            if len(processed_ids) == len(existing_ids):
                break
        
        # Search background recordings if needed
        if include_background and len(processed_ids) < len(existing_ids):
            print(f"\nSearching for background recording metadata...")
            print(f"Still looking for {len(existing_ids) - len(processed_ids)} recordings...")
            
            for rec in self.crawler.search_background(self.species, extra_query):
                xcid = rec.get("id", "")
                
                if xcid in existing_ids and xcid not in processed_ids:
                    metadata = RecordingMetadata.from_xc_record(rec, self.species, is_background=True)
                    
                    # Verify our species is in the 'also' field
                    also_text = metadata.also if isinstance(metadata.also, str) else str(metadata.also)
                    if also_text and self.species.lower() in also_text.lower():
                        all_metadata.append(metadata)
                        processed_ids.add(xcid)
                        
                        if len(processed_ids) % 50 == 0:
                            print(f"  Processed {len(processed_ids)}/{len(existing_ids)} recordings...")
                
                # Stop if we've found all our files
                if len(processed_ids) == len(existing_ids):
                    break
        
        # Check for missing metadata
        missing_ids = existing_ids - processed_ids
        if missing_ids:
            print(f"\nWarning: Could not find metadata for {len(missing_ids)} recordings:")
            for mid in sorted(missing_ids)[:10]:  # Show first 10
                print(f"  - XC{mid}")
            if len(missing_ids) > 10:
                print(f"  ... and {len(missing_ids) - 10} more")
        
        # Save metadata
        if all_metadata:
            self._save_metadata(all_metadata)
            
            print(f"\n{'='*60}")
            print(f"Metadata rebuild complete!")
            print(f"Total recordings with metadata: {len(all_metadata)}")
            print(f"Foreground: {sum(1 for m in all_metadata if not m.is_background)}")
            print(f"Background: {sum(1 for m in all_metadata if m.is_background)}")
            print(f"Missing metadata: {len(missing_ids)}")
            print(f"Metadata saved to: {self.download_dir / 'metadata.csv'}")
            print(f"{'='*60}\n")
        else:
            print("\nNo metadata found for existing recordings!")
    
    def _get_existing_ids(self) -> Set[str]:
        """Get IDs of already downloaded recordings"""
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
    
    def _save_metadata(self, metadata_list: List[RecordingMetadata]):
        """Save metadata to CSV"""
        if not metadata_list:
            return
        
        csv_path = self.download_dir / "metadata.csv"
        
        # Backup existing metadata if it exists
        if csv_path.exists():
            backup_path = csv_path.with_suffix('.csv.bak')
            print(f"Backing up existing metadata to: {backup_path}")
            csv_path.rename(backup_path)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            # Convert dataclasses to dicts
            rows = [asdict(m) for m in metadata_list]
            
            # Write CSV
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def normalize_url(url: str) -> str:
    """Normalize URL (handle protocol-relative URLs)"""
    if url.startswith("//"):
        return "https:" + url
    return url


def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from file or use defaults"""
    
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
            "timeout": 15,
            "pause": 0.5
        }
    }
    
    if config_path:
        requested_path = config_path
        resolved_path = config_path if config_path.is_absolute() else (Path.cwd() / config_path)
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


def main():
    parser = argparse.ArgumentParser(description="Rebuild metadata for existing Xeno-Canto recordings")
    parser.add_argument("--config", type=Path, help="Path to config.yaml file")
    parser.add_argument("--species", type=str, help="Scientific name (e.g., 'Emberiza citrinella')")
    parser.add_argument("--common-name", type=str, help="Common name (e.g., 'Yellowhammer')")
    parser.add_argument("--include-background", action="store_true", help="Include background recordings")
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
    if args.include_background:
        config["xeno_canto"]["include_background"] = True
    if args.root:
        config["paths"]["root"] = str(args.root)
    
    # Run metadata rebuilder
    try:
        rebuilder = MetadataRebuilder(config)
        rebuilder.rebuild_metadata()
    except ValueError as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main()
