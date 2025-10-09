#!/usr/bin/env python3
"""
Process bird recordings from folder structure: extract clips and generate embeddings.
Designed for Raspberry Pi exports organised by date and species folders.

Typical usage (from repository root):
    python birdnet_pi/process_species_pi.py --species "Merel" --root /path/to/birdnet_all_recordings --metadata birdnet_pi/detections.csv
    for me:
    python birdnet_pi/process_species_pi.py --species "Bonte_Kraai" --root /Users/masjansma/birdnet_all_recordings --metadata birdnet_pi/detections.csv
    python birdnet_pi/process_species_pi.py --species "Bonte_Kraai" --root /Users/masjansma/birdnet_all_recordings --metadata birdnet_pi/detections.csv --output /Users/masjansma/Desktop/pi_clips_embeds
    
    The script defaults to storing outputs in `birdnet_pi/pi_clips_embeds/<species_slug>/`.
    Override with `--output` if you want the output data elsewhere.
    
    Note that --metadata specifies the input location of the CSV file with metadata,
    inputting metadata is not optional, it is necessary.
    when running one should see: 'Found 778 MP3 files for Bonte_Kraai
    Loading metadata from birdnet_pi/detections.csv
    Loaded 134031 metadata entries'. # If no metadata is seen here, no good.
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import yaml
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from datetime import datetime
import re


@dataclass
class ClipInfo:
    """Information about a single clip"""
    source_file: Path
    clip_file: Path
    date: str
    time: str
    species: str
    clip_start: float
    clip_end: float
    
    def to_dict(self):
        return {
            'source_file': str(self.source_file),
            'clip_file': str(self.clip_file),
            'date': self.date,
            'time': self.time,
            'species': self.species,
            'clip_start_s': self.clip_start,
            'clip_end_s': self.clip_end
        }


class SpeciesFolderProcessor:
    """Process recordings from folder structure for a specific species"""
    
    def __init__(self, species: str, root_dir: Path, output_dir: Path, 
                 metadata_csv: Optional[Path] = None, config: Dict = None):
        self.species = species
        self.species_slug = species.lower().replace(" ", "_")
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.metadata_csv = metadata_csv
        
        # Setup output paths
        self.clips_dir = self.output_dir / "clips" / self.species_slug
        self.embeddings_dir = self.output_dir / "embeddings" / self.species_slug
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters (with defaults)
        if config is None:
            config = {}
        self.clip_duration = config.get("clip_duration_sec", 3.0)
        self.threads = config.get("threads", 8)
        self.batch_size = config.get("batch_size", 32)
        self.delete_originals = config.get("delete_originals", False)
        
    def process(self, skip_confirm: bool = False):
        """Main processing pipeline"""
        print(f"\n{'='*60}")
        print(f"Processing recordings for: {self.species}")
        print(f"Input root: {self.root_dir}")
        print(f"Output: {self.clips_dir}")
        print(f"{'='*60}\n")
        
        # Step 1: Find all MP3 files for this species across all date folders
        mp3_files = self._find_species_files()
        if not mp3_files:
            print(f"No files found for species: {self.species}")
            return
        
        print(f"Found {len(mp3_files)} MP3 files for {self.species}")
        
        # Step 2: Load metadata if provided
        metadata_df = None
        if self.metadata_csv and self.metadata_csv.exists():
            print(f"Loading metadata from {self.metadata_csv}")
            metadata_df = pd.read_csv(self.metadata_csv)
            print(f"  Loaded {len(metadata_df)} metadata entries")
        
        # Step 3: Create clips from MP3 files
        print("\nCreating clips...")
        all_clips = []
        
        for mp3_file in tqdm(mp3_files, desc="Processing files"):
            # Extract metadata from filepath and filename
            file_metadata = self._extract_file_metadata(mp3_file)
            
            # Create a random 3s clip
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clip = self._create_random_clip(mp3_file, file_metadata)
            
            if clip:
                all_clips.append(clip)
        
        print(f"Created {len(all_clips)} clips")
        
        # Step 4: Generate embeddings in batch
        if all_clips:
            print("\nGenerating embeddings...")
            embeddings_csv = self._generate_embeddings_batch()
            
            if not embeddings_csv:
                print("Failed to generate embeddings!")
                return
            
            # Step 5: Create aligned metadata file
            self._create_aligned_metadata(all_clips, embeddings_csv, metadata_df)
        
        # Step 6: Optionally delete originals
        if self.delete_originals and mp3_files:
            print(f"\n{'='*60}")
            print(f"Ready to delete {len(mp3_files)} original MP3 files")
            print(f"Clips created: {len(all_clips)}")
            print(f"{'='*60}")
            
            if not skip_confirm:
                response = input("\nDelete original MP3 files? (yes/no): ")
                if response.lower() != 'yes':
                    print("Keeping original files.")
                    return
            
            print(f"Deleting {len(mp3_files)} original files...")
            for f in tqdm(mp3_files, desc="Deleting"):
                f.unlink()
        
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"Clips: {len(all_clips)} files in {self.clips_dir}")
        print(f"Embeddings: {self.embeddings_dir / 'embeddings.csv'}")
        print(f"Metadata: {self.embeddings_dir / 'metadata.csv'}")
        print(f"{'='*60}\n")
    
    def _find_species_files(self) -> List[Path]:
        """Find all MP3 files for the species across all date folders"""
        mp3_files = []
        
        # Look for date folders (YYYY-MM-DD format)
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        
        for item in self.root_dir.iterdir():
            if item.is_dir() and date_pattern.match(item.name):
                # Check if this date folder has our species
                species_dir = item / self.species
                if species_dir.exists() and species_dir.is_dir():
                    # Collect all MP3 files in this species folder
                    mp3_files.extend(species_dir.glob("*.mp3"))
        
        return sorted(mp3_files)
    
    def _extract_file_metadata(self, filepath: Path) -> Dict:
        """Extract metadata from filepath and filename"""
        # Example: /Users/masjansma/birdnet_all_recordings/2025-06-10/Ekster/Ekster-72-2025-06-10-birdnet-20/18/40.mp3
        # The filename contains time information in an unusual format with slashes
        
        # Get the date from parent directories
        date_folder = filepath.parent.parent.name  # Should be like "2025-06-10"
        species_folder = filepath.parent.name  # Should be species name
        
        # Parse the filename to extract time
        # Format appears to be: Species-ID-Date-birdnet-HH/MM/SS.mp3
        filename = filepath.stem  # Remove .mp3
        
        # Try to extract time from the filename
        time_str = "00:00:00"  # Default
        
        # The filename might have slashes in it for time, need to handle carefully
        # Since Path automatically handles slashes, we need to reconstruct
        full_filename = filepath.name
        
        # Try to parse time from the end of filename
        if "-birdnet-" in full_filename:
            parts = full_filename.split("-birdnet-")
            if len(parts) == 2:
                time_part = parts[1].replace(".mp3", "")
                # Convert format like "20/18/40" to "20:18:40"
                time_str = time_part.replace("/", ":")
        
        return {
            'date': date_folder,
            'time': time_str,
            'species': self.species
        }
    
    def _create_random_clip(self, mp3_file: Path, metadata: Dict) -> Optional[ClipInfo]:
        """Create a random 3s clip from the MP3 file"""
        try:
            # Load audio
            y, sr = librosa.load(str(mp3_file), sr=None, mono=True)
            duration = len(y) / sr
            
            # If file is shorter than clip duration, use the whole file
            if duration <= self.clip_duration:
                clip_start = 0
                clip_end = duration
            else:
                # Random start position
                max_start = duration - self.clip_duration
                clip_start = random.uniform(0, max_start)
                clip_end = clip_start + self.clip_duration
            
            # Extract clip
            start_sample = int(clip_start * sr)
            end_sample = int(clip_end * sr)
            clip_audio = y[start_sample:end_sample]
            
            # Create clip filename
            # Use date and time to make unique filename
            safe_time = metadata['time'].replace(':', '')
            clip_filename = f"{self.species_slug}_{metadata['date']}_{safe_time}.wav"
            clip_path = self.clips_dir / clip_filename
            
            # Save clip
            sf.write(str(clip_path), clip_audio, sr)
            
            return ClipInfo(
                source_file=mp3_file,
                clip_file=clip_path,
                date=metadata['date'],
                time=metadata['time'],
                species=metadata['species'],
                clip_start=clip_start,
                clip_end=clip_end
            )
            
        except Exception as e:
            print(f"Error processing {mp3_file.name}: {e}")
            return None
    
    def _generate_embeddings_batch(self) -> Optional[Path]:
        """Generate embeddings using BirdNET-Analyzer in batch mode"""
        embeddings_csv = self.embeddings_dir / "embeddings.csv"
        embeds_db = self.embeddings_dir / "embeds_db"
        
        # Use BirdNET-Analyzer's embedding mode
        cmd = [
            sys.executable, "-m", "birdnet_analyzer.embeddings",
            "-i", str(self.clips_dir),  # Input directory with clips
            "-db", str(embeds_db),      # Database folder for storing embeddings
            "--file_output", str(embeddings_csv),  # Output CSV file
            "-t", str(self.threads),    # Threads
            "-b", str(self.batch_size)  # Batch size
        ]
        
        print(f"Generating embeddings with {self.threads} threads...")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error generating embeddings: {result.stderr}")
                # Try fallback without batch size parameter
                cmd_fallback = [
                    sys.executable, "-m", "birdnet_analyzer.embeddings",
                    "-i", str(self.clips_dir),
                    "-db", str(embeds_db),
                    "--file_output", str(embeddings_csv),
                    "-t", str(self.threads)
                ]
                result = subprocess.run(cmd_fallback, capture_output=True, text=True)
                
            if embeddings_csv.exists():
                return embeddings_csv
            else:
                print("Embeddings file not created!")
                return None
                
        except Exception as e:
            print(f"Failed to run embedding generation: {e}")
            return None
    
    def _create_aligned_metadata(self, clips: List[ClipInfo], 
                                embeddings_csv: Path, 
                                original_metadata: Optional[pd.DataFrame]):
        """Create metadata file aligned with embeddings order"""
        
        # Read embeddings to get the order
        embeddings_df = pd.read_csv(embeddings_csv)
        
        # Create a mapping from clip filename to ClipInfo
        clip_map = {clip.clip_file.name: clip for clip in clips}
        
        # Create metadata rows in the same order as embeddings
        metadata_rows = []
        
        for _, row in embeddings_df.iterrows():
            filepath = Path(row.iloc[0])
            filename = filepath.name
            
            if filename in clip_map:
                clip = clip_map[filename]
                
                # Build metadata row
                meta_row = {
                    'Date': clip.date,
                    'Time': clip.time,
                    'Species': clip.species,
                    'Clip_File': str(clip.clip_file.name),
                    'Source_File': str(clip.source_file),
                    'Clip_Start_s': clip.clip_start,
                    'Clip_End_s': clip.clip_end
                }
                
                # If we have original metadata, try to find matching entry
                if original_metadata is not None:
                    # Try to match by filename or other criteria
                    source_name = clip.source_file.name
                    
                    # Look for matching row in original metadata
                    # This might need adjustment based on how File_Name is formatted
                    matching = original_metadata[
                        original_metadata['File_Name'].str.contains(source_name, na=False)
                    ]
                    
                    if not matching.empty:
                        # Add additional metadata fields
                        orig_row = matching.iloc[0]
                        meta_row.update({
                            'Sci_Name': orig_row.get('Sci_Name', clip.species),
                            'Com_Name': orig_row.get('Com_Name', ''),
                            'Confidence': orig_row.get('Confidence', 0.0),
                            'Lat': orig_row.get('Lat', 0.0),
                            'Lon': orig_row.get('Lon', 0.0),
                            'Cutoff': orig_row.get('Cutoff', 0.0),
                            'Week': orig_row.get('Week', 0.0),
                            'Sens': orig_row.get('Sens', 0.0),
                            'Overlap': orig_row.get('Overlap', 0.0)
                        })
                
                metadata_rows.append(meta_row)
        
        # Save aligned metadata
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_path = self.embeddings_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        print(f"Saved aligned metadata with {len(metadata_df)} entries")


def main():
    parser = argparse.ArgumentParser(
        description="Process bird recordings from folder structure: extract clips and generate embeddings"
    )
    parser.add_argument("--species", type=str, required=True,
                       help="Species name (should match folder names)")
    parser.add_argument("--root", type=Path, 
                       default=Path.home() / "birdnet_all_recordings",
                       help="Root directory containing date folders")
    parser.add_argument("--output", type=Path,
                       help="Species output directory (defaults to birdnet_pi/pi_clips_embeds/<species_slug>)")
    parser.add_argument("--metadata", type=Path,
                       help="Path to detections CSV file with metadata")
    parser.add_argument("--clip-duration", type=float, default=3.0,
                       help="Clip duration in seconds")
    parser.add_argument("--threads", type=int, default=8,
                       help="Number of threads for processing")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for BirdNET embeddings")
    parser.add_argument("--delete-originals", action="store_true",
                       help="Delete original MP3 files after processing")
    parser.add_argument("--skip-confirm", action="store_true",
                       help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = {
        "clip_duration_sec": args.clip_duration,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "delete_originals": args.delete_originals
    }
    
    species_slug = args.species.lower().replace(" ", "_")
    output_arg = Path(args.output).expanduser() if args.output else None
    default_root = Path(__file__).resolve().parent / "pi_clips_embeds"
    if output_arg:
        if output_arg.name.lower().replace(" ", "_") == species_slug:
            output_dir = output_arg
        else:
            output_dir = output_arg / species_slug
    else:
        output_dir = default_root / species_slug

    # Create processor
    root_dir = Path(args.root).expanduser()
    metadata_path = Path(args.metadata).expanduser() if args.metadata else None

    processor = SpeciesFolderProcessor(
        species=args.species,
        root_dir=root_dir,
        output_dir=output_dir,
        metadata_csv=metadata_path,
        config=config
    )
    
    # Run processing
    processor.process(skip_confirm=args.skip_confirm)


if __name__ == "__main__":
    main()
