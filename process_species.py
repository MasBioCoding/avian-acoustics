#!/usr/bin/env python3
"""
Process downloaded bird recordings: detect, clip, and generate embeddings.
Uses BirdNET-Analyzer CLI for efficient batch processing.

Usage:
    python process_species.py --config config_limosa_limosa.yaml
    python process_species.py --config config_emberiza_citrinella.yaml
    python process_species.py --config config_sylvia_atricapilla.yaml
    python process_species.py --species "Sylvia atricapilla" --skip-confirm
    python process_species.py --config config_turdus_merula.yaml
    python process_species.py --config config_parus_major.yaml
    python process_species.py --config config_corvus_corax.yaml
    python process_species.py --config config_passer_montanus.yaml
    python process_species.py --config config_passer_domesticus.yaml
    python process_species.py --config config_strix_aluco.yaml
    python process_species.py --config config_asio_otus.yaml
    python process_species.py --config config_chloris_chloris.yaml
    python process_species.py --config config_phylloscopus_collybita.yaml
    python process_species.py --config config_phylloscopus_trochilus.yaml
    python process_species.py --config config_acrocephalus_scirpaceus.yaml
    python process_species.py --config config_curruca_communis.yaml
    python process_species.py --config config_cettia_cetti.yaml

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


@dataclass
class ClipInfo:
    """Information about a single clip"""
    xcid: str
    clip_index: int
    start_time: float
    end_time: float
    confidence: float
    source_file: Path
    clip_file: Path
    
    def to_dict(self):
        return {
            'xcid': self.xcid,
            'clip_index': self.clip_index,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'confidence': self.confidence,
            'source_file': str(self.source_file),
            'clip_file': str(self.clip_file)
        }


class SpeciesProcessor:
    """Process recordings using BirdNET-Analyzer CLI for efficiency"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.species = config["species"]["scientific_name"]
        self.slug = config["species"].get("slug", self.species.lower().replace(" ", "_"))
        
        # Setup paths
        self.root = Path(config["paths"]["root"]).expanduser()
        self.download_dir = self.root / "xc_downloads" / self.slug
        self.clips_dir = self.root / "clips" / self.slug
        self.embeddings_dir = self.root / "embeddings" / self.slug
        
        # Create output directories
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters
        self.max_clips = config["processing"].get("max_clips_per_recording", 5)
        self.clip_duration = config["processing"].get("clip_duration_sec", 3.0)
        self.min_confidence = config["processing"].get("min_confidence", 0.20)
        self.merge_within = config["processing"].get("merge_within_sec", 1.0)
        self.delete_originals = config["processing"].get("delete_originals", True)
        
        # BirdNET parameters
        self.threads = config.get("birdnet", {}).get("threads", 4)
        self.batch_size = config.get("birdnet", {}).get("batch_size", 16)
        
    def process(self, skip_confirm: bool = False):
        """Main processing pipeline"""
        print(f"\n{'='*60}")
        print(f"Processing recordings for: {self.species}")
        print(f"Input: {self.download_dir}")
        print(f"Output: {self.clips_dir}")
        print(f"{'='*60}\n")
        
        # Step 1: Load existing metadata
        metadata_df = self._load_metadata()
        if metadata_df is None:
            print("No metadata.csv found in download directory!")
            return
        
        # Step 2: Get audio files to process
        audio_files = self._get_audio_files()
        if not audio_files:
            print("No audio files found to process!")
            return
        
        print(f"Found {len(audio_files)} audio files")
        
        # Step 3: Run batch detection with BirdNET
        print("\nRunning BirdNET detection...")
        detections_by_file = self._batch_detect_species(audio_files)
        
        # Step 4: Process detections and create clips
        print(f"\nFound {self.species} in {len(detections_by_file)} files")
        if not detections_by_file:
            print("No target species detected in any files!")
            return
            
        print("Creating clips...")
        all_clips = []
        files_to_delete = []
        
        # Progress bar for files with detections only
        for audio_path_str in tqdm(detections_by_file.keys(), desc="Processing files"):
            audio_file = Path(audio_path_str)
            xcid = self._extract_xcid(audio_file)
            
            # Get detections for this file
            detections = detections_by_file[audio_path_str]
            
            # Select random non-overlapping detections
            selected_detections = self._select_random_detections(detections)
            
            # Create clips (suppress librosa warnings)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clips = self._create_clips(audio_file, xcid, selected_detections)
            
            all_clips.extend(clips)
            if clips:
                files_to_delete.append(audio_file)
        
        print(f"\nCreated {len(all_clips)} clips from {len(files_to_delete)} recordings")
        
        # Step 5: Generate embeddings in batch
        print("\nGenerating embeddings...")
        embeddings_csv = self._generate_embeddings_batch()
        
        if not embeddings_csv:
            print("Failed to generate embeddings!")
            return
        
        # Step 6: Create aligned metadata file
        self._create_aligned_metadata(all_clips, embeddings_csv, metadata_df)
        
        # Step 7: Delete ALL originals if requested
        if self.delete_originals and audio_files:
            print(f"\n{'='*60}")
            print(f"Ready to delete ALL {len(audio_files)} original recordings")
            print(f"Files with detections: {len(files_to_delete)}")
            print(f"Files without detections: {len(audio_files) - len(files_to_delete)}")
            print(f"Clips created: {len(all_clips)}")
            print(f"Output directory: {self.clips_dir}")
            print(f"{'='*60}")
            
            if not skip_confirm:
                response = input("\nDelete ALL original recordings? (yes/no): ")
                if response.lower() != 'yes':
                    print("Keeping original files.")
                    return
            
            print(f"Deleting all {len(audio_files)} original recordings...")
            for f in tqdm(audio_files, desc="Deleting"):
                f.unlink()
        
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"Clips: {len(all_clips)} files in {self.clips_dir}")
        print(f"Embeddings: {self.embeddings_dir / 'embeddings.csv'}")
        print(f"Metadata: {self.embeddings_dir / 'metadata.csv'}")
        print(f"{'='*60}\n")
    
    def _load_metadata(self) -> Optional[pd.DataFrame]:
        """Load metadata from download directory"""
        metadata_path = self.download_dir / "metadata.csv"
        if not metadata_path.exists():
            return None
        return pd.read_csv(metadata_path, dtype={'xcid': str})
    
    def _get_audio_files(self) -> List[Path]:
        """Get list of audio files to process"""
        audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        files = []
        for ext in audio_exts:
            files.extend(self.download_dir.glob(f"*{ext}"))
        return sorted(files)
    
    def _extract_xcid(self, filepath: Path) -> str:
        """Extract XC ID from filename"""
        # Format: slug_xcid.ext
        parts = filepath.stem.split('_')
        if len(parts) >= 2:
            return parts[-1]
        return filepath.stem
    
    def _batch_detect_species(self, audio_files: List[Path]) -> Dict[str, List[Dict]]:
        """
        Run BirdNET detection on all files using CLI for efficiency.
        Returns dict mapping file path to list of detections.
        """
        # Create temp directory for results
        temp_dir = self.embeddings_dir / "temp_detections"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # BirdNET CLI creates a directory when processing a directory
            results_dir = temp_dir / "results"
            
            # Build command for batch analysis - NO SPECIES FILTER
            cmd = [
                sys.executable, "-m", "birdnet_analyzer.analyze",
                "--output", str(results_dir),  # Output directory
                "--min_conf", str(self.min_confidence),
                "--threads", str(self.threads),
                "--batch_size", str(self.batch_size),
                "--rtype", "csv",
                "--locale", "en",
                str(self.download_dir)  # INPUT is positional at the end
            ]
            
            print(f"  Using {self.threads} threads, confidence threshold {self.min_confidence}")
            print(f"  Processing {len(audio_files)} files...")
            
            # Run detection and capture output to monitor progress
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                     text=True, bufsize=1)
            
            # Monitor output and show progress
            files_processed = 0
            for line in process.stdout:
                if "Finished" in line:  # BirdNET prints "Finished /path/to/file.mp3 in X seconds"
                    files_processed += 1
                    if files_processed % 100 == 0 or files_processed == len(audio_files):
                        print(f"    Analyzed: {files_processed}/{len(audio_files)} files", end='\r')
            
            process.wait()
            
            if files_processed > 0:
                print(f"    Analyzed: {files_processed}/{len(audio_files)} files - Complete")
            
            if process.returncode != 0:
                print("  BirdNET CLI failed, falling back to slower method")
                return self._fallback_detect(audio_files)
            
            # Parse results from individual CSV files
            detections_by_file = {}
            
            if results_dir.exists() and results_dir.is_dir():
                csv_files = list(results_dir.glob("*.csv"))
                
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        
                        if df.empty:
                            continue
                        
                        # Skip metadata files - only process detection files
                        required_columns = ['Start (s)', 'End (s)', 'Scientific name', 'Common name', 'Confidence']
                        if not all(col in df.columns for col in required_columns):
                            continue
                        
                        # Parse filename to find audio file
                        csv_name = csv_file.stem
                        if '.BirdNET.results' in csv_name:
                            base_name = csv_name.replace('.BirdNET.results', '')
                        else:
                            base_name = csv_name
                        
                        # Find the actual audio file
                        audio_path = None
                        for ext in ['.mp3', '.MP3', '.wav', '.ogg', '.m4a', '.flac']:
                            potential_path = self.download_dir / f"{base_name}{ext}"
                            if potential_path.exists():
                                audio_path = potential_path
                                break
                        
                        if not audio_path or audio_path not in audio_files:
                            continue
                        
                        # Filter to target species
                        target_detections = []
                        for _, row in df.iterrows():
                            sci_name = str(row['Scientific name']).strip()
                            
                            # Check if this is our target species (exact match, case-insensitive)
                            if sci_name.lower() == self.species.lower():
                                target_detections.append({
                                    'start_time': float(row['Start (s)']),
                                    'end_time': float(row['End (s)']),
                                    'confidence': float(row['Confidence']),
                                    'scientific_name': sci_name,
                                    'common_name': row['Common name']
                                })
                        
                        if target_detections:
                            detections_by_file[str(audio_path)] = target_detections
                                
                    except Exception as e:
                        continue
            
            return detections_by_file
            
        finally:
            # Clean up temp files after we're done
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _fallback_detect(self, audio_files: List[Path]) -> Dict[str, List[Dict]]:
        """Fallback to birdnetlib if CLI fails"""
        print("Falling back to birdnetlib (slower)...")
        
        try:
            from birdnetlib import Recording
            from birdnetlib.analyzer import Analyzer
        except ImportError:
            print("Error: birdnetlib not available for fallback")
            return {}
        
        analyzer = Analyzer()
        detections_by_file = {}
        
        for audio_file in tqdm(audio_files, desc="Detecting"):
            try:
                recording = Recording(
                    analyzer,
                    str(audio_file),
                    min_conf=self.min_confidence
                )
                recording.analyze()
                
                # Filter to target species
                detections = []
                for det in recording.detections:
                    sci_name = det.get('scientific_name', '').lower()
                    if self.species.lower() in sci_name:
                        detections.append(det)
                
                if detections:
                    detections_by_file[str(audio_file)] = detections
                    
            except Exception as e:
                print(f"Error analyzing {audio_file.name}: {e}")
        
        return detections_by_file
    
    def _select_random_detections(self, detections: List[Dict]) -> List[Dict]:
        """Randomly select N non-overlapping detections"""
        if not detections:
            return []
        
        # First, find ALL non-overlapping detections
        sorted_dets = sorted(detections, 
                           key=lambda x: float(x.get('start_time', 0)))
        
        non_overlapping = []
        for det in sorted_dets:
            start = float(det.get('start_time', 0))
            end = float(det.get('end_time', 0))
            
            overlap = False
            for existing in non_overlapping:
                ex_start = float(existing.get('start_time', 0))
                ex_end = float(existing.get('end_time', 0))
                
                # Check for overlap (with merge_within buffer)
                if not (end + self.merge_within < ex_start or 
                       start > ex_end + self.merge_within):
                    overlap = True
                    break
            
            if not overlap:
                non_overlapping.append(det)
        
        # Randomly sample up to max_clips from all non-overlapping
        if len(non_overlapping) <= self.max_clips:
            selected = non_overlapping
        else:
            selected = random.sample(non_overlapping, self.max_clips)
        
        # Return in temporal order for consistent clip numbering
        return sorted(selected, key=lambda x: float(x.get('start_time', 0)))
    
    def _create_clips(self, audio_file: Path, xcid: str, 
                     detections: List[Dict]) -> List[ClipInfo]:
        """Create clip files from detections"""
        clips = []
        
        # Load audio once
        try:
            y, sr = librosa.load(str(audio_file), sr=None, mono=True)
            duration = len(y) / sr
        except Exception as e:
            print(f"Error loading {audio_file.name}: {e}")
            return []
        
        for idx, det in enumerate(detections, 1):
            start = float(det.get('start_time', 0))
            end = float(det.get('end_time', 0))
            confidence = float(det.get('confidence', 0))
            
            # Center clip on detection midpoint
            mid = (start + end) / 2
            clip_start = max(0, mid - self.clip_duration / 2)
            clip_end = min(duration, clip_start + self.clip_duration)
            clip_start = max(0, clip_end - self.clip_duration)  # Adjust if at end
            
            # Extract clip
            start_sample = int(clip_start * sr)
            end_sample = int(clip_end * sr)
            clip_audio = y[start_sample:end_sample]
            
            # Save clip
            clip_filename = f"{self.slug}_{xcid}_{idx:02d}.wav"
            clip_path = self.clips_dir / clip_filename
            
            sf.write(str(clip_path), clip_audio, sr)
            
            clips.append(ClipInfo(
                xcid=xcid,
                clip_index=idx,
                start_time=clip_start,
                end_time=clip_end,
                confidence=confidence,
                source_file=audio_file,
                clip_file=clip_path
            ))
        
        return clips
    
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
            "-t", str(self.threads),    # Use short form for threads
            "-b", str(self.batch_size)  # Use short form for batch_size
        ]
        
        print(f"Generating embeddings with {self.threads} threads...")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error generating embeddings: {result.stderr}")
                return None
            
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
                                 original_metadata: pd.DataFrame):
        """Create metadata file aligned with embeddings order"""
        
        # Read embeddings to get the order
        embeddings_df = pd.read_csv(embeddings_csv)
        
        # Parse the file paths to get xcid and clip_index
        ordered_clips = []
        for _, row in embeddings_df.iterrows():
            filepath = Path(row.iloc[0])
            # Extract xcid and clip_index from filename
            parts = filepath.stem.split('_')
            if len(parts) >= 3:
                xcid = parts[-2]
                clip_idx = int(parts[-1])
                
                # Find matching clip
                for clip in clips:
                    if clip.xcid == xcid and clip.clip_index == clip_idx:
                        ordered_clips.append(clip)
                        break
        
        # Create metadata DataFrame in the same order as embeddings
        metadata_rows = []
        for clip in ordered_clips:
            # Get original recording metadata
            orig_meta = original_metadata[original_metadata['xcid'] == clip.xcid]
            
            if not orig_meta.empty:
                orig_dict = orig_meta.iloc[0].to_dict()
            else:
                orig_dict = {}
            
            # Combine with clip info
            row = {
                'xcid': clip.xcid,
                'clip_index': clip.clip_index,
                'confidence': clip.confidence,
                'clip_start_s': clip.start_time,
                'clip_end_s': clip.end_time,
                **{k: v for k, v in orig_dict.items() 
                   if k != 'xcid'}  # Only exclude xcid since we already have it
            }
            metadata_rows.append(row)
        
        # Save aligned metadata
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_path = self.embeddings_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        print(f"Saved aligned metadata with {len(metadata_df)} entries")


def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from file or use defaults"""
    
    default_config = {
        "species": {
            "scientific_name": "Sylvia atricapilla",
            "common_name": "Blackcap",
            "slug": "sylvia_atricapilla"
        },
        "paths": {
            "root": "/Volumes/Z Slim/zslim_birdcluster"
        },
        "processing": {
            "max_clips_per_recording": 5,
            "clip_duration_sec": 3.0,
            "min_confidence": 0.20,
            "merge_within_sec": 1.0,
            "delete_originals": True
        },
        "birdnet": {
            "version": "2.4",
            "threads": 4,
            "batch_size": 16
        }
    }
    
    if config_path and config_path.exists():
        with open(config_path) as f:
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
    parser = argparse.ArgumentParser(description="Process bird recordings: detect, clip, embed")
    parser.add_argument("--config", type=Path, help="Path to config.yaml file")
    parser.add_argument("--species", type=str, help="Scientific name")
    parser.add_argument("--skip-confirm", action="store_true", 
                       help="Skip confirmation before deleting originals")
    parser.add_argument("--threads", type=int, help="Number of threads for processing")
    parser.add_argument("--batch-size", type=int, help="Batch size for BirdNET")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.species:
        config["species"]["scientific_name"] = args.species
        config["species"]["slug"] = args.species.lower().replace(" ", "_")
    if args.threads:
        config["birdnet"]["threads"] = args.threads
    if args.batch_size:
        config["birdnet"]["batch_size"] = args.batch_size
    
    # Run processor
    processor = SpeciesProcessor(config)
    processor.process(skip_confirm=args.skip_confirm)


if __name__ == "__main__":
    main()