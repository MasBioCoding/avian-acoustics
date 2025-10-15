#!/usr/bin/env python
"""
UMAP Yellowhammer Visualization with Interactive Zoom
Bokeh Server Application - COMPLETE VERSION

To run:
1. Save this file as 'xc_scripts/umap_app.py'
2. Start audio server in separate terminal:
    cd "/path/to/clips/<species_slug>"
    for me:
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/sylvia_atricapilla"
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/turdus_merula"
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/corvus_corax"
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/passer_montanus"
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/passer_domesticus"
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/parus_major"
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/buteo_buteo"
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/fringilla_coelebs"
    python3 -m http.server 8765
3. Run the Bokeh app:
    cd /path/to/birdnet_data_pipeline
    for me: cd /Users/masjansma/Desktop/birdnetcluster1folder/birdnet_data_pipeline
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_limosa_limosa.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_emberiza_citrinella.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_fringilla_coelebs.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_sylvia_atricapilla.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_turdus_merula.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_parus_major.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_corvus_corax.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_passer_montanus.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_passer_domesticus.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_chloris_chloris.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_strix_aluco.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_asio_otus.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_curruca_communis.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_cettia_cetti.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_phylloscopus_collybita.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_phylloscopus_trochilus.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_acrocephalus_scirpaceus.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_buteo_buteo.yaml
    bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config.yaml
    Or without config (uses defaults):
    bokeh serve --show xc_scripts/umap_app.py
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Optional
from importlib_metadata import metadata
import pandas as pd
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import traceback
import colorsys

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, Button, Div, DateRangeSlider, HoverTool, BoxSelectTool,
    TapTool, Toggle, Select, CheckboxGroup, CustomJS, RangeSlider, CDSView, BooleanFilter,
    Spinner
)
from bokeh.plotting import figure
import hdbscan

print("=" * 80)
print("STARTING BOKEH SERVER APP")
print("=" * 80)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------


def load_config(config_path: Optional[Path] = None):
    """Load configuration from file or use defaults"""
    
    # Default configuration with analysis parameters
    default_config = {
        "species": {
            "scientific_name": "Emberiza citrinella",
            "common_name": "Yellowhammer",
            "slug": "emberiza_citrinella"
        },
        "paths": {
            "root": "/Volumes/Z Slim/zslim_birdcluster"
        },
        "audio": {
            "base_url": "http://localhost:8765",
            "port": 8765
        },
        "analysis": {
            "umap_n_neighbors": 10,
            "umap_min_dist": 0.0,
            "umap_n_components": 2,
            "kmeans_clusters": 4,
            "kmeans_n_init": 10,
            "point_size": 10,
            "point_alpha": 0.3
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

# Parse command line arguments to get config file
parser = argparse.ArgumentParser(description="UMAP Visualization App")
parser.add_argument("--config", type=Path, help="Path to config.yaml file")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Set paths based on config
ROOT_PATH = Path(config["paths"]["root"]).expanduser()
SPECIES_SLUG = config["species"]["slug"]
EMBEDDINGS_FILE = ROOT_PATH / "embeddings" / SPECIES_SLUG / "embeddings.csv"
METADATA_FILE = ROOT_PATH / "embeddings" / SPECIES_SLUG / "metadata.csv"
CLIPS_DIR = ROOT_PATH / "clips" / SPECIES_SLUG
AUDIO_BASE_URL = config.get("audio", {}).get("base_url", "http://localhost:8765")

# UMAP parameters from config
ANALYSIS_PARAMS = config.get("analysis", {})

print(f"Configuration loaded:")
print(f"  Species: {config['species']['scientific_name']} ({config['species']['common_name']})")
print(f"  Embeddings: {EMBEDDINGS_FILE}")
print(f"  Metadata: {METADATA_FILE}")
print(f"  Clips: {CLIPS_DIR}")
print(f"  Audio URL: {AUDIO_BASE_URL}")
print(f"  Analysis parameters:")
print(f"    UMAP neighbors: {ANALYSIS_PARAMS.get('umap_n_neighbors', 10)}")
print(f"    UMAP min_dist: {ANALYSIS_PARAMS.get('umap_min_dist', 0.0)}")
print(f"    UMAP n_components: {ANALYSIS_PARAMS.get('umap_n_components', 2)}")
print(f"    KMeans clusters: {ANALYSIS_PARAMS.get('kmeans_clusters', 4)}")
print(f"    KMeans n_init: {ANALYSIS_PARAMS.get('kmeans_n_init', 10)}")
print(f"    Point size: {ANALYSIS_PARAMS.get('point_size', 10)}")
print(f"    Point alpha: {ANALYSIS_PARAMS.get('point_alpha', 0.3)}")

# -----------------------------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------------------------

class AppState:
    """Manages the application state"""
    def __init__(self):
        self.original_embeddings = None
        self.original_meta = None
        self.current_embeddings = None
        self.current_meta = None
        self.current_indices = None
        self.mapper = None
        self.projection = None
        self.is_zoomed = False  # Simple flag instead of complex tracking
        self.current_umap_params = {  # Track current UMAP parameters
            'n_neighbors': ANALYSIS_PARAMS.get('umap_n_neighbors', 10),
            'min_dist': ANALYSIS_PARAMS.get('umap_min_dist', 0.0)
        }

state = AppState()
print("AppState initialized")


class FilterManager:
    """Centralises all visibility filtering for the application."""

    def __init__(
        self,
        source: ColumnDataSource,
        filters: list[BooleanFilter],
        stats_callback,
        base_alpha: float,
    ) -> None:
        self.source = source
        self.filters = filters
        self.stats_callback = stats_callback
        self.base_alpha = base_alpha
        self.current_alpha = base_alpha
        self.dataframe = pd.DataFrame()
        self.available: dict[str, set[str] | None] = {}
        self.selected: dict[str, set[str] | None] = {}
        self.time_range: tuple[float, float] = (0.0, 24.0)
        self.full_time_range: tuple[float, float] = (0.0, 24.0)
        self.date_range: tuple[pd.Timestamp | None, pd.Timestamp | None] = (None, None)
        self.full_date_range: tuple[pd.Timestamp | None, pd.Timestamp | None] = (None, None)
        self.current_mask: np.ndarray = np.array([], dtype=bool)

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------
    def update_dataset(self, df: pd.DataFrame) -> None:
        """Replace the managed dataset and reset all filters."""

        self.dataframe = df.reset_index(drop=True)
        n_points = len(self.dataframe)
        mask = np.ones(n_points, dtype=bool)
        self.current_mask = mask

        # Ensure the backing filters know about the new length
        booleans = mask.tolist()
        for flt in self.filters:
            flt.booleans = booleans

        # Determine available categories for each filter dimension
        self.available = {
            'season': set(self.dataframe['season'].dropna().astype(str)) if 'season' in self.dataframe else None,
            'cluster': set(self.dataframe['kmeans3_str'].dropna().astype(str)) if 'kmeans3_str' in self.dataframe else None,
            'hdbscan': set(self.dataframe['hdbscan_str'].dropna().astype(str)) if 'hdbscan_str' in self.dataframe else None,
            'sex': set(self.dataframe['sex'].dropna().astype(str)) if 'sex' in self.dataframe else None,
            'type': set(self.dataframe['type'].dropna().astype(str)) if 'type' in self.dataframe else None,
        }

        self.selected = {
            key: (set(values) if values is not None else None)
            for key, values in self.available.items()
        }

        # Reset time and date ranges
        time_values = pd.to_numeric(self.dataframe.get('time_hour', pd.Series([], dtype=float)), errors='coerce')
        if len(time_values) and np.isfinite(time_values).any():
            valid_times = time_values[np.isfinite(time_values)]
            self.full_time_range = (float(valid_times.min()), float(valid_times.max()))
        else:
            self.full_time_range = (0.0, 24.0)
        self.time_range = self.full_time_range

        ts_values = pd.to_numeric(self.dataframe.get('ts', pd.Series([], dtype=float)), errors='coerce')
        valid_ts = ts_values[np.isfinite(ts_values)]
        if len(valid_ts):
            start = pd.to_datetime(valid_ts.min(), unit='ms')
            end = pd.to_datetime(valid_ts.max(), unit='ms')
            self.full_date_range = (start, end)
        else:
            self.full_date_range = (None, None)
        self.date_range = self.full_date_range

        self._update_alpha(mask)

    # ------------------------------------------------------------------
    # Filter state updates
    # ------------------------------------------------------------------
    def set_allowed(self, key: str, allowed: set[str]) -> None:
        if self.available.get(key) is None:
            return
        self.selected[key] = set(allowed)
        self.apply()

    def set_time_range(self, start: float, end: float) -> None:
        self.time_range = (start, end)
        self.apply()

    def set_date_range(self, start: pd.Timestamp | None, end: pd.Timestamp | None) -> None:
        self.date_range = (start, end)
        self.apply()

    def set_alpha(self, alpha: float) -> None:
        self.current_alpha = alpha
        self.apply(update_mask=False)

    def clear_hdbscan(self) -> None:
        self.available['hdbscan'] = None
        self.selected['hdbscan'] = None
        self.apply()

    def update_hdbscan(self, labels: list[str]) -> None:
        values = set(labels)
        self.available['hdbscan'] = values
        self.selected['hdbscan'] = set(values)
        self.apply()

    # ------------------------------------------------------------------
    # Core filtering
    # ------------------------------------------------------------------
    def apply(self, update_mask: bool = True) -> None:
        if self.dataframe.empty:
            return

        if update_mask:
            mask = np.ones(len(self.dataframe), dtype=bool)

            for key, column in (
                ('season', 'season'),
                ('cluster', 'kmeans3_str'),
                ('hdbscan', 'hdbscan_str'),
                ('sex', 'sex'),
                ('type', 'type'),
            ):
                allowed = self.selected.get(key)
                available = self.available.get(key)
                if available is not None and allowed is not None:
                    mask &= self.dataframe[column].astype(str).isin(allowed)

            # Time filter (allow invalid values represented by negatives)
            time_values = pd.to_numeric(self.dataframe.get('time_hour'), errors='coerce')
            if len(time_values):
                low, high = self.time_range
                valid_time = (time_values < 0) | ((time_values >= low) & (time_values <= high))
                mask &= valid_time.fillna(True).to_numpy()

            # Date filter operates on timestamps in milliseconds
            ts_values = pd.to_numeric(self.dataframe.get('ts'), errors='coerce')
            start, end = self.date_range
            if start is not None and end is not None and len(ts_values):
                start_ms = start.value // 10**6
                end_ms = end.value // 10**6
                within_range = (ts_values >= start_ms) & (ts_values <= end_ms)
                mask &= within_range.fillna(True).to_numpy()

            self.current_mask = mask
        else:
            mask = self.current_mask

        self._update_alpha(mask)

    # ------------------------------------------------------------------
    def _update_alpha(self, mask: np.ndarray) -> None:
        booleans = mask.tolist()
        for flt in self.filters:
            flt.booleans = booleans

        alpha = np.where(mask, self.current_alpha, 0.0).tolist()
        new_data = dict(self.source.data)
        new_data['alpha'] = alpha
        self.source.data = new_data

        if self.stats_callback is not None:
            self.stats_callback(mask)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def time_to_color(time_str):
    """Convert time string (HH:MM) to a color representing day/night cycle"""
    try:
        if pd.isna(time_str) or time_str == '':
            return "#808080"  # Gray for unknown
        
        # Parse time
        parts = str(time_str).split(':')
        if len(parts) < 2:
            return "#808080"
        
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        
        # Convert to fraction of day (0-1)
        time_fraction = (hour + minute/60) / 24
        
        # Create a color gradient for day/night cycle
        # Night (0:00-4:00): Dark blue to purple
        # Dawn (4:00-8:00): Purple to orange
        # Day (8:00-16:00): Orange to yellow to light blue
        # Dusk (16:00-20:00): Light blue to purple
        # Night (20:00-24:00): Purple to dark blue
        
        if hour < 4:  # Night
            hue = 0.65  # Blue
            saturation = 0.8
            lightness = 0.2 + (hour/4) * 0.2
        elif hour < 8:  # Dawn
            hue = 0.75 - ((hour-4)/4) * 0.15  # Purple to orange
            saturation = 0.7
            lightness = 0.3 + ((hour-4)/4) * 0.3
        elif hour < 12:  # Morning
            hue = 0.15 - ((hour-8)/4) * 0.05  # Orange to yellow
            saturation = 0.6
            lightness = 0.6
        elif hour < 16:  # Afternoon
            hue = 0.10 + ((hour-12)/4) * 0.45  # Yellow to light blue
            saturation = 0.5
            lightness = 0.65
        elif hour < 20:  # Dusk
            hue = 0.55 + ((hour-16)/4) * 0.20  # Light blue to purple
            saturation = 0.6
            lightness = 0.5 - ((hour-16)/4) * 0.2
        else:  # Night
            hue = 0.75 - ((hour-20)/4) * 0.10  # Purple to blue
            saturation = 0.8
            lightness = 0.3 - ((hour-20)/4) * 0.1
            
        # Convert HSL to RGB to hex
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
        
        return hex_color
        
    except:
        return "#808080"  # Gray for any error

def update_stats_display(metadata, stats_div, zoom_level, date_slider=None, source=None):
    """Update the statistics display with UMAP and selection info"""
    dates = pd.to_datetime(metadata['date'], errors='coerce')
    valid_dates = dates[dates.notna()]
    
    # Season counts for UMAP data
    months = dates.dt.month
    spring_count = months.isin([4, 5]).sum()
    summer_count = months.isin([6, 7, 8]).sum()
    autumn_count = months.isin([9, 10, 11]).sum()
    winter_count = months.isin([12, 1, 2, 3]).sum()
    
    # Date range of UMAP data
    if len(valid_dates) > 0:
        date_min = valid_dates.min().strftime('%Y-%m-%d')
        date_max = valid_dates.max().strftime('%Y-%m-%d')
        date_range_str = f"{date_min} to {date_max}"
    else:
        date_range_str = "No valid dates"
    
    # Get currently visible points if source is provided
    visible_info = ""
    if source is not None and 'alpha' in source.data:
        visible_count = sum(1 for a in source.data['alpha'] if a > 0)
        visible_info = f" (Visible: {visible_count})"
    
    # Get date slider range if provided
    slider_info = ""
    if date_slider is not None:
        try:
            start_date = pd.Timestamp(date_slider.value[0], unit='ms').strftime('%Y-%m-%d')
            end_date = pd.Timestamp(date_slider.value[1], unit='ms').strftime('%Y-%m-%d')
            slider_info = f"<br><b>Date Filter:</b> {start_date} to {end_date}"
        except:
            pass
        
    # Include species name in the display
    species_info = f"{config['species']['scientific_name']}"
    if config['species'].get('common_name'):
        species_info += f" ({config['species']['common_name']})"
    
    stats_html = f"""
    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border: 1px solid #dee2e6;">
        <b>Current UMAP Statistics</b> (Zoom Level: {zoom_level})<br>
        <div style="display: flex; gap: 20px; margin-top: 8px;">
            <div><b>Total Points:</b> {len(metadata)}{visible_info}</div>
            <div><b>UMAP Date Range:</b> {date_range_str}</div>
            <div style="display: flex; gap: 10px;">
                <span style="color: #88CC88;">Spring: {spring_count}</span>
                <span style="color: #F6C667;">Summer: {summer_count}</span>
                <span style="color: #D98859;">Autumn: {autumn_count}</span>
                <span style="color: #6BAED6;">Winter: {winter_count}</span>
            </div>
        </div>
        {slider_info}
    </div>
    """
    stats_div.text = stats_html

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def load_data():
    """Load embeddings and metadata"""
    print("\n" + "-" * 40)
    print("LOADING DATA...")
    
    try:
        # Check if files exist
        if not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        if not METADATA_FILE.exists():
            raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
        
        # Load embeddings
        print("  Loading embeddings CSV...")
        embeddings_df = pd.read_csv(EMBEDDINGS_FILE)
        print(f"  Loaded {len(embeddings_df)} rows")
        
        print("  Parsing embedding strings...")
        embeddings = embeddings_df['embedding'].apply(lambda x: np.fromstring(x, sep=',')).values
        embeddings_array = np.stack(embeddings)
        print(f"  Embeddings array shape: {embeddings_array.shape}")
        
        # Load metadata
        print("  Loading metadata CSV...")
        metadata = pd.read_csv(METADATA_FILE)
        print(f"  Metadata shape: {metadata.shape}")
        
        # Add audio URLs based on clip filenames
        if 'clip_file' not in metadata.columns:
            # If no clip_file column, create from xcid and clip_index
            metadata['audio_url'] = metadata.apply(
                lambda row: f"{AUDIO_BASE_URL}/{SPECIES_SLUG}_{row['xcid']}_{row['clip_index']:02d}.wav",
                axis=1
            )
        else:
            # Use the actual filename from clip_file
            metadata['audio_url'] = metadata['clip_file'].apply(
                lambda f: f"{AUDIO_BASE_URL}/{Path(f).name}"
            )
        
        # Store in state
        state.original_embeddings = embeddings_array
        state.original_meta = metadata.copy()
        state.current_embeddings = embeddings_array
        state.current_meta = metadata.copy()
        state.current_indices = np.arange(len(embeddings_array))
        
        print(f"Successfully loaded {len(embeddings_array)} embeddings")
        return embeddings_array, metadata
        
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        traceback.print_exc()
        raise

def compute_initial_umap(embeddings_array, metadata, n_neighbors=None, min_dist=None):
    """Compute initial UMAP projection and clustering"""
    print("\n" + "-" * 40)
    print("COMPUTING INITIAL UMAP...")
    
    try:
        # Only compute if we haven't already
        if state.projection is None:
            # UMAP projection with parameters from config or user input
            print("  Fitting UMAP...")
            
            # Use provided parameters or fall back to config/defaults
            if n_neighbors is None:
                n_neighbors = ANALYSIS_PARAMS.get("umap_n_neighbors", 10)
            if min_dist is None:
                min_dist = ANALYSIS_PARAMS.get("umap_min_dist", 0.0)
            n_components = ANALYSIS_PARAMS.get("umap_n_components", 2)
            
            print(f"    n_neighbors: {n_neighbors}")
            print(f"    min_dist: {min_dist}")
            print(f"    n_components: {n_components}")
            
            state.mapper = umap.UMAP(
                n_neighbors=n_neighbors, 
                min_dist=min_dist, 
                n_components=n_components, 
                n_jobs=1
            )
            state.projection = state.mapper.fit_transform(embeddings_array)
            print(f"  UMAP projection shape: {state.projection.shape}")
            
            # K-means clustering
            print("  Computing K-means clustering...")
            n_clusters = ANALYSIS_PARAMS.get("kmeans_clusters", 4)
            n_init = ANALYSIS_PARAMS.get("kmeans_n_init", 10)
            
            print(f"    n_clusters: {n_clusters}")
            print(f"    n_init: {n_init}")
            
            kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
            km_labels = kmeans.fit_predict(state.projection)
            
            metadata['kmeans3'] = km_labels
            metadata['kmeans3_str'] = metadata['kmeans3'].astype(str)
            
            sil = silhouette_score(state.projection, km_labels)
            print(f"  Silhouette score: {sil:.3f}")
            
            print("UMAP computation complete")
        else:
            print("  Using existing UMAP projection")
            
        return state.projection, metadata
        
    except Exception as e:
        print(f"  ERROR computing UMAP: {e}")
        traceback.print_exc()
        raise


# -----------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------

def prepare_hover_data(metadata, projection):
    """Prepare data for hover tooltips and interactions"""
    print("\n" + "-" * 40)
    print("PREPARING HOVER DATA...")
    
    try:
        # Parse dates
        print("  Parsing dates...")
        dates = pd.to_datetime(pd.Series(metadata['date']), errors='coerce')
        months = dates.dt.month
        valid = dates.notna()
        ts_ms = (dates.astype('int64', copy=False) // 10**6).astype('float64')
        ts_ms[~valid] = np.nan
        
        # Define seasons
        print("  Assigning seasons...")
        SPRING_MONTHS = {4, 5}
        SUMMER_MONTHS = {6, 7, 8}
        AUTUMN_MONTHS = {9, 10, 11}
        WINTER_MONTHS = {12, 1, 2, 3}
        
        is_spring = months.isin(SPRING_MONTHS).fillna(False)
        is_summer = months.isin(SUMMER_MONTHS).fillna(False)
        is_autumn = months.isin(AUTUMN_MONTHS).fillna(False)
        is_winter = months.isin(WINTER_MONTHS).fillna(False)
        
        # Assign season labels
        season_arr = np.array(['Unknown'] * len(metadata), dtype=object)
        season_arr[is_spring] = 'Spring'
        season_arr[is_summer] = 'Summer'
        season_arr[is_autumn] = 'Autumn'
        season_arr[is_winter] = 'Winter'
        
        # Add to metadata for consistency
        metadata['season'] = season_arr
        
        # Convert coordinates to Web Mercator for map
        print("  Converting coordinates to Web Mercator...")
        def lonlat_to_mercator_arrays(lon_deg, lat_deg):
            lon = pd.to_numeric(lon_deg, errors='coerce')
            lat = pd.to_numeric(lat_deg, errors='coerce')
            k = 6378137.0
            x = lon * (np.pi/180.0) * k
            y = np.log(np.tan((np.pi/4.0) + (lat * (np.pi/180.0) / 2.0))) * k
            return x, y
        
        x3857, y3857 = lonlat_to_mercator_arrays(metadata['lon'], metadata['lat'])
        
        # Color palettes
        print("  Assigning colors...")
        season_palette = {
            "Spring": "#88CC88",
            "Summer": "#F6C667", 
            "Autumn": "#D98859",
            "Winter": "#6BAED6",
            "Unknown": "#999999"
        }
        
        cluster_palette = ["#66c2a5", "#8da0cb", "#fc8d62", "#e78ac3", 
                          "#a6d854", "#ffd92f", "#a1d99b", "#9ecae1"]
        
        # Assign colors for different categories
        factors = sorted(metadata['kmeans3_str'].unique())
        print(f"    Cluster factors: {factors}")
        
        cluster_color_map = {lab: cluster_palette[i % len(cluster_palette)] 
                            for i, lab in enumerate(factors)}
        
        # Season colors
        season_colors = [season_palette.get(s, "#999999") for s in season_arr]
        
        # Cluster colors
        cluster_colors = [cluster_color_map.get(str(c), "#999999") 
                         for c in metadata['kmeans3_str']]
        
        # Sex colors
        sex_palette = {"M": "#4B8BBE", "F": "#F07C7C", "?": "#A0A0A0"}
        sex_colors = [sex_palette.get(str(s).upper(), "#A0A0A0") for s in metadata['sex']]
        sex_labels = [str(s) if pd.notna(s) else "?" for s in metadata['sex']]
        
        # Type colors (song, call, etc.)
        type_palette = {
            "S": "#FF6B6B",  # Song - Red
            "C": "#4ECDC4",  # Call - Teal
            "F": "#FFE66D",  # Flight - Yellow
            "A": "#95E77E",  # Alarm - Green
            "?": "#A0A0A0"   # Unknown - Gray
        }
        type_colors = [type_palette.get(str(t).upper(), "#A0A0A0") for t in metadata['type']]
        type_labels = [str(t) if pd.notna(t) else "?" for t in metadata['type']]
        
        # Time of day colors (continuous gradient)
        time_colors = [time_to_color(t) for t in metadata['time']]
        
        # Parse time to hours for filtering
        time_hours = []
        for t in metadata['time']:
            try:
                if pd.notna(t) and t != '':
                    parts = str(t).split(':')
                    hour = float(parts[0]) + (float(parts[1])/60 if len(parts) > 1 else 0)
                    time_hours.append(hour)
                else:
                    time_hours.append(-1)  # Invalid time
            except:
                time_hours.append(-1)
        
        # Initial alpha (visibility)
        alpha = np.full(len(metadata), ANALYSIS_PARAMS.get("point_alpha", 0.3))

        # Build data dictionary
        print("  Building data dictionary...")
        data = {
            'x': projection[:, 0],
            'y': projection[:, 1],
            'xcid': metadata['xcid'],
            'sex': sex_labels,
            'type': type_labels,
            'cnt': metadata['country'],
            'lat': metadata['lat'],
            'lon': metadata['lon'],
            'x3857': x3857,
            'y3857': y3857,
            'alt': metadata['alt'],
            'date': metadata['date'],
            'time': metadata['time'],
            'time_hour': time_hours,
            'also': metadata['also'],
            'rmk': metadata['remarks'],
            'month': months,
            'ts': ts_ms,
            'valid_date': valid,
            'kmeans3': metadata['kmeans3'],
            'kmeans3_str': metadata['kmeans3_str'],
            'audio_url': metadata['audio_url'] if 'audio_url' in metadata else [''] * len(metadata),
            'season': season_arr,
            'season_color': season_colors,
            'cluster_color': cluster_colors,
            'sex_color': sex_colors,
            'type_color': type_colors,
            'time_color': time_colors,
            'active_color': season_colors,  # Start with season colors
            'alpha': alpha,
            'original_index': np.arange(len(metadata))
        }

        df = pd.DataFrame(data)
        print(f"Dataframe created with {len(df.columns)} columns")
        return df, factors
        
    except Exception as e:
        print(f"  ERROR preparing data: {e}")
        traceback.print_exc()
        raise

# -----------------------------------------------------------------------------
# CREATE PLOTS
# -----------------------------------------------------------------------------

def create_umap_plot(source: ColumnDataSource, visible_filter: BooleanFilter):
    """Create the main UMAP scatter plot"""
    print("\n" + "-" * 40)
    print("CREATING UMAP PLOT...")
    
    point_size = ANALYSIS_PARAMS.get("point_size", 10)
    point_alpha = ANALYSIS_PARAMS.get("point_alpha", 0.3)
    
    p = figure(
        title="UMAP Projection",
        width=600, height=600,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        background_fill_color="white",
        border_fill_color="#ffe88c"
    )
    
    # Add box select tool
    box_select = BoxSelectTool()
    p.add_tools(box_select)
    p.toolbar.active_drag = box_select
    
    # Filter selections to only include visible points
    selection_filter_callback = CustomJS(args=dict(source=source), code="""
        const indices = source.selected.indices;
        if (indices.length === 0) return;
        
        const alpha = source.data['alpha'];
        const filtered = [];
        
        for (let i of indices) {
            if (alpha[i] > 0) {
                filtered.push(i);
            }
        }
        
        if (filtered.length !== indices.length) {
            source.selected.indices = filtered;
        }
    """)
    source.selected.js_on_change('indices', selection_filter_callback)
    
    view = CDSView(filter=visible_filter)

    scatter = p.scatter('x', 'y', source=source, view=view,
                        size=point_size,
                        fill_color={'field':'active_color'},
                        line_color=None,
                        alpha='alpha',
                        hover_line_color="black", 
                        hover_alpha=1.0, 
                        hover_line_width=1.5)
        
    # Hover tool - will only work on visible points (alpha > 0)
    hover = HoverTool(
        renderers=[scatter],
        tooltips=[
            ("xcid", "@xcid"),
            ("season", "@season"),
            ("date", "@date"),
            ("time", "@time"),
            ("cluster", "@kmeans3_str"),
            ("sex", "@sex"),
            ("type", "@type"),
            ("lat, lon", "@lat, @lon")
        ]
    )
    
    p.add_tools(hover)
    p.add_tools(TapTool())
    
    print("  UMAP plot created")
    return p, hover, view

def create_map_plot(source: ColumnDataSource, visible_filter: BooleanFilter):
    """Create the geographic map plot"""
    print("\n" + "-" * 40)
    print("CREATING MAP PLOT...")
    
    # Europe bounds
    lon_min, lon_max = -25.0, 40.0
    lat_min, lat_max = 34.0, 72.0
    
    def lonlat_to_mercator(lon, lat):
        k = 6378137.0
        x = lon * (np.pi/180.0) * k
        y = np.log(np.tan((np.pi/4.0) + (lat * (np.pi/180.0) / 2.0))) * k
        return x, y
    
    x0, y0 = lonlat_to_mercator(lon_min, lat_min)
    x1, y1 = lonlat_to_mercator(lon_max, lat_max)
    
    map_fig = figure(
        title="Recording locations (Europe)",
        width=600, height=600,
        x_axis_label="Longitude", y_axis_label="Latitude",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        match_aspect=True,
        background_fill_color="#ffe88c",
        border_fill_color="#ffe88c"
    )
    
    try:
        map_fig.add_tile("CartoDB Positron", retina=True)
    except:
        pass
    
    map_fig.x_range.start, map_fig.x_range.end = x0, x1
    map_fig.y_range.start, map_fig.y_range.end = y0, y1
    
    hv_line_width = 1.5
    
    view = CDSView(filter=visible_filter)

    map_scatter = map_fig.scatter('x3857','y3857', source=source, view=view,
                                 size=8, # Slightly bigger points on map
                                 fill_color={'field':'active_color'},
                                 line_color=None,
                                 alpha='alpha',
                                 hover_line_color="black",
                                 hover_alpha=1.0, 
                                 hover_line_width=1.5)
    
    # Store view reference

    # Hover tool
    hover_map = HoverTool(
        renderers=[map_scatter],
        tooltips=[
            ("xcid", "@xcid"),
            ("season", "@season"),
            ("date", "@date"),
            ("time", "@time"),
            ("lat, lon", "@lat, @lon"),
            ("alt", "@alt")
        ]
    )
        
    map_fig.add_tools(hover_map)
    map_fig.add_tools(TapTool())
    
    print("  Map plot created")
    return map_fig, hover_map, view # Return views too

# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------

def create_app():
    """Create the Bokeh application"""
    print("\n" + "=" * 80)
    print("CREATING BOKEH APPLICATION")
    print("=" * 80)
    
    try:
        # Load data
        embeddings_array, metadata = load_data()
        projection, metadata = compute_initial_umap(embeddings_array, metadata)
        
        # Prepare data
        data_df, factors = prepare_hover_data(metadata, projection)
        source = ColumnDataSource(data=data_df.to_dict(orient='list'))
        print(f"\nColumnDataSource created with {len(source.data['x'])} points")

        # Create shared visibility filters for the plots
        initial_mask = [True] * len(data_df)
        umap_filter = BooleanFilter(booleans=initial_mask.copy())
        map_filter = BooleanFilter(booleans=initial_mask.copy())

        # Create plots with shared filters
        umap_plot, umap_hover, umap_view = create_umap_plot(source, umap_filter)
        map_plot, map_hover, map_view = create_map_plot(source, map_filter)
        
        # --- CREATE ALL WIDGETS ---
        print("\n" + "-" * 40)
        print("CREATING WIDGETS...")
        
        # Statistics display
        stats_div = Div(width=1400, height=60,
                       styles={'margin-bottom': '10px'})
        update_stats_display(metadata, stats_div, 0)  # Initial stats
        
        # Zoom controls (should this be at top or bottom of widget order? what about with callback order?)
        zoom_button = Button(label="Zoom to Selection", button_type="success", width=150)
        reset_button = Button(label="Reset to Full Dataset", button_type="warning", width=150)
        zoom_status = Div(text="Viewing: Full dataset", width=400, height=30,
                         styles={'padding':'4px', 'background':'#f0f0f0', 'border-radius':'4px'})
        
        # Get unique values for categorical filters
        unique_seasons = sorted(set(source.data['season']))
        unique_clusters = sorted(set(source.data['kmeans3_str']))
        unique_sex = sorted(set(source.data['sex']))
        unique_type = sorted(set(source.data['type']))

        
        # --- CHECKBOXES and TIMERANGE SLIDER---
        # Season checkboxes
        season_checks = CheckboxGroup(
            labels=unique_seasons,
            active=list(range(len(unique_seasons))),  # All active by default
            visible=True,  # Hidden by default
            name="season_checks"
        )

        # Cluster checkboxes
        cluster_checks = CheckboxGroup(
            labels=unique_clusters,
            active=list(range(len(unique_clusters))),
            visible=False,
            name="cluster_checks"
        )
        
        # Sex checkboxes
        sex_checks = CheckboxGroup(
            labels=unique_sex,
            active=list(range(len(unique_sex))),
            visible=False,  # Hidden by default
            name="sex_checks"
        )
        
        # Type checkboxes
        type_checks = CheckboxGroup(
            labels=unique_type,
            active=list(range(len(unique_type))),
            visible=False,  # Hidden by default
            name="type_checks"
        )
        
        # Time range slider (0-24 hours) (no check for len(unique times), necessary?( tested and does indeed cause the same bug, maybe fix later))
        time_range_slider = RangeSlider(
            start=0, end=24, value=(0, 24), step=0.5,
            title="Time of day (hours)",
            visible=False,  # Hidden by default
            width=300
        )
        
        # Color selector - expanded options
        color_select = Select(
            title="Color by", 
            value="Season", 
            options=["Season", "KMeans", "HDBSCAN", "Sex", "Type", "Time of Day"]
        )
        
        # --- SLIDERS ---
        # Bounds slider (first create valid range)
        dates = pd.to_datetime(pd.Series(source.data['date']), errors='coerce')
        valid = dates.notna()

        if valid.any():
            # Use ACTUAL min/max from the data
            start_dt = dates[valid].min()
            end_dt = dates[valid].max()
        else:
            # Fallback if no valid dates
            start_dt = pd.Timestamp("2000-01-01")
            end_dt = pd.Timestamp("2025-01-01")

        # Bounds control slider - this controls the range of the main slider
        date_bounds_slider = DateRangeSlider(
            title="Adjust date slider range (zoom timeline)",
            start=start_dt,  # Actual earliest recording
            end=end_dt,      # Actual latest recording
            value=(start_dt, end_dt),  # Start with full range
            step=1, 
            width=1400
        )

        # Main date filter slider
        date_slider = DateRangeSlider(
            title="Filter recordings between",
            start=start_dt, end=end_dt,
            value=(start_dt, end_dt),
            step=1, width=1400
        )

        reset_bounds_btn = Button(label="Reset timeline range", button_type="default", width=150)
        
        # --- OTHER WIDGETS ---
        # Hover toggle
        hover_toggle = Toggle(label="Show hover info", active=True)
        
        # Audio controls
        test_audio_btn = Button(label="Test audio server", button_type="primary")
        audio_status = Div(text="Audio server: <i>not tested</i>", width=500, height=50,
                          styles={'border':'1px solid #ddd','padding':'6px','background':'#fff'})
        
        # Playlist panel
        playlist_panel = Div(width=420, height=320,
                           styles={'border':'1px solid #ddd','padding':'8px','background':'#fff'})
        playlist_panel.text = "<i>Click a point in either plot to list nearby recordings...</i>"
        
        # Alpha toggle for point transparency
        alpha_toggle = Toggle(label="Full opacity", active=False, width=120)

        # UMAP parameter inputs
        umap_params_div = Div(text="<b>UMAP Parameters:</b>", width=200)
        umap_neighbors_input = Spinner(
            title="Nearest neighbors", 
            low=2, high=100, step=1, 
            value=ANALYSIS_PARAMS.get('umap_n_neighbors', 10),
            width=150
        )
        umap_mindist_input = Spinner(
            title="Min distance", 
            low=0.0, high=1.0, step=0.01, 
            value=ANALYSIS_PARAMS.get('umap_min_dist', 0.0),
            width=150
        )
        
        # HDBSCAN parameters box
        hdbscan_params_div = Div(text="<b>HDBSCAN Parameters:</b>", width=200)
        hdbscan_min_cluster_size = Spinner(
            title="Min cluster size",
            low=2, high=500, step=1,
            value=15,
            width=150
        )
        hdbscan_min_samples = Spinner(
            title="Min samples",
            low=1, high=100, step=1,
            value=5,
            width=150
        )
        hdbscan_apply_btn = Button(
            label="Apply HDBSCAN",
            button_type="primary",
            width=150
        )
        hdbscan_stats_div = Div(width=300, height=30, text="<i>HDBSCAN not yet computed</i>")

        # HDBSCAN checkboxes (initially hidden)
        hdbscan_checks = CheckboxGroup(
            labels=[],
            active=[],
            visible=False,
            name="hdbscan_checks"
        )

        def _to_datetime(ts):
            if ts is None or (isinstance(ts, float) and np.isnan(ts)):
                return None
            return pd.Timestamp(ts).to_pydatetime()

        filter_manager = FilterManager(
            source=source,
            filters=[umap_filter, map_filter],
            stats_callback=None,
            base_alpha=ANALYSIS_PARAMS.get('point_alpha', 0.3),
        )

        def stats_callback(mask: np.ndarray) -> None:
            try:
                visible_meta = state.current_meta.loc[mask]
            except Exception:
                visible_meta = state.current_meta
            update_stats_display(visible_meta, stats_div, 1 if state.is_zoomed else 0, date_slider, source)

        def refresh_controls_from_filter_manager() -> None:
            time_start, time_end = filter_manager.full_time_range
            time_range_slider.start = time_start
            time_range_slider.end = time_end
            time_range_slider.value = (time_start, time_end)

            full_start, full_end = filter_manager.full_date_range
            if full_start is not None and full_end is not None:
                start_dt = _to_datetime(full_start)
                end_dt = _to_datetime(full_end)
                date_bounds_slider.start = start_dt
                date_bounds_slider.end = end_dt
                date_bounds_slider.value = (start_dt, end_dt)
                date_slider.start = start_dt
                date_slider.end = end_dt
                date_slider.value = (start_dt, end_dt)
                date_slider.title = f"Filter recordings between ({start_dt.date()} to {end_dt.date()})"
            else:
                date_slider.title = "Filter recordings between"

        filter_manager.stats_callback = stats_callback
        filter_manager.update_dataset(data_df)
        refresh_controls_from_filter_manager()

        print("  All widgets created")

        # --- SETUP CALLBACKS ---
        print("\n" + "-" * 40)
        print("SETTING UP CALLBACKS...")

        # --- CHECKBOX CALLBACKS ---

        def _selected_labels(widget: CheckboxGroup) -> set[str]:
            labels = list(widget.labels)
            return {labels[i] for i in widget.active if i < len(labels)}

        def on_season_change(attr, old, new):
            filter_manager.set_allowed('season', _selected_labels(season_checks))

        def on_cluster_change(attr, old, new):
            filter_manager.set_allowed('cluster', _selected_labels(cluster_checks))

        def on_sex_change(attr, old, new):
            filter_manager.set_allowed('sex', _selected_labels(sex_checks))

        def on_type_change(attr, old, new):
            filter_manager.set_allowed('type', _selected_labels(type_checks))

        def on_hdbscan_change(attr, old, new):
            filter_manager.set_allowed('hdbscan', _selected_labels(hdbscan_checks))

        season_checks.on_change('active', on_season_change)
        cluster_checks.on_change('active', on_cluster_change)
        sex_checks.on_change('active', on_sex_change)
        type_checks.on_change('active', on_type_change)
        hdbscan_checks.on_change('active', on_hdbscan_change)

        # --- SLIDER CALLBACKS ---

        def on_time_range_change(attr, old, new):
            start, end = new
            filter_manager.set_time_range(float(start), float(end))

        def on_date_change(attr, old, new):
            start = pd.Timestamp(new[0]) if new and new[0] is not None else None
            end = pd.Timestamp(new[1]) if new and new[1] is not None else None
            filter_manager.set_date_range(start, end)

        def on_bounds_change(attr, old, new):
            start, end = new
            date_slider.start = start
            date_slider.end = end
            current_start, current_end = date_slider.value
            new_start = max(current_start, start)
            new_end = min(current_end, end)
            if (new_start, new_end) != date_slider.value:
                date_slider.value = (new_start, new_end)
            if start and end:
                start_label = pd.Timestamp(start).date()
                end_label = pd.Timestamp(end).date()
                date_slider.title = f"Filter recordings between ({start_label} to {end_label})"

        def on_reset_bounds():
            start, end = filter_manager.full_date_range
            if start is None or end is None:
                return
            start_dt = _to_datetime(start)
            end_dt = _to_datetime(end)
            date_bounds_slider.value = (start_dt, end_dt)
            date_slider.start = start_dt
            date_slider.end = end_dt
            date_slider.value = (start_dt, end_dt)
            date_slider.title = f"Filter recordings between ({start_dt.date()} to {end_dt.date()})"

        time_range_slider.on_change('value', on_time_range_change)
        date_slider.on_change('value', on_date_change)
        date_bounds_slider.on_change('value', on_bounds_change)
        reset_bounds_btn.on_click(on_reset_bounds)

        # --- COLOR SELECTION ---

        color_columns = {
            "Season": "season_color",
            "KMeans": "cluster_color",
            "HDBSCAN": "hdbscan_color",
            "Sex": "sex_color",
            "Type": "type_color",
            "Time of Day": "time_color",
        }

        def on_color_change(attr, old, new):
            if new == "HDBSCAN" and 'hdbscan_color' not in source.data:
                hdbscan_stats_div.text = "<b>HDBSCAN:</b> Compute clusters to enable colouring."
                target = old if old in color_select.options else "Season"
                if color_select.value != target:
                    color_select.value = target
                return

            for widget in (season_checks, cluster_checks, hdbscan_checks, sex_checks, type_checks):
                widget.visible = False
            time_range_slider.visible = False

            if new == "Season":
                season_checks.visible = True
            elif new == "KMeans":
                cluster_checks.visible = True
            elif new == "HDBSCAN":
                hdbscan_checks.visible = True
            elif new == "Sex":
                sex_checks.visible = True
            elif new == "Type":
                type_checks.visible = True
            elif new == "Time of Day":
                time_range_slider.visible = True

            column = color_columns.get(new)
            if column is None or column not in source.data:
                return

            new_data = dict(source.data)
            new_data['active_color'] = list(source.data[column])
            source.data = new_data

        color_select.on_change('value', on_color_change)

# --- OTHER CALLBACKS ---
        # Hover toggle callback
        hover_toggle_callback = CustomJS(args=dict(
            h_u=umap_hover, h_m=map_hover, toggle=hover_toggle
        ), code="""
            const showInfo = toggle.active;
            if (!window.original_umap_tooltips) {
                window.original_umap_tooltips = h_u.tooltips;
                window.original_map_tooltips = h_m.tooltips;
            }
            if (h_u) h_u.tooltips = showInfo ? window.original_umap_tooltips : null;
            if (h_m) h_m.tooltips = showInfo ? window.original_map_tooltips : null;
        """)
        hover_toggle.js_on_change('active', hover_toggle_callback)
        
        # Audio test callback
        test_audio_callback = CustomJS(args=dict(src=source, status=audio_status, base=AUDIO_BASE_URL), code="""
            let testUrl = "";
            const d = src.data;
            if (d['audio_url']) {
                for (let i = 0; i < d['audio_url'].length; i++) {
                    const u = d['audio_url'][i];
                    if (u && typeof u === 'string' && u.trim().length) { 
                        testUrl = u; 
                        break; 
                    }
                }
            }
            if (!testUrl && base) testUrl = String(base);
            
            if (!testUrl) {
                status.text = "Audio server: No URL available to test.";
                return;
            }
            
            testUrl += (testUrl.includes("?") ? "&" : "?") + "t=" + Date.now();
            status.text = "Audio server: testing...";
            
            const a = new Audio();
            let done = false;
            const finish = (ok, msg) => {
                if (done) return; 
                done = true;
                status.text = ok ? "Audio server: reachable" : "Audio server: " + msg;
            };
            
            a.oncanplaythrough = () => finish(true);
            a.onerror = () => finish(false, "unreachable");
            a.src = testUrl;
            a.load();
        """)
        test_audio_btn.js_on_click(test_audio_callback)
        
        
        # Playlist callback - FIXED to only include visible points
        # Track plot clicks
        umap_plot.js_on_event('tap', CustomJS(args=dict(src=source), code="""
            src.data._ctx = 'umap';
        """))
        
        map_plot.js_on_event('tap', CustomJS(args=dict(src=source), code="""
            src.data._ctx = 'map';
        """))
        
        playlist_callback = CustomJS(args=dict(src=source, pane=playlist_panel), code="""
            const d = src.data;
            const inds = src.selected.indices;
            if (!inds.length) return;
            const i = inds[0];
            
            // Check if the clicked point is visible
            if (d['alpha'][i] <= 0) {
                pane.text = "<i>Selected point is hidden by current filters</i>";
                return;
            }
            
            const KM_RADIUS = 15.0;
            const UMAP_RADIUS = 0.4;
            
            const toRad = (deg) => deg * Math.PI / 180.0;
            function hav_km(lat1, lon1, lat2, lon2) {
                if (!isFinite(lat1) || !isFinite(lon1) || 
                    !isFinite(lat2) || !isFinite(lon2)) return Infinity;
                const R = 6371.0088;
                const dlat = toRad(lat2 - lat1);
                const dlon = toRad(lon2 - lon1);
                const a = Math.sin(dlat/2)**2 +
                        Math.cos(toRad(lat1))*Math.cos(toRad(lat2))*Math.sin(dlon/2)**2;
                return 2 * R * Math.asin(Math.sqrt(a));
            }
            
            const ctx = d._ctx || 'umap';
            const N = d['xcid'].length;
            const items = [];
            let centerInfo = "";
            
            if (ctx === 'map') {
                const lat0 = Number(d['lat'][i]);
                const lon0 = Number(d['lon'][i]);
                centerInfo = `Map click – ${KM_RADIUS} km radius`;
                
                for (let j = 0; j < N; j++) {
                    // Only include visible points
                    if (d['alpha'][j] <= 0) continue;
                    
                    const km = hav_km(lat0, lon0, Number(d['lat'][j]), Number(d['lon'][j]));
                    if (km <= KM_RADIUS) items.push([j, km]);
                }
                items.sort((a,b) => a[1]-b[1]);
            } else {
                const x0 = Number(d['x'][i]);
                const y0 = Number(d['y'][i]);
                centerInfo = `UMAP click – radius ${UMAP_RADIUS}`;
                
                for (let j = 0; j < N; j++) {
                    // Only include visible points
                    if (d['alpha'][j] <= 0) continue;
                    
                    const dx = Number(d['x'][j]) - x0;
                    const dy = Number(d['y'][j]) - y0;
                    const r = Math.hypot(dx, dy);
                    if (isFinite(r) && r <= UMAP_RADIUS) items.push([j, r]);
                }
                items.sort((a,b) => a[1]-b[1]);
            }
            
            if (!items.length) {
                pane.text = `<b>${centerInfo}</b><br>No visible neighbors found.`;
                return;
            }
            
            let html = `<b>${centerInfo}</b> – ${items.length} visible recordings<br>
                    <div style="max-height:280px; overflow:auto; margin-top:8px;">`;
            
            for (const [j, dist] of items) {
                const xcid = d['xcid'][j];
                const date = d['date'][j];
                const url = d['audio_url'] ? d['audio_url'][j] : "";
                
                html += `<div style="display:flex;align-items:center;margin:4px 0;">
                    <button onclick="(function(u){
                        if(!u) return;
                        if(window._BN_audio) window._BN_audio.pause();
                        window._BN_audio = new Audio(u);
                        window._BN_audio.play();
                    })('${url}')">Play</button>
                    <div style="margin-left:8px">
                        <b>${xcid}</b><br>
                        <small>${date}</small>
                    </div>
                </div>`;
            }
            
            html += '</div>';
            pane.text = html;
        """)
        source.selected.js_on_change('indices', playlist_callback)
        
        def on_alpha_toggle(attr, old, new):
            alpha_value = 1.0 if alpha_toggle.active else ANALYSIS_PARAMS.get('point_alpha', 0.3)
            filter_manager.set_alpha(alpha_value)

        alpha_toggle.on_change('active', on_alpha_toggle)
        
        # Zoom and reset callbacks
        # Define zoom functions inline in create_app scope
        def zoom_to_visible():
            """Recompute UMAP on only the currently visible points"""
            print("\n[ZOOM] Computing UMAP on visible points...")
            
            # Get selected indices that are also visible
            selected = source.selected.indices
            if not selected:
                zoom_status.text = "No points selected"
                return
                
            # Filter to only visible selected points
            visible_selected = [i for i in selected if source.data['alpha'][i] > 0]
            
            if len(visible_selected) < 10:
                zoom_status.text = f"Need at least 10 visible points (found {len(visible_selected)})"
                return
            
            print(f"[ZOOM] Selected {len(selected)} points, {len(visible_selected)} are visible")
            
            # Get the original data indices for the visible selected points
            actual_indices = [state.current_indices[i] for i in visible_selected]
            
            # Get subset of original data
            subset_embeddings = state.original_embeddings[actual_indices]
            subset_meta = state.original_meta.iloc[actual_indices].copy().reset_index(drop=True)
            
            print(f"[ZOOM] Computing UMAP on {len(actual_indices)} points from original data")
            
            # Recompute UMAP with user-specified parameters
            n_neighbors = min(umap_neighbors_input.value, len(actual_indices) - 1)
            min_dist = umap_mindist_input.value
            
            print(f"[ZOOM] Using UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")
            
            subset_mapper = umap.UMAP(
                n_neighbors=n_neighbors, 
                min_dist=min_dist, 
                n_components=2, 
                n_jobs=1
            )
            new_projection = subset_mapper.fit_transform(subset_embeddings)
            
            # Recompute clusters (this should also change when redefining kmeans nclusters)
            if len(actual_indices) >= 4:
                n_clusters = min(4, len(actual_indices))
                kmeans = KMeans(n_clusters=n_clusters, n_init=10)
                new_labels = kmeans.fit_predict(new_projection)
                subset_meta['kmeans3'] = new_labels
                subset_meta['kmeans3_str'] = subset_meta['kmeans3'].astype(str)
            
            # Update state (not sure yet how indices play into this)
            state.current_embeddings = subset_embeddings
            state.current_meta = subset_meta
            state.current_indices = np.array(actual_indices)
            state.projection = new_projection
            state.is_zoomed = True
            
            had_hdbscan = 'hdbscan_str' in source.data

            # Update visualization
            zoom_df, _ = prepare_hover_data(subset_meta, new_projection)

            print(f"[ZOOM] New seasons: {sorted(zoom_df['season'].astype(str).unique())}")
            print(f"[ZOOM] New clusters: {sorted(zoom_df['kmeans3_str'].astype(str).unique())}")

            source.data = zoom_df.to_dict(orient='list')
            filter_manager.update_dataset(zoom_df)
            refresh_controls_from_filter_manager()

            def _reset_widget(widget: CheckboxGroup, values: list[str]) -> None:
                widget.labels = values
                widget.active = list(range(len(values)))

            _reset_widget(season_checks, sorted(zoom_df['season'].astype(str).unique()))
            _reset_widget(cluster_checks, sorted(zoom_df['kmeans3_str'].astype(str).unique()))
            _reset_widget(sex_checks, sorted(zoom_df['sex'].astype(str).unique()))
            _reset_widget(type_checks, sorted(zoom_df['type'].astype(str).unique()))

            if had_hdbscan:
                compute_hdbscan_clustering()
            else:
                hdbscan_checks.labels = []
                hdbscan_checks.active = []
                filter_manager.clear_hdbscan()

            if color_select.value == "HDBSCAN" and 'hdbscan_color' not in source.data:
                color_select.value = "Season"

            zoom_status.text = f"Viewing: Zoomed subset ({len(actual_indices)} points)"
            print(f"[ZOOM] Complete. Showing {len(actual_indices)} points")
        
        def reset_to_full_dataset():
            """Simple reset - reload everything from scratch"""
            print("\n[RESET] Resetting to full dataset...")
            
            # Reset state to original
            state.current_embeddings = state.original_embeddings
            state.current_meta = state.original_meta.copy()
            state.current_indices = np.arange(len(state.original_embeddings))
            state.is_zoomed = False
            
            # Force recomputation by clearing the projection first
            state.projection = None  # This ensures compute_initial_umap will actually compute

            # Recompute fresh UMAP on full dataset with user-specified parameters
            n_neighbors = umap_neighbors_input.value
            min_dist = umap_mindist_input.value

            print(f"[RESET] Using UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")

            state.projection, state.current_meta = compute_initial_umap(
                state.original_embeddings,
                state.current_meta,
                n_neighbors=n_neighbors,
                min_dist=min_dist
            )

            full_df, _ = prepare_hover_data(state.current_meta, state.projection)

            source.data = full_df.to_dict(orient='list')
            filter_manager.update_dataset(full_df)
            refresh_controls_from_filter_manager()

            def _reset_widget(widget: CheckboxGroup, values: list[str]) -> None:
                widget.labels = values
                widget.active = list(range(len(values)))

            _reset_widget(season_checks, sorted(full_df['season'].astype(str).unique()))
            _reset_widget(cluster_checks, sorted(full_df['kmeans3_str'].astype(str).unique()))
            _reset_widget(sex_checks, sorted(full_df['sex'].astype(str).unique()))
            _reset_widget(type_checks, sorted(full_df['type'].astype(str).unique()))

            hdbscan_checks.labels = []
            hdbscan_checks.active = []
            filter_manager.clear_hdbscan()

            if color_select.value == "HDBSCAN":
                color_select.value = "Season"

            zoom_status.text = "Viewing: Full dataset"
            print(f"[RESET] Complete - showing {len(state.current_meta)} points")
        
        # Simple zoom callbacks
        def on_zoom_click():
            selected = source.selected.indices
            if selected:
                # Only zoom on selected AND visible points
                visible_selected = [i for i in selected if source.data['alpha'][i] > 0]
                if len(visible_selected) < 10:
                    zoom_status.text = "Need at least 10 visible points selected"
                    source.selected.indices = []
                    return
                
                zoom_to_visible()
                source.selected.indices = []
        
        def on_reset_click():
            reset_to_full_dataset()
        
        zoom_button.on_click(on_zoom_click)
        reset_button.on_click(on_reset_click)
        
        def compute_hdbscan_clustering():
            """Compute HDBSCAN clustering on current projection"""
            print("\n[HDBSCAN] Computing clusters...")

            min_cluster_size = hdbscan_min_cluster_size.value
            min_samples = hdbscan_min_samples.value

            print(f"  min_cluster_size: {min_cluster_size}")
            print(f"  min_samples: {min_samples}")

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )

            cluster_labels = clusterer.fit_predict(state.projection)
            cluster_labels_str = [str(label) if label >= 0 else "Noise" for label in cluster_labels]

            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = sum(1 for label in cluster_labels if label == -1)

            print(f"  Found {n_clusters} clusters")
            print(f"  Noise points: {n_noise} ({100*n_noise/len(cluster_labels):.1f}%)")

            unique_hdbscan = sorted(set(cluster_labels_str))
            hdbscan_palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                               "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
            hdbscan_color_map = {}
            for i, label in enumerate(unique_hdbscan):
                if label == "Noise":
                    hdbscan_color_map[label] = "#CCCCCC"
                else:
                    hdbscan_color_map[label] = hdbscan_palette[i % len(hdbscan_palette)]

            df = filter_manager.dataframe.copy()
            df['hdbscan'] = cluster_labels.tolist()
            df['hdbscan_str'] = cluster_labels_str
            df['hdbscan_color'] = [hdbscan_color_map[str(c)] for c in cluster_labels_str]

            if color_select.value == "HDBSCAN":
                df['active_color'] = df['hdbscan_color']

            source.data = df.to_dict(orient='list')
            filter_manager.dataframe = df
            filter_manager.update_hdbscan(cluster_labels_str)

            hdbscan_checks.labels = unique_hdbscan
            hdbscan_checks.active = list(range(len(unique_hdbscan)))

            if color_select.value == "HDBSCAN" and 'hdbscan_color' in source.data:
                new_data = dict(source.data)
                new_data['active_color'] = list(source.data['hdbscan_color'])
                source.data = new_data

            hdbscan_stats_div.text = (
                f"<b>HDBSCAN:</b> {n_clusters} clusters, {n_noise} noise points "
                f"({100*n_noise/len(cluster_labels):.1f}%)"
            )



        # Add callback for apply button
        hdbscan_apply_btn.on_click(compute_hdbscan_clustering)
        
        print("  All callbacks set up")
        
        # --- LAYOUT ---
        print("\n" + "-" * 40)
        print("CREATING LAYOUT...")
        
        # Create UMAP parameters box
        umap_params_box = column(
            umap_params_div,
            umap_neighbors_input,
            umap_mindist_input,
            styles={'border': '1px solid #ddd', 'padding': '10px', 'border-radius': '5px'}
        )
        
        # Create HDBSCAN parameters box
        hdbscan_params_box = column(
            hdbscan_params_div,
            hdbscan_min_cluster_size,
            hdbscan_min_samples,
            hdbscan_apply_btn,
            hdbscan_stats_div,
            styles={'border': '1px solid #ddd', 'padding': '10px', 'border-radius': '5px'}
        )
        
        zoom_controls = row(zoom_button, reset_button, alpha_toggle, zoom_status)
        
        # Create a column that contains all filter widgets
        filter_widgets = column(season_checks, cluster_checks, hdbscan_checks, sex_checks, type_checks, time_range_slider)
        controls = row(color_select, filter_widgets, hover_toggle, test_audio_btn, umap_params_box, hdbscan_params_box)
        
        plots = row(umap_plot, map_plot, playlist_panel)
        date_bounds_controls = row(date_bounds_slider, reset_bounds_btn)
        
        layout = column(
            stats_div,  # Statistics at the top
            zoom_controls,
            controls,
            audio_status,
            date_bounds_controls,  # Timeline zoom control
            date_slider,  # Main date filter
            plots,
            sizing_mode="stretch_width"
        )
        
        print("  Layout created successfully")
        print("\n" + "=" * 80)
        print("BOKEH APP READY!")
        print("=" * 80)
        
        return layout
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        error_div = Div(text=f"<h2>Error: {str(e)}</h2>", width=800, height=600)
        return column(error_div)

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

doc = curdoc()
doc.title = f"UMAP {config['species']['common_name'] or config['species']['scientific_name']} Visualization"
layout = create_app()
doc.add_root(layout)

print("\nServer ready at: http://localhost:5006/umap_yellowhammer_app")

#   bokeh serve --show umap_yellowhammer_app.py
