#!/usr/bin/env python
"""
UMAP Yellowhammer Visualization with Interactive Zoom
Bokeh Server Application - COMPLETE VERSION

To run:
1. Save this file as 'xc_scripts/umap_app.py'
2. Start audio server in separate terminal:
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/sylvia_atricapilla"
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/turdus_merula"
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/corvus_corax"
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/passer_montanus"
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/passer_domesticus"
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/parus_major"
    cd "/Volumes/Z Slim/zslim_birdcluster/clips/buteo_buteo"
    python3 -m http.server 8765
3. Run the Bokeh app:
    cd /Users/masjansma/Desktop/birdnetcluster1folder/xc_pipeline
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
        
        # Season visibility flags        
        season_on = [True] * len(metadata)

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
        earliest_date = dates[valid].min() if valid.any() else pd.Timestamp("2000-01-01")
        alpha = np.where(valid & (dates >= earliest_date), 0.3, 0.0)
        
        # Build data dictionary
        print("  Building data dictionary...")
        data = {
            'x': projection[:, 0].tolist(),
            'y': projection[:, 1].tolist(),
            'xcid': metadata['xcid'].tolist(),
            'sex': sex_labels,
            'type': type_labels,
            'cnt': metadata['country'].tolist(),
            'lat': metadata['lat'].tolist(),
            'lon': metadata['lon'].tolist(),
            'x3857': x3857.tolist(),
            'y3857': y3857.tolist(),
            'alt': metadata['alt'].tolist(),
            'date': metadata['date'].tolist(),
            'time': metadata['time'].tolist(),
            'time_hour': time_hours,
            'also': metadata['also'].tolist(),
            'rmk': metadata['remarks'].tolist(),
            'month': months.tolist(),
            'ts': ts_ms.tolist(),
            'valid_date': valid.tolist(),
            'kmeans3': metadata['kmeans3'].tolist(),
            'kmeans3_str': metadata['kmeans3_str'].tolist(),
            'hdbscan_on': [True] * len(metadata),
            'audio_url': metadata['audio_url'].tolist() if 'audio_url' in metadata else [''] * len(metadata),
            'season': season_arr.tolist(),
            'season_on': season_on,
            'season_color': season_colors,
            'cluster_color': cluster_colors,
            'sex_color': sex_colors,
            'type_color': type_colors,
            'time_color': time_colors,
            'active_color': season_colors,  # Start with season colors
            'alpha': alpha.tolist(),
            'alpha_base': alpha.tolist(),
            'cluster_on': [True] * len(metadata),
            'sex_on': [True] * len(metadata),
            'type_on': [True] * len(metadata),
            'time_on': [True] * len(metadata),
            'original_index': np.arange(len(metadata)).tolist()
        }

        print(f"Data dictionary created with {len(data)} keys")
        return data, factors
        
    except Exception as e:
        print(f"  ERROR preparing data: {e}")
        traceback.print_exc()
        raise

# -----------------------------------------------------------------------------
# CREATE PLOTS
# -----------------------------------------------------------------------------

def create_umap_plot(source):
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
    
    # Create a CDSView with boolean filter based on alpha
    view_filter = BooleanFilter(booleans=[a > 0 for a in source.data['alpha']])
    view = CDSView(filter=view_filter)
    
    # Single scatter renderer using the view
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

def create_map_plot(source):
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
    
    # Create a CDSView with boolean filter based on alpha
    view_filter = BooleanFilter(booleans=[a > 0 for a in source.data['alpha']])
    view = CDSView(filter=view_filter)
    
    # Single scatter renderer using the view
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
        data, factors = prepare_hover_data(metadata, projection)
        source = ColumnDataSource(data=data)
        print(f"\nColumnDataSource created with {len(source.data['x'])} points")
        
        # Create plots - now they return season views
        umap_plot, umap_hover, umap_view = create_umap_plot(source)
        map_plot, map_hover, map_view = create_map_plot(source)
        
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
        
        print("  All widgets created")
        
        # --- SETUP CALLBACKS ---
        print("\n" + "-" * 40)
        print("SETTING UP CALLBACKS...")
        
        # --- CHECKBOX CALLBACKS ---
        # Season checkbox callback - exactly like cluster callback
        season_callback = CustomJS(args=dict(src=source, cb=season_checks,
                                   umap_view=umap_view, map_view=map_view), code="""
            const d = src.data;
            const labs = cb.labels;  // Read current labels from widget
            const active_indices = cb.active;
            
            // Build set of active labels
            const active = new Set();
            for (let idx of active_indices) {
                if (idx < labs.length) {
                    active.add(labs[idx]);
                }
            }
            
            const season = d['season'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const cluster_on = d['cluster_on'] || new Array(season.length).fill(true);
            const hdbscan_on = d['hdbscan_on'] || new Array(season.length).fill(true);  // ADD THIS
            const sex_on = d['sex_on'] || new Array(season.length).fill(true);
            const type_on = d['type_on'] || new Array(season.length).fill(true);
            const time_on = d['time_on'] || new Array(season.length).fill(true);
            
            const n = season.length;
            for (let i = 0; i < n; i++) {
                const season_visible = active.has(String(season[i]));
                d['season_on'][i] = season_visible;

                if (cluster_on[i] && hdbscan_on[i] && sex_on[i] && type_on[i] && time_on[i] && season_visible) {
                    alpha[i] = alpha_base[i];
                } else {
                    alpha[i] = 0.0;
                }
            }
            
            // Update views at the end
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }
            src.change.emit();
        """)
        season_checks.js_on_change('active', season_callback)
        
        # Cluster checkbox callback - reads labels dynamically
        cluster_callback = CustomJS(args=dict(src=source, cb=cluster_checks,
                                   umap_view=umap_view, map_view=map_view), code="""
            const d = src.data;
            const labs = cb.labels;  // Read current labels from widget
            const active_indices = cb.active;
            
            // Build set of active labels
            const active = new Set();
            for (let idx of active_indices) {
                if (idx < labs.length) {
                    active.add(labs[idx]);
                }
            }
            
            const km = d['kmeans3_str'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const hdbscan_on = d['hdbscan_on'] || new Array(km.length).fill(true);
            const sex_on = d['sex_on'] || new Array(km.length).fill(true);
            const type_on = d['type_on'] || new Array(km.length).fill(true);
            const time_on = d['time_on'] || new Array(km.length).fill(true);
            const season_on = d['season_on'] || new Array(km.length).fill(true);
            
            const n = km.length;
            for (let i = 0; i < n; i++) {
                const cluster_visible = active.has(String(km[i]));
                d['cluster_on'][i] = cluster_visible;

                if (cluster_visible && hdbscan_on[i] && season_on[i] && sex_on[i] && type_on[i] && time_on[i]) {
                    alpha[i] = alpha_base[i];
                } else {
                    alpha[i] = 0.0;
                }
            }
            
            // Update views at the end
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }
            src.change.emit();
        """)
        cluster_checks.js_on_change('active', cluster_callback)

        # Sex checkbox callback - reads labels dynamically
        sex_callback = CustomJS(args=dict(src=source, cb=sex_checks,
                                   umap_view=umap_view, map_view=map_view), code="""
            const d = src.data;
            const labs = cb.labels;  // Read current labels from widget
            const active_indices = cb.active;
            
            // Build set of active labels
            const active = new Set();
            for (let idx of active_indices) {
                if (idx < labs.length) {
                    active.add(labs[idx]);
                }
            }
            
            const sex = d['sex'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const cluster_on = d['cluster_on'] || new Array(sex.length).fill(true);
            const hdbscan_on = d['hdbscan_on'] || new Array(sex.length).fill(true);
            const type_on = d['type_on'] || new Array(sex.length).fill(true);
            const time_on = d['time_on'] || new Array(sex.length).fill(true);
            const season_on = d['season_on'] || new Array(sex.length).fill(true);

            
            const n = sex.length;
            for (let i = 0; i < n; i++) {
                const sex_visible = active.has(String(sex[i]));
                d['sex_on'][i] = sex_visible;

                if (cluster_on[i] && hdbscan_on[i] && sex_visible && type_on[i] && time_on[i] && season_on[i]) {
                    alpha[i] = alpha_base[i];
                } else {
                    alpha[i] = 0.0;
                }
            }
            
            // Update views at the end
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }
            src.change.emit();
        """)
        sex_checks.js_on_change('active', sex_callback)

        # Type checkbox callback - reads labels dynamically
        type_callback = CustomJS(args=dict(src=source, cb=type_checks,
                                   umap_view=umap_view, map_view=map_view), code="""
            const d = src.data;
            const labs = cb.labels;  // Read current labels from widget
            const active_indices = cb.active;
            
            // Build set of active labels
            const active = new Set();
            for (let idx of active_indices) {
                if (idx < labs.length) {
                    active.add(labs[idx]);
                }
            }
            
            const type = d['type'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const cluster_on = d['cluster_on'] || new Array(type.length).fill(true);
            const hdbscan_on = d['hdbscan_on'] || new Array(type.length).fill(true);
            const sex_on = d['sex_on'] || new Array(type.length).fill(true);
            const time_on = d['time_on'] || new Array(type.length).fill(true);
            const season_on = d['season_on'] || new Array(type.length).fill(true);

            const n = type.length;
            for (let i = 0; i < n; i++) {
                const type_visible = active.has(String(type[i]));
                d['type_on'][i] = type_visible;

                if (cluster_on[i] && hdbscan_on[i] && sex_on[i] && type_visible && time_on[i] && season_on[i]) {
                    alpha[i] = alpha_base[i];
                } else {
                    alpha[i] = 0.0;
                }
            }
            
            // Update views at the end
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }
            src.change.emit();
        """)
        type_checks.js_on_change('active', type_callback)
        
        # Time range slider callback (what would happen if selection contains invalid times?, like with checkboxes should we read dynamically?)
        time_range_callback = CustomJS(args=dict(src=source, slider=time_range_slider,
                                   umap_view=umap_view, map_view=map_view), code="""
            const d = src.data;
            const time_hour = d['time_hour'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            
            // Add fallback initialization for all _on fields
            const cluster_on = d['cluster_on'] || new Array(time_hour.length).fill(true);
            const hdbscan_on = d['hdbscan_on'] || new Array(time_hour.length).fill(true);
            const sex_on = d['sex_on'] || new Array(time_hour.length).fill(true);
            const type_on = d['type_on'] || new Array(time_hour.length).fill(true);
            const season_on = d['season_on'] || new Array(time_hour.length).fill(true);

            const min_hour = slider.value[0];
            const max_hour = slider.value[1];
            
            const n = time_hour.length;
            for (let i = 0; i < n; i++) {
                const hour = time_hour[i];
                // Allow invalid times (-1) or times within range
                const time_visible = (hour < 0) || (hour >= min_hour && hour <= max_hour);
                d['time_on'][i] = time_visible;
                
                // Alpha is visible only if ALL filters allow it
                if (cluster_on[i] && hdbscan_on[i] && sex_on[i] && type_on[i] && time_visible && season_on[i]) {
                    alpha[i] = alpha_base[i];
                } else {
                    alpha[i] = 0.0;
                }
            }
            
            // Update views at the end
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }
            src.change.emit();
        """)
        time_range_slider.js_on_change('value', time_range_callback)
        
        color_callback = CustomJS(args=dict(
            src=source, 
            sel=color_select,
            season_checks=season_checks,
            cluster_checks=cluster_checks,
            hdbscan_checks=hdbscan_checks,  # Add this
            sex_checks=sex_checks,
            type_checks=type_checks,
            time_slider=time_range_slider
        ), code="""
            const d = src.data;
            const mode = sel.value;
            
            // Hide all filter widgets first
            season_checks.visible = false;
            cluster_checks.visible = false;
            hdbscan_checks.visible = false;
            sex_checks.visible = false;
            type_checks.visible = false;
            time_slider.visible = false;
            
            // Map mode to color column and show appropriate widget
            let from_col;
            switch(mode) {
                case "Season":
                    from_col = "season_color";
                    season_checks.visible = true;
                    break;
                case "KMeans":
                    from_col = "cluster_color";
                    cluster_checks.visible = true;
                    break;
                case "HDBSCAN":
                    if (d['hdbscan_color']) {
                        from_col = "hdbscan_color";
                        hdbscan_checks.visible = true;
                    } else {
                        alert("HDBSCAN not yet computed. Please click 'Apply HDBSCAN' first.");
                        return;
                    }
                    break;
                case "Sex":
                    from_col = "sex_color";
                    sex_checks.visible = true;
                    break;
                case "Type":
                    from_col = "type_color";
                    type_checks.visible = true;
                    break;
                case "Time of Day":
                    from_col = "time_color";
                    time_slider.visible = true;
                    break;
            }
            
            // Update colors
            const n = d['active_color'].length;
            for (let i = 0; i < n; i++) {
                d['active_color'][i] = d[from_col][i];
            }
            src.change.emit();
        """)
        color_select.js_on_change('value', color_callback)

        # --- SLIDER CALLBACKS ---
        # Bounds slider callback - update main slider range
        bounds_callback = CustomJS(args=dict(
            bounds=date_bounds_slider,
            main=date_slider,
            source=source
        ), code="""
            // Get new bounds from the bounds slider
            const new_start = bounds.value[0];
            const new_end = bounds.value[1];
            
            // Update the main slider's range
            main.start = new_start;
            main.end = new_end;
            
            // Ensure current value stays within new bounds
            const current_start = main.value[0];
            const current_end = main.value[1];
            
            // Constrain current selection to new bounds if needed
            const adjusted_start = Math.max(current_start, new_start);
            const adjusted_end = Math.min(current_end, new_end);
            
            // Only update value if it changed
            if (adjusted_start != current_start || adjusted_end != current_end) {
                main.value = [adjusted_start, adjusted_end];
            }
            
            // Optional: Update the main slider's title to show the range
            const start_date = new Date(new_start).toISOString().split('T')[0];
            const end_date = new Date(new_end).toISOString().split('T')[0];
            main.title = `Filter recordings between (${start_date} to ${end_date})`;
        """)
        date_bounds_slider.js_on_change('value', bounds_callback)

        # Also add a reset button for the bounds slider
        reset_bounds_btn = Button(label="Reset timeline range", button_type="default", width=150)
        reset_bounds_callback = CustomJS(args=dict(
            bounds=date_bounds_slider,
            main=date_slider,
            source=source
        ), code="""
            // Find actual data bounds
            const dates = source.data['ts'];
            let data_min = Infinity;
            let data_max = -Infinity;
            
            for (let ts of dates) {
                if (!Number.isNaN(ts)) {
                    if (ts < data_min) data_min = ts;
                    if (ts > data_max) data_max = ts;
                }
            }
            
            // Reset both sliders to full data range
            bounds.value = [data_min, data_max];
            main.start = data_min;
            main.end = data_max;
            main.title = "Filter recordings between";
        """)
        reset_bounds_btn.js_on_click(reset_bounds_callback)
                
        # Date slider callback - update stats when slider changes
        date_callback = CustomJS(args=dict(src=source, s=date_slider, stats=stats_div,
                                   umap_view=umap_view, map_view=map_view), code="""
            const d = src.data;
            const ts = d['ts'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const cluster_on = d['cluster_on'] || new Array(ts.length).fill(true);
            const hdbscan_on = d['hdbscan_on'] || new Array(ts.length).fill(true);
            const sex_on = d['sex_on'] || new Array(ts.length).fill(true);
            const type_on = d['type_on'] || new Array(ts.length).fill(true);
            const time_on = d['time_on'] || new Array(ts.length).fill(true);
            const season_on = d['season_on'] || new Array(ts.length).fill(true);
            
            const cut0 = Number(s.value[0]);
            const cut1 = Number(s.value[1]);
            
            let visible_count = 0;
            for (let i = 0; i < ts.length; i++) {
                let base = Number.isNaN(ts[i]) ? 0.0 : 
                          (ts[i] >= cut0 && ts[i] <= cut1) ? 0.3 : 0;
                alpha_base[i] = base;
                
                // Final alpha depends on ALL filters
                if (cluster_on[i] && hdbscan_on[i] && sex_on[i] && type_on[i] && time_on[i] && season_on[i] && base > 0) {
                    alpha[i] = base;
                } else {
                    alpha[i] = 0.0;
                }
                
                if (alpha[i] > 0) visible_count++;
            }
            
            // Update stats display to show visible count
            const start_date = new Date(cut0).toISOString().split('T')[0];
            const end_date = new Date(cut1).toISOString().split('T')[0];
            
            // Keep existing HTML but update the visible count
            let html = stats.text;
            if (html.includes('(Visible:')) {
                html = html.replace(/\(Visible: \d+\)/, `(Visible: ${visible_count})`);
            } else {
                html = html.replace(/Total Points:<\/b> (\d+)/, 
                                  `Total Points:</b> $1 (Visible: ${visible_count})`);
            }
            
            // Update date filter line
            if (html.includes('Date Filter:')) {
                html = html.replace(/Date Filter:<\/b> [\d-]+ to [\d-]+/, 
                                  `Date Filter:</b> ${start_date} to ${end_date}`);
            } else {
                html = html.replace('</div>', 
                                  `<br><b>Date Filter:</b> ${start_date} to ${end_date}</div>`);
            }
            
            stats.text = html;
            
            
            // Update views at the end
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < ts.length; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < ts.length; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }
            src.change.emit();
        """)
        date_slider.js_on_change('value', date_callback)
        
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
        
        # Alpha toggle callback - instant change
        alpha_toggle_callback = CustomJS(args=dict(src=source, toggle=alpha_toggle), code="""
            const d = src.data;
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const new_alpha = toggle.active ? 1.0 : 0.3;
            
            // Update base alpha for all visible points
            for (let i = 0; i < alpha_base.length; i++) {
                if (alpha_base[i] > 0) {
                    alpha_base[i] = new_alpha;
                    // Also update current alpha if it's visible
                    if (alpha[i] > 0) {
                        alpha[i] = new_alpha;
                    }
                }
            }
            
            src.change.emit();
        """)
        alpha_toggle.js_on_change('active', alpha_toggle_callback)
        
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
            
            # Update visualization
            new_data, new_factors = prepare_hover_data(subset_meta, new_projection)
            
            # Get new unique values             
            new_unique_seasons = sorted(set(new_data['season']))
            new_unique_clusters = sorted(set(new_data['kmeans3_str']))
            new_unique_sex = sorted(set(new_data['sex']))
            new_unique_type = sorted(set(new_data['type']))
            
            print(f"[ZOOM] New seasons: {new_unique_seasons}")
            print(f"[ZOOM] New factors: {new_factors}")
            print(f"[ZOOM] New sex values: {new_unique_sex}")
            print(f"[ZOOM] New type values: {new_unique_type}")
            
            # All points start visible after zoom
            new_data['alpha'] = [0.3] * len(new_data['x'])
            new_data['alpha_base'] = [0.3] * len(new_data['x'])
            
            # Reset all filter states to True
            new_data['season_on'] = [True] * len(new_data['x'])
            new_data['cluster_on'] = [True] * len(new_data['x'])
            new_data['hdbscan_on'] = [True] * len(new_data['x'])
            new_data['sex_on'] = [True] * len(new_data['x'])
            new_data['type_on'] = [True] * len(new_data['x'])
            new_data['time_on'] = [True] * len(new_data['x'])
            
            # If HDBSCAN was computed, recompute for zoomed data
            if 'hdbscan_str' in source.data:
                # Recompute HDBSCAN on zoomed projection
                compute_hdbscan_clustering()  # This will update the zoomed data
            
            # When we zoom, and we recreate widgets, should this source data be changed before their creation? Does this currently happen?
            source.data = new_data
            
            # Update filter widgets with new unique values and reset to all active
            season_checks.labels = new_unique_seasons
            season_checks.active = list(range(len(new_unique_seasons)))

            cluster_checks.labels = new_unique_clusters
            cluster_checks.active = list(range(len(new_unique_clusters)))
            
            sex_checks.labels = new_unique_sex
            sex_checks.active = list(range(len(new_unique_sex)))
            
            type_checks.labels = new_unique_type
            type_checks.active = list(range(len(new_unique_type)))

            print(f"[ZOOM] Updated widgets - seasons: {len(new_unique_seasons)}, clusters: {len(new_unique_clusters)}, sex: {len(new_unique_sex)}, type: {len(new_unique_type)}")

            # Update displays
            update_stats_display(subset_meta, stats_div, 1, source=source)
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
            
            # Prepare full data with original factors
            new_data, original_factors = prepare_hover_data(state.current_meta, state.projection)
            
            # Get unique values for the full dataset
            original_unique_seasons = sorted(set(new_data['season']))            
            original_unique_clusters = sorted(set(new_data['kmeans3_str']))
            original_unique_sex = sorted(set(new_data['sex']))
            original_unique_type = sorted(set(new_data['type']))
            original_unique_seasons = sorted(set(new_data['season']))

            print(f"[RESET] Original seasons: {original_unique_seasons}")
            print(f"[RESET] Original factors: {original_factors}")
            print(f"[RESET] Original sex values: {original_unique_sex}")
            print(f"[RESET] Original type values: {original_unique_type}")
            
            # Reset all filter states to True for full dataset
            new_data['season_on'] = [True] * len(new_data['x'])
            new_data['cluster_on'] = [True] * len(new_data['x'])
            new_data['hdbscan_on'] = [True] * len(new_data['x'])
            new_data['sex_on'] = [True] * len(new_data['x'])
            new_data['type_on'] = [True] * len(new_data['x'])
            new_data['time_on'] = [True] * len(new_data['x'])
            
            # same comment as in zoom_to_visible
            source.data = new_data
            
            # Reset filter widgets with original values and set all active
            season_checks.labels = original_unique_seasons
            season_checks.active = list(range(len(original_unique_seasons)))
    
            cluster_checks.labels = original_unique_clusters
            cluster_checks.active = list(range(len(original_unique_clusters)))

            sex_checks.labels = original_unique_sex
            sex_checks.active = list(range(len(original_unique_sex)))
            
            type_checks.labels = original_unique_type
            type_checks.active = list(range(len(original_unique_type)))
            
            # Reset time range slider to full range
            time_range_slider.value = (0, 24)

            print(f"[RESET] Reset widgets - seasons: {len(original_unique_seasons)}, clusters: {len(original_unique_clusters)}, sex: {len(original_unique_sex)}, type: {len(original_unique_type)}")

            # Update displays
            update_stats_display(state.current_meta, stats_div, 0, source=source)
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
            
            # Run HDBSCAN on current projection
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            
            cluster_labels = clusterer.fit_predict(state.projection)
            
            # HDBSCAN uses -1 for noise points
            cluster_labels_str = [str(label) if label >= 0 else "Noise" for label in cluster_labels]
            
            # Calculate metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = sum(1 for label in cluster_labels if label == -1)
            
            print(f"  Found {n_clusters} clusters")
            print(f"  Noise points: {n_noise} ({100*n_noise/len(cluster_labels):.1f}%)")
            
            # Update the source data
            current_data = dict(source.data)
            current_data['hdbscan'] = cluster_labels.tolist()
            current_data['hdbscan_str'] = cluster_labels_str
            
            # Create color mapping for HDBSCAN
            unique_hdbscan = sorted(set(cluster_labels_str))
            hdbscan_palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", 
                            "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
            
            hdbscan_color_map = {}
            for i, label in enumerate(unique_hdbscan):
                if label == "Noise":
                    hdbscan_color_map[label] = "#CCCCCC"  # Gray for noise
                else:
                    hdbscan_color_map[label] = hdbscan_palette[i % len(hdbscan_palette)]
            
            current_data['hdbscan_color'] = [hdbscan_color_map[str(c)] for c in cluster_labels_str]
            current_data['hdbscan_on'] = [True] * len(cluster_labels)
            
            # Update HDBSCAN checkboxes
            hdbscan_checks.labels = unique_hdbscan
            hdbscan_checks.active = list(range(len(unique_hdbscan)))
            
            source.data = current_data
            
            # Update stats
            hdbscan_stats_div.text = f"<b>HDBSCAN:</b> {n_clusters} clusters, {n_noise} noise points ({100*n_noise/len(cluster_labels):.1f}%)"

        # HDBSCAN checkbox callback
        hdbscan_callback = CustomJS(args=dict(src=source, cb=hdbscan_checks,
                                umap_view=umap_view, map_view=map_view), code="""
            const d = src.data;
            const labs = cb.labels;
            const active_indices = cb.active;
            
            // Build set of active labels
            const active = new Set();
            for (let idx of active_indices) {
                if (idx < labs.length) {
                    active.add(labs[idx]);
                }
            }
            
            const hdbscan = d['hdbscan_str'];
            const alpha_base = d['alpha_base'];
            const alpha = d['alpha'];
            const season_on = d['season_on'] || new Array(hdbscan.length).fill(true);
            const cluster_on = d['cluster_on'] || new Array(hdbscan.length).fill(true);
            const sex_on = d['sex_on'] || new Array(hdbscan.length).fill(true);
            const type_on = d['type_on'] || new Array(hdbscan.length).fill(true);
            const time_on = d['time_on'] || new Array(hdbscan.length).fill(true);
            
            const n = hdbscan.length;
            for (let i = 0; i < n; i++) {
                const hdbscan_visible = active.has(String(hdbscan[i]));
                d['hdbscan_on'][i] = hdbscan_visible;
                
                if (season_on[i] && cluster_on[i] && sex_on[i] && type_on[i] && 
                    time_on[i] && hdbscan_visible) {
                    alpha[i] = alpha_base[i];
                } else {
                    alpha[i] = 0.0;
                }
            }
            
            // Update views
            if (typeof umap_view !== 'undefined' && umap_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                umap_view.filter.booleans = new_booleans;
            }
            if (typeof map_view !== 'undefined' && map_view.filter) {
                const new_booleans = [];
                for (let i = 0; i < n; i++) {
                    new_booleans.push(alpha[i] > 0);
                }
                map_view.filter.booleans = new_booleans;
            }
            src.change.emit();
        """)
        hdbscan_checks.js_on_change('active', hdbscan_callback)

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
