# BirdNET cluster Data Pipeline

This repository collects tooling for building clustered embeddings of bird vocalizations. It now bundles both the Xeno-Canto ingestion workflow and Raspberry Pi processing utilities in a single tree.

## Repository Layout

```
birdnet_data_pipeline/
├── environment.yml          # Conda specification for the pipeline runtime
├── conda_packages_ext.txt   # Optional personal package additions
├── README.md                # You are here
├── birdnet_pi/              # Raspberry Pi specific helpers
│   ├── process_species_pi.py
│   ├── umap_app_pi.py
│   ├── detections.csv       # BirdNET Pi detection metadata
│   └── pi_clips_embeds/     # Local Pi embeddings & clips (gitignored, user-supplied). Other paths possible.
├── xc_configs/              # Species-specific configuration templates
├── xc_scripts/              # Download, process, and visualization scripts
├── testbirdnet/             # Sample inputs/outputs for quick checks
└── BirdNET-Analyzer/        # BirdNET CLI source (installed inside the env)
```

> `birdnet_pi/pi_clips_embeds/` remains empty in version control; populate it with the assets I distribute separately.

The pipeline expects a writable data root (see `paths.root` in the configs) and will create three folders beneath it: `xc_downloads/`, `clips/`, and `embeddings/`. An external drive is recommended because audio files are numerous.

## Managing the Conda Environment

`environment.yml` at the repository root is the canonical specification for the end-to-end workflow. It pins Python 3.11 together with TensorFlow, librosa, UMAP, Bokeh, HDBSCAN, and command-line helpers such as `tqdm`. Always create or update your environment from this file to guarantee package compatibility with the scripts:

```bash
cd /path/to/birdnet_data_pipeline  # replace with your checkout path
conda env create -f environment.yml
# or for updating use
conda env update -f environment.yml
# but note pruning (--prune) requires reinstalling BirdNET-Analyzer dependencies manually
conda activate birdnetcluster1
```

If the environment already exists, replace the first command with `conda env update -f environment.yml --prune`. Pruning removes unlisted packages, so reinstall BirdNET-Analyzer afterwards with `pip install -e ./BirdNET-Analyzer --no-deps`. The optional `conda_packages_ext.txt` can be used to track personal additions; keep the shared `environment.yml` authoritative.

## Installing BirdNET-Analyzer

https://birdnet-team.github.io/BirdNET-Analyzer/installation.html
The processing script shells out to the BirdNET CLI (`birdnet_analyzer.analyze` and `birdnet_analyzer.embeddings`). Install the bundled copy in editable mode **after** activating the Conda environment: 

```bash
cd /path/to/birdnet_data_pipeline
git clone https://github.com/birdnet-team/BirdNET-Analyzer.git
pip install -e ./BirdNET-Analyzer --no-deps
```

This exposes the `birdnet_analyzer` module to the Python interpreter used by the pipeline. Verify the installation with `python -m birdnet_analyzer.analyze --help`. Quick sanity checks:

```bash
python -m birdnet_analyzer.analyze testbirdnet/input -o testbirdnet/output
python -m birdnet_analyzer.embeddings -db testbirdnet/output -i testbirdnet/input
```

Delete the temporary outputs after successful runs.

## Working with Config Files

Configuration lives in `xc_configs/`. Start from `config.yaml` or one of the species-specific examples and adjust the blocks below:

- `species`: scientific/common names plus an optional `slug` used for folder names.
- `paths.root`: absolute path where downloads, clips, and embeddings are stored.
- `xeno_canto`: API parameters such as `max_recordings`, `include_background`, and extra query filters (e.g., `len:3-200`). Max recordings can exceed available recordings without issue. In my experience 5k raw files, assuming maxlen 180, will be around ... 20GB, and subsequent clips, assuming max 5 per rec, will be around 4GB
- `processing`: clip duration, confidence thresholds, and whether to delete originals after clipping.
- `birdnet`: CLI thread/batch sizing and window overlap. Set `overlap_sec: 2.0` to slide BirdNET's default 3s window every ~1s; lower values reduce overlap if you need fewer detections. For a 10 core M1MAX apple sillicon I use 8 cores for maximum speed, 4-5 cores if I want to do other tasks during processing. thread size i keep 32, have not played around with it.
- `analysis`: UMAP and clustering defaults consumed by the visualization.
- `audio`: host/port for the HTTP server that makes clips playable inside the Bokeh app.

Store custom configs alongside the templates and reference them with the `--config` flag when running scripts.

## Xeno Canto
I suggest this page to gain insight into the available recordings and the name usage of Xeno Canto.
https://xeno-canto.org/collection/species/all?area=europe

## Running the Pipeline

All commands below assume `conda activate birdnetcluster1` and that you're inside the repository root.

1. **Download Xeno-Canto audio**  
   Set `XENO_CANTO_API_KEY` (or add `xeno_canto.api_key` in the config) before
   running; API v3 rejects requests without a key.  
   ```bash
   python xc_scripts/download_species.py --config xc_configs/config_turdus_merula.yaml
   ```  
   Replace the config to target another species, or use `--species "Genus species"` for ad-hoc runs.

2. **Regenerate metadata (optional, for existing downloads)**  
   ```bash
   python xc_scripts/metadata_species.py --config xc_configs/config_turdus_merula.yaml
   ```

3. **Detect, clip, and embed recordings**  
   ```bash
   python xc_scripts/process_species.py --config xc_configs/config_turdus_merula.yaml
   ```  
   This populates `clips/<slug>/` and `embeddings/<slug>/` inside your data root.

4. **Serve audio locally for the visualization**  
   ```bash
   cd "<paths.root>/clips/turdus_merula"
   python -m http.server 8765
   ```  
   Keep this terminal running; the UMAP app streams audio from `http://localhost:8765`.
   This directory should be where you stored the processed audio files, aka the clips.

5. **Launch the interactive UMAP explorer**  
   ```bash
   cd /path/to/birdnet_data_pipeline
   bokeh serve --show xc_scripts/umap_app.py --args --config xc_configs/config_emberiza_citrinella.yaml
   ```  
   The script accepts the same `--config` files used earlier and falls back to built-in defaults if none are provided.

Each script prints progress along with the locations it reads from and writes to; refer to the usage blocks embedded in the file headers for additional examples.

## Using the Browser App

Once the Bokeh server is running, the browser app loads a UMAP projection on the left and a map of Europe on the right, the EUMAP if you will. Both maps are backed by the same data source.

### Plot navigation
- The UMAP view groups recordings by embedding similarity; points that sit close together are acoustically similar.  
- The map view plots the same recordings by latitude/longitude over a Europe basemap.  
- Use the Bokeh toolbar to explore both plots: select the `Pan` tool (hand icon) to drag, and use your mouse scroll or trackpad pinch to zoom. The `Reset` tool returns to the full extent if you get lost.
- It might be nice to have zoomed in and out a little in the UMAP, so that the sliders wont affect the zoom.

### Audio playback
- Click any visible point in either plot to populate the playlist panel. UMAP clicks gather neighbours within the embedding space, while map clicks gather recordings within roughly 15 km.  
- Each playlist row includes a `Play` button that streams audio from the HTTP server you started earlier. Recordings filtered out of view are omitted from the playlist.  
- Use the `Test audio server` button to confirm that the audio host is reachable - the status panel updates once the test clip loads (or fails).

### Hover details and highlighting
- Hovering over a point shows its metadata in a tooltip and simultaneously highlights the same recording on both plots, making it easy to cross-reference geographic and embedding context.

### Filters and display widgets
- **Date range sliders:** `Adjust date slider range (zoom timeline)` changes the bounds available to the main `Filter recordings between` slider. After narrowing the bounds, drag the main slider handles to keep only recordings within the desired date interval.  
- **Color by:** The drop-down sets the active coloring scheme. Choosing `Season`, `KMeans`, `HDBSCAN`, `Sex`, or `Type` reveals the corresponding checkboxes; unchecking a value hides those recordings. Selecting `Time of Day` enables the slider described below. If colors are not loading, it helps to switch back and forth.
- **Time range slider:** When coloring by time, use the 0-24 hour slider to keep dawn, night, or other parts of the day. Clips without a known time remain visible.  
- **Show hover info toggle:** Disable this toggle if you want to reduce tooltip clutter; it hides hover popovers on both plots without affecting selections.

### Analysis widgets
- **UMAP params:** The `Nearest neighbors` and `Min distance` spinners control how the app recomputes the projection during zooms. See the [UMAP parameter guide](https://umap-learn.readthedocs.io/en/latest/parameters.html) for tuning advice.  
- **HDBSCAN controls:** Adjust `Min cluster size` and `Min samples`, then click `Apply HDBSCAN` to label the currently visible projection. The resulting checkboxes let you hide specific clusters or noise. Refer to the [HDBSCAN parameter selection guide](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html) when experimenting.

### Zoom controls and known issue
- `Zoom to Selection` recomputes the UMAP layout using only the currently selected (use box selection tool), visible points, while `Reset to Full Dataset` restores the original dataset.  
- Zooming can occasionally trigger a bug that leaves most points invisible; this is a known issue slated for a future fix. If it happens, reset to the full dataset to recover.

## Raspberry Pi Workflow

The `birdnet_pi` directory mirrors the Xeno-Canto tooling but targets BirdNET exports generated on a Raspberry Pi.

1. **Process Pi detections into clips and embeddings**  
   ```bash
   python birdnet_pi/process_species_pi.py --species "Merel" --root /path/to/birdnet_all_recordings --metadata birdnet_pi/detections.csv
   ```  
   By default the script writes to `birdnet_pi/pi_clips_embeds/<species_slug>/`. Pass `--output /alternate/path` to direct results elsewhere; the script adds the species slug automatically if you supply the parent directory.

2. **Serve Pi clips for the visualization**  
   ```bash
   cd birdnet_pi/pi_clips_embeds/merel/clips/merel # Or wherever youve stored them
   python3 -m http.server 8765
   ```

3. **Launch the Pi UMAP viewer**  
   ```bash
   bokeh serve --show birdnet_pi/umap_app_pi.py --args --species "Merel"
   ```  
   Use `--output` if the embeddings reside outside the repository structure.

## Additional Notes

- Reruns are incremental: the downloader skips files it already has, and `process_species.py` uses `metadata.csv` to map clips back to their Xeno-Canto IDs. Still, as of now, I would suggest running scripts in one go, and not trust its incremental capabillities
- Set `delete_originals: false` in the config if you prefer to retain the raw downloads after clipping. If true, you will be prompted at the end of the script with the option to delete or retain originals.
- The visualization expects `embeddings.csv` and `metadata.csv` in the species embeddings directory; regenerate them via the processing step if the app reports missing files.
- Network access is required only during the download stage; subsequent steps operate on local data.
