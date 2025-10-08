# Xeno-Canto Pipeline

This repository collects tooling for building clustered embeddings of bird vocalizations. At present this README focuses on the Xeno-Canto ingestion and analysis workflow housed in `xc_pipeline/`.

## Repository Layout

```
xc_pipeline/
├── environment.yml          # Conda specification for the pipeline runtime
├── conda_packages_ext.txt   # Optional personal package additions
├── xc_configs/              # Species-specific configuration templates
├── xc_scripts/              # Download, process, and visualization scripts
└── BirdNET-Analyzer/        # BirdNET CLI source (installed inside the env)
birdnet_pi/                # Pi-specific tooling (out of scope here)
```

The pipeline expects a writable data root (see `paths.root` in the configs) and will create three folders beneath it: `xc_downloads/`, `clips/`, and `embeddings/`. An external drive is recommended because audio files are numerous.

## Managing the Conda Environment

`xc_pipeline/environment.yml` is the canonical specification for the end-to-end workflow. It pins Python 3.11 together with TensorFlow, librosa, UMAP, Bokeh, HDBSCAN, and command-line helpers such as `tqdm`. Always create or update your environment from this file to guarantee package compatibility with the scripts:

```bash
cd /Users/masjansma/Desktop/birdnetcluster1folder/xc_pipeline # replace with path to your wd
conda env create -f environment.yml
# or for updating use
conda env update -f environment.yml
# but note pruning (--prune) requires reinstalling BirdNET-Analyzer dependencies manually
conda activate birdnetcluster1
```

If the environment already exists, replace the first command with `conda env update -f environment.yml --prune` but note pruning requires reinstalling BirdNET-Analyzer build, use `pip install -e ./BirdNET-Analyzer --no-deps`. The optional `conda_packages_ext.txt` can be used to track personal additions; keep the shared `environment.yml` authoritative.

## Installing BirdNET-Analyzer

https://birdnet-team.github.io/BirdNET-Analyzer/installation.html
The processing script shells out to the BirdNET CLI (`birdnet_analyzer.analyze` and `birdnet_analyzer.embeddings`). Install the bundled copy in editable mode **after** activating the Conda environment: 

```bash
cd /Users/masjansma/Desktop/birdnetcluster1folder/xc_pipeline
git clone https://github.com/birdnet-team/BirdNET-Analyzer.git
pip install -e ./BirdNET-Analyzer --no-deps
```

This exposes the `birdnet_analyzer` module to the Python interpreter used by the pipeline. Verify the installation with `python -m birdnet_analyzer.analyze --help`. Make sure to cd back to wd (xc_pipeline as of now)
Further tests: 
python -m birdnet_analyzer.analyze /Users/masjansma/Desktop/birdnetcluster1folder/xc_pipeline/testbirdnet/input -o /Users/masjansma/Desktop/birdnetcluster1folder/xc_pipeline/testbirdnet/output
also test embeddings:
python -m birdnet_analyzer.embeddings -db /Users/masjansma/Desktop/birdnetcluster1folder/xc_pipeline/testbirdnet/output -i /Users/masjansma/Desktop/birdnetcluster1folder/xc_pipeline/testbirdnet/input
Can delete output after succesful test

## Working with Config Files

Configuration lives in `xc_pipeline/xc_configs/`. Start from `config.yaml` or one of the species-specific examples and adjust the blocks below:

- `species`: scientific/common names plus an optional `slug` used for folder names.
- `paths.root`: absolute path where downloads, clips, and embeddings are stored.
- `xeno_canto`: API parameters such as `max_recordings`, `include_background`, and extra query filters (e.g., `len:3-200`). Max recordings can exceed available recordings without issue. In my experience 5k raw files, assuming maxlen 180, will be around ... 20GB, and subsequent clips, assuming max 5 per rec, will be around 4GB
- `processing`: clip duration, confidence thresholds, and whether to delete originals after clipping.
- `birdnet`: CLI thread/batch sizing for your hardware. For a 10 core M1MAX apple sillicon I use 8 cores for maximum speed, 4-5 cores if I want to do other tasks during processing. thread size i keep 32, have not played around with it.
- `analysis`: UMAP and clustering defaults consumed by the visualization.
- `audio`: host/port for the HTTP server that makes clips playable inside the Bokeh app.

Store custom configs alongside the templates and reference them with the `--config` flag when running scripts.

## Running the Pipeline

All commands below assume `conda activate birdnetcluster1` and `cd /Users/masjansma/Desktop/birdnetcluster1folder/xc_pipeline`.

1. **Download Xeno-Canto audio**  
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
   python xc_scripts/process_species.py --config xc_configs/config_turdus_merula.yaml --skip-confirm
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
   cd /Users/masjansma/Desktop/birdnetcluster1folder/xc_pipeline
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

## Additional Notes

- Reruns are incremental: the downloader skips files it already has, and `process_species.py` uses `metadata.csv` to map clips back to their Xeno-Canto IDs. Still, as of now, I would suggest running scripts in one go, and not trust its incremental capabillities
- Set `delete_originals: false` in the config if you prefer to retain the raw downloads after clipping. If true, you will be prompted at the end of the script with the option to delete or retain originals.
- The visualization expects `embeddings.csv` and `metadata.csv` in the species embeddings directory; regenerate them via the processing step if the app reports missing files.
- Network access is required only during the download stage; subsequent steps operate on local data.
