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

The pipeline expects a writable data root (see `paths.root` in the configs) and will create three folders beneath it: `xc_downloads/`, `clips/`, and `embeddings/`. An external drive is recommended because audio files are numerous

## Managing the Conda Environment

`xc_pipeline/environment.yml` is the canonical specification for the end-to-end workflow. It pins Python 3.11 together with TensorFlow, librosa, UMAP, Bokeh, HDBSCAN, and command-line helpers such as `tqdm`. Always create or update your environment from this file to guarantee package compatibility with the scripts:

```bash
cd /Users/masjansma/Desktop/birdnetcluster1folder/xc_pipeline
conda env create -f environment.yml
conda activate birdnetcluster1
```

If the environment already exists, replace the first command with `conda env update -f environment.yml --prune`. The optional `conda_packages_ext.txt` can be used to track personal additions; keep the shared `environment.yml` authoritative.

## Installing BirdNET-Analyzer

The processing script shells out to the BirdNET CLI (`birdnet_analyzer.analyze` and `birdnet_analyzer.embeddings`). Install the bundled copy in editable mode **after** activating the Conda environment:

```bash
cd /Users/masjansma/Desktop/birdnetcluster1folder/xc_pipeline
git clone https://github.com/kahst/BirdNET-Analyzer.git
```

This exposes the `birdnet_analyzer` module to the Python interpreter used by the pipeline. Verify the installation with `python -m birdnet_analyzer.analyze --help`.

## Working with Config Files

Configuration lives in `xc_pipeline/xc_configs/`. Start from `config.yaml` or one of the species-specific examples and adjust the blocks below:

- `species`: scientific/common names plus an optional `slug` used for folder names.
- `paths.root`: absolute path where downloads, clips, and embeddings are stored.
- `xeno_canto`: API parameters such as `max_recordings`, `include_background`, and extra query filters (e.g., `len:3-200`).
- `processing`: clip duration, confidence thresholds, and whether to delete originals after clipping.
- `birdnet`: CLI thread/batch sizing for your hardware.
- `analysis`: UMAP and clustering defaults consumed by the visualization.
- `audio`: host/port for the HTTP server that makes clips playable inside the Bokeh app.

Store custom configs alongside the templates and reference them with the `--config` flag when running scripts.

## Running the Pipeline

All commands below assume `conda activate birdnetcluster1` and `cd /Users/masjansma/Desktop/birdnetcluster1folder/xc_pipeline`.

1. **Download Xeno-Canto audio**  
   ```bash
   python xc_scripts/download_species.py --config xc_configs/config_emberiza_citrinella.yaml
   ```  
   Replace the config to target another species, or use `--species "Genus species"` for ad-hoc runs.

2. **Regenerate metadata (optional, for existing downloads)**  
   ```bash
   python xc_scripts/metadata_species.py --config xc_configs/config_emberiza_citrinella.yaml
   ```

3. **Detect, clip, and embed recordings**  
   ```bash
   python xc_scripts/process_species.py --config xc_configs/config_emberiza_citrinella.yaml --skip-confirm
   ```  
   This populates `clips/<slug>/` and `embeddings/<slug>/` inside your data root.

4. **Serve audio locally for the visualization**  
   ```bash
   cd "<paths.root>/clips/emberiza_citrinella"
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

## Additional Notes

- Reruns are incremental: the downloader skips files it already has, and `process_species.py` uses `metadata.csv` to map clips back to their Xeno-Canto IDs. Still, as of now, I would suggest running scripts in one go, and not trust its incremental capabillities
- Set `delete_originals: false` in the config if you prefer to retain the raw downloads after clipping. If true, you will be prompted at the end of the script with the option to delete or retain originals.
- The visualization expects `embeddings.csv` and `metadata.csv` in the species embeddings directory; regenerate them via the processing step if the app reports missing files.
- Network access is required only during the download stage; subsequent steps operate on local data.
