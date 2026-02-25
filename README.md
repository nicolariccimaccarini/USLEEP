# USLEEP — Unsupervised Sleep Spindle Detection via EEG Patterns

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-f7931e?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/MNE-EEG-green" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## Overview

**USLEEP** is a research pipeline for **automatic sleep spindle detection** in EEG signals using unsupervised machine learning. Sleep spindles, bursts of oscillatory activity in the 11–16 Hz range occurring during NREM stage 2 sleep, are clinically relevant biomarkers for neurological research and sleep medicine.

The pipeline combines two complementary stages:

1. **Feature extraction** via a Morlet wavelet-based LSTM autoencoder
2. **Unsupervised detection** via K-means clustering with a Morlet envelope refinement step

No labeled training data is required. The system learns compact time-frequency representations of EEG segments and separates spindle-like patterns from background activity through clustering and biophysical constraints.

---

## Pipeline Architecture

```
EEG Recording (.edf)
        │
        ▼
┌───────────────────────────────┐
│  Morlet CWT (fc = 13.5 Hz)    │  ← autoencoder_morlet_wavelet.py
│  Envelope extraction          │
│  Segment features             │
│  LSTM Autoencoder training    │
└──────────────┬────────────────┘
               │  Learned weights (.h5 per channel)
               ▼
┌───────────────────────────────┐
│  Autoencoder inference        │  ← clustering_with_spindle_detection.py
│  Reconstruction error         │
│  K-means clustering           │
│  Spindle cluster identification│
│  Morlet envelope refinement   │
│  Duration + amplitude filters │
│  Merge adjacent events        │
└──────────────┬────────────────┘
               │
               ▼
     start_end_per_channel.csv
     (spindle timestamps + metrics)
```

---

## Repository Structure

```
USLEEP/
├── autoencoder/
│   ├── autoencoder_morlet_wavelet.py       # ⭐ Primary: Morlet wavelet LSTM autoencoder
│   ├── trasformazione.py                   # Convolutional autoencoder on raw EEG
│   ├── autoencoder_psd.py                  # LSTM autoencoder on PSD features
│   ├── autoencoder_CI_psd.py               # Per-channel independent LSTM on PSD
│   └── autoencoder_CI_sovra_psd.py         # Per-channel LSTM with overlapping windows
├── clustering/
│   ├── clustering_with_spindle_detection.py # ⭐ Primary: Hybrid clustering + spindle detection
│   ├── clustering.py                        # Clustering on autoencoder embeddings
│   ├── clustering_no_ae.py                  # Clustering on raw data (no autoencoder)
│   ├── k-means.py                           # Standalone K-means
│   └── find_K.py                            # Elbow method for optimal K
├── utils/
│   └── signal_processing.py                # Shared DSP utilities (Morlet, segmentation, thresholding)
├── src/
│   ├── main.py                             # Orchestration entry point
│   ├── letturaEDF.py                       # EDF reading, segmentation, spectrum computation
│   ├── calcolo_acc.py                      # Accuracy evaluation vs. ground-truth labels
│   ├── rebuild_autoencoder.py              # Rebuild autoencoder from saved weights
│   ├── h5_to_csv.py                        # Convert HDF5 output to CSV
│   ├── test_model.py                       # Model integrity checks
│   └── rinonima.py                         # Sequential renaming of EDF files
├── Data/
│   ├── Preprocessed_Edf/                   # Input EEG recordings (.edf)
│   ├── Output/
│   │   ├── model/canali_individuali/       # Saved autoencoder weights (.h5 per channel)
│   │   ├── images/                         # Training plots and visualizations
│   │   └── cluster/                        # Detection results
│   │       └── start_end_per_channel.csv   # ← Final spindle event output
├── environment.yml                         # Conda environment specification
├── requirements.txt                        # pip fallback dependencies
└── README.md
```

---

## Environment Setup

> **Conda is the recommended environment manager** for this project due to its reliable handling of compiled dependencies (`numpy`, `scipy`, `tensorflow`).

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/nicolariccimaccarini/USLEEP.git
cd USLEEP

# Create the conda environment
conda env create -f environment.yml

# Activate it
conda activate usleep
```

If `environment.yml` is not yet available, create the environment manually:

```bash
conda create -n usleep python=3.10 -y
conda activate usleep

conda install -c conda-forge numpy scipy matplotlib pandas scikit-learn -y
pip install mne tensorflow
```

### Using pip (Alternative)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Core Dependencies

| Package | Purpose |
|---|---|
| `mne` | EDF file I/O, EEG signal processing |
| `tensorflow >= 2.10` | LSTM autoencoder training and inference |
| `scikit-learn` | K-means clustering, StandardScaler |
| `numpy`, `scipy` | Numerical computation, wavelet transforms |
| `matplotlib` | Training curves and visualization plots |
| `pandas` | Tabular output and CSV export |

---

## Directory Preparation

Before running any script, create the expected output directories:

```bash
mkdir -p Data/Preprocessed_Edf
mkdir -p Data/Output/model/canali_individuali
mkdir -p Data/Output/images
mkdir -p Data/Output/cluster
```

Place your EEG recordings (`.edf` format) in `Data/Preprocessed_Edf/`.

Optionally rename files sequentially:

```bash
python src/rinonima.py
```

---

## Usage

### Option A — Full Pipeline via `main.py` (Recommended)

The `src/main.py` script orchestrates the entire pipeline end-to-end: autoencoder training followed by clustering and spindle detection.

```bash
python src/main.py
```

Environment variables can be used to override default paths:

```bash
DATA_PATH=Data/Preprocessed_Edf \
OUTPUT_PATH=Data/Output \
CURRENT_FILE=subject_01.edf \
python src/main.py
```

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `Data/Preprocessed_Edf` | Directory containing `.edf` input files |
| `OUTPUT_PATH` | `Data/Output` | Root directory for all outputs |
| `CURRENT_FILE` | *(unset)* | Process a single file; if unset, processes all `.edf` files in batch |

---

### Option B — Step-by-Step Execution

#### Step 1 — Autoencoder Training

Train the Morlet wavelet LSTM autoencoder per EEG channel:

```bash
python autoencoder/autoencoder_morlet_wavelet.py
```

This script:
- Reads all `.edf` files from `DATA_PATH`
- Applies Morlet CWT at 13.5 Hz to each EEG channel
- Extracts envelope segments with 50% overlap
- Trains one LSTM autoencoder per channel
- Saves model weights to `OUTPUT_PATH/model/canali_individuali/autoencoder_<channel>.h5`

**Key parameters** (edit in script):

| Parameter | Default | Description |
|---|---|---|
| `window_size` | `0.5` | Segment length in seconds |
| `overlap_ratio` | `0.5` | Overlap between consecutive windows |
| `epoche` | `200` | Maximum training epochs |
| `batch_size` | `16` | Training batch size |
| `pazienza` | `20` | Early stopping patience |
| `wavelet_fc` | `13.5` | Morlet central frequency (Hz) |
| `wavelet_n_cycles` | `7` | Number of Morlet cycles |

---

#### Step 2 — Spindle Detection

Run the hybrid clustering and detection pipeline:

```bash
python clustering/clustering_with_spindle_detection.py
```

This script:
- Loads each per-channel autoencoder from `OUTPUT_PATH/model/canali_individuali/`
- Applies Morlet CWT and extracts 4 statistical envelope features per segment  
  (`mean`, `std`, `max`, `median`)
- Concatenates features with autoencoder reconstruction error
- Runs K-means (k=2: spindles vs. background)
- Identifies the spindle cluster by highest mean envelope amplitude
- Refines detections using biophysical constraints:
  - Duration: 0.5 s – 3.0 s
  - Amplitude: ≥ 95% of samples above adaptive RMS threshold
- Merges adjacent spindle events (gap < 1.0 s)
- Exports results to `OUTPUT_PATH/cluster/start_end_per_channel.csv`

**Key parameters** (edit `CONFIG` dict):

| Parameter | Default | Description |
|---|---|---|
| `window_size` | `0.5` | Segment length (must match training) |
| `overlap_ratio` | `0.5` | Overlap ratio (must match training) |
| `num_clusters` | `2` | K-means clusters (spindle vs. background) |
| `min_spindle_duration` | `0.5` | Minimum spindle duration (s) |
| `max_spindle_duration` | `3.0` | Maximum spindle duration (s) |
| `wavelet_fc` | `13.5` | Morlet central frequency (Hz) |
| `rms_percentile` | `0.25` | Adaptive threshold percentile |
| `min_amplitude_ratio` | `0.95` | Fraction of samples above RMS threshold |
| `merge_gap_sec` | `1.0` | Max gap to merge adjacent spindles (s) |

---

#### Step 3 — Evaluate Results

Compare detected spindles against ground-truth annotations:

```bash
python src/calcolo_acc.py
```

---

## Output Format

The primary output is `Data/Output/cluster/start_end_per_channel.csv`:

| Column | Description |
|---|---|
| `Canale` | EEG channel name |
| `Start_Time(s)` | Spindle onset (seconds from recording start) |
| `End_Time(s)` | Spindle offset (seconds) |
| `Duration(s)` | Event duration |
| `Peak_Amplitude(µV)` | Peak Morlet envelope amplitude |
| `Mean_Amplitude(µV)` | Mean Morlet envelope amplitude |
| `RMS_Threshold(µV)` | Adaptive RMS threshold applied |
| `Confidence` | Fraction of samples exceeding threshold |

---

## Alternative Approaches

USLEEP includes additional scripts for experimental comparison:

| Script | Approach |
|---|---|
| `autoencoder/autoencoder_psd.py` | LSTM autoencoder on Power Spectral Density |
| `autoencoder/autoencoder_CI_psd.py` | Per-channel PSD LSTM autoencoder |
| `autoencoder/trasformazione.py` | Convolutional autoencoder on raw EEG segments |
| `clustering/clustering.py` | Clustering on latent embeddings (no detection refinement) |
| `clustering/clustering_no_ae.py` | Direct clustering on raw signal (no autoencoder) |
| `clustering/find_K.py` | Elbow + silhouette method for optimal K selection |

The `autoencoder_morlet_wavelet` + `clustering_with_spindle_detection` combination is the recommended and most accurate pipeline.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| **Out of memory** | Reduce `batch_size` or `window_size` in the autoencoder config |
| **Model not found for channel** | Ensure Step 1 (autoencoder training) completed for all channels |
| **window_size`/`overlap_ratio` mismatch** | Both scripts must use identical values; defaults are pre-aligned |
| **Corrupted or unreadable EDF** | Validate files with `src/test_model.py` |
| **Path errors** | Verify all output directories exist (see *Directory Preparation*) |
| **GPU/CUDA errors** | Force CPU: add `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'` at script start |
| **TensorFlow serialization warning** | Handled automatically via `tf.keras.config.enable_unsafe_deserialization()` |

---

## Research Context

This project explores fully unsupervised spindle detection using multiple paradigms:

- **Raw signal approaches** — direct clustering without feature extraction
- **Spectral approaches** — PSD-based LSTM autoencoders
- **Time-frequency approaches** — Morlet wavelet envelope autoencoders *(recommended)*

The Morlet wavelet approach is specifically suited to spindle detection because it isolates the 11–16 Hz oscillatory band while preserving temporal dynamics, providing richer representations than static PSD features.

---

## License

This project is released under the [MIT License](LICENSE).
