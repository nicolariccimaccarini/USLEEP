# USLEEP – Unsupervised Spindle Learning via EEG Patterns

## Project Overview

**USLEEP** is a Python-based pipeline for the **automatic detection of sleep spindles** in EEG signals using **unsupervised learning** techniques. Sleep spindles are characteristic bursts of oscillatory neural activity (typically 11–16 Hz) occurring during NREM stage 2 sleep and are relevant biomarkers for neurological research and sleep medicine.

The pipeline combines two core strategies:
1. **Feature extraction via autoencoders** (including a novel Morlet wavelet-based approach)
2. **Unsupervised clustering with integrated spindle detection**

---

## ⭐ Featured Scripts

### 🔬 `autoencoder_morlet_wavelet`

> **Path:** `autoencoder/autoencoder_morlet_wavelet.py`

This is the most advanced autoencoder model in the pipeline. It applies the **Morlet continuous wavelet transform (CWT)** to EEG segments before feeding them into the autoencoder, enabling rich time-frequency feature extraction specifically suited for spindle-like oscillations.

**Key characteristics:**
- Applies **Morlet CWT** to each EEG segment, producing a 2D time-frequency representation
- Feeds the wavelet scalograms into a **LSTM autoencoder** (encoder–decoder architecture)
- The encoder's latent space captures compact, discriminative representations of EEG oscillations
- Particularly effective at isolating the 11–16 Hz spindle frequency band from background EEG noise
- Supports **per-channel independent processing** for multi-channel EEG recordings
- Outputs learned embeddings to be consumed downstream by the clustering step

**Typical usage:**
```bash
python autoencoder/autoencoder_morlet_wavelet.py
```

**Configurable parameters:**
```python
window_size   = 0.5    # Segment length in seconds
overlap       = 0.5    # Overlap between consecutive windows (10%)
epoche        = 200    # Training epochs
batch_size    = 16     # Batch size for training
pazienza      = 20     # Early stopping patience
```

**Output:**
- Trained model weights saved in `Data/model/`
- Latent-space embeddings used by the clustering step
- Reconstruction plots saved in `Data/images/`

---

### 🔍 `clustering_with_spindle_detection`

> **Path:** `clustering/clustering_with_spindle_detection.py`

This is the core analysis script that combines **K-means clustering** with a dedicated **spindle detection** post-processing step. It takes the latent embeddings produced by the autoencoder (typically from `autoencoder_morlet_wavelet`) and identifies which clusters correspond to sleep spindle activity.

**Key characteristics:**
- Loads pre-trained autoencoder weights and extracts latent embeddings from EEG data
- Applies **K-means clustering** on the embedding space to group EEG segments by pattern similarity
- Implements a **spindle-detection heuristic**: after clustering, it evaluates each cluster's frequency content and temporal characteristics to label spindle clusters automatically
- Uses **PCA** for 2D/3D visualization of the cluster structure
- Computes **silhouette scores** to assess clustering quality
- Generates annotated output files marking detected spindle events with their timestamps

**Typical usage:**
```bash
python clustering/clustering_with_spindle_detection.py
```

**Configurable parameters:**
```python
num_clusters  = 2      # Number of K-means clusters
batch_size    = 8      # Batch size for embedding extraction
n_components  = 2      # PCA components for visualization
```

**Output:**
- Cluster assignment files in `Data/cluster/`
- Visualization plots (PCA scatter, cluster waveforms) in `Data/images/clustering/`
- Spindle event timestamps exported to `Data/cluster/<subject>/spindle_events.csv`

---

## Repository Structure

```
USLEEP/
├── autoencoder/
│   ├── autoencoder_morlet_wavelet.py       # ⭐ Morlet wavelet autoencoder (primary model)
│   ├── trasformazione.py                   # Convolutional autoencoder on raw EEG segments
│   ├── autoencoder_psd.py                  # LSTM autoencoder on Power Spectral Density
│   ├── autoencoder_CI_psd.py               # Per-channel independent LSTM autoencoder
│   └── autoencoder_CI_sovra_psd.py         # Per-channel autoencoder with overlapping windows
├── clustering/
│   ├── clustering_with_spindle_detection.py # ⭐ Clustering + spindle detection (primary script)
│   ├── clustering.py                        # Clustering on autoencoder-extracted features
│   ├── clustering_no_ae.py                  # Direct clustering on raw data (no autoencoder)
│   ├── k-means.py                           # Standalone K-means implementation
│   └── find_K.py                            # Optimal cluster number estimation
├── src/
│   ├── letturaEDF.py                        # EDF file reading, segmentation, spectrum computation
│   ├── rinonima.py                          # Sequential renaming of EDF files
│   ├── calcolo_acc.py                       # Accuracy evaluation against ground truth labels
│   ├── rebuild_autoencoder.py               # Utility to rebuild autoencoder from saved weights
│   ├── h5_to_csv.py                         # Convert HDF5 output to CSV format
│   ├── test_model.py                        # Model integrity tests
│   └── main.py                              # Main entry point
├── Data/
│   ├── Edf/                                 # Raw EDF input files
│   ├── Temp/                                # Temporary working files
│   ├── images/                              # Output visualizations
│   ├── model/                               # Saved model architectures
│   ├── weigths/                             # Saved model weights
│   └── cluster/                             # Clustering results and spindle event files
├── requirements.txt
└── README.md
```

---

## Environment Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**Main dependencies:**
- `mne` – EEG/MEG data processing and EDF reading
- `tensorflow` / `keras` – Autoencoder model training
- `scikit-learn` – Clustering, PCA, evaluation metrics
- `numpy`, `scipy` – Numerical computation and signal processing
- `matplotlib` – Visualization

---

## Directory Preparation

Before running any script, create the expected output directory structure:

```bash
mkdir -p Data/{Edf,Temp,images,model,weigths,cluster}
mkdir -p Data/images/{canali_individuali,canali_individuali_sovrapposizione,clustering,clustering-No-Ae}
```

---

## Execution Workflow

### Step 1 – Prepare EEG data

Place your EEG recordings (`.edf` format) into `Data/Edf/`. Optionally rename them sequentially:

```bash
python src/rinonima.py
```

Copy a subset for quick testing:

```bash
cp Data/Edf/*.edf Data/Temp/
```

### Step 2 – Feature Extraction (Recommended: Morlet Wavelet Autoencoder)

```bash
python autoencoder/autoencoder_morlet_wavelet.py
```

Alternative autoencoder approaches:

```bash
# Convolutional autoencoder on raw segments
python autoencoder/trasformazione.py

# LSTM autoencoder on PSD features
python autoencoder/autoencoder_psd.py

# Per-channel independent LSTM autoencoder
python autoencoder/autoencoder_CI_psd.py

# Per-channel with overlapping windows
python autoencoder/autoencoder_CI_sovra_psd.py
```

### Step 3 – Clustering and Spindle Detection (Recommended)

```bash
python clustering/clustering_with_spindle_detection.py
```

Alternative clustering approaches:

```bash
# Clustering on autoencoder features (without integrated spindle detection)
python clustering/clustering.py

# Direct clustering on raw data (no autoencoder)
python clustering/clustering_no_ae.py

# Find optimal number of clusters K
python clustering/find_K.py
```

### Step 4 – Evaluate Results

```bash
python src/calcolo_acc.py
```

---

## Configurable Parameters

The following key parameters can be adjusted in the respective scripts:

| Parameter     | Default | Description                            |
|---------------|---------|----------------------------------------|
| `window_size` | `5`     | EEG segment length in seconds          |
| `overlap`     | `0.10`  | Overlap ratio between windows (10%)    |
| `epoche`      | `200`   | Number of training epochs              |
| `batch_size`  | `16`    | Batch size during training             |
| `num_clusters`| `5`     | Number of K-means clusters             |
| `pazienza`    | `20`    | Early stopping patience (epochs)       |

---

## Output Locations

| Artifact                    | Path                                        |
|-----------------------------|---------------------------------------------|
| Training plots & waveforms  | `Data/images/`                              |
| Saved model architectures   | `Data/model/`                               |
| Saved model weights         | `Data/weigths/`                             |
| Cluster assignments         | `Data/cluster/`                             |
| Spindle event timestamps    | `Data/cluster/<subject>/spindle_events.csv` |

---

## Troubleshooting

| Issue                   | Solution                                                             |
|-------------------------|----------------------------------------------------------------------|
| **Out of memory**       | Reduce `batch_size` or `window_size`                                 |
| **Corrupted EDF files** | Check file integrity using `src/test_model.py`                       |
| **Path errors**         | Ensure all required directories exist (see *Directory Preparation*)  |
| **GPU issues**          | Force CPU mode: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`          |
| **TF serialization**    | Already handled via `tf.keras.config.enable_unsafe_deserialization()` |

---

## Notes

This project implements multiple strategies for automatic sleep spindle recognition, enabling systematic comparison between:
- **Raw signal approaches** (direct clustering without feature extraction)
- **Spectral approaches** (PSD-based LSTM autoencoders)
- **Time-frequency approaches** (Morlet wavelet convolutional autoencoders — recommended)

The `autoencoder_morlet_wavelet` + `clustering_with_spindle_detection` pipeline represents the most complete and accurate approach for spindle detection in this repository.
