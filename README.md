# USLEEP — Unsupervised Spindle Learning via EEG Patterns

> **USLEEP** is a spindle detection algorithm for polysomnographic sleep recordings based on unsupervised learning. It combines autoencoder-based feature extraction with K-means clustering to identify sleep spindles in EEG signals — without the need for labelled training data.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Directory Preparation](#directory-preparation)
- [Usage](#usage)
- [Configurable Parameters](#configurable-parameters)
- [Output](#output)
- [Troubleshooting](#troubleshooting)

---

## Overview

Sleep spindles are brief bursts of oscillatory neural activity (11–16 Hz) that occur during NREM sleep and are associated with memory consolidation and sleep quality. USLEEP detects spindles automatically by:

1. **Preprocessing** — reading polysomnographic recordings in EDF format, segmenting EEG signals into short windows, and computing Power Spectral Density (PSD).
2. **Feature extraction** — training an autoencoder (convolutional or LSTM-based) on EEG segments to learn a compact latent representation.
3. **Clustering** — applying K-means to the latent features (or directly to PSD features) to separate spindle-containing segments from background activity.
4. **Evaluation** — comparing predicted clusters against ground-truth annotations to compute detection accuracy.

---

## Project Structure

```
USLEEP/
├── autoencoder/
│   ├── trasformazione.py          # Convolutional autoencoder on raw EEG segments
│   ├── autoencoder_psd.py         # LSTM autoencoder on Power Spectral Density (PSD)
│   ├── autoencoder_CI_psd.py      # Per-channel autoencoder on PSD
│   └── autoencoder_CI_sovra_psd.py# Per-channel autoencoder with overlapping windows
├── clustering/
│   ├── clustering.py              # K-means clustering on autoencoder latent features
│   ├── clustering_no_ae.py        # Direct K-means clustering on raw/PSD features
│   ├── k-means.py                 # Standalone K-means implementation
│   └── find_K.py                  # Elbow method to determine optimal number of clusters
├── src/
│   ├── main.py                    # Main entry point
│   ├── letturaEDF.py              # EDF file reading, segmentation, and spectrum computation
│   ├── rinonima.py                # Sequential renaming of EDF files
│   ├── h5_to_csv.py               # Convert HDF5 model outputs to CSV
│   ├── rebuild_autoencoder.py     # Reload and rebuild a saved autoencoder model
│   ├── calcolo_acc.py             # Accuracy computation against ground-truth annotations
│   ├── test_model.py              # Model integrity tests
│   └── prova1.ipynb               # Exploratory Jupyter notebook
├── Data/
│   ├── Edf/                       # Input EDF recordings
│   ├── Temp/                      # Temporary / test subset of EDF files
│   ├── images/                    # Generated plots and figures
│   ├── model/                     # Saved Keras model files
│   ├── weigths/                   # Saved model weights
│   └── cluster/                   # Clustering output files
├── requirements.txt
└── README.md
```

---

## Environment Setup

[Conda](https://docs.conda.io/en/latest/) is the recommended way to manage dependencies.

### 1. Create and activate a Conda environment

```bash
conda create -n usleep python=3.11 -y
conda activate usleep
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU support** — USLEEP is compatible with NVIDIA GPUs via CUDA 12. The `requirements.txt` already includes the required CUDA libraries (`nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, etc.) so no separate CUDA toolkit installation is needed when using pip. If you are on a CPU-only machine, see [Troubleshooting](#troubleshooting).

---

## Directory Preparation

Create the required output directories before the first run:

```bash
mkdir -p Data/{Edf,Temp,images,model,weigths,cluster}
mkdir -p Data/images/{individual_channels,individual_channels_overlap,clustering,clustering_no_ae}
```

Place your EDF polysomnographic recordings in `Data/Edf/`.

---

## Usage

### Step 1 — (Optional) Rename EDF files sequentially

```bash
python src/rinonima.py
```

### Step 2 — (Optional) Copy a subset for quick testing

```bash
cp Data/Edf/1.edf Data/Temp/
```

---

### Approach A — Convolutional Autoencoder (raw EEG segments)

Trains a convolutional autoencoder directly on time-domain EEG windows.

```bash
python autoencoder/trasformazione.py
```

---

### Approach B — LSTM Autoencoder on PSD

Trains an LSTM autoencoder on the Power Spectral Density of each EEG segment.

```bash
# All channels jointly
python autoencoder/autoencoder_psd.py

# Per-channel (independent autoencoders)
python autoencoder/autoencoder_CI_psd.py

# Per-channel with overlapping windows
python autoencoder/autoencoder_CI_sovra_psd.py
```

---

### Approach C — Direct Clustering (no autoencoder)

Applies K-means directly to raw or PSD features, without any neural network.

```bash
python clustering/clustering_no_ae.py
```

---

### Step 3 — Find the optimal number of clusters (elbow method)

```bash
python clustering/find_K.py
```

---

### Step 4 — Run clustering on autoencoder features

```bash
python clustering/clustering.py
```

---

### Step 5 — Evaluate detection accuracy

```bash
python src/calcolo_acc.py
```

---

### Running via the main entry point

```bash
python src/main.py
```

---

## Configurable Parameters

The following parameters can be adjusted at the top of the relevant scripts:

| Parameter       | Default | Description                                      |
|-----------------|---------|--------------------------------------------------|
| `window_size`   | `5`     | EEG segment length in seconds                    |
| `overlap`       | `0.10`  | Fractional overlap between consecutive windows   |
| `epochs`        | `200`   | Number of autoencoder training epochs            |
| `batch_size`    | `16`    | Mini-batch size for training                     |
| `num_clusters`  | `5`     | Number of K-means clusters                       |
| `patience`      | `20`    | Early stopping patience (epochs without improvement) |

---

## Output

| Location | Contents |
|---|---|
| `Data/images/` | Plots: PSD spectra, training loss curves, cluster visualisations |
| `Data/model/` | Saved Keras model files (`.keras`) |
| `Data/weigths/` | Saved model weights (`.weights.h5`) |
| `Data/cluster/` | Cluster assignment files (`.csv`) |
| Console / stdout | Training logs and accuracy metrics |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| **Out of memory** | Reduce `batch_size` or `window_size` |
| **Corrupt or unreadable EDF files** | Validate files with `src/test_model.py` or MNE's `mne.io.read_raw_edf` |
| **Path / directory errors** | Ensure all directories exist (see [Directory Preparation](#directory-preparation)) |
| **CUDA / GPU errors** | Force CPU execution: add `import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'` at the top of the script |
| **TensorFlow version mismatch** | Recreate the Conda environment and reinstall from `requirements.txt` |
| **Module not found** | Make sure the `usleep` Conda environment is activated: `conda activate usleep` |

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| TensorFlow / Keras | 2.20 / 3.11 | Autoencoder model training |
| MNE | 1.10 | EDF file reading and EEG preprocessing |
| scikit-learn | 1.7 | K-means clustering and evaluation metrics |
| NumPy | 2.2 | Numerical operations |
| SciPy | 1.15 | Signal processing and PSD computation |
| pandas | 2.3 | Data manipulation and CSV I/O |
| matplotlib | 3.10 | Plotting and visualisation |
| h5py | 3.14 | HDF5 model weight storage |

---

## License

This project is released for academic and research use. Please cite appropriately if you use USLEEP in your work.
