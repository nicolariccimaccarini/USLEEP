# Guida Setup e Documentazione Progetto Riconoscimento Spindles EEG

## Panoramica del Progetto

Questo progetto si occupa del riconoscimento automatico degli **spindles** nei segnali EEG tramite tecniche di **apprendimento non supervisionato** (autoencoder + clustering). Gli spindles sono oscillazioni cerebrali tipiche del sonno profondo.

---

## Struttura dei File

### Lettura e Preprocessing

- **letturaEDF.py**: Lettura file EDF, segmentazione e calcolo spettro.
- **rinonima.py**: Rinomina i file EDF in modo sequenziale.

### Modelli Autoencoder

- **trasformazione.py**: Autoencoder convoluzionale su segmenti EEG.
- **autoencoder_psd.py**: Autoencoder LSTM su Power Spectral Density.
- **autoencoder_CI_psd.py**: Autoencoder per ogni canale EEG.
- **autoencoder_CI_sovra_psd.py**: Autoencoder canali indipendenti con segmentazione sovrapposta.

### Clustering e Analisi

- **clustering.py**: Clustering con features estratte da autoencoder.
- **clustering_no_ae.py**: Clustering diretto sui dati raw.
- **k-means.py**: Implementazione standalone di K-means.
- **find_K.py**: Determina il numero ottimale di cluster.

### Valutazione

- **calcolo_acc.py**: Calcola accuratezza confrontando cluster e ground truth.

### Test e Utilities

- **test.py**: Verifica integrità modelli e test vari.

---

## Struttura delle Cartelle

```
Data/
├── images/
├── model/
├── weigths/
├── Edf/
├── Temp/
└── cluster/
```

---

## Setup Ambiente

```bash
# Crea ambiente conda
python -m venv venv
source venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt
```

---

## Preparazione Directory

```bash
mkdir -p Data/{Edf,Temp,images,model,weigths,cluster}
mkdir -p Data/images/{canali_individuali,canali_individuali_sovrapposizione,clustering,clustering-No-Ae}
```

---

## Workflow di Esecuzione

1. **Posiziona i file EEG (.edf) in `Data/Edf/`**
2. **Rinomina i file (opzionale):**
   ```bash
   python rinonima.py
   ```
3. **Copia alcuni file per test in `Data/Temp/`:**
   ```bash
   cp Data/Edf/*.edf Data/Temp/
   ```

### Approccio Convoluzionale

```bash
python autoencoder/trasformazione.py
```

### Approccio PSD

```bash
python autoencoder/autoencoder_psd.py
python autoencoder/autoencoder_CI_psd.py
python autoencoder/autoencoder_CI_sovra_psd.py
```

### Clustering Diretto

```bash
python clustering/clustering_no_ae.py
```

### Analisi Risultati

```bash
python clustering/find_K.py
python clustering/clustering.py
python clustering/calcolo_acc.py
```

---

## Parametri Modificabili

```python
window_size = 5      # Lunghezza finestra (secondi)
overlap = 0.10       # Sovrapposizione (10%)
epoche = 200         # Epoche training
batch_size = 16      # Batch size
num_clusters = 5     # Numero cluster K-means
pazienza = 20        # Early stopping patience
```

---

## Dove Trovare i Risultati

- **Grafici**: `Data/images/`
- **Modelli**: `Data/model/`
- **Log**: Output console

---

## Troubleshooting

- **Errore memoria**: Riduci `batch_size` o `window_size`
- **File corrotti**: Verifica con `test.py`
- **Path errors**: Controlla che tutte le directory esistano
- **GPU issues**: Forza CPU con `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

---

## Workflow Consigliato

```bash
cp Data/Edf/1.edf Data/Temp/
python trasformazione.py
# Analizza risultati in Data/images/
# Espandi a dataset completo se soddisfatto
```

---

## Note

Il progetto implementa diverse strategie per il riconoscimento automatico degli spindles, permettendo il confronto tra approcci basati su segnali raw, PSD e autoencoder.