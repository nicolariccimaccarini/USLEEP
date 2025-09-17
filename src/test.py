import numpy as np
import h5py
import tensorflow as tf


# print(tf.__version__)

try:
    with h5py.File('Data/weigths/autoencoder_model.h5', 'r') as f:
        print("Il file è integro.")
except Exception as e:
    print(f"Errore nell'aprire il file: {e}")


# # Esempio: dati EEG con 100 campioni e 64 caratteristiche
# eeg_features = np.random.rand(100, 64)

# # Ottieni il numero di caratteristiche
# num_features = eeg_features.shape[1]
# print(f"Numero di caratteristiche: {num_features}")