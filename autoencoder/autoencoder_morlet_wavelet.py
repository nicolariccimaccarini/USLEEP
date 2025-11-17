import mne 
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
import gc
import sys

# Aggiungi il percorso utils al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from signal_processing import (
    get_file_output_path, segment_signal_with_overlap,
    compute_morlet_features  # NUOVA FUNZIONE
)

# ...existing code...

CONFIG = {
    'window_size': 0.5,
    'overlap_ratio': 0.2,
    'batch_size': 256,
    'epochs': 200,
    'patience': 20,
    'wavelet_fc_range': [11, 12.5, 14, 15],         # Range frequenze centrali
    'wavelet_n_cycles': 7,                          # Numero cicli Morlet
    'channels_to_exclude': {'EEG A1', 'EEG A2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2'}
}

def create_autoencoder_model(n_features):
    """
    Crea l'architettura dell'autoencoder per feature Morlet
    
    Args:
        n_features: numero di feature in input (ampiezza + fase + freq_inst per ogni fc)
    """
    input_layer = Input(shape=(1, n_features))
    
    # Encoder LSTM
    encoded = LSTM(128, activation='relu', return_sequences=True, name='encoder_lstm_1')(input_layer)
    encoded = LSTM(64, activation='relu', return_sequences=True, name='encoder_lstm_2')(encoded)
    encoded = Dropout(0.2, name='encoder_dropout')(encoded)
    encoded = LSTM(32, activation='relu', return_sequences=False, name='encoder_lstm_3')(encoded)
    encoded = Dense(32, activation='relu', name='encoder_dense')(encoded)
    
    # Decoder
    decoded = RepeatVector(1, name='repeat_vector')(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True, name='decoded_lstm_1')(decoded)
    decoded = LSTM(128, activation='relu', return_sequences=True, name='decoded_lstm_2')(decoded)
    decoded = TimeDistributed(Dense(n_features), name='time_distributed_output')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss=MeanSquaredError())
    
    return autoencoder

def process_edf_files():
    """Processa i file EDF preprocessati e addestra gli autoencoder per canale"""
    
    # Configurazione percorsi
    path_edf = os.environ.get('DATA_PATH', 'Data/Preprocessed_Edf')
    output_path = os.environ.get('OUTPUT_PATH', 'Data/Output')
    current_file = os.environ.get('CURRENT_FILE', None)
    
    print(f"📂 Path EDF preprocessati: {path_edf}")
    
    if not os.path.exists(path_edf):
        print(f"❌ Errore: cartella {path_edf} non trovata!")
        return
    
    # Determina i file da processare
    if current_file:
        dirData = get_file_output_path(output_path, current_file)
        filenames = [current_file]
        print(f"📁 Modalità file singolo: {current_file}")
    else:
        dirData = output_path
        filenames = [f for f in os.listdir(path_edf) if f.endswith('.edf')]
        print(f"📁 Modalità batch: {len(filenames)} file preprocessati")
    
    if not filenames:
        print(f"❌ Nessun file .edf trovato in {path_edf}")
        return
    
    # Crea struttura cartelle
    weights_path = os.path.join(dirData, "model", "canali_individuali")
    images_path = os.path.join(dirData, "images", "canali_individuali")
    
    os.makedirs(weights_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    
    # Aggregazione dati per canale
    aggregated_data = {}
    
    print("\n📊 Processamento file EDF con Morlet Wavelet Features...")
    for file in filenames:
        if not file.endswith('.edf'):
            continue
            
        file_path = os.path.join(path_edf, file)
        print(f"📁 Processando: {file}")
        
        # Carica file EDF
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        
        # Filtra canali
        channels_to_include = [ch for ch in raw.ch_names if ch not in CONFIG['channels_to_exclude']]
        raw.pick_channels(channels_to_include)
        
        # Ottieni dati
        raw_data = raw.get_data()
        
        # Segmentazione con sliding window
        segment_length = int(CONFIG['window_size'] * sfreq)
        segments = segment_signal_with_overlap(
            raw_data,
            segment_length, 
            CONFIG['overlap_ratio']
        )
        
        print(f"   📏 Segmenti generati: {len(segments)}")
        
        morlet_features_all = []
        
        for segment in segments:
            # Calcola feature Morlet per ogni segmento
            segment_features = compute_morlet_features(
                segment, 
                sfreq, 
                fc_range=CONFIG['wavelet_fc_range'],
                n_cycles=CONFIG['wavelet_n_cycles']
            )
            morlet_features_all.append(segment_features)
        
        morlet_features_all = np.array(morlet_features_all)
        n_features = morlet_features_all.shape[2]  # feature per canale
        
        print(f"   🌊 Feature Morlet estratte: {morlet_features_all.shape}")
        print(f"   📊 Feature per segmento: {n_features}")
        
        # Aggregazione per canale
        for idx, channel in enumerate(raw.ch_names):
            if channel not in aggregated_data:
                aggregated_data[channel] = []
            for morlet_feat in morlet_features_all:
                aggregated_data[channel].append(morlet_feat[idx])
    
    # Training degli autoencoder
    strategy = tf.distribute.MirroredStrategy()
    
    print(f"\n🤖 Addestramento autoencoder su feature Morlet per {len(aggregated_data)} canali...")
    for channel_idx, (channel, data) in enumerate(aggregated_data.items(), 1):
        print(f"\n🔧 Canale {channel} ({channel_idx}/{len(aggregated_data)})")
        
        # Preparazione dati
        data = np.array(data)
        n_features = data.shape[1]
        all_segments_standardized = data.reshape((-1, 1, n_features))
        
        print(f"   📊 Campioni: {len(data)}, Feature Morlet: {n_features}")
        
        # Percorsi di salvataggio
        model_path = os.path.join(weights_path, f"autoencoder_{channel}.h5")
        plot_path = os.path.join(images_path, f"training_{channel}.png")
        
        # Creazione e addestramento modello
        with strategy.scope():
            autoencoder = create_autoencoder_model(n_features)
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=CONFIG['patience'], 
            verbose=1, 
            restore_best_weights=True
        )
        
        history = autoencoder.fit(
            all_segments_standardized, all_segments_standardized,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Salvataggio modello e plot
        autoencoder.save(model_path)
        print(f"   💾 Modello salvato: {model_path}")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history["loss"], 'r', marker='.', label="Train Loss")
        plt.plot(history.history["val_loss"], 'b--', marker='.', label="Validation Loss")
        plt.title(f"Training History (Morlet Features) - {channel}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Plot salvato: {plot_path}")
        
        # Pulizia memoria
        del autoencoder
        tf.keras.backend.clear_session()
        gc.collect()
        
        print(f"   ✅ Completato {channel}")

if __name__ == "__main__":
    process_edf_files()
    print("\n🎉 Addestramento completato con feature Morlet!")