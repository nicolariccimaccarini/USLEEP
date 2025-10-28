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
    apply_sigma_band_filter, compute_sigma_power_spectrum, normalize_spectrum
)

# Configurazione GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Configurate {len(gpus)} GPU con crescita memoria graduale")
    except RuntimeError as e:
        print(f"⚠️ Errore configurazione GPU: {e}")

# Parametri ottimizzati per spindle detection
CONFIG = {
    'window_size': 0.5,      # Finestra di 0.5s (durata minima spindle)
    'overlap_ratio': 0.8,    # 80% overlap -> step 0.1s per risoluzione temporale
    'batch_size': 512,
    'epochs': 300,
    'patience': 30,
    'sigma_low': 9,          # Banda sigma: 9-15 Hz
    'sigma_high': 15,
    'channels_to_exclude': {'EEG A1', 'EEG A2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2'}
}

def create_sigma_autoencoder(n_sigma_features):
    """
    Autoencoder profondo ottimizzato per features sigma discriminative
    """
    input_layer = Input(shape=(1, n_sigma_features), name='sigma_input')
    
    # Encoder LSTM profondo 
    encoded = LSTM(128, activation='relu', return_sequences=True, name='encoder_lstm_1')(input_layer)
    encoded = LSTM(64, activation='relu', return_sequences=True, name='encoder_lstm_2')(encoded)
    encoded = Dropout(0.2, name='encoder_dropout')(encoded)
    encoded = LSTM(32, activation='relu', return_sequences=False, name='encoder_lstm_3')(encoded)
    encoded = Dense(32, activation='relu', name='encoder_dense')(encoded)  # Layer chiave per clustering
    
    # Decoder simmetrico
    decoded = RepeatVector(1, name='repeat_vector')(encoded)
    decoded = LSTM(32, activation='relu', return_sequences=True, name='decoder_lstm_1')(decoded)
    decoded = LSTM(64, activation='relu', return_sequences=True, name='decoder_lstm_2')(decoded)
    decoded = LSTM(128, activation='relu', return_sequences=True, name='decoder_lstm_3')(decoded)
    decoded = TimeDistributed(Dense(n_sigma_features), name='sigma_output')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded, name='sigma_autoencoder')
    autoencoder.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])
    
    return autoencoder

def process_edf_for_spindles():
    """Processa EEG con features consistenti per training e detection"""
    
    # Configurazione percorsi
    path_edf = os.environ.get('DATA_PATH', 'Data/Edf')
    output_path = os.environ.get('OUTPUT_PATH', 'Data/Output')
    current_file = os.environ.get('CURRENT_FILE', None)
    
    # Determina i file da processare
    if current_file:
        dirData = get_file_output_path(output_path, current_file)
        filenames = [current_file]
    else:
        dirData = output_path
        filenames = [f for f in os.listdir(path_edf) if f.endswith('.edf')]
    
    # Crea struttura cartelle
    weights_path = os.path.join(dirData, "model", "sigma_band")
    images_path = os.path.join(dirData, "images", "sigma_band")
    
    os.makedirs(weights_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    
    # Aggregazione dati sigma per canale
    aggregated_sigma_data = {}
    
    print("🧠 Processamento EEG con features discriminative...")
    for file in filenames:
        if not file.endswith('.edf'):
            continue
            
        file_path = os.path.join(path_edf, file)
        print(f"📁 Processando: {file}")
        
        # Carica e filtra EEG
        raw = mne.io.read_raw_edf(file_path, preload=True)
        sfreq = raw.info['sfreq']
        
        channels_to_include = [ch for ch in raw.ch_names if ch not in CONFIG['channels_to_exclude']]
        raw.pick_channels(channels_to_include)
        
        # Filtro banda sigma + segmentazione
        print("🔧 Applicando filtro banda sigma...")
        sigma_filtered_data = apply_sigma_band_filter(
            raw.get_data(), sfreq, CONFIG['sigma_low'], CONFIG['sigma_high']
        )
        
        segment_length = int(CONFIG['window_size'] * sfreq)
        segments = segment_signal_with_overlap(
            sigma_filtered_data, segment_length, CONFIG['overlap_ratio']
        )
        
        # 🎯 CRUCIALE: Usa stesse features per training e detection
        sigma_powers, _ = compute_sigma_power_spectrum(segments, sfreq)
        
        # Aggregazione per canale con features consistenti
        for idx, channel in enumerate(raw.ch_names):
            if channel not in aggregated_sigma_data:
                aggregated_sigma_data[channel] = []
            
            # Estrai features per ogni segmento di questo canale
            for segment_idx in range(len(segments)):
                # sigma_powers[segment_idx] contiene già le 5 features per tutti i canali
                # Prendi solo quelle del canale corrente
                segment_powers = []
                for power_group in sigma_powers:
                    if len(power_group) > idx:
                        segment_powers.append(power_group[idx])
                
                if segment_powers:
                    aggregated_sigma_data[channel].extend(segment_powers)
    
    # Training con features discriminative
    strategy = tf.distribute.MirroredStrategy()
    
    print(f"\n🤖 Training autoencoder con features discriminative...")
    for channel_idx, (channel, data) in enumerate(aggregated_sigma_data.items(), 1):
        print(f"\n🧠 Canale {channel} ({channel_idx}/{len(aggregated_sigma_data)})")
        
        # Preparazione dati: ogni sample ha 5 features discriminative
        data = np.array(data)
        print(f"📊 Shape dati grezzi: {data.shape}")
        
        # Assicurati che abbiamo 5 features per segmento
        if data.ndim == 1:
            # Se è 1D, reshape in (n_samples, 5)
            n_samples = len(data) // 5
            data = data.reshape(n_samples, 5)
        elif data.ndim == 2 and data.shape[1] != 5:
            print(f"⚠️ Dimensioni features inaspettate: {data.shape}")
            continue
            
        n_sigma_features = 5  # Sempre 5 features discriminative
        all_sigma_segments = data.reshape((-1, 1, n_sigma_features))
        
        print(f"📊 Features discriminative: 5 (media, std, max, picchi, area)")
        print(f"📊 Segmenti totali: {all_sigma_segments.shape[0]}")
        
        # Percorsi di salvataggio
        model_path = os.path.join(weights_path, f"sigma_autoencoder_{channel}.h5")
        plot_path = os.path.join(images_path, f"sigma_training_{channel}.png")
        
        # Creazione e addestramento modello sigma-specifico
        with strategy.scope():
            sigma_autoencoder = create_sigma_autoencoder(n_sigma_features)
        
        print(f"🏗️ Architettura autoencoder sigma per {channel}:")
        sigma_autoencoder.summary()
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=CONFIG['patience'], 
            verbose=1, 
            restore_best_weights=True
        )
        
        history = sigma_autoencoder.fit(
            all_sigma_segments, all_sigma_segments,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Salvataggio modello
        sigma_autoencoder.save(model_path)
        
        # Plot training history con focus su convergenza
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], 'r-', label="Train Loss")
        plt.plot(history.history["val_loss"], 'b--', label="Validation Loss")
        plt.title(f"Sigma Band Training - {channel}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history["mae"], 'g-', label="Train MAE")
        plt.plot(history.history["val_mae"], 'orange', linestyle='--', label="Validation MAE")
        plt.title(f"Sigma Band MAE - {channel}")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Pulizia memoria
        del sigma_autoencoder
        tf.keras.backend.clear_session()
        gc.collect()
        
        print(f"✅ Completato training sigma per {channel}")

if __name__ == "__main__":
    process_edf_for_spindles()
    print("\n🎉 Training autoencoder banda sigma completato!")