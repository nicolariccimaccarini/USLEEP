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
    Crea autoencoder specializzato per features della banda sigma
    """
    input_layer = Input(shape=(1, n_sigma_features), name='sigma_input')
    
    # Encoder più profondo per catturare patterns sigma
    encoded = LSTM(256, activation='tanh', return_sequences=True, name='encoder_lstm_1')(input_layer)
    encoded = Dropout(0.3, name='encoder_dropout_1')(encoded)
    encoded = LSTM(128, activation='tanh', return_sequences=True, name='encoder_lstm_2')(encoded)
    encoded = Dropout(0.3, name='encoder_dropout_2')(encoded)
    encoded = LSTM(64, activation='tanh', return_sequences=False, name='encoder_lstm_3')(encoded)
    
    # Bottleneck per feature compatte degli spindles
    encoded = Dense(32, activation='tanh', name='sigma_bottleneck')(encoded)
    encoded = Dense(16, activation='tanh', name='spindle_features')(encoded)  # Features finali spindle
    
    # Decoder
    decoded = RepeatVector(1, name='repeat_vector')(encoded)
    decoded = LSTM(64, activation='tanh', return_sequences=True, name='decoder_lstm_1')(decoded)
    decoded = Dropout(0.3, name='decoder_dropout_1')(decoded)
    decoded = LSTM(128, activation='tanh', return_sequences=True, name='decoder_lstm_2')(decoded)
    decoded = Dropout(0.3, name='decoder_dropout_2')(decoded)
    decoded = LSTM(256, activation='tanh', return_sequences=True, name='decoder_lstm_3')(decoded)
    decoded = TimeDistributed(Dense(n_sigma_features, activation='linear'), name='sigma_output')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded, name='sigma_autoencoder')
    autoencoder.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])
    
    return autoencoder

def process_edf_for_spindles():
    """Processa i file EEG focalizzandosi sulla banda sigma per spindle detection"""
    
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
    
    print("🧠 Processamento EEG per banda sigma (9-15 Hz)...")
    for file in filenames:
        if not file.endswith('.edf'):
            continue
            
        file_path = os.path.join(path_edf, file)
        print(f"📁 Processando: {file}")
        
        # Carica file EEG
        raw = mne.io.read_raw_edf(file_path, preload=True)
        sfreq = raw.info['sfreq']
        
        print(f"📊 Frequenza campionamento: {sfreq} Hz")
        
        # Filtra canali EEG
        channels_to_include = [ch for ch in raw.ch_names if ch not in CONFIG['channels_to_exclude']]
        raw.pick_channels(channels_to_include)
        
        # Applica filtro banda sigma (9-15 Hz) - STEP FONDAMENTALE
        print("🔧 Applicando filtro banda sigma (9-15 Hz)...")
        sigma_filtered_data = apply_sigma_band_filter(
            raw.get_data(), 
            sfreq, 
            CONFIG['sigma_low'], 
            CONFIG['sigma_high']
        )
        
        # Segmentazione con alta risoluzione temporale (0.1s step)
        segment_length = int(CONFIG['window_size'] * sfreq)
        segments = segment_signal_with_overlap(
            sigma_filtered_data, 
            segment_length, 
            CONFIG['overlap_ratio']
        )
        
        print(f"📏 Segmenti generati: {len(segments)} (risoluzione: {CONFIG['window_size'] * (1-CONFIG['overlap_ratio'])}s)")
        
        # Calcola potenza nella banda sigma
        sigma_powers, sigma_freqs = compute_sigma_power_spectrum(segments, sfreq)
        normalized_sigma_powers = [normalize_spectrum(power) for power in sigma_powers]
        
        # Aggregazione per canale
        for idx, channel in enumerate(raw.ch_names):
            if channel not in aggregated_sigma_data:
                aggregated_sigma_data[channel] = []
            for norm_power in normalized_sigma_powers:
                aggregated_sigma_data[channel].append(norm_power[idx])
    
    # Training degli autoencoder specializzati per banda sigma
    strategy = tf.distribute.MirroredStrategy()
    
    print(f"\n🤖 Training autoencoder sigma-specifici per {len(aggregated_sigma_data)} canali...")
    for channel_idx, (channel, data) in enumerate(aggregated_sigma_data.items(), 1):
        print(f"\n🧠 Canale {channel} ({channel_idx}/{len(aggregated_sigma_data)})")
        
        # Preparazione dati sigma
        data = np.array(data)
        n_sigma_features = data.shape[1]
        all_sigma_segments = data.reshape((-1, 1, n_sigma_features))
        
        print(f"📊 Features sigma per segmento: {n_sigma_features}")
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