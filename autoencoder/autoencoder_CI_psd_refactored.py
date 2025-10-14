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
    compute_spectrum_numpy, normalize_spectrum
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

# Parametri di configurazione
CONFIG = {
    'window_size': 0.5,  # Finestra sliding window (0.5s)
    'overlap_ratio': 0.2,  # 20% di sovrapposizione (0.1s overlap su 0.5s window)
    'batch_size': 256,
    'epochs': 200,
    'patience': 20,
    'channels_to_exclude': {'EEG A1', 'EEG A2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2'}
}

def create_autoencoder_model(n_frequencies):
    """Crea l'architettura dell'autoencoder"""
    input_layer = Input(shape=(1, n_frequencies))
    
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
    decoded = TimeDistributed(Dense(n_frequencies), name='time_distributed_output')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss=MeanSquaredError())
    
    return autoencoder

def process_edf_files():
    """Processa i file EDF e addestra gli autoencoder per canale"""
    
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
    weights_path = os.path.join(dirData, "model", "canali_individuali")
    images_path = os.path.join(dirData, "images", "canali_individuali")
    
    os.makedirs(weights_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    
    # Aggregazione dati per canale
    aggregated_data = {}
    
    print("📊 Processamento file EDF...")
    for file in filenames:
        if not file.endswith('.edf'):
            continue
            
        file_path = os.path.join(path_edf, file)
        print(f"📁 Processando: {file}")
        
        # Carica file EDF
        raw = mne.io.read_raw_edf(file_path, preload=True)
        sfreq = raw.info['sfreq']
        
        # Filtra canali
        channels_to_include = [ch for ch in raw.ch_names if ch not in CONFIG['channels_to_exclude']]
        raw.pick_channels(channels_to_include)
        
        # Segmentazione con sliding window
        segment_length = int(CONFIG['window_size'] * sfreq)
        segments = segment_signal_with_overlap(
            raw.get_data(), 
            segment_length, 
            CONFIG['overlap_ratio']
        )
        
        # Calcolo spettri
        spectrums, _ = compute_spectrum_numpy(segments, sfreq)
        normalized_spectrums = [normalize_spectrum(spectrum) for spectrum in spectrums]
        
        # Aggregazione per canale
        for idx, channel in enumerate(raw.ch_names):
            if channel not in aggregated_data:
                aggregated_data[channel] = []
            for norm_spectrum in normalized_spectrums:
                aggregated_data[channel].append(norm_spectrum[idx])
    
    # Training degli autoencoder
    strategy = tf.distribute.MirroredStrategy()
    
    print(f"\n🤖 Addestramento autoencoder per {len(aggregated_data)} canali...")
    for channel_idx, (channel, data) in enumerate(aggregated_data.items(), 1):
        print(f"\n🔧 Canale {channel} ({channel_idx}/{len(aggregated_data)})")
        
        # Preparazione dati
        data = np.array(data)
        n_frequencies = data.shape[1]
        all_segments_standardized = data.reshape((-1, 1, n_frequencies))
        
        # Percorsi di salvataggio
        model_path = os.path.join(weights_path, f"autoencoder_{channel}.h5")
        plot_path = os.path.join(images_path, f"training_{channel}.png")
        
        # Creazione e addestramento modello
        with strategy.scope():
            autoencoder = create_autoencoder_model(n_frequencies)
        
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
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history["loss"], 'r', marker='.', label="Train Loss")
        plt.plot(history.history["val_loss"], 'b--', marker='.', label="Validation Loss")
        plt.title(f"Training History - {channel}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Pulizia memoria
        del autoencoder
        tf.keras.backend.clear_session()
        gc.collect()
        
        print(f"✅ Completato {channel}")

if __name__ == "__main__":
    process_edf_files()
    print("\n🎉 Addestramento completato!")