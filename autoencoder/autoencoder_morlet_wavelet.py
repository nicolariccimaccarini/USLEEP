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
    compute_morlet_wavelet, mne_bandpass_filter
)


CONFIG = {
    'window_size': 0.5,                             # Finestra segmento (s)
    'overlap_ratio': 0.5,                           # 50% overlap per smoothing
    'batch_size': 256,
    'epochs': 200,
    'patience': 20,
    'wavelet_fc': 13.5,                             # Frequenza centrale (Hz) - centro banda sigma
    'wavelet_n_cycles': 7,                          # Numero cicli Morlet
    'n_envelope_features': 4,                       # Feature estratte da envelope
    'channels_to_exclude': {'EEG A1', 'EEG A2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2'}
}


def extract_envelope_features(envelope_segment):
    """
    Estrae feature statistiche semplici dall'envelope Morlet
    
    Args:
        envelope_segment: array 1D di ampiezza Morlet
    
    Returns:
        array di 4 feature: [mean, std, max, median]
    """
    return np.array([
        np.mean(envelope_segment),      # Ampiezza media
        np.std(envelope_segment),       # Variabilità
        np.max(envelope_segment),       # Picco massimo
        np.median(envelope_segment)     # Valore mediano
    ])


def create_autoencoder_model(n_features=4):
    """
    Crea l'architettura dell'autoencoder per feature envelope Morlet
    
    Args:
        n_features: numero di feature in input (default: 4 statistiche base)
    
    Returns:
        modello autoencoder compilato
    """
    input_layer = Input(shape=(1, n_features))
    
    # Encoder LSTM 
    encoded = LSTM(64, activation='relu', return_sequences=True, name='encoder_lstm_1')(input_layer)
    encoded = LSTM(32, activation='relu', return_sequences=True, name='encoder_lstm_2')(encoded)
    encoded = Dropout(0.2, name='encoder_dropout')(encoded)
    encoded = LSTM(16, activation='relu', return_sequences=False, name='encoder_lstm_3')(encoded)
    encoded = Dense(8, activation='relu', name='encoder_dense')(encoded)
    
    # Decoder 
    decoded = RepeatVector(1, name='repeat_vector')(encoded)
    decoded = LSTM(16, activation='relu', return_sequences=True, name='decoder_lstm_1')(decoded)
    decoded = LSTM(32, activation='relu', return_sequences=True, name='decoder_lstm_2')(decoded)
    decoded = LSTM(64, activation='relu', return_sequences=True, name='decoder_lstm_3')(decoded)
    decoded = TimeDistributed(Dense(n_features), name='time_distributed_output')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss=MeanSquaredError())
    
    return autoencoder


def process_edf_files():
    """
    Processa i file EDF con Morlet Wavelet e addestra autoencoder su envelope
    
    Pipeline:
    1. Carica segnale grezzo (NO pre-filtering)
    2. Applica Morlet Wavelet per canale
    3. Estrai envelope (ampiezza)
    4. Segmenta envelope
    5. Estrai feature statistiche da ogni segmento
    6. Addestra autoencoder per anomaly detection
    """
    
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
    
    print("\n📊 Processamento file EDF con Morlet Envelope...")
    print(f"   🌊 Parametri Morlet: fc={CONFIG['wavelet_fc']} Hz, cycles={CONFIG['wavelet_n_cycles']}")
    
    for file in filenames:
        if not file.endswith('.edf'):
            continue
            
        file_path = os.path.join(path_edf, file)
        print(f"\n📁 Processando: {file}")
        
        # Carica file EDF
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        
        # Filtra canali
        channels_to_include = [ch for ch in raw.ch_names if ch not in CONFIG['channels_to_exclude']]
        raw.pick_channels(channels_to_include)
        
        print(f"   📊 Canali: {len(channels_to_include)}, Frequenza: {sfreq} Hz")
        
        raw_data = raw.get_data()
        
        segment_length = int(CONFIG['window_size'] * sfreq)
        
        print(f"   🔧 Lunghezza segmento: {segment_length} campioni ({CONFIG['window_size']}s)")
        
        # Processa ogni canale
        for ch_idx, channel in enumerate(raw.ch_names):
            channel_data = raw_data[ch_idx]
            
            # Morlet Wavelet Transform
            wavelet_complex = compute_morlet_wavelet(
                channel_data, 
                sfreq, 
                fc=CONFIG['wavelet_fc'],
                n_cycles=CONFIG['wavelet_n_cycles']
            )
            
            # Estrai envelope (ampiezza)
            envelope = np.abs(wavelet_complex)
            
            # Segmenta envelope
            envelope_reshaped = envelope.reshape(1, -1)
            envelope_segments = segment_signal_with_overlap(
                envelope_reshaped,
                segment_length,
                CONFIG['overlap_ratio']
            )
            
            # Estrai feature da ogni segmento envelope
            channel_features = []
            for seg in envelope_segments:
                seg_1d = seg.flatten()
                features = extract_envelope_features(seg_1d)
                channel_features.append(features)
            
            if channel not in aggregated_data:
                aggregated_data[channel] = []
            
            # Aggiungi feature
            aggregated_data[channel].extend(channel_features)
        
        print(f"   ✅ File processato: {len(envelope_segments)} segmenti per canale")
    
    # Verifica aggregazione
    if not aggregated_data:
        print("❌ Nessun dato aggregato!")
        return
    
    print(f"\n📊 Dati aggregati per {len(aggregated_data)} canali")
    
    # Training degli autoencoder
    strategy = tf.distribute.MirroredStrategy()
    
    print(f"\n🤖 Addestramento autoencoder su envelope features...")
    for channel_idx, (channel, data) in enumerate(aggregated_data.items(), 1):
        print(f"\n🔧 Canale {channel} ({channel_idx}/{len(aggregated_data)})")
        
        # Preparazione dati
        data = np.array(data)
        n_samples = data.shape[0]
        n_features = CONFIG['n_envelope_features']
        
        # Reshape per autoencoder: (n_samples, 1, n_features)
        data_reshaped = data.reshape((-1, 1, n_features))
        
        print(f"   📊 Campioni: {n_samples}, Feature: {n_features}")
        print(f"   📐 Shape input: {data_reshaped.shape}")
        
        # Percorsi di salvataggio
        model_path = os.path.join(weights_path, f"autoencoder_{channel}.h5")
        plot_path = os.path.join(images_path, f"training_{channel}.png")
        
        # Creazione e addestramento modello
        with strategy.scope():
            autoencoder = create_autoencoder_model(n_features)
        
        print(f"   🏗️ Modello creato: {autoencoder.count_params()} parametri")
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=CONFIG['patience'], 
            verbose=1, 
            restore_best_weights=True
        )
        
        history = autoencoder.fit(
            data_reshaped, data_reshaped,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        print(f"   📉 Loss finale: {final_loss:.6f}, Val Loss: {final_val_loss:.6f}")
        
        # Salvataggio modello
        autoencoder.save(model_path)
        print(f"   💾 Modello salvato: {model_path}")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history["loss"], 'r', marker='.', label="Train Loss", linewidth=2)
        plt.plot(history.history["val_loss"], 'b--', marker='.', label="Validation Loss", linewidth=2)
        plt.title(f"Training History (Morlet Envelope) - {channel}", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss (MSE)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
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
