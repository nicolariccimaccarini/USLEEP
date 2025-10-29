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
    'batch_size': 256,
    'epochs': 200,
    'patience': 20,
    'sigma_low': 9,          # Banda sigma: 9-15 Hz
    'sigma_high': 15,
    'channels_to_exclude': {'EEG A1', 'EEG A2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2'}
}

def create_sigma_autoencoder(n_sigma_features):
    """
    Autoencoder con regolarizzazione per features più discriminative
    """
    input_layer = Input(shape=(1, n_sigma_features), name='sigma_input')
    
    # Encoder con regolarizzazione per variabilità
    encoded = LSTM(64, activation='tanh', return_sequences=True, 
                   kernel_regularizer=tf.keras.regularizers.l2(0.001),
                   name='encoder_lstm_1')(input_layer)
    encoded = Dropout(0.3, name='encoder_dropout_1')(encoded)
    
    encoded = LSTM(32, activation='tanh', return_sequences=False,
                   kernel_regularizer=tf.keras.regularizers.l2(0.001), 
                   name='encoder_lstm_2')(encoded)
    encoded = Dropout(0.2, name='encoder_dropout_2')(encoded)
    
    # Bottleneck 
    encoded = Dense(16, activation='tanh', 
                   kernel_regularizer=tf.keras.regularizers.l2(0.001),
                   name='encoder_dense')(encoded)
    
    # Decoder
    decoded = RepeatVector(1, name='repeat_vector')(encoded)
    decoded = LSTM(32, activation='tanh', return_sequences=True, name='decoder_lstm_1')(decoded)
    decoded = LSTM(64, activation='tanh', return_sequences=True, name='decoder_lstm_2')(decoded)
    decoded = TimeDistributed(Dense(n_sigma_features, activation='linear'), name='sigma_output')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded, name='sigma_autoencoder')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=optimizer, loss=MeanSquaredError(), metrics=['mae'])
    
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
    
    print("🧠 Processamento EEG con features discriminative...")
    for file in filenames:
        if not file.endswith('.edf'):
            continue
            
        file_path = os.path.join(path_edf, file)
        print(f"📁 Processando: {file}")
        
        # Carica EEG
        raw = mne.io.read_raw_edf(file_path, preload=True)
        sfreq = raw.info['sfreq']
        
        channels_to_include = [ch for ch in raw.ch_names if ch not in CONFIG['channels_to_exclude']]
        raw.pick_channels(channels_to_include)
        
        strategy = tf.distribute.MirroredStrategy()
        
        for channel_idx, channel in enumerate(raw.ch_names):
            print(f"\n🧠 Processando canale {channel} ({channel_idx+1}/{len(raw.ch_names)})")
            
            # Estrai solo i dati di questo canale
            channel_data = raw.get_data()[channel_idx:channel_idx+1, :]
            
            # Filtro banda sigma + segmentazione
            print("🔧 Applicando filtro banda sigma...")
            sigma_filtered_data = apply_sigma_band_filter(
                channel_data, sfreq, CONFIG['sigma_low'], CONFIG['sigma_high']
            )
            
            segment_length = int(CONFIG['window_size'] * sfreq)
            segments = segment_signal_with_overlap(
                sigma_filtered_data, segment_length, CONFIG['overlap_ratio']
            )
            
            # Features per questo canale
            sigma_powers, _ = compute_sigma_power_spectrum(segments, sfreq)
            
            if not sigma_powers:
                print(f"⚠️ Nessuna feature per {channel}")
                continue
            
            print(f"🔍 DEBUG: Tipo sigma_powers: {type(sigma_powers)}")
            print(f"🔍 DEBUG: Lunghezza sigma_powers: {len(sigma_powers)}")
            if len(sigma_powers) > 0:
                print(f"🔍 DEBUG: Tipo primo elemento: {type(sigma_powers[0])}")
                print(f"🔍 DEBUG: Shape primo elemento: {sigma_powers[0].shape if hasattr(sigma_powers[0], 'shape') else 'N/A'}")
            
            channel_features = []
            for power_features in sigma_powers:
                channel_features.append(power_features)
            
            if not channel_features:
                print(f"⚠️ Nessuna feature estratta per {channel}")
                continue
                
            # Reshape per training: (n_segments, 1, 5_features)
            all_sigma_segments = np.array(channel_features).reshape(-1, 1, 7)
            
            print(f"📊 Segmenti per {channel}: {all_sigma_segments.shape[0]}")
            
            # Training
            model_path = os.path.join(weights_path, f"sigma_autoencoder_{channel}.h5")
            plot_path = os.path.join(images_path, f"sigma_training_{channel}.png")
            
            try:
                with strategy.scope():
                    sigma_autoencoder = create_sigma_autoencoder(7)
                
                early_stopping = EarlyStopping(
                    monitor='val_loss', 
                    patience=CONFIG['patience'], 
                    verbose=1, 
                    restore_best_weights=True,
                    min_delta=0.0001  # Soglia minima per miglioramento
                )

                # Training più lungo con learning rate scheduling
                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
                )
                
                history = sigma_autoencoder.fit(
                    all_sigma_segments, all_sigma_segments,
                    epochs=CONFIG['epochs'],
                    batch_size=CONFIG['batch_size'],
                    validation_split=0.2,
                    callbacks=[early_stopping, lr_scheduler],
                    verbose=1
                )
                
                # Salva modello
                sigma_autoencoder.save(model_path)
                
                # Plot training
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(history.history["loss"], 'r-', label="Train Loss")
                plt.plot(history.history["val_loss"], 'b--', label="Val Loss")
                plt.title(f"Training - {channel}")
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(history.history["mae"], 'g-', label="Train MAE")
                plt.plot(history.history["val_mae"], 'orange', linestyle='--', label="Val MAE")
                plt.title(f"MAE - {channel}")
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Completato training per {channel}")
                
            except Exception as e:
                print(f"❌ Errore training {channel}: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                # Pulizia memoria aggressiva
                if 'sigma_autoencoder' in locals():
                    del sigma_autoencoder
                if 'history' in locals():
                    del history
                tf.keras.backend.clear_session()
                gc.collect()
        
        # Pulizia finale
        del raw
        gc.collect()

if __name__ == "__main__":
    process_edf_for_spindles()
    print("\n🎉 Training autoencoder banda sigma completato!")
