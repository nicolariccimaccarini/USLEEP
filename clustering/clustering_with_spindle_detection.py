import mne 
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model, load_model
import sys
from scipy.signal import savgol_filter
import gc

# Aggiungi il percorso utils al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from signal_processing import (
    get_file_output_path, segment_signal_with_overlap,
    apply_smoothing, detect_spindle_regions, convert_regions_to_time
)

# Abilita deserializzazione
tf.keras.config.enable_unsafe_deserialization()

# Configurazione
CONFIG = {
    'window_size': 0.5,                 # Finestra di 0.5s per risoluzione freq ~2Hz
    'overlap_ratio': 0.5,               # 50% overlap -> step 0.25s per precisione temporale
    'num_clusters': 3,
    'spindle_threshold_type': 'z_score', # Usa z-score normalizzato
    'spindle_threshold': 2.3,           # z >= 2.3 (95-99° percentile)
    'min_spindle_duration': 0.5,        # Durata minima spindle (standard letteratura)
    'max_spindle_duration': 2.0,        # Durata massima (più conservativo)
    'smoothing_window_sec': 0.25,       # 0.25s smoothing window
    'sigma_band': (11, 16),             # Banda sigma per spindles (Hz)
    'context_window_sec': 30,           # Finestra per normalizzazione locale
    'channels_to_exclude': {'EEG A1', 'EEG A2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2'}
}

def load_autoencoder_with_fallback(model_path):
    """Carica il modello autoencoder con fallback"""
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"⚠️ Errore caricamento {model_path}: {e}")
        return None

def compute_anomaly_scores(encoder, segments, threshold_percentile=90):
    """
    Calcola i punteggi di anomalia per ogni segmento
    
    Args:
        encoder: modello encoder
        segments: segmenti preprocessati
        threshold_percentile: percentile per definire la soglia di anomalia
    
    Returns:
        array di punteggi normalizzati (0-1)
    """
    # Ottieni features codificate
    features = encoder.predict(segments, verbose=0)
    features_flat = features.reshape(features.shape[0], -1)
    
    # Clustering per identificare pattern anomali
    kmeans = KMeans(n_clusters=CONFIG['num_clusters'], random_state=42)
    cluster_labels = kmeans.fit_predict(features_flat)
    
    # Calcola distanze dai centroidi
    distances = []
    for i, feature in enumerate(features_flat):
        cluster_center = kmeans.cluster_centers_[cluster_labels[i]]
        distance = np.linalg.norm(feature - cluster_center)
        distances.append(distance)
    
    distances = np.array(distances)
    
    # Normalizza i punteggi (distanza maggiore = più anomalo = più probabile spindle)
    scores = (distances - distances.min()) / (distances.max() - distances.min())
    
    return scores, cluster_labels

def compute_sigma_power(segments, sfreq, sigma_band=(11, 16)):
    """
    Calcola la potenza nella banda sigma per ogni segmento
    
    Args:
        segments: segmenti EEG
        sfreq: frequenza di campionamento
        sigma_band: tuple (freq_min, freq_max) per banda sigma
    
    Returns:
        array di potenze sigma normalizzate
    """
    from scipy.signal import welch
    
    sigma_powers = []
    
    for segment in segments:
        # Calcola PSD usando Welch
        freqs, psd = welch(segment, fs=sfreq, nperseg=len(segment)//2)
        
        # Trova indici per banda sigma
        sigma_mask = (freqs >= sigma_band[0]) & (freqs <= sigma_band[1])
        
        # Calcola potenza media nella banda sigma
        sigma_power = np.mean(psd[sigma_mask])
        sigma_powers.append(sigma_power)
    
    return np.array(sigma_powers)

def compute_z_score_threshold(signal, context_window_samples, threshold_z=2.3):
    """
    Calcola soglia basata su z-score con normalizzazione locale
    
    Args:
        signal: segnale di potenza sigma
        context_window_samples: finestra per calcolo statistiche locali
        threshold_z: soglia z-score
    
    Returns:
        array di z-scores, soglia binaria
    """
    z_scores = np.zeros_like(signal)
    
    for i in range(len(signal)):
        # Definisci finestra di contesto attorno al campione corrente
        start_idx = max(0, i - context_window_samples // 2)
        end_idx = min(len(signal), i + context_window_samples // 2)
        
        context = signal[start_idx:end_idx]
        
        # Calcola statistiche robuste
        mean_context = np.mean(context)
        std_context = np.std(context)
        
        # Calcola z-score
        if std_context > 0:
            z_scores[i] = (signal[i] - mean_context) / std_context
        else:
            z_scores[i] = 0
    
    # Applica soglia
    threshold_binary = z_scores >= threshold_z
    
    return z_scores, threshold_binary

def process_channel_for_spindles(channel_name, data, sfreq, encoder):
    """
    Processa un singolo canale per rilevare spindles con parametri ottimizzati
    
    Returns:
        DataFrame con i risultati dei spindles rilevati
    """
    print(f"🔍 Analisi spindles per canale: {channel_name}")
    
    # Prepara segmenti con overlap ottimizzato
    segment_length = int(CONFIG['window_size'] * sfreq)
    segments = segment_signal_with_overlap(data, segment_length, CONFIG['overlap_ratio'])
    
    if len(segments) == 0:
        return pd.DataFrame(columns=['Canale', 'Start_Time(s)', 'End_Time(s)'])
    
    # Metodo 1: Calcola potenza sigma diretta (più affidabile per spindles)
    sigma_powers = compute_sigma_power(segments, sfreq, CONFIG['sigma_band'])
    
    # Metodo 2: Usa anche features dell'autoencoder (per confronto)
    from signal_processing import compute_spectrum_numpy, normalize_spectrum
    
    spectrums, _ = compute_spectrum_numpy(segments, sfreq)
    normalized_spectrums = [normalize_spectrum(spectrum) for spectrum in spectrums]
    
    channel_spectra = np.array([spec[0] for spec in normalized_spectrums])
    encoder_input = channel_spectra.reshape(-1, 1, channel_spectra.shape[1])
    
    # Ottieni features encoder
    features = encoder.predict(encoder_input, verbose=0)
    features_flat = features.reshape(features.shape[0], -1)
    
    # Combina potenza sigma e features encoder
    # Usa correlazione pesata: sigma power (70%) + encoder features (30%)
    encoder_anomaly = np.linalg.norm(features_flat, axis=1)
    encoder_anomaly = (encoder_anomaly - encoder_anomaly.min()) / (encoder_anomaly.max() - encoder_anomaly.min())
    
    combined_signal = 0.7 * sigma_powers + 0.3 * encoder_anomaly
    
    # Normalizzazione locale con z-score
    context_window_samples = int(CONFIG['context_window_sec'] / CONFIG['window_size'] * (1 - CONFIG['overlap_ratio']))
    
    if CONFIG['spindle_threshold_type'] == 'z_score':
        z_scores, threshold_binary = compute_z_score_threshold(
            combined_signal, 
            context_window_samples, 
            CONFIG['spindle_threshold']
        )
        detection_signal = z_scores
        binary_threshold = threshold_binary
    else:
        # Fallback a percentile
        threshold_val = np.percentile(combined_signal, CONFIG['spindle_threshold'])
        detection_signal = combined_signal
        binary_threshold = combined_signal >= threshold_val
    
    # Smoothing temporale
    smoothing_window_samples = int(CONFIG['smoothing_window_sec'] / CONFIG['window_size'] * (1 - CONFIG['overlap_ratio']))
    smoothed_binary = apply_smoothing(
        binary_threshold.astype(float), 
        window_size=smoothing_window_samples, 
        method='moving_average'
    ) >= 0.5  # Riconverti a binario
    
    # Rileva regioni spindle con durate corrette
    step_duration = CONFIG['window_size'] * (1 - CONFIG['overlap_ratio'])
    min_duration_samples = int(CONFIG['min_spindle_duration'] / step_duration)
    max_duration_samples = int(CONFIG['max_spindle_duration'] / step_duration)
    
    spindle_regions = detect_spindle_regions(
        smoothed_binary.astype(float), 
        threshold=0.5,  # Già binario
        min_duration_samples=min_duration_samples
    )
    
    # Filtra per durata massima
    spindle_regions = [
        (start, end) for start, end in spindle_regions 
        if end - start <= max_duration_samples
    ]
    
    # Converti in tempi reali
    time_regions = convert_regions_to_time(
        spindle_regions, 
        segment_length, 
        CONFIG['overlap_ratio'], 
        sfreq
    )
    
    # Crea DataFrame risultati
    results = []
    for start_time, end_time in time_regions:
        # Verifica durata finale
        duration = end_time - start_time
        if CONFIG['min_spindle_duration'] <= duration <= CONFIG['max_spindle_duration']:
            results.append({
                'Canale': channel_name,
                'Start_Time(s)': round(start_time, 3),
                'End_Time(s)': round(end_time, 3)
            })
    
    print(f"✅ Rilevati {len(results)} spindles per {channel_name}")
    
    # Debug info
    print(f"   📊 Sigma power - Media: {np.mean(sigma_powers):.3f}, Std: {np.std(sigma_powers):.3f}")
    print(f"   📊 Z-scores - Max: {np.max(detection_signal):.2f}, Soglia: {CONFIG['spindle_threshold']}")
    print(f"   📊 Regioni candidate: {len(spindle_regions)}")
    
    return pd.DataFrame(results)

def main():
    """Funzione principale per clustering e rilevamento spindles"""
    
    # Configurazione percorsi
    path_edf = os.environ.get('DATA_PATH', 'Data/Edf')
    output_path = os.environ.get('OUTPUT_PATH', 'Data/Output')
    current_file = os.environ.get('CURRENT_FILE', None)
    
    # Determina file da processare
    if current_file:
        dirData = get_file_output_path(output_path, current_file)
        filenames = [current_file]
        print(f"📁 Modalità file singolo: {current_file}")
    else:
        dirData = output_path
        filenames = [f for f in os.listdir(path_edf) if f.endswith('.edf')]
        print(f"📁 Modalità batch: {len(filenames)} file")
    
    # Percorsi
    weights_path = os.path.join(dirData, "model", "canali_individuali")
    cluster_output_path = os.path.join(dirData, "cluster")
    images_path = os.path.join(dirData, "images", "clustering")
    
    os.makedirs(cluster_output_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    
    # Lista per raccogliere tutti i risultati
    all_spindle_results = []
    
    for file in filenames:
        if not file.endswith('.edf'):
            continue
            
        file_path = os.path.join(path_edf, file)
        print(f"\n📊 Processando: {file}")
        
        try:
            # Carica dati EEG
            raw = mne.io.read_raw_edf(file_path, preload=True)
            sfreq = raw.info['sfreq']
            
            # Filtra canali
            channels_to_include = [ch for ch in raw.ch_names if ch not in CONFIG['channels_to_exclude']]
            raw.pick_channels(channels_to_include)
            
            # Processa ogni canale
            for channel_name in raw.ch_names:
                # Carica modello per il canale
                model_path = os.path.join(weights_path, f"autoencoder_{channel_name}.h5")
                
                if not os.path.exists(model_path):
                    print(f"⚠️ Modello non trovato per {channel_name}: {model_path}")
                    continue
                
                autoencoder = load_autoencoder_with_fallback(model_path)
                if autoencoder is None:
                    continue
                
                # Crea encoder
                encoder = Model(
                    inputs=autoencoder.input, 
                    outputs=autoencoder.get_layer('encoder_dense').output
                )
                
                # Estrai dati del canale
                channel_idx = raw.ch_names.index(channel_name)
                channel_data = raw.get_data()[channel_idx:channel_idx+1, :]  # Mantieni dimensione 2D
                
                # Processa per spindles
                channel_results = process_channel_for_spindles(
                    channel_name, channel_data, sfreq, encoder
                )
                
                if len(channel_results) > 0:
                    all_spindle_results.append(channel_results)
                
                # Pulizia memoria
                del autoencoder, encoder
                gc.collect()
                
        except Exception as e:
            print(f"❌ Errore processando {file}: {e}")
            continue
    
    # Combina e salva risultati
    if all_spindle_results:
        final_results = pd.concat(all_spindle_results, ignore_index=True)
        
        # Ordina per canale e tempo di inizio
        final_results = final_results.sort_values(['Canale', 'Start_Time(s)'])
        
        # Salva CSV
        csv_path = os.path.join(cluster_output_path, 'start_end_per_channel.csv')
        final_results.to_csv(csv_path, index=False)
        
        print(f"\n🎉 Risultati salvati in: {csv_path}")
        print(f"📈 Totale spindles rilevati: {len(final_results)}")
        print("\n📊 Riepilogo per canale:")
        summary = final_results.groupby('Canale').size()
        for channel, count in summary.items():
            print(f"  {channel}: {count} spindles")
        
    else:
        print("❌ Nessun spindle rilevato")

if __name__ == "__main__":
    main()