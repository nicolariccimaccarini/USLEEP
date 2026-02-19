import mne 
import pandas as pd
import os
import sys
import gc
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Aggiungi il percorso utils al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from signal_processing import (
    get_file_output_path, segment_signal_with_overlap,
    compute_morlet_wavelet, compute_adaptive_threshold, 
    merge_close_spindles, mne_bandpass_filter
)

# Abilita deserializzazione
tf.keras.config.enable_unsafe_deserialization()

# Configurazione
CONFIG = {
    'window_size': 0.5,                         # Ampiezza finestra temporale (s)
    'overlap_ratio': 0.5,                       # 50% overlap (ALLINEATO con training)
    'num_clusters': 2,                          # (spindles vs non-spindles)
    'min_spindle_duration': 0.5,                # Durata minima spindle (s)
    'max_spindle_duration': 3.0,                # Durata massima spindle (s)
    'wavelet_fc': 13.5,                         # Frequenza centrale Morlet (Hz)
    'wavelet_n_cycles': 7,                      # Numero cicli Morlet
    'rms_window_sec': 30.0,                     # Finestra RMS per threshold adattivo
    'rms_percentile': 0.25,                       # Percentile per threshold
    'merge_gap_sec': 1.0,                       # Gap minimo per merge spindles
    'min_amplitude_ratio': 0.95,                # % campioni sopra threshold RMS
    'channels_to_exclude': {'EEG A1', 'EEG A2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2'}
}


def load_autoencoder_with_fallback(model_path):
    """Carica il modello autoencoder con fallback"""
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"⚠️ Errore caricamento {model_path}: {e}")
        return None


def extract_envelope_features(envelope_segment):
    """
    Estrae feature statistiche dall'envelope (IDENTICHE al training)
    
    Args:
        envelope_segment: array 1D di ampiezza Morlet
    
    Returns:
        array di 4 feature: [mean, std, max, median]
    """
    return np.array([
        np.mean(envelope_segment),
        np.std(envelope_segment),
        np.max(envelope_segment),
        np.median(envelope_segment)
    ])


def detect_continuous_regions_from_mask(mask, step_samples, sfreq):
    """
    Converte maschera binaria in regioni temporali continue
    
    Args:
        mask: array booleano (True = spindle)
        step_samples: passo tra segmenti
        sfreq: frequenza campionamento
    
    Returns:
        lista di tuple (start_time, end_time)
    """
    regions = []
    in_region = False
    start_idx = 0
    
    for i, is_spindle in enumerate(mask):
        if is_spindle and not in_region:
            start_idx = i
            in_region = True
        elif not is_spindle and in_region:
            start_time = start_idx * step_samples / sfreq
            end_time = i * step_samples / sfreq
            regions.append((start_time, end_time))
            in_region = False
    
    # Chiudi ultima regione se aperta
    if in_region:
        start_time = start_idx * step_samples / sfreq
        end_time = len(mask) * step_samples / sfreq
        regions.append((start_time, end_time))
    
    return regions


def process_channel_for_spindles_hybrid(channel_name, data, sfreq, encoder):
    """
    Approccio ibrido ML + Morlet (nvelope-based)
    
    Pipeline:
    1. Morlet Wavelet Transform (fc=13.5 Hz)
    2. Estrazione envelope (ampiezza)
    3. Segmentazione envelope
    4. Estrazione feature statistiche (mean, std, max, median)
    5. Autoencoder: anomaly detection (reconstruction error)
    6. Clustering: spindles vs non-spindles
    7. Refinement: criteri Morlet (durata + RMS threshold)
    8. Merge spindles vicini
    """
    print(f"🔍 Analisi spindles IBRIDA (Envelope) per: {channel_name}")
    
    channel_data_1d = data.flatten()
    
    # Morlet Wavelet Transform 
    wavelet_complex = compute_morlet_wavelet(
        channel_data_1d, 
        sfreq, 
        fc=CONFIG['wavelet_fc'],
        n_cycles=CONFIG['wavelet_n_cycles']
    )
    
    # Estrai envelope (ampiezza)
    envelope = np.abs(wavelet_complex)
    
    print(f"   🌊 Morlet Wavelet applicata (fc={CONFIG['wavelet_fc']} Hz, {CONFIG['wavelet_n_cycles']} cycles)")
    
    # Segmentazione envelope (STESSO metodo del training)
    segment_length = int(CONFIG['window_size'] * sfreq)
    envelope_reshaped = envelope.reshape(1, -1)
    envelope_segments = segment_signal_with_overlap(
        envelope_reshaped,
        segment_length,
        CONFIG['overlap_ratio']
    )
    
    print(f"   📏 Segmenti envelope: {len(envelope_segments)}")
    
    # Estrai feature da ogni segmento (IDENTICHE al training)
    segment_features = []
    for seg in envelope_segments:
        seg_1d = seg.flatten()
        features = extract_envelope_features(seg_1d)
        segment_features.append(features)
    
    segment_features = np.array(segment_features)
    
    print(f"   📊 Feature estratte: {segment_features.shape}")
    
    # Normalizza features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(segment_features)
    
    # Reshape per autoencoder: (n_samples, 1, 4)
    features_reshaped = features_normalized.reshape(-1, 1, 4)
    
    # Autoencoder prediction (reconstruction error)
    try:
        predictions = encoder.predict(features_reshaped, verbose=0)
        reconstruction_errors = np.mean((features_reshaped - predictions)**2, axis=(1, 2))
        print(f"   🤖 Reconstruction error - mean: {np.mean(reconstruction_errors):.6f}, std: {np.std(reconstruction_errors):.6f}")
    except Exception as e:
        print(f"   ⚠️ Errore prediction: {e}")
        reconstruction_errors = np.zeros(len(features_normalized))
    
    # Clustering (combina features + reconstruction error)
    cluster_input = np.column_stack([
        features_normalized,
        reconstruction_errors.reshape(-1, 1)
    ])
    
    kmeans = KMeans(n_clusters=CONFIG['num_clusters'], random_state=42, n_init=10)
    labels = kmeans.fit_predict(cluster_input)
    
    # Identifica cluster spindles (maggiore ampiezza media)
    cluster_means = []
    for cluster_id in range(CONFIG['num_clusters']):
        cluster_mask = (labels == cluster_id)
        cluster_features = segment_features[cluster_mask]
        
        if len(cluster_features) > 0:
            # Usa feature 0 = mean amplitude
            cluster_mean = np.mean(cluster_features[:, 0])
        else:
            cluster_mean = 0
        
        cluster_means.append(cluster_mean)
    
    spindle_cluster = np.argmax(cluster_means)
    
    print(f"   🎯 Cluster spindle: {spindle_cluster}")
    print(f"   📊 Ampiezza media cluster 0: {cluster_means[0]:.6f}")
    print(f"   📊 Ampiezza media cluster 1: {cluster_means[1]:.6f}")
    print(f"   📊 Segmenti spindle: {np.sum(labels == spindle_cluster)}/{len(labels)}")
    
    # Converti cluster in regioni temporali
    spindle_mask = (labels == spindle_cluster)
    step_samples = int(segment_length * (1 - CONFIG['overlap_ratio']))
    
    regions = detect_continuous_regions_from_mask(spindle_mask, step_samples, sfreq)
    
    print(f"   📍 Regioni iniziali: {len(regions)}")
    
    # Refinement con criteri Morlet
    # Calcola threshold RMS adattivo (30s window, 95° percentile)
    rms_threshold = compute_adaptive_threshold(
        envelope, 
        window_sec=CONFIG['rms_window_sec'],
        sfreq=sfreq,
        percentile=CONFIG['rms_percentile']
    )
    
    print(f"   🔬 Threshold RMS (95° percentile): {rms_threshold:.6f}")
    
    refined_regions = []
    
    for start_time, end_time in regions:
        duration = end_time - start_time
        
        if not (CONFIG['min_spindle_duration'] <= duration <= CONFIG['max_spindle_duration']):
            continue
        
        start_sample = int(start_time * sfreq)
        end_sample = int(end_time * sfreq)
        end_sample = min(end_sample, len(envelope))
        
        if start_sample >= len(envelope):
            continue
        
        region_envelope = envelope[start_sample:end_sample]
        
        if len(region_envelope) == 0:
            continue
        
        above_threshold_ratio = np.sum(region_envelope > rms_threshold) / len(region_envelope)
        
        if above_threshold_ratio < CONFIG['min_amplitude_ratio']:
            continue
        
        refined_regions.append((start_time, end_time, above_threshold_ratio))
    
    print(f"   🔬 Dopo refinement: {len(refined_regions)} regioni")
    
    # Merge spindles vicini
    regions_to_merge = [(s, e) for s, e, _ in refined_regions]
    merged_regions = merge_close_spindles(
        regions_to_merge,
        min_gap_sec=CONFIG['merge_gap_sec'],
        max_total_duration=CONFIG['max_spindle_duration']
    )
    
    print(f"   🔗 Dopo merge: {len(merged_regions)} spindles")
    
    # Crea risultati finali con metriche
    results = []
    for start_time, end_time in merged_regions:
        duration = end_time - start_time
        
        start_sample = int(start_time * sfreq)
        end_sample = int(end_time * sfreq)
        end_sample = min(end_sample, len(envelope))
        
        if start_sample >= len(envelope):
            continue
        
        region_envelope = envelope[start_sample:end_sample]
        
        if len(region_envelope) == 0:
            continue
        
        # Calcola metriche qualità
        peak_amplitude = np.max(region_envelope)
        mean_amplitude = np.mean(region_envelope)
        confidence = np.sum(region_envelope > rms_threshold) / len(region_envelope)
        
        results.append({
            'Canale': channel_name,
            'Start_Time(s)': round(start_time, 3),
            'End_Time(s)': round(end_time, 3),
            'Duration(s)': round(duration, 3),
            'Peak_Amplitude(µV)': round(peak_amplitude * 1e6, 3),
            'Mean_Amplitude(µV)': round(mean_amplitude * 1e6, 3),
            'RMS_Threshold(µV)': round(rms_threshold * 1e6, 3),
            'Confidence': round(confidence, 3)
        })
    
    print(f"✅ Rilevati {len(results)} spindles finali")
    
    return pd.DataFrame(results)


def main():
    """Funzione principale per clustering ibrido e rilevamento spindles"""
    
    # Configurazione percorsi
    path_edf = os.environ.get('DATA_PATH', 'Data/Preprocessed_Edf')
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
    
    os.makedirs(cluster_output_path, exist_ok=True)
    
    # Lista per raccogliere tutti i risultati
    all_spindle_results = []
    
    for file in filenames:
        if not file.endswith('.edf'):
            continue
            
        file_path = os.path.join(path_edf, file)
        print(f"\n📊 Processando: {file}")
        
        try:
            # Carica dati EEG
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            sfreq = raw.info['sfreq']
            
            # Filtra canali
            channels_to_include = [ch for ch in raw.ch_names if ch not in CONFIG['channels_to_exclude']]
            raw.pick_channels(channels_to_include)
            
            print(f"   📊 Canali: {len(channels_to_include)}, Frequenza: {sfreq} Hz")
            
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
                
                print(f"\n   🤖 Modello caricato: {channel_name}")
                print(f"      Input shape: {autoencoder.input_shape}")
                print(f"      Output shape: {autoencoder.output_shape}")
                
                # Estrai dati del canale
                channel_idx = raw.ch_names.index(channel_name)
                channel_data = raw.get_data()[channel_idx:channel_idx+1, :]
                
                channel_results = process_channel_for_spindles_hybrid(
                    channel_name, channel_data, sfreq, autoencoder
                )
                
                if len(channel_results) > 0:
                    all_spindle_results.append(channel_results)
                
                # Pulizia memoria
                del autoencoder
                tf.keras.backend.clear_session()
                gc.collect()
                
        except Exception as e:
            print(f"❌ Errore processando {file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combina e salva risultati
    if all_spindle_results:
        final_results = pd.concat(all_spindle_results, ignore_index=True)
        
        # Ordina per canale e tempo di inizio
        final_results = final_results.sort_values(['Canale', 'Start_Time(s)'])
        
        # Salva CSV
        csv_path = os.path.join(cluster_output_path, 'start_end_per_channel.csv')
        final_results.to_csv(csv_path, index=False)
        
        print("\n📊 Riepilogo per canale:")
        summary = final_results.groupby('Canale').agg({
            'Start_Time(s)': 'count',
            'Duration(s)': ['mean', 'std'],
            'Mean_Amplitude(µV)': 'mean',
            'Confidence': 'mean'
        }).round(3)
        summary.columns = ['N_Spindles', 'Mean_Duration(s)', 'Std_Duration(s)', 'Mean_Amplitude(µV)', 'Mean_Confidence']
        print(summary)
        
    else:
        print("\n❌ Nessun spindle rilevato")


if __name__ == "__main__":
    main()