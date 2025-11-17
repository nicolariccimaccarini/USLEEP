import mne 
import pandas as pd
import os
import sys
import gc
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model

# Aggiungi il percorso utils al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from signal_processing import (
    get_file_output_path, segment_signal_with_overlap,
    compute_morlet_wavelet, compute_morlet_features,
    compute_adaptive_threshold, merge_close_spindles
)

# Abilita deserializzazione
tf.keras.config.enable_unsafe_deserialization()

# Configurazione
CONFIG = {
    'window_size': 0.5,                         # Ampiezza finestra temporale (s)
    'overlap_ratio': 0.2,                       # Percentuale sovrapposizione segmenti
    'num_clusters': 2,                          # (spindles vs non-spindles)
    'min_spindle_duration': 0.5,                # Durata minima spindle
    'max_spindle_duration': 3.0,                # Durata massima spindle
    'wavelet_fc_range': [11, 12.5, 14, 15],     # Range frequenze centrali
    'wavelet_n_cycles': 7,                      # Numero cicli
    'threshold_multiplier': 4.5,                # 4.5x media
    'threshold_window_sec': 0.1,                # Finestra media mobile
    'merge_gap_sec': 1.0,                       # Gap minimo per merge
    'min_threshold_ratio': 0.7,                 # % campioni sopra threshold per validazione
    'channels_to_exclude': {'EEG A1', 'EEG A2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2'}
}


def load_autoencoder_with_fallback(model_path):
    """Carica il modello autoencoder con fallback"""
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"⚠️ Errore caricamento {model_path}: {e}")
        return None


def detect_continuous_regions(indices):
    """Converte indici sparsi in regioni continue"""
    if len(indices) == 0:
        return []
    
    regions = []
    start = indices[0]
    prev = indices[0]
    
    for idx in indices[1:]:
        if idx != prev + 1:
            regions.append((start, prev + 1))
            start = idx
        prev = idx
    
    regions.append((start, prev + 1))
    return regions


def process_channel_for_spindles_hybrid(channel_name, data, sfreq, encoder):
    """
    Approccio ibrido: ML feature learning + Morlet criteria
    
    Pipeline:
    1. Estrazione feature Morlet multi-scala
    2. Segmentazione con overlap
    3. Calcolo anomaly score (reconstruction error dell'autoencoder)
    4. Clustering binario (spindles vs non-spindles)
    5. Refinement con criteri Morlet (durata, threshold adattivo)
    6. Merge spindles vicini
    """
    print(f"🔍 Analisi spindles IBRIDA per: {channel_name}")
    
    channel_data_1d = data.flatten()
    
    # Calcola Morlet wavelet (multi-scala per robustezza)
    wavelet_features = []
    for fc in CONFIG['wavelet_fc_range']:
        wavelet_complex = compute_morlet_wavelet(channel_data_1d, sfreq, fc=fc, n_cycles=CONFIG['wavelet_n_cycles'])
        wavelet_amp = np.abs(wavelet_complex)
        wavelet_features.append(wavelet_amp)
    
    # Combina feature multi-scala (media)
    wavelet_combined = np.mean(wavelet_features, axis=0)
    
    print(f"   🌊 Morlet Wavelet applicata - fc range: {CONFIG['wavelet_fc_range']} Hz")
    
    segment_length = int(CONFIG['window_size'] * sfreq)
    
    # Segmenta il segnale wavelet
    wavelet_reshaped = wavelet_combined.reshape(1, -1)
    wavelet_segments_raw = segment_signal_with_overlap(
        wavelet_reshaped,
        segment_length,
        CONFIG['overlap_ratio']
    )
    
    # Segmenta anche il segnale originale per feature extraction
    data_segments = segment_signal_with_overlap(
        data,
        segment_length,
        CONFIG['overlap_ratio']
    )
    
    print(f"   📏 Segmenti generati: {len(wavelet_segments_raw)}")
    
    # Estrai feature Morlet per ogni segmento
    segment_features_list = []
    for seg in data_segments:
        morlet_feat = compute_morlet_features(
            seg,
            sfreq,
            fc_range=CONFIG['wavelet_fc_range'],
            n_cycles=CONFIG['wavelet_n_cycles']
        )
        segment_features_list.append(morlet_feat.flatten())  # flatten per singolo canale
    
    segment_features = np.array(segment_features_list)
    
    print(f"   📊 Feature Morlet estratte per segmento: {segment_features.shape}")
    
    # Normalizza features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(segment_features)
    
    # Reshape per encoder
    features_reshaped = features_normalized.reshape(-1, 1, features_normalized.shape[1])
    
    try:
        predictions = encoder.predict(features_reshaped, verbose=0)
        
        # Calcola reconstruction error (MSE tra input e output ricostruito)
        reconstruction_errors = np.mean((features_reshaped - predictions)**2, axis=(1, 2))        
    except Exception as e:
        print(f"   ⚠️ Errore prediction autoencoder: {e}")
    
    # Combina feature normalizzate + reconstruction error per clustering
    cluster_input = np.column_stack([
        features_normalized,
        reconstruction_errors.reshape(-1, 1)
    ])
    
    kmeans = KMeans(n_clusters=CONFIG['num_clusters'], random_state=42, n_init=10)
    labels = kmeans.fit_predict(cluster_input)
    
    # Identifica cluster spindles (quello con maggiore ampiezza wavelet media)
    cluster_wavelet_means = []
    for cluster_id in range(CONFIG['num_clusters']):
        cluster_mask = labels == cluster_id
        cluster_wavelet_segments = [wavelet_segments_raw[i] for i, m in enumerate(cluster_mask) if m]
        
        if len(cluster_wavelet_segments) > 0:
            cluster_mean_amp = np.mean([np.mean(seg) for seg in cluster_wavelet_segments])
        else:
            cluster_mean_amp = 0
        
        cluster_wavelet_means.append(cluster_mean_amp)
    
    spindle_cluster = np.argmax(cluster_wavelet_means)
    
    print(f"   🎯 Cluster identificato: {spindle_cluster} (ampiezza media: {cluster_wavelet_means[spindle_cluster]:.3f})")
    print(f"   📊 Segmenti cluster spindle: {np.sum(labels == spindle_cluster)}/{len(labels)}")
    
    # Prendi solo i segmenti classificati come spindles
    spindle_indices = np.where(labels == spindle_cluster)[0]
    
    # Converti in regioni temporali continue
    spindle_regions = detect_continuous_regions(spindle_indices)
    
    # Calcola tempi reali
    step_samples = int(segment_length * (1 - CONFIG['overlap_ratio']))
    spindle_regions_time = []
    
    for start_idx, end_idx in spindle_regions:
        start_time = start_idx * step_samples / sfreq
        end_time = end_idx * step_samples / sfreq
        duration = end_time - start_time
        
        # CRITERIO 1: Durata 0.5-3s
        if not (CONFIG['min_spindle_duration'] <= duration <= CONFIG['max_spindle_duration']):
            continue
        
        # CRITERIO 2: Verifica threshold adattivo sulla regione wavelet
        start_sample = int(start_time * sfreq)
        end_sample = int(end_time * sfreq)
        end_sample = min(end_sample, len(wavelet_combined))  # safety check
        
        if start_sample >= len(wavelet_combined):
            continue
        
        region_wavelet = wavelet_combined[start_sample:end_sample]
        
        if len(region_wavelet) == 0:
            continue
        
        # Calcola threshold adattivo per questa regione
        threshold = compute_adaptive_threshold(
            region_wavelet,
            window_sec=CONFIG['threshold_window_sec'],
            sfreq=sfreq,
            threshold_multiplier=CONFIG['threshold_multiplier']
        )
        
        # Verifica che almeno X% dei campioni superi threshold
        above_threshold_ratio = np.sum(region_wavelet > threshold) / len(region_wavelet)
        
        if above_threshold_ratio < CONFIG['min_threshold_ratio']:
            continue
        
        spindle_regions_time.append((start_time, end_time, above_threshold_ratio))
    
    print(f"   🔬 Dopo refinement Morlet: {len(spindle_regions_time)} regioni")
    
    # Rimuovi la confidenza temporaneamente per merge
    regions_for_merge = [(s, e) for s, e, _ in spindle_regions_time]
    
    merged_regions = merge_close_spindles(
        regions_for_merge,
        min_gap_sec=CONFIG['merge_gap_sec'],
        max_total_duration=CONFIG['max_spindle_duration']
    )
    
    print(f"   🔗 Dopo merge: {len(merged_regions)} spindles")
    
    # Creazione risultati
    results = []
    for start_time, end_time in merged_regions:
        duration = end_time - start_time
        
        # Calcola metriche qualità sulla regione finale
        start_sample = int(start_time * sfreq)
        end_sample = int(end_time * sfreq)
        end_sample = min(end_sample, len(wavelet_combined))
        
        if start_sample >= len(wavelet_combined):
            continue
        
        region_wavelet = wavelet_combined[start_sample:end_sample]
        
        if len(region_wavelet) == 0:
            continue
        
        # Ricalcola threshold e confidenza
        threshold = compute_adaptive_threshold(
            region_wavelet,
            window_sec=CONFIG['threshold_window_sec'],
            sfreq=sfreq,
            threshold_multiplier=CONFIG['threshold_multiplier']
        )
        
        confidence = np.sum(region_wavelet > threshold) / len(region_wavelet)
        
        results.append({
            'Canale': channel_name,
            'Start_Time(s)': round(start_time, 3),
            'End_Time(s)': round(end_time, 3),
            'Duration(s)': round(duration, 3),
            'Mean_Amplitude': round(np.mean(region_wavelet), 3),
            'Max_Amplitude': round(np.max(region_wavelet), 3),
            'ML_Confidence': round(confidence, 3)
        })
    
    print(f"✅ Rilevati {len(results)} spindles finali (IBRIDO)")
    
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
                
                print(f"   🤖 Modello caricato: {model_path}")
                print(f"   📊 Input shape: {autoencoder.input_shape}")
                print(f"   📊 Output shape: {autoencoder.output_shape}")
                
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
        csv_path = os.path.join(cluster_output_path, 'start_end_duration_hybrid.csv')
        final_results.to_csv(csv_path, index=False)
        
        print(f"\n🎉 Risultati salvati in: {csv_path}")
        print(f"📈 Totale spindles rilevati: {len(final_results)}")
        print(f"📊 Durata media: {final_results['Duration(s)'].mean():.3f}s")
        print(f"📊 Ampiezza media: {final_results['Mean_Amplitude'].mean():.3f}")
        print(f"📊 Confidenza media: {final_results['ML_Confidence'].mean():.3f}")
        
        print("\n📊 Riepilogo per canale:")
        summary = final_results.groupby('Canale').agg({
            'Start_Time(s)': 'count',
            'Duration(s)': ['mean', 'std'],
            'ML_Confidence': 'mean'
        }).round(3)
        summary.columns = ['Count', 'Mean_Duration(s)', 'Std_Duration(s)', 'Mean_Confidence']
        print(summary)
        
    else:
        print("❌ Nessun spindle rilevato")


if __name__ == "__main__":
    main()