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
    apply_smoothing, detect_spindle_regions, convert_regions_to_time,
    compute_spectrum_numpy, normalize_spectrum
)

# Abilita deserializzazione
tf.keras.config.enable_unsafe_deserialization()

# Configurazione
CONFIG = {
    'window_size': 0.5,
    'overlap_ratio': 0.2,
    'num_clusters': 2,  
    'min_spindle_duration': 0.5,
    'max_spindle_duration': 3.0,
    'smoothing_window_sec': 0.25,
    'min_amplitude_uv': 10,  # Ampiezza minima spindle in microvolts
    'max_amplitude_uv': 60,  # Ampiezza massima spindle in microvolts
    'channels_to_exclude': {'EEG A1', 'EEG A2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2'}
}


def load_autoencoder_with_fallback(model_path):
    """Carica il modello autoencoder con fallback"""
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"⚠️ Errore caricamento {model_path}: {e}")
        return None


def compute_segment_amplitudes(segments, sfreq):
    """
    Calcola l'ampiezza peak-to-peak per ogni segmento in microvolts
    
    Args:
        segments: array di segmenti (n_segments, n_samples)
        sfreq: frequenza di campionamento
    
    Returns:
        amplitudes: array di ampiezze in microvolts
    """
    amplitudes = []
    
    for segment in segments:
        # Rimuovi la dimensione extra se presente
        if segment.ndim > 1:
            segment = segment.flatten()
        
        # Calcola ampiezza peak-to-peak
        peak_to_peak = np.ptp(segment)
        
        # Converti in microvolts (assumendo che i dati siano già in Volts)
        # I dati MNE sono tipicamente in Volts, quindi moltiplichiamo per 1e6
        amplitude_uv = peak_to_peak * 1e6
        
        amplitudes.append(amplitude_uv)
    
    return np.array(amplitudes)


def compute_binary_clustering_with_amplitude(encoder, segments, segment_amplitudes):
    """
    Clustering binario identificando spindles tramite ampiezza
    
    Args:
        encoder: modello encoder
        segments: segmenti preprocessati (shape: n_samples, 1, n_features)
        segment_amplitudes: ampiezze dei segmenti in microvolts
    
    Returns:
        cluster_labels: etichette cluster binarie (0 o 1)
        spindle_cluster: ID del cluster identificato come "spindle"
        kmeans: oggetto KMeans fitted
        amplitude_stats: statistiche ampiezze per cluster
        spindle_segments_info: informazioni sui segmenti spindle rilevati
    """
    # Ottieni features codificate
    features = encoder.predict(segments, verbose=0)
    features_flat = features.reshape(features.shape[0], -1)
    
    # Clustering binario
    kmeans = KMeans(n_clusters=CONFIG['num_clusters'], random_state=42)
    cluster_labels = kmeans.fit_predict(features_flat)
    
    # Analizza ampiezze per ogni cluster
    amplitude_stats = {}
    spindle_scores = []
    
    for i in range(CONFIG['num_clusters']):
        cluster_mask = cluster_labels == i
        if cluster_mask.sum() > 0:
            cluster_amplitudes = segment_amplitudes[cluster_mask]
            
            # Conta quanti segmenti hanno ampiezza nel range spindle
            in_spindle_range = np.sum(
                (cluster_amplitudes >= CONFIG['min_amplitude_uv']) & 
                (cluster_amplitudes <= CONFIG['max_amplitude_uv'])
            )
            
            # Percentuale di segmenti nel range spindle
            spindle_percentage = (in_spindle_range / cluster_mask.sum()) * 100
            
            amplitude_stats[i] = {
                'mean': np.mean(cluster_amplitudes),
                'median': np.median(cluster_amplitudes),
                'std': np.std(cluster_amplitudes),
                'min': np.min(cluster_amplitudes),
                'max': np.max(cluster_amplitudes),
                'count': cluster_mask.sum(),
                'spindle_range_count': in_spindle_range,
                'spindle_percentage': spindle_percentage
            }
            
            spindle_scores.append(spindle_percentage)
        else:
            amplitude_stats[i] = None
            spindle_scores.append(0)
    
    # Il cluster spindle è quello con maggiore percentuale di segmenti nel range 10-60 µV
    spindle_cluster = np.argmax(spindle_scores)
    
    # **NUOVA PARTE: Raccolta info sui segmenti spindle**
    spindle_mask = cluster_labels == spindle_cluster
    spindle_segments_info = {
        'indices': np.where(spindle_mask)[0],  # Indici dei segmenti spindle
        'amplitudes': segment_amplitudes[spindle_mask],  # Ampiezze dei segmenti spindle
        'count': spindle_mask.sum()
    }
    
    print(f"   🎯 Cluster spindle identificato: {spindle_cluster}")
    print(f"   📊 Segmenti nel cluster spindle: {spindle_segments_info['count']}/{len(cluster_labels)} "
          f"({100*spindle_segments_info['count']/len(cluster_labels):.1f}%)")
    print(f"\n   📊 Statistiche ampiezze per cluster:")
    for i, stats in amplitude_stats.items():
        if stats:
            label = "SPINDLE" if i == spindle_cluster else "NON-SPINDLE"
            print(f"      Cluster {i} ({label}): Media={stats['mean']:.2f}µV, "
                  f"Mediana={stats['median']:.2f}µV, "
                  f"Range=[{stats['min']:.2f}, {stats['max']:.2f}]µV")
            print(f"                 {'':15} Segmenti totali: {stats['count']}")
            print(f"                 {'':15} Segmenti in range spindle (10-60µV): {stats['spindle_range_count']} "
                  f"({stats['spindle_percentage']:.1f}%)")
    
    # **STAMPA DETTAGLIO SEGMENTI SPINDLE RILEVATI**
    print(f"\n   🔍 Dettaglio primi 10 segmenti spindle rilevati:")
    for idx in spindle_segments_info['indices'][:10]:
        amp = segment_amplitudes[idx]
        in_range = "✓" if CONFIG['min_amplitude_uv'] <= amp <= CONFIG['max_amplitude_uv'] else "✗"
        print(f"      Segmento #{idx:4d}: Ampiezza={amp:6.2f}µV {in_range}")
    
    if len(spindle_segments_info['indices']) > 10:
        print(f"      ... e altri {len(spindle_segments_info['indices'])-10} segmenti")
    
    return cluster_labels, spindle_cluster, kmeans, amplitude_stats, spindle_segments_info


def process_channel_for_spindles(channel_name, data, sfreq, encoder):
    """
    Processa un singolo canale per rilevare spindles usando clustering binario con validazione ampiezza
    
    Args:
        channel_name: nome del canale
        data: dati del canale (in Volts)
        sfreq: frequenza di campionamento
        encoder: modello encoder
    
    Returns:
        DataFrame con i risultati dei spindles rilevati
    """
    print(f"🔍 Analisi spindles per canale: {channel_name}")
    print(f"📊 Frequenza di campionamento: {sfreq} Hz")
    
    # Prepara segmenti con overlap ottimizzato
    segment_length = int(CONFIG['window_size'] * sfreq)
    segments = segment_signal_with_overlap(
        data.reshape(1, -1),
        segment_length, 
        CONFIG['overlap_ratio']
    )
    
    if len(segments) == 0:
        print(f"⚠️ Nessun segmento generato per {channel_name}")
        return pd.DataFrame(columns=['Canale', 'Start_Time(s)', 'End_Time(s)'])
    
    print(f"📏 Segmenti generati: {len(segments)}")
    
    # Converti lista in array numpy
    segments = np.array(segments)
    
    # Rimuovi dimensioni extra per calcolo ampiezze
    segments_for_amplitude = segments.squeeze()
    if segments_for_amplitude.ndim == 1:
        segments_for_amplitude = segments_for_amplitude.reshape(1, -1)
    
    # Calcola ampiezze dei segmenti
    segment_amplitudes = compute_segment_amplitudes(segments_for_amplitude, sfreq)
    
    print(f"   📊 Ampiezze segmenti - Min: {segment_amplitudes.min():.2f}µV, "
          f"Max: {segment_amplitudes.max():.2f}µV, "
          f"Media: {segment_amplitudes.mean():.2f}µV")
    
    # Conta segmenti nel range spindle
    in_range = np.sum((segment_amplitudes >= CONFIG['min_amplitude_uv']) & 
                      (segment_amplitudes <= CONFIG['max_amplitude_uv']))
    print(f"   📊 Segmenti con ampiezza nel range spindle ({CONFIG['min_amplitude_uv']}-{CONFIG['max_amplitude_uv']}µV): "
          f"{in_range}/{len(segment_amplitudes)} ({100*in_range/len(segment_amplitudes):.1f}%)")
    
    try:
        # Calcola spettri
        spectrums, frequencies = compute_spectrum_numpy(segments, sfreq)
        
        # Normalizza spettri
        normalized_spectrums = [normalize_spectrum(spectrum) for spectrum in spectrums]
        
        channel_spectra = np.array([spec[0] for spec in normalized_spectrums])
        encoder_input = channel_spectra.reshape(-1, 1, channel_spectra.shape[1])
        
        print(f"   📊 Shape encoder input: {encoder_input.shape}")
        
        # Clustering binario con validazione ampiezza (aggiornato)
        cluster_labels, spindle_cluster, kmeans, amplitude_stats, spindle_segments_info = compute_binary_clustering_with_amplitude(
            encoder, encoder_input, segment_amplitudes
        )
        
        print(f"   🎯 K-Means clustering binario completato")
        print(f"   📊 Distribuzione cluster:")
        for i in range(CONFIG['num_clusters']):
            label = "SPINDLE" if i == spindle_cluster else "NON-SPINDLE"
            count = np.sum(cluster_labels == i)
            print(f"      Cluster {i} ({label}): {count} segmenti ({100*count/len(cluster_labels):.1f}%)")
        
        # Calcola silhouette score per valutare qualità clustering
        if len(np.unique(cluster_labels)) > 1:
            features = encoder.predict(encoder_input, verbose=0)
            features_flat = features.reshape(features.shape[0], -1)
            silhouette_avg = silhouette_score(features_flat, cluster_labels)
            print(f"   📈 Silhouette Score: {silhouette_avg:.3f}")
        
        # Crea segnale binario: 1 se appartiene al cluster spindle E ha ampiezza valida
        cluster_binary = (cluster_labels == spindle_cluster).astype(float)
        amplitude_binary = (
            (segment_amplitudes >= CONFIG['min_amplitude_uv']) & 
            (segment_amplitudes <= CONFIG['max_amplitude_uv'])
        ).astype(float)
        
        # Combinazione: deve essere sia nel cluster spindle che nel range di ampiezza
        binary_detection = cluster_binary * amplitude_binary
        
        # Smoothing temporale per ridurre falsi positivi
        smoothing_window_samples = int(CONFIG['smoothing_window_sec'] / CONFIG['window_size'] * (1 - CONFIG['overlap_ratio']))
        smoothing_window_samples = max(1, min(smoothing_window_samples, len(binary_detection)))
        
        smoothed_binary = apply_smoothing(
            binary_detection, 
            window_size=smoothing_window_samples, 
            method='moving_average'
        ) >= 0.5
        
        # Rileva regioni spindle
        step_duration = CONFIG['window_size'] * (1 - CONFIG['overlap_ratio'])
        min_duration_samples = int(CONFIG['min_spindle_duration'] / step_duration)
        max_duration_samples = int(CONFIG['max_spindle_duration'] / step_duration)
        
        spindle_regions = detect_spindle_regions(
            smoothed_binary.astype(float), 
            threshold=0.5,
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
        
        # Crea DataFrame risultati con ampiezze medie
        results = []
        for start_time, end_time in time_regions:
            duration = end_time - start_time
            if CONFIG['min_spindle_duration'] <= duration <= CONFIG['max_spindle_duration']:
                # Trova indici segmenti corrispondenti a questa regione
                start_seg = int(start_time / (CONFIG['window_size'] * (1 - CONFIG['overlap_ratio'])))
                end_seg = int(end_time / (CONFIG['window_size'] * (1 - CONFIG['overlap_ratio'])))
                
                # Calcola ampiezza media della regione
                region_amplitudes = segment_amplitudes[start_seg:end_seg+1]
                mean_amplitude = np.mean(region_amplitudes) if len(region_amplitudes) > 0 else 0
                
                results.append({
                    'Canale': channel_name,
                    'Start_Time(s)': round(start_time, 3),
                    'End_Time(s)': round(end_time, 3),
                    'Duration(s)': round(duration, 3),
                    'Mean_Amplitude(µV)': round(mean_amplitude, 2)
                })
        
        spindle_percentage = (binary_detection.sum() / len(binary_detection)) * 100
        print(f"✅ Rilevati {len(results)} spindles per {channel_name}")
        print(f"   📊 Percentuale segmenti classificati come spindle: {spindle_percentage:.2f}%")
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"❌ Errore durante il processing di {channel_name}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['Canale', 'Start_Time(s)', 'End_Time(s)', 'Duration(s)', 'Mean_Amplitude(µV)'])


def main():
    """Funzione principale per clustering binario e rilevamento spindles"""
    
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
    cluster_output_path = os.path.join(dirData, "cluster_binary")
    
    os.makedirs(cluster_output_path, exist_ok=True)
    
    print(f"\n🔬 Modalità: Clustering Binario con Validazione Ampiezza")
    print(f"📊 Numero cluster: {CONFIG['num_clusters']}")
    print(f"⏱️ Durata spindle: {CONFIG['min_spindle_duration']}-{CONFIG['max_spindle_duration']}s")
    print(f"📏 Range ampiezza spindle: {CONFIG['min_amplitude_uv']}-{CONFIG['max_amplitude_uv']}µV")
    
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
                
                # Crea encoder
                encoder = Model(
                    inputs=autoencoder.input, 
                    outputs=autoencoder.get_layer('encoder_dense').output
                )
                
                # Estrai dati del canale
                channel_idx = raw.ch_names.index(channel_name)
                channel_data = raw.get_data()[channel_idx:channel_idx+1, :]
                
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
            import traceback
            traceback.print_exc()
            continue
    
    # Combina e salva risultati
    if all_spindle_results:
        final_results = pd.concat(all_spindle_results, ignore_index=True)
        
        # Ordina per canale e tempo di inizio
        final_results = final_results.sort_values(['Canale', 'Start_Time(s)'])
        
        # Salva CSV
        csv_path = os.path.join(cluster_output_path, 'spindles_binary_clustering_amplitude.csv')
        final_results.to_csv(csv_path, index=False)
        
        print(f"\n🎉 Risultati salvati in: {csv_path}")
        print(f"📈 Totale spindles rilevati: {len(final_results)}")
        print(f"📊 Durata media spindles: {final_results['Duration(s)'].mean():.3f}s")
        print(f"📊 Durata mediana spindles: {final_results['Duration(s)'].median():.3f}s")
        print(f"📊 Ampiezza media spindles: {final_results['Mean_Amplitude(µV)'].mean():.2f}µV")
        print(f"📊 Ampiezza mediana spindles: {final_results['Mean_Amplitude(µV)'].median():.2f}µV")
        print("\n📊 Riepilogo per canale:")
        summary = final_results.groupby('Canale').agg({
            'Start_Time(s)': 'count',
            'Duration(s)': ['mean', 'std'],
            'Mean_Amplitude(µV)': ['mean', 'std']
        }).round(3)
        summary.columns = ['Count', 'Mean_Duration(s)', 'Std_Duration(s)', 
                          'Mean_Amplitude(µV)', 'Std_Amplitude(µV)']
        print(summary)
        
    else:
        print("❌ Nessun spindle rilevato")


if __name__ == "__main__":
    main()