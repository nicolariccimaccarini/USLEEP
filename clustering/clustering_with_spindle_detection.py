import mne 
import pandas as pd
import os
import sys
import gc
import numpy as np
import tensorflow as tf
import scipy.signal as signal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras.models import Model, load_model
from scipy.signal import morlet2

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
    'window_size': 0.5,                         # Ampiezza finestra temporale (s)
    'overlap_ratio': 0.2,                       # Percentuale di sovrapposizione segmenti
    'num_clusters': 3,                          # Numero cluster
    'min_spindle_duration': 0.5,                # Durata minima spindle
    'max_spindle_duration': 3.0,                # Durata massima spindle
    'wavelet_fc': 12.5,                         # Frequenza centrale Morlet
    'wavelet_n_cycles': 7,                      # Numero cicli
    'threshold_multiplier': 4.5,                # 4.5x media
    'threshold_window_sec': 0.1,                # Finestra media mobile
    'merge_gap_sec': 1.0,                       # Gap minimo per merge
    'channels_to_exclude': {'EEG A1', 'EEG A2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2'}
}


def load_autoencoder_with_fallback(model_path):
    """Carica il modello autoencoder con fallback"""
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"⚠️ Errore caricamento {model_path}: {e}")
        return None


def process_channel_for_spindles(channel_name, data, sfreq, encoder):
    """
    Processa un singolo canale per rilevare spindles usando Morlet Wavelet
    
    Args:
        channel_name: nome del canale
        data: dati del canale
        sfreq: frequenza di campionamento
        encoder: modello encoder
    
    Returns:
        DataFrame con i risultati dei spindles rilevati
    """
    print(f"🔍 Analisi spindles per canale: {channel_name}")
    
    # Applica Morlet Wavelet su segnale originale
    channel_data_1d = data.flatten()
    wavelet_signal = compute_morlet_wavelet(channel_data_1d, sfreq, fc=12.5, n_cycles=7)
    
    print(f"   🌊 Morlet Wavelet applicata (fc=12.5 Hz, n=7 cycles)")
    
    # Calcola threshold adattivo (4.5x media mobile con finestra 0.1s)
    threshold_signal = compute_adaptive_threshold(
        wavelet_signal, 
        window_sec=0.1, 
        sfreq=sfreq, 
        threshold_multiplier=4.5
    )
    
    # Rileva superamenti threshold
    above_threshold = wavelet_signal > threshold_signal
    
    # Converti in regioni temporali
    diff = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # Filtra per durata (0.5-3.0 secondi)
    spindle_regions_time = []
    for start_idx, end_idx in zip(starts, ends):
        start_time = start_idx / sfreq
        end_time = end_idx / sfreq
        duration = end_time - start_time
        
        if CONFIG['min_spindle_duration'] <= duration <= CONFIG['max_spindle_duration']:
            spindle_regions_time.append((start_time, end_time))
    
    # Merge spindles vicini (gap < 1s, durata totale < 3s)
    merged_regions = merge_close_spindles(
        spindle_regions_time, 
        min_gap_sec=1.0, 
        max_total_duration=3.0
    )
    
    # Crea DataFrame risultati
    results = []
    for start_time, end_time in merged_regions:
        duration = end_time - start_time
        results.append({
            'Canale': channel_name,
            'Start_Time(s)': round(start_time, 3),
            'End_Time(s)': round(end_time, 3),
            'Duration(s)': round(duration, 3)
        })
    
    print(f"✅ Rilevati {len(results)} spindles per {channel_name}")
    print(f"   📊 Wavelet amplitude - Media: {np.mean(wavelet_signal):.3f}, Max: {np.max(wavelet_signal):.3f}")
    
    return pd.DataFrame(results)


def compute_morlet_wavelet(data, sfreq, fc=12.5, n_cycles=7):
    """
    Applica la trasformata Morlet wavelet al segnale
    
    Args:
        data: segnale (1D array)
        sfreq: frequenza di campionamento
        fc: frequenza centrale (Hz) - per spindles tipicamente 11-15 Hz
        n_cycles: numero di cicli del wavelet (default 7)
    
    Returns:
        wavelet_signal: segnale trasformato
    """
    # Calcola parametri bandwidth
    s = n_cycles / (2 * np.pi * fc)
    fb = 2 * s**2
    
    # Crea il vettore tempo
    w = 2 * np.pi * fc
    
    # Lunghezza della wavelet in secondi
    wavelet_duration = n_cycles / fc
    wavelet_samples = int(wavelet_duration * sfreq * 2)
    
    # Crea morlet wavelet
    t = np.arange(-wavelet_samples/2, wavelet_samples/2) / sfreq
    morlet_wav = (np.pi * fb)**(-0.5) * np.exp(2j * np.pi * fc * t) * np.exp(-t**2 / fb)
    
    # Convoluzione
    wavelet_signal = np.convolve(data, morlet_wav, mode='same')
    
    return np.abs(wavelet_signal)


def compute_adaptive_threshold(signal, window_sec=0.1, sfreq=200, threshold_multiplier=4.5):
    """
    Calcola threshold adattivo con media mobile
    
    Args:
        signal: segnale wavelet (ampiezza)
        window_sec: finestra per media mobile (secondi)
        sfreq: frequenza di campionamento
        threshold_multiplier: moltiplicatore per la soglia (4.5x)
    
    Returns:
        threshold_signal: array con valori di soglia adattiva
    """
    window_samples = int(window_sec * sfreq)
    
    # Calcola media mobile
    from scipy.ndimage import uniform_filter1d
    moving_avg = uniform_filter1d(signal, size=window_samples, mode='nearest')
    
    # Threshold = 4.5 * media mobile
    threshold_signal = threshold_multiplier * moving_avg
    
    return threshold_signal


def merge_close_spindles(regions, min_gap_sec=1.0, max_total_duration=3.0, step_duration=0.4):
    """
    Unisce spindles vicini secondo criteri
    
    Args:
        regions: lista di tuple (start_time, end_time)
        min_gap_sec: distanza minima tra spindles (secondi)
        max_total_duration: durata massima dopo merge (secondi)
        step_duration: durata step per conversione indici
    
    Returns:
        merged_regions: lista di regioni unite
    """
    if len(regions) == 0:
        return regions
    
    # Ordina per tempo di inizio
    sorted_regions = sorted(regions, key=lambda x: x[0])
    merged = [sorted_regions[0]]
    
    for current_start, current_end in sorted_regions[1:]:
        last_start, last_end = merged[-1]
        
        # Calcola gap e durata totale se unite
        gap = current_start - last_end
        total_duration = current_end - last_start
        
        # Unisci se gap < 1s e durata totale < 3s
        if gap < min_gap_sec and total_duration <= max_total_duration:
            merged[-1] = (last_start, current_end)
        else:
            merged.append((current_start, current_end))
    
    return merged


def main():
    """Funzione principale per clustering e rilevamento spindles"""
    
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
        csv_path = os.path.join(cluster_output_path, 'start_end_duration.csv')
        final_results.to_csv(csv_path, index=False)
        
        print(f"\n🎉 Risultati salvati in: {csv_path}")
        print(f"📈 Totale spindles rilevati: {len(final_results)}")
        print(f"📊 Durata media spindles: {final_results['Duration(s)'].mean():.3f}s")
        print(f"📊 Durata mediana spindles: {final_results['Duration(s)'].median():.3f}s")
        print("\n📊 Riepilogo per canale:")
        summary = final_results.groupby('Canale').agg({
            'Start_Time(s)': 'count',
            'Duration(s)': ['mean', 'std']
        }).round(3)
        summary.columns = ['Count', 'Mean_Duration(s)', 'Std_Duration(s)']
        print(summary)
        
    else:
        print("❌ Nessun spindle rilevato")


if __name__ == "__main__":
    main()