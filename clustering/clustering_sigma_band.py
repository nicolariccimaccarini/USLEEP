import mne 
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras.models import Model, load_model
import sys
import gc

# Aggiungi il percorso utils al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from signal_processing import (
    get_file_output_path, segment_signal_with_overlap,
    apply_sigma_band_filter, compute_sigma_power_spectrum, normalize_spectrum
)

# Abilita deserializzazione
tf.keras.config.enable_unsafe_deserialization()

# Configurazione per spindle detection
SPINDLE_CONFIG = {
    'window_size': 0.5,                 # Finestra 0.5s (durata minima spindle)
    'overlap_ratio': 0.8,               # Step 0.1s per risoluzione temporale
    'num_clusters': 2,                  # Binario: spindle vs non-spindle
    'sigma_low': 9,                     # Banda sigma
    'sigma_high': 15,
    'min_spindle_duration': 0.5,        # Durata minima spindle
    'max_spindle_duration': 3.0,        # Durata massima spindle
    'channels_to_exclude': {'EEG A1', 'EEG A2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2'}
}

def extract_spindle_features_from_sigma(encoder, segments):
    """
    Estrae features specifiche per spindles dalla banda sigma usando l'encoder
    
    Args:
        encoder: modello encoder addestrato sulla banda sigma
        segments: segmenti preprocessati nella banda sigma
    
    Returns:
        features estratte, labels del clustering
    """
    # Ottieni features dall'encoder
    spindle_features = encoder.predict(segments, verbose=0)
    spindle_features_flat = spindle_features.reshape(spindle_features.shape[0], -1)
    
    print(f"📊 Features estratte: {spindle_features_flat.shape}")
    
    # 🔍 DEBUG: Analizza variabilità features
    feature_std = np.std(spindle_features_flat, axis=0)
    feature_mean = np.mean(spindle_features_flat, axis=0)
    
    print(f"🔍 DEBUG Features:")
    print(f"   Media features: {feature_mean}")
    print(f"   Std features: {feature_std}")
    print(f"   Range features: {np.ptp(spindle_features_flat, axis=0)}")
    print(f"   Features costanti: {np.sum(feature_std < 1e-6)}/{len(feature_std)}")
    
    # Rimuovi features costanti
    valid_features_mask = feature_std > 1e-6
    if np.sum(valid_features_mask) < 2:
        print("❌ ERRORE: Troppo poche features variabili per clustering!")
        return spindle_features_flat, np.zeros(len(spindle_features_flat)), None
    
    valid_features = spindle_features_flat[:, valid_features_mask]
    print(f"✅ Features valide per clustering: {valid_features.shape[1]}/{spindle_features_flat.shape[1]}")
    
    # Prova clustering con diverse strategie
    try:
        # Strategia 1: K-means standard
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=20, max_iter=500)
        cluster_labels = kmeans.fit_predict(valid_features)
        
        unique_clusters = len(np.unique(cluster_labels))
        print(f"🎯 Cluster trovati: {unique_clusters}")
        
        if unique_clusters < 2:
            print("⚠️ K-means fallito, provo clustering basato su percentile...")
            # Strategia fallback: usa varianza delle features
            feature_variance = np.var(valid_features, axis=1)
            threshold = np.percentile(feature_variance, 75)  # Top 25% come spindles
            cluster_labels = (feature_variance > threshold).astype(int)
            
        print(f"📊 Distribuzione finale: {np.bincount(cluster_labels)}")
        
    except Exception as e:
        print(f"❌ Errore clustering: {e}")
        return spindle_features_flat, np.zeros(len(spindle_features_flat)), None
    
    # Identifica cluster spindles (quello meno numeroso se bilanciato)
    cluster_counts = np.bincount(cluster_labels)
    if len(cluster_counts) == 2 and min(cluster_counts) > 0:
        # Scegli il cluster meno frequente come spindles
        spindle_cluster_id = np.argmin(cluster_counts)
    else:
        spindle_cluster_id = 0
    
    print(f"🎯 Cluster spindle: {spindle_cluster_id}")
    spindle_binary_labels = (cluster_labels == spindle_cluster_id).astype(int)
    
    return spindle_features_flat, spindle_binary_labels, kmeans

def process_channel_for_spindle_detection(channel_name, data, sfreq, sigma_encoder):
    """
    Processa un canale per rilevare spindles usando solo features ML
    
    Returns:
        DataFrame con i risultati degli spindles rilevati
    """
    print(f"🔍 Rilevamento spindles per canale: {channel_name}")
    
    # Applica filtro banda sigma
    sigma_filtered_data = apply_sigma_band_filter(
        data, sfreq, 
        SPINDLE_CONFIG['sigma_low'], 
        SPINDLE_CONFIG['sigma_high']
    )
    
    # Segmentazione con alta risoluzione
    segment_length = int(SPINDLE_CONFIG['window_size'] * sfreq)
    segments = segment_signal_with_overlap(
        sigma_filtered_data, 
        segment_length, 
        SPINDLE_CONFIG['overlap_ratio']
    )
    
    if len(segments) == 0:
        print(f"⚠️ Nessun segmento per {channel_name}")
        return pd.DataFrame(columns=['Canale', 'Start_Time(s)', 'End_Time(s)', 'Confidence'])
    
    print(f"📏 Segmenti: {len(segments)}")
    
    try:
        # Calcola potenza sigma
        sigma_powers, _ = compute_sigma_power_spectrum(segments, sfreq)
        normalized_powers = [normalize_spectrum(power) for power in sigma_powers]
        
        # Prepara input per l'encoder
        channel_powers = np.array([power[0] for power in normalized_powers])
        encoder_input = channel_powers.reshape(-1, 1, channel_powers.shape[1])
        
        # Estrai features e classifica
        features, spindle_labels, kmeans_model = extract_spindle_features_from_sigma(
            sigma_encoder, encoder_input
        )
        
        if kmeans_model is None:
            print(f"❌ Clustering fallito per {channel_name}")
            return pd.DataFrame(columns=['Canale', 'Start_Time(s)', 'End_Time(s)', 'Confidence'])
        
        # 🔧 DEBUG: Verifica labels
        spindle_count = np.sum(spindle_labels)
        print(f"🔍 DEBUG: {spindle_count} segmenti classificati come spindles su {len(spindle_labels)}")
        
        if spindle_count == 0:
            print(f"⚠️ Nessun segmento spindle trovato per {channel_name}")
            return pd.DataFrame(columns=['Canale', 'Start_Time(s)', 'End_Time(s)', 'Confidence'])
        
        # Rileva regioni continue
        step_duration = SPINDLE_CONFIG['window_size'] * (1 - SPINDLE_CONFIG['overlap_ratio'])
        spindle_regions = []
        
        # Trova transizioni
        spindle_diff = np.diff(np.concatenate(([0], spindle_labels, [0])))
        starts = np.where(spindle_diff == 1)[0]
        ends = np.where(spindle_diff == -1)[0]
        
        for start_idx, end_idx in zip(starts, ends):
            duration = (end_idx - start_idx) * step_duration
            
            if (SPINDLE_CONFIG['min_spindle_duration'] <= duration <= 
                SPINDLE_CONFIG['max_spindle_duration']):
                
                start_time = start_idx * step_duration
                end_time = end_idx * step_duration
                
                spindle_regions.append({
                    'Canale': channel_name,
                    'Start_Time(s)': round(start_time, 3),
                    'End_Time(s)': round(end_time, 3),
                    'Confidence': 0.8  # Placeholder
                })
        
        print(f"✅ Rilevati {len(spindle_regions)} spindles per {channel_name}")
        
        # Debug info
        spindle_count = np.sum(spindle_labels)
        spindle_percentage = spindle_count / len(spindle_labels) * 100
        print(f"   📊 Segmenti spindle: {spindle_count}/{len(spindle_labels)} ({spindle_percentage:.1f}%)")
        
        return pd.DataFrame(spindle_regions)
        
    except Exception as e:
        print(f"❌ Errore processando {channel_name}: {e}")
        return pd.DataFrame(columns=['Canale', 'Start_Time(s)', 'End_Time(s)', 'Confidence'])

def main():
    """Funzione principale per spindle detection ML-based"""
    
    # Configurazione percorsi
    path_edf = os.environ.get('DATA_PATH', 'Data/Edf')
    output_path = os.environ.get('OUTPUT_PATH', 'Data/Output')
    current_file = os.environ.get('CURRENT_FILE', None)
    
    # Determina file da processare
    if current_file:
        dirData = get_file_output_path(output_path, current_file)
        filenames = [current_file]
    else:
        dirData = output_path
        filenames = [f for f in os.listdir(path_edf) if f.endswith('.edf')]
    
    # Percorsi
    sigma_weights_path = os.path.join(dirData, "model", "sigma_band")
    cluster_output_path = os.path.join(dirData, "cluster")
    images_path = os.path.join(dirData, "images", "spindle_detection")
    
    os.makedirs(cluster_output_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    
    # Risultati
    all_spindle_results = []
    
    for file in filenames:
        if not file.endswith('.edf'):
            continue
            
        file_path = os.path.join(path_edf, file)
        print(f"\n🧠 Processando per spindles: {file}")
        
        try:
            # Carica EEG
            raw = mne.io.read_raw_edf(file_path, preload=True)
            sfreq = raw.info['sfreq']
            
            # Filtra canali
            channels_to_include = [ch for ch in raw.ch_names 
                                 if ch not in SPINDLE_CONFIG['channels_to_exclude']]
            raw.pick_channels(channels_to_include)
            
            # Processa ogni canale
            for channel_name in raw.ch_names:
                # Carica modello sigma per il canale
                model_path = os.path.join(sigma_weights_path, f"sigma_autoencoder_{channel_name}.h5")
                
                if not os.path.exists(model_path):
                    print(f"⚠️ Modello sigma non trovato per {channel_name}")
                    continue
                
                sigma_autoencoder = load_model(model_path)
                
                # Crea encoder sigma
                sigma_encoder = Model(
                    inputs=sigma_autoencoder.input, 
                    outputs=sigma_autoencoder.get_layer('spindle_features').output
                )
                
                # Estrai dati del canale
                channel_idx = raw.ch_names.index(channel_name)
                channel_data = raw.get_data()[channel_idx:channel_idx+1, :]
                
                # Rileva spindles
                channel_results = process_channel_for_spindle_detection(
                    channel_name, channel_data, sfreq, sigma_encoder
                )
                
                if len(channel_results) > 0:
                    all_spindle_results.append(channel_results)
                
                # Pulizia memoria
                del sigma_autoencoder, sigma_encoder
                gc.collect()
                
        except Exception as e:
            print(f"❌ Errore processando {file}: {e}")
            continue
    
    # Salva risultati
    if all_spindle_results:
        final_results = pd.concat(all_spindle_results, ignore_index=True)
        final_results = final_results.sort_values(['Canale', 'Start_Time(s)'])
        
        csv_path = os.path.join(cluster_output_path, 'spindles_detected_ml.csv')
        final_results.to_csv(csv_path, index=False)
        
        print(f"\n🎉 Spindles rilevati e salvati in: {csv_path}")
        print(f"📈 Totale spindles: {len(final_results)}")
        
        # Statistiche per canale
        print("\n📊 Riepilogo spindles per canale:")
        summary = final_results.groupby('Canale').agg({
            'Start_Time(s)': 'count',
            'Confidence': 'mean'
        }).round(3)
        summary.columns = ['Count', 'Avg_Confidence']
        print(summary)
        
    else:
        print("❌ Nessun spindle rilevato")

if __name__ == "__main__":
    main()