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
    # Ottieni features dall'encoder specializzato
    spindle_features = encoder.predict(segments, verbose=0)
    spindle_features_flat = spindle_features.reshape(spindle_features.shape[0], -1)
    
    print(f"📊 Features spindle estratte: {spindle_features_flat.shape}")
    
    # Clustering binario: spindle vs non-spindle
    kmeans = KMeans(
        n_clusters=SPINDLE_CONFIG['num_clusters'], 
        random_state=42,
        n_init=20,  # Più inizializzazioni per stabilità
        max_iter=500
    )
    
    cluster_labels = kmeans.fit_predict(spindle_features_flat)
    
    # Calcola silhouette score per validare la qualità del clustering
    if len(set(cluster_labels)) > 1:
        sil_score = silhouette_score(spindle_features_flat, cluster_labels)
        print(f"📈 Silhouette Score: {sil_score:.3f}")
    
    # Determina quale cluster rappresenta gli spindles
    # Gli spindles tendono ad avere features più concentrate/anomale
    cluster_distances = []
    for cluster_id in range(SPINDLE_CONFIG['num_clusters']):
        cluster_mask = cluster_labels == cluster_id
        if np.sum(cluster_mask) > 0:
            cluster_center = kmeans.cluster_centers_[cluster_id]
            cluster_points = spindle_features_flat[cluster_mask]
            avg_distance = np.mean([
                np.linalg.norm(point - cluster_center) 
                for point in cluster_points
            ])
            cluster_distances.append(avg_distance)
        else:
            cluster_distances.append(float('inf'))
    
    # Il cluster con distanza minore (più compatto) è probabilmente quello degli spindles
    spindle_cluster_id = np.argmin(cluster_distances)
    
    print(f"🎯 Cluster spindle identificato: {spindle_cluster_id}")
    print(f"📊 Distribuzione cluster: {np.bincount(cluster_labels)}")
    
    # Crea labels binari: 1 = spindle, 0 = non-spindle
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
        
        # Estrai features e classifica usando SOLO ML
        features, spindle_labels, kmeans_model = extract_spindle_features_from_sigma(
            sigma_encoder, encoder_input
        )
        
        # Calcola confidence score basato sulla distanza dal centroide
        confidence_scores = []
        for i, label in enumerate(spindle_labels):
            if label == 1:  # Solo per i segmenti classificati come spindles
                centroid = kmeans_model.cluster_centers_[np.argmin([
                    np.linalg.norm(kmeans_model.cluster_centers_[j] - features[i]) 
                    for j in range(len(kmeans_model.cluster_centers_))
                ])]
                distance = np.linalg.norm(features[i] - centroid)
                # Converti distanza in confidence (più vicino = più confidence)
                max_distance = np.max([
                    np.linalg.norm(kmeans_model.cluster_centers_[j] - centroid) 
                    for j in range(len(kmeans_model.cluster_centers_))
                ])
                confidence = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
                confidence_scores.append(max(0.5, confidence))  # Min confidence 0.5
            else:
                confidence_scores.append(0.0)
        
        # Rileva regioni continue di spindles
        step_duration = SPINDLE_CONFIG['window_size'] * (1 - SPINDLE_CONFIG['overlap_ratio'])
        
        spindle_regions = []
        start_idx = None
        
        for i, is_spindle in enumerate(spindle_labels):
            if is_spindle == 1 and start_idx is None:
                start_idx = i
            elif is_spindle == 0 and start_idx is not None:
                # Fine regione spindle
                duration = (i - start_idx) * step_duration
                if (SPINDLE_CONFIG['min_spindle_duration'] <= duration <= 
                    SPINDLE_CONFIG['max_spindle_duration']):
                    
                    start_time = start_idx * step_duration
                    end_time = i * step_duration
                    avg_confidence = np.mean(confidence_scores[start_idx:i])
                    
                    spindle_regions.append({
                        'Canale': channel_name,
                        'Start_Time(s)': round(start_time, 3),
                        'End_Time(s)': round(end_time, 3),
                        'Confidence': round(avg_confidence, 3)
                    })
                
                start_idx = None
        
        # Gestisci ultimo segmento se necessario
        if start_idx is not None:
            duration = (len(spindle_labels) - start_idx) * step_duration
            if (SPINDLE_CONFIG['min_spindle_duration'] <= duration <= 
                SPINDLE_CONFIG['max_spindle_duration']):
                
                start_time = start_idx * step_duration
                end_time = len(spindle_labels) * step_duration
                avg_confidence = np.mean(confidence_scores[start_idx:])
                
                spindle_regions.append({
                    'Canale': channel_name,
                    'Start_Time(s)': round(start_time, 3),
                    'End_Time(s)': round(end_time, 3),
                    'Confidence': round(avg_confidence, 3)
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