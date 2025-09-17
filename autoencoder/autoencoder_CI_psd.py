import mne 
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.models import load_model
import gc



### funzioni

def checkMaxMin(data_normalized):
    # Verifica minimo e massimo per ciascun canale
    for i in range(data_normalized.shape[0]):
        min_val = np.min(data_normalized[i, :])  # Minimo del canale i
        max_val = np.max(data_normalized[i, :])  # Massimo del canale i
        print(f"Canale {i}: Min = {min_val:.6f}, Max = {max_val:.6f}")


def checkMediaDeviazione(data_normalized):
    # Verifica media e deviazione standard per ciascun canale
    for i in range(data_normalized.shape[0]):
        mean = np.mean(data_normalized[i, :])  # Media del canale i
        std = np.std(data_normalized[i, :])    # Deviazione standard del canale i
        print(f"Canale {i}: Media = {mean:.6f}, Deviazione Standard = {std:.6f}")      

def checkDistribuzione(data):
    # Supponiamo che 'data' sia il segnale di un canale specifico
    channel_data = data[0, :]  # Selezioniamo il primo canale, ad esempio

    # Creiamo l'istogramma
    plt.hist(channel_data, bins=50, density=True, alpha=0.6, color='g')
    plt.title('Istogramma del segnale EEG (primo canale)')
    plt.xlabel('Valori del segnale')
    plt.ylabel('Densità')
    plt.show()


def checkLunghezzaSegmenti(segments):
    
    for i, seg in enumerate(segments):
        print(f"Segmento {i} forma: {seg.shape}")

#inutile 
def pad_or_trim(segment, window_size):
    # Funzione per ritagliare o riempire i segmenti alla lunghezza desiderata (window_size)
    if segment.shape[1] > window_size:  # Se il segmento è più lungo
        return segment[:, :window_size]  # Ritaglia
    elif segment.shape[1] < window_size:  # Se il segmento è più corto
        # Riempie con zeri fino a window_size
        return np.pad(segment, ((0, 0), (0, window_size - segment.shape[1])), mode='constant')
    else:
        return segment  # Se ha già la dimensione corretta     

# Funzione per applicare il padding all'ultimo segmento se necessario
def pad_last_segment(segment, window_size):
    if segment.shape[1] < window_size:
        # Applica padding con zeri fino a raggiungere window_size
        return np.pad(segment, ((0, 0), (0, window_size - segment.shape[1])), mode='constant')
    return segment    


def segment_signal(data,segment_length):
    # Lista per contenere i segmenti
    segments = []

    for start in range(0, data.shape[1], segment_length):

        segment = data[:, start:start + segment_length]  # Estrai un segmento
        # Se questo è l'ultimo segmento e non ha la dimensione corretta, applica padding
        if start + segment_length > data.shape[1]:
            segment = pad_last_segment(segment, segment_length)

        # Controlla la lunghezza del segmento
        if segment.shape[1] != segment_length:
            print(f"Warning: Segment of incorrect length {segment.shape[1]} instead of {segment_length}")
            

        segments.append(segment)
    
    # checkLunghezzaSegmenti(segments)

    return segments

def segment_signal_with_overlap(data, segment_length, step):
    segments = []
    num_samples = data.shape[1]

    for start in range(0, data.shape[1] - segment_length, step):
        segment = data[:, start:start + segment_length]  # Estrai la finestra con sovrapposizione

        if start + segment_length > data.shape[1]:
            last_segment_start = num_samples - segment_length
            last_segment = data[:, last_segment_start:]

            padding_size = segment_length - last_segment.shape[1]
            last_segment_padded = np.pad(last_segment, ((0, 0), (0, padding_size)), mode='constant')
            segments.append(last_segment_padded)
        
        segments.append(segment)
    return segments


def check_overlap(segments, segment_length, overlap, sfreq):
    step = int(segment_length * (1 - overlap))
    num_segments = segments.shape[0]
    overlap_length = segment_length - step  # Numero di campioni sovrapposti teoricamente
    
    for i in range(num_segments - 1):
        # Confronta la fine del segmento i con l'inizio del segmento i+1
        segment_end = segments[i][:, -overlap_length:]  # Fine del segmento corrente
        next_segment_start = segments[i+1][:, :overlap_length]  # Inizio del segmento successivo
        
        # Confronto dei segmenti sovrapposti
        if not np.array_equal(segment_end, next_segment_start):
            # print(f"Segmenti {i} e {i+1} hanno la corretta sovrapposizione.")
            print(f"ATTENZIONE: Segmenti {i} e {i+1} NON hanno la corretta sovrapposizione!")
           

def compute_spectrum_numpy(segments, freq_sample):
    spectrums = []
    
    # Frequenze per l'asse delle X, calcolate una sola volta
    segment_length = segments[0].shape[1]
    frequencies = np.fft.fftfreq(segment_length, d=1/freq_sample)
    frequencies = frequencies[:segment_length // 2]

    # print(len(frequencies))
    
    # Loop per calcolare lo spettro di ogni segmento
    for segment in segments:
        # Applica la trasformata di Fourier a ogni canale separatamente
        segment_spectrum = np.abs(np.fft.fft(segment, axis=1))  # Calcola il modulo della trasformata
        segment_spectrum = np.abs(segment_spectrum[:, :segment_length // 2])
        # print(len(segment_spectrum))
        spectrums.append(segment_spectrum)
    
    return spectrums, frequencies   


def normalize_spectrum(spectrum):
    scaler = MinMaxScaler()
    normalized_spectrum = []
    for channel_spectrum in spectrum:
        channel_spectrum = channel_spectrum.reshape(-1, 1)
        normalized_channel = scaler.fit_transform(channel_spectrum).flatten()
        normalized_spectrum.append(normalized_channel)
    return np.array(normalized_spectrum)

##### script

dirData = "Data/" #dirEdf = "Data/Edf"
overlap = 0.10   #percentuale di sovrapposzione
window_size = 5 # Lunghezza della finestra in secondi
epoche = 200
batch_size = 16
pazienza = 20
c = 1

channels_to_exclude = {'EEG A1', 'EEG A2', 'Oculo', 'MK', 'ECG', 'EMG1', 'EMG2'}

scaler = MinMaxScaler()

# for dirpath, dirnames, filenames in os.walk(dirData):
#     print(f"Directory: {dirpath}")

script_dir = os.path.dirname(os.path.abspath(__file__))
dirData = os.path.join(script_dir, 'Data')

# dirData = os.path.abspath('Data') #<-- corretto il percorso 

# Path relativo alla cartella 'edf'
path_edf = os.path.join(dirData, "Edf")

# print(f"percorso cartella edf {path_edf}")

images_path = os.path.join(dirData, "images", "canali_individuali")
weights_path = os.path.join(dirData, "model", "canali_individuali")
cluster_path = os.path.join(dirData, "cluster")
grafico_cluster_path = os.path.join(images_path, 'grafici_clustering')
grafico_apprendimento_path = os.path.join(images_path, 'grafici_apprendimento')

if not os.path.exists(images_path):
    os.makedirs(images_path)
    
if not os.path.exists(weights_path):
    os.makedirs(weights_path)

if not os.path.exists(cluster_path):
    os.makedirs(cluster_path)  

if not os.path.exists(grafico_cluster_path):
    os.makedirs(grafico_cluster_path)  

if not os.path.exists(grafico_apprendimento_path):
    os.makedirs(grafico_apprendimento_path)          


# path_edf = os.path.join(dirData, "Temp")
# selected_channels = ['EEG Fp1', 'EEG Fp2']

filenames = [f for f in os.listdir(path_edf) if "edf" in f]

aggregated_data = {}

for file in filenames:
    if file.endswith('.edf'):  # Controlla se il file ha estensione .edf
        file_path = os.path.join(path_edf, file)
        print(f"\tFile: {file_path}")
        #raw = mne.io.read_raw_edf(file_path, preload=True)

        raw = mne.io.read_raw_edf(file_path, preload=True)
        data, times = raw[:]
        sfreq = raw.info['sfreq']  # Frequenza di campionamento 

        psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq)

        channels_to_include = [ch for ch in raw.ch_names if ch not in channels_to_exclude]
        raw.pick_channels(channels_to_include)

        # ### per testing 
        # raw.pick_channels(selected_channels)
        # data = raw.get_data()
        # ###

        ##segmentazione
        segment_length = int(window_size * sfreq) 
        segments = segment_signal(raw.get_data(), segment_length)

        ##calcolo dello spettro
        spectrums, frequencies = compute_spectrum_numpy(segments, sfreq)
        normalized_spectrums = [normalize_spectrum(spectrum) for spectrum in spectrums]


        for idx, channel in enumerate(raw.ch_names):
            if channel not in aggregated_data:
                aggregated_data[channel] = []
            for norm_spectrum in normalized_spectrums:
                aggregated_data[channel].append(norm_spectrum[idx])


for channel, data in aggregated_data.items():
        print(f"Dati aggregati per il canale '{channel}' (numero di spettri {len(aggregated_data[channel])}, shape di ogni spettro {aggregated_data[channel][0].shape}):")

print("\n")

strategy = tf.distribute.MirroredStrategy()

for channel, data in aggregated_data.items():
    print(f"\nAddestramento dell'autoencoder per il canale: {channel}")
    
    print(f"addestramento {c} di 19")
    c+=1
    # Normalizza i dati del canale
    # data = scaler.fit_transform(data)

    data = np.array(data)

    # print(data.shape)
    
    n_frequencies = data.shape[1]  # Presume che data sia di forma (numero_segmenti, numero_frequenze)
    # print(n_frequencies)

    # Ridimensiona i dati per adattarli all'input dell'autoencoder (numero_segmenti, n_channels, n_frequencies)
    all_segments_standardized = data.reshape((-1, 1, n_frequencies))
    
    print(all_segments_standardized.shape) #segmenti,canali,spettro


    channel_weights_path = os.path.join(weights_path, f"autoencoder_{channel}.h5")
    channel_images_path = os.path.join(grafico_apprendimento_path, f"grafico_apprendimento_{channel}.png")

    early_stopping = EarlyStopping(monitor='val_loss', patience=pazienza, verbose=1, restore_best_weights=True) 

    # Configura il modello per ogni canale
    with strategy.scope():
        input_layer = Input(shape=(1, n_frequencies))

        # Encoder LSTM
        encoded = LSTM(128, activation='relu', return_sequences=True, name='encoder_lstm_1')(input_layer)
        encoded = LSTM(64, activation='relu', return_sequences=True, name='encoder_lstm_2')(encoded)
        encoded = Dropout(0.2, name='encoder_dropout')(encoded)
        encoded = LSTM(32, activation='relu', return_sequences=False, name='encoder_lstm_3')(encoded)
        encoded = Dense(32, activation='relu', name='encoder_dense')(encoded)  # Riduzione ulteriore della dimensione

        # Decodifica e ricostruzione
        decoded = RepeatVector(1, name='repeat_vector')(encoded)
        decoded = LSTM(64, activation='relu', return_sequences=True, name='decoded_lstm_1')(decoded)
        decoded = LSTM(128, activation='relu', return_sequences=True, name='decoded_lstm_2')(decoded)
        decoded = TimeDistributed(Dense(n_frequencies), name='time_distributed_output')(decoded)

        # Crea il modello autoencoder
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss=MeanSquaredError())
    
    # Mostra il sommario del modello
    autoencoder.summary()
    
    # Addestramento del modello
    history = autoencoder.fit(all_segments_standardized, all_segments_standardized, 
                              epochs=epoche, 
                              batch_size=batch_size, 
                              validation_split=0.2, 
                              callbacks=[early_stopping])
    
    # Salva il grafico dell'apprendimento
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], 'r', marker='.', label="Train Loss")
    ax.plot(history.history["val_loss"], 'r--', marker='.', label="Val Loss")
    ax.set_title(f"Loss per il canale {channel}")
    ax.legend()
    plt.savefig(channel_images_path)
    plt.close()
    
    # Salvataggio del modello
    autoencoder.save(channel_weights_path)  
    autoencoder = None
    gc.collect()


#######clustering per trovare il valore di k ideale

k_range = range(1, 11)

for channel, data in aggregated_data.items():
    print(f"\nEsecuzione del clustering per il canale: {channel}")
    
    # Carica il modello del canale corrente
    model_path = os.path.join(weights_path, f"autoencoder_{channel}.h5")
    autoencoder = load_model(model_path)

    print(f"caricato il modello {model_path}")
    
    # Crea il modello encoder dal autoencoder
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder_dense').output)
    
    data = np.array(data) 

    n_frequencies = data.shape[1] 

    all_segments_standardized = data.reshape((-1, 1, n_frequencies))

    eeg_features = encoder.predict(all_segments_standardized)
    eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)  # Ridimensiona in 2D
    
    # Inizializzazione delle liste per il WCSS e i punteggi di silhouette
    wcss = []
    silhouette_scores = []

    # Clustering per ogni valore di k in k_range
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(eeg_features)
        cluster_assignments = kmeans.labels_
        wcss.append(kmeans.inertia_)  # Calcolo del WCSS
        
        # Calcola il silhouette score se il numero di cluster è valido
        if len(set(kmeans.labels_)) > 1:
            sil_score = silhouette_score(eeg_features, cluster_assignments)
        else:
            sil_score = -1  # Silhouette score non valido per k=1
        silhouette_scores.append(sil_score)

    # Determina il k ottimale in base al silhouette score
    max_silhouette = max(silhouette_scores)
    max_index = silhouette_scores.index(max_silhouette)
    max_k = k_range[max_index]

    # Creazione del grafico per il metodo del gomito e silhouette score
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharex=True)

    # Grafico WCSS (gomito)
    ax1.plot(k_range, wcss, marker='o', color='blue', label='WCSS')
    ax1.set_title(f'Metodo del Gomito e Silhouette Score per il Canale {channel}')
    ax1.set_xlabel('Numero di cluster (k)')
    ax1.set_ylabel('WCSS', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid()

    # Grafico Silhouette Score
    ax2.plot(k_range, silhouette_scores, marker='o', color='orange')
    ax2.set_title('Silhouette Score per Numero di Cluster')
    ax2.set_xlabel('Numero di cluster (k)')
    ax2.set_ylabel('Silhouette Score', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.grid()
    ax2.plot(max_k, max_silhouette, marker='o', color='red', markersize=10, label=f'Massimo Silhouette ({max_silhouette})')
    ax2.legend()

    plt.tight_layout()

    # Percorso di salvataggio per il grafico del clustering per il canale corrente
    grafico_cluster_path_channel = os.path.join(grafico_cluster_path, f'grafico_cluster_{channel}.png')
    plt.savefig(grafico_cluster_path_channel, dpi=300, bbox_inches='tight')
    plt.close()

    del autoencoder
    del encoder
    gc.collect()

