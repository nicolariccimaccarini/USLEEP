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

##### script

#file_path = "Data/Sartini_Daisy/SARTINI^DAISY.edf"

# all_data = []
dirData = "Data/" #dirEdf = "Data/Edf"
dirEdf = "Data/Edf"
dirEdf = "Data/Temp"
psd_all = []
overlap = 0.10   #percentuale di sovrapposzione
window_size = 5 # Lunghezza della finestra in secondi
epoche = 200
batch_size = 16
num_clusters = 5
pazienza = 20


# for dirpath, dirnames, filenames in os.walk(dirData):
#     print(f"Directory: {dirpath}")

script_dir = os.path.dirname(os.path.abspath(__file__))
dirData = os.path.join(script_dir, 'Data')

# dirData = os.path.abspath('Data') #<-- corretto il percorso 

# Path relativo alla cartella 'edf'
path_edf = os.path.join(dirData, "Edf")

# print(f"percorso cartella edf {path_edf}")

images_path = os.path.join(dirData, "images")
weights_path = os.path.join(dirData, "model")
cluster_path = os.path.join(dirData, "cluster")

if not os.path.exists(images_path):
    os.makedirs(images_path)
    
if not os.path.exists(weights_path):
    os.makedirs(weights_path)

if not os.path.exists(cluster_path):
    os.makedirs(cluster_path)  


# print(f"percorso dirData = {dirData}")
# print(f"percorso dirEdf = {path_edf}")
# print(f"percorso images_path = {images_path}")
# print(f"percorso weights_path = {weights_path}")

path_edf = os.path.join(dirData, "Temp")

filenames = [f for f in os.listdir(path_edf) if "edf" in f]

# print(filenames)

# Salvataggio del modello Keras
model_path = os.path.join(weights_path, 'autoencoder_psd_model.h5')
grafico_app_path = os.path.join(images_path, 'grafico_apprendimento_psd.png')
grafico_cluster_path = os.path.join(images_path, 'grafico_cluster_psd.png')
grafico_cluster_path_no_auto = os.path.join(images_path, 'grafico_cluster_psd_no_auto.png')


segment_split_temp = []

for file in filenames:
    if file.endswith('.edf'):  # Controlla se il file ha estensione .edf
        file_path = os.path.join(path_edf, file)
        print(f"\tFile: {file_path}")
        #raw = mne.io.read_raw_edf(file_path, preload=True)

        raw = mne.io.read_raw_edf(file_path, preload=True)
        data, times = raw[:]
        sfreq = raw.info['sfreq']  # Frequenza di campionamento 

        psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq)

        # checkDistribuzione(data)

        #######  normalizzazione dei segnali 
        scaler = MinMaxScaler()
        # data_normalized = scaler.fit_transform(psds) 


         #-> salvattagio in array di tutti gli egg letti e post trasformazione
        sfreq = raw.info['sfreq']  # Frequenza di campionamento 

        #shape[0] -> righe |||| shape[1] -> colonne

        segment_length = int(window_size * sfreq)   

        step = int(segment_length * (1 - overlap))

        segment_split = segment_signal(data,segment_length)

        # segment_split = segment_signal_with_overlap(data_normalized,segment_length,step)
        # print(segment_split.shape) #formato (dati, canali, time_steps) 


        #parte di controllo della sovrapposzione
        # segment = np.array(segment_split)

        # check_overlap(segment, segment_length, overlap, sfreq)


        s,f = compute_spectrum_numpy(segment_split,sfreq)



        segment_split_temp.append(s)


# print(segment_split_temp)
for i in segment_split_temp:
    psd_all.extend(i)


# print(psd_all)
# print(psd_all.shape)
all_segments_standardized = np.array(psd_all)
print(all_segments_standardized.shape)



n_files, n_channels, n_frequencies = all_segments_standardized.shape

# scaler = StandardScaler()
all_segments_standardized = scaler.fit_transform(all_segments_standardized.reshape(-1, n_frequencies)).reshape(n_files, n_channels, n_frequencies)

early_stopping = EarlyStopping(monitor='val_loss', patience=pazienza, verbose=1, restore_best_weights=True) 

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    input_layer = Input(shape=(n_channels, n_frequencies))

    # Encoder LSTM
    encoded = LSTM(128, activation='relu', return_sequences=True)(input_layer)
    encoded = LSTM(64, activation='relu', return_sequences=True)(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = LSTM(32, activation='relu', return_sequences=False)(encoded)
    # encoded = LSTM(64, activation='relu', return_sequences=True)(encoded)
    encoded = Dense(32, activation='relu')(encoded)  # Riduzione ulteriorme della dimensione

    # Espansione della dimensione latente nello spazio originale
    decoded = RepeatVector(n_channels)(encoded)  # 
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)

    # Applicazione di un livello TimeDistributed per tornare al numero originale di frequenze
    decoded = TimeDistributed(Dense(n_frequencies))(decoded)



    # Crea il modello Autoencoder
    autoencoder = Model(inputs=input_layer, outputs=decoded)

    # Compila il modello
    autoencoder.compile(optimizer='adam', loss=MeanSquaredError())

    autoencoder.summary()

# Addestramento del modello
history = autoencoder.fit(all_segments_standardized, all_segments_standardized, 
                epochs=epoche, 
                batch_size=batch_size, 
                validation_split=0.2,
                callbacks=[early_stopping])



# #grafico dell'apprendimento
fig, ax = plt.subplots()
ax.plot(history.history["loss"],'r', marker='.', label="Model 1 Train Loss")
ax.plot(history.history["val_loss"],'r--', marker='.', label="Model 1 Val Loss")

ax.legend()

plt.savefig(grafico_app_path)
plt.close()


#salvataggio modello
autoencoder.save(model_path)

##### estrazione delle feature 
encoder = Model(inputs=autoencoder.input, outputs=encoded)

encoder.summary()

# Ottenere le feature codificate 
eeg_features = encoder.predict(all_segments_standardized)

eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)

print(f"numero delle feature {eeg_features.shape}")


##### clustering 
# kmeans = KMeans(n_clusters=num_clusters)

# Standardizzazione delle feature 
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(eeg_features)


# Inizializzazione della lista per il WCSS
wcss = []
silhouette_scores = []
k_range = range(1, 11)

for k in k_range:

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(eeg_features)
    cluster_assignments = kmeans.labels_
    wcss.append(kmeans.inertia_)  # WCSS

    #### valutazione dei risultati
    if len(set(kmeans.labels_)) > 1:
        sil_score = silhouette_score(eeg_features, cluster_assignments)
    else:
        sil_score = -1 
    
    silhouette_scores.append(sil_score)

# # applicazione clustering sui dati codificati
# kmeans.fit(eeg_features)

# # Assegna ogni segmento EEG al cluster più vicino
# cluster_assignments = kmeans.labels_

max_silhouette = max(silhouette_scores)
max_index = silhouette_scores.index(max_silhouette)
max_k = k_range[max_index]


# Creazione della figura
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12),sharex=True)

# Grafico WCSS (gomito)
ax1.plot(k_range, wcss, marker='o', color='blue', label='WCSS')
ax1.set_title('Metodo del Gomito e Silhouette Score')
ax1.set_xlabel('Numero di cluster (k)')
ax1.set_ylabel('WCSS', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid()

# Creazione del secondo asse y

# Grafico Silhouette Score
ax2.plot(k_range, silhouette_scores, marker='o', color='orange')
ax2.set_title('Silhouette Score per Numero di Cluster')
ax2.set_xlabel('Numero di cluster (k)')
ax2.set_ylabel('Silhouette Score', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.grid()

ax2.plot(max_k, max_silhouette, marker='o', color='red', markersize=10, label=f'Massimo Silhouette ({max_k}, {max_silhouette})')

plt.tight_layout()

# Salvataggio dell'immagine
plt.savefig(grafico_cluster_path, dpi=300, bbox_inches='tight')
# plt.show()


####find k senza autoencoder
wcss = []
silhouette_scores = []

print("inzio k senza autoencoder")
all_segments_standardized = all_segments_standardized.reshape(all_segments_standardized.shape[0], -1)

print(all_segments_standardized.shape)

for k in k_range:

    print(k)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(all_segments_standardized)
    cluster_assignments = kmeans.labels_
    wcss.append(kmeans.inertia_)  # WCSS

    #### valutazione dei risultati
    if len(set(kmeans.labels_)) > 1:
        sil_score = silhouette_score(all_segments_standardized, cluster_assignments)
    else:
        sil_score = -1 
    
    silhouette_scores.append(sil_score)

max_silhouette = max(silhouette_scores)
max_index = silhouette_scores.index(max_silhouette)
max_k = k_range[max_index]


# Creazione della figura
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12),sharex=True)

# Grafico WCSS (gomito)
ax1.plot(k_range, wcss, marker='o', color='blue', label='WCSS')
ax1.set_title('Metodo del Gomito e Silhouette Score')
ax1.set_xlabel('Numero di cluster (k)')
ax1.set_ylabel('WCSS', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid()

# Creazione del secondo asse y

# Grafico Silhouette Score
ax2.plot(k_range, silhouette_scores, marker='o', color='orange')
ax2.set_title('Silhouette Score per Numero di Cluster')
ax2.set_xlabel('Numero di cluster (k)')
ax2.set_ylabel('Silhouette Score', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.grid()

ax2.plot(max_k, max_silhouette, marker='o', color='red', markersize=10, label=f'Massimo Silhouette ({max_k}, {max_silhouette})')

plt.tight_layout()

# Salvataggio dell'immagine
plt.savefig(grafico_cluster_path_no_auto, dpi=300, bbox_inches='tight')

