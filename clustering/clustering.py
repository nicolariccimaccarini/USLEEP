import mne 
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.models import load_model

# Abilita la deserializzazione unsafe per i layer Lambda
tf.keras.config.enable_unsafe_deserialization()

print(f"Versione TensorFlow: {tf.__version__}")

def get_file_output_path(base_data_path, filename=None):
    """Crea e restituisce il percorso per l'output specifico del file"""
    if filename:
        file_base_name = os.path.splitext(filename)[0]
        file_output_path = os.path.join(base_data_path, file_base_name)
        os.makedirs(os.path.join(file_output_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(file_output_path, "model"), exist_ok=True)
        os.makedirs(os.path.join(file_output_path, "cluster"), exist_ok=True)
        return file_output_path
    else:
        return base_data_path

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
           




##### script

segment_split_all = []
overlap = 0.10  #percentuale di sovrapposzione
window_size = 15 # Lunghezza della finestra in secondi
num_clusters = 3

# Path relativo alla cartella 'edf'
path_edf = os.environ.get('DATA_PATH', 'Data/Edf')
output_path = os.environ.get('OUTPUT_PATH', 'Data/Output')
base_path = os.environ.get('BASE_PATH', '.')
current_file = os.environ.get('CURRENT_FILE', None)

# Usa il percorso specifico del file se disponibile
if current_file:
    dirData = get_file_output_path(output_path, current_file)
    # Per clustering, usa solo il file corrente dalla cartella EDF originale
    filenames = [current_file]  # Processa solo il file corrente
    print(f"Modalità file singolo: processando {current_file}")
else:
    dirData = output_path
    filenames = [f for f in os.listdir(path_edf) if "edf" in f]
    print(f"Modalità batch: processando {len(filenames)} file")

segment_split_temp = []

images_path = os.path.join(dirData, "images")
weights_path = os.path.join(base_path, 'Data', 'weights')

images_clus_path = os.path.join(images_path, "clustering")

model_path = os.path.join(weights_path, 'autoencoder_model.h5')

if not os.path.exists(images_clus_path):
    os.makedirs(images_clus_path)

for file in filenames:
    if file.endswith('.edf'):  # Controlla se il file ha estensione .edf
        # Quando current_file è specificato, leggi dalla cartella EDF originale
        if current_file:
            file_path = os.path.join(path_edf, file)
        else:
            # In modalità batch, usa il cluster path se esiste, altrimenti EDF originale
            cluster_path = os.path.join(dirData, "cluster")
            if os.path.exists(os.path.join(cluster_path, file)):
                file_path = os.path.join(cluster_path, file)
            else:
                file_path = os.path.join(path_edf, file)
        
        print(f"\tFile: {file_path}")
        #raw = mne.io.read_raw_edf(file_path, preload=True)

        raw = mne.io.read_raw_edf(file_path, preload=True)
        data, times = raw[:]
        channel_names = raw.ch_names  # Ottieni i nomi dei canali dal file EDF
    
        # checkDistribuzione(data)

        #######  normalizzazione dei segnali 
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data.T).T #.T fa la trasposta perchè sklearn vuole i dati disposti per colonna

        #### fine normalizzazione


        ####### standardizzazione

        # scaler = StandardScaler()
        # data_normalized = scaler.fit_transform(data.T).T #.T fa la trasposta perchè sklearn vuole i dati disposti per colonna


        # checkMaxMin(data_normalized) #min = 0 e max = 1

        # checkMediaDeviazione(data_normalized) #media =~ 0 e std =~ 1


        #### fine standarizzazione

        #-> salvattagio in array di tutti gli egg letti e post trasformazione
        sfreq = raw.info['sfreq']  # Frequenza di campionamento 

        #shape[0] -> righe |||| shape[1] -> colonne

        segment_length = int(window_size * sfreq)   

        step = int(segment_length * (1 - overlap))

        # segment_split = segment_signal(data_normalized,segment_length)

        segment_split = segment_signal_with_overlap(data_normalized,segment_length,step)
        # print(segment_split.shape) #formato (dati, canali, time_steps) 


        #parte di controllo della sovrapposzione
        # segment = np.array(segment_split)

        # check_overlap(segment, segment_length, overlap, sfreq)

        segment_split_temp.append(segment_split)


# print(segment_split_temp)
for i in segment_split_temp:
    segment_split_all.extend(i)


# print(segment_split_all)
all_segments_standardized = np.array(segment_split_all)
# print(all_segments_standardized.shape)

eeg_segments = np.expand_dims(all_segments_standardized, axis=-1)

#cancellazione della lista originale 
# del segment_split_all


###caricamento dell'autoencoder
autoencoder = load_model(model_path, safe_mode=False)

autoencoder.summary()

##### estrazione delle feature 
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('conv2d_8').output)

encoder.summary()

# Ottenere le feature codificate 
eeg_features = encoder.predict(eeg_segments)

print("fine del predict")

eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)

##### clustering 
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(eeg_features)

# Ottieni le etichette dei cluster
cluster_labels = kmeans.labels_

# Puoi anche ottenere i centri dei cluster se necessario
cluster_centers = kmeans.cluster_centers_

print(f"labels dei cluster {cluster_labels}\nCentro dei cluster {cluster_centers}")


sil_score = silhouette_score(eeg_features, cluster_labels)

print(f"valore della sil_score -> {sil_score}")


###### PCA


# Riduci le dimensioni a 2D
pca = PCA(n_components=2)
eeg_features_2d = pca.fit_transform(eeg_features)

#grafico PCA
plt.figure(figsize=(10, 8))
plt.scatter(eeg_features_2d[:, 0], eeg_features_2d[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Visualizzazione dei Cluster con PCA')
plt.xlabel('Componente Principale 1')
plt.ylabel('Componente Principale 2')
plt.colorbar(label='Cluster Label')

grafico_PCA_path = os.path.join(images_clus_path, 'grafico_PCA.png')
plt.savefig(grafico_PCA_path, dpi=300, bbox_inches='tight')
# plt.show()
plt.close()



####creazione grafico per mostrare i segnali

# Imposta il numero massimo di segmenti per cluster
num_segments_per_cluster = 15

# Mostra il segnale associato a un punto di ogni cluster
for cluster in range(num_clusters):
    # Trova gli indici dei segmenti nel cluster
    cluster_indices = np.where(cluster_labels == cluster)[0]
    
    selected_indices = cluster_indices[:num_segments_per_cluster]
    
    all_segment = segment_signal(data,segment_length)
    # print(all_segment.shape)

    # selected_segment = all_segment[selected_index]

    try:
        # Get unique clusters and their counts
        unique_clusters = np.unique(cluster_labels)  # Changed from 'labels' to 'cluster_labels'
        print(f"Cluster univoci trovati: {unique_clusters}")
        print(f"Numero totale segmenti: {len(all_segment)}")
        
        selected_segments = []
        for cluster_id in unique_clusters:
            # Find indices for this cluster
            cluster_indices_for_id = np.where(cluster_labels == cluster_id)[0]  # Changed from 'labels' to 'cluster_labels'
            
            # Ensure we have valid indices
            valid_indices = cluster_indices_for_id[cluster_indices_for_id < len(all_segment)]
            
            if len(valid_indices) > 0:
                # Select the first valid segment for this cluster
                selected_index = valid_indices[0]
                selected_segment = all_segment[selected_index]
                selected_segments.append(selected_segment)
                print(f"Cluster {cluster_id}: selezionato segmento {selected_index}")
            else:
                print(f"Warning: Nessun segmento valido trovato per cluster {cluster_id}")
        
    except IndexError as e:
        print(f"Errore nell'accesso ai segmenti: {e}")
        print(f"Cluster_labels shape: {cluster_labels.shape if hasattr(cluster_labels, 'shape') else len(cluster_labels)}")
        print(f"All_segment length: {len(all_segment)}")
        print(f"Selected_index: {selected_index if 'selected_index' in locals() else 'Not defined'}")
        raise

    for i, selected_index in enumerate(selected_indices):
        # Check if selected_index is within bounds
        if selected_index >= len(all_segment):
            print(f"Warning: selected_index {selected_index} out of bounds for all_segment (length: {len(all_segment)})")
            continue
            
        # Seleziona il segmento corrente
        selected_segment = all_segment[selected_index]
        
        segment_length_actual = selected_segment.shape[1]
        time_array = np.arange(0, segment_length_actual) / sfreq

        # Visualizza i segnali di tutti i canali
        for channel in range(len(channel_names)):  # Supponiamo di avere 26 canali
            plt.figure(figsize=(12, 6))
            plt.plot(time_array, selected_segment[channel, :], label=f'Canale {channel_names[channel]}')
            plt.title(f'Segnale Associato al Cluster {cluster} - Canale {channel_names[channel]}')
            plt.xlabel('Tempo')
            plt.ylabel('Ampiezza')
            plt.grid()
            plt.legend()      
            nome = f"Cluster {cluster} - Segmento {i+1} - Canale {channel_names[channel]}.png"
            grafico_cluster_path = os.path.join(images_clus_path, nome)

            plt.savefig(grafico_cluster_path, dpi=300, bbox_inches='tight')
            plt.close()
            # plt.show()



