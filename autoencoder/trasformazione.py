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
           

##### script

#file_path = "Data/Sartini_Daisy/SARTINI^DAISY.edf"

# all_data = []
dirData = "Data/" #dirEdf = "Data/Edf"
dirEdf = "Data/Edf"
dirEdf = "Data/Temp"
segment_split_all = []
overlap = 0.10   #percentuale di sovrapposzione
window_size = 5 # Lunghezza della finestra in secondi
epoche = 60
batch_size = 512
num_clusters = 5
pazienza = 5


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

# path_edf = os.path.join(dirData, "Temp")

filenames = [f for f in os.listdir(path_edf) if "edf" in f]

# print(filenames)

# Salvataggio del modello Keras
model_path = os.path.join(weights_path, 'autoencoder_model.h5')
grafico_app_path = os.path.join(images_path, 'grafico_apprendimento.png')
grafico_cluster_path = os.path.join(images_path, 'grafico_cluster.png')


segment_split_temp = []

for file in filenames:
    if file.endswith('.edf'):  # Controlla se il file ha estensione .edf
        file_path = os.path.join(path_edf, file)
        print(f"\tFile: {file_path}")
        #raw = mne.io.read_raw_edf(file_path, preload=True)

        raw = mne.io.read_raw_edf(file_path, preload=True)
        data, times = raw[:]

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
print(all_segments_standardized.shape)

#cancellazione della lista originale 
del segment_split_all

######### rete neurale

#aggiunta di una dimensione
eeg_segments = np.expand_dims(all_segments_standardized, axis=-1)

print(eeg_segments.shape)

input_shape = eeg_segments.shape[1:]  # Restituisce (canali, campioni_temporali, 1)

print("Forma dell'input:", input_shape)


# Primo step: Dividere il dataset in 80% train+validation e 20% test    
eeg_train_val, eeg_test = train_test_split(eeg_segments, test_size=0.2, random_state=42)

# Secondo step: Dividere il train+validation set in 80% train e 20% validation
eeg_train, eeg_val = train_test_split(eeg_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2 del totale

# Verifica delle forme dei dati
print("Forma dei dati di training:", eeg_train.shape)
print("Forma dei dati di validation:", eeg_val.shape)
print("Forma dei dati di test:", eeg_test.shape)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    input_eeg = layers.Input(shape=input_shape)

    # Encoder convoluzionale
    # Blocchi Conv2D + Pooling
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_eeg)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x) 

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 1))(x)  

    # Bottleneck (la rappresentazione latente, codifica)
    encoded = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    # Decoder convoluzionale
    # Blocchi di UpSampling2D + Conv2D Trasposta (Conv2DTranspose)
    x = layers.Conv2DTranspose(256, (3, 3), padding='same', activation='relu')(encoded)
    x = layers.Conv2DTranspose(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(3, 1))(x)  

    x = layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)  

    x = layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)  

    x = layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)  

    # Strato finale per la ricostruzione
    decoded = layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)  

    # Reshape finale per ottenere la forma corretta
    decoded = layers.Lambda(lambda x: tf.image.resize(x, (26, segment_length)), output_shape=(26, segment_length, 1))(decoded)
    
    

    # Creazione del modello Autoencoder
    autoencoder = models.Model(input_eeg, decoded)

    # Configurare l'early stopping                                  #verbose stampa il messegggio di attivazione di early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=pazienza, verbose=1, restore_best_weights=True)

    # Compilazione del modello
    autoencoder.compile(optimizer='adam', loss=MeanSquaredError())


# autoencoder.summary()


history = autoencoder.fit(eeg_train, eeg_train, 
                epochs=epoche, 
                batch_size=batch_size, 
                shuffle=True, 
                validation_data=(eeg_val, eeg_val),
                callbacks=[early_stopping])



#grafico dell'apprendimento
fig, ax = plt.subplots()
ax.plot(history.history["loss"],'r', marker='.', label="Model 1 Train Loss")
ax.plot(history.history["val_loss"],'r--', marker='.', label="Model 1 Val Loss")

ax.legend()

plt.savefig(grafico_app_path)
plt.close()


#salvataggio modello
autoencoder.save(model_path)

# # Salvataggio solo dei pesi
# autoencoder.save_weights('Data/weigths/autoencoder_weights')

##### estrazione delle feature 
encoder = Model(inputs=autoencoder.input, outputs=encoded)

encoder.summary()

# Ottenere le feature codificate 
eeg_features = encoder.predict(eeg_test)

eeg_features = eeg_features.reshape(eeg_features.shape[0], -1)

print(f"numero delle feature {eeg_features.shape}")

##### clustering 
kmeans = KMeans(n_clusters=num_clusters)

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
