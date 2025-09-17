import mne 
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

def compute_spectrum_numpy(segments, freq_sample):
    spectrums = []
    
    # Frequenze per l'asse delle X, calcolate una sola volta
    segment_length = segments[0].shape[1]
    frequencies = np.fft.fftfreq(segment_length, d=1/freq_sample)
    # frequencies = frequencies[:segment_length // 2]

    # print(len(frequencies))
    
    # Loop per calcolare lo spettro di ogni segmento
    for segment in segments:
        # Applica la trasformata di Fourier a ogni canale separatamente
        segment_spectrum = np.abs(np.fft.fft(segment, axis=1))  # Calcola il modulo della trasformata
        # segment_spectrum = np.abs(segment_spectrum[:, :segment_length // 2])
        # print(len(segment_spectrum))
        spectrums.append(segment_spectrum)
    
    return spectrums, frequencies




dirData = "Data/Temp"
window_size = 0.5

### metodo per la lettura con file in divisi per cartelle

all_data = []
for dirpath, dirnames, filenames in os.walk(dirData):
    print(f"Directory: {dirpath}")
    for file in filenames:
        if file.endswith('.edf'):  # Controlla se il file ha estensione .edf
            file_path = os.path.join(dirpath, file)
            print(f"\tFile: {file_path}")
            raw = mne.io.read_raw_edf(file_path, preload=True)

            # # all_data.append(raw)

            # sfreq = raw.info['sfreq']  # Frequenza di campionamento 

            # data, times = raw[:]
            # # data = raw.get_data()


            # scaler = MinMaxScaler()
            # data_normalized = scaler.fit_transform(data.T).T #.T fa la trasposta perchè sklearn vuole i dati disposti per colonna

            # #### fine normalizzazione

            # #-> salvattagio in array di tutti gli egg letti e post trasformazione
            # sfreq = raw.info['sfreq']  # Frequenza di campionamento 

            # #shape[0] -> righe |||| shape[1] -> colonne

            # segment_length = int(window_size * sfreq)   

        

            # segment_split = segment_signal(data_normalized,segment_length)

            # segment_split = segment_signal(data_normalized,segment_length)

            # # print(segment_split)

            # s,f = compute_spectrum_numpy(segment_split,sfreq)

            # print(len(np.abs(s[:segment_length // 2])))
            # print(len(s[0]))

            # print(len(f[:segment_length // 2]))

            


            # plt.plot(f, s[15][15])  # Primo canale del primo segmento
            # plt.xlabel("Frequenza (Hz)")
            # plt.ylabel("Ampiezza")
            # plt.title("Spettro del primo segmento")
            # plt.show()

            raw = mne.io.read_raw_edf(file_path, preload=True)

            # Mostra le informazioni sul file
            print(raw.info)

            # Visualizza i dati del file EDF
            raw.plot()

            # Esplicita la visualizzazione del grafico
            plt.show()