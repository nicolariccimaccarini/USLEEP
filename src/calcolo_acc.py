import os
import pandas as pd
import re
from sklearn.metrics import accuracy_score, classification_report


def parse_image_title(img_file):
    # Estrai cluster, inizio e fine dai titoli
    match = re.match(r"Cluster (\d+) - Segmento (\d+) - Canale (EEG[\w\s]+) - \((\d+\.\d+) -> (\d+\.\d+)\)\.png", img_file)
    if match:
        cluster = int(match.group(1))
        segment = int(match.group(2))
        channel = match.group(3).replace(' ', '')  # Rimuove gli spazi extra nel nome del canale (ad esempio "EEG Pz" -> "EEGPz")
        start = float(match.group(4))
        end = float(match.group(5))
        return cluster, channel, start, end
    return None

def match_clusters_with_ground_truth(cluster_data, ground_truth):
    results = []

    for img_file in cluster_data:
        parsed = parse_image_title(img_file)
        if parsed is None:
            print(f"Errore nel parsing del file: {img_file}")
            continue

        
        
        cluster, channel, start, end = parsed

        if cluster != 2:
            # print(channel)

            # Filtra il ground truth per il canale specifico
            ground_truth_channel = ground_truth[ground_truth['Canale'] == channel]

            # Trova gli spindle che si sovrappongono con il segmento [start, end]
            overlapping_spindles = ground_truth_channel[
                (ground_truth['Start_Time(s)'] < end) & (ground_truth['End_Time(s)'] > start)
            ]

            # true_label = 1 if not overlapping_spindles.empty else 0

            if not overlapping_spindles.empty:
                print(f"Cluster -> {cluster}, canale {channel}, Sovrapposizione trovata: {overlapping_spindles}")
            else:
                print(f"Cluster -> {cluster}, canale {channel}, Nessuna sovrapposizione per il segmento {start} -> {end}")

                # Associa la corretta etichetta a seconda del cluster:
            if cluster == 0:  # Cluster 0 corrisponde agli "spindle"
                true_label = 1 if not overlapping_spindles.empty else 0
            elif cluster == 1:  # Cluster 1 corrisponde ai "non spindle"
                true_label = 0 if overlapping_spindles.empty else 1
            else:
                # Se hai altre classi (ad esempio cluster 2), puoi gestirle qui
                true_label = 2  # Solo un esempio; gestisci come necessario

            results.append((true_label, cluster))  # Label reale e cluster predetto
    
    return results


# Lista di file immagine

ground_truth = pd.read_csv('Data/98/dati_start_end.csv') 

image_folder = 'Data/Rclustering'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# print(image_files)

# print(ground_truth.head())
# print(ground_truth.columns)


matched_data = match_clusters_with_ground_truth(image_files, ground_truth)

if not matched_data:
    print("Nessun dato corrisponde. Controlla il ground truth o i nomi dei file.")
    exit()

# Separate true e predicted
true_labels = [x[0] for x in matched_data]
predicted_labels = [x[1] for x in matched_data]

# Accuratezza e report
accuracy = accuracy_score(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels)

print(f"Accuratezza: {accuracy}")
print(report)

