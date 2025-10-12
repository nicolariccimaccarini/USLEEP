import os
import pandas as pd
import re
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report


def parse_image_title(img_file):
    # Estrai cluster, inizio e fine dai titoli
    match = re.match(r"Cluster (\d+) - Segmento (\d+) - Canale (EEG[\w\s]+)\.png", img_file)
    if match:
        cluster = int(match.group(1))
        segment = int(match.group(2))
        channel = match.group(3).replace(' ', '')  # Rimuove gli spazi extra nel nome del canale (ad esempio "EEG Pz" -> "EEGPz")
        
        # setto start e end manualmente visto che ogni segmento dura 15 secondi
        start = (segment - 1) * 15.0
        end = segment * 15.0
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
            # Filtra il ground truth per il canale specifico
            ground_truth_channel = ground_truth[ground_truth['Canale'] == channel]

            # Trova gli spindle che si sovrappongono con il segmento [start, end]
            overlapping_spindles = ground_truth_channel[
                (ground_truth['Start_Time(s)'] < end) & (ground_truth['End_Time(s)'] > start)
            ]

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
                true_label = 2

            # Aggiungi anche il canale ai risultati
            results.append((true_label, cluster, channel))  # Label reale, cluster predetto, e canale
    
    return results

def calculate_accuracy_by_channel(matched_data):
    """Calcola l'accuratezza per ogni canale"""
    from collections import defaultdict
    
    channel_data = defaultdict(lambda: {'true': [], 'pred': []})
    
    # Raggruppa i dati per canale
    for true_label, predicted_label, channel in matched_data:
        channel_data[channel]['true'].append(true_label)
        channel_data[channel]['pred'].append(predicted_label)
    
    # Calcola l'accuratezza per ogni canale
    channel_accuracies = {}
    for channel, data in channel_data.items():
        if data['true']:  # Se ci sono dati per questo canale
            accuracy = accuracy_score(data['true'], data['pred'])
            channel_accuracies[channel] = {
                'accuracy': accuracy,
                'samples': len(data['true']),
                'true_labels': data['true'],
                'pred_labels': data['pred']
            }
    
    return channel_accuracies

def process_patient(patient_id):
    """Processa un singolo paziente e calcola l'accuratezza"""
    print(f"\n--- Processando paziente {patient_id} ---")
    
    # Percorsi per il paziente corrente
    image_folder = f'Data/Output/{patient_id}/images/clustering'
    ground_truth_path = f'Data/Mat_Output/{patient_id}/dati_start_end.csv'
    
    # Verifica che esistano i percorsi necessari
    if not os.path.exists(image_folder):
        print(f"Cartella immagini non trovata: {image_folder}")
        return None
    
    if not os.path.exists(ground_truth_path):
        print(f"File ground truth non trovato: {ground_truth_path}")
        return None
    
    # Carica i dati
    try:
        ground_truth = pd.read_csv(ground_truth_path)
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
        
        if not image_files:
            print(f"Nessuna immagine PNG trovata in {image_folder}")
            return None
            
    except Exception as e:
        print(f"Errore nel caricare i dati per il paziente {patient_id}: {e}")
        return None
    
    # Calcola le corrispondenze
    matched_data = match_clusters_with_ground_truth(image_files, ground_truth)
    
    if not matched_data:
        print(f"Nessun dato corrisponde per il paziente {patient_id}")
        return None
    
    # Separa true e predicted per l'accuratezza globale
    true_labels = [x[0] for x in matched_data]
    predicted_labels = [x[1] for x in matched_data]
    
    # Calcola accuratezza globale e report
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    overall_report = classification_report(true_labels, predicted_labels)
    
    # Calcola accuratezza per canale
    channel_accuracies = calculate_accuracy_by_channel(matched_data)
    
    return overall_accuracy, overall_report, channel_accuracies

def save_results(patient_id, overall_accuracy, overall_report, channel_accuracies):
    """Salva i risultati nella cartella appropriata"""
    output_dir = f'Data/Accuracy/{patient_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva accuratezza globale e per canale
    with open(f'{output_dir}/accuracy.txt', 'w') as f:
        f.write(f"=== ACCURATEZZA PAZIENTE {patient_id} ===\n\n")
        f.write(f"Accuratezza globale: {overall_accuracy:.4f}\n\n")
        
        f.write("=== ACCURATEZZA PER CANALE ===\n")
        f.write("-" * 50 + "\n")
        
        # Ordina i canali per nome
        sorted_channels = sorted(channel_accuracies.items())
        
        for channel, data in sorted_channels:
            f.write(f"Canale {channel}:\n")
            f.write(f"  Accuratezza: {data['accuracy']:.4f}\n")
            f.write(f"  Campioni: {data['samples']}\n")
            f.write(f"  True labels: {data['true_labels']}\n")
            f.write(f"  Pred labels: {data['pred_labels']}\n")
            f.write("-" * 30 + "\n")
    
    # Salva report globale
    with open(f'{output_dir}/classification_report.txt', 'w') as f:
        f.write(f"Classification Report - Paziente {patient_id}\n")
        f.write("=" * 50 + "\n")
        f.write(overall_report)
    
    # Salva report dettagliato per canale
    with open(f'{output_dir}/channel_detailed_report.txt', 'w') as f:
        f.write(f"Detailed Channel Report - Paziente {patient_id}\n")
        f.write("=" * 60 + "\n\n")
        
        for channel, data in sorted(channel_accuracies.items()):
            f.write(f"CANALE: {channel}\n")
            f.write("-" * 30 + "\n")
            try:
                channel_report = classification_report(data['true_labels'], data['pred_labels'])
                f.write(channel_report)
            except Exception as e:
                f.write(f"Errore nel generare il report per {channel}: {e}\n")
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"Risultati salvati in {output_dir}")

def main():
    """Funzione principale che processa tutti i pazienti"""
    # Crea la cartella principale per i risultati
    os.makedirs('Data/Accuracy', exist_ok=True)
    
    # Liste per raccogliere le accuratezze di tutti i pazienti
    all_accuracies = []
    all_channel_accuracies = defaultdict(list)
    
    # Processa tutti i pazienti da 1 a 100
    for patient_id in range(1, 101):
        result = process_patient(patient_id)
        
        if result is not None:
            overall_accuracy, overall_report, channel_accuracies = result
            all_accuracies.append(overall_accuracy)
            
            # Raccogli le accuratezze per canale
            for channel, data in channel_accuracies.items():
                all_channel_accuracies[channel].append(data['accuracy'])
            
            print(f"Paziente {patient_id} - Accuratezza globale: {overall_accuracy:.4f}")
            
            # Salva i risultati
            save_results(patient_id, overall_accuracy, overall_report, channel_accuracies)
        else:
            print(f"Impossibile processare il paziente {patient_id}")
    
    # Calcola e salva le statistiche generali
    if all_accuracies:
        mean_accuracy = sum(all_accuracies) / len(all_accuracies)
        max_accuracy = max(all_accuracies)
        min_accuracy = min(all_accuracies)
        
        # Salva statistiche generali
        with open('Data/Accuracy/overall_statistics.txt', 'w') as f:
            f.write("=== STATISTICHE GENERALI - TUTTI I PAZIENTI ===\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Numero di pazienti processati: {len(all_accuracies)}\n")
            f.write(f"Accuratezza media globale: {mean_accuracy:.4f}\n")
            f.write(f"Accuratezza massima globale: {max_accuracy:.4f}\n")
            f.write(f"Accuratezza minima globale: {min_accuracy:.4f}\n\n")
            
            f.write("=== STATISTICHE PER CANALE ===\n")
            f.write("-" * 60 + "\n")
            
            for channel, accuracies in sorted(all_channel_accuracies.items()):
                if accuracies:
                    mean_ch_acc = sum(accuracies) / len(accuracies)
                    max_ch_acc = max(accuracies)
                    min_ch_acc = min(accuracies)
                    f.write(f"\nCanale {channel}:\n")
                    f.write(f"  Pazienti con dati: {len(accuracies)}\n")
                    f.write(f"  Accuratezza media: {mean_ch_acc:.4f}\n")
                    f.write(f"  Accuratezza massima: {max_ch_acc:.4f}\n")
                    f.write(f"  Accuratezza minima: {min_ch_acc:.4f}\n")
            
            f.write(f"\n{'='*60}\n")
            f.write(f"Accuratezze globali individuali:\n")
            for i, acc in enumerate(all_accuracies, 1):
                f.write(f"  Paziente {i}: {acc:.4f}\n")
        
        print(f"\n--- Statistiche Finali ---")
        print(f"Pazienti processati: {len(all_accuracies)}")
        print(f"Accuratezza media globale: {mean_accuracy:.4f}")
        print(f"Statistiche salvate in Data/Accuracy/overall_statistics.txt")
    else:
        print("Nessun paziente è stato processato con successo.")

if __name__ == "__main__":
    main()