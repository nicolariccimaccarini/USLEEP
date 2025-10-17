import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def load_data(python_csv_path, matlab_csv_path):
    """Carica i file CSV di Python e MATLAB"""
    try:
        # Carica risultati Python
        python_data = pd.read_csv(python_csv_path)
        print(f"✅ Caricati {len(python_data)} spindles da Python")
        
        # Carica risultati MATLAB (adatta il formato se necessario)
        matlab_data = pd.read_csv(matlab_csv_path)
        print(f"✅ Caricati {len(matlab_data)} spindles da MATLAB")
        
        return python_data, matlab_data
        
    except Exception as e:
        print(f"❌ Errore caricamento dati: {e}")
        return None, None

def find_matching_spindles(python_spindles, matlab_spindles, tolerance=0.5):
    """
    Trova spindles corrispondenti tra Python e MATLAB
    
    Args:
        python_spindles: DataFrame con spindles Python
        matlab_spindles: DataFrame con spindles MATLAB  
        tolerance: tolleranza in secondi per considerare due spindles come corrispondenti
    
    Returns:
        dict con statistiche di corrispondenza per canale
    """
    results = defaultdict(lambda: {
        'python_count': 0,
        'matlab_count': 0,
        'matched': 0,
        'python_only': 0,
        'matlab_only': 0,
        'matches': []
    })
    
    # Raggruppa per canale
    python_by_channel = python_spindles.groupby('Canale')
    matlab_by_channel = matlab_spindles.groupby('Canale') if 'Canale' in matlab_spindles.columns else matlab_spindles.groupby(matlab_spindles.columns[0])
    
    all_channels = set(python_spindles['Canale'].unique()) | set(matlab_spindles.iloc[:, 0].unique())
    
    for channel in all_channels:
        print(f"🔍 Analizzando canale: {channel}")
        
        # Ottieni spindles per il canale corrente
        py_channel = python_by_channel.get_group(channel) if channel in python_by_channel.groups else pd.DataFrame()
        mat_channel = matlab_by_channel.get_group(channel) if channel in matlab_by_channel.groups else pd.DataFrame()
        
        results[channel]['python_count'] = len(py_channel)
        results[channel]['matlab_count'] = len(mat_channel)
        
        if len(py_channel) == 0 and len(mat_channel) == 0:
            continue
            
        if len(py_channel) == 0:
            results[channel]['matlab_only'] = len(mat_channel)
            continue
            
        if len(mat_channel) == 0:
            results[channel]['python_only'] = len(py_channel)
            continue
        
        # Trova corrispondenze
        matched_python = set()
        matched_matlab = set()
        
        for py_idx, py_row in py_channel.iterrows():
            py_start = py_row['Start_Time(s)']
            py_end = py_row['End_Time(s)']
            
            best_match = None
            best_overlap = 0
            
            for mat_idx, mat_row in mat_channel.iterrows():
                if mat_idx in matched_matlab:
                    continue
                    
                mat_start = mat_row.iloc[1]  # Assumendo che la seconda colonna sia start
                mat_end = mat_row.iloc[2]    # Assumendo che la terza colonna sia end
                
                # Calcola sovrapposizione
                overlap_start = max(py_start, mat_start)
                overlap_end = min(py_end, mat_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                # Calcola distanza tra centri
                py_center = (py_start + py_end) / 2
                mat_center = (mat_start + mat_end) / 2
                center_distance = abs(py_center - mat_center)
                
                # Considera match se:
                # 1. I centri sono vicini (entro tolleranza)
                # 2. C'è sovrapposizione significativa
                if center_distance <= tolerance and overlap_duration > 0:
                    if overlap_duration > best_overlap:
                        best_overlap = overlap_duration
                        best_match = mat_idx
            
            if best_match is not None:
                matched_python.add(py_idx)
                matched_matlab.add(best_match)
                
                # Salva dettagli del match
                mat_row = mat_channel.loc[best_match]
                results[channel]['matches'].append({
                    'py_start': py_start,
                    'py_end': py_end,
                    'mat_start': mat_row.iloc[1],
                    'mat_end': mat_row.iloc[2],
                    'start_diff': abs(py_start - mat_row.iloc[1]),
                    'end_diff': abs(py_end - mat_row.iloc[2]),
                    'overlap': best_overlap
                })
        
        results[channel]['matched'] = len(matched_python)
        results[channel]['python_only'] = len(py_channel) - len(matched_python)
        results[channel]['matlab_only'] = len(mat_channel) - len(matched_matlab)
    
    return dict(results)

def print_comparison_results(results):
    """Stampa i risultati del confronto"""
    print("\n" + "="*80)
    print("📊 RISULTATI CONFRONTO SPINDLES")
    print("="*80)
    
    total_python = 0
    total_matlab = 0
    total_matched = 0
    
    for channel, stats in results.items():
        print(f"\n🧠 Canale: {channel}")
        print(f"   Python: {stats['python_count']} spindles")
        print(f"   MATLAB: {stats['matlab_count']} spindles")
        print(f"   Corrispondenti: {stats['matched']}")
        print(f"   Solo Python: {stats['python_only']}")
        print(f"   Solo MATLAB: {stats['matlab_only']}")
        
        if stats['matched'] > 0:
            if stats['python_count'] > 0:
                precision = stats['matched'] / stats['python_count'] * 100
                print(f"   Precisione: {precision:.1f}%")
            if stats['matlab_count'] > 0:
                recall = stats['matched'] / stats['matlab_count'] * 100
                print(f"   Recall: {recall:.1f}%")
        
        total_python += stats['python_count']
        total_matlab += stats['matlab_count']
        total_matched += stats['matched']
    
    # Statistiche globali
    print(f"\n🌍 STATISTICHE GLOBALI:")
    print(f"   Totale Python: {total_python}")
    print(f"   Totale MATLAB: {total_matlab}")
    print(f"   Totale corrispondenti: {total_matched}")
    
    if total_python > 0:
        overall_precision = total_matched / total_python * 100
        print(f"   Precisione globale: {overall_precision:.1f}%")
    
    if total_matlab > 0:
        overall_recall = total_matched / total_matlab * 100
        print(f"   Recall globale: {overall_recall:.1f}%")
    
    if total_matched > 0:
        f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        print(f"   F1-Score: {f1_score:.1f}%")

def analyze_time_differences(results):
    """Analizza le differenze temporali nei match"""
    print("\n" + "="*80)
    print("⏱️ ANALISI DIFFERENZE TEMPORALI")
    print("="*80)
    
    all_start_diffs = []
    all_end_diffs = []
    
    for channel, stats in results.items():
        if not stats['matches']:
            continue
            
        start_diffs = [m['start_diff'] for m in stats['matches']]
        end_diffs = [m['end_diff'] for m in stats['matches']]
        
        all_start_diffs.extend(start_diffs)
        all_end_diffs.extend(end_diffs)
        
        print(f"\n🧠 {channel}:")
        print(f"   Diff. tempo inizio - Media: {np.mean(start_diffs):.3f}s, Max: {np.max(start_diffs):.3f}s")
        print(f"   Diff. tempo fine - Media: {np.mean(end_diffs):.3f}s, Max: {np.max(end_diffs):.3f}s")
    
    if all_start_diffs:
        print(f"\n🌍 GLOBALE:")
        print(f"   Diff. inizio - Media: {np.mean(all_start_diffs):.3f}s ± {np.std(all_start_diffs):.3f}s")
        print(f"   Diff. fine - Media: {np.mean(all_end_diffs):.3f}s ± {np.std(all_end_diffs):.3f}s")

def create_comparison_plot(results, output_path):
    """Crea un grafico di confronto"""
    channels = list(results.keys())
    python_counts = [results[ch]['python_count'] for ch in channels]
    matlab_counts = [results[ch]['matlab_count'] for ch in channels]
    matched_counts = [results[ch]['matched'] for ch in channels]
    
    x = np.arange(len(channels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, python_counts, width, label='Python', alpha=0.8)
    ax.bar(x, matlab_counts, width, label='MATLAB', alpha=0.8)
    ax.bar(x + width, matched_counts, width, label='Corrispondenti', alpha=0.8)
    
    ax.set_xlabel('Canali')
    ax.set_ylabel('Numero Spindles')
    ax.set_title('Confronto Rilevamento Spindles: Python vs MATLAB')
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Grafico salvato in: {output_path}")

def main():
    """Funzione principale"""
    # Configurazione percorsi
    base_path = "/mnt/c/Users/nicol/OneDrive/Documenti/GitHub/ML-for-Spindle-Detection-in-EEESWAS"
    
    python_csv = os.path.join(base_path, "Data/Output/1/cluster/start_end_per_channel.csv")
    
    matlab_csv = os.path.join(base_path, "Data/Mat_Output/1/dati_start_end.csv")
    
    # Tolleranza per il matching (secondi)
    tolerance = 0.5
    
    print(f"\n🔍 Confrontando:")
    print(f"   Python: {python_csv}")
    print(f"   MATLAB: {matlab_csv}")
    print(f"   Tolleranza: {tolerance}s")
    
    # Carica dati
    python_data, matlab_data = load_data(python_csv, matlab_csv)
    if python_data is None or matlab_data is None:
        return
    
    # Esegui confronto
    results = find_matching_spindles(python_data, matlab_data, tolerance)
    
    # Mostra risultati
    print_comparison_results(results)
    analyze_time_differences(results)
    
    # Crea cartella di output per i risultati di accuratezza
    output_dir = os.path.join(base_path, "Data/Accuracy")
    os.makedirs(output_dir, exist_ok=True)
    
    # Crea grafico
    plot_path = os.path.join(output_dir, "spindle_comparison.png")
    create_comparison_plot(results, plot_path)
    
    # Salva risultati dettagliati
    detailed_results = []
    for channel, stats in results.items():
        detailed_results.append({
            'Canale': channel,
            'Python_Count': stats['python_count'],
            'MATLAB_Count': stats['matlab_count'],
            'Matched': stats['matched'],
            'Python_Only': stats['python_only'],
            'MATLAB_Only': stats['matlab_only'],
            'Precision(%)': (stats['matched'] / stats['python_count'] * 100) if stats['python_count'] > 0 else 0,
            'Recall(%)': (stats['matched'] / stats['matlab_count'] * 100) if stats['matlab_count'] > 0 else 0
        })
    
    results_df = pd.DataFrame(detailed_results)
    results_csv_path = os.path.join(output_dir, "comparison_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    print(f"\n💾 Risultati dettagliati salvati in: {results_csv_path}")
    print(f"📊 Grafico salvato in: {plot_path}")
    print(f"📁 Tutti i file di output sono in: {output_dir}")
    print("✅ Analisi completata!")

if __name__ == "__main__":
    main()