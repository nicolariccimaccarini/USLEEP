import subprocess
import os
import time
import json
from datetime import datetime

def setup_environment():
    """Configura l'ambiente necessario per l'esecuzione degli script"""
    print("🔧 Configurazione ambiente...")
    
    # Directory necessarie (percorsi corretti)
    required_dirs = [
        "Data/Edf",           # Percorso corretto: Data è fuori da autoencoder
        "Data/Output",
        "clustering/Data",
        "results"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Creata directory: {dir_path}")
        else:
            print(f"✅ Directory esistente: {dir_path}")
    
    print("🔧 Setup completato!\n")

def check_script_prerequisites(script_path):
    """Verifica i prerequisiti per l'esecuzione di uno script"""
    if "autoencoder/trasformazione.py" in script_path:
        # Verifica esistenza directory Data/Edf (percorso corretto)
        edf_path = "Data/Edf"  # Corretto: Data è al livello root del progetto
        if not os.path.exists(edf_path):
            print(f"❌ Directory mancante: {edf_path}")
            return False
        
        # Verifica che ci siano file EDF
        try:
            edf_files = [f for f in os.listdir(edf_path) if f.endswith('.edf')]
            if not edf_files:
                print(f"❌ Nessun file .edf trovato in {edf_path}")
                return False
            else:
                print(f"✅ Trovati {len(edf_files)} file .edf in {edf_path}")
        except Exception as e:
            print(f"❌ Errore nell'accesso a {edf_path}: {e}")
            return False
    
    return True

def run_script(script_path):
    """Esegue uno script e misura il tempo di esecuzione"""
    print(f"Eseguo: {script_path}")
    
    # Verifica prerequisiti prima dell'esecuzione
    if not check_script_prerequisites(script_path):
        print(f"⚠️  Prerequisiti mancanti per {script_path}, salto l'esecuzione")
        return "Script saltato - prerequisiti mancanti", 0, -2
    
    start_time = time.time()
    
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Tempo di esecuzione: {execution_time:.2f} secondi")
    if result.stderr:
        print(f"Errori: {result.stderr}")
    print(result.stdout)
    print("-" * 40)
    
    return result.stdout, execution_time, result.returncode

def main():
    """
    Script principale per eseguire tutti gli algoritmi di ML e confrontare le prestazioni
    """
    print("=" * 60)
    print("CONFRONTO ALGORITMI ML - SPINDLE DETECTION")
    print("=" * 60)
    print(f"Inizio esecuzione: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup ambiente
    setup_environment()
    
    # Lista degli script ML principali
    scripts = [
        "autoencoder/trasformazione.py",
        "autoencoder/autoencoder_psd.py",
        "autoencoder/autoencoder_CI_psd.py",
        "autoencoder/autoencoder_CI_sovra_psd.py",
        "clustering/clustering.py",
        "clustering/clustering_no_ae.py",
        "clustering/k-means.py",
        "clustering/find_K.py",
        "src/calcolo_acc.py"
    ]

    results = {}

    # Esecuzione degli script
    for script in scripts:
        if os.path.exists(script):
            try:
                output, exec_time, return_code = run_script(script)
                metrics = extract_metrics(output)
                
                results[script] = {
                    "execution_time": exec_time,
                    "return_code": return_code,
                    "metrics": metrics,
                    "success": return_code == 0
                }
            except Exception as e:
                print(f"Errore nell'esecuzione di {script}: {e}")
                results[script] = {
                    "execution_time": 0,
                    "return_code": -1,
                    "metrics": {},
                    "success": False,
                    "error": str(e)
                }
        else:
            print(f"Script non trovato: {script}")

    # Confronto dettagliato dei risultati
    print("\n" + "=" * 60)
    print("CONFRONTO DETTAGLIATO PRESTAZIONI")
    print("=" * 60)
    
    # Tabella riassuntiva
    print(f"{'Script':<40} {'Tempo (s)':<12} {'Status':<10} {'Metriche'}")
    print("-" * 80)
    
    for script, data in results.items():
        script_name = script.split('/')[-1]  # Solo nome file
        time_str = f"{data['execution_time']:.2f}" if data['execution_time'] > 0 else "N/A"
        status = "✓ OK" if data['success'] else "✗ FAIL"
        metrics_summary = ", ".join([f"{k}: {v:.3f}" for k, v in data['metrics'].items()])
        
        print(f"{script_name:<40} {time_str:<12} {status:<10} {metrics_summary}")
    
    # Analisi delle prestazioni temporali
    successful_runs = {k: v for k, v in results.items() if v['success']}
    if successful_runs:
        print(f"\n--- ANALISI TEMPORALE ---")
        times = [(k.split('/')[-1], v['execution_time']) for k, v in successful_runs.items()]
        times.sort(key=lambda x: x[1])
        
        print(f"Script più veloce: {times[0][0]} ({times[0][1]:.2f}s)")
        print(f"Script più lento: {times[-1][0]} ({times[-1][1]:.2f}s)")
        print(f"Tempo totale: {sum([t[1] for t in times]):.2f}s")
    
    # Analisi delle metriche
    print(f"\n--- ANALISI METRICHE ---")
    best_accuracy = max([(k, v['metrics'].get('accuracy', 0)) for k, v in successful_runs.items()], 
                       key=lambda x: x[1], default=(None, 0))
    best_silhouette = max([(k, v['metrics'].get('silhouette_score', -1)) for k, v in successful_runs.items()], 
                         key=lambda x: x[1], default=(None, -1))
    
    if best_accuracy[1] > 0:
        print(f"Migliore accuratezza: {best_accuracy[0].split('/')[-1]} ({best_accuracy[1]:.3f})")
    if best_silhouette[1] > -1:
        print(f"Migliore silhouette score: {best_silhouette[0].split('/')[-1]} ({best_silhouette[1]:.3f})")
    
    # Salva risultati in JSON
    output_file = "results_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRisultati dettagliati salvati in: {output_file}")
    
    print(f"\nFine esecuzione: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()