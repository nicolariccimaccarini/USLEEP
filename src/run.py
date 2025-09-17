import subprocess
import os

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

def run_script(script_path):
    print(f"Eseguo: {script_path}")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(result.stdout)
    return result.stdout

def extract_metrics(output):
    # Estrai le metriche principali (accuracy, silhouette score) dai log
    metrics = {}
    if "Accuratezza:" in output:
        try:
            acc = float(output.split("Accuratezza:")[1].split("\n")[0].strip())
            metrics["accuracy"] = acc
        except:
            pass
    if "valore della sil_score" in output:
        try:
            sil = float(output.split("valore della sil_score ->")[1].split("\n")[0].strip())
            metrics["silhouette"] = sil
        except:
            pass
    return metrics

for script in scripts:
    if os.path.exists(script):
        output = run_script(script)
        metrics = extract_metrics(output)
        results[script] = metrics
    else:
        print(f"Script non trovato: {script}")

print("\n--- Confronto Metriche ---")
for script, metrics in results.items():
    print(f"{script}: {metrics}")