import subprocess
import os
import sys
from datetime import datetime

class MLPipelineRunner:
    def __init__(self):
        self.base_path = "/hpc/home/nmaccarini/ML-for-Spindle-Detection-in-EEESWAS/"
        self.data_path = os.path.join(self.base_path, "Data/Edf")
        self.output_path = os.path.join(self.base_path, "Data/Output")

        os.makedirs(self.output_path, exist_ok=True)
        
    def run_script(self, script_path, script_name, current_file=None):
        """Esegue uno script Python"""
        print(f"\n{'='*60}")
        print(f"Eseguendo: {script_name}")
        if current_file:
            print(f"File corrente: {current_file}")
        print(f"{'='*60}")
        
        full_path = os.path.join(self.base_path, script_path)
        
        # Verifica che il file esista
        if not os.path.exists(full_path):
            print(f"❌ ERRORE: File non trovato: {full_path}")
            return False
        
        # Salva directory corrente
        original_cwd = os.getcwd()
        
        try:
            # Vai nella directory dello script
            if script_path.startswith('src/'):
                working_dir = self.base_path
                script_to_run = script_path
            else:
                working_dir = os.path.dirname(full_path)
                script_to_run = os.path.basename(full_path)
            
            os.chdir(working_dir)
            
            # Prepara variabili d'ambiente
            env = os.environ.copy()
            env['DATA_PATH'] = self.data_path
            env['OUTPUT_PATH'] = self.output_path
            env['BASE_PATH'] = self.base_path
            if current_file:
                env['CURRENT_FILE'] = current_file
            
            # Esegui lo script
            result = subprocess.run(
                [sys.executable, script_to_run],
                env=env,
                cwd=working_dir,
                capture_output=True,
                text=True
            )
            
            # Ripristina directory originale
            os.chdir(original_cwd)
            
            if result.stdout:
                print("📋 OUTPUT:")
                print(result.stdout)

            if result.stderr:
                print("⚠️ ERRORI:")
                print(result.stderr)

            if result.returncode == 0:
                print(f"✅ {script_name} completato")
                return True
            else:
                print(f"❌ {script_name} fallito (codice: {result.returncode})")
                return False
                
        except Exception as e:
            os.chdir(original_cwd)
            print(f"❌ Errore durante l'esecuzione: {str(e)}")
            return False
    
    def run_pipeline(self):
        """Esegue tutti gli script nell'ordine per ogni file EDF"""
        scripts = [
            ("src/letturaEDF.py", "Lettura EDF"),
            ("autoencoder/trasformazione.py", "Trasformazione Autoencoder"),
            ("autoencoder/autoencoder_psd.py", "Autoencoder PSD"),
            ("autoencoder/autoencoder_CI_psd.py", "Autoencoder CI PSD"),
            # ("autoencoder/autoencoder_CI_sovra_psd.py", "Autoencoder CI Sovra PSD"),
            # ("clustering/find_K.py", "Find K Clustering"),
            # ("clustering/clustering.py", "Clustering"),
            # ("clustering/clustering_no_ae.py", "Clustering No AE"),
            # ("clustering/k-means.py", "K-Means"),
            ("src/calcolo_acc.py", "Calcolo Accuratezza")
        ]
        
        print(f"🚀 Avvio pipeline ML - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Ottieni lista dei file EDF
        edf_files = [f for f in os.listdir(self.data_path) if f.lower().endswith('.edf')]
        
        if not edf_files:
            print("❌ Nessun file EDF trovato!")
            return
        
        total_successful = 0
        
        for edf_file in edf_files:
            print(f"\n🔄 Processando file: {edf_file}")
            
            successful = 0
            for script_path, script_name in scripts:
                success = self.run_script(script_path, script_name, edf_file)
                if success:
                    successful += 1
                else:
                    response = input(f"\n⚠️ {script_name} fallito per {edf_file}. Continuare con questo file? (y/n): ")
                    if response.lower() not in ['y', 'yes']:
                        break
            
            print(f"\n📊 File {edf_file} completato: {successful}/{len(scripts)} script eseguiti con successo")
            total_successful += successful
        
        print(f"\n🎉 Pipeline completata: {total_successful}/{len(scripts) * len(edf_files)} script totali eseguiti con successo")

def main():
    runner = MLPipelineRunner()
    runner.run_pipeline()

if __name__ == "__main__":
    main()