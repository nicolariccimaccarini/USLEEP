import subprocess
import os
import sys
from datetime import datetime


class MLPipelineRunner:
    def __init__(self):
        # self.base_path = "/hpc/groups/users-ai/EEG/ML-for-Spindle-Detection-in-EEESWAS/"
        self.base_path = "/mnt/c/Users/nicol/OneDrive/Documenti/GitHub/ML-for-Spindle-Detection-in-EEESWAS"
        self.data_path = os.environ.get('DATA_PATH', os.path.join(self.base_path, "Data/Preprocessed_Edf"))
        self.output_path = os.environ.get('OUTPUT_PATH', os.path.join(self.base_path, "Data/Output"))
        self.current_file = os.environ.get('CURRENT_FILE', None)
        self.pipeline_mode = os.environ.get('PIPELINE_MODE', 'full')  # 'full', 'training_only', 'detection_only'

        os.makedirs(self.output_path, exist_ok=True)
        
        # Verifica che la cartella utils esista
        utils_path = os.path.join(self.base_path, "utils")
        if not os.path.exists(utils_path):
            os.makedirs(utils_path, exist_ok=True)


    def get_output_folder_name(self, edf_file):
        """Genera il nome della cartella di output basato sul file EDF"""
        return os.path.splitext(edf_file)[0]


    def check_training_completed(self, edf_file):
        """Controlla se il training è già completato per un file"""
        output_folder = self.get_output_folder_name(edf_file)
        output_folder_path = os.path.join(self.output_path, output_folder)
        
        # Controlla se esistono i modelli addestrati
        model_paths = [
            os.path.join(output_folder_path, "model", "canali_individuali"),
            os.path.join(output_folder_path, "model", "canali_individuali_sovrapposti")
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                model_files = [f for f in os.listdir(model_path) if f.endswith('.h5')]
                if len(model_files) > 0:
                    return True
        
        return False


    def check_detection_completed(self, edf_file):
        """Controlla se la detection è già completata per un file"""
        output_folder = self.get_output_folder_name(edf_file)
        output_folder_path = os.path.join(self.output_path, output_folder)
        
        csv_path = os.path.join(output_folder_path, "cluster", "start_end_per_channel.csv")
        return os.path.exists(csv_path)


    def output_exists(self, edf_file):
        """Controlla se esiste già una cartella di output completa per il file EDF"""
        return self.check_training_completed(edf_file) and self.check_detection_completed(edf_file)


    def run_script(self, script_path, script_name, current_file=None, required_for_next=True):
        """Esegue uno script Python"""
        print(f"\n{'='*60}")
        print(f"🔄 Eseguendo: {script_name}")
        if current_file:
            print(f"📁 File corrente: {current_file}")
        print(f"{'='*60}")
        
        full_path = os.path.join(self.base_path, script_path)

        if not os.path.exists(full_path):
            print(f"❌ ERRORE: File non trovato: {full_path}")
            return False
        
        original_cwd = os.getcwd()
        
        try:
            # Determina la directory di lavoro
            if script_path.startswith('src/'):
                working_dir = self.base_path
                script_to_run = script_path
            else:
                working_dir = os.path.dirname(full_path)
                script_to_run = os.path.basename(full_path)
            
            os.chdir(working_dir)
            
            # Configura variabili d'ambiente
            env = os.environ.copy()
            env['DATA_PATH'] = self.data_path
            env['OUTPUT_PATH'] = self.output_path
            env['BASE_PATH'] = self.base_path
            env['PYTHONPATH'] = self.base_path + ':' + env.get('PYTHONPATH', '')
            
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
            
            os.chdir(original_cwd)
            
            # Gestione output
            if result.stdout:
                print("📋 OUTPUT:")
                print(result.stdout)

            if result.stderr:
                print("⚠️ STDERR:")
                print(result.stderr)

            if result.returncode == 0:
                print(f"✅ {script_name} completato con successo")
                return True
            else:
                print(f"❌ {script_name} fallito (codice: {result.returncode})")
                return False
                
        except Exception as e:
            os.chdir(original_cwd)
            print(f"❌ Errore durante l'esecuzione: {str(e)}")
            return False


    def get_pipeline_scripts(self):
        """Restituisce la lista degli script da eseguire in base alla modalità"""
        training_scripts = [
            ("autoencoder/autoencoder_CI_psd_refactored.py", "Autoencoder PSD Refactored", True),
            # ("autoencoder/autoencoder_CI_sovra_psd_refactored.py", "Autoencoder PSD con Alta Sovrapposizione", False)
        ]
        
        detection_scripts = [
<<<<<<< HEAD
            # ("clustering/clustering_with_spindle_detection.py", "Clustering e Rilevamento Spindles", True)
            ("clustering/binary_clustering_no_th.py", "Clustering Binario senza Threshold", True)
            # ("clustering/hybrid_clustering_amplitude.py", "Clustering ibrido che combina calcolo ampiezza e distanza centroidi tramite percentile", True)
=======
            ("clustering/clustering_with_spindle_detection.py", "Clustering e Rilevamento Spindles", True)
            ("clustering/binary_clustering_no_th.py", "Clustering Binario senza Threshold", True)
            ("clustering/hybrid_clustering_amplitude.py", "Clustering ibrido che combina calcolo ampiezza e distanza centroidi tramite percentile", True)
>>>>>>> 8f176d2 (changes)
        ]
        
        if self.pipeline_mode == 'training_only':
            return training_scripts
        elif self.pipeline_mode == 'detection_only':
            return detection_scripts
        else:  # 'full'
            return training_scripts + detection_scripts


    def should_skip_file(self, edf_file):
        """Determina se un file deve essere saltato basandosi sulla modalità e sullo stato completamento"""
        if self.pipeline_mode == 'full':
            return self.output_exists(edf_file)
        elif self.pipeline_mode == 'training_only':
            return self.check_training_completed(edf_file)
        elif self.pipeline_mode == 'detection_only':
            return self.check_detection_completed(edf_file)
        return False


    def validate_prerequisites(self, edf_file):
        """Valida che i prerequisiti per la detection siano soddisfatti"""
        if self.pipeline_mode == 'detection_only':
            if not self.check_training_completed(edf_file):
                print(f"❌ Training non completato per {edf_file}. Detection non possibile.")
                return False
        return True


    def run_pipeline(self):
        """Esegue il pipeline completo"""
        scripts = self.get_pipeline_scripts()
        
        print(f"🚀 Avvio pipeline ML - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 Modalità: {self.pipeline_mode}")
        print(f"📂 Path dati: {self.data_path}")
        print(f"📁 Path output: {self.output_path}")
        
        # Determina i file da processare
        if self.current_file:
            edf_files = [self.current_file]
            print(f"📋 Modalità job array - processando: {self.current_file}")
        else:
            try:
                edf_files = [f for f in os.listdir(self.data_path) if f.lower().endswith('.edf')]
                print(f"📋 Modalità batch - {len(edf_files)} file trovati")
            except FileNotFoundError:
                print(f"❌ Directory dati non trovata: {self.data_path}")
                return
        
        if not edf_files:
            print("❌ Nessun file EDF trovato!")
            return
        
        # Statistiche
        total_successful = 0
        files_processed = 0
        files_skipped = 0
        files_failed = 0
        
        for edf_file in edf_files:
            print(f"\n{'='*80}")
            print(f"📊 File {edf_files.index(edf_file)+1}/{len(edf_files)}: {edf_file}")
            print(f"{'='*80}")
            
            # Controlla se saltare il file
            if self.should_skip_file(edf_file):
                status_msg = {
                    'full': 'già completamente processato',
                    'training_only': 'training già completato',
                    'detection_only': 'detection già completata'
                }
                print(f"⏭️ File {edf_file} {status_msg.get(self.pipeline_mode, 'già processato')}. Saltando...")
                files_skipped += 1
                continue
            
            # Valida prerequisiti
            if not self.validate_prerequisites(edf_file):
                files_failed += 1
                continue
            
            print(f"🔄 Processando file: {edf_file}")
            files_processed += 1
            
            # Esegui script per il file corrente
            file_successful = 0
            file_failed = False
            
            for script_path, script_name, required in scripts:
                success = self.run_script(script_path, script_name, edf_file, required)
                
                if success:
                    file_successful += 1
                else:
                    if required:
                        print(f"❌ Script obbligatorio {script_name} fallito per {edf_file}")
                        file_failed = True
                        
                        # Solo in modalità interattiva chiedi conferma
                        if not self.current_file:
                            response = input(f"⚠️ Continuare con il prossimo file? (y/n): ")
                            if response.lower() not in ['y', 'yes']:
                                print("🛑 Pipeline interrotta dall'utente")
                                return
                        break
                    else:
                        print(f"⚠️ Script opzionale {script_name} fallito per {edf_file}, continuando...")
            
            if file_failed:
                files_failed += 1
            else:
                total_successful += file_successful
            
            # Riepilogo del file
            status = "✅ SUCCESSO" if not file_failed else "❌ FALLITO"
            print(f"\n📊 {status} - File {edf_file}: {file_successful}/{len(scripts)} script completati")
        
        # Riepilogo finale
        print(f"\n{'='*80}")
        print(f"🎉 Pipeline {self.pipeline_mode} completata - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"📁 File totali trovati: {len(edf_files)}")
        print(f"🔄 File processati: {files_processed}")
        print(f"⏭️ File saltati (già processati): {files_skipped}")
        print(f"❌ File falliti: {files_failed}")
        print(f"✅ Script eseguiti con successo: {total_successful}")
        
        # Mostra statistiche dettagliate
        if files_processed > 0:
            success_rate = (files_processed - files_failed) / files_processed * 100
            print(f"📈 Tasso di successo: {success_rate:.1f}%")


def main():
    """Punto di ingresso principale"""
    # Mostra informazioni sulla configurazione
    print("🔧 Configurazione Pipeline:")
    print(f"   PIPELINE_MODE: {os.environ.get('PIPELINE_MODE', 'full')}")
    print(f"   DATA_PATH: {os.environ.get('DATA_PATH', 'default')}")
    print(f"   OUTPUT_PATH: {os.environ.get('OUTPUT_PATH', 'default')}")
    print(f"   CURRENT_FILE: {os.environ.get('CURRENT_FILE', 'None (batch mode)')}")
    
    runner = MLPipelineRunner()
    
    try:
        runner.run_pipeline()
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrotta dall'utente")
    except Exception as e:
        print(f"\n❌ Errore fatale nel pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
