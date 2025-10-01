import subprocess
import os
import sys
from datetime import datetime


class MLPipelineRunner:
    def __init__(self):
        self.base_path = "/hpc/groups/users-ai/EEG/ML-for-Spindle-Detection-in-EEESWAS/"
        # self.base_path = "/mnt/c/Users/nicol/OneDrive/Documenti/GitHub/ML-for-Spindle-Detection-in-EEESWAS"
        self.data_path = os.environ.get('DATA_PATH', os.path.join(self.base_path, "Data/Edf"))
        self.output_path = os.environ.get('OUTPUT_PATH', os.path.join(self.base_path, "Data/Output"))
        self.current_file = os.environ.get('CURRENT_FILE', None)

        os.makedirs(self.output_path, exist_ok=True)


    def get_output_folder_name(self, edf_file):
        """Genera il nome della cartella di output basato sul file EDF"""
        return os.path.splitext(edf_file)[0]


    def output_exists(self, edf_file):
        """Controlla se esiste già una cartella di output per il file EDF"""
        output_folder = self.get_output_folder_name(edf_file)
        output_folder_path = os.path.join(self.output_path, output_folder)
        return os.path.exists(output_folder_path) and os.path.isdir(output_folder_path)


    def run_script(self, script_path, script_name, current_file=None):
        """Esegue uno script Python"""
        print(f"\n{'='*60}")
        print(f"Eseguendo: {script_name}")
        if current_file:
            print(f"File corrente: {current_file}")
        print(f"{'='*60}")
        
        full_path = os.path.join(self.base_path, script_path)

        if not os.path.exists(full_path):
            print(f"❌ ERRORE: File non trovato: {full_path}")
            return False
        
        original_cwd = os.getcwd()
        
        try:
            if script_path.startswith('src/'):
                working_dir = self.base_path
                script_to_run = script_path
            else:
                working_dir = os.path.dirname(full_path)
                script_to_run = os.path.basename(full_path)
            
            os.chdir(working_dir)
            
            env = os.environ.copy()
            env['DATA_PATH'] = self.data_path
            env['OUTPUT_PATH'] = self.output_path
            env['BASE_PATH'] = self.base_path
            if current_file:
                env['CURRENT_FILE'] = current_file
            
            result = subprocess.run(
                [sys.executable, script_to_run],
                env=env,
                cwd=working_dir,
                capture_output=True,
                text=True
            )
            
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
        scripts = [
            ("autoencoder/autoencoder_CI_psd.py", "Autoencoder CI PSD"),
            ("autoencoder/autoencoder_CI_sovra_psd.py", "Autoencoder CI Sovra PSD")
        ]
        
        print(f"🚀 Avvio pipeline ML - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        edf_files = [f for f in os.listdir(self.data_path) if f.lower().endswith('.edf')]
        
        if not edf_files:
            print("❌ Nessun file EDF trovato!")
            return
        
        total_successful = 0
        files_processed = 0
        files_skipped = 0
        
        for edf_file in edf_files:
            if self.output_exists(edf_file):
                output_folder = self.get_output_folder_name(edf_file)
                print(f"\n⏭️ File {edf_file} già processato (cartella {output_folder} esistente). Saltando...")
                files_skipped += 1
                continue
            
            print(f"\n🔄 Processando file: {edf_file}")
            files_processed += 1
            
            successful = 0
            for script_path, script_name in scripts:
                success = self.run_script(script_path, script_name, edf_file)
                if success:
                    successful += 1
                else:
                    # Solo in modalità interattiva
                    if not self.current_file:
                        response = input(f"\n⚠️ {script_name} fallito per {edf_file}. Continuare? (y/n): ")
                        if response.lower() not in ['y', 'yes']:
                            break
                    else:
                        # In job array, interrompi se fallisce
                        print(f"❌ {script_name} fallito per {edf_file}. Interruzione.")
                        break
            
            print(f"\n📊 File {edf_file} completato: {successful}/{len(scripts)} script eseguiti con successo")
            total_successful += successful
        
        print(f"\n🎉 Pipeline completata:")
        print(f"   📁 File processati: {files_processed}")
        print(f"   ⏭️ File saltati (già processati): {files_skipped}")
        print(f"   ✅ Script eseguiti con successo: {total_successful}/{len(scripts) * files_processed}")


def main():
    runner = MLPipelineRunner()
    runner.run_pipeline()


if __name__ == "__main__":
    main()
