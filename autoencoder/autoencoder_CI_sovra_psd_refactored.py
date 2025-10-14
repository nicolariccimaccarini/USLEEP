import os
import sys

# Aggiungi il percorso del modulo principale
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Importa e usa la versione refactored con parametri di overlap diversi
from autoencoder_CI_psd_refactored import process_edf_files, CONFIG

# Modifica la configurazione per sovrapposizione maggiore
CONFIG['overlap_ratio'] = 0.9  # 90% di sovrapposizione
CONFIG['window_size'] = 0.5    # Finestra ancora più piccola

def main():
    """Wrapper per eseguire con parametri di sovrapposizione elevata"""
    print("🔄 Esecuzione autoencoder con alta sovrapposizione...")
    print(f"📐 Finestra: {CONFIG['window_size']}s, Overlap: {CONFIG['overlap_ratio']*100}%")
    
    # Modifica i percorsi di output per distinguere questa versione
    original_get_file_output_path = sys.modules['signal_processing'].get_file_output_path
    
    def modified_get_file_output_path(base_data_path, filename=None):
        if filename:
            file_base_name = os.path.splitext(filename)[0]
            file_output_path = os.path.join(base_data_path, file_base_name)
            os.makedirs(os.path.join(file_output_path, "images", "canali_individuali_sovrapposti"), exist_ok=True)
            os.makedirs(os.path.join(file_output_path, "model", "canali_individuali_sovrapposti"), exist_ok=True)
            os.makedirs(os.path.join(file_output_path, "cluster"), exist_ok=True)
            return file_output_path
        else:
            return base_data_path
    
    # Sostituisci temporaneamente la funzione
    sys.modules['signal_processing'].get_file_output_path = modified_get_file_output_path
    
    try:
        process_edf_files()
    finally:
        # Ripristina la funzione originale
        sys.modules['signal_processing'].get_file_output_path = original_get_file_output_path

if __name__ == "__main__":
    main()