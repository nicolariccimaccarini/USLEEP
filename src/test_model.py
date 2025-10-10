import h5py
import tensorflow as tf
import json

print(f"Versione TensorFlow: {tf.__version__}")

try:
    with h5py.File('Data/weights/autoencoder_model.h5', 'r') as f:
        print("Il file è integro.")
        
        # Verifica la struttura del file
        print("\nStruttura del file H5:")
        def print_structure(name, obj):
            print(name)
        f.visititems(print_structure)
        
        # Prova a leggere la configurazione del modello
        if 'model_config' in f.attrs:
            config = f.attrs['model_config']
            if isinstance(config, bytes):
                config = config.decode('utf-8')
            print(f"\nConfigurazione modello trovata: {len(config)} caratteri")
            
            # Prova a parsare la configurazione JSON
            try:
                config_dict = json.loads(config)
                print("Configurazione JSON valida")
                print(f"Classe modello: {config_dict.get('class_name', 'N/A')}")
            except json.JSONDecodeError as e:
                print(f"Errore nel parsing JSON: {e}")
        
except Exception as e:
    print(f"Errore nell'aprire il file: {e}")

# Test di caricamento con diverse opzioni
print("\n--- Test caricamento modello ---")

try:
    print("Tentativo 1: Caricamento standard...")
    model = tf.keras.models.load_model('Data/weights/autoencoder_model.h5')
    print("✅ Caricamento riuscito!")
except Exception as e:
    print(f"❌ Fallito: {e}")

try:
    print("\nTentativo 2: Caricamento senza safe_mode...")
    model = tf.keras.models.load_model('Data/weights/autoencoder_model.h5', safe_mode=False)
    print("✅ Caricamento riuscito!")
except Exception as e:
    print(f"❌ Fallito: {e}")

try:
    print("\nTentativo 3: Caricamento con compile=False...")
    model = tf.keras.models.load_model('Data/weights/autoencoder_model.h5', compile=False)
    print("✅ Caricamento riuscito!")
except Exception as e:
    print(f"❌ Fallito: {e}")