import h5py
import tensorflow as tf
import json

print(f"Versione TensorFlow: {tf.__version__}")

tf.keras.config.enable_unsafe_deserialization()

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
print("--- Test caricamento modello con deserializzazione unsafe abilitata ---")

try:
    print("Tentativo 1: Caricamento con unsafe deserialization abilitata...")
    model = tf.keras.models.load_model('Data/weights/autoencoder_model.h5', safe_mode=False)
    print("✅ Caricamento riuscito!")
    print(f"Modello caricato: {model.__class__.__name__}")
except Exception as e:
    print(f"❌ Fallito: {e}")

try:
    print("\nTentativo 2: Caricamento con compile=False e unsafe deserialization...")
    model = tf.keras.models.load_model('Data/weights/autoencoder_model.h5', compile=False, safe_mode=False)
    print("✅ Caricamento riuscito!")
    print(f"Modello caricato: {model.__class__.__name__}")
except Exception as e:
    print(f"❌ Fallito: {e}")

print("\n--- Verifica configurazione ---")
print(f"Unsafe deserialization abilitata: {tf.keras.config.get_enable_unsafe_deserialization()}")