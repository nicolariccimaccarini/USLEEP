import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Lambda
from tensorflow.keras.models import Model
import numpy as np

def create_autoencoder_architecture():
    """Ricreare l'architettura dell'autoencoder dal tuo modello originale"""
    
    # Input layer
    input_layer = Input(shape=(26, 1920, 1), name='input_layer')
    
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1')(x)
    x = MaxPooling2D((2, 2), padding='same', name='max_pooling2d')(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_3')(x)
    x = MaxPooling2D((2, 2), padding='same', name='max_pooling2d_1')(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_4')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_5')(x)
    x = MaxPooling2D((2, 2), padding='same', name='max_pooling2d_2')(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_6')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_7')(x)
    x = MaxPooling2D((2, 2), padding='same', name='max_pooling2d_3')(x)
    
    # Bottleneck
    encoded = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_8')(x)
    
    # Decoder
    x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='conv2d_transpose')(encoded)
    x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', name='conv2d_transpose_1')(x)
    x = UpSampling2D((3, 1), name='up_sampling2d')(x)
    
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', name='conv2d_transpose_2')(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', name='conv2d_transpose_3')(x)
    x = UpSampling2D((2, 2), name='up_sampling2d_1')(x)
    
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='conv2d_transpose_4')(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', name='conv2d_transpose_5')(x)
    x = UpSampling2D((2, 2), name='up_sampling2d_2')(x)
    
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', name='conv2d_transpose_6')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', name='conv2d_transpose_7')(x)
    x = UpSampling2D((2, 2), name='up_sampling2d_3')(x)
    
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='conv2d_9')(x)
    
    # Layer Lambda per il padding - sostituito con cropping manuale
    def crop_to_original_size(x):
        return x[:, :26, :1920, :]
    
    output = Lambda(crop_to_original_size, name='lambda')(decoded)
    
    autoencoder = Model(input_layer, output)
    return autoencoder

def load_weights_from_h5(model, h5_path):
    """Carica i pesi dal file H5 esistente"""
    try:
        model.load_weights(h5_path)
        print("✅ Pesi caricati con successo")
        return True
    except Exception as e:
        print(f"❌ Errore nel caricamento dei pesi: {e}")
        return False

# Test dello script
if __name__ == "__main__":
    print("🔧 Ricostruzione dell'autoencoder...")
    
    # Crea il modello
    autoencoder = create_autoencoder_architecture()
    
    # Prova a caricare i pesi
    h5_path = 'Data/weights/autoencoder_model.h5'
    success = load_weights_from_h5(autoencoder, h5_path)
    
    if success:
        print("✅ Autoencoder ricostruito con successo!")
        autoencoder.summary()
        
        # Salva il modello ricostruito
        new_path = 'Data/weights/autoencoder_model_rebuilt.h5'
        autoencoder.save(new_path, save_format='h5')
        print(f"💾 Modello ricostruito salvato in: {new_path}")
    else:
        print("❌ Fallimento nella ricostruzione")