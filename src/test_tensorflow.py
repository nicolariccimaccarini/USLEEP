import os
import subprocess
import sys

# Configura TensorFlow PRIMA dell'import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Mostra più info per debug
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print("🔍 Verifica sistema per GPU...")

# Verifica NVIDIA driver
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Driver NVIDIA installati")
        print("GPU info:")
        lines = result.stdout.split('\n')
        for line in lines:
            if 'GeForce' in line or 'Quadro' in line or 'Tesla' in line or 'RTX' in line:
                print(f"   {line.strip()}")
    else:
        print("❌ nvidia-smi non funziona - driver non installati?")
except FileNotFoundError:
    print("❌ nvidia-smi non trovato - driver NVIDIA non installati")

# Verifica CUDA
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ CUDA Toolkit installato")
        version_line = [line for line in result.stdout.split('\n') if 'release' in line]
        if version_line:
            print(f"   {version_line[0].strip()}")
    else:
        print("❌ CUDA Toolkit non installato o non nel PATH")
except FileNotFoundError:
    print("❌ nvcc non trovato - CUDA Toolkit non installato")

print("\n🧪 Test TensorFlow...")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} importato")
    
    # Info dettagliate su dispositivi
    print("\n📊 Dispositivi fisici:")
    physical_devices = tf.config.list_physical_devices()
    for device in physical_devices:
        print(f"   {device}")
    
    # Test specifico GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n🎮 GPU TensorFlow: {len(gpus)} trovate")
        
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
            try:
                # Configura crescita memoria
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Info dettagliate
                details = tf.config.experimental.get_device_details(gpu)
                print(f"     Dettagli: {details}")
                
            except Exception as e:
                print(f"     ⚠️ Errore configurazione: {e}")
        
        # Test operazione su GPU
        print("\n🧪 Test operazione su GPU...")
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"✅ Moltiplicazione su GPU: {c.numpy().tolist()}")
                
                # Test più impegnativo
                big_matrix = tf.random.normal([1000, 1000])
                result = tf.linalg.matmul(big_matrix, big_matrix)
                print(f"✅ Moltiplicazione matrice 1000x1000 completata")
                
        except Exception as e:
            print(f"❌ Errore operazione GPU: {e}")
            
    else:
        print("❌ Nessuna GPU rilevata da TensorFlow")
        print("\n💡 Possibili cause:")
        print("   - Driver NVIDIA non installati o non compatibili")
        print("   - CUDA Toolkit non installato o versione non compatibile")
        print("   - cuDNN non installato")
        print("   - TensorFlow installato senza supporto GPU")
    
    # Mostra librerie caricate
    print(f"\n📚 Build info TensorFlow:")
    print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"   CUDA version: {tf.sysconfig.get_build_info().get('cuda_version', 'N/A')}")
    print(f"   cuDNN version: {tf.sysconfig.get_build_info().get('cudnn_version', 'N/A')}")
    
    # Test se possiamo creare tensori GPU
    if gpus:
        try:
            with tf.device('/GPU:0'):
                test_tensor = tf.ones([100, 100])
                print(f"✅ Tensor creato su GPU: {test_tensor.device}")
        except:
            print("❌ Impossibile creare tensor su GPU")
    
except ImportError as e:
    print(f"❌ Errore import TensorFlow: {e}")
except Exception as e:
    print(f"❌ Errore TensorFlow: {e}")

print("\n" + "="*60)
print("SOLUZIONI se GPU non funziona:")
print("="*60)
print("1. WSL2: Installa driver NVIDIA Windows + CUDA WSL")
print("2. Linux: sudo apt install nvidia-driver-XXX cuda-toolkit-11-8")
print("3. Reinstalla TF: pip uninstall tensorflow && pip install tensorflow==2.20.0")
print("4. Verifica compatibilità: TF 2.20 richiede CUDA 11.8 + cuDNN 8.6+")
print("5. Se non hai GPU: usa CPU, è comunque veloce per molti task")