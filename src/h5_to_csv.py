import h5py
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path


def h5_to_csv(h5_file_path, output_dir=None):
    """
    Converte un file .h5 in uno o più file .csv
    
    Args:
        h5_file_path (str): Percorso del file .h5
        output_dir (str): Directory di output (opzionale)
    """
    h5_path = Path(h5_file_path)
    
    if not h5_path.exists():
        print(f"Errore: Il file {h5_file_path} non esiste")
        return
    
    if output_dir is None:
        output_dir = h5_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = h5_path.stem
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            print(f"Conversione di {h5_file_path}...")
            
            def save_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  Dataset trovato: {name}")
                    
                    data = obj[:]
                    
                    if data.ndim == 1:
                        df = pd.DataFrame(data, columns=[name.split('/')[-1]])
                    elif data.ndim == 2:
                        df = pd.DataFrame(data)
                    else:
                        data_flat = data.flatten()
                        df = pd.DataFrame(data_flat, columns=[f"{name.split('/')[-1]}_flattened"])
                    
                    dataset_name = name.replace('/', '_').strip('_')
                    csv_filename = f"{base_name}_{dataset_name}.csv"
                    csv_path = output_dir / csv_filename
                    
                    df.to_csv(csv_path, index=False)
                    print(f"    Salvato: {csv_path}")
            
            f.visititems(save_dataset)
            
    except Exception as e:
        print(f"Errore durante la conversione: {e}")


def convert_multiple_h5_files(directory_path, output_dir=None):
    """
    Converte tutti i file .h5 in una directory
    
    Args:
        directory_path (str): Percorso della directory contenente i file .h5
        output_dir (str): Directory di output (opzionale)
    """
    dir_path = Path(directory_path)
    h5_files = list(dir_path.glob("*.h5"))
    
    if not h5_files:
        print(f"Nessun file .h5 trovato in {directory_path}")
        return
    
    print(f"Trovati {len(h5_files)} file .h5")
    
    for h5_file in h5_files:
        h5_to_csv(h5_file, output_dir)


def main():
    """Funzione principale con interfaccia da riga di comando"""
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python h5_to_csv.py <file.h5> [output_directory]")
        print("  python h5_to_csv.py <directory> [output_directory]")
        return
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    input_path = Path(input_path)
    
    if input_path.is_file() and input_path.suffix == '.h5':
        h5_to_csv(input_path, output_dir)
    elif input_path.is_dir():
        convert_multiple_h5_files(input_path, output_dir)
    else:
        print("Errore: Specificare un file .h5 valido o una directory")


if __name__ == "__main__":
    main()