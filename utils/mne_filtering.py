import mne
import sys
import os
import argparse

# Add utils to path
sys.path.append(os.path.dirname(__file__))
from signal_processing import mne_bandpass_filter, crop_and_save_edf


def main():
    parser = argparse.ArgumentParser(description='Filtra e croppa file EDF con MNE')
    parser.add_argument('--input', required=True, help='File EDF di input')
    parser.add_argument('--output', required=True, help='File EDF di output')
    # parser.add_argument('--lowcut', type=float, default=9, help='Freq. taglio bassa (Hz)')
    # parser.add_argument('--highcut', type=float, default=15, help='Freq. taglio alta (Hz)')
    parser.add_argument('--tmin', type=float, default=None, help='Tempo inizio crop (s)')
    parser.add_argument('--tmax', type=float, default=None, help='Tempo fine crop (s)')
    
    args = parser.parse_args()
    
    print(f"🔧 Configurazione:")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    # print(f"   Bandpass: {args.lowcut}-{args.highcut} Hz")
    if args.tmin is not None and args.tmax is not None:
        print(f"   Crop: {args.tmin}-{args.tmax} s")
    
    # Carica
    raw = mne.io.read_raw_edf(args.input, preload=True, verbose=False)
    print(f"\n✅ File caricato: durata {raw.times[-1]:.2f} s")
    
    # Filtra
    # print(f"🔊 Applicando bandpass filter...")
    # mne_bandpass_filter(raw, lowcut=args.lowcut, highcut=args.highcut)
    
    # Croppa e salva
    if args.tmin is not None and args.tmax is not None:
        print(f"✂️ Croppando segnale e salvando...")
        crop_and_save_edf(raw, args.output, tmin=args.tmin, tmax=args.tmax)
    else:
        # Salva senza cropping
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        raw.export(args.output, overwrite=True)
    
    print(f"✅ Salvato: {args.output}")


if __name__ == "__main__":
    main()
