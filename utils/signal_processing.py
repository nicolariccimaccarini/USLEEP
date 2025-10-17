import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def get_file_output_path(base_data_path, filename=None):
    """Crea e restituisce il percorso per l'output specifico del file"""
    if filename:
        file_base_name = os.path.splitext(filename)[0]
        file_output_path = os.path.join(base_data_path, file_base_name)
        os.makedirs(os.path.join(file_output_path, "images", "canali_individuali"), exist_ok=True)
        os.makedirs(os.path.join(file_output_path, "model", "canali_individuali"), exist_ok=True)
        os.makedirs(os.path.join(file_output_path, "cluster"), exist_ok=True)
        return file_output_path
    else:
        return base_data_path

def pad_last_segment(segment, window_size):
    """Applica padding con zeri all'ultimo segmento se necessario"""
    if segment.shape[1] < window_size:
        return np.pad(segment, ((0, 0), (0, window_size - segment.shape[1])), mode='constant')
    return segment

def segment_signal_with_overlap(data, segment_length, overlap_ratio):
    """
    Segmenta il segnale con sovrapposizione usando ratio di overlap
    
    Args:
        data: array numpy (n_channels, n_samples)
        segment_length: lunghezza del segmento in campioni
        overlap_ratio: percentuale di sovrapposizione (0.0-1.0)
    
    Returns:
        list di segmenti
    """
    step = int(segment_length * (1 - overlap_ratio))
    segments = []
    
    for start in range(0, data.shape[1] - segment_length + 1, step):
        segment = data[:, start:start + segment_length]
        segments.append(segment)
    
    # Gestisce l'ultimo segmento se necessario
    if data.shape[1] % step != 0:
        last_start = data.shape[1] - segment_length
        if last_start > 0 and last_start not in range(0, data.shape[1] - segment_length + 1, step):
            last_segment = data[:, last_start:]
            last_segment = pad_last_segment(last_segment, segment_length)
            segments.append(last_segment)
    
    return segments

def compute_spectrum_numpy(segments, freq_sample):
    """Calcola lo spettro di potenza per ogni segmento"""
    spectrums = []
    segment_length = segments[0].shape[1]
    frequencies = np.fft.fftfreq(segment_length, d=1/freq_sample)
    frequencies = frequencies[:segment_length // 2]
    
    for segment in segments:
        segment_spectrum = np.abs(np.fft.fft(segment, axis=1))
        segment_spectrum = segment_spectrum[:, :segment_length // 2]
        spectrums.append(segment_spectrum)
    
    return spectrums, frequencies

def normalize_spectrum(spectrum):
    """Normalizza lo spettro usando MinMaxScaler"""
    scaler = MinMaxScaler()
    normalized_spectrum = []
    for channel_spectrum in spectrum:
        channel_spectrum = channel_spectrum.reshape(-1, 1)
        normalized_channel = scaler.fit_transform(channel_spectrum).flatten()
        normalized_spectrum.append(normalized_channel)
    return np.array(normalized_spectrum)

def apply_smoothing(signal, window_size, method='moving_average'):
    """
    Applica smoothing al segnale
    
    Args:
        signal: segnale da smussare
        window_size: dimensione finestra (in campioni)
        method: 'moving_average', 'gaussian', 'savgol'
    
    Returns:
        segnale smussato
    """
    if len(signal) == 0:
        return signal
    
    # Assicurati che window_size sia valido
    window_size = max(1, int(window_size))
    window_size = min(window_size, len(signal))
    
    if window_size <= 1:
        return signal
        
    if method == 'moving_average':
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(signal.astype(float), size=window_size, mode='nearest')
    
    elif method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        sigma = max(0.5, window_size / 4)  # Evita sigma troppo piccolo
        return gaussian_filter1d(signal.astype(float), sigma=sigma, mode='nearest')
    
    elif method == 'savgol':
        from scipy.signal import savgol_filter
        # Savgol richiede finestra dispari e almeno 3 campioni
        if window_size % 2 == 0:
            window_size += 1
        window_size = max(3, window_size)
        window_size = min(window_size, len(signal))
        
        # Se la finestra è ancora troppo grande, usa il massimo possibile
        if window_size >= len(signal):
            window_size = len(signal) - 1 if len(signal) > 1 else 1
            if window_size % 2 == 0:
                window_size -= 1
        
        polyorder = min(2, window_size - 1)
        if polyorder < 1:
            polyorder = 1
            
        return savgol_filter(signal, window_size, polyorder)
    
    else:
        return signal

def detect_spindle_regions(signal, threshold, min_duration_samples):
    """
    Rileva regioni continue sopra la soglia
    
    Args:
        signal: segnale binario o continuo
        threshold: soglia per rilevamento
        min_duration_samples: durata minima in campioni
    
    Returns:
        lista di tuple (start_idx, end_idx)
    """
    binary_signal = signal >= threshold
    
    # Trova transizioni
    diff = np.diff(np.concatenate(([False], binary_signal, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # Filtra per durata minima
    regions = []
    for start, end in zip(starts, ends):
        if end - start >= min_duration_samples:
            regions.append((start, end))
    
    return regions

def convert_regions_to_time(regions, segment_length, overlap_ratio, sfreq):
    """
    Converte regioni da indici di segmenti a tempi in secondi
    
    Args:
        regions: lista di tuple (start_idx, end_idx) in indici di segmenti
        segment_length: lunghezza segmento in campioni
        overlap_ratio: rapporto di sovrapposizione
        sfreq: frequenza di campionamento
    
    Returns:
        lista di tuple (start_time, end_time) in secondi
    """
    step_samples = int(segment_length * (1 - overlap_ratio))
    step_time = step_samples / sfreq
    
    time_regions = []
    for start_idx, end_idx in regions:
        start_time = start_idx * step_time
        end_time = end_idx * step_time
        time_regions.append((start_time, end_time))
    
    return time_regions