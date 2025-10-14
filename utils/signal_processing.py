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

def apply_smoothing(scores, window_size=5, method='moving_average'):
    """
    Applica smoothing ai punteggi
    
    Args:
        scores: array di punteggi
        window_size: dimensione della finestra per lo smoothing
        method: 'moving_average' o 'gaussian'
    
    Returns:
        array di punteggi smussati
    """
    if method == 'moving_average':
        smoothed = np.convolve(scores, np.ones(window_size)/window_size, mode='same')
    elif method == 'gaussian':
        from scipy import ndimage
        smoothed = ndimage.gaussian_filter1d(scores, sigma=window_size/3)
    else:
        smoothed = scores
    
    return smoothed

def detect_spindle_regions(scores, threshold=0.5, min_duration_samples=10):
    """
    Rileva regioni contigue sopra la soglia che rappresentano spindles
    
    Args:
        scores: array di punteggi per ogni finestra
        threshold: soglia per considerare una finestra come spindle
        min_duration_samples: durata minima in campioni per considerare una regione valida
    
    Returns:
        list di tuple (start_idx, end_idx) per ogni spindle rilevato
    """
    above_threshold = scores > threshold
    regions = []
    
    if not np.any(above_threshold):
        return regions
    
    # Trova inizio e fine delle regioni contigue
    diff = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # Filtra per durata minima
    for start, end in zip(starts, ends):
        if end - start >= min_duration_samples:
            regions.append((start, end))
    
    return regions

def convert_regions_to_time(regions, segment_length, overlap_ratio, sfreq):
    """
    Converte gli indici delle regioni in tempi di inizio e fine
    
    Args:
        regions: list di tuple (start_idx, end_idx)
        segment_length: lunghezza del segmento in campioni
        overlap_ratio: percentuale di sovrapposizione
        sfreq: frequenza di campionamento
    
    Returns:
        list di tuple (start_time, end_time) in secondi
    """
    step = int(segment_length * (1 - overlap_ratio))
    time_regions = []
    
    for start_idx, end_idx in regions:
        start_time = start_idx * step / sfreq
        end_time = (end_idx * step + segment_length) / sfreq
        time_regions.append((start_time, end_time))
    
    return time_regions