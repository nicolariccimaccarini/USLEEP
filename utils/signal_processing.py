import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt, find_peaks

def apply_sigma_band_filter(data, sfreq, low_freq=9, high_freq=15):
    """
    Applica filtro passa-banda per isolare la banda sigma (9-15 Hz)
    
    Args:
        data: segnale EEG (n_channels, n_samples)
        sfreq: frequenza di campionamento
        low_freq: frequenza minima banda sigma
        high_freq: frequenza massima banda sigma
    
    Returns:
        segnale filtrato nella banda sigma
    """
    # Progetta filtro Butterworth passa-banda
    nyquist = sfreq / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Assicurati che le frequenze normalizzate siano valide
    if low >= 1.0 or high >= 1.0:
        raise ValueError(f"Frequenze troppo alte per sfreq={sfreq}. Sigma band: {low_freq}-{high_freq} Hz")
    
    b, a = butter(4, [low, high], btype='band')
    
    # Applica il filtro a ogni canale
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered_data[ch] = filtfilt(b, a, data[ch])
    
    return filtered_data

def compute_sigma_power_spectrum(segments, freq_sample):
    """
    Calcola la potenza spettrale focalizzata sulla banda sigma con features piu' discriminative
    
    Args:
        segments: lista di segmenti EEG
        freq_sample: frequenza di campionamento
    
    Returns:
        potenze sigma, frequenze sigma
    """
    sigma_powers = []
    sigma_low, sigma_high = 9, 15
    
    for segment in segments:
        # FFT per ogni canale del segmento
        segment_fft = np.fft.fft(segment, axis=1)
        freqs = np.fft.fftfreq(segment.shape[1], d=1/freq_sample)
        
        # Banda sigma
        sigma_mask = (freqs >= sigma_low) & (freqs <= sigma_high)
        sigma_fft = segment_fft[:, sigma_mask]
        sigma_power = np.abs(sigma_fft) ** 2
        
        for ch in range(segment.shape[0]):
            channel_sigma_power = sigma_power[ch, :]
            
            mean_power = np.mean(channel_sigma_power)
            std_power = np.std(channel_sigma_power)
            max_power = np.max(channel_sigma_power)
            
            # Features 
            power_ratio = max_power / (mean_power + 1e-10)  # Rapporto picco/media
            spectral_centroid = np.sum(freqs[sigma_mask] * channel_sigma_power) / (np.sum(channel_sigma_power) + 1e-10)
            spectral_bandwidth = np.sqrt(np.sum(((freqs[sigma_mask] - spectral_centroid) ** 2) * channel_sigma_power) / (np.sum(channel_sigma_power) + 1e-10))
            
            # Peak detection 
            peaks, _ = find_peaks(channel_sigma_power, height=np.percentile(channel_sigma_power, 75))
            peak_count = len(peaks)
            
            # Energia relativa nella banda centrale sigma (11-13 Hz)
            central_mask = (freqs >= 11) & (freqs <= 13)
            central_energy = np.sum(np.abs(segment_fft[ch, central_mask]) ** 2)
            total_energy = np.sum(channel_sigma_power)
            central_ratio = central_energy / (total_energy + 1e-10)
            
            features = [
                np.log1p(mean_power),        # Log per stabilizzare
                np.log1p(std_power),         # Log per stabilizzare  
                np.log1p(power_ratio),       # Rapporto discriminativo
                spectral_centroid,           # Frequenza centrale
                spectral_bandwidth,          # Larghezza spettrale
                peak_count,                  # Numero di picchi
                central_ratio                # Energia banda centrale
            ]
            
            sigma_powers.append(np.array(features))
    
    return sigma_powers, freqs[sigma_mask]

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
    Segmenta il segnale con sovrapposizione per risoluzione temporale di 0.1s
    """
    step = int(segment_length * (1 - overlap_ratio))
    segments = []
    
    for start in range(0, data.shape[1] - segment_length + 1, step):
        segment = data[:, start:start + segment_length]
        segments.append(segment)
    
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
    """
    Normalizza le features usando MinMaxScaler
    """
    if isinstance(spectrum, (list, tuple)) and len(spectrum) > 0:
        # Se è una lista di array (come nel nostro caso)
        if isinstance(spectrum[0], np.ndarray):
            # Converte in array 2D: (n_samples, n_features)
            features_array = np.array(spectrum)
            if features_array.ndim == 1:
                features_array = features_array.reshape(-1, 1)
        else:
            # Se è già un array
            features_array = np.array(spectrum)
            if features_array.ndim == 1:
                features_array = features_array.reshape(-1, 1)
    else:
        # Gestione caso singolo spectrum (backward compatibility)
        if isinstance(spectrum, np.ndarray):
            if spectrum.ndim == 1:
                features_array = spectrum.reshape(-1, 1)
            else:
                features_array = spectrum
        else:
            # Vecchia logica per compatibilità
            scaler = MinMaxScaler()
            normalized_spectrum = []
            for channel_spectrum in spectrum:
                if len(channel_spectrum.shape) == 1:
                    channel_spectrum = channel_spectrum.reshape(-1, 1)
                normalized_channel = scaler.fit_transform(channel_spectrum).flatten()
                normalized_spectrum.append(normalized_channel)
            return np.array(normalized_spectrum)
    
    # Normalizza le features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features_array)
    
    return normalized_features

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