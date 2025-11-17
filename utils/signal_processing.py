from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mne
import os


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
    """
    Calcola lo spettro di potenza per ogni segmento
    
    Note:
        Per segnali già filtrati in banda spindle (9-15 Hz), 
        tutto lo spettro è rilevante
    """
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
        sigma = max(0.5, window_size / 4)
        return gaussian_filter1d(signal.astype(float), sigma=sigma, mode='nearest')
    
    elif method == 'savgol':
        from scipy.signal import savgol_filter
        if window_size % 2 == 0:
            window_size += 1
        window_size = max(3, window_size)
        window_size = min(window_size, len(signal))
        
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


def mne_bandpass_filter(raw, lowcut=9, highcut=15, filter_length='auto', 
                        l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                        method='fir', phase='zero', fir_design='firwin'):
    """
    Applica un filtro bandpass usando MNE
    
    Args:
        raw: mne.io.Raw object
        lowcut: Frequenza di taglio bassa (Hz)
        highcut: Frequenza di taglio alta (Hz)
        filter_length: Lunghezza del filtro ('auto' o numero di campioni)
        l_trans_bandwidth: Larghezza banda di transizione bassa ('auto' o Hz)
        h_trans_bandwidth: Larghezza banda di transizione alta ('auto' o Hz)
        method: 'fir' o 'iir'
        phase: 'zero' (zero-phase) o 'minimum'
        fir_design: 'firwin' o 'firwin2'
    
    Returns:
        mne.io.Raw object filtrato (in-place modification)
    """
    raw.filter(
        l_freq=lowcut,
        h_freq=highcut,
        filter_length=filter_length,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
        method=method,
        phase=phase,
        fir_design=fir_design,
        verbose=False
    )
    return raw


def crop_and_save_edf(raw_or_path, output_path, tmin, tmax, include_tmax=True, overwrite=True):
    """
    Croppa e salva un file EDF
    
    Args:
        raw_or_path: mne.io.Raw object oppure percorso file EDF di input
        output_path: Percorso file EDF di output
        tmin: Tempo di inizio in secondi
        tmax: Tempo di fine in secondi
        include_tmax: Se True, include il campione a tmax
        overwrite: Se True, sovrascrive file esistente
    
    Returns:
        mne.io.Raw object croppato
    """
    if isinstance(raw_or_path, str):
        raw = mne.io.read_raw_edf(raw_or_path, preload=True, verbose=False)
    else:
        raw = raw_or_path
    
    raw.crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    raw.export(output_path, overwrite=overwrite)
    
    return raw


def compute_morlet_wavelet(data, sfreq, fc=12.5, n_cycles=7):
    """
    Applica la trasformata Morlet wavelet al segnale
    
    Args:
        data: segnale (1D array)
        sfreq: frequenza di campionamento
        fc: frequenza centrale (Hz)
        n_cycles: numero di cicli del wavelet
    
    Returns:
        wavelet_complex: segnale wavelet complesso (per estrarre ampiezza e fase)
    """
    # Calcola parametri bandwidth
    s = n_cycles / (2 * np.pi * fc)
    fb = 2 * s**2
    
    # Lunghezza della wavelet in secondi
    wavelet_duration = n_cycles / fc
    wavelet_samples = int(wavelet_duration * sfreq * 2)
    
    # Crea morlet wavelet
    t = np.arange(-wavelet_samples/2, wavelet_samples/2) / sfreq
    morlet_wav = (np.pi * fb)**(-0.5) * np.exp(2j * np.pi * fc * t) * np.exp(-t**2 / fb)
    
    # Convoluzione (mantieni risultato complesso)
    wavelet_signal = np.convolve(data, morlet_wav, mode='same')
    
    return wavelet_signal


def compute_morlet_features(segments, sfreq, fc_range=[11, 12.5, 14, 15], n_cycles=7):
    """
    Estrae feature Morlet multi-scala da segmenti
    
    Args:
        segments: array (n_channels, n_samples)
        sfreq: frequenza campionamento
        fc_range: lista di frequenze centrali
        n_cycles: numero di cicli
    
    Returns:
        features: array (n_channels, n_features)
            Per ogni fc: [ampiezza_media, ampiezza_max, fase_media, freq_inst_media]
    """
    n_channels = segments.shape[0]
    n_features_per_fc = 4  # ampiezza_media, ampiezza_max, fase_media, freq_inst_media
    n_features_total = len(fc_range) * n_features_per_fc
    
    features = np.zeros((n_channels, n_features_total))
    
    for ch_idx in range(n_channels):
        channel_data = segments[ch_idx, :]
        feature_idx = 0
        
        for fc in fc_range:
            # Calcola wavelet complesso
            wavelet_complex = compute_morlet_wavelet(channel_data, sfreq, fc, n_cycles)
            
            # Estrai componenti
            amplitude = np.abs(wavelet_complex)
            phase = np.angle(wavelet_complex)
            
            # Calcola frequenza istantanea
            phase_diff = np.diff(np.unwrap(phase))
            instantaneous_freq = (sfreq / (2 * np.pi)) * phase_diff
            
            # Feature estratte
            features[ch_idx, feature_idx] = np.mean(amplitude)      # ampiezza media
            features[ch_idx, feature_idx + 1] = np.max(amplitude)   # ampiezza max
            features[ch_idx, feature_idx + 2] = np.mean(phase)      # fase media
            features[ch_idx, feature_idx + 3] = np.mean(instantaneous_freq) if len(instantaneous_freq) > 0 else 0  # freq inst media
            
            feature_idx += n_features_per_fc
    
    return features


def compute_adaptive_threshold(signal, window_sec=0.1, sfreq=200, threshold_multiplier=4.5):
    """
    Calcola threshold adattivo con media mobile
    
    Args:
        signal: segnale wavelet (ampiezza)
        window_sec: finestra per media mobile (secondi)
        sfreq: frequenza di campionamento
        threshold_multiplier: moltiplicatore per la soglia (4.5x)
    
    Returns:
        threshold_signal: array con valori di soglia adattiva
    """
    window_samples = int(window_sec * sfreq)
    
    # Calcola media mobile
    from scipy.ndimage import uniform_filter1d
    moving_avg = uniform_filter1d(signal, size=window_samples, mode='nearest')
    
    # Threshold = 4.5 * media mobile
    threshold_signal = threshold_multiplier * moving_avg
    
    return threshold_signal


def merge_close_spindles(regions, min_gap_sec=1.0, max_total_duration=3.0):
    """
    Unisce spindles vicini secondo criteri
    
    Args:
        regions: lista di tuple (start_time, end_time)
        min_gap_sec: distanza minima tra spindles (secondi)
        max_total_duration: durata massima dopo merge (secondi)
    
    Returns:
        merged_regions: lista di regioni unite
    """
    if len(regions) == 0:
        return regions
    
    # Ordina per tempo di inizio
    sorted_regions = sorted(regions, key=lambda x: x[0])
    merged = [sorted_regions[0]]
    
    for current_start, current_end in sorted_regions[1:]:
        last_start, last_end = merged[-1]
        
        # Calcola gap e durata totale se unite
        gap = current_start - last_end
        total_duration = current_end - last_start
        
        # Unisci se gap < 1s e durata totale < 3s
        if gap < min_gap_sec and total_duration <= max_total_duration:
            merged[-1] = (last_start, current_end)
        else:
            merged.append((current_start, current_end))
    
    return merged