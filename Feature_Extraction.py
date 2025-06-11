
import Audio_Extraction
import librosa
from sklearn.linear_model import LinearRegression
import numpy as np
from audio2numpy import open_audio


def extract_combined_features(file_path: str):

    # Get Mel Spectrogram and Waveform
    waveform_df = Audio_Extraction.process_audio_file(file_path)
    mel_df = Audio_Extraction.compute_mel_spectrogram_df(waveform_df)
    y, sr = open_audio(file_path)

    # Mel Spectrogram-Based SDE Features
    filename = mel_df["filename"].iloc[0]

    pivot = mel_df.pivot(index="time", columns="mel_frequency", values="power_db").sort_index()
    features_per_band = []

    for band in pivot.columns:
        band_series = pivot[band].dropna().values
        if len(band_series) < 3:
            continue

        diff = np.diff(band_series)

        # SDE residuals (OU process)
        x_mid = band_series[:-1]
        dx = diff
        x_mean = np.mean(x_mid)
        theta_hat = np.sum(dx * (x_mid - x_mean)) / (np.sum((x_mid - x_mean) ** 2) + 1e-8)
        mu_hat = x_mean
        dt = 1.0
        predicted_dx = theta_hat * (mu_hat - x_mid) * dt
        residuals = dx - predicted_dx
        residual_sigma = np.std(residuals)

        # Additional SDE features
        mean_reversion_strength = np.abs(theta_hat)
        variance_ratio = np.var(diff) / (np.var(band_series) + 1e-8)
        noise_to_signal = residual_sigma / (np.std(predicted_dx) + 1e-8)

        features = {
            'residsual_sigma': residual_sigma,
            'mean_reversion_strength': mean_reversion_strength,
            'variance_ratio': variance_ratio,
            'noise_to_signal': noise_to_signal
        }
        features_per_band.append(features)

    sde_features = {
        'filename': filename,
        **{f'mean_{k}': np.mean([f[k] for f in features_per_band]) for k in features_per_band[0]},
        **{f'median_{k}': np.median([f[k] for f in features_per_band]) for k in features_per_band[0]}
    }

    # Spectral Features from Raw Audio
    hop_length = 512
    duration = len(y) / sr

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).flatten()
    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)
    centroid_range = np.max(centroid) - np.min(centroid)
    increments = np.diff(centroid)
    std_increment = np.std(increments)
    max_abs_increment = np.max(np.abs(increments))
    smoothness_ratio = np.mean(np.abs(increments)) / (std_increment + 1e-8)

    times = librosa.times_like(centroid, sr=sr, hop_length=hop_length).reshape(-1, 1)
    model = LinearRegression().fit(times, centroid)
    predictions = model.predict(times)
    residuals = centroid - predictions
    residual_energy = np.mean(residuals ** 2)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length).flatten()
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length).flatten()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length).flatten()
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length).flatten()
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    speech_rate = len(onsets) / duration if duration > 0 else 0
    zero_crossings = np.sum(librosa.zero_crossings(y, pad=False))
    zero_crossings_per_sec = zero_crossings / duration if duration > 0 else 0
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

    spectral_features = {
        "spectral_centroid_mean": centroid_mean,
        "spectral_centroid_std": centroid_std,
        "centroid_range": centroid_range,
        "std_increment": std_increment,
        "max_abs_increment": max_abs_increment,
        "smoothness_ratio_centroid": smoothness_ratio,
        "residual_energy": residual_energy,
        "spectral_bandwidth_mean": np.mean(bandwidth),
        "spectral_bandwidth_std": np.std(bandwidth),
        "spectral_flatness_mean": np.mean(flatness),
        "spectral_flatness_std": np.std(flatness),
        "spectral_rolloff_mean": np.mean(rolloff),
        "spectral_rolloff_std": np.std(rolloff),
        "zcr_mean": np.mean(zcr),
        "zcr_std": np.std(zcr),
        "speech_rate": speech_rate,
        "zero_crossings_per_sec": zero_crossings_per_sec
    }
    spectral_features.update({f"mfcc_{i+1}_mean": np.mean(mfccs[i]) for i in range(13)})
    spectral_features.update({f"mfcc_{i+1}_std": np.std(mfccs[i]) for i in range(13)})

    return {**sde_features, **spectral_features}











