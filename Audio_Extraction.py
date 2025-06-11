from audio2numpy import open_audio
import os
import numpy as np
import pandas as pd
import librosa

def audio_to_df(path_to_audio: str):
    samples, sample_rate = open_audio(path_to_audio)
    times = np.arange(len(samples)) / sample_rate
    return pd.DataFrame({"time": times,"amplitude": samples.astype(float)})

def normalize_by_peak(df: pd.DataFrame):
    df_norm = df.copy()
    max_abs = df_norm['amplitude'].abs().max()

    if max_abs == 0:
        df_norm['amplitude'] = 0.0 #Defensive for division by zero
    else:
        df_norm['amplitude'] = df_norm['amplitude'] / max_abs

    return df_norm

def process_audio_file(file_path: str):
    fname = os.path.basename(file_path)
    df_raw = audio_to_df(file_path)
    df_norm = normalize_by_peak(df_raw)
    df_norm["filename"] = fname
    df_norm = df_norm[["filename", "time", "amplitude"]]
    return df_norm


def compute_mel_spectrogram_df(df: pd.DataFrame,n_fft: int = 1024,hop_length: int = 512,n_mels: int = 128,fmin: float = 0.0,fmax: float = None):

    fname = df["filename"].iloc[0]
    grp = df.sort_values("time")
    times = grp["time"].values
    amps = grp["amplitude"].values


    dt = np.mean(np.diff(times))
    fs = 1.0 / dt
    this_fmax = fmax if fmax is not None else fs / 2.0

    # Mel spectrogram (power)
    S = librosa.feature.melspectrogram(y=amps,sr=int(round(fs)),n_fft=n_fft,hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=this_fmax,power=2.0)

    # Convert to dB
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Frame times
    frame_times = librosa.frames_to_time(np.arange(S_dB.shape[1]),sr=int(round(fs)),hop_length=hop_length,n_fft=n_fft)

    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=this_fmax)
    n_frames = S_dB.shape[1]
    M = n_mels

    power_vals = S_dB.flatten(order='F')
    time_vals = np.repeat(frame_times, M)
    mel_vals = np.tile(mel_freqs, n_frames)

    return pd.DataFrame({"filename": np.repeat(fname, M * n_frames),"time": time_vals,"mel_frequency": mel_vals,"power_db": power_vals})
