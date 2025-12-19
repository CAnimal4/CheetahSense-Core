from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio)) + 1e-9
    return (audio / peak).astype(np.float32)


def load_audio_mono(file_path: Path, target_sr: int = 16000) -> Tuple[int, np.ndarray]:
    sr, data = wavfile.read(file_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)
    # Convert integer PCM to [-1,1]
    if data.dtype != np.float32:
        max_val = np.iinfo(data.dtype).max
        data = data / float(max_val)
    if sr != target_sr:
        data = signal.resample_poly(data, target_sr, sr)
        sr = target_sr
    data = normalize_audio(data)
    return sr, data
