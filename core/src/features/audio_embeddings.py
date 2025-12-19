from typing import Dict

import numpy as np


def compute_features(waveform: np.ndarray, sample_rate: int) -> Dict[str, float]:
    eps = 1e-9
    rms = float(np.sqrt(np.mean(np.square(waveform)) + eps))

    magnitude = np.abs(np.fft.rfft(waveform))
    freqs = np.fft.rfftfreq(len(waveform), d=1.0 / sample_rate)
    spectral_centroid = float(np.sum(freqs * magnitude) / (np.sum(magnitude) + eps))

    zero_crossings = np.where(np.diff(np.sign(waveform)))[0]
    zcr = float(len(zero_crossings)) / (len(waveform) + eps)

    return {"rms": rms, "spectral_centroid": spectral_centroid, "zero_cross_rate": zcr}


def to_feature_vector(features: Dict[str, float]) -> np.ndarray:
    # Scale features to similar ranges for stability.
    rms = features["rms"] * 10
    centroid = features["spectral_centroid"] / 4000.0
    zcr = features["zero_cross_rate"] * 5.0
    return np.array([rms, centroid, zcr], dtype=np.float32)
