from pathlib import Path

import numpy as np
from scipy.io import wavfile

from src.preprocess.audio_preprocess import load_audio_mono, normalize_audio


def test_load_audio_mono_resamples_and_normalizes(tmp_path):
    sr = 8000
    t = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
    wave = 0.8 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    wavfile.write(path, sr, wave.astype(np.float32))

    target_sr, audio = load_audio_mono(path, target_sr=16000)
    assert target_sr == 16000
    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) <= 1.0 + 1e-6


def test_normalize_audio_handles_zero_vector():
    audio = np.zeros(10, dtype=np.float32)
    norm = normalize_audio(audio)
    assert np.allclose(norm, 0.0)
