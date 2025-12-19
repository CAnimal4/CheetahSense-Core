import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile

OUTPUT_DIR = Path("data/synth")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def synth_wave(freq: float, sr: int, duration: float) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    noise = 0.02 * np.random.default_rng(42).normal(size=wave.shape)
    return (wave + noise).astype(np.float32)


def main():
    sr = 16000
    duration = 0.6
    clips = {
        "resting": 220.0,
        "hunting": 880.0,
        "distress": 440.0,
    }
    labels_path = OUTPUT_DIR / "labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "label"],
        )
        writer.writeheader()
        for label, freq in clips.items():
            filename = f"{label}_0.wav"
            wave = synth_wave(freq, sr, duration)
            wavfile.write(OUTPUT_DIR / filename, sr, wave)
            writer.writerow({"filename": filename, "label": label})
    print(f"Wrote synthetic data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
