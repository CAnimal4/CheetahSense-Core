import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from src.features.audio_embeddings import compute_features, to_feature_vector
from src.preprocess.audio_preprocess import load_audio_mono

CONFIG_PATH = Path("configs/train_config.yaml")
CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(data_dir: Path) -> List[tuple[Path, str]]:
    labels_path = data_dir / "labels.csv"
    samples: List[tuple[Path, str]] = []
    with labels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((data_dir / row["filename"], row["label"]))
    return samples


def train():
    config = load_config()
    data_dir = Path(config["data_dir"])
    samples = load_dataset(data_dir)
    labels = sorted(list({label for _, label in samples}))

    feature_sums: Dict[str, np.ndarray] = {label: np.zeros(3, dtype=np.float32) for label in labels}
    counts: Dict[str, int] = {label: 0 for label in labels}

    for path, label in samples:
        sr, audio = load_audio_mono(path, target_sr=config["sample_rate"])
        feats = compute_features(audio, sr)
        vec = to_feature_vector(feats)
        feature_sums[label] += vec
        counts[label] += 1

    weights = []
    bias = []
    for label in labels:
        mean_vec = feature_sums[label] / max(counts[label], 1)
        weights.append(mean_vec.tolist())
        bias.append(0.0)

    checkpoint = {"labels": labels, "weights": weights, "bias": bias}
    ckpt_path = CHECKPOINT_DIR / "trained_weights.json"
    with ckpt_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"Wrote checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()
