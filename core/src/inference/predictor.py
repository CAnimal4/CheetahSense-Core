import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.features.audio_embeddings import compute_features, to_feature_vector
from src.preprocess.audio_preprocess import load_audio_mono

DEFAULT_LABELS = ["resting", "hunting", "distress"]
DEFAULT_WEIGHTS = np.array(
    [
        [1.2, -0.4, -0.2],   # resting
        [-0.3, 1.0, -0.1],   # hunting
        [-0.6, -0.2, 1.3],   # distress
    ],
    dtype=np.float32,
)
DEFAULT_BIAS = np.array([0.0, 0.0, 0.0], dtype=np.float32)


class Predictor:
    def __init__(self, checkpoint_path: Path | None = None):
        self.labels: List[str] = DEFAULT_LABELS
        self.weights, self.bias = self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: Path | None) -> Tuple[np.ndarray, np.ndarray]:
        if checkpoint_path is None:
            checkpoint_path = Path("models/checkpoints/trained_weights.json")
        if checkpoint_path.exists():
            with checkpoint_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            weights = np.array(data.get("weights", DEFAULT_WEIGHTS)).astype(np.float32)
            bias = np.array(data.get("bias", DEFAULT_BIAS)).astype(np.float32)
            return weights, bias
        return DEFAULT_WEIGHTS, DEFAULT_BIAS

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        denom = np.sum(exp) + 1e-9
        return exp / denom

    def predict_from_file(self, file_path: Path) -> Dict:
        sample_rate, audio = load_audio_mono(file_path)
        feats = compute_features(audio, sample_rate)
        vec = to_feature_vector(feats)
        logits = np.dot(self.weights, vec) + self.bias
        probs = self._softmax(logits)
        top_idx = int(np.argmax(probs))
        label = self.labels[top_idx]
        return {
            "label": label,
            "probs": {lbl: float(prob) for lbl, prob in zip(self.labels, probs)},
            "confidence": float(probs[top_idx]),
            "features": feats,
        }
