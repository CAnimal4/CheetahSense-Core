import json
from pathlib import Path

from src.inference.predictor import DEFAULT_BIAS, DEFAULT_LABELS, DEFAULT_WEIGHTS

CHECKPOINT_DIR = Path("models/checkpoints")


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"labels": DEFAULT_LABELS, "weights": DEFAULT_WEIGHTS.tolist(), "bias": DEFAULT_BIAS.tolist()}
    path = CHECKPOINT_DIR / "trained_weights.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Placeholder checkpoint written to {path}")


if __name__ == "__main__":
    main()
