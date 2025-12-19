import json
import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from scipy.io import wavfile

from app.fastapi_app import UPLOAD_PREFIX, app
from src.inference.predictor import Predictor


def make_tone(tmp_path, freq=440.0, sr=16000, duration=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    path = tmp_path / "tone.wav"
    wavfile.write(path, sr, wave.astype(np.float32))
    return path


def test_predictor_output_schema(tmp_path):
    path = make_tone(tmp_path, freq=440.0)
    pred = Predictor().predict_from_file(path)
    assert "label" in pred and "probs" in pred and "confidence" in pred
    assert 0.0 <= pred["confidence"] <= 1.0
    assert abs(sum(pred["probs"].values()) - 1.0) < 1e-4


def test_fastapi_upload_deletes_temp_file(tmp_path):
    client = TestClient(app)
    before = list(Path(tempfile.gettempdir()).glob(f"{UPLOAD_PREFIX}*"))
    path = make_tone(tmp_path, freq=220.0)
    with path.open("rb") as f:
        response = client.post("/upload", files={"file": ("tone.wav", f, "audio/wav")}, data={"contribute": "false"})
    assert response.status_code == 200
    after = list(Path(tempfile.gettempdir()).glob(f"{UPLOAD_PREFIX}*"))
    # Ensure no new lingering temp files remain with the prefix
    assert len(after) == len(before)

    payload = json.loads(response.content.decode("utf-8"))
    assert "contribution" in payload
    assert payload["contribution"]["status"] in {"skipped", "error", "pushed"}
