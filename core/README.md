# CheetahSense Core

Vocalization → intent translator with an ephemeral-by-default upload flow. Uploads are processed then immediately deleted. Contributions are opt-in and only pushed to the private dataset repo when the contributor asks **and** the prediction confidence is below the configured threshold.

## Quick start
```bash
# 1) Bootstrap environment
bash bootstrap.sh

# 2) Make small synthetic clips and labels for local runs/tests
python scripts/generate_synthetic_data.py

# 3) Run tests
pytest

# 4) Launch FastAPI (ephemeral uploads)
uvicorn app.fastapi_app:app --reload --port 8000

# 5) Launch Streamlit (simple UI)
streamlit run app/streamlit_app.py
```

## Endpoint
`POST /upload` (FastAPI)
- multipart `file` (audio/video; wav recommended), `contribute` (bool), optional `contributor`, `label`, `notes`.
- Rejects files > 5MB.
- Returns JSON: label, probs, confidence, and contribution status/url/message.
- Temp files are always deleted in a `finally` block.

## Contribution rules
- Default: no contribution.
- Contribution attempted only when **contribute==True** AND **confidence < CONTRIB_THRESHOLD**.
- Files go to `pending/` of `cheetahsense-dataset` plus a CSV row.
- If GH env vars are missing, contribution returns an error status but inference still succeeds.

## Environment variables
- `GH_TOKEN` (required for contributions; GitHub token with `repo` scope)
- `GITHUB_OWNER` (required; your GitHub user/org)
- `DATASET_REPO` (optional, default `cheetahsense-dataset`)
- `COMMITTER_EMAIL` (required; placeholder like `<EMAIL>` until you set a real one)
- `CONTRIB_THRESHOLD` (optional, default `0.85`)
- `UVICORN_HOST`/`UVICORN_PORT` (optional for CLI runs)

No secrets are stored in code; GH token is only read from the environment.

## Colab note
All dependencies are free/open-source and compatible with Python 3.10+ and Google Colab GPUs. In Colab, clone the repo, run `bash bootstrap.sh`, then `python scripts/generate_synthetic_data.py`.

## Project layout
- `app/fastapi_app.py` — API upload endpoint, ephemeral by default.
- `app/streamlit_app.py` — lightweight UI for local demos.
- `src/inference/predictor.py` — deterministic prototype predictor (RMS + spectral centroid + ZCR).
- `src/utils/github_push.py` — GitHub REST PUT helper for `pending/` uploads + `labels.csv` append.
- `scripts/generate_synthetic_data.py` — tiny synthetic wav clips + labels.
- `models/create_placeholder_checkpoint.py` — creates a placeholder checkpoint.
- `tests/` — pytest suite (preprocess, inference, temp cleanup).

## Training (toy)
A tiny trainer builds class-mean feature centroids from `data/synth`. Configurable via `configs/train_config.yaml`. Outputs checkpoint JSON in `models/checkpoints/`.

## Safety
- Files >5MB are rejected.
- Uploads are processed from secure temp files and deleted in `finally`.
- Contribution is opt-in and gated by confidence threshold.
