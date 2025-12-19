import io
import os
import tempfile
from pathlib import Path

import requests
import streamlit as st

from src.inference.predictor import Predictor
from src.utils.github_push import push_pending_clip

API_URL = os.getenv("CHEETAHSENSE_API", "http://localhost:8000/upload")
predictor = Predictor()
MAX_FILE_SIZE = 5 * 1024 * 1024  # keep in sync with API

st.set_page_config(page_title="CheetahSense", page_icon="🐆", layout="centered")
st.title("CheetahSense — Vocalization → Intent")

st.markdown("Uploads are processed ephemerally. Contributions are opt-in and gated by a confidence threshold.")


def run_local_inference(file_bytes: bytes, suffix: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="cheetahsense_streamlit_") as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        result = predictor.predict_from_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return result


with st.form("upload_form"):
    uploaded = st.file_uploader("Upload audio/video (wav preferred, <=5MB)", type=["wav", "mp3", "mp4", "m4a"])
    contribute = st.checkbox("Contribute to dataset if confidence is low", value=False)
    contributor = st.text_input("Contributor (optional)")
    label = st.text_input("Your label (optional, used only when contributing)")
    notes = st.text_area("Notes (optional)")
    use_api = st.checkbox("Send via FastAPI instead of local-only processing", value=False)
    submitted = st.form_submit_button("Run")

if submitted:
    if not uploaded:
        st.error("Please upload a file.")
    else:
        data = uploaded.read()
        if len(data) > MAX_FILE_SIZE:
            st.error("File too large (>5MB).")
        else:
            if use_api:
                files = {"file": (uploaded.name, io.BytesIO(data), uploaded.type)}
                form = {
                    "contribute": str(contribute).lower(),
                    "contributor": contributor,
                    "label": label,
                    "notes": notes,
                }
                resp = requests.post(API_URL, files=files, data=form, timeout=30)
                if resp.ok:
                    output = resp.json()
                else:
                    st.error(f"API error: {resp.text}")
                    output = None
            else:
                output = run_local_inference(data, Path(uploaded.name).suffix or ".bin")
                output = {
                    "label": output["label"],
                    "probs": output["probs"],
                    "confidence": output["confidence"],
                    "contribution": {"status": "skipped", "url": None, "message": "Local inference only."},
                }
                if contribute and output["confidence"] < float(os.getenv("CONTRIB_THRESHOLD", "0.85")):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                            tmp.write(data)
                            temp_path = Path(tmp.name)
                        push_result = push_pending_clip(
                            file_path=temp_path,
                            contributor=contributor,
                            provided_label=label,
                            notes=notes,
                            predicted_label=output["label"],
                            confidence=output["confidence"],
                        )
                        output["contribution"] = push_result.to_dict()
                    except Exception as exc:  # noqa: BLE001
                        output["contribution"] = {"status": "error", "url": None, "message": str(exc)}
                    finally:
                        if "temp_path" in locals():
                            temp_path.unlink(missing_ok=True)

            if output:
                st.success(f"Predicted: {output['label']} (confidence {output['confidence']:.2f})")
                st.json(output)
