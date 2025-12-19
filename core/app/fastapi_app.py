import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.inference.predictor import Predictor
from src.utils.github_push import PushResult, push_pending_clip

app = FastAPI(title="CheetahSense API", version="0.1.0")

predictor = Predictor()

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
CONTRIB_THRESHOLD = float(os.getenv("CONTRIB_THRESHOLD", "0.85"))
UPLOAD_PREFIX = "cheetahsense_upload_"


@app.get("/health")
def health():
    return {"status": "ok", "threshold": CONTRIB_THRESHOLD}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    contribute: bool = Form(False),
    contributor: Optional[str] = Form(None),
    label: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    contribution_result = PushResult(status="skipped", url=None, message="Contribution not requested.")
    temp_path: Optional[Path] = None

    try:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Empty upload.")
        if len(payload) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large (>5MB).")

        suffix = Path(file.filename or "upload.bin").suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, prefix=UPLOAD_PREFIX, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            tmp.write(payload)

        inference = predictor.predict_from_file(temp_path)

        if contribute:
            if inference["confidence"] < CONTRIB_THRESHOLD:
                try:
                    contribution_result = push_pending_clip(
                        file_path=temp_path,
                        contributor=contributor,
                        provided_label=label,
                        notes=notes,
                        predicted_label=inference["label"],
                        confidence=inference["confidence"],
                    )
                except Exception as exc:  # noqa: BLE001
                    contribution_result = PushResult(
                        status="error",
                        url=None,
                        message=f"Contribution failed: {exc}",
                    )
            else:
                contribution_result = PushResult(
                    status="skipped",
                    url=None,
                    message=f"Confidence {inference['confidence']:.2f} above threshold {CONTRIB_THRESHOLD:.2f}; not contributed.",
                )

        response = {
            "label": inference["label"],
            "probs": inference["probs"],
            "confidence": inference["confidence"],
            "contribution": contribution_result.to_dict(),
        }
        return JSONResponse(content=response)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    host = os.getenv("UVICORN_HOST", "0.0.0.0")
    port = int(os.getenv("UVICORN_PORT", "8000"))
    import uvicorn

    uvicorn.run("app.fastapi_app:app", host=host, port=port, reload=True)
