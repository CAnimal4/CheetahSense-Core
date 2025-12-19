import base64
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests


@dataclass
class PushResult:
    status: str
    url: Optional[str]
    message: str

    def to_dict(self):
        return {"status": self.status, "url": self.url, "message": self.message}


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value or value == "<EMAIL>":
        raise EnvironmentError(f"Environment variable {name} is required for contributions.")
    return value


def _github_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _put_content(owner: str, repo: str, path: str, content_bytes: bytes, message: str, token: str, committer_email: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "committer": {"name": "CheetahSense Bot", "email": committer_email},
    }
    resp = requests.put(url, headers=_github_headers(token), json=payload, timeout=15)
    if not resp.ok:
        raise RuntimeError(f"GitHub content upload failed: {resp.status_code} {resp.text}")
    return resp.json()


def _append_labels_csv(
    owner: str,
    repo: str,
    path: str,
    row: str,
    token: str,
    committer_email: str,
):
    get_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = _github_headers(token)
    existing = requests.get(get_url, headers=headers, timeout=15)
    sha = None
    if existing.status_code == 200:
        content = base64.b64decode(existing.json()["content"]).decode("utf-8")
        if not content.endswith("\n"):
            content += "\n"
        new_content = content + row
        sha = existing.json().get("sha")
    elif existing.status_code == 404:
        header = "filename,label,contributor,notes,confidence,predicted_label,created_utc\n"
        new_content = header + row
    else:
        raise RuntimeError(f"Could not fetch labels.csv: {existing.status_code} {existing.text}")

    payload = {
        "message": "Add pending label metadata",
        "content": base64.b64encode(new_content.encode("utf-8")).decode("utf-8"),
        "committer": {"name": "CheetahSense Bot", "email": committer_email},
    }
    if sha:
        payload["sha"] = sha
    put = requests.put(get_url, headers=headers, json=payload, timeout=15)
    if not put.ok:
        raise RuntimeError(f"GitHub labels append failed: {put.status_code} {put.text}")
    return put.json()


def _safe_filename(name: str) -> str:
    name = name or "upload.wav"
    name = name.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]", "", name)


def push_pending_clip(
    file_path: Path,
    contributor: Optional[str],
    provided_label: Optional[str],
    notes: Optional[str],
    predicted_label: str,
    confidence: float,
    dataset_repo: Optional[str] = None,
) -> PushResult:
    token = _require_env("GH_TOKEN")
    owner = _require_env("GITHUB_OWNER")
    committer_email = _require_env("COMMITTER_EMAIL")
    repo = dataset_repo or os.getenv("DATASET_REPO", "cheetahsense-dataset")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_name = f"{timestamp}_{_safe_filename(file_path.name)}"
    pending_path = f"pending/{safe_name}"

    with file_path.open("rb") as f:
        content_bytes = f.read()

    upload_resp = _put_content(
        owner=owner,
        repo=repo,
        path=pending_path,
        content_bytes=content_bytes,
        message=f"Add pending clip {safe_name}",
        token=token,
        committer_email=committer_email,
    )

    row = ",".join(
        [
            pending_path,
            provided_label or "",
            contributor or "",
            (notes or "").replace("\n", " "),
            f"{confidence:.4f}",
            predicted_label,
            timestamp,
        ]
    )
    row += "\n"
    labels_resp = _append_labels_csv(
        owner=owner,
        repo=repo,
        path="labels.csv",
        row=row,
        token=token,
        committer_email=committer_email,
    )

    url = upload_resp.get("content", {}).get("html_url") or upload_resp.get("commit", {}).get("html_url")
    meta_url = labels_resp.get("content", {}).get("html_url") or labels_resp.get("commit", {}).get("html_url")
    combined_url = url or meta_url

    return PushResult(
        status="pushed",
        url=combined_url,
        message="Contribution pushed to pending/ and labels.csv updated.",
    )
