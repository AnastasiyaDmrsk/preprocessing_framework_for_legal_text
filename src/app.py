# src/app.py
from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, send_file, url_for

from pipeline.run_pipeline import run_pipeline

APP_ROOT = Path(__file__).resolve().parent
RUNS_DIR = (APP_ROOT / "runs").resolve()
RUNS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_ARTIFACTS = {
    "process.txt",
    "organigram.xml",
    "role_task_mapping.xml",
    "preprocess.txt",
    "references.csv",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB


def safe_job_dir(job_id: str) -> Path:
    if not job_id:
        abort(400)
    if any(sep in job_id for sep in ["/", "\\"]):
        abort(400)
    job_dir = (RUNS_DIR / job_id).resolve()
    if not str(job_dir).startswith(str(RUNS_DIR)):
        abort(400)
    return job_dir


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/submit")
def submit():
    text = (request.form.get("text") or "").strip()

    file = request.files.get("file")
    if file and getattr(file, "filename", ""):
        raw = file.read() or b""
        try:
            text = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace").strip()

    if not text:
        abort(400)

    job_id = uuid.uuid4().hex
    job_dir = safe_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    (job_dir / "input.txt").write_text(text, encoding="utf-8")

    run_pipeline(text=text, out_dir=job_dir)

    return redirect(url_for("job", job_id=job_id), code=303)


@app.get("/jobs/<job_id>")
def job(job_id: str):
    job_dir = safe_job_dir(job_id)
    if not job_dir.exists():
        abort(404)

    def read_optional(name: str, limit: int = None) -> str:
        p = job_dir / name
        if p.exists():
            content = p.read_text(encoding="utf-8", errors="replace")
            return content if limit is None else content[:limit]
        return ""

    # Calculate statistics
    input_text = read_optional("input.txt")
    preprocessed_text = read_optional("preprocess.txt")

    # Count articles from INPUT (not preprocessed) - lines with Article/Art. pattern
    num_articles = 0
    for match in re.finditer(r'^(?:Article|Art\.)\s+\d+', input_text, re.MULTILINE | re.IGNORECASE):
        num_articles += 1

    # Count references
    refs_file = job_dir / "references.csv"
    num_references = 0
    if refs_file.exists():
        with open(refs_file, 'r', encoding='utf-8') as f:
            num_references = max(0, len(f.readlines()) - 1)  # Subtract header

    # Count tokens (simple word-based tokenization)
    input_tokens = len(input_text.split())
    output_tokens = len(preprocessed_text.split())

    return render_template(
        "job.html",
        job_id=job_id,
        input_text=input_text,
        preprocessed_text=preprocessed_text,
        process_preview=read_optional("process.txt"),
        num_articles=num_articles,
        num_references=num_references,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


@app.get("/jobs/<job_id>/download/<name>")
def download(job_id: str, name: str):
    if name not in ALLOWED_ARTIFACTS:
        abort(400)

    job_dir = safe_job_dir(job_id)
    file_path = (job_dir / name).resolve()
    if not file_path.exists():
        abort(404)
    if not str(file_path).startswith(str(job_dir)):
        abort(400)

    return send_file(file_path, as_attachment=True, download_name=name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
