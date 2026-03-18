from __future__ import annotations

import csv
import io
import re
import uuid
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, send_file, url_for

from pipeline.run_pipeline import run_pipeline
from src.pipeline.organigram.organigram_parser import parse_organigram
from src.pipeline.role_task_mapping.role_task_parser import parse_role_task_mapping

APP_ROOT = Path(__file__).resolve().parent
RUNS_DIR  = (APP_ROOT / "runs").resolve()
RUNS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_ARTIFACTS = {
    "statistics.txt",
    "preprocess.txt",
    "organigram.xml",
    "role_task_mapping.xml",
    "process_description.txt",
    "references.csv",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB


def safe_job_dir(job_id: str) -> Path:
    if not job_id:
        abort(400)
    if any(sep in job_id for sep in ["/", "\\"]):
        abort(400)
    job_dir = (RUNS_DIR / job_id).resolve()
    if not str(job_dir).startswith(str(RUNS_DIR)):
        abort(400)
    return job_dir


def parse_csv_table(content: str) -> tuple[list[str], list[list[str]]]:
    """Return (headers, rows) from a CSV string; empty lists if blank."""
    if not content.strip():
        return [], []
    reader = csv.reader(io.StringIO(content))
    rows   = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


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

    job_id  = uuid.uuid4().hex
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

    def read_optional(name: str) -> str:
        p = job_dir / name
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

    input_text        = read_optional("input.txt")
    preprocessed_text = read_optional("preprocess.txt")
    organigram_xml    = read_optional("organigram.xml")
    role_task_xml     = read_optional("role_task_mapping.xml")
    references_raw    = read_optional("references.csv")

    num_articles = sum(
        1 for _ in re.finditer(
            r'^(?:Article|Art\.)\s+\d+', input_text, re.MULTILINE | re.IGNORECASE
        )
    )

    refs_file = job_dir / "references.csv"
    num_references = 0
    if refs_file.exists():
        with open(refs_file, "r", encoding="utf-8") as f:
            num_references = max(0, len(f.readlines()) - 1)

    input_tokens  = len(input_text.split())
    output_tokens = len(preprocessed_text.split())

    stats_text = (
        f"Number of articles processed : {num_articles}\n"
        f"Number of references found   : {num_references}\n"
        f"Number of tokens in input    : {input_tokens}\n"
        f"Number of tokens in output   : {output_tokens}\n"
    )
    (job_dir / "statistics.txt").write_text(stats_text, encoding="utf-8")

    ref_headers, ref_rows = parse_csv_table(references_raw)
    organigram_graph      = parse_organigram(organigram_xml)
    role_task_rows        = parse_role_task_mapping(role_task_xml)

    return render_template(
        "job.html",
        job_id=job_id,
        num_articles=num_articles,
        num_references=num_references,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stats_text=stats_text,
        preprocessed_text=preprocessed_text,
        organigram_content=organigram_xml,
        organigram_graph=organigram_graph,
        role_task_content=role_task_xml,
        role_task_rows=role_task_rows,
        process_description=read_optional("process_description.txt"),
        references_raw=references_raw,
        ref_headers=ref_headers,
        ref_rows=ref_rows,
    )


@app.get("/jobs/<job_id>/download/<name>")
def download(job_id: str, name: str):
    if name not in ALLOWED_ARTIFACTS:
        abort(400)
    job_dir   = safe_job_dir(job_id)
    file_path = (job_dir / name).resolve()
    if not file_path.exists():
        abort(404)
    if not str(file_path).startswith(str(job_dir)):
        abort(400)
    return send_file(file_path, as_attachment=True, download_name=name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
