from __future__ import annotations

import secrets
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

from .config import CompressionConfig, ExportConfig
from .demo import run_demo
from .manifest import ModelManifest, read_manifest, write_manifest
from .pipeline import CompressionPipeline

app = FastAPI(title="LLM Edge Compression Server")

# Allow web UI served from localhost (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:8081", "http://127.0.0.1:8081"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunDemoRequest(BaseModel):
    output_dir: str = "artifacts/demo"
    rank_ratio: float = 0.5


def _safe_slug(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "model"


def _extract_uploaded_archive(upload_dir: Path, upload: UploadFile) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    archive_path = upload_dir / (upload.filename or "model.zip")
    with archive_path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)

    if not zipfile.is_zipfile(archive_path):
        raise HTTPException(status_code=400, detail="uploaded file must be a zip archive containing a model directory")

    extract_dir = upload_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    config_paths = list(extract_dir.rglob("config.json"))
    if config_paths:
        return config_paths[0].parent
    return extract_dir


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/artifacts")
def list_artifacts(base: str = "artifacts") -> list[dict[str, Any]]:
    base_path = Path(base)
    if not base_path.exists():
        return []
    results: list[dict[str, Any]] = []
    for child in sorted(base_path.iterdir()):
        manifest_path = child / "manifest.json"
        if manifest_path.exists():
            try:
                m = read_manifest(manifest_path)
                results.append({"name": child.name, "manifest": m})
            except Exception:
                continue
    return results


@app.get("/manifest/{artifact_name}")
def get_manifest(artifact_name: str, base: str = "artifacts") -> ModelManifest:
    manifest_path = Path(base) / artifact_name / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="manifest not found")
    return read_manifest(manifest_path)


@app.get("/compare")
def compare(a: str, b: str, base: str = "artifacts") -> dict[str, Any]:
    pa = Path(base) / a / "manifest.json"
    pb = Path(base) / b / "manifest.json"
    if not pa.exists() or not pb.exists():
        raise HTTPException(status_code=404, detail="one or both manifests not found")
    ma = read_manifest(pa)
    mb = read_manifest(pb)
    oa = ma.metrics.get("original_parameters")
    ca = ma.metrics.get("compressed_parameters")
    ob = mb.metrics.get("original_parameters")
    cb = mb.metrics.get("compressed_parameters")
    return {
        "a": ma,
        "b": mb,
        "a_metrics": ma.metrics,
        "b_metrics": mb.metrics,
        "difference": {
            "original_parameters_diff": (oa or 0) - (ob or 0),
            "compressed_parameters_diff": (ca or 0) - (cb or 0),
            "parameter_ratio_a": ma.metrics.get("parameter_ratio"),
            "parameter_ratio_b": mb.metrics.get("parameter_ratio"),
        },
    }


@app.get("/chart/{artifact_name}", response_class=HTMLResponse)
def chart_artifact(artifact_name: str, base: str = "artifacts") -> HTMLResponse:
        """Return a simple HTML page with a Chart.js bar chart comparing
        original vs compressed parameter counts for the given artifact.
        """
        manifest_path = Path(base) / artifact_name / "manifest.json"
        if not manifest_path.exists():
                raise HTTPException(status_code=404, detail="manifest not found")
        m = read_manifest(manifest_path)
        orig = int(m.metrics.get("original_parameters") or 0)
        comp = int(m.metrics.get("compressed_parameters") or 0)
        title = f"Compression comparison: {artifact_name}"
        data = {"labels": ["Original", "Compressed"], "values": [orig, comp]}
        json_data = json.dumps(data)
        html = (
                "<!doctype html>"
                "<html>"
                "<head>"
                "<meta charset='utf-8' />"
                f"<title>{title}</title>"
                "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>"
                "</head>"
                "<body>"
                f"<h2>{title}</h2>"
                "<canvas id='barChart' width='600' height='300'></canvas>"
                "<script>"
                "const data = "
                + json_data
                + ";"
                "const ctx = document.getElementById('barChart').getContext('2d');"
                "new Chart(ctx, {"
                "type: 'bar',"
                "data: { labels: data.labels, datasets: [{ label: 'Parameter count', data: data.values, backgroundColor: ['#4e79a7', '#f28e2b'] }] },"
                "options: { scales: { y: { beginAtZero: true } } }"
                "});"
                "</script>"
                "</body></html>"
        )
        return HTMLResponse(content=html)


@app.post("/run-demo")
def api_run_demo(req: RunDemoRequest) -> dict[str, Any]:
    out = Path(req.output_dir)
    res = run_demo(out, rank_ratio=req.rank_ratio)
    return {"manifest": str(res.manifest_path), "parameter_ratio": res.parameter_ratio}


@app.post("/compress-upload")
async def compress_upload(
    model_archive: UploadFile = File(...),
    model_id: str = Form("uploaded-model"),
    method: str = Form("tensor_inspired"),
    rank_ratio: float = Form(0.5),
    target_device: str = Form("cpu"),
    heal_steps: int = Form(0),
    heal_learning_rate: float = Form(1e-4),
    calibration_batches: int = Form(8),
    calibration_batch_size: int = Form(2),
    calibration_sequence_length: int = Form(16),
    output_root: str = Form("artifacts/uploads"),
) -> dict[str, Any]:
    slug = _safe_slug(model_id)
    request_id = secrets.token_hex(4)
    output_dir = Path(output_root) / f"{slug}-{request_id}"

    with tempfile.TemporaryDirectory(prefix="llm-edge-upload-") as temp_dir:
        upload_dir = Path(temp_dir)
        extracted_model_dir = _extract_uploaded_archive(upload_dir, model_archive)
        compression = CompressionConfig(
            model_id=extracted_model_dir.as_posix(),
            output_dir=output_dir,
            method=method,
            rank_ratio=rank_ratio,
            target_device=target_device,
            heal_steps=heal_steps,
            heal_learning_rate=heal_learning_rate,
            calibration_batches=calibration_batches,
            calibration_batch_size=calibration_batch_size,
            calibration_sequence_length=calibration_sequence_length,
        )
        export = ExportConfig(output_dir=output_dir)
        try:
            result = CompressionPipeline(compression, export).run()
        except ValueError as ve:
            # Likely a malformed model config (e.g. missing model_type)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as exc:
            # For unexpected errors, return a 500 with the message for easier debugging
            raise HTTPException(status_code=500, detail=str(exc))
        result.manifest.model_id = model_id
        write_manifest(result.manifest, result.manifest_path.parent)

    return {
        "manifest": result.manifest_path.as_posix(),
        "output_dir": result.export_dir.as_posix(),
        "parameter_ratio": result.manifest.metrics.get("parameter_ratio"),
        "metrics": result.manifest.metrics,
        "model_id": model_id,
    }


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("llm_edge_compression.server:app", host=host, port=port, log_level="info")
