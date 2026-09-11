"""FastAPI service for the Edge AI model-compression platform.

The service never fabricates ML metrics. Endpoints return measured values only
when a real artifact and dataset are supplied; otherwise they return an explicit
unmeasured state with a reason.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml_pipeline import (
    DEFAULT_STUDENT,
    DEFAULT_TEACHER,
    DistillationConfig,
    benchmark_local_model,
    build_teacher_student,
    distill_student,
    evaluate_local_model,
    export_onnx,
    export_torchscript,
    load_local_model,
    load_sms_dataset,
    metric_state,
    transform_local_model,
    predict_text,
)

SCHEMA_VERSION = "2.0"
MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"

app = FastAPI(title="Edge AI Model Compression API", version=SCHEMA_VERSION)
origins = [origin.strip() for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://localhost:5173,https://edge-ai-knowledge-distillation-ruddy.vercel.app").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


class DistillationRequest(BaseModel):
    teacher_model: str = Field(DEFAULT_TEACHER, min_length=3, max_length=200)
    student_model: str = Field(DEFAULT_STUDENT, min_length=3, max_length=200)
    temperature: float = Field(4.0, ge=1.0, le=20.0)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    learning_rate: float = Field(3e-5, ge=1e-7, le=1e-2)
    epochs: int = Field(3, ge=1, le=100)
    batch_size: int = Field(16, ge=1, le=256)
    max_sequence_length: int = Field(128, ge=16, le=512)
    weight_decay: float = Field(0.01, ge=0.0, le=1.0)
    warmup_ratio: float = Field(0.1, ge=0.0, le=1.0)
    random_seed: int = Field(42, ge=0, le=2**32 - 1)

    def to_config(self) -> DistillationConfig:
        return DistillationConfig(**self.model_dump())


class BenchmarkRequest(BaseModel):
    model_path: str = Field(..., min_length=1, max_length=500)
    dataset_path: str = Field(..., min_length=1, max_length=500)
    iterations: int = Field(40, ge=1, le=1000)
    warmup_iterations: int = Field(8, ge=0, le=100)
    max_sequence_length: int = Field(128, ge=16, le=512)


class OptimizationRequest(BaseModel):
    model_path: str = Field(..., min_length=1, max_length=500)
    pruning_amount: float = Field(0.2, gt=0.0, lt=1.0)


class ExportRequest(BaseModel):
    model_path: str = Field(..., min_length=1, max_length=500)
    output_path: str = Field(..., min_length=1, max_length=500)
    max_sequence_length: int = Field(128, ge=16, le=512)


class PredictionRequest(BaseModel):
    model_path: str = Field(..., min_length=1, max_length=500)
    text: str = Field(..., min_length=1, max_length=2000)
    max_sequence_length: int = Field(128, ge=16, le=512)


def not_measured(reason: str) -> dict:
    return {"measured": False, "value": "Not measured", "reason": reason}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "edge-ai-model-compression", "schemaVersion": SCHEMA_VERSION}


@app.get("/api/v1/metadata")
def metadata() -> dict:
    return {"schemaVersion": SCHEMA_VERSION, "modelId": MODEL_ID, "teacherDefault": DEFAULT_TEACHER, "studentDefault": DEFAULT_STUDENT, "device": "cpu", "capabilities": {"distillation": True, "evaluation": True, "dynamicQuantization": True, "pruning": True, "onnxExport": True, "torchscriptExport": True}, "metricPolicy": "Only measured values are reported; unavailable values are marked Not measured."}


@app.post("/api/v1/distillation/validate")
def validate_distillation(request: DistillationRequest) -> dict:
    config = request.to_config()
    config.validate()
    return {"valid": True, "schemaVersion": SCHEMA_VERSION, "config": request.model_dump(), "loss": {"formula": "alpha * hardCrossEntropy + (1 - alpha) * temperature^2 * KL(teacher || student)", "temperature": request.temperature, "alpha": request.alpha}}


@app.post("/api/v1/distillation/run")
def run_distillation(request: DistillationRequest, dataset_path: str, output_dir: str = "artifacts/student") -> dict:
    if not Path(dataset_path).exists():
        return {"schemaVersion": SCHEMA_VERSION, "status": "unavailable", "result": not_measured("dataset_path does not exist")}
    try:
        result = distill_student(request.to_config(), dataset_path, output_dir)
        return {"schemaVersion": SCHEMA_VERSION, "status": "measured", "result": result}
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/v1/evaluate")
def evaluate_artifact(model_path: str, dataset_path: str) -> dict:
    """Evaluate a local model artifact against a local labeled CSV.

    The endpoint is intentionally explicit about local paths; production should
    replace this with an authenticated object-storage reference.
    """
    model_file, dataset_file = Path(model_path), Path(dataset_path)
    if not model_file.exists() or not dataset_file.exists():
        return {"schemaVersion": SCHEMA_VERSION, "status": "unavailable", "teacher": not_measured("Provide an existing model_path and dataset_path"), "student": not_measured("Provide an existing model_path and dataset_path")}
    try:
        result = evaluate_local_model(model_file, dataset_file)
        return {"schemaVersion": SCHEMA_VERSION, "status": "measured", "student": result}
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/v1/benchmark")
def benchmark(request: BenchmarkRequest) -> dict:
    model_file, dataset_file = Path(request.model_path), Path(request.dataset_path)
    if not model_file.exists() or not dataset_file.exists():
        return {"schemaVersion": SCHEMA_VERSION, "status": "unavailable", "fp32": not_measured("Both model_path and dataset_path must exist"), "int8": not_measured("Both model_path and dataset_path must exist")}
    try:
        fp32 = benchmark_local_model(model_file, dataset_file, request.iterations, request.warmup_iterations, request.max_sequence_length)
        quantized_dir = model_file.parent / f"{model_file.name}-int8-runtime"
        int8_artifact = transform_local_model(model_file, quantized_dir, "quantize")
        int8 = benchmark_local_model(quantized_dir, dataset_file, request.iterations, request.warmup_iterations, request.max_sequence_length)
        return {"schemaVersion": SCHEMA_VERSION, "status": "measured", "fp32": fp32, "int8": int8, "quantizedArtifact": int8_artifact}
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/v1/optimization/quantize")
def quantize_artifact(request: OptimizationRequest) -> dict:
    if not Path(request.model_path).exists():
        return {"schemaVersion": SCHEMA_VERSION, "status": "unavailable", "result": not_measured("model_path does not exist")}
    output = f"{request.model_path}-int8"
    try:
        return {"schemaVersion": SCHEMA_VERSION, "status": "measured", "result": transform_local_model(request.model_path, output, "quantize")}
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/v1/optimization/prune")
def prune_artifact(request: OptimizationRequest) -> dict:
    if not Path(request.model_path).exists():
        return {"schemaVersion": SCHEMA_VERSION, "status": "unavailable", "result": not_measured("model_path does not exist")}
    output = f"{request.model_path}-pruned"
    try:
        return {"schemaVersion": SCHEMA_VERSION, "status": "measured", "result": transform_local_model(request.model_path, output, "prune", request.pruning_amount)}
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/v1/export/torchscript")
def export_torchscript_route(request: ExportRequest) -> dict:
    try:
        tokenizer, model = load_local_model(request.model_path)
        example = tokenizer("export example", return_tensors="pt", truncation=True, max_length=request.max_sequence_length)
        output = export_torchscript(model, example, request.output_path)
        return {"schemaVersion": SCHEMA_VERSION, "status": "measured", "format": "torchscript", "artifactPath": output}
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/v1/export/onnx")
def export_onnx_route(request: ExportRequest) -> dict:
    try:
        tokenizer, model = load_local_model(request.model_path)
        example = tokenizer("export example", return_tensors="pt", truncation=True, max_length=request.max_sequence_length)
        output = export_onnx(model, example, request.output_path)
        return {"schemaVersion": SCHEMA_VERSION, "status": "measured", "format": "onnx", "artifactPath": output}
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/v1/predict")
def predict(request: PredictionRequest) -> dict:
    model_file = Path(request.model_path)
    if not model_file.exists():
        return {"schemaVersion": SCHEMA_VERSION, "status": "unavailable", "result": not_measured("Provide an existing model_path")}
    try:
        result = predict_text(model_file, request.text, request.max_sequence_length)
        return {"schemaVersion": SCHEMA_VERSION, "status": "measured", "result": result}
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/v1/dataset/inspect")
async def inspect_dataset(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a labeled CSV dataset")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        handle.write(await file.read())
        path = handle.name
    try:
        splits = load_sms_dataset(path)
        return {"schemaVersion": SCHEMA_VERSION, "measured": True, "rows": {name: len(frame) for name, frame in splits.items()}, "columns": ["text", "label"]}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        Path(path).unlink(missing_ok=True)


