"""Real ML workflows for the Edge AI model-compression platform.

No metrics are invented here: evaluation and benchmark functions return values
only after running against a supplied dataset or artifact. The API layer may
report "Not measured" when an artifact or dataset is not available.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils import prune
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_TEACHER = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
DEFAULT_STUDENT = "prajjwal1/bert-tiny"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_TEXT_COLUMN = "text"


@dataclass(frozen=True)
class DistillationConfig:
    teacher_model: str = DEFAULT_TEACHER
    student_model: str = DEFAULT_STUDENT
    temperature: float = 4.0
    alpha: float = 0.5
    learning_rate: float = 3e-5
    epochs: int = 3
    batch_size: int = 16
    max_sequence_length: int = 128
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    random_seed: int = 42

    def validate(self) -> None:
        if not 1.0 <= self.temperature <= 20.0:
            raise ValueError("temperature must be between 1 and 20")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 1e-7 <= self.learning_rate <= 1e-2:
            raise ValueError("learning_rate must be between 1e-7 and 1e-2")
        if not 1 <= self.epochs <= 100:
            raise ValueError("epochs must be between 1 and 100")
        if not 1 <= self.batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256")
        if not 16 <= self.max_sequence_length <= 512:
            raise ValueError("max_sequence_length must be between 16 and 512")
        if not 0.0 <= self.weight_decay <= 1.0:
            raise ValueError("weight_decay must be between 0 and 1")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if not 0 <= self.random_seed <= 2**32 - 1:
            raise ValueError("random_seed is out of range")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sms_dataset(path: str | Path, text_column: str = DEFAULT_TEXT_COLUMN, label_column: str = DEFAULT_LABEL_COLUMN, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Load a user-supplied CSV and return deterministic train/validation/test splits."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset file not found: {source}")
    frame = pd.read_csv(source)
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame[[text_column, label_column]].dropna().copy()
    frame[text_column] = frame[text_column].astype(str).str.strip()
    frame = frame[frame[text_column].ne("")]
    frame[label_column] = frame[label_column].astype(int)
    if frame[label_column].nunique() < 2:
        raise ValueError("Dataset must contain at least two labels")
    if len(frame) < 10 or frame[label_column].value_counts().min() < 2:
        raise ValueError("Dataset must contain at least 10 rows and two examples per label for deterministic stratified splits")
    train, remainder = train_test_split(frame, test_size=0.2, random_state=seed, stratify=frame[label_column])
    validation, test = train_test_split(remainder, test_size=0.5, random_state=seed, stratify=remainder[label_column])
    return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True), "test": test.reset_index(drop=True)}


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, temperature: float, alpha: float) -> tuple[torch.Tensor, dict[str, float]]:
    """L = alpha * hard CE + (1-alpha) * T^2 * KL(teacher || student)."""
    hard_loss = F.cross_entropy(student_logits, labels)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)
    total = alpha * hard_loss + (1.0 - alpha) * soft_loss
    return total, {"total": float(total.detach()), "hardCrossEntropy": float(hard_loss.detach()), "softKLDivergence": float(soft_loss.detach())}


def build_teacher_student(config: DistillationConfig, num_labels: int = 2):
    config.validate()
    tokenizer = AutoTokenizer.from_pretrained(config.student_model)
    teacher = AutoModelForSequenceClassification.from_pretrained(config.teacher_model, num_labels=num_labels).eval()
    student = AutoModelForSequenceClassification.from_pretrained(config.student_model, num_labels=num_labels)
    return tokenizer, teacher, student


def tokenize_frame(frame: pd.DataFrame, tokenizer, max_length: int, text_column: str = DEFAULT_TEXT_COLUMN) -> dict[str, torch.Tensor]:
    encoded = tokenizer(frame[text_column].tolist(), truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    encoded["labels"] = torch.tensor(frame[DEFAULT_LABEL_COLUMN].tolist(), dtype=torch.long)
    return encoded


def evaluate_model(model: nn.Module, encoded: dict[str, torch.Tensor], batch_size: int = 16) -> dict:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(encoded["labels"]), batch_size):
            batch = {key: value[start : start + batch_size] for key, value in encoded.items()}
            logits_parts.append(model(**{key: value for key, value in batch.items() if key != "labels"}).logits.cpu())
    logits = torch.cat(logits_parts)
    predictions = logits.argmax(dim=-1).numpy()
    labels = encoded["labels"].numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {"measured": True, "sampleCount": int(len(labels)), "accuracy": float(accuracy_score(labels, predictions)), "precision": float(precision), "recall": float(recall), "f1": float(f1), "confusionMatrix": confusion_matrix(labels, predictions).tolist()}


def quantize_dynamic(model: nn.Module) -> nn.Module:
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8).eval()


def prune_linear_layers(model: nn.Module, amount: float = 0.2) -> nn.Module:
    if not 0.0 < amount < 1.0:
        raise ValueError("pruning amount must be between 0 and 1")
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model.eval()


def export_torchscript(model: nn.Module, example_inputs: dict[str, torch.Tensor], destination: str | Path) -> str:
    destination = str(destination)
    model.eval()
    traced = torch.jit.trace(lambda input_ids, attention_mask: model(input_ids=input_ids, attention_mask=attention_mask).logits, (example_inputs["input_ids"], example_inputs["attention_mask"]))
    traced.save(destination)
    return destination


def export_onnx(model: nn.Module, example_inputs: dict[str, torch.Tensor], destination: str | Path, opset: int = 17) -> str:
    destination = str(destination)
    model.eval()
    torch.onnx.export(model, (example_inputs["input_ids"], example_inputs["attention_mask"]), destination, input_names=["input_ids", "attention_mask"], output_names=["logits"], dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}}, opset_version=opset)
    return destination


def benchmark_model(model: nn.Module, encoded: dict[str, torch.Tensor], iterations: int = 40, warmup: int = 8) -> dict:
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")
    model.eval()
    example = {key: value[:1] for key, value in encoded.items() if key != "labels"}
    with torch.inference_mode():
        for _ in range(warmup):
            model(**example)
        samples: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            model(**example)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
    ordered = sorted(samples)
    return {"measured": True, "sampleCount": iterations, "warmupIterations": warmup, "medianMs": float(np.median(samples)), "p95Ms": float(ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))]), "minMs": float(min(samples)), "maxMs": float(max(samples)), "runtime": "cpu"}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / (1024 * 1024)


def metric_state(value: float | None) -> float | str:
    return value if value is not None else "Not measured"


def distill_student(config: DistillationConfig, dataset_path: str | Path, output_dir: str | Path) -> dict:
    """Train a student with hard labels plus temperature-scaled teacher targets.

    This is intentionally a local/offline-capable job primitive. Callers should
    run it in a worker or job queue for production workloads rather than inside
    a short HTTP request.
    """
    config.validate()
    seed_everything(config.random_seed)
    splits = load_sms_dataset(dataset_path, seed=config.random_seed)
    tokenizer, teacher, student = build_teacher_student(config)
    train_encoded = tokenize_frame(splits["train"], tokenizer, config.max_sequence_length)
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        student.train()
        epoch_losses: list[dict[str, float]] = []
        order = torch.randperm(len(train_encoded["labels"]))
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            batch = {key: value[indices] for key, value in train_encoded.items()}
            with torch.inference_mode():
                teacher_logits = teacher(**{key: value for key, value in batch.items() if key != "labels"}).logits
            student_logits = student(**{key: value for key, value in batch.items() if key != "labels"}).logits
            loss, detail = distillation_loss(student_logits, teacher_logits, batch["labels"], config.temperature, config.alpha)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(detail)
        history.append({"epoch": epoch + 1, **{key: float(np.mean([row[key] for row in epoch_losses])) for key in epoch_losses[0]}})
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    with (destination / "distillation_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": asdict(config), "history": history}, handle, indent=2)
    return {"measured": True, "artifactPath": str(destination), "epochs": config.epochs, "trainRows": len(splits["train"]), "history": history, "parameterCount": parameter_count(student), "modelSizeMb": model_size_mb(student)}


def load_local_model(model_path: str | Path):
    """Load a local Hugging Face directory or a serialized transformed Torch module."""
    source = Path(model_path)
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Model directory not found: {source}")
    tokenizer = AutoTokenizer.from_pretrained(source)
    serialized = source / "model.pt"
    if serialized.exists():
        model = torch.load(serialized, map_location="cpu", weights_only=False).eval()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(source).eval()
    return tokenizer, model


def evaluate_local_model(model_path: str | Path, dataset_path: str | Path, max_sequence_length: int = 128, batch_size: int = 16) -> dict:
    tokenizer, model = load_local_model(model_path)
    splits = load_sms_dataset(dataset_path)
    encoded = tokenize_frame(splits["test"], tokenizer, max_sequence_length)
    result = evaluate_model(model, encoded, batch_size)
    return {"modelPath": str(model_path), "datasetPath": str(dataset_path), "split": "test", "parameters": parameter_count(model), "modelSizeMb": model_size_mb(model), **result}


def benchmark_local_model(model_path: str | Path, dataset_path: str | Path, iterations: int = 40, warmup: int = 8, max_sequence_length: int = 128) -> dict:
    tokenizer, model = load_local_model(model_path)
    splits = load_sms_dataset(dataset_path)
    encoded = tokenize_frame(splits["test"], tokenizer, max_sequence_length)
    return {"modelPath": str(model_path), "datasetPath": str(dataset_path), "parameters": parameter_count(model), "modelSizeMb": model_size_mb(model), **benchmark_model(model, encoded, iterations, warmup)}


def transform_local_model(model_path: str | Path, output_dir: str | Path, method: str, pruning_amount: float = 0.2) -> dict:
    tokenizer, model = load_local_model(model_path)
    if method == "quantize":
        transformed = quantize_dynamic(model)
    elif method == "prune":
        transformed = prune_linear_layers(model, pruning_amount)
    else:
        raise ValueError("method must be quantize or prune")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(destination)
    if method == "quantize":
        torch.save(transformed, destination / "model.pt")
    else:
        transformed.save_pretrained(destination)
    return {"measured": True, "method": method, "artifactPath": str(destination), "parameters": parameter_count(transformed), "modelSizeMb": model_size_mb(transformed)}


def predict_text(model_path: str | Path, text: str, max_sequence_length: int = 128) -> dict:
    """Run one real prediction against a local model artifact."""
    if not text.strip():
        raise ValueError("text must not be empty")
    tokenizer, model = load_local_model(model_path)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_sequence_length)
    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
    label_index = int(probabilities.argmax().item())
    return {
        "measured": True,
        "label": "spam" if label_index == 1 else "ham",
        "labelIndex": label_index,
        "confidence": float(probabilities[label_index].item()),
        "probabilities": {"ham": float(probabilities[0].item()), "spam": float(probabilities[1].item())},
        "modelPath": str(model_path),
    }
