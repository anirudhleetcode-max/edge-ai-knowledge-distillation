from pathlib import Path
import json
import sys

from ml_pipeline import (
    DEFAULT_STUDENT,
    DEFAULT_TEACHER,
    DistillationConfig,
    benchmark_local_model,
    build_teacher_student,
    distill_student,
    evaluate_local_model,
    evaluate_model,
    load_sms_dataset,
    tokenize_frame,
    transform_local_model,
)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "sms_spam_uci.csv"
RUN = ROOT.parent.parent / "model_runs" / "uci_sms_real_run"
STUDENT = RUN / "student"
INT8 = RUN / "student-int8"
config = DistillationConfig(
    teacher_model=DEFAULT_TEACHER,
    student_model=DEFAULT_STUDENT,
    temperature=4.0,
    alpha=0.5,
    learning_rate=3e-5,
    epochs=1,
    batch_size=32,
    max_sequence_length=64,
    random_seed=42,
)
RUN.mkdir(parents=True, exist_ok=True)
result = {
    "dataset": str(DATASET),
    "teacher": DEFAULT_TEACHER,
    "student": DEFAULT_STUDENT,
    "distillation": distill_student(config, DATASET, STUDENT),
}
_, teacher, _ = build_teacher_student(config)
teacher_tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(DEFAULT_TEACHER)
test_frame = load_sms_dataset(DATASET, seed=42)["test"]
result["teacherEvaluation"] = evaluate_model(teacher, tokenize_frame(test_frame, teacher_tokenizer, 64), 32)
result["studentEvaluation"] = evaluate_local_model(STUDENT, DATASET, 64, 32)
result["studentBenchmark"] = benchmark_local_model(STUDENT, DATASET, iterations=10, warmup=3, max_sequence_length=64)
quantized = transform_local_model(STUDENT, INT8, "quantize")
result["quantizedArtifact"] = quantized
result["int8Benchmark"] = benchmark_local_model(INT8, DATASET, iterations=10, warmup=3, max_sequence_length=64)
with (RUN / "run_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result, indent=2))
