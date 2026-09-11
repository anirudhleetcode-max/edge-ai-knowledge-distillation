# Edge AI Model Compression Platform

A research-grade workspace for turning teacher-model signal into edge-ready evidence. The platform separates **knowledge distillation**, **quantization**, **pruning**, **evaluation**, **export**, and **benchmarking** so each technique has a clear implementation and an honest measurement boundary.

The frontend is a React/Vite observability workspace. The ML service is FastAPI/PyTorch/Transformers and is intended for Render or another Python-capable runtime. Vercel builds only the frontend/Node shell; it must not auto-detect or deploy `api/main.py` as a Python serverless function.

## Truthfulness policy

The interface never invents accuracy, F1, precision, recall, latency, throughput, memory, parameter count, compression ratio, or edge-device performance. Before a real artifact and evaluation dataset are supplied, the UI displays **Not measured**, **Requires benchmark**, or an explicit unavailable state. The local preview may show capabilities and configuration defaults, but it does not present representative ML metrics as measured results.

> A metric is authoritative only when it is returned by a completed evaluation or benchmark against a concrete artifact, dataset, and runtime.

## Architecture

```text
React/Vite frontend on Vercel
  ├─ research workspace UI and accessibility
  ├─ explicit measured/unmeasured/unavailable states
  └─ API base URL from VITE_API_BASE_URL
                 │ HTTPS JSON
                 ▼
FastAPI/PyTorch service on Render
  ├─ /health and /api/v1/metadata
  ├─ distillation config validation and training job primitive
  ├─ labeled CSV inspection and deterministic dataset splitting
  ├─ evaluation and benchmark adapters
  ├─ dynamic INT8 quantization and L1 pruning primitives
  └─ ONNX/TorchScript export helpers
                 │
                 ▼
Local or object-storage model artifacts + labeled datasets
```

## Optimization methods

**Knowledge distillation** trains a smaller student using hard labels and teacher logits. The implemented objective is:

```text
L = alpha * CE(y, student_logits)
  + (1 - alpha) * T^2 * KL(teacher_logits / T || student_logits / T)
```

`DistillationConfig` validates teacher and student identifiers, temperature, alpha, learning rate, epochs, batch size, sequence length, weight decay, warmup ratio, and deterministic seed. `distill_student()` loads a labeled CSV, tokenizes it, performs teacher inference with gradients disabled, trains the student, and writes a Hugging Face-compatible artifact plus `distillation_config.json` containing the run history.

**Quantization** uses dynamic `torch.qint8` quantization for linear layers. It changes inference representation and execution; it is not distillation. **Pruning** applies L1 unstructured pruning to linear weights and removes the reparameterization before saving. These transformations are independently callable and can be evaluated as separate variants.

**Export** helpers are provided for TorchScript and ONNX. Export is only meaningful after a real model artifact is loaded and an example input has been prepared. The UI therefore keeps export actions visible but reports that an artifact is required rather than pretending an export completed.

## Dataset pipeline

`load_sms_dataset()` accepts a user-supplied CSV with `text` and `label` columns, validates non-empty rows and binary labels, and creates deterministic stratified train, validation, and test splits using the configured seed. The API includes `/api/v1/dataset/inspect` for checking an uploaded CSV. No arbitrary dataset is silently downloaded by the service.

For local work, place a documented dataset at `data/sms_spam.csv` and keep it outside the repository if it contains restricted or licensed data. Production deployments should use authenticated object storage and pass a controlled artifact reference instead of an unrestricted local filesystem path.

## Evaluation and benchmark policy

Evaluation is expected to compute accuracy, precision, recall, F1, and a confusion matrix from model predictions and labels. The service evaluates local Hugging Face sequence-classification directories with their tokenizer and `test` split, and benchmarks the same artifact against a generated dynamic-INT8 directory. If an artifact or dataset path is missing, it returns an explicit unavailable/unmeasured response instead of inventing metrics.

Benchmarking reports CPU wall-clock samples only after warm-up. Median, p95, minimum, maximum, sample count, and warm-up count are recorded. Model-size values are serialized-weight proxies, not GPU VRAM, process RSS, or a deployment guarantee. Raspberry Pi, Jetson, Android, GPU, and other edge claims require separate measured runners and are never inferred from a local CPU run.

## API routes

| Route | Purpose | Result policy |
|---|---|---|
| `GET /health` | Render health check | Service status only |
| `GET /api/v1/metadata` | Capabilities and metric policy | No ML metrics |
| `POST /api/v1/distillation/validate` | Validate hyperparameters and show loss formula | Deterministic validation |
| `POST /api/v1/distillation/run` | Train a student on a supplied labeled CSV | Real artifact or explicit unavailable/error |
| `POST /api/v1/dataset/inspect` | Validate uploaded CSV and report split row counts | Measured dataset structure |
| `POST /api/v1/predict` | Classify one message with a local artifact | Label and confidence only after real inference |
| `POST /api/v1/evaluate` | Evaluate a local Hugging Face artifact against a labeled CSV | Accuracy, precision, recall, F1, and confusion matrix after real evaluation |
| `POST /api/v1/benchmark` | Benchmark FP32 and generated INT8 artifact | CPU timing only after a real benchmark |
| `POST /api/v1/optimization/quantize` | Save a dynamic INT8 artifact | Artifact metadata after transformation |
| `POST /api/v1/optimization/prune` | Save an L1-pruned artifact | Artifact metadata after transformation |
| `POST /api/v1/export/onnx` | Export a local artifact to ONNX | Export path only after successful export |
| `POST /api/v1/export/torchscript` | Export a local artifact to TorchScript | Export path only after successful export |

## Local development

```bash
pnpm install
pnpm run check
pnpm test
pnpm run build
pnpm dev
```

The web shell runs independently of the Python service and shows `API unavailable` when `VITE_API_BASE_URL` is not configured. To run the ML service locally:

```bash
cd api
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
$env:FRONTEND_ORIGINS="http://localhost:3000"  # PowerShell
# export FRONTEND_ORIGINS=http://localhost:3000 # macOS/Linux
uvicorn main:app --reload --port 8000
```

Set `VITE_API_BASE_URL=http://localhost:8000` for the frontend when live metadata and benchmark requests are desired. The frontend should not be treated as evidence of model quality until a real dataset and artifact are connected.

## Deployment

**Vercel:** deploy the React/Vite/Node shell only. `vercel.json` pins Corepack/pnpm, uses the frozen lockfile, and runs typecheck, tests, and the production build. Set `VITE_API_BASE_URL` to the public FastAPI origin. If the Vercel dashboard has a custom `npm install --legacy-peer-deps` command, remove it so the committed pnpm lockfile remains authoritative.

**Render:** deploy `api/` with its Dockerfile or a Python service. Set `FRONTEND_ORIGINS` to the exact Vercel origin and use `/health` for health checks. Model downloads and distillation jobs can be slow or memory-intensive; long-running training should move to a worker/job system rather than a short request lifecycle. Keep artifact storage authenticated and do not expose unrestricted filesystem paths in a public API.

## Reproducing the real SMS-spam measured run

The repository includes a real end-to-end run path using the public UCI SMS Spam Collection and the specified Hugging Face teacher/student checkpoints. Install the Python dependencies from `api/requirements.txt` on a CPU-capable machine with internet access; the first run downloads the model weights from Hugging Face and the dataset from UCI. Then run `python api/prepare_uci_sms.py` followed by `PYTHONPATH=api python api/run_real_sms_experiment.py` from the repository root. The workflow performs one deterministic distillation epoch, evaluates the teacher and trained student on the held-out split, creates a dynamic `torch.qint8` linear-layer artifact, and measures CPU latency after warm-up. Training and model conversion are synchronous local primitives and can require several minutes and multiple gigabytes of working memory; production deployments should move these jobs to a worker with authenticated artifact storage.

The authoritative output from the completed local run is stored in `api/real_run_summary.json`. It records the dataset path, checkpoint identifiers, training loss history, held-out accuracy/precision/recall/F1, confusion matrices, parameter/model-size proxies, and warm-up-aware CPU benchmark measurements. These values are machine-specific and must not be presented as universal performance claims.

## Verification

The repository includes Vitest coverage for the Node shell, TypeScript checks, Python syntax validation, FastAPI route contracts, and a production bundle command. Install `api/requirements.txt` before running the Python suite; otherwise the tests deliberately skip rather than pretending the ML environment exists. Run `PYTHONPATH=api python -m unittest discover -s api -p 'test_*.py'` to execute route and pipeline coverage. Add adapter-specific fixtures with a cached model artifact before enabling a production CI gate; those tests should assert metric keys, confusion-matrix shape, artifact existence, and the explicit unmeasured state when inputs are missing.

## Known limitations

The default UI is an honest research workspace, not a completed training run. It does not claim that a student has been trained, that a dataset has been evaluated, or that a target device has been benchmarked until those jobs run against real inputs. The current distillation job primitive is synchronous and should be moved behind a queued worker for production-scale training. Evaluation, benchmark, export, and optimization routes are intended for trusted internal artifact paths or an authenticated object-storage adapter, not arbitrary public filesystem access. Confidence is not calibration, and prediction agreement is not a substitute for a labeled test set.


## Internship-ready workspace additions

The frontend now presents the complete compression workflow without fabricating measurements. It includes validated distillation controls for temperature, alpha, learning rate, batch size, epochs, and maximum sequence length; evaluation summaries with accuracy, precision, recall, F1, and a confusion-matrix view; a prediction playground; CPU benchmark statistics; a compression pipeline; experiment details; a local saved-run history; a truthful Pareto-frontier empty state; and estimated edge profiles that are explicitly not hardware measurements.

The FastAPI service exposes `POST /api/v1/predict` for a real local model artifact. It returns the predicted ham/spam label, confidence, and class probabilities only when the artifact can be loaded. Missing artifacts remain an explicit unavailable response. The API continues to expose the real distillation, evaluation, benchmark, quantization, pruning, dataset-inspection, TorchScript, and ONNX routes.

Experiment history is intentionally stored in browser local storage for this deployment-safe workspace. It records completed API responses from the current browser and is not a server-side experiment database. A production deployment should replace this with authenticated database-backed records or object-storage manifests. The trade-off and Pareto panels remain empty until enough structured measured experiment results are available.

Benchmark reports distinguish median latency, p95 latency, repeated-run count, runtime, batch size, and input length. Throughput is shown as unavailable because the current API does not return a measured throughput value. Edge profiles are planning estimates only; no mobile, low-power, or memory-constrained hardware claim is made.
