/** Shared product contracts for the Edge / Inference console. */
export type * from "../drizzle/schema";
export * from "./_core/errors";

export const BENCHMARK_SCHEMA_VERSION = "1.0" as const;
export const MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection" as const;

export type RuntimeStatus = "reachable" | "unavailable" | "loading" | "stale" | "loaded" | "available";
export type BenchmarkMode = "live" | "demo";
export type PredictionLabel = "spam" | "ham";

export interface RuntimeMetadata {
  schemaVersion: typeof BENCHMARK_SCHEMA_VERSION;
  modelId: typeof MODEL_ID;
  task: "text-classification";
  device: "cpu";
  quantization: "dynamic-qint8-linear";
  apiStatus: RuntimeStatus;
  modelStatus: RuntimeStatus;
  supportedTokenLength: { min: number; max: number; recommended: number };
  supportedIterations: { min: number; max: number; recommended: number };
}

export interface LatencyStats {
  medianMs: number;
  p95Ms: number;
  minMs?: number;
  maxMs?: number;
  sampleCount: number;
}

export interface VariantMetrics {
  variant: "fp32" | "int8";
  latency: LatencyStats;
  serializedSizeMb: number;
}

export interface BenchmarkResult {
  schemaVersion: typeof BENCHMARK_SCHEMA_VERSION;
  mode: BenchmarkMode;
  runId: string;
  modelId: typeof MODEL_ID;
  tokenLength: number;
  measuredIterations: number;
  warmupIterations: number;
  runtime: "cpu";
  fp32: VariantMetrics;
  int8: VariantMetrics;
  speedup: number;
  latencyDeltaPercent: number;
  sizeDeltaPercent: number;
  labelAgreementPercent?: number;
  startedAt: string;
  completedAt: string;
  limitations: string[];
}

export interface PredictionVariant {
  variant: "fp32" | "int8";
  label: PredictionLabel;
  confidence: number;
  probabilities: Record<PredictionLabel, number>;
}

export interface DualPredictionResult {
  schemaVersion: typeof BENCHMARK_SCHEMA_VERSION;
  mode: BenchmarkMode;
  modelId: typeof MODEL_ID;
  input: string;
  fp32: PredictionVariant;
  int8: PredictionVariant;
  agrees: boolean;
  limitations: string[];
}

export interface BenchmarkRequest {
  tokenLength: number;
  measuredIterations: number;
  signal?: AbortSignal;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  service: "edge-inference-benchmark";
  schemaVersion: typeof BENCHMARK_SCHEMA_VERSION;
  modelLoaded: boolean;
  quantizedPathAvailable: boolean;
}
