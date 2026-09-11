/**
 * Central FastAPI client for Edge AI Knowledge Distillation.
 *
 * The frontend talks to the FastAPI backend through this module.
 *
 * Production:
 *   VITE_API_BASE_URL=https://your-render-service.onrender.com
 *
 * Local development:
 *   VITE_API_BASE_URL=http://127.0.0.1:8000
 *
 * Do not put secrets/API keys in this file.
 */

const configuredBaseUrl =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || "";

export const API_BASE_URL = configuredBaseUrl.replace(/\/+$/, "");

export class ApiError extends Error {
  status: number;
  details: unknown;

  constructor(
    message: string,
    status = 0,
    details: unknown = undefined,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

function getApiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  if (!API_BASE_URL) {
    throw new ApiError(
      "FastAPI backend is not configured. Set VITE_API_BASE_URL in the frontend deployment environment.",
      0,
    );
  }

  return `${API_BASE_URL}${normalizedPath}`;
}

async function parseResponse(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  try {
    return await response.text();
  } catch {
    return null;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const url = getApiUrl(path);

  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 30000);

  try {
    const response = await fetch(url, {
      ...options,
      signal: options.signal ?? controller.signal,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
    });

    const data = await parseResponse(response);

    if (!response.ok) {
      let message = `FastAPI request failed with HTTP ${response.status}.`;

      if (typeof data === "object" && data !== null) {
        const body = data as Record<string, unknown>;

        if (typeof body.detail === "string") {
          message = body.detail;
        } else if (typeof body.message === "string") {
          message = body.message;
        }
      } else if (typeof data === "string" && data.trim()) {
        message = data;
      }

      throw new ApiError(message, response.status, data);
    }

    return data as T;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(
        "The FastAPI backend request timed out. Please check whether the backend is running.",
        408,
      );
    }

    if (error instanceof TypeError) {
      throw new ApiError(
        "Unable to reach the FastAPI backend. Check VITE_API_BASE_URL, Render status, and CORS configuration.",
        0,
        error,
      );
    }

    throw new ApiError(
      error instanceof Error
        ? error.message
        : "An unexpected API error occurred.",
      0,
      error,
    );
  } finally {
    window.clearTimeout(timeout);
  }
}

/**
 * Generic GET request.
 */
export function apiGet<T>(path: string): Promise<T> {
  return request<T>(path, {
    method: "GET",
  });
}

/**
 * Generic POST request.
 */
export function apiPost<T>(
  path: string,
  body?: unknown,
): Promise<T> {
  return request<T>(path, {
    method: "POST",
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

/**
 * Generic PUT request.
 */
export function apiPut<T>(
  path: string,
  body?: unknown,
): Promise<T> {
  return request<T>(path, {
    method: "PUT",
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

/**
 * Generic DELETE request.
 */
export function apiDelete<T>(path: string): Promise<T> {
  return request<T>(path, {
    method: "DELETE",
  });
}

/**
 * Backend health check.
 */
export async function getHealth<T = unknown>(): Promise<T> {
  return apiGet<T>("/health");
}

/**
 * Model/application metadata.
 */
export async function getMetadata<T = unknown>(): Promise<T> {
  return apiGet<T>("/api/v1/metadata");
}

/**
 * Model evaluation.
 *
 * The backend can evolve its exact request/response schema without
 * requiring the frontend client architecture to change.
 */
export async function evaluateModel<T = unknown>(
  payload: unknown,
): Promise<T> {
  return apiPost<T>("/api/v1/evaluate", payload);
}

/**
 * Benchmark models.
 */
export async function benchmarkModels<T = unknown>(
  payload: unknown,
): Promise<T> {
  return apiPost<T>("/api/v1/benchmark", payload);
}

/**
 * Run knowledge distillation.
 */
export async function runDistillation<T = unknown>(
  payload: unknown,
): Promise<T> {
  return apiPost<T>("/api/v1/distill", payload);
}

/**
 * Inspect a dataset.
 */
export async function inspectDataset<T = unknown>(
  payload: unknown,
): Promise<T> {
  return apiPost<T>("/api/v1/dataset/inspect", payload);
}

/**
 * Export a generated model/artifact.
 */
export async function exportModel<T = unknown>(
  payload: unknown,
): Promise<T> {
  return apiPost<T>("/api/v1/export", payload);
}

export default {
  API_BASE_URL,
  apiGet,
  apiPost,
  apiPut,
  apiDelete,
  getHealth,
  getMetadata,
  evaluateModel,
  benchmarkModels,
  runDistillation,
  inspectDataset,
  exportModel,
};
