/**
 * Typed client for the moderation API (docs/api.md).
 *
 * Runs against fixtures when NEXT_PUBLIC_USE_MOCK is set, which is the default
 * in development. That is deliberate: the backend is not deployed yet, and a
 * frontend that cannot be built or demoed until it is would block all UI work.
 */

import {
  MOCK_DETAILS,
  MOCK_QUEUE_RESPONSE,
  MOCK_STATS,
} from "./mock";
import type {
  Attributions,
  DecisionAction,
  DecisionResponse,
  Explanation,
  Health,
  ItemDetail,
  QueueFilters,
  QueueResponse,
  Stats,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
export const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK !== "false";

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

class ApiClientError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly retryAfter?: number,
  ) {
    super(message);
    this.name = "ApiClientError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}/api/v1${path}`, {
    ...init,
    headers: { Accept: "application/json", ...init?.headers },
  });

  if (!res.ok) {
    // 503 means the container is awake but weights are still loading. It is a
    // routine state on a cold start, not a failure, so the retry hint is
    // surfaced rather than swallowed.
    const retryAfter = Number(res.headers.get("Retry-After")) || undefined;
    let message = res.statusText;
    try {
      const body = await res.json();
      message = body?.error?.message ?? message;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiClientError(message, res.status, retryAfter);
  }

  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<Health> {
  if (USE_MOCK) {
    await delay(120);
    return {
      status: "ok",
      models_loaded: true,
      warm: true,
      device: "mock",
      version: "0.1.0-mock",
      loaded_at: new Date().toISOString(),
    };
  }
  return request<Health>("/health");
}

export async function getQueue(filters: QueueFilters = {}): Promise<QueueResponse> {
  if (USE_MOCK) {
    await delay(180);
    let items = MOCK_QUEUE_RESPONSE.items;
    if (filters.emergent_only) items = items.filter((i) => i.is_emergent);
    if (filters.head) items = items.filter((i) => i.top_head === filters.head);
    if (filters.min_priority != null) {
      items = items.filter((i) => i.verdict.priority_score >= filters.min_priority!);
    }
    // The ranking is the product: never fall back to chronological order.
    items = [...items].sort(
      (a, b) => b.verdict.priority_score - a.verdict.priority_score,
    );
    return { items, next_cursor: null, total_pending: items.length };
  }

  const params = new URLSearchParams();
  Object.entries(filters).forEach(([k, v]) => {
    if (v !== undefined && v !== null) params.set(k, String(v));
  });
  return request<QueueResponse>(`/queue?${params}`);
}

export async function getItem(itemId: string): Promise<ItemDetail> {
  if (USE_MOCK) {
    await delay(220);
    const item = MOCK_DETAILS[itemId];
    if (!item) throw new ApiClientError("item not found", 404);
    return item;
  }
  return request<ItemDetail>(`/items/${itemId}`);
}

export async function getExplanation(itemId: string): Promise<Explanation> {
  if (USE_MOCK) {
    // Deliberately slow, so the split-render path is exercised in development
    // rather than only discovered against the real LLM.
    await delay(1800);
    return (
      MOCK_DETAILS[itemId]?.explanation ?? {
        item_id: itemId,
        status: "unavailable",
        narrative: null,
        key_factors: [],
        model: null,
        generated_at: null,
        latency_ms: null,
      }
    );
  }
  return request<Explanation>(`/items/${itemId}/explanation`);
}

export async function getAttributions(itemId: string): Promise<Attributions> {
  if (USE_MOCK) {
    await delay(400);
    return (
      MOCK_DETAILS[itemId]?.attributions ?? {
        item_id: itemId,
        text: null,
        image: null,
        cross_attention: null,
      }
    );
  }
  return request<Attributions>(`/items/${itemId}/attributions`);
}

export async function submitDecision(
  itemId: string,
  body: {
    action: DecisionAction;
    moderator_id: string;
    rationale?: string;
    agreed_with_model?: boolean;
    explanation_was_useful?: boolean;
  },
): Promise<DecisionResponse> {
  if (USE_MOCK) {
    await delay(260);
    return {
      item_id: itemId,
      status: "resolved",
      action: body.action,
      decided_at: new Date().toISOString(),
      time_to_decision_seconds: 138,
    };
  }
  return request<DecisionResponse>(`/items/${itemId}/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function analyze(form: FormData): Promise<ItemDetail> {
  if (USE_MOCK) {
    await delay(900);
    const text = String(form.get("text") ?? "");
    const base = MOCK_DETAILS["itm_harassment_01"];
    return {
      ...base,
      item_id: `itm_live_${Date.now().toString(36)}`,
      created_at: new Date().toISOString(),
      input: { ...base.input, text: text || base.input.text },
    };
  }
  return request<ItemDetail>("/analyze", { method: "POST", body: form });
}

export async function getStats(): Promise<Stats> {
  if (USE_MOCK) {
    await delay(150);
    return MOCK_STATS;
  }
  return request<Stats>("/stats");
}

export { ApiClientError };
