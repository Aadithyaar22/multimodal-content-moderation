/**
 * Typed client for the moderation API (docs/api.md).
 *
 * Talks to the deployed API by default. Set NEXT_PUBLIC_USE_MOCK=true to run
 * against fixtures instead, which is useful for UI work with no backend up.
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

/**
 * Deployed API, baked in as the default.
 *
 * Neither of these is a secret. The browser must know the API origin to call it
 * at all, so it appears in network traffic no matter how it is stored — keeping
 * it in a dashboard buys nothing and costs a deployment step that is easy to get
 * wrong. Marking it "sensitive" there actively breaks the build, because Vercel
 * then refuses to inline it into client JS and the app silently falls back to
 * fixtures.
 *
 * Real secrets — GROQ_API_KEY, GEMINI_API_KEY, MONGODB_URI — live on Cloud Run
 * and are never sent to the browser.
 *
 * Both remain overridable by environment for local development and for anyone
 * pointing the frontend at their own backend.
 */
const DEFAULT_API_BASE =
  "https://vanguard-moderation-api-2wr445ogxq-el.a.run.app";

const BASE = process.env.NEXT_PUBLIC_API_BASE || DEFAULT_API_BASE;

// Opt *in* to fixtures. Defaulting to mock meant a deployment that forgot the
// variable looked like it worked while serving invented numbers, which is the
// worse failure: a broken API is obvious, fabricated data is not.
export const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK === "true";

/**
 * Resolve a server-relative path against the API origin.
 *
 * The backend returns image_url as "/api/v1/items/{id}/image". Rendered
 * directly it resolves against the *frontend* origin, so every evidence image
 * 404s once the app is pointed at a separately-hosted API — which is the
 * deployed topology, Vercel in front of Render.
 */
export function resolveApiUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  if (/^(https?:|data:|blob:)/.test(path)) return path;
  return `${BASE}${path.startsWith("/") ? "" : "/"}${path}`;
}

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Results minted by a mock analyze() run, keyed by their generated id.
 *
 * Without this, a submission gets a fresh item_id that is absent from the
 * fixtures, so the follow-up explanation and detail lookups miss and the demo
 * surface reports "no narrative available" for every item a viewer submits —
 * precisely the screen where the reasoning most needs to appear.
 */
const mockLiveItems = new Map<string, ItemDetail>();

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
    const item = MOCK_DETAILS[itemId] ?? mockLiveItems.get(itemId);
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
      (MOCK_DETAILS[itemId] ?? mockLiveItems.get(itemId))?.explanation ?? {
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
      (MOCK_DETAILS[itemId] ?? mockLiveItems.get(itemId))?.attributions ?? {
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
    const hasImage = form.get("image") instanceof File;
    const base = MOCK_DETAILS["itm_harassment_01"];
    const item: ItemDetail = {
      ...base,
      item_id: `itm_live_${Date.now().toString(36)}`,
      created_at: new Date().toISOString(),
      input: {
        ...base.input,
        text: text || base.input.text,
        has_image: hasImage,
        modalities: hasImage ? ["image", "text"] : ["text"],
      },
    };
    mockLiveItems.set(item.item_id, item);
    return item;
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
