/**
 * Deterministic fixtures so the whole UI is buildable and demoable before the
 * backend exists.
 *
 * The two worked examples from PROJECT_CONTEXT.md Sec. 1 are included verbatim
 * as the first two items, because they are the cases the project is premised
 * on: harm that neither modality reveals alone. Their numbers are set so
 * `is_emergent` is true — both unimodal scores below threshold, fused score
 * well above — which is exactly what the interface has to make legible.
 *
 * Numbers here are illustrative fixtures, not model outputs. Real measured
 * results live in reports/results/.
 */

import type {
  Attributions,
  Explanation,
  ItemDetail,
  QueueItem,
  QueueResponse,
  Stats,
} from "./types";

const now = Date.parse("2026-08-21T09:14:22Z");
const ago = (s: number) => new Date(now - s * 1000).toISOString();

export const MOCK_QUEUE: QueueItem[] = [
  {
    item_id: "itm_harassment_01",
    thumbnail_url: null,
    text_preview: "Sending them a little gift 🎁 they won't forget 😂",
    verdict: { label: "review", confidence: 0.71, priority_score: 0.88 },
    top_head: "toxicity",
    is_emergent: true,
    status: "pending",
    created_at: ago(412),
    age_seconds: 412,
  },
  {
    item_id: "itm_misinfo_02",
    thumbnail_url: null,
    text_preview:
      "This is what's happening RIGHT NOW because of the new policy — share before they delete this!!",
    verdict: { label: "review", confidence: 0.68, priority_score: 0.81 },
    top_head: "misinformation",
    is_emergent: true,
    status: "pending",
    created_at: ago(1180),
    age_seconds: 1180,
  },
  {
    item_id: "itm_slur_03",
    thumbnail_url: null,
    text_preview: "go back to where you came from, nobody wants you here",
    verdict: { label: "harmful", confidence: 0.94, priority_score: 0.79 },
    top_head: "toxicity",
    is_emergent: false,
    status: "pending",
    created_at: ago(2050),
    age_seconds: 2050,
  },
  {
    item_id: "itm_satire_04",
    thumbnail_url: null,
    text_preview: "Local man declares himself Emperor of the parking lot",
    verdict: { label: "benign", confidence: 0.83, priority_score: 0.34 },
    top_head: "misinformation",
    is_emergent: false,
    status: "pending",
    created_at: ago(3400),
    age_seconds: 3400,
  },
  {
    item_id: "itm_listing_05",
    thumbnail_url: null,
    text_preview: "Barely used, DM before it's gone 😉 no refunds, cash only",
    verdict: { label: "review", confidence: 0.62, priority_score: 0.58 },
    top_head: "misinformation",
    is_emergent: true,
    status: "pending",
    created_at: ago(5200),
    age_seconds: 5200,
  },
  {
    item_id: "itm_news_06",
    thumbnail_url: null,
    text_preview: "Flooding right now near the east ward, no one's doing anything!!",
    verdict: { label: "review", confidence: 0.55, priority_score: 0.51 },
    top_head: "misinformation",
    is_emergent: false,
    status: "pending",
    created_at: ago(7100),
    age_seconds: 7100,
  },
];

export const MOCK_QUEUE_RESPONSE: QueueResponse = {
  items: MOCK_QUEUE,
  next_cursor: null,
  total_pending: MOCK_QUEUE.length,
};

const EXPLANATIONS: Record<string, Explanation> = {
  itm_harassment_01: {
    item_id: "itm_harassment_01",
    status: "ready",
    narrative:
      "The caption's playful framing — 'a little gift', a laughing emoji — sits against footage of someone approaching a private doorway while filming covertly. Neither element is objectionable alone: the language reads as friendly banter, and the imagery shows no violence. Together they match a harassment pattern, where the sarcasm reframes a covert approach to someone's home as intimidation rather than a favour.",
    key_factors: [
      { modality: "text", factor: "sarcastic minimiser ('little gift')", weight: 0.34 },
      { modality: "image", factor: "covert approach to a residence", weight: 0.29 },
      { modality: "cross", factor: "tone contradicts depicted action", weight: 0.37 },
    ],
    model: "gemini-2.5-pro",
    generated_at: ago(400),
    latency_ms: 3120,
  },
  itm_misinfo_02: {
    item_id: "itm_misinfo_02",
    status: "ready",
    narrative:
      "The image is a crowded hospital corridor, presented as current evidence of a policy's effect. Reverse-embedding search places the same photograph in circulation roughly eight months ago, so the claim of immediacy is not supported by the image itself. The caption additionally carries urgency-pressure phrasing ('share before they delete this'), a known distribution marker rather than a claim about the facts.",
    key_factors: [
      { modality: "image", factor: "image predates the claimed event", weight: 0.41 },
      { modality: "text", factor: "urgency-pressure framing", weight: 0.28 },
      { modality: "cross", factor: "old image asserted as current", weight: 0.31 },
    ],
    model: "gemini-2.5-pro",
    generated_at: ago(1160),
    latency_ms: 2870,
  },
};

const ATTRIBUTIONS: Record<string, Attributions> = {
  itm_harassment_01: {
    item_id: "itm_harassment_01",
    text: {
      method: "shap",
      tokens: [
        { token: "Sending", score: 0.03 },
        { token: "them", score: 0.01 },
        { token: "a", score: 0.0 },
        { token: "little", score: 0.18 },
        { token: "gift", score: 0.24 },
        { token: "🎁", score: 0.06 },
        { token: "they", score: 0.02 },
        { token: "won't", score: 0.19 },
        { token: "forget", score: 0.33 },
        { token: "😂", score: 0.21 },
      ],
    },
    image: {
      method: "grad-cam",
      heatmap_url: "",
      regions: [
        { bbox: [0.41, 0.22, 0.68, 0.74], score: 0.62, label: "doorway approach" },
        { bbox: [0.08, 0.55, 0.3, 0.9], score: 0.28, label: "obscured hand" },
      ],
    },
    cross_attention: {
      available: true,
      top_links: [
        { text_token: "gift", image_region: [0.41, 0.22, 0.68, 0.74], weight: 0.28 },
        { text_token: "forget", image_region: [0.08, 0.55, 0.3, 0.9], weight: 0.19 },
      ],
    },
  },
};

function detailFor(item: QueueItem): ItemDetail {
  const isTox = item.top_head === "toxicity";
  const fused = item.verdict.confidence;
  // Emergent items are constructed so both unimodal arms sit below threshold
  // while the fused score clears it — the case the project exists to catch.
  const cv = item.is_emergent ? fused * 0.31 : fused * 0.78;
  const nlp = item.is_emergent ? fused * 0.44 : fused * 0.86;

  return {
    item_id: item.item_id,
    created_at: item.created_at,
    input: {
      text: item.text_preview,
      has_image: true,
      image_url: null,
      ocr_text: null,
      modalities: ["image", "text"],
    },
    verdict: {
      label: item.verdict.label,
      confidence: fused,
      priority_score: item.verdict.priority_score,
      recommended_action: "queue_for_review",
      auto_action: null,
    },
    heads: {
      toxicity: {
        label: isTox && fused > 0.5 ? "harmful" : "benign",
        score: isTox ? fused : 0.12,
        classes: isTox
          ? { benign: 1 - fused, harmful: fused }
          : { benign: 0.88, harmful: 0.12 },
      },
      misinformation: {
        label: !isTox && fused > 0.5 ? "misleading" : "true",
        score: !isTox ? fused : 0.09,
        classes: !isTox
          ? { true: 1 - fused, satire: 0.06, misleading: fused - 0.06 }
          : { true: 0.91, satire: 0.05, misleading: 0.04 },
      },
    },
    modality_scores: {
      cv_only: isTox ? { toxicity: cv } : { misinformation: cv },
      nlp_only: isTox ? { toxicity: nlp } : { misinformation: nlp },
      fusion: isTox ? { toxicity: fused } : { misinformation: fused },
    },
    fusion_signal: {
      is_emergent: item.is_emergent,
      delta_over_best_unimodal: fused - Math.max(cv, nlp),
      note: item.is_emergent
        ? "Neither modality alone crosses threshold; the signal appears only jointly."
        : undefined,
    },
    deepfake: { checked: true, score: 0.04, label: "authentic" },
    explanation_status: EXPLANATIONS[item.item_id] ? "ready" : "unavailable",
    latency_ms: { total: 612, cv: 210, nlp: 95, fusion: 18, ocr: 289 },
    explanation: EXPLANATIONS[item.item_id] ?? null,
    attributions: ATTRIBUTIONS[item.item_id] ?? null,
    decisions: [],
    status: item.status,
  };
}

export const MOCK_DETAILS: Record<string, ItemDetail> = Object.fromEntries(
  MOCK_QUEUE.map((i) => [i.item_id, detailFor(i)]),
);

export const MOCK_STATS: Stats = {
  queue: { pending: 6, resolved_24h: 148, median_time_to_decision_s: 96 },
  model: {
    emergent_case_rate: 0.22,
    agreement_rate: 0.84,
    explanation_useful_rate: 0.79,
  },
  distribution: {
    toxicity: { benign: 0.72, harmful: 0.28 },
    misinformation: { true: 0.61, satire: 0.14, misleading: 0.25 },
  },
};
