/**
 * Mirrors docs/api.md exactly. Keep the two in step — this file is the only
 * place the wire format is described on the frontend.
 */

export type VerdictLabel = "benign" | "review" | "harmful";
export type ToxicityClass = "benign" | "harmful";
export type MisinfoClass = "true" | "satire" | "misleading";
export type DecisionAction = "approve" | "remove" | "escalate" | "defer";
export type ItemStatus = "pending" | "resolved";
export type ExplanationStatus = "pending" | "ready" | "failed" | "unavailable";

export interface Health {
  status: string;
  models_loaded: boolean;
  warm: boolean;
  device: string;
  version: string;
  loaded_at: string;
}

export interface HeadScore<T extends string> {
  label: T;
  score: number;
  classes: Record<T, number>;
}

/**
 * The project's central claim in wire form.
 *
 * `is_emergent` marks items where the fused score materially exceeds both
 * unimodal scores — harm that exists only in the combination. This is the field
 * the UI is built around, not a detail.
 */
export interface FusionSignal {
  is_emergent: boolean;
  delta_over_best_unimodal: number;
  note?: string;
}

export interface ModalityScores {
  cv_only: { toxicity?: number; misinformation?: number };
  nlp_only: { toxicity?: number; misinformation?: number };
  fusion: { toxicity?: number; misinformation?: number };
}

export interface AnalysisResult {
  item_id: string;
  created_at: string;
  input: {
    text: string;
    has_image: boolean;
    image_url: string | null;
    ocr_text: string | null;
    modalities: Array<"image" | "text">;
  };
  verdict: {
    label: VerdictLabel;
    confidence: number;
    priority_score: number;
    recommended_action: string;
    /** Always null. The system never acts on its own; see docs/api.md. */
    auto_action: null;
  };
  heads: {
    toxicity: HeadScore<ToxicityClass>;
    misinformation: HeadScore<MisinfoClass>;
  };
  modality_scores: ModalityScores;
  fusion_signal: FusionSignal;
  deepfake: { checked: boolean; score: number; label: string };
  explanation_status: ExplanationStatus;
  latency_ms: Record<string, number>;
}

export interface Explanation {
  item_id: string;
  status: ExplanationStatus;
  narrative: string | null;
  key_factors: Array<{
    modality: "text" | "image" | "cross";
    factor: string;
    weight: number;
  }>;
  model: string | null;
  generated_at: string | null;
  latency_ms: number | null;
}

export interface Attributions {
  item_id: string;
  text: {
    method: string;
    /** Signed: positive pushes toward harmful, negative toward benign. */
    tokens: Array<{ token: string; score: number }>;
  } | null;
  image: {
    method: string;
    heatmap_url: string;
    regions: Array<{ bbox: [number, number, number, number]; score: number; label: string }>;
  } | null;
  cross_attention: {
    available: boolean;
    top_links: Array<{
      text_token: string;
      image_region: [number, number, number, number];
      weight: number;
    }>;
  } | null;
}

export interface QueueItem {
  item_id: string;
  thumbnail_url: string | null;
  text_preview: string;
  verdict: { label: VerdictLabel; confidence: number; priority_score: number };
  top_head: "toxicity" | "misinformation";
  is_emergent: boolean;
  status: ItemStatus;
  created_at: string;
  age_seconds: number;
}

export interface QueueResponse {
  items: QueueItem[];
  next_cursor: string | null;
  total_pending: number;
}

export interface QueueFilters {
  status?: ItemStatus | "all";
  min_priority?: number;
  head?: "toxicity" | "misinformation";
  emergent_only?: boolean;
  limit?: number;
  cursor?: string;
}

export interface ItemDetail extends AnalysisResult {
  explanation: Explanation | null;
  attributions: Attributions | null;
  decisions: Decision[];
  status: ItemStatus;
}

export interface Decision {
  action: DecisionAction;
  moderator_id: string;
  rationale?: string;
  agreed_with_model?: boolean;
  explanation_was_useful?: boolean;
  decided_at: string;
}

export interface DecisionResponse {
  item_id: string;
  status: ItemStatus;
  action: DecisionAction;
  decided_at: string;
  time_to_decision_seconds: number;
}

export interface Stats {
  queue: { pending: number; resolved_24h: number; median_time_to_decision_s: number };
  model: {
    emergent_case_rate: number;
    agreement_rate: number;
    explanation_useful_rate: number;
  };
  distribution: {
    toxicity: Record<ToxicityClass, number>;
    misinformation: Record<MisinfoClass, number>;
  };
}

export interface ApiError {
  error: { code: string; message: string; detail?: Record<string, unknown> };
}
