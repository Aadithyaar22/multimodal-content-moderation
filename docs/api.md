# API Contract

Backend: FastAPI on HuggingFace Spaces or Render.
Frontend: Next.js on Vercel.

Base URL: `{NEXT_PUBLIC_API_BASE}/api/v1`

Everything here is designed around one constraint from the problem statement:
this is a **decision-support tool for human moderators**, not an autonomous
remover. No endpoint deletes content. The system produces a ranked queue and an
explanation; a person makes the call and that decision is logged.

---

## Design decisions the frontend depends on

**Compute first, explain second.** `POST /analyze` returns model scores in
roughly 300–800ms. The LLM narrative takes 2–5s and lives behind a separate
call. Do not block the verdict UI on the explanation — render scores
immediately, then fill the narrative in. This mirrors the CivicPulse pattern.

**Cold starts are real.** On Render's free tier the container sleeps and takes
30–50s to wake. `GET /health` is cheap and unauthenticated; call it on app mount
and show a warming state rather than letting the first real request look like a
hang.

**Confidence is not a verdict.** Every response carries `confidence` and a
`recommended_action` derived from thresholds. The UI must never present a score
as a decision. `auto_action` is always `null` in this system — it exists in the
schema so the contract is explicit about the fact that nothing is auto-removed.

---

## Endpoints

### `GET /health`

Liveness plus model readiness. No auth.

```json
{
  "status": "ok",
  "models_loaded": true,
  "warm": true,
  "device": "cpu",
  "version": "0.1.0",
  "loaded_at": "2026-08-21T09:12:04Z"
}
```

`models_loaded: false` means the container is up but weights are still loading —
show a warming state and poll every 2s.

---

### `POST /analyze`

The core call. Accepts an image, text, or both.

`Content-Type: multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | no | Caption, post body, or OCR override |
| `image` | file | no | JPEG/PNG/WebP, ≤10MB |
| `run_ocr` | bool | no | Default `true`. Extracts embedded meme text |
| `source` | string | no | Free-form tag, e.g. `"marketplace"`, `"citizen-report"` |

At least one of `text` or `image` must be present. A request with neither
returns `422`.

**Response `200`:**

```json
{
  "item_id": "itm_01J8XQ2K3M",
  "created_at": "2026-08-21T09:14:22Z",
  "input": {
    "text": "Sending them a little gift 🎁 they won't forget 😂",
    "has_image": true,
    "image_url": "/api/v1/items/itm_01J8XQ2K3M/image",
    "ocr_text": "GIFT INCOMING",
    "modalities": ["image", "text"]
  },
  "verdict": {
    "label": "review",
    "confidence": 0.71,
    "priority_score": 0.83,
    "recommended_action": "queue_for_review",
    "auto_action": null
  },
  "heads": {
    "toxicity": {
      "label": "harmful",
      "score": 0.71,
      "classes": { "benign": 0.29, "harmful": 0.71 }
    },
    "misinformation": {
      "label": "true",
      "score": 0.12,
      "classes": { "true": 0.88, "satire": 0.06, "misleading": 0.06 }
    }
  },
  "modality_scores": {
    "cv_only": { "toxicity": 0.22 },
    "nlp_only": { "toxicity": 0.31 },
    "fusion": { "toxicity": 0.71 }
  },
  "fusion_signal": {
    "is_emergent": true,
    "delta_over_best_unimodal": 0.40,
    "note": "Neither modality alone crosses threshold; harm appears only jointly."
  },
  "deepfake": {
    "checked": true,
    "score": 0.04,
    "label": "authentic"
  },
  "explanation_status": "pending",
  "latency_ms": { "total": 612, "cv": 210, "nlp": 95, "fusion": 18, "ocr": 289 }
}
```

**`fusion_signal` is the money field for your demo.** It flags cases where the
fused score materially exceeds both unimodal scores — exactly the "signals only
make sense together" claim. Give it a visible treatment in the UI; it is the
thing that distinguishes this system from a pair of ordinary classifiers.

**Errors:** `413` file too large, `415` unsupported media type, `422` no input,
`503` models still loading (retry after `Retry-After` seconds).

---

### `GET /items/{item_id}/explanation`

The slow half. Call immediately after `/analyze` returns.

**Response `200`:**

```json
{
  "item_id": "itm_01J8XQ2K3M",
  "status": "ready",
  "narrative": "The caption's playful framing ('a little gift', laughing emoji) sits against footage of someone approaching a private doorway while filming covertly. Neither element is alone objectionable, but together they match a harassment-prank pattern: the sarcasm reframes the covert delivery as intimidation.",
  "key_factors": [
    { "modality": "text", "factor": "sarcastic minimizer", "weight": 0.34 },
    { "modality": "image", "factor": "covert approach to residence", "weight": 0.29 },
    { "modality": "cross", "factor": "tone/action mismatch", "weight": 0.37 }
  ],
  "model": "gemini-2.5-pro",
  "generated_at": "2026-08-21T09:14:26Z",
  "latency_ms": 3120
}
```

`status` is one of `pending` | `ready` | `failed` | `unavailable`. On `failed`,
render the scores and attribution maps without the narrative — the verdict does
not depend on the LLM.

---

### `GET /items/{item_id}/attributions`

Explainability artifacts for the detail view.

```json
{
  "item_id": "itm_01J8XQ2K3M",
  "text": {
    "method": "shap",
    "tokens": [
      { "token": "Sending", "score": 0.02 },
      { "token": "gift", "score": 0.21 },
      { "token": "won't", "score": 0.08 },
      { "token": "forget", "score": 0.33 }
    ]
  },
  "image": {
    "method": "grad-cam",
    "heatmap_url": "/api/v1/items/itm_01J8XQ2K3M/heatmap.png",
    "regions": [
      { "bbox": [0.41, 0.22, 0.68, 0.74], "score": 0.62, "label": "doorway approach" }
    ]
  },
  "cross_attention": {
    "available": true,
    "top_links": [
      { "text_token": "gift", "image_region": [0.41, 0.22, 0.68, 0.74], "weight": 0.28 }
    ]
  }
}
```

Token `score` is signed: positive pushes toward harmful, negative toward benign.
Render as a diverging colour scale, not a single-hue intensity ramp — sign
carries meaning here.

`bbox` is `[x0, y0, x1, y1]` normalized 0–1, origin top-left.

`cross_attention.top_links` is what makes the fusion visible: which words
attended to which image regions. Draw these as connectors in the detail view.

---

### `GET /queue`

The ranked moderator queue. **Ranked, not chronological** — that ordering is the
product.

Query params:

| Param | Type | Default | Notes |
|---|---|---|---|
| `status` | enum | `pending` | `pending` \| `resolved` \| `all` |
| `min_priority` | float | `0.0` | Filter by `priority_score` |
| `head` | enum | — | `toxicity` \| `misinformation` |
| `emergent_only` | bool | `false` | Only items where `fusion_signal.is_emergent` |
| `limit` | int | `25` | Max 100 |
| `cursor` | string | — | Opaque, from previous response |

```json
{
  "items": [
    {
      "item_id": "itm_01J8XQ2K3M",
      "thumbnail_url": "/api/v1/items/itm_01J8XQ2K3M/thumb",
      "text_preview": "Sending them a little gift 🎁 they won't…",
      "verdict": { "label": "review", "confidence": 0.71, "priority_score": 0.83 },
      "top_head": "toxicity",
      "is_emergent": true,
      "status": "pending",
      "created_at": "2026-08-21T09:14:22Z",
      "age_seconds": 412
    }
  ],
  "next_cursor": "eyJvIjoyNX0",
  "total_pending": 143
}
```

---

### `GET /items/{item_id}`

Full record: everything `/analyze` returned, plus explanation, attributions, and
decision history in one payload. Use this for deep links into the detail view so
a refresh doesn't require replaying `/analyze`.

---

### `POST /items/{item_id}/decision`

Records a human decision. This is the only state-changing endpoint.

```json
{
  "action": "remove",
  "moderator_id": "mod_7f3a",
  "rationale": "Confirmed targeted harassment of an identifiable person.",
  "agreed_with_model": true,
  "explanation_was_useful": true
}
```

`action` ∈ `approve` | `remove` | `escalate` | `defer`.

**Response `200`:**

```json
{
  "item_id": "itm_01J8XQ2K3M",
  "status": "resolved",
  "action": "remove",
  "decided_at": "2026-08-21T09:16:40Z",
  "time_to_decision_seconds": 138
}
```

`agreed_with_model` and `explanation_was_useful` are optional but feed the
human-agreement metric in report Sec. 6. Make them one-click, not a form — a
required field here will just get clicked through and poison the data.

---

### `GET /stats`

Dashboard aggregates.

```json
{
  "queue": { "pending": 143, "resolved_24h": 512, "median_time_to_decision_s": 96 },
  "model": {
    "emergent_case_rate": 0.18,
    "agreement_rate": 0.84,
    "explanation_useful_rate": 0.79
  },
  "distribution": {
    "toxicity": { "benign": 0.72, "harmful": 0.28 },
    "misinformation": { "true": 0.61, "satire": 0.14, "misleading": 0.25 }
  }
}
```

---

### `GET /model-card`

Static metadata for an "about this model" panel — architecture, training data,
and the ablation numbers. Populate from `reports/results/`. Showing known
limitations in-product is part of the responsible-deployment framing, not
decoration.

---

## Conventions

- All timestamps ISO-8601 UTC with `Z`.
- All scores are floats in `[0, 1]`.
- Errors: `{ "error": { "code": "string", "message": "human readable", "detail": {} } }`
- `item_id` is opaque — do not parse it.
- CORS: allow the Vercel origin and `localhost:3000`.
- No auth in v1. Add a header-based API key before any public deployment; do not
  ship a public write endpoint without one.
