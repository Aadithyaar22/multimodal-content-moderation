# API Contract

Backend: FastAPI on Google Cloud Run. Frontend: Next.js on Vercel. See
[deployment.md](deployment.md) for why Cloud Run rather than the free tiers
named in PROJECT_CONTEXT Sec. 7.

Base URL: `{NEXT_PUBLIC_API_BASE}/api/v1`

Everything here is designed around one constraint from the problem statement:
this is a **decision-support tool for human moderators**, not an autonomous
remover. No endpoint deletes content. The system produces a ranked queue and an
explanation; a person makes the call and that decision is logged.

---

## Design decisions the frontend depends on

**Compute first, explain second.** `POST /analyze` returns model scores in
roughly 300–800ms. The LLM narrative lives behind a separate call and takes
longer than an early draft of this doc assumed: `gemini-3.1-pro-preview`
measured ~16s in production; the deployed default, `gemini-2.5-flash`, measured
~7–8s on the same prompts with no perceptible quality loss on the cases tried.
Neither is the 2–5s originally estimated. Do not block the verdict UI on the
explanation under any estimate — render scores immediately, then fill the
narrative in whenever it lands, and treat "the fetch failed" as a distinct,
retryable state from "still generating" rather than the same one (a client that
conflates them shows an infinite spinner in front of an answer that may already
be sitting in the response cache). This mirrors the CivicPulse pattern.

**Cold starts are real.** Cloud Run scales to zero between demos and takes
15–20s to wake. `GET /health` is cheap and unauthenticated; call it on app mount
and show a warming state rather than letting the first real request look like a
hang. `models_loaded: false` while `status: "ok"` means "still loading" — poll
again. `status: "error"` with an `error` message means loading finished and
failed; stop polling and show the failure, since it will not resolve on its own.

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
  "loaded_at": "2026-08-21T09:12:04Z",
  "error": null
}
```

`error` is `null` unless `status: "error"`, in which case it carries the reason
loading failed. This is what lets a client tell a 20s cold start apart from a
crashed deployment — both otherwise present as `models_loaded: false` forever.

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
  "active_heads": ["toxicity"],
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

**`active_heads` is every head that independently clears threshold, not just
the loudest one.** The `heads` object always reports both toxicity and
misinformation, but they are structurally different classifiers (2-way vs
3-way) and their raw scores are not on a comparable scale — picking "whichever
is highest" to decide what an item is about is not sound. A real example
caught this directly: a meme scored toxicity=0.76 (correctly harmful) while
misinformation happened to read 0.94, so a single "lead" selection would have
filed it as a misinformation case and made it invisible to a moderator
filtering for harassment specifically. `GET /queue?head=toxicity` matches
against `active_heads`, so this item is findable under either head it actually
clears. Show every entry in `active_heads` in the UI, not only whichever the
verdict headline happens to name — an item with two active heads is two
separate reasons to look at it, not one.

**Errors:** `413` file too large, `415` unsupported media type, `422` no input,
`429` too many requests from this client (retry after `Retry-After` seconds —
this is the only rate-limited endpoint, since it is the only one that costs a
CLIP forward pass and, once an explanation is requested, a paid LLM call),
`503` models still loading (retry after `Retry-After` seconds).

---

### `GET /items/{item_id}/explanation`

The slow half. Call immediately after `/analyze` returns.

**Response `200`:**

```json
{
  "item_id": "itm_01J8XQ2K3M",
  "status": "ready",
  "narrative": "This item is flagged on two independent grounds. The caption reads as harassment on its own — 'go back to where you came from' — and the model scores it harmful at 0.76. Separately, the same post is flagged as misleading at 0.94, the stronger of the two signals. Both should be treated as real concerns rather than the harassment reading being incidental to a misinformation case.",
  "key_factors": [
    { "modality": "image", "factor": "vision-only signal", "weight": 0.73, "head": "toxicity" },
    { "modality": "text", "factor": "language-only signal", "weight": 0.81, "head": "toxicity" },
    { "modality": "image", "factor": "vision-only signal", "weight": 0.62, "head": "misinformation" },
    { "modality": "text", "factor": "language-only signal", "weight": 0.55, "head": "misinformation" },
    { "modality": "cross", "factor": "gain from modelling the pair jointly", "weight": 0.13, "head": "misinformation" }
  ],
  "model": "gemini-2.5-flash",
  "generated_at": "2026-08-21T09:14:26Z",
  "latency_ms": 3120
}
```

`status` is one of `pending` | `ready` | `failed` | `unavailable`. On `failed`,
render the scores and attribution maps without the narrative — the verdict does
not depend on the LLM.

**`key_factors` covers every entry in `active_heads`, not just `top_head`.**
Each factor's `head` says which one it belongs to. This example is the same
real case `active_heads` itself was fixed for: toxicity clears threshold at
0.76 (harassment) while misinformation reads higher at 0.94 — both are genuine
findings, and the narrative is instructed to discuss each one it is given, not
just the strongest. Group `key_factors` by `head` in the UI rather than
rendering them as one flat list once there is more than one head present.

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
| `head` | enum | — | `toxicity` \| `misinformation` — matches `active_heads`, not `top_head` |
| `emergent_only` | bool | `false` | Only items where `fusion_signal.is_emergent` |
| `limit` | int | `25` | Max 100 |
| `cursor` | string | — | Opaque, from previous response |

Two counts come back, and they answer different questions. `total_matching` is
how many rows satisfied *this call's* filters — what `next_cursor` paginates
over. `total_pending` is the system-wide pending count regardless of any filter
applied — the number for a badge or header. Filtering to `emergent_only=true`
must not make a moderator's overall backlog count look smaller than it is.

```json
{
  "items": [
    {
      "item_id": "itm_01J8XQ2K3M",
      "thumbnail_url": "/api/v1/items/itm_01J8XQ2K3M/thumb",
      "text_preview": "Sending them a little gift 🎁 they won't…",
      "verdict": { "label": "review", "confidence": 0.71, "priority_score": 0.83 },
      "top_head": "toxicity",
      "active_heads": ["toxicity"],
      "is_emergent": true,
      "status": "pending",
      "created_at": "2026-08-21T09:14:22Z",
      "age_seconds": 412
    }
  ],
  "next_cursor": "eyJvIjoyNX0",
  "total_matching": 25,
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
