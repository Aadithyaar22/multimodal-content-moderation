# Frontend Build Brief

Paste this into a fresh session to start the frontend. It assumes
[`docs/api.md`](api.md) is available alongside it.

---

## Prompt

Build the frontend for a multimodal content-moderation system. Next.js
(App Router) + TypeScript + Tailwind, deploying to Vercel. The FastAPI backend
is specified in `docs/api.md` — build strictly against that contract.

**What the product is.** Moderators at a small platform face a queue of
image+text posts. A model scores each one for harassment and misinformation,
ranks them by urgency, and explains its reasoning. The moderator decides;
the system never removes anything on its own. Build for that: every screen
should make a human faster, not replace them.

**The one idea the UI must convey.** Ordinary moderation tools run separate
classifiers per modality and miss harm that only exists in the *combination* —
an innocuous photo plus an innocuous caption that together imply a threat.
This system detects that. When `fusion_signal.is_emergent` is true, both
individual scores are low but the fused score is high. That case needs a
distinct, prominent visual treatment showing the three numbers side by side
(CV-only, NLP-only, Fusion) so the jump is obvious at a glance. If a viewer
takes away one thing from a demo, it should be this.

### Screens

**1. Review Queue** — the default route.
Ranked list from `GET /queue`, ordered by `priority_score`, not time. Each row:
thumbnail, text preview, priority indicator, which head fired, age, and a badge
when `is_emergent`. Filters for head, minimum priority, and an "emergent only"
toggle. Cursor pagination. Empty and loading states that don't collapse layout.

**2. Item Detail** — `/items/[id]`, from `GET /items/{id}`.
Three regions:
- *Evidence*: the image with a Grad-CAM heatmap overlay that can be toggled and
  opacity-adjusted, and the caption with SHAP token highlighting. Token scores
  are signed — use a diverging scale (e.g. blue negative / red positive), never
  a single-hue ramp, or you invert the meaning of half the data.
- *Scores*: the three-way modality comparison, per-head class distributions,
  and the deepfake score when `checked` is true.
- *Reasoning*: the LLM narrative plus `key_factors`. This arrives on a separate,
  slower call — render the rest immediately and stream this in. Never block the
  page on it, and degrade cleanly when `status` is `failed`.

If `cross_attention.top_links` is available, draw connectors between highlighted
tokens and their attended image regions. That visualization *is* the thesis made
visible; give it room.

**3. Decision Bar** — persistent on the detail view.
Approve / Remove / Escalate / Defer, plus an optional one-line rationale and two
one-click toggles: "agreed with model" and "explanation was useful". Keep those
optional — a required field gets clicked through and the data becomes worthless.
`POST /items/{id}/decision`, optimistic update, advance to the next queue item
on success. Keyboard shortcuts (a/r/e/d, j/k to move) — moderators work fast.

**4. Analyze** — `/analyze`, a manual submission form.
Drag-drop image and/or textarea, calls `POST /analyze`, renders the same detail
layout. This is the demo surface; make it the most polished screen.

**5. Dashboard** — `/stats` aggregates. Queue depth, median time-to-decision,
agreement rate, class distributions. Keep it to a handful of real numbers.

### Non-negotiables

- **Cold start.** The backend sleeps on free tier and takes 30–50s to wake. Poll
  `GET /health` on mount; show an explicit warming state with progress. A first
  request that looks like a hang will read as a broken app in a demo.
- **Never present a score as a verdict.** Language is "flagged for review",
  "recommended action", never "removed" or "violation confirmed". `auto_action`
  is always null; the UI should make it evident nothing acts autonomously.
- **Uncertainty is visible.** Show confidence alongside every label. A 0.51 and
  a 0.97 must not look the same.
- **Accessibility.** Never encode meaning in colour alone — pair every colour
  signal with text or an icon. Heatmaps need a legend. Keyboard-navigable
  throughout, visible focus rings, WCAG AA contrast in both themes.
- **Content warning.** The datasets contain hate speech and slurs by
  construction. Blur flagged imagery by default with a click-to-reveal, and
  put a dismissible advisory on first load.

### Technical

- Server Components for queue/detail fetches; Client Components for the
  interactive overlay, filters, and decision bar.
- `NEXT_PUBLIC_API_BASE` env var. No hardcoded URLs.
- Typed API client in `lib/api.ts` with types mirroring `docs/api.md` exactly.
- Loading skeletons, error boundaries, retry on 503 honouring `Retry-After`.
- Responsive from 375px up; the queue collapses to cards on mobile.
- Dark and light themes, tokens defined once.
- Mock mode: a `USE_MOCK_API` flag serving fixtures, so the frontend is
  buildable before the backend is deployed. Build this first — it unblocks all
  UI work and gives you deterministic demo data.

### Build order

1. Types + API client + mock fixtures
2. Queue (mock)
3. Item detail with attribution overlays (mock)
4. Analyze form (mock)
5. Decision flow
6. Dashboard
7. Swap to live backend, handle cold start
8. Polish: keyboard nav, a11y pass, theming

Start with 1–2 and show me the queue before going further.
