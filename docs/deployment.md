# Deployment

Backend on **Google Cloud Run**, frontend on **Vercel**.

```
Vercel (Next.js)  ──REST──▶  Cloud Run (FastAPI + CLIP + trained heads)
                                   │
                                   └──▶ Gemini / Groq for the narrative
```

---

## Why Cloud Run and not Render

PROJECT_CONTEXT Sec. 7 named Render or HF Spaces. Render was tried first and
does not fit, for a measurable reason.

The built image is **690MB resident with every branch loaded**, against Render's
free-tier ceiling of **512MB**. As configured the service would have been
OOM-killed on its first request. Dropping the deepfake branch brought it to
473MB and did fit — that was the state at commit `12d2136` — but it meant
shipping the system with a documented feature switched off.

Cloud Run allows 2GB on a scale-to-zero service, so nothing has to be disabled.
The deepfake detector is baked into the image and active in production.

The cost of the move is a real billing account rather than a free tier, which is
why `--max-instances` is capped: a traffic spike or a runaway loop cannot run up
an unbounded bill.

---

## 1. Backend → Cloud Run

One command. `cloudbuild.yaml` builds the image, pushes it to Artifact
Registry, and deploys the service.

```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_SHORT_SHA=$(git rev-parse --short HEAD)
```

The substitution is what tags the image with the commit it was built from,
which matters when tracing a bug back to a deploy. It is optional — omitting it
falls back to the tag `manual` — but the plain form is only there so the
command still runs at all outside a git checkout; always pass it when you have
one.

First run needs the APIs and the registry to exist:

```bash
gcloud services enable cloudbuild.googleapis.com run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create vanguard --repository-format=docker --location=asia-south1
```

Build takes 15–25 minutes. Most of it is installing CPU-only torch and baking
the model cache into the image, so cold starts do not wait on the Hub.

### What the settings are for

| Setting | Value | Reason |
|---|---|---|
| `--memory` | 2Gi | 690MB resident; the headroom absorbs request-time allocation, it is not spare |
| `--cpu` | 2 | Model load is ~14s on CPU; two cores keeps first-request latency tolerable |
| `--min-instances` | 0 | Idle service costs nothing. Accepts a cold start in exchange |
| `--max-instances` | 3 | Ceiling on spend. This is a real billing account, not a free tier |
| `--concurrency` | 4 | Each instance holds CLIP plus six heads; more in-flight requests contend for the same 2 vCPU |
| `--timeout` | 120 | Well above the ~600ms analyse path, with room for the LLM call |
| `machineType` | E2_HIGHCPU_8 | The default builder runs out of memory installing torch |
| `diskSizeGb` | 60 | The image plus its layers exceed the default builder disk |

### Two things that are easy to get wrong

**The push is an explicit step.** `cloudbuild.yaml` deliberately has no
top-level `images:` block. Those push only after every step completes, so the
deploy step would look for a tag that does not exist yet and fail with "image
not found". The push is ordered before the deploy instead.

**The container must honour `$PORT`.** Cloud Run injects it and ignores
`EXPOSE`. The Dockerfile defaults it to 8080 and the start command reads it:

```
CMD ["sh", "-c", "exec uvicorn mcm.serving.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
```

Hardcoding 8000 makes the service fail its health check with no useful log line.

### Trained weights are not in git

The heads are 69MB of binary artefacts and live in a HuggingFace model repo:
[`Aadithya1122/vanguard-moderation-checkpoints`](https://huggingface.co/Aadithya1122/vanguard-moderation-checkpoints).
The Dockerfile fetches them by name at build time.

This matters for reproducibility. An earlier revision copied them from the build
context, which worked only on the machine that trained them and failed on any
fresh clone.

To publish new weights, push to that repo and rebuild; override the source with
`--build-arg CHECKPOINT_REPO=...` if needed.

---

## 2. Frontend → Vercel

```bash
cd frontend && npx vercel --prod
```

Or import the repo at [vercel.com/new](https://vercel.com/new) with root
directory `frontend`.

Set one environment variable:

```
NEXT_PUBLIC_API_BASE=https://vanguard-moderation-api-<hash>-el.a.run.app
```

`NEXT_PUBLIC_*` values are inlined at build time, not read at runtime, so
changing it requires a redeploy. There is no runtime fallback by design — a
silent switch to mock data in production would be worse than an obvious failure.

### Working without a backend

```
NEXT_PUBLIC_USE_MOCK=true
```

Serves the fixtures in `src/lib/mock.ts`, including both worked examples from
PROJECT_CONTEXT Sec. 1. The whole UI is buildable and demoable with no backend
running.

---

## 3. Close the loop

**`--set-env-vars` replaces the service's entire environment; `--update-env-vars`
merges into it.** Every command below uses the merging form on purpose.
`cloudbuild.yaml`'s own deploy step made this exact mistake with the plain
`--set-env-vars` form for months — invisibly, because until `GEMINI_API_KEY` and
`GEMINI_MODEL` existed as separately-added variables, there was nothing else on
the service for it to destroy. The first time there was, a routine rebuild
silently deleted both, with no error and no symptom until the explanation
endpoint was actually exercised, since `/health` does not depend on that key.
Treat a CI/CD deploy step as exactly the kind of "user" this warning is for —
it runs `gcloud` unattended, every time, whether or not you're watching.

CORS is set at deploy time to accept any Vercel deployment:

```
--update-env-vars=MCM_ALLOWED_ORIGIN_REGEX=https://.*\.vercel\.app
```

The regex form covers preview deployments, whose URLs change per commit. For a
custom domain, use the explicit list instead:

```bash
gcloud run services update vanguard-moderation-api --region=asia-south1 \
  --update-env-vars=MCM_ALLOWED_ORIGINS=https://yourdomain.com
```

Optional keys, all degrading cleanly when absent:

| Variable | Effect when unset |
|---|---|
| `GEMINI_API_KEY` / `GROQ_API_KEY` | Explanations return `status: "unavailable"`; scores and attributions are unaffected |
| `MONGODB_URI` | Decisions are held in memory and lost on restart |

Set any of these with the same merging flag:

```bash
gcloud run services update vanguard-moderation-api --region=asia-south1 \
  --update-env-vars=GEMINI_API_KEY=your-key-here
```

If an explanation returns `"unavailable"` with a key set and the deployed image
predates this note, check that `deploy/requirements-serve.txt` actually installs
`google-genai` and `groq` — the Dockerfile builds with `pip install --no-deps -e .`,
so `pyproject.toml`'s own `[serve]` extra, which lists both, is never read at
build time. Both packages missing was the state of every build before this was
caught: the endpoint always returned `"unavailable"` regardless of whether a key
was configured, because the import itself failed and was swallowed by the same
per-backend `except Exception` that is meant to catch an unset key.

`/analyze` is rate-limited per client to 20 requests/minute — it is the only
endpoint that costs a CLIP forward pass and, once an explanation is requested,
a paid LLM call. The limit is in-process (`mcm/serving/ratelimit.py`), so it
resets per instance and under-counts once more than one instance is running;
that is an accepted tradeoff at demo scale, not an oversight.

---

## Verifying

```bash
curl https://<service-url>/api/v1/health
```

```json
{ "status": "ok", "models_loaded": true, "warm": true, "device": "cpu" }
```

`models_loaded: false` means the container is up but weights are still loading.
That is a normal cold-start state, not a failure; the frontend polls and shows a
warming indicator rather than letting the first request look like a hang.

An end-to-end check:

```bash
curl -X POST https://<service-url>/api/v1/analyze \
  -F 'text=Sending them a little gift they wont forget' \
  -F 'image=@some-image.jpg'
```

Look for `fusion_signal.is_emergent` and the three `modality_scores` — those are
what the interface is built around.

---

## Cost

Cloud Run scales to zero, so an idle service is free and billing is per request.
At demo traffic this is cents per month. Vercel's hobby tier covers the frontend.

The tradeoff for scale-to-zero is a cold start: roughly 15–20s while the
container boots and loads the models. `--min-instances=1` removes it and costs
about $15/month, which is worth setting for a live viva and turning back off
afterwards.
