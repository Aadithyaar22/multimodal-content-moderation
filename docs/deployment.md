# Deployment

Everything needed to deploy is committed. Both steps below need an account
login, which is why they are instructions rather than something already done —
see "What blocked automation" at the end.

Total time: about five minutes.

---

## 1. Backend → Render

The repo contains a `render.yaml` blueprint, so Render configures the service
itself.

1. Go to **https://dashboard.render.com/blueprints** and sign in with GitHub.
2. **New Blueprint Instance** → pick `Aadithyaar22/multimodal-content-moderation`.
3. Render reads `render.yaml` and proposes one Docker web service on the free
   plan. Apply it.
4. First build takes 10–15 minutes: it installs CPU-only torch and bakes CLIP
   plus the deepfake weights into the image so cold starts do not wait on the
   Hub.

Copy the resulting URL, e.g. `https://vanguard-moderation-api.onrender.com`.

Check it:

```bash
curl https://YOUR-SERVICE.onrender.com/api/v1/health
```

Expect `{"status":"ok","models_loaded":true,...}`. The first call after idle
takes 30–50s while the instance wakes; that is the free tier, and the frontend
shows an explicit warming state for it rather than appearing to hang.

### Optional environment variables

| Variable | Effect if absent |
|---|---|
| `MCM_ALLOWED_ORIGINS` | **Set this** to the Vercel URL, or the browser blocks every request |
| `GROQ_API_KEY` or `GEMINI_API_KEY` | Narrative reports "unavailable"; scores unaffected |
| `MONGODB_URI` | Records live in a bounded in-memory store, lost on restart |

---

## 2. Frontend → Vercel

1. Go to **https://vercel.com/new** and sign in with GitHub.
2. Import `Aadithyaar22/multimodal-content-moderation`.
3. Set **Root Directory** to `frontend`. Vercel detects Next.js on its own.
4. Add environment variables:

   ```
   NEXT_PUBLIC_API_BASE = https://YOUR-SERVICE.onrender.com
   NEXT_PUBLIC_USE_MOCK = false
   ```

5. Deploy.

Importing from GitHub rather than uploading files is deliberate: every push to
`main` then redeploys automatically, so the deployed site cannot drift from the
repo.

### Deploying without a backend

Omit both variables and the frontend runs on its fixtures — a complete,
self-contained demo including the two worked examples from PROJECT_CONTEXT
Sec. 1. Useful for showing the interface before the API is up. The nav shows a
"Mock" badge whenever it is in that mode, so demo data is never mistaken for
model output.

---

## 3. Close the loop

Set `MCM_ALLOWED_ORIGINS` on Render to the Vercel URL and let it redeploy.
Without it the API is running and the site is loading, but every request fails
in the browser with an opaque CORS error.

Preview deployments are already covered: `render.yaml` sets
`MCM_ALLOWED_ORIGIN_REGEX` to `https://.*\.vercel\.app`, since Vercel gives each
preview its own generated subdomain that cannot be listed in advance.

---

## What blocked automation

Three routes were tried; each needs an account action that cannot be done from
here.

**HuggingFace Spaces** returns `402 Payment Required`. Spaces now requires a PRO
subscription for the Docker SDK — only static Spaces remain free. This was the
first choice in PROJECT_CONTEXT Sec. 7, hence Render instead.

**Vercel** returns `403 Forbidden — you don't have permission to create a
project` for the connected account.

**Render** has no API credentials configured, and its free tier is dashboard-only
in practice.

Nothing is missing from the repo; the remaining work is authenticating.

---

## Verifying a deployment

```bash
BASE=https://YOUR-SERVICE.onrender.com

curl -s $BASE/api/v1/health

curl -s -X POST $BASE/api/v1/analyze \
  -F 'text=Sending them a little gift they will not forget'

curl -s $BASE/api/v1/model-card
```

`/model-card` returns the measured ablation including the p-values, so a
deployment can be checked against the reported numbers rather than trusted.

## Cost

Both free tiers. Render sleeps after 15 minutes idle and wakes in 30–50s. For a
viva, hit the health endpoint a few minutes beforehand so the first live request
is warm.
