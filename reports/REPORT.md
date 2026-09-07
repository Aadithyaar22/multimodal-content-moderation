# Multimodal Content Moderation: Cross-Attention Fusion for Decision-Support Review

**Author:** Aadithya A R (Aadi), B.Tech CSE (AI & ML), Global Academy of Technology, Bengaluru
**Repository:** [github.com/Aadithyaar22/multimodal-content-moderation](https://github.com/Aadithyaar22/multimodal-content-moderation)
**Live backend:** Google Cloud Run (`vanguard-moderation-api`, `asia-south1`)
**Model registry:** [huggingface.co/Aadithya1122/vanguard-moderation-checkpoints](https://huggingface.co/Aadithya1122/vanguard-moderation-checkpoints)

All numbers in this report are measured against the committed checkpoints and datasets in this repository (`reports/results/*.json`, `checkpoints/*.pt`), not estimated or rounded up. Where a result is not statistically significant, it is reported as such rather than described in language that implies otherwise — this is a deliberate editorial choice explained in §9.5 and §10.

---

## 1. Introduction & Problem Statement

### 1.1 Limitations of single-modality moderation

Production content-moderation pipelines at most platforms run per-modality classifiers in isolation: a text model scores captions for toxicity, a vision model scores images for graphic or sexual content, a separate pipeline flags manipulated media. Each branch discards everything about its input except a scalar harm score before any combination happens. This produces two symmetric failure modes.

First, **false negatives when harm exists only in the relationship between modalities**. A caption that reads as friendly on its own, paired with an image that is unremarkable on its own, can jointly describe a threat, a harassment pattern, or a scam that neither branch has any way to see, because by the time the branches are combined, the information needed to see it has already been thrown away.

Second, **false positives when context is stripped from a genuinely graphic image**. A violent photograph accompanying a news report or a documented human-rights account is visually indistinguishable, in isolation, from the same photograph used to glorify the violence. A vision-only classifier cannot tell these apart; only the caption can, and only if the two are read together.

### 1.2 Motivating examples

Two cases from the original project brief anchor this report and recur as fixtures throughout the shipped system (`frontend/src/lib/mock.ts`) and in the deployed demo:

**Harassment (fusion catches it, neither modality alone does).** Footage of someone leaving an item at another person's door while covertly filming, captioned *"Sending them a little gift 🎁 they won't forget 😂."* Read alone, the caption is playful. Read alone, the footage shows no violence. Read together, a sarcastic minimiser attached to a covert approach to someone's residence is a harassment pattern a person recognizes immediately and a per-modality classifier cannot, because neither modality's evidence crosses a harm threshold on its own.

**Misinformation (stale image, urgency framing).** A photograph of a crowded hospital captioned *"This is what's happening RIGHT NOW because of the new policy — share before they delete this!!"* CLIP-similarity image-reuse search (§6.3, §9.3) can place the same photograph in circulation months before the claimed event; the caption independently carries urgency-pressure phrasing, a documented misinformation marker. Neither fact alone proves the post false; together they are strong evidence the "current" framing is fabricated.

### 1.3 Objectives & scope

The thesis this project tests: **a fusion mechanism that keeps per-modality representations alive through the point of combination — cross-attention — catches cases a score-level (late) fusion baseline structurally cannot**, because late fusion has already collapsed each modality to a single number before any interaction between them can be modeled. The system is built, trained, evaluated, and deployed to test this thesis honestly, including reporting the result when it does not clear statistical significance (§6.4, §9.2).

Scope is explicitly bounded to a **decision-support tool**. Every API response's `verdict.auto_action` field is hard-typed to `None` at the schema level (`src/mcm/serving/schemas.py`) — not a policy the frontend happens to follow, but a contract the backend cannot violate without changing the type. No code path in this system removes, hides, or blocks content; every flagged item is queued for a human moderator's own decision (`POST /items/{id}/decision`).

---

## 2. Literature Survey

### 2.1 Unimodal moderation systems

Production toxicity classifiers (Jigsaw's Perspective API, DistilBERT/RoBERTa fine-tunes on Jigsaw Toxic Comment) and vision-only NSFW/violence classifiers are mature, cheap to serve, and blind by construction to any harm signal that depends on the pairing of image and text. Hateful Memes (Kiela et al., 2020) was purpose-built to make this failure mode measurable: memes in the dataset are selected specifically so that neither the image nor the text alone is offensive, and only the combination is. It is the primary benchmark used throughout this project for exactly that reason.

### 2.2 Multimodal fusion approaches

Three families are relevant here. **Early fusion** concatenates raw or shallow features before any task-specific processing; expensive to train from scratch and rarely used with frozen backbones. **Late fusion** combines per-modality *decisions* or *pooled embeddings* after each branch has processed its own modality independently — cheap, easy to implement, and the baseline this project builds explicitly (§6.1) to falsify or confirm cross-attention's advantage against. **Cross-attention fusion** lets one modality's token-level representations attend to the other's before pooling, so a joint pattern across modalities can influence the representation itself rather than only a downstream combination rule. This project's core architectural claim (§6.2) is that only the third family can, even in principle, learn the "sarcastic caption + covert action" interaction in the harassment example above, because the first two never retain the token-level structure that pattern lives in.

### 2.3 Explainable moderation systems

A verdict without a reason is not usable by a human moderator who has to decide whether to trust it — this is the practical justification, independent of any regulatory one, for the explainability layer (§7). This project draws on three established XAI techniques rather than inventing new ones: leave-one-out occlusion for text-token attribution (a close cousin of, but distinct from, SHAP — the naming distinction is deliberate and documented in `src/mcm/explain/text.py`), Grad-CAM adapted for a Vision Transformer's patch tokens (§7.1), and an LLM-generated natural-language narrative layered on top of both (§7.2) — a "compute-first, explain-second" pattern where the narrative never gates or can override the computed verdict.

---

## 3. Dataset & Preprocessing

### 3.1 Datasets used

| Dataset | Task | Train | Val | Test |
|---|---|---|---|---|
| Hateful Memes (Meta/FB) | Toxicity (binary: benign / harmful) | 8,500 | 500 | 1,000 |
| Fakeddit | Misinformation (3-way: true / satire / misleading) | 4,799 | 2,376 | 1,422 |

HateXplain, listed in the original brief as an explainability-focused stretch dataset, was not brought into the shared multi-task manifest — Hateful Memes and Fakeddit alone were sufficient to test the fusion thesis on both heads, and adding a third label source would have widened the multi-task masking surface (§3.3) without adding a modality combination the two existing datasets don't already exercise.

### 3.2 Preprocessing pipeline

Every source dataset normalizes into one canonical record schema (`src/mcm/data/schema.py`) before anything downstream touches it: `uid`, `dataset`, `split`, `text`, `image_path`, `has_image`, and three label columns (`label_toxicity`, `label_misinfo_3`, `label_misinfo_6`). A label that does not apply to a given sample — Fakeddit rows have no toxicity label, Hateful Memes rows have no misinformation label — is set to an explicit `IGNORE_INDEX` sentinel (`-1`), never dropped and never invented. `validate_frame()` enforces this schema, the label ranges, and uid uniqueness at manifest-build time, so a malformed manifest fails immediately rather than deep inside a training loop.

Images are encoded through CLIP ViT-B/32's own processor (resize, center-crop, normalize) via `FrozenCLIP.preprocess_images`; text through the same model's tokenizer, padded/truncated to CLIP's 77-token context window. Both unimodal arms and the cross-attention arm consume features from this one frozen encoder, which is itself the control that makes the ablation comparison meaningful (§6.4): every arm sees identical input representations, so a difference in outcome is attributable to the fusion mechanism, not to one arm having learned better features.

### 3.3 Class imbalance and the multi-task masking

The shared-trunk, multi-head design (§6.2) means a single training batch can mix Hateful Memes rows (toxicity label present, misinformation label = `IGNORE_INDEX`) and Fakeddit rows (the reverse). The loss (`CrossEntropyLoss(ignore_index=IGNORE_INDEX)`) masks out exactly the positions that don't apply, so a Hateful Memes batch trains the toxicity head and the shared fused representation while leaving the misinformation head's gradient untouched for those rows. This is what makes the multi-task setup honest: the alternative — inventing a placeholder label or training two fully disjoint single-task models — would either corrupt the loss or throw away the one thing multi-task learning is meant to test, namely whether a representation shared across both heads helps either one.

---

## 4. Computer Vision Module

### 4.1 CLIP-based feature extraction

CLIP ViT-B/32 (`openai/clip-vit-base-patch32`, via HuggingFace `transformers`) is used as a frozen backbone throughout — every parameter has `requires_grad=False`, and `FrozenCLIP.train()` is overridden to always call `super().train(False)` so that wrapping it inside a larger module and calling `.train()` on the parent can never silently re-enable dropout inside the backbone (`src/mcm/models/clip_encoder.py`). Two granularities are exposed: **pooled** 512-d projected embeddings (`get_image_features`/`get_text_features`), cheap enough to cache for the full corpus and sufficient for the unimodal and late-fusion arms; and **token-level** hidden states (50×768 image patches including CLS, 77×512 text tokens), computed on demand because cross-attention needs the pre-pooling structure and caching it wholesale would be roughly 100× the storage cost of the pooled cache.

Freezing CLIP is a working-constraint decision, not only a design one: the target hardware is a MacBook Air M4 with 24GB unified memory and no CUDA, and full backbone fine-tuning does not fit that budget. It also serves the ablation's validity directly — every arm sharing one frozen encoder means the fusion mechanism is the only thing varying between arms.

### 4.2 Violence/NSFW classifier and OCR

A dedicated violence/NSFW frame classifier (EfficientNet-B0/YOLOv8-cls, as scoped in the original brief) was not built as a separate branch; CLIP's own frozen vision features feed the CV-only arm of the toxicity and misinformation heads directly, and the deepfake branch (§4.3) covers the manipulated-media case. EasyOCR extracts embedded meme text at request time (`src/mcm/serving/app.py::_run_ocr`) and is surfaced to the moderator alongside the caption — but it is deliberately never spliced into the text the classifiers score. The heads were trained on captions only; feeding OCR text into that same input path at inference time would be an untested train/inference distribution shift disguised as a feature, so OCR output stays purely informational (`docs/api.md`'s note on `ocr_text`).

### 4.3 Deepfake detection

The original brief specified a pretrained Xception checkpoint fine-tuned on FaceForensics++. That checkpoint is EULA-gated with no public HuggingFace mirror, so the deployed detector uses `dima806/deepfake_vs_real_image_detection` — a ViT-based facial-manipulation classifier — instead, a documented substitution rather than a silent one (surfaced in `GET /model-card`'s `limitations` list, and in §9.5 below). The branch is deliberately excluded from the cross-attention block: it reasons about pixel/temporal artifacts entirely unrelated to caption text, so forcing it through a text-conditioned fusion mechanism would not add signal, only noise. It joins the final verdict at score level only (`deepfake` field in every `AnalysisResult`).

Because the detector is trained specifically on facial manipulation, it declines to score an image with no face detected rather than returning a number with nothing behind it — verified live during this session: an image with no face returned `{"checked": false, "label": "not_applicable", "reason": "no face detected; this detector is trained on facial manipulation and its output would not be meaningful here", "face_confidence": -0.039}`. Stating "not checked" explicitly, rather than defaulting to a score that would read as "checked and found authentic," is the same design principle used for `image_reuse.checked` (§6.3) and every other optional branch in the serving layer.

### 4.4 CV-only baseline results

See the master ablation table in §6.4 and §9.2. CV-only macro-F1: 0.6217 on Hateful Memes, 0.6863 on Fakeddit — both well above chance but clearly the weakest of the four arms on their respective benchmark, consistent with each benchmark's design intent (Hateful Memes' images are largely unremarkable on their own; Fakeddit's misinformation cues lean textual).

---

## 5. NLP Module

### 5.1–5.2 Toxicity and misinformation classifiers

The originally scoped separate DistilBERT (toxicity) and RoBERTa/DeBERTa-v3-small (misinformation) fine-tunes were superseded by the frozen-CLIP-text-encoder design once the fusion architecture was locked (§6): using CLIP's own text tower for the NLP-only arm, rather than a second independently-trained text model, keeps every arm's text representation identical to what the cross-attention arm consumes, which is what makes the three-way ablation an apples-to-apples comparison of *fusion mechanisms* rather than a comparison partly confounded by different text encoders. This is a deliberate deviation from the original brief, made for internal validity, and is stated as such rather than left implicit.

### 5.3 NLP-only baseline results

NLP-only macro-F1: 0.6283 on Hateful Memes, 0.7031 on Fakeddit — the stronger of the two unimodal arms on both benchmarks, and markedly closer to the fused arms' performance on Fakeddit specifically (0.703 vs. 0.771–0.773), which foreshadows the fusion architecture's most important negative result in §6.4: on a benchmark where the text carries most of the label on its own, fusion has comparatively little headroom to add.

---

## 6. Deep Learning Fusion Architecture (core contribution)

### 6.1 Late fusion baseline

The late-fusion arm concatenates the pooled 512-d image and text embeddings from the shared frozen CLIP encoder and passes the concatenation through a small MLP head (`hidden_dim=256`, `dropout=0.3`, 133,637 trainable parameters — identical head size to each unimodal arm, so a difference in outcome is attributable to *having both modalities*, not to a larger head). This is the same-modality-input, no-cross-attention control the cross-attention arm is measured against; it is not a strawman — the same PROJECT_CONTEXT.md brief that names cross-attention as the core contribution requires this baseline specifically so the claim is falsifiable rather than assumed.

### 6.2 Cross-attention fusion (proposed architecture)

```
Image ──► CLIP Vision Encoder (frozen) ──► image_tokens (50×768)
Text  ──► CLIP Text Encoder (frozen)   ──► text_tokens (77×512)
                    │
                    ▼
       Cross-Attention Block (2 layers, bidirectional)
       image attends to text, text attends to image
                    │
                    ▼
              fused_emb
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Toxicity Head (→2)   Misinformation Head (→3)
```

Both attention directions run in every layer — image tokens attend to text tokens and text tokens attend to image tokens — because the harassment example (§1.2) needs the model to ask both "which words does this region of the image support" and "which region does this word describe," not only one direction. The two task heads share one fused embedding, the multi-task design whose masking mechanics are described in §3.3.

Hyperparameters differ by dataset because a single configuration tuned on one benchmark performed poorly on the other during an early hyperparameter sweep (a real finding from earlier in this project's development, not a default): Hateful Memes' cross-attention arm uses `d_model=256`, `lr=3e-4`, `fusion_dropout=0.3` (3,687,173 trainable parameters); Fakeddit's uses `d_model=512` at the same learning rate and dropout (13,926,405 trainable parameters). Every arm's checkpoint records its own configuration (`checkpoints/*.pt`'s `config` key), so this asymmetry is reproducible and auditable rather than a remembered fact.

### 6.3 Score-level branches: deepfake and image-reuse

Two signals join the verdict without passing through the cross-attention block at all, each for a stated reason rather than by omission. The deepfake branch (§4.3) reasons about pixel-level artifacts unrelated to caption semantics. Image-reuse detection — added this session to close a gap between what the API contract already documented (`docs/api.md`) and what the service actually computed — is a CLIP cosine-similarity search against every previously analyzed image's normalized embedding (`src/mcm/serving/store.py::find_similar_image`), flagging a re-upload of the same or a near-duplicate image, possibly under a new caption. The similarity threshold (0.90) was set empirically against real Hateful Memes images rather than assumed: the same image reloaded scores 1.00, JPEG-recompressed 0.988, downsized-then-upsized 0.973 — the actual degradation a real re-upload produces — while different images sharing the same meme-template visual style (the hardest case, since the domain is stylistically homogeneous) top out around 0.74. 0.90 sits with a wide margin on both sides rather than splitting a close call.

Both branches are structurally informational: `deepfake` and `image_reuse` are reported alongside the verdict but never feed back into `priority_score` or `verdict.label`. A moderator reading `image_reuse.first_seen_text` next to the current caption can distinguish "same image, different (possibly deceptive) claim" from "same image, legitimately re-shared" — a distinction a fixed similarity threshold alone cannot make, which is exactly why this stays a signal for a human rather than an automatic judgment.

### 6.4 Ablation study (unimodal vs. late-fusion vs. cross-attention)

Measured on the committed test splits, across every recorded training seed (`scripts/ablation_table.py`, `reports/results/cross_attention__*.json` etc.):

**Hateful Memes (toxicity head)**

| Arm | Seeds | Test macro-F1 | 95% CI | AUC |
|---|---|---|---|---|
| CV-only | 3 | 0.6217 ± 0.0064 | [0.606, 0.637] | 0.678 |
| NLP-only | 3 | 0.6283 ± 0.0098 | [0.604, 0.653] | 0.686 |
| Late fusion | 5 | 0.6910 ± 0.0074 | [0.682, 0.700] | 0.764 |
| Cross-attention | 7 | 0.7035 ± 0.0120 | [0.692, 0.715] | 0.769 |

Cross-attention − late fusion = **+0.0125 macro-F1**, Welch's t-test t=2.217, **p=0.051** — not significant at the conventional α=0.05 threshold. At 7 seeds, this gap cannot honestly be called a win over late fusion, even though it points the direction the thesis predicts.

**Fakeddit (misinformation head)**

| Arm | Seeds | Test macro-F1 | 95% CI | AUC |
|---|---|---|---|---|
| CV-only | 3 | 0.6863 ± 0.0040 | [0.676, 0.696] | 0.883 |
| NLP-only | 3 | 0.7031 ± 0.0030 | [0.696, 0.711] | 0.875 |
| Late fusion | 5 | 0.7732 ± 0.0085 | [0.763, 0.784] | 0.926 |
| Cross-attention | 5 | 0.7705 ± 0.0069 | [0.762, 0.779] | 0.914 |

Cross-attention − late fusion = **−0.0027 macro-F1**, t=−0.553, **p=0.596** — indistinguishable from noise, and numerically in the *wrong* direction for the thesis. §9.5 and §10 discuss why, and what it does and doesn't say about the cross-attention hypothesis.

---

## 7. Explainability Layer

### 7.1 Token- and region-level attribution

Text attribution uses leave-one-out occlusion (`src/mcm/explain/text.py`): each token is removed in turn and the resulting change in the fused harm score is that token's attribution, positive meaning "removing it lowered the score, so it was pushing toward harmful." This is explicitly not SHAP — SHAP averages a token's marginal contribution over coalitions of the others, occlusion measures it once against the full context — and the API field name (`method: "occlusion"`) says so rather than misdescribing the method in a report meant to be examined on it. KernelSHAP is available behind `method="shap"` for offline analysis where its ~100 model evaluations per item are affordable; at request time it is not.

Image attribution uses Grad-CAM adapted for CLIP's Vision Transformer (`src/mcm/explain/vision.py`): since a ViT has no convolutional feature map, the gradient-times-activation weighting is taken over the last vision block's 49 patch tokens (the CLS token is dropped first — it has no spatial location, and including it would silently shift every patch by one), ReLU'd so only evidence *for* the target class is shown, and reshaped to the 7×7 patch grid CLIP ViT-B/32 produces at 224px. The backbone stays frozen throughout attribution; gradients are taken with respect to activations, never parameters, so nothing about this process trains anything.

### 7.2 LLM-generated natural-language verdicts

A "compute-first, explain-second" pattern: the verdict, scores, and attribution maps are computed and returned to the moderator immediately; a natural-language narrative (Gemini 2.5 Flash primary, Groq as fallback) is generated separately and asynchronously, and its absence or failure never blocks or alters the verdict (`explanation_status` can be `pending`/`ready`/`failed`/`unavailable` independently of the verdict itself being final). When an item clears threshold on more than one head independently (§9.3), the narrative is instructed to discuss every active head explicitly rather than only the loudest one — a fix made this session after a real production example (§9.3, §9.5) showed the earlier single-head narrative silently dropping a second, independently valid finding.

### 7.3 Deletion-test faithfulness (human evaluation of explanation quality)

PROJECT_CONTEXT.md §6 specifies this and explicitly flags it as the check most likely to be skipped under time pressure. It was run for real this session (`scripts/faithfulness_eval.py`), against the actual trained checkpoints and the actual `grad_cam`/`occlusion_attribution` code the live explanation endpoint calls — not a hand-picked or simulated example.

**Method.** For n=40 sampled test items per dataset: mask the region the explanation names as most important (top-25% Grad-CAM patches for the image; top-25% occlusion-ranked tokens for the text) and re-score; separately mask an equal-sized *random* region and re-score. A faithful explanation should see the score move more from the named region than from an arbitrary one of the same size — reporting only the top-region drop, with no control, would not distinguish "the explanation is faithful" from "the model is simply sensitive to losing any 25% of its input." Both conditions run against the same sampled items, so the two drops are paired, and a paired t-test (the same significance-testing standard §6.4's ablation table is held to) says whether the gap is separable from noise.

**Results:**

| Dataset | Modality | n | Top-region drop | Random-region drop | Gap | p |
|---|---|---|---|---|---|---|
| Hateful Memes | image | 40 | +0.0193 | −0.0102 | +0.0295 | 0.162 |
| Hateful Memes | text | 40 | +0.1412 | +0.0434 | +0.0978 | **0.0001** |
| Fakeddit | image | 40 | −0.0176 | −0.0372 | +0.0195 | 0.310 |
| Fakeddit | text | 40 | +0.0684 | +0.0140 | +0.0544 | **0.018** |

Text attribution is faithful with high confidence on both benchmarks. Image attribution shows the right-signed gap (top-region masking hurts the score more than random masking, on average) on both benchmarks, but the effect is not statistically separable from chance at n=40. This is reported honestly rather than rounded up to "faithful" — it is a genuine limitation, discussed further in §9.5.

Calibration is a prerequisite for any of this being meaningful in production: measured expected calibration error before temperature scaling was 0.28 for the Hateful Memes fusion arm and 0.17 for Fakeddit, against accuracies of 0.69 and 0.82 — a "90% confident" prediction was right closer to 60% of the time. `scripts/calibrate.py` fits one temperature scalar per arm on validation (never on test, to avoid reporting a calibration the deployed model doesn't have) by minimizing validation NLL; because dividing logits by a positive scalar cannot change an argmax, this changes only the reported probabilities, not a single prediction, so the ablation table above is untouched by it.

---

## 8. System Architecture & Deployment

### 8.1 Full stack

```
Vercel (Next.js 16 frontend)
   │ REST, CORS-restricted to the Vercel origin regex
   ▼
Google Cloud Run (FastAPI backend, asia-south1)
   ├── Frozen CLIP ViT-B/32 (shared by every arm)
   ├── Toxicity + misinformation heads × {cv_only, nlp_only, cross_attention}
   ├── Deepfake branch (ViT classifier, score-level)
   ├── OCR (EasyOCR, informational)
   ├── Image-reuse index (CLIP cosine similarity)
   └── LLM explanation calls → Gemini 2.5 Flash (primary) / Groq (fallback)
   ▼
MongoDB Atlas (optional — degrades to a bounded in-memory store when unset)
```

The backend was moved from Render to Cloud Run mid-project: Render's free tier caps memory at 512MB, and the service's measured resident set with every branch loaded is 690MB, so something would have had to be disabled to fit. Cloud Run's configuration (`cloudbuild.yaml`) allows 2Gi/2 vCPU, scales to zero when idle (min-instances=0, so an idle demo costs nothing), and caps at 3 instances (so a traffic spike or a runaway loop cannot run up an unbounded bill).

### 8.2 API design

The full contract is documented in `docs/api.md` and mirrored in three places that are required to move together: the FastAPI Pydantic models (`src/mcm/serving/schemas.py`), the frontend's TypeScript types (`frontend/src/lib/types.ts`), and the doc itself. `POST /analyze` is the only rate-limited endpoint (20 requests/minute per client, in-process) since it is the only one that costs a CLIP forward pass and, once an explanation is requested, a paid LLM call. Every response's `verdict.auto_action` is typed `None` at the schema level — enforced by the type system, not by convention (§1.3).

Two fields added this session close a gap between what the contract already promised and what the service actually computed: `active_heads` (every head that independently clears threshold, not just the single loudest one — §9.3 walks through the real bug this fixes) and `image_reuse` (§6.3).

### 8.3 Latency and cold-start analysis

A cold Cloud Run instance loads: frozen CLIP, the deepfake ViT classifier, the OCR reader, and six small `.pt` checkpoints (cv_only/nlp_only/cross_attention × 2 tasks). Every one of these is baked into the Docker image at build time specifically so a cold start never touches the network — CLIP and the deepfake classifier's weights are pre-fetched during the image build (`deploy/Dockerfile`), for the same reason production services generally avoid depending on an external registry's availability at request time. Model loading runs in a background thread (`app.py`'s lifespan handler) rather than blocking startup, so `/health` can answer immediately and report `loading` state while it works.

Verifying this session's own OCR and image-reuse work by actually redeploying surfaced two real, compounding bugs, found in the order the evidence pointed to them rather than assumed upfront:

**First finding.** EasyOCR's `Reader()` defaults to `download_enabled=True`, so any cache miss or checksum mismatch at runtime silently falls through to a live download instead of raising. A first redeploy took approximately twelve minutes to become ready instead of the previously-documented 30–50 seconds, with the background loading thread producing no log output for the entire gap between the deepfake detector finishing and the first checkpoint-loaded line. `download_enabled=False` (`src/mcm/serving/inference.py`) was the fix — a real bug worth closing regardless (a bad cache should now fail fast into the same "continuing without it" degradation every other optional branch already has, not hang), but redeploying to verify it showed the cold start was still slow: roughly 8.5 minutes.

**Second, deeper finding.** Reading Cloud Logging timestamps against wall-clock time showed the background loading thread sitting idle for minutes between log lines, advancing only in short bursts that lined up with incoming health-check requests. That pattern is the signature of Cloud Run's default CPU allocation: a container instance's CPU is throttled to near-zero *except while it is actively handling a request*. This service's own background-loading design — deliberately not blocking `/health` on model load — means the load thread runs almost entirely *outside* any request context, so under the default setting it was starved of CPU and only inched forward when an incoming health-check request happened to grant the container a brief window. The first finding was a genuine bug, but it was not the dominant cause; it had been diagnosed as the culprit because it was the first anomaly visible in the logs, not because timing it in isolation confirmed it explained the majority of the gap.

The actual fix is `--no-cpu-throttling` on the Cloud Run deploy step (`cloudbuild.yaml`; billed only while an instance is up, and `min-instances=0` still scales fully to zero when idle). Verified live after redeploying with it: cold start dropped to **12 seconds** — faster than the originally documented 30–50 second Render-era estimate, and closer to what loading these models actually costs in CPU time once the process is allowed to have any.

This two-step misdiagnosis is included in full, not condensed to "fixed a cold-start bug," because the more interesting fact for a deployment chapter is *how* a plausible-looking first fix (real, correctly reasoned, and shipped) turned out to explain only part of a symptom, and what it took to notice the difference between "this bug is gone" and "the cold start is actually fast now" — the redeploy-and-measure step, not code review, is what caught it.

---

## 9. Results & Evaluation

### 9.1 Per-modality baseline metrics

See §4.4, §5.3, and the full table in §9.2.

### 9.2 Fusion architecture comparison table (headline result)

The full four-arm table is in §6.4. Restated as the single most important figure in this report: cross-attention fusion outperforms the late-fusion baseline on Hateful Memes by 1.25 macro-F1 points, a gap that is directionally consistent with the fusion thesis but falls just short of conventional significance (p=0.051, 7 seeds). On Fakeddit the two are statistically indistinguishable (p=0.596) and numerically reversed. §9.5 discusses why a flow-specific, honest reading of this — rather than a single blended conclusion — is the more useful one.

### 9.3 Flow-specific evaluation

**Harassment detection (the headline flow).** The metric PROJECT_CONTEXT.md §6 names as the single most important figure in the report — fusion-vs-unimodal recall delta on samples where both individual models score below threshold — is operationalized in this system as the *emergent-signal flag*: an item is marked emergent when both unimodal arms score below 0.5 **and** the fused arm clears 0.5 **and** the fused score exceeds the better unimodal arm by at least 0.15 (`EMERGENT_MARGIN`, `src/mcm/serving/inference.py::emergent_signal`). The margin is deliberate: a bare inequality would fire on rounding noise, and calling a marginal, decision-irrelevant gain "emergent" would inflate the headline rate with cases no moderator would ever actually see differently. Emergent items are lifted in queue priority (`priority_score = min(1.0, score + 0.12)`) specifically because they are the cases a single-signal system would have missed outright, not merely scored lower.

**Misinformation detection.** Image-reuse precision/recall was validated empirically against real near-duplicate degradations rather than assumed (§6.3's threshold derivation). Multiclass F1 on Fakeddit is reported in §6.4 (fusion: 0.7705 macro-F1 test).

**Explainability (moderator-facing).** Faithfulness is reported in full in §7.3. Human agreement rate — the second check PROJECT_CONTEXT.md §6 flags as likely to be skipped — has the full collection infrastructure built and tested (`POST /items/{id}/decision`'s `agreed_with_model`/`explanation_was_useful` fields, `GET /stats`'s `aggregate_stats`, tested in `tests/test_serving.py::TestStore::test_rates_are_over_responders_not_all_decisions`), including a fix made this session so that real raters are individually distinguishable (`frontend/src/lib/moderator.ts` — previously every decision was silently attributed to a single hardcoded `"mod_demo"` id regardless of who clicked it). What has **not** happened is a real pass with real raters; `docs/human-agreement-study.md` is the exact runbook for running one, and §9.5 states this gap plainly rather than filling it with fabricated numbers.

**Researcher/ablation.** The full master comparison table is in §6.4, including parameter counts fine-tuned per arm: 133,637 for each unimodal arm (identical across both benchmarks and both modalities, by design — see §6.1), 3,687,173 for cross-attention on Hateful Memes (`d_model=256`), 13,926,405 for cross-attention on Fakeddit (`d_model=512`).

### 9.4 Case studies

**Case A — Harassment (§1.2, Example A), as implemented.** Fused score 0.71 on toxicity vs. 0.22 (CV-only) and 0.31 (NLP-only) — both unimodal arms below threshold, fused score 0.40 above the better of them, correctly flagged `is_emergent: true`. The generated narrative: *"The caption's playful framing — 'a little gift', a laughing emoji — sits against footage of someone approaching a private doorway while filming covertly... Together they match a harassment pattern, where the sarcasm reframes a covert approach to someone's home as intimidation rather than a favour."*

**Case B — Misinformation (§1.2, Example B), as implemented.** Fused score 0.68 on misinformation, `is_emergent: true` (CV-only 0.21, NLP-only 0.30). This case doubles as the demo fixture for image-reuse (§6.3): the item is marked as a 99% match to an image "first seen" roughly eight months earlier under an unrelated caption, alongside a narrative that independently reasons about the same staleness from the image-reuse-search framing described in the original brief.

**Case C — a real production bug, not a scripted example.** During a comprehensive functionality audit this session, a real analyzed item scored toxicity=0.76 (correctly "harmful," clearing the 2-way head's threshold) while misinformation read 0.94 (the 3-way head's own, differently-scaled confidence). The system's `top_head` selection — "whichever score is numerically higher" — is not a sound way to pick a single lead category when the two heads are structurally different classifiers (2-way vs. 3-way) whose raw scores are not on a comparable scale; filing this item under `top_head="misinformation"` alone made it invisible to a moderator specifically filtering the queue for harassment cases (`GET /queue?head=toxicity`), even though it was a fully independent, threshold-clearing harassment finding. The fix — `active_heads`, every head that independently clears its own threshold, not just the loudest one — now drives the queue filter, the badge rendering, and the explanation narrative (which discusses each active head by name rather than only the highest-scoring one). This case is included deliberately: it is evidence the evaluation and testing built around this system catches real, non-hypothetical failure modes, not only the ones anticipated at design time.

### 9.5 Limitations & failure cases

Stated plainly, in the same spirit as `GET /model-card`'s own `limitations` field:

- **The headline ablation result is not statistically significant on either benchmark** (Hateful Memes p=0.051, Fakeddit p=0.596, reversed in direction). The honest reading is that cross-attention shows a small, right-signed, non-significant edge on the benchmark purpose-built to require joint reasoning (Hateful Memes), and no edge — possibly a very small real disadvantage from the extra parameters' harder optimization — on a benchmark where the label is substantially recoverable from text alone (Fakeddit). This is consistent with the fusion thesis being about *which cases* fusion helps on, not a universal improvement, but the current sample size (5-7 seeds) cannot separate the Hateful Memes gap from seed noise at conventional significance.
- **Image-attribution faithfulness is not statistically significant at n=40** (§7.3), though text attribution is (p<0.02 on both benchmarks). More sampled items, or a coarser/finer masking granularity than the fixed 25% used here, would be the direct next step to sharpen this.
- **The deepfake branch uses a ViT facial-manipulation classifier, not the originally-scoped Xception/FaceForensics++ checkpoint** (§4.3) — a substitution made because the original checkpoint is EULA-gated with no public mirror, stated in the live `GET /model-card` response rather than only in this document.
- **Human-agreement data has not been collected** (§9.3). The infrastructure is built, tested, and — as of this session — fixed to support real multi-rater attribution; the collection itself requires recruiting real people, which is outside what this report can respectably claim.
- **The in-memory store is the default and is wiped on every redeploy or scale-to-zero cycle.** `MONGODB_URI` persists across restarts once configured; as of this report, it has not been (`docs/deployment.md`'s MongoDB section is the exact runbook for doing so).
- **A cold-start regression (up to twelve minutes, against a 30-50s baseline) was found and fixed this session in two passes, not one** (§8.3) — included here as a limitation of the process, not only a resolved bug. The first fix (EasyOCR's default network-download fallback) was real and worth keeping, but redeploying to verify it showed the actual dominant cause was untouched: Cloud Run's default CPU throttling starving this service's own background model-loading thread between requests. Only measuring after each fix, rather than trusting that a plausible, correctly-diagnosed bug was *the* bug, caught the gap — the final verified cold start (12s) is faster than the original documented estimate, not just restored to it.
- **Token attribution is leave-one-out occlusion, not Shapley values** (§7.1) — a legitimate, well-established, but distinct method, named accurately rather than as "SHAP" in a report meant to be checked on it.

---

## 10. Conclusion & Future Work

This project set out to test a specific, falsifiable claim: that keeping per-modality representations alive through cross-attention lets a model learn harm patterns that exist only in the *relationship* between an image and a caption, in a way a late-fusion baseline — which has already collapsed each modality to a score before combining — structurally cannot. The honest result is a **partial, benchmark-dependent confirmation**: on Hateful Memes, a benchmark purpose-built so that neither modality is offensive alone, cross-attention shows the predicted direction of improvement over late fusion but does not clear conventional statistical significance at the sample size trained here (p=0.051, 7 seeds). On Fakeddit, where a large share of the misinformation signal is recoverable from text alone, the two fusion mechanisms are indistinguishable. Reporting this as an unambiguous win would have been the easiest way to make the project's central claim unfalsifiable; reporting it as it actually measured is the more defensible contribution.

Around that central result, the system built to test it turned out to be a substantial deliverable in its own right: a decision-support API (never a black-box auto-moderator, enforced at the type level) with two independently working explainability methods, one of which (text occlusion) is measurably faithful by a controlled deletion test with a random-region baseline, not merely plausible; a working image-reuse detector with an empirically-derived threshold; a deployed, cost-bounded cloud service; and a test suite (148 backend tests at time of writing) built substantially around real bugs this project encountered and fixed during its own development — a CORS-wiping deploy-config bug caught twice, an operator-precedence bug caught before it shipped, a genuine 12-minute cold-start regression caught by actually redeploying and checking, and the `active_heads` blind spot (§9.4, Case C) caught on a real analyzed item rather than a synthetic test case.

**Future work**, in the order it would most change the headline conclusion: (1) more training seeds on the Hateful Memes cross-attention arm specifically, since p=0.051 is close enough to significance that a modest increase in seed count could resolve it either way; (2) a real human-agreement study now that the infrastructure correctly attributes decisions to distinct raters (`docs/human-agreement-study.md`); (3) a larger-n faithfulness eval for the image modality, where the signed direction is right but the current sample size cannot separate it from chance; (4) provisioning `MONGODB_URI` so moderator decisions and the resulting agreement statistics persist past a redeploy; and (5), if a public Xception/FaceForensics++ checkpoint becomes available, replacing the current ViT-based deepfake classifier to close the one remaining documented deviation from the original architecture brief.

---

## Appendix: reproducing these results

```bash
# Ablation table (§6.4, §9.2)
python scripts/ablation_table.py --datasets hateful_memes fakeddit --markdown

# Faithfulness eval (§7.3)
python scripts/faithfulness_eval.py --datasets hateful_memes fakeddit --n 40 --markdown

# Full backend test suite (148 tests at time of writing)
pytest -q
```

Raw results backing every table in this report are committed at `reports/results/*.json` — the ablation runs (`cross_attention__*.json`, `late_fusion__*.json`, etc., one file per architecture/dataset/seed) and `faithfulness.json` (per-dataset, per-item deletion-test results).
