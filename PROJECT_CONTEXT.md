# Multimodal Content Moderation / Media Intelligence Platform

**Project owner:** Aadi (Aadithya A R) — B.Tech CSE (AI & ML), Global Academy of Technology, Bengaluru
**Machine:** MacBook Air M4, 24GB unified memory, zero CUDA — all tooling must be MPS/CPU compatible
**Goal:** A single, coherent DL + CV + NLP project (with a GenAI/LLM layer bolted on) that can be written up as a multi-chapter academic report, and also stands as a strong resume/portfolio piece.

This file is the full context dump of everything decided so far. Read this end-to-end before writing any code.

---

## 1. Problem Statement

Social/community platforms moderate content using **single-signal systems**: a text classifier flags toxic captions, a separate image classifier flags NSFW/violent images, a separate pipeline flags deepfakes. These run in isolation, causing two failure modes:

1. **False negatives when signals only make sense together** — e.g. an innocuous caption + an innocuous image that together imply harassment or a threat.
2. **False positives when context is missing** — e.g. a violent-looking image with a caption that's clearly news reporting/satire gets wrongly flagged because modalities never "talk" to each other.

**Thesis of this project:** detect harm from the *relationship between modalities*, not just per-modality scores, and produce **explainable verdicts** instead of a black-box confidence number.

**Framing (important for ethics/report):** this is a **decision-support tool for human moderators**, not an autonomous ban-hammer. It surfaces and prioritizes likely-violating content for faster human review — it does not auto-delete at low confidence. This mirrors how real platforms (e.g. Meta) actually deploy moderation ML.

### Real-time example walkthroughs (use these as case studies in report Ch. 9.4)

**Example A — Harassment (neither modality alone flags it):**
Video of someone leaving a "gift" at a person's door while secretly filming, captioned *"Sending them a little gift 🎁 they won't forget 😂"*. CV alone: passes (no violence/NSFW). NLP alone: passes (playful tone, no slurs). Only jointly modeling "covert delivery action" + "sarcastic threatening language" reveals a harassment/prank-gone-wrong pattern.

**Example B — Misinformation (recycled image + urgency framing):**
Image of a crowded hospital captioned *"This is what's happening right now because of [policy X] — share before they delete this!!"*. CV: image-reuse detection shows it's an old/recycled photo. NLP: urgency-pressure language is a known misinformation marker. Fusion of "old image" + "claimed as current" is the real signal.

---

## 2. Target Users & End-to-End Flows

Position the system for these five user types. Each maps to a report chapter/section and a specific eval metric set (see Section 6).

### 1. Trust & Safety teams (small/mid platforms — marketplaces, community apps)
Flow: user reports a listing (image + "Barely used, DM before it's gone 😉") → CV checks image against reference catalog (CLIP similarity) → NLP flags urgency-pressure language → fusion elevates scam-likelihood → LLM explains → item pushed to **priority review queue** (not auto-removed).

### 2. Ed-tech / student community platforms
Flow: candid photo of a student + mocking caption "lol look at this guy 💀 someone tag him" → CV alone passes, NLP alone borderline → fusion jointly models "non-consensual candid photo of identifiable person" + "mocking/tagging call" → cyberbullying pattern flagged → post **soft-flagged** (held from trending, not deleted) pending review.

### 3. News/citizen journalism aggregators (ties to Aadi's CivicPulse hackathon project)
Flow: flood photo + "Flooding right now near [X] ward, no one's doing anything!!" → CV reverse-embedding search finds the image is 8 months old → NLP flags unverified causal claim → fusion combines "recycled image" + "urgent unverifiable claim" → submission held in **unverified queue** for editor fact-check.

### 4. Human moderators (cuts across all platforms)
Flow: moderator sees a **ranked queue** (not flat list) → clicks item → sees per-modality scores, fused verdict, Grad-CAM heatmap (CV), SHAP token highlights (NLP), and LLM plain-English reasoning → makes final call (approve/remove/escalate) → decision logged back (future work: active learning loop).

### 5. Researchers / ML practitioners
Consumers of the ablation study and GitHub repo/paper — comparing unimodal vs late-fusion vs cross-attention-fusion results on Hateful Memes / Fakeddit.

---

## 3. Datasets

| Modality/Task | Dataset | Size | Notes |
|---|---|---|---|
| NLP — Toxicity/Hate speech | **HateXplain** | ~20K posts | Has span-level rationales — useful for explainability chapter |
| NLP — Toxicity | **Jigsaw Toxic Comment Classification** (Kaggle) | ~160K comments | Easy baseline |
| CV — NSFW/Violence | **Real Life Violence Situations** (Kaggle) or NudeNet dataset | ~2K videos | Violence/NSFW frame classification |
| Multimodal — Fake news | **Fakeddit** | ~1M posts (image+text+label) | Primary multimodal dataset; fine-grained true/satire/misleading labels |
| Multimodal — Hateful memes | **Hateful Memes Challenge (Meta/FB)** | 10K memes | Purpose-built: neither modality alone is offensive — core to the fusion thesis |
| Deepfake/manipulation | **FaceForensics++** | Use a curated ~500MB–1GB subset, NOT the full 1.8TB | Prefer a pretrained HF checkpoint, fine-tune only |
| OCR (meme text) | none | — | Use EasyOCR/Tesseract directly, no training needed |

**Priority order to actually build:** Hateful Memes + Fakeddit first (they directly support the core "signals only make sense together" thesis) → add HateXplain for explainability → deepfake branch last, using a pretrained checkpoint rather than training from scratch.

---

## 4. Model Architecture (LOCKED — do not redesign without reason)

### Per-modality models

**CV:**
- Feature extraction: **CLIP ViT-B/32** (frozen backbone, fine-tune only attention/heads)
- Violence/NSFW frame classifier: EfficientNet-B0 fine-tuned, or YOLOv8-cls
- OCR: EasyOCR for embedded meme text
- Deepfake detection: pretrained **Xception** (FaceForensics++ checkpoint from HuggingFace) — fine-tune only, don't train from scratch

**NLP:**
- Toxicity classifier: **DistilBERT** fine-tuned on Jigsaw + HateXplain
- Misinformation framing detector: **RoBERTa** (or DeBERTa-v3-small for MPS-friendliness) fine-tuned on Fakeddit text
- Optional stretch: sarcasm/context reasoning via Gemini/Groq zero-shot

### Fusion architecture — THE CORE CONTRIBUTION

**Decision: Cross-attention fusion is primary. Late fusion is the ablation baseline.**

Rationale: Late fusion combines already-computed per-modality *scores*. By the time fusion happens, each branch has discarded everything except its own harm score — so the "sarcastic caption + covert action" interaction (Example A above) can never be learned, because both branches independently say "safe" and averaging two safe scores stays safe. Cross-attention keeps the raw *representations* alive so the model can learn joint patterns — this is exactly why Hateful Memes was built as a dataset (neither modality alone is offensive).

```
Image ──► CLIP Vision Encoder (frozen) ──► image_emb (512-dim)
Text  ──► CLIP Text Encoder (frozen)   ──► text_emb (512-dim)
                    │
                    ▼
       Cross-Attention Block (2 layers)
       (image attends to text, text attends to image)
                    │
                    ▼
          fused_emb (512-dim)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Toxicity/Harassment    Misinformation Head
  Head (fused_emb→2)     (fused_emb→3)
        │                       │
        └───────────┬───────────┘
                     ▼
            Final harm verdict
                     ▲
                     │ (score-level combine only, NOT cross-attention)
          Deepfake branch (separate, frame-level CNN — 
          has nothing to do with caption text, so it stays
          outside the cross-attention block by design)
```

**Key design decisions to preserve in implementation:**
1. CLIP backbone stays **frozen**; only fine-tune ~2-4 cross-attention layers + small classification heads. Full CLIP fine-tuning is not realistic on M4 24GB unified memory.
2. Deepfake branch is deliberately **excluded** from cross-attention (it's about pixel/temporal artifacts, unrelated to caption text) and joins the final verdict via score-level combination — this is a documented "right tool for the right job" decision, not an oversight.
3. Multi-head output (toxicity head + misinformation head) shares **one fused embedding** — multi-task learning, gives report material on task interference/synergy.
4. **Late fusion model must also be built**, reusing the same CLIP embeddings but with simple concat + MLP instead of cross-attention — this produces the direct ablation table (unimodal vs late-fusion vs cross-attention) that's the single most important result in the report.

### Explainability layer (reuse patterns from Aadi's CardioLens/SilentSigns SHAP/LIME code)
- SHAP for NLP token-level attribution
- Grad-CAM for CV attention regions
- LLM-generated natural-language verdict explanation (Gemini 2.5 Pro or Groq LLaMA-3.3-70B), same "compute-first, explain-second" pattern as Aadi's CivicPulse hackathon project

---

## 5. Report / Documentation Structure

```
1. Introduction & Problem Statement
   1.1 Limitations of single-modality moderation
   1.2 Real-world motivating examples (harassment + misinformation, see Section 2)
   1.3 Objectives & scope

2. Literature Survey
   2.1 Unimodal moderation systems (prior work)
   2.2 Multimodal fusion approaches (late / early / cross-attention)
   2.3 Explainable moderation systems

3. Dataset & Preprocessing
   3.1 Hateful Memes, Fakeddit, HateXplain, FaceForensics++ subset
   3.2 Preprocessing pipeline per modality
   3.3 Class imbalance handling

4. Computer Vision Module
   4.1 CLIP-based feature extraction
   4.2 Violence/NSFW classifier architecture & training
   4.3 Deepfake detection (Xception fine-tune)
   4.4 CV-only baseline results

5. NLP Module
   5.1 Toxicity classifier (DistilBERT)
   5.2 Misinformation framing detector (RoBERTa)
   5.3 NLP-only baseline results

6. Deep Learning Fusion Architecture (core contribution)
   6.1 Late fusion baseline
   6.2 Cross-attention fusion (proposed architecture)
   6.3 Deepfake branch integration strategy
   6.4 Ablation study (unimodal vs late-fusion vs cross-attention)

7. Explainability Layer
   7.1 SHAP (text), Grad-CAM (image)
   7.2 LLM-generated natural language verdicts
   7.3 Human evaluation of explanation quality

8. System Architecture & Deployment
   8.1 Full stack diagram (Vercel + Render/HF Spaces + MongoDB)
   8.2 API design
   8.3 Latency/cold-start analysis

9. Results & Evaluation
   9.1 Per-modality baseline metrics
   9.2 Fusion architecture comparison table (headline result)
   9.3 Flow-specific evaluation (see Section 6 below)
   9.4 Case studies (Examples A & B from Section 1, walked through end-to-end)
   9.5 Limitations & failure cases

10. Conclusion & Future Work
```

---

## 6. Evaluation Metrics (per target-user flow)

| Flow | Metrics |
|---|---|
| **Trust & Safety triage** | Recall @ high-confidence threshold (missing real scams is worse than false alarms); Precision @ top-K queue; False Positive Rate; time-to-review proxy (items resolvable from explanation alone vs needing full read) |
| **Harassment detection (headline result)** | **Fusion vs Unimodal Recall Delta** — % more harassment cases caught by cross-attention fusion vs NLP-only/CV-only on samples where both individual models scored below threshold. This is the single most important figure in the report. Also: F1 on Hateful Memes test set; calibration/reliability diagram for soft-flag confidence |
| **Misinformation detection** | Image-reuse detection precision/recall (CLIP-similarity check); multiclass F1 on Fakeddit (true/satire/misleading/etc.); false-positive rate specifically on genuine fresh/breaking content wrongly flagged as recycled |
| **Explainability (moderator-facing)** | Explanation faithfulness via deletion/insertion test (remove Grad-CAM/SHAP-highlighted region, verify score drops); human agreement rate (small Likert-scale study, 10-15 raters); explanation readability (word count / reading-grade-level) |
| **Researcher/ablation** | Master comparison table: CV-only, NLP-only, Late Fusion, Cross-Attention Fusion × {F1 on Hateful Memes, F1 on Fakeddit, AUC, params fine-tuned} |

**Honest scope note:** the human-agreement study and deletion-test faithfulness check are the two most likely to get skipped under time pressure (need extra setup: a small survey, a scripted deletion-test loop) — but they're what differentiates this from "just another classifier," so keep at least a lightweight version of both.

---

## 7. Tech Stack

- **CV:** CLIP ViT-B/32 (HuggingFace), EfficientNet-B0/YOLOv8-cls, EasyOCR, OpenCV, Xception (pretrained deepfake checkpoint)
- **NLP:** DistilBERT, RoBERTa/DeBERTa-v3-small (HuggingFace Transformers)
- **Fusion:** custom PyTorch cross-attention block (2 layers) + late-fusion MLP baseline
- **XAI:** SHAP, Grad-CAM, LIME (reuse existing implementations from CardioLens/SilentSigns)
- **LLM reasoning layer:** Gemini 2.5 Pro or Groq LLaMA-3.3-70B API
- **Backend:** FastAPI
- **Frontend:** React (or Next.js)
- **Database:** MongoDB Atlas (free tier) — moderation logs, flagged posts
- **Experiment tracking:** Weights & Biases (W&B)
- **Deployment:**
  - Frontend → **Vercel**
  - Backend (model inference: CV+NLP+Fusion) → **HuggingFace Spaces (Docker)** or **Render Web Service** — NOT Vercel serverless (50MB/10s limits won't fit PyTorch models)
  - Deepfake model → separate HF Spaces endpoint if needed, to avoid bloating main API cold-start
  - LLM calls → direct API calls from backend (Gemini/Groq), no hosting needed
  - Note the Render free-tier cold-start tradeoff (~30-50s sleep/wake) in the deployment chapter if used

```
Vercel (Next.js frontend)
   ↓ REST calls
Render or HF Spaces (FastAPI backend)
   ├── CV models (loaded once, cached)
   ├── NLP models (loaded once, cached)
   ├── Fusion layer
   └── calls out to Gemini/Groq for explanation
   ↓
MongoDB Atlas (moderation history) + W&B (training metrics)
```

---

## 8. Aadi's Working Constraints (apply throughout implementation)

- Mac M4, 24GB unified memory, **zero CUDA dependencies** — everything must run on MPS/CPU
- Strongly prefers **terminal heredoc** (`cat > file << 'EOF'`) and complete implementations over partial patches
- On Mac: use `set +H` before heredocs to disable zsh history expansion
- Prefers direct, command-by-command instructions with expected output shown
- Prefers copy-paste ready outputs, absolute paths, minimal explanation when the task is clear
- For large files (250+ lines), prefers downloading and `mv`-swapping over heredoc
- Dislikes cosmetic redesigns when only functional changes are requested
- Has reusable SHAP/LIME/Grad-CAM code patterns from CardioLens and SilentSigns projects — check those repos before re-implementing XAI from scratch
- Has an existing "Conflict Resolver" gating pattern from MoodScript that the late-fusion baseline should reuse conceptually
- GitHub: Aadithyaar22, HuggingFace: Aadithya1122

## 9. Suggested Build Order (not yet started — first step for Claude Code)

1. Dataset download + preprocessing (Hateful Memes + Fakeddit first, then HateXplain)
2. Unimodal baselines (CV-only, NLP-only) — establish baseline metrics
3. Late fusion baseline (concat + MLP)
4. Cross-attention fusion model (the core contribution)
5. Deepfake branch (pretrained checkpoint, score-level integration)
6. Explainability layer (SHAP, Grad-CAM, LLM explanation generation)
7. Ablation study + evaluation metrics (Section 6)
8. FastAPI backend + MongoDB logging
9. React/Next.js frontend
10. Deployment (Vercel + HF Spaces/Render)
11. Report writing, using Section 5 structure

Nothing has been implemented yet — this document is pure planning/architecture context. Start with step 1.
