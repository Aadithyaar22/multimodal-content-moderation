# Multimodal Content Moderation

Harm detection from the **relationship between image and text**, not from
per-modality scores — with explainable verdicts for human moderators.

Single-signal moderation fails in two directions. It misses harm that only
exists jointly (an innocuous photo plus an innocuous caption that together imply
a threat), and it over-flags when context is missing (a violent news photograph
with a caption that makes clear it is reporting). Averaging two "safe" scores
stays safe, so late fusion structurally cannot catch the first case. This project
keeps the raw representations alive through a cross-attention block so those
joint patterns can actually be learned, and measures the difference against a
late-fusion baseline.

Framing: this is a **decision-support tool for human moderators**. It ranks and
explains content for faster review. It does not auto-remove anything.

## Status

| Stage | State |
|---|---|
| 1. Dataset download + preprocessing | in progress |
| 2. Unimodal baselines (CV-only, NLP-only) | not started |
| 3. Late fusion baseline | not started |
| 4. Cross-attention fusion (core contribution) | not started |
| 5. Deepfake branch | not started |
| 6. Explainability layer (SHAP / Grad-CAM / LLM) | not started |
| 7. Ablation study + evaluation | not started |
| 8. FastAPI backend | not started |
| 9. Frontend | not started |
| 10. Deployment | not started |

Full architecture and rationale: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

## Setup

Targets Apple Silicon (MPS) with no CUDA anywhere.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Optional subsystems install separately so the base environment stays small:

```bash
uv pip install -e ".[xai]"    # SHAP, Grad-CAM, LIME
uv pip install -e ".[cv]"     # OpenCV, EasyOCR, YOLO
uv pip install -e ".[serve]"  # FastAPI, MongoDB, Gemini/Groq
uv pip install -e ".[track]"  # Weights & Biases
```

Copy `.env.example` to `.env` and fill in the keys you need.

## Data

```bash
python scripts/prepare_data.py --all
```

Each dataset normalizes into one canonical record format
(`src/mcm/data/schema.py`) written as parquet manifests under
`data/processed/<dataset>/<split>.parquet`.

| Dataset | Role | Source |
|---|---|---|
| Hateful Memes | Core fusion thesis — neither modality alone is offensive | `neuralcatcher/hateful_memes` |
| Fakeddit | Misinformation head, 6-way labels collapsed to 3 | `AdoCleanCode/Fakeddit`, `ams-99/fakeddit_9k` |
| HateXplain | Human token-level rationales — explanation ground truth | hate-alert/HateXplain |

Fakeddit runs in two tiers. `offline` (default) uses a mirror that bundles its
images, so the pipeline needs no scraping and is reproducible from a clean
clone. `scale` samples the 794k-row metadata table and fetches images from their
original URLs; a meaningful share of 2019-era Reddit links are dead, so the
realized sample is always smaller than requested.

```bash
python scripts/prepare_data.py --dataset fakeddit --tier scale --sample-size 40000
```

### The label sentinel

Hateful Memes has no misinformation label; Fakeddit has no harassment label.
Rather than inventing labels or training two disjoint models, inapplicable
labels are set to `IGNORE_INDEX` and masked out of the loss. Both heads sit on
one shared fused embedding, so every dataset trains the shared trunk while only
the applicable head receives gradient. This is what makes the multi-task setup
in the architecture honest, and it is the reason a single training loop can
produce the whole ablation table.

### Dataset licensing

These are research datasets with their own terms, and this repo ships none of
their content — only code that fetches it.

- **Hateful Memes** is released by Meta under a research licence via DrivenData.
  Using it here does not waive those terms; read and accept them before running
  the pipeline.
- **Fakeddit** is CC-BY-4.0 metadata referencing user-posted Reddit images.
- **HateXplain** is MIT-licensed and contains slurs and hate speech by
  construction, since that is what it annotates.

## Layout

```
configs/data.yaml        dataset sources and preparation options
src/mcm/config.py        paths, dataset specs
src/mcm/utils/device.py  MPS/CPU selection (no CUDA anywhere)
src/mcm/data/schema.py   the canonical record format + validation
src/mcm/data/prepare/    one normalization pipeline per dataset
scripts/prepare_data.py  build manifests
```

## Hardware

MacBook Air M4, 24GB unified memory, MPS. The CLIP backbone stays frozen and
only the cross-attention layers and heads are trained — full CLIP fine-tuning is
not realistic in this memory budget, and freezing it is also what keeps the
unimodal and fusion arms of the ablation comparable.
