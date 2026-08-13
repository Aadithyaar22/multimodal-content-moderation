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
| 1. Dataset download + preprocessing | done |
| 2. Unimodal baselines (CV-only, NLP-only) | done |
| 3. Late fusion baseline | done |
| 4. Cross-attention fusion (core contribution) | next |
| 5. Deepfake branch | not started |
| 6. Explainability layer (SHAP / Grad-CAM / LLM) | not started |
| 7. Ablation study + evaluation | not started |
| 8. FastAPI backend | not started |
| 9. Frontend | not started |
| 10. Deployment | not started |

Full architecture and rationale: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

## Results so far

Test macro-F1, mean ± sd over 3 seeds. All arms share an identical
`MultiTaskHead` on a frozen CLIP ViT-B/32 backbone, so differences reflect what
feeds the head, not head capacity.

| Arm | Hateful Memes | Fakeddit |
|---|---|---|
| CV-only | 0.6217 ± 0.0064 | 0.6863 ± 0.0040 |
| NLP-only | 0.6283 ± 0.0098 | 0.7031 ± 0.0030 |
| Late fusion | **0.6899** ± 0.0121 | **0.7650** ± 0.0031 |
| Cross-attention | *pending* | *pending* |

Both benchmarks show the same ordering: either modality alone is weak, and
combining them helps substantially. On Hateful Memes that is by construction —
the dataset was built so neither modality alone is offensive — so the ~6-point
unimodal-to-fusion gap is the effect the benchmark exists to produce. The open
question, and the point of the next stage, is how much of the remaining headroom
comes from letting the modalities attend to each other rather than merely be
concatenated.

```bash
python scripts/encode_features.py --all
python scripts/train_baseline.py --all-arms --datasets hateful_memes --seeds 1 2 3
```

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

| Dataset | Role | train / val / test | Source |
|---|---|---|---|
| Hateful Memes | Core fusion thesis — neither modality alone is offensive | 8,500 / 500 / 1,000 | `neuralcatcher/hateful_memes` + `limjiayi/hateful_memes_expanded` |
| Fakeddit | Misinformation head, 6-way labels collapsed to 3 | 4,799 / 2,376 / 1,422 | `AdoCleanCode/Fakeddit`, `ams-99/fakeddit_9k` |
| HateXplain | Human token-level rationales — explanation ground truth | 15,383 / 1,922 / 1,924 | hate-alert/HateXplain |

37,826 rows total: 18,597 with images, 19,229 text-only.

Verify a prepared build at any time:

```bash
python scripts/inspect_data.py
```

### Two data problems this pipeline corrects

Both were present in the upstream sources and both would have quietly corrupted
the headline result, so they are fixed in code and reported loudly at build time
rather than left to be discovered later.

**Hateful Memes: 2,043 missing images.** The primary mirror ships 9,664 images
against a 10,000-row dataset, which looks complete. It is not — 1,707 of those
belong to the `unseen` splits, and 2,043 images referenced by
`train`/`dev_seen`/`test_seen` are absent. Skipping those rows would train on
79.6% of the data and evaluate on 815 of 1,000 test memes, with a shifted class
balance. The expanded mirror carries exactly those 2,043, so they are backfilled.

**Fakeddit: cross-split image leakage.** 25 images appear in more than one split
in the source partition (12 train/val, 10 train/test, 3 val/test). Training on
an image and then evaluating on it rewards memorization, so leaked images are
removed from the eval side.

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
