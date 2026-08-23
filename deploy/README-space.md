---
title: Vanguard Moderation API
emoji: 🛡️
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Vanguard — Multimodal Content Moderation API

Decision support for human moderators. Detects harm in the relationship between
an image and its caption, and explains the reasoning.

**No endpoint removes content.** `auto_action` is present in every verdict and is
always `null`.

## Endpoints

`GET /api/v1/health` · `POST /api/v1/analyze` · `GET /api/v1/queue` ·
`GET /api/v1/items/{id}` · `GET /api/v1/items/{id}/explanation` ·
`GET /api/v1/items/{id}/attributions` · `POST /api/v1/items/{id}/decision` ·
`GET /api/v1/stats` · `GET /api/v1/model-card`

Interactive docs at `/docs`.

## Measured results

Test macro-F1, mean over seeds, cross-attention fusion over frozen CLIP ViT-B/32:

| Arm | Hateful Memes | Fakeddit |
|---|---|---|
| CV-only | 0.6217 | 0.6863 |
| NLP-only | 0.6283 | 0.7031 |
| Late fusion | 0.6910 | 0.7732 |
| Cross-attention | 0.7035 | 0.7705 |

Cross-attention beats late fusion on Hateful Memes by 0.0125 at **p = 0.051**,
which is *not* significant at the conventional threshold. On Fakeddit the two are
indistinguishable (p = 0.596). The split is consistent with Hateful Memes being
built so neither modality alone is offensive, while Fakeddit's text often carries
the label by itself.

Served probabilities are temperature-scaled (fitted on validation); this reduced
expected calibration error from 0.28 to 0.036 on the Hateful Memes fusion arm
without changing any prediction.

## Limitations

- The fusion gain is not statistically significant at n = 8,500.
- The deepfake branch detects facial manipulation only and is skipped, with a
  stated reason, when no face is present.
- Token attribution is leave-one-out occlusion, not Shapley values.
- Free tier sleeps; the first request after idle takes 30–50s.

Source: https://github.com/Aadithyaar22/multimodal-content-moderation
