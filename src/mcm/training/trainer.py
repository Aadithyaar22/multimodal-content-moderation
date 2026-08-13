"""Training loop for the cached-feature ablation arms.

Because the CLIP backbone is frozen, every arm trains on precomputed 512-d
vectors. The whole corpus is ~155MB, so it lives on-device for the entire run
and minibatching is just index shuffling — no dataloader, no image decoding, no
per-epoch I/O. An arm trains in seconds, which is what makes it practical to run
every configuration across several seeds and report a mean and spread rather
than a single number that might be luck.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mcm.config import REPORT_DIR
from mcm.data.datasets import class_weights
from mcm.data.features import load_mixture
from mcm.data.manifest import load_splits
from mcm.data.schema import IGNORE_INDEX, MISINFO_3_LABELS, TOXICITY_LABELS
from mcm.models.baselines import build_model
from mcm.models.heads import MaskedMultiTaskLoss
from mcm.training.metrics import TaskMetrics, evaluate_task
from mcm.utils.device import empty_cache, get_device
from mcm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrainConfig:
    arch: str
    datasets: list[str]
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    dropout: float = 0.3
    seed: int = 42
    patience: int = 10
    normalize_input: bool = True
    task_weights: tuple[float, float] = (1.0, 1.0)
    use_class_weights: bool = True

    def tag(self) -> str:
        return f"{self.arch}__{'+'.join(self.datasets)}__seed{self.seed}"


@dataclass
class RunResult:
    config: dict
    best_epoch: int
    best_val_score: float
    train_seconds: float
    val: dict = field(default_factory=dict)
    test: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed) if torch.backends.mps.is_available() else None


class SplitTensors:
    """Features and labels for one split, resident on the compute device."""

    def __init__(self, frame: pd.DataFrame, datasets: list[str], split: str, device: torch.device):
        cache = load_mixture(datasets, split, frame)
        self.uid = frame["uid"].tolist()
        self.dataset = frame["dataset"].tolist()
        self.image_emb = cache.image_emb.to(device)
        self.text_emb = cache.text_emb.to(device)
        self.image_mask = cache.image_mask.to(device)
        self.label_toxicity = torch.tensor(
            frame["label_toxicity"].to_numpy(), dtype=torch.long, device=device
        )
        self.label_misinfo = torch.tensor(
            frame["label_misinfo_3"].to_numpy(), dtype=torch.long, device=device
        )
        self.frame = frame

    def __len__(self) -> int:
        return len(self.uid)


def train(cfg: TrainConfig, save_dir: Path | None = None) -> RunResult:
    set_seed(cfg.seed)
    device = get_device()

    frames = {s: load_splits(cfg.datasets, s) for s in ("train", "val", "test")}
    data = {s: SplitTensors(frames[s], cfg.datasets, s, device) for s in frames}
    log.info(
        "%s | train=%d val=%d test=%d on %s",
        cfg.tag(),
        len(data["train"]),
        len(data["val"]),
        len(data["test"]),
        device,
    )

    model = build_model(
        cfg.arch,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        normalize_input=cfg.normalize_input,
    ).to(device)

    tox_w = mis_w = None
    if cfg.use_class_weights:
        tox_w = class_weights(frames["train"], "label_toxicity")
        mis_w = class_weights(frames["train"], "label_misinfo_3")
        tox_w = tox_w.to(device) if tox_w is not None else None
        mis_w = mis_w.to(device) if mis_w is not None else None

    criterion = MaskedMultiTaskLoss(
        toxicity_weight=tox_w, misinfo_weight=mis_w, task_weights=cfg.task_weights
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_score, best_epoch, best_state = -1.0, -1, None
    stale = 0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(data["train"]), device=device)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            out = model(
                image_emb=data["train"].image_emb[idx],
                text_emb=data["train"].text_emb[idx],
                image_mask=data["train"].image_mask[idx],
            )
            loss = criterion(
                out, data["train"].label_toxicity[idx], data["train"].label_misinfo[idx]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            optimizer.step()

            epoch_loss += float(loss.total)
            n_batches += 1

        val_metrics = evaluate(model, data["val"])
        score = selection_score(val_metrics)

        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "  epoch %3d  loss=%.4f  val_score=%.4f  (best %.4f @ %d)",
                epoch,
                epoch_loss / max(1, n_batches),
                score,
                best_score,
                best_epoch,
            )

        if stale >= cfg.patience:
            log.info("  early stop at epoch %d (no val gain for %d epochs)", epoch, cfg.patience)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    result = RunResult(
        config=asdict(cfg),
        best_epoch=best_epoch,
        best_val_score=best_score,
        train_seconds=elapsed,
        val={k: v.to_dict() for k, v in evaluate(model, data["val"]).items()},
        test={k: v.to_dict() for k, v in evaluate(model, data["test"]).items()},
    )

    log.info("  done in %.1fs | best val %.4f @ epoch %d", elapsed, best_score, best_epoch)
    for task, m in result.test.items():
        if m["n"]:
            log.info("  test %s: macro_f1=%.4f acc=%.4f auc=%s n=%d",
                     task, m["macro_f1"], m["accuracy"],
                     f"{m['auc']:.4f}" if m["auc"] is not None else "n/a", m["n"])

    if save_dir is not None:
        save_result(result, save_dir, cfg.tag())

    empty_cache()
    return result


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: SplitTensors) -> dict[str, TaskMetrics]:
    model.eval()
    out = model(
        image_emb=data.image_emb, text_emb=data.text_emb, image_mask=data.image_mask
    )
    return {
        "toxicity": evaluate_task(
            "toxicity",
            out.toxicity_logits.float().cpu().numpy(),
            data.label_toxicity.cpu().numpy(),
            TOXICITY_LABELS,
        ),
        "misinformation": evaluate_task(
            "misinformation",
            out.misinfo_logits.float().cpu().numpy(),
            data.label_misinfo.cpu().numpy(),
            MISINFO_3_LABELS,
        ),
    }


def selection_score(metrics: dict[str, TaskMetrics]) -> float:
    """Model-selection criterion: mean macro-F1 over heads that have labels.

    Averaging only the applicable heads matters for mixed-dataset runs. Counting
    a head with zero labelled rows as 0.0 would drag the score down by a
    constant and make early stopping track the wrong thing.
    """
    active = [m.macro_f1 for m in metrics.values() if m.n > 0]
    return float(np.mean(active)) if active else 0.0


@torch.no_grad()
def predict_logits(model: torch.nn.Module, data: SplitTensors) -> dict[str, np.ndarray]:
    """Raw logits, kept for the fusion-vs-unimodal delta and error analysis."""
    model.eval()
    out = model(image_emb=data.image_emb, text_emb=data.text_emb, image_mask=data.image_mask)
    return {
        "toxicity": out.toxicity_logits.float().cpu().numpy(),
        "misinformation": out.misinfo_logits.float().cpu().numpy(),
    }


def save_result(result: RunResult, save_dir: Path, tag: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{tag}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2))
    log.info("  saved -> %s", path)
    return path


def results_dir() -> Path:
    d = REPORT_DIR / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


__all__ = [
    "RunResult",
    "SplitTensors",
    "TrainConfig",
    "evaluate",
    "predict_logits",
    "results_dir",
    "selection_score",
    "set_seed",
    "train",
    "IGNORE_INDEX",
]
