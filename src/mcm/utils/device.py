"""Device selection for Apple Silicon.

The whole project targets a MacBook Air M4 with unified memory and no CUDA, so
MPS is the primary accelerator and CPU is the fallback. Nothing here should ever
import or assume CUDA.
"""

from __future__ import annotations

import os

import torch

# Several ops used by CLIP/transformers still lack MPS kernels. Without this the
# process hard-crashes instead of silently running that one op on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device.

    prefer: force a specific device ("mps", "cpu"); None means auto-detect.
    """
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report() -> dict[str, object]:
    """Diagnostics printed at the start of every training run."""
    dev = get_device()
    return {
        "torch": torch.__version__,
        "device": str(dev),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "threads": torch.get_num_threads(),
    }


def empty_cache() -> None:
    """Release cached MPS allocations between training phases."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def autocast_dtype(device: torch.device) -> torch.dtype:
    """MPS autocast is only reliable in float16; CPU prefers bfloat16."""
    return torch.float16 if device.type == "mps" else torch.bfloat16
