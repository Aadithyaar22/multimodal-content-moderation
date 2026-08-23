# Backend image for HuggingFace Spaces (Docker SDK).
#
# Two things are deliberate here.
#
# CLIP weights are baked into the image rather than downloaded at boot. They are
# ~600MB; fetching them on every cold start would add minutes to a wake that is
# already 30-50s on free tier, and would fail the whole Space if the Hub were
# briefly unreachable. Baking them makes the image larger and the boot fast,
# which is the right trade for a service that sleeps between demos.
#
# The container runs on CPU. get_device() falls back automatically, so no code
# path is conditional on the accelerator — the same image runs on MPS locally
# and CPU in the Space.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Spaces runs as a non-root user; caches must live somewhere writable.
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    MCM_CHECKPOINT_DIR=/home/user/app/checkpoints

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only torch. The default wheel pulls the full CUDA stack, which is ~2GB of
# libraries this image will never load.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0 torchvision==0.21.0

COPY deploy/requirements-serve.txt ./requirements-serve.txt
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=user:user src/ ./src/
COPY --chown=user:user checkpoints/ ./checkpoints/
COPY --chown=user:user pyproject.toml ./
COPY --chown=user:user README.md ./

RUN pip install --no-cache-dir --no-deps -e .

USER user

# Warm the model cache into the image so a cold start does not wait on the Hub.
RUN python -c "\
from transformers import CLIPModel, CLIPProcessor, AutoModelForImageClassification, AutoImageProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
AutoModelForImageClassification.from_pretrained('dima806/deepfake_vs_real_image_detection'); \
AutoImageProcessor.from_pretrained('dima806/deepfake_vs_real_image_detection'); \
print('model cache warmed')"

# 7860 is the port Spaces routes to.
EXPOSE 7860

# One worker: each would load its own copy of CLIP plus six heads, and the free
# tier does not have the memory for two.
CMD ["uvicorn", "mcm.serving.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
